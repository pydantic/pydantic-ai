"""Tests for `ReadOnlySandbox`, the policy wrapper that blocks execution and file mutation.

Unit tests rather than VCR: the wrapper is sandbox infrastructure that never talks to a model
API, so there is no provider behavior to record.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from pydantic_ai import Agent, RunContext, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.sandboxes import LocalSandbox, ReadOnlySandbox, Sandbox, SandboxRef, SupportsFilesystem, SupportsStart

from .sandbox_fakes import FakeSandboxResult, RecordingSandboxBackend

pytestmark = pytest.mark.anyio


@dataclass(frozen=True)
class _Entry:
    name: str
    path: str
    is_dir: bool
    size: int | None = None


class _Fs:
    def __init__(self, files: dict[str, bytes] | None = None) -> None:
        self.files = files or {}

    def _content(self, path: str) -> bytes:
        """Honors the protocol's missing-path contract: `FileNotFoundError`, not `KeyError`."""
        if path not in self.files:  # pragma: no cover
            raise FileNotFoundError(path)
        return self.files[path]

    async def read_bytes(self, path: str) -> bytes:
        return self._content(path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        self.files[path] = data

    async def stat(self, path: str) -> _Entry:
        return _Entry(name=path.rsplit('/', 1)[-1], path=path, is_dir=False, size=len(self._content(path)))

    async def list_dir(self, path: str) -> Sequence[_Entry]:
        return [await self.stat(p) for p in self.files]

    async def make_dir(self, path: str) -> None:
        pass  # pragma: no cover

    async def remove(self, path: str) -> None:  # pragma: no cover
        self._content(path)
        del self.files[path]

    async def exists(self, path: str) -> bool:
        return path in self.files


class _FilesystemBackend:
    """A read-write in-memory backend, recording every executed command."""

    provider = 'fake'
    sandbox_id = 'fake-rw'

    def __init__(self, files: dict[str, bytes] | None = None) -> None:
        self.fs = _Fs(files)
        self.commands: list[str | Sequence[str]] = []

    async def run(
        self,
        command: str | Sequence[str],
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> FakeSandboxResult:
        self.commands.append(command)
        return FakeSandboxResult()

    async def working_dir(self) -> str:
        return '/workspace'


async def test_reads_forward_and_mutations_raise():
    """Reads pass through to the wrapped backend; every mutating or executing operation raises."""
    backend = _FilesystemBackend({'/workspace/data.csv': b'a,b\n1,2\n'})
    sandbox = Sandbox(ReadOnlySandbox(backend))

    assert await sandbox.read_text('data.csv') == 'a,b\n1,2\n'
    assert (await sandbox.fs.stat('/workspace/data.csv')).size == 8
    assert [entry.name for entry in await sandbox.fs.list_dir('/workspace')] == ['data.csv']
    assert await sandbox.fs.exists('/workspace/data.csv')

    with pytest.raises(UserError, match='read-only'):
        await sandbox.write_text('data.csv', 'overwritten')
    with pytest.raises(UserError, match='read-only'):
        await sandbox.fs.make_dir('/workspace/new')
    with pytest.raises(UserError, match='read-only'):
        await sandbox.fs.remove('/workspace/data.csv')
    with pytest.raises(UserError, match='read-only'):
        await sandbox.run(['rm', '-rf', '/workspace'])
    # `UserError` with the policy reason, not the facade's `NotImplementedError` advice to
    # background the command over `run()` — which is also blocked here.
    with pytest.raises(UserError, match='read-only'):
        await sandbox.start(['sleep', '60'])

    assert backend.fs.files == {'/workspace/data.csv': b'a,b\n1,2\n'}
    assert backend.commands == []


async def test_caller_owned_backend_stays_fully_usable_outside_the_wrapper():
    """The caller's backend stays read-write while its model-facing wrapper is read-only."""
    backend = _FilesystemBackend()
    read_only = ReadOnlySandbox(backend)

    await backend.fs.write_bytes('/workspace/data.csv', b'a,b\n')
    result = await backend.run(['touch', 'marker'])
    assert result.exit_code == 0
    assert backend.commands == [['touch', 'marker']]

    sandbox = Sandbox(read_only)
    assert await sandbox.read_text('data.csv') == 'a,b\n'  # the application's write is visible
    with pytest.raises(UserError, match='read-only'):
        await sandbox.run(['touch', 'marker2'])


async def test_identity_and_working_dir_forward():
    """The wrapper keeps the wrapped backend's identity: policy is not part of it."""
    backend = _FilesystemBackend()
    read_only = ReadOnlySandbox(backend)

    assert read_only.provider == 'fake'
    assert read_only.sandbox_id == 'fake-rw'
    assert await read_only.working_dir() == '/workspace'


async def test_connection_close_forwards_detach_but_blocks_termination():
    class ClosableBackend(_FilesystemBackend):
        def __init__(self) -> None:
            super().__init__()
            self.close_calls: list[bool] = []

        async def close(self, *, terminate: bool) -> None:
            self.close_calls.append(terminate)

    backend = ClosableBackend()
    read_only = ReadOnlySandbox(backend)

    async def connect(_: Any) -> ReadOnlySandbox:
        return read_only

    sandbox = Sandbox._from_ref(SandboxRef(sandbox_id='fake'), connect)  # pyright: ignore[reportPrivateUsage]
    assert await sandbox.working_dir() == '/workspace'
    await sandbox._close_connected_backend()  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(UserError, match='cannot terminate'):
        await read_only.close(terminate=True)

    assert backend.close_calls == [False]


async def test_connection_close_ignores_unrelated_incompatible_close_method():
    class BackendWithUnrelatedClose(_FilesystemBackend):
        async def close(self) -> None:
            raise AssertionError('incompatible close must not be called')  # pragma: no cover

    backend = BackendWithUnrelatedClose()
    read_only = ReadOnlySandbox(backend)

    async def connect(_: Any) -> ReadOnlySandbox:
        return read_only

    sandbox = Sandbox._from_ref(SandboxRef(sandbox_id='fake'), connect)  # pyright: ignore[reportPrivateUsage]
    await sandbox.working_dir()
    await sandbox._close_connected_backend()  # pyright: ignore[reportPrivateUsage]


async def test_filesystem_support_mirrors_wrapped_backend():
    """The wrapper claims `SupportsFilesystem` only when the wrapped backend does."""
    assert isinstance(ReadOnlySandbox(_FilesystemBackend()), SupportsFilesystem)
    assert isinstance(ReadOnlySandbox(_FilesystemBackend()), SupportsStart)

    without_filesystem = ReadOnlySandbox(RecordingSandboxBackend('no-fs'))
    assert not isinstance(without_filesystem, SupportsFilesystem)
    with pytest.raises(NotImplementedError, match='SupportsFilesystem'):
        await Sandbox(without_filesystem).read_text('data.csv')


async def test_windowed_read_uses_internal_non_mutating_command(tmp_path: Path):
    (tmp_path / 'lines.txt').write_text('one\ntwo\nthree\nfour\n')
    window = await Sandbox(ReadOnlySandbox(LocalSandbox(tmp_path))).read_file('lines.txt', offset=2, limit=2)

    assert window.lines == ('two', 'three')
    assert window.start_line == 2
    assert window.has_more
    assert window.total_lines is None


async def test_agent_run_surfaces_read_only_reason():
    """A tool writing through `ctx.sandbox` fails the run with the policy reason."""

    def model_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('rewrite', {})])
        return ModelResponse(parts=[TextPart('done')])  # pragma: no cover

    agent: Agent = Agent(FunctionModel(model_func))

    @agent.tool
    async def rewrite(ctx: RunContext[Any]) -> str:
        await ctx.sandbox.write_text('data.csv', 'overwritten')
        return 'wrote'  # pragma: no cover

    backend = _FilesystemBackend({'/workspace/data.csv': b'a,b\n'})
    with pytest.raises(UserError, match='read-only'):
        await agent.run('Rewrite the data file.', sandbox=ReadOnlySandbox(backend))
    assert backend.fs.files == {'/workspace/data.csv': b'a,b\n'}
