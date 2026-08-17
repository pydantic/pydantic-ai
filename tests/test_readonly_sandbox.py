"""Tests for `ReadOnlySandbox`, the policy wrapper that blocks execution and file mutation.

Unit tests rather than VCR: the wrapper is sandbox infrastructure that never talks to a model
API, so there is no provider behavior to record.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pytest

from pydantic_ai import Agent, ReadOnlySandbox, RunContext, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.sandboxes import Sandbox, SupportsFilesystem, SupportsStart

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


async def test_identity_and_working_dir_forward():
    """The wrapper keeps the wrapped backend's identity: policy is not part of it."""
    backend = _FilesystemBackend()
    read_only = ReadOnlySandbox(backend)

    assert read_only.wrapped is backend
    assert read_only.provider == 'fake'
    assert read_only.sandbox_id == 'fake-rw'
    assert await read_only.working_dir() == '/workspace'


async def test_filesystem_support_mirrors_wrapped_backend():
    """The wrapper claims `SupportsFilesystem` only when the wrapped backend does."""
    assert isinstance(ReadOnlySandbox(_FilesystemBackend()), SupportsFilesystem)
    assert isinstance(ReadOnlySandbox(_FilesystemBackend()), SupportsStart)

    without_filesystem = ReadOnlySandbox(RecordingSandboxBackend('no-fs'))
    assert not isinstance(without_filesystem, SupportsFilesystem)
    with pytest.raises(NotImplementedError, match='SupportsFilesystem'):
        await Sandbox(without_filesystem).read_text('data.csv')


async def test_windowed_read_falls_back_to_filesystem():
    """`read_file` still works read-only: the blocked `sed` optimization falls back to `fs`."""
    backend = _FilesystemBackend({'/workspace/lines.txt': b'one\ntwo\nthree\nfour\n'})
    window = await Sandbox(ReadOnlySandbox(backend)).read_file('lines.txt', offset=2, limit=2)

    assert window.lines == ('two', 'three')
    assert window.start_line == 2
    assert window.has_more
    assert window.total_lines == 4
    assert backend.commands == []


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
