"""Tests for sandbox backends, the rich facade, and read-only `RunContext.sandbox` propagation."""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from pydantic_ai import Agent, LocalSandbox, RunContext, SandboxResolutionContext, UnavailableSandbox, UserError
from pydantic_ai.agent import WrapperAgent
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, WrapperCapability
from pydantic_ai.durable_exec._sandbox import contributes_sandbox
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.sandboxes import (
    Sandbox,
    SandboxBackend,
    SandboxConnector,
    SandboxRef,
    SandboxResult,
    SupportsFilesystem,
    SupportsStart,
)
from pydantic_ai.toolsets import FunctionToolset, WrapperToolset
from pydantic_ai.usage import RunUsage

pytestmark = pytest.mark.anyio


@dataclass(frozen=True)
class _Result:
    exit_code: int
    stdout: str
    stderr: str
    stdout_dropped: int = 0
    stderr_dropped: int = 0


@dataclass(frozen=True)
class _Entry:
    name: str
    path: str
    is_dir: bool
    size: int | None = None


class _Fs:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    async def read_bytes(self, path: str) -> bytes:
        return self.files[path]

    async def write_bytes(self, path: str, data: bytes) -> None:
        self.files[path] = data

    async def stat(self, path: str) -> _Entry:
        return _Entry(name=path.rsplit('/', 1)[-1], path=path, is_dir=False, size=len(self.files[path]))

    async def list_dir(self, path: str) -> Sequence[_Entry]:
        return [await self.stat(p) for p in self.files]

    async def make_dir(self, path: str) -> None:
        pass

    async def remove(self, path: str) -> None:
        self.files.pop(path, None)

    async def exists(self, path: str) -> bool:
        return path in self.files


class _RangeFs(_Fs):
    def __init__(self) -> None:
        super().__init__()
        self.ranges: list[tuple[int, int]] = []

    async def read_bytes_range(self, path: str, start: int, end: int) -> bytes:
        self.ranges.append((start, end))
        return self.files[path][start:end]


class FakeSandbox:
    """A minimal in-memory implementation of the `SandboxBackend` protocol."""

    provider = 'fake'

    def __init__(self, name: str, fs: _Fs | None = None) -> None:
        self.name = name
        self._fs = fs or _Fs()

    @property
    def sandbox_id(self) -> str:
        return f'fake-{self.name}'

    @property
    def fs(self) -> _Fs:
        return self._fs

    async def run(
        self,
        command: str | Sequence[str],
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        output_limit: int | None = None,
    ) -> _Result:
        return _Result(exit_code=0, stdout=f'ran:{command}', stderr='')

    async def start(
        self,
        command: str | Sequence[str],
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        output_limit: int | None = None,
    ) -> Any:
        raise NotImplementedError('FakeSandbox cannot start background processes; use `run` instead.')

    async def working_dir(self) -> str:
        return '/workspace'


def _describe(sandbox: Sandbox) -> str:
    return getattr(sandbox.backend, 'name', sandbox.sandbox_id)


def _tool_call_then_text(tool_name: str = 'probe') -> FunctionModel:
    """A model that calls `tool_name` on the first step and returns text on the second."""

    def model_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name, {})])
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(model_func)


def make_probe_agent(seen: list[str], **kwargs: Any) -> Agent:
    agent: Agent = Agent(_tool_call_then_text(), **kwargs)

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        seen.append(_describe(ctx.sandbox))
        return 'ok'

    return agent


def make_identity_probe_agent(seen: list[Sandbox], **kwargs: Any) -> Agent:
    agent: Agent = Agent(_tool_call_then_text(), **kwargs)

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        seen.append(ctx.sandbox)
        return 'ok'

    return agent


def make_connecting_probe_agent(seen: list[str], **kwargs: Any) -> Agent:
    agent: Agent = Agent(_tool_call_then_text(), **kwargs)

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        await ctx.sandbox.run(['true'])
        seen.append(_describe(ctx.sandbox))
        return 'ok'

    return agent


def test_fake_sandbox_conforms_to_protocol():
    sandbox = FakeSandbox('x')
    assert isinstance(sandbox, SandboxBackend)
    assert isinstance(sandbox, SupportsFilesystem)
    assert isinstance(sandbox, SupportsStart)
    # Static conformance too: pyright checks this assignment because tests are type-checked.
    typed: SandboxBackend = sandbox
    assert typed.provider == 'fake'


async def test_fake_sandbox_protocol_surface():
    """Exercise the in-memory protocol implementation used by the run tests."""
    backend = FakeSandbox('surface')
    sandbox = Sandbox(backend)
    await backend.fs.write_bytes('/workspace/data.bin', b'123')
    assert await backend.fs.read_bytes('/workspace/data.bin') == b'123'
    await sandbox.write_text('notes.txt', 'hello')
    assert await sandbox.read_text('notes.txt') == 'hello'
    assert await backend.fs.stat('/workspace/notes.txt') == _Entry(
        name='notes.txt', path='/workspace/notes.txt', is_dir=False, size=5
    )
    assert {entry.name for entry in await backend.fs.list_dir('/workspace')} == {'data.bin', 'notes.txt'}
    await backend.fs.make_dir('/workspace/subdir')
    assert await backend.fs.exists('/workspace/notes.txt')
    await backend.fs.remove('/workspace/notes.txt')
    assert not await backend.fs.exists('/workspace/notes.txt')

    assert await sandbox.working_dir() == '/workspace'
    assert await sandbox.resolve('data.bin') == '/workspace/data.bin'
    assert await sandbox.resolve('data.bin', base='/tmp') == '/tmp/data.bin'
    assert await sandbox.resolve('/absolute') == '/absolute'
    assert (await sandbox.run(['echo', 'ok'])).stdout == "ran:['echo', 'ok']"
    with pytest.raises(NotImplementedError, match='cannot start background processes'):
        await sandbox.start(['echo', 'ok'])


def test_sandbox_wrap_is_idempotent():
    backend = FakeSandbox('wrapped')
    sandbox = Sandbox.wrap(backend)
    assert isinstance(sandbox, SandboxBackend)
    assert sandbox.backend is backend
    assert Sandbox.wrap(sandbox) is sandbox


async def test_facade_text_helpers_resolve_relative_paths():
    backend = FakeSandbox('text')
    sandbox = Sandbox(backend)
    await sandbox.write_text('notes.txt', 'héllo', encoding='utf-16')
    assert backend.fs.files['/workspace/notes.txt'] == 'héllo'.encode('utf-16')
    assert await sandbox.read_text('notes.txt', encoding='utf-16') == 'héllo'


class _FloorOnlySandbox:
    """Command-execution floor backed by `LocalSandbox`, without native extensions."""

    provider = 'floor-only'

    def __init__(self, local: LocalSandbox) -> None:
        self._local = local
        self.shell_commands: list[str] = []

    @property
    def sandbox_id(self) -> str:
        return self._local.sandbox_id

    async def run(
        self,
        command: str | Sequence[str],
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        output_limit: int | None = None,
    ) -> SandboxResult:
        assert isinstance(command, str), 'the shell fallback always issues shell strings'
        self.shell_commands.append(command)
        return await self._local.run(
            command,
            shell=shell,
            cwd=cwd,
            env=env,
            timeout=timeout,
            output_limit=output_limit,
        )

    async def working_dir(self) -> str:
        return await self._local.working_dir()


async def test_floor_only_backend_gets_shell_filesystem_fallback(tmp_path: Path):
    local = LocalSandbox(tmp_path)
    backend = _FloorOnlySandbox(local)
    typed: SandboxBackend = backend
    assert typed.provider == 'floor-only'
    assert typed.sandbox_id == local.sandbox_id
    assert not isinstance(backend, SupportsFilesystem)
    assert not isinstance(backend, SupportsStart)

    sandbox = Sandbox(backend)
    assert (sandbox.provider, sandbox.sandbox_id) == ('floor-only', local.sandbox_id)
    filesystem = sandbox.fs
    assert sandbox.fs is filesystem  # the shell adapter is lazy and cached

    content = 'héllo\n' + 'line\n' * 20_000
    await sandbox.write_text('nested/notes.txt', content)
    assert await sandbox.read_text('nested/notes.txt') == content
    assert sum("printf '%s'" in command for command in backend.shell_commands) > 1

    window = await sandbox.read_file('nested/notes.txt', offset=2, limit=2)
    assert window.lines == ('line', 'line')
    assert window.has_more is True
    assert window.total_lines is None
    assert any('tail -c +1' in command and 'head -c 65536' in command for command in backend.shell_commands)

    eof_window = await sandbox.read_file('nested/notes.txt', offset=20_000, limit=5)
    assert eof_window.lines == ('line', 'line')
    assert eof_window.has_more is False
    assert eof_window.total_lines == 20_001

    await sandbox.write_text('nested/empty.txt', '')
    empty_window = await sandbox.read_file('nested/empty.txt', limit=5)
    assert empty_window.lines == ()
    assert empty_window.total_lines == 0

    payload = bytes(range(256)) * 200
    binary_path = await sandbox.resolve("nested/weird ' blob.bin")
    await filesystem.write_bytes(binary_path, payload)
    assert await filesystem.read_bytes(binary_path) == payload

    file_entry = await filesystem.stat(binary_path)
    assert (file_entry.name, file_entry.path, file_entry.is_dir, file_entry.size) == (
        "weird ' blob.bin",
        binary_path,
        False,
        len(payload),
    )
    directory_path = await sandbox.resolve('nested')
    directory_entry = await filesystem.stat(directory_path)
    assert (directory_entry.name, directory_entry.is_dir, directory_entry.size) == ('nested', True, None)

    subdirectory_path = await sandbox.resolve('nested/subdir')
    await filesystem.make_dir(subdirectory_path)
    newline_path = await sandbox.resolve('nested/line\nbreak.txt')
    await filesystem.write_bytes(newline_path, b'newline')
    entries = {entry.name: entry for entry in await filesystem.list_dir(directory_path)}
    assert set(entries) == {'notes.txt', 'empty.txt', 'subdir', 'line\nbreak.txt', "weird ' blob.bin"}
    assert entries['subdir'].is_dir is True
    assert all(entry.size is None for entry in entries.values())

    made_path = await sandbox.resolve('made/deep')
    await filesystem.make_dir(made_path)
    assert await filesystem.exists(made_path)
    await filesystem.remove(await sandbox.resolve('made'))
    assert not await filesystem.exists(made_path)

    missing_path = await sandbox.resolve('missing')
    with pytest.raises(FileNotFoundError):
        await filesystem.read_bytes(missing_path)
    with pytest.raises(FileNotFoundError):
        await filesystem.stat(missing_path)
    with pytest.raises(FileNotFoundError):
        await filesystem.list_dir(missing_path)
    with pytest.raises(FileNotFoundError):
        await filesystem.remove(missing_path)

    dangling_path = tmp_path / 'dangling'
    dangling_path.symlink_to(tmp_path / 'missing-target')
    await filesystem.remove(str(dangling_path))
    assert not dangling_path.is_symlink()


async def test_floor_only_windowed_read_reports_hidden_pipeline_failure(tmp_path: Path):
    path = tmp_path / 'secret.txt'
    path.write_text('secret')
    path.chmod(0o000)
    if os.access(path, os.R_OK):  # pragma: no cover — only taken when the test runs as root
        path.chmod(0o600)
        pytest.skip('this platform can read files regardless of their mode')

    sandbox = Sandbox(_FloorOnlySandbox(LocalSandbox(tmp_path)))
    try:
        with pytest.raises(OSError, match=r'secret\.txt'):
            await sandbox.read_file('secret.txt', limit=1)
        with pytest.raises(OSError, match=r'secret\.txt'):
            await sandbox.read_text('secret.txt')
    finally:
        path.chmod(0o600)


async def test_floor_only_windowed_read_rejects_empty_result_for_in_range_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A `tail`/`head` failure hidden by `base64` exiting 0 must not read as an empty window.

    Unlike the permissions test above, the file stays readable, so the stat probe succeeds and
    proves the requested window is within the file.
    """
    (tmp_path / 'file.txt').write_text('content')
    backend = _FloorOnlySandbox(LocalSandbox(tmp_path))
    original_run = backend.run

    async def swallow_tail(
        command: str | Sequence[str],
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        output_limit: int | None = None,
    ) -> SandboxResult:
        if isinstance(command, str) and 'tail -c' in command:
            command = 'true'
        return await original_run(command, shell=shell, cwd=cwd, env=env, timeout=timeout, output_limit=output_limit)

    monkeypatch.setattr(backend, 'run', swallow_tail)
    with pytest.raises(OSError, match=r'file\.txt'):
        await Sandbox(backend).read_file('file.txt', limit=1)


async def test_floor_only_exists_but_failed_read_raises_oserror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / 'unreadable.txt'
    path.write_text('content')
    backend = _FloorOnlySandbox(LocalSandbox(tmp_path))
    original_run = backend.run

    async def fail_read(
        command: str | Sequence[str],
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        output_limit: int | None = None,
    ) -> SandboxResult:
        if isinstance(command, str) and command.startswith('base64 <'):
            return _Result(exit_code=1, stdout='', stderr='read failed')
        return await original_run(
            command,
            shell=shell,
            cwd=cwd,
            env=env,
            timeout=timeout,
            output_limit=output_limit,
        )

    monkeypatch.setattr(backend, 'run', fail_read)
    with pytest.raises(OSError, match='read failed'):
        await Sandbox(backend).read_text('unreadable.txt')


async def test_floor_only_write_cancellation_cleans_up_temporary_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    backend = _FloorOnlySandbox(LocalSandbox(tmp_path))
    original_run = backend.run

    async def cancel_write(
        command: str | Sequence[str],
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        output_limit: int | None = None,
    ) -> SandboxResult:
        if isinstance(command, str) and "printf '%s'" in command:
            raise asyncio.CancelledError
        return await original_run(
            command,
            shell=shell,
            cwd=cwd,
            env=env,
            timeout=timeout,
            output_limit=output_limit,
        )

    monkeypatch.setattr(backend, 'run', cancel_write)
    with pytest.raises(asyncio.CancelledError):
        await Sandbox(backend).write_text('cancelled.txt', 'content')
    assert list(tmp_path.glob('.pydantic-ai-*.tmp')) == []


async def test_floor_only_backend_start_is_unavailable(tmp_path: Path):
    sandbox = Sandbox(_FloorOnlySandbox(LocalSandbox(tmp_path)))
    with pytest.raises(NotImplementedError, match='shell backgrounding'):
        await sandbox.start(['echo', 'hello'])


@pytest.mark.parametrize(
    ('content', 'offset', 'limit', 'expected'),
    [
        (b'one\ntwo\nthree\nfour\n', 2, 2, (('two', 'three'), True, 4)),
        (b'one\ntwo\nthree\n', 1, 2, (('one', 'two'), True, 3)),
        (b'one\ntwo\n', 4, 2, ((), False, 2)),
        (b'one\ntwo\nthree', 2, None, (('two', 'three'), False, 3)),
        (b'one\ntwo\nthree', 3, 1, (('three',), False, 3)),
        (b'one\ntwo\nthree\n', 3, 1, (('three',), False, 3)),
        (b'one\r\ntwo\r\n', 1, None, (('one', 'two'), False, 2)),
        (b'x\r', 1, None, (('x',), False, 1)),
        (b'', 1, None, ((), False, 0)),
        (b'one\xfftwo\n', 1, None, (('one�two',), False, 1)),
    ],
)
async def test_read_file_slow_path(
    content: bytes,
    offset: int,
    limit: int | None,
    expected: tuple[tuple[str, ...], bool, int],
):
    backend = FakeSandbox('slow')
    backend.fs.files['/workspace/file'] = content
    window = await Sandbox(backend).read_file('file', offset=offset, limit=limit)
    lines, has_more, total_lines = expected
    assert window.lines == lines
    assert window.text == '\n'.join(lines)
    assert window.start_line == offset
    assert window.has_more is has_more
    assert window.total_lines == total_lines


@pytest.mark.parametrize(('kwargs', 'message'), [({'offset': 0}, 'offset'), ({'limit': 0}, 'limit')])
async def test_read_file_rejects_invalid_windows(kwargs: dict[str, int], message: str):
    with pytest.raises(ValueError, match=message):
        await Sandbox(FakeSandbox('invalid')).read_file('file', **kwargs)


async def test_read_file_fast_path_stops_before_eof():
    filesystem = _RangeFs()
    backend = FakeSandbox('fast', filesystem)
    filesystem.files['/workspace/file'] = b'line\n' * 40_000

    window = await Sandbox(backend).read_file('file', offset=1, limit=2)

    assert window.lines == ('line', 'line')
    assert window.has_more is True
    assert window.total_lines is None
    assert len(filesystem.ranges) <= 2
    assert sum(end - start for start, end in filesystem.ranges) <= 2 * 64 * 1024


async def test_read_file_fast_path_does_not_overread_past_window():
    filesystem = _RangeFs()
    backend = FakeSandbox('fast-window', filesystem)
    filesystem.files['/workspace/file'] = b'a\nb\n' + b'x' * (3 * 64 * 1024)

    window = await Sandbox(backend).read_file('file', offset=1, limit=2)

    assert filesystem.ranges == [(0, 64 * 1024)]
    assert window.lines == ('a', 'b')
    assert window.has_more is True


async def test_read_file_fast_path_reaching_eof_reports_totals():
    filesystem = _RangeFs()
    backend = FakeSandbox('fast-eof', filesystem)
    filesystem.files['/workspace/file'] = b'one\ntwo'

    window = await Sandbox(backend).read_file('file', offset=2, limit=3)

    assert window.lines == ('two',)
    assert window.has_more is False
    assert window.total_lines == 2


async def test_read_file_fast_path_reports_totals_when_first_read_reaches_eof():
    filesystem = _RangeFs()
    backend = FakeSandbox('fast-eof-window', filesystem)
    filesystem.files['/workspace/file'] = b'one\ntwo\nthree\n'

    window = await Sandbox(backend).read_file('file', offset=1, limit=2)

    assert window.lines == ('one', 'two')
    assert window.has_more is True
    assert window.total_lines == 3


@pytest.mark.parametrize(
    ('suffix', 'has_more', 'total_lines'),
    [(b'more', True, 3), (b'', False, 2)],
)
async def test_read_file_fast_path_resolves_chunk_aligned_window_boundary(
    suffix: bytes, has_more: bool, total_lines: int
):
    filesystem = _RangeFs()
    backend = FakeSandbox('fast-boundary', filesystem)
    second_line = b'x' * (64 * 1024 - 3)
    filesystem.files['/workspace/file'] = b'a\n' + second_line + b'\n' + suffix

    window = await Sandbox(backend).read_file('file', offset=1, limit=2)

    assert filesystem.ranges == [(0, 64 * 1024), (64 * 1024, 2 * 64 * 1024)]
    assert window.lines == ('a', second_line.decode())
    assert window.has_more is has_more
    assert window.total_lines == total_lines


async def test_read_file_fast_and_slow_paths_have_window_parity():
    content = b'one\r\ntwo\r\nthree\nfour\r\nfive'
    slow_backend = FakeSandbox('slow')
    slow_backend.fs.files['/workspace/file'] = content
    range_filesystem = _RangeFs()
    range_filesystem.files['/workspace/file'] = content
    fast_backend = FakeSandbox('fast', range_filesystem)

    for offset in (1, 2, 4, 8):
        for limit in (1, 2, 5):
            slow = await Sandbox(slow_backend).read_file('file', offset=offset, limit=limit)
            fast = await Sandbox(fast_backend).read_file('file', offset=offset, limit=limit)
            assert (fast.lines, fast.start_line, fast.has_more) == (slow.lines, slow.start_line, slow.has_more)
            # The file is smaller than the read chunk size, so the range path always reaches
            # EOF and knows the total.
            assert fast.total_lines == slow.total_lines


async def test_bare_run_context_gets_working_default_sandbox():
    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())
    assert ctx.sandbox.provider == 'local'
    await ctx.sandbox.write_text('hello.txt', 'hello')
    assert await ctx.sandbox.read_text('hello.txt') == 'hello'

    backend = ctx.sandbox.backend
    assert isinstance(backend, LocalSandbox)
    async with backend:
        pass


async def test_unavailable_sandbox_surfaces_reason_for_every_operation():
    reason = 'sandbox disabled by policy'
    backend = UnavailableSandbox(reason)
    assert isinstance(backend, SandboxBackend)
    assert isinstance(backend, SupportsFilesystem)
    assert isinstance(backend, SupportsStart)
    typed_backend: SandboxBackend = backend
    typed_filesystem: SupportsFilesystem = backend
    typed_start: SupportsStart = backend
    assert (typed_backend.provider, typed_backend.sandbox_id) == ('unavailable', 'unavailable')
    assert typed_filesystem.fs is backend
    assert typed_start is backend

    sandbox = Sandbox(backend)
    operations = [
        sandbox.run(['echo', 'hello']),
        sandbox.working_dir(),
        sandbox.start(['echo', 'hello']),
        sandbox.fs.read_bytes('/file'),
        sandbox.fs.write_bytes('/file', b'data'),
        sandbox.fs.stat('/file'),
        sandbox.fs.list_dir('/'),
        sandbox.fs.make_dir('/dir'),
        sandbox.fs.remove('/file'),
        sandbox.fs.exists('/file'),
    ]
    for operation in operations:
        with pytest.raises(UserError, match=reason):
            await operation


async def test_run_argument_sandbox_reaches_tools():
    seen: list[str] = []
    agent = make_probe_agent(seen)
    sandbox = FakeSandbox('direct')
    result = await agent.run('go', sandbox=sandbox)
    assert result.output == 'done'
    assert seen == ['direct']


async def test_run_argument_backend_is_exposed_through_facade():
    observed: list[Sandbox] = []
    backend = FakeSandbox('direct')
    await make_identity_probe_agent(observed).run('go', sandbox=backend)
    assert len(observed) == 1
    assert isinstance(observed[0], Sandbox)
    assert observed[0].backend is backend


async def test_existing_facade_passes_through_run_unchanged():
    observed: list[Sandbox] = []
    sandbox = Sandbox(FakeSandbox('rich'))
    await make_identity_probe_agent(observed).run('go', sandbox=sandbox)
    assert observed == [sandbox]


async def test_run_without_sandbox_gets_fresh_local_sandbox():
    roots: list[Path] = []
    providers: list[str] = []
    agent: Agent = Agent(_tool_call_then_text('use_default'))

    @agent.tool
    async def use_default(ctx: RunContext[Any]) -> str:
        providers.append(ctx.sandbox.provider)
        await ctx.sandbox.write_text('notes.txt', 'hello')
        assert await ctx.sandbox.read_text('notes.txt') == 'hello'
        roots.append(Path(await ctx.sandbox.working_dir()))
        return 'ok'

    await agent.run('go')
    assert providers == ['local']
    assert len(roots) == 1
    assert not roots[0].exists()


async def test_default_sandbox_is_torn_down_when_tool_raises():
    roots: list[Path] = []

    def model_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('explode', {})])

    agent: Agent = Agent(FunctionModel(model_func))

    @agent.tool
    async def explode(ctx: RunContext[Any]) -> str:
        await ctx.sandbox.write_text('created.txt', 'content')
        roots.append(Path(await ctx.sandbox.working_dir()))
        raise RuntimeError('boom')

    with pytest.raises(RuntimeError, match='boom'):
        await agent.run('go')
    assert len(roots) == 1
    assert not roots[0].exists()


async def test_non_posix_default_is_present_and_reports_platform_reason(monkeypatch: pytest.MonkeyPatch):
    from pydantic_ai import agent as agent_module

    reason = 'default local sandbox requires POSIX; attach a container sandbox'
    monkeypatch.setattr(
        agent_module,
        'default_sandbox_backend',
        lambda: UnavailableSandbox(reason),
    )
    agent: Agent = Agent(_tool_call_then_text())

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        assert ctx.sandbox.provider == 'unavailable'
        await ctx.sandbox.run(['echo', 'hello'])
        return 'unreachable'

    with pytest.raises(UserError, match=reason):
        await agent.run('go')


async def test_unavailable_sandbox_run_argument_opts_out_of_default():
    reason = 'local execution disabled'
    agent: Agent = Agent(_tool_call_then_text())

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        await ctx.sandbox.run(['echo', 'hello'])
        return 'unreachable'

    with pytest.raises(UserError, match=reason):
        await agent.run('go', sandbox=UnavailableSandbox(reason))


async def test_wrapper_agent_forwards_sandbox():
    seen: list[str] = []
    agent = make_probe_agent(seen)
    await WrapperAgent(agent).run('go', sandbox=FakeSandbox('wrapped'))
    assert seen == ['wrapped']


async def test_run_argument_sandbox_available_in_all_hooks():
    """A run-argument sandbox is available from `for_run` through `after_run`."""
    log: list[str] = []

    @dataclass
    class Watcher(AbstractCapability[Any]):
        async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
            log.append(f'for_run:{_describe(ctx.sandbox)}')
            return self

        async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
            log.append(f'wrap_enter:{_describe(ctx.sandbox)}')
            return await handler()

        async def after_run(self, ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
            log.append(f'after_run:{_describe(ctx.sandbox)}')
            return result

    seen: list[str] = []
    agent = make_probe_agent(seen, capabilities=[Watcher()])
    await agent.run('go', sandbox=FakeSandbox('direct'))
    assert log == ['for_run:direct', 'wrap_enter:direct', 'after_run:direct']


async def test_sandbox_identity_stable_across_steps():
    """Two tool calls in different run steps observe the same sandbox object."""
    observed: list[Any] = []

    def model_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) in (1, 3):
            return ModelResponse(parts=[ToolCallPart('probe', {})])
        return ModelResponse(parts=[TextPart('done')])

    agent: Agent = Agent(FunctionModel(model_func))

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        observed.append(ctx.sandbox)
        return 'ok'

    sandbox = FakeSandbox('stable')
    await agent.run('go', sandbox=sandbox)
    assert len(observed) == 2
    assert isinstance(observed[0], Sandbox)
    assert observed[0].backend is sandbox
    assert observed[1] is observed[0]


async def test_sandbox_available_during_streamed_run():
    seen: list[str] = []
    agent: Agent = Agent(TestModel())  # TestModel calls every registered tool, then streams output

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        seen.append(_describe(ctx.sandbox))
        return 'ok'

    async with agent.run_stream('go', sandbox=FakeSandbox('streamed')) as stream:
        async for _chunk in stream.stream_text():
            pass
    assert seen == ['streamed']


@dataclass
class SandboxCapability(AbstractCapability[Any]):
    """Canonical supplier: a fresh sandbox per run, entered and exited by the run itself."""

    name: str = 'cap'
    events: list[str] = field(default_factory=lambda: [])
    backend: FakeSandbox | None = field(default=None, init=False)

    def get_sandbox(self, ctx: SandboxResolutionContext[Any]) -> AbstractAsyncContextManager[SandboxBackend]:
        self.events.append(f'{self.name}:offered')

        @asynccontextmanager
        async def per_run_sandbox() -> AsyncGenerator[SandboxBackend]:
            self.events.append(f'{self.name}:enter')
            self.backend = FakeSandbox(self.name)
            try:
                yield self.backend
            finally:
                self.events.append(f'{self.name}:exit')

        return per_run_sandbox()


@dataclass
class FakeSandboxConnector:
    backend: SandboxBackend
    provider: str = 'fake'
    calls: list[str] = field(default_factory=lambda: list[str]())
    error: Exception | None = None

    async def connect(self, sandbox_id: str) -> SandboxBackend:
        self.calls.append(sandbox_id)
        await asyncio.sleep(0)
        if self.error is not None:
            raise self.error
        return self.backend


@dataclass
class SandboxConnectorCapability(AbstractCapability[Any]):
    connectors: Sequence[SandboxConnector]

    def get_sandbox_connectors(self) -> Sequence[SandboxConnector]:
        return self.connectors


def test_fake_sandbox_connector_conforms_to_protocol():
    connector: SandboxConnector = FakeSandboxConnector(FakeSandbox('connector'))
    assert connector.provider == 'fake'


async def test_sandbox_ref_connects_once_and_exposes_identity_before_connection():
    backend = FakeSandbox('deferred')
    connector = FakeSandboxConnector(backend)
    observed: list[Sandbox] = []
    agent: Agent = Agent(
        _tool_call_then_text(),
        capabilities=[SandboxConnectorCapability([connector])],
    )

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        sandbox = ctx.sandbox
        observed.append(sandbox)
        assert (sandbox.provider, sandbox.sandbox_id) == ('fake', 'fake-deferred')
        with pytest.raises(UserError, match='has not connected yet'):
            sandbox.backend

        run_results = await asyncio.gather(
            sandbox.run(['one']),
            sandbox.run(['two']),
            sandbox.working_dir(),
            sandbox.fs.exists('/workspace/missing'),
        )
        assert run_results[2:] == ['/workspace', False]
        assert sandbox.backend is backend
        return 'ok'

    result = await agent.run('go', sandbox=SandboxRef('fake', 'fake-deferred'))
    assert result.output == 'done'
    assert len(observed) == 1
    assert connector.calls == ['fake-deferred']


@pytest.mark.parametrize(
    ('ref', 'connectors', 'registered'),
    [
        (SandboxRef('missing', 'sandbox-1'), [], '(none)'),
        (
            SandboxRef('missing', 'sandbox-1'),
            [FakeSandboxConnector(FakeSandbox('other'), provider='other')],
            "'other'",
        ),
    ],
)
async def test_sandbox_ref_requires_matching_connector(
    ref: SandboxRef, connectors: Sequence[SandboxConnector], registered: str
):
    agent = make_connecting_probe_agent([], capabilities=[SandboxConnectorCapability(connectors)])
    with pytest.raises(
        UserError,
        match=rf"No sandbox connector is registered for provider 'missing'.+{re.escape(registered)}",
    ):
        await agent.run('go', sandbox=ref)


async def test_sandbox_ref_connector_failure_is_chained():
    error = RuntimeError('expired')
    connector = FakeSandboxConnector(FakeSandbox('unused'), error=error)
    agent = make_connecting_probe_agent([], capabilities=[SandboxConnectorCapability([connector])])
    with pytest.raises(
        UserError,
        match=re.escape("Failed to connect to sandbox provider 'fake' for sandbox 'expired'.") + '$',
    ) as exc_info:
        await agent.run('go', sandbox=SandboxRef('fake', 'expired'))
    assert exc_info.value.__cause__ is error


async def test_sandbox_ref_wins_over_sandbox_supplier():
    supplier = SandboxCapability(name='loser')
    connector = FakeSandboxConnector(FakeSandbox('winner'))
    seen: list[str] = []
    agent = make_connecting_probe_agent(seen, capabilities=[supplier, SandboxConnectorCapability([connector])])
    await agent.run('go', sandbox=SandboxRef('fake', 'fake-winner'))
    assert seen == ['winner']
    assert supplier.events == []


def test_sandbox_connectors_compose_and_latest_duplicate_wins():
    first = FakeSandboxConnector(FakeSandbox('first'))
    other = FakeSandboxConnector(FakeSandbox('other'), provider='other')
    last = FakeSandboxConnector(FakeSandbox('last'))
    combined = CombinedCapability(
        [
            SandboxConnectorCapability([first]),
            WrapperCapability(wrapped=SandboxConnectorCapability([other, last])),
        ]
    )
    connectors = combined.get_sandbox_connectors()
    assert connectors == [first, other, last]
    assert {connector.provider: connector for connector in connectors} == {'fake': last, 'other': other}
    assert contributes_sandbox(combined) is False


async def test_capability_sandbox_reaches_tools_and_is_bracketed_by_the_run():
    cap = SandboxCapability()
    seen: list[str] = []
    agent = make_probe_agent(seen, capabilities=[cap])
    result = await agent.run('go')
    assert result.output == 'done'
    assert seen == ['cap']
    assert cap.events == ['cap:offered', 'cap:enter', 'cap:exit']


async def test_context_manager_served_backend_is_exposed_through_facade():
    cap = SandboxCapability()
    observed: list[Sandbox] = []
    await make_identity_probe_agent(observed, capabilities=[cap]).run('go')
    assert len(observed) == 1
    assert isinstance(observed[0], Sandbox)
    assert observed[0].backend is cap.backend


async def test_capability_sandbox_live_through_after_run():
    """The run exits the sandbox after `after_run`, so hooks never see a dead handle."""

    @dataclass
    class ContributingWatcher(SandboxCapability):
        async def after_run(self, ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
            self.events.append(f'after_run:{_describe(ctx.sandbox)}')
            return result

    cap = ContributingWatcher()
    await make_probe_agent([], capabilities=[cap]).run('go')
    assert cap.events == ['cap:offered', 'cap:enter', 'after_run:cap', 'cap:exit']


async def test_run_argument_wins_over_capability():
    cap = SandboxCapability(name='loser')
    seen: list[str] = []
    agent = make_probe_agent(seen, capabilities=[cap])
    await agent.run('go', sandbox=FakeSandbox('direct'))
    assert seen == ['direct']
    assert cap.events == []


async def test_last_capability_in_chain_wins_and_losers_are_never_consulted():
    first = SandboxCapability(name='first')
    last = SandboxCapability(name='last')
    seen: list[str] = []
    await make_probe_agent(seen, capabilities=[first, last]).run('go')
    assert seen == ['last']
    assert first.events == []


async def test_capability_without_sandbox_does_not_mask_supplier():
    @dataclass
    class NonSupplier(AbstractCapability[Any]):
        pass

    contributor = SandboxCapability(name='contributor')
    seen: list[str] = []
    await make_probe_agent(seen, capabilities=[contributor, NonSupplier()]).run('go')
    assert seen == ['contributor']


async def test_warm_sandbox_shared_across_runs():
    warm = FakeSandbox('warm')

    @dataclass
    class WarmSandboxCapability(AbstractCapability[Any]):
        def get_sandbox(self, ctx: SandboxResolutionContext[Any]) -> AbstractAsyncContextManager[SandboxBackend]:
            return nullcontext(warm)

    observed: list[Any] = []
    agent: Agent = Agent(_tool_call_then_text(), capabilities=[WarmSandboxCapability()])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        observed.append(ctx.sandbox)
        return 'ok'

    await agent.run('one')
    await agent.run('two')
    assert len(observed) == 2
    assert all(isinstance(sandbox, Sandbox) and sandbox.backend is warm for sandbox in observed)


async def test_bare_sandbox_is_used_without_being_entered():
    """A capability may serve a bare backend; the run does not bracket its lifecycle."""
    warm = FakeSandbox('bare')

    @dataclass
    class BareSandboxCapability(AbstractCapability[Any]):
        def get_sandbox(self, ctx: SandboxResolutionContext[Any]) -> SandboxBackend:
            return warm

    observed: list[Sandbox] = []
    await make_identity_probe_agent(observed, capabilities=[BareSandboxCapability()]).run('go')
    assert len(observed) == 1
    assert isinstance(observed[0], Sandbox)
    assert observed[0].backend is warm


async def test_deferred_capability_never_contributes():
    cap = SandboxCapability(name='deferred')
    cap.id = 'deferred-sandbox'
    cap.defer_loading = True
    seen: list[str] = []
    await make_probe_agent(seen, capabilities=[cap]).run('go')
    assert len(seen) == 1
    assert seen[0].startswith('local-')
    assert cap.events == []


async def test_wrapper_capability_forwards_get_sandbox():
    inner = SandboxCapability(name='inner')
    seen: list[str] = []
    await make_probe_agent(seen, capabilities=[WrapperCapability(wrapped=inner)]).run('go')
    assert seen == ['inner']
    assert inner.events == ['inner:offered', 'inner:enter', 'inner:exit']


async def test_capability_sandbox_exits_when_a_tool_raises():
    cap = SandboxCapability()

    def model_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('explode', {})])

    agent: Agent = Agent(FunctionModel(model_func), capabilities=[cap])

    @agent.tool
    async def explode(ctx: RunContext[Any]) -> str:
        raise RuntimeError('boom')

    with pytest.raises(RuntimeError, match='boom'):
        await agent.run('go')
    assert cap.events == ['cap:offered', 'cap:enter', 'cap:exit']


async def test_capability_sandbox_exits_when_toolset_entry_fails():
    """The exit stack owns the bracket, so teardown runs even when the run never starts."""

    class ExplodingToolset(WrapperToolset[Any]):
        async def __aenter__(self) -> Any:
            raise RuntimeError('toolset entry failed')

    cap = SandboxCapability()
    agent: Agent = Agent(TestModel(), toolsets=[ExplodingToolset(wrapped=FunctionToolset())], capabilities=[cap])
    with pytest.raises(RuntimeError, match='toolset entry failed'):
        await agent.run('go')
    assert cap.events == ['cap:offered', 'cap:enter', 'cap:exit']
