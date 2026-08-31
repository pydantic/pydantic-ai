"""Tests for sandbox backends, the rich facade, and read-only `RunContext.sandbox` propagation."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
import pytest

from pydantic_ai import Agent, RunContext, UnavailableSandbox, UserError
from pydantic_ai.agent import WrapperAgent
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, WrapperCapability
from pydantic_ai.capabilities._sandbox import connect_sandbox_provider, find_sandbox_ref_connector
from pydantic_ai.durable_exec._sandbox import contributes_sandbox, guard_workflow_sandbox
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.sandboxes import (
    Sandbox,
    SandboxBackend,
    SandboxOutputChunk,
    SandboxProcess,
    SandboxRef,
    SandboxResult,
    SupportsFilesystem,
    SupportsStart,
    SupportsStream,
)
from pydantic_ai.toolsets import FunctionToolset, WrapperToolset
from pydantic_ai.usage import RunUsage

from .sandbox_fakes import (
    AcquireOnlySandboxCapability,
    ConnectOnlySandboxCapability,
    DecliningSandboxCapability,
    FakeSandboxHandle,
    FakeSandboxResult,
    LifecycleSandboxCapability,
)

pytestmark = pytest.mark.anyio


@dataclass(frozen=True)
class _Entry:
    name: str
    path: str
    is_dir: bool
    size: int | None = None


class _Fs:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.reads: list[str] = []

    def _content(self, path: str) -> bytes:
        """Honors the protocol's missing-path contract: `FileNotFoundError`, not `KeyError`."""
        try:
            return self.files[path]
        except KeyError:
            raise FileNotFoundError(path) from None

    async def read_bytes(self, path: str) -> bytes:
        self.reads.append(path)
        return self._content(path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        self.files[path] = data

    async def stat(self, path: str) -> _Entry:
        return _Entry(name=path.rsplit('/', 1)[-1], path=path, is_dir=False, size=len(self._content(path)))

    async def list_dir(self, path: str) -> Sequence[_Entry]:
        return [await self.stat(p) for p in self.files]

    async def make_dir(self, path: str) -> None:
        pass

    async def remove(self, path: str) -> None:
        self._content(path)
        del self.files[path]

    async def exists(self, path: str) -> bool:
        return path in self.files


class _WaitOnlyProcess:
    # Conformance-only test double: never called.
    pid = None

    async def wait(self) -> SandboxResult:
        raise NotImplementedError  # pragma: no cover

    async def kill(self) -> None:
        raise NotImplementedError  # pragma: no cover


class _StreamingProcess(_WaitOnlyProcess):
    def stream(self) -> AsyncIterator[SandboxOutputChunk]:
        raise NotImplementedError  # pragma: no cover


async def test_stream_support_is_separate_from_process_protocol():
    wait_only: SandboxProcess = _WaitOnlyProcess()
    streaming: SupportsStream = _StreamingProcess()
    assert not isinstance(wait_only, SupportsStream)
    assert isinstance(streaming, SupportsStream)


async def test_local_filesystem_rejects_relative_paths(tmp_path: Path):
    from pydantic_ai.sandboxes import LocalSandbox

    sandbox = LocalSandbox(tmp_path)
    with pytest.raises(ValueError, match=r"path must be absolute, got 'outside\.txt'"):
        await sandbox.fs.write_bytes('outside.txt', b'escape')


async def test_sandbox_resolve_rejects_relative_base():
    sandbox = Sandbox(FakeSandbox('resolve-base'))
    with pytest.raises(ValueError, match="base must be an absolute path, got 'relative'"):
        await sandbox.resolve('file.txt', base='relative')


# The facade's bounded slice form: print the window, then quit at its last line.
_SED_WINDOW_EXPR = re.compile(r'^(\d+),(\d+)p;\2q$')


class FakeSandbox:
    """A minimal in-memory implementation of the `SandboxBackend` protocol.

    Honors the protocol's one-environment contract: the `sed` line-window form the
    `Sandbox` facade emits is served from the same files `fs` exposes. `sed=False`
    models an environment without a usable `sed` (exit 127). Other commands echo
    `ran:<command>` for forwarding tests.
    """

    provider = 'fake'

    def __init__(self, name: str, fs: _Fs | None = None, *, sed: bool = True) -> None:
        self.name = name
        self._fs = fs or _Fs()
        self._sed = sed

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
    ) -> FakeSandboxResult:
        if not isinstance(command, str) and list(command[:2]) == ['sed', '-n']:
            if not self._sed:
                return FakeSandboxResult(exit_code=127, stdout='', stderr='sed: not found')
            expr, path = command[2], command[3]
            window = _SED_WINDOW_EXPR.match(expr)
            assert window is not None, f'FakeSandbox only emulates the line-window sed form, got {expr!r}'
            if path not in self._fs.files:
                return FakeSandboxResult(exit_code=2, stdout='', stderr=f'sed: {path}: No such file or directory')
            # `sed` splits on `\n` only (`\r` stays line content), prints selected lines,
            # and preserves the absence of a trailing newline on the file's final line.
            text = self._fs.files[path].decode('utf-8', errors='replace')
            lines = text.split('\n')
            if lines[-1] == '':
                lines.pop()
            start, end = int(window[1]) - 1, int(window[2])
            selected = lines[start:end]
            stdout = '\n'.join(selected)
            if selected and (start + len(selected) < len(lines) or text.endswith('\n')):
                stdout += '\n'
            return FakeSandboxResult(exit_code=0, stdout=stdout, stderr='')
        return FakeSandboxResult(exit_code=0, stdout=f'ran:{command}', stderr='')

    async def start(
        self,
        command: str | Sequence[str],
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        raise NotImplementedError('FakeSandbox cannot start background processes; use `run` instead.')

    async def working_dir(self) -> str:
        return '/workspace'


class _ClosableFakeSandbox(FakeSandbox):
    def __init__(self, name: str, close_calls: list[bool]) -> None:
        super().__init__(name)
        self.close_calls = close_calls

    async def close(self, *, terminate: bool) -> None:
        await asyncio.sleep(0)
        self.close_calls.append(terminate)


def _describe(sandbox: Sandbox) -> str:
    # `sandbox_id` is readable both before and after a deferred facade connects.
    return sandbox.sandbox_id.removeprefix('fake-')


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


async def test_fake_sandbox_facade_surface():
    backend = FakeSandbox('surface')
    sandbox = Sandbox(backend)
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


class _MinimalBackend:
    """A backend that implements only the members `SandboxBackend` requires."""

    provider = 'minimal'
    sandbox_id = 'minimal-1'

    async def run(
        self,
        command: str | Sequence[str],
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> SandboxResult:
        return FakeSandboxResult(exit_code=0, stdout='', stderr='')

    async def working_dir(self) -> str:
        return '/workspace'


async def test_backend_without_supports_filesystem_raises_on_fs_access():
    """A backend that doesn't implement `SupportsFilesystem` fails at `.fs` access with a
    clear pointer, mirroring how `.start()` behaves for backends without `SupportsStart`.
    """
    backend = _MinimalBackend()
    typed: SandboxBackend = backend
    assert not isinstance(typed, SupportsFilesystem)
    assert not isinstance(typed, SupportsStart)

    sandbox = Sandbox(backend)
    # The required members still work: `.run` reaches the backend directly.
    assert (await sandbox.run(['true'])).exit_code == 0
    with pytest.raises(NotImplementedError, match=r'does not implement `SupportsFilesystem`'):
        _ = sandbox.fs
    with pytest.raises(NotImplementedError, match=r'does not implement `SupportsFilesystem`'):
        await sandbox.read_text('anything.txt')


@pytest.mark.parametrize(
    ('content', 'offset', 'limit', 'expected'),
    [
        (b'one\ntwo\nthree', 2, None, (('two', 'three'), False, 3)),
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
    backend = FakeSandbox('slow', sed=False)
    backend.fs.files['/workspace/file'] = content
    window = await Sandbox(backend).read_file('file', offset=offset, limit=limit)
    lines, has_more, total_lines = expected
    assert window.lines == lines
    assert window.text == '\n'.join(lines)
    assert window.start_line == offset
    assert window.has_more is has_more
    assert window.total_lines == total_lines


async def test_bounded_read_falls_back_to_filesystem_without_shell_support():
    backend = FakeSandbox('no-sed', sed=False)
    backend.fs.files['/workspace/file'] = b'one\ntwo\nthree\n'

    window = await Sandbox(backend).read_file('file', offset=2, limit=1)

    assert window.lines == ('two',)
    assert window.has_more is True
    assert window.total_lines == 3
    assert backend.fs.reads == ['/workspace/file']


@pytest.mark.parametrize(('kwargs', 'message'), [({'offset': 0}, 'offset'), ({'limit': 0}, 'limit')])
async def test_read_file_rejects_invalid_windows(kwargs: dict[str, int], message: str):
    with pytest.raises(ValueError, match=message):
        await Sandbox(FakeSandbox('invalid')).read_file('file', **kwargs)


async def test_read_file_slices_inside_the_sandbox_when_no_range_support():
    """Without native range support, the window is produced by `sed` inside the sandbox:
    only the requested lines cross the wire, and the filesystem is never asked for the
    whole file.
    """
    backend = FakeSandbox('sliced')
    backend.fs.files['/workspace/file'] = b'line\n' * 200_000

    window = await Sandbox(backend).read_file('file', offset=99_999, limit=2)

    assert window.lines == ('line', 'line')
    assert window.start_line == 99_999
    assert window.has_more is True
    assert window.total_lines is None
    assert backend.fs.reads == []


async def test_read_file_shell_slice_reports_totals_at_eof():
    backend = FakeSandbox('sliced-eof')
    backend.fs.files['/workspace/file'] = b'one\ntwo\nthree\n'

    window = await Sandbox(backend).read_file('file', offset=2, limit=5)

    assert window.lines == ('two', 'three')
    assert window.has_more is False
    assert window.total_lines == 3
    assert backend.fs.reads == []


async def test_read_file_empty_shell_window_stays_bounded():
    backend = FakeSandbox('sliced-past-eof')
    backend.fs.files['/workspace/file'] = b'one\n'

    window = await Sandbox(backend).read_file('file', offset=10, limit=2)

    assert window.lines == ()
    assert window.start_line == 10
    assert window.has_more is False
    assert window.total_lines is None
    assert backend.fs.reads == []


async def test_read_file_empty_window_without_filesystem_support():
    class EmptyRunOnly(_RunOnlySandbox):
        async def run(
            self,
            command: str | Sequence[str],
            *,
            shell: bool = False,
            cwd: str | None = None,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
        ) -> FakeSandboxResult:
            return FakeSandboxResult(stdout='')

    window = await Sandbox(EmptyRunOnly(FakeSandbox('empty-run-only'))).read_file('/missing', limit=1)

    assert window.lines == ()


async def test_shell_slice_without_trailing_newline():
    backend = FakeSandbox('no-trailing-newline')
    backend.fs.files['/workspace/file'] = b'one'

    window = await Sandbox(backend).read_file('file', limit=2)

    assert window.lines == ('one',)
    assert window.total_lines == 1


class _RunOnlySandbox:
    """The smallest legal backend — the four required members, no `fs`, no `start` —
    delegating to a `FakeSandbox` so its `sed` emulation serves the slice."""

    provider = 'run-only'

    def __init__(self, inner: FakeSandbox) -> None:
        self._inner = inner

    @property
    def sandbox_id(self) -> str:
        return self._inner.sandbox_id

    async def run(
        self,
        command: str | Sequence[str],
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> FakeSandboxResult:
        return await self._inner.run(command, shell=shell, cwd=cwd, env=env, timeout=timeout)

    async def working_dir(self) -> str:
        return await self._inner.working_dir()


async def test_read_file_windowed_works_without_filesystem_support():
    """A backend with only the required members still serves windowed reads through the
    `sed` slice; only the fallback (and `limit=None`) needs `SupportsFilesystem`, so the
    filesystem must not be resolved before the slice is attempted.
    """
    inner = FakeSandbox('run-only')
    inner.fs.files['/workspace/file'] = b'one\ntwo\nthree\n'
    sandbox = Sandbox(_RunOnlySandbox(inner))
    assert sandbox.sandbox_id == 'fake-run-only'  # identity forwards without filesystem support

    window = await sandbox.read_file('file', offset=1, limit=2)
    assert window.lines == ('one', 'two')
    assert window.has_more is True
    assert inner.fs.reads == []

    with pytest.raises(NotImplementedError, match='SupportsFilesystem'):
        await sandbox.read_file('file')


async def test_windowed_read_without_shell_or_filesystem_has_targeted_error():
    inner = FakeSandbox('run-only-no-sed', sed=False)

    with pytest.raises(NotImplementedError, match=r'working `sed`.*SupportsFilesystem'):
        await Sandbox(_RunOnlySandbox(inner)).read_file('file', limit=1)


async def test_shell_slice_is_bounded_and_stops_at_the_window():
    """The slice is an optimization, so it must not cost more than what it optimizes: a
    deadline bounds a wedged path (a FIFO never produces output), and the quit command
    stops `sed` at the window instead of scanning a large file to EOF.
    """
    backend = FakeSandbox('bounded')
    backend.fs.files['/workspace/file'] = b'one\ntwo\nthree\n'
    calls: list[tuple[list[str], float | None]] = []

    class _Recorder(_RunOnlySandbox):
        async def run(
            self,
            command: str | Sequence[str],
            *,
            shell: bool = False,
            cwd: str | None = None,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
        ) -> FakeSandboxResult:
            assert not isinstance(command, str)
            calls.append((list(command), timeout))
            return await super().run(command, shell=shell, cwd=cwd, env=env, timeout=timeout)

    window = await Sandbox(_Recorder(backend)).read_file('file', offset=1, limit=2)
    assert window.lines == ('one', 'two')
    ((argv, timeout),) = calls
    assert argv == ['sed', '-n', '1,3p;3q', '/workspace/file']
    assert timeout is not None


async def test_run_and_start_reject_a_relative_cwd():
    """A relative `cwd` has no sandbox meaning — a backend resolving it against ambient
    host state is exactly the one-environment break the protocol forbids — so the facade
    rejects it before any backend sees it.
    """
    sandbox = Sandbox(FakeSandbox('cwd'))
    with pytest.raises(ValueError, match='absolute'):
        await sandbox.run(['true'], cwd='subdir')
    with pytest.raises(ValueError, match='absolute'):
        await sandbox.start(['true'], cwd='subdir')


async def test_read_file_missing_file_surfaces_filesystem_error_without_reading():
    backend = FakeSandbox('missing')

    with pytest.raises(FileNotFoundError):
        await Sandbox(backend).read_file('nope.txt', limit=3)

    assert backend.fs.reads == []


async def test_read_file_on_unavailable_sandbox_surfaces_reason():
    sandbox = Sandbox(UnavailableSandbox('sandbox disabled by policy'))
    with pytest.raises(UserError, match='sandbox disabled by policy'):
        await sandbox.read_file('/file', limit=3)


async def test_bare_run_context_sandbox_is_unavailable():
    """A `RunContext` not backed by a run grants no execution: sandboxes are attached by run
    assembly only, so synthetic contexts (e.g. in user test suites) can't silently execute
    on the host.
    """
    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())
    assert ctx.sandbox.provider == 'unavailable'
    assert ctx.sandbox._is_framework_default()  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(UserError, match='created outside an agent run'):
        await ctx.sandbox.run(['echo', 'hello'])

    backend = ctx.sandbox.backend
    assert isinstance(backend, UnavailableSandbox)
    explicit = Sandbox(UnavailableSandbox(backend.reason))
    assert not explicit._is_framework_default()  # pyright: ignore[reportPrivateUsage]


async def test_unavailable_sandbox_surfaces_reason_for_every_operation():
    reason = 'sandbox disabled by policy'
    backend = UnavailableSandbox(reason)
    assert isinstance(backend, SandboxBackend)
    assert isinstance(backend, SupportsStart)
    typed_backend: SandboxBackend = backend
    typed_start: SupportsStart = backend
    assert (typed_backend.provider, typed_backend.sandbox_id) == ('unavailable', 'unavailable')
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


async def test_run_argument_backend_is_exposed_through_facade():
    observed: list[Sandbox] = []
    backend = FakeSandbox('direct')
    result = await make_identity_probe_agent(observed).run('go', sandbox=backend)
    assert result.output == 'done'
    assert len(observed) == 1
    assert isinstance(observed[0], Sandbox)
    assert observed[0].backend is backend


async def test_existing_facade_passes_through_run_unchanged():
    observed: list[Sandbox] = []
    sandbox = Sandbox(FakeSandbox('rich'))
    await make_identity_probe_agent(observed).run('go', sandbox=sandbox)
    assert observed == [sandbox]


async def test_run_without_sandbox_is_unavailable_with_attachment_instructions():
    """No run ever gets implicit access to the host: without an explicit sandbox, every
    operation raises with the ways to attach one.
    """
    agent: Agent = Agent(_tool_call_then_text('use_default'))
    providers: list[str] = []

    @agent.tool
    async def use_default(ctx: RunContext[Any]) -> str:
        providers.append(ctx.sandbox.provider)
        await ctx.sandbox.run(['echo', 'hello'])
        return 'ok'  # pragma: no cover

    with pytest.raises(UserError, match=r'No sandbox is attached to this run.+`sandbox=LocalSandbox\(\)`'):
        await agent.run('go')
    assert providers == ['unavailable']


async def test_unavailable_sandbox_run_argument_overrides_default_reason():
    reason = 'local execution disabled'
    agent: Agent = Agent(_tool_call_then_text())

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        return (await ctx.sandbox.run(['echo', 'hello'])).stdout

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
    """Canonical supplier: a fresh sandbox per run through the three lifecycle hooks."""

    name: str = 'cap'
    events: list[str] = field(default_factory=lambda: [])
    backend: FakeSandbox | None = field(default=None, init=False)

    async def acquire_sandbox(self, ctx: RunContext[Any]) -> SandboxRef:
        self.events.append(f'{self.name}:acquire')
        self.backend = FakeSandbox(self.name)
        return SandboxRef(provider='fake', sandbox_id=self.backend.sandbox_id)

    async def get_sandbox(self, ctx: RunContext[Any], ref: SandboxRef | None) -> SandboxBackend | None:
        if ref is None or ref.provider != 'fake' or self.backend is None or ref.sandbox_id != self.backend.sandbox_id:
            return None
        self.events.append(f'{self.name}:connect')
        return self.backend

    async def release_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> None:
        self.events.append(f'{self.name}:release')


async def test_sandbox_ref_connects_once_and_exposes_identity_before_connection():
    connector = ConnectOnlySandboxCapability()
    observed: list[Sandbox] = []
    agent: Agent = Agent(_tool_call_then_text(), capabilities=[connector])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        sandbox = ctx.sandbox
        observed.append(sandbox)
        assert (sandbox.provider, sandbox.sandbox_id) == ('fake', 'fake-deferred')
        with pytest.raises(UserError, match='has not connected yet'):
            sandbox.backend
        deferred_fs = sandbox.fs
        assert sandbox.fs is deferred_fs  # the pre-connection adapter is cached

        run_results = await asyncio.gather(
            sandbox.run(['one']),
            sandbox.run(['two']),
            sandbox.working_dir(),
        )
        assert [result.stdout for result in run_results[:2]] == ['connected', 'connected']
        assert run_results[2] == '/workspace'
        assert sandbox.backend is connector.backends[0]
        return 'ok'

    result = await agent.run('go', sandbox=SandboxRef(provider='fake', sandbox_id='fake-deferred'))
    assert result.output == 'done'
    assert len(observed) == 1
    assert connector.sandbox_ids == ['fake-deferred']  # concurrent first ops connect exactly once


@pytest.mark.parametrize('sandbox_arg', [None, SandboxRef(provider='fake', sandbox_id='fake-connected')])
async def test_capability_connection_is_detached_without_terminating_sandbox(sandbox_arg: SandboxRef | None):
    close_calls: list[bool] = []

    class Connector(AbstractCapability[Any]):
        async def get_sandbox(self, ctx: RunContext[Any], ref: SandboxRef | None) -> SandboxBackend | None:
            return _ClosableFakeSandbox('connected', close_calls)

    await make_connecting_probe_agent([], capabilities=[Connector()]).run('go', sandbox=sandbox_arg)
    assert close_calls == [False]


async def test_caller_owned_backend_is_not_closed_by_run():
    close_calls: list[bool] = []
    backend = _ClosableFakeSandbox('direct', close_calls)
    facade = Sandbox(backend)

    await make_probe_agent([], capabilities=[]).run('go', sandbox=facade)
    await facade._close_connected_backend()  # pyright: ignore[reportPrivateUsage]
    assert close_calls == []


async def test_deferred_filesystem_proxy_serves_every_operation():
    """A file operation as the FIRST act on a deferred facade must connect and work, and an
    `fs` handle obtained before connection stays valid afterwards — tools may capture one and
    use it across the connect boundary. Unit-level via `from_ref` because the durable suites
    always run a command before touching files, leaving the pre-connection proxy unexercised.
    """
    backend = FakeSandbox('proxy')
    connected: list[str] = []

    async def resolver(ref: SandboxRef) -> SandboxBackend:
        connected.append(ref.sandbox_id)
        return backend

    sandbox = Sandbox._from_ref(  # pyright: ignore[reportPrivateUsage]
        SandboxRef(provider='fake', sandbox_id='fake-proxy'), resolver
    )
    fs = sandbox.fs  # obtained before any connection: the deferred proxy
    assert connected == []

    await fs.write_bytes('/workspace/notes.txt', b'hello')
    assert connected == ['fake-proxy']  # the first operation connected
    assert await fs.read_bytes('/workspace/notes.txt') == b'hello'
    assert (await fs.stat('/workspace/notes.txt')).size == 5
    assert [entry.path for entry in await fs.list_dir('/workspace')] == ['/workspace/notes.txt']
    await fs.make_dir('/workspace/sub')
    assert await fs.exists('/workspace/notes.txt')
    await fs.remove('/workspace/notes.txt')
    assert not await fs.exists('/workspace/notes.txt')
    assert connected == ['fake-proxy']  # connected exactly once across all operations


async def test_ref_resolution_skips_deferred_capabilities():
    """A deferred capability's contributions are inert until it loads, so ref resolution walks
    past it to a loaded capability that recognizes the provider."""
    deferred = SandboxCapability(id='deferred-sandbox', defer_loading=True)
    connector = ConnectOnlySandboxCapability()
    seen: list[str] = []
    # The connector sits earlier in the chain, so resolution (which walks latest-first)
    # reaches the deferred capability first and must skip, not consult, it.
    agent = make_connecting_probe_agent(seen, capabilities=[connector, deferred])
    await agent.run('go', sandbox=SandboxRef(provider='fake', sandbox_id='fake-1'))
    assert seen == ['1']
    assert connector.sandbox_ids == ['fake-1']


@pytest.mark.parametrize('provider_kind', ['none', 'connect-only', 'lifecycle', 'stateful'])
async def test_sandbox_ref_requires_recognizing_capability(provider_kind: str):
    """No provider, or any provider declining a foreign ref, produces the attachment error."""
    provider = {
        'none': None,
        'connect-only': ConnectOnlySandboxCapability(),
        'lifecycle': AcquireOnlySandboxCapability(),
        'stateful': SandboxCapability(),
    }[provider_kind]
    capabilities = [provider] if provider is not None else []
    agent = make_connecting_probe_agent([], capabilities=capabilities)
    with pytest.raises(
        UserError,
        match=r"No capability recognizes the sandbox reference for provider 'missing'",
    ):
        await agent.run('go', sandbox=SandboxRef(provider='missing', sandbox_id='sandbox-1'))


async def test_sandbox_ref_connection_failure_is_chained():
    error = RuntimeError('expired')

    @dataclass
    class FailingConnectCapability(AbstractCapability[Any]):
        async def get_sandbox(self, ctx: RunContext[Any], ref: SandboxRef | None) -> SandboxBackend | None:
            await asyncio.sleep(0)
            raise error

    agent = make_connecting_probe_agent([], capabilities=[FailingConnectCapability()])
    with pytest.raises(
        UserError,
        match=re.escape("Failed to connect to sandbox 'expired' for provider 'fake'.") + '$',
    ) as exc_info:
        await agent.run('go', sandbox=SandboxRef(provider='fake', sandbox_id='expired'))
    assert exc_info.value.__cause__ is error


async def test_sandbox_ref_user_error_is_preserved():
    class UserErrorConnector(AbstractCapability[Any]):
        async def get_sandbox(self, ctx: RunContext[Any], ref: SandboxRef | None) -> SandboxBackend | None:
            raise UserError('credentials are missing')

    agent = Agent(_tool_call_then_text(), capabilities=[UserErrorConnector()])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        await ctx.sandbox.run(['true'])
        return 'ok'  # pragma: no cover

    with pytest.raises(UserError, match='credentials are missing'):
        await agent.run('go', sandbox=SandboxRef(provider='fake', sandbox_id='broken'))


async def test_sandbox_ref_wins_over_sandbox_supplier():
    class SupplierAndConnector(ConnectOnlySandboxCapability):
        async def acquire_sandbox(self, ctx: RunContext[Any]) -> SandboxRef:
            raise AssertionError('an explicit ref must skip acquisition')  # pragma: no cover

    supplier = SupplierAndConnector()
    seen: list[str] = []
    agent = make_connecting_probe_agent(seen, capabilities=[supplier])
    await agent.run('go', sandbox=SandboxRef(provider='fake', sandbox_id='fake-winner'))
    assert seen == ['winner']
    assert supplier.sandbox_ids == ['fake-winner']


async def test_multiple_ref_connectors_are_rejected_before_connection():
    first = ConnectOnlySandboxCapability()
    last = ConnectOnlySandboxCapability()
    agent = make_connecting_probe_agent([], capabilities=[first, WrapperCapability(wrapped=last)])
    with pytest.raises(UserError, match=r'Exactly one capability may provide sandbox hooks; found 2:'):
        await agent.run('go', sandbox=SandboxRef(provider='fake', sandbox_id='fake-1'))
    assert first.sandbox_ids == []
    assert last.sandbox_ids == []


async def test_sandbox_ref_capability_id_routes_to_exact_connector():
    other = ConnectOnlySandboxCapability()
    other.id = 'other'
    connector = ConnectOnlySandboxCapability()
    connector.id = 'connector'
    seen: list[str] = []
    agent = make_connecting_probe_agent(seen, capabilities=[other, connector])
    await agent.run('go', sandbox=SandboxRef(provider='fake', sandbox_id='fake-1', capability_id='connector'))
    assert seen == ['1']
    assert other.sandbox_ids == []
    assert connector.sandbox_ids == ['fake-1']


async def test_sandbox_ref_capability_id_must_be_available():
    agent = make_connecting_probe_agent([], capabilities=[ConnectOnlySandboxCapability()])
    with pytest.raises(
        UserError,
        match=r"Cannot reconnect sandbox 'fake-1': expected one capability with id 'missing', found 0\.",
    ):
        await agent.run('go', sandbox=SandboxRef(provider='fake', sandbox_id='fake-1', capability_id='missing'))


async def test_sandbox_ref_capability_id_cannot_activate_deferred_connector():
    connector = ConnectOnlySandboxCapability()
    connector.id = 'deferred'
    connector.defer_loading = True
    agent = make_connecting_probe_agent([], capabilities=[connector])
    with pytest.raises(UserError, match=r'deferred capabilities cannot provide the run sandbox\.'):
        await agent.run('go', sandbox=SandboxRef(provider='fake', sandbox_id='fake-1', capability_id='deferred'))


def test_contributes_sandbox_detection():
    assert contributes_sandbox(ConnectOnlySandboxCapability()) is True
    assert contributes_sandbox(WrapperCapability(wrapped=SandboxCapability())) is True
    assert contributes_sandbox(SandboxCapability(id='deferred-sandbox', defer_loading=True)) is False
    assert contributes_sandbox(CombinedCapability([SandboxCapability(), ConnectOnlySandboxCapability()])) is True


def test_contributes_sandbox_handles_a_capability_reachable_twice():
    supplier = SandboxCapability()
    tree = CombinedCapability([supplier, WrapperCapability(wrapped=supplier)])
    assert contributes_sandbox(tree)


async def test_lifecycle_capability_creates_at_run_start_and_tears_down_at_run_end():
    """A capability owning the lifecycle gives the run the whole bracket: setup before the
    acquisition first, release after the run. The tool reaches the sandbox by reconnecting
    through `get_sandbox`, exactly as it would inside a durable engine's activity.
    """
    lifecycle = LifecycleSandboxCapability()
    agent: Agent = Agent(_tool_call_then_text(), capabilities=[lifecycle])
    seen: list[str] = []

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        seen.append(ctx.sandbox.sandbox_id)
        # Created but not yet torn down while the run is still going.
        assert lifecycle.events == ['acquire:created-1']
        await ctx.sandbox.run(['echo', 'hello'])
        assert lifecycle.events == ['acquire:created-1', 'connect:created-1']
        return 'ok'

    result = await agent.run('go')
    assert result.output == 'done'
    assert seen == ['created-1']
    assert lifecycle.events == ['acquire:created-1', 'connect:created-1', 'release:created-1']
    assert [backend.commands for backend in lifecycle.backends] == [[['echo', 'hello']]]


async def test_created_sandbox_ref_is_stamped_with_supplier_id():
    lifecycle = LifecycleSandboxCapability()
    observed: list[SandboxRef] = []
    agent: Agent = Agent(_tool_call_then_text(), capabilities=[lifecycle])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        identity = ctx.sandbox._durable_identity()  # pyright: ignore[reportPrivateUsage]
        assert isinstance(identity, SandboxRef)
        observed.append(identity)
        return 'ok'

    await agent.run('go')
    assert observed == [SandboxRef(provider='fake', sandbox_id='created-1', capability_id='test-sandbox')]


async def test_created_sandbox_ref_is_stamped_with_derived_supplier_id():
    supplier = SandboxCapability()
    observed: list[SandboxRef] = []
    agent: Agent = Agent(_tool_call_then_text(), capabilities=[supplier])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        identity = ctx.sandbox._durable_identity()  # pyright: ignore[reportPrivateUsage]
        assert isinstance(identity, SandboxRef)
        observed.append(identity)
        return 'ok'

    await agent.run('go')
    assert observed == [SandboxRef(provider='fake', sandbox_id='fake-cap', capability_id='sandbox_capability')]


async def test_lifecycle_capability_tears_down_when_a_tool_raises():
    lifecycle = LifecycleSandboxCapability()

    def model_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('explode', {})])

    agent: Agent = Agent(FunctionModel(model_func), capabilities=[lifecycle])

    @agent.tool
    async def explode(ctx: RunContext[Any]) -> str:
        raise RuntimeError('boom')

    with pytest.raises(RuntimeError, match='boom'):
        await agent.run('go')
    assert lifecycle.events == ['acquire:created-1', 'release:created-1']


async def test_lifecycle_capability_tears_down_when_run_preparation_fails():
    lifecycle = LifecycleSandboxCapability()
    agent = Agent(TestModel(), capabilities=[lifecycle])

    def failing_metadata(ctx: RunContext[Any]) -> dict[str, Any]:
        raise RuntimeError('metadata preparation failed')

    with pytest.raises(RuntimeError, match='metadata preparation failed'):
        await agent.run('go', metadata=failing_metadata)
    assert lifecycle.events == ['acquire:created-1', 'release:created-1']


async def test_lifecycle_capability_tears_down_when_capability_resolution_fails():
    lifecycle = LifecycleSandboxCapability()

    @dataclass
    class FailingForRun(AbstractCapability[Any]):
        async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
            raise RuntimeError('capability resolution failed')

    agent = Agent(TestModel(), capabilities=[lifecycle, FailingForRun()])
    with pytest.raises(RuntimeError, match='capability resolution failed'):
        await agent.run('go')
    assert lifecycle.events == ['acquire:created-1', 'release:created-1']


async def test_failing_sandbox_acquisition_exits_whole_run_context():
    events: list[str] = []

    @dataclass
    class Bracket(AbstractCapability[Any]):
        @asynccontextmanager
        async def wrap_entire_run(self, ctx: Any) -> AsyncGenerator[None]:
            events.append('enter')
            try:
                yield
            finally:
                events.append('exit')

    @dataclass
    class FailingAcquirer(AbstractCapability[Any]):
        async def acquire_sandbox(self, ctx: RunContext[Any]) -> SandboxRef:
            raise RuntimeError('acquisition failed')

    with pytest.raises(RuntimeError, match='acquisition failed'):
        await Agent(TestModel(), capabilities=[Bracket(), FailingAcquirer()]).run('go')
    assert events == ['enter', 'exit']


async def test_pre_sandbox_validation_failure_exits_whole_run_context():
    events: list[str] = []

    @dataclass
    class Bracket(AbstractCapability[Any]):
        @asynccontextmanager
        async def wrap_entire_run(self, ctx: Any) -> AsyncGenerator[None]:
            events.append('enter')
            try:
                yield
            finally:
                events.append('exit')

    agent = Agent(TestModel(), capabilities=[Bracket()])
    with pytest.raises(UserError, match=r"`tool_choice='required'` prevents"):
        await agent.run('go', model_settings={'tool_choice': 'required'})
    assert events == ['enter', 'exit']


async def test_duplicate_per_run_supplier_ids_fail_before_acquisition():
    creator = LifecycleSandboxCapability()
    creator.id = 'duplicate'
    decliner = DecliningSandboxCapability()
    decliner.id = 'duplicate'
    agent = Agent(TestModel(), capabilities=[creator])

    with pytest.raises(UserError, match=r"Capability id 'duplicate' is used by multiple capabilities\."):
        await agent.run('go', capabilities=[decliner])
    assert creator.events == []
    assert decliner.acquire_calls == 0


async def test_acquire_only_capability_leans_on_platform_reaping():
    """The inherited no-op `release_sandbox` is what lets a capability lean on its
    platform's idle timeout instead of destroying anything itself.
    """
    creator = AcquireOnlySandboxCapability()
    seen: list[str] = []
    agent = make_probe_agent(seen, capabilities=[creator])
    result = await agent.run('go')
    assert result.output == 'done'
    assert seen == ['created-1']
    assert creator.events == ['acquire:created-1']


async def test_get_only_capability_supplies_an_already_live_sandbox_without_a_ref():
    backend = FakeSandbox('always-live')
    refs: list[SandboxRef | None] = []

    @dataclass
    class LiveSandboxCapability(AbstractCapability[Any]):
        async def get_sandbox(self, ctx: RunContext[Any], ref: SandboxRef | None) -> SandboxBackend | None:
            refs.append(ref)
            return backend if ref is None else None

    seen: list[tuple[str, str, SandboxBackend]] = []
    agent: Agent = Agent(_tool_call_then_text(), capabilities=[LiveSandboxCapability()])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        for attribute in ('provider', 'sandbox_id', 'backend'):
            with pytest.raises(UserError, match='has not connected yet'):
                getattr(ctx.sandbox, attribute)
        await ctx.sandbox.run(['true'])
        seen.append((ctx.sandbox.provider, ctx.sandbox.sandbox_id, ctx.sandbox.backend))
        return 'ok'

    await agent.run('go')
    assert seen == [('fake', 'fake-always-live', backend)]
    assert refs == [None]


@pytest.mark.parametrize('failure', ['raise', 'return-none'])
async def test_get_only_capability_connection_failure_is_explained(failure: str):
    error = RuntimeError('connection failed')

    @dataclass
    class FailingLiveSandboxCapability(AbstractCapability[Any]):
        async def get_sandbox(self, ctx: RunContext[Any], ref: SandboxRef | None) -> SandboxBackend | None:
            assert ref is None
            if failure == 'raise':
                raise error
            return None

    capability = FailingLiveSandboxCapability()
    capability.id = 'failing-live-sandbox'
    agent: Agent = Agent(_tool_call_then_text(), capabilities=[capability])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        return (await ctx.sandbox.run(['true'])).stdout

    expected = (
        "Capability 'failing-live-sandbox' failed to connect its sandbox."
        if failure == 'raise'
        else "Capability 'failing-live-sandbox' provides sandbox hooks but its `get_sandbox` hook returned "
        '`None` without a `SandboxRef`.'
    )
    with pytest.raises(UserError, match=re.escape(expected)) as exc_info:
        await agent.run('go')
    assert exc_info.value.__cause__ is (error if failure == 'raise' else None)


async def test_declining_sandbox_capability_leaves_the_default_unavailable():
    decliner = DecliningSandboxCapability()
    result = await Agent(TestModel(), capabilities=[decliner]).run('go')
    assert result.output == 'success (no tool calls)'
    assert decliner.acquire_calls == 1


async def test_base_and_combined_get_sandbox_route_or_decline_cleanly():
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    base = AbstractCapability[Any]()
    assert await base.get_sandbox(ctx, None) is None

    deferred = ConnectOnlySandboxCapability()
    deferred.id = 'deferred'
    deferred.defer_loading = True
    declining = AbstractCapability[Any]()
    connector = ConnectOnlySandboxCapability()
    connector.id = 'connector'
    combined = CombinedCapability([deferred, declining, connector])
    assert combined._has_get_sandbox  # pyright: ignore[reportPrivateUsage]
    backend = await combined.get_sandbox(ctx, SandboxRef(provider='fake', sandbox_id='combined'))
    assert backend is not None and backend.sandbox_id == 'combined'

    assert await CombinedCapability([deferred, declining]).get_sandbox(ctx, None) is None


async def test_connect_provider_requires_exact_active_capability():
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    connector = ConnectOnlySandboxCapability()
    connector.id = 'connector'

    with pytest.raises(UserError, match='found 0'):
        await connect_sandbox_provider(connector, ctx, 'missing')

    connector.defer_loading = True
    with pytest.raises(UserError, match='deferred capabilities cannot provide'):
        await connect_sandbox_provider(connector, ctx, 'connector')


def test_exact_ref_connector_must_implement_get_sandbox():
    plain = AbstractCapability[Any]()
    plain.id = 'plain'

    with pytest.raises(UserError, match='does not implement `get_sandbox`'):
        find_sandbox_ref_connector(
            plain,
            SandboxRef(provider='fake', sandbox_id='sandbox', capability_id='plain'),
        )


async def test_sandbox_test_fakes_reset_state():
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    assert FakeSandboxHandle('identity').sandbox_id == 'identity'

    connector = ConnectOnlySandboxCapability()
    await connector.get_sandbox(ctx, SandboxRef(provider='fake', sandbox_id='one'))
    connector.reset()
    assert connector.sandbox_ids == [] and connector.backends == []

    acquirer = AcquireOnlySandboxCapability()
    await acquirer.acquire_sandbox(ctx)
    acquirer.reset()
    assert acquirer.events == [] and acquirer.backends == [] and acquirer._created == 0  # pyright: ignore[reportPrivateUsage]

    decliner = DecliningSandboxCapability()
    decliner.acquire_calls = 1
    decliner.reset()
    assert decliner.acquire_calls == 0


def test_durable_workflow_sandbox_guard():
    with pytest.raises(UserError, match='contribution'):
        guard_workflow_sandbox(
            None,
            [AcquireOnlySandboxCapability()],
            static_contributes_sandbox=False,
            contribution_error='contribution blocked',
            live_error='live blocked',
        )
    with pytest.raises(UserError, match='live'):
        guard_workflow_sandbox(
            FakeSandbox('live'),
            None,
            static_contributes_sandbox=False,
            contribution_error='contribution blocked',
            live_error='live blocked',
        )
    ref = SandboxRef(provider='fake', sandbox_id='ref')
    assert (
        guard_workflow_sandbox(
            ref,
            None,
            static_contributes_sandbox=False,
            contribution_error='contribution blocked',
            live_error='live blocked',
        )
        is ref
    )


async def test_lifecycle_capability_also_connects_ref_run_arguments():
    """The same capability serves both jobs: with a ref run argument its `acquire_sandbox` is
    skipped (the caller owns the lifecycle), but its `get_sandbox` still connects.
    """
    lifecycle = LifecycleSandboxCapability()
    seen: list[str] = []
    agent = make_connecting_probe_agent(seen, capabilities=[lifecycle])
    await agent.run('go', sandbox=SandboxRef(provider='fake', sandbox_id='pre-existing'))
    assert seen == ['pre-existing']
    # The run argument wins outright: nothing was created, and the run does not destroy a
    # sandbox it was merely lent.
    assert lifecycle.events == ['connect:pre-existing']


async def test_capability_supplied_sandbox_is_exposed_through_facade():
    cap = SandboxCapability()
    observed: list[Sandbox] = []
    agent: Agent = Agent(_tool_call_then_text(), capabilities=[cap])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        observed.append(ctx.sandbox)
        await ctx.sandbox.run(['true'])  # first operation connects the deferred facade
        return 'ok'

    result = await agent.run('go')
    assert result.output == 'done'
    assert len(observed) == 1
    assert isinstance(observed[0], Sandbox)
    assert observed[0].backend is cap.backend
    assert cap.events == ['cap:acquire', 'cap:connect', 'cap:release']


async def test_capability_supplied_sandbox_connects_lazily():
    """A run that never touches `ctx.sandbox` pays acquisition and release, but no connection."""
    cap = SandboxCapability()
    await make_probe_agent([], capabilities=[cap]).run('go')
    assert cap.events == ['cap:acquire', 'cap:release']


async def test_capability_sandbox_live_through_after_run():
    """Teardown runs after `after_run`, so hooks never see a torn-down sandbox."""

    @dataclass
    class ContributingWatcher(SandboxCapability):
        async def after_run(self, ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
            self.events.append(f'after_run:{_describe(ctx.sandbox)}')
            return result

    cap = ContributingWatcher()
    await make_probe_agent([], capabilities=[cap]).run('go')
    assert cap.events == ['cap:acquire', 'after_run:cap', 'cap:release']


async def test_run_argument_wins_over_capability():
    first = SandboxCapability(name='first-loser')
    second = SandboxCapability(name='second-loser')
    seen: list[str] = []
    agent = make_probe_agent(seen, capabilities=[first, second])
    await agent.run('go', sandbox=FakeSandbox('direct'))
    assert seen == ['direct']
    assert first.events == []
    assert second.events == []


async def test_multiple_sandbox_providers_fail_before_acquisition():
    first = SandboxCapability(name='first')
    last = SandboxCapability(name='last')
    with pytest.raises(UserError, match=r'Exactly one capability may provide sandbox hooks; found 2:'):
        await make_probe_agent([], capabilities=[first, last]).run('go')
    assert first.events == []
    assert last.events == []


async def test_capability_without_sandbox_does_not_mask_supplier():
    @dataclass
    class NonSupplier(AbstractCapability[Any]):
        pass

    contributor = SandboxCapability(name='contributor')
    seen: list[str] = []
    await make_probe_agent(seen, capabilities=[contributor, NonSupplier()]).run('go')
    assert seen == ['contributor']


async def test_warm_sandbox_shared_across_runs():
    """A warm capability returns the identity of the backend it already holds from
    `acquire_sandbox` and leaves `release_sandbox` alone, so the same environment serves
    every run.
    """
    warm = FakeSandbox('warm')

    @dataclass
    class WarmSandboxCapability(AbstractCapability[Any]):
        async def acquire_sandbox(self, ctx: RunContext[Any]) -> SandboxRef:
            return SandboxRef(provider='fake', sandbox_id=warm.sandbox_id)

        async def get_sandbox(self, ctx: RunContext[Any], ref: SandboxRef | None) -> SandboxBackend | None:
            return warm if ref is not None and ref.sandbox_id == warm.sandbox_id else None

    observed: list[Sandbox] = []
    agent: Agent = Agent(_tool_call_then_text(), capabilities=[WarmSandboxCapability()])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        await ctx.sandbox.run(['true'])
        observed.append(ctx.sandbox)
        return 'ok'

    await agent.run('one')
    await agent.run('two')
    assert len(observed) == 2
    assert all(isinstance(sandbox, Sandbox) and sandbox.backend is warm for sandbox in observed)


async def test_deferred_capability_never_contributes():
    cap = SandboxCapability(name='deferred', id='deferred-sandbox', defer_loading=True)
    seen: list[str] = []
    await make_probe_agent(seen, capabilities=[cap]).run('go')
    assert seen == ['unavailable']
    assert cap.events == []


async def test_wrapper_capability_forwards_sandbox_lifecycle():
    inner = SandboxCapability(name='inner')
    seen: list[str] = []
    await make_connecting_probe_agent(seen, capabilities=[WrapperCapability(wrapped=inner)]).run('go')
    assert seen == ['inner']
    assert inner.events == ['inner:acquire', 'inner:connect', 'inner:release']


async def test_capability_sandbox_tears_down_when_toolset_entry_fails():
    """The exit stack owns the bracket, so release runs even when the run never starts."""

    class ExplodingToolset(WrapperToolset[Any]):
        async def __aenter__(self) -> Any:
            raise RuntimeError('toolset entry failed')

    cap = SandboxCapability()
    agent: Agent = Agent(TestModel(), toolsets=[ExplodingToolset(wrapped=FunctionToolset())], capabilities=[cap])
    with pytest.raises(RuntimeError, match='toolset entry failed'):
        await agent.run('go')
    assert cap.events == ['cap:acquire', 'cap:release']


async def test_failing_release_propagates():
    """An in-process release failure surfaces to the caller, exactly like a toolset exit
    error; durable engines make their own call (Temporal logs instead, because the platform
    idle timeout is the backstop and the run's work is already done).
    """
    from .sandbox_fakes import FailingReleaseSandboxCapability

    failing = FailingReleaseSandboxCapability()
    agent = make_probe_agent([], capabilities=[failing])
    with pytest.raises(RuntimeError, match="sandbox 'created-1' is already gone"):
        await agent.run('go')
    assert failing.events == ['acquire:created-1', 'release-failed:created-1']


async def test_release_survives_run_cancellation():
    """Cancelling the run must not abort `release_sandbox`: the exit stack unwinds inside an
    already-cancelled scope, so an unshielded release would die at its first await and leak
    the sandbox.
    """

    class AwaitingTeardownCapability(SandboxCapability):
        async def acquire_sandbox(self, ctx: RunContext[Any]) -> SandboxRef:
            self.events.append(f'{self.name}:acquire')
            self.backend = _ClosableFakeSandbox(self.name, close_calls)
            return SandboxRef(provider='fake', sandbox_id=self.backend.sandbox_id)

        async def release_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> None:
            await asyncio.sleep(0)  # a real release awaits provider I/O; an unshielded cancel lands here
            assert close_calls == [False]
            await super().release_sandbox(ctx, ref)

    close_calls: list[bool] = []
    capability = AwaitingTeardownCapability()
    agent: Agent = Agent(_tool_call_then_text(), capabilities=[capability])
    entered = anyio.Event()

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        await ctx.sandbox.run(['true'])
        entered.set()
        await asyncio.sleep(10)
        return 'never'  # pragma: no cover

    # An anyio cancel scope is level-triggered: once cancelled, every await during the
    # unwind re-raises, unlike a one-shot `task.cancel()`.
    async with anyio.create_task_group() as tg:

        async def run_agent() -> None:
            await agent.run('go')

        tg.start_soon(run_agent)
        await entered.wait()
        tg.cancel_scope.cancel()
    assert capability.events == ['cap:acquire', 'cap:connect', 'cap:release']
    assert close_calls == [False]


async def test_setup_declined_falls_through_to_default():
    """A supplier that returns `None` contributes nothing, and the run falls back to the
    unavailable default rather than treating the decline as an error.
    """

    @dataclass
    class DecliningSupplier(AbstractCapability[Any]):
        async def acquire_sandbox(self, ctx: RunContext[Any]) -> SandboxRef | None:
            return None

    seen: list[str] = []
    await make_probe_agent(seen, capabilities=[DecliningSupplier()]).run('go')
    assert seen == ['unavailable']


def test_managed_and_provider_concepts_are_gone():
    """The lifecycle lives on capability hooks; the adapter types must not resurface."""
    import pydantic_ai.sandboxes as sandboxes

    assert not hasattr(sandboxes, 'ManagedSandbox')
    assert not hasattr(sandboxes, 'SandboxProvider')
    assert 'SandboxRef' in sandboxes.__all__
