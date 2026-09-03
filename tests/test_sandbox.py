"""Tests for sandbox backends, the rich facade, and read-only `RunContext.sandbox` propagation."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import anyio
import pytest

from pydantic_ai import Agent, RunContext, UserError
from pydantic_ai.agent import WrapperAgent
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, WrapperCapability, durable_operation
from pydantic_ai.durable_exec._sandbox import contributes_sandbox, guard_workflow_sandbox
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.sandboxes import (
    Sandbox,
    SandboxBackend,
    SandboxError,
    SandboxRef,
    SandboxResult,
    SandboxTimeoutError,
    SandboxUnavailableError,
    SupportsFilesystem,
    UnavailableSandbox,
)
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset, WrapperToolset
from pydantic_ai.usage import RunUsage
from pydantic_graph import End

from .sandbox_fakes import (
    AcquireOnlySandboxCapability,
    ConnectOnlySandboxCapability,
    DecliningSandboxCapability,
    FailingReleaseSandboxCapability,
    FakeSandbox,
    FakeSandboxResult,
    LifecycleSandboxCapability,
)

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize(
    ('path', 'base', 'expected'),
    [
        ('data.bin', None, '/workspace/data.bin'),
        ('sub/../notes.txt', None, '/workspace/notes.txt'),
        ('/abs/./x', None, '/abs/x'),
        ('x', '/elsewhere', '/elsewhere/x'),
    ],
)
async def test_resolve_normalizes_against_the_working_directory(path: str, base: str | None, expected: str):
    sandbox = Sandbox(FakeSandbox('resolve'))
    assert await sandbox.resolve(path, base=base) == expected


async def test_sandbox_resolve_rejects_relative_base():
    sandbox = Sandbox(FakeSandbox('resolve-base'))
    with pytest.raises(ValueError, match="base must be an absolute path, got 'relative'"):
        await sandbox.resolve('file.txt', base='relative')


async def test_working_dir_is_cached():
    class CountingBackend(FakeSandbox):
        calls = 0

        async def working_dir(self) -> str:
            self.calls += 1
            return await super().working_dir()

    backend = CountingBackend('working-dir')
    sandbox = Sandbox(backend)

    assert await sandbox.working_dir() == '/workspace'
    assert await sandbox.resolve('file.txt') == '/workspace/file.txt'
    assert backend.calls == 1


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


def make_connecting_probe_agent(seen: list[str], **kwargs: Any) -> Agent:
    agent: Agent = Agent(_tool_call_then_text(), **kwargs)

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        await ctx.sandbox.run(['true'])
        seen.append(_describe(ctx.sandbox))
        return 'ok'

    return agent


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


async def test_read_text_is_strict_about_encoding():
    backend = FakeSandbox('strict', {'/workspace/bad': b'one\xff'})
    with pytest.raises(UnicodeDecodeError):
        await Sandbox(backend).read_text('bad')


class _MinimalBackend:
    """A backend that implements only the members `SandboxBackend` requires."""

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
    """A backend that doesn't implement `SupportsFilesystem` fails at `.fs` access with a clear pointer."""
    backend = _MinimalBackend()
    typed: SandboxBackend = backend
    assert not isinstance(typed, SupportsFilesystem)

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
async def test_full_read_decodes_and_splits_lines(
    content: bytes,
    offset: int,
    limit: int | None,
    expected: tuple[tuple[str, ...], bool, int],
):
    backend = FakeSandbox('full')
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


async def test_bounded_read_slices_inside_the_sandbox():
    """Only the requested lines cross the wire; the filesystem is never asked for the whole file."""
    backend = FakeSandbox('sliced', {'/workspace/file': b'line\n' * 5})

    window = await Sandbox(backend).read_file('file', offset=2, limit=2)

    assert window.lines == ('line', 'line')
    assert window.start_line == 2
    assert window.has_more is True
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


class _RunOnlySandbox:
    """The smallest legal backend — the three required members and no `fs` —
    delegating to a `FakeSandbox` so its `sed` emulation serves the slice."""

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


async def test_read_file_windowed_preserves_no_trailing_newline():
    inner = FakeSandbox('run-only-no-trailing-newline')
    inner.fs.files['/workspace/file'] = b'one\ntwo\nthree'

    window = await Sandbox(_RunOnlySandbox(inner)).read_file('file', offset=2, limit=2)

    assert (window.lines, window.text, window.has_more, window.total_lines) == (
        ('two', 'three'),
        'two\nthree',
        False,
        3,
    )


async def test_windowed_read_without_shell_or_filesystem_has_targeted_error():
    inner = FakeSandbox('run-only-no-sed', sed=False)

    with pytest.raises(NotImplementedError, match=r'working `sed`.*SupportsFilesystem'):
        await Sandbox(_RunOnlySandbox(inner)).read_file('file', limit=1)


async def test_shell_slice_is_bounded_and_stops_at_the_window():
    """The quit command stops `sed` at the window instead of scanning to EOF, and the slice
    carries a deadline so it never costs more than the read it optimizes."""
    backend = FakeSandbox('bounded', {'/workspace/file': b'one\ntwo\nthree\n'})
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


async def test_slice_that_times_out_falls_back_to_the_filesystem_window():
    """A wedged path (a FIFO that never produces output) hits the slice deadline; the
    window is then served from the filesystem instead of failing the read."""

    class WedgedSed(FakeSandbox):
        async def run(
            self,
            command: str | Sequence[str],
            *,
            shell: bool = False,
            cwd: str | None = None,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
        ) -> FakeSandboxResult:
            raise SandboxTimeoutError('sed hung', timeout=timeout)

    backend = WedgedSed('wedged', {'/workspace/file': b'one\ntwo\nthree\n'})

    window = await Sandbox(backend).read_file('file', offset=2, limit=1)

    assert window.lines == ('two',)
    assert backend.fs.reads == ['/workspace/file']


async def test_run_rejects_a_relative_cwd():
    """A relative `cwd` has no sandbox meaning — a backend resolving it against ambient
    host state is exactly the one-environment break the protocol forbids — so the facade
    rejects it before any backend sees it.
    """
    sandbox = Sandbox(FakeSandbox('cwd'))
    with pytest.raises(ValueError, match='absolute'):
        await sandbox.run(['true'], cwd='subdir')


async def test_read_file_missing_file_surfaces_filesystem_error_without_reading():
    backend = FakeSandbox('missing')

    with pytest.raises(FileNotFoundError):
        await Sandbox(backend).read_file('nope.txt', limit=3)

    assert backend.fs.reads == []


async def test_bare_run_context_sandbox_is_unavailable():
    """A `RunContext` not backed by a run grants no execution: sandboxes are attached by run
    assembly only, so synthetic contexts (e.g. in user test suites) can't silently execute
    on the host.
    """
    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())
    with pytest.raises(UserError, match='created outside an agent run'):
        await ctx.sandbox.run(['echo', 'hello'])


async def test_unavailable_sandbox_surfaces_reason_for_every_operation():
    reason = 'sandbox disabled by policy'
    sandbox = Sandbox(UnavailableSandbox(reason))
    operations = [
        sandbox.run(['echo', 'hello']),
        sandbox.working_dir(),
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


def test_sandbox_errors_share_the_common_exception_family():
    unavailable = SandboxUnavailableError('unavailable')
    timeout = SandboxTimeoutError('timed out')

    assert isinstance(unavailable, SandboxError)
    assert isinstance(timeout, SandboxError)
    assert issubclass(SandboxError, RuntimeError)
    assert isinstance(timeout, TimeoutError)


async def test_run_without_sandbox_is_unavailable_with_attachment_instructions():
    """No run ever gets implicit access to the host: without an explicit sandbox, every
    operation raises with the ways to attach one.
    """
    agent: Agent = Agent(_tool_call_then_text('use_default'))

    @agent.tool
    async def use_default(ctx: RunContext[Any]) -> str:
        await ctx.sandbox.run(['echo', 'hello'])
        return 'ok'  # pragma: no cover

    with pytest.raises(UserError, match=r'No sandbox is attached to this run.+`LocalSandbox\(\)`'):
        await agent.run('go')


async def test_unavailable_sandbox_run_argument_overrides_default_reason():
    """A policy `UnavailableSandbox` is the run's sandbox, refusing with its own reason."""
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


@pytest.mark.parametrize('wrapped', [False, True])
async def test_run_argument_sandbox_is_the_same_object_in_every_step(wrapped: bool):
    """Tool calls in different run steps observe one `Sandbox` over the caller's backend; a
    caller-built `Sandbox` is used as-is."""
    observed: list[Sandbox] = []

    def model_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) in (1, 3):
            return ModelResponse(parts=[ToolCallPart('probe', {})])
        return ModelResponse(parts=[TextPart('done')])

    agent: Agent = Agent(FunctionModel(model_func))

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        observed.append(ctx.sandbox)
        return 'ok'

    backend = FakeSandbox('stable')
    sandbox: SandboxBackend = Sandbox(backend) if wrapped else backend
    await agent.run('go', sandbox=sandbox)
    assert len(observed) == 2
    assert observed[0].backend is backend
    assert observed[1] is observed[0]
    if wrapped:
        assert observed[0] is sandbox


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
        return SandboxRef(sandbox_id=self.backend.sandbox_id)

    def resolve_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> SandboxBackend | None:
        if self.backend is None or ref.sandbox_id != self.backend.sandbox_id:
            return None
        self.events.append(f'{self.name}:connect')
        return self.backend

    async def release_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> None:
        self.events.append(f'{self.name}:release')


async def test_sandbox_ref_connects_once_and_exposes_identity():
    connector = ConnectOnlySandboxCapability()
    observed: list[Sandbox] = []
    agent: Agent = Agent(_tool_call_then_text(), capabilities=[connector])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        sandbox = ctx.sandbox
        observed.append(sandbox)
        assert sandbox.sandbox_id == 'fake-deferred'
        assert sandbox.ref == SandboxRef(sandbox_id='fake-deferred')
        assert sandbox.backend is connector.backends[0]

        run_results = await asyncio.gather(
            sandbox.run(['one']),
            sandbox.run(['two']),
            sandbox.working_dir(),
        )
        assert [result.stdout for result in run_results[:2]] == ['connected', 'connected']
        assert run_results[2] == '/workspace'
        return 'ok'

    result = await agent.run('go', sandbox=SandboxRef(sandbox_id='fake-deferred'))
    assert result.output == 'done'
    assert len(observed) == 1
    assert connector.sandbox_ids == ['fake-deferred']  # concurrent first ops connect exactly once


@pytest.mark.parametrize('sandbox_arg', [None, SandboxRef(sandbox_id='fake-connected')])
async def test_capability_connection_is_detached_without_terminating_sandbox(sandbox_arg: SandboxRef | None):
    close_calls: list[bool] = []

    class Connector(AbstractCapability[Any]):
        async def acquire_sandbox(self, ctx: RunContext[Any]) -> SandboxRef:
            return SandboxRef(sandbox_id='fake-connected')

        def resolve_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> SandboxBackend | None:
            return _ClosableFakeSandbox('connected', close_calls)

    await make_connecting_probe_agent([], capabilities=[Connector()]).run('go', sandbox=sandbox_arg)
    assert close_calls == [False]


async def test_caller_owned_backend_is_not_closed_by_run():
    close_calls: list[bool] = []
    backend = _ClosableFakeSandbox('direct', close_calls)
    facade = Sandbox(backend)

    await make_probe_agent([], capabilities=[]).run('go', sandbox=facade)
    assert close_calls == []


async def test_capability_sandbox_filesystem_serves_every_operation():
    """A capability-constructed backend exposes its filesystem for the whole run."""
    backend = FakeSandbox('proxy')
    connected: list[str] = []

    class Connector(AbstractCapability[Any]):
        async def acquire_sandbox(self, ctx: RunContext[Any]) -> SandboxRef:
            return SandboxRef(sandbox_id=backend.sandbox_id)

        def resolve_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> SandboxBackend | None:
            connected.append(ref.sandbox_id)
            return backend

    agent: Agent = Agent(_tool_call_then_text(), capabilities=[Connector()])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        fs = ctx.sandbox.fs
        assert connected == ['fake-proxy']
        await fs.write_bytes('/workspace/notes.txt', b'hello')
        assert await fs.read_bytes('/workspace/notes.txt') == b'hello'
        assert (await fs.stat('/workspace/notes.txt')).size == 5
        assert [entry.path for entry in await fs.list_dir('/workspace')] == ['/workspace/notes.txt']
        await fs.make_dir('/workspace/sub')
        assert await fs.exists('/workspace/notes.txt')
        await fs.remove('/workspace/notes.txt')
        assert not await fs.exists('/workspace/notes.txt')
        return 'ok'

    await agent.run('go')
    assert connected == ['fake-proxy']  # connected exactly once across all operations


async def test_ref_resolution_skips_deferred_capabilities():
    """A deferred capability's contributions are inert until it loads, so it is never asked to
    connect a ref; the loaded connector answers alone."""
    deferred = SandboxCapability(id='deferred-sandbox', defer_loading=True)
    connector = ConnectOnlySandboxCapability()
    seen: list[str] = []
    agent = make_connecting_probe_agent(seen, capabilities=[connector, deferred])
    await agent.run('go', sandbox=SandboxRef(sandbox_id='fake-1'))
    assert seen == ['1']
    assert connector.sandbox_ids == ['fake-1']


@pytest.mark.parametrize('capabilities', [[], [DecliningSandboxCapability()], [SandboxCapability()]])
async def test_sandbox_ref_requires_connecting_capability(capabilities: list[AbstractCapability[Any]]):
    """No capability, or every capability declining the ref, produces the attachment error."""
    agent = make_connecting_probe_agent([], capabilities=capabilities)
    with pytest.raises(UserError, match=r"No capability on this agent can resolve sandbox 'sandbox-1'"):
        await agent.run('go', sandbox=SandboxRef(sandbox_id='sandbox-1'))


async def test_sandbox_ref_connection_failure_is_chained():
    error = RuntimeError('expired')

    @dataclass
    class FailingConnectCapability(AbstractCapability[Any]):
        def resolve_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> SandboxBackend | None:
            raise error

    agent = make_connecting_probe_agent([], capabilities=[FailingConnectCapability()])
    with pytest.raises(
        UserError,
        match=re.escape("Failed to connect to sandbox 'expired'.") + '$',
    ) as exc_info:
        await agent.run('go', sandbox=SandboxRef(sandbox_id='expired'))
    assert exc_info.value.__cause__ is error


async def test_sandbox_ref_user_error_is_preserved():
    class UserErrorConnector(AbstractCapability[Any]):
        def resolve_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> SandboxBackend | None:
            raise UserError('credentials are missing')

    agent = Agent(_tool_call_then_text(), capabilities=[UserErrorConnector()])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        await ctx.sandbox.run(['true'])
        return 'ok'  # pragma: no cover

    with pytest.raises(UserError, match='credentials are missing'):
        await agent.run('go', sandbox=SandboxRef(sandbox_id='broken'))


async def test_sandbox_ref_wins_over_sandbox_supplier():
    class SupplierAndConnector(ConnectOnlySandboxCapability):
        async def acquire_sandbox(self, ctx: RunContext[Any]) -> SandboxRef:
            raise AssertionError('an explicit ref must skip acquisition')  # pragma: no cover

    supplier = SupplierAndConnector()
    seen: list[str] = []
    agent = make_connecting_probe_agent(seen, capabilities=[supplier])
    await agent.run('go', sandbox=SandboxRef(sandbox_id='fake-winner'))
    assert seen == ['winner']
    assert supplier.sandbox_ids == ['fake-winner']


async def test_multiple_ref_connectors_are_rejected():
    first = ConnectOnlySandboxCapability()
    last = ConnectOnlySandboxCapability()
    agent = make_connecting_probe_agent([], capabilities=[first, WrapperCapability(wrapped=last)])
    with pytest.raises(
        UserError,
        match=r'Several capabilities can resolve sandbox refs '
        r'\(connect_only_sandbox_capability, wrapper_capability\)',
    ):
        await agent.run('go', sandbox=SandboxRef(sandbox_id='fake-1'))
    assert first.sandbox_ids == []
    assert last.sandbox_ids == []


def test_contributes_sandbox_detection():
    class ConfiguredSandbox(AbstractCapability[Any]):
        async def acquire_sandbox(self, ctx: RunContext[Any]) -> SandboxRef:
            return SandboxRef(sandbox_id='configured')

    class DurableSandbox(ConfiguredSandbox):
        @durable_operation('acquire_sandbox')
        async def acquire_sandbox(self, ctx: RunContext[Any]) -> SandboxRef:
            return await super().acquire_sandbox(ctx)

    assert contributes_sandbox(ConnectOnlySandboxCapability()) is False
    assert contributes_sandbox(ConfiguredSandbox()) is False
    assert contributes_sandbox(WrapperCapability(wrapped=ConfiguredSandbox())) is False
    assert contributes_sandbox(WrapperCapability(wrapped=DurableSandbox())) is True
    deferred = DurableSandbox()
    deferred.id = 'deferred-sandbox'
    deferred.defer_loading = True
    assert contributes_sandbox(deferred) is False
    assert contributes_sandbox(CombinedCapability([DurableSandbox(), ConnectOnlySandboxCapability()])) is True
    supplier = DurableSandbox()
    assert contributes_sandbox(CombinedCapability([supplier, WrapperCapability(wrapped=supplier)])) is True


async def test_lifecycle_capability_creates_at_run_start_and_tears_down_at_run_end():
    """A capability owning the lifecycle gives the run the whole bracket: setup before the
    acquisition first, release after the run. The tool reaches the sandbox by reconnecting
    through `resolve_sandbox`, exactly as it would inside a durable engine's activity.
    """
    lifecycle = LifecycleSandboxCapability()
    agent: Agent = Agent(_tool_call_then_text(), capabilities=[lifecycle])
    seen: list[str] = []

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        seen.append(ctx.sandbox.sandbox_id)
        # Created but not yet torn down while the run is still going.
        assert lifecycle.events == ['acquire:created-1', 'connect:created-1']
        await ctx.sandbox.run(['echo', 'hello'])
        assert lifecycle.events == ['acquire:created-1', 'connect:created-1']
        return 'ok'

    result = await agent.run('go')
    assert result.output == 'done'
    assert seen == ['created-1']
    assert lifecycle.events == ['acquire:created-1', 'connect:created-1', 'release:created-1']
    assert [backend.commands for backend in lifecycle.backends] == [[['echo', 'hello']]]


@pytest.mark.parametrize('failure', ['tool', 'metadata', 'for_run', 'toolset_entry'])
async def test_sandbox_is_released_when_the_run_fails(failure: str):
    """The exit stack owns the bracket, so release runs however the run fails."""
    lifecycle = LifecycleSandboxCapability()
    capabilities: list[AbstractCapability[Any]] = [lifecycle]
    toolsets: list[AbstractToolset[Any]] = []
    run_kwargs: dict[str, Any] = {}

    if failure == 'tool':

        def model_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart('explode', {})])

        agent: Agent = Agent(FunctionModel(model_func), capabilities=capabilities)

        @agent.tool
        async def explode(ctx: RunContext[Any]) -> str:
            raise RuntimeError('boom')

    else:
        if failure == 'metadata':

            def failing_metadata(ctx: RunContext[Any]) -> dict[str, Any]:
                raise RuntimeError('boom')

            run_kwargs['metadata'] = failing_metadata
        elif failure == 'for_run':

            @dataclass
            class FailingForRun(AbstractCapability[Any]):
                async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
                    raise RuntimeError('boom')

            capabilities.append(FailingForRun())
        else:

            class ExplodingToolset(WrapperToolset[Any]):
                async def __aenter__(self) -> Any:
                    raise RuntimeError('boom')

            toolsets.append(ExplodingToolset(wrapped=FunctionToolset()))
        agent = Agent(TestModel(), capabilities=capabilities, toolsets=toolsets)

    with pytest.raises(RuntimeError, match='boom'):
        await agent.run('go', **run_kwargs)
    assert lifecycle.events == ['acquire:created-1', 'connect:created-1', 'release:created-1']


@pytest.mark.parametrize('run_mode', ['run', 'iter', 'run_stream'])
async def test_sandbox_lifecycle_brackets_the_run(run_mode: str) -> None:
    """The sandbox is acquired before `for_run` and released after the run's own teardown."""
    events: list[str] = []

    class LifecycleToolset(WrapperToolset[Any]):
        async def __aenter__(self) -> LifecycleToolset:
            events.append('toolset_enter')
            await self.wrapped.__aenter__()
            return self

        async def __aexit__(self, *args: Any) -> bool | None:
            events.append('toolset_exit')
            return await self.wrapped.__aexit__(*args)

    @dataclass
    class OrderingSandboxCapability(AbstractCapability[Any]):
        async def acquire_sandbox(self, ctx: RunContext[Any]) -> SandboxRef:
            events.append('acquire_sandbox')
            return SandboxRef(sandbox_id='lifecycle')

        def resolve_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> SandboxBackend | None:
            events.append('resolve_sandbox')
            return FakeSandbox('lifecycle')

        async def release_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> None:
            events.append('release_sandbox')

        async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
            events.append('for_run')
            return self

        def get_toolset(self) -> AbstractToolset[Any]:
            return LifecycleToolset(wrapped=FunctionToolset())

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        events.append('model')
        return ModelResponse(parts=[TextPart('done')])

    async def stream_function(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        events.append('model')
        yield 'done'

    agent = Agent(
        FunctionModel(model_function, stream_function=stream_function), capabilities=[OrderingSandboxCapability()]
    )

    if run_mode == 'run':
        await agent.run('hello')
    elif run_mode == 'iter':
        async with agent.iter('hello') as agent_run:
            node = agent_run.next_node
            while not isinstance(node, End):
                node = await agent_run.next(node)
    else:
        async with agent.run_stream('hello') as stream:
            await stream.get_output()

    assert events == [
        'acquire_sandbox',
        'resolve_sandbox',
        'for_run',
        'toolset_enter',
        'model',
        'toolset_exit',
        'release_sandbox',
    ]


async def test_duplicate_per_run_supplier_ids_fail_and_release():
    creator = LifecycleSandboxCapability()
    creator.id = 'duplicate'
    decliner = DecliningSandboxCapability()
    decliner.id = 'duplicate'
    agent = Agent(TestModel(), capabilities=[creator])

    with pytest.raises(UserError, match=r"Capability id 'duplicate' is used by multiple capabilities"):
        await agent.run('go', capabilities=[decliner])
    assert creator.events == ['acquire:created-1', 'connect:created-1', 'release:created-1']


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
    assert creator.events == ['acquire:created-1', 'connect:created-1']


@pytest.mark.parametrize('failure', ['raise', 'return-none'])
async def test_acquired_sandbox_connection_failure_is_explained(failure: str):
    error = RuntimeError('connection failed')

    @dataclass
    class OwnedButUnreachable(AbstractCapability[Any]):
        async def acquire_sandbox(self, ctx: RunContext[Any]) -> SandboxRef:
            return SandboxRef(sandbox_id='owned')

        def resolve_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> SandboxBackend | None:
            if failure == 'raise':
                raise error
            return None

    agent = make_connecting_probe_agent([], capabilities=[OwnedButUnreachable()])
    expected = (
        "Failed to connect to sandbox 'owned'."
        if failure == 'raise'
        else "No capability with id 'owned_but_unreachable' is attached to this agent to connect sandbox 'owned'"
    )
    with pytest.raises(UserError, match=re.escape(expected)) as exc_info:
        await agent.run('go')
    assert exc_info.value.__cause__ is (error if failure == 'raise' else None)


def test_durable_workflow_sandbox_guard():
    class DurableSupplier(AcquireOnlySandboxCapability):
        @durable_operation('acquire_sandbox')
        async def acquire_sandbox(self, ctx: RunContext[Any]) -> SandboxRef:
            return await super().acquire_sandbox(ctx)

    with pytest.raises(UserError, match='contribution'):
        guard_workflow_sandbox(
            None,
            [DurableSupplier()],
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
    ref = SandboxRef(sandbox_id='ref')
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
    with pytest.raises(UserError, match=r'`sandbox=` takes a `SandboxRef` or a `SandboxBackend`'):
        guard_workflow_sandbox(
            AcquireOnlySandboxCapability(),  # pyright: ignore[reportArgumentType]
            None,
            static_contributes_sandbox=False,
            contribution_error='contribution blocked',
            live_error='live blocked',
        )


async def test_combined_capability_cannot_release_unstamped_ref():
    capability: CombinedCapability[Any] = CombinedCapability([])
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())

    with pytest.raises(UserError, match=r"Sandbox ref 'caller-built' without a `capability_id` cannot be released"):
        await capability.release_sandbox(ctx, SandboxRef(sandbox_id='caller-built'))


async def test_lifecycle_capability_also_connects_ref_run_arguments():
    """The same capability serves both jobs: with a ref run argument its `acquire_sandbox` is
    skipped (the caller owns the lifecycle), but its `resolve_sandbox` still connects.
    """
    lifecycle = LifecycleSandboxCapability()
    seen: list[str] = []
    agent = make_connecting_probe_agent(seen, capabilities=[lifecycle])
    await agent.run('go', sandbox=SandboxRef(sandbox_id='pre-existing'))
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


async def test_capability_supplied_sandbox_is_constructed_eagerly():
    """A run constructs the backend before hooks even when no sandbox operation follows."""
    cap = SandboxCapability()
    await make_probe_agent([], capabilities=[cap]).run('go')
    assert cap.events == ['cap:acquire', 'cap:connect', 'cap:release']


async def test_capability_sandbox_live_through_after_run():
    """Teardown runs after `after_run`, so hooks never see a torn-down sandbox."""

    @dataclass
    class ContributingWatcher(SandboxCapability):
        async def after_run(self, ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
            self.events.append(f'after_run:{_describe(ctx.sandbox)}')
            return result

    cap = ContributingWatcher()
    await make_probe_agent([], capabilities=[cap]).run('go')
    assert cap.events == ['cap:acquire', 'cap:connect', 'after_run:cap', 'cap:release']


async def test_run_argument_wins_over_capability():
    first = SandboxCapability(name='first-loser')
    second = SandboxCapability(name='second-loser')
    seen: list[str] = []
    agent = make_probe_agent(seen, capabilities=[first, second])
    await agent.run('go', sandbox=FakeSandbox('direct'))
    assert seen == ['direct']
    assert first.events == []
    assert second.events == []


async def test_wrapper_sandbox_uses_middleware_order_and_keeps_ref():
    events: list[str] = []

    class MiddlewareBackend(FakeSandbox):
        def __init__(self, label: str, wrapped: SandboxBackend) -> None:
            super().__init__(label)
            self.wrapped = wrapped

        @property
        def sandbox_id(self) -> str:
            return self.wrapped.sandbox_id

        async def run(
            self,
            command: str | Sequence[str],
            *,
            shell: bool = False,
            cwd: str | None = None,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
        ) -> FakeSandboxResult:
            events.append(f'{self.name}:before')
            result = await self.wrapped.run(command, shell=shell, cwd=cwd, env=env, timeout=timeout)
            events.append(f'{self.name}:after')
            return FakeSandboxResult(exit_code=result.exit_code, stdout=result.stdout, stderr=result.stderr)

        async def working_dir(self) -> str:
            return await self.wrapped.working_dir()

    @dataclass
    class Middleware(AbstractCapability[Any]):
        label: str

        def get_wrapper_sandbox(self, ctx: RunContext[Any], sandbox: Sandbox) -> Sandbox:
            return Sandbox(MiddlewareBackend(self.label, sandbox.backend), ref=sandbox.ref)

    connector = ConnectOnlySandboxCapability()
    agent = Agent(_tool_call_then_text(), capabilities=[Middleware('first'), Middleware('second'), connector])
    seen_ref: list[SandboxRef | None] = []

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        seen_ref.append(ctx.sandbox.ref)
        await ctx.sandbox.run(['true'])
        return 'ok'

    ref = SandboxRef(sandbox_id='wrapped')
    await agent.run('go', sandbox=ref)

    assert seen_ref == [ref]
    assert events == ['first:before', 'second:before', 'second:after', 'first:after']


async def test_first_sandbox_supplier_wins_and_later_supplier_is_not_asked():
    first = SandboxCapability(name='first')
    last = SandboxCapability(name='last')
    seen: list[str] = []

    await make_probe_agent(seen, capabilities=[first, last]).run('go')

    assert seen == ['first']
    assert first.events == ['first:acquire', 'first:connect', 'first:release']
    assert last.events == []


async def test_declining_sandbox_supplier_falls_through_to_next_supplier():
    decliner = DecliningSandboxCapability()
    winner = SandboxCapability(name='winner')
    seen: list[str] = []

    await make_probe_agent(seen, capabilities=[decliner, winner]).run('go')

    assert decliner.acquire_calls == 1
    assert seen == ['winner']
    assert winner.events == ['winner:acquire', 'winner:connect', 'winner:release']


async def test_acquired_sandbox_ref_is_stamped_and_released_only_by_winner():
    received: list[SandboxRef] = []

    class Winner(SandboxCapability):
        async def release_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> None:
            received.append(ref)
            await super().release_sandbox(ctx, ref)

    winner = Winner(name='winner')
    await make_probe_agent([], capabilities=[winner]).run('go')

    assert received == [SandboxRef(sandbox_id='fake-winner', capability_id='winner')]


async def test_wrapper_around_sandbox_supplier_preserves_routing_identity():
    supplier = SandboxCapability(name='wrapped-supplier')
    wrapper = WrapperCapability(wrapped=supplier)

    await make_probe_agent([], capabilities=[wrapper]).run('go')

    assert wrapper.id is None
    assert supplier.events == ['wrapped-supplier:acquire', 'wrapped-supplier:connect', 'wrapped-supplier:release']


async def test_stamped_sandbox_ref_with_unknown_capability_id_names_it():
    with pytest.raises(
        UserError,
        match=re.escape("No capability with id 'missing' is attached to this agent to connect sandbox 'orphan'"),
    ):
        await make_probe_agent([]).run('go', sandbox=SandboxRef(sandbox_id='orphan', capability_id='missing'))


async def test_sandbox_argument_rejects_capability_with_lifecycle_guidance():
    capability = SandboxCapability()
    with pytest.raises(
        UserError,
        match=re.escape(
            '`sandbox=` takes a `SandboxRef` or a `SandboxBackend`, not a capability. '
            'Pass `SandboxCapability` through `capabilities=[...]` so Pydantic AI can manage its lifecycle.'
        ),
    ):
        await make_probe_agent([]).run('go', sandbox=capability)  # pyright: ignore[reportArgumentType]


async def test_wrapper_sandbox_must_preserve_ref():
    class DropsRef(AbstractCapability[Any]):
        def get_wrapper_sandbox(self, ctx: RunContext[Any], sandbox: Sandbox) -> Sandbox:
            return Sandbox(sandbox.backend)

    agent = make_probe_agent([], capabilities=[DropsRef(), ConnectOnlySandboxCapability()])
    with pytest.raises(UserError, match=r"DropsRef\.get_wrapper_sandbox must preserve the sandbox's `ref`"):
        await agent.run('go', sandbox=SandboxRef(sandbox_id='wrapped'))


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
            return SandboxRef(sandbox_id=warm.sandbox_id)

        def resolve_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> SandboxBackend | None:
            return warm if ref.sandbox_id == warm.sandbox_id else None

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


async def test_failing_release_propagates():
    """A release failure surfaces to the caller, exactly like a toolset exit error."""
    failing = FailingReleaseSandboxCapability()
    agent = make_probe_agent([], capabilities=[failing])
    with pytest.raises(RuntimeError, match="sandbox 'created-1' is already gone"):
        await agent.run('go')
    assert failing.events == ['acquire:created-1', 'connect:created-1', 'release-failed:created-1']


async def test_release_survives_run_cancellation():
    """Cancelling the run must not abort `release_sandbox`: the exit stack unwinds inside an
    already-cancelled scope, so an unshielded release would die at its first await and leak
    the sandbox.
    """

    class AwaitingTeardownCapability(SandboxCapability):
        async def acquire_sandbox(self, ctx: RunContext[Any]) -> SandboxRef:
            self.events.append(f'{self.name}:acquire')
            self.backend = _ClosableFakeSandbox(self.name, close_calls)
            return SandboxRef(sandbox_id=self.backend.sandbox_id)

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


async def test_declining_supplier_falls_through_to_default():
    """A supplier that returns `None` contributes nothing; the run keeps the unavailable default."""
    decliner = DecliningSandboxCapability()
    seen: list[str] = []
    await make_probe_agent(seen, capabilities=[decliner]).run('go')
    assert seen == ['unavailable']
    assert decliner.acquire_calls == 1
