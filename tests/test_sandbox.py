"""Tests for sandbox backends, the rich facade, and read-only `RunContext.sandbox` propagation."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import anyio
import pytest

from pydantic_ai import Agent, RunContext, UnavailableSandbox, UserError
from pydantic_ai.agent import WrapperAgent
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, WrapperCapability
from pydantic_ai.durable_exec._sandbox import contributes_sandbox, run_sandbox_supplier
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
    ConnectOnlySandboxCapability,
    CreateOnlySandboxCapability,
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


_SED_WINDOW_EXPR = re.compile(r'^(\d+),(\d+)p$')


class FakeSandbox:
    """A minimal in-memory implementation of the `SandboxBackend` protocol.

    Honors the protocol's one-environment contract: the `sed` line-window form the
    `Sandbox` facade emits is served from the same files `fs` exposes. `sed=False`
    models an environment without a usable `sed` (exit 127), forcing the facade's
    full-read fallback. Other commands echo `ran:<command>` for forwarding tests.
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
    backend = FakeSandbox('slow', sed=False)  # no usable `sed`: pins the full-read fallback
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


async def test_read_file_shell_slice_empty_window_has_unknown_total():
    """`sed` prints nothing both for an offset past EOF and for a short file, so an empty
    shell window cannot claim a total.
    """
    backend = FakeSandbox('sliced-past-eof')
    backend.fs.files['/workspace/file'] = b'one\n'

    window = await Sandbox(backend).read_file('file', offset=10, limit=2)

    assert window.lines == ()
    assert window.start_line == 10
    assert window.has_more is False
    assert window.total_lines is None


async def test_read_file_falls_back_to_full_read_without_sed():
    """A failed slice attempt (no `sed` in the environment) falls back to the
    authoritative filesystem read.
    """
    backend = FakeSandbox('no-sed', sed=False)
    backend.fs.files['/workspace/file'] = b'one\ntwo\n'

    window = await Sandbox(backend).read_file('file', offset=1, limit=1)

    assert window.lines == ('one',)
    assert window.has_more is True
    assert window.total_lines == 2
    assert backend.fs.reads == ['/workspace/file']


async def test_read_file_missing_file_falls_back_to_filesystem_error():
    """A missing file makes the slice attempt exit non-zero, so the filesystem read runs
    and surfaces the authoritative error: `FileNotFoundError`, as the protocol requires of
    every filesystem.
    """
    backend = FakeSandbox('missing')

    with pytest.raises(FileNotFoundError):
        await Sandbox(backend).read_file('nope.txt', limit=3)

    assert backend.fs.reads == ['/workspace/nope.txt']


async def test_read_file_on_unavailable_sandbox_surfaces_reason():
    """The slice attempt swallows `run()`'s failure so the filesystem read surfaces the
    authoritative policy reason instead.
    """
    sandbox = Sandbox(UnavailableSandbox('sandbox disabled by policy'))
    with pytest.raises(UserError, match='sandbox disabled by policy'):
        await sandbox.read_file('/file', limit=3)


async def test_read_file_shell_and_full_paths_have_window_parity():
    content = b'one\r\ntwo\r\nthree\nfour\r\nfive'
    shell_backend = FakeSandbox('shell')
    shell_backend.fs.files['/workspace/file'] = content
    full_backend = FakeSandbox('full', sed=False)
    full_backend.fs.files['/workspace/file'] = content

    for offset in (1, 2, 4, 8):
        for limit in (1, 2, 5):
            full = await Sandbox(full_backend).read_file('file', offset=offset, limit=limit)
            shell = await Sandbox(shell_backend).read_file('file', offset=offset, limit=limit)
            assert (shell.lines, shell.start_line, shell.has_more) == (full.lines, full.start_line, full.has_more)
            # The shell slice only knows the total when it provably reached EOF.
            assert shell.total_lines in (full.total_lines, None)


async def test_bare_run_context_sandbox_is_unavailable():
    """A `RunContext` not backed by a run grants no execution: sandboxes are attached by run
    assembly only, so synthetic contexts (e.g. in user test suites) can't silently execute
    on the host.
    """
    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())
    assert ctx.sandbox.provider == 'unavailable'
    with pytest.raises(UserError, match='created outside an agent run'):
        await ctx.sandbox.run(['echo', 'hello'])


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
        return (await ctx.sandbox.run(['echo', 'hello'])).stdout  # pragma: no cover

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

    async def create_sandbox(self, ctx: RunContext[Any]) -> SandboxRef:
        self.events.append(f'{self.name}:setup')
        self.backend = FakeSandbox(self.name)
        return SandboxRef(provider='fake', sandbox_id=self.backend.sandbox_id)

    async def get_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> SandboxBackend | None:
        if ref.provider != 'fake' or self.backend is None or ref.sandbox_id != self.backend.sandbox_id:
            return None
        self.events.append(f'{self.name}:connect')
        return self.backend

    async def destroy_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> None:
        self.events.append(f'{self.name}:teardown')


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


@pytest.mark.parametrize('with_other_capability', [False, True], ids=['no-capabilities', 'unrelated-capability'])
async def test_sandbox_ref_requires_recognizing_capability(with_other_capability: bool):
    @dataclass
    class OtherProviderCapability(AbstractCapability[Any]):
        async def get_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> SandboxBackend | None:
            return FakeSandbox('other') if ref.provider == 'other' else None

    capabilities = [OtherProviderCapability()] if with_other_capability else []
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
        async def get_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> SandboxBackend | None:
            await asyncio.sleep(0)
            raise error

    agent = make_connecting_probe_agent([], capabilities=[FailingConnectCapability()])
    with pytest.raises(
        UserError,
        match=re.escape("Failed to connect to sandbox 'expired' for provider 'fake'.") + '$',
    ) as exc_info:
        await agent.run('go', sandbox=SandboxRef(provider='fake', sandbox_id='expired'))
    assert exc_info.value.__cause__ is error


async def test_sandbox_ref_wins_over_sandbox_supplier():
    supplier = SandboxCapability(name='loser')
    connector = ConnectOnlySandboxCapability()
    seen: list[str] = []
    agent = make_connecting_probe_agent(seen, capabilities=[supplier, connector])
    await agent.run('go', sandbox=SandboxRef(provider='fake', sandbox_id='fake-winner'))
    assert seen == ['winner']
    assert supplier.events == []
    assert connector.sandbox_ids == ['fake-winner']


async def test_ref_resolution_prefers_the_latest_recognizing_capability():
    """Two capabilities recognize the same provider name; the latest in the chain connects,
    matching supplier precedence everywhere else.
    """
    first = ConnectOnlySandboxCapability()
    last = ConnectOnlySandboxCapability()
    seen: list[str] = []
    agent = make_connecting_probe_agent(seen, capabilities=[first, WrapperCapability(wrapped=last)])
    await agent.run('go', sandbox=SandboxRef(provider='fake', sandbox_id='fake-1'))
    assert seen == ['1']
    assert first.sandbox_ids == []
    assert last.sandbox_ids == ['fake-1']


def test_contributes_sandbox_detection():
    assert contributes_sandbox(ConnectOnlySandboxCapability()) is False  # connecting alone supplies nothing
    assert contributes_sandbox(WrapperCapability(wrapped=SandboxCapability())) is True
    assert contributes_sandbox(SandboxCapability(id='deferred-sandbox', defer_loading=True)) is False
    assert contributes_sandbox(CombinedCapability([SandboxCapability(), ConnectOnlySandboxCapability()])) is True


def test_run_sandbox_supplier_only_reports_the_winner():
    """Durable engines may only route the sandbox a non-durable run would actually have used."""
    lifecycle = LifecycleSandboxCapability()
    other = SandboxCapability(name='other')
    assert run_sandbox_supplier(CombinedCapability([other, lifecycle])) is lifecycle
    assert run_sandbox_supplier(WrapperCapability(wrapped=lifecycle)) is lifecycle
    assert run_sandbox_supplier(CombinedCapability([lifecycle, other])) is other
    assert run_sandbox_supplier(ConnectOnlySandboxCapability()) is None


async def test_lifecycle_capability_creates_at_run_start_and_tears_down_at_run_end():
    """A capability owning the lifecycle gives the run the whole bracket: setup before the
    first hook, teardown after the run. The tool reaches the sandbox by reconnecting
    through `get_sandbox`, exactly as it would inside a durable engine's activity.
    """
    lifecycle = LifecycleSandboxCapability()
    agent: Agent = Agent(_tool_call_then_text(), capabilities=[lifecycle])
    seen: list[str] = []

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        seen.append(ctx.sandbox.sandbox_id)
        # Created but not yet torn down while the run is still going.
        assert lifecycle.events == ['create:created-1']
        await ctx.sandbox.run(['echo', 'hello'])
        assert lifecycle.events == ['create:created-1', 'connect:created-1']
        return 'ok'

    result = await agent.run('go')
    assert result.output == 'done'
    assert seen == ['created-1']
    assert lifecycle.events == ['create:created-1', 'connect:created-1', 'teardown:created-1']
    assert [backend.commands for backend in lifecycle.backends] == [[['echo', 'hello']]]


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
    assert lifecycle.events == ['create:created-1', 'teardown:created-1']


async def test_create_only_capability_leans_on_platform_reaping():
    """The inherited no-op `destroy_sandbox` is what lets a capability lean on its
    platform's idle timeout instead of destroying anything itself.
    """
    creator = CreateOnlySandboxCapability()
    seen: list[str] = []
    agent = make_probe_agent(seen, capabilities=[creator])
    result = await agent.run('go')
    assert result.output == 'done'
    assert seen == ['created-1']
    assert creator.events == ['create:created-1']


async def test_lifecycle_capability_also_connects_ref_run_arguments():
    """The same capability serves both jobs: with a ref run argument its `create_sandbox` is
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
    assert cap.events == ['cap:setup', 'cap:connect', 'cap:teardown']


async def test_capability_supplied_sandbox_connects_lazily():
    """A run that never touches `ctx.sandbox` pays setup and teardown, but no connection."""
    cap = SandboxCapability()
    await make_probe_agent([], capabilities=[cap]).run('go')
    assert cap.events == ['cap:setup', 'cap:teardown']


async def test_capability_sandbox_live_through_after_run():
    """Teardown runs after `after_run`, so hooks never see a torn-down sandbox."""

    @dataclass
    class ContributingWatcher(SandboxCapability):
        async def after_run(self, ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
            self.events.append(f'after_run:{_describe(ctx.sandbox)}')
            return result

    cap = ContributingWatcher()
    await make_probe_agent([], capabilities=[cap]).run('go')
    assert cap.events == ['cap:setup', 'after_run:cap', 'cap:teardown']


async def test_run_argument_wins_over_capability():
    cap = SandboxCapability(name='loser')
    seen: list[str] = []
    agent = make_probe_agent(seen, capabilities=[cap])
    await agent.run('go', sandbox=FakeSandbox('direct'))
    assert seen == ['direct']
    assert cap.events == []


async def test_last_capability_in_chain_wins_and_losers_are_never_set_up():
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
    """A warm capability returns the identity of the backend it already holds from
    `create_sandbox` and leaves `destroy_sandbox` alone, so the same environment serves
    every run.
    """
    warm = FakeSandbox('warm')

    @dataclass
    class WarmSandboxCapability(AbstractCapability[Any]):
        async def create_sandbox(self, ctx: RunContext[Any]) -> SandboxRef:
            return SandboxRef(provider='fake', sandbox_id=warm.sandbox_id)

        async def get_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> SandboxBackend | None:
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
    assert inner.events == ['inner:setup', 'inner:connect', 'inner:teardown']


async def test_capability_sandbox_tears_down_when_a_tool_raises():
    cap = SandboxCapability()

    def model_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('explode', {})])

    agent: Agent = Agent(FunctionModel(model_func), capabilities=[cap])

    @agent.tool
    async def explode(ctx: RunContext[Any]) -> str:
        raise RuntimeError('boom')

    with pytest.raises(RuntimeError, match='boom'):
        await agent.run('go')
    assert cap.events == ['cap:setup', 'cap:teardown']


async def test_capability_sandbox_tears_down_when_toolset_entry_fails():
    """The exit stack owns the bracket, so teardown runs even when the run never starts."""

    class ExplodingToolset(WrapperToolset[Any]):
        async def __aenter__(self) -> Any:
            raise RuntimeError('toolset entry failed')

    cap = SandboxCapability()
    agent: Agent = Agent(TestModel(), toolsets=[ExplodingToolset(wrapped=FunctionToolset())], capabilities=[cap])
    with pytest.raises(RuntimeError, match='toolset entry failed'):
        await agent.run('go')
    assert cap.events == ['cap:setup', 'cap:teardown']


async def test_failing_teardown_propagates():
    """An in-process teardown failure surfaces to the caller, exactly like a toolset exit
    error; durable engines make their own call (Temporal logs instead, because the platform
    idle timeout is the backstop and the run's work is already done).
    """
    from .sandbox_fakes import FailingTeardownSandboxCapability

    failing = FailingTeardownSandboxCapability()
    agent = make_probe_agent([], capabilities=[failing])
    with pytest.raises(RuntimeError, match="sandbox 'created-1' is already gone"):
        await agent.run('go')
    assert failing.events == ['create:created-1', 'teardown-failed:created-1']


async def test_teardown_survives_run_cancellation():
    """Cancelling the run must not abort `destroy_sandbox`: the exit stack unwinds inside an
    already-cancelled scope, so an unshielded teardown would die at its first await and leak
    the sandbox.
    """

    class AwaitingTeardownCapability(SandboxCapability):
        async def destroy_sandbox(self, ctx: RunContext[Any], ref: SandboxRef) -> None:
            await asyncio.sleep(0)  # a real teardown awaits provider I/O; an unshielded cancel lands here
            await super().destroy_sandbox(ctx, ref)

    capability = AwaitingTeardownCapability()
    agent: Agent = Agent(_tool_call_then_text(), capabilities=[capability])
    entered = anyio.Event()

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:  # pragma: no cover
        entered.set()
        await asyncio.sleep(10)
        return 'never'

    # An anyio cancel scope is level-triggered: once cancelled, every await during the
    # unwind re-raises, unlike a one-shot `task.cancel()`.
    async with anyio.create_task_group() as tg:

        async def run_agent() -> None:  # pragma: no cover
            await agent.run('go')

        tg.start_soon(run_agent)
        await entered.wait()
        tg.cancel_scope.cancel()
    assert capability.events == ['cap:setup', 'cap:teardown']


async def test_setup_declined_falls_through_to_default():
    """A supplier that returns `None` contributes nothing, and the run falls back to the
    unavailable default rather than treating the decline as an error.
    """

    @dataclass
    class DecliningSupplier(AbstractCapability[Any]):
        async def create_sandbox(self, ctx: RunContext[Any]) -> SandboxRef | None:
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
