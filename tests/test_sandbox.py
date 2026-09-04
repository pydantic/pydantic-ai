"""Tests for the sandbox facade and its lazy backend contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import anyio
import pytest
from pydantic import TypeAdapter

from pydantic_ai import Agent, RunContext, UserError
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.durable_exec._sandbox import guard_workflow_sandbox
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.sandboxes import (
    ReadOnlySandbox,
    Sandbox,
    SandboxBackend,
    SandboxRef,
    SandboxTimeoutError,
    UnavailableSandbox,
)
from pydantic_ai.usage import RunUsage

from .sandbox_fakes import (
    ConnectOnlySandboxCapability,
    DecliningSandboxCapability,
    FakeSandbox,
    FakeSandboxResult,
    SandboxCapability,
)

pytestmark = pytest.mark.anyio


def _tool_call_model(tool_name: str = 'probe') -> FunctionModel:
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name, {})])
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(model)


async def test_resolve_normalizes_paths_against_the_working_directory() -> None:
    sandbox = Sandbox(FakeSandbox('resolve'))

    assert await sandbox.resolve('sub/../notes.txt') == '/workspace/notes.txt'
    assert await sandbox.resolve('/abs/./x') == '/abs/x'
    assert await sandbox.resolve('x', base='/elsewhere') == '/elsewhere/x'


async def test_resolve_rejects_a_relative_base() -> None:
    with pytest.raises(ValueError, match="base must be an absolute path, got 'relative'"):
        await Sandbox(FakeSandbox('resolve')).resolve('file.txt', base='relative')


async def test_flat_file_operations_use_the_backend_filesystem() -> None:
    backend = FakeSandbox('files', {'/workspace/data.txt': b'hello'})
    sandbox = Sandbox(backend)

    assert await sandbox.read_bytes('data.txt') == b'hello'
    assert (await sandbox.stat('data.txt')).path == '/workspace/data.txt'
    assert await sandbox.exists('data.txt')
    assert (await sandbox.list_dir('.'))[0].path == '/workspace/data.txt'
    await sandbox.make_dir('new-dir')
    await sandbox.write_bytes('new.txt', b'new')
    await sandbox.write_text('data.txt', 'updated')
    await sandbox.remove('new.txt')

    assert backend.fs.files['/workspace/data.txt'] == b'updated'
    assert not await sandbox.exists('new.txt')


async def test_text_helpers_resolve_relative_paths() -> None:
    backend = FakeSandbox('text', {'/workspace/data.txt': b'old'})
    sandbox = Sandbox(backend)

    await sandbox.write_text('data.txt', 'updated')

    assert await sandbox.read_text('data.txt') == 'updated'
    assert backend.fs.files['/workspace/data.txt'] == b'updated'


async def test_run_only_backend_supports_bounded_reads_through_shell() -> None:
    inner = FakeSandbox('run-only', {'/workspace/data.txt': b'one\ntwo\nthree\n'})
    commands: list[str | Sequence[str]] = []

    class RunOnlyBackend:
        @property
        def ref(self) -> SandboxRef | None:
            return inner.ref

        async def run(
            self,
            command: str | Sequence[str],
            *,
            shell: bool = False,
            cwd: str | None = None,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
        ) -> FakeSandboxResult:
            commands.append(command)
            return await inner.run(command, shell=shell, cwd=cwd, env=env, timeout=timeout)

        async def working_dir(self) -> str:
            return await inner.working_dir()

    sandbox = Sandbox(RunOnlyBackend())
    window = await sandbox.read_file('data.txt', limit=2)

    assert window.lines == ('one', 'two')
    assert commands == [['sed', '-n', '1,3p;3q', '/workspace/data.txt']]
    assert inner.fs.reads == []
    with pytest.raises(NotImplementedError, match='SupportsFilesystem'):
        await sandbox.read_file('data.txt')


@pytest.mark.parametrize(
    'result',
    [
        pytest.param(FakeSandboxResult(exit_code=127, stderr='sed: not found'), id='no-sed'),
        pytest.param(FakeSandboxResult(exit_code=2), id='nonzero'),
        pytest.param(FakeSandboxResult(stderr='warning'), id='stderr'),
    ],
)
async def test_bounded_read_shell_failures_fall_back_to_filesystem(result: FakeSandboxResult) -> None:
    class FailedSed(FakeSandbox):
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
                return result
            return await super().run(command, shell=shell, cwd=cwd, env=env, timeout=timeout)

    backend = FailedSed('failed-sed', {'/workspace/data.txt': b'one\ntwo\nthree\n'})

    window = await Sandbox(backend).read_file('data.txt', offset=2, limit=1)

    assert window.lines == ('two',)
    assert window.has_more is True
    assert window.total_lines == 3
    assert backend.fs.reads == ['/workspace/data.txt']


async def test_bounded_read_without_shell_or_filesystem_names_both_options() -> None:
    inner = FakeSandbox('no-shell-or-filesystem', sed=False)

    class RunOnlyBackend:
        @property
        def ref(self) -> SandboxRef | None:
            return inner.ref

        async def run(
            self,
            command: str | Sequence[str],
            *,
            shell: bool = False,
            cwd: str | None = None,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
        ) -> FakeSandboxResult:
            return await inner.run(command, shell=shell, cwd=cwd, env=env, timeout=timeout)

        async def working_dir(self) -> str:
            return await inner.working_dir()

    with pytest.raises(NotImplementedError, match=r'working `sed`.*SupportsFilesystem'):
        await Sandbox(RunOnlyBackend()).read_file('data.txt', limit=1)


async def test_slice_timeout_falls_back_to_filesystem() -> None:
    class TimedOutSed(FakeSandbox):
        async def run(
            self,
            command: str | Sequence[str],
            *,
            shell: bool = False,
            cwd: str | None = None,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
        ) -> FakeSandboxResult:
            raise SandboxTimeoutError('sed timed out', timeout=timeout)

    backend = TimedOutSed('timed-out-sed', {'/workspace/data.txt': b'one\ntwo\nthree\n'})

    window = await Sandbox(backend).read_file('data.txt', offset=2, limit=1)

    assert window.lines == ('two',)
    assert backend.fs.reads == ['/workspace/data.txt']


@pytest.mark.parametrize('kwargs', [{'offset': 0}, {'limit': 0}])
async def test_read_file_rejects_invalid_window_values(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        await Sandbox(FakeSandbox('invalid-window')).read_file('data.txt', **kwargs)


@pytest.mark.parametrize('offset', [1, 4])
async def test_bounded_read_returns_empty_window_at_or_past_empty_file(offset: int) -> None:
    backend = FakeSandbox('empty-file', {'/workspace/data.txt': b''})

    window = await Sandbox(backend).read_file('data.txt', offset=offset, limit=2)

    assert (window.lines, window.start_line, window.has_more, window.total_lines) == ((), offset, False, None)


async def test_bounded_read_reports_more_lines_only_when_the_window_is_short() -> None:
    backend = FakeSandbox('window', {'/workspace/data.txt': b'one\ntwo\nthree\n'})
    sandbox = Sandbox(backend)

    partial = await sandbox.read_file('data.txt', offset=1, limit=2)
    ending = await sandbox.read_file('data.txt', offset=2, limit=2)

    assert (partial.lines, partial.has_more, partial.total_lines) == (('one', 'two'), True, None)
    assert (ending.lines, ending.has_more, ending.total_lines) == (('two', 'three'), False, 3)


async def test_full_read_uses_filesystem_and_preserves_decoding_contracts() -> None:
    backend = FakeSandbox('full-read', {'/workspace/data.txt': b'one\ntwo\nthree'})
    sandbox = Sandbox(backend)

    window = await sandbox.read_file('data.txt', offset=2)

    assert (window.lines, window.has_more, window.total_lines) == (('two', 'three'), False, 3)
    assert window.text == 'two\nthree'
    assert backend.fs.reads == ['/workspace/data.txt']

    backend.fs.files['/workspace/bad.txt'] = b'one\xfftwo\n'
    assert (await sandbox.read_file('bad.txt')).lines == ('one�two',)
    with pytest.raises(UnicodeDecodeError):
        await sandbox.read_text('bad.txt')


async def test_bounded_read_through_read_only_sandbox_uses_filesystem() -> None:
    backend = FakeSandbox('read-only', {'/workspace/data.txt': b'one\ntwo\nthree\n'})
    sandbox = Sandbox(ReadOnlySandbox(backend))

    window = await sandbox.read_file('data.txt', offset=2, limit=1)

    assert window.lines == ('two',)
    assert backend.fs.reads == ['/workspace/data.txt']


async def test_unavailable_sandbox_uses_the_configured_reason_for_every_operation() -> None:
    reason = 'sandbox disabled by policy'
    backend = UnavailableSandbox(reason)
    # No environment exists, so there is no identity a later run could reconnect to.
    assert backend.ref is None
    operations = [
        backend.run(['true']),
        backend.working_dir(),
        backend.fs.read_bytes('/file'),
        backend.fs.write_bytes('/file', b'data'),
        backend.fs.stat('/file'),
        backend.fs.list_dir('/'),
        backend.fs.make_dir('/dir'),
        backend.fs.remove('/file'),
        backend.fs.exists('/file'),
    ]

    for operation in operations:
        with pytest.raises(UserError, match='sandbox disabled by policy'):
            await operation


async def test_bare_run_context_sandbox_explains_how_to_attach_one() -> None:
    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())

    with pytest.raises(UserError, match=r'pass `sandbox=`.*capability'):
        await ctx.sandbox.run(['true'])


async def test_caller_owned_only_marks_an_explicit_backend() -> None:
    capability = SandboxCapability()
    observed: list[bool] = []
    agent = Agent(_tool_call_model(), capabilities=[capability])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        observed.append(ctx.sandbox.caller_owned)
        return 'ok'

    await agent.run('capability')
    await agent.run('explicit', sandbox=FakeSandbox('explicit'))

    assert observed == [False, True]


async def test_explicit_backend_wins_over_a_capability_backend() -> None:
    capability = SandboxCapability()
    explicit = FakeSandbox('explicit')
    observed: list[Sandbox] = []
    agent = Agent(_tool_call_model(), capabilities=[capability])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        observed.append(ctx.sandbox)
        return 'ok'

    await agent.run('go', sandbox=explicit)

    assert observed[0].backend is explicit
    assert capability.refs == []


async def test_the_result_carries_the_sandbox_the_run_used() -> None:
    """`result.sandbox` is the same object tools saw, so a caller can keep working in it."""
    capability = SandboxCapability()
    observed: list[Sandbox] = []
    agent = Agent(_tool_call_model(), capabilities=[capability])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        observed.append(ctx.sandbox)
        return (await ctx.sandbox.run(['true'])).stdout

    result = await agent.run('go')

    assert result.sandbox is observed[0]
    assert result.sandbox.ref == SandboxRef(sandbox_id='fake-capability')

    # Handing it to a second run continues in the same environment rather than making a new one.
    second = await agent.run('again', sandbox=result.sandbox)
    assert second.sandbox is result.sandbox
    assert capability.refs == [None]
    assert capability.backend.create_calls == 1


async def test_a_result_still_round_trips_through_json_when_a_sandbox_was_used() -> None:
    """The sandbox is a live handle, so it is left out of the serialized result rather than breaking it."""
    agent = Agent(_tool_call_model(), capabilities=[SandboxCapability()])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        return (await ctx.sandbox.run(['true'])).stdout

    result = await agent.run('go')
    adapter = TypeAdapter(AgentRunResult[str])
    restored = adapter.validate_json(adapter.dump_json(result))

    assert restored == result
    with pytest.raises(UserError, match='created outside an agent run'):
        await restored.sandbox.run(['true'])


async def test_a_result_built_outside_a_run_explains_that_no_sandbox_is_attached() -> None:
    result = AgentRunResult[str]('output')

    with pytest.raises(UserError, match='created outside an agent run'):
        await result.sandbox.run(['true'])


async def test_two_capabilities_supplying_a_sandbox_name_both() -> None:
    """One run, one sandbox: a second supplier is a configuration mistake, not a silent winner."""

    class SecondSandboxCapability(SandboxCapability):
        id = 'second-sandbox'

    agent = Agent(TestModel(), capabilities=[SandboxCapability(), SecondSandboxCapability()])

    with pytest.raises(UserError, match='SandboxCapability and SecondSandboxCapability both did'):
        await agent.run('go')


async def test_deferred_capability_never_contributes_a_backend() -> None:
    capability = SandboxCapability()
    capability.defer_loading = True
    observed: list[Sandbox] = []
    agent = Agent(_tool_call_model(), capabilities=[capability])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        observed.append(ctx.sandbox)
        return 'ok'

    await agent.run('go')

    assert isinstance(observed[0].backend, UnavailableSandbox)
    assert capability.refs == []


def test_backend_construction_does_no_io() -> None:
    backend = FakeSandbox('lazy')

    assert backend.ref is None
    assert backend.create_calls == 0
    assert backend.attach_calls == 0


async def test_first_operation_creates_the_environment_once() -> None:
    backend = FakeSandbox('fresh')

    await backend.run(['true'])
    await backend.working_dir()

    assert backend.create_calls == 1
    assert backend.attach_calls == 0
    assert backend.ref is not None


async def test_concurrent_first_operations_create_one_environment() -> None:
    backend = FakeSandbox('concurrent')

    async with anyio.create_task_group() as tg:
        tg.start_soon(backend.run, ['true'])
        tg.start_soon(backend.working_dir)

    assert backend.create_calls == 1
    assert backend.ref is not None


async def test_create_backend_ref_is_set_after_its_first_operation() -> None:
    backend = FakeSandbox('identity')
    assert backend.ref is None

    await backend.run(['true'])

    assert backend.ref == SandboxRef(sandbox_id='fake-identity')


async def test_sandbox_ref_forwards_backend_identity() -> None:
    backend = FakeSandbox('ref')
    sandbox = Sandbox(backend)

    assert sandbox.ref is None
    await sandbox.run(['true'])

    assert sandbox.ref == SandboxRef(sandbox_id='fake-ref')


def test_sandbox_wrap_is_idempotent() -> None:
    backend = FakeSandbox('wrapped')
    sandbox = Sandbox.wrap(backend)

    assert isinstance(sandbox, Sandbox)
    assert sandbox.backend is backend
    assert Sandbox.wrap(sandbox) is sandbox


async def test_run_rejects_relative_cwd() -> None:
    with pytest.raises(ValueError, match='absolute'):
        await Sandbox(FakeSandbox('cwd')).run(['true'], cwd='relative')


async def test_make_unavailable_updates_existing_references() -> None:
    sandbox = Sandbox(FakeSandbox('unavailable'))
    existing_reference = sandbox
    reason = 'sandbox is no longer available'

    sandbox._make_unavailable(reason)  # pyright: ignore[reportPrivateUsage]

    assert existing_reference.backend is sandbox.backend
    operations = [
        existing_reference.run(['true']),
        existing_reference.working_dir(),
        existing_reference.read_bytes('file.txt'),
        existing_reference.write_bytes('file.txt', b'data'),
        existing_reference.stat('file.txt'),
        existing_reference.list_dir('.'),
        existing_reference.make_dir('dir'),
        existing_reference.remove('file.txt'),
        existing_reference.exists('file.txt'),
        existing_reference.read_file('file.txt', limit=1),
    ]
    for operation in operations:
        with pytest.raises(UserError, match='no longer available'):
            await operation


async def test_two_capabilities_cannot_supply_the_sandbox() -> None:
    class FirstSandboxCapability(AbstractCapability[Any]):
        def get_sandbox(self, ctx: RunContext[Any], *, ref: SandboxRef | None) -> SandboxBackend:
            return FakeSandbox('first')

    class SecondSandboxCapability(AbstractCapability[Any]):
        def get_sandbox(self, ctx: RunContext[Any], *, ref: SandboxRef | None) -> SandboxBackend:
            return FakeSandbox('second')

    agent = Agent(_tool_call_model(), capabilities=[FirstSandboxCapability(), SecondSandboxCapability()])

    with pytest.raises(UserError, match=r'FirstSandboxCapability.*SecondSandboxCapability'):
        await agent.run('go')


async def test_declining_capability_leaves_the_run_sandbox_unavailable() -> None:
    capability = DecliningSandboxCapability()
    agent = Agent(_tool_call_model(), capabilities=[capability])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        assert isinstance(ctx.sandbox.backend, UnavailableSandbox)
        await ctx.sandbox.run(['true'])
        return 'unreachable'  # pragma: no cover

    with pytest.raises(UserError, match='No sandbox is attached'):
        await agent.run('go')
    assert capability.calls == 1


async def test_unrecognized_sandbox_ref_is_rejected() -> None:
    agent = Agent(_tool_call_model(), capabilities=[DecliningSandboxCapability()])

    with pytest.raises(UserError, match="No capability can supply sandbox 'missing'"):
        await agent.run('go', sandbox=SandboxRef(sandbox_id='missing'))


async def test_capability_backend_is_available_without_connecting_during_run_setup() -> None:
    capability = SandboxCapability()
    seen: list[Sandbox] = []
    agent = Agent(_tool_call_model(), capabilities=[capability])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        seen.append(ctx.sandbox)
        return 'ok'

    await agent.run('go')

    assert seen[0].backend is capability.backend
    assert capability.backend.create_calls == 0


async def test_run_never_cleans_up_the_sandbox() -> None:
    backend = FakeSandbox('persistent')
    agent = Agent(_tool_call_model(), capabilities=[])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        await ctx.sandbox.run(['true'])
        return 'ok'

    await agent.run('go', sandbox=backend)

    assert backend.cleanup_calls == []


async def test_failed_run_never_cleans_up_the_sandbox() -> None:
    backend = FakeSandbox('failed')
    agent = Agent(_tool_call_model('explode'))

    @agent.tool
    async def explode(ctx: RunContext[Any]) -> str:
        await ctx.sandbox.run(['true'])
        raise RuntimeError('boom')

    with pytest.raises(RuntimeError, match='boom'):
        await agent.run('go', sandbox=backend)

    assert backend.cleanup_calls == []


async def test_cancelled_run_never_cleans_up_the_sandbox() -> None:
    backend = FakeSandbox('cancelled')
    agent = Agent(_tool_call_model())
    entered = anyio.Event()

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        await ctx.sandbox.run(['true'])
        entered.set()
        await anyio.sleep(60)
        return 'unreachable'  # pragma: no cover

    async with anyio.create_task_group() as tg:

        async def run_agent() -> None:
            await agent.run('go', sandbox=backend)

        tg.start_soon(run_agent)
        await entered.wait()
        tg.cancel_scope.cancel()

    assert backend.cleanup_calls == []


async def test_guard_workflow_sandbox_only_rejects_a_live_handle() -> None:
    ref = SandboxRef(sandbox_id='existing')

    assert guard_workflow_sandbox(ref, live_error='live sandbox') is ref
    assert guard_workflow_sandbox(None, live_error='live sandbox') is None
    with pytest.raises(UserError, match='live sandbox'):
        guard_workflow_sandbox(FakeSandbox('live'), live_error='live sandbox')


async def test_capability_can_supply_a_backend_for_an_explicit_ref() -> None:
    capability = ConnectOnlySandboxCapability()
    agent = Agent(_tool_call_model(), capabilities=[capability])

    @agent.tool
    async def probe(ctx: RunContext[Any]) -> str:
        return (await ctx.sandbox.run(['true'])).stdout

    result: AgentRunResult[Any] = await agent.run('go', sandbox=SandboxRef(sandbox_id='existing'))

    assert result.output == 'done'
    assert capability.sandbox_ids == ['existing']
