"""Public E2B workspace capability tests."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import anyio
import pytest
from e2b.exceptions import SandboxException
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import ReadOnlyWorkspace, Workspace, WorkspaceError, WorkspaceReadOnlyError, WorkspaceRef

from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.e2b_sandbox import E2BSandbox, E2BSandboxBackend

from .._tool_calls import call_tools
from .fake_e2b import FakeE2B

pytestmark = pytest.mark.anyio


def test_env_is_not_shown_in_repr() -> None:
    secret = 'sensitive-credential-value'
    capability = E2BSandbox(env={'TOKEN': secret})
    assert secret not in repr(capability)
    assert secret not in str(capability)
    assert secret not in repr(E2BSandboxBackend(env={'TOKEN': secret}))


async def test_destroy_ref_without_attaching(fake_e2b: FakeE2B) -> None:
    provider = E2BSandbox()
    fake_e2b.new_sandbox('sbx-keep')
    backend = provider.backend(WorkspaceRef(provider='e2b', id='sbx-keep'))
    assert isinstance(backend, E2BSandboxBackend)
    assert backend.ref is not None
    await provider.destroy(backend.ref)
    assert fake_e2b.sandboxes[0].killed
    assert not fake_e2b.connect_calls


def test_capability_uses_workspace_contract() -> None:
    capability = E2BSandbox()
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    backend = capability.get_workspace(ctx, ref=WorkspaceRef(provider='e2b', id='known'))
    assert isinstance(backend, E2BSandboxBackend)
    assert backend.ref == WorkspaceRef(provider='e2b', id='known')


def test_foreign_reference_is_rejected() -> None:
    with pytest.raises(ValueError, match="expected 'e2b'"):
        E2BSandboxBackend(ref=WorkspaceRef(provider='modal', id='other'))


@pytest.mark.parametrize(
    ('settings', 'message'),
    [
        ({'working_dir': 'repo'}, "working_dir must be an absolute POSIX path or None, got 'repo'"),
        ({'sandbox_timeout': 0}, 'sandbox_timeout must be an integer of at least 1, got 0'),
        ({'sandbox_timeout': 1.5}, 'sandbox_timeout must be an integer of at least 1, got 1.5'),
        ({'defer_loading': True, 'id': 'e2b'}, '`defer_loading` is not supported on `E2BSandbox`'),
    ],
)
def test_invalid_settings_fail_at_construction(settings: dict[str, object], message: str) -> None:
    with pytest.raises(UserError, match=re.escape(message)):
        E2BSandbox(**settings)  # pyright: ignore[reportArgumentType]


def test_capability_declines_foreign_reference() -> None:
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    assert E2BSandbox().get_workspace(ctx, ref=WorkspaceRef(provider='modal', id='other')) is None


async def test_native_handle_is_exposed_without_creation(fake_e2b: FakeE2B) -> None:
    seed_backend = E2BSandboxBackend()
    native = await seed_backend.get_sandbox()
    fake_e2b.create_calls.clear()
    backend = E2BSandboxBackend(sandbox=native)
    assert backend.ref == WorkspaceRef(provider='e2b', id='sbx-1')
    assert await backend.get_sandbox() is native
    assert not fake_e2b.create_calls
    with pytest.raises(ValueError, match='either `sandbox` or `ref`'):
        E2BSandboxBackend(sandbox=native, ref=WorkspaceRef(provider='e2b', id='other'))


async def test_agent_without_workspace_use_does_not_create(fake_e2b: FakeE2B) -> None:
    agent = Agent(TestModel(custom_output_text='done'), capabilities=[E2BSandbox()])
    result = await agent.run('hello')
    assert result.output == 'done'
    assert not fake_e2b.create_calls


async def test_concurrent_first_use_creates_once(fake_e2b: FakeE2B) -> None:
    backend = E2BSandboxBackend()
    results: list[object] = []

    async def acquire() -> None:
        results.append(await backend.get_sandbox())

    async with anyio.create_task_group() as tg:
        tg.start_soon(acquire)
        tg.start_soon(acquire)

    assert len(fake_e2b.create_calls) == 1
    assert results[0] is results[1]


async def test_capability_forwards_creation_options_and_run_does_not_kill(fake_e2b: FakeE2B) -> None:
    capability = E2BSandbox(
        template='base',
        sandbox_timeout=120,
        working_dir='/work',
        env={'FOO': 'bar'},
        allow_internet_access=False,
    )
    agent = Agent(TestModel(call_tools=['run_command']), capabilities=[capability])

    @agent.tool
    async def run_command(ctx: RunContext[object]) -> str:
        return (await ctx.workspace.run(['echo', 'ok'])).stdout

    result = await agent.run('run', deps=None)
    assert 'ok' in result.output
    call = fake_e2b.create_calls[0]
    assert (call.template, call.timeout, call.envs, call.allow_internet_access) == ('base', 120, {'FOO': 'bar'}, False)
    assert fake_e2b.sandboxes[0].commands.calls[-1].cwd == '/work'
    assert fake_e2b.sandboxes[0].killed is False
    assert isinstance(result.workspace, Workspace)
    backend = result.workspace.backend
    assert isinstance(backend, E2BSandboxBackend)
    native = await backend.get_sandbox()
    await native.kill()
    assert fake_e2b.sandboxes[0].killed is True


async def test_explicit_ref_uses_attach_even_with_creation_options(fake_e2b: FakeE2B) -> None:
    capability = E2BSandbox(template='base', env={'FOO': 'bar'})
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    backend = capability.get_workspace(ctx, ref=WorkspaceRef(provider='e2b', id='existing'))
    assert isinstance(backend, E2BSandboxBackend)
    await backend.get_sandbox()
    assert fake_e2b.connect_calls == [('existing', 3_600)]
    assert not fake_e2b.create_calls


async def test_post_acquisition_operations_overlap(fake_e2b: FakeE2B) -> None:
    backend = E2BSandboxBackend()
    await backend.get_sandbox()
    fake_e2b.command_hangs = True

    async def run_command() -> None:
        await backend.run(['sleep', '1'])

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_command)
        tg.start_soon(run_command)
        await anyio.wait_all_tasks_blocked()
        assert len(fake_e2b.sandboxes[0].commands.calls) == 2
        tg.cancel_scope.cancel()
    calls = fake_e2b.sandboxes[0].commands.calls
    assert len([call for call in calls if call.command.startswith('setsid sh -c ')]) == 2
    assert len(fake_e2b.sandboxes[0].commands.group_stops) == 2


async def test_agent_preserves_explicit_read_only_workspace(fake_e2b: FakeE2B) -> None:
    backend = E2BSandboxBackend()
    facade = ReadOnlyWorkspace(Workspace(backend))
    agent = Agent(TestModel(call_tools=['check_workspace']), capabilities=[E2BSandbox()])

    @agent.tool
    async def check_workspace(ctx: RunContext[object]) -> str:
        assert ctx.workspace is facade
        with pytest.raises(WorkspaceReadOnlyError, match='read-only'):
            await ctx.workspace.run(['echo', 'blocked'])
        return 'checked'

    result = await agent.run('check', workspace=facade)
    assert 'checked' in result.output
    assert result.workspace is facade
    assert not fake_e2b.create_calls


async def test_failed_acquisition_can_retry(fake_e2b: FakeE2B) -> None:
    backend = E2BSandboxBackend()
    fake_e2b.create_error = SandboxException('temporary')
    with pytest.raises(WorkspaceError, match='temporary'):
        await backend.get_sandbox()
    fake_e2b.create_error = None
    assert await backend.get_sandbox() is fake_e2b.sandboxes[0]
    assert len(fake_e2b.create_calls) == 2


@pytest.mark.parametrize('anyio_backend', ['asyncio', 'trio'])
async def test_cancelled_creation_keeps_the_sandbox_it_made(fake_e2b: FakeE2B, anyio_backend: str) -> None:
    # E2B can make the sandbox before its response arrives. A caller cancelled in between still
    # records it, so `ref` names it and a retry reuses it instead of creating a second one.
    fake_e2b.create_response_held = held = anyio.Event()
    backend = E2BSandboxBackend()
    cancelled: list[bool] = []

    async def acquire() -> None:
        try:
            await backend.get_sandbox()
        except anyio.get_cancelled_exc_class():
            cancelled.append(True)
            raise

    async with anyio.create_task_group() as tg:
        tg.start_soon(acquire)
        with anyio.fail_after(1):
            while not fake_e2b.sandboxes:
                await anyio.sleep(0)
        tg.cancel_scope.cancel()
        held.set()

    assert cancelled == [True]
    assert backend.ref == WorkspaceRef(provider='e2b', id='sbx-1')
    assert await backend.get_sandbox() is fake_e2b.sandboxes[0]
    assert len(fake_e2b.create_calls) == 1


async def test_native_cancellation_during_creation_retains_sandbox(fake_e2b: FakeE2B) -> None:
    fake_e2b.create_response_held = held = anyio.Event()
    backend = E2BSandboxBackend()
    first = asyncio.create_task(backend.get_sandbox())
    with anyio.fail_after(5):
        while not fake_e2b.sandboxes:
            await anyio.sleep(0)
        first.cancel()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        held.set()
        assert await backend.get_sandbox() is fake_e2b.sandboxes[0]
    assert backend.ref == WorkspaceRef(provider='e2b', id='sbx-1')
    assert len(fake_e2b.create_calls) == 1


async def test_agent_runs_without_history_create_fresh_workspaces(fake_e2b: FakeE2B) -> None:
    agent = Agent(TestModel(call_tools=['run_command']), capabilities=[E2BSandbox()])

    @agent.tool
    async def run_command(ctx: RunContext[object]) -> str:
        return (await ctx.workspace.run(['printf', 'ok'])).stdout

    await agent.run('go')
    await agent.run('go')
    assert len(fake_e2b.sandboxes) == 2


async def test_agent_history_attaches_same_workspace(fake_e2b: FakeE2B) -> None:
    def model(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        if isinstance(request, ModelRequest) and any(isinstance(part, ToolReturnPart) for part in request.parts):
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='run_command', args={}, tool_call_id='call')])

    agent = Agent(FunctionModel(model), capabilities=[E2BSandbox()])
    tool_results: list[str] = []

    @agent.tool
    async def run_command(ctx: RunContext[object]) -> str:
        if await ctx.workspace.exists('/marker.txt'):
            result = await ctx.workspace.read_text('/marker.txt')
        else:
            await ctx.workspace.write_text('/marker.txt', 'persisted')
            result = 'created'
        tool_results.append(result)
        return result

    first = await agent.run('go')
    second = await agent.run('go', message_history=first.all_messages())
    assert first.output == 'done'
    assert second.output == 'done'
    assert tool_results == ['created', 'persisted']
    assert len(fake_e2b.sandboxes) == 1
    assert first.workspace.ref is not None
    assert fake_e2b.connect_calls == [('sbx-1', 3_600)]


async def test_coder_works_in_the_sandbox(fake_e2b: FakeE2B, tmp_path: Path) -> None:
    # Host mode: the fake sandbox runs commands and file calls under `tmp_path`, so the model's
    # shell and file tools are observable end to end.
    fake_e2b.host_root = tmp_path.resolve()
    shell_output, read_output = await call_tools(
        [E2BSandbox(), Coder()],
        [('shell', {'command': 'echo made-in-sandbox > note.txt'}), ('read_file', {'path': 'note.txt'})],
    )
    assert '"exit_code": 0' in shell_output
    assert 'made-in-sandbox' in read_output
    assert (tmp_path / 'note.txt').read_text() == 'made-in-sandbox\n'
    assert len(fake_e2b.sandboxes) == 1
