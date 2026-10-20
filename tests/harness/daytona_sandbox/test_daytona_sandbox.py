"""Public Daytona workspace capability tests."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
import daytona
import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import ReadOnlyWorkspace, Workspace, WorkspaceReadOnlyError, WorkspaceRef
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox, DaytonaSandboxBackend
from pydantic_ai_harness.daytona_sandbox._backend import _close_client  # pyright: ignore[reportPrivateUsage]

from .._tool_calls import call_tools
from .fake_daytona import FakeDaytona

pytestmark = pytest.mark.anyio(backends=['asyncio'])


def test_capability_repr_hides_command_environment() -> None:
    assert 'secret-sentinel' not in repr(DaytonaSandbox(env={'TOKEN': 'secret-sentinel'}))


def _ctx() -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage())


async def test_capability_is_lazy_and_forwards_options(fake_daytona: FakeDaytona) -> None:
    capability = DaytonaSandbox(
        snapshot='python',
        auto_stop_interval=0,
        working_dir='/work',
        env={'A': 'b'},
        allow_internet_access=False,
    )
    backend = capability.get_workspace(_ctx(), ref=None)
    assert isinstance(backend, DaytonaSandboxBackend)
    assert not fake_daytona.create_params
    assert await backend.working_dir() == '/work'
    params = fake_daytona.create_params[0]
    assert (
        params.snapshot,
        params.auto_stop_interval,
        params.env_vars,
        params.network_block_all,
    ) == ('python', 0, {'A': 'b'}, True)


@pytest.mark.parametrize(
    ('settings', 'message'),
    [
        ({'working_dir': 'repo'}, 'working_dir must be an absolute POSIX path'),
        ({'auto_stop_interval': -1}, 'auto_stop_interval must be an integer of at least 0'),
        ({'auto_stop_interval': 1.5}, 'auto_stop_interval must be an integer of at least 0'),
        ({'defer_loading': True, 'id': 'sandbox'}, 'does not support `defer_loading=True`'),
    ],
)
def test_invalid_settings_fail_at_construction(settings: dict[str, Any], message: str) -> None:
    with pytest.raises(UserError, match=message):
        DaytonaSandbox(**settings)


@pytest.mark.parametrize('env', [{'PORT': 3000}, {1: 'one'}, {'TOKEN': None}])
def test_non_string_env_fails_at_construction(env: dict[object, object]) -> None:
    with pytest.raises(TypeError, match='env keys and values must be strings'):
        DaytonaSandboxBackend(env=env)  # pyright: ignore[reportArgumentType]


async def test_explicit_ref_attaches_without_creation(fake_daytona: FakeDaytona) -> None:
    existing = fake_daytona.sandbox('existing')
    backend = DaytonaSandboxBackend(ref=WorkspaceRef(provider='daytona', id=existing.id))
    await backend.get_sandbox()
    assert backend.ref == WorkspaceRef(provider='daytona', id=existing.id)
    assert not fake_daytona.create_params


async def test_foreign_ref_is_declined_and_backend_rejects_it() -> None:
    assert DaytonaSandbox().get_workspace(_ctx(), ref=WorkspaceRef(provider='other', id='x')) is None
    with pytest.raises(ValueError, match="expected 'daytona'"):
        DaytonaSandboxBackend(ref=WorkspaceRef(provider='other', id='x'))


async def test_destroy_rejects_a_foreign_or_blank_ref(fake_daytona: FakeDaytona) -> None:
    with pytest.raises(ValueError, match=r"^unsupported workspace provider 'other'; expected 'daytona'$"):
        await DaytonaSandbox().destroy(WorkspaceRef(provider='other', id='x'))
    with pytest.raises(ValueError, match=r'^Daytona workspace ref id cannot be empty$'):
        await DaytonaSandbox().destroy(WorkspaceRef(provider='daytona', id=' '))
    assert fake_daytona.opened_clients == 0


async def test_destroy_with_a_supplied_client_leaves_it_open(fake_daytona: FakeDaytona) -> None:
    sandbox = fake_daytona.sandbox()
    capability = DaytonaSandbox[None](client=daytona.AsyncDaytona())
    await capability.destroy(WorkspaceRef(provider='daytona', id=sandbox.id))
    assert sandbox.state == daytona.SandboxState.DESTROYING
    assert (fake_daytona.opened_clients, fake_daytona.closed_clients) == (1, 0)


async def test_backend_for_a_ref_attaches_with_the_capability_settings(fake_daytona: FakeDaytona) -> None:
    existing = fake_daytona.sandbox('existing')
    capability = DaytonaSandbox[None](working_dir='/work', env={'A': 'b'})
    backend = capability.backend(WorkspaceRef(provider='daytona', id=existing.id))
    assert await backend.working_dir() == '/work'
    await backend.run(['true'])
    assert 'A=b' in existing.process_command
    assert not fake_daytona.create_params


async def test_native_ref_conflict_and_native_identity(fake_daytona: FakeDaytona) -> None:
    seed = DaytonaSandboxBackend()
    native = await seed.get_sandbox()
    backend = DaytonaSandboxBackend(sandbox=native)
    assert await backend.get_sandbox() is native
    assert backend.ref == WorkspaceRef(provider='daytona', id=native.id)
    with pytest.raises(ValueError, match='either `sandbox` or `ref`'):
        DaytonaSandboxBackend(sandbox=native, ref=backend.ref)


async def test_agent_without_workspace_use_does_not_create(fake_daytona: FakeDaytona) -> None:
    result = await Agent(TestModel(custom_output_text='done'), capabilities=[DaytonaSandbox()]).run('hello')
    assert result.output == 'done'
    assert not fake_daytona.create_params


async def test_agent_runs_create_fresh_resources_without_history(fake_daytona: FakeDaytona) -> None:
    agent = Agent(TestModel(call_tools=['run_command']), capabilities=[DaytonaSandbox()])

    @agent.tool
    async def run_command(ctx: RunContext[object]) -> str:
        return (await ctx.workspace.run(['echo', 'ok'])).stdout

    await agent.run('run')
    await agent.run('run')
    assert len(fake_daytona.create_params) == 2


async def test_agent_history_attaches_persisted_workspace(fake_daytona: FakeDaytona) -> None:
    def model(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        if isinstance(messages[-1], ModelRequest) and any(isinstance(p, ToolReturnPart) for p in messages[-1].parts):
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='remember', args={}, tool_call_id='call')])

    agent = Agent(FunctionModel(model), capabilities=[DaytonaSandbox()])

    @agent.tool
    async def remember(ctx: RunContext[object]) -> str:
        if await ctx.workspace.exists('/marker'):
            return (await ctx.workspace.read_bytes('/marker')).decode()
        await ctx.workspace.write_bytes('/marker', b'saved')
        return 'created'

    first = await agent.run('remember')
    second = await agent.run('continue', message_history=first.all_messages())
    assert 'done' in first.output and 'done' in second.output
    assert fake_daytona.create_params and len(fake_daytona.create_params) == 1
    assert len(fake_daytona.sandboxes) == 1
    assert fake_daytona.sandboxes[0].start_calls == [60]


async def test_read_only_facade_identity_and_denied_command(fake_daytona: FakeDaytona) -> None:
    backend = DaytonaSandboxBackend()
    facade = ReadOnlyWorkspace(Workspace(backend))
    agent = Agent(TestModel(call_tools=['check']), capabilities=[DaytonaSandbox()])

    @agent.tool
    async def check(ctx: RunContext[object]) -> str:
        assert ctx.workspace is facade
        with pytest.raises(WorkspaceReadOnlyError, match='read-only'):
            await ctx.workspace.run(['echo', 'blocked'])
        return 'ok'

    result = await agent.run('check', workspace=facade)
    assert result.workspace is facade
    assert 'ok' in result.output
    assert not fake_daytona.create_params


async def test_concurrent_acquisition_and_operation_overlap(fake_daytona: FakeDaytona) -> None:
    backend = DaytonaSandboxBackend()
    results: list[object] = []

    async def acquire() -> None:
        results.append(await backend.get_sandbox())

    async with anyio.create_task_group() as tg:
        tg.start_soon(acquire)
        tg.start_soon(acquire)
    assert len(fake_daytona.create_params) == 1
    assert results[0] is results[1]


async def _touch(ctx: RunContext[None]) -> str:
    await ctx.workspace.write_bytes('/note', b'kept')
    return 'ok'


async def test_cancelled_run_closes_owned_client_mid_command(fake_daytona: FakeDaytona) -> None:
    async def long_command(ctx: RunContext[None]) -> str:
        return (await ctx.workspace.run(['sleep', '30'])).stdout

    agent = Agent(
        TestModel(call_tools=['long_command']),
        deps_type=type(None),
        tools=[long_command],
        capabilities=[DaytonaSandbox()],
    )
    task = asyncio.create_task(agent.run('go'))
    with anyio.fail_after(5):
        while not fake_daytona.sandboxes:
            await asyncio.sleep(0)
        sandbox = fake_daytona.sandboxes[0]
        sandbox.process_hangs = True
        await sandbox.process_status_polled.wait()
    task.cancel()
    # Promptly: the run's client close must not wait on the command it is cancelling.
    with anyio.fail_after(10), pytest.raises(asyncio.CancelledError):
        await task
    assert fake_daytona.closed_clients == 1
    assert sandbox.process_sessions == set()


async def test_closing_a_client_waits_for_its_cancelled_event_connection() -> None:
    connecting = asyncio.create_task(asyncio.Event().wait())

    class Client:
        _event_dispatcher = SimpleNamespace(_connect_task=connecting)

        async def close(self) -> None:
            connecting.cancel()  # what `AsyncDaytona.close()` does, without waiting

    await _close_client(Client())  # pyright: ignore[reportArgumentType]
    assert connecting.cancelled()


async def test_run_closes_the_client_it_opened_and_the_result_workspace_reopens(fake_daytona: FakeDaytona) -> None:
    agent = Agent(
        TestModel(call_tools=['_touch']), deps_type=type(None), tools=[_touch], capabilities=[DaytonaSandbox()]
    )
    result = await agent.run('go')
    # The run keeps one client across its operations and closes it at the end.
    assert (fake_daytona.opened_clients, fake_daytona.closed_clients) == (1, 1)

    # After the run nothing would close the backend, so each operation closes its own client.
    assert await result.workspace.read_bytes('/note') == b'kept'
    assert (fake_daytona.opened_clients, fake_daytona.closed_clients) == (2, 2)
    assert len(fake_daytona.sandboxes) == 1


async def test_backend_used_outside_a_run_closes_its_client_after_each_operation(fake_daytona: FakeDaytona) -> None:
    # What a Temporal activity does: rebuild the backend from the ref, use it, and drop it.
    backend = DaytonaSandbox[None]().get_workspace(_ctx(), ref=None)
    assert isinstance(backend, DaytonaSandboxBackend)
    await backend.write_bytes('/note', b'hi')
    assert (fake_daytona.opened_clients, fake_daytona.closed_clients) == (1, 1)

    attached = DaytonaSandbox[None]().get_workspace(_ctx(), ref=backend.ref)
    assert isinstance(attached, DaytonaSandboxBackend)
    # One operation, however many requests it makes, and concurrent operations share one client,
    # closed once the last of them ends.
    first, second = await asyncio.gather(attached.read_bytes('/note'), attached.exists('/note'))
    assert (first, second) == (b'hi', True)
    assert (fake_daytona.opened_clients, fake_daytona.closed_clients) == (2, 2)


@pytest.mark.parametrize('anyio_backend', ['trio'])
async def test_trio_is_refused_before_any_daytona_call(fake_daytona: FakeDaytona) -> None:
    """The Daytona SDK runs its calls on asyncio tasks, so Trio gets a clear `UserError` instead."""
    message = r'^Daytona needs the asyncio event loop: the Daytona SDK runs its calls on asyncio tasks\.$'
    backend = DaytonaSandbox[None]().get_workspace(_ctx(), ref=None)
    assert isinstance(backend, DaytonaSandboxBackend)
    with pytest.raises(UserError, match=message):
        await backend.run(['true'])
    with pytest.raises(UserError, match=message):
        await backend.read_bytes('/file')
    with pytest.raises(UserError, match=message):
        await backend.get_sandbox()
    with pytest.raises(UserError, match=message):
        await DaytonaSandbox[None]().destroy(WorkspaceRef(provider='daytona', id='target'))
    await backend.aclose()
    assert fake_daytona.opened_clients == 0


@pytest.mark.parametrize('supplied', ['client', 'workspace'])
async def test_run_leaves_a_supplied_client_or_workspace_open(fake_daytona: FakeDaytona, supplied: str) -> None:
    capability = DaytonaSandbox[None](client=daytona.AsyncDaytona()) if supplied == 'client' else DaytonaSandbox[None]()
    workspace = DaytonaSandboxBackend() if supplied == 'workspace' else None
    agent = Agent(TestModel(call_tools=['_touch']), deps_type=type(None), tools=[_touch], capabilities=[capability])
    await agent.run('go', workspace=workspace)
    assert fake_daytona.closed_clients == 0


async def test_a_child_run_leaves_the_parent_runs_client_open(fake_daytona: FakeDaytona) -> None:
    child = Agent(
        TestModel(call_tools=['_touch']), deps_type=type(None), tools=[_touch], capabilities=[DaytonaSandbox()]
    )
    closed_after_child: list[int] = []

    async def delegate(ctx: RunContext[None]) -> str:
        await child.run('go', workspace=ctx.workspace)
        closed_after_child.append(fake_daytona.closed_clients)
        return 'ok'

    parent = Agent(
        TestModel(call_tools=['_touch', 'delegate']),
        deps_type=type(None),
        tools=[_touch, delegate],
        capabilities=[DaytonaSandbox()],
    )
    await parent.run('go')
    assert closed_after_child == [0]
    assert fake_daytona.closed_clients == 1


async def test_coder_works_in_the_sandbox(fake_daytona: FakeDaytona, tmp_path: Path) -> None:
    fake_daytona.host_root = tmp_path.resolve()
    results = await call_tools(
        [DaytonaSandbox(), Coder()],
        [
            ('write_file', {'path': 'hello.py', 'content': "print('hi from the sandbox')\n"}),
            ('shell', {'command': 'python3 hello.py'}),
        ],
    )
    assert 'hi from the sandbox' in results[1]
    assert (tmp_path / 'hello.py').read_text() == "print('hi from the sandbox')\n"
