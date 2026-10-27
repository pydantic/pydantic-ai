"""Public Vercel Sandbox capability tests."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

import anyio
import pytest
from vercel import sandbox as vercel_sdk

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import WorkspaceError, WorkspaceRef
from pydantic_ai_harness.vercel_sandbox import VercelSandbox, VercelSandboxBackend

from .fake_vercel import FakeVercel

pytestmark = pytest.mark.anyio(backends=['asyncio'])


def _ctx() -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage())


def test_capability_repr_hides_command_environment() -> None:
    assert 'secret-sentinel' not in repr(VercelSandbox(env={'TOKEN': 'secret-sentinel'}))


@pytest.mark.parametrize(
    ('settings', 'message'),
    [
        ({'working_dir': 'repo'}, 'working_dir must be an absolute POSIX path'),
        ({'sandbox_timeout': 0}, 'sandbox_timeout must be an integer of at least 1'),
        ({'sandbox_timeout': 1.5}, 'sandbox_timeout must be an integer of at least 1'),
        ({'env': {'PORT': 3000}}, '`env` keys and values must be strings'),
        ({'defer_loading': True, 'id': 'sandbox'}, 'does not support `defer_loading=True`'),
    ],
)
def test_invalid_settings_fail_at_construction(settings: dict[str, Any], message: str) -> None:
    with pytest.raises(UserError, match=message):
        VercelSandbox(**settings)


async def test_capability_is_lazy_and_forwards_options(fake_vercel: FakeVercel) -> None:
    working_dir = str(fake_vercel.host_root / 'project')
    capability = VercelSandbox[None](
        image='vercel/sandbox/ubuntu',
        sandbox_timeout=600,
        allow_internet_access=False,
        working_dir=working_dir,
        env={'A': 'b'},
    )
    backend = capability.get_workspace(_ctx(), ref=None)
    assert isinstance(backend, VercelSandboxBackend)
    assert not fake_vercel.create_calls
    assert await backend.working_dir() == working_dir
    [call] = fake_vercel.create_calls
    assert call == {
        'name': call['name'],
        'image': 'vercel/sandbox/ubuntu',
        'execution_time_limit': timedelta(seconds=600),
        'persistent': True,
        'network_policy': vercel_sdk.NetworkPolicy.deny_all(),
        'env': {'A': 'b'},
    }
    assert backend.ref == WorkspaceRef(provider='vercel', id=call['name'])


async def test_foreign_ref_is_declined_and_backend_rejects_it() -> None:
    assert VercelSandbox().get_workspace(_ctx(), ref=WorkspaceRef(provider='other', id='x')) is None
    with pytest.raises(ValueError, match="expected 'vercel'"):
        VercelSandboxBackend(ref=WorkspaceRef(provider='other', id='x'))


async def test_backend_for_a_ref_attaches_with_the_capability_settings(fake_vercel: FakeVercel) -> None:
    existing = fake_vercel.sandbox('existing')
    (fake_vercel.host_root / 'work').mkdir()
    capability = VercelSandbox[None](working_dir=str(fake_vercel.host_root / 'work'), env={'A': 'b'})
    backend = capability.backend(WorkspaceRef(provider='vercel', id='existing'))
    result = await backend.run(['sh', '-c', 'printf %s "$A"; pwd -P'])
    assert result.stdout == f'b{fake_vercel.host_root}/work\n'
    assert existing.command_envs == [{'A': 'b'}]
    assert not fake_vercel.create_calls


async def test_destroy_deletes_the_sandbox_and_its_snapshots(fake_vercel: FakeVercel) -> None:
    sandbox = fake_vercel.sandbox()
    await VercelSandbox().destroy(WorkspaceRef(provider='vercel', id=sandbox.name))
    assert sandbox.destroy_calls == [True]
    # The client opened for the call is closed again.
    assert (fake_vercel.opened_clients, fake_vercel.closed_clients) == (1, 1)
    # A sandbox that is already gone returns quietly.
    await VercelSandbox().destroy(WorkspaceRef(provider='vercel', id=sandbox.name))


async def test_destroy_with_a_supplied_client_leaves_it_open(fake_vercel: FakeVercel) -> None:
    sandbox = fake_vercel.sandbox()
    capability = VercelSandbox[None](client=vercel_sdk.SandboxClient.create())
    await capability.destroy(WorkspaceRef(provider='vercel', id=sandbox.name))
    assert sandbox.deleted
    assert (fake_vercel.opened_clients, fake_vercel.closed_clients) == (1, 0)


async def test_destroy_rejects_a_foreign_or_blank_ref(fake_vercel: FakeVercel) -> None:
    with pytest.raises(ValueError, match=r"^unsupported workspace provider 'other'; expected 'vercel'$"):
        await VercelSandbox().destroy(WorkspaceRef(provider='other', id='x'))
    with pytest.raises(ValueError, match=r'^Vercel workspace ref id cannot be empty$'):
        await VercelSandbox().destroy(WorkspaceRef(provider='vercel', id=' '))
    assert fake_vercel.opened_clients == 0


async def test_agent_without_workspace_use_does_not_create(fake_vercel: FakeVercel) -> None:
    result = await Agent(TestModel(custom_output_text='done'), capabilities=[VercelSandbox()]).run('hello')
    assert result.output == 'done'
    assert not fake_vercel.create_calls


async def _touch(ctx: RunContext[None]) -> str:
    await ctx.workspace.write_bytes('note', b'kept')
    return 'ok'


async def test_run_closes_the_client_it_opened_and_the_result_workspace_reopens(fake_vercel: FakeVercel) -> None:
    agent = Agent(
        TestModel(call_tools=['_touch']), deps_type=type(None), tools=[_touch], capabilities=[VercelSandbox()]
    )
    result = await agent.run('go')
    # The run keeps one client across its operations and closes it at the end.
    assert (fake_vercel.opened_clients, fake_vercel.closed_clients) == (1, 1)

    # After the run nothing would close the backend, so each operation closes its own client.
    assert await result.workspace.read_bytes('note') == b'kept'
    assert (fake_vercel.opened_clients, fake_vercel.closed_clients) == (2, 2)
    assert len(fake_vercel.sandboxes) == 1


async def test_agent_history_attaches_persisted_workspace(fake_vercel: FakeVercel) -> None:
    def model(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        if isinstance(messages[-1], ModelRequest) and any(isinstance(p, ToolReturnPart) for p in messages[-1].parts):
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='remember', args={}, tool_call_id='call')])

    agent = Agent(FunctionModel(model), capabilities=[VercelSandbox()])

    @agent.tool
    async def remember(ctx: RunContext[object]) -> str:
        if await ctx.workspace.exists('marker'):
            return (await ctx.workspace.read_bytes('marker')).decode()
        await ctx.workspace.write_bytes('marker', b'saved')
        return 'created'

    first = await agent.run('remember')
    second = await agent.run('continue', message_history=first.all_messages())
    assert 'done' in first.output and 'done' in second.output
    assert len(fake_vercel.create_calls) == 1
    # The second run looked the sandbox up by the name in the history's ref.
    assert fake_vercel.get_calls == [fake_vercel.create_calls[0]['name']]


@pytest.mark.parametrize('supplied', ['client', 'workspace'])
async def test_run_leaves_a_supplied_client_or_workspace_open(fake_vercel: FakeVercel, supplied: str) -> None:
    capability = (
        VercelSandbox[None](client=vercel_sdk.SandboxClient.create()) if supplied == 'client' else VercelSandbox[None]()
    )
    workspace = VercelSandboxBackend() if supplied == 'workspace' else None
    agent = Agent(TestModel(call_tools=['_touch']), deps_type=type(None), tools=[_touch], capabilities=[capability])
    await agent.run('go', workspace=workspace)
    assert (fake_vercel.opened_clients, fake_vercel.closed_clients) == (1, 0)
    if workspace is not None:
        await workspace.aclose()
        assert fake_vercel.closed_clients == 1


async def test_backend_used_outside_a_run_closes_its_client_after_each_operation(fake_vercel: FakeVercel) -> None:
    # What a Temporal activity does: rebuild the backend from the ref, use it, and drop it.
    backend = VercelSandbox[None]().get_workspace(_ctx(), ref=None)
    assert isinstance(backend, VercelSandboxBackend)
    assert (await backend.run(['true'])).exit_code == 0
    assert (fake_vercel.opened_clients, fake_vercel.closed_clients) == (1, 1)

    attached = VercelSandbox[None]().get_workspace(_ctx(), ref=backend.ref)
    assert isinstance(attached, VercelSandboxBackend)
    # Concurrent operations share one client, closed once the last of them ends.
    first, second = await asyncio.gather(attached.run(['true']), attached.working_dir())
    assert (first.exit_code, second) == (0, str(fake_vercel.host_root))
    assert (fake_vercel.opened_clients, fake_vercel.closed_clients) == (2, 2)


async def test_a_client_closed_during_a_direct_get_sandbox_skips_the_sandbox_lookup(fake_vercel: FakeVercel) -> None:
    """A direct `get_sandbox()` is not counted as an operation, so the last operation ending while it
    makes the working directory closes the client. A signal exit then goes without the sandbox lookup."""
    backend = VercelSandbox[None](working_dir=str(fake_vercel.host_root / 'project')).get_workspace(_ctx(), ref=None)
    assert isinstance(backend, VercelSandboxBackend)
    # `mkdir` dies by a signal, so its exit would be checked against the sandbox.
    fake_vercel.rewrite = lambda argv: ['sh', '-c', 'kill -TERM $$'] if argv[0] == 'mkdir' else argv
    fake_vercel.process_gate = asyncio.Event()
    acquiring = asyncio.create_task(backend.get_sandbox())
    await fake_vercel.process_requested.wait()
    # An operation waiting for the sandbox, cancelled: it is the only one in flight, so ending it closes the client.
    waiting = asyncio.create_task(backend.run(['true']))
    for _ in range(3):
        await asyncio.sleep(0)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert fake_vercel.closed_clients == 1
    fake_vercel.process_gate.set()
    with pytest.raises(WorkspaceError, match=r"^Could not create working_dir '.*/project' in Vercel sandbox "):
        await acquiring
    assert fake_vercel.get_calls == []


async def test_cancelled_run_closes_owned_client_after_killing_the_command(fake_vercel: FakeVercel) -> None:
    async def long_command(ctx: RunContext[None]) -> str:
        return (await ctx.workspace.run(['sleep', '30'])).stdout

    agent = Agent(
        TestModel(call_tools=['long_command']),
        deps_type=type(None),
        tools=[long_command],
        capabilities=[VercelSandbox()],
    )
    task = asyncio.create_task(agent.run('go'))
    with anyio.fail_after(10):
        await fake_vercel.process_started.wait()
    task.cancel()
    with anyio.fail_after(10), pytest.raises(asyncio.CancelledError):
        await task
    [sandbox] = fake_vercel.sandboxes.values()
    [process] = sandbox.processes
    assert process.kill_calls == 1
    assert fake_vercel.closed_clients == 1


@pytest.mark.parametrize('anyio_backend', ['trio'])
async def test_trio_is_refused_before_any_vercel_call(fake_vercel: FakeVercel) -> None:
    message = r'^Vercel Sandbox needs the asyncio event loop'
    backend = VercelSandbox[None]().get_workspace(_ctx(), ref=None)
    assert isinstance(backend, VercelSandboxBackend)
    with pytest.raises(UserError, match=message):
        await backend.run(['true'])
    with pytest.raises(UserError, match=message):
        await backend.get_sandbox()
    with pytest.raises(UserError, match=message):
        await VercelSandbox[None]().destroy(WorkspaceRef(provider='vercel', id='target'))
    await backend.aclose()
    assert fake_vercel.opened_clients == 0
