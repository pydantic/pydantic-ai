"""Public `BoatSandbox` capability tests, over the fake Boat API.

They drive the capability through `Agent` runs and its own methods. They are not VCR tests: what
they pin is which client a run opens and closes, and which sandbox a run creates or attaches to,
which no recorded exchange shows.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

import anyio
import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import (
    ReadOnlyWorkspace,
    Workspace,
    WorkspaceReadOnlyError,
    WorkspaceRef,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness.boat_sandbox import BoatSandbox, BoatSandboxBackend

# The error type a transient Boat failure propagates as; it is not exported.
from pydantic_ai_harness.boat_sandbox._api import BoatAPIError
from pydantic_ai_harness.coder import Coder

from .._tool_calls import call_tools
from .fake_boat import FakeBoat, FakeSandbox, boat_error

pytestmark = pytest.mark.anyio(backends=['asyncio'])

_AUTH = 'Boat rejected the credentials. Set BOAT_API_KEY, or pass a configured `httpx.AsyncClient` as `client=`.'


def _ctx() -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage())


def only(fake: FakeBoat) -> FakeSandbox:
    [sandbox] = fake.sandboxes.values()
    return sandbox


def test_capability_repr_hides_command_environment() -> None:
    assert 'secret-sentinel' not in repr(BoatSandbox(env={'TOKEN': 'secret-sentinel'}))


async def test_capability_is_lazy_and_forwards_options(fake_boat: FakeBoat, tmp_path: Path) -> None:
    directory = f'{tmp_path.resolve()}/work'
    fake_boat.snapshots.add('python')
    capability = BoatSandbox[None](
        machine_type='large', snapshot='python', auto_stop=600, working_dir=directory, env={'A': 'b'}
    )
    backend = capability.get_workspace(_ctx(), ref=None)
    assert isinstance(backend, BoatSandboxBackend)
    assert fake_boat.calls == []
    assert await backend.working_dir() == directory
    assert fake_boat.create_bodies == [{'noEnv': True, 'type': 'large', 'from': 'python', 'ttlSeconds': 600}]
    # The env reaches commands, not the create request.
    assert (await backend.run(['sh', '-c', 'printf %s "$A"'])).stdout == 'b'


@pytest.mark.parametrize(
    ('build', 'error', 'message'),
    [
        pytest.param(
            lambda: BoatSandbox(working_dir='repo'),
            UserError,
            r"^working_dir must be an absolute POSIX path or None, got 'repo'\.$",
            id='relative-working-dir',
        ),
        pytest.param(
            lambda: BoatSandbox(auto_stop=-1),
            UserError,
            r'^auto_stop must be an integer of at least 0 or None, got -1\.$',
            id='negative-auto-stop',
        ),
        pytest.param(
            lambda: BoatSandbox(auto_stop=1.5),  # pyright: ignore[reportArgumentType]
            UserError,
            r'^auto_stop must be an integer of at least 0 or None, got 1\.5\.$',
            id='fractional-auto-stop',
        ),
        pytest.param(
            lambda: BoatSandbox(defer_loading=True, id='sandbox'),
            UserError,
            r'^`BoatSandbox` does not support `defer_loading=True`: '
            r'the workspace is selected before deferred capabilities load\.$',
            id='defer-loading',
        ),
        pytest.param(
            lambda: BoatSandbox(env={'PORT': 3000}),  # pyright: ignore[reportArgumentType]
            TypeError,
            r'^env keys and values must be strings$',
            id='non-string-env',
        ),
    ],
)
def test_invalid_settings_fail_at_construction(
    build: Callable[[], object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        build()


async def test_backend_for_a_ref_attaches_with_the_capability_settings(fake_boat: FakeBoat) -> None:
    existing = fake_boat.sandbox()
    (existing.home / 'work').mkdir()
    capability = BoatSandbox[None](working_dir=f'{existing.home}/work', env={'A': 'b'}, machine_type='large')
    backend = capability.backend(WorkspaceRef(provider='boat', id=existing.id))
    assert (await backend.run(['sh', '-c', 'printf "%s %s" "$A" "$(pwd -P)"'])).stdout == f'b {existing.home}/work'
    assert fake_boat.create_bodies == []


def test_foreign_ref_is_declined_and_the_backend_rejects_it() -> None:
    assert BoatSandbox[None]().get_workspace(_ctx(), ref=WorkspaceRef(provider='other', id='x')) is None
    with pytest.raises(ValueError, match="expected 'boat'"):
        BoatSandbox[None]().backend(WorkspaceRef(provider='other', id='x'))


async def test_destroy_rejects_a_foreign_or_blank_ref(fake_boat: FakeBoat) -> None:
    with pytest.raises(ValueError, match=r"^unsupported workspace provider 'other'; expected 'boat'$"):
        await BoatSandbox[None]().destroy(WorkspaceRef(provider='other', id='x'))
    with pytest.raises(ValueError, match=r'^Boat workspace ref id cannot be empty$'):
        await BoatSandbox[None]().destroy(WorkspaceRef(provider='boat', id=' '))
    assert fake_boat.opened_clients == 0


async def test_destroy_deletes_a_stopped_sandbox_without_resuming_it(fake_boat: FakeBoat) -> None:
    existing = fake_boat.sandbox('archived')
    capability = BoatSandbox[None]()
    await capability.destroy(WorkspaceRef(provider='boat', id=existing.id))
    assert existing.id not in fake_boat.sandboxes
    assert existing.resume_calls == 0
    # Boat deletes only with the target's ID repeated in a header.
    [delete] = [call.request for call in fake_boat.calls if call.route == 'delete']
    assert delete.headers['x-ascii-confirm-delete'] == existing.id
    # A sandbox already deleted, or one Boat never knew, is already gone.
    await capability.destroy(WorkspaceRef(provider='boat', id=existing.id))
    await capability.destroy(WorkspaceRef(provider='boat', id='bx_unknown'))
    # Each call opened its own client and closed it.
    assert (fake_boat.opened_clients, fake_boat.closed_clients) == (3, 3)


async def test_destroy_translates_rejected_credentials_and_propagates_transient_errors(fake_boat: FakeBoat) -> None:
    existing = fake_boat.sandbox()
    ref = WorkspaceRef(provider='boat', id=existing.id)
    fake_boat.errors['delete'] = [
        boat_error(401, 'unauthorized', 'expired token secret-value'),
        boat_error(503, 'out_of_capacity', 'No capacity.'),
    ]
    with pytest.raises(WorkspaceUnavailableError) as exc_info:
        await BoatSandbox[None]().destroy(ref)
    assert str(exc_info.value) == f'Credential expired. {_AUTH}'
    with pytest.raises(BoatAPIError) as transient:
        await BoatSandbox[None]().destroy(ref)
    assert transient.value.status == 503
    assert existing.id in fake_boat.sandboxes
    # The client each call opened is closed on the way out of a failure too.
    assert (fake_boat.opened_clients, fake_boat.closed_clients) == (2, 2)


async def test_destroy_with_a_supplied_client_leaves_it_open(fake_boat: FakeBoat) -> None:
    existing = fake_boat.sandbox()
    client = fake_boat.supplied_client()
    await BoatSandbox[None](client=client).destroy(WorkspaceRef(provider='boat', id=existing.id))
    assert existing.id not in fake_boat.sandboxes
    assert not client.is_closed
    assert fake_boat.opened_clients == 0


async def test_agent_without_workspace_use_does_not_create(fake_boat: FakeBoat) -> None:
    result = await Agent(TestModel(custom_output_text='done'), capabilities=[BoatSandbox()]).run('hello')
    assert result.output == 'done'
    assert fake_boat.calls == []


async def test_agent_runs_without_history_create_fresh_sandboxes(fake_boat: FakeBoat) -> None:
    agent = Agent(TestModel(call_tools=['run_command']), capabilities=[BoatSandbox()])

    @agent.tool
    async def run_command(ctx: RunContext[object]) -> str:
        return (await ctx.workspace.run(['echo', 'ok'])).stdout

    await agent.run('run')
    await agent.run('run')
    assert len(fake_boat.create_bodies) == 2


async def test_agent_history_attaches_and_resumes_the_persisted_sandbox(fake_boat: FakeBoat) -> None:
    def model(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        if isinstance(messages[-1], ModelRequest) and any(isinstance(p, ToolReturnPart) for p in messages[-1].parts):
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='remember', args={}, tool_call_id='call')])

    agent = Agent(FunctionModel(model), capabilities=[BoatSandbox()])

    @agent.tool
    async def remember(ctx: RunContext[object]) -> str:
        if await ctx.workspace.exists('marker'):
            return (await ctx.workspace.read_bytes('marker')).decode()
        await ctx.workspace.write_bytes('marker', b'saved')
        return 'created'

    first = await agent.run('remember')
    sandbox = only(fake_boat)
    # Boat stops an idle sandbox after its auto-stop; the next run attaching to it resumes it.
    fake_boat.stop(sandbox.id)
    second = await agent.run('continue', message_history=first.all_messages())
    assert (first.output, second.output) == ('done', 'done')
    returned = [
        part.content for message in second.new_messages() for part in message.parts if isinstance(part, ToolReturnPart)
    ]
    assert returned == ['saved']
    assert len(fake_boat.create_bodies) == 1
    assert sandbox.resume_calls == 1
    assert second.workspace.ref == first.workspace.ref == WorkspaceRef(provider='boat', id=sandbox.id)


async def test_read_only_facade_identity_and_denied_command(fake_boat: FakeBoat) -> None:
    backend = BoatSandboxBackend()
    facade = ReadOnlyWorkspace(Workspace(backend))
    agent = Agent(TestModel(call_tools=['check']), capabilities=[BoatSandbox()])

    @agent.tool
    async def check(ctx: RunContext[object]) -> str:
        assert ctx.workspace is facade
        with pytest.raises(WorkspaceReadOnlyError, match='read-only'):
            await ctx.workspace.run(['echo', 'blocked'])
        return 'ok'

    result = await agent.run('check', workspace=facade)
    assert result.workspace is facade
    assert 'ok' in result.output
    assert fake_boat.calls == []


async def _touch(ctx: RunContext[None]) -> str:
    await ctx.workspace.write_bytes('note', b'kept')
    return 'ok'


async def test_a_cancelled_run_closes_its_client_after_stopping_the_command(fake_boat: FakeBoat) -> None:
    async def long_command(ctx: RunContext[None]) -> str:
        return (await ctx.workspace.run(['sleep', '30'])).stdout

    agent = Agent(
        TestModel(call_tools=['long_command']), deps_type=type(None), tools=[long_command], capabilities=[BoatSandbox()]
    )
    task = asyncio.create_task(agent.run('go'))
    with anyio.fail_after(10):
        await fake_boat.reached['status'].wait()
    task.cancel()
    # Promptly: the run's client close must not wait on the command it is cancelling.
    with anyio.fail_after(10), pytest.raises(asyncio.CancelledError):
        await task
    assert fake_boat.closed_clients == 1
    [command] = only(fake_boat).commands.values()
    assert command.stopped and command.process.poll() is not None


async def test_a_run_closes_the_client_it_opened_and_the_result_workspace_reopens(fake_boat: FakeBoat) -> None:
    agent = Agent(TestModel(call_tools=['_touch']), deps_type=type(None), tools=[_touch], capabilities=[BoatSandbox()])
    result = await agent.run('go')
    # The run keeps one client across its operations and closes it at the end.
    assert (fake_boat.opened_clients, fake_boat.closed_clients) == (1, 1)

    # After the run nothing would close the backend, so each operation closes its own client:
    # resolving the relative path against the working directory, then reading the file.
    assert await result.workspace.read_bytes('note') == b'kept'
    assert (fake_boat.opened_clients, fake_boat.closed_clients) == (3, 3)
    assert len(fake_boat.create_bodies) == 1


async def test_a_backend_used_outside_a_run_closes_its_client_after_each_operation(fake_boat: FakeBoat) -> None:
    # What a Temporal activity does: rebuild the backend from the ref, use it, and drop it.
    backend = BoatSandbox[None]().get_workspace(_ctx(), ref=None)
    assert isinstance(backend, BoatSandboxBackend)
    path = f'{await backend.working_dir()}/note'
    assert (fake_boat.opened_clients, fake_boat.closed_clients) == (1, 1)
    await backend.write_bytes(path, b'hi')
    assert (fake_boat.opened_clients, fake_boat.closed_clients) == (2, 2)

    attached = BoatSandbox[None]().get_workspace(_ctx(), ref=backend.ref)
    assert isinstance(attached, BoatSandboxBackend)
    # One operation, however many requests it makes, and concurrent operations share one client,
    # closed once the last of them ends.
    first, second = await asyncio.gather(attached.read_bytes(path), attached.exists(path))
    assert (first, second) == (b'hi', True)
    assert (fake_boat.opened_clients, fake_boat.closed_clients) == (3, 3)


async def test_a_stop_a_repeated_cancellation_leaves_running_is_an_operation_of_its_own(fake_boat: FakeBoat) -> None:
    # Outside a run the client closes after each operation; the stop keeps it open until Boat answers.
    backend = BoatSandbox[None]().get_workspace(_ctx(), ref=None)
    assert isinstance(backend, BoatSandboxBackend)
    await backend.working_dir()
    release = fake_boat.gates['kill'] = asyncio.Event()
    task = asyncio.create_task(backend.run(['sleep', '30']))
    with anyio.fail_after(10):
        await fake_boat.reached['status'].wait()
        task.cancel()
        await fake_boat.reached['kill'].wait()
        # The second cancellation ends the run's wait for the stop, which keeps going.
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert fake_boat.closed_clients == 1
        release.set()
        while fake_boat.closed_clients < 2:
            await anyio.sleep(0.001)
    [command] = only(fake_boat).commands.values()
    assert command.stopped


@pytest.mark.parametrize('anyio_backend', ['trio'])
async def test_trio_is_refused_before_any_boat_call(fake_boat: FakeBoat) -> None:
    """The backend finishes creation and command cleanup on asyncio tasks, so Trio gets a clear `UserError`."""
    message = (
        r'^Boat needs the asyncio event loop: the backend finishes sandbox creation and command cleanup '
        r'on asyncio tasks of its own\.$'
    )
    backend = BoatSandbox[None]().get_workspace(_ctx(), ref=None)
    assert isinstance(backend, BoatSandboxBackend)
    with pytest.raises(UserError, match=message):
        await backend.run(['true'])
    with pytest.raises(UserError, match=message):
        await backend.read_bytes('/home/user/file')
    with pytest.raises(UserError, match=message):
        await backend.working_dir()
    with pytest.raises(UserError, match=message):
        await BoatSandbox[None]().destroy(WorkspaceRef(provider='boat', id='bx_target'))
    await backend.aclose()
    assert fake_boat.opened_clients == 0


@pytest.mark.parametrize('supplied', ['client', 'workspace'])
async def test_a_run_leaves_a_supplied_client_or_workspace_open(fake_boat: FakeBoat, supplied: str) -> None:
    client = fake_boat.supplied_client() if supplied == 'client' else None
    workspace = BoatSandboxBackend() if supplied == 'workspace' else None
    agent = Agent(
        TestModel(call_tools=['_touch']),
        deps_type=type(None),
        tools=[_touch],
        capabilities=[BoatSandbox[None](client=client)],
    )
    await agent.run('go', workspace=workspace)
    assert fake_boat.closed_clients == 0
    assert client is None or not client.is_closed


async def test_a_child_run_leaves_the_parent_runs_client_open(fake_boat: FakeBoat) -> None:
    child = Agent(TestModel(call_tools=['_touch']), deps_type=type(None), tools=[_touch], capabilities=[BoatSandbox()])
    closed_after_child: list[int] = []

    async def delegate(ctx: RunContext[None]) -> str:
        await child.run('go', workspace=ctx.workspace)
        closed_after_child.append(fake_boat.closed_clients)
        return 'ok'

    parent = Agent(
        TestModel(call_tools=['_touch', 'delegate']),
        deps_type=type(None),
        tools=[_touch, delegate],
        capabilities=[BoatSandbox()],
    )
    await parent.run('go')
    assert closed_after_child == [0]
    assert fake_boat.closed_clients == 1
    assert len(fake_boat.create_bodies) == 1


async def test_coder_works_in_the_sandbox(fake_boat: FakeBoat) -> None:
    results = await call_tools(
        [BoatSandbox(), Coder()],
        [
            ('write_file', {'path': 'greet.sh', 'content': 'echo hi from the sandbox\n'}),
            ('shell', {'command': 'sh greet.sh'}),
        ],
    )
    assert 'hi from the sandbox' in results[1]
    assert (only(fake_boat).home / 'greet.sh').read_text() == 'echo hi from the sandbox\n'
