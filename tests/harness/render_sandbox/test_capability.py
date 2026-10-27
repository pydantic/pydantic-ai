"""Public Render sandbox capability tests, against the fake Render SDK."""

from __future__ import annotations

import re

import pytest

from ...conftest import try_import

with try_import() as imports_successful:
    from render.client.errors import ClientError, RateLimitError, ServerError
    from render.experimental.sandbox import SandboxNotFoundError

    from pydantic_ai import Agent
    from pydantic_ai.exceptions import UserError
    from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.tools import RunContext
    from pydantic_ai.usage import RunUsage
    from pydantic_ai.workspaces import WorkspaceRef, WorkspaceUnavailableError
    from pydantic_ai_harness.filesystem import FileSystem
    from pydantic_ai_harness.render_sandbox import RenderSandbox, RenderSandboxBackend

    from .._tool_calls import call_tools
    from .workspace_helpers import CreateCall, WorkspaceHarness, client_is_closed, use_host_credentials

pytestmark = pytest.mark.skipif(not imports_successful(), reason='render not installed')


def _ctx() -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage())


def test_root_package_exports_the_capability_and_backend() -> None:
    import pydantic_ai_harness

    assert pydantic_ai_harness.RenderSandbox is RenderSandbox
    assert pydantic_ai_harness.RenderSandboxBackend is RenderSandboxBackend
    assert {'RenderSandbox', 'RenderSandboxBackend'} <= set(pydantic_ai_harness.__all__)


def test_construction_needs_no_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('RENDER_API_KEY', raising=False)
    monkeypatch.delenv('RENDER_WORKSPACE_ID', raising=False)
    capability = RenderSandbox()
    assert isinstance(capability.get_workspace(_ctx(), ref=None), RenderSandboxBackend)


def test_env_is_not_shown_in_repr() -> None:
    secret = 'sensitive-credential-value'
    assert secret not in repr(RenderSandbox(env={'TOKEN': secret}))


@pytest.mark.parametrize(
    ('settings', 'message'),
    [
        ({'working_dir': 'repo'}, "working_dir must be an absolute POSIX path or None, got 'repo'"),
        ({'sandbox_timeout': 0}, 'sandbox_timeout must be an integer of at least 1, got 0'),
        ({'sandbox_timeout': 1.5}, 'sandbox_timeout must be an integer of at least 1, got 1.5'),
        ({'sandbox_timeout': True}, 'sandbox_timeout must be an integer of at least 1, got True'),
        ({'sandbox_timeout': 86_401}, 'sandbox_timeout must be at most 86400 seconds, got 86401'),
        (
            {'defer_loading': True, 'id': 'render'},
            (
                '`RenderSandbox` does not support `defer_loading=True`: '
                'the workspace is selected before deferred capabilities load.'
            ),
        ),
    ],
)
def test_invalid_settings_fail_at_construction(settings: dict[str, object], message: str) -> None:
    with pytest.raises(UserError, match=re.escape(message)):
        RenderSandbox(**settings)  # pyright: ignore[reportArgumentType]


def test_capability_declines_foreign_reference() -> None:
    assert RenderSandbox().get_workspace(_ctx(), ref=WorkspaceRef(provider='e2b', id='other')) is None


@pytest.mark.parametrize(('allow_internet_access', 'policy'), [(False, 'deny-all'), (True, 'allow-all')])
async def test_settings_shape_a_new_sandbox(
    workspace_harness: WorkspaceHarness, allow_internet_access: bool, policy: str
) -> None:
    capability = RenderSandbox(
        client=workspace_harness.client,
        allow_internet_access=allow_internet_access,
        sandbox_timeout=120,
        working_dir=str(workspace_harness.root),
    )
    backend = capability.get_workspace(_ctx(), ref=None)
    assert backend is not None
    assert (await backend.working_dir()) == str(workspace_harness.root)
    assert workspace_harness.create_calls['sb-1'] == CreateCall(120, policy)


async def test_backend_attaches_without_creating(workspace_harness: WorkspaceHarness) -> None:
    capability = RenderSandbox(client=workspace_harness.client, working_dir=str(workspace_harness.root))
    owned = workspace_harness.backend()
    await owned.working_dir()
    assert owned.ref is not None
    attached = capability.backend(owned.ref)
    assert attached.ref == owned.ref
    assert (await attached.run(['printf', 'attached'])).stdout == 'attached'
    assert workspace_harness.created == ['sb-1']


async def test_destroy_terminates_by_id_and_treats_a_missing_sandbox_as_gone(
    workspace_harness: WorkspaceHarness,
) -> None:
    capability = RenderSandbox(client=workspace_harness.client)
    backend = workspace_harness.backend()
    await backend.working_dir()
    assert backend.ref is not None
    await capability.destroy(backend.ref)
    assert workspace_harness.terminated == ['sb-1']
    with pytest.raises(WorkspaceUnavailableError):
        await backend.run(['true'])
    workspace_harness.terminate_error = SandboxNotFoundError('gone')
    await capability.destroy(backend.ref)


@pytest.mark.parametrize(
    ('error', 'expected'),
    [
        (ClientError('denied'), WorkspaceUnavailableError),
        (RateLimitError('try later'), RateLimitError),
        (ServerError('transient'), ServerError),
    ],
    ids=repr,
)
async def test_destroy_maps_a_rejection_and_propagates_transient_errors(
    workspace_harness: WorkspaceHarness, error: Exception, expected: type[Exception]
) -> None:
    workspace_harness.terminate_error = error
    with pytest.raises(expected):
        await RenderSandbox(client=workspace_harness.client).destroy(WorkspaceRef(provider='render', id='sb-1'))


@pytest.mark.parametrize('anyio_backend', ['trio'])
async def test_destroy_refuses_trio(workspace_harness: WorkspaceHarness) -> None:
    with pytest.raises(UserError, match='asyncio event loop'):
        await RenderSandbox(client=workspace_harness.client).destroy(WorkspaceRef(provider='render', id='sb-1'))
    assert workspace_harness.terminated == []


async def test_foreign_reference_is_rejected() -> None:
    foreign = WorkspaceRef(provider='e2b', id='other')
    message = re.escape("unsupported workspace provider 'e2b'; expected 'render'")
    with pytest.raises(ValueError, match=message):
        RenderSandbox().backend(foreign)
    with pytest.raises(ValueError, match=message):
        await RenderSandbox().destroy(foreign)


async def test_destroy_opens_and_closes_its_own_client(
    workspace_harness: WorkspaceHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = workspace_harness.backend()
    await backend.working_dir()
    assert backend.ref is not None
    use_host_credentials(workspace_harness, monkeypatch)
    await RenderSandbox().destroy(backend.ref)
    assert workspace_harness.terminated == ['sb-1']
    assert len(workspace_harness.opened_clients) == 1
    assert client_is_closed(workspace_harness.opened_clients[0])


@pytest.mark.parametrize('missing', ['RENDER_API_KEY', 'RENDER_WORKSPACE_ID'])
async def test_destroy_without_credentials_is_unavailable(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    monkeypatch.setenv('RENDER_API_KEY', 'test-only')
    monkeypatch.setenv('RENDER_WORKSPACE_ID', 'tea-test')
    monkeypatch.delenv(missing)
    with pytest.raises(WorkspaceUnavailableError, match='RENDER_API_KEY and RENDER_WORKSPACE_ID'):
        await RenderSandbox().destroy(WorkspaceRef(provider='render', id='sb-1'))


async def test_agent_run_works_in_the_sandbox_and_leaves_it_running(workspace_harness: WorkspaceHarness) -> None:
    capability = RenderSandbox(client=workspace_harness.client, working_dir=str(workspace_harness.root))
    results = await call_tools(
        [capability, FileSystem()],
        [('write_file', {'path': 'answer.txt', 'content': '42'}), ('read_file', {'path': 'answer.txt'})],
    )
    assert '42' in results[-1]
    assert (workspace_harness.root / 'answer.txt').read_text() == '42'
    assert workspace_harness.created == ['sb-1']
    assert workspace_harness.terminated == []


async def test_later_runs_reattach_to_the_same_sandbox(workspace_harness: WorkspaceHarness) -> None:
    # Continuing from message history, or passing the stored ref, must reach the first run's sandbox.
    call = ToolCallPart('write_file', {'path': 'note.txt', 'content': 'kept'})

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        last = messages[-1]
        if isinstance(last, ModelRequest) and any(isinstance(part, ToolReturnPart) for part in last.parts):
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[call])

    capability = RenderSandbox(client=workspace_harness.client, working_dir=str(workspace_harness.root))
    agent = Agent(FunctionModel(respond), capabilities=[capability, FileSystem()])
    first = await agent.run('write')
    call = ToolCallPart('read_file', {'path': 'note.txt'})
    from_history = await agent.run('read', message_history=first.all_messages())
    from_ref = await agent.run('read', workspace=first.workspace.ref)

    assert workspace_harness.created == ['sb-1']
    assert first.workspace.ref == from_history.workspace.ref == from_ref.workspace.ref
    returns = [
        part.content
        for message in from_ref.new_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]
    assert 'kept' in str(returns)
