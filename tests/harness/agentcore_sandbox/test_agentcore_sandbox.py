"""Public AgentCore workspace capability tests."""

from __future__ import annotations

import re

import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import Workspace, WorkspaceRef, WorkspaceUnavailableError
from pydantic_ai_harness.agentcore_sandbox import AgentCoreSandbox, AgentCoreSandboxBackend
from pydantic_ai_harness.coder import Coder

from .._tool_calls import call_tools
from .fake_agentcore import FakeAgentCore, client_error


def _ctx() -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage())


def test_root_package_exports_the_capability_and_backend() -> None:
    import pydantic_ai_harness

    assert pydantic_ai_harness.AgentCoreSandbox is AgentCoreSandbox
    assert pydantic_ai_harness.AgentCoreSandboxBackend is AgentCoreSandboxBackend
    assert {'AgentCoreSandbox', 'AgentCoreSandboxBackend'} <= set(pydantic_ai_harness.__all__)


def test_env_is_not_shown_in_repr() -> None:
    secret = 'sensitive-credential-value'
    assert secret not in repr(AgentCoreSandbox(env={'TOKEN': secret}))


@pytest.mark.parametrize(
    ('settings', 'message'),
    [
        ({'working_dir': 'repo'}, "working_dir must be an absolute POSIX path or None, got 'repo'"),
        ({'session_timeout': 0}, 'session_timeout must be an integer of at least 1, got 0'),
        ({'session_timeout': True}, 'session_timeout must be an integer of at least 1, got True'),
        ({'session_timeout': 28_801}, 'session_timeout must be at most 28800, got 28801'),
        ({'env': {'PORT': 3000}}, '`env`: env keys and values must be strings.'),
        ({'env': {'A=B': 'x'}}, '`env`: env keys must be non-empty without `=` or NUL'),
        ({'defer_loading': True}, '`AgentCoreSandbox` does not support `defer_loading=True`'),
    ],
)
def test_invalid_settings_fail_at_construction(settings: dict[str, object], message: str) -> None:
    with pytest.raises(UserError, match=re.escape(message)):
        AgentCoreSandbox(**settings)  # pyright: ignore[reportArgumentType]


def test_capability_declines_a_foreign_reference() -> None:
    assert AgentCoreSandbox[None]().get_workspace(_ctx(), ref=WorkspaceRef(provider='e2b', id='x')) is None


def test_capability_attaches_to_its_own_reference() -> None:
    ref = WorkspaceRef(provider='agentcore', id='my-interpreter-abc/session1')
    backend = AgentCoreSandbox[None]().get_workspace(_ctx(), ref=ref)
    assert isinstance(backend, AgentCoreSandboxBackend)
    assert backend.ref == ref
    assert AgentCoreSandbox[None]().backend(ref).ref == ref


async def test_agent_without_workspace_use_starts_no_session(fake_agentcore: FakeAgentCore) -> None:
    agent = Agent(TestModel(custom_output_text='done'), capabilities=[AgentCoreSandbox()])
    assert (await agent.run('hello')).output == 'done'
    assert not fake_agentcore.sessions


async def test_capability_forwards_its_settings_and_a_run_does_not_stop_the_session(
    fake_agentcore: FakeAgentCore,
) -> None:
    fake_agentcore.code_interpreters.add('my-interpreter-abc')
    work = fake_agentcore.home / 'work'
    capability = AgentCoreSandbox(
        region_name='eu-central-1',
        code_interpreter='my-interpreter-abc',
        session_timeout=3600,
        working_dir=str(work),
        env={'GREETING': 'hello'},
    )
    agent = Agent(TestModel(call_tools=['run_command']), capabilities=[capability])

    @agent.tool
    async def run_command(ctx: RunContext[object]) -> str:
        return (await ctx.workspace.run(['sh', '-c', 'echo "$GREETING from $(pwd)"'])).stdout

    result = await agent.run('run')
    assert f'hello from {work}' in result.output
    [session] = fake_agentcore.sessions.values()
    assert (session.code_interpreter, session.timeout, session.active) == ('my-interpreter-abc', 3600, True)
    assert fake_agentcore.clients[0]['region_name'] == 'eu-central-1'
    assert isinstance(result.workspace, Workspace)
    assert result.workspace.ref == WorkspaceRef(provider='agentcore', id=f'my-interpreter-abc/{session.session_id}')


async def test_agent_history_attaches_the_same_session(fake_agentcore: FakeAgentCore) -> None:
    def model(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        if isinstance(request, ModelRequest) and any(isinstance(part, ToolReturnPart) for part in request.parts):
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='remember', args={}, tool_call_id='call')])

    agent = Agent(FunctionModel(model), capabilities=[AgentCoreSandbox()])
    seen: list[str] = []

    @agent.tool
    async def remember(ctx: RunContext[object]) -> str:
        if await ctx.workspace.exists('marker.txt'):
            seen.append(await ctx.workspace.read_text('marker.txt'))
        else:
            await ctx.workspace.write_text('marker.txt', 'persisted')
            seen.append('created')
        return seen[-1]

    first = await agent.run('go')
    await agent.run('go', message_history=first.all_messages())
    assert seen == ['created', 'persisted']
    assert len(fake_agentcore.sessions) == 1


async def test_coder_works_in_the_session(fake_agentcore: FakeAgentCore) -> None:
    shell_output, read_output = await call_tools(
        [AgentCoreSandbox(), Coder()],
        [('shell', {'command': 'echo made-in-session > note.txt'}), ('read_file', {'path': 'note.txt'})],
    )
    assert '"exit_code": 0' in shell_output
    assert 'made-in-session' in read_output
    assert (fake_agentcore.home / 'note.txt').read_text() == 'made-in-session\n'


async def test_destroy_stops_the_session_and_is_idempotent(fake_agentcore: FakeAgentCore) -> None:
    backend = AgentCoreSandboxBackend()
    await backend.working_dir()
    assert backend.ref is not None
    capability = AgentCoreSandbox[None]()
    await capability.destroy(backend.ref)
    await capability.destroy(backend.ref)
    await capability.destroy(WorkspaceRef(provider='agentcore', id='aws.codeinterpreter.v1/never-existed'))
    assert not list(fake_agentcore.active())
    with pytest.raises(WorkspaceUnavailableError, match='is not active in this region'):
        await AgentCoreSandboxBackend(ref=backend.ref).working_dir()


async def test_destroy_maps_refused_credentials_and_passes_transient_errors(fake_agentcore: FakeAgentCore) -> None:
    ref = WorkspaceRef(provider='agentcore', id='aws.codeinterpreter.v1/session1')
    fake_agentcore.queue('stop', client_error('AccessDeniedException', 'not allowed', 'StopCodeInterpreterSession'))
    with pytest.raises(
        WorkspaceUnavailableError, match=r'^AgentCore refused the AWS credentials \(AccessDeniedException\)\.$'
    ):
        await AgentCoreSandbox[None]().destroy(ref)
    throttled = client_error('ThrottlingException', 'slow down', 'StopCodeInterpreterSession')
    fake_agentcore.queue('stop', throttled)
    with pytest.raises(type(throttled), match='slow down'):
        await AgentCoreSandbox[None]().destroy(ref)


async def test_destroy_rejects_a_foreign_reference(fake_agentcore: FakeAgentCore) -> None:
    with pytest.raises(ValueError, match=re.escape("unsupported workspace provider 'e2b'; expected 'agentcore'")):
        await AgentCoreSandbox[None]().destroy(WorkspaceRef(provider='e2b', id='x'))
    assert not fake_agentcore.clients
