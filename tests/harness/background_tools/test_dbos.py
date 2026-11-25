"""`BackgroundTools` under `DBOSDurability`: a background step keeps running while the run waits for an approval."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Generator

import anyio
import pytest

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextContent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import DeferredToolRequests
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai_harness import BackgroundTools
from tests._timeouts import READINESS_WAIT_TIMEOUT

try:
    from dbos import DBOS, DBOSConfig, SetWorkflowID, WorkflowHandleAsync
    from dbos._core import WorkflowHandleAsyncTask

    from pydantic_ai.durable_exec.dbos import DBOSDurability, resolve_external_event
except ImportError:  # pragma: lax no cover
    pytest.skip('dbos not installed', allow_module_level=True)

pytestmark = pytest.mark.xdist_group(name='harness-dbos')

_research_calls: list[str] = []
_gated_calls: list[str] = []
_tokens: list[str] = []
_asked: list[anyio.Event] = []
_research_may_finish: list[anyio.Event] = []
_research_finished: list[anyio.Event] = []


@DBOS.step()
async def research() -> str:
    """Research the topic."""
    _research_calls.append('research')
    # Still running when the run starts waiting for the approval, and finishing while it waits.
    await _research_may_finish[-1].wait()
    _research_finished[-1].set()
    return 'durable result'


def gated() -> str:
    """Do something that needs approval."""
    _gated_calls.append('gated')
    return 'approved'


def _background_result(messages: list[ModelMessage]) -> bool:
    return any(
        isinstance(part, UserPromptPart)
        and not isinstance(part.content, str)
        and any(
            isinstance(item, TextContent)
            and item.content == '<tool_result tool_name="research" tool_call_id="research-call">'
            for item in part.content
        )
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
    )


def _model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if _background_result(messages):
        return ModelResponse(parts=[TextPart(content='done')])
    if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
        return ModelResponse(parts=[TextPart(content='waiting')])
    return ModelResponse(
        parts=[
            ToolCallPart(tool_name='research', args={}, tool_call_id='research-call'),
            ToolCallPart(tool_name='gated', args={}, tool_call_id='gated-call'),
        ]
    )


async def _hand_on(ctx: RunContext[None], requests: DeferredToolRequests, token: str) -> None:
    _tokens.append(token)
    _asked[-1].set()


agent = Agent(
    FunctionModel(_model),
    name='background_tools_dbos_agent',
    deps_type=type(None),
    toolsets=[
        FunctionToolset[None](tools=[research], id='background-tools', metadata={'background': True}),
        FunctionToolset[None](tools=[Tool(gated, requires_approval=True)], id='gated-tools'),
    ],
    capabilities=[BackgroundTools[None](), DBOSDurability[None](), HandleDeferredToolCalls[None](_hand_on)],
)


@DBOS.workflow()
async def background_workflow(prompt: str) -> str:
    return (await agent.run(prompt)).output


@pytest.fixture(scope='module')
def dbos(tmp_path_factory: pytest.TempPathFactory) -> Generator[DBOS]:
    config: DBOSConfig = {
        'name': 'pydantic_ai_harness_background_tools_dbos',
        'system_database_url': f'sqlite:///{tmp_path_factory.mktemp("dbos") / "dbos.sqlite"}',
        'enable_otlp': False,
    }
    instance = DBOS(config=config)
    DBOS.launch()
    # On SQLite, DBOS polls for messages every second; poll more often so a resolved wait ends right away.
    instance._sys_db._notification_listener_polling_interval_sec = 0.01  # pyright: ignore[reportPrivateUsage]
    try:
        yield instance
    finally:
        DBOS.destroy()


async def _stop_while_waiting(dbos: DBOS, handle: WorkflowHandleAsync[str]) -> None:
    """Stop the workflow's execution while it waits, as if its process died, leaving the workflow pending.

    It waits until the workflow recorded `ask`'s step and set up the receive, whose timeout DBOS records as a sleep:
    `ask` hands the token on before its step records the result, so a stop before then would run the step again.
    """
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        # real-wait: nothing in the process signals that DBOS recorded the step, so poll the system database
        while 'DBOS.sleep' not in [
            step['function_name'] for step in await dbos.list_workflow_steps_async(handle.workflow_id)
        ]:
            await anyio.sleep(0.01)
    # The handle of a workflow started in this process wraps the task running it.
    assert isinstance(handle, WorkflowHandleAsyncTask)
    task: asyncio.Future[str] = handle.task  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    status = await DBOS.get_workflow_status_async(handle.workflow_id)
    assert status is not None and status.status == 'PENDING'


async def test_a_background_step_runs_through_an_approval_wait_and_recovery_does_not_ask_again(dbos: DBOS) -> None:
    _asked.append(anyio.Event())
    _research_may_finish.append(anyio.Event())
    _research_finished.append(anyio.Event())
    workflow_id = f'background-{uuid.uuid4()}'
    with SetWorkflowID(workflow_id):
        handle = await DBOS.start_workflow_async(background_workflow, 'go')
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        await _asked[-1].wait()
        assert _research_calls == ['research'], 'the background step started before the wait'
        # The background step finishes while the run waits for the approval.
        _research_may_finish[-1].set()
        await _research_finished[-1].wait()
    [token] = _tokens

    # The execution stops while it waits, as if its process died, and a new one takes over.
    await _stop_while_waiting(dbos, handle)
    resumed = await DBOS.resume_workflow_async(workflow_id)
    await resolve_external_event(token, {'approvals': {'gated-call': True}})
    # The model only answers `'done'` once it has the background result, which comes with the approved call's.
    assert await resumed.get_result(polling_interval_sec=0.01) == 'done'
    assert _tokens == [token], 'the resumed workflow does not ask again'
    assert _research_calls == ['research'], 'the resumed workflow replays the background step'
    assert _gated_calls == ['gated']

    steps = await dbos.list_workflow_steps_async(workflow_id)
    # The background step was numbered when it started, before the wait, so the replay numbers it the same.
    assert [step['function_name'] for step in steps] == [
        'background_tools_dbos_agent__model.request',
        'research',
        'background_tools_dbos_agent__external_event.deferred_tool_calls',
        'background_tools_dbos_agent__external_event.deferred_tool_calls.ask',
        'DBOS.recv',
        'DBOS.sleep',
        'background_tools_dbos_agent__model.request',
    ]

    # A fork from the last step replays the whole run, the background step and the wait included.
    fork = await DBOS.fork_workflow_async(workflow_id, steps[-1]['function_id'])
    assert await fork.get_result(polling_interval_sec=0.01) == 'done'
    assert _tokens == [token]
    assert _research_calls == ['research']
