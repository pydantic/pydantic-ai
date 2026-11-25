"""Waiting for a decision inside a DBOS workflow, with `HandleDeferredToolCalls`.

The run draws the wait's key in a step, and calls the handler with a token that names the workflow in another
step that records its answer. When the handler answers, the run continues. Otherwise the workflow receives the
message a test sends with `resolve_external_event` or `fail_external_event`, or the timeout elapses.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Generator
from dataclasses import dataclass, field
from datetime import timedelta

import anyio
import pytest
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolDenied
from tests._timeouts import READINESS_WAIT_TIMEOUT

pytest.importorskip('dbos')

from dbos import DBOS, DBOSClient, DBOSConfig, SetWorkflowID, WorkflowHandleAsync
from dbos._core import WorkflowHandleAsyncTask
from dbos._error import DBOSNonExistentWorkflowError

from pydantic_ai.durable_exec.dbos import DBOSDurability, fail_external_event, resolve_external_event
from pydantic_ai.durable_exec.dbos._external_events import _TOPIC_PREFIX  # pyright: ignore[reportPrivateUsage]

pytestmark = [pytest.mark.xdist_group(name='dbos')]


@dataclass
class Records:
    """What the agents' functions did in the current test."""

    send_token: MemoryObjectSendStream[str]
    """Each token `ask` hands on, for the test to receive as the approver would."""
    receive_token: MemoryObjectReceiveStream[str]
    asked_about: list[list[str]] = field(default_factory=list[list[str]])
    deleted: list[str] = field(default_factory=list[str])
    model_calls: int = 0
    attempts: int = 0


_records: list[Records] = []


def _current() -> Records:
    return _records[-1]


def _delete_then_report(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    _current().model_calls += 1
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('delete', {'path': 'a.txt'}, tool_call_id='call-1')])
    returned = next(part for part in messages[-1].parts if isinstance(part, ToolReturnPart))
    return ModelResponse(parts=[TextPart(f'{returned.outcome}: {returned.content}')])


async def _hand_on(ctx: RunContext[None], requests: DeferredToolRequests, token: str) -> None:
    records = _current()
    records.asked_about.append([call.tool_call_id for call in requests.approvals])
    records.send_token.send_nowait(token)


def _delete(path: str) -> str:
    _current().deleted.append(path)
    return f'deleted {path}'


_approvals_agent = Agent(
    FunctionModel(_delete_then_report),
    name='dbos_approvals',
    deps_type=type(None),
    capabilities=[DBOSDurability[None](), HandleDeferredToolCalls[None](_hand_on, name='approve_delete')],
)
_approvals_agent.tool_plain(name='delete', requires_approval=True)(_delete)


class _Crash(BaseException):
    """Leave the first attempt pending, so resuming the workflow replays its recorded steps."""


@DBOS.workflow()
async def approvals_workflow(prompt: str) -> str:
    return (await _approvals_agent.run(prompt)).output


_timeout_agent = Agent(
    FunctionModel(_delete_then_report),
    name='dbos_approval_timeout',
    deps_type=type(None),
    capabilities=[
        DBOSDurability[None](),
        HandleDeferredToolCalls[None](_hand_on, timeout=timedelta(milliseconds=1)),  # real-wait: the timeout elapses
    ],
)
_timeout_agent.tool_plain(name='delete', requires_approval=True)(_delete)


@DBOS.workflow()
async def timeout_workflow(prompt: str) -> str:
    return (await _timeout_agent.run(prompt)).output


def _delete_twice_then_report(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    returned = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
    if len(returned) < 2:
        path = 'ab'[len(returned)] + '.txt'
        return ModelResponse(parts=[ToolCallPart('delete', {'path': path}, tool_call_id=f'call-{path}')])
    return ModelResponse(parts=[TextPart(', '.join(str(part.content) for part in returned))])


def _approve_all(ctx: RunContext[None], requests: DeferredToolRequests, token: str) -> DeferredToolResults:
    _current().asked_about.append([call.tool_call_id for call in requests.approvals])
    return requests.build_results(approvals={call.tool_call_id: True for call in requests.approvals})


_policy_agent = Agent(
    FunctionModel(_delete_twice_then_report),
    name='dbos_approval_policy',
    deps_type=type(None),
    capabilities=[DBOSDurability[None](), HandleDeferredToolCalls[None](_approve_all, timeout=timedelta(minutes=5))],
)
_policy_agent.tool_plain(name='delete', requires_approval=True)(_delete)


@DBOS.workflow()
async def policy_workflow(prompt: str) -> str:
    output = (await _policy_agent.run(prompt)).output
    _current().attempts += 1
    if _current().attempts == 1:
        raise _Crash
    return output


_two_approvals_agent = Agent(
    FunctionModel(_delete_twice_then_report),
    name='dbos_two_approvals',
    deps_type=type(None),
    capabilities=[DBOSDurability[None](), HandleDeferredToolCalls[None](_hand_on)],
)
_two_approvals_agent.tool_plain(name='delete', requires_approval=True)(_delete)


@DBOS.workflow()
async def two_approvals_workflow(prompt: str) -> str:
    return (await _two_approvals_agent.run(prompt)).output


@pytest.fixture(scope='module')
def dbos(tmp_path_factory: pytest.TempPathFactory) -> Generator[DBOS]:
    config: DBOSConfig = {
        'name': 'pydantic_ai_dbos_external_events',
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


@pytest.fixture(autouse=True)
async def records() -> AsyncIterator[Records]:
    send_token, receive_token = anyio.create_memory_object_stream[str](10)
    async with send_token, receive_token:
        _records.append(Records(send_token=send_token, receive_token=receive_token))
        yield _current()


async def _next_token(records: Records) -> str:
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        return await records.receive_token.receive()


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


async def test_the_run_waits_for_the_decision_and_a_resumed_workflow_does_not_ask_again(
    dbos: DBOS, records: Records
) -> None:
    workflow_id = f'approvals-{uuid.uuid4()}'
    with SetWorkflowID(workflow_id):
        handle = await DBOS.start_workflow_async(approvals_workflow, 'clean up')
    token = await _next_token(records)
    assert token.startswith(f'{workflow_id}:'), 'the token names the workflow'
    assert records.asked_about == [['call-1']]
    assert records.deleted == []

    # The execution that asked stops while it waits, as if its process died, and a new one takes over.
    await _stop_while_waiting(dbos, handle)
    resumed = await DBOS.resume_workflow_async(workflow_id)
    decision = {'approvals': {'call-1': {'kind': 'tool-approved', 'override_args': {'path': 'b.txt'}}}}
    await resolve_external_event(token, decision)
    assert await resumed.get_result(polling_interval_sec=0.01) == 'success: deleted b.txt'
    assert records.asked_about == [['call-1']], 'the resumed workflow does not ask again'
    assert records.deleted == ['b.txt']
    assert records.model_calls == 2, 'the first model response came from the recorded step'
    steps = await dbos.list_workflow_steps_async(workflow_id)
    assert [step['function_name'] for step in steps] == [
        'dbos_approvals__model.request',
        'dbos_approvals__external_event.approve_delete',
        'dbos_approvals__external_event.approve_delete.ask',
        'DBOS.recv',
        'DBOS.sleep',
        'dbos_approvals__model.request',
    ]

    # A fork from the last step replays the wait from its recorded steps, message included.
    fork = await DBOS.fork_workflow_async(workflow_id, steps[-1]['function_id'])
    assert await fork.get_result(polling_interval_sec=0.01) == 'success: deleted b.txt'
    assert records.asked_about == [['call-1']]
    # The tool is a function tool, which DBOS runs in workflow code, so the fork runs it again with the recorded decision.
    assert records.deleted == ['b.txt', 'b.txt']


async def test_a_denial_sent_with_a_client_reaches_the_model_without_running_the_tool(
    dbos: DBOS, records: Records
) -> None:
    """A `DBOSClient` sends the decision as an approval endpoint in another process would."""
    with SetWorkflowID(f'approvals-{uuid.uuid4()}'):
        handle = await DBOS.start_workflow_async(approvals_workflow, 'clean up')
    token = await _next_token(records)
    system_database_url = dbos._sys_db.engine.url.render_as_string(hide_password=False)  # pyright: ignore[reportPrivateUsage]
    client = DBOSClient(system_database_url=system_database_url)
    try:
        await resolve_external_event(
            token, DeferredToolResults(approvals={'call-1': ToolDenied('Not today.')}), client=client
        )
    finally:
        client.destroy()
    assert await handle.get_result() == 'denied: Not today.'
    assert records.deleted == []


async def test_a_failed_wait_denies_the_call(dbos: DBOS, records: Records) -> None:
    with SetWorkflowID(f'approvals-{uuid.uuid4()}'):
        handle = await DBOS.start_workflow_async(approvals_workflow, 'clean up')
    token = await _next_token(records)
    await fail_external_event(token, 'The approver is away.')
    assert await handle.get_result() == 'denied: No decision on this tool call arrived: The approver is away.'
    assert records.deleted == []


async def test_a_wait_that_times_out_denies_the_call(dbos: DBOS, records: Records) -> None:
    with SetWorkflowID(f'timeout-{uuid.uuid4()}'):
        handle = await DBOS.start_workflow_async(timeout_workflow, 'clean up')
    assert await handle.get_result() == (
        'denied: No decision on this tool call arrived: No resolution arrived within 0:00:00.001000.'
    )
    assert records.asked_about == [['call-1']]
    assert records.deleted == []


async def test_a_message_from_elsewhere_on_the_topic_denies_the_call(dbos: DBOS, records: Records) -> None:
    workflow_id = f'approvals-{uuid.uuid4()}'
    with SetWorkflowID(workflow_id):
        handle = await DBOS.start_workflow_async(approvals_workflow, 'clean up')
    token = await _next_token(records)
    event_key = token.rpartition(':')[2]
    await DBOS.send_async(workflow_id, 'approve', f'{_TOPIC_PREFIX}{event_key}')
    assert await handle.get_result() == (
        'denied: No decision on this tool call arrived: The message sent to resolve the wait was not sent by '
        '`resolve_external_event` or `fail_external_event`.'
    )
    assert records.deleted == []


async def test_answers_from_ask_are_recorded_and_the_run_continues_without_waiting(
    dbos: DBOS, records: Records
) -> None:
    workflow_id = f'approval-policy-{uuid.uuid4()}'
    with SetWorkflowID(workflow_id), pytest.raises(_Crash):
        await policy_workflow('clean up')
    assert records.deleted == ['a.txt', 'b.txt']
    assert records.asked_about == [['call-a.txt'], ['call-b.txt']]
    # Each wait drew a key and called `ask` in steps, and nothing was received.
    steps = [step['function_name'] for step in await dbos.list_workflow_steps_async(workflow_id)]
    assert steps.count('dbos_approval_policy__external_event.deferred_tool_calls.ask') == 2
    assert 'DBOS.recv' not in steps

    resumed = await DBOS.resume_workflow_async(workflow_id)
    assert await resumed.get_result(polling_interval_sec=0.01) == 'deleted a.txt, deleted b.txt'
    assert records.asked_about == [['call-a.txt'], ['call-b.txt']], 'the replay reuses the recorded answers'


async def test_a_resolution_for_a_wait_that_ended_is_discarded(dbos: DBOS, records: Records) -> None:
    """A late failure of the first wait's token, sent while the run waits again, changes nothing."""
    with SetWorkflowID(f'two-approvals-{uuid.uuid4()}'):
        handle = await DBOS.start_workflow_async(two_approvals_workflow, 'clean up')
    first = await _next_token(records)
    await resolve_external_event(first, {'approvals': {'call-a.txt': True}})
    second = await _next_token(records)
    await fail_external_event(first, 'Sent too late.')
    await resolve_external_event(second, {'approvals': {'call-b.txt': True}})
    # A second resolution of the same wait is ignored too: the wait received the first one.
    await fail_external_event(second, 'Sent twice.')
    assert await handle.get_result() == 'deleted a.txt, deleted b.txt'
    assert records.asked_about == [['call-a.txt'], ['call-b.txt']]
    assert records.deleted == ['a.txt', 'b.txt']


async def test_a_token_from_elsewhere_is_refused(dbos: DBOS) -> None:
    with pytest.raises(UserError, match='not handed out by a wait for an external event in a DBOS workflow'):
        await resolve_external_event('not-a-token', {'approvals': {}})
    with pytest.raises(UserError, match='not handed out by a wait for an external event in a DBOS workflow'):
        await fail_external_event(f'workflow:{"0" * 31}', 'No.')


async def test_a_token_for_a_workflow_that_does_not_exist_is_refused(dbos: DBOS) -> None:
    with pytest.raises(DBOSNonExistentWorkflowError):
        await resolve_external_event(f'no-such-workflow:{uuid.uuid4().hex}', {'approvals': {}})
