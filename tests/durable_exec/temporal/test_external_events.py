"""Waiting for a decision inside a Temporal workflow, with `RequestInput`.

The run calls `ask` with a token that names the workflow, in a local activity whose result the history
records. When `ask` answers, the run continues. Otherwise the workflow waits until a test resolves the
token with `resolve_external_event` or `fail_external_event`, which signal the workflow, or the
timeout elapses, which Temporal's time-skipping test server fast-forwards to.
"""

from __future__ import annotations

import sys
import uuid
from collections.abc import AsyncIterator, Iterator
from datetime import timedelta
from typing import Any

import anyio
import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import RequestInput
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolDenied
from tests.temporal_utils import temporal_dev_server_cache_dir

try:
    from temporalio import workflow
    from temporalio.client import Client, WorkflowHandle
    from temporalio.common import RetryPolicy
    from temporalio.contrib.pydantic import pydantic_data_converter
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Replayer, UnsandboxedWorkflowRunner, Worker, WorkflowRunner
    from temporalio.workflow import ActivityConfig

    from pydantic_ai.durable_exec.temporal import (
        AgentPlugin,
        PydanticAIPlugin,
        TemporalDurability,
        fail_external_event,
        resolve_external_event,
    )
except ImportError:  # pragma: lax no cover
    pytest.skip('temporal not installed', allow_module_level=True)

if sys.version_info >= (3, 14):  # pragma: lax no cover
    pytest.skip(
        'temporalio sandbox is incompatible with Python 3.14: '
        'sandbox module state accumulates across validation cycles causing import failures after ~22 workflows '
        '(remove when https://github.com/temporalio/sdk-python/issues/1326 closes)',
        allow_module_level=True,
    )

pytestmark = [pytest.mark.xdist_group(name='temporal-external-events')]

ACTIVITY_CONFIG = ActivityConfig(
    start_to_close_timeout=timedelta(seconds=60), retry_policy=RetryPolicy(maximum_attempts=1)
)

_tokens: list[str] = []
_asked: list[list[str]] = []
_deleted: list[str] = []
_model_calls: list[int] = []


def _delete_then_report(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    _model_calls.append(len(messages))
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('delete', {'path': 'a.txt'}, tool_call_id='call-1')])
    returned = next(part for part in messages[-1].parts if isinstance(part, ToolReturnPart))
    return ModelResponse(parts=[TextPart(f'{returned.outcome}: {returned.content}')])


async def _hand_on(ctx: RunContext[None], requests: DeferredToolRequests, token: str) -> None:
    _tokens.append(token)
    _asked.append([call.tool_call_id for call in requests.approvals])


def _delete(path: str) -> str:
    _deleted.append(path)
    return f'deleted {path}'


_approvals_agent = Agent(
    FunctionModel(_delete_then_report),
    name='temporal_approvals',
    deps_type=type(None),
    capabilities=[
        TemporalDurability[None](activity_config=ACTIVITY_CONFIG),
        RequestInput[None](_hand_on, timeout=timedelta(minutes=30)),
    ],
)
_approvals_agent.tool_plain(name='delete', requires_approval=True)(_delete)


@workflow.defn
class ApprovalsWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await _approvals_agent.run(prompt)).output


def _delete_twice_then_report(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    returned = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
    if len(returned) < 2:
        path = 'ab'[len(returned)] + '.txt'
        return ModelResponse(parts=[ToolCallPart('delete', {'path': path}, tool_call_id=f'call-{path}')])
    return ModelResponse(parts=[TextPart(', '.join(str(part.content) for part in returned))])


def _approve_all(ctx: RunContext[None], requests: DeferredToolRequests, token: str) -> DeferredToolResults:
    return requests.build_results(approvals={call.tool_call_id: True for call in requests.approvals})


_policy_agent = Agent(
    FunctionModel(_delete_twice_then_report),
    name='temporal_approval_policy',
    deps_type=type(None),
    capabilities=[
        TemporalDurability[None](activity_config=ACTIVITY_CONFIG),
        RequestInput[None](_approve_all, timeout=timedelta(minutes=5)),
    ],
)
_policy_agent.tool_plain(name='delete', requires_approval=True)(_delete)


@workflow.defn
class ApprovalPolicyWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await _policy_agent.run(prompt)).output


@pytest.fixture(scope='module')
def anyio_backend() -> str:
    """Temporal's Python SDK runs on asyncio."""
    return 'asyncio'


@pytest.fixture(scope='module')
async def env() -> AsyncIterator[WorkflowEnvironment]:
    async with await WorkflowEnvironment.start_time_skipping(
        plugins=[PydanticAIPlugin()], download_dest_dir=temporal_dev_server_cache_dir()
    ) as env:
        yield env


@pytest.fixture(autouse=True)
def reset_records() -> Iterator[None]:
    yield
    for records in (_tokens, _asked, _deleted, _model_calls):
        records.clear()


def _worker(
    client: Client,
    task_queue: str,
    workflow_class: type,
    agent: Agent[None, str],
    runner: WorkflowRunner | None = None,
) -> Worker:
    return Worker(
        client,
        task_queue=task_queue,
        workflows=[workflow_class],
        plugins=[AgentPlugin(agent)],
        workflow_runner=runner or UnsandboxedWorkflowRunner(),
    )


async def _start(client: Client, task_queue: str) -> WorkflowHandle[Any, str]:
    return await client.start_workflow(
        ApprovalsWorkflow.run, args=['clean up'], id=f'approvals-{uuid.uuid4()}', task_queue=task_queue
    )


async def _until_asked(handle: WorkflowHandle[Any, str]) -> str:
    """Wait until the history records the local activity that called `ask`, so the run is waiting, and return the token."""
    with anyio.fail_after(30):
        while not any(
            event.HasField('marker_recorded_event_attributes') for event in (await handle.fetch_history()).events
        ):
            await anyio.sleep(0.05)
    [token] = _tokens
    return token


async def test_the_run_waits_for_the_decision_and_a_new_worker_resumes_it_without_asking_again(
    env: WorkflowEnvironment,
) -> None:
    client = env.client
    task_queue = f'approvals-{uuid.uuid4()}'
    async with _worker(client, task_queue, ApprovalsWorkflow, _approvals_agent):
        handle = await _start(client, task_queue)
        token = await _until_asked(handle)
    assert token.startswith(f'{handle.id}:'), 'the token names the workflow'
    assert _asked == [['call-1']]
    assert _deleted == []

    # The worker that asked is gone. The decision is signalled to the workflow, and a new worker replays it.
    decision = {'approvals': {'call-1': {'kind': 'tool-approved', 'override_args': {'path': 'b.txt'}}}}
    await resolve_external_event(client, token, decision)
    async with _worker(client, task_queue, ApprovalsWorkflow, _approvals_agent):
        assert await handle.result() == 'success: deleted b.txt'

    assert _deleted == ['b.txt']
    assert _asked == [['call-1']], 'the replay does not ask again'
    assert len(_model_calls) == 2, 'the first model response came from the history'

    await Replayer(
        workflows=[ApprovalsWorkflow],
        workflow_runner=UnsandboxedWorkflowRunner(),
        data_converter=pydantic_data_converter,
    ).replay_workflow(await handle.fetch_history())
    assert _asked == [['call-1']]
    assert _deleted == ['b.txt']


async def test_a_denial_reaches_the_model_without_running_the_tool(env: WorkflowEnvironment) -> None:
    client = env.client
    task_queue = f'approvals-{uuid.uuid4()}'
    async with _worker(client, task_queue, ApprovalsWorkflow, _approvals_agent):
        handle = await _start(client, task_queue)
        token = await _until_asked(handle)
        await resolve_external_event(client, token, DeferredToolResults(approvals={'call-1': ToolDenied('Not today.')}))
        assert await handle.result() == 'denied: Not today.'

    assert _deleted == []


async def test_a_failed_wait_denies_the_call(env: WorkflowEnvironment) -> None:
    client = env.client
    task_queue = f'approvals-{uuid.uuid4()}'
    async with _worker(client, task_queue, ApprovalsWorkflow, _approvals_agent):
        handle = await _start(client, task_queue)
        token = await _until_asked(handle)
        await fail_external_event(client, token, 'The approver is away.')
        assert await handle.result() == 'denied: No decision on this tool call arrived: The approver is away.'

    assert _deleted == []


async def test_a_wait_that_times_out_denies_the_call(env: WorkflowEnvironment) -> None:
    """Awaiting the result lets the time-skipping server jump over the 30-minute timeout."""
    client = env.client
    task_queue = f'approvals-{uuid.uuid4()}'
    async with _worker(client, task_queue, ApprovalsWorkflow, _approvals_agent):
        handle = await _start(client, task_queue)
        assert await handle.result() == (
            'denied: No decision on this tool call arrived: No resolution arrived within 0:30:00.'
        )

    assert _asked == [['call-1']]
    assert _deleted == []


async def test_answers_from_ask_are_recorded_and_the_run_continues_without_waiting(
    env: WorkflowEnvironment,
) -> None:
    """In the workflow sandbox, as a production worker runs it: each wait's `ask` answers in place, and a replay reuses the answers."""
    client = env.client
    task_queue = f'approval-policy-{uuid.uuid4()}'
    async with Worker(
        client, task_queue=task_queue, workflows=[ApprovalPolicyWorkflow], plugins=[AgentPlugin(_policy_agent)]
    ):
        handle = await client.start_workflow(
            ApprovalPolicyWorkflow.run, args=['clean up'], id=f'approval-policy-{uuid.uuid4()}', task_queue=task_queue
        )
        assert await handle.result() == 'deleted a.txt, deleted b.txt'

    history = await handle.fetch_history()
    # One local activity per wait called `ask`, and nothing was signalled.
    assert len([event for event in history.events if event.HasField('marker_recorded_event_attributes')]) == 2
    assert not any(event.HasField('workflow_execution_signaled_event_attributes') for event in history.events)
    assert _deleted == ['a.txt', 'b.txt']

    await Replayer(
        workflows=[ApprovalPolicyWorkflow],
        workflow_runner=UnsandboxedWorkflowRunner(),
        data_converter=pydantic_data_converter,
    ).replay_workflow(history)
    assert _deleted == ['a.txt', 'b.txt']


async def test_a_token_from_elsewhere_is_refused(env: WorkflowEnvironment) -> None:
    with pytest.raises(UserError, match='not handed out by a wait for an external event in a Temporal workflow'):
        await resolve_external_event(env.client, 'not-a-token', {'approvals': {}})
