"""Waiting for a decision inside a Temporal workflow, with `HandleDeferredToolCalls`.

The run calls the handler with a token that names the workflow, in a local activity whose result the history
records. When the handler answers, the run continues. Otherwise the workflow waits until a test resolves the
token with `resolve_external_event` or `fail_external_event`, which signal the workflow, or the
timeout elapses, which Temporal's time-skipping test server fast-forwards to.
"""

from __future__ import annotations

import sys
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import anyio
import pytest
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, HandleDeferredToolCalls
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolDenied
from tests._timeouts import READINESS_WAIT_TIMEOUT
from tests.temporal_utils import temporal_dev_server_cache_dir

try:
    from temporalio import workflow
    from temporalio.client import Client, WorkflowHandle
    from temporalio.common import RetryPolicy
    from temporalio.contrib.pydantic import pydantic_data_converter
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Replayer, UnsandboxedWorkflowRunner, Worker, WorkflowRunner
    from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions
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


@dataclass
class Records:
    """What the agents' functions did in the current test, as the worker's copy of this module saw it."""

    send_token: MemoryObjectSendStream[str]
    """Each token `ask` hands on, for the test to receive as the approver would."""
    receive_token: MemoryObjectReceiveStream[str]
    tokens: list[str] = field(default_factory=list[str])
    asked_about: list[list[str]] = field(default_factory=list[list[str]])
    deleted: list[str] = field(default_factory=list[str])
    model_calls: int = 0
    log: list[str] = field(default_factory=list[str])
    """What the model and the run's in-flight work did, in order."""


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
    records.tokens.append(token)
    records.asked_about.append([call.tool_call_id for call in requests.approvals])
    records.send_token.send_nowait(token)


def _delete(path: str) -> str:
    _current().deleted.append(path)
    return f'deleted {path}'


_approvals_agent = Agent(
    FunctionModel(_delete_then_report),
    name='temporal_approvals',
    deps_type=type(None),
    capabilities=[
        TemporalDurability[None](activity_config=ACTIVITY_CONFIG),
        HandleDeferredToolCalls[None](_hand_on, timeout=timedelta(minutes=30)),
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
        HandleDeferredToolCalls[None](_approve_all, timeout=timedelta(minutes=5)),
    ],
)
_policy_agent.tool_plain(name='delete', requires_approval=True)(_delete)


@workflow.defn
class ApprovalPolicyWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await _policy_agent.run(prompt)).output


_two_approvals_agent = Agent(
    FunctionModel(_delete_twice_then_report),
    name='temporal_two_approvals',
    deps_type=type(None),
    capabilities=[
        TemporalDurability[None](activity_config=ACTIVITY_CONFIG),
        HandleDeferredToolCalls[None](_hand_on, timeout=timedelta(minutes=30)),
    ],
)
_two_approvals_agent.tool_plain(name='delete', requires_approval=True)(_delete)


@workflow.defn
class TwoApprovalsWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await _two_approvals_agent.run(prompt)).output


def _logged_delete_then_report(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    _current().log.append('model request')
    return _delete_then_report(messages, info)


@dataclass
class _StartsWork(AbstractCapability[None]):
    """Starts in-flight work from `before_run`, in workflow code: a `'cancel'` piece and a `'drain'` piece."""

    async def before_run(self, ctx: RunContext[None]) -> None:
        log = _current().log

        async def never_finishes() -> None:
            try:
                await workflow.wait_condition(lambda: False)
            finally:
                log.append('cancel work stopped')

        async def reports_once_the_wait_began() -> None:
            # Workflow state, so a replay reaches it at the same point: the wait cancels this work as it begins.
            await workflow.wait_condition(lambda: cancel_work.state == 'cancelled')
            # A durable timer, which the time-skipping server only fires once the workflow has nothing else to do:
            # a run that went on past the wait would reach its next model request first.
            await workflow.sleep(timedelta(hours=1))
            log.append('drain work finished')
            ctx.enqueue('background result')

        cancel_work = ctx.in_flight.start(
            never_finishes, kind='test', name='never finishes', id='cancel-work', on_end='cancel'
        )
        ctx.in_flight.start(reports_once_the_wait_began, kind='test', name='reports', id='drain-work')


_in_flight_approvals_agent = Agent(
    FunctionModel(_logged_delete_then_report),
    name='temporal_in_flight_approvals',
    deps_type=type(None),
    capabilities=[
        _StartsWork(),
        TemporalDurability[None](activity_config=ACTIVITY_CONFIG),
        HandleDeferredToolCalls[None](_approve_all, timeout=timedelta(minutes=5)),
    ],
)
_in_flight_approvals_agent.tool_plain(name='delete', requires_approval=True)(_delete)


@workflow.defn
class InFlightApprovalsWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> list[str]:
        result = await _in_flight_approvals_agent.run(prompt)
        return [
            str(part.content)
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, UserPromptPart) and part.content != prompt
        ]


def _waits_for_the_background_result(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    texts = [part.content for message in messages for part in message.parts if isinstance(part, UserPromptPart)]
    return ModelResponse(parts=[TextPart('got it' if 'background result' in texts else 'waiting')])


@dataclass
class _StartsDrainWork(AbstractCapability[None]):
    """Starts `'drain'` work in workflow code that queues its result after the run's first model request."""

    async def before_run(self, ctx: RunContext[None]) -> None:
        async def report_later() -> None:
            # Fired by the time-skipping server once the model request's activity has completed.
            await workflow.sleep(timedelta(hours=1))
            ctx.enqueue('background result')

        ctx.in_flight.start(report_later, kind='test', name='reports later', id='drain-work')


_in_flight_end_agent = Agent(
    FunctionModel(_waits_for_the_background_result),
    name='temporal_in_flight_end',
    deps_type=type(None),
    capabilities=[_StartsDrainWork(), TemporalDurability[None](activity_config=ACTIVITY_CONFIG)],
)


@workflow.defn
class InFlightEndWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await _in_flight_end_agent.run(prompt)).output


@pytest.fixture(scope='module')
async def temporal_env() -> AsyncIterator[WorkflowEnvironment]:
    async with await WorkflowEnvironment.start_time_skipping(
        plugins=[PydanticAIPlugin()], download_dest_dir=temporal_dev_server_cache_dir()
    ) as env:
        yield env


@pytest.fixture(autouse=True)
async def records() -> AsyncIterator[Records]:
    send_token, receive_token = anyio.create_memory_object_stream[str](10)
    async with send_token, receive_token:
        _records.append(Records(send_token=send_token, receive_token=receive_token))
        yield _current()


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


async def _until_asked(handle: WorkflowHandle[Any, str], records: Records) -> str:
    """Wait until `ask` handed the token on and the history recorded it, and return the token."""
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        token = await records.receive_token.receive()
    # Temporal answers a query once the workflow task in progress, which ran `ask`, has completed.
    await handle.query('__stack_trace')
    return token


async def test_the_run_waits_for_the_decision_and_a_new_worker_resumes_it_without_asking_again(
    temporal_env: WorkflowEnvironment, records: Records
) -> None:
    client = temporal_env.client
    task_queue = f'approvals-{uuid.uuid4()}'
    # Caching nothing keeps the workflow off this worker's sticky queue, so the next worker picks it up right away.
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ApprovalsWorkflow],
        plugins=[AgentPlugin(_approvals_agent)],
        workflow_runner=UnsandboxedWorkflowRunner(),
        max_cached_workflows=0,
    ):
        handle = await _start(client, task_queue)
        token = await _until_asked(handle, records)
    assert token.startswith(f'{handle.id}:'), 'the token names the workflow'
    assert records.asked_about == [['call-1']]
    assert records.deleted == []

    # The worker that asked is gone. The decision is signalled to the workflow, and a new worker replays it.
    decision = {'approvals': {'call-1': {'kind': 'tool-approved', 'override_args': {'path': 'b.txt'}}}}
    await resolve_external_event(client, token, decision)
    async with _worker(client, task_queue, ApprovalsWorkflow, _approvals_agent):
        assert await handle.result() == 'success: deleted b.txt'

    assert records.deleted == ['b.txt']
    assert records.asked_about == [['call-1']], 'the replay does not ask again'
    assert records.model_calls == 2, 'the first model response came from the history'

    await Replayer(
        workflows=[ApprovalsWorkflow],
        workflow_runner=UnsandboxedWorkflowRunner(),
        data_converter=pydantic_data_converter,
    ).replay_workflow(await handle.fetch_history())
    assert records.asked_about == [['call-1']]
    assert records.deleted == ['b.txt']


async def test_a_denial_reaches_the_model_without_running_the_tool(
    temporal_env: WorkflowEnvironment, records: Records
) -> None:
    """In the workflow sandbox, with the module that defines `ask` passed through, so `ask` shares the worker's state."""
    client = temporal_env.client
    task_queue = f'approvals-{uuid.uuid4()}'
    runner = SandboxedWorkflowRunner(restrictions=SandboxRestrictions.default.with_passthrough_modules(__name__))
    async with _worker(client, task_queue, ApprovalsWorkflow, _approvals_agent, runner):
        handle = await _start(client, task_queue)
        with anyio.fail_after(READINESS_WAIT_TIMEOUT):
            token = await records.receive_token.receive()
        await resolve_external_event(client, token, DeferredToolResults(approvals={'call-1': ToolDenied('Not today.')}))
        assert await handle.result() == 'denied: Not today.'

    assert records.deleted == []


async def test_a_failed_wait_denies_the_call(temporal_env: WorkflowEnvironment, records: Records) -> None:
    client = temporal_env.client
    task_queue = f'approvals-{uuid.uuid4()}'
    async with _worker(client, task_queue, ApprovalsWorkflow, _approvals_agent):
        handle = await _start(client, task_queue)
        token = await _until_asked(handle, records)
        await fail_external_event(client, token, 'The approver is away.')
        assert await handle.result() == 'denied: No decision on this tool call arrived: The approver is away.'

    assert records.deleted == []


async def test_a_wait_that_times_out_denies_the_call(temporal_env: WorkflowEnvironment, records: Records) -> None:
    """Awaiting the result lets the time-skipping server jump over the 30-minute timeout."""
    client = temporal_env.client
    task_queue = f'approvals-{uuid.uuid4()}'
    async with _worker(client, task_queue, ApprovalsWorkflow, _approvals_agent):
        handle = await _start(client, task_queue)
        assert await handle.result() == (
            'denied: No decision on this tool call arrived: No resolution arrived within 0:30:00.'
        )

    assert records.asked_about == [['call-1']]
    assert records.deleted == []


async def test_answers_from_ask_are_recorded_and_the_run_continues_without_waiting(
    temporal_env: WorkflowEnvironment, records: Records
) -> None:
    """In the default workflow sandbox: each wait's `ask` answers in place, and a replay reuses the answers."""
    client = temporal_env.client
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
    markers = [event.marker_recorded_event_attributes.marker_name for event in history.events]
    assert markers.count('core_local_activity') == 2
    assert not any(event.HasField('workflow_execution_signaled_event_attributes') for event in history.events)
    assert records.deleted == ['a.txt', 'b.txt']

    await Replayer(
        workflows=[ApprovalPolicyWorkflow],
        workflow_runner=UnsandboxedWorkflowRunner(),
        data_converter=pydantic_data_converter,
    ).replay_workflow(history)
    assert records.deleted == ['a.txt', 'b.txt']


async def test_a_resolution_for_a_wait_that_ended_is_discarded(
    temporal_env: WorkflowEnvironment, records: Records
) -> None:
    """A late failure of the first wait's token, sent while the run waits again, changes nothing."""
    client = temporal_env.client
    task_queue = f'two-approvals-{uuid.uuid4()}'
    async with _worker(client, task_queue, TwoApprovalsWorkflow, _two_approvals_agent):
        handle = await client.start_workflow(
            TwoApprovalsWorkflow.run, args=['clean up'], id=f'two-approvals-{uuid.uuid4()}', task_queue=task_queue
        )
        first = await _until_asked(handle, records)
        await resolve_external_event(client, first, {'approvals': {'call-a.txt': True}})
        second = await _until_asked(handle, records)
        await fail_external_event(client, first, 'Sent too late.')
        await resolve_external_event(client, second, {'approvals': {'call-b.txt': True}})
        assert await handle.result() == 'deleted a.txt, deleted b.txt'

    assert records.asked_about == [['call-a.txt'], ['call-b.txt']]
    assert records.deleted == ['a.txt', 'b.txt']


async def test_the_wait_cancels_cancel_work_and_returns_once_drain_work_has_finished(
    temporal_env: WorkflowEnvironment, records: Records
) -> None:
    """In-flight work started in workflow code keeps running across the wait, which applies each piece's `on_end`."""
    client = temporal_env.client
    task_queue = f'in-flight-approvals-{uuid.uuid4()}'
    async with _worker(client, task_queue, InFlightApprovalsWorkflow, _in_flight_approvals_agent):
        handle = await client.start_workflow(
            InFlightApprovalsWorkflow.run,
            args=['clean up'],
            id=f'in-flight-approvals-{uuid.uuid4()}',
            task_queue=task_queue,
        )
        assert await handle.result() == ['background result']

    assert records.log == [
        'model request',
        'cancel work stopped',
        'drain work finished',
        'model request',
    ]
    assert records.deleted == ['a.txt']

    await Replayer(
        workflows=[InFlightApprovalsWorkflow],
        workflow_runner=UnsandboxedWorkflowRunner(),
        data_converter=pydantic_data_converter,
    ).replay_workflow(await handle.fetch_history())


async def test_the_run_waits_to_end_for_drain_work_started_in_workflow_code(temporal_env: WorkflowEnvironment) -> None:
    """The model ends the run before the work queued its result: the run waits, then gives the model a turn on it."""
    client = temporal_env.client
    task_queue = f'in-flight-end-{uuid.uuid4()}'
    async with _worker(client, task_queue, InFlightEndWorkflow, _in_flight_end_agent):
        handle = await client.start_workflow(
            InFlightEndWorkflow.run, args=['go'], id=f'in-flight-end-{uuid.uuid4()}', task_queue=task_queue
        )
        assert await handle.result() == 'got it'

    await Replayer(
        workflows=[InFlightEndWorkflow],
        workflow_runner=UnsandboxedWorkflowRunner(),
        data_converter=pydantic_data_converter,
    ).replay_workflow(await handle.fetch_history())


async def test_a_token_from_elsewhere_is_refused(temporal_env: WorkflowEnvironment) -> None:
    with pytest.raises(UserError, match='not handed out by a wait for an external event in a Temporal workflow'):
        await resolve_external_event(temporal_env.client, 'not-a-token', {'approvals': {}})
