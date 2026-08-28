from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from datetime import timedelta
from typing import Protocol

import anyio
import pytest
from temporalio import activity, workflow
from temporalio.client import Client, WorkflowFailureError, WorkflowHistory
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.exceptions import CancelledError as TemporalCancelledError, TerminatedError
from temporalio.worker import Replayer, UnsandboxedWorkflowRunner, Worker
from temporalio.workflow import ChildWorkflowCancellationType, ChildWorkflowConfig

from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.durable_exec.temporal import AgentPlugin, TemporalDurability
from pydantic_ai.models.test import TestModel

from ._shared import TASK_QUEUE

pytestmark = pytest.mark.anyio

CHILD_WORKFLOW_TYPE = 'agent__child_workflow_agent__toolset__child_workflow_tools__call_tool__child_workflow'


@activity.defn
async def nested_child_activity(value: str) -> str:
    return f'delegated: {value}'


toolset = FunctionToolset(id='child_workflow_tools')


@toolset.tool_plain(
    metadata={
        'temporal': {
            'child_workflow': ChildWorkflowConfig(execution_timeout=timedelta(seconds=30)),
        }
    }
)
async def durable_delegate(value: str) -> str:
    return await workflow.execute_activity(
        nested_child_activity,
        value,
        start_to_close_timeout=timedelta(seconds=10),
    )


agent = Agent(
    TestModel(call_tools='all'),
    name='child_workflow_agent',
    toolsets=[toolset],
    capabilities=[TemporalDurability()],
)


@workflow.defn
class ChildWorkflowAgentWorkflow:
    @workflow.run
    async def run(self) -> str:
        return (await agent.run('delegate this')).output


def _child_start(history: WorkflowHistory):
    starts = [
        event.start_child_workflow_execution_initiated_event_attributes
        for event in history.events
        if event.HasField('start_child_workflow_execution_initiated_event_attributes')
    ]
    assert len(starts) == 1
    return starts[0]


class _HistoryHandle(Protocol):
    async def fetch_history(self) -> WorkflowHistory: ...


async def _wait_for_child_start(handle: _HistoryHandle) -> tuple[str, str]:
    with anyio.fail_after(10):
        while True:
            history = await handle.fetch_history()
            starts = [
                event.start_child_workflow_execution_initiated_event_attributes
                for event in history.events
                if event.HasField('start_child_workflow_execution_initiated_event_attributes')
            ]
            if starts:
                start = starts[0]
                return start.workflow_type.name, start.workflow_id
            await asyncio.sleep(0.01)


async def test_tool_call_runs_as_child_workflow(client: Client) -> None:
    workflow_id = 'test_tool_call_runs_as_child_workflow'
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ChildWorkflowAgentWorkflow],
        activities=[nested_child_activity],
        plugins=[AgentPlugin(agent)],
    ):
        result = await client.execute_workflow(
            ChildWorkflowAgentWorkflow.run,
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )

    assert 'delegated:' in result
    history = await client.get_workflow_handle(workflow_id).fetch_history()
    child_start = _child_start(history)
    assert child_start.workflow_type.name == CHILD_WORKFLOW_TYPE
    assert re.fullmatch(
        rf'{workflow_id}--[0-9a-f]{{8}}-[0-9a-f]{{4}}-4[0-9a-f]{{3}}-[89ab][0-9a-f]{{3}}-[0-9a-f]{{12}}',
        child_start.workflow_id,
    )

    child_history = await client.get_workflow_handle(child_start.workflow_id).fetch_history()
    nested_activity_types = [
        event.activity_task_scheduled_event_attributes.activity_type.name
        for event in child_history.events
        if event.HasField('activity_task_scheduled_event_attributes')
    ]
    assert nested_activity_types == ['nested_child_activity']

    # Replaying pins the workflow-generated UUID: a non-deterministic child ID would mismatch history.
    await Replayer(
        workflows=[ChildWorkflowAgentWorkflow],
        data_converter=pydantic_data_converter,
        workflow_runner=UnsandboxedWorkflowRunner(),
    ).replay_workflow(history)


def _cancellation_agent(name: str, cancellation_type: ChildWorkflowCancellationType) -> Agent[None, str]:
    cancellation_toolset = FunctionToolset(id='cancellation_tools')

    @cancellation_toolset.tool_plain(
        name='wait_forever',
        metadata={
            'temporal': {
                'child_workflow': ChildWorkflowConfig(
                    execution_timeout=timedelta(seconds=30),
                    cancellation_type=cancellation_type,
                ),
            }
        },
    )
    async def wait_forever() -> str:
        await workflow.wait_condition(lambda: False)
        return 'unreachable'  # pragma: no cover

    return Agent(
        TestModel(call_tools='all'),
        name=name,
        toolsets=[cancellation_toolset],
        capabilities=[TemporalDurability()],
    )


wait_cancellation_agent = _cancellation_agent(
    'wait_cancellation_agent', ChildWorkflowCancellationType.WAIT_CANCELLATION_COMPLETED
)
try_cancel_agent = _cancellation_agent('try_cancel_agent', ChildWorkflowCancellationType.TRY_CANCEL)


@workflow.defn
class WaitCancellationWorkflow:
    @workflow.run
    async def run(self) -> str:
        return (await wait_cancellation_agent.run('wait')).output


@workflow.defn
class TryCancelWorkflow:
    @workflow.run
    async def run(self) -> str:
        return (await try_cancel_agent.run('wait')).output


@pytest.mark.parametrize(
    ('parent_workflow', 'cancel_agent'),
    [
        (WaitCancellationWorkflow, wait_cancellation_agent),
        (TryCancelWorkflow, try_cancel_agent),
    ],
    ids=['wait-cancellation-completed', 'try-cancel'],
)
async def test_parent_cancellation_reaches_child_without_livelock(
    client: Client,
    parent_workflow: type[WaitCancellationWorkflow] | type[TryCancelWorkflow],
    cancel_agent: Agent[None, str],
) -> None:
    workflow_id = f'test_child_cancellation_{parent_workflow.__name__}'
    run: Callable[..., object] = parent_workflow.run
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[parent_workflow],
        plugins=[AgentPlugin(cancel_agent)],
    ):
        handle = await client.start_workflow(run, id=workflow_id, task_queue=TASK_QUEUE)
        child_type, child_id = await _wait_for_child_start(handle)
        assert child_type.endswith('__call_tool__child_workflow')

        await handle.cancel()
        with anyio.fail_after(10):
            with pytest.raises(WorkflowFailureError) as parent_error:
                await handle.result()
        assert isinstance(parent_error.value.__cause__, TemporalCancelledError)

        child_handle = client.get_workflow_handle(child_id)
        with anyio.fail_after(10):
            with pytest.raises(WorkflowFailureError) as child_error:
                await child_handle.result()

        parent_history = await handle.fetch_history()

    assert not [event for event in parent_history.events if 'WORKFLOW_TASK_FAILED' in str(event.event_type)]
    assert any(
        event.HasField('request_cancel_external_workflow_execution_initiated_event_attributes')
        for event in parent_history.events
    )
    if parent_workflow is WaitCancellationWorkflow:
        assert isinstance(child_error.value.__cause__, TemporalCancelledError)
    else:
        assert isinstance(child_error.value.__cause__, TemporalCancelledError | TerminatedError)
