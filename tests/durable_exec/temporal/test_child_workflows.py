from __future__ import annotations

from datetime import timedelta

import pytest
from temporalio import workflow
from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.workflow import ChildWorkflowConfig

from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.durable_exec.temporal import AgentPlugin, TemporalDurability
from pydantic_ai.models.test import TestModel

from ._shared import TASK_QUEUE

pytestmark = pytest.mark.anyio

toolset = FunctionToolset(id='child_workflow_tools')


@toolset.tool_plain(
    metadata={
        'temporal': {
            'child_workflow': ChildWorkflowConfig(execution_timeout=timedelta(seconds=30)),
        }
    }
)
async def durable_delegate(value: str) -> str:
    return f'delegated: {value}'


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


async def test_tool_call_runs_as_child_workflow(client: Client) -> None:
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ChildWorkflowAgentWorkflow],
        plugins=[AgentPlugin(agent)],
    ):
        result = await client.execute_workflow(
            ChildWorkflowAgentWorkflow.run,
            id='test_tool_call_runs_as_child_workflow',
            task_queue=TASK_QUEUE,
        )

    assert 'delegated:' in result
    handle = client.get_workflow_handle('test_tool_call_runs_as_child_workflow')
    history = await handle.fetch_history()
    assert any(event.HasField('start_child_workflow_execution_initiated_event_attributes') for event in history.events)
