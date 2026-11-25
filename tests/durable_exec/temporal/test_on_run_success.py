"""`on_run_success` in a Temporal workflow, committing through a durable operation that runs as an activity.

In a module of its own: building a durable operation for a hook takes a while, and a Temporal workflow imports its
module again in the sandbox for every run, so an agent with one makes every workflow defined beside it slower.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from pydantic_ai import Agent, AgentRunResult, RunContext
from pydantic_ai.capabilities import AbstractCapability, durable_operation
from pydantic_ai.models.test import TestModel

try:
    from temporalio import activity, workflow
    from temporalio.client import Client
    from temporalio.worker import Worker

    from pydantic_ai.durable_exec.temporal import AgentPlugin, TemporalDurability

    from ._shared import BASE_ACTIVITY_CONFIG, TASK_QUEUE
except ImportError:  # pragma: lax no cover
    pytest.skip('temporal not installed', allow_module_level=True)

pytestmark = [pytest.mark.xdist_group(name='temporal-on-run-success')]

_saved: list[tuple[str, bool]] = []


class _SaveOnSuccess(AbstractCapability[Any]):
    """Save each run's output once the run has succeeded, through an activity.

    Example:
        agent = Agent(model, capabilities=[_SaveOnSuccess(), TemporalDurability()])
    """

    id = 'save_on_success'

    async def on_run_success(self, ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> None:
        await self.save(ctx, output=str(result.output))

    @durable_operation('save')
    async def save(self, ctx: RunContext[Any], *, output: str) -> None:
        _saved.append((output, activity.in_activity()))


_agent = Agent(
    TestModel(custom_output_text='done'),
    name='on_run_success_agent',
    capabilities=[_SaveOnSuccess(), TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@workflow.defn
class OnRunSuccessWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await _agent.run(prompt)).output


async def test_on_run_success_commits_through_an_activity(client: Client):
    async with Worker(client, task_queue=TASK_QUEUE, workflows=[OnRunSuccessWorkflow], plugins=[AgentPlugin(_agent)]):
        output = await client.execute_workflow(
            OnRunSuccessWorkflow.run,
            args=['Hello'],
            id=OnRunSuccessWorkflow.__name__,
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=30),
        )

    assert output == 'done'
    assert _saved == [('done', True)]
