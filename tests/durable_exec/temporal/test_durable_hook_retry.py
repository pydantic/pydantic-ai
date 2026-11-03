"""A `RetryModelRequest` raised by a capability hook that runs as a durable operation under Temporal.

In a module of its own: building a durable operation for a hook takes a while, and a Temporal workflow imports its
module again in the sandbox for every run, so an agent with one makes every workflow defined beside it slower.
The test runs on a time-skipping environment so that this real bootstrap time never consumes the workflow's
execution deadline.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

import pytest

from pydantic_ai import Agent, ModelMessage, ModelResponse, RunContext, TextPart
from pydantic_ai.capabilities import AbstractCapability, durable_operation
from pydantic_ai.exceptions import RetryModelRequest
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

try:
    from temporalio import workflow
    from temporalio.client import Client
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Worker

    from pydantic_ai.durable_exec.temporal import AgentPlugin, PydanticAIPlugin, TemporalDurability

    from ._shared import BASE_ACTIVITY_CONFIG, TASK_QUEUE
except ImportError:  # pragma: lax no cover
    pytest.skip('temporal not installed', allow_module_level=True)

pytestmark = [pytest.mark.xdist_group(name='temporal-durability')]


def _rerouted_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('from the rerouted model')])


class _RerouteFirstAttempt(AbstractCapability[Any]):
    id = 'reroute_first_attempt'

    @durable_operation('prepare')
    async def prepare_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        if request_context.attempt == 1:
            raise RetryModelRequest('rerouted')
        return request_context


_reroute_agent = Agent(
    # Never called: the hook moves the first attempt to the rerouted model before anything is sent.
    TestModel(),
    name='durable_hook_reroute',
    capabilities=[
        _RerouteFirstAttempt(),
        TemporalDurability(
            activity_config=BASE_ACTIVITY_CONFIG,
            models={'rerouted': FunctionModel(_rerouted_model_fn, model_name='rerouted')},
        ),
    ],
)


@workflow.defn
class DurableHookRerouteWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await _reroute_agent.run(prompt)).output


@pytest.fixture(scope='module')
async def time_skipping_client() -> AsyncIterator[Client]:
    """A client whose environment skips time, so real bootstrap time on slow runners cannot consume the deadline."""
    async with await WorkflowEnvironment.start_time_skipping(plugins=[PydanticAIPlugin()]) as env:
        yield env.client


async def test_durability_hook_running_as_an_activity_can_reroute_the_request(time_skipping_client: Client):
    """A `RetryModelRequest` a durable capability hook raises in its activity reaches the attempt loop as itself."""
    async with Worker(
        time_skipping_client,
        task_queue=TASK_QUEUE,
        workflows=[DurableHookRerouteWorkflow],
        plugins=[AgentPlugin(_reroute_agent)],
    ):
        output = await time_skipping_client.execute_workflow(
            DurableHookRerouteWorkflow.run,
            args=['Hello'],
            id=DurableHookRerouteWorkflow.__name__,
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=30),
        )
    assert output == 'from the rerouted model'
