"""Capability hooks that run as durable operations under Temporal: a `RetryModelRequest` one raises, and an
`after_model_request` that returns its response.

In a module of its own: building a durable operation for a hook takes a while, and a Temporal workflow imports its
module again in the sandbox for every run, so an agent with one makes every workflow defined beside it slower.
"""

from __future__ import annotations

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
    from temporalio.worker import Worker

    from pydantic_ai.durable_exec.temporal import AgentPlugin, TemporalDurability

    from ._shared import BASE_ACTIVITY_CONFIG, TASK_QUEUE
except ImportError:  # pragma: lax no cover
    pytest.skip('temporal not installed', allow_module_level=True)

pytestmark = [pytest.mark.xdist_group(name='temporal-durable-hook-retry')]


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


async def test_durability_hook_running_as_an_activity_can_reroute_the_request(client: Client):
    """A `RetryModelRequest` a durable capability hook raises in its activity reaches the attempt loop as itself."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurableHookRerouteWorkflow],
        plugins=[AgentPlugin(_reroute_agent)],
    ):
        output = await client.execute_workflow(
            DurableHookRerouteWorkflow.run,
            args=['Hello'],
            id=DurableHookRerouteWorkflow.__name__,
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=30),
        )
    assert output == 'from the rerouted model'


class _ChecksTheResponseDurably(AbstractCapability[Any]):
    id = 'checks_the_response'

    @durable_operation('after_model_request')
    async def after_model_request(
        self, ctx: RunContext[Any], *, request_context: ModelRequestContext, response: ModelResponse
    ) -> ModelResponse:
        return ModelResponse(parts=[TextPart(f'Checked: {response.text}')])


_checks_agent = Agent(
    TestModel(custom_output_text='Done.'),
    name='durable_after_model_request',
    capabilities=[
        _ChecksTheResponseDurably(),
        TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG),
    ],
)


@workflow.defn
class DurableAfterModelRequestWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await _checks_agent.run(prompt)).output


async def test_durability_after_model_request_running_as_an_activity_returns_its_response(client: Client):
    """The activity takes the request context but returns a response, which comes back as the response."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurableAfterModelRequestWorkflow],
        plugins=[AgentPlugin(_checks_agent)],
    ):
        output = await client.execute_workflow(
            DurableAfterModelRequestWorkflow.run,
            args=['Hello'],
            id=DurableAfterModelRequestWorkflow.__name__,
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=30),
        )
    assert output == 'Checked: Done.'
