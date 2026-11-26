"""Capability hooks that run as durable operations under Temporal: a `RetryModelRequest` one raises, and one held to a
lower authority that hands its request on unchanged.

In a module of its own: building a durable operation for a hook takes a while, and a Temporal workflow imports its
module again in the sandbox for every run, so an agent with one makes every workflow defined beside it slower.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from pydantic_ai import Agent, ModelMessage, ModelRequest, ModelResponse, RunContext, TextPart, UserPromptPart
from pydantic_ai.capabilities import AbstractCapability, durable_operation
from pydantic_ai.exceptions import RetryModelRequest
from pydantic_ai.messages import SystemPromptPart
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


class _HandsOnDurably(AbstractCapability[Any]):
    id = 'hands_on_durably'

    @durable_operation('before_model_request')
    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        return request_context


_sent: list[list[str]] = []
"""The parts of the first request each model call was sent, by type: the model runs in an activity of this process."""


def _recording_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    first = messages[0]
    assert isinstance(first, ModelRequest)
    _sent.append([type(part).__name__ for part in first.parts])
    return ModelResponse(parts=[TextPart('done')])


_hands_on_agent = Agent(
    FunctionModel(_recording_model_fn),
    name='durable_hook_hands_on',
    capabilities=[_HandsOnDurably().with_authority('user'), TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)

_T = datetime(2026, 1, 1, tzinfo=UTC)


@workflow.defn
class DurableHookHandsOnWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        history: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    # Comes back from the activity a list, which the model doesn't read.
                    SystemPromptPart('Be brief.', metadata={'pair': (1, 2)}, timestamp=_T),
                    UserPromptPart('Hi.', timestamp=_T),
                ],
                timestamp=_T,
            ),
            ModelResponse(parts=[TextPart('Hello.')], timestamp=_T),
        ]
        return (await _hands_on_agent.run(prompt, message_history=history)).output


async def test_durability_a_lowered_capability_can_hand_on_the_request_its_durable_hook_gets(client: Client):
    """The activity sends the messages back rebuilt, which isn't a change, so the system prompt keeps its voice."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurableHookHandsOnWorkflow],
        plugins=[AgentPlugin(_hands_on_agent)],
    ):
        _sent.clear()
        output = await client.execute_workflow(
            DurableHookHandsOnWorkflow.run,
            args=['Hello'],
            id=DurableHookHandsOnWorkflow.__name__,
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=30),
        )
    assert output == 'done'
    assert _sent == [['SystemPromptPart', 'UserPromptPart']]
