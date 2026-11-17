"""A `RetryModelRequest` raised by a capability hook that runs as a durable operation under Temporal, and
capability operations that run as local activities.

In a module of its own: building a durable operation for a hook takes a while, and a Temporal workflow imports its
module again in the sandbox for every run, so an agent with one makes every workflow defined beside it slower.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, ModelMessage, ModelResponse, RunContext, TextPart
from pydantic_ai.capabilities import AbstractCapability, durable_operation
from pydantic_ai.exceptions import RetryModelRequest
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

try:
    from temporalio import activity, workflow
    from temporalio.api.enums.v1 import EventType
    from temporalio.client import Client
    from temporalio.worker import Replayer, Worker

    from pydantic_ai.durable_exec.temporal import AgentPlugin, PydanticAIPlugin, TemporalDurability

    from ._shared import BASE_ACTIVITY_CONFIG, TASK_QUEUE, scheduled_activity_types
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


def _echo_instructions_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(info.instructions or '')])


def _where() -> str:
    info = activity.info()
    return f'{info.activity_type} local={info.is_local}'


class _LocalOperations(AbstractCapability[Any]):
    id = 'local_operations'

    @durable_operation('turn', local=True)
    async def turn(self, ctx: RunContext[Any], value: str) -> str:
        return f'{value}: {_where()}'

    @durable_operation('remote')
    async def remote(self, ctx: RunContext[Any], value: str) -> str:
        return f'{value}: {_where()}'

    @durable_operation('prepare', local=True)
    async def prepare_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        if request_context.attempt == 1:
            raise RetryModelRequest('rerouted')
        return request_context

    def get_instructions(self):
        async def instructions(ctx: RunContext[Any]) -> str:
            return f'{await self.turn(ctx, "a")}\n{await self.remote(ctx, "b")}'

        return instructions


_local_agent = Agent(
    # Never called: the local hook moves the first attempt to the rerouted model.
    TestModel(),
    name='local_operations',
    capabilities=[
        _LocalOperations(),
        TemporalDurability(
            activity_config=BASE_ACTIVITY_CONFIG,
            models={'rerouted': FunctionModel(_echo_instructions_model_fn, model_name='rerouted')},
        ),
    ],
)


@workflow.defn
class LocalOperationsWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await _local_agent.run(prompt)).output


async def test_durability_local_operation_runs_as_a_local_activity(client: Client):
    """A `local=True` operation runs as a local activity, its errors rebuilt; an unmarked one stays an activity."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[LocalOperationsWorkflow],
        plugins=[AgentPlugin(_local_agent)],
    ):
        handle = await client.start_workflow(
            LocalOperationsWorkflow.run,
            args=['Hello'],
            id=LocalOperationsWorkflow.__name__,
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=30),
        )
        output = await handle.result()
    assert output == (
        'a: agent__local_operations__capability__local_operations__turn local=True\n'
        'b: agent__local_operations__capability__local_operations__remote local=False'
    )

    history = await handle.fetch_history()
    local_activity_markers = [
        event
        for event in history.events
        if event.event_type == EventType.EVENT_TYPE_MARKER_RECORDED
        and event.marker_recorded_event_attributes.marker_name == 'core_local_activity'
    ]
    # The `turn` operation once, and the `prepare` hook twice (its first attempt asks for a retry).
    assert len(local_activity_markers) == 3
    assert scheduled_activity_types(history) == snapshot(
        ['agent__local_operations__capability__local_operations__remote', 'agent__local_operations__model_request']
    )

    replay = await Replayer(workflows=[LocalOperationsWorkflow], plugins=[PydanticAIPlugin()]).replay_workflow(history)
    assert replay.replay_failure is None
