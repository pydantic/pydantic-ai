"""Session resources must execute in activities, never while replaying workflow history."""

from __future__ import annotations

import sys
import uuid
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any

import pytest

try:
    from temporalio import activity, workflow
    from temporalio.client import Client
    from temporalio.worker import Replayer, Worker

    from pydantic_ai.durable_exec.temporal import (
        AgentPlugin,
        PydanticAIPlugin,
        TemporalAgent,  # pyright: ignore[reportDeprecated]
        TemporalDurability,
    )
except ImportError:
    pytest.skip('temporal not installed', allow_module_level=True)

if sys.version_info >= (3, 14):
    pytest.skip('Temporal sandbox does not support Python 3.14', allow_module_level=True)

with workflow.unsafe.imports_passed_through():
    from pydantic_ai import Agent, RunContext
    from pydantic_ai._warnings import PydanticAIDeprecationWarning
    from pydantic_ai.capabilities import AbstractCapability, WrapperCapability
    from pydantic_ai.exceptions import UserError
    from pydantic_ai.messages import ModelMessage, ToolReturnPart
    from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters, ModelSelectionContext
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.realtime import RealtimeModel, RealtimeModelProfile, RealtimeModelSettings
    from pydantic_ai.realtime.codec import RealtimeConnection

    from ._shared import BASE_ACTIVITY_CONFIG


pytestmark = pytest.mark.xdist_group(name='temporal-session')

resource_events: list[str] = []


class ActivitySessionModel(TestModel):
    async def __aenter__(self):
        assert not workflow.in_workflow()
        resource_events.append('client enter')
        return self

    async def __aexit__(self, *args: Any):
        resource_events.append('client exit')

    @asynccontextmanager
    async def open_session(self) -> AsyncGenerator[Model]:
        assert activity.in_activity()
        resource_events.append('interaction enter')
        try:
            yield TestModel(custom_output_text='bound inside activity')
        finally:
            resource_events.append('interaction exit')


async def record_tool() -> str:
    assert activity.in_activity()
    resource_events.append('tool executed')
    return 'recorded'


registered_model = ActivitySessionModel()


class SelectSessionModel(AbstractCapability[None]):
    def get_model(self):
        def select(ctx: ModelSelectionContext[None]) -> Model:
            return registered_model

        return select


class ReplaceSessionModel(AbstractCapability[None]):
    async def before_model_request(self, ctx: RunContext[None], request_context: ModelRequestContext):
        request_context.model = registered_model
        return request_context


session_agents = {
    mode: Agent(
        ActivitySessionModel(),
        deps_type=type(None),
        name=f'session_activity_owner_{mode}',
        tools=[record_tool],
        capabilities=[
            *capabilities,
            WrapperCapability(
                TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG, models={'registered': registered_model})
            ),
        ],
    )
    for mode, capabilities in {
        'direct': [],
        'selector': [SelectSessionModel()],
        'hook': [ReplaceSessionModel()],
    }.items()
}

with pytest.warns(PydanticAIDeprecationWarning, match='`TemporalAgent` is deprecated'):
    legacy_session_agent = TemporalAgent(  # pyright: ignore[reportDeprecated]
        Agent(ActivitySessionModel(), name='legacy_session_activity_owner', tools=[record_tool]),
        activity_config=BASE_ACTIVITY_CONFIG,
    )


class UnreachableRealtimeModel(RealtimeModel):
    @property
    def model_name(self) -> str:
        return 'unreachable'

    @property
    def system(self) -> str:
        return 'test'

    @property
    def profile(self) -> RealtimeModelProfile:
        return RealtimeModelProfile()

    @asynccontextmanager
    async def connect(
        self,
        *,
        messages: Sequence[ModelMessage],
        model_settings: RealtimeModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncGenerator[RealtimeConnection]:
        resource_events.append('unexpected realtime connection')
        raise UserError('Unexpected realtime connection inside workflow')
        yield  # pragma: no cover


@workflow.defn
class RealtimeGuardWorkflow:
    @workflow.run
    async def run(self, mode: str, entry: str) -> str:
        agent = legacy_session_agent if mode == 'legacy' else session_agents['direct']
        model = UnreachableRealtimeModel()
        try:
            if entry == 'session':
                async with agent.realtime(model).session():
                    pass
            else:
                async with agent.session() as owner:
                    if entry == 'owned-session':
                        async with owner.realtime(model).session():
                            pass
                    else:
                        async with owner.realtime(model).connect() as connection:
                            async with connection.run():
                                pass
        except UserError as exc:
            return str(exc)
        return 'Unexpected successful realtime entry'


@pytest.mark.parametrize('mode', ['direct', 'legacy'])
@pytest.mark.parametrize('entry', ['session', 'owned-session', 'connection'])
async def test_session_realtime_rejected_before_connection(client: Client, mode: str, entry: str):
    resource_events.clear()
    task_queue = f'realtime-guard-{uuid.uuid4()}'
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[RealtimeGuardWorkflow],
        plugins=[AgentPlugin(legacy_session_agent if mode == 'legacy' else session_agents['direct'])],
    ):
        error = await client.execute_workflow(
            RealtimeGuardWorkflow.run,
            args=[mode, entry],
            id=f'realtime-guard-{uuid.uuid4()}',
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=60),
        )
    assert 'cannot be used inside a Temporal workflow' in error
    assert resource_events == []


@workflow.defn
class SessionWorkflow:
    @workflow.run
    async def run(self, mode: str) -> list[str]:
        agent = legacy_session_agent if mode == 'legacy' else session_agents[mode]
        async with agent.session() as session:
            first = await session.run('first')
            if mode == 'legacy':
                second = await session.run('second')
            else:
                async with session.run_stream_events('second') as events:
                    async for _ in events:
                        pass
                    second = events.result
            assert second is not None
            assert first.run_id != second.run_id
            assert len(session.conversation.messages) == 6
            (operation,) = session.state.operations
            assert operation.run_id == first.run_id
            assert (operation.execution, operation.delivery) == ('completed', 'committed')
            # These assertions also run during Replayer: engine-cached outcomes reconstruct the
            # projection without either opening an interaction or re-executing the tool.
            returned = operation.result[0].parts[0]
            assert isinstance(returned, ToolReturnPart)
            assert returned.content == 'recorded'
            return [first.output, second.output]


@pytest.mark.parametrize('mode', ['direct', 'selector', 'hook', 'legacy'])
async def test_session_resources_are_activity_owned_and_not_replayed(client: Client, mode: str):
    resource_events.clear()
    task_queue = f'session-{uuid.uuid4()}'
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[SessionWorkflow],
        plugins=[AgentPlugin(legacy_session_agent if mode == 'legacy' else session_agents[mode])],
    ):
        handle = await client.start_workflow(
            SessionWorkflow.run,
            mode,
            id=f'session-{uuid.uuid4()}',
            task_queue=task_queue,
            execution_timeout=timedelta(seconds=60),
        )
        assert await handle.result() == ['bound inside activity', 'bound inside activity']
        history = await handle.fetch_history()

    # Registered clients retain their external owner, but interactions belong to each activity.
    assert resource_events == [
        'interaction enter',
        'interaction exit',
        'tool executed',
        'interaction enter',
        'interaction exit',
        'interaction enter',
        'interaction exit',
    ]
    replay = await Replayer(workflows=[SessionWorkflow], plugins=[PydanticAIPlugin()]).replay_workflow(history)
    assert replay.replay_failure is None
    assert resource_events == [
        'interaction enter',
        'interaction exit',
        'tool executed',
        'interaction enter',
        'interaction exit',
        'interaction enter',
        'interaction exit',
    ]
