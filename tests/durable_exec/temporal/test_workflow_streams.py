from __future__ import annotations

import asyncio
import sys
import uuid
from collections.abc import AsyncIterable, Awaitable, Callable
from contextlib import aclosing
from dataclasses import replace
from datetime import timedelta
from typing import Any

import anyio
import pytest

from pydantic_ai import Agent, AgentStreamEvent, PartDeltaEvent, PartEndEvent, PartStartEvent, RunContext, RunUsage
from pydantic_ai._utils import BaseExceptionGroup
from pydantic_ai.capabilities import ProcessEventStream
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel

try:
    from temporalio import workflow
    from temporalio.client import Client, WorkflowHandle
    from temporalio.contrib.pydantic import pydantic_data_converter
    from temporalio.contrib.workflow_streams import WorkflowStream
    from temporalio.testing import ActivityEnvironment
    from temporalio.worker import Replayer, UnsandboxedWorkflowRunner, Worker

    from pydantic_ai.durable_exec.temporal import (
        AgentPlugin,
        TemporalDurability,
        stream_agent_events,
        workflow_stream_event_handler,
    )
    from pydantic_ai.durable_exec.temporal._event_stream import combine_event_stream_handlers
except ImportError:  # pragma: lax no cover
    pytest.skip('temporal not installed', allow_module_level=True)

if sys.version_info >= (3, 14):  # pragma: lax no cover
    pytest.skip(
        'temporalio sandbox is incompatible with Python 3.14: '
        'sandbox module state accumulates across validation cycles causing import failures after ~22 workflows '
        '(remove when https://github.com/temporalio/sdk-python/issues/1326 closes)',
        allow_module_level=True,
    )

with workflow.unsafe.imports_passed_through():
    from ._shared import BASE_ACTIVITY_CONFIG, TASK_QUEUE, _stream_fn_model  # pyright: ignore[reportPrivateUsage]

pytestmark = [pytest.mark.anyio, pytest.mark.xdist_group(name='temporal-durability')]


_teed_events: list[AgentStreamEvent] = []


async def _teed_handler(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
    async for event in stream:
        _teed_events.append(event)


_stream_agent = Agent(
    _stream_fn_model,
    name='durability_workflow_stream_agent',
    capabilities=[
        TemporalDurability(
            activity_config=BASE_ACTIVITY_CONFIG,
            event_stream_topic='agent_events',
            event_stream_handler=_teed_handler,
        )
    ],
)


@workflow.defn
class WorkflowStreamAgentWorkflow:
    @workflow.init
    def __init__(self, prompt: str) -> None:
        self.stream = WorkflowStream()
        self._released = False

    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await _stream_agent.run(prompt)
        try:
            await workflow.wait_condition(lambda: self._released, timeout=timedelta(seconds=30))
        except asyncio.TimeoutError:
            pass
        self.stream.detach_pollers()
        await workflow.wait_condition(workflow.all_handlers_finished)
        return result.output

    @workflow.signal
    def release(self) -> None:
        self._released = True


async def _collect_events(
    client: Client,
    handle: WorkflowHandle[Any, Any],
    release: Callable[[], Awaitable[Any]],
    *,
    topic: str = 'agent_events',
) -> list[AgentStreamEvent]:
    events: list[AgentStreamEvent] = []
    released = False
    async for event in stream_agent_events(client, handle, topic, poll_cooldown=timedelta(milliseconds=50)):
        events.append(event)
        if isinstance(event, PartEndEvent) and not released:
            released = True
            await release()
    return events


async def test_streaming_to_workflow_stream(client: Client) -> None:
    """The topic publisher and a user handler both receive model stream events."""
    _teed_events.clear()
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[WorkflowStreamAgentWorkflow],
        plugins=[AgentPlugin(_stream_agent)],
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        handle = await client.start_workflow(
            WorkflowStreamAgentWorkflow.run,
            args=['Hello'],
            id=f'{WorkflowStreamAgentWorkflow.__name__}-{uuid.uuid4()}',
            task_queue=TASK_QUEUE,
        )
        events = await asyncio.wait_for(
            _collect_events(client, handle, lambda: handle.signal(WorkflowStreamAgentWorkflow.release)),
            timeout=60,
        )
        output = await handle.result()

    assert output == 'Streamed response'
    assert any(isinstance(event, PartStartEvent) for event in events)
    assert any(isinstance(event, PartDeltaEvent) for event in events)
    assert any(isinstance(event, PartStartEvent) for event in _teed_events)


async def test_publisher_failure_cancels_handler_and_fails_activity() -> None:
    handler_started = anyio.Event()
    handler_cancelled = anyio.Event()

    async def publisher(ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]) -> None:
        await handler_started.wait()
        raise RuntimeError('publisher failed')

    async def handler(ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]) -> None:
        handler_started.set()
        try:
            await anyio.sleep_forever()
        finally:
            handler_cancelled.set()

    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage(), run_id='publisher-failure')
    combined = combine_event_stream_handlers(publisher, handler)
    send, stream = anyio.create_memory_object_stream[AgentStreamEvent]()
    send.close()

    async with stream:
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await combined(ctx, stream)

    assert exc_info.value.subgroup(RuntimeError) is not None
    assert handler_cancelled.is_set()


async def test_handler_failure_interrupts_publisher_and_fails_activity() -> None:
    publisher_started = anyio.Event()
    publisher_cancelled = anyio.Event()

    async def publisher(ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]) -> None:
        publisher_started.set()
        try:
            await anyio.sleep_forever()
        finally:
            publisher_cancelled.set()

    async def handler(ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]) -> None:
        await publisher_started.wait()
        raise RuntimeError('handler failed')

    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage(), run_id='handler-failure')
    combined = combine_event_stream_handlers(publisher, handler)
    send, stream = anyio.create_memory_object_stream[AgentStreamEvent]()
    send.close()

    async with stream:
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await combined(ctx, stream)

    assert exc_info.value.subgroup(RuntimeError) is not None
    assert publisher_cancelled.is_set()


_offset_agent = Agent(
    _stream_fn_model,
    name='durability_offsets_stream_agent',
    capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG, event_stream_topic='agent_events')],
)


@workflow.defn
class WorkflowStreamOffsetsWorkflow:
    @workflow.init
    def __init__(self, prompt: str) -> None:
        self.stream = WorkflowStream()
        self._released = False
        other = self.stream.topic('other_events')
        for i in range(3):
            other.publish(f'noise-{i}')

    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await _offset_agent.run(prompt)
        try:
            await workflow.wait_condition(lambda: self._released, timeout=timedelta(seconds=30))
        except asyncio.TimeoutError:
            pass
        self.stream.detach_pollers()
        await workflow.wait_condition(workflow.all_handlers_finished)
        return result.output

    @workflow.signal
    def release(self) -> None:
        self._released = True

    @workflow.signal
    def publish_noise(self) -> None:
        other = self.stream.topic('other_events')
        for i in range(3, 6):
            other.publish(f'noise-{i}')


async def test_workflow_stream_offsets_support_resume(client: Client) -> None:
    """A consumer can resume at the next global stream offset without gaps or duplicates."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[WorkflowStreamOffsetsWorkflow],
        plugins=[AgentPlugin(_offset_agent)],
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        handle = await client.start_workflow(
            WorkflowStreamOffsetsWorkflow.run,
            args=['Hello'],
            id=f'{WorkflowStreamOffsetsWorkflow.__name__}-{uuid.uuid4()}',
            task_queue=TASK_QUEUE,
        )
        async with aclosing(
            stream_agent_events(
                client,
                handle,
                'agent_events',
                poll_cooldown=timedelta(milliseconds=50),
                with_offsets=True,
            )
        ) as subscription:
            last_offset, _ = await asyncio.wait_for(anext(subscription), timeout=60)
        await handle.signal(WorkflowStreamOffsetsWorkflow.publish_noise)

        rest: list[tuple[int, AgentStreamEvent]] = []
        async for offset, event in stream_agent_events(
            client,
            handle,
            'agent_events',
            from_offset=last_offset + 1,
            poll_cooldown=timedelta(milliseconds=50),
            with_offsets=True,
        ):
            rest.append((offset, event))
            if isinstance(event, PartEndEvent):
                await handle.signal(WorkflowStreamOffsetsWorkflow.release)
        output = await handle.result()

    assert output == 'Streamed response'
    assert rest[0][0] > last_offset
    offsets = [last_offset, *(offset for offset, _ in rest)]
    assert offsets == sorted(set(offsets))
    assert offsets[0] == 3
    assert offsets != list(range(len(offsets)))


_workflow_handler_invocations = 0
_workflow_topic_publisher = workflow_stream_event_handler('agent_events')


async def _workflow_side_handler(ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
    global _workflow_handler_invocations
    _workflow_handler_invocations += 1
    await _workflow_topic_publisher(ctx, stream)


_replay_agent = Agent(
    _stream_fn_model,
    name='durability_replay_stream_agent',
    capabilities=[
        ProcessEventStream(_workflow_side_handler),
        TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG, event_stream_topic='agent_events'),
    ],
)


@workflow.defn
class ReplayStreamWorkflow:
    @workflow.init
    def __init__(self, prompt: str) -> None:
        self.stream = WorkflowStream()
        self._released = False

    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await _replay_agent.run(prompt)
        try:
            await workflow.wait_condition(lambda: self._released, timeout=timedelta(seconds=30))
        except asyncio.TimeoutError:
            pass
        self.stream.detach_pollers()
        await workflow.wait_condition(workflow.all_handlers_finished)
        return result.output

    @workflow.signal
    def release(self) -> None:
        self._released = True


async def test_workflow_stream_publishes_only_from_activities(client: Client) -> None:
    """Workflow replay drains the publisher without duplicating activity-published events."""
    global _workflow_handler_invocations
    _workflow_handler_invocations = 0

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ReplayStreamWorkflow],
        plugins=[AgentPlugin(_replay_agent)],
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        handle = await client.start_workflow(
            ReplayStreamWorkflow.run,
            args=['Hello'],
            id=f'{ReplayStreamWorkflow.__name__}-{uuid.uuid4()}',
            task_queue=TASK_QUEUE,
        )
        events = await asyncio.wait_for(
            _collect_events(client, handle, lambda: handle.signal(ReplayStreamWorkflow.release)),
            timeout=60,
        )
        await handle.result()
        history = await handle.fetch_history()

    assert _workflow_handler_invocations > 0
    assert len([event for event in events if isinstance(event, PartStartEvent)]) == 1
    assert len([event for event in events if isinstance(event, PartDeltaEvent)]) == 2

    _workflow_handler_invocations = 0
    await Replayer(
        workflows=[ReplayStreamWorkflow],
        workflow_runner=UnsandboxedWorkflowRunner(),
        data_converter=pydantic_data_converter,
    ).replay_workflow(history)
    assert _workflow_handler_invocations > 0


_filtered_agent = Agent(
    _stream_fn_model,
    name='durability_filtered_stream_agent',
    capabilities=[
        TemporalDurability(
            activity_config=BASE_ACTIVITY_CONFIG,
            event_stream_topic='agent_events',
            event_stream_events=lambda event: not isinstance(event, PartDeltaEvent),
        )
    ],
)


@workflow.defn
class FilteredWorkflowStream:
    @workflow.init
    def __init__(self, prompt: str) -> None:
        self.stream = WorkflowStream()
        self._released = False

    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await _filtered_agent.run(prompt)
        try:
            await workflow.wait_condition(lambda: self._released, timeout=timedelta(seconds=30))
        except asyncio.TimeoutError:
            pass
        self.stream.detach_pollers()
        await workflow.wait_condition(workflow.all_handlers_finished)
        return result.output

    @workflow.signal
    def release(self) -> None:
        self._released = True


async def test_event_stream_topic_filter(client: Client) -> None:
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[FilteredWorkflowStream],
        plugins=[AgentPlugin(_filtered_agent)],
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        handle = await client.start_workflow(
            FilteredWorkflowStream.run,
            args=['Hello'],
            id=f'{FilteredWorkflowStream.__name__}-{uuid.uuid4()}',
            task_queue=TASK_QUEUE,
        )
        events = await asyncio.wait_for(
            _collect_events(client, handle, lambda: handle.signal(FilteredWorkflowStream.release)),
            timeout=60,
        )
        await handle.result()

    assert any(isinstance(event, PartStartEvent) for event in events)
    assert any(isinstance(event, PartEndEvent) for event in events)
    assert not any(isinstance(event, PartDeltaEvent) for event in events)


async def test_event_stream_topic_outside_workflow() -> None:
    events: list[AgentStreamEvent] = []

    async def handler(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(event)

    agent = Agent(
        TestModel(custom_output_text='done'),
        name='outside_topic',
        capabilities=[TemporalDurability(event_stream_topic='agent_events', event_stream_handler=handler)],
    )
    await agent.run('Hello')
    assert any(isinstance(event, PartStartEvent) for event in events)


async def test_event_stream_topic_without_handler_outside_workflow() -> None:
    agent = Agent(
        TestModel(custom_output_text='done'),
        name='outside_topic_no_handler',
        capabilities=[TemporalDurability(event_stream_topic='agent_events')],
    )
    result = await agent.run('Hello')
    assert result.output == 'done'


async def test_workflow_stream_event_handler_is_composable() -> None:
    agent = Agent(
        TestModel(custom_output_text='done'),
        name='factory_handler',
        capabilities=[TemporalDurability(event_stream_handler=workflow_stream_event_handler('agent_events'))],
    )
    result = await agent.run('Hello')
    assert result.output == 'done'


@pytest.mark.parametrize('batch_interval', [timedelta(0), timedelta(milliseconds=-1)])
def test_workflow_stream_event_handler_rejects_non_positive_batch_interval(batch_interval: timedelta) -> None:
    """The guard runs before any provider request, so a VCR recording would add no coverage."""
    with pytest.raises(UserError, match='Workflow Stream batch interval must be greater than zero'):
        workflow_stream_event_handler('agent_events', batch_interval=batch_interval)


async def test_workflow_stream_event_handler_rejects_standalone_activity() -> None:
    handler = workflow_stream_event_handler('agent_events')
    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage(), run_id='standalone-run')
    env = ActivityEnvironment()
    env.info = replace(env.info, workflow_id=None, workflow_run_id=None)

    send, receive = anyio.create_memory_object_stream[AgentStreamEvent](0)
    async with send, receive:
        with pytest.raises(UserError, match='can only publish from an activity scheduled by a workflow'):
            await env.run(handler, ctx, receive)
