from __future__ import annotations

import sys
import uuid
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import replace
from datetime import timedelta
from typing import Any, cast

import anyio
import pytest

from pydantic_ai import (
    Agent,
    AgentStreamEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    RunContext,
    RunUsage,
)
from pydantic_ai.capabilities import Hooks, ProcessEventStream
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResult, AgentRunResultEvent

try:
    from temporalio import workflow
    from temporalio.client import Client, WorkflowFailureError, WorkflowHandle
    from temporalio.contrib.pydantic import pydantic_data_converter
    from temporalio.contrib.workflow_streams import WorkflowStreamItem
    from temporalio.exceptions import ApplicationError
    from temporalio.service import RPCError, RPCStatusCode
    from temporalio.testing import ActivityEnvironment
    from temporalio.worker import Replayer, UnsandboxedWorkflowRunner, Worker

    from pydantic_ai.durable_exec.temporal import (
        AgentEventStream,
        AgentPlugin,
        DurableAgentRunEvents,
        TemporalDurability,
        WorkflowStreamTopic,
        stream_agent_events,
        workflow_stream_event_handler,
    )

    # Direct construction covers the consumer's RPC failure branch without a racy server shutdown.
    from pydantic_ai.durable_exec.temporal._event_stream import (
        _DurableAgentRunResultEvent,  # pyright: ignore[reportPrivateUsage]
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

with workflow.unsafe.imports_passed_through():
    from ._shared import BASE_ACTIVITY_CONFIG, TASK_QUEUE

pytestmark = [pytest.mark.anyio, pytest.mark.xdist_group(name='temporal-durability')]

TOPIC = 'agent_events'


# --- A model that calls a tool before answering ---------------------------------------------------
#
# Model events reach the topic from inside the model-request activity, but tool-call and tool-result
# events are produced in workflow code and published from there. A text-only model exercises only the
# first path, so every workflow below runs a model that takes a tool-calling step first.


def _tool_calling_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:  # pragma: no cover
    raise AssertionError('these tests always stream')


async def _tool_calling_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
    if len(messages) == 1:
        yield {0: DeltaToolCall(name='get_answer', json_args='{}')}
    else:
        yield 'Stream'
        yield 'ed '
        yield 'response'


async def _failing_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
    yield {0: DeltaToolCall(name='fail', json_args='{}')}


_model = FunctionModel(_tool_calling_model, stream_function=_tool_calling_stream)


async def get_answer() -> str:
    return '42'


def _kinds(events: list[Any]) -> list[str]:
    return [type(event).__name__ for event in events]


async def _collect(events: AsyncIterable[Any]) -> list[Any]:
    return [event async for event in events]


# --- Publishing, fan-out to a handler, and the terminal event -------------------------------------

_handler_events: list[AgentStreamEvent] = []


async def _handler(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
    async for event in stream:
        _handler_events.append(event)


_durability = TemporalDurability(
    activity_config=BASE_ACTIVITY_CONFIG,
    event_stream_topic=TOPIC,
    event_stream_handler=_handler,
)
_agent = Agent(_model, name='workflow_stream_agent', tools=[get_answer], capabilities=[_durability])


@workflow.defn
class StreamingWorkflow:
    @workflow.init
    def __init__(self, prompt: str) -> None:
        self.events = AgentEventStream()

    @workflow.run
    async def run(self, prompt: str) -> str:
        async with self.events:
            result = await _agent.run(prompt)
        return result.output


async def test_consumer_receives_the_whole_run(client: Client) -> None:
    """A consumer outside the workflow sees model *and* tool events, then the run's result.

    Tool events take a different route to the topic than model events -- they are produced in
    workflow code and published from there, rather than from inside the model-request activity -- so
    a run without a tool call would leave that half of the feature untested.
    """
    _handler_events.clear()
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[StreamingWorkflow],
        plugins=[AgentPlugin(_agent)],
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        handle = await client.start_workflow(
            StreamingWorkflow.run,
            args=['Hello'],
            id=f'{StreamingWorkflow.__name__}-{uuid.uuid4()}',
            task_queue=TASK_QUEUE,
        )
        events = _durability.stream_agent_events(
            client, handle, output_type=str, poll_cooldown=timedelta(milliseconds=50)
        )
        received = await _collect(events)
        output = await handle.result()

    assert output == 'Streamed response'

    # The run in full: the tool call streamed from the first model request, the workflow-side call
    # and result events, the text streamed from the second, and the terminal event.
    assert _kinds(received) == [
        'PartStartEvent',
        'PartEndEvent',
        'FunctionToolCallEvent',
        'FunctionToolResultEvent',
        'PartStartEvent',
        'FinalResultEvent',
        'PartDeltaEvent',
        'PartDeltaEvent',
        'PartEndEvent',
        'AgentRunResultEvent',
    ]

    # The terminal event carries the real result, decoded into the agent's output type.
    terminal = cast(AgentRunResultEvent[str], received[-1])
    assert isinstance(terminal, AgentRunResultEvent)
    assert terminal.result.output == 'Streamed response'
    assert events.result is not None and events.result.output == 'Streamed response'
    assert len(terminal.result.all_messages()) == 4

    # A topic is orthogonal to an `event_stream_handler`: the handler still sees every event.
    assert any(isinstance(event, PartDeltaEvent) for event in _handler_events)
    assert any(isinstance(event, FunctionToolCallEvent) for event in _handler_events)
    assert any(isinstance(event, FunctionToolResultEvent) for event in _handler_events)


async def test_a_late_consumer_still_gets_the_whole_run(client: Client) -> None:
    """The workflow holds itself open until a subscriber has drained the stream.

    A Workflow Stream can only be read while its workflow is running, so a consumer that connects
    after the run has finished would otherwise find nothing at all. Waiting until the terminal event
    is in the log means the run is over and the workflow is parked on the drain.
    """
    from temporalio.contrib.workflow_streams import WorkflowStreamClient

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[StreamingWorkflow],
        plugins=[AgentPlugin(_agent)],
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        handle = await client.start_workflow(
            StreamingWorkflow.run,
            args=['Hello'],
            id=f'{StreamingWorkflow.__name__}-{uuid.uuid4()}',
            task_queue=TASK_QUEUE,
        )
        stream_client = WorkflowStreamClient(handle, client=client)
        while await stream_client.get_offset() < 10:
            await anyio.sleep(0.05)

        received = await _collect(
            _durability.stream_agent_events(client, handle, poll_cooldown=timedelta(milliseconds=50))
        )
        output = await handle.result()

    assert output == 'Streamed response'
    assert len(received) == 10
    assert isinstance(received[-1], AgentRunResultEvent)


_impatient_durability = TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG, event_stream_topic=TOPIC)
_impatient_agent = Agent(
    TestModel(custom_output_text='done'), name='impatient_stream_agent', capabilities=[_impatient_durability]
)


@workflow.defn
class ImpatientWorkflow:
    @workflow.init
    def __init__(self, prompt: str) -> None:
        self.events = AgentEventStream(drain_timeout=timedelta(milliseconds=1))

    @workflow.run
    async def run(self, prompt: str) -> str:
        async with self.events:
            result = await _impatient_agent.run(prompt)
        return result.output


async def test_a_run_nobody_is_watching_finishes_anyway(client: Client) -> None:
    """`drain_timeout` bounds the wait, so an unwatched run can't hang on a subscriber that never comes."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ImpatientWorkflow],
        plugins=[AgentPlugin(_impatient_agent)],
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        handle = await client.start_workflow(
            ImpatientWorkflow.run,
            args=['Hello'],
            id=f'{ImpatientWorkflow.__name__}-{uuid.uuid4()}',
            task_queue=TASK_QUEUE,
        )
        assert await handle.result() == 'done'


async def fail() -> str:
    raise RuntimeError('tool exploded')


_failing_durability = TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG, event_stream_topic=TOPIC)
_failing_agent = Agent(
    FunctionModel(_tool_calling_model, stream_function=_failing_stream),
    name='failing_stream_agent',
    tools=[fail],
    capabilities=[_failing_durability],
)


@workflow.defn
class FailingWorkflow:
    @workflow.init
    def __init__(self, prompt: str) -> None:
        self.events = AgentEventStream()

    @workflow.run
    async def run(self, prompt: str) -> None:
        # The run always raises, so this never returns a value; leaving the `async with` on the
        # exception path is the behaviour under test.
        async with self.events:
            await _failing_agent.run(prompt)


async def test_a_failed_run_ends_the_stream_without_a_result(client: Client) -> None:
    """A run that fails publishes no terminal event, so the subscription just ends.

    `result` staying `None` is how a consumer tells "the run finished" from "the workflow ended some
    other way"; the workflow handle carries the actual failure.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[FailingWorkflow],
        plugins=[AgentPlugin(_failing_agent)],
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        handle = await client.start_workflow(
            FailingWorkflow.run,
            args=['Hello'],
            id=f'{FailingWorkflow.__name__}-{uuid.uuid4()}',
            task_queue=TASK_QUEUE,
        )
        events = _failing_durability.stream_agent_events(client, handle, poll_cooldown=timedelta(milliseconds=50))
        received = await _collect(events)
        with pytest.raises(WorkflowFailureError):
            await handle.result()

    assert events.result is None
    assert not any(isinstance(event, AgentRunResultEvent) for event in received)


_final_result_hooks = Hooks[bool]()


@_final_result_hooks.on.after_run
async def _replace_final_result(ctx: RunContext[bool], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
    if ctx.deps:
        raise ApplicationError('after_run exploded', non_retryable=True)
    return replace(result, output='finalized')


_final_result_durability = TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG, event_stream_topic=TOPIC)
_final_result_agent = Agent(
    TestModel(custom_output_text='original'),
    name='final_result_stream_agent',
    deps_type=bool,
    capabilities=[_final_result_hooks, _final_result_durability],
)


@workflow.defn
class FinalResultWorkflow:
    @workflow.init
    def __init__(self, prompt: str) -> None:
        self.events = AgentEventStream()

    @workflow.run
    async def run(self, prompt: str) -> str:
        async with self.events:
            result = await _final_result_agent.run(prompt, deps=prompt == 'fail')
        return result.output


async def test_terminal_event_carries_the_finalized_result(client: Client) -> None:
    """The terminal event is published after every capability has transformed the result."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[FinalResultWorkflow],
        plugins=[AgentPlugin(_final_result_agent)],
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        handle = await client.start_workflow(
            FinalResultWorkflow.run,
            args=['Hello'],
            id=f'{FinalResultWorkflow.__name__}-{uuid.uuid4()}',
            task_queue=TASK_QUEUE,
        )
        received = await _collect(
            _final_result_durability.stream_agent_events(
                client, handle, output_type=str, poll_cooldown=timedelta(milliseconds=50)
            )
        )
        output = await handle.result()

    terminal = cast(AgentRunResultEvent[str], received[-1])
    assert output == 'finalized'
    assert terminal.result.output == output


async def test_after_run_failure_publishes_no_terminal_event(client: Client) -> None:
    """A later `after_run` failure must not publish a false successful result."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[FinalResultWorkflow],
        plugins=[AgentPlugin(_final_result_agent)],
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        handle = await client.start_workflow(
            FinalResultWorkflow.run,
            args=['fail'],
            id=f'{FinalResultWorkflow.__name__}-{uuid.uuid4()}',
            task_queue=TASK_QUEUE,
        )
        events = _final_result_durability.stream_agent_events(client, handle, poll_cooldown=timedelta(milliseconds=50))
        received = await _collect(events)
        with pytest.raises(WorkflowFailureError):
            await handle.result()

    assert events.result is None
    assert not any(isinstance(event, AgentRunResultEvent) for event in received)


_topic_only_durability = TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG, event_stream_topic=TOPIC)
_topic_only_agent = Agent(
    _model, name='topic_only_stream_agent', tools=[get_answer], capabilities=[_topic_only_durability]
)


@workflow.defn
class TopicOnlyWorkflow:
    @workflow.init
    def __init__(self, prompt: str) -> None:
        self.events = AgentEventStream()

    @workflow.run
    async def run(self, prompt: str) -> str:
        async with self.events:
            result = await _topic_only_agent.run(prompt)
        return result.output


async def test_workflow_side_events_cost_no_activity(client: Client) -> None:
    """A topic on its own adds no durable unit: workflow-side events are published from workflow code.

    Only the two model requests and the tool call are scheduled. An `event_stream_handler` is what
    puts workflow-side events through an event-handler activity; publishing to a topic appends to the
    workflow's own log instead of going out to an activity and coming back as a signal.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[TopicOnlyWorkflow],
        plugins=[AgentPlugin(_topic_only_agent)],
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        handle = await client.start_workflow(
            TopicOnlyWorkflow.run,
            args=['Hello'],
            id=f'{TopicOnlyWorkflow.__name__}-{uuid.uuid4()}',
            task_queue=TASK_QUEUE,
        )
        await _collect(
            _topic_only_durability.stream_agent_events(client, handle, poll_cooldown=timedelta(milliseconds=50))
        )
        await handle.result()
        history = await handle.fetch_history()

    scheduled = [
        event.activity_task_scheduled_event_attributes.activity_type.name
        for event in history.events
        if event.HasField('activity_task_scheduled_event_attributes')
    ]
    assert sorted(scheduled) == [
        'agent__topic_only_stream_agent__model_request_stream',
        'agent__topic_only_stream_agent__model_request_stream',
        'agent__topic_only_stream_agent__toolset__<agent>__call_tool',
    ]


# --- Filtering ------------------------------------------------------------------------------------

_filtered_durability = TemporalDurability(
    activity_config=BASE_ACTIVITY_CONFIG,
    # One event from each publish path: `PartDeltaEvent` goes out from the model-request activity,
    # `FunctionToolCallEvent` from workflow code. A filter has to reach both.
    event_stream_topic=WorkflowStreamTopic(
        TOPIC, events=lambda event: not isinstance(event, (PartDeltaEvent, FunctionToolCallEvent))
    ),
)
_filtered_agent = Agent(_model, name='filtered_stream_agent', tools=[get_answer], capabilities=[_filtered_durability])


@workflow.defn
class FilteredWorkflow:
    @workflow.init
    def __init__(self, prompt: str) -> None:
        self.events = AgentEventStream()

    @workflow.run
    async def run(self, prompt: str) -> str:
        async with self.events:
            result = await _filtered_agent.run(prompt)
        return result.output


async def test_topic_filter_keeps_the_terminal_event(client: Client) -> None:
    """`events=` drops what it rejects, but never the terminal event, which ends the subscription."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[FilteredWorkflow],
        plugins=[AgentPlugin(_filtered_agent)],
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        handle = await client.start_workflow(
            FilteredWorkflow.run,
            args=['Hello'],
            id=f'{FilteredWorkflow.__name__}-{uuid.uuid4()}',
            task_queue=TASK_QUEUE,
        )
        received = await _collect(
            _filtered_durability.stream_agent_events(client, handle, poll_cooldown=timedelta(milliseconds=50))
        )
        await handle.result()

    assert not any(isinstance(event, PartDeltaEvent) for event in received)  # activity-side
    assert not any(isinstance(event, FunctionToolCallEvent) for event in received)  # workflow-side
    assert any(isinstance(event, PartStartEvent) for event in received)
    assert any(isinstance(event, FunctionToolResultEvent) for event in received)
    assert isinstance(received[-1], AgentRunResultEvent)


# --- More than one run on one stream --------------------------------------------------------------


@workflow.defn
class TwoRunWorkflow:
    @workflow.init
    def __init__(self, prompt: str) -> None:
        self.events = AgentEventStream()
        self.finished = False

    @workflow.run
    async def run(self, prompt: str) -> str:
        async with self.events:
            await _topic_only_agent.run(prompt)
            second = await _topic_only_agent.run(prompt)
        self.finished = True
        return second.output

    @workflow.query
    def runs_finished(self) -> bool:
        return self.finished


async def test_each_run_gets_its_own_terminal_event(client: Client) -> None:
    """One iterator covers one run; the next run is picked up by reconnecting at the next offset.

    Each terminal event has its own idempotent acknowledgment, so replaying the first event cannot
    release the second run's drain barrier.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[TwoRunWorkflow],
        plugins=[AgentPlugin(_topic_only_agent)],
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        handle = await client.start_workflow(
            TwoRunWorkflow.run,
            args=['Hello'],
            id=f'{TwoRunWorkflow.__name__}-{uuid.uuid4()}',
            task_queue=TASK_QUEUE,
        )
        first = _topic_only_durability.stream_agent_events(client, handle, poll_cooldown=timedelta(milliseconds=50))
        first_events = await _collect(first)
        duplicate = _topic_only_durability.stream_agent_events(
            client, handle, from_offset=first.offset, poll_cooldown=timedelta(milliseconds=50)
        )
        duplicate_events = await _collect(duplicate)
        assert await handle.query(TwoRunWorkflow.runs_finished) is False
        second = _topic_only_durability.stream_agent_events(
            client, handle, from_offset=first.offset + 1, poll_cooldown=timedelta(milliseconds=50)
        )
        second_events = await _collect(second)
        assert await handle.result() == 'Streamed response'

    # Each subscription ends at its own run's terminal event rather than running on into the next.
    assert sum(1 for event in first_events if isinstance(event, AgentRunResultEvent)) == 1
    first_terminal = cast(AgentRunResultEvent[str], first_events[-1])
    assert isinstance(first_terminal, AgentRunResultEvent)
    assert len(duplicate_events) == 1
    duplicate_terminal = cast(AgentRunResultEvent[str], duplicate_events[0])
    assert isinstance(duplicate_terminal, AgentRunResultEvent)
    assert duplicate_terminal.result.run_id == first_terminal.result.run_id
    assert sum(1 for event in second_events if isinstance(event, AgentRunResultEvent)) == 1
    assert isinstance(second_events[-1], AgentRunResultEvent)
    assert _kinds(first_events) == _kinds(second_events)


async def test_acknowledgment_failure_does_not_hide_the_terminal_event() -> None:
    """A failed drain signal only delays workflow completion; the received result stays usable."""

    async def subscription() -> AsyncIterator[WorkflowStreamItem[AgentStreamEvent | _DurableAgentRunResultEvent[str]]]:
        result = AgentRunResult(output='done')
        yield WorkflowStreamItem(
            topic=TOPIC,
            data=_DurableAgentRunResultEvent(result, drain_token='run:0'),
            offset=7,
        )

    class FailingHandle:
        async def signal(self, signal: str, arg: str) -> None:
            raise RPCError('workflow completed', RPCStatusCode.NOT_FOUND, b'')

    stream = DurableAgentRunEvents(
        cast(
            'AsyncIterator[WorkflowStreamItem[AgentStreamEvent | AgentRunResultEvent[Any]]]',
            subscription(),
        ),
        cast('WorkflowHandle[Any, Any]', FailingHandle()),
    )
    with pytest.warns(RuntimeWarning, match='Failed to acknowledge the terminal agent event'):
        terminal = await anext(stream)

    assert isinstance(terminal, AgentRunResultEvent)
    assert terminal.result.output == 'done'
    assert stream.result is terminal.result
    assert stream.offset == 7


# --- Resuming from an offset ----------------------------------------------------------------------


@workflow.defn
class NoisyWorkflow:
    """Publishes to a second topic, so the agent's offsets are not simply 0, 1, 2, ..."""

    @workflow.init
    def __init__(self, prompt: str) -> None:
        self.events = AgentEventStream()
        other = self.events.stream.topic('other_events')
        for i in range(3):
            other.publish(f'noise-{i}')

    @workflow.run
    async def run(self, prompt: str) -> str:
        async with self.events:
            result = await _topic_only_agent.run(prompt)
        return result.output


async def test_a_consumer_can_resume_at_the_next_offset(client: Client) -> None:
    """Reconnecting with `from_offset=offset + 1` continues without gaps or duplicates."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[NoisyWorkflow],
        plugins=[AgentPlugin(_topic_only_agent)],
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        handle = await client.start_workflow(
            NoisyWorkflow.run,
            args=['Hello'],
            id=f'{NoisyWorkflow.__name__}-{uuid.uuid4()}',
            task_queue=TASK_QUEUE,
        )
        # `async with` is how a consumer that stops early releases its long-poll immediately.
        async with _topic_only_durability.stream_agent_events(
            client, handle, poll_cooldown=timedelta(milliseconds=50)
        ) as first:
            await anext(first)

        rest = _topic_only_durability.stream_agent_events(
            client, handle, from_offset=first.offset + 1, poll_cooldown=timedelta(milliseconds=50)
        )
        received = await _collect(rest)
        await handle.result()

    # Offsets run over the whole stream, so the three items on the other topic are skipped rather
    # than renumbered: the agent's first event is at offset 3, and the sequence has gaps.
    assert first.offset == 3
    assert rest.offset > first.offset
    assert isinstance(received[-1], AgentRunResultEvent)
    assert 'PartStartEvent' not in _kinds(received[:1])  # the first event was consumed above


# --- Replay safety --------------------------------------------------------------------------------

_replay_handler_runs = 0


async def _workflow_side_handler(ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
    """A workflow-side handler installed with the same publisher the capability uses.

    `ProcessEventStream` runs in workflow code, which re-runs on replay. The publisher must not
    publish there -- and the whole run must not be re-published when a history is replayed.
    """
    global _replay_handler_runs
    _replay_handler_runs += 1
    await workflow_stream_event_handler(TOPIC)(ctx, stream)


_replay_durability = TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG, event_stream_topic=TOPIC)
_replay_agent = Agent(
    _model,
    name='replay_stream_agent',
    tools=[get_answer],
    capabilities=[ProcessEventStream(_workflow_side_handler), _replay_durability],
)


@workflow.defn
class ReplayWorkflow:
    @workflow.init
    def __init__(self, prompt: str) -> None:
        self.events = AgentEventStream()

    @workflow.run
    async def run(self, prompt: str) -> str:
        async with self.events:
            result = await _replay_agent.run(prompt)
        return result.output


async def test_replay_does_not_duplicate_events(client: Client) -> None:
    global _replay_handler_runs
    _replay_handler_runs = 0

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ReplayWorkflow],
        plugins=[AgentPlugin(_replay_agent)],
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        handle = await client.start_workflow(
            ReplayWorkflow.run,
            args=['Hello'],
            id=f'{ReplayWorkflow.__name__}-{uuid.uuid4()}',
            task_queue=TASK_QUEUE,
        )
        received = await _collect(
            _replay_durability.stream_agent_events(client, handle, poll_cooldown=timedelta(milliseconds=50))
        )
        await handle.result()
        history = await handle.fetch_history()

    assert _replay_handler_runs > 0
    assert len([event for event in received if isinstance(event, PartDeltaEvent)]) == 2
    assert sum(1 for event in received if isinstance(event, AgentRunResultEvent)) == 1

    _replay_handler_runs = 0
    await Replayer(
        workflows=[ReplayWorkflow],
        workflow_runner=UnsandboxedWorkflowRunner(),
        data_converter=pydantic_data_converter,
    ).replay_workflow(history)
    # The workflow-side handler ran again, and publishing from it was a no-op rather than a failure
    # or a second copy of the run on the topic.
    assert _replay_handler_runs > 0


# --- Guards and validation ------------------------------------------------------------------------


@workflow.defn
class StreamlessWorkflow:
    """A workflow that forgot to host an `AgentEventStream`."""

    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await _streamless_agent.run(prompt)).output


_streamless_durability = TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG, event_stream_topic=TOPIC)
_streamless_agent = Agent(
    TestModel(custom_output_text='done'), name='streamless', capabilities=[_streamless_durability]
)


async def test_a_workflow_without_an_event_stream_fails_clearly(client: Client) -> None:

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[StreamlessWorkflow],
        plugins=[AgentPlugin(_streamless_agent)],
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        handle = await client.start_workflow(
            StreamlessWorkflow.run,
            args=['Hello'],
            id=f'{StreamlessWorkflow.__name__}-{uuid.uuid4()}',
            task_queue=TASK_QUEUE,
        )
        with pytest.raises(WorkflowFailureError) as exc_info:
            await handle.result()

    assert 'needs its workflow to host an `AgentEventStream`' in str(exc_info.value.__cause__)


async def test_the_topic_is_transparent_outside_a_workflow() -> None:
    """An agent configured for Temporal streaming still runs normally outside one."""
    events: list[AgentStreamEvent] = []

    async def handler(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(event)

    agent = Agent(
        TestModel(custom_output_text='done'),
        name='outside_topic',
        capabilities=[TemporalDurability(event_stream_topic=TOPIC, event_stream_handler=handler)],
    )
    assert (await agent.run('Hello')).output == 'done'
    assert any(isinstance(event, PartStartEvent) for event in events)

    # Without a handler the topic is inert outside a workflow, rather than forcing a streamed run.
    plain = Agent(
        TestModel(custom_output_text='done'),
        name='outside_topic_no_handler',
        capabilities=[TemporalDurability(event_stream_topic=TOPIC)],
    )
    assert (await plain.run('Hello')).output == 'done'


async def test_the_publisher_rejects_a_standalone_activity() -> None:
    """An activity started directly on the client has no workflow stream to publish to."""
    handler = workflow_stream_event_handler(TOPIC)
    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage(), run_id='standalone-run')
    env = ActivityEnvironment()
    env.info = replace(env.info, workflow_id=None, workflow_run_id=None)

    send, receive = anyio.create_memory_object_stream[AgentStreamEvent](0)
    async with send, receive:
        with pytest.raises(UserError, match='can only publish from an activity scheduled by a workflow'):
            await env.run(handler, ctx, receive)


@pytest.mark.parametrize('batch_interval', [timedelta(0), timedelta(milliseconds=-1)])
def test_a_topic_rejects_a_non_positive_batch_interval(batch_interval: timedelta) -> None:
    with pytest.raises(UserError, match='batch interval must be greater than zero'):
        WorkflowStreamTopic(TOPIC, batch_interval=batch_interval)


def test_a_topic_needs_a_name() -> None:
    with pytest.raises(UserError, match='needs a name'):
        WorkflowStreamTopic('')


async def test_streaming_without_a_topic_is_rejected(client: Client) -> None:
    durability = TemporalDurability[None]()
    with pytest.raises(UserError, match='has no `event_stream_topic`'):
        durability.stream_agent_events(client, client.get_workflow_handle('some-workflow'))


@pytest.mark.parametrize('poll_cooldown', [timedelta(0), timedelta(milliseconds=-1)])
async def test_streaming_rejects_a_non_positive_poll_cooldown(client: Client, poll_cooldown: timedelta) -> None:
    with pytest.raises(UserError, match='poll cooldown must be greater than zero'):
        stream_agent_events(
            client,
            client.get_workflow_handle('some-workflow'),
            TOPIC,
            poll_cooldown=poll_cooldown,
        )


async def test_the_publisher_passes_a_wrapped_handler_the_stream_outside_an_activity() -> None:
    """Composed explicitly and run outside an activity, the wrapped handler still sees every event."""
    seen: list[AgentStreamEvent] = []

    async def inner(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            seen.append(event)

    agent = Agent(
        TestModel(custom_output_text='done'),
        name='composed_handler',
        capabilities=[TemporalDurability(event_stream_handler=workflow_stream_event_handler(TOPIC, handler=inner))],
    )
    assert (await agent.run('Hello')).output == 'done'
    assert any(isinstance(event, PartStartEvent) for event in seen)


# --- Driving a UI protocol over the workflow boundary ---------------------------------------------


async def test_the_events_drive_a_ui_adapter(client: Client) -> None:
    """The point of the terminal event: the stream is what a `UIAdapter` already consumes.

    An HTTP handler can start the workflow and serve the protocol stream straight from the topic,
    with `on_complete` receiving the run result exactly as it would for an in-process run.
    """
    from pydantic_ai.ui.vercel_ai import VercelAIAdapter
    from pydantic_ai.ui.vercel_ai.request_types import SubmitMessage

    completed: list[str] = []

    async def on_complete(result: Any) -> None:
        completed.append(result.output)

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[StreamingWorkflow],
        plugins=[AgentPlugin(_agent)],
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        handle = await client.start_workflow(
            StreamingWorkflow.run,
            args=['Hello'],
            id=f'{StreamingWorkflow.__name__}-{uuid.uuid4()}',
            task_queue=TASK_QUEUE,
        )
        adapter = VercelAIAdapter(agent=_agent, run_input=SubmitMessage(id='chat-1', messages=[]))
        chunks = await _collect(
            adapter.transform_stream(
                _durability.stream_agent_events(client, handle, poll_cooldown=timedelta(milliseconds=50)),
                on_complete=on_complete,
            )
        )
        await handle.result()

    assert 'TextDeltaChunk' in _kinds(chunks)
    assert 'ToolInputAvailableChunk' in _kinds(chunks)
    assert 'ToolOutputAvailableChunk' in _kinds(chunks)
    assert completed == ['Streamed response']
