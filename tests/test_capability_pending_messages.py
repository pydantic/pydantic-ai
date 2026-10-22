"""Tests for the capability pending-message queue.

Split out of `test_capabilities.py` per #7304.
"""

from __future__ import annotations

import copy
import pickle
import threading
from collections.abc import AsyncIterable, AsyncIterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

import anyio
import pytest
from pydantic import BaseModel, TypeAdapter

from pydantic_ai import _agent_graph
from pydantic_ai._enqueue import PendingMessage, PendingMessagePriority, PendingMessageQueue
from pydantic_ai._run_context import RunContext
from pydantic_ai.agent import Agent
from pydantic_ai.capabilities import (
    ProcessHistory,
    ToolSearch,
    WrapModelRequestHandler,
)
from pydantic_ai.capabilities.abstract import (
    AbstractCapability,
    AgentNode,
    CapabilityOrdering,
    NodeResult,
)
from pydantic_ai.exceptions import (
    ModelRetry,
    UserError,
)
from pydantic_ai.messages import (
    AgentStreamEvent,
    EnqueuedMessagesEvent,
    ImageUrl,
    InstructionPart,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponseStreamEvent,
    PartStartEvent,
    RetryFeedbackPart,
    RetryInfo,
    SystemPromptPart,
    TextPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    _clean_message_history,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai.models import (
    ModelRequestContext,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.result import AgentStream
from pydantic_ai.run import AgentRunResult, AgentRunResultEvent
from pydantic_ai.usage import RequestUsage, RunUsage
from pydantic_graph import End

from ._inline_snapshot import snapshot
from .conftest import IsDatetime, IsStr, iter_message_parts, try_import

with try_import() as anthropic_imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

_SEARCH_TOOLS_NAME = ToolSearch.function_tool_name

pytestmark = []


# ===== Pending Message Queue Tests =====


def test_pending_message_queue_state_and_copy_round_trip():
    pending = PendingMessage(messages=[ModelRequest(parts=[UserPromptPart(content='queued')])])
    state = _agent_graph.GraphAgentState(pending_messages=[pending])
    adapter = TypeAdapter(_agent_graph.GraphAgentState)

    queues = [
        state.pending_messages,
        adapter.validate_python(adapter.dump_python(state)).pending_messages,
        copy.deepcopy(state.pending_messages),
        pickle.loads(pickle.dumps(state.pending_messages)),
    ]

    for queue in queues:
        assert isinstance(queue, PendingMessageQueue)
        assert queue == [pending]


@pytest.mark.parametrize('at_end', [False, True], ids=['between-nodes', 'run-ending'])
def test_sync_enqueue_does_not_race_pending_message_drain(at_end: bool):
    """A worker enqueue either survives a mid-run drain or is rejected at run end."""
    drain_started = threading.Event()
    enqueue_started = threading.Event()
    release_drain = threading.Event()

    class BlockingQueue(PendingMessageQueue):
        def append(self, pending: PendingMessage) -> None:
            enqueue_started.set()
            super().append(pending)

        def _pop_priority(self, priority: PendingMessagePriority) -> list[PendingMessage]:
            if priority == 'asap' and not drain_started.is_set():
                drain_started.set()
                assert release_drain.wait(timeout=5)
            return super()._pop_priority(priority)

    queue = BlockingQueue()
    ctx = RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        pending_messages=queue,
    )

    drain = queue.drain_at_end if at_end else lambda: queue.pop_priority('asap')
    with ThreadPoolExecutor(max_workers=2) as executor:
        drain_future = executor.submit(drain)
        assert drain_started.wait(timeout=5)
        enqueue_future = executor.submit(ctx.enqueue, 'from sync tool')
        assert enqueue_started.wait(timeout=5)
        assert not enqueue_future.done()
        release_drain.set()
        drain_future.result(timeout=5)
        if at_end:
            with pytest.raises(UserError, match='run has ended'):
                enqueue_future.result(timeout=5)
        else:
            enqueue_future.result(timeout=5)

    assert len(queue) == (0 if at_end else 1)


@pytest.mark.parametrize('finish_run', [False, True], ids=['early-exit', 'finished'])
async def test_agent_run_enqueue_after_run_ends_raises(finish_run: bool):
    agent = Agent(TestModel())

    async with agent.iter('hello') as agent_run:
        if finish_run:
            async for _ in agent_run:
                pass

    with pytest.raises(UserError, match='run has ended'):
        agent_run.enqueue('too late')


@pytest.mark.parametrize('hook_redirects', [True, False], ids=['hook-redirects', 'drain-redirects'])
async def test_outermost_hook_can_enqueue_on_end(hook_redirects: bool):
    """The final drain runs after every hook, so an outermost hook can still enqueue on `End`."""

    class RedirectCapability(AbstractCapability[object]):
        enqueued = False

        def get_ordering(self) -> CapabilityOrdering:
            return CapabilityOrdering(position='outermost', wraps=[AbstractCapability])

        async def after_node_run(
            self, ctx: RunContext[object], *, node: AgentNode[object], result: NodeResult[object]
        ) -> NodeResult[object]:
            if isinstance(result, End) and not self.enqueued:
                self.enqueued = True
                ctx.enqueue('after redirect')
                if hook_redirects:
                    return _agent_graph.ModelRequestNode(
                        request=ModelRequest(parts=[UserPromptPart(content='redirect')])
                    )
            return result

    agent = Agent(TestModel(), capabilities=[RedirectCapability()])
    result = await agent.run('hello')

    assert any(
        isinstance(part, UserPromptPart) and part.content == 'after redirect'
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
    )


async def test_enqueue_asap_message_from_tool():
    """`'asap'` messages enqueued from a tool land right after its `ToolReturnPart`, in the same request."""

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            return ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        return ModelResponse(
            parts=[ToolCallPart(tool_name='inject_msg', args='{}')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    agent = Agent(FunctionModel(model_fn))

    @agent.tool
    def inject_msg(ctx: RunContext[object]) -> str:
        ctx.enqueue('Injected asap message')
        return 'ok'

    result = await agent.run('Hello')
    assert result.output == 'done'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='inject_msg', args='{}', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='inject_msg',
                        content='ok',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(content='Injected asap message', timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_enqueue_asap_delivery_event_from_tool():
    """An `EnqueuedMessagesEvent` is emitted when an `'asap'` message is delivered, before the next model response."""
    events: list[AgentStreamEvent] = []
    enqueue_id: str | None = None

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            yield 'done'
            return
        yield {0: DeltaToolCall(name='inject_msg', json_args='{}')}

    agent = Agent(FunctionModel(stream_function=stream_fn))

    @agent.tool
    def inject_msg(ctx: RunContext[Any]) -> str:
        nonlocal enqueue_id
        enqueue_id = ctx.enqueue('Injected asap message')
        return 'ok'

    async def event_stream_handler(_: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(event)

    result = await agent.run('Hello', event_stream_handler=event_stream_handler)

    assert enqueue_id is not None
    delivery_events = [event for event in events if isinstance(event, EnqueuedMessagesEvent)]
    tool_request = result.all_messages()[2]
    assert isinstance(tool_request, ModelRequest)
    assert delivery_events == [
        EnqueuedMessagesEvent(
            enqueue_id=enqueue_id,
            messages=(
                ModelRequest(
                    parts=[tool_request.parts[1]],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ),
            parts=(tool_request.parts[1],),
        )
    ]
    # `parts` are the very objects in history; `messages` holds them in a request of their own.
    assert delivery_events[0].parts[0] is tool_request.parts[1]
    assert delivery_events[0].messages[0] not in result.all_messages()
    # The delivery event precedes the model response that can depend on the delivered message.
    delivery_index = events.index(delivery_events[0])
    done_index = next(
        i
        for i, event in enumerate(events)
        if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart) and event.part.content == 'done'
    )
    assert delivery_index < done_index


async def test_enqueue_when_idle_delivery_event_during_iter_streaming():
    """A `'when_idle'` delivery surfaces as an `EnqueuedMessagesEvent` during `agent.iter` streaming."""
    call_count = 0

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        nonlocal call_count
        call_count += 1
        yield f'response {call_count}'

    agent = Agent(FunctionModel(stream_function=stream_fn))
    events: list[AgentStreamEvent] = []

    async with agent.iter('Hello') as agent_run:
        enqueue_id = agent_run.enqueue('External follow-up', priority='when_idle')
        # Drive with `next()` (not bare `async for`) so `when_idle` messages drain, while
        # streaming each model-request node to observe its events.
        node = agent_run.next_node
        while not isinstance(node, End):
            if Agent.is_model_request_node(node):
                async with node.stream(agent_run.ctx) as stream:
                    async for event in stream:
                        events.append(event)
            node = await agent_run.next(node)

    assert enqueue_id is not None
    assert agent_run.result is not None
    delivery_events = [event for event in events if isinstance(event, EnqueuedMessagesEvent)]
    assert delivery_events == [
        EnqueuedMessagesEvent(enqueue_id=enqueue_id, messages=(agent_run.result.all_messages()[2],))
    ]
    delivery_index = events.index(delivery_events[0])
    response_2_index = next(
        i
        for i, event in enumerate(events)
        if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart) and event.part.content == 'response 2'
    )
    assert delivery_index < response_2_index


async def test_multiple_enqueue_delivery_events_keep_order():
    """Multiple `enqueue` calls each emit one `EnqueuedMessagesEvent`, in enqueue order, via `run_stream_events`."""
    enqueue_ids: list[str] = []

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            yield 'done'
            return
        yield {0: DeltaToolCall(name='inject_msgs', json_args='{}')}

    agent = Agent(FunctionModel(stream_function=stream_fn))

    @agent.tool
    def inject_msgs(ctx: RunContext[Any]) -> str:
        first = ctx.enqueue('First injected message')
        second = ctx.enqueue('Second injected message')
        assert first is not None and second is not None
        enqueue_ids.extend([first, second])
        return 'ok'

    delivery_events: list[EnqueuedMessagesEvent] = []
    result: AgentRunResult[Any] | None = None
    async with agent.run_stream_events('Hello') as stream:
        async for event in stream:
            if isinstance(event, EnqueuedMessagesEvent):
                delivery_events.append(event)
            elif isinstance(event, AgentRunResultEvent):
                result = event.result

    assert result is not None
    tool_request = result.all_messages()[2]
    assert isinstance(tool_request, ModelRequest)
    assert delivery_events == [
        EnqueuedMessagesEvent(
            enqueue_id=enqueue_ids[0],
            messages=(
                ModelRequest(
                    parts=[tool_request.parts[1]],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ),
            parts=(tool_request.parts[1],),
        ),
        EnqueuedMessagesEvent(
            enqueue_id=enqueue_ids[1],
            messages=(
                ModelRequest(
                    parts=[tool_request.parts[2]],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ),
            parts=(tool_request.parts[2],),
        ),
    ]


async def test_enqueue_delivery_event_survives_history_processor_rebuild():
    """The delivery event still matches final history when a history processor rebuilds message objects."""
    events: list[AgentStreamEvent] = []
    enqueue_id: str | None = None

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            yield 'done'
            return
        yield {0: DeltaToolCall(name='inject_msg', json_args='{}')}

    def rebuild_messages(messages: list[ModelMessage]) -> list[ModelMessage]:
        # Round-trip through JSON so every message is a fresh, equal-but-not-identical object.
        return ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(messages))

    agent = Agent(FunctionModel(stream_function=stream_fn), capabilities=[ProcessHistory(rebuild_messages)])

    @agent.tool
    def inject_msg(ctx: RunContext[Any]) -> str:
        nonlocal enqueue_id
        enqueue_id = ctx.enqueue('Injected asap message')
        return 'ok'

    async def event_stream_handler(_: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(event)

    result = await agent.run('Hello', event_stream_handler=event_stream_handler)

    assert enqueue_id is not None
    delivery_events = [event for event in events if isinstance(event, EnqueuedMessagesEvent)]
    tool_request = result.all_messages()[2]
    assert isinstance(tool_request, ModelRequest)
    assert delivery_events == [
        EnqueuedMessagesEvent(
            enqueue_id=enqueue_id,
            messages=(
                ModelRequest(
                    parts=[tool_request.parts[1]],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ),
            parts=(tool_request.parts[1],),
        )
    ]


async def test_empty_enqueue_emits_no_delivery_event():
    """An empty `enqueue()` call delivers nothing and emits no `EnqueuedMessagesEvent`."""

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            yield 'done'
            return
        yield {0: DeltaToolCall(name='noop_enqueue', json_args='{}')}

    agent = Agent(FunctionModel(stream_function=stream_fn))

    @agent.tool
    def noop_enqueue(ctx: RunContext[Any]) -> str:
        assert ctx.enqueue() is None
        return 'ok'

    events: list[AgentStreamEvent] = []

    async def event_stream_handler(_: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(event)

    await agent.run('Hello', event_stream_handler=event_stream_handler)

    assert [event for event in events if isinstance(event, EnqueuedMessagesEvent)] == []


def test_enqueued_messages_event_serialization_roundtrip():
    """`EnqueuedMessagesEvent` round-trips through the `AgentStreamEvent` union as JSON.

    Durable execution (e.g. Temporal's per-event `event_stream_handler` wrapping) serializes
    events to JSON across the activity boundary, so JSON mode is the actual constraint.
    """
    event = EnqueuedMessagesEvent(
        enqueue_id='enq-1',
        messages=(ModelRequest(parts=[UserPromptPart(content='hi')]),),
    )
    adapter = TypeAdapter[AgentStreamEvent](AgentStreamEvent)
    dumped = adapter.dump_python(event)
    assert dumped['event_kind'] == 'enqueued_messages'
    assert adapter.validate_python(dumped) == event
    assert adapter.validate_json(adapter.dump_json(event)) == event


def test_pending_message_positional_construction_keeps_priority_second():
    """`PendingMessage(messages, priority)` positional construction still sets `priority`.

    Guards the field order: `enqueue_id` (which has a generated default) must stay after
    `priority`, or positional callers would silently assign their priority to `enqueue_id`.
    """
    pending = PendingMessage([ModelRequest(parts=[UserPromptPart(content='hi')])], 'when_idle')
    assert pending.priority == 'when_idle'
    assert pending.enqueue_id != 'when_idle'
    assert UUID(pending.enqueue_id).version == 7


async def test_single_enqueue_with_multiple_messages_emits_one_event():
    """One `enqueue` call carrying multiple messages emits a single event with all delivered messages."""
    events: list[AgentStreamEvent] = []
    enqueue_id: str | None = None

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            yield 'done'
            return
        yield {0: DeltaToolCall(name='inject_exchange', json_args='{}')}

    agent = Agent(FunctionModel(stream_function=stream_fn))

    @agent.tool
    def inject_exchange(ctx: RunContext[Any]) -> str:
        nonlocal enqueue_id
        # A synthetic prior turn (a complete response) followed by a fresh user request:
        # one enqueue call, two delivered messages.
        enqueue_id = ctx.enqueue(
            ModelResponse(parts=[TextPart(content='synthetic recap')]),
            'Follow up on the recap',
        )
        return 'ok'

    async def event_stream_handler(_: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(event)

    result = await agent.run('Hello', event_stream_handler=event_stream_handler)

    assert enqueue_id is not None
    delivery_events = [event for event in events if isinstance(event, EnqueuedMessagesEvent)]
    assert delivery_events == [EnqueuedMessagesEvent(enqueue_id=enqueue_id, messages=tuple(result.all_messages()[3:5]))]
    assert isinstance(delivery_events[0].messages[0], ModelResponse)
    assert isinstance(delivery_events[0].messages[1], ModelRequest)


async def test_enqueue_delivery_event_via_run_stream():
    """The delivery event surfaces through `agent.run_stream`'s `event_stream_handler`."""
    events: list[AgentStreamEvent] = []
    enqueue_id: str | None = None

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            yield 'done'
            return
        yield {0: DeltaToolCall(name='inject_msg', json_args='{}')}

    agent = Agent(FunctionModel(stream_function=stream_fn))

    @agent.tool
    def inject_msg(ctx: RunContext[Any]) -> str:
        nonlocal enqueue_id
        enqueue_id = ctx.enqueue('Injected asap message')
        return 'ok'

    async def event_stream_handler(_: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(event)

    async with agent.run_stream('Hello', event_stream_handler=event_stream_handler) as result:
        output = await result.get_output()

    assert output == 'done'
    assert enqueue_id is not None
    delivery_events = [event for event in events if isinstance(event, EnqueuedMessagesEvent)]
    tool_request = result.all_messages()[2]
    assert isinstance(tool_request, ModelRequest)
    assert delivery_events == [
        EnqueuedMessagesEvent(
            enqueue_id=enqueue_id,
            messages=(
                ModelRequest(
                    parts=[tool_request.parts[1]],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ),
            parts=(tool_request.parts[1],),
        )
    ]


async def test_with_event_stream_buffer_drains_around_node_stream():
    """`_with_event_stream_buffer` yields buffered events before and after the node stream.

    It deliberately does not drain between node events: a live node stream yields buffered events
    itself as they are emitted, and draining here as well could invert emission order.
    """
    buffer: list[AgentStreamEvent] = [initial := EnqueuedMessagesEvent(enqueue_id='initial', messages=())]
    during = EnqueuedMessagesEvent(enqueue_id='during', messages=())
    after = EnqueuedMessagesEvent(enqueue_id='after', messages=())
    model_event = PartStartEvent(index=0, part=TextPart(content='done'))

    async def stream() -> AsyncIterator[AgentStreamEvent]:
        buffer.append(during)
        yield model_event
        buffer.append(after)

    drained = [event async for event in _agent_graph._with_event_stream_buffer(stream(), buffer)]  # pyright: ignore[reportPrivateUsage]
    assert drained == [initial, model_event, during, after]


async def test_agent_stream_events_iter_drains_buffer_before_each_pull():
    """`AgentStream._events_iter` drains buffered run events before each pull from the model stream.

    Events buffered while a pull is in flight surface on the next pull; events buffered after the
    last model event are not drained here — they flow through the response-handling node's stream
    (`_with_event_stream_buffer`'s trailing drain) once this stream is exhausted.
    """
    initial = EnqueuedMessagesEvent(enqueue_id='initial', messages=())
    during = EnqueuedMessagesEvent(enqueue_id='during', messages=())
    after = EnqueuedMessagesEvent(enqueue_id='after', messages=())
    model_event = PartStartEvent(index=0, part=TextPart(content='done'))
    buffer: list[AgentStreamEvent] = [initial]

    async def base_iter() -> AsyncIterator[ModelResponseStreamEvent]:
        buffer.append(during)
        yield model_event
        buffer.append(after)

    stream = cast(AgentStream[Any, str], object.__new__(AgentStream))
    stream._event_stream_buffer_getter = lambda: buffer  # pyright: ignore[reportPrivateUsage]
    stream._anext_lock = anyio.Lock()  # pyright: ignore[reportPrivateUsage]

    drained = [event async for event in stream._events_iter(base_iter())]  # pyright: ignore[reportPrivateUsage]
    assert drained == [initial, model_event, during]
    # `after` stays buffered for the response-handling node's stream to deliver.
    assert buffer == [after]


class _FixedEventsAgentStream(AgentStream[Any, str]):
    """An `AgentStream` whose event stream is a fixed list, for testing the event filters."""

    def __init__(self, events: list[AgentStreamEvent]) -> None:
        self._events = events

    def __aiter__(self) -> AsyncIterator[AgentStreamEvent]:
        return self._iter_events()

    async def _iter_events(self) -> AsyncIterator[AgentStreamEvent]:
        for event in self._events:
            yield event


async def test_agent_stream_model_response_events_skips_buffered_events():
    """`AgentStream._model_response_events` filters buffered run events out of the model response stream."""
    buffered = EnqueuedMessagesEvent(enqueue_id='buffered', messages=())
    model_event = PartStartEvent(index=0, part=TextPart(content='done'))
    stream = _FixedEventsAgentStream([buffered, model_event])

    drained = [event async for event in stream._model_response_events()]  # pyright: ignore[reportPrivateUsage]
    assert drained == [model_event]


async def test_enqueue_when_idle_message_prevents_end():
    """`'when_idle'` messages prevent the agent from ending and are drained into a new ModelRequest."""
    call_count = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(
                parts=[ToolCallPart(tool_name='inject_follow_up', args='{}')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        elif call_count == 2:
            # Agent produces final result, but follow-up is pending
            return ModelResponse(
                parts=[TextPart(content='premature end')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        else:
            # After follow-up is drained, agent produces real final result
            return ModelResponse(
                parts=[TextPart(content='final answer after follow-up')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )

    agent = Agent(FunctionModel(model_fn))

    @agent.tool
    def inject_follow_up(ctx: RunContext[object]) -> str:
        ctx.enqueue('Follow-up context', priority='when_idle')
        return 'ok'

    result = await agent.run('Hello')
    assert result.output == 'final answer after follow-up'
    assert call_count == 3
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='inject_follow_up', args='{}', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='inject_follow_up',
                        content='ok',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='premature end')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='Follow-up context', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='final answer after follow-up')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_enqueue_when_idle_redirects_after_output_tool_end():
    """A `when_idle` message redirects the run even when it would end via an output tool.

    The run terminates when the model calls an output tool (`ToolOutput` mode), which produces
    an `End` from `CallToolsNode`. The drain's `after_node_run` still sees that `End` and
    redirects into a fresh request, so the agent gets another turn after the structured output —
    and the final `result.output` comes from that later turn.
    """

    class Answer(BaseModel):
        value: int

    call_count = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        output_tool = info.output_tools[0].name
        if call_count == 1:
            return ModelResponse(
                parts=[ToolCallPart(tool_name='inject_follow_up', args='{}')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        if call_count == 2:
            # Would end the run via the output tool, but a `when_idle` message is pending.
            return ModelResponse(
                parts=[ToolCallPart(tool_name=output_tool, args='{"value": 1}')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        # After the follow-up is drained, the model produces the real final output.
        return ModelResponse(
            parts=[ToolCallPart(tool_name=output_tool, args='{"value": 2}')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    agent = Agent(FunctionModel(model_fn), output_type=Answer)

    @agent.tool
    def inject_follow_up(ctx: RunContext[object]) -> str:
        ctx.enqueue('Follow-up context', priority='when_idle')
        return 'ok'

    result = await agent.run('Hello')

    assert result.output == Answer(value=2)
    assert call_count == 3
    # The `when_idle` follow-up lands as its own request after the first (superseded) output-tool
    # call, redirecting the run so the second output-tool call produces the real output.
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='inject_follow_up', args='{}', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='inject_follow_up',
                        content='ok',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"value": 1}',
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='Follow-up context', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"value": 2}',
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_enqueue_from_agent_run():
    """Messages can be enqueued from external code via AgentRun.enqueue."""
    call_count = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        return ModelResponse(
            parts=[TextPart(content=f'response {call_count}')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    agent = Agent(FunctionModel(model_fn))

    async with agent.iter('Hello') as agent_run:
        assert agent_run.pending_messages == []
        # Enqueue a when_idle message from external code before iteration
        agent_run.enqueue('External follow-up', priority='when_idle')
        assert len(agent_run.pending_messages) == 1
        # Use next() to drive iteration so after_node_run fires
        node = agent_run.next_node
        while not isinstance(node, End):
            node = await agent_run.next(node)

    assert agent_run.result is not None
    assert call_count == 2  # First response triggers End, follow-up prevents it, second response is final
    assert agent_run.result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='response 1')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='External follow-up', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='response 2')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_bare_async_for_drains_pending_messages():
    """Bare `async for` drains `when_idle` messages, because it advances through `next()`.

    `when_idle` messages (and end-of-step `asap` leftovers) drain in `after_node_run`. Bare
    iteration used to skip the node hooks and strand them, raising `UndrainedPendingMessagesError`
    instead; it now fires the same hooks as `agent.run()`, so the message is delivered.
    """

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart(content='done')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    agent = Agent(FunctionModel(model_fn))

    async with agent.iter('hi') as agent_run:
        agent_run.enqueue('stranded follow-up', priority='when_idle')
        async for _ in agent_run:
            pass

        assert agent_run.pending_messages == []
        assert any(
            isinstance(part, UserPromptPart) and part.content == 'stranded follow-up'
            for message in agent_run.all_messages()
            for part in message.parts
        )


async def test_pending_messages_accessible_on_run_context():
    """RunContext.pending_messages is accessible and initially empty."""

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            return ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        return ModelResponse(
            parts=[ToolCallPart(tool_name='check_queue', args='{}')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    agent = Agent(FunctionModel(model_fn))

    @agent.tool
    def check_queue(ctx: RunContext[object]) -> str:
        # The queue must be live (mutations from inside a tool reach the drain).
        assert ctx.pending_messages is not None
        assert len(ctx.pending_messages) == 0
        ctx.enqueue('observed', priority='asap')
        assert len(ctx.pending_messages) == 1
        return 'done'

    result = await agent.run('Test')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Test', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='check_queue', args='{}', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='check_queue',
                        content='done',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(content='observed', timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_enqueue_with_no_args_is_a_noop():
    """`ctx.enqueue()` and `agent_run.enqueue()` with no content are silent no-ops.

    Producers that conditionally enqueue (e.g. "announce if new tools were discovered")
    can call `enqueue(*maybe_items)` without guarding for the empty case — `enqueue`
    simply doesn't append a `PendingMessage` when there's nothing to send.
    """

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            return ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        return ModelResponse(
            parts=[ToolCallPart(tool_name='from_tool', args='{}')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    agent = Agent(FunctionModel(model_fn))

    @agent.tool
    def from_tool(ctx: RunContext[Any]) -> str:
        assert ctx.enqueue() is None  # no-op, no exception, no id
        assert ctx.pending_messages == []
        return 'ok'

    async with agent.iter('hi') as agent_run:
        assert agent_run.enqueue() is None  # no-op, no exception, no id
        assert agent_run.pending_messages == []
        async for _ in agent_run:
            pass


async def test_enqueue_coerces_string_to_user_prompt():
    """A bare string passed to `enqueue` is wrapped in a `UserPromptPart`."""

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            return ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        return ModelResponse(
            parts=[ToolCallPart(tool_name='inject_msg', args='{}')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    agent = Agent(FunctionModel(model_fn))

    @agent.tool
    def inject_msg(ctx: RunContext[object]) -> str:
        ctx.enqueue('steering as plain string')
        return 'ok'

    result = await agent.run('Hello')
    injected = [
        part
        for part in iter_message_parts(result.all_messages(), ModelRequest, UserPromptPart)
        if part.content == 'steering as plain string'
    ]
    assert len(injected) == 1, 'string-coerced enqueue did not land as a UserPromptPart'


async def test_enqueue_accepts_multimodal_user_content():
    """Adjacent user-content args (text + multi-modal) are gathered into one `UserPromptPart`."""
    image = ImageUrl(url='https://example.com/image.png')

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            return ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        return ModelResponse(
            parts=[ToolCallPart(tool_name='inject_msg', args='{}')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    agent = Agent(FunctionModel(model_fn))

    @agent.tool
    def inject_msg(ctx: RunContext[object]) -> str:
        ctx.enqueue('look at this', image)
        return 'ok'

    result = await agent.run('Hello')
    injected = [
        part
        for part in iter_message_parts(result.all_messages(), ModelRequest, UserPromptPart)
        if part.content == ['look at this', image]
    ]
    assert len(injected) == 1


async def test_enqueue_accepts_model_request_passthrough():
    """A full `ModelRequest` is enqueued verbatim, preserving `instructions`/`metadata`.

    Two passthroughs cover both branches of the fill-in-if-unset stamping logic:
    one with `timestamp`/`run_id` unset (drain stamps them); one with both set
    (drain leaves them alone).
    """

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            return ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        return ModelResponse(
            parts=[ToolCallPart(tool_name='inject_msg', args='{}')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    agent = Agent(FunctionModel(model_fn))
    unstamped = ModelRequest(
        parts=[UserPromptPart(content='wire-level payload')],
        instructions='do this carefully',
        metadata={'origin': 'webhook-42'},
    )
    preset_timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    prestamped = ModelRequest(
        parts=[UserPromptPart(content='already stamped')],
        instructions='preserve me',
        timestamp=preset_timestamp,
        run_id='caller-run-id',
        conversation_id='caller-conv-id',
    )

    @agent.tool
    def inject_msg(ctx: RunContext[object]) -> str:
        ctx.enqueue(unstamped)
        ctx.enqueue(prestamped)
        return 'ok'

    result = await agent.run('Hello')

    injected_unstamped = next(
        msg
        for msg in result.all_messages()
        if isinstance(msg, ModelRequest) and msg.instructions == 'do this carefully'
    )
    assert injected_unstamped.metadata == {'origin': 'webhook-42'}
    # Drain should have stamped timestamp/run_id/conversation_id since the user didn't set them.
    assert injected_unstamped.timestamp is not None
    assert injected_unstamped.run_id is not None
    assert injected_unstamped.conversation_id is not None

    injected_prestamped = next(
        msg for msg in result.all_messages() if isinstance(msg, ModelRequest) and msg.instructions == 'preserve me'
    )
    # Producer-supplied timestamp/run_id/conversation_id are preserved (drain doesn't overwrite).
    assert injected_prestamped.timestamp == preset_timestamp
    assert injected_prestamped.run_id == 'caller-run-id'
    assert injected_prestamped.conversation_id == 'caller-conv-id'


def test_pending_message_drain_capability_is_not_spec_constructible():
    """`PendingMessageDrainCapability` is auto-injected only; can't be in an `AgentSpec`."""
    from pydantic_ai.capabilities._pending_messages import PendingMessageDrainCapability

    assert PendingMessageDrainCapability.get_serialization_name() is None


def test_pending_message_allows_empty_request():
    """`PendingMessage` doesn't validate its `messages`; empty-parts requests are tolerated.

    `enqueue()` already filters out the no-args case (no `PendingMessage` is appended).
    An empty `ModelRequest` reaching the queue is harmless — the drain stamps and forwards
    it, and downstream wire-merging absorbs zero-part messages as a natural no-op.
    """
    msg = PendingMessage(messages=[ModelRequest(parts=[])])
    assert msg.priority == 'asap'
    assert msg.messages[0].parts == []


def test_enqueue_without_live_queue_raises():
    """`ctx.enqueue` raises when the `RunContext` isn't backed by a running agent's queue.

    Synthetic contexts (e.g. the one `Agent.system_prompt_parts` builds to resolve system
    prompts outside a run) have no queue to drain to, so enqueue fails loudly instead of
    silently dropping the message.
    """
    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[])
    assert ctx.pending_messages is None
    with pytest.raises(UserError, match='only available during an agent run'):
        ctx.enqueue('this has nowhere to go')


async def test_enqueue_parts_style_calls_keep_their_own_parts():
    """Each `enqueue` call keeps its own part: two calls from one tool aren't coalesced into one prompt.

    Both land in the tool's request, after its return, in the order they were enqueued.
    """

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            return ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        return ModelResponse(
            parts=[ToolCallPart(tool_name='inject_msg', args='{}')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    agent = Agent(FunctionModel(model_fn))

    @agent.tool
    def inject_msg(ctx: RunContext[object]) -> str:
        ctx.enqueue('first hint')
        ctx.enqueue('second hint')
        return 'ok'

    result = await agent.run('Hello')
    drained = [
        msg
        for msg in result.all_messages()
        if isinstance(msg, ModelRequest)
        and any(isinstance(p, UserPromptPart) and p.content in ('first hint', 'second hint') for p in msg.parts)
    ]
    assert [[type(p).__name__ for p in msg.parts] for msg in drained] == [
        ['ToolReturnPart', 'UserPromptPart', 'UserPromptPart']
    ]
    assert [p.content for p in iter_message_parts(drained, ModelRequest, UserPromptPart)] == [
        'first hint',
        'second hint',
    ]


async def test_enqueue_passthrough_stays_separate_from_parts_style():
    """A passthrough `ModelRequest` stays its own message, and nothing enqueued after it moves ahead of it.

    It may carry its own `instructions` or other message-level fields, so it's never put into the tool's
    request. Content enqueued before it goes into the tool's request; content enqueued after it follows it as
    its own message, so the model gets all three in enqueue order.
    """
    received: list[list[ModelMessage]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        received.append(messages)
        if any(isinstance(msg, ModelResponse) for msg in messages):
            return ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        return ModelResponse(
            parts=[ToolCallPart(tool_name='inject_msg', args='{}')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    agent = Agent(FunctionModel(model_fn))

    @agent.tool
    def inject_msg(ctx: RunContext[object]) -> str:
        ctx.enqueue('before')
        ctx.enqueue(
            ModelRequest(parts=[UserPromptPart(content='passthrough')], instructions='careful'),
        )
        ctx.enqueue('after')
        return 'ok'

    def contents(messages: Sequence[ModelMessage]) -> list[list[str]]:
        return [
            [
                f'return:{p.content}' if isinstance(p, ToolReturnPart) else str(p.content)
                for p in message.parts
                if isinstance(p, ToolReturnPart | UserPromptPart)
            ]
            for message in messages
            if isinstance(message, ModelRequest)
        ]

    result = await agent.run('Hello')
    assert contents(result.all_messages()[2:]) == [['return:ok', 'before'], ['passthrough'], ['after']]
    passthrough = result.all_messages()[3]
    assert isinstance(passthrough, ModelRequest)
    # Passthrough preserved its instructions.
    assert passthrough.instructions == 'careful'
    assert contents(received[-1])[-1] == ['return:ok', 'before', 'passthrough', 'after']


async def test_parallel_tool_enqueues_land_after_their_own_returns():
    """Content enqueued by parallel tool calls lands after its own call's return, in enqueue order per call.

    The calls interleave their enqueues (`a1`, `b1`, `a2`) and their returns are stored in call order
    (`tool_b` first), so `b1` goes in ahead of `tool_a`'s content and must not push `a2` ahead of `a1`.
    `tool_c` enqueues nothing and asks for a retry, so its result gets nothing after it. The model still
    gets every tool result ahead of the enqueued content, as providers require.
    """
    model_requests: list[ModelRequest] = []

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        assert isinstance(messages[-1], ModelRequest)
        model_requests.append(messages[-1])
        if any(isinstance(msg, ModelResponse) for msg in messages):
            yield 'done'
            return
        yield {
            0: DeltaToolCall(name='tool_b', json_args='{}', tool_call_id='call-b'),
            1: DeltaToolCall(name='tool_a', json_args='{}', tool_call_id='call-a'),
            2: DeltaToolCall(name='tool_c', json_args='{}', tool_call_id='call-c'),
        }

    agent = Agent(FunctionModel(stream_function=stream_fn))
    a1_enqueued, b1_enqueued = anyio.Event(), anyio.Event()

    @agent.tool
    async def tool_a(ctx: RunContext[object]) -> str:
        ctx.enqueue('a1')
        a1_enqueued.set()
        await b1_enqueued.wait()
        ctx.enqueue('a2')
        return 'a'

    @agent.tool
    async def tool_b(ctx: RunContext[object]) -> str:
        await a1_enqueued.wait()
        ctx.enqueue('b1')
        b1_enqueued.set()
        return 'b'

    @agent.tool_plain
    def tool_c() -> str:
        raise ModelRetry('try again')

    events: list[AgentStreamEvent] = []

    async def event_stream_handler(_: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(event)

    result = await agent.run('Hello', event_stream_handler=event_stream_handler)

    def describe(request: ModelMessage) -> list[str]:
        assert isinstance(request, ModelRequest)
        return [
            f'{part.part_kind}:{part.tool_name}'
            if isinstance(part, ToolReturnPart)
            else f'{part.part_kind}:{part.content}'
            for part in request.parts
            if isinstance(part, ToolReturnPart | UserPromptPart)
        ]

    assert describe(result.all_messages()[2]) == snapshot(
        [
            'tool-return:tool_b',
            'user-prompt:b1',
            'tool-return:tool_a',
            'user-prompt:a1',
            'user-prompt:a2',
            'tool-return:tool_c',
        ]
    )
    # One event per `enqueue` call, in the order the content lands in history.
    assert [
        [part.content for part in event.parts if isinstance(part, UserPromptPart)]
        for event in events
        if isinstance(event, EnqueuedMessagesEvent)
    ] == [['b1'], ['a1'], ['a2']]
    assert describe(model_requests[-1]) == snapshot(
        [
            'tool-return:tool_b',
            'tool-return:tool_a',
            'tool-return:tool_c',
            'user-prompt:b1',
            'user-prompt:a1',
            'user-prompt:a2',
        ]
    )


async def test_enqueue_from_tool_is_its_own_request_when_the_request_was_rebuilt():
    """When a wrapper has replaced the request's messages with copies, the tool's request isn't there to add to.

    The enqueued content is then added as its own request, so the model still sees it and history keeps it.
    """
    received: list[list[ModelMessage]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        received.append(messages)
        if any(isinstance(msg, ModelResponse) for msg in messages):
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='inject_msg', args='{}', tool_call_id='call-1')])

    @dataclass
    class CopyMessages(AbstractCapability[Any]):
        async def wrap_model_request(
            self,
            ctx: RunContext[Any],
            *,
            request_context: ModelRequestContext,
            handler: WrapModelRequestHandler,
        ) -> ModelResponse:
            request_context.messages = copy.deepcopy(request_context.messages)
            return await handler(request_context)

    agent = Agent(FunctionModel(model_fn), capabilities=[CopyMessages()])

    @agent.tool
    def inject_msg(ctx: RunContext[object]) -> str:
        ctx.enqueue('Injected asap message')
        return 'ok'

    result = await agent.run('Hello')
    assert [[part.part_kind for part in message.parts] for message in result.all_messages()[2:4]] == [
        ['tool-return'],
        ['user-prompt'],
    ]
    assert [part.part_kind for part in received[-1][-1].parts] == ['tool-return', 'user-prompt']


@pytest.mark.vcr
@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
async def test_parallel_tool_enqueues_reach_anthropic(allow_model_requests: None, anthropic_api_key: str):
    """Anthropic accepts the request after parallel tool calls that each enqueue content.

    It requires every tool result to lead the message answering the calls, and in history each call's
    content sits between the results.
    """
    model = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(model)

    @agent.tool
    def get_weather(ctx: RunContext[object], city: str) -> str:
        ctx.enqueue(f'The {city} reading came from a backup station.')
        return f'Sunny in {city}'

    result = await agent.run(
        'Call get_weather for Paris and for London in parallel, then summarize both in one short sentence.'
    )
    tool_request = result.all_messages()[2]
    assert isinstance(tool_request, ModelRequest)
    assert [type(part).__name__ for part in tool_request.parts] == snapshot(
        ['ToolReturnPart', 'UserPromptPart', 'ToolReturnPart', 'UserPromptPart']
    )
    assert result.output == snapshot(
        'Thank you for that clarification. Both weather readings are from backup stations—Paris is sunny (via backup station) and London is sunny (via backup station). Since both cities are reporting sunny conditions from their respective backup weather stations, the summary remains the same: Both Paris and London are experiencing sunny weather according to their backup station readings.'
    )


async def test_late_enqueue_from_an_earlier_call_with_a_reused_id_is_its_own_request():
    """Content an earlier call enqueues late isn't put next to a later call's return that reuses its id.

    The model calls `tool_a` with `call_0` twice, in two steps, and each call's own content follows its return.
    The first call also starts a task that enqueues while the second call runs, so that content still carries
    `call_0`, but it doesn't belong to the second return.
    """
    model_calls = 0
    second_call_started, late_enqueued = anyio.Event(), anyio.Event()

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls <= 2:
            return ModelResponse(parts=[ToolCallPart('tool_a', {'call': model_calls}, tool_call_id='call_0')])
        return ModelResponse(parts=[TextPart(content='done')])

    agent = Agent(FunctionModel(model_fn))

    async with anyio.create_task_group() as task_group:

        @agent.tool
        async def tool_a(ctx: RunContext[object], call: int) -> str:
            ctx.enqueue(f'from call {call}')
            if call == 1:

                async def report_later() -> None:
                    await second_call_started.wait()
                    ctx.enqueue('late from the first call')
                    late_enqueued.set()

                task_group.start_soon(report_later)
            else:
                second_call_started.set()
                await late_enqueued.wait()
            return f'result {call}'

        result = await agent.run('Hello')
        assert [
            [part.content for part in message.parts if isinstance(part, ToolReturnPart | UserPromptPart)]
            for message in result.all_messages()
            if isinstance(message, ModelRequest)
        ] == [
            ['Hello'],
            ['result 1', 'from call 1'],
            ['result 2', 'from call 2'],
            ['late from the first call'],
        ]


async def test_tool_call_content_delivered_in_two_drains_keeps_enqueue_order():
    """Content a call enqueues after the first delivery of the step still lands after what that delivery put in.

    A task the tool started enqueues `b` while a `before_model_request` hook runs, between the delivery ahead of
    the hooks and the one after them.
    """
    first_delivered, b_enqueued = anyio.Event(), anyio.Event()

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='inject_msg', args='{}', tool_call_id='call-1')])

    @dataclass
    class WaitForLateContent(AbstractCapability[Any]):
        async def before_model_request(
            self, ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            if any(isinstance(part, ToolReturnPart) for part in ctx.messages[-1].parts):
                first_delivered.set()
                await b_enqueued.wait()
            return request_context

    agent = Agent(FunctionModel(model_fn), capabilities=[WaitForLateContent()])

    async with anyio.create_task_group() as task_group:

        @agent.tool
        async def inject_msg(ctx: RunContext[object]) -> str:
            ctx.enqueue('a')

            async def enqueue_later() -> None:
                await first_delivered.wait()
                ctx.enqueue('b')
                b_enqueued.set()

            task_group.start_soon(enqueue_later)
            return 'ok'

        result = await agent.run('Hello')
        tool_request = result.all_messages()[2]
        assert isinstance(tool_request, ModelRequest)
        assert [part.content for part in tool_request.parts if isinstance(part, ToolReturnPart | UserPromptPart)] == [
            'ok',
            'a',
            'b',
        ]


async def test_request_with_tool_call_content_records_what_was_sent():
    """The request tool-call content went into is the step's request: what's recorded on it lands in history.

    A `before_model_request` hook adds an instruction part each step, and the instructions sent are recorded on
    the step's request, which the content was put into by a copy.
    """

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='inject_msg', args='{}', tool_call_id='call-1')])

    @dataclass
    class StepInstructions(AbstractCapability[Any]):
        async def before_model_request(
            self, ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            parameters = request_context.model_request_parameters
            request_context.model_request_parameters = replace(
                parameters,
                instruction_parts=[*(parameters.instruction_parts or []), InstructionPart(f'step {ctx.run_step}')],
            )
            return request_context

    agent = Agent(FunctionModel(model_fn), instructions='base', capabilities=[StepInstructions()])

    @agent.tool
    def inject_msg(ctx: RunContext[object]) -> str:
        ctx.enqueue('side effect')
        return 'ok'

    nodes: list[_agent_graph.ModelRequestNode[Any, Any]] = []
    async with agent.iter('Hello') as run:
        async for node in run:
            if isinstance(node, _agent_graph.ModelRequestNode):
                nodes.append(node)
    assert run.result is not None
    history = run.result.all_messages()
    assert [any(node.request is message for message in history) for node in nodes] == [True, True]
    tool_request = history[2]
    assert isinstance(tool_request, ModelRequest)
    assert [part.part_kind for part in tool_request.parts] == ['tool-return', 'user-prompt']
    assert tool_request.instructions == 'base\n\nstep 2'


async def test_directly_built_pending_message_with_several_messages_is_delivered_whole():
    """A `PendingMessage` built by hand, holding more than one request's worth, is added as its own messages."""

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='inject_msg', args='{}', tool_call_id='call-1')])

    agent = Agent(FunctionModel(model_fn))

    @agent.tool
    def inject_msg(ctx: RunContext[object]) -> str:
        assert ctx.pending_messages is not None and ctx.tool_call_id is not None
        ctx.pending_messages.append(
            PendingMessage(
                messages=[
                    ModelResponse(parts=[TextPart(content='synthetic')]),
                    ModelRequest(parts=[UserPromptPart(content='follow-up')]),
                ],
                _tool_call=(ctx.tool_call_id, ctx.run_step),
            )
        )
        return 'ok'

    result = await agent.run('Hello')
    assert [message.kind for message in result.all_messages()[2:5]] == ['request', 'response', 'request']


async def test_enqueue_from_tool_renders_like_its_own_request():
    """The model gets the same bytes as when the enqueued content was its own request after the tool's."""
    received: list[list[ModelMessage]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        received.append(messages)
        if any(isinstance(msg, ModelResponse) for msg in messages):
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='inject_msg', args='{}', tool_call_id='call-1')])

    agent = Agent(FunctionModel(model_fn))

    @agent.tool
    def inject_msg(ctx: RunContext[object]) -> str:
        ctx.enqueue('first hint')
        ctx.enqueue('second hint')
        return 'ok'

    result = await agent.run('Hello')
    *before, tool_request, _ = result.all_messages()
    assert isinstance(tool_request, ModelRequest)
    tool_return, *enqueued = tool_request.parts
    own_request = [*before, replace(tool_request, parts=[tool_return]), replace(tool_request, parts=enqueued)]
    expected = _clean_message_history(own_request, repair_last_response=True)
    # Merging two requests drops their `run_id` / `conversation_id`, which are never sent to the model.
    sent = [*received[-1][:-1], replace(received[-1][-1], run_id=None, conversation_id=None)]
    assert ModelMessagesTypeAdapter.dump_json(sent) == ModelMessagesTypeAdapter.dump_json(expected)


@pytest.mark.parametrize(
    'parts',
    [
        pytest.param([UserPromptPart('Then this.'), ToolReturnPart('tool_a', 'a', 'call-a')], id='prompt-first'),
        pytest.param([ToolReturnPart('tool_a', 'a', 'call-a'), UserPromptPart('Then this.')], id='prompt-last'),
        pytest.param(
            [
                RetryFeedbackPart('Try again.', retry=RetryInfo(cause='model_retry')),
                ToolReturnPart('tool_a', 'a', 'call-a'),
            ],
            id='plain-retry-first',
        ),
        pytest.param(
            [
                ToolReturnPart('tool_a', 'a', 'call-a'),
                ToolAvailabilityDeltaPart(tools_added=['tool_c']),
                ToolReturnPart('tool_b', 'b', 'call-b'),
            ],
            id='delta-between-results',
        ),
    ],
)
def test_requests_without_parts_between_tool_results_are_sent_unchanged(parts: list[ModelRequestPart]):
    """Only parts sitting between a request's tool results move; any other request goes out exactly as stored.

    A unit test because it pins the wire shape of histories that enqueue doesn't produce, such as persisted ones.
    """
    calls = [
        ToolCallPart(part.tool_name, {}, tool_call_id=part.tool_call_id)
        for part in parts
        if isinstance(part, ToolReturnPart)
    ]
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('Hello')]),
        ModelResponse(parts=calls),
        ModelRequest(parts=parts),
    ]
    cleaned = _clean_message_history(history)
    assert ModelMessagesTypeAdapter.dump_json(cleaned) == ModelMessagesTypeAdapter.dump_json(history)


async def test_enqueue_system_prompt_part():
    """A bare `SystemPromptPart` is coalesced into a `ModelRequest` and delivered.

    Now that mid-conversation `SystemPromptPart`s are rendered inline (not hoisted) on all
    providers, `enqueue` accepts request parts directly — no `ModelRequest` wrapper needed.
    """

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            return ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        return ModelResponse(
            parts=[ToolCallPart(tool_name='announce', args='{}')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    agent = Agent(FunctionModel(model_fn))

    @agent.tool
    def announce(ctx: RunContext[object]) -> str:
        ctx.enqueue(SystemPromptPart(content='New tools are now available.'))
        return 'ok'

    result = await agent.run('Hello')
    injected = next(
        msg
        for msg in result.all_messages()
        if isinstance(msg, ModelRequest)
        and any(isinstance(p, SystemPromptPart) and p.content == 'New tools are now available.' for p in msg.parts)
    )
    assert injected is not None


async def test_enqueue_tool_availability_delta_part():
    """A `ToolAvailabilityDeltaPart` enqueues as a request part, not as user content.

    It's a `ModelRequestPart` like the rest, so it has to be coalesced into the `ModelRequest`
    alongside a user prompt. Falling through to the user-content branch instead would bury the
    change inside a `UserPromptPart`, where every adapter's delta rendering would miss it.
    """

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            return ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        return ModelResponse(
            parts=[ToolCallPart(tool_name='announce', args='{}')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    agent = Agent(FunctionModel(model_fn))

    @agent.tool
    def announce(ctx: RunContext[object]) -> str:
        ctx.enqueue(ToolAvailabilityDeltaPart(tools_added=['lookup_exchange_rate']), 'Use it.')
        return 'ok'

    result = await agent.run('Hello')
    injected = next(
        msg
        for msg in result.all_messages()
        if isinstance(msg, ModelRequest) and any(isinstance(p, ToolAvailabilityDeltaPart) for p in msg.parts)
    )
    assert [type(part).__name__ for part in injected.parts] == snapshot(
        ['ToolReturnPart', 'ToolAvailabilityDeltaPart', 'UserPromptPart']
    )


async def test_enqueue_interleaved_response_and_request():
    """One `enqueue` call can inject an interleaved `ModelResponse` + `ModelRequest` exchange.

    This is the synthetic "tool-search call + result" shape (a `ModelResponse` carrying the call
    followed by a `ModelRequest` carrying the return). Both land in history in order, and the
    trailing request is what the agent responds to next.
    """

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            return ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        return ModelResponse(
            parts=[ToolCallPart(tool_name='inject_exchange', args='{}')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    agent = Agent(FunctionModel(model_fn))
    synthetic_response = ModelResponse(
        parts=[TextPart(content='synthetic prior turn')],
        usage=RequestUsage(input_tokens=1, output_tokens=1),
    )

    @agent.tool
    def inject_exchange(ctx: RunContext[object]) -> str:
        ctx.enqueue(
            synthetic_response,
            ModelRequest(parts=[UserPromptPart(content='follow-up after synthetic turn')]),
            priority='when_idle',
        )
        return 'ok'

    result = await agent.run('Hello')
    # The synthetic response is appended to history immediately before its paired request.
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='inject_exchange', args='{}', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='inject_exchange',
                        content='ok',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='synthetic prior turn')],
                usage=RequestUsage(input_tokens=1, output_tokens=1),
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='follow-up after synthetic turn', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_enqueue_rejects_content_not_ending_in_request():
    """Enqueued content must end in a `ModelRequest`; a lone `ModelResponse` is rejected.

    The agent needs a request to respond to — content that ends in a `ModelResponse` (with no
    trailing request/part-style items) would leave nothing for the model to react to.
    """

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            return ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        return ModelResponse(
            parts=[ToolCallPart(tool_name='from_tool', args='{}')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    agent = Agent(FunctionModel(model_fn))
    lone_response = ModelResponse(
        parts=[TextPart(content='synthetic')], usage=RequestUsage(input_tokens=1, output_tokens=1)
    )

    @agent.tool
    def from_tool(ctx: RunContext[object]) -> str:
        with pytest.raises(UserError, match='must end with a `ModelRequest`'):
            ctx.enqueue(lone_response)
        return 'ok'

    async with agent.iter('hi') as agent_run:
        with pytest.raises(UserError, match='must end with a `ModelRequest`'):
            agent_run.enqueue(lone_response)
        async for _ in agent_run:
            pass


async def test_drain_rejects_directly_queued_content_not_ending_in_request():
    """Directly appending a malformed `PendingMessage` raises a `UserError` at end-of-run drain.

    `enqueue` enforces the "ends in a `ModelRequest`" rule up front, but `RunContext.pending_messages`
    is public, so a producer can append a `PendingMessage` directly. The end-of-run drain catches a
    request-less message with a helpful `UserError` rather than a bare assertion.
    """
    lone_response = ModelResponse(
        parts=[TextPart(content='synthetic')], usage=RequestUsage(input_tokens=1, output_tokens=1)
    )

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(p, ToolReturnPart) for m in messages if isinstance(m, ModelRequest) for p in m.parts):
            return ModelResponse(parts=[TextPart(content='done')], usage=RequestUsage(input_tokens=10, output_tokens=5))
        return ModelResponse(
            parts=[ToolCallPart(tool_name='queue_bad', args='{}')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    agent = Agent(FunctionModel(model_fn))

    @agent.tool
    def queue_bad(ctx: RunContext[object]) -> str:
        assert ctx.pending_messages is not None
        ctx.pending_messages.append(PendingMessage(messages=[lone_response], priority='when_idle'))
        return 'ok'

    with pytest.raises(UserError, match='must end with a `ModelRequest`'):
        await agent.run('hi')


async def test_enqueue_asap_with_rich_message_history_tail():
    """`'asap'` enqueue lands as its own `ModelRequest` in history *and* gets wire-merged into the rich tail.

    The history keeps the un-merged view (drain's request is a separate `ModelRequest`
    after the rich tail) so `all_messages()` reflects per-call structure. On the wire,
    `_clean_message_history` merges the two adjacent `ModelRequest`s and sorts
    `ToolReturnPart`s first — non-tool parts keep arrival order, so the
    enqueued content lands at the *end* of the merged turn (not interleaved between
    existing parts). Captures the `messages` arg `FunctionModel` actually received to
    validate the wire-level merge through the public path.
    """
    captured_wire_messages: list[list[ModelMessage]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captured_wire_messages.append(messages)
        return ModelResponse(
            parts=[TextPart(content='done')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    agent = Agent(FunctionModel(model_fn))
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='original prompt')]),
        ModelResponse(
            parts=[ToolCallPart(tool_name='hint', args='{}', tool_call_id='call-1')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
            model_name='function:model_fn:',
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name='hint', content='ok', tool_call_id='call-1'),
                UserPromptPart(content='follow-up question'),
            ],
        ),
    ]

    async with agent.iter(message_history=history) as agent_run:
        agent_run.enqueue('injected after rich tail')
        async for _ in agent_run:
            pass

    assert agent_run.result is not None
    # `all_messages()` keeps the un-merged view (drain's request is a separate
    # `ModelRequest` after the rich tail).
    assert agent_run.result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='original prompt', timestamp=IsDatetime())]),
            ModelResponse(
                parts=[ToolCallPart(tool_name='hint', args='{}', tool_call_id='call-1')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name='hint', content='ok', tool_call_id='call-1', timestamp=IsDatetime()),
                    UserPromptPart(content='follow-up question', timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='injected after rich tail', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    # And the wire-level view: the rich tail and the drained request merged into one
    # `ModelRequest`, with `ToolReturnPart` first and the user-prompt parts in arrival
    # order (so the enqueued content lands at the end, not interleaved).
    assert len(captured_wire_messages) == 1
    assert captured_wire_messages[0] == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='original prompt', timestamp=IsDatetime())]),
            ModelResponse(
                parts=[ToolCallPart(tool_name='hint', args='{}', tool_call_id='call-1')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name='hint', content='ok', tool_call_id='call-1', timestamp=IsDatetime()),
                    UserPromptPart(content='follow-up question', timestamp=IsDatetime()),
                    UserPromptPart(content='injected after rich tail', timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
            ),
        ]
    )


async def test_enqueue_asap_drains_at_end_if_arrived_during_final_step():
    """`'asap'` arriving during the final step (after its `before_model_request` drain) still gets delivered.

    Simulates the background-tools pattern: a long-running task completes *during* what
    would have been the model's final response. The enqueue happens after the step's
    `before_model_request` drain has already fired, so the message can only be picked up
    by the end-of-run drain (matching pi-mono's drain-on-end). Without this fallback the
    message would be lost. `'asap'` semantically means "deliver at the earliest opportunity"
    — including redirecting if the agent would otherwise terminate before another call.
    """
    call_count = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(
                parts=[TextPart(content='would-have-ended')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        return ModelResponse(
            parts=[TextPart(content='final after late asap')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    @dataclass
    class BackgroundTaskCap(AbstractCapability[Any]):
        """Simulates a background task that completes mid-model-response on the first call only."""

        fired: bool = False

        async def after_model_request(
            self,
            ctx: RunContext[Any],
            *,
            request_context: ModelRequestContext,
            response: ModelResponse,
        ) -> ModelResponse:
            if not self.fired:
                ctx.enqueue('background task result', priority='asap')
                self.fired = True
            return response

    agent = Agent(FunctionModel(model_fn), capabilities=[BackgroundTaskCap()])

    result = await agent.run('Hello')
    assert result.output == 'final after late asap'
    assert call_count == 2
    # The 'asap' message landed in its own ModelRequest before the final response,
    # not lost despite the agent producing a no-tool-call response.
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='would-have-ended')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='background task result', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='final after late asap')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_enqueue_when_idle_drains_after_leftover_asap():
    """If both `'asap'` and `'when_idle'` are queued at end-of-run, `'asap'` drains first."""

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # Only fire enqueues once.
        already_enqueued = any(
            isinstance(p, UserPromptPart) and p.content in ('A', 'B')
            for msg in messages
            if isinstance(msg, ModelRequest)
            for p in msg.parts
        )
        # If we've already seen our injected messages, just terminate.
        if already_enqueued:
            return ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        return ModelResponse(
            parts=[ToolCallPart(tool_name='inject', args='{}')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    agent = Agent(FunctionModel(model_fn))

    @agent.tool
    def inject(ctx: RunContext[object]) -> str:
        ctx.enqueue('B', priority='when_idle')
        ctx.enqueue('A', priority='asap')
        return 'ok'

    result = await agent.run('Hello')
    # Both A and B should appear in history. `'asap'` (A) drains in `before_model_request`
    # before the second call. `'when_idle'` (B) drains at end-of-run when the second
    # response has no tool calls.
    requests_with_injected = [
        msg
        for msg in result.all_messages()
        if isinstance(msg, ModelRequest)
        and any(isinstance(p, UserPromptPart) and p.content in ('A', 'B') for p in msg.parts)
    ]
    contents = [
        [p.content for p in r.parts if isinstance(p, UserPromptPart) and p.content in ('A', 'B')]
        for r in requests_with_injected
    ]
    assert contents == [['A'], ['B']], f'expected A before B in separate requests, got {contents}'


async def test_enqueue_priorities_stay_separate_when_both_drain_at_end_of_run():
    """When both `'asap'` and `'when_idle'` parts-style payloads drain together at end-of-run,
    they land in separate `ModelRequest`s — the priority split stays visible in history.

    Reaches the case Devin flagged: a tool enqueues `'when_idle'` (which sits until
    end-of-run), and a capability `after_model_request` hook enqueues `'asap'` during the
    final step (after that step's `before_model_request` drain has already fired). Both
    arrive at `after_node_run`. Without the per-priority split they'd merge into one
    synthesized request, blurring the priority distinction in the persisted history.
    On the wire `_clean_message_history` still merges them for the model.
    """
    call_count = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(
                parts=[ToolCallPart(tool_name='inject', args='{}')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        if call_count == 2:
            return ModelResponse(
                parts=[TextPart(content='would-have-ended')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        return ModelResponse(
            parts=[TextPart(content='final')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    @dataclass
    class LateAsapCap(AbstractCapability[Any]):
        """Enqueues an `'asap'` message during `after_model_request` of the no-tool-call step.

        Fires after the step's `before_model_request` drain, so the message can only be
        delivered via the end-of-run drain in `after_node_run`.
        """

        fired: bool = False

        async def after_model_request(
            self,
            ctx: RunContext[Any],
            *,
            request_context: ModelRequestContext,
            response: ModelResponse,
        ) -> ModelResponse:
            if not self.fired and any(
                isinstance(p, TextPart) and p.content == 'would-have-ended' for p in response.parts
            ):
                ctx.enqueue('asap-from-cap')
                self.fired = True
            return response

    agent = Agent(FunctionModel(model_fn), capabilities=[LateAsapCap()])

    @agent.tool
    def inject(ctx: RunContext[object]) -> str:
        ctx.enqueue('when-idle-from-tool', priority='when_idle')
        return 'ok'

    result = await agent.run('Hello')
    assert result.output == 'final'

    # Find the two end-of-run drained requests: one with the 'asap' content, one with 'when_idle'.
    drained = [
        msg
        for msg in result.all_messages()
        if isinstance(msg, ModelRequest)
        and any(
            isinstance(p, UserPromptPart) and p.content in ('asap-from-cap', 'when-idle-from-tool') for p in msg.parts
        )
    ]
    contents = [
        next(
            p.content
            for p in r.parts
            if isinstance(p, UserPromptPart) and p.content in ('asap-from-cap', 'when-idle-from-tool')
        )
        for r in drained
    ]
    assert contents == ['asap-from-cap', 'when-idle-from-tool'], (
        f'asap and when_idle should land in separate ModelRequests with asap first, got {contents}'
    )
    # Each priority bucket got its own ModelRequest (not merged into one).
    assert all(len([p for p in r.parts if isinstance(p, UserPromptPart)]) == 1 for r in drained)
