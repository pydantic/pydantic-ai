"""Tests for the `ProcessEventStream` capability.

Split out of `test_capabilities.py`, which had grown past the repository's file-size limit.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast

import anyio
import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, ProcessEventStream
from pydantic_ai.exceptions import SkipModelRequest, UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    ModelMessage,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
)
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters, ModelSelectionContext
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings as _ModelSettings
from pydantic_ai.usage import RunUsage
from pydantic_graph import End

from ._inline_snapshot import snapshot
from .capability_models import (
    simple_model_function,
    simple_stream_function,
    tool_calling_model,
    tool_calling_stream_function,
)

pytestmark = [
    pytest.mark.anyio,
]


class _RequestOnlyModel(Model):
    @property
    def model_name(self) -> str:
        return 'request-only'

    @property
    def system(self) -> str:
        return 'test'

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: _ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content='hi')], model_name='request-only')  # pragma: no cover


class TestProcessEventStream:
    """Tests for the ProcessEventStream capability."""

    async def test_handler_receives_events(self):
        """Handler registered via capability receives events from model streaming."""
        handler_events: list[AgentStreamEvent] = []

        async def handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                handler_events.append(event)

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[ProcessEventStream(handler=handler)],
        )

        # No event_stream_handler arg — capability should drive streaming
        result = await agent.run('hello')
        assert result.output is not None
        assert any(isinstance(e, PartStartEvent) for e in handler_events)

    async def test_multiple_handlers_and_param_all_observe(self):
        """Multiple ProcessEventStream capabilities and an explicit event_stream_handler all see the same events."""
        cap1_events: list[AgentStreamEvent] = []
        cap2_events: list[AgentStreamEvent] = []
        param_events: list[AgentStreamEvent] = []

        async def cap1_handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                cap1_events.append(event)

        async def cap2_handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                cap2_events.append(event)

        async def param_handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                param_events.append(event)

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[ProcessEventStream(handler=cap1_handler), ProcessEventStream(handler=cap2_handler)],
        )

        await agent.run('hello', event_stream_handler=param_handler)
        assert len(cap1_events) > 0
        assert cap1_events == cap2_events == param_events

    async def test_handler_sees_events_after_inner_wrappers(self):
        """Events passed to the handler go through inner wrap_run_event_stream wrappers."""
        transformed_calls: list[AgentStreamEvent] = []
        handler_events: list[AgentStreamEvent] = []

        @dataclass
        class InnerWrapper(AbstractCapability[Any]):
            async def wrap_run_event_stream(
                self,
                ctx: RunContext[Any],
                *,
                stream: AsyncIterable[AgentStreamEvent],
            ) -> AsyncIterable[AgentStreamEvent]:
                async for event in stream:
                    transformed_calls.append(event)
                    yield event

        async def handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                handler_events.append(event)

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[ProcessEventStream(handler=handler), InnerWrapper()],
        )

        await agent.run('hello')
        assert handler_events == transformed_calls
        assert len(handler_events) > 0

    async def test_transformer_handler_replaces_stream(self):
        """An async-generator handler transforms the stream seen by downstream wrappers and the param handler."""
        downstream_events: list[AgentStreamEvent] = []

        async def transformer(
            _ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]
        ) -> AsyncIterator[AgentStreamEvent]:
            async for event in stream:
                if isinstance(event, PartStartEvent):
                    # Drop PartStart events — downstream should never see them.
                    continue
                yield event

        async def param_handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                downstream_events.append(event)

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[ProcessEventStream(handler=transformer)],
        )

        await agent.run('hello', event_stream_handler=param_handler)
        assert len(downstream_events) > 0
        assert not any(isinstance(e, PartStartEvent) for e in downstream_events)

    async def test_callable_instance_processor(self):
        """A callable-class processor (not a plain async-generator function) is detected via its return type."""
        captured: list[AgentStreamEvent] = []

        class Transformer:
            async def __call__(
                self, _ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]
            ) -> AsyncIterator[AgentStreamEvent]:
                async for event in stream:
                    captured.append(event)
                    yield event

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[ProcessEventStream(handler=Transformer())],
        )
        await agent.run('hello')
        assert any(isinstance(e, PartStartEvent) for e in captured)

    async def test_observer_bailout_does_not_break_downstream(self):
        """If an observer stops iterating early, downstream consumers still see all events."""
        received_by_observer: list[AgentStreamEvent] = []
        received_downstream: list[AgentStreamEvent] = []

        async def bail_after_first(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                received_by_observer.append(event)
                return

        async def downstream(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                received_downstream.append(event)

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[ProcessEventStream(handler=bail_after_first)],
        )
        await agent.run('hello', event_stream_handler=downstream)
        assert len(received_by_observer) == 1
        assert len(received_downstream) > 1

    async def test_failing_stream_tears_down_the_handler(self):
        """When the stream being wrapped fails, the observer is cancelled rather than left parked.

        The handler runs in its own task waiting on the teed stream, so an error that stops the
        wrapper mid-flight has to tear it down; otherwise it lingers, blocked on `receive`. Asserting
        the handler didn't *finish* would prove nothing (it never finishes either way), so this
        records how it ended.
        """
        state = 'not started'

        async def observer(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            nonlocal state
            try:
                state = 'receiving'
                async for _event in stream:
                    pass
            except asyncio.CancelledError:
                state = 'cancelled'
                raise

        async def exploding_stream() -> AsyncIterator[AgentStreamEvent]:
            yield PartStartEvent(index=0, part=TextPart(content='hi'))
            raise RuntimeError('stream boom')

        capability = ProcessEventStream[None](handler=observer)
        run_ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())

        with pytest.raises(RuntimeError, match='stream boom'):
            async for _event in capability.wrap_run_event_stream(run_ctx, stream=exploding_stream()):
                pass

        assert state == snapshot('cancelled')

    async def test_abandoned_model_request_stream_tears_down_the_handler(self):
        """Walking away from a node stream closes the capability chain instead of stranding its observer."""
        state = 'not started'
        torn_down = anyio.Event()

        async def observer(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            nonlocal state
            try:
                state = 'receiving'
                async for _event in stream:
                    pass
            except asyncio.CancelledError:
                state = 'cancelled'
                torn_down.set()
                raise

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[ProcessEventStream(handler=observer)],
        )

        async with agent.iter('hello') as agent_run:
            node = agent_run.next_node
            while not Agent.is_model_request_node(node):
                assert not isinstance(node, End)
                node = await agent_run.next(node)
            async with node.stream(agent_run.ctx) as stream:
                async for _event in stream:  # pragma: no branch
                    break

        # Closing the chain reaches the observer via the wrapping capability's own teardown, which
        # asyncio finalizes a tick later, so wait for it rather than asserting on the same tick.
        with anyio.fail_after(5):
            await torn_down.wait()
        assert state == snapshot('cancelled')

    async def test_referenced_inner_wrapper_does_not_pin_observer(self):
        """Closing the chain must propagate through a wrapper that retains its input stream."""
        torn_down = anyio.Event()
        held_streams: list[AsyncIterable[AgentStreamEvent]] = []

        async def observer(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            try:
                async for _event in stream:
                    pass
            except asyncio.CancelledError:
                torn_down.set()
                raise

        @dataclass
        class Stasher(AbstractCapability[Any]):
            async def wrap_run_event_stream(
                self,
                ctx: RunContext[Any],
                *,
                stream: AsyncIterable[AgentStreamEvent],
            ) -> AsyncIterable[AgentStreamEvent]:
                held_streams.append(stream)
                async for event in stream:  # pragma: no branch
                    yield event

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[Stasher(), ProcessEventStream(handler=observer)],
        )

        async with agent.iter('hello') as agent_run:
            node = agent_run.next_node
            while not Agent.is_model_request_node(node):
                assert not isinstance(node, End)
                node = await agent_run.next(node)
            async with node.stream(agent_run.ctx) as stream:
                await anext(aiter(stream))

        assert held_streams
        with anyio.fail_after(5):
            await torn_down.wait()

    @pytest.mark.parametrize('consumer_error', [False, True])
    async def test_abandoned_call_tools_stream_tears_down_the_handler(self, consumer_error: bool):
        """Leaving a response-handling stream closes its memoized capability chain."""
        states: list[str] = []
        torn_down = anyio.Event()

        async def observer(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            invocation = len(states)
            states.append('receiving')
            try:
                async for _event in stream:
                    pass
                states[invocation] = 'finished'
            except asyncio.CancelledError:
                states[invocation] = 'cancelled'
                raise
            finally:
                if invocation == 1:
                    torn_down.set()

        agent = Agent(
            FunctionModel(tool_calling_model, stream_function=tool_calling_stream_function),
            capabilities=[ProcessEventStream(handler=observer)],
        )

        @agent.tool_plain
        def get_thing() -> str:
            return 'thing'

        # Held so refcounting can't finalize the abandoned chain for us -- teardown has to come from
        # the node closing it, not from the garbage collector.
        held_streams: list[AsyncIterator[AgentStreamEvent]] = []

        async def consume() -> None:
            async with agent.iter('hello') as agent_run:
                async for node in agent_run:  # pragma: no branch
                    if Agent.is_call_tools_node(node):
                        async with node.stream(agent_run.ctx) as stream:
                            held_streams.append(stream)
                            async for _event in stream:  # pragma: no branch
                                if consumer_error:
                                    raise RuntimeError('consumer exploded')
                                break
                        break

        if consumer_error:
            with pytest.raises(RuntimeError, match='consumer exploded'):
                await consume()
        else:
            await consume()

        # Closing the chain reaches the observer via the wrapping capability's own teardown, which
        # asyncio finalizes a tick later, so wait for it rather than asserting on the same tick.
        with anyio.fail_after(5):
            await torn_down.wait()
        assert held_streams
        assert states[1] == snapshot('cancelled' if consumer_error else 'finished')

    async def test_abandoned_stream_text_does_not_deadlock_run_stream(self):
        """Walking away from `stream_text()` mid-stream must not wedge the node's teardown.

        `stream_text()` debounces through `group_by_temporal`, which parks a prefetch task inside
        `anext()` on the shared iterator — holding the lock `aclose_events()` wants. Waiting for it
        would never return, because nothing is left to finish that pull.
        """

        async def stalled_stream(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
            yield 'first'
            await asyncio.sleep(30)

        agent = Agent(FunctionModel(stream_function=stalled_stream))
        # Held so refcounting can't finalize the abandoned generator for us: the parked prefetch has
        # to still be holding the lock when the node closes its stream.
        held: list[AsyncIterator[str]] = []

        with anyio.fail_after(5):
            async with agent.run_stream('hello') as result:
                deltas = result.stream_text(delta=True)
                held.append(deltas)
                async for _text in deltas:  # pragma: no branch
                    break

    async def test_post_model_request_error_tears_down_the_handler(self):
        """An error raised after the model stream completes still closes the capability chain."""
        state = 'not started'
        torn_down = anyio.Event()

        async def observer(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            nonlocal state
            try:
                state = 'receiving'
                async for _event in stream:
                    pass
            except asyncio.CancelledError:
                state = 'cancelled'
                torn_down.set()
                raise

        @dataclass
        class PostProcessError(AbstractCapability[Any]):
            async def wrap_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                handler: Callable[[ModelRequestContext], Awaitable[ModelResponse]],
            ) -> ModelResponse:
                await handler(request_context)
                raise RuntimeError('post-processing exploded')

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[ProcessEventStream(handler=observer), PostProcessError()],
        )

        with pytest.raises(RuntimeError, match='post-processing exploded'):
            async with agent.iter('hello') as agent_run:
                node = agent_run.next_node
                while not Agent.is_model_request_node(node):
                    assert not isinstance(node, End)
                    node = await agent_run.next(node)
                # Never exits normally: the capability's error surfaces from the node's teardown.
                async with node.stream(agent_run.ctx) as stream:  # pragma: no branch
                    async for _event in stream:  # pragma: no branch
                        break

        with anyio.fail_after(5):
            await torn_down.wait()
        assert state == snapshot('cancelled')

    async def test_abandoned_short_circuited_stream_tears_down_the_handler(self):
        """A short-circuited request replays its response as events, so that stream needs closing too."""
        state = 'not started'
        torn_down = anyio.Event()

        async def observer(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            nonlocal state
            try:
                state = 'receiving'
                async for _event in stream:
                    pass
            except asyncio.CancelledError:
                state = 'cancelled'
                torn_down.set()
                raise

        @dataclass
        class CachedResponse(AbstractCapability[Any]):
            async def wrap_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                handler: Callable[[ModelRequestContext], Awaitable[ModelResponse]],
            ) -> ModelResponse:
                return ModelResponse(parts=[TextPart(content='cached'), TextPart(content='still cached')])

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[ProcessEventStream(handler=observer), CachedResponse()],
        )

        async with agent.iter('hello') as agent_run:
            node = agent_run.next_node
            while not Agent.is_model_request_node(node):
                assert not isinstance(node, End)
                node = await agent_run.next(node)
            async with node.stream(agent_run.ctx) as stream:
                async for _event in stream:  # pragma: no branch
                    break

        with anyio.fail_after(5):
            await torn_down.wait()
        assert state == snapshot('cancelled')

    async def test_raised_short_circuited_stream_tears_down_the_handler(self):
        """A consumer error while replaying a short-circuited response still closes the event stream."""
        state = 'not started'
        torn_down = anyio.Event()

        async def observer(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            nonlocal state
            try:
                state = 'receiving'
                async for _event in stream:
                    pass
            except asyncio.CancelledError:
                state = 'cancelled'
                torn_down.set()
                raise

        @dataclass
        class CachedResponse(AbstractCapability[Any]):
            async def wrap_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                handler: Callable[[ModelRequestContext], Awaitable[ModelResponse]],
            ) -> ModelResponse:
                return ModelResponse(parts=[TextPart(content='cached'), TextPart(content='still cached')])

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[ProcessEventStream(handler=observer), CachedResponse()],
        )

        with pytest.raises(RuntimeError, match='consumer exploded'):
            async with agent.iter('hello') as agent_run:
                node = agent_run.next_node
                while not Agent.is_model_request_node(node):
                    assert not isinstance(node, End)
                    node = await agent_run.next(node)
                # The consumer error is expected to exit both loops and context managers.
                async with node.stream(agent_run.ctx) as stream:  # pragma: no branch
                    async for _event in stream:  # pragma: no branch
                        raise RuntimeError('consumer exploded')

        with anyio.fail_after(5):
            await torn_down.wait()
        assert state == snapshot('cancelled')

    async def test_cancelled_consumer_closes_stalled_source(self):
        """Cancellation drains the in-flight upstream pull before it reaches the consumer."""
        pulling = anyio.Event()
        closed = anyio.Event()

        async def observer(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for _event in stream:
                pass

        async def stalled_stream() -> AsyncIterator[AgentStreamEvent]:
            try:
                yield PartStartEvent(index=0, part=TextPart(content='hi'))
                pulling.set()
                await anyio.sleep_forever()
            finally:
                closed.set()

        capability = ProcessEventStream[None](handler=observer)
        run_ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())

        async def consume() -> None:
            async for _event in capability.wrap_run_event_stream(run_ctx, stream=stalled_stream()):
                pass

        consumer = asyncio.create_task(consume())
        await pulling.wait()
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
        assert closed.is_set()

    async def test_failing_observer_interrupts_stalled_stream(self):
        """An observer failure propagates without waiting for a stalled upstream pull."""

        class ObserverError(Exception):
            pass

        async def observer(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for _event in stream:  # pragma: no branch
                raise ObserverError('observer boom')

        async def stalled_stream() -> AsyncIterator[AgentStreamEvent]:
            yield PartStartEvent(index=0, part=TextPart(content='hi'))
            await asyncio.sleep(30)

        capability = ProcessEventStream[None](handler=observer)
        run_ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())

        async def consume() -> None:
            async for _event in capability.wrap_run_event_stream(run_ctx, stream=stalled_stream()):
                pass

        with pytest.raises(ObserverError, match='observer boom'):
            await asyncio.wait_for(consume(), timeout=5)

    async def test_failing_observer_interrupts_a_stream_that_cannot_be_closed(self):
        """Interrupting the source is best-effort: a plain `AsyncIterator` has no `aclose()` to call."""

        class ObserverError(Exception):
            pass

        class UncloseableStream:
            """What a custom capability may hand down: an `AsyncIterator` that isn't a generator."""

            def __init__(self) -> None:
                self.sent = False

            def __aiter__(self) -> UncloseableStream:
                return self

            async def __anext__(self) -> AgentStreamEvent:
                if self.sent:
                    await asyncio.sleep(30)
                self.sent = True
                return PartStartEvent(index=0, part=TextPart(content='hi'))

        async def observer(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for _event in stream:  # pragma: no branch
                raise ObserverError('observer boom')

        capability = ProcessEventStream[None](handler=observer)
        run_ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())

        async def consume() -> None:
            async for _event in capability.wrap_run_event_stream(run_ctx, stream=UncloseableStream()):
                pass

        with pytest.raises(ObserverError, match='observer boom'):
            await asyncio.wait_for(consume(), timeout=5)

    async def test_observer_closing_its_own_stream_does_not_break_the_run(self):
        """The observer owns the stream handed to it; closing it just stops its own delivery."""
        closed = anyio.Event()
        resume = anyio.Event()
        seen: list[AgentStreamEvent] = []

        async def observer(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:  # pragma: no branch
                seen.append(event)
                break
            await cast('Any', stream).aclose()
            closed.set()
            # Stay alive past the next send, so it's the closed stream that ends delivery.
            await resume.wait()

        async def source() -> AsyncIterator[AgentStreamEvent]:
            yield PartStartEvent(index=0, part=TextPart(content='one'))
            await closed.wait()
            yield PartStartEvent(index=1, part=TextPart(content='two'))
            resume.set()
            yield PartStartEvent(index=2, part=TextPart(content='three'))

        capability = ProcessEventStream[None](handler=observer)
        run_ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())

        downstream = [event async for event in capability.wrap_run_event_stream(run_ctx, stream=source())]

        assert len(downstream) == snapshot(3)
        assert len(seen) == snapshot(1)

    async def test_not_spec_serializable(self):
        """ProcessEventStream holds a callable so it cannot participate in spec-based construction."""
        assert ProcessEventStream.get_serialization_name() is None

    @pytest.mark.parametrize('drive', ['run', 'bare_async_for', 'next', 'manual_stream'])
    async def test_handler_fires_under_every_drive_mode(self, drive: str):
        """Every way of driving a run delivers the same events to the capability's handler.

        `wrap_run_event_stream` used to be applied by `run()`/`run_stream()` rather than by the
        node stream primitives, so `agent.iter()` silently skipped the handler no matter how the
        caller advanced the run.
        """
        handler_events: list[AgentStreamEvent] = []

        async def handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                handler_events.append(event)

        agent = Agent(
            FunctionModel(tool_calling_model, stream_function=tool_calling_stream_function),
            capabilities=[ProcessEventStream(handler=handler)],
        )

        @agent.tool_plain
        def get_thing() -> str:
            return 'thing'

        if drive == 'run':
            await agent.run('hello')
        else:
            async with agent.iter('hello') as agent_run:
                if drive == 'bare_async_for':
                    async for _node in agent_run:
                        pass
                else:
                    node = agent_run.next_node
                    while not isinstance(node, End):
                        if drive == 'manual_stream' and (
                            Agent.is_model_request_node(node) or Agent.is_call_tools_node(node)
                        ):
                            # Streaming the node by hand must not stop `next()` from advancing it,
                            # and must not stream it a second time.
                            async with node.stream(agent_run.ctx) as stream:
                                async for _event in stream:
                                    pass
                        node = await agent_run.next(node)

        assert [type(event).__name__ for event in handler_events] == snapshot(
            [
                'PartStartEvent',
                'PartEndEvent',
                'FunctionToolCallEvent',
                'FunctionToolResultEvent',
                'PartStartEvent',
                'FinalResultEvent',
                'PartEndEvent',
            ]
        )

    @pytest.mark.parametrize('drive', ['run', 'bare_async_for', 'next', 'manual_stream'])
    async def test_handler_invoked_once_per_streamed_node(self, drive: str):
        """The handler is invoked once per streamed node, not once per `stream()` entry.

        `CallToolsNode.run()` enters `stream()` itself, so the node's stream is entered a second
        time after it has been consumed. Rebuilding the capability chain there would re-run every
        handler over an exhausted stream, duplicating any setup it does outside its own iteration —
        invisible in the event list (the second pass yields nothing) but not in its side effects.
        """
        invocations: list[int] = []

        async def handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            count = 0
            async for _event in stream:
                count += 1
            invocations.append(count)

        agent = Agent(
            FunctionModel(tool_calling_model, stream_function=tool_calling_stream_function),
            capabilities=[ProcessEventStream(handler=handler)],
        )

        @agent.tool_plain
        def get_thing() -> str:
            return 'thing'

        if drive == 'run':
            await agent.run('hello')
        else:
            async with agent.iter('hello') as agent_run:
                if drive == 'bare_async_for':
                    async for _node in agent_run:
                        pass
                else:
                    node = agent_run.next_node
                    while not isinstance(node, End):
                        if drive == 'manual_stream' and (
                            Agent.is_model_request_node(node) or Agent.is_call_tools_node(node)
                        ):
                            async with node.stream(agent_run.ctx) as stream:
                                async for _event in stream:
                                    pass
                        node = await agent_run.next(node)

        # One entry per streamed node: two model requests and the two response-handling nodes
        # between and after them. No trailing zero-event entries from a re-entered stream.
        assert invocations == snapshot([2, 2, 3, 0])

    async def test_streamed_result_can_be_consumed_in_another_task(self):
        """A `run_stream()` result stays usable when consumed from a different task.

        The wrapped stream is memoized and shared, so it can be resumed by whichever task pulls next.
        Holding an `anyio` task group open across the handler's yields would bind the generator to the
        task that created it, and exiting that scope elsewhere raises `RuntimeError: Attempted to exit
        a cancel scope that isn't the current task's current cancel scope` — turning an ordinary
        `asyncio.create_task(result.get_output())` into a crash.
        """
        seen: list[AgentStreamEvent] = []

        async def observer(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                seen.append(event)

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[ProcessEventStream(handler=observer)],
        )

        async with agent.run_stream('hello') as result:
            output = await asyncio.create_task(result.get_output())

        assert output == snapshot('streamed response')
        assert seen != []

    async def test_processor_shapes_streamed_text_but_not_the_output(self):
        """A processor reaches `stream_text()` but not the run's output.

        Pins the boundary the docs promise: the `ModelResponse` accumulates from the raw model stream
        before a processor sees the events, so rewriting a delta changes the streamed text without
        changing what the run actually returns.
        """

        async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            yield 'hello '
            yield 'world'

        async def rewriter(
            _ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]
        ) -> AsyncIterator[AgentStreamEvent]:
            async for event in stream:
                if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                    yield PartDeltaEvent(index=event.index, delta=TextPartDelta(content_delta='XXX'))
                else:
                    yield event

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=stream_fn),
            capabilities=[ProcessEventStream(handler=rewriter)],
        )

        async with agent.run_stream('hello') as result:
            streamed = [text async for text in result.stream_text(delta=True)]
            output = await result.get_output()

        assert streamed == snapshot(['hello XXX'])
        assert output == snapshot('hello world')

    async def test_non_streaming_model_raises_a_clear_error(self):
        """A model that can't stream fails with an actionable error, not an opaque `NotImplementedError`.

        A capability with this hook needs the model to stream so there are events to observe, so
        `next()` starts streaming — which a model implementing only `request()` cannot do.
        """

        async def observer(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for _event in stream:  # pragma: no cover
                pass

        model = _RequestOnlyModel()
        assert (model.model_name, model.system) == snapshot(('request-only', 'test'))

        agent = Agent(model, capabilities=[ProcessEventStream(handler=observer)])

        with pytest.raises(UserError, match='does not support streamed requests'):
            async with agent.iter('hello') as agent_run:
                node = agent_run.next_node
                while not isinstance(node, End):  # pragma: no branch — the error ends the loop
                    node = await agent_run.next(node)

    async def test_non_streaming_model_error_does_not_blame_a_capability(self):
        """The guard also covers plain streamed runs, so it must not assert a cause that isn't there.

        `run_stream()` needs the model to stream on its own account. Saying "a capability registers a
        `wrap_run_event_stream` hook" would be simply false here, and sends the reader looking for a
        capability they never added.
        """

        model = _RequestOnlyModel()
        agent = Agent(model)

        with pytest.raises(UserError) as exc_info:
            async with agent.run_stream('hello') as result:
                await result.get_output()  # pragma: no cover — the guard raises on entry

        assert 'does not support streamed requests' in str(exc_info.value)
        assert 'either because the run itself is streamed' in str(exc_info.value)

    async def test_non_streaming_model_can_be_short_circuited_during_streaming(self):
        """A cached response does not require the configured model to support streaming."""

        @dataclass
        class CachedResponse(AbstractCapability[Any]):
            async def wrap_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                handler: Callable[[ModelRequestContext], Awaitable[ModelResponse]],
            ) -> ModelResponse:
                return ModelResponse(parts=[TextPart(content='cached')])

        model = _RequestOnlyModel()
        agent = Agent(model, capabilities=[CachedResponse()])

        async with agent.run_stream('hello') as result:
            output = await result.get_output()

        assert output == snapshot('cached')

    async def test_non_streaming_model_can_be_skipped_during_streaming(self):
        """Skipping a request does not require the configured model to support streaming."""

        @dataclass
        class SkipCapability(AbstractCapability[Any]):
            async def wrap_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                handler: Callable[[ModelRequestContext], Awaitable[ModelResponse]],
            ) -> ModelResponse:
                raise SkipModelRequest(ModelResponse(parts=[TextPart(content='skipped')]))

        agent = Agent(_RequestOnlyModel(), capabilities=[SkipCapability()])

        async with agent.run_stream('hello') as result:
            output = await result.get_output()

        assert output == snapshot('skipped')

    async def test_wrap_model_request_can_replace_non_streaming_model(self):
        """The streaming guard checks the model selected by request middleware."""
        selected = FunctionModel(simple_model_function, stream_function=simple_stream_function)

        @dataclass
        class ReplaceModel(AbstractCapability[Any]):
            async def wrap_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                handler: Callable[[ModelRequestContext], Awaitable[ModelResponse]],
            ) -> ModelResponse:
                request_context.model = selected
                return await handler(request_context)

        agent = Agent(_RequestOnlyModel(), capabilities=[ReplaceModel()])

        async with agent.run_stream('hello') as result:
            output = await result.get_output()

        assert output == snapshot('streamed response')

    async def test_model_selector_can_replace_non_streaming_model(self):
        """The streaming guard checks the model selected for the current step."""

        selected = FunctionModel(simple_model_function, stream_function=simple_stream_function)

        @dataclass
        class SelectStreamingModel(AbstractCapability[Any]):
            def get_model(self) -> Callable[[ModelSelectionContext[Any]], Model]:
                return lambda _ctx: selected

        handler_events: list[AgentStreamEvent] = []

        async def handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                handler_events.append(event)

        unusable = _RequestOnlyModel()
        agent = Agent(
            unusable,
            capabilities=[SelectStreamingModel(), ProcessEventStream(handler=handler)],
        )

        result = await agent.run('hello')

        assert result.output is not None
        assert handler_events

    async def test_processor_transforms_events_seen_by_manual_stream(self):
        """A processor's transformations reach a caller streaming a node by hand under `iter()`.

        The processor form replaces the stream for downstream consumers, so a dropped event must
        not surface to the `node.stream()` consumer either.
        """

        async def drop_part_starts(
            _ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]
        ) -> AsyncIterator[AgentStreamEvent]:
            async for event in stream:
                if not isinstance(event, PartStartEvent):
                    yield event

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[ProcessEventStream(handler=drop_part_starts)],
        )

        seen: list[AgentStreamEvent] = []
        async with agent.iter('hello') as agent_run:
            node = agent_run.next_node
            while not isinstance(node, End):
                if Agent.is_model_request_node(node):
                    async with node.stream(agent_run.ctx) as stream:
                        async for event in stream:
                            seen.append(event)
                node = await agent_run.next(node)

        assert seen != []
        assert not any(isinstance(event, PartStartEvent) for event in seen)

    async def test_next_does_not_force_streaming_without_event_hooks(self):
        """`next()` only streams when a capability registers event-stream hooks.

        Without them there are no events to deliver, so the run keeps using non-streamed model
        requests — `FunctionModel` without a `stream_function` would fail if it streamed.
        """
        agent = Agent(FunctionModel(simple_model_function))

        async with agent.iter('hello') as agent_run:
            node = agent_run.next_node
            while not isinstance(node, End):
                node = await agent_run.next(node)

        assert agent_run.result is not None
