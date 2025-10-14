"""Base classes for UI event stream protocols.

This module provides abstract base classes for implementing UI event stream adapters
that transform Pydantic AI agent events into protocol-specific events (e.g., AG-UI, Vercel AI).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Generic, Literal, TypeVar
from uuid import uuid4

from ..messages import (
    AgentStreamEvent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    FilePart,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelResponsePart,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)
from ..run import AgentRunResult, AgentRunResultEvent
from ..tools import AgentDepsT

__all__ = [
    'BaseEventStream',
]

EventT = TypeVar('EventT')
"""Type variable for protocol-specific event types."""

RunRequestT = TypeVar('RunRequestT')
"""Type variable for request types."""

SourceEvent = AgentStreamEvent | AgentRunResultEvent


@dataclass
class BaseEventStream(ABC, Generic[RunRequestT, EventT, AgentDepsT]):
    """TODO (DouwM): Docstring."""

    request: RunRequestT
    result: AgentRunResult | None = None

    message_id: str = field(default_factory=lambda: str(uuid4()))

    def new_message_id(self) -> str:
        """Generate and store a new message ID.

        Returns:
            A new UUID-based message ID.
        """
        self.message_id = str(uuid4())
        return self.message_id

    @abstractmethod
    def encode_event(self, event: EventT, accept: str | None = None) -> str:
        """Encode an event as a string.

        Args:
            event: The event to encode.
            accept: The accept header value for encoding format.
        """
        raise NotImplementedError

    async def encode_stream(self, stream: AsyncIterator[EventT], accept: str | None = None) -> AsyncIterator[str]:
        """Encode a stream of events as SSE strings.

        Args:
            stream: The stream of events to encode.
            accept: The accept header value for encoding format.
        """
        async for event in stream:
            print(event)
            yield self.encode_event(event, accept)

    async def handle_stream(self, stream: AsyncIterator[SourceEvent]) -> AsyncIterator[EventT]:  # noqa: C901
        """Handle a stream of agent events.

        Args:
            stream: The stream of agent events to handle.

        Yields:
            Protocol-specific events.
        """
        async for e in self.before_stream():
            yield e

        part: ModelResponsePart | None = None
        turn: Literal['request', 'response'] | None = None
        try:
            async for event in stream:
                # TODO (DouweM): Introduce PartEndEvent, possible MessageStartEvent, MessageEndEvent with ModelRequest/Response
                previous_part = part
                next_turn = turn
                next_part = part
                if isinstance(event, PartStartEvent):
                    next_turn = 'request'
                    next_part = event.part
                elif isinstance(event, FunctionToolCallEvent):
                    next_turn = 'response'
                    next_part = None
                elif isinstance(event, AgentRunResultEvent):
                    next_turn = None
                    next_part = None

                if next_part != part:
                    if part:
                        async for e in self.handle_part_end(part, next_part):
                            yield e

                    part = next_part

                if turn != next_turn:
                    if turn == 'request':
                        async for e in self.after_request():
                            yield e
                    elif turn == 'response':
                        async for e in self.after_response():
                            yield e

                    turn = next_turn

                    if turn == 'request':
                        async for e in self.before_request():
                            yield e
                    elif turn == 'response':
                        async for e in self.before_response():
                            yield e

                async for e in self.handle_event(event, previous_part):
                    yield e
        except Exception as e:
            async for e in self.on_error(e):
                yield e
        else:
            async for e in self.after_stream():
                yield e

    async def handle_event(
        self, event: SourceEvent, previous_part: ModelResponsePart | None = None
    ) -> AsyncIterator[EventT]:
        """Transform a Pydantic AI agent event into protocol-specific events.

        This method dispatches to specific `handle_*` methods based on event and part type.
        Subclasses should implement the individual handler methods rather than overriding this.

        Args:
            event: The Pydantic AI agent event to transform.
            previous_part: The previous part.

        Yields:
            Protocol-specific events.
        """
        async for e in self.before_event(event):
            yield e

        match event:
            case PartStartEvent():
                async for e in self.handle_part_start(event, previous_part):
                    yield e
            case PartDeltaEvent():
                async for e in self.handle_part_delta(event):
                    yield e
            case FinalResultEvent():
                async for e in self.handle_final_result(event):
                    yield e
            case FunctionToolCallEvent():
                async for e in self.handle_function_tool_call(event):
                    yield e
            case FunctionToolResultEvent():
                async for e in self.handle_function_tool_result(event):
                    yield e
            case AgentRunResultEvent():
                self.result = event.result
                async for e in self.handle_run_result(event):
                    yield e
            case _:
                pass

        async for e in self.after_event(event):
            yield e

    async def handle_part_start(
        self, event: PartStartEvent, previous_part: ModelResponsePart | None = None
    ) -> AsyncIterator[EventT]:
        """Handle a PartStartEvent.

        Args:
            event: The PartStartEvent.
            previous_part: The previous part.
        """
        part = event.part
        match part:
            case TextPart():
                async for e in self.handle_text_start(part, follows_text=isinstance(previous_part, TextPart)):
                    yield e
            case ThinkingPart():
                async for e in self.handle_thinking_start(
                    part, follows_thinking=isinstance(previous_part, ThinkingPart)
                ):
                    yield e
            case ToolCallPart():
                async for e in self.handle_tool_call_start(part):
                    yield e
            case BuiltinToolCallPart():
                async for e in self.handle_builtin_tool_call_start(part):
                    yield e
            case BuiltinToolReturnPart():
                async for e in self.handle_builtin_tool_return(part):
                    yield e
            case FilePart():
                async for e in self.handle_file(part):
                    yield e

    async def handle_part_delta(self, event: PartDeltaEvent) -> AsyncIterator[EventT]:
        """Handle a PartDeltaEvent.

        Args:
            event: The PartDeltaEvent.
        """
        delta = event.delta
        match delta:
            case TextPartDelta():
                async for e in self.handle_text_delta(delta):
                    yield e
            case ThinkingPartDelta():
                async for e in self.handle_thinking_delta(delta):
                    yield e
            case ToolCallPartDelta():
                async for e in self.handle_tool_call_delta(delta):
                    yield e

    async def handle_part_end(
        self, part: ModelResponsePart, next_part: ModelResponsePart | None = None
    ) -> AsyncIterator[EventT]:
        """Handle the end of a part.

        Args:
            part: The part that ended.
            next_part: The new part that started.
        """
        # TODO (DouweM): Make this a proper event. Figure out a proper way to do context manager style wrapping
        match part:
            case TextPart():
                async for e in self.handle_text_end(part, followed_by_text=isinstance(next_part, TextPart)):
                    yield e
            case ThinkingPart():
                async for e in self.handle_thinking_end(part, followed_by_thinking=isinstance(next_part, ThinkingPart)):
                    yield e
            case ToolCallPart():
                async for e in self.handle_tool_call_end(part):
                    yield e
            case BuiltinToolCallPart():
                async for e in self.handle_builtin_tool_call_end(part):
                    yield e
            case BuiltinToolReturnPart() | FilePart():
                # These don't have deltas, so they don't need to be ended.
                pass

    async def handle_text_start(self, part: TextPart, follows_text: bool = False) -> AsyncIterator[EventT]:
        """Handle a TextPart at start.

        Args:
            part: The TextPart.
            follows_text: Whether the part follows a text part.

        Yields:
            Protocol-specific events.
        """
        return
        yield  # Make this an async generator

    async def handle_text_delta(self, delta: TextPartDelta) -> AsyncIterator[EventT]:
        """Handle a TextPartDelta.

        Args:
            delta: The TextPartDelta.

        Yields:
            Protocol-specific events.
        """
        return
        yield  # Make this an async generator

    async def handle_thinking_start(self, part: ThinkingPart, follows_thinking: bool = False) -> AsyncIterator[EventT]:
        """Handle a ThinkingPart at start.

        Args:
            part: The ThinkingPart.
            follows_thinking: Whether the part follows a thinking part.

        Yields:
            Protocol-specific events.
        """
        return
        yield  # Make this an async generator

    async def handle_thinking_delta(self, delta: ThinkingPartDelta) -> AsyncIterator[EventT]:
        """Handle a ThinkingPartDelta.

        Args:
            delta: The ThinkingPartDelta.

        Yields:
            Protocol-specific events.
        """
        return
        yield  # Make this an async generator

    async def handle_tool_call_start(self, part: ToolCallPart) -> AsyncIterator[EventT]:
        """Handle a ToolCallPart at start.

        Args:
            part: The tool call part.

        Yields:
            Protocol-specific events.
        """
        return
        yield  # Make this an async generator

    async def handle_builtin_tool_call_start(self, part: BuiltinToolCallPart) -> AsyncIterator[EventT]:
        """Handle a BuiltinToolCallPart at start.

        Args:
            part: The tool call part.

        Yields:
            Protocol-specific events.
        """
        return
        yield  # Make this an async generator

    async def handle_tool_call_delta(self, delta: ToolCallPartDelta) -> AsyncIterator[EventT]:
        """Handle a ToolCallPartDelta.

        Args:
            delta: The ToolCallPartDelta.

        Yields:
            Protocol-specific events.
        """
        return
        yield  # Make this an async generator

    async def handle_builtin_tool_return(self, part: BuiltinToolReturnPart) -> AsyncIterator[EventT]:
        """Handle a BuiltinToolReturnPart.

        Args:
            part: The BuiltinToolReturnPart.

        Yields:
            Protocol-specific events.
        """
        return
        yield  # Make this an async generator

    async def handle_file(self, part: FilePart) -> AsyncIterator[EventT]:
        """Handle a FilePart.

        Args:
            part: The FilePart.

        Yields:
            Protocol-specific events.
        """
        return
        yield  # Make this an async generator

    async def handle_text_end(self, part: TextPart, followed_by_text: bool = False) -> AsyncIterator[EventT]:
        """Handle the end of a TextPart."""
        return
        yield  # Make this an async generator

    async def handle_thinking_end(
        self, part: ThinkingPart, followed_by_thinking: bool = False
    ) -> AsyncIterator[EventT]:
        """Handle the end of a ThinkingPart."""
        return
        yield  # Make this an async generator

    async def handle_tool_call_end(self, part: ToolCallPart) -> AsyncIterator[EventT]:
        """Handle the end of a ToolCallPart."""
        return
        yield  # Make this an async generator

    async def handle_builtin_tool_call_end(self, part: BuiltinToolCallPart) -> AsyncIterator[EventT]:
        """Handle the end of a BuiltinToolCallPart."""
        return
        yield  # Make this an async generator

    async def handle_final_result(self, event: FinalResultEvent) -> AsyncIterator[EventT]:
        """Handle a FinalResultEvent.

        Args:
            event: The final result event.

        Yields:
            Protocol-specific events.
        """
        return
        yield  # Make this an async generator

    async def handle_function_tool_call(self, event: FunctionToolCallEvent) -> AsyncIterator[EventT]:
        """Handle a FunctionToolCallEvent.

        Args:
            event: The function tool call event.

        Yields:
            Protocol-specific events.
        """
        return
        yield  # Make this an async generator

    async def handle_function_tool_result(self, event: FunctionToolResultEvent) -> AsyncIterator[EventT]:
        """Handle a FunctionToolResultEvent.

        Args:
            event: The function tool result event.

        Yields:
            Protocol-specific events.
        """
        return
        yield  # Make this an async generator

    async def handle_run_result(self, event: AgentRunResultEvent) -> AsyncIterator[EventT]:
        """Handle an AgentRunResultEvent (final event with result/usage).

        Args:
            event: The agent run result event.

        Yields:
            Protocol-specific events.
        """
        return
        yield  # Make this an async generator

    # Lifecycle hooks (optional overrides)

    async def before_event(self, event: SourceEvent) -> AsyncIterator[EventT]:
        """Handle an event before it is processed.

        Args:
            event: The event to handle.
        """
        return
        yield  # Make this an async generator

    async def after_event(self, event: SourceEvent) -> AsyncIterator[EventT]:
        """Handle an event after it is processed.

        Args:
            event: The event to handle.
        """
        return
        yield  # Make this an async generator

    async def before_request(self) -> AsyncIterator[EventT]:
        """Handle a request before it is processed."""
        return
        yield  # Make this an async generator

    async def after_request(self) -> AsyncIterator[EventT]:
        """Handle a request after it is processed."""
        return
        yield  # Make this an async generator

    async def before_response(self) -> AsyncIterator[EventT]:
        """Handle a response before it is processed."""
        return
        yield  # Make this an async generator

    async def after_response(self) -> AsyncIterator[EventT]:
        """Yield events after agent streaming completes."""
        return
        yield  # Make this an async generator

    async def before_stream(self) -> AsyncIterator[EventT]:
        """Yield events before agent streaming starts.

        This hook is called before any agent events are processed.
        Override this to inject custom events at the start of the stream.

        Yields:
            Protocol-specific events to emit before streaming.
        """
        return
        yield  # Make this an async generator

    async def after_stream(self) -> AsyncIterator[EventT]:
        """Yield events after agent streaming completes.

        This hook is called after all agent events have been processed.
        Override this to inject custom events at the end of the stream.

        Yields:
            Protocol-specific events to emit after streaming.
        """
        return
        yield  # Make this an async generator

    async def on_error(self, error: Exception) -> AsyncIterator[EventT]:
        """Handle errors that occur during streaming (after stream has started).

        Args:
            error: The error that occurred during streaming.

        Yields:
            Protocol-specific error events.
        """
        return
        yield  # Make this an async generator
