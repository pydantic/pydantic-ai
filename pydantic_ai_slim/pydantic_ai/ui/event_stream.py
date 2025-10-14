"""Base classes for UI event stream protocols.

This module provides abstract base classes for implementing UI event stream adapters
that transform Pydantic AI agent events into protocol-specific events (e.g., AG-UI, Vercel AI).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Generic, TypeVar
from uuid import uuid4

from ..messages import (
    AgentStreamEvent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    FilePart,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
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

    _final_result_event: FinalResultEvent | None = None

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
            yield self.encode_event(event, accept)

    async def handle_stream(self, stream: AsyncIterator[SourceEvent]) -> AsyncIterator[EventT]:
        """Handle a stream of agent events.

        Args:
            stream: The stream of agent events to handle.

        Yields:
            Protocol-specific events.
        """
        async for e in self.before_stream():
            yield e

        try:
            async for event in stream:
                async for e in self.handle_event(event):
                    yield e
        except Exception as e:
            async for e in self.on_error(e):
                yield e
        else:
            async for e in self.after_stream():
                yield e

    async def handle_event(self, event: SourceEvent) -> AsyncIterator[EventT]:  # noqa: C901
        """Transform a Pydantic AI agent event into protocol-specific events.

        This method dispatches to specific `handle_*` methods based on event and part type.
        Subclasses should implement the individual handler methods rather than overriding this.

        Args:
            event: The Pydantic AI agent event to transform.

        Yields:
            Protocol-specific events.
        """
        async for e in self.before_event(event):
            yield e

        match event:
            case PartStartEvent():
                async for e in self.handle_part_start(event):
                    yield e
            case PartDeltaEvent():
                async for e in self.handle_part_delta(event):
                    yield e
            case PartEndEvent():
                async for e in self.handle_part_end(event):
                    yield e
            case FinalResultEvent():
                self._final_result_event = event
                async for e in self.handle_final_result(event):
                    yield e
            case FunctionToolCallEvent():
                async for e in self.handle_function_tool_call(event):
                    yield e
            case FunctionToolResultEvent():
                async for e in self.handle_function_tool_result(event):
                    yield e
            case AgentRunResultEvent():
                if (
                    self._final_result_event
                    and (tool_call_id := self._final_result_event.tool_call_id)
                    and (tool_name := self._final_result_event.tool_name)
                ):
                    self._final_result_event = None
                    output_tool_result_event = FunctionToolResultEvent(
                        result=ToolReturnPart(
                            tool_call_id=tool_call_id,
                            tool_name=tool_name,
                            content='Final result processed.',
                        )
                    )
                    async for e in self.handle_function_tool_result(output_tool_result_event):
                        yield e

                self.result = event.result
                async for e in self.handle_run_result(event):
                    yield e
            case _:
                pass

        async for e in self.after_event(event):
            yield e

    async def handle_part_start(self, event: PartStartEvent) -> AsyncIterator[EventT]:
        """Handle a PartStartEvent.

        Args:
            event: The PartStartEvent.
        """
        part = event.part
        previous_part_kind = event.previous_part_kind
        match part:
            case TextPart():
                async for e in self.handle_text_start(part, follows_text=previous_part_kind == 'text'):
                    yield e
            case ThinkingPart():
                async for e in self.handle_thinking_start(part, follows_thinking=previous_part_kind == 'thinking'):
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

    async def handle_part_end(self, event: PartEndEvent) -> AsyncIterator[EventT]:
        """Handle a PartEndEvent.

        Args:
            event: The PartEndEvent.
        """
        part = event.part
        next_part_kind = event.next_part_kind
        match part:
            case TextPart():
                async for e in self.handle_text_end(part, followed_by_text=next_part_kind == 'text'):
                    yield e
            case ThinkingPart():
                async for e in self.handle_thinking_end(part, followed_by_thinking=next_part_kind == 'thinking'):
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
