"""Base classes for UI event stream protocols.

This module provides abstract base classes for implementing UI event stream adapters
that transform Pydantic AI agent events into protocol-specific events (e.g., AG-UI, Vercel AI).
"""

# pyright: reportIncompatibleMethodOverride=false, reportUnknownVariableType=false, reportGeneralTypeIssues=false

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable
from uuid import uuid4

from ..messages import (
    AgentStreamEvent,
    BuiltinToolCallEvent,  # type: ignore[reportDeprecated]
    BuiltinToolCallPart,
    BuiltinToolResultEvent,  # type: ignore[reportDeprecated]
    BuiltinToolReturnPart,
    FilePart,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)
from ..run import AgentRunResultEvent
from ..tools import AgentDepsT

__all__ = [
    'SSEEvent',
    'BaseEventStream',
    'BaseAdapter',
]

EventT = TypeVar('EventT', bound='SSEEvent')
"""Type variable for protocol-specific event types."""


@runtime_checkable
class SSEEvent(Protocol):
    """Protocol for events that can be encoded as Server-Sent Events (SSE)."""

    def sse(self) -> str:
        """Encode event as a Server-Sent Event string.

        Returns:
            The SSE-formatted string representation of the event.
        """
        ...


class BaseEventStream(ABC, Generic[EventT, AgentDepsT]):
    """Base class for transforming pAI agent events into protocol-specific events.

    This class provides a granular method-per-part-type pattern that makes it easy to customize
    specific event transformations by overriding individual handler methods.

    Subclasses should:
    1. Initialize state attributes in `__init__` if needed
    2. Implement all abstract `handle_*` methods for event transformation
    3. Implement error handling methods
    4. Optionally override lifecycle hooks (`before_stream`, `after_stream`)

    Example:
        ```python
        class MyEventStream(BaseEventStream[MyEvent, MyDeps]):
            def __init__(self):
                super().__init__()
                self.custom_state = {}

            async def handle_text_start(self, part: TextPart):
                yield MyTextStartEvent(id=self.new_message_id(), text=part.content)
        ```
    """

    def __init__(self) -> None:
        """Initialize event stream state.

        Subclasses can add additional state attributes for tracking streaming context.
        """
        self.message_id: str = ''

    def new_message_id(self) -> str:
        """Generate and store a new message ID.

        Returns:
            A new UUID-based message ID.
        """
        self.message_id = str(uuid4())
        return self.message_id

    async def agent_event_to_events(self, event: AgentStreamEvent | AgentRunResultEvent) -> AsyncIterator[EventT]:  # noqa: C901
        """Transform a pAI agent event into protocol-specific events.

        This method dispatches to specific `handle_*` methods based on event and part type.
        Subclasses should implement the individual handler methods rather than overriding this.

        Args:
            event: The pAI agent event to transform.

        Yields:
            Protocol-specific events.
        """
        match event:
            case PartStartEvent(part=part):
                # Dispatch based on part type
                match part:
                    case TextPart():
                        async for e in self.handle_text_start(part):
                            yield e
                    case ThinkingPart():
                        async for e in self.handle_thinking_start(part):
                            yield e
                    case ToolCallPart() | BuiltinToolCallPart():
                        async for e in self.handle_tool_call_start(part):
                            yield e
                    case BuiltinToolReturnPart():
                        async for e in self.handle_builtin_tool_return(part):
                            yield e
                    case FilePart():
                        # FilePart is not currently handled by UI protocols
                        pass
            case PartDeltaEvent(delta=delta):
                # Dispatch based on delta type
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
            case FunctionToolCallEvent():
                async for e in self.handle_function_tool_call(event):
                    yield e
            case FunctionToolResultEvent():
                async for e in self.handle_function_tool_result(event):
                    yield e
            case BuiltinToolCallEvent():  # type: ignore[reportDeprecated]
                async for e in self.handle_builtin_tool_call(event):
                    yield e
            case BuiltinToolResultEvent():  # type: ignore[reportDeprecated]
                async for e in self.handle_builtin_tool_result(event):
                    yield e
            case FinalResultEvent():
                async for e in self.handle_final_result(event):
                    yield e
            case AgentRunResultEvent():
                async for e in self.handle_run_result(event):
                    yield e

    # Granular part handlers (abstract - must implement)

    @abstractmethod
    async def handle_text_start(self, part: TextPart) -> AsyncIterator[EventT]:
        """Handle a TextPart at start.

        Args:
            part: The TextPart.

        Yields:
            Protocol-specific events.
        """

    @abstractmethod
    async def handle_text_delta(self, delta: TextPartDelta) -> AsyncIterator[EventT]:
        """Handle a TextPartDelta.

        Args:
            delta: The TextPartDelta.

        Yields:
            Protocol-specific events.
        """

    @abstractmethod
    async def handle_thinking_start(self, part: ThinkingPart) -> AsyncIterator[EventT]:
        """Handle a ThinkingPart at start.

        Args:
            part: The ThinkingPart.

        Yields:
            Protocol-specific events.
        """

    @abstractmethod
    async def handle_thinking_delta(self, delta: ThinkingPartDelta) -> AsyncIterator[EventT]:
        """Handle a ThinkingPartDelta.

        Args:
            delta: The ThinkingPartDelta.

        Yields:
            Protocol-specific events.
        """

    @abstractmethod
    async def handle_tool_call_start(self, part: ToolCallPart | BuiltinToolCallPart) -> AsyncIterator[EventT]:
        """Handle a ToolCallPart or BuiltinToolCallPart at start.

        Args:
            part: The tool call part.

        Yields:
            Protocol-specific events.
        """

    @abstractmethod
    async def handle_tool_call_delta(self, delta: ToolCallPartDelta) -> AsyncIterator[EventT]:
        """Handle a ToolCallPartDelta.

        Args:
            delta: The ToolCallPartDelta.

        Yields:
            Protocol-specific events.
        """

    @abstractmethod
    async def handle_builtin_tool_return(self, part: BuiltinToolReturnPart) -> AsyncIterator[EventT]:
        """Handle a BuiltinToolReturnPart.

        Args:
            part: The BuiltinToolReturnPart.

        Yields:
            Protocol-specific events.
        """

    # Tool event handlers (abstract - must implement)

    @abstractmethod
    async def handle_function_tool_call(self, event: FunctionToolCallEvent) -> AsyncIterator[EventT]:
        """Handle a FunctionToolCallEvent.

        Args:
            event: The function tool call event.

        Yields:
            Protocol-specific events.
        """

    @abstractmethod
    async def handle_function_tool_result(self, event: FunctionToolResultEvent) -> AsyncIterator[EventT]:
        """Handle a FunctionToolResultEvent.

        Args:
            event: The function tool result event.

        Yields:
            Protocol-specific events.
        """

    @abstractmethod
    async def handle_builtin_tool_call(self, event: BuiltinToolCallEvent) -> AsyncIterator[EventT]:  # type: ignore[reportDeprecated]
        """Handle a BuiltinToolCallEvent.

        Args:
            event: The builtin tool call event.

        Yields:
            Protocol-specific events.
        """

    @abstractmethod
    async def handle_builtin_tool_result(self, event: BuiltinToolResultEvent) -> AsyncIterator[EventT]:  # type: ignore[reportDeprecated]
        """Handle a BuiltinToolResultEvent.

        Args:
            event: The builtin tool result event.

        Yields:
            Protocol-specific events.
        """

    @abstractmethod
    async def handle_final_result(self, event: FinalResultEvent) -> AsyncIterator[EventT]:
        """Handle a FinalResultEvent.

        Args:
            event: The final result event.

        Yields:
            Protocol-specific events.
        """

    @abstractmethod
    async def handle_run_result(self, event: AgentRunResultEvent) -> AsyncIterator[EventT]:
        """Handle an AgentRunResultEvent (final event with result/usage).

        Args:
            event: The agent run result event.

        Yields:
            Protocol-specific events.
        """

    # Lifecycle hooks (optional overrides)

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

    # Error handling (must implement)

    @abstractmethod
    async def on_validation_error(self, error: Exception) -> AsyncIterator[EventT]:
        """Handle validation errors that occur before streaming starts.

        Args:
            error: The validation error that occurred.

        Yields:
            Protocol-specific error events.
        """

    @abstractmethod
    async def on_stream_error(self, error: Exception) -> AsyncIterator[EventT]:
        """Handle errors that occur during streaming (after stream has started).

        Args:
            error: The error that occurred during streaming.

        Yields:
            Protocol-specific error events.
        """


RequestT = TypeVar('RequestT')
"""Type variable for protocol-specific request types."""

MessageT = TypeVar('MessageT')
"""Type variable for protocol-specific message types."""


class BaseAdapter(ABC, Generic[RequestT, MessageT, EventT, AgentDepsT]):
    """Base adapter for handling UI protocol requests and streaming responses.

    This class provides a unified interface for request/response handling across different
    UI protocols (AG-UI, Vercel AI, etc.). It handles:
    - Request parsing and validation
    - Message format conversion (protocol messages → pAI messages)
    - Agent execution and event streaming
    - Error handling (validation errors vs streaming errors)
    - SSE encoding

    Type Parameters:
        RequestT: Protocol-specific request type (e.g., RunAgentInput, RequestData)
        MessageT: Protocol-specific message type (e.g., ag_ui.Message, UIMessage)
        EventT: Protocol-specific event type (e.g., ag_ui.BaseEvent, AbstractSSEChunk)
        AgentDepsT: Agent dependencies type

    Example:
        ```python
        class MyAdapter(BaseAdapter[MyRequest, MyMessage, MyEvent, MyDeps]):
            def create_event_stream(self) -> BaseEventStream[MyEvent, MyDeps]:
                return MyEventStream()

            def parse_request_messages(self, request: MyRequest) -> list[MyMessage]:
                return request.messages

            def protocol_messages_to_pai_messages(self, messages: list[MyMessage]) -> list[ModelMessage]:
                # Convert protocol messages to pAI messages
                ...
        ```
    """

    @abstractmethod
    def create_event_stream(self) -> BaseEventStream[EventT, AgentDepsT]:
        """Create a new event stream for this protocol.

        Returns:
            A protocol-specific event stream instance.
        """

    @abstractmethod
    def parse_request_messages(self, request: RequestT) -> list[MessageT]:
        """Extract messages from the protocol request.

        Args:
            request: The protocol-specific request.

        Returns:
            List of protocol-specific messages.
        """

    @abstractmethod
    def protocol_messages_to_pai_messages(self, messages: list[MessageT]) -> list[ModelMessage]:
        """Convert protocol messages to Pydantic AI messages.

        Args:
            messages: List of protocol-specific messages.

        Returns:
            List of Pydantic AI ModelMessage objects.
        """

    @abstractmethod
    def encode_event(self, event: EventT) -> str:
        """Encode a protocol event as an SSE string.

        Args:
            event: The protocol-specific event.

        Returns:
            SSE-formatted string.
        """

    @abstractmethod
    async def dispatch_request(self, request: Any, deps: AgentDepsT | None = None) -> Any:
        """Handle a request and return a response.

        This method should handle the full request/response cycle:
        - Parse and validate the request
        - Run the agent with the request data
        - Return an appropriate response (e.g., StreamingResponse, EventSourceResponse)

        Args:
            request: The protocol-specific request object (e.g., Starlette Request).
            deps: Optional dependencies to pass to the agent.

        Returns:
            A protocol-specific response object.
        """
