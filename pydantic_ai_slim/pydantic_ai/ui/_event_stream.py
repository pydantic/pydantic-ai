from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeAlias, TypeVar, cast
from uuid import uuid4

from pydantic_ai import _utils

from ..messages import (
    AgentStreamEvent,
    BuiltinToolCallEvent,  # pyright: ignore[reportDeprecated]
    BuiltinToolCallPart,
    BuiltinToolResultEvent,  # pyright: ignore[reportDeprecated]
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
from ..output import OutputDataT
from ..run import AgentRunResult, AgentRunResultEvent
from ..tools import AgentDepsT

if TYPE_CHECKING:
    from starlette.responses import StreamingResponse


SSE_CONTENT_TYPE = 'text/event-stream'
"""Content type header value for Server-Sent Events (SSE)."""

EventT = TypeVar('EventT')
"""Type variable for protocol-specific event types."""

RunInputT = TypeVar('RunInputT')
"""Type variable for protocol-specific run input types."""

NativeEvent: TypeAlias = AgentStreamEvent | AgentRunResultEvent[Any]
"""Type alias for the native event type, which is either an `AgentStreamEvent` or an `AgentRunResultEvent`."""

OnCompleteFunc: TypeAlias = (
    Callable[[AgentRunResult[Any]], None]
    | Callable[[AgentRunResult[Any]], Awaitable[None]]
    | Callable[[AgentRunResult[Any]], AsyncIterator[EventT]]
)
"""Callback function type that receives the `AgentRunResult` of the completed run. Can be sync, async, or an async generator of protocol-specific events."""


@dataclass
class UIEventStream(ABC, Generic[RunInputT, EventT, AgentDepsT, OutputDataT]):
    """Base class for UI event stream transformers.

    This class is responsible for transforming Pydantic AI events into protocol-specific events.
    """

    run_input: RunInputT

    accept: str | None = None
    """The `Accept` header value of the request, used to determine how to encode the protocol-specific events for the streaming response."""

    message_id: str = field(default_factory=lambda: str(uuid4()))
    """The message ID to use for the next event."""

    _turn: Literal['request', 'response'] | None = None

    _result: AgentRunResult[OutputDataT] | None = None
    _final_result_event: FinalResultEvent | None = None

    def new_message_id(self) -> str:
        """Generate and store a new message ID."""
        self.message_id = str(uuid4())
        return self.message_id

    @property
    def response_headers(self) -> Mapping[str, str] | None:
        """Response headers to return to the frontend."""
        return None

    @property
    def content_type(self) -> str:
        """Get the content type for the event stream, compatible with the `Accept` header value.

        By default, this returns the Server-Sent Events content type (`text/event-stream`).
        If a subclass supports other types as well, it should consider `self.accept` in [`encode_event()`][pydantic_ai.ui.UIEventStream.encode_event] and return the resulting content type.
        """
        return SSE_CONTENT_TYPE

    @abstractmethod
    def encode_event(self, event: EventT) -> str:
        """Encode a protocol-specific event as a string."""
        raise NotImplementedError

    async def encode_stream(self, stream: AsyncIterator[EventT]) -> AsyncIterator[str]:
        """Encode a stream of protocol-specific events as strings according to the `Accept` header value."""
        async for event in stream:
            yield self.encode_event(event)

    def streaming_response(self, stream: AsyncIterator[EventT]) -> StreamingResponse:
        """Generate a streaming response from a stream of protocol-specific events."""
        try:
            from starlette.responses import StreamingResponse
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                'Please install the `starlette` package to use the `streaming_response()` method, '
                'you can use the `ui` optional group — `pip install "pydantic-ai-slim[ui]"`'
            ) from e

        return StreamingResponse(
            self.encode_stream(stream),
            headers=self.response_headers,
            media_type=self.content_type,
        )

    async def transform_stream(  # noqa: C901
        self, stream: AsyncIterator[NativeEvent], on_complete: OnCompleteFunc[EventT] | None = None
    ) -> AsyncIterator[EventT]:
        """Transform a stream of Pydantic AI events into protocol-specific events.

        This method dispatches to specific hooks and `handle_*` methods that subclasses can override:
        - [`before_stream()`][pydantic_ai.ui.UIEventStream.before_stream]
        - [`after_stream()`][pydantic_ai.ui.UIEventStream.after_stream]
        - [`on_error()`][pydantic_ai.ui.UIEventStream.on_error]
        - [`before_request()`][pydantic_ai.ui.UIEventStream.before_request]
        - [`after_request()`][pydantic_ai.ui.UIEventStream.after_request]
        - [`before_response()`][pydantic_ai.ui.UIEventStream.before_response]
        - [`after_response()`][pydantic_ai.ui.UIEventStream.after_response]
        - [`handle_event()`][pydantic_ai.ui.UIEventStream.handle_event]

        Args:
            stream: The stream of Pydantic AI events to transform.
            on_complete: Optional callback function called when the agent run completes successfully.
                The callback receives the completed [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] and can optionally yield additional protocol-specific events.
        """
        async for e in self.before_stream():
            yield e

        try:
            async for event in stream:
                if isinstance(event, PartStartEvent):
                    async for e in self._turn_to('response'):
                        yield e
                elif isinstance(event, FunctionToolCallEvent):
                    async for e in self._turn_to('request'):
                        yield e
                elif isinstance(event, AgentRunResultEvent):
                    if (
                        self._final_result_event
                        and (tool_call_id := self._final_result_event.tool_call_id)
                        and (tool_name := self._final_result_event.tool_name)
                    ):
                        async for e in self._turn_to('request'):
                            yield e

                        self._final_result_event = None
                        # Ensure the stream does not end on a dangling tool call without a result.
                        output_tool_result_event = FunctionToolResultEvent(
                            result=ToolReturnPart(
                                tool_call_id=tool_call_id,
                                tool_name=tool_name,
                                content='Final result processed.',
                            )
                        )
                        async for e in self.handle_function_tool_result(output_tool_result_event):
                            yield e

                    result = cast(AgentRunResult[OutputDataT], event.result)
                    self._result = result

                    async for e in self._turn_to(None):
                        yield e

                    if on_complete is not None:
                        if inspect.isasyncgenfunction(on_complete):
                            async for e in on_complete(result):
                                yield e
                        elif _utils.is_async_callable(on_complete):
                            await on_complete(result)
                        else:
                            await _utils.run_in_executor(on_complete, result)
                elif isinstance(event, FinalResultEvent):
                    self._final_result_event = event

                if isinstance(event, BuiltinToolCallEvent | BuiltinToolResultEvent):  # pyright: ignore[reportDeprecated]
                    # These events were deprecated before this feature was introduced
                    continue

                async for e in self.handle_event(event):
                    yield e
        except Exception as e:
            async for e in self.on_error(e):
                yield e
        finally:
            async for e in self._turn_to(None):
                yield e

            async for e in self.after_stream():
                yield e

    async def _turn_to(self, to_turn: Literal['request', 'response'] | None) -> AsyncIterator[EventT]:
        """Fire hooks when turning from request to response or vice versa."""
        if to_turn == self._turn:
            return

        if self._turn == 'request':
            async for e in self.after_request():
                yield e
        elif self._turn == 'response':
            async for e in self.after_response():
                yield e

        self._turn = to_turn

        if to_turn == 'request':
            async for e in self.before_request():
                yield e
        elif to_turn == 'response':
            async for e in self.before_response():
                yield e

    async def handle_event(self, event: NativeEvent) -> AsyncIterator[EventT]:
        """Transform a Pydantic AI event into one or more protocol-specific events.

        This method dispatches to specific `handle_*` methods based on event type:

        - [`PartStartEvent`][pydantic_ai.messages.PartStartEvent] -> [`handle_part_start()`][pydantic_ai.ui.UIEventStream.handle_part_start]
        - [`PartDeltaEvent`][pydantic_ai.messages.PartDeltaEvent] -> `handle_part_delta`
        - [`PartEndEvent`][pydantic_ai.messages.PartEndEvent] -> `handle_part_end`
        - [`FinalResultEvent`][pydantic_ai.messages.FinalResultEvent] -> `handle_final_result`
        - [`FunctionToolCallEvent`][pydantic_ai.messages.FunctionToolCallEvent] -> `handle_function_tool_call`
        - [`FunctionToolResultEvent`][pydantic_ai.messages.FunctionToolResultEvent] -> `handle_function_tool_result`
        - [`AgentRunResultEvent`][pydantic_ai.run.AgentRunResultEvent] -> `handle_run_result`

        Subclasses are encouraged to override the individual `handle_*` methods rather than this one.
        If you need specific behavior for all events, make sure you call the super method.
        """
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
                async for e in self.handle_final_result(event):
                    yield e
            case FunctionToolCallEvent():
                async for e in self.handle_function_tool_call(event):
                    yield e
            case FunctionToolResultEvent():
                async for e in self.handle_function_tool_result(event):
                    yield e
            case AgentRunResultEvent():
                async for e in self.handle_run_result(event):
                    yield e
            case _:
                pass

    async def handle_part_start(self, event: PartStartEvent) -> AsyncIterator[EventT]:
        """Handle a `PartStartEvent`.

        This method dispatches to specific `handle_*` methods based on part type:

        - [`TextPart`][pydantic_ai.messages.TextPart] -> [`handle_text_start()`][pydantic_ai.ui.UIEventStream.handle_text_start]
        - [`ThinkingPart`][pydantic_ai.messages.ThinkingPart] -> [`handle_thinking_start()`][pydantic_ai.ui.UIEventStream.handle_thinking_start]
        - [`ToolCallPart`][pydantic_ai.messages.ToolCallPart] -> [`handle_tool_call_start()`][pydantic_ai.ui.UIEventStream.handle_tool_call_start]
        - [`BuiltinToolCallPart`][pydantic_ai.messages.BuiltinToolCallPart] -> [`handle_builtin_tool_call_start()`][pydantic_ai.ui.UIEventStream.handle_builtin_tool_call_start]
        - [`BuiltinToolReturnPart`][pydantic_ai.messages.BuiltinToolReturnPart] -> [`handle_builtin_tool_return()`][pydantic_ai.ui.UIEventStream.handle_builtin_tool_return]
        - [`FilePart`][pydantic_ai.messages.FilePart] -> [`handle_file()`][pydantic_ai.ui.UIEventStream.handle_file]

        Subclasses are encouraged to override the individual `handle_*` methods rather than this one.
        If you need specific behavior for all part start events, make sure you call the super method.

        Args:
            event: The part start event.
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
            case FilePart():  # pragma: no branch
                async for e in self.handle_file(part):
                    yield e

    async def handle_part_delta(self, event: PartDeltaEvent) -> AsyncIterator[EventT]:
        """Handle a PartDeltaEvent.

        This method dispatches to specific `handle_*_delta` methods based on part delta type:

        - [`TextPartDelta`][pydantic_ai.messages.TextPartDelta] -> [`handle_text_delta()`][pydantic_ai.ui.UIEventStream.handle_text_delta]
        - [`ThinkingPartDelta`][pydantic_ai.messages.ThinkingPartDelta] -> [`handle_thinking_delta()`][pydantic_ai.ui.UIEventStream.handle_thinking_delta]
        - [`ToolCallPartDelta`][pydantic_ai.messages.ToolCallPartDelta] -> [`handle_tool_call_delta()`][pydantic_ai.ui.UIEventStream.handle_tool_call_delta]

        Subclasses are encouraged to override the individual `handle_*_delta` methods rather than this one.
        If you need specific behavior for all part delta events, make sure you call the super method.

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
            case ToolCallPartDelta():  # pragma: no branch
                async for e in self.handle_tool_call_delta(delta):
                    yield e

    async def handle_part_end(self, event: PartEndEvent) -> AsyncIterator[EventT]:
        """Handle a `PartEndEvent`.

        This method dispatches to specific `handle_*_end` methods based on part type:

        - [`TextPart`][pydantic_ai.messages.TextPart] -> [`handle_text_end()`][pydantic_ai.ui.UIEventStream.handle_text_end]
        - [`ThinkingPart`][pydantic_ai.messages.ThinkingPart] -> [`handle_thinking_end()`][pydantic_ai.ui.UIEventStream.handle_thinking_end]
        - [`ToolCallPart`][pydantic_ai.messages.ToolCallPart] -> [`handle_tool_call_end()`][pydantic_ai.ui.UIEventStream.handle_tool_call_end]
        - [`BuiltinToolCallPart`][pydantic_ai.messages.BuiltinToolCallPart] -> [`handle_builtin_tool_call_end()`][pydantic_ai.ui.UIEventStream.handle_builtin_tool_call_end]

        Subclasses are encouraged to override the individual `handle_*_end` methods rather than this one.
        If you need specific behavior for all part end events, make sure you call the super method.

        Args:
            event: The part end event.
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
            case BuiltinToolReturnPart() | FilePart():  # pragma: no cover
                # These don't have deltas, so they don't need to be ended.
                pass

    async def before_stream(self) -> AsyncIterator[EventT]:
        """Yield events before agent streaming starts.

        This hook is called before any agent events are processed.
        Override this to inject custom events at the start of the stream.
        """
        return  # pragma: no cover
        yield  # Make this an async generator

    async def after_stream(self) -> AsyncIterator[EventT]:
        """Yield events after agent streaming completes.

        This hook is called after all agent events have been processed.
        Override this to inject custom events at the end of the stream.
        """
        return  # pragma: no cover
        yield  # Make this an async generator

    async def on_error(self, error: Exception) -> AsyncIterator[EventT]:
        """Handle errors that occur during streaming.

        Args:
            error: The error that occurred during streaming.
        """
        return  # pragma: no cover
        yield  # Make this an async generator

    async def before_request(self) -> AsyncIterator[EventT]:
        """Yield events before a model request is processed.

        Override this to inject custom events at the start of the request.
        """
        return  # pragma: lax no cover
        yield  # Make this an async generator

    async def after_request(self) -> AsyncIterator[EventT]:
        """Yield events after a model request is processed.

        Override this to inject custom events at the end of the request.
        """
        return  # pragma: lax no cover
        yield  # Make this an async generator

    async def before_response(self) -> AsyncIterator[EventT]:
        """Yield events before a model response is processed.

        Override this to inject custom events at the start of the response.
        """
        return  # pragma: no cover
        yield  # Make this an async generator

    async def after_response(self) -> AsyncIterator[EventT]:
        """Yield events after a model response is processed.

        Override this to inject custom events at the end of the response.
        """
        return  # pragma: lax no cover
        yield  # Make this an async generator

    async def handle_text_start(self, part: TextPart, follows_text: bool = False) -> AsyncIterator[EventT]:
        """Handle the start of a `TextPart`.

        Args:
            part: The text part.
            follows_text: Whether the part is directly preceded by another text part. In this case, you may want to yield a "text-delta" event instead of a "text-start" event.
        """
        return  # pragma: no cover
        yield  # Make this an async generator

    async def handle_text_delta(self, delta: TextPartDelta) -> AsyncIterator[EventT]:
        """Handle a `TextPartDelta`.

        Args:
            delta: The text part delta.
        """
        return  # pragma: no cover
        yield  # Make this an async generator

    async def handle_text_end(self, part: TextPart, followed_by_text: bool = False) -> AsyncIterator[EventT]:
        """Handle the end of a `TextPart`.

        Args:
            part: The text part.
            followed_by_text: Whether the part is directly followed by another text part. In this case, you may not want to yield a "text-end" event yet.
        """
        return  # pragma: no cover
        yield  # Make this an async generator

    async def handle_thinking_start(self, part: ThinkingPart, follows_thinking: bool = False) -> AsyncIterator[EventT]:
        """Handle the start of a `ThinkingPart`.

        Args:
            part: The thinking part.
            follows_thinking: Whether the part is directly preceded by another thinking part. In this case, you may want to yield a "thinking-delta" event instead of a "thinking-start" event.
        """
        return  # pragma: no cover
        yield  # Make this an async generator

    async def handle_thinking_delta(self, delta: ThinkingPartDelta) -> AsyncIterator[EventT]:
        """Handle a `ThinkingPartDelta`.

        Args:
            delta: The thinking part delta.
        """
        return  # pragma: no cover
        yield  # Make this an async generator

    async def handle_thinking_end(
        self, part: ThinkingPart, followed_by_thinking: bool = False
    ) -> AsyncIterator[EventT]:
        """Handle the end of a `ThinkingPart`.

        Args:
            part: The thinking part.
            followed_by_thinking: Whether the part is directly followed by another thinking part. In this case, you may not want to yield a "thinking-end" event yet.
        """
        return  # pragma: no cover
        yield  # Make this an async generator

    async def handle_tool_call_start(self, part: ToolCallPart) -> AsyncIterator[EventT]:
        """Handle the start of a `ToolCallPart`.

        Args:
            part: The tool call part.
        """
        return  # pragma: no cover
        yield  # Make this an async generator

    async def handle_tool_call_delta(self, delta: ToolCallPartDelta) -> AsyncIterator[EventT]:
        """Handle a `ToolCallPartDelta`.

        Args:
            delta: The tool call part delta.
        """
        return  # pragma: no cover
        yield  # Make this an async generator

    async def handle_tool_call_end(self, part: ToolCallPart) -> AsyncIterator[EventT]:
        """Handle the end of a `ToolCallPart`.

        Args:
            part: The tool call part.
        """
        return  # pragma: no cover
        yield  # Make this an async generator

    async def handle_builtin_tool_call_start(self, part: BuiltinToolCallPart) -> AsyncIterator[EventT]:
        """Handle a `BuiltinToolCallPart` at start.

        Args:
            part: The builtin tool call part.
        """
        return  # pragma: no cover
        yield  # Make this an async generator

    async def handle_builtin_tool_call_end(self, part: BuiltinToolCallPart) -> AsyncIterator[EventT]:
        """Handle the end of a `BuiltinToolCallPart`.

        Args:
            part: The builtin tool call part.
        """
        return  # pragma: no cover
        yield  # Make this an async generator

    async def handle_builtin_tool_return(self, part: BuiltinToolReturnPart) -> AsyncIterator[EventT]:
        """Handle a `BuiltinToolReturnPart`.

        Args:
            part: The builtin tool return part.
        """
        return  # pragma: no cover
        yield  # Make this an async generator

    async def handle_file(self, part: FilePart) -> AsyncIterator[EventT]:
        """Handle a `FilePart`.

        Args:
            part: The file part.
        """
        return  # pragma: no cover
        yield  # Make this an async generator

    async def handle_final_result(self, event: FinalResultEvent) -> AsyncIterator[EventT]:
        """Handle a `FinalResultEvent`.

        Args:
            event: The final result event.
        """
        return
        yield  # Make this an async generator

    async def handle_function_tool_call(self, event: FunctionToolCallEvent) -> AsyncIterator[EventT]:
        """Handle a `FunctionToolCallEvent`.

        Args:
            event: The function tool call event.
        """
        return
        yield  # Make this an async generator

    async def handle_function_tool_result(self, event: FunctionToolResultEvent) -> AsyncIterator[EventT]:
        """Handle a `FunctionToolResultEvent`.

        Args:
            event: The function tool result event.
        """
        return  # pragma: no cover
        yield  # Make this an async generator

    async def handle_run_result(self, event: AgentRunResultEvent) -> AsyncIterator[EventT]:
        """Handle an `AgentRunResultEvent`.

        Args:
            event: The agent run result event.
        """
        return
        yield  # Make this an async generator
