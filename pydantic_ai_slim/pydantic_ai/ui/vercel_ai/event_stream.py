"""Vercel AI event stream implementation."""

# pyright: reportIncompatibleMethodOverride=false

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from pydantic_core import to_json

from ...messages import (
    BuiltinToolCallEvent,  # type: ignore[reportDeprecated]
    BuiltinToolCallPart,
    BuiltinToolResultEvent,  # type: ignore[reportDeprecated]
    BuiltinToolReturnPart,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
)
from ...run import AgentRunResultEvent
from ...tools import AgentDepsT
from .. import BaseEventStream
from .request_types import TextUIPart, UIMessage
from .response_types import (
    AbstractSSEChunk,
    DoneChunk,
    ErrorChunk,
    FinishChunk,
    ReasoningDeltaChunk,
    ReasoningStartChunk,
    TextDeltaChunk,
    TextStartChunk,
    ToolInputDeltaChunk,
    ToolInputStartChunk,
    ToolOutputAvailableChunk,
)

__all__ = ['VercelAIEventStream', 'protocol_messages_to_pai_messages']


def _json_dumps(obj: Any) -> str:
    """Dump an object to JSON string."""
    return to_json(obj).decode('utf-8')


class VercelAIEventStream(BaseEventStream[AbstractSSEChunk | DoneChunk, AgentDepsT]):
    """Transforms Pydantic AI agent events into Vercel AI protocol events.

    This class handles the stateful transformation of streaming agent events
    into the Vercel AI protocol format, managing message IDs and final result tool tracking.

    Example:
        ```python
        event_stream = VercelAIEventStream()
        async for vercel_event in event_stream.agent_event_to_events(pai_event):
            print(vercel_event.sse())
        ```
    """

    def __init__(self) -> None:
        """Initialize Vercel AI event stream state."""
        super().__init__()
        self.new_message_id()  # Generate a message ID at initialization
        self._final_result_tool_id: str | None = None

    # Granular handlers implementation

    async def handle_text_start(self, part: TextPart) -> AsyncIterator[AbstractSSEChunk]:
        """Handle a TextPart at start."""
        yield TextStartChunk(id=self.message_id or self.new_message_id())
        if part.content:
            yield TextDeltaChunk(id=self.message_id, delta=part.content)

    async def handle_text_delta(self, delta: TextPartDelta) -> AsyncIterator[AbstractSSEChunk]:
        """Handle a TextPartDelta."""
        if delta.content_delta:
            yield TextDeltaChunk(id=self.message_id, delta=delta.content_delta)

    async def handle_thinking_start(self, part: ThinkingPart) -> AsyncIterator[AbstractSSEChunk]:
        """Handle a ThinkingPart at start."""
        if not self.message_id:
            self.new_message_id()
        yield ReasoningStartChunk(id=self.message_id)
        if part.content:
            yield ReasoningDeltaChunk(id=self.message_id, delta=part.content)

    async def handle_thinking_delta(self, delta: ThinkingPartDelta) -> AsyncIterator[AbstractSSEChunk]:
        """Handle a ThinkingPartDelta."""
        if delta.content_delta:
            yield ReasoningDeltaChunk(id=self.message_id, delta=delta.content_delta)

    async def handle_tool_call_start(self, part: ToolCallPart | BuiltinToolCallPart) -> AsyncIterator[AbstractSSEChunk]:
        """Handle a ToolCallPart or BuiltinToolCallPart at start."""
        yield ToolInputStartChunk(tool_call_id=part.tool_call_id, tool_name=part.tool_name)
        if isinstance(part.args, str):
            yield ToolInputDeltaChunk(tool_call_id=part.tool_call_id, input_text_delta=part.args)
        elif part.args is not None:
            yield ToolInputDeltaChunk(tool_call_id=part.tool_call_id, input_text_delta=_json_dumps(part.args))

    async def handle_tool_call_delta(self, delta: ToolCallPartDelta) -> AsyncIterator[AbstractSSEChunk]:
        """Handle a ToolCallPartDelta."""
        tool_call_id = delta.tool_call_id or ''
        if isinstance(delta.args_delta, str):
            yield ToolInputDeltaChunk(tool_call_id=tool_call_id, input_text_delta=delta.args_delta)
        elif delta.args_delta is not None:
            yield ToolInputDeltaChunk(tool_call_id=tool_call_id, input_text_delta=_json_dumps(delta.args_delta))

    async def handle_builtin_tool_return(self, part: BuiltinToolReturnPart) -> AsyncIterator[AbstractSSEChunk]:
        """Handle a BuiltinToolReturnPart."""
        yield ToolOutputAvailableChunk(tool_call_id=part.tool_call_id, output=part.content)

    async def handle_function_tool_call(self, event: FunctionToolCallEvent) -> AsyncIterator[AbstractSSEChunk]:
        """Handle a FunctionToolCallEvent.

        No Vercel AI events are emitted at this stage since tool calls are handled in PartStartEvent.
        """
        return
        yield  # Make this an async generator

    async def handle_function_tool_result(self, event: FunctionToolResultEvent) -> AsyncIterator[AbstractSSEChunk]:
        """Handle a FunctionToolResultEvent, emitting tool result events."""
        result = event.result
        if isinstance(result, ToolReturnPart):
            yield ToolOutputAvailableChunk(tool_call_id=result.tool_call_id, output=result.content)
        elif isinstance(result, RetryPromptPart):
            # For retry prompts, emit the error content as tool output
            yield ToolOutputAvailableChunk(tool_call_id=result.tool_call_id, output=result.content)

    async def handle_builtin_tool_call(self, event: BuiltinToolCallEvent) -> AsyncIterator[AbstractSSEChunk]:  # type: ignore[reportDeprecated]
        """Handle a BuiltinToolCallEvent, emitting tool input events."""
        part = event.part
        yield ToolInputStartChunk(tool_call_id=part.tool_call_id, tool_name=part.tool_name)
        if isinstance(part.args, str):
            yield ToolInputDeltaChunk(tool_call_id=part.tool_call_id, input_text_delta=part.args)
        elif part.args is not None:
            yield ToolInputDeltaChunk(tool_call_id=part.tool_call_id, input_text_delta=_json_dumps(part.args))

    async def handle_builtin_tool_result(self, event: BuiltinToolResultEvent) -> AsyncIterator[AbstractSSEChunk]:  # type: ignore[reportDeprecated]
        """Handle a BuiltinToolResultEvent, emitting tool output events."""
        result = event.result
        yield ToolOutputAvailableChunk(tool_call_id=result.tool_call_id, output=result.content)

    async def handle_final_result(self, event: FinalResultEvent) -> AsyncIterator[AbstractSSEChunk]:
        """Handle a FinalResultEvent, tracking the final result tool."""
        if event.tool_call_id and event.tool_name:
            self._final_result_tool_id = event.tool_call_id
            yield ToolInputStartChunk(tool_call_id=event.tool_call_id, tool_name=event.tool_name)

    async def handle_run_result(self, event: AgentRunResultEvent) -> AsyncIterator[AbstractSSEChunk]:
        """Handle an AgentRunResultEvent.

        No additional Vercel AI events are emitted at this stage.
        """
        return
        yield  # Make this an async generator

    async def after_stream(self) -> AsyncIterator[AbstractSSEChunk | DoneChunk]:
        """Yield events after agent streaming completes."""
        # Close the final result tool if there was one
        if tool_call_id := self._final_result_tool_id:
            yield ToolOutputAvailableChunk(tool_call_id=tool_call_id, output=None)
        yield FinishChunk()
        yield DoneChunk()

    async def on_validation_error(self, error: Exception) -> AsyncIterator[AbstractSSEChunk]:
        """Handle validation errors before stream starts."""
        yield ErrorChunk(error_text=str(error))

    async def on_stream_error(self, error: Exception) -> AsyncIterator[AbstractSSEChunk]:
        """Handle errors during streaming."""
        yield ErrorChunk(error_text=str(error))


def protocol_messages_to_pai_messages(messages: list[UIMessage]) -> list[ModelMessage]:
    """Convert Vercel AI protocol messages to Pydantic AI messages.

    Args:
        messages: List of Vercel AI UIMessage objects.

    Returns:
        List of Pydantic AI ModelMessage objects.

    Raises:
        ValueError: If message format is not supported.
    """
    from ...messages import ModelRequest, ModelResponse, SystemPromptPart, TextPart, UserPromptPart

    pai_messages: list[ModelMessage] = []

    for msg in messages:
        if msg.role == 'user':
            # User message - extract text from parts
            texts: list[str] = []
            for part in msg.parts:
                if isinstance(part, TextUIPart):
                    texts.append(part.text)
                else:
                    raise ValueError(f'Only text parts are supported for user messages, got {type(part).__name__}')

            if texts:
                pai_messages.append(ModelRequest(parts=[UserPromptPart(content='\n'.join(texts))]))

        elif msg.role == 'assistant':
            # Assistant message - for now, just extract text
            # Full reconstruction of ModelResponse with tool calls would require more complex logic
            texts: list[str] = []
            for part in msg.parts:
                if isinstance(part, TextUIPart):
                    texts.append(part.text)
                # TODO: Handle ToolOutputAvailablePart for full message history reconstruction

            if texts:
                pai_messages.append(ModelResponse(parts=[TextPart(content='\n'.join(texts))]))

        elif msg.role == 'system':
            # System message - not in standard Vercel AI protocol but might be custom
            texts: list[str] = []
            for part in msg.parts:
                if isinstance(part, TextUIPart):
                    texts.append(part.text)

            if texts:
                pai_messages.append(ModelRequest(parts=[SystemPromptPart(content='\n'.join(texts))]))

    return pai_messages
