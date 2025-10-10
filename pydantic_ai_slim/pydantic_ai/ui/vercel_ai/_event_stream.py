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
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
)
from ...tools import AgentDepsT
from .. import BaseEventStream
from ._request_types import RequestData
from ._response_types import (
    BaseChunk,
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

__all__ = ['VercelAIEventStream']


def _json_dumps(obj: Any) -> str:
    """Dump an object to JSON string."""
    return to_json(obj).decode('utf-8')


class VercelAIEventStream(BaseEventStream[RequestData, BaseChunk, AgentDepsT]):
    """TODO (DouwM): Docstring."""

    def __init__(self, request: RequestData) -> None:
        """Initialize Vercel AI event stream state."""
        super().__init__(request)
        self._final_result_tool_id: str | None = None

    async def after_stream(self) -> AsyncIterator[BaseChunk]:
        """Yield events after agent streaming completes."""
        # Close the final result tool if there was one
        if tool_call_id := self._final_result_tool_id:
            yield ToolOutputAvailableChunk(tool_call_id=tool_call_id, output=None)
        yield FinishChunk()

    async def on_error(self, error: Exception) -> AsyncIterator[BaseChunk]:
        """Handle errors during streaming."""
        yield ErrorChunk(error_text=str(error))

    # Granular handlers implementation

    async def handle_text_start(self, part: TextPart) -> AsyncIterator[BaseChunk]:
        """Handle a TextPart at start."""
        yield TextStartChunk(id=self.message_id or self.new_message_id())
        if part.content:
            yield TextDeltaChunk(id=self.message_id, delta=part.content)

    async def handle_text_delta(self, delta: TextPartDelta) -> AsyncIterator[BaseChunk]:
        """Handle a TextPartDelta."""
        if delta.content_delta:
            yield TextDeltaChunk(id=self.message_id, delta=delta.content_delta)

    async def handle_thinking_start(self, part: ThinkingPart) -> AsyncIterator[BaseChunk]:
        """Handle a ThinkingPart at start."""
        if not self.message_id:
            self.new_message_id()
        yield ReasoningStartChunk(id=self.message_id)
        if part.content:
            yield ReasoningDeltaChunk(id=self.message_id, delta=part.content)

    async def handle_thinking_delta(self, delta: ThinkingPartDelta) -> AsyncIterator[BaseChunk]:
        """Handle a ThinkingPartDelta."""
        if delta.content_delta:
            yield ReasoningDeltaChunk(id=self.message_id, delta=delta.content_delta)

    async def handle_tool_call_start(self, part: ToolCallPart | BuiltinToolCallPart) -> AsyncIterator[BaseChunk]:
        """Handle a ToolCallPart or BuiltinToolCallPart at start."""
        yield ToolInputStartChunk(tool_call_id=part.tool_call_id, tool_name=part.tool_name)
        if isinstance(part.args, str):
            yield ToolInputDeltaChunk(tool_call_id=part.tool_call_id, input_text_delta=part.args)
        elif part.args is not None:
            yield ToolInputDeltaChunk(tool_call_id=part.tool_call_id, input_text_delta=_json_dumps(part.args))

    def handle_builtin_tool_call_start(self, part: BuiltinToolCallPart) -> AsyncIterator[BaseChunk]:
        return self.handle_tool_call_start(part)

    async def handle_tool_call_delta(self, delta: ToolCallPartDelta) -> AsyncIterator[BaseChunk]:
        """Handle a ToolCallPartDelta."""
        tool_call_id = delta.tool_call_id or ''
        if isinstance(delta.args_delta, str):
            yield ToolInputDeltaChunk(tool_call_id=tool_call_id, input_text_delta=delta.args_delta)
        elif delta.args_delta is not None:
            yield ToolInputDeltaChunk(tool_call_id=tool_call_id, input_text_delta=_json_dumps(delta.args_delta))

    async def handle_builtin_tool_return(self, part: BuiltinToolReturnPart) -> AsyncIterator[BaseChunk]:
        """Handle a BuiltinToolReturnPart."""
        yield ToolOutputAvailableChunk(tool_call_id=part.tool_call_id, output=part.content)

    async def handle_function_tool_call(self, event: FunctionToolCallEvent) -> AsyncIterator[BaseChunk]:
        """Handle a FunctionToolCallEvent.

        No Vercel AI events are emitted at this stage since tool calls are handled in PartStartEvent.
        """
        return
        yield  # Make this an async generator

    async def handle_function_tool_result(self, event: FunctionToolResultEvent) -> AsyncIterator[BaseChunk]:
        """Handle a FunctionToolResultEvent, emitting tool result events."""
        result = event.result
        if isinstance(result, ToolReturnPart):
            yield ToolOutputAvailableChunk(tool_call_id=result.tool_call_id, output=result.content)
        elif isinstance(result, RetryPromptPart):
            # For retry prompts, emit the error content as tool output
            yield ToolOutputAvailableChunk(tool_call_id=result.tool_call_id, output=result.content)

    async def handle_builtin_tool_call(self, event: BuiltinToolCallEvent) -> AsyncIterator[BaseChunk]:  # type: ignore[reportDeprecated]
        """Handle a BuiltinToolCallEvent, emitting tool input events."""
        part = event.part
        yield ToolInputStartChunk(tool_call_id=part.tool_call_id, tool_name=part.tool_name)
        if isinstance(part.args, str):
            yield ToolInputDeltaChunk(tool_call_id=part.tool_call_id, input_text_delta=part.args)
        elif part.args is not None:
            yield ToolInputDeltaChunk(tool_call_id=part.tool_call_id, input_text_delta=_json_dumps(part.args))

    async def handle_builtin_tool_result(self, event: BuiltinToolResultEvent) -> AsyncIterator[BaseChunk]:  # type: ignore[reportDeprecated]
        """Handle a BuiltinToolResultEvent, emitting tool output events."""
        result = event.result
        yield ToolOutputAvailableChunk(tool_call_id=result.tool_call_id, output=result.content)

    async def handle_final_result(self, event: FinalResultEvent) -> AsyncIterator[BaseChunk]:
        """Handle a FinalResultEvent, tracking the final result tool."""
        if event.tool_call_id and event.tool_name:
            self._final_result_tool_id = event.tool_call_id
            # TODO (DouweM): Stream output tool result once it's ready
            yield ToolInputStartChunk(tool_call_id=event.tool_call_id, tool_name=event.tool_name)
