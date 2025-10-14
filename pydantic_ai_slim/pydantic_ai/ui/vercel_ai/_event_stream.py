"""Vercel AI event stream implementation."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from pydantic_core import to_json

from ...messages import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    FilePart,
    FunctionToolResultEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)
from ...tools import AgentDepsT
from .. import BaseEventStream
from ._request_types import RequestData
from ._response_types import (
    BaseChunk,
    ErrorChunk,
    FileChunk,
    FinishChunk,
    ReasoningDeltaChunk,
    ReasoningEndChunk,
    ReasoningStartChunk,
    StartChunk,
    TextDeltaChunk,
    TextEndChunk,
    TextStartChunk,
    ToolInputAvailableChunk,
    ToolInputDeltaChunk,
    ToolInputStartChunk,
    ToolOutputAvailableChunk,
    ToolOutputErrorChunk,
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

    def encode_event(self, event: BaseChunk, accept: str | None = None) -> str:
        return f'data: {event.model_dump_json(by_alias=True, exclude_none=True)}\n\n'

    async def before_stream(self) -> AsyncIterator[BaseChunk]:
        """Yield events before agent streaming starts."""
        yield StartChunk()

    async def after_stream(self) -> AsyncIterator[BaseChunk]:
        """Yield events after agent streaming completes."""
        yield FinishChunk()

    async def on_error(self, error: Exception) -> AsyncIterator[BaseChunk]:
        """Handle errors during streaming."""
        yield ErrorChunk(error_text=str(error))

    async def handle_text_start(self, part: TextPart, follows_text: bool = False) -> AsyncIterator[BaseChunk]:
        """Handle a TextPart at start."""
        if follows_text:
            message_id = self.message_id
        else:
            message_id = self.new_message_id()
            yield TextStartChunk(id=message_id)

        if part.content:
            yield TextDeltaChunk(id=message_id, delta=part.content)

    async def handle_text_delta(self, delta: TextPartDelta) -> AsyncIterator[BaseChunk]:
        """Handle a TextPartDelta."""
        if delta.content_delta:
            yield TextDeltaChunk(id=self.message_id, delta=delta.content_delta)

    async def handle_text_end(self, part: TextPart, followed_by_text: bool = False) -> AsyncIterator[BaseChunk]:
        """Handle a TextPart at end."""
        if not followed_by_text:
            yield TextEndChunk(id=self.message_id)

    async def handle_thinking_start(
        self, part: ThinkingPart, follows_thinking: bool = False
    ) -> AsyncIterator[BaseChunk]:
        """Handle a ThinkingPart at start."""
        message_id = self.new_message_id()
        yield ReasoningStartChunk(id=message_id)
        if part.content:
            yield ReasoningDeltaChunk(id=message_id, delta=part.content)

    async def handle_thinking_delta(self, delta: ThinkingPartDelta) -> AsyncIterator[BaseChunk]:
        """Handle a ThinkingPartDelta."""
        if delta.content_delta:
            yield ReasoningDeltaChunk(id=self.message_id, delta=delta.content_delta)

    async def handle_thinking_end(
        self, part: ThinkingPart, followed_by_thinking: bool = False
    ) -> AsyncIterator[BaseChunk]:
        """Handle a ThinkingPart at end."""
        yield ReasoningEndChunk(id=self.message_id)

    def handle_tool_call_start(self, part: ToolCallPart | BuiltinToolCallPart) -> AsyncIterator[BaseChunk]:
        """Handle a ToolCallPart or BuiltinToolCallPart at start."""
        return self._handle_tool_call_start(part)

    def handle_builtin_tool_call_start(self, part: BuiltinToolCallPart) -> AsyncIterator[BaseChunk]:
        """Handle a BuiltinToolCallEvent, emitting tool input events."""
        return self._handle_tool_call_start(part, provider_executed=True)

    async def _handle_tool_call_start(
        self,
        part: ToolCallPart | BuiltinToolCallPart,
        tool_call_id: str | None = None,
        provider_executed: bool | None = None,
    ) -> AsyncIterator[BaseChunk]:
        """Handle a ToolCallPart or BuiltinToolCallPart at start."""
        tool_call_id = tool_call_id or part.tool_call_id
        yield ToolInputStartChunk(
            tool_call_id=tool_call_id,
            tool_name=part.tool_name,
            provider_executed=provider_executed,
        )
        if part.args:
            yield ToolInputDeltaChunk(tool_call_id=tool_call_id, input_text_delta=part.args_as_json_str())

    async def handle_tool_call_delta(self, delta: ToolCallPartDelta) -> AsyncIterator[BaseChunk]:
        """Handle a ToolCallPartDelta."""
        tool_call_id = delta.tool_call_id or ''
        assert tool_call_id, '`ToolCallPartDelta.tool_call_id` must be set'
        yield ToolInputDeltaChunk(
            tool_call_id=tool_call_id,
            input_text_delta=delta.args_delta if isinstance(delta.args_delta, str) else _json_dumps(delta.args_delta),
        )

    async def handle_tool_call_end(self, part: ToolCallPart) -> AsyncIterator[BaseChunk]:
        """Handle a ToolCallPart at end."""
        yield ToolInputAvailableChunk(tool_call_id=part.tool_call_id, tool_name=part.tool_name, input=part.args)

    async def handle_builtin_tool_call_end(self, part: BuiltinToolCallPart) -> AsyncIterator[BaseChunk]:
        """Handle a BuiltinToolCallPart at end."""
        yield ToolInputAvailableChunk(
            tool_call_id=part.tool_call_id,
            tool_name=part.tool_name,
            input=part.args,
            provider_executed=True,
            provider_metadata={'pydantic_ai': {'provider_name': part.provider_name}},
        )

    async def handle_builtin_tool_return(self, part: BuiltinToolReturnPart) -> AsyncIterator[BaseChunk]:
        """Handle a BuiltinToolReturnPart."""
        yield ToolOutputAvailableChunk(
            tool_call_id=part.tool_call_id,
            output=part.content,
            provider_executed=True,
        )

    async def handle_file(self, part: FilePart) -> AsyncIterator[BaseChunk]:
        """Handle a FilePart."""
        file = part.content
        yield FileChunk(url=file.data_uri, media_type=file.media_type)

    async def handle_function_tool_result(self, event: FunctionToolResultEvent) -> AsyncIterator[BaseChunk]:
        """Handle a FunctionToolResultEvent, emitting tool result events."""
        result = event.result
        if isinstance(result, RetryPromptPart):
            yield ToolOutputErrorChunk(tool_call_id=result.tool_call_id, error_text=result.model_response())
        else:
            yield ToolOutputAvailableChunk(tool_call_id=result.tool_call_id, output=result.content)

        # TODO (DouweM): Stream ToolCallResultEvent.content as user parts?
