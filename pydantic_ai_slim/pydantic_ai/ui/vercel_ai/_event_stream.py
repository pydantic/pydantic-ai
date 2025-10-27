"""Vercel AI event stream implementation."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
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
from ...output import OutputDataT
from ...tools import AgentDepsT
from .. import UIEventStream
from ._request_types import RequestData
from ._response_types import (
    BaseChunk,
    DoneChunk,
    ErrorChunk,
    FileChunk,
    FinishChunk,
    FinishStepChunk,
    ReasoningDeltaChunk,
    ReasoningEndChunk,
    ReasoningStartChunk,
    StartChunk,
    StartStepChunk,
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

# See https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol#data-stream-protocol
VERCEL_AI_DSP_HEADERS = {'x-vercel-ai-ui-message-stream': 'v1'}


def _json_dumps(obj: Any) -> str:
    """Dump an object to JSON string."""
    return to_json(obj).decode('utf-8')


@dataclass
class VercelAIEventStream(UIEventStream[RequestData, BaseChunk, AgentDepsT, OutputDataT]):
    """TODO (DouweM): Docstring."""

    _step_started: bool = False

    @property
    def response_headers(self) -> Mapping[str, str] | None:
        """Get the response headers for the adapter."""
        return VERCEL_AI_DSP_HEADERS

    def encode_event(self, event: BaseChunk) -> str:
        if isinstance(event, DoneChunk):
            return 'data: [DONE]\n\n'
        return f'data: {event.model_dump_json(by_alias=True, exclude_none=True)}\n\n'

    async def before_stream(self) -> AsyncIterator[BaseChunk]:
        """Yield events before agent streaming starts."""
        yield StartChunk()

    async def before_response(self) -> AsyncIterator[BaseChunk]:
        """Yield events before the request is processed."""
        self._step_started = True
        yield StartStepChunk()

    async def after_request(self) -> AsyncIterator[BaseChunk]:
        """Yield events after the response is processed."""
        if self._step_started:  # TODO (DouweM): coverage
            yield FinishStepChunk()
            self._step_started = False

    async def after_stream(self) -> AsyncIterator[BaseChunk]:
        """Yield events after agent streaming completes."""
        if self._step_started:  # TODO (DouweM): coverage branch
            yield FinishStepChunk()

        yield FinishChunk()
        yield DoneChunk()

    async def on_error(self, error: Exception) -> AsyncIterator[BaseChunk]:
        """Handle errors during streaming."""
        yield ErrorChunk(error_text=str(error))

    async def handle_text_start(self, part: TextPart, follows_text: bool = False) -> AsyncIterator[BaseChunk]:
        """Handle a TextPart at start."""
        if follows_text:
            message_id = self.message_id  # TODO (DouweM): coverage
        else:
            message_id = self.new_message_id()
            yield TextStartChunk(id=message_id)

        if part.content:  # TODO (DouweM): coverage branch
            yield TextDeltaChunk(id=message_id, delta=part.content)

    async def handle_text_delta(self, delta: TextPartDelta) -> AsyncIterator[BaseChunk]:
        """Handle a TextPartDelta."""
        if delta.content_delta:  # TODO (DouweM): coverage branch
            yield TextDeltaChunk(id=self.message_id, delta=delta.content_delta)

    async def handle_text_end(self, part: TextPart, followed_by_text: bool = False) -> AsyncIterator[BaseChunk]:
        """Handle a TextPart at end."""
        if not followed_by_text:  # TODO (DouweM): coverage branch
            yield TextEndChunk(id=self.message_id)

    async def handle_thinking_start(
        self, part: ThinkingPart, follows_thinking: bool = False
    ) -> AsyncIterator[BaseChunk]:
        """Handle a ThinkingPart at start."""
        message_id = self.new_message_id()
        yield ReasoningStartChunk(id=message_id)
        if part.content:
            yield ReasoningDeltaChunk(id=message_id, delta=part.content)  # TODO (DouweM): coverage

    async def handle_thinking_delta(self, delta: ThinkingPartDelta) -> AsyncIterator[BaseChunk]:
        """Handle a ThinkingPartDelta."""
        if delta.content_delta:  # TODO (DouweM): coverage
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
            yield ToolInputDeltaChunk(
                tool_call_id=tool_call_id, input_text_delta=part.args_as_json_str()
            )  # TODO (DouweM): coverage

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
        yield ToolInputAvailableChunk(
            tool_call_id=part.tool_call_id, tool_name=part.tool_name, input=part.args
        )  # TODO (DouweM): coverage

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
        yield FileChunk(url=file.data_uri, media_type=file.media_type)  # TODO (DouweM): coverage

    async def handle_function_tool_result(
        self, event: FunctionToolResultEvent
    ) -> AsyncIterator[BaseChunk]:  # TODO (DouweM): coverage
        """Handle a FunctionToolResultEvent, emitting tool result events."""
        result = event.result
        if isinstance(result, RetryPromptPart):
            yield ToolOutputErrorChunk(tool_call_id=result.tool_call_id, error_text=result.model_response())
        else:
            yield ToolOutputAvailableChunk(tool_call_id=result.tool_call_id, output=result.content)

        # ToolCallResultEvent.content may hold user parts (e.g. text, images) that Vercel AI does not currently have events for
