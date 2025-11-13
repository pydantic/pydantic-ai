"""Vercel AI event stream implementation."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic_core import to_json

from ...messages import (
    BaseToolReturnPart,
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
from .request_types import RequestData
from .response_types import (
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
    """UI event stream transformer for the Vercel AI protocol."""

    _step_started: bool = False

    @property
    def response_headers(self) -> Mapping[str, str] | None:
        return VERCEL_AI_DSP_HEADERS

    def encode_event(self, event: BaseChunk) -> str:
        return f'data: {event.encode()}\n\n'

    async def before_stream(self) -> AsyncIterator[BaseChunk]:
        yield StartChunk()

    async def before_response(self) -> AsyncIterator[BaseChunk]:
        if self._step_started:
            yield FinishStepChunk()

        self._step_started = True
        yield StartStepChunk()

    async def after_stream(self) -> AsyncIterator[BaseChunk]:
        yield FinishStepChunk()

        yield FinishChunk()
        yield DoneChunk()

    async def on_error(self, error: Exception) -> AsyncIterator[BaseChunk]:
        yield ErrorChunk(error_text=str(error))

    async def handle_text_start(self, part: TextPart, follows_text: bool = False) -> AsyncIterator[BaseChunk]:
        if follows_text:
            message_id = self.message_id
        else:
            message_id = self.new_message_id()
            yield TextStartChunk(id=message_id)

        if part.content:
            yield TextDeltaChunk(id=message_id, delta=part.content)

    async def handle_text_delta(self, delta: TextPartDelta) -> AsyncIterator[BaseChunk]:
        if delta.content_delta:  # pragma: no branch
            yield TextDeltaChunk(id=self.message_id, delta=delta.content_delta)

    async def handle_text_end(self, part: TextPart, followed_by_text: bool = False) -> AsyncIterator[BaseChunk]:
        if not followed_by_text:
            yield TextEndChunk(id=self.message_id)

    async def handle_thinking_start(
        self, part: ThinkingPart, follows_thinking: bool = False
    ) -> AsyncIterator[BaseChunk]:
        message_id = self.new_message_id()
        yield ReasoningStartChunk(id=message_id)
        if part.content:
            yield ReasoningDeltaChunk(id=message_id, delta=part.content)

    async def handle_thinking_delta(self, delta: ThinkingPartDelta) -> AsyncIterator[BaseChunk]:
        if delta.content_delta:  # pragma: no branch
            yield ReasoningDeltaChunk(id=self.message_id, delta=delta.content_delta)

    async def handle_thinking_end(
        self, part: ThinkingPart, followed_by_thinking: bool = False
    ) -> AsyncIterator[BaseChunk]:
        yield ReasoningEndChunk(id=self.message_id)

    def handle_tool_call_start(self, part: ToolCallPart | BuiltinToolCallPart) -> AsyncIterator[BaseChunk]:
        return self._handle_tool_call_start(part)

    def handle_builtin_tool_call_start(self, part: BuiltinToolCallPart) -> AsyncIterator[BaseChunk]:
        return self._handle_tool_call_start(part, provider_executed=True)

    async def _handle_tool_call_start(
        self,
        part: ToolCallPart | BuiltinToolCallPart,
        tool_call_id: str | None = None,
        provider_executed: bool | None = None,
    ) -> AsyncIterator[BaseChunk]:
        tool_call_id = tool_call_id or part.tool_call_id
        yield ToolInputStartChunk(
            tool_call_id=tool_call_id,
            tool_name=part.tool_name,
            provider_executed=provider_executed,
        )
        if part.args:
            yield ToolInputDeltaChunk(tool_call_id=tool_call_id, input_text_delta=part.args_as_json_str())

    async def handle_tool_call_delta(self, delta: ToolCallPartDelta) -> AsyncIterator[BaseChunk]:
        tool_call_id = delta.tool_call_id or ''
        assert tool_call_id, '`ToolCallPartDelta.tool_call_id` must be set'
        yield ToolInputDeltaChunk(
            tool_call_id=tool_call_id,
            input_text_delta=delta.args_delta if isinstance(delta.args_delta, str) else _json_dumps(delta.args_delta),
        )

    async def handle_tool_call_end(self, part: ToolCallPart) -> AsyncIterator[BaseChunk]:
        yield ToolInputAvailableChunk(
            tool_call_id=part.tool_call_id, tool_name=part.tool_name, input=part.args_as_dict()
        )

    async def handle_builtin_tool_call_end(self, part: BuiltinToolCallPart) -> AsyncIterator[BaseChunk]:
        yield ToolInputAvailableChunk(
            tool_call_id=part.tool_call_id,
            tool_name=part.tool_name,
            input=part.args_as_dict(),
            provider_executed=True,
            provider_metadata={'pydantic_ai': {'provider_name': part.provider_name}},
        )

    async def handle_builtin_tool_return(self, part: BuiltinToolReturnPart) -> AsyncIterator[BaseChunk]:
        yield ToolOutputAvailableChunk(
            tool_call_id=part.tool_call_id,
            output=self._tool_return_output(part),
            provider_executed=True,
        )

    async def handle_file(self, part: FilePart) -> AsyncIterator[BaseChunk]:
        file = part.content
        yield FileChunk(url=file.data_uri, media_type=file.media_type)

    async def handle_function_tool_result(self, event: FunctionToolResultEvent) -> AsyncIterator[BaseChunk]:
        part = event.result
        if isinstance(part, RetryPromptPart):
            yield ToolOutputErrorChunk(tool_call_id=part.tool_call_id, error_text=part.model_response())
        else:
            yield ToolOutputAvailableChunk(tool_call_id=part.tool_call_id, output=self._tool_return_output(part))

        # ToolCallResultEvent.content may hold user parts (e.g. text, images) that Vercel AI does not currently have events for

    def _tool_return_output(self, part: BaseToolReturnPart) -> Any:
        output = part.model_response_object()
        # Unwrap the return value from the output dictionary if it exists
        return output.get('return_value', output)
