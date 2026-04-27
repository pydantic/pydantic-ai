"""OpenAI Responses protocol event stream transformer for Pydantic AI agents."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

try:
    from openai.types.responses import (
        Response as ResponseObject,
        ResponseCompletedEvent,
        ResponseContentPartAddedEvent,
        ResponseContentPartDoneEvent,
        ResponseCreatedEvent,
        ResponseFunctionToolCall,
        ResponseOutputItemAddedEvent,
        ResponseOutputItemDoneEvent,
        ResponseOutputMessage,
        ResponseOutputText,
        ResponseTextDeltaEvent,
        ResponseTextDoneEvent,
        ResponseUsage,
    )
    from openai.types.responses.response_create_params import ResponseCreateParamsStreaming
    from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the Responses integration, '
        'you can use the `responses` optional group — `pip install "pydantic-ai-slim[responses]"`'
    ) from e

from ...messages import (
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)
from ...output import OutputDataT
from ...tools import AgentDepsT
from .. import UIEventStream

__all__ = ['ResponsesEventStream']


@dataclass
class ResponsesEventStream(UIEventStream[ResponseCreateParamsStreaming, Any, AgentDepsT, OutputDataT]):
    """UI event stream transformer for the OpenAI Responses protocol."""

    _response_id: str = ''
    _text_item_id: str = ''
    _output_index: int = 0
    _content_part_added: bool = False
    _message_item_added: bool = False
    _sequence_number: int = 0
    _error: bool = False
    _accumulated_text: list[str] = field(default_factory=list[str])

    @property
    def content_type(self) -> str:
        return 'text/event-stream'

    def encode_event(self, event: Any) -> str:
        event_type = getattr(event, 'type', 'message')
        event_data = event.model_dump_json(exclude_unset=True)
        return f'event: {event_type}\ndata: {event_data}\n\n'

    async def encode_stream(self, stream: AsyncIterator[Any]) -> AsyncIterator[str]:
        async for event in stream:
            yield self.encode_event(event)
        yield 'event: done\ndata: [DONE]\n\n'

    def _next_seq(self) -> int:
        n = self._sequence_number
        self._sequence_number += 1
        return n

    def _model_name(self) -> str:
        model = self.run_input.get('model')
        return str(model) if model is not None else 'unknown'

    def _instructions(self) -> str | list[Any] | None:
        return self.run_input.get('instructions')

    def _empty_usage(self) -> ResponseUsage:
        return ResponseUsage(
            input_tokens=0,
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens=0,
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
            total_tokens=0,
        )

    def _build_response(self, *, status: str, usage: ResponseUsage) -> ResponseObject:
        # `tool_choice` and `tools` come from the request-side TypedDict union, which is
        # structurally compatible with but distinct from the response-side pydantic union;
        # the runtime accepts both, the static unions don't intersect.
        return ResponseObject(
            id=self._response_id,
            object='response',
            created_at=time.time(),
            model=self._model_name(),
            output=[],
            status=status,  # pyright: ignore[reportArgumentType]
            usage=usage,
            instructions=self._instructions(),
            parallel_tool_calls=bool(self.run_input.get('parallel_tool_calls', True)),
            tool_choice=self.run_input.get('tool_choice') or 'auto',  # pyright: ignore[reportArgumentType]
            tools=self.run_input.get('tools') or [],  # pyright: ignore[reportArgumentType]
        )

    async def before_stream(self) -> AsyncIterator[Any]:
        self._response_id = self.new_message_id()
        self._text_item_id = self.new_message_id()
        yield ResponseCreatedEvent(
            type='response.created',
            response=self._build_response(status='in_progress', usage=self._empty_usage()),
            sequence_number=self._next_seq(),
        )

    async def after_stream(self) -> AsyncIterator[Any]:
        # Close any still-open content/output items so the spec ordering is preserved.
        async for ev in self._close_open_message_item():
            yield ev

        usage = self._empty_usage()
        if self._result is not None:
            run_usage = self._result.usage()
            usage = ResponseUsage(
                input_tokens=run_usage.input_tokens or 0,
                input_tokens_details=InputTokensDetails(cached_tokens=run_usage.cache_read_tokens or 0),
                output_tokens=run_usage.output_tokens or 0,
                output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                total_tokens=run_usage.total_tokens or 0,
            )

        status = 'failed' if self._error else 'completed'
        yield ResponseCompletedEvent(
            type='response.completed',
            response=self._build_response(status=status, usage=usage),
            sequence_number=self._next_seq(),
        )

    async def on_error(self, error: Exception) -> AsyncIterator[Any]:
        self._error = True
        return
        yield  # pyright: ignore[reportUnreachable]

    async def handle_text_start(self, part: TextPart, follows_text: bool = False) -> AsyncIterator[Any]:
        if not self._message_item_added:
            yield ResponseOutputItemAddedEvent(
                type='response.output_item.added',
                item=ResponseOutputMessage(
                    type='message',
                    id=self._text_item_id,
                    role='assistant',
                    status='in_progress',
                    content=[],
                ),
                output_index=self._output_index,
                sequence_number=self._next_seq(),
            )
            self._message_item_added = True

        if not self._content_part_added:
            yield ResponseContentPartAddedEvent(
                type='response.content_part.added',
                item_id=self._text_item_id,
                output_index=self._output_index,
                content_index=0,
                part=ResponseOutputText(type='output_text', text='', annotations=[]),
                sequence_number=self._next_seq(),
            )
            self._content_part_added = True

        if part.content:
            self._accumulated_text.append(part.content)
            yield ResponseTextDeltaEvent(
                type='response.output_text.delta',
                item_id=self._text_item_id,
                output_index=self._output_index,
                content_index=0,
                delta=part.content,
                logprobs=[],
                sequence_number=self._next_seq(),
            )

    async def handle_text_delta(self, delta: TextPartDelta) -> AsyncIterator[Any]:
        if not delta.content_delta:
            return
        self._accumulated_text.append(delta.content_delta)
        yield ResponseTextDeltaEvent(
            type='response.output_text.delta',
            item_id=self._text_item_id,
            output_index=self._output_index,
            content_index=0,
            delta=delta.content_delta,
            logprobs=[],
            sequence_number=self._next_seq(),
        )

    async def handle_text_end(self, part: TextPart, followed_by_text: bool = False) -> AsyncIterator[Any]:
        if followed_by_text:
            return
        async for ev in self._close_open_message_item():
            yield ev

    async def _close_open_message_item(self) -> AsyncIterator[Any]:
        if not self._message_item_added:
            return

        full_text = ''.join(self._accumulated_text)
        if self._content_part_added:
            yield ResponseTextDoneEvent(
                type='response.output_text.done',
                item_id=self._text_item_id,
                output_index=self._output_index,
                content_index=0,
                text=full_text,
                logprobs=[],
                sequence_number=self._next_seq(),
            )
            yield ResponseContentPartDoneEvent(
                type='response.content_part.done',
                item_id=self._text_item_id,
                output_index=self._output_index,
                content_index=0,
                part=ResponseOutputText(type='output_text', text=full_text, annotations=[]),
                sequence_number=self._next_seq(),
            )
            self._content_part_added = False

        yield ResponseOutputItemDoneEvent(
            type='response.output_item.done',
            item=ResponseOutputMessage(
                type='message',
                id=self._text_item_id,
                role='assistant',
                status='completed',
                content=[ResponseOutputText(type='output_text', text=full_text, annotations=[])],
            ),
            output_index=self._output_index,
            sequence_number=self._next_seq(),
        )

        self._message_item_added = False
        self._accumulated_text = []
        self._output_index += 1

    async def handle_tool_call_start(self, part: ToolCallPart) -> AsyncIterator[Any]:
        # If a message item is still open, finalise it before starting a tool-call output item.
        async for ev in self._close_open_message_item():
            yield ev

        tool_call_item = ResponseFunctionToolCall(
            type='function_call',
            call_id=part.tool_call_id,
            name=part.tool_name,
            arguments=part.args_as_json_str(),
        )

        yield ResponseOutputItemAddedEvent(
            type='response.output_item.added',
            item=tool_call_item,
            output_index=self._output_index,
            sequence_number=self._next_seq(),
        )
        yield ResponseOutputItemDoneEvent(
            type='response.output_item.done',
            item=tool_call_item,
            output_index=self._output_index,
            sequence_number=self._next_seq(),
        )
        self._output_index += 1

    async def handle_tool_call_delta(self, delta: ToolCallPartDelta) -> AsyncIterator[Any]:
        # Tool call arguments are sent in full at start; streaming deltas are not surfaced.
        return
        yield  # pyright: ignore[reportUnreachable]

    async def handle_tool_call_end(self, part: ToolCallPart) -> AsyncIterator[Any]:
        return
        yield  # pyright: ignore[reportUnreachable]
