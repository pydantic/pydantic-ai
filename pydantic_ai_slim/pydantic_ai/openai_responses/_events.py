"""Translate a Pydantic AI agent run into OpenAI Responses API objects and streaming events."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseError,
    ResponseFailedEvent,
    ResponseInProgressEvent,
    ResponseOutputItem,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseUsage,
)
from openai.types.responses.response_status import ResponseStatus
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from ..messages import AgentStreamEvent, PartDeltaEvent, PartEndEvent, PartStartEvent, TextPart, TextPartDelta
from ..run import AgentRunResultEvent
from ..usage import RunUsage

__all__ = ['build_response', 'build_failed_response', 'build_usage', 'response_event_stream', 'encode_sse']

_OUTPUT_INDEX = 0


def build_usage(usage: RunUsage) -> ResponseUsage:
    """Map a Pydantic AI [`RunUsage`][pydantic_ai.usage.RunUsage] to an OpenAI `ResponseUsage`."""
    return ResponseUsage(
        input_tokens=usage.input_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=usage.cache_read_tokens),
        output_tokens=usage.output_tokens,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=usage.details.get('reasoning_tokens', 0)),
        total_tokens=usage.input_tokens + usage.output_tokens,
    )


def _new_response(
    *,
    response_id: str,
    model: str,
    created_at: float,
    status: ResponseStatus,
    output: Sequence[ResponseOutputItem],
    usage: RunUsage | None = None,
    error: ResponseError | None = None,
) -> Response:
    return Response(
        id=response_id,
        created_at=created_at,
        model=model,
        object='response',
        status=status,
        output=list(output),
        error=error,
        parallel_tool_calls=False,
        tool_choice='auto',
        tools=[],
        usage=build_usage(usage) if usage is not None else None,
    )


def _text_message(message_id: str, text_parts: Sequence[str]) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id=message_id,
        role='assistant',
        status='completed',
        type='message',
        content=[ResponseOutputText(text=text, type='output_text', annotations=[]) for text in text_parts],
    )


def build_response(
    *,
    response_id: str,
    model: str,
    created_at: float,
    text_parts: Sequence[str],
    message_id: str,
    usage: RunUsage | None,
) -> Response:
    """Build a completed OpenAI `Response`.

    Carries a single assistant message with one content part per text segment, or no output at
    all when the run produced no text.
    """
    return _new_response(
        response_id=response_id,
        model=model,
        created_at=created_at,
        status='completed',
        output=[_text_message(message_id, text_parts)] if text_parts else [],
        usage=usage,
    )


def build_failed_response(*, response_id: str, model: str, created_at: float, error: str) -> Response:
    """Build a failed OpenAI `Response` for an error raised during the run."""
    return _new_response(
        response_id=response_id,
        model=model,
        created_at=created_at,
        status='failed',
        output=[],
        error=ResponseError(code='server_error', message=error),
    )


def encode_sse(event: ResponseStreamEvent) -> str:
    """Encode a Responses streaming event as a Server-Sent Events frame."""
    return f'event: {event.type}\ndata: {event.model_dump_json()}\n\n'


class _ResponseEventState:
    def __init__(self, *, response_id: str, model: str, message_id: str, created_at: float) -> None:
        self.response_id = response_id
        self.model = model
        self.message_id = message_id
        self.created_at = created_at
        self.sequence = 0
        self.text_parts: list[str] = []
        self.item_open = False
        self.current_content_index: int | None = None
        self.usage: RunUsage | None = None

    def next_sequence(self) -> int:
        self.sequence, current = self.sequence + 1, self.sequence
        return current

    def in_progress_response(self) -> Response:
        return _new_response(
            response_id=self.response_id, model=self.model, created_at=self.created_at, status='in_progress', output=[]
        )

    def started_events(self) -> list[ResponseStreamEvent]:
        return [
            ResponseCreatedEvent(
                response=self.in_progress_response(), sequence_number=self.next_sequence(), type='response.created'
            ),
            ResponseInProgressEvent(
                response=self.in_progress_response(),
                sequence_number=self.next_sequence(),
                type='response.in_progress',
            ),
        ]

    def open_item(self) -> ResponseOutputItemAddedEvent:
        return ResponseOutputItemAddedEvent(
            item=ResponseOutputMessage(
                id=self.message_id, role='assistant', status='in_progress', type='message', content=[]
            ),
            output_index=_OUTPUT_INDEX,
            sequence_number=self.next_sequence(),
            type='response.output_item.added',
        )

    def add_content_part(self, content_index: int) -> ResponseContentPartAddedEvent:
        return ResponseContentPartAddedEvent(
            content_index=content_index,
            item_id=self.message_id,
            output_index=_OUTPUT_INDEX,
            part=ResponseOutputText(text='', type='output_text', annotations=[]),
            sequence_number=self.next_sequence(),
            type='response.content_part.added',
        )

    def close_current_part(self) -> list[ResponseStreamEvent]:
        if self.current_content_index is None:
            return []

        content_index = self.current_content_index
        self.current_content_index = None
        text = self.text_parts[content_index]
        return [
            ResponseTextDoneEvent(
                content_index=content_index,
                item_id=self.message_id,
                logprobs=[],
                output_index=_OUTPUT_INDEX,
                text=text,
                sequence_number=self.next_sequence(),
                type='response.output_text.done',
            ),
            ResponseContentPartDoneEvent(
                content_index=content_index,
                item_id=self.message_id,
                output_index=_OUTPUT_INDEX,
                part=ResponseOutputText(text=text, type='output_text', annotations=[]),
                sequence_number=self.next_sequence(),
                type='response.content_part.done',
            ),
        ]

    def start_text_part(self) -> list[ResponseStreamEvent]:
        response_events: list[ResponseStreamEvent] = []
        if not self.item_open:
            self.item_open = True
            response_events.append(self.open_item())

        self.current_content_index = len(self.text_parts)
        self.text_parts.append('')
        response_events.append(self.add_content_part(self.current_content_index))
        return response_events

    def add_text_delta(self, delta: str) -> list[ResponseStreamEvent]:
        if not delta:
            return []

        response_events: list[ResponseStreamEvent] = []
        if self.current_content_index is None:
            response_events.extend(self.start_text_part())

        content_index = self.current_content_index
        assert content_index is not None
        self.text_parts[content_index] += delta
        response_events.append(
            ResponseTextDeltaEvent(
                content_index=content_index,
                delta=delta,
                item_id=self.message_id,
                logprobs=[],
                output_index=_OUTPUT_INDEX,
                sequence_number=self.next_sequence(),
                type='response.output_text.delta',
            )
        )
        return response_events

    def handle_event(self, event: AgentStreamEvent | AgentRunResultEvent[Any]) -> list[ResponseStreamEvent]:
        if isinstance(event, AgentRunResultEvent):
            self.usage = event.result.usage
            output = event.result.output
            if not self.item_open and isinstance(output, str):
                return self.add_text_delta(output)
            return []

        if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
            # A new text part only opens a Responses content part once it has content, so a
            # text part that stays empty (e.g. alongside a tool call) is never surfaced.
            return [*self.close_current_part(), *self.add_text_delta(event.part.content)]
        if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
            return self.add_text_delta(event.delta.content_delta)
        if isinstance(event, PartEndEvent) and isinstance(event.part, TextPart):
            return self.close_current_part()
        return []

    def finish_events(self) -> list[ResponseStreamEvent]:
        response_events = self.close_current_part()
        if self.item_open:
            response_events.append(
                ResponseOutputItemDoneEvent(
                    item=_text_message(self.message_id, self.text_parts),
                    output_index=_OUTPUT_INDEX,
                    sequence_number=self.next_sequence(),
                    type='response.output_item.done',
                )
            )
        response_events.append(
            ResponseCompletedEvent(
                response=build_response(
                    response_id=self.response_id,
                    model=self.model,
                    created_at=self.created_at,
                    text_parts=self.text_parts,
                    message_id=self.message_id,
                    usage=self.usage,
                ),
                sequence_number=self.next_sequence(),
                type='response.completed',
            )
        )
        return response_events

    def failed_event(self, error: Exception) -> ResponseFailedEvent:
        return ResponseFailedEvent(
            response=build_failed_response(
                response_id=self.response_id, model=self.model, created_at=self.created_at, error=str(error)
            ),
            sequence_number=self.next_sequence(),
            type='response.failed',
        )


async def response_event_stream(
    events: AsyncIterator[AgentStreamEvent | AgentRunResultEvent[Any]],
    *,
    model: str,
    response_id: str,
    message_id: str,
    created_at: float,
    catch_run_errors: bool = True,
) -> AsyncIterator[ResponseStreamEvent]:
    """Translate a Pydantic AI agent event stream into OpenAI Responses streaming events.

    The agent runs its tool loop server-side, so only its assistant text is surfaced. Each
    Pydantic AI `TextPart` becomes its own `output_text` content part within one assistant
    message. A streaming run that raises ends with `response.failed` instead.
    """
    state = _ResponseEventState(response_id=response_id, model=model, message_id=message_id, created_at=created_at)
    for event in state.started_events():
        yield event

    try:
        async for event in events:
            for response_event in state.handle_event(event):
                yield response_event
    except Exception as error:
        if not catch_run_errors:
            raise
        yield state.failed_event(error)
        return

    for response_event in state.finish_events():
        yield response_event
