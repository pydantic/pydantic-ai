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

from ..messages import AgentStreamEvent, PartDeltaEvent, PartStartEvent, TextPart, TextPartDelta
from ..run import AgentRunResultEvent
from ..usage import RunUsage

__all__ = ['build_response', 'build_failed_response', 'build_usage', 'response_event_stream', 'encode_sse']

_OUTPUT_INDEX = 0
_CONTENT_INDEX = 0


def build_usage(usage: RunUsage) -> ResponseUsage:
    """Map a Pydantic AI [`RunUsage`][pydantic_ai.usage.RunUsage] to an OpenAI `ResponseUsage`."""
    return ResponseUsage(
        input_tokens=usage.input_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=usage.cache_read_tokens),
        output_tokens=usage.output_tokens,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
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


def _text_message(message_id: str, text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id=message_id,
        role='assistant',
        status='completed',
        type='message',
        content=[ResponseOutputText(text=text, type='output_text', annotations=[])],
    )


def build_response(
    *, response_id: str, model: str, created_at: float, text: str, message_id: str, usage: RunUsage | None
) -> Response:
    """Build a completed OpenAI `Response`.

    Carries a single assistant text message, or no output at all when the run produced no text.
    """
    return _new_response(
        response_id=response_id,
        model=model,
        created_at=created_at,
        status='completed',
        output=[_text_message(message_id, text)] if text else [],
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
    """Encode a Responses streaming event as a Server-Sent Events `data:` line."""
    return f'data: {event.model_dump_json()}\n\n'


async def response_event_stream(
    events: AsyncIterator[AgentStreamEvent | AgentRunResultEvent[Any]],
    *,
    model: str,
    response_id: str,
    message_id: str,
    created_at: float,
) -> AsyncIterator[ResponseStreamEvent]:
    """Translate a Pydantic AI agent event stream into OpenAI Responses streaming events.

    The agent runs its tool loop server-side, so only its assistant text is surfaced: it is
    accumulated into a single output message that opens on the first non-empty token. The sequence
    is `response.created` → `response.in_progress` → (lazily) `response.output_item.added` →
    `response.content_part.added` → `response.output_text.delta`… → the matching `done` events →
    `response.completed`. A run that raises ends with `response.failed` instead.
    """
    sequence = 0

    def next_sequence() -> int:
        nonlocal sequence
        sequence, current = sequence + 1, sequence
        return current

    def in_progress() -> Response:
        return _new_response(
            response_id=response_id, model=model, created_at=created_at, status='in_progress', output=[]
        )

    yield ResponseCreatedEvent(response=in_progress(), sequence_number=next_sequence(), type='response.created')
    yield ResponseInProgressEvent(response=in_progress(), sequence_number=next_sequence(), type='response.in_progress')

    text = ''
    item_open = False
    usage: RunUsage | None = None

    try:
        async for event in events:
            if isinstance(event, AgentRunResultEvent):
                usage = event.result.usage
                continue

            if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                delta = event.part.content
            elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                delta = event.delta.content_delta
            else:
                continue

            if not delta:
                continue

            if not item_open:
                item_open = True
                yield ResponseOutputItemAddedEvent(
                    item=ResponseOutputMessage(
                        id=message_id, role='assistant', status='in_progress', type='message', content=[]
                    ),
                    output_index=_OUTPUT_INDEX,
                    sequence_number=next_sequence(),
                    type='response.output_item.added',
                )
                yield ResponseContentPartAddedEvent(
                    content_index=_CONTENT_INDEX,
                    item_id=message_id,
                    output_index=_OUTPUT_INDEX,
                    part=ResponseOutputText(text='', type='output_text', annotations=[]),
                    sequence_number=next_sequence(),
                    type='response.content_part.added',
                )

            text += delta
            yield ResponseTextDeltaEvent(
                content_index=_CONTENT_INDEX,
                delta=delta,
                item_id=message_id,
                logprobs=[],
                output_index=_OUTPUT_INDEX,
                sequence_number=next_sequence(),
                type='response.output_text.delta',
            )
    except Exception as error:
        yield ResponseFailedEvent(
            response=build_failed_response(
                response_id=response_id, model=model, created_at=created_at, error=str(error)
            ),
            sequence_number=next_sequence(),
            type='response.failed',
        )
        return

    if item_open:
        yield ResponseTextDoneEvent(
            content_index=_CONTENT_INDEX,
            item_id=message_id,
            logprobs=[],
            output_index=_OUTPUT_INDEX,
            text=text,
            sequence_number=next_sequence(),
            type='response.output_text.done',
        )
        yield ResponseContentPartDoneEvent(
            content_index=_CONTENT_INDEX,
            item_id=message_id,
            output_index=_OUTPUT_INDEX,
            part=ResponseOutputText(text=text, type='output_text', annotations=[]),
            sequence_number=next_sequence(),
            type='response.content_part.done',
        )
        yield ResponseOutputItemDoneEvent(
            item=_text_message(message_id, text),
            output_index=_OUTPUT_INDEX,
            sequence_number=next_sequence(),
            type='response.output_item.done',
        )

    yield ResponseCompletedEvent(
        response=build_response(
            response_id=response_id, model=model, created_at=created_at, text=text, message_id=message_id, usage=usage
        ),
        sequence_number=next_sequence(),
        type='response.completed',
    )
