"""`OpenAIResponsesStreamedResponse` collects raw reasoning text (`response.reasoning_text.delta`) into `raw_content`.

gpt-oss models send it, and `ThinkingPart.provider_details['raw_content']` keeps one entry per `content_index`. The
tests use synthetic events: they pin event-level output and inputs no recorded model sends, a `content_index` past the
next entry or before the first. `test_openai_responses_stream_parity.py` replays the recorded gpt-oss streams.
"""

from __future__ import annotations as _annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Any

import pytest

from pydantic_ai._parts_manager import ModelResponsePartsManager
from pydantic_ai.messages import (
    FinalResultEvent,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    ProviderDetailsDelta,
    TextPart,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters

from .._inline_snapshot import snapshot
from ..conftest import try_import

with try_import() as imports_successful:
    from openai.types import responses

    import pydantic_ai.models.openai._responses_streaming as responses_streaming
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

    from .mock_openai import MockOpenAIResponses, response_message

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def _text_delta(item_id: str, text: str) -> responses.ResponseTextDeltaEvent:
    return responses.ResponseTextDeltaEvent(
        content_index=0,
        delta=text,
        item_id=item_id,
        output_index=1,
        logprobs=[],
        type='response.output_text.delta',
        sequence_number=0,
    )


def _raw_reasoning_stream(
    deltas: Sequence[tuple[int, str] | responses.ResponseStreamEvent],
    *,
    encrypted_content: str | None = None,
    text: str | None = None,
) -> list[responses.ResponseStreamEvent]:
    """A reasoning item `rs_1`, then an optional message with `text`.

    Each `(content_index, delta)` pair in `deltas` is a raw reasoning delta of `rs_1`; each event is sent as it is.
    """
    reasoning = responses.ResponseReasoningItem(id='rs_1', summary=[], type='reasoning')
    events: list[responses.ResponseStreamEvent] = [
        responses.ResponseOutputItemAddedEvent(
            item=reasoning, output_index=0, type='response.output_item.added', sequence_number=0
        ),
        *(
            responses.ResponseReasoningTextDeltaEvent(
                content_index=delta[0],
                delta=delta[1],
                item_id='rs_1',
                output_index=0,
                type='response.reasoning_text.delta',
                sequence_number=0,
            )
            if isinstance(delta, tuple)
            else delta
            for delta in deltas
        ),
        responses.ResponseOutputItemDoneEvent(
            item=reasoning.model_copy(update={'encrypted_content': encrypted_content}),
            output_index=0,
            type='response.output_item.done',
            sequence_number=0,
        ),
    ]
    if text is not None:
        events.append(_text_delta('msg_1', text))
    response = response_message([])
    return [
        responses.ResponseCreatedEvent(response=response, type='response.created', sequence_number=0),
        *(event.model_copy(update={'sequence_number': index}) for index, event in enumerate(events, 1)),
        responses.ResponseCompletedEvent(
            response=response.model_copy(update={'status': 'completed'}),
            type='response.completed',
            sequence_number=len(events) + 1,
        ),
    ]


@dataclass
class _Recorded:
    events: list[ModelResponseStreamEvent]
    """The events as yielded, each delta's `provider_details` updater replaced by what it gives the consumer's part."""
    raw_content: list[object]
    """`raw_content` of the thinking part in `get()` after each event, when it's read while streaming."""
    response: ModelResponse


async def _record(stream: list[responses.ResponseStreamEvent], *, read_while_streaming: bool = True) -> _Recorded:
    model = OpenAIResponsesModel(
        'gpt-oss-120b', provider=OpenAIProvider(openai_client=MockOpenAIResponses.create_mock_stream(stream))
    )
    events: list[ModelResponseStreamEvent] = []
    raw_content: list[object] = []
    consumer_parts: dict[int, ModelResponsePart] = {}
    async with model.request_stream(
        [ModelRequest(parts=[UserPromptPart(content='hi')])], None, ModelRequestParameters()
    ) as streamed:
        async for event in streamed:
            if isinstance(event, PartStartEvent):
                consumer_parts[event.index] = event.part
            elif isinstance(event, PartDeltaEvent):
                # A consumer that rebuilds the parts from the events applies the updater to its own copy.
                consumer_part = consumer_parts[event.index] = event.delta.apply(consumer_parts[event.index])
                if isinstance(event.delta, ThinkingPartDelta) and callable(event.delta.provider_details):
                    event = replace(event, delta=replace(event.delta, provider_details=consumer_part.provider_details))
            events.append(event)
            if read_while_streaming:
                raw_content.append(
                    next(
                        (
                            (p.provider_details or {}).get('raw_content')
                            for p in streamed.get().parts
                            if isinstance(p, ThinkingPart)
                        ),
                        None,
                    )
                )
    response = streamed.get()
    assert list(consumer_parts.values()) == response.parts
    return _Recorded(events, raw_content, response)


async def test_in_order_stream(allow_model_requests: None):
    """One entry per `content_index`, built up as the deltas arrive: in every event, in `get()`, and in the response."""
    recorded = await _record(
        _raw_reasoning_stream(
            [(0, 'Let me '), (0, 'think.'), (1, 'The answer '), (1, 'is 4.')], encrypted_content='sig', text='4'
        )
    )
    assert recorded.events == snapshot(
        [
            PartStartEvent(
                index=0,
                part=ThinkingPart(
                    content='', id='rs_1', provider_name='openai', provider_details={'raw_content': ['Let me ']}
                ),
                canonical_system='openai',
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(provider_details={'raw_content': ['Let me think.']})),
            PartDeltaEvent(
                index=0, delta=ThinkingPartDelta(provider_details={'raw_content': ['Let me think.', 'The answer ']})
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(provider_details={'raw_content': ['Let me think.', 'The answer is 4.']}),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(signature_delta='sig')),
            PartEndEvent(
                index=0,
                part=ThinkingPart(
                    content='',
                    id='rs_1',
                    signature='sig',
                    provider_name='openai',
                    provider_details={'raw_content': ['Let me think.', 'The answer is 4.']},
                ),
                next_part_kind='text',
                canonical_system='openai',
            ),
            PartStartEvent(
                index=1,
                part=TextPart(content='4', id='msg_1', provider_name='openai'),
                previous_part_kind='thinking',
                canonical_system='openai',
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartEndEvent(
                index=1, part=TextPart(content='4', id='msg_1', provider_name='openai'), canonical_system='openai'
            ),
        ]
    )
    assert recorded.raw_content == snapshot(
        [
            ['Let me '],
            ['Let me think.'],
            ['Let me think.', 'The answer '],
            ['Let me think.', 'The answer is 4.'],
            ['Let me think.', 'The answer is 4.'],
            ['Let me think.', 'The answer is 4.'],
            ['Let me think.', 'The answer is 4.'],
            ['Let me think.', 'The answer is 4.'],
            ['Let me think.', 'The answer is 4.'],
        ]
    )
    assert recorded.response.parts == snapshot(
        [
            ThinkingPart(
                content='',
                id='rs_1',
                signature='sig',
                provider_name='openai',
                provider_details={'raw_content': ['Let me think.', 'The answer is 4.']},
            ),
            TextPart(content='4', id='msg_1', provider_name='openai'),
        ]
    )


async def test_content_index_past_the_next_entry(allow_model_requests: None):
    """Each `content_index` gets one entry, in the order the indices first arrive, rather than the list being padded."""
    recorded = await _record(_raw_reasoning_stream([(1_000_000, 'a'), (1_000_000, 'b'), (3, 'c'), (1, 'd'), (3, 'e')]))
    assert recorded.response.parts == snapshot(
        [
            ThinkingPart(
                content='', id='rs_1', provider_name='openai', provider_details={'raw_content': ['ab', 'ce', 'd']}
            )
        ]
    )


async def test_negative_content_index(allow_model_requests: None):
    """A negative index gets an entry of its own too, rather than counting from the end of the list."""
    recorded = await _record(_raw_reasoning_stream([(-1, 'a'), (0, 'b'), (-1, 'c')]))
    assert recorded.response.parts == snapshot(
        [ThinkingPart(content='', id='rs_1', provider_name='openai', provider_details={'raw_content': ['ac', 'b']})]
    )


async def test_item_reusing_the_reasoning_item_id(allow_model_requests: None):
    """A function call that reuses the reasoning item's id gets a part of its own, and the reasoning part keeps its text.

    Nothing reads the parts until the stream ends, so the reasoning part's raw text is still waiting to be written
    when the call takes over the id.
    """
    call = responses.ResponseFunctionToolCall(
        id='rs_1', call_id='call_1', name='lookup', arguments='{}', type='function_call'
    )
    added = responses.ResponseOutputItemAddedEvent(
        item=call, output_index=1, type='response.output_item.added', sequence_number=0
    )
    recorded = await _record(_raw_reasoning_stream([(0, 'a'), (0, 'b'), added]), read_while_streaming=False)
    assert recorded.response.parts == snapshot(
        [
            ThinkingPart(content='', id='rs_1', provider_name='openai', provider_details={'raw_content': ['ab']}),
            ToolCallPart(tool_name='lookup', args='{}', tool_call_id='call_1', id='rs_1', provider_name='openai'),
        ]
    )


@pytest.mark.parametrize('between_parts', [False, True], ids=['deltas', 'deltas-between-new-parts'])
@pytest.mark.parametrize('deltas', [10, 100])
async def test_raw_content_is_not_rebuilt_per_delta(
    allow_model_requests: None, monkeypatch: pytest.MonkeyPatch, deltas: int, between_parts: bool
):
    """The stream applies only the first delta's updater and writes the rest into the part once, so it takes linear time.

    Each delta event still carries an updater, which a consumer that rebuilds the parts applies itself. Both counts
    stay put at both sizes, also when another part starts after each delta, where rebuilding `raw_content` for each
    delta or each new part would make them grow with the number of deltas.
    """
    calls = 0
    written = 0
    make_updater = responses_streaming.make_raw_content_updater

    def counting_updater(delta: str, index: int) -> Callable[[dict[str, Any] | None], dict[str, Any]]:
        update = make_updater(delta, index)

        def counted(existing: dict[str, Any] | None) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            return update(existing)

        return counted

    handle_thinking_delta = ModelResponsePartsManager.handle_thinking_delta

    def counting_handle_thinking_delta(
        self: ModelResponsePartsManager, *, provider_details: ProviderDetailsDelta = None, **kwargs: Any
    ) -> Iterator[ModelResponseStreamEvent]:
        nonlocal written
        if provider_details is not None and not callable(provider_details):
            raw_content: list[str] = provider_details['raw_content']
            written += sum(len(entry) for entry in raw_content)
        return handle_thinking_delta(self, provider_details=provider_details, **kwargs)

    monkeypatch.setattr(responses_streaming, 'make_raw_content_updater', counting_updater)
    monkeypatch.setattr(ModelResponsePartsManager, 'handle_thinking_delta', counting_handle_thinking_delta)
    stream = _raw_reasoning_stream(
        [delta for i in range(deltas) for delta in ((0, 'x'), _text_delta(f'msg_{i}', 'y'))]
        if between_parts
        else [(0, 'x')] * deltas
    )
    model = OpenAIResponsesModel(
        'gpt-oss-120b', provider=OpenAIProvider(openai_client=MockOpenAIResponses.create_mock_stream(stream))
    )
    async with model.request_stream(
        [ModelRequest(parts=[UserPromptPart(content='hi')])], None, ModelRequestParameters()
    ) as streamed:
        async for _ in streamed:
            pass
    assert streamed.get().parts[0] == ThinkingPart(
        content='', id='rs_1', provider_name='openai', provider_details={'raw_content': ['x' * deltas]}
    )
    assert (calls, written) == (1, deltas)
