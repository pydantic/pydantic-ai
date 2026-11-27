"""The OpenAI Chat Completions and Responses streams collect the text they keep across deltas in linear time.

Appending each delta to a string with `+=` copies the whole string each time, so the work grows with the square of
the text's length. The tests stream deltas as a `str` subclass that counts the characters each concatenation onto it
copies, which turns that cost into an operation count rather than a timing: a linear stream copies each character at
most once. They use synthetic events, since the cost only shows on streams far longer than any recording.
"""

from __future__ import annotations as _annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pytest

from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models import Model, ModelRequestParameters

from ..conftest import try_import

with try_import() as imports_successful:
    from openai.types import chat, responses
    from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice, ChoiceDelta

    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.profiles.openai import OpenAIModelProfile
    from pydantic_ai.providers.openai import OpenAIProvider

    from .mock_openai import MockOpenAI, MockOpenAIResponses, response_message

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')

DELTA = 'abcdefgh'


@dataclass
class _Copies:
    chars: int = 0

    def delta(self, text: str = DELTA) -> str:
        """`text` as a delta that adds the length of the string it's appended to to `chars`."""
        copies = self

        class CountingStr(str):
            def __radd__(self, other: str) -> str:
                # Python calls this before `str.__add__` for `other + self` and `other += self`.
                copies.chars += len(other)
                return other + str(self)

        return CountingStr(text)


def _responses_model(events: Sequence[responses.ResponseStreamEvent]) -> Model:
    response = response_message([])
    stream = [
        responses.ResponseCreatedEvent(response=response, type='response.created', sequence_number=0),
        *(event.model_copy(update={'sequence_number': index}) for index, event in enumerate(events, 1)),
        responses.ResponseCompletedEvent(
            response=response.model_copy(update={'status': 'completed'}),
            type='response.completed',
            sequence_number=len(events) + 1,
        ),
    ]
    return OpenAIResponsesModel(
        'gpt-5', provider=OpenAIProvider(openai_client=MockOpenAIResponses.create_mock_stream(stream))
    )


def _chat_model(deltas: list[ChoiceDelta], profile: OpenAIModelProfile | None = None) -> Model:
    chunks = [
        chat.ChatCompletionChunk(
            id='chunk',
            created=0,
            model='gpt-4o',
            object='chat.completion.chunk',
            choices=[ChunkChoice(index=0, delta=delta, finish_reason='stop' if final else None)],
        )
        for delta, final in [*((delta, False) for delta in deltas), (ChoiceDelta(), True)]
    ]
    return OpenAIChatModel(
        'gpt-4o', provider=OpenAIProvider(openai_client=MockOpenAI.create_mock_stream(chunks)), profile=profile
    )


def _responses_function_call_args(copies: _Copies, n: int) -> Model:
    call = responses.ResponseFunctionToolCall(
        id='fc_1', call_id='call_1', name='lookup', arguments='', type='function_call'
    )
    delta = responses.ResponseFunctionCallArgumentsDeltaEvent(
        item_id='fc_1', output_index=0, delta='', type='response.function_call_arguments.delta', sequence_number=0
    )
    return _responses_model(
        [
            responses.ResponseOutputItemAddedEvent(
                item=call, output_index=0, type='response.output_item.added', sequence_number=0
            ),
            *(delta.model_copy(update={'delta': copies.delta()}) for _ in range(n)),
            responses.ResponseFunctionCallArgumentsDoneEvent(
                arguments=DELTA * n,
                item_id='fc_1',
                output_index=0,
                type='response.function_call_arguments.done',
                sequence_number=0,
            ),
        ]
    )


def _responses_refusal(copies: _Copies, n: int) -> Model:
    delta = responses.ResponseRefusalDeltaEvent(
        content_index=0, delta='', item_id='msg_1', output_index=0, type='response.refusal.delta', sequence_number=0
    )
    return _responses_model([delta.model_copy(update={'delta': copies.delta()}) for _ in range(n)])


def _chat_text(copies: _Copies, n: int) -> Model:
    return _chat_model([ChoiceDelta(content='').model_copy(update={'content': copies.delta()}) for _ in range(n)])


def _chat_refusal(copies: _Copies, n: int) -> Model:
    return _chat_model([ChoiceDelta(refusal='').model_copy(update={'refusal': copies.delta()}) for _ in range(n)])


def _chat_whitespace_after_reasoning(copies: _Copies, n: int) -> Model:
    """Whitespace-only text after reasoning that interrupted a text part is held for the next text part."""
    reasoning = ChoiceDelta.model_validate({'reasoning_content': 'thinking'})
    return _chat_model(
        [
            ChoiceDelta(content='a'),
            reasoning,
            *(ChoiceDelta(content='').model_copy(update={'content': copies.delta(' ' * 8)}) for _ in range(n)),
            ChoiceDelta(content='b'),
        ],
        OpenAIModelProfile(ignore_streamed_leading_whitespace=True),
    )


def _tool_call_args(response: ModelResponse) -> object:
    return [part.args for part in response.parts if isinstance(part, ToolCallPart)]


def _refusal(response: ModelResponse) -> object:
    return (response.provider_details or {}).get('refusal')


def _texts(response: ModelResponse) -> object:
    return [part.content for part in response.parts if isinstance(part, TextPart)]


@dataclass(frozen=True)
class Case:
    id: str
    model: Callable[[_Copies, int], Model]
    output: Callable[[ModelResponse], object]
    expected: Callable[[int], object]


CASES = [
    Case('responses-function-call-args', _responses_function_call_args, _tool_call_args, lambda n: [DELTA * n]),
    Case('responses-refusal', _responses_refusal, _refusal, lambda n: DELTA * n),
    Case('chat-text', _chat_text, _texts, lambda n: [DELTA * n]),
    Case('chat-refusal', _chat_refusal, _refusal, lambda n: DELTA * n),
    Case(
        'chat-whitespace-after-reasoning', _chat_whitespace_after_reasoning, _texts, lambda n: ['a', ' ' * 8 * n + 'b']
    ),
]


def test_counting_delta_counts_the_copied_characters():
    """The control for the test below: the counter sees `+=` onto a plain string copy that string."""
    copies = _Copies()
    text = 'abc'
    text += copies.delta('d')
    assert (text, copies.chars) == ('abcd', 3)


@pytest.mark.parametrize('n', [10, 100])
@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in CASES])
async def test_stream_copies_each_character_at_most_once(allow_model_requests: None, case: Case, n: int):
    copies = _Copies()
    model = case.model(copies, n)
    async with model.request_stream(
        [ModelRequest(parts=[UserPromptPart(content='hi')])], None, ModelRequestParameters()
    ) as streamed:
        async for _ in streamed:
            pass
    assert case.output(streamed.get()) == case.expected(n)
    assert copies.chars <= 8 * n
