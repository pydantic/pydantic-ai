"""Pin how the Chat Completions-shaped providers map streamed tool-call deltas onto parts.

Groq, OpenAI and Hugging Face each stream a tool call as a run of per-chunk entries and read the name
and the argument fragment off each entry's `function` object. The three loops are line-for-line the
same, so they are tested together here: one scripted chunk sequence is replayed through each SDK's
mocked stream and the parts that come out are asserted, which is what keeps the three in step.

The shapes that matter are the ones the SDK types understate. An entry whose `function` is absent
carries only an index and must contribute nothing, and an empty-string name or argument fragment must
reach the parts manager as `''` rather than as `None` — the two are not interchangeable there, since
`''` opens a part where `None` leaves a bare delta.

These are unit tests rather than VCR tests because those shapes are not reachable from a recording.
Nothing makes a live provider emit a `function`-less entry or an empty-string fragment on demand, and
a cassette that happened to hold one would keep replaying green after the mapping stopped
distinguishing `''` from `None`, since a cassette pins the request rather than how the response is
read. Scripting the chunks is what makes each shape reachable and each assertion meaningful.
"""

from __future__ import annotations as _annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Literal

import pytest
from typing_extensions import TypedDict

from pydantic_ai import (
    Agent,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.direct import model_request_stream
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model
from pydantic_ai.usage import RequestUsage, RunUsage

from .._inline_snapshot import snapshot
from ..conftest import IsDatetime, IsStr, try_import
from .mock_openai import MockOpenAI

with try_import() as openai_imports_successful:
    from openai.types import chat as openai_chat
    from openai.types.chat.chat_completion_chunk import (
        Choice as OpenAIChunkChoice,
        ChoiceDelta as OpenAIChoiceDelta,
        ChoiceDeltaToolCall as OpenAIChoiceDeltaToolCall,
        ChoiceDeltaToolCallFunction as OpenAIChoiceDeltaToolCallFunction,
    )
    from openai.types.completion_usage import CompletionUsage as OpenAICompletionUsage

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as groq_imports_successful:
    from groq.types import chat as groq_chat
    from groq.types.chat.chat_completion_chunk import (
        Choice as GroqChunkChoice,
        ChoiceDelta as GroqChoiceDelta,
        ChoiceDeltaToolCall as GroqChoiceDeltaToolCall,
        ChoiceDeltaToolCallFunction as GroqChoiceDeltaToolCallFunction,
    )
    from groq.types.completion_usage import CompletionUsage as GroqCompletionUsage

    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.providers.groq import GroqProvider

    from .test_groq import MockGroq

with try_import() as huggingface_imports_successful:
    from huggingface_hub import (
        ChatCompletionStreamOutput,
        ChatCompletionStreamOutputChoice,
        ChatCompletionStreamOutputDelta,
        ChatCompletionStreamOutputDeltaToolCall,
        ChatCompletionStreamOutputFunction,
        ChatCompletionStreamOutputUsage,
    )

    from pydantic_ai.models.huggingface import HuggingFaceModel
    from pydantic_ai.providers.huggingface import HuggingFaceProvider

    from .test_huggingface import MockHuggingFace

pytestmark = pytest.mark.anyio

FinishReason = Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call']


@dataclass(frozen=True)
class ToolCallDelta:
    """One tool-call entry inside a streamed chunk's delta, in the terms the three SDKs share."""

    index: int = 0
    name: str | None = None
    arguments: str | None = None
    function: bool = True
    """Whether the entry carries a `function` object at all — providers send entries without one."""


@dataclass(frozen=True)
class Chunk:
    """One streamed chunk, in the terms the three SDKs share."""

    tool_calls: tuple[ToolCallDelta, ...] | None = None
    """`None` builds a delta carrying no `tool_calls` at all; `()` builds one carrying an empty list."""

    finish_reason: FinishReason | None = None

    has_choice: bool = True
    """`False` builds a chunk with no choices, which providers send as a usage-only trailer."""


NAME_THEN_ARGUMENTS = (
    Chunk(),
    Chunk(tool_calls=()),
    Chunk(tool_calls=(ToolCallDelta(function=False),)),
    Chunk(tool_calls=(ToolCallDelta(function=False),)),
    Chunk(tool_calls=(ToolCallDelta(name='final_result'),)),
    Chunk(tool_calls=(ToolCallDelta(function=False),)),
    Chunk(tool_calls=(ToolCallDelta(arguments='{"first": "One'),)),
    Chunk(tool_calls=(ToolCallDelta(arguments='", "second": "Two"'),)),
    Chunk(tool_calls=(ToolCallDelta(arguments='}'),)),
    Chunk(has_choice=False),
)
"""One tool call spread over a stream that also carries every empty shape a provider interleaves."""

ARGUMENTS_THEN_FINISH_REASON = (
    Chunk(tool_calls=(ToolCallDelta(name='final_result'),)),
    Chunk(tool_calls=(ToolCallDelta(arguments='{"first": "One'),)),
    Chunk(tool_calls=(ToolCallDelta(arguments='", "second": "Two"'),)),
    Chunk(tool_calls=(ToolCallDelta(arguments='}'),)),
    Chunk(tool_calls=(ToolCallDelta(),), finish_reason='stop'),
)
"""The same call, closed by a `function` object whose name and arguments are both `None`."""

EMPTY_STRING_DELTAS = (
    Chunk(tool_calls=(ToolCallDelta(index=0, arguments=''),)),
    Chunk(tool_calls=(ToolCallDelta(index=0, name='my_tool'),)),
    Chunk(tool_calls=(ToolCallDelta(index=1, name=''),)),
)
"""Empty-string fragments, at the two points where `''` and `None` produce different parts."""


def _openai_chunk(chunk: Chunk) -> openai_chat.ChatCompletionChunk:
    choices: list[OpenAIChunkChoice] = []
    if chunk.has_choice:
        tool_calls = (
            None
            if chunk.tool_calls is None
            else [
                OpenAIChoiceDeltaToolCall(
                    index=delta.index,
                    function=OpenAIChoiceDeltaToolCallFunction(name=delta.name, arguments=delta.arguments)
                    if delta.function
                    else None,
                )
                for delta in chunk.tool_calls
            ]
        )
        choices.append(
            OpenAIChunkChoice(
                index=0, delta=OpenAIChoiceDelta(tool_calls=tool_calls), finish_reason=chunk.finish_reason
            )
        )
    return openai_chat.ChatCompletionChunk(
        id='123',
        choices=choices,
        created=1704067200,  # 2024-01-01
        model='gpt-4o-123',
        object='chat.completion.chunk',
        usage=OpenAICompletionUsage(completion_tokens=1, prompt_tokens=2, total_tokens=3),
    )


def _openai_model(chunks: Sequence[Chunk]) -> Model:
    client = MockOpenAI.create_mock_stream([_openai_chunk(chunk) for chunk in chunks])
    return OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=client))


def _groq_chunk(chunk: Chunk) -> groq_chat.ChatCompletionChunk:
    choices: list[GroqChunkChoice] = []
    if chunk.has_choice:
        tool_calls = (
            None
            if chunk.tool_calls is None
            else [
                GroqChoiceDeltaToolCall(
                    index=delta.index,
                    function=GroqChoiceDeltaToolCallFunction(name=delta.name, arguments=delta.arguments)
                    if delta.function
                    else None,
                )
                for delta in chunk.tool_calls
            ]
        )
        choices.append(
            GroqChunkChoice(index=0, delta=GroqChoiceDelta(tool_calls=tool_calls), finish_reason=chunk.finish_reason)
        )
    return groq_chat.ChatCompletionChunk(
        id='x',
        choices=choices,
        created=1704067200,  # 2024-01-01
        x_groq=None,
        model='llama-3.3-70b-versatile',
        object='chat.completion.chunk',
        usage=GroqCompletionUsage(completion_tokens=1, prompt_tokens=2, total_tokens=3),
    )


def _groq_model(chunks: Sequence[Chunk]) -> Model:
    client = MockGroq.create_mock_stream([_groq_chunk(chunk) for chunk in chunks])
    return GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(groq_client=client))


def _huggingface_tool_call(delta: ToolCallDelta) -> ChatCompletionStreamOutputDeltaToolCall:
    """Build the entry through `parse_obj_as_instance`, the only way past the SDK's required fields.

    The SDK types both `function` and its `arguments` as required, which is what the streamed shapes
    contradict, and the dict path applies no such constraint.
    """
    entry: dict[str, object] = {'id': str(delta.index), 'type': 'function', 'index': delta.index}
    if delta.function:
        entry['function'] = ChatCompletionStreamOutputFunction.parse_obj_as_instance(  # pyright: ignore[reportUnknownMemberType]
            {'name': delta.name, 'arguments': delta.arguments}
        )
    return ChatCompletionStreamOutputDeltaToolCall.parse_obj_as_instance(entry)  # pyright: ignore[reportUnknownMemberType]


def _huggingface_chunk(chunk: Chunk) -> ChatCompletionStreamOutput:
    choices: list[ChatCompletionStreamOutputChoice] = []
    if chunk.has_choice:
        tool_calls = None if chunk.tool_calls is None else [_huggingface_tool_call(delta) for delta in chunk.tool_calls]
        choices.append(
            ChatCompletionStreamOutputChoice(
                index=0,
                delta=ChatCompletionStreamOutputDelta(role='assistant', tool_calls=tool_calls),
                finish_reason=chunk.finish_reason,
            )
        )
    return ChatCompletionStreamOutput.parse_obj_as_instance(  # pyright: ignore[reportUnknownMemberType]
        {
            'id': 'x',
            'choices': choices,
            'created': 1704067200,  # 2024-01-01
            'model': 'hf-model',
            'object': 'chat.completion.chunk',
            'usage': ChatCompletionStreamOutputUsage(completion_tokens=1, prompt_tokens=2, total_tokens=3),
        }
    )


def _huggingface_model(chunks: Sequence[Chunk]) -> Model:
    client = MockHuggingFace.create_stream_mock([_huggingface_chunk(chunk) for chunk in chunks])
    return HuggingFaceModel('hf-model', provider=HuggingFaceProvider(hf_client=client, api_key='x'))


ModelFactory = Callable[[Sequence[Chunk]], Model]
"""Builds a model whose SDK client replays the given chunks as one streamed response."""


@dataclass(frozen=True)
class Case:
    """One provider, and what the shared chunk scripts produce through it."""

    id: str
    build_model: ModelFactory
    usage: RunUsage
    messages: list[ModelMessage]
    output_tokens_per_chunk: int
    """Groq counts streamed usage only off its `x_groq` trailer, which the scripted chunks don't carry."""

    marks: tuple[pytest.MarkDecorator, ...] = ()


CASES = [
    Case(
        id='openai',
        build_model=_openai_model,
        usage=snapshot(RunUsage(requests=1, input_tokens=20, output_tokens=10)),
        messages=snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"first": "One", "second": "Two"}',
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(output_tokens=10, input_tokens=20),
                    model_name='gpt-4o-123',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1',
                    provider_details={'timestamp': IsDatetime()},
                    provider_response_id='123',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        ),
        output_tokens_per_chunk=1,
        marks=(pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed'),),
    ),
    Case(
        id='groq',
        build_model=_groq_model,
        usage=snapshot(RunUsage(requests=1, cost=Decimal('0.00'))),
        messages=snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"first": "One", "second": "Two"}',
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(cost=Decimal('0.00')),
                    model_name='llama-3.3-70b-versatile',
                    timestamp=IsDatetime(),
                    provider_name='groq',
                    provider_url='https://api.groq.com',
                    provider_details={'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)},
                    provider_response_id='x',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        ),
        output_tokens_per_chunk=0,
        marks=(pytest.mark.skipif(not groq_imports_successful(), reason='groq not installed'),),
    ),
    Case(
        id='huggingface',
        build_model=_huggingface_model,
        usage=snapshot(RunUsage(requests=1, input_tokens=20, output_tokens=10)),
        messages=snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        # Hugging Face is the one SDK whose entries carry an id, so the part keeps it
                        # instead of the generated one the other two get.
                        ToolCallPart(
                            tool_name='final_result', args='{"first": "One", "second": "Two"}', tool_call_id='0'
                        )
                    ],
                    usage=RequestUsage(output_tokens=10, input_tokens=20),
                    model_name='hf-model',
                    timestamp=IsDatetime(),
                    provider_name='huggingface',
                    provider_url='https://api-inference.huggingface.co',
                    provider_details={'timestamp': IsDatetime()},
                    provider_response_id='x',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='0',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        ),
        output_tokens_per_chunk=1,
        marks=(pytest.mark.skipif(not huggingface_imports_successful(), reason='huggingface_hub not installed'),),
    ),
]

CASE_PARAMS = [pytest.param(case, id=case.id, marks=case.marks) for case in CASES]


class MyTypedDict(TypedDict, total=False):
    first: str
    second: str


@pytest.mark.parametrize('case', CASE_PARAMS)
async def test_tool_call_deltas_accumulate_into_one_part(case: Case, allow_model_requests: None):
    """A name-only entry opens the call and later argument-only entries fill it in.

    The script interleaves the empty shapes a provider sends around a tool call — a delta with no
    `tool_calls`, one with an empty list, entries carrying no `function`, and a trailing chunk with no
    choices at all — none of which may disturb the call being accumulated.
    """
    agent = Agent(case.build_model(NAME_THEN_ARGUMENTS), output_type=MyTypedDict)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [dict(c) async for c in result.stream_output(debounce_by=None)] == snapshot(
            [
                {},
                {'first': 'One'},
                {'first': 'One', 'second': 'Two'},
                {'first': 'One', 'second': 'Two'},
                {'first': 'One', 'second': 'Two'},
            ]
        )
        assert result.is_complete

    assert result.usage == case.usage
    # Every scripted chunk carries the same usage, so the total is what proves each one was consumed.
    assert result.usage.output_tokens == case.output_tokens_per_chunk * len(NAME_THEN_ARGUMENTS)
    assert result.all_messages() == case.messages


@pytest.mark.parametrize('case', CASE_PARAMS)
async def test_tool_call_deltas_closed_by_an_empty_function(case: Case, allow_model_requests: None):
    """A `function` carrying neither a name nor arguments leaves the accumulated call as it stands.

    It rides on the chunk that also carries the finish reason, which is how a provider closes a tool
    call: the run has to end on the arguments it already has rather than on a further chunk.
    """
    agent = Agent(case.build_model(ARGUMENTS_THEN_FINISH_REASON), output_type=MyTypedDict)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [dict(c) async for c in result.stream_output(debounce_by=None)] == snapshot(
            [
                {'first': 'One'},
                {'first': 'One', 'second': 'Two'},
                {'first': 'One', 'second': 'Two'},
                {'first': 'One', 'second': 'Two'},
            ]
        )
        assert result.is_complete


@pytest.mark.parametrize('case', CASE_PARAMS)
async def test_empty_string_tool_call_deltas_reach_the_parts_manager(case: Case, allow_model_requests: None):
    """`''` is forwarded as `''`, which the resulting parts distinguish from `None` in both fields.

    An empty argument fragment arriving before the name has to survive until the name completes the
    call, and an empty name has to open a part on its own — collapsing either to `None` would leave
    the first call with `args=None` and the second as a bare delta contributing no part at all.

    The stream is driven through `model_request_stream` rather than an agent because neither call
    names a tool an agent could route, which is the point: the mapping is what's under test, not what
    the run does with it.
    """
    async with model_request_stream(
        case.build_model(EMPTY_STRING_DELTAS), [ModelRequest.user_text_prompt('')]
    ) as stream:
        async for _ in stream:
            pass
        response = stream.get()

    assert response.parts == snapshot(
        [
            ToolCallPart(tool_name='my_tool', args='', tool_call_id=IsStr()),
            ToolCallPart(tool_name='', tool_call_id=IsStr()),
        ]
    )
