from __future__ import annotations as _annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import cached_property
from typing import Any, cast

import httpx
import pytest
from pydantic import BaseModel
from typing_extensions import NotRequired, TypedDict
from vcr.cassette import Cassette

from pydantic_ai import (
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UploadedFile,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.agent import Agent
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, ModelRetry, UnexpectedModelBehavior
from pydantic_ai.messages import BinaryImage
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ThinkingLevel
from pydantic_ai.usage import RequestUsage, RunUsage

from .._inline_snapshot import snapshot
from ..conftest import IsDatetime, IsInstance, IsNow, IsStr, raise_if_exception, try_import
from .mock_async_stream import MockAsyncStream

with try_import() as imports_successful:
    from mistralai.client import Mistral
    from mistralai.client.errors import SDKError
    from mistralai.client.models import (
        AssistantMessage as MistralAssistantMessage,
        ChatCompletionChoice as MistralChatCompletionChoice,
        ChatCompletionResponse as MistralChatCompletionResponse,
        CompletionChunk as MistralCompletionChunk,
        CompletionEvent as MistralCompletionEvent,
        CompletionResponseStreamChoice as MistralCompletionResponseStreamChoice,
        CompletionResponseStreamChoiceFinishReason as MistralCompletionResponseStreamChoiceFinishReason,
        ContentChunk as MistralContentChunk,
        DeltaMessage as MistralDeltaMessage,
        FunctionCall as MistralFunctionCall,
        ImageURL as MistralImageURL,
        ImageURLChunk as MistralImageURLChunk,
        ReferenceChunk as MistralReferenceChunk,
        TextChunk,
        TextChunk as MistralTextChunk,
        ToolCall as MistralToolCall,
        UsageInfo as MistralUsageInfo,
        UserMessage,
    )
    from mistralai.client.types.basemodel import Unset as MistralUnset

    from pydantic_ai.models.mistral import (
        MistralModel,
        MistralModelSettings,
        MistralStreamedResponse,
        _map_content,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
    from pydantic_ai.providers.mistral import MistralProvider
    from pydantic_ai.providers.openai import OpenAIProvider

    MockChatCompletion = MistralChatCompletionResponse | Exception
    MockCompletionEvent = MistralCompletionEvent | Exception

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='mistral or openai not installed'),
    pytest.mark.anyio,
]


@dataclass
class MockSdkConfiguration:
    def get_server_details(self) -> tuple[str, ...]:
        return ('https://api.mistral.ai',)


@dataclass
class MockMistralAI:
    completions: MockChatCompletion | Sequence[MockChatCompletion] | None = None
    stream: Sequence[MockCompletionEvent] | Sequence[Sequence[MockCompletionEvent]] | None = None
    index: int = 0
    chat_completion_kwargs: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])

    @cached_property
    def sdk_configuration(self) -> MockSdkConfiguration:
        return MockSdkConfiguration()

    @cached_property
    def chat(self) -> Any:
        if self.stream:
            return type(
                'Chat',
                (),
                {'stream_async': self.chat_completions_create, 'complete_async': self.chat_completions_create},
            )
        else:
            return type('Chat', (), {'complete_async': self.chat_completions_create})

    @classmethod
    def create_mock(cls, completions: MockChatCompletion | Sequence[MockChatCompletion]) -> Mistral:
        return cast(Mistral, cls(completions=completions))

    @classmethod
    def create_stream_mock(
        cls, completions_streams: Sequence[MockCompletionEvent] | Sequence[Sequence[MockCompletionEvent]]
    ) -> Mistral:
        return cast(Mistral, cls(stream=completions_streams))

    async def chat_completions_create(  # pragma: lax no cover
        self, *_args: Any, stream: bool = False, **kwargs: Any
    ) -> MistralChatCompletionResponse | MockAsyncStream[MockCompletionEvent]:
        self.chat_completion_kwargs.append(kwargs)
        if stream or self.stream:
            assert self.stream is not None, 'you can only use `stream=True` if `stream` is provided'
            if isinstance(self.stream[0], list):
                response = MockAsyncStream(iter(cast(list[MockCompletionEvent], self.stream[self.index])))
            else:
                response = MockAsyncStream(iter(cast(list[MockCompletionEvent], self.stream)))
        else:
            assert self.completions is not None, 'you can only use `stream=False` if `completions` are provided'
            if isinstance(self.completions, Sequence):
                raise_if_exception(self.completions[self.index])
                response = cast(MistralChatCompletionResponse, self.completions[self.index])
            else:
                raise_if_exception(self.completions)
                response = cast(MistralChatCompletionResponse, self.completions)
        self.index += 1
        return response


def completion_message(
    message: MistralAssistantMessage, *, usage: MistralUsageInfo | None = None, with_created: bool = True
) -> MistralChatCompletionResponse:
    return MistralChatCompletionResponse(
        id='123',
        choices=[MistralChatCompletionChoice(finish_reason='stop', index=0, message=message)],
        created=1704067200 if with_created else 0,  # 2024-01-01
        model='mistral-large-123',
        object='chat.completion',
        usage=usage or MistralUsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=1),
    )


def chunk(
    delta: list[MistralDeltaMessage],
    finish_reason: MistralCompletionResponseStreamChoiceFinishReason | None = None,
    with_created: bool = True,
) -> MistralCompletionEvent:
    return MistralCompletionEvent(
        data=MistralCompletionChunk(
            id='x',
            choices=[
                MistralCompletionResponseStreamChoice(index=index, delta=delta, finish_reason=finish_reason)
                for index, delta in enumerate(delta)
            ],
            created=1704067200 if with_created else 0,  # 2024-01-01
            model='gpt-4',
            object='chat.completion.chunk',
            usage=MistralUsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=1),
        )
    )


def text_chunk(
    text: str, finish_reason: MistralCompletionResponseStreamChoiceFinishReason | None = None
) -> MistralCompletionEvent:
    return chunk([MistralDeltaMessage(content=text, role='assistant')], finish_reason=finish_reason)


def text_chunkk(
    text: str, finish_reason: MistralCompletionResponseStreamChoiceFinishReason | None = None
) -> MistralCompletionEvent:
    return chunk(
        [MistralDeltaMessage(content=[MistralTextChunk(text=text)], role='assistant')], finish_reason=finish_reason
    )


def func_chunk(
    tool_calls: list[MistralToolCall], finish_reason: MistralCompletionResponseStreamChoiceFinishReason | None = None
) -> MistralCompletionEvent:
    return chunk([MistralDeltaMessage(tool_calls=tool_calls, role='assistant')], finish_reason=finish_reason)


#####################
## Init
#####################


def test_init():
    provider = MistralProvider(api_key='foobar')
    m = MistralModel('mistral-large-latest', provider=provider)
    assert m.client is provider.client
    assert m.model_name == 'mistral-large-latest'
    assert m.base_url == 'https://api.mistral.ai'


#####################
## Completion
#####################


async def test_multiple_completions(allow_model_requests: None):
    completions = [
        # First completion: created is "now" (simulate IsNow)
        completion_message(
            MistralAssistantMessage(content='world'),
            usage=MistralUsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=1),
            with_created=False,
        ),
        # Second completion: created is fixed 2024-01-01 00:00:00 UTC
        completion_message(MistralAssistantMessage(content='hello again')),
    ]
    mock_client = MockMistralAI.create_mock(completions)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model=model)

    result = await agent.run('hello')

    assert result.output == 'world'
    assert result.usage.input_tokens == 1
    assert result.usage.output_tokens == 1

    result = await agent.run('hello again', message_history=result.new_messages())
    assert result.output == 'hello again'
    assert result.usage.input_tokens == 1
    assert result.usage.output_tokens == 1
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='world')],
                usage=RequestUsage(input_tokens=1, output_tokens=1),
                model_name='mistral-large-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={'finish_reason': 'stop'},
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='hello again', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='hello again')],
                usage=RequestUsage(input_tokens=1, output_tokens=1),
                model_name='mistral-large-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_three_completions(allow_model_requests: None):
    completions = [
        completion_message(
            MistralAssistantMessage(content='world'),
            usage=MistralUsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=1),
        ),
        completion_message(MistralAssistantMessage(content='hello again')),
        completion_message(MistralAssistantMessage(content='final message')),
    ]
    mock_client = MockMistralAI.create_mock(completions)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model=model)

    result = await agent.run('hello')

    assert result.output == 'world'
    assert result.usage.input_tokens == 1
    assert result.usage.output_tokens == 1

    result = await agent.run('hello again', message_history=result.all_messages())
    assert result.output == 'hello again'
    assert result.usage.input_tokens == 1
    assert result.usage.output_tokens == 1

    result = await agent.run('final message', message_history=result.all_messages())
    assert result.output == 'final message'
    assert result.usage.input_tokens == 1
    assert result.usage.output_tokens == 1
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='world')],
                usage=RequestUsage(input_tokens=1, output_tokens=1),
                model_name='mistral-large-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='hello again', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='hello again')],
                usage=RequestUsage(input_tokens=1, output_tokens=1),
                model_name='mistral-large-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='final message', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='final message')],
                usage=RequestUsage(input_tokens=1, output_tokens=1),
                model_name='mistral-large-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_usage_with_cached_tokens(allow_model_requests: None):
    # Mistral reports prompt-cache hits nested under `prompt_tokens_details.cached_tokens`,
    # which genai-prices maps to the first-class `cache_read_tokens` field.
    # https://docs.mistral.ai/studio-api/conversations/advanced/prompt-caching
    usage = MistralUsageInfo.model_validate(
        {
            'prompt_tokens': 1013,
            'completion_tokens': 30,
            'total_tokens': 1043,
            'prompt_tokens_details': {'cached_tokens': 1008},
        }
    )
    completion = completion_message(MistralAssistantMessage(content='world'), usage=usage)
    mock_client = MockMistralAI.create_mock(completion)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model=model)

    result = await agent.run('hello')

    assert result.usage == snapshot(RunUsage(input_tokens=1013, cache_read_tokens=1008, output_tokens=30, requests=1))


#####################
## Completion Stream
#####################


async def test_stream_text(allow_model_requests: None):
    stream = [
        text_chunk('hello '),
        text_chunk('world '),
        text_chunk('welcome '),
        text_chunkk('mistral'),
        chunk([]),
    ]
    mock_client = MockMistralAI.create_stream_mock(stream)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model=model)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(
            ['hello ', 'hello world ', 'hello world welcome ', 'hello world welcome mistral']
        )
        assert result.is_complete
        assert result.usage.input_tokens == 5
        assert result.usage.output_tokens == 5


@pytest.mark.parametrize('with_tool', [False, True])
async def test_stream_forwards_model_settings(allow_model_requests: None, with_tool: bool):
    """The mock captures request fields that VCR matching does not compare."""
    stream = [text_chunk('hello'), chunk([])]
    mock_client = MockMistralAI.create_stream_mock(stream)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(
        model,
        model_settings=MistralModelSettings(
            temperature=0.0,
            top_p=1.0,
            max_tokens=100,
            timeout=2.5,
            seed=42,
            presence_penalty=0.3,
            frequency_penalty=0.1,
            stop_sequences=['STOP'],
        ),
    )

    if with_tool:

        @agent.tool_plain
        def echo(value: str) -> str:
            return value  # pragma: no cover

    async with agent.run_stream('hello') as result:
        await result.get_output()

    kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert {
        'temperature': kwargs['temperature'],
        'top_p': kwargs['top_p'],
        'max_tokens': kwargs['max_tokens'],
        'timeout_ms': kwargs['timeout_ms'],
        'random_seed': kwargs['random_seed'],
        'presence_penalty': kwargs['presence_penalty'],
        'frequency_penalty': kwargs['frequency_penalty'],
        'stop': kwargs['stop'],
    } == snapshot(
        {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': 100,
            'timeout_ms': 2500,
            'random_seed': 42,
            'presence_penalty': 0.3,
            'frequency_penalty': 0.1,
            'stop': ['STOP'],
        }
    )
    if with_tool:
        assert len(kwargs['tools']) == 1
        assert kwargs['tool_choice'] == 'auto'
    else:
        assert isinstance(kwargs['tools'], MistralUnset)
        assert kwargs['tool_choice'] is None


@pytest.mark.parametrize('with_tool', [False, True])
async def test_stream_preserves_unset_model_settings(allow_model_requests: None, with_tool: bool):
    """Consolidating request paths must not add defaults to no-tool requests."""
    stream = [text_chunk('hello'), chunk([])]
    mock_client = MockMistralAI.create_stream_mock(stream)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model)

    if with_tool:

        @agent.tool_plain
        def echo(value: str) -> str:
            return value  # pragma: no cover

    async with agent.run_stream('hello') as result:
        await result.get_output()

    kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    if with_tool:
        assert kwargs['top_p'] == 1
        assert kwargs['n'] == 1
    else:
        assert kwargs['top_p'] is None
        assert isinstance(kwargs['n'], MistralUnset)
    assert isinstance(kwargs['temperature'], MistralUnset)
    assert isinstance(kwargs['max_tokens'], MistralUnset)
    assert isinstance(kwargs['random_seed'], MistralUnset)


async def test_stream_usage_with_cached_tokens(allow_model_requests: None):
    stream = [
        MistralCompletionEvent(
            data=MistralCompletionChunk(
                id='x',
                choices=[
                    MistralCompletionResponseStreamChoice(
                        index=0,
                        delta=MistralDeltaMessage(content='world', role='assistant'),
                        finish_reason='stop',
                    )
                ],
                created=1704067200,
                model='mistral-large-latest',
                object='chat.completion.chunk',
                usage=MistralUsageInfo.model_validate(
                    {
                        'prompt_tokens': 1013,
                        'completion_tokens': 30,
                        'total_tokens': 1043,
                        'prompt_tokens_details': {'cached_tokens': 1008},
                    }
                ),
            )
        ),
    ]
    mock_client = MockMistralAI.create_stream_mock(stream)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model=model)

    async with agent.run_stream('') as result:
        async for _ in result.stream_text(debounce_by=None):
            pass

    # `prompt_tokens_details.cached_tokens` is surfaced as first-class `cache_read_tokens`.
    assert result.usage == snapshot(RunUsage(input_tokens=1013, cache_read_tokens=1008, output_tokens=30, requests=1))


async def test_stream_text_finish_reason(allow_model_requests: None):
    stream = [
        text_chunk('hello '),
        text_chunkk('world'),
        text_chunk('.', finish_reason='stop'),
    ]
    mock_client = MockMistralAI.create_stream_mock(stream)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model=model)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(
            ['hello ', 'hello world', 'hello world.']
        )
        assert result.is_complete


async def test_no_delta(allow_model_requests: None):
    stream = [
        chunk([], with_created=False),
        text_chunk('hello '),
        text_chunk('world'),
    ]
    mock_client = MockMistralAI.create_stream_mock(stream)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model=model)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(['hello ', 'hello world'])
        assert result.is_complete
        assert result.usage.input_tokens == 3
        assert result.usage.output_tokens == 3


#####################
## Completion Model Structured
#####################


async def test_request_native_with_arguments_dict_response(allow_model_requests: None):
    class CityLocation(BaseModel):
        city: str
        country: str

    completion = completion_message(
        MistralAssistantMessage(
            content=None,
            role='assistant',
            tool_calls=[
                MistralToolCall(
                    id='123',
                    function=MistralFunctionCall(arguments={'city': 'paris', 'country': 'france'}, name='final_result'),
                    type='function',
                )
            ],
        ),
        usage=MistralUsageInfo(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )
    mock_client = MockMistralAI.create_mock(completion)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model=model, output_type=CityLocation)

    result = await agent.run('User prompt value')

    assert result.output == CityLocation(city='paris', country='france')
    assert result.usage.input_tokens == 1
    assert result.usage.output_tokens == 2
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='User prompt value', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args={'city': 'paris', 'country': 'france'},
                        tool_call_id='123',
                    )
                ],
                usage=RequestUsage(input_tokens=1, output_tokens=2),
                model_name='mistral-large-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='123',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_request_native_with_arguments_str_response(allow_model_requests: None):
    class CityLocation(BaseModel):
        city: str
        country: str

    completion = completion_message(
        MistralAssistantMessage(
            content=None,
            role='assistant',
            tool_calls=[
                MistralToolCall(
                    id='123',
                    function=MistralFunctionCall(
                        arguments='{"city": "paris", "country": "france"}', name='final_result'
                    ),
                    type='function',
                )
            ],
        )
    )
    mock_client = MockMistralAI.create_mock(completion)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model=model, output_type=CityLocation)

    result = await agent.run('User prompt value')

    assert result.output == CityLocation(city='paris', country='france')
    assert result.usage.input_tokens == 1
    assert result.usage.output_tokens == 1
    assert result.usage.details == {}
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='User prompt value', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"city": "paris", "country": "france"}',
                        tool_call_id='123',
                    )
                ],
                usage=RequestUsage(input_tokens=1, output_tokens=1),
                model_name='mistral-large-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='123',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_request_output_type_with_arguments_str_response(allow_model_requests: None):
    completion = completion_message(
        MistralAssistantMessage(
            content=None,
            role='assistant',
            tool_calls=[
                MistralToolCall(
                    id='123',
                    function=MistralFunctionCall(arguments='{"response": 42}', name='final_result'),
                    type='function',
                )
            ],
        )
    )
    mock_client = MockMistralAI.create_mock(completion)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model=model, output_type=int, instructions='System prompt value')

    result = await agent.run('User prompt value')

    assert result.output == 42
    assert result.usage.input_tokens == 1
    assert result.usage.output_tokens == 1
    assert result.usage.details == {}
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='User prompt value', timestamp=IsNow(tz=timezone.utc)),
                ],
                instructions='System prompt value',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"response": 42}',
                        tool_call_id='123',
                    )
                ],
                usage=RequestUsage(input_tokens=1, output_tokens=1),
                model_name='mistral-large-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='123',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


#####################
## Completion Model Structured Stream (JSON Mode)
#####################


async def test_stream_structured_with_all_type(allow_model_requests: None):
    class MyTypedDict(TypedDict, total=False):
        first: str
        second: int
        bool_value: bool
        nullable_value: int | None
        array_value: list[str]
        dict_value: dict[str, Any]
        dict_int_value: dict[str, int]
        dict_str_value: dict[int, str]

    stream = [
        text_chunk('{'),
        text_chunk('"first": "One'),
        text_chunk(
            '", "second": 2',
        ),
        text_chunk(
            ', "bool_value": true',
        ),
        text_chunk(
            ', "nullable_value": null',
        ),
        text_chunk(
            ', "array_value": ["A", "B", "C"]',
        ),
        text_chunk(
            ', "dict_value": {"A": "A", "B":"B"}',
        ),
        text_chunk(
            ', "dict_int_value": {"A": 1, "B":2}',
        ),
        text_chunk('}'),
        chunk([]),
    ]

    mock_client = MockMistralAI.create_stream_mock(stream)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model, output_type=MyTypedDict)

    async with agent.run_stream('User prompt value') as result:
        assert not result.is_complete
        v = [dict(c) async for c in result.stream_output(debounce_by=None)]
        assert v == snapshot(
            [
                {'first': 'One'},
                {'first': 'One', 'second': 2},
                {'first': 'One', 'second': 2, 'bool_value': True},
                {'first': 'One', 'second': 2, 'bool_value': True, 'nullable_value': None},
                {
                    'first': 'One',
                    'second': 2,
                    'bool_value': True,
                    'nullable_value': None,
                    'array_value': ['A', 'B', 'C'],
                },
                {
                    'first': 'One',
                    'second': 2,
                    'bool_value': True,
                    'nullable_value': None,
                    'array_value': ['A', 'B', 'C'],
                    'dict_value': {'A': 'A', 'B': 'B'},
                },
                {
                    'first': 'One',
                    'second': 2,
                    'bool_value': True,
                    'nullable_value': None,
                    'array_value': ['A', 'B', 'C'],
                    'dict_value': {'A': 'A', 'B': 'B'},
                    'dict_int_value': {'A': 1, 'B': 2},
                },
                {
                    'first': 'One',
                    'second': 2,
                    'bool_value': True,
                    'nullable_value': None,
                    'array_value': ['A', 'B', 'C'],
                    'dict_value': {'A': 'A', 'B': 'B'},
                    'dict_int_value': {'A': 1, 'B': 2},
                },
                {
                    'first': 'One',
                    'second': 2,
                    'bool_value': True,
                    'nullable_value': None,
                    'array_value': ['A', 'B', 'C'],
                    'dict_value': {'A': 'A', 'B': 'B'},
                    'dict_int_value': {'A': 1, 'B': 2},
                },
            ]
        )
        assert result.is_complete
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 10

        # double check usage matches stream count
        assert result.usage.output_tokens == len(stream)


async def test_stream_result_type_primitif_dict(allow_model_requests: None):
    """This test tests the primitif result with the pydantic ai format model response"""

    class MyTypedDict(TypedDict, total=False):
        first: str
        second: str

    stream = [
        text_chunk('{'),
        text_chunk('"'),
        text_chunk('f'),
        text_chunk('i'),
        text_chunk('r'),
        text_chunk('s'),
        text_chunk('t'),
        text_chunk('"'),
        text_chunk(':'),
        text_chunk(' '),
        text_chunk('"'),
        text_chunk('O'),
        text_chunk('n'),
        text_chunk('e'),
        text_chunk('"'),
        text_chunk(','),
        text_chunk(' '),
        text_chunk('"'),
        text_chunk('s'),
        text_chunk('e'),
        text_chunk('c'),
        text_chunk('o'),
        text_chunk('n'),
        text_chunk('d'),
        text_chunk('"'),
        text_chunk(':'),
        text_chunk(' '),
        text_chunk('"'),
        text_chunk('T'),
        text_chunk('w'),
        text_chunk('o'),
        text_chunk('"'),
        text_chunk('}'),
        chunk([]),
    ]

    mock_client = MockMistralAI.create_stream_mock(stream)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model=model, output_type=MyTypedDict)

    async with agent.run_stream('User prompt value') as result:
        assert not result.is_complete
        v = [c async for c in result.stream_output(debounce_by=None)]
        assert v == snapshot(
            [
                {'first': ''},
                {'first': 'O'},
                {'first': 'On'},
                {'first': 'One'},
                {'first': 'One'},
                {'first': 'One'},
                {'first': 'One'},
                {'first': 'One'},
                {'first': 'One'},
                {'first': 'One'},
                {'first': 'One'},
                {'first': 'One'},
                {'first': 'One'},
                {'first': 'One'},
                {'first': 'One'},
                {'first': 'One'},
                {'first': 'One'},
                {'first': 'One', 'second': ''},
                {'first': 'One', 'second': 'T'},
                {'first': 'One', 'second': 'Tw'},
                {'first': 'One', 'second': 'Two'},
                {'first': 'One', 'second': 'Two'},
                {'first': 'One', 'second': 'Two'},
                {'first': 'One', 'second': 'Two'},
            ]
        )
        assert result.is_complete
        assert result.usage.input_tokens == 34
        assert result.usage.output_tokens == 34

        # double check usage matches stream count
        assert result.usage.output_tokens == len(stream)


async def test_stream_result_type_primitif_int(allow_model_requests: None):
    """This test tests the primitif result with the pydantic ai format model response"""

    stream = [
        # {'response':
        text_chunk('{'),
        text_chunk('"resp'),
        text_chunk('onse":'),
        text_chunk('1'),
        text_chunk('}'),
        chunk([]),
    ]

    mock_client = MockMistralAI.create_stream_mock(stream)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model=model, output_type=int)

    async with agent.run_stream('User prompt value') as result:
        assert not result.is_complete
        v = [c async for c in result.stream_output(debounce_by=None)]
        assert v == snapshot([1, 1])
        assert result.is_complete
        assert result.usage.input_tokens == 6
        assert result.usage.output_tokens == 6

        # double check usage matches stream count
        assert result.usage.output_tokens == len(stream)


@pytest.mark.parametrize(
    'output_type, json_chunks, expected',
    [
        pytest.param(float, ('{"response":20', '}'), 20.0, id='number-accepts-integer'),
        pytest.param(float, ('{"response":1', '.5}'), 1.5, id='number-accepts-decimal-continuation'),
        pytest.param(int, ('{"response":1.0', '}'), 1, id='integer-accepts-zero-fraction'),
    ],
)
async def test_stream_result_type_numeric_json(
    allow_model_requests: None,
    output_type: type[int] | type[float],
    json_chunks: tuple[str, ...],
    expected: int | float,
) -> None:
    """Use mock chunks because a live model cannot reliably emit the exact numeric spellings and boundaries."""
    stream = [text_chunk(text) for text in json_chunks[:-1]]
    stream.append(text_chunk(json_chunks[-1], finish_reason='stop'))
    mock_client = MockMistralAI.create_stream_mock(stream)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model=model, output_type=output_type)

    async with agent.run_stream('User prompt value') as result:
        assert await result.get_output() == expected


@pytest.mark.parametrize(
    'output_type, partial_value, valid_value, expected',
    [
        pytest.param(float, '1', '2.5', 2.5, id='number-with-integer-prefix'),
        pytest.param(int, '1.0', '2', 2, id='integer-with-integral-float-prefix'),
    ],
)
async def test_stream_result_type_numeric_json_retries_malformed_continuation(
    allow_model_requests: None,
    output_type: type[int] | type[float],
    partial_value: str,
    valid_value: str,
    expected: int | float,
) -> None:
    """Reject a compatible numeric prefix when the completed JSON is malformed.

    A live model cannot reliably reproduce the exact chunk boundary and malformed retry sequence.
    """
    streams = [
        [text_chunk(f'{{"response":{partial_value}'), text_chunk('x}', finish_reason='stop')],
        [text_chunk(f'{{"response":{valid_value}}}', finish_reason='stop')],
    ]
    mock_client = MockMistralAI.create_stream_mock(streams)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model=model, output_type=output_type)

    async with agent.run_stream('User prompt value') as result:
        assert await result.get_output() == expected

    assert len(get_mock_chat_completion_kwargs(mock_client)) == 2


class _StaleIntField(TypedDict):
    value: int


class _DuplicateIntField(TypedDict):
    a: int
    b: str


class _NullableIntField(TypedDict):
    value: int | None


class _StringField(TypedDict):
    value: str


@pytest.mark.parametrize(
    'first_chunk, partial_value, complete_value',
    [
        pytest.param('{"value":"item 1', 'item 1', 'item 12', id='plain'),
        pytest.param('{"value":"item \\"1', 'item "1', 'item "12', id='escaped-quote'),
    ],
)
async def test_stream_output_keeps_digit_ending_partial_string(
    allow_model_requests: None,
    first_chunk: str,
    partial_value: str,
    complete_value: str,
) -> None:
    """Use mock chunks because a live model cannot reliably reproduce the exact string chunk boundary."""
    stream = [text_chunk(first_chunk), text_chunk('2"}', finish_reason='stop')]
    mock_client = MockMistralAI.create_stream_mock(stream)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model=model, output_type=_StringField)

    async with agent.run_stream('User prompt value') as result:
        assert [item async for item in result.stream_output(debounce_by=None)] == [
            {'value': partial_value},
            {'value': complete_value},
            {'value': complete_value},
        ]


async def test_stream_output_keeps_incomplete_unicode_escape(allow_model_requests: None) -> None:
    """Use mock chunks because a live model cannot reliably reproduce the exact escape boundary."""
    stream = [text_chunk('{"value":"\\u12'), text_chunk('34"}', finish_reason='stop')]
    mock_client = MockMistralAI.create_stream_mock(stream)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model=model, output_type=_StringField)

    async with agent.run_stream('User prompt value') as result:
        assert [item async for item in result.stream_output(debounce_by=None)] == [
            {'value': '\u1234'},
            {'value': '\u1234'},
        ]


class _PartialIntField(TypedDict):
    value: int
    label: NotRequired[str]


async def test_stream_output_keeps_integer_followed_by_whitespace(allow_model_requests: None) -> None:
    """Use mock chunks because a live model cannot reliably reproduce the exact whitespace boundary."""
    stream = [text_chunk('{"value":1 '), text_chunk(',"label":"x"}', finish_reason='stop')]
    mock_client = MockMistralAI.create_stream_mock(stream)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model=model, output_type=_PartialIntField)

    async with agent.run_stream('User prompt value') as result:
        assert [item async for item in result.stream_output(debounce_by=None)] == [
            {'value': 1},
            {'value': 1, 'label': 'x'},
            {'value': 1, 'label': 'x'},
        ]


class _FloatField(TypedDict):
    value: float


async def test_stream_output_keeps_partial_float(allow_model_requests: None) -> None:
    """Use mock chunks because a live model cannot reliably reproduce the exact numeric boundary."""
    stream = [text_chunk('{"value":1.2'), text_chunk('3}', finish_reason='stop')]
    mock_client = MockMistralAI.create_stream_mock(stream)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model=model, output_type=_FloatField)

    async with agent.run_stream('User prompt value') as result:
        assert [item async for item in result.stream_output(debounce_by=None)] == [
            {'value': 1.2},
            {'value': 1.23},
            {'value': 1.23},
        ]


async def test_stream_output_nullable_integer_prefix_fails_validation(allow_model_requests: None) -> None:
    """Use mock chunks because a live model cannot reliably reproduce the exact numeric boundary."""
    stream = [text_chunk('{"value":1'), text_chunk('.5}', finish_reason='stop')]
    mock_client = MockMistralAI.create_stream_mock(stream)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model=model, output_type=_NullableIntField)

    async with agent.run_stream('User prompt value') as result:
        with pytest.raises(UnexpectedModelBehavior, match='Output validation failed during streaming'):
            await result.get_output()


@pytest.mark.parametrize(
    'output_type, json_chunks',
    [
        pytest.param(int, ('{"response":1', '.5}'), id='top-level-int'),
        pytest.param(list[int], ('{"response":[1', '.5]}'), id='array-of-int'),
        pytest.param(_StaleIntField, ('{"value":1', '.5}'), id='object-int-field'),
        pytest.param(_DuplicateIntField, ('{"a":0,"b":"x","a":1', '.5}'), id='duplicate-key'),
    ],
)
async def test_stream_output_defers_stale_integer_prefix(
    allow_model_requests: None,
    output_type: Any,
    json_chunks: tuple[str, str],
) -> None:
    """Regression test for https://github.com/pydantic/pydantic-ai/issues/6504.

    A live model cannot reliably reproduce the exact numeric spelling and chunk boundary, so this
    uses mocked SDK chunks.

    When a chunk boundary falls inside a number right after an integral prefix (`1`) and the
    completed value is non-integral (`1.5`), the partial parse (e.g. `{"response": 1`) used to be
    emitted as a stale `1`. The completed `1.5` then fails `integer` validation, so nothing replaced
    the stale args and the run silently returned `1`, a value the model never produced. The emission
    must instead be deferred until the number is complete, so the invalid `1.5` reaches the
    output-retry path rather than being silently truncated. As the issue notes, a top-level integer,
    an integer array item, and an integer object field all share this hazard.
    """
    stream = [text_chunk(json_chunks[0]), text_chunk(json_chunks[1], finish_reason='stop')]
    mock_client = MockMistralAI.create_stream_mock(stream)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent: Agent[None, Any] = Agent(model=model, output_type=output_type)

    # `run_stream` exhausts output retries and raises while entering the context manager, so the
    # body never runs.
    with pytest.raises(UnexpectedModelBehavior, match='Exceeded maximum output retries'):
        async with agent.run_stream('User prompt value'):
            pass  # pragma: no cover


async def test_stream_result_type_primitif_array(allow_model_requests: None):
    """This test tests the primitif result with the pydantic ai format model response"""

    stream = [
        # {'response':
        text_chunk('{'),
        text_chunk('"resp'),
        text_chunk('onse":'),
        text_chunk('['),
        text_chunk('"'),
        text_chunk('f'),
        text_chunk('i'),
        text_chunk('r'),
        text_chunk('s'),
        text_chunk('t'),
        text_chunk('"'),
        text_chunk(','),
        text_chunk('"'),
        text_chunk('O'),
        text_chunk('n'),
        text_chunk('e'),
        text_chunk('"'),
        text_chunk(','),
        text_chunk('"'),
        text_chunk('s'),
        text_chunk('e'),
        text_chunk('c'),
        text_chunk('o'),
        text_chunk('n'),
        text_chunk('d'),
        text_chunk('"'),
        text_chunk(','),
        text_chunk('"'),
        text_chunk('T'),
        text_chunk('w'),
        text_chunk('o'),
        text_chunk('"'),
        text_chunk(']'),
        text_chunk('}'),
        chunk([]),
    ]

    mock_client = MockMistralAI.create_stream_mock(stream)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model, output_type=list[str])

    async with agent.run_stream('User prompt value') as result:
        assert not result.is_complete
        v = [c async for c in result.stream_output(debounce_by=None)]
        assert v == snapshot(
            [
                [],
                [''],
                ['f'],
                ['fi'],
                ['fir'],
                ['firs'],
                ['first'],
                ['first'],
                ['first'],
                ['first', ''],
                ['first', 'O'],
                ['first', 'On'],
                ['first', 'One'],
                ['first', 'One'],
                ['first', 'One'],
                ['first', 'One', ''],
                ['first', 'One', 's'],
                ['first', 'One', 'se'],
                ['first', 'One', 'sec'],
                ['first', 'One', 'seco'],
                ['first', 'One', 'secon'],
                ['first', 'One', 'second'],
                ['first', 'One', 'second'],
                ['first', 'One', 'second'],
                ['first', 'One', 'second', ''],
                ['first', 'One', 'second', 'T'],
                ['first', 'One', 'second', 'Tw'],
                ['first', 'One', 'second', 'Two'],
                ['first', 'One', 'second', 'Two'],
                ['first', 'One', 'second', 'Two'],
                ['first', 'One', 'second', 'Two'],
                ['first', 'One', 'second', 'Two'],
            ]
        )
        assert result.is_complete
        assert result.usage.input_tokens == 35
        assert result.usage.output_tokens == 35

        # double check usage matches stream count
        assert result.usage.output_tokens == len(stream)


async def test_stream_result_type_basemodel_with_default_params(allow_model_requests: None):
    class MyTypedBaseModel(BaseModel):
        first: str = ''  # Note: Default, set value.
        second: str = ''  # Note: Default, set value.

    stream = [
        text_chunk('{'),
        text_chunk('"'),
        text_chunk('f'),
        text_chunk('i'),
        text_chunk('r'),
        text_chunk('s'),
        text_chunk('t'),
        text_chunk('"'),
        text_chunk(':'),
        text_chunk(' '),
        text_chunk('"'),
        text_chunk('O'),
        text_chunk('n'),
        text_chunk('e'),
        text_chunk('"'),
        text_chunk(','),
        text_chunk(' '),
        text_chunk('"'),
        text_chunk('s'),
        text_chunk('e'),
        text_chunk('c'),
        text_chunk('o'),
        text_chunk('n'),
        text_chunk('d'),
        text_chunk('"'),
        text_chunk(':'),
        text_chunk(' '),
        text_chunk('"'),
        text_chunk('T'),
        text_chunk('w'),
        text_chunk('o'),
        text_chunk('"'),
        text_chunk('}'),
        chunk([]),
    ]

    mock_client = MockMistralAI.create_stream_mock(stream)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model=model, output_type=MyTypedBaseModel)

    async with agent.run_stream('User prompt value') as result:
        assert not result.is_complete
        v = [c async for c in result.stream_output(debounce_by=None)]
        assert v == snapshot(
            [
                MyTypedBaseModel(first='', second=''),
                MyTypedBaseModel(first='O', second=''),
                MyTypedBaseModel(first='On', second=''),
                MyTypedBaseModel(first='One', second=''),
                MyTypedBaseModel(first='One', second=''),
                MyTypedBaseModel(first='One', second=''),
                MyTypedBaseModel(first='One', second=''),
                MyTypedBaseModel(first='One', second=''),
                MyTypedBaseModel(first='One', second=''),
                MyTypedBaseModel(first='One', second=''),
                MyTypedBaseModel(first='One', second=''),
                MyTypedBaseModel(first='One', second=''),
                MyTypedBaseModel(first='One', second=''),
                MyTypedBaseModel(first='One', second=''),
                MyTypedBaseModel(first='One', second=''),
                MyTypedBaseModel(first='One', second=''),
                MyTypedBaseModel(first='One', second=''),
                MyTypedBaseModel(first='One', second=''),
                MyTypedBaseModel(first='One', second='T'),
                MyTypedBaseModel(first='One', second='Tw'),
                MyTypedBaseModel(first='One', second='Two'),
                MyTypedBaseModel(first='One', second='Two'),
                MyTypedBaseModel(first='One', second='Two'),
                MyTypedBaseModel(first='One', second='Two'),
            ]
        )
        assert result.is_complete
        assert result.usage.input_tokens == 34
        assert result.usage.output_tokens == 34

        # double check usage matches stream count
        assert result.usage.output_tokens == len(stream)


async def test_stream_result_type_basemodel_with_required_params(allow_model_requests: None):
    class MyTypedBaseModel(BaseModel):
        first: str  # Note: Required params
        second: str  # Note: Required params

    stream = [
        text_chunk('{'),
        text_chunk('"'),
        text_chunk('f'),
        text_chunk('i'),
        text_chunk('r'),
        text_chunk('s'),
        text_chunk('t'),
        text_chunk('"'),
        text_chunk(':'),
        text_chunk(' '),
        text_chunk('"'),
        text_chunk('O'),
        text_chunk('n'),
        text_chunk('e'),
        text_chunk('"'),
        text_chunk(','),
        text_chunk(' '),
        text_chunk('"'),
        text_chunk('s'),
        text_chunk('e'),
        text_chunk('c'),
        text_chunk('o'),
        text_chunk('n'),
        text_chunk('d'),
        text_chunk('"'),
        text_chunk(':'),
        text_chunk(' '),
        text_chunk('"'),
        text_chunk('T'),
        text_chunk('w'),
        text_chunk('o'),
        text_chunk('"'),
        text_chunk('}'),
        chunk([]),
    ]

    mock_client = MockMistralAI.create_stream_mock(stream)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model=model, output_type=MyTypedBaseModel)

    async with agent.run_stream('User prompt value') as result:
        assert not result.is_complete
        v = [c async for c in result.stream_output(debounce_by=None)]
        assert v == snapshot(
            [
                MyTypedBaseModel(first='One', second=''),
                MyTypedBaseModel(first='One', second='T'),
                MyTypedBaseModel(first='One', second='Tw'),
                MyTypedBaseModel(first='One', second='Two'),
                MyTypedBaseModel(first='One', second='Two'),
                MyTypedBaseModel(first='One', second='Two'),
                MyTypedBaseModel(first='One', second='Two'),
            ]
        )
        assert result.is_complete
        assert result.usage.input_tokens == 34
        assert result.usage.output_tokens == 34

        # double check cost matches stream count
        assert result.usage.output_tokens == len(stream)


#####################
## Completion Function call
#####################


async def test_request_tool_call(allow_model_requests: None):
    completion = [
        completion_message(
            MistralAssistantMessage(
                content=None,
                role='assistant',
                tool_calls=[
                    MistralToolCall(
                        id='1',
                        function=MistralFunctionCall(arguments='{"loc_name": "San Fransisco"}', name='get_location'),
                        type='function',
                    )
                ],
            ),
            usage=MistralUsageInfo(
                completion_tokens=1,
                prompt_tokens=2,
                total_tokens=3,
            ),
        ),
        completion_message(
            MistralAssistantMessage(
                content=None,
                role='assistant',
                tool_calls=[
                    MistralToolCall(
                        id='2',
                        function=MistralFunctionCall(arguments='{"loc_name": "London"}', name='get_location'),
                        type='function',
                    )
                ],
            ),
            usage=MistralUsageInfo(
                completion_tokens=2,
                prompt_tokens=3,
                total_tokens=6,
            ),
        ),
        completion_message(MistralAssistantMessage(content='final response', role='assistant')),
    ]
    mock_client = MockMistralAI.create_mock(completion)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model, system_prompt='this is the system prompt')

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        if loc_name == 'London':
            return json.dumps({'lat': 51, 'lng': 0})
        else:
            raise ModelRetry('Wrong location, please try again')

    result = await agent.run('Hello')

    assert result.output == 'final response'
    assert result.usage.input_tokens == 6
    assert result.usage.output_tokens == 4
    assert result.usage.total_tokens == 10
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='this is the system prompt', timestamp=IsNow(tz=timezone.utc)),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_location',
                        args='{"loc_name": "San Fransisco"}',
                        tool_call_id='1',
                    )
                ],
                usage=RequestUsage(input_tokens=2, output_tokens=1),
                model_name='mistral-large-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Wrong location, please try again',
                        tool_name='get_location',
                        tool_call_id='1',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_location',
                        args='{"loc_name": "London"}',
                        tool_call_id='2',
                    )
                ],
                usage=RequestUsage(input_tokens=3, output_tokens=2),
                model_name='mistral-large-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_location',
                        content='{"lat": 51, "lng": 0}',
                        tool_call_id='2',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='final response')],
                usage=RequestUsage(input_tokens=1, output_tokens=1),
                model_name='mistral-large-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_request_tool_call_with_result_type(allow_model_requests: None):
    class MyTypedDict(TypedDict, total=False):
        lat: int
        lng: int

    completion = [
        completion_message(
            MistralAssistantMessage(
                content=None,
                role='assistant',
                tool_calls=[
                    MistralToolCall(
                        id='1',
                        function=MistralFunctionCall(arguments='{"loc_name": "San Fransisco"}', name='get_location'),
                        type='function',
                    )
                ],
            ),
            usage=MistralUsageInfo(
                completion_tokens=1,
                prompt_tokens=2,
                total_tokens=3,
            ),
        ),
        completion_message(
            MistralAssistantMessage(
                content=None,
                role='assistant',
                tool_calls=[
                    MistralToolCall(
                        id='2',
                        function=MistralFunctionCall(arguments='{"loc_name": "London"}', name='get_location'),
                        type='function',
                    )
                ],
            ),
            usage=MistralUsageInfo(
                completion_tokens=2,
                prompt_tokens=3,
                total_tokens=6,
            ),
        ),
        completion_message(
            MistralAssistantMessage(
                content=None,
                role='assistant',
                tool_calls=[
                    MistralToolCall(
                        id='1',
                        function=MistralFunctionCall(arguments='{"lat": 51, "lng": 0}', name='final_result'),
                        type='function',
                    )
                ],
            ),
            usage=MistralUsageInfo(
                completion_tokens=1,
                prompt_tokens=2,
                total_tokens=3,
            ),
        ),
    ]
    mock_client = MockMistralAI.create_mock(completion)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model, instructions='this is the system prompt', output_type=MyTypedDict)

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        if loc_name == 'London':
            return json.dumps({'lat': 51, 'lng': 0})
        else:
            raise ModelRetry('Wrong location, please try again')

    result = await agent.run('Hello')

    assert result.output == {'lat': 51, 'lng': 0}
    assert result.usage.input_tokens == 7
    assert result.usage.output_tokens == 4
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
                ],
                instructions='this is the system prompt',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_location',
                        args='{"loc_name": "San Fransisco"}',
                        tool_call_id='1',
                    )
                ],
                usage=RequestUsage(input_tokens=2, output_tokens=1),
                model_name='mistral-large-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Wrong location, please try again',
                        tool_name='get_location',
                        tool_call_id='1',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                instructions='this is the system prompt',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_location',
                        args='{"loc_name": "London"}',
                        tool_call_id='2',
                    )
                ],
                usage=RequestUsage(input_tokens=3, output_tokens=2),
                model_name='mistral-large-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_location',
                        content='{"lat": 51, "lng": 0}',
                        tool_call_id='2',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                instructions='this is the system prompt',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"lat": 51, "lng": 0}',
                        tool_call_id='1',
                    )
                ],
                usage=RequestUsage(input_tokens=2, output_tokens=1),
                model_name='mistral-large-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='1',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


#####################
## Completion Function call Stream
#####################


async def test_stream_tool_call_with_return_type(allow_model_requests: None):
    class MyTypedDict(TypedDict, total=False):
        won: bool

    completion = [
        [
            chunk(
                delta=[MistralDeltaMessage(role=MistralUnset(), content='', tool_calls=MistralUnset())],
                finish_reason='tool_calls',
            ),
            func_chunk(
                tool_calls=[
                    MistralToolCall(
                        id='1',
                        function=MistralFunctionCall(arguments='{"loc_name": "San Fransisco"}', name='get_location'),
                        type='function',
                    )
                ],
                finish_reason='tool_calls',
            ),
        ],
        [
            chunk(
                delta=[MistralDeltaMessage(role=MistralUnset(), content='', tool_calls=MistralUnset())],
                finish_reason='tool_calls',
            ),
            func_chunk(
                tool_calls=[
                    MistralToolCall(
                        id='1',
                        function=MistralFunctionCall(arguments='{"won": true}', name='final_result'),
                        type=None,
                    )
                ],
                finish_reason='tool_calls',
            ),
        ],
    ]

    mock_client = MockMistralAI.create_stream_mock(completion)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model, instructions='this is the system prompt', output_type=MyTypedDict)

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        return json.dumps({'lat': 51, 'lng': 0})

    async with agent.run_stream('User prompt value') as result:
        assert not result.is_complete
        v = [c async for c in result.stream_output(debounce_by=None)]
        assert v == snapshot([{'won': True}, {'won': True}])
        assert result.is_complete
        assert result.timestamp == IsNow(tz=timezone.utc)
        assert result.usage.input_tokens == 4
        assert result.usage.output_tokens == 4

        # double check usage matches stream count
        assert result.usage.output_tokens == 4

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='User prompt value', timestamp=IsNow(tz=timezone.utc)),
                ],
                instructions='this is the system prompt',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_location',
                        args='{"loc_name": "San Fransisco"}',
                        tool_call_id='1',
                    )
                ],
                usage=RequestUsage(input_tokens=2, output_tokens=2),
                model_name='gpt-4',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'tool_calls',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='x',
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_location',
                        content='{"lat": 51, "lng": 0}',
                        tool_call_id='1',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                instructions='this is the system prompt',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='final_result', args='{"won": true}', tool_call_id='1')],
                usage=RequestUsage(input_tokens=2, output_tokens=2),
                model_name='gpt-4',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'tool_calls',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='x',
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='1',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    assert await result.get_output() == {'won': True}


async def test_stream_tool_call(allow_model_requests: None):
    completion = [
        [
            chunk(
                delta=[MistralDeltaMessage(role=MistralUnset(), content='', tool_calls=MistralUnset())],
                finish_reason='tool_calls',
            ),
            func_chunk(
                tool_calls=[
                    MistralToolCall(
                        id='1',
                        function=MistralFunctionCall(arguments='{"loc_name": "San Fransisco"}', name='get_location'),
                        type='function',
                    )
                ],
                finish_reason='tool_calls',
            ),
        ],
        [
            chunk(delta=[MistralDeltaMessage(role='assistant', content='', tool_calls=MistralUnset())]),
            chunk(delta=[MistralDeltaMessage(role=MistralUnset(), content='final ', tool_calls=MistralUnset())]),
            chunk(delta=[MistralDeltaMessage(role=MistralUnset(), content='response', tool_calls=MistralUnset())]),
            chunk(
                delta=[MistralDeltaMessage(role=MistralUnset(), content='', tool_calls=MistralUnset())],
                finish_reason='stop',
            ),
        ],
    ]

    mock_client = MockMistralAI.create_stream_mock(completion)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model, instructions='this is the system prompt')

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        return json.dumps({'lat': 51, 'lng': 0})

    async with agent.run_stream('User prompt value') as result:
        assert not result.is_complete
        v = [c async for c in result.stream_output(debounce_by=None)]
        assert v == snapshot(['final ', 'final response', 'final response'])
        assert result.is_complete
        assert result.timestamp == IsNow(tz=timezone.utc)
        assert result.usage.input_tokens == 6
        assert result.usage.output_tokens == 6

        # double check usage matches stream count
        assert result.usage.output_tokens == 6

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='User prompt value', timestamp=IsNow(tz=timezone.utc)),
                ],
                instructions='this is the system prompt',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_location',
                        args='{"loc_name": "San Fransisco"}',
                        tool_call_id='1',
                    )
                ],
                usage=RequestUsage(input_tokens=2, output_tokens=2),
                model_name='gpt-4',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'tool_calls',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='x',
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_location',
                        content='{"lat": 51, "lng": 0}',
                        tool_call_id='1',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                instructions='this is the system prompt',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='final response')],
                usage=RequestUsage(input_tokens=4, output_tokens=4),
                model_name='gpt-4',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='x',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_stream_tool_call_with_retry(allow_model_requests: None):
    completion = [
        [
            chunk(
                delta=[MistralDeltaMessage(role=MistralUnset(), content='', tool_calls=MistralUnset())],
                finish_reason='tool_calls',
            ),
            func_chunk(
                tool_calls=[
                    MistralToolCall(
                        id='1',
                        function=MistralFunctionCall(arguments='{"loc_name": "San Fransisco"}', name='get_location'),
                        type='function',
                    )
                ],
                finish_reason='tool_calls',
            ),
        ],
        [
            func_chunk(
                tool_calls=[
                    MistralToolCall(
                        id='2',
                        function=MistralFunctionCall(arguments='{"loc_name": "London"}', name='get_location'),
                        type='function',
                    )
                ],
                finish_reason='tool_calls',
            ),
        ],
        [
            chunk(delta=[MistralDeltaMessage(role='assistant', content='', tool_calls=MistralUnset())]),
            chunk(delta=[MistralDeltaMessage(role=MistralUnset(), content='final ', tool_calls=MistralUnset())]),
            chunk(delta=[MistralDeltaMessage(role=MistralUnset(), content='response', tool_calls=MistralUnset())]),
            chunk(
                delta=[MistralDeltaMessage(role=MistralUnset(), content='', tool_calls=MistralUnset())],
                finish_reason='stop',
            ),
        ],
    ]

    mock_client = MockMistralAI.create_stream_mock(completion)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model, instructions='this is the system prompt')

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        if loc_name == 'London':
            return json.dumps({'lat': 51, 'lng': 0})
        else:
            raise ModelRetry('Wrong location, please try again')

    async with agent.run_stream('User prompt value') as result:
        assert not result.is_complete
        v = [c async for c in result.stream_text(debounce_by=None)]
        assert v == snapshot(['final ', 'final response'])
        assert result.is_complete
        assert result.timestamp == IsNow(tz=timezone.utc)
        assert result.usage.input_tokens == 7
        assert result.usage.output_tokens == 7

        # double check usage matches stream count
        assert result.usage.output_tokens == 7

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='User prompt value', timestamp=IsNow(tz=timezone.utc)),
                ],
                instructions='this is the system prompt',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_location',
                        args='{"loc_name": "San Fransisco"}',
                        tool_call_id='1',
                    )
                ],
                usage=RequestUsage(input_tokens=2, output_tokens=2),
                model_name='gpt-4',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'tool_calls',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='x',
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Wrong location, please try again',
                        tool_name='get_location',
                        tool_call_id='1',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                instructions='this is the system prompt',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_location',
                        args='{"loc_name": "London"}',
                        tool_call_id='2',
                    )
                ],
                usage=RequestUsage(input_tokens=1, output_tokens=1),
                model_name='gpt-4',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'tool_calls',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='x',
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_location',
                        content='{"lat": 51, "lng": 0}',
                        tool_call_id='2',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                instructions='this is the system prompt',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='final response')],
                usage=RequestUsage(input_tokens=4, output_tokens=4),
                model_name='gpt-4',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='x',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


#####################
## Test methods
#####################


def test_generate_user_output_format_complex(mistral_api_key: str):
    """
    Single test that includes properties exercising every branch
    in _get_python_type (anyOf, arrays, objects with additionalProperties, etc.).
    """
    schema = {
        'properties': {
            'prop_anyOf': {'anyOf': [{'type': 'string'}, {'type': 'integer'}]},
            'prop_no_type': {
                # no 'type' key
            },
            'prop_simple_string': {'type': 'string'},
            'prop_array_booleans': {'type': 'array', 'items': {'type': 'boolean'}},
            'prop_object_simple': {'type': 'object', 'additionalProperties': {'type': 'boolean'}},
            'prop_object_array': {
                'type': 'object',
                'additionalProperties': {'type': 'array', 'items': {'type': 'integer'}},
            },
            'prop_object_object': {'type': 'object', 'additionalProperties': {'type': 'object'}},
            'prop_object_unknown': {'type': 'object', 'additionalProperties': {'type': 'someUnknownType'}},
            'prop_unrecognized_type': {'type': 'customSomething'},
        }
    }
    m = MistralModel('', json_mode_schema_prompt='{schema}', provider=MistralProvider(api_key=mistral_api_key))
    result = m._generate_user_output_format([schema])  # pyright: ignore[reportPrivateUsage]
    assert result.content == (
        "{'prop_anyOf': 'Optional[str]', "
        "'prop_no_type': 'Any', "
        "'prop_simple_string': 'str', "
        "'prop_array_booleans': 'list[bool]', "
        "'prop_object_simple': 'dict[str, bool]', "
        "'prop_object_array': 'dict[str, list[int]]', "
        "'prop_object_object': 'dict[str, dict[str, Any]]', "
        "'prop_object_unknown': 'dict[str, Any]', "
        "'prop_unrecognized_type': 'Any'}"
    )


def test_generate_user_output_format_multiple(mistral_api_key: str):
    schema = {'properties': {'prop_anyOf': {'anyOf': [{'type': 'string'}, {'type': 'integer'}]}}}
    m = MistralModel('', json_mode_schema_prompt='{schema}', provider=MistralProvider(api_key=mistral_api_key))
    result = m._generate_user_output_format([schema, schema])  # pyright: ignore[reportPrivateUsage]
    assert result.content == "[{'prop_anyOf': 'Optional[str]'}, {'prop_anyOf': 'Optional[str]'}]"


@pytest.mark.parametrize(
    'desc, schema, data, expected',
    [
        (
            'Missing required parameter',
            {
                'required': ['name', 'age'],
                'properties': {
                    'name': {'type': 'string'},
                    'age': {'type': 'integer'},
                },
            },
            {'name': 'Alice'},  # Missing "age"
            False,
        ),
        (
            'Type mismatch (expected string, got int)',
            {'required': ['name'], 'properties': {'name': {'type': 'string'}}},
            {'name': 123},  # Should be a string, got int
            False,
        ),
        (
            'Array parameter check (param not a list)',
            {'required': ['tags'], 'properties': {'tags': {'type': 'array', 'items': {'type': 'string'}}}},
            {'tags': 'not a list'},  # Not a list
            False,
        ),
        (
            'Array item type mismatch',
            {'required': ['tags'], 'properties': {'tags': {'type': 'array', 'items': {'type': 'string'}}}},
            {'tags': ['ok', 123, 'still ok']},  # One item is int, not str
            False,
        ),
        (
            'Nested object fails',
            {
                'required': ['user'],
                'properties': {
                    'user': {
                        'type': 'object',
                        'required': ['id', 'profile'],
                        'properties': {
                            'id': {'type': 'integer'},
                            'profile': {
                                'type': 'object',
                                'required': ['address'],
                                'properties': {'address': {'type': 'string'}},
                            },
                        },
                    }
                },
            },
            {'user': {'id': 101, 'profile': {}}},  # Missing "address" in the nested profile
            False,
        ),
        (
            'All requirements met (success)',
            {
                'required': ['name', 'age', 'tags', 'user'],
                'properties': {
                    'name': {'type': 'string'},
                    'age': {'type': 'integer'},
                    'tags': {'type': 'array', 'items': {'type': 'string'}},
                    'user': {
                        'type': 'object',
                        'required': ['id', 'profile'],
                        'properties': {
                            'id': {'type': 'integer'},
                            'profile': {
                                'type': 'object',
                                'required': ['address'],
                                'properties': {'address': {'type': 'string'}},
                            },
                        },
                    },
                },
            },
            {
                'name': 'Alice',
                'age': 30,
                'tags': ['tag1', 'tag2'],
                'user': {'id': 101, 'profile': {'address': '123 Street'}},
            },
            True,
        ),
        (
            'Number accepts integer',
            {'required': ['value'], 'properties': {'value': {'type': 'number'}}},
            {'value': 20},
            True,
        ),
        (
            'Number accepts float',
            {'required': ['value'], 'properties': {'value': {'type': 'number'}}},
            {'value': 20.5},
            True,
        ),
        (
            'Number rejects boolean',
            {'required': ['value'], 'properties': {'value': {'type': 'number'}}},
            {'value': True},
            False,
        ),
        (
            'Integer accepts float with zero fractional part',
            {'required': ['value'], 'properties': {'value': {'type': 'integer'}}},
            {'value': 1.0},
            True,
        ),
        (
            'Integer rejects float with fractional part',
            {'required': ['value'], 'properties': {'value': {'type': 'integer'}}},
            {'value': 1.5},
            False,
        ),
        (
            'Integer rejects boolean',
            {'required': ['value'], 'properties': {'value': {'type': 'integer'}}},
            {'value': True},
            False,
        ),
        (
            'Boolean accepts booleans',
            {
                'required': ['true_value', 'false_value'],
                'properties': {
                    'true_value': {'type': 'boolean'},
                    'false_value': {'type': 'boolean'},
                },
            },
            {'true_value': True, 'false_value': False},
            True,
        ),
        (
            'Boolean rejects integer',
            {'required': ['value'], 'properties': {'value': {'type': 'boolean'}}},
            {'value': 1},
            False,
        ),
        (
            'Nested number accepts integer',
            {
                'required': ['outer'],
                'properties': {
                    'outer': {
                        'type': 'object',
                        'required': ['inner'],
                        'properties': {'inner': {'type': 'number'}},
                    }
                },
            },
            {'outer': {'inner': 20}},
            True,
        ),
        (
            'Array of number accepts integers',
            {'required': ['values'], 'properties': {'values': {'type': 'array', 'items': {'type': 'number'}}}},
            {'values': [1, 2, 3]},
            True,
        ),
    ],
)
def test_validate_required_json_schema(desc: str, schema: dict[str, Any], data: dict[str, Any], expected: bool) -> None:
    result = MistralStreamedResponse._validate_required_json_schema(data, schema)  # pyright: ignore[reportPrivateUsage]
    assert result == expected, f'{desc} — expected {expected}, got {result}'


@pytest.mark.vcr()
async def test_image_as_binary_content_tool_response(
    allow_model_requests: None, mistral_api_key: str, image_content: BinaryContent
):
    m = MistralModel('pixtral-12b-latest', provider=MistralProvider(api_key=mistral_api_key))
    agent = Agent(m)

    @agent.tool_plain
    async def get_image() -> BinaryContent:
        return image_content

    result = await agent.run(['What fruit is in the image you can get from the get_image tool? Call the tool.'])
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=['What fruit is in the image you can get from the get_image tool? Call the tool.'],
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_image', args='{}', tool_call_id='FI5qQGzDE')],
                usage=RequestUsage(input_tokens=65, output_tokens=16),
                model_name='pixtral-12b-latest',
                timestamp=IsDatetime(),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={'finish_reason': 'tool_calls', 'timestamp': IsDatetime()},
                provider_response_id='20c656d7c70e4362858160d9d241ce92',
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_image',
                        content=IsInstance(BinaryImage),
                        tool_call_id='FI5qQGzDE',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='The image shows a kiwi fruit that has been cut in half. Kiwis are small, oval-shaped fruits with a bright green flesh and tiny black seeds. They have a sweet and tangy flavor and are known for being rich in vitamin C and fiber.'
                    )
                ],
                usage=RequestUsage(input_tokens=1540, output_tokens=54),
                model_name='pixtral-12b-latest',
                timestamp=IsDatetime(),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={'finish_reason': 'stop', 'timestamp': IsDatetime()},
                provider_response_id='b9df7d6167a74543aed6c27557ab0a29',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_text_content_input(allow_model_requests: None):
    c = completion_message(MistralAssistantMessage(content='world', role='assistant'))
    mock_client = MockMistralAI.create_mock(c)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))

    part = UserPromptPart(
        content=[
            'Hello',
            TextContent(content='This is some text content.', metadata={'key': 'value'}),
        ]
    )
    m = await model._map_user_prompt(part)  # pyright: ignore[reportPrivateUsage]
    assert m == snapshot(UserMessage(content=[TextChunk(text='Hello'), TextChunk(text='This is some text content.')]))


async def test_image_url_input(allow_model_requests: None):
    c = completion_message(MistralAssistantMessage(content='world', role='assistant'))
    mock_client = MockMistralAI.create_mock(c)
    m = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(m)

    result = await agent.run(
        [
            'hello',
            ImageUrl(url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg'),
        ]
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'hello',
                            ImageUrl(
                                url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg',
                                identifier='bd38f5',
                            ),
                        ],
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='world')],
                usage=RequestUsage(input_tokens=1, output_tokens=1),
                model_name='mistral-large-123',
                timestamp=IsDatetime(),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_image_as_binary_content_input(allow_model_requests: None):
    c = completion_message(MistralAssistantMessage(content='world', role='assistant'))
    mock_client = MockMistralAI.create_mock(c)
    m = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(m)

    # Fake image bytes for testing
    image_bytes = b'fake image data'

    result = await agent.run(['hello', BinaryContent(data=image_bytes, media_type='image/jpeg')])
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'hello',
                            BinaryContent(data=image_bytes, media_type='image/jpeg'),
                        ],
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='world')],
                usage=RequestUsage(input_tokens=1, output_tokens=1),
                model_name='mistral-large-123',
                timestamp=IsDatetime(),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


def get_mock_chat_completion_kwargs(mistral_client: Mistral) -> list[dict[str, Any]]:
    if isinstance(mistral_client, MockMistralAI):
        return mistral_client.chat_completion_kwargs
    else:  # pragma: no cover
        raise RuntimeError('Not a MockMistralAI instance')


async def test_image_detail_vendor_metadata(allow_model_requests: None):
    """`vendor_metadata['detail']` is forwarded to the Mistral API for image inputs."""
    c = completion_message(MistralAssistantMessage(content='done', role='assistant'))
    mock_client = MockMistralAI.create_mock(c)
    m = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(m)

    image_url = ImageUrl('https://example.com/image.png', vendor_metadata={'detail': 'high'})
    binary_image = BinaryContent(b'\x89PNG', media_type='image/png', vendor_metadata={'detail': 'low'})

    await agent.run(['Describe these images.', image_url, binary_image])

    messages = get_mock_chat_completion_kwargs(mock_client)[0]['messages']
    details = [
        chunk.image_url.detail
        for chunk in messages[0].content
        if isinstance(chunk, MistralImageURLChunk) and isinstance(chunk.image_url, MistralImageURL)
    ]
    assert details == snapshot(['high', 'low'])


async def test_pdf_url_input(allow_model_requests: None):
    c = completion_message(MistralAssistantMessage(content='world', role='assistant'))
    mock_client = MockMistralAI.create_mock(c)
    m = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(m)

    result = await agent.run(
        [
            'hello',
            DocumentUrl(url='https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf'),
        ]
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'hello',
                            DocumentUrl(
                                url='https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf',
                                identifier='c6720d',
                            ),
                        ],
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='world')],
                usage=RequestUsage(input_tokens=1, output_tokens=1),
                model_name='mistral-large-123',
                timestamp=IsDatetime(),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_pdf_as_binary_content_input(allow_model_requests: None):
    c = completion_message(MistralAssistantMessage(content='world', role='assistant'))
    mock_client = MockMistralAI.create_mock(c)
    m = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(m)

    base64_content = b'%PDF-1.\rtrailer<</Root<</Pages<</Kids[<</MediaBox[0 0 3 3]>>>>>>>>>'

    result = await agent.run(['hello', BinaryContent(data=base64_content, media_type='application/pdf')])
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'hello',
                            BinaryContent(data=base64_content, media_type='application/pdf', identifier='b9d976'),
                        ],
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='world')],
                usage=RequestUsage(input_tokens=1, output_tokens=1),
                model_name='mistral-large-123',
                timestamp=IsDatetime(),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_txt_url_input(allow_model_requests: None):
    c = completion_message(MistralAssistantMessage(content='world', role='assistant'))
    mock_client = MockMistralAI.create_mock(c)
    m = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(m)

    with pytest.raises(
        NotImplementedError, match='DocumentUrl other than PDF is not supported in Mistral user prompts'
    ):
        await agent.run(
            [
                'hello',
                DocumentUrl(url='https://examplefiles.org/files/documents/plaintext-example-file-download.txt'),
            ]
        )


async def test_audio_as_binary_content_input(allow_model_requests: None):
    c = completion_message(MistralAssistantMessage(content='world', role='assistant'))
    mock_client = MockMistralAI.create_mock(c)
    m = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(m)

    base64_content = b'//uQZ'

    with pytest.raises(
        NotImplementedError, match='BinaryContent other than image or PDF is not supported in Mistral user prompts'
    ):
        await agent.run(['hello', BinaryContent(data=base64_content, media_type='audio/wav')])


async def test_video_url_input(allow_model_requests: None):
    c = completion_message(MistralAssistantMessage(content='world', role='assistant'))
    mock_client = MockMistralAI.create_mock(c)
    m = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(m)

    with pytest.raises(NotImplementedError, match='VideoUrl is not supported in Mistral user prompts'):
        await agent.run(['hello', VideoUrl(url='https://www.google.com')])


async def test_uploaded_file_input(allow_model_requests: None):
    c = completion_message(MistralAssistantMessage(content='world', role='assistant'))
    mock_client = MockMistralAI.create_mock(c)
    m = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(m)

    with pytest.raises(NotImplementedError, match='UploadedFile is not supported in Mistral user prompts'):
        await agent.run(['hello', UploadedFile(file_id='file-123', provider_name='anthropic')])


def test_model_status_error(allow_model_requests: None) -> None:
    response = httpx.Response(500, content=b'test error')
    mock_client = MockMistralAI.create_mock(SDKError('test error', raw_response=response))
    m = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(m)
    with pytest.raises(ModelHTTPError) as exc_info:
        agent.run_sync('hello')
    assert str(exc_info.value) == snapshot('status_code: 500, model_name: mistral-large-latest, body: test error')


def test_model_non_http_error(allow_model_requests: None) -> None:
    response = httpx.Response(300, content=b'redirect')
    mock_client = MockMistralAI.create_mock(SDKError('Connection error', raw_response=response))
    m = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(m)
    with pytest.raises(ModelAPIError) as exc_info:
        agent.run_sync('hello')
    assert exc_info.value.model_name == 'mistral-large-latest'


async def test_mistral_model_instructions(allow_model_requests: None, mistral_api_key: str):
    c = completion_message(MistralAssistantMessage(content='world', role='assistant'))
    mock_client = MockMistralAI.create_mock(c)
    m = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(m, instructions='You are a helpful assistant.')

    result = await agent.run('hello')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                timestamp=IsNow(tz=timezone.utc),
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='world')],
                usage=RequestUsage(input_tokens=1, output_tokens=1),
                model_name='mistral-large-123',
                timestamp=IsDatetime(),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


@pytest.mark.vcr()
async def test_mistral_forwards_penalties(allow_model_requests: None, mistral_api_key: str, vcr: Cassette):
    m = MistralModel('mistral-large-latest', provider=MistralProvider(api_key=mistral_api_key))
    agent = Agent(m, model_settings=MistralModelSettings(presence_penalty=0.5, frequency_penalty=0.25))

    result = await agent.run('hello')

    assert result.output
    sent = json.loads(vcr.requests[0].body)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    assert sent['presence_penalty'] == 0.5
    assert sent['frequency_penalty'] == 0.25


@pytest.mark.vcr()
async def test_mistral_model_thinking_part(allow_model_requests: None, openai_api_key: str, mistral_api_key: str):
    openai_model = OpenAIResponsesModel('o3-mini', provider=OpenAIProvider(api_key=openai_api_key))
    settings = OpenAIResponsesModelSettings(openai_reasoning_effort='high', openai_reasoning_summary='detailed')
    agent = Agent(openai_model, model_settings=settings)

    result = await agent.run('How do I cross the street?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='How do I cross the street?', timestamp=IsDatetime())],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68bb645d50f48196a0c49fd603b87f4503498c8aa840cf12',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68bb645d50f48196a0c49fd603b87f4503498c8aa840cf12',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68bb645d50f48196a0c49fd603b87f4503498c8aa840cf12',
                        provider_name='openai',
                    ),
                    TextPart(
                        content=IsStr(),
                        id='msg_68bb64663d1c8196b9c7e78e7018cc4103498c8aa840cf12',
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(input_tokens=13, output_tokens=1616, details={'reasoning_tokens': 1344}),
                model_name='o3-mini-2025-01-31',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 5, 22, 29, 38, tzinfo=timezone.utc),
                },
                provider_response_id='resp_68bb6452990081968f5aff503a55e3b903498c8aa840cf12',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    mistral_model = MistralModel('magistral-medium-latest', provider=MistralProvider(api_key=mistral_api_key))
    result = await agent.run(
        'Considering the way to cross the street, analogously, how do I cross the river?',
        model=mistral_model,
        message_history=result.all_messages(),
    )
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Considering the way to cross the street, analogously, how do I cross the river?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content=IsStr()),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(input_tokens=664, output_tokens=747),
                model_name='magistral-medium-latest',
                timestamp=IsDatetime(),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2025, 9, 5, 22, 30, tzinfo=timezone.utc),
                },
                provider_response_id='9abe8b736bff46af8e979b52334a57cd',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


@pytest.mark.vcr()
async def test_mistral_model_thinking_part_iter(allow_model_requests: None, mistral_api_key: str):
    model = MistralModel('magistral-medium-latest', provider=MistralProvider(api_key=mistral_api_key))
    agent = Agent(model)

    async with agent.iter(user_prompt='How do I cross the street?') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for _ in request_stream:
                        pass

    assert agent_run.result is not None
    assert agent_run.result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='How do I cross the street?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='Okay, the user is asking how to cross the street. I know that crossing the street safely involves a few key steps: first, look both ways to check for oncoming traffic; second, use a crosswalk if one is available; third, obey any traffic signals or signs that may be present; and finally, proceed with caution until you have safely reached the other side. Let me compile this information into a clear and concise response.'
                    ),
                    TextPart(
                        content="""\
To cross the street safely, follow these steps:

1. Look both ways to check for oncoming traffic.
2. Use a crosswalk if one is available.
3. Obey any traffic signals or signs that may be present.
4. Proceed with caution until you have safely reached the other side.

```markdown
To cross the street safely, follow these steps:

1. Look both ways to check for oncoming traffic.
2. Use a crosswalk if one is available.
3. Obey any traffic signals or signs that may be present.
4. Proceed with caution until you have safely reached the other side.
```

By following these steps, you can ensure a safe crossing.\
"""
                    ),
                ],
                usage=RequestUsage(input_tokens=10, output_tokens=232),
                model_name='magistral-medium-latest',
                timestamp=IsDatetime(),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2025, 11, 28, 2, 19, 53, tzinfo=timezone.utc),
                },
                provider_response_id='9f9d90210f194076abeee223863eaaf0',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_image_url_force_download() -> None:
    """Test that force_download=True calls download_item for ImageUrl in MistralModel."""
    from unittest.mock import AsyncMock, patch

    m = MistralModel('mistral-large-2512', provider=MistralProvider(api_key='test-key'))

    with patch('pydantic_ai.models.mistral.download_item', new_callable=AsyncMock) as mock_download:
        mock_download.return_value = {
            'data': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
            'data_type': 'image/png',
        }

        messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'Test image',
                            ImageUrl(
                                url='https://example.com/image.png',
                                media_type='image/png',
                                force_download=True,
                            ),
                        ]
                    )
                ]
            )
        ]

        await m._map_messages(messages, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]

        mock_download.assert_called_once()
        assert mock_download.call_args[0][0].url == 'https://example.com/image.png'
        assert mock_download.call_args[1]['data_format'] == 'base64_uri'


async def test_image_url_no_force_download() -> None:
    """Test that force_download=False does not call download_item for ImageUrl in MistralModel."""
    from unittest.mock import AsyncMock, patch

    m = MistralModel('mistral-large-2512', provider=MistralProvider(api_key='test-key'))

    with patch('pydantic_ai.models.mistral.download_item', new_callable=AsyncMock) as mock_download:
        messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'Test image',
                            ImageUrl(
                                url='https://example.com/image.png',
                                media_type='image/png',
                                force_download=False,
                            ),
                        ]
                    )
                ]
            )
        ]

        await m._map_messages(messages, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]

        mock_download.assert_not_called()


async def test_document_url_force_download() -> None:
    """Test that force_download=True calls download_item for DocumentUrl PDF in MistralModel."""
    from unittest.mock import AsyncMock, patch

    m = MistralModel('mistral-large-2512', provider=MistralProvider(api_key='test-key'))

    with patch('pydantic_ai.models.mistral.download_item', new_callable=AsyncMock) as mock_download:
        mock_download.return_value = {
            'data': 'data:application/pdf;base64,JVBERi0xLjQKJdPr6eEKMSAwIG9iago8PC9UeXBlL',
            'data_type': 'application/pdf',
        }

        messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'Test PDF',
                            DocumentUrl(
                                url='https://example.com/document.pdf',
                                media_type='application/pdf',
                                force_download=True,
                            ),
                        ]
                    )
                ]
            )
        ]

        await m._map_messages(messages, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]

        mock_download.assert_called_once()
        assert mock_download.call_args[0][0].url == 'https://example.com/document.pdf'
        assert mock_download.call_args[1]['data_format'] == 'base64_uri'


async def test_document_url_no_force_download() -> None:
    """Test that force_download=False does not call download_item for DocumentUrl PDF in MistralModel."""
    from unittest.mock import AsyncMock, patch

    m = MistralModel('mistral-large-2512', provider=MistralProvider(api_key='test-key'))

    with patch('pydantic_ai.models.mistral.download_item', new_callable=AsyncMock) as mock_download:
        messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'Test PDF',
                            DocumentUrl(
                                url='https://example.com/document.pdf',
                                media_type='application/pdf',
                                force_download=False,
                            ),
                        ]
                    )
                ]
            )
        ]

        await m._map_messages(messages, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]

        mock_download.assert_not_called()


def test_map_content_concatenates_text_chunks() -> None:
    """Test that _map_content correctly concatenates multiple MistralTextChunks."""
    content: list[MistralContentChunk] = [
        MistralTextChunk(text='Hello'),
        MistralTextChunk(text=' world'),
    ]

    text, thinking = _map_content(content)

    assert text == 'Hello world'
    assert thinking == []


def test_get_timeout_ms() -> None:
    assert MistralModel._get_timeout_ms(None) is None  # pyright: ignore[reportPrivateUsage]
    assert MistralModel._get_timeout_ms(30) == 30000  # pyright: ignore[reportPrivateUsage]
    assert MistralModel._get_timeout_ms(1.5) == 1500  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(NotImplementedError, match=re.escape('Timeout object is not yet supported for MistralModel.')):
        MistralModel._get_timeout_ms(httpx.Timeout(30))  # pyright: ignore[reportPrivateUsage]


def test_map_content_handles_reference_chunk() -> None:
    """Test that _map_content does not fail when encountering a MistralReferenceChunk."""
    content: list[MistralContentChunk] = [
        MistralTextChunk(text='Hello'),
        MistralReferenceChunk(reference_ids=[1, 2, 3]),
        MistralTextChunk(text=' world'),
    ]

    text, thinking = _map_content(content)

    assert text == 'Hello world'
    assert thinking == []


async def test_stream_cancel(allow_model_requests: None):
    stream = [text_chunk('hello '), text_chunk('world'), chunk([])]
    mock_client = MockMistralAI.create_stream_mock(stream)
    m = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('') as result:
        async for _ in result.stream_text(delta=True, debounce_by=None):  # pragma: no branch
            break
        await result.cancel()
        await result.cancel()  # double cancel is a no-op
        assert result.cancelled

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='hello ')],
                usage=RequestUsage(input_tokens=1, output_tokens=1),
                model_name='gpt-4',
                timestamp=IsDatetime(),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={'timestamp': IsDatetime()},
                provider_response_id='x',
                run_id=IsStr(),
                conversation_id=IsStr(),
                state='interrupted',
            ),
        ]
    )


async def test_mistral_empty_response_skipped_in_history(allow_model_requests: None):
    """An empty `ModelResponse(parts=[])` must not be sent back as an assistant message with
    neither content nor tool calls, which Mistral rejects with a 400. The agent graph retries
    empty responses by emitting a `RetryPromptPart`, relying on the model adapter to omit the
    empty response from the API payload.
    """
    completions = [
        completion_message(MistralAssistantMessage(content=None, role='assistant')),
        completion_message(MistralAssistantMessage(content='hello back', role='assistant')),
    ]
    mock_client = MockMistralAI.create_mock(completions)
    m = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(m)

    result = await agent.run('hello')
    assert result.output == 'hello back'

    # The empty response is omitted from the payload (no assistant message with neither content nor
    # tool calls, which would trigger a 400); a retry prompt is appended instead so the model can
    # self-correct.
    second_call_messages = get_mock_chat_completion_kwargs(mock_client)[1]['messages']
    assert not any(message.role == 'assistant' for message in second_call_messages)
    assert [message.role for message in second_call_messages] == ['user', 'user']


#####################
## Reasoning effort
#####################

# These are kwarg-level unit tests rather than VCR tests: the cassette matcher isn't sensitive to
# the request body, so a dropped or mis-mapped `reasoning_effort` would still replay green against
# a recording. Asserting the SDK kwarg directly is what pins the mapping; the mistralai SDK omits
# `UNSET` fields from the request body at serialization. The wire-level pinning (mapped values,
# UNSET omission, magistral absence) lives in tests/test_thinking_wire_contract.py.


@pytest.mark.parametrize(
    'thinking,expected',
    [
        pytest.param(True, 'high', id='true'),
        pytest.param(False, 'none', id='false'),
        pytest.param('minimal', 'high', id='minimal'),
        pytest.param('low', 'high', id='low'),
        pytest.param('medium', 'high', id='medium'),
        pytest.param('high', 'high', id='high'),
        pytest.param('xhigh', 'high', id='xhigh'),
    ],
)
async def test_reasoning_effort_with_unified_thinking(
    allow_model_requests: None, thinking: ThinkingLevel, expected: str
) -> None:
    """Unified `thinking` values map to Mistral's `reasoning_effort` ('high' for any enabled level, 'none' only for `False`)."""
    c = completion_message(MistralAssistantMessage(content='thought deeply', role='assistant'))
    mock_client = MockMistralAI(completions=c)
    m = MistralModel('mistral-small-latest', provider=MistralProvider(mistral_client=cast(Mistral, mock_client)))
    agent = Agent(m)

    result = await agent.run('hello', model_settings=MistralModelSettings(thinking=thinking))
    assert result.output == 'thought deeply'
    assert mock_client.chat_completion_kwargs[-1]['reasoning_effort'] == expected


async def test_reasoning_effort_not_sent_for_unsupported_model(allow_model_requests: None) -> None:
    """`thinking` is silently ignored on models without adjustable reasoning, so `reasoning_effort` stays UNSET."""
    c = completion_message(MistralAssistantMessage(content='hello', role='assistant'))
    mock_client = MockMistralAI(completions=c)
    m = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=cast(Mistral, mock_client)))
    agent = Agent(m)

    result = await agent.run('hello', model_settings=MistralModelSettings(thinking='high'))
    assert result.output == 'hello'
    assert isinstance(mock_client.chat_completion_kwargs[-1]['reasoning_effort'], MistralUnset)


@pytest.mark.parametrize('thinking', [True, False, 'minimal', 'low', 'medium', 'high', 'xhigh'])
async def test_reasoning_effort_not_sent_for_always_on_model(
    allow_model_requests: None, thinking: ThinkingLevel
) -> None:
    """`magistral` always reasons, so `reasoning_effort` is never sent for any `thinking` value.

    Enabled levels reach `_translate_thinking` and are dropped by its always-on guard;
    `thinking=False` is stripped upstream in `prepare_request` and never reaches the model.
    """
    c = completion_message(MistralAssistantMessage(content='hello', role='assistant'))
    mock_client = MockMistralAI(completions=c)
    m = MistralModel('magistral-medium-latest', provider=MistralProvider(mistral_client=cast(Mistral, mock_client)))
    agent = Agent(m)

    result = await agent.run('hello', model_settings=MistralModelSettings(thinking=thinking))
    assert result.output == 'hello'
    assert isinstance(mock_client.chat_completion_kwargs[-1]['reasoning_effort'], MistralUnset)


async def test_reasoning_effort_not_sent_without_config(allow_model_requests: None) -> None:
    """Without any thinking config, reasoning_effort should be UNSET."""
    c = completion_message(MistralAssistantMessage(content='hello', role='assistant'))
    mock_client = MockMistralAI(completions=c)
    m = MistralModel('mistral-small-latest', provider=MistralProvider(mistral_client=cast(Mistral, mock_client)))
    agent = Agent(m)

    result = await agent.run('hello')
    assert result.output == 'hello'
    assert isinstance(mock_client.chat_completion_kwargs[-1]['reasoning_effort'], MistralUnset)


async def test_reasoning_effort_stream_with_unified_thinking(allow_model_requests: None) -> None:
    """Unified thinking='high' should pass reasoning_effort='high' in streaming mode."""
    stream = [text_chunk('hello '), text_chunk('world', finish_reason='stop')]
    mock_client = MockMistralAI(stream=stream)
    m = MistralModel('mistral-small-latest', provider=MistralProvider(mistral_client=cast(Mistral, mock_client)))
    agent = Agent(m)

    async with agent.run_stream('hello', model_settings=MistralModelSettings(thinking='high')) as result:
        text = await result.get_output()
    assert text == 'hello world'
    assert mock_client.chat_completion_kwargs[-1]['reasoning_effort'] == 'high'


async def test_reasoning_effort_stream_not_sent_without_config(allow_model_requests: None) -> None:
    """Without any thinking config, reasoning_effort should be UNSET in streaming mode."""
    stream = [text_chunk('hello', finish_reason='stop')]
    mock_client = MockMistralAI(stream=stream)
    m = MistralModel('mistral-small-latest', provider=MistralProvider(mistral_client=cast(Mistral, mock_client)))
    agent = Agent(m)

    async with agent.run_stream('hello') as result:
        text = await result.get_output()
    assert text == 'hello'
    assert isinstance(mock_client.chat_completion_kwargs[-1]['reasoning_effort'], MistralUnset)


async def test_reasoning_effort_stream_with_tools(allow_model_requests: None) -> None:
    """thinking=False maps to reasoning_effort='none' on the function-calling streaming path.

    This pins reasoning_effort on every request of a tool-call round trip, which issues
    multiple sequential streaming requests.
    """
    streams = [
        [
            func_chunk(
                [
                    MistralToolCall(
                        id='1',
                        function=MistralFunctionCall(arguments='{"loc_name": "London"}', name='get_location'),
                        type='function',
                    )
                ],
                finish_reason='tool_calls',
            )
        ],
        [text_chunk('done', finish_reason='stop')],
    ]
    mock_client = MockMistralAI(stream=streams)
    m = MistralModel('mistral-small-latest', provider=MistralProvider(mistral_client=cast(Mistral, mock_client)))
    agent = Agent(m)

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        return json.dumps({'lat': 51, 'lng': 0})

    async with agent.run_stream('hello', model_settings=MistralModelSettings(thinking=False)) as result:
        text = await result.get_output()
    assert text == 'done'
    assert len(mock_client.chat_completion_kwargs) == 2
    assert all(kwargs['reasoning_effort'] == 'none' for kwargs in mock_client.chat_completion_kwargs)
