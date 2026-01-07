from __future__ import annotations as _annotations

import base64
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal, cast
from unittest.mock import patch

import httpx
import pytest
from inline_snapshot import snapshot
from pydantic import AnyUrl, BaseModel, ConfigDict, Discriminator, Field, Tag
from typing_extensions import NotRequired, TypedDict

from pydantic_ai import (
    Agent,
    AudioUrl,
    BinaryContent,
    CachePoint,
    DocumentUrl,
    ImageUrl,
    ModelAPIError,
    ModelHTTPError,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelRetry,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UnexpectedModelBehavior,
    UserError,
    UserPromptPart,
)
from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.builtin_tools import ImageGenerationTool, WebSearchTool
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.output import NativeOutput, PromptedOutput, TextOutput, ToolOutput
from pydantic_ai.profiles.openai import OpenAIModelProfile, openai_model_profile
from pydantic_ai.result import RunUsage
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

from ..conftest import IsDatetime, IsNow, IsStr, TestEnv, try_import
from .mock_openai import (
    MockOpenAI,
    MockOpenAIResponses,
    completion_message,
    get_mock_chat_completion_kwargs,
    get_mock_responses_kwargs,
    response_message,
)

with try_import() as imports_successful:
    from openai import APIConnectionError, APIStatusError, AsyncOpenAI
    from openai.types import chat
    from openai.types.chat.chat_completion import ChoiceLogprobs
    from openai.types.chat.chat_completion_chunk import (
        Choice as ChunkChoice,
        ChoiceDelta,
        ChoiceDeltaToolCall,
        ChoiceDeltaToolCallFunction,
    )
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
    from openai.types.chat.chat_completion_message_function_tool_call import ChatCompletionMessageFunctionToolCall
    from openai.types.chat.chat_completion_message_tool_call import Function
    from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
    from openai.types.completion_usage import CompletionUsage, PromptTokensDetails

    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.models.openai import (
        OpenAIChatModel,
        OpenAIChatModelSettings,
        OpenAIResponsesModel,
        OpenAIResponsesModelSettings,
        OpenAISystemPromptRole,
        _resolve_openai_image_generation_size,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer
    from pydantic_ai.providers.cerebras import CerebrasProvider
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.ollama import OllamaProvider
    from pydantic_ai.providers.openai import OpenAIProvider

    MockChatCompletion = chat.ChatCompletion | Exception
    MockChatCompletionChunk = chat.ChatCompletionChunk | Exception

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


def test_init():
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key='foobar'))
    assert m.base_url == 'https://api.openai.com/v1/'
    assert m.client.api_key == 'foobar'
    assert m.model_name == 'gpt-4o'


@pytest.mark.parametrize(
    'aspect_ratio,size,expected',
    [
        # aspect_ratio is None, various sizes
        (None, None, 'auto'),
        (None, 'auto', 'auto'),
        (None, '1024x1024', '1024x1024'),
        (None, '1024x1536', '1024x1536'),
        (None, '1536x1024', '1536x1024'),
        # Valid aspect_ratios with no size
        ('1:1', None, '1024x1024'),
        ('2:3', None, '1024x1536'),
        ('3:2', None, '1536x1024'),
        # Valid aspect_ratios with compatible sizes
        ('1:1', 'auto', '1024x1024'),
        ('1:1', '1024x1024', '1024x1024'),
        ('2:3', '1024x1536', '1024x1536'),
        ('3:2', '1536x1024', '1536x1024'),
    ],
)
def test_openai_image_generation_size_valid_combinations(
    aspect_ratio: Literal['1:1', '2:3', '3:2'] | None,
    size: Literal['auto', '1024x1024', '1024x1536', '1536x1024'] | None,
    expected: Literal['auto', '1024x1024', '1024x1536', '1536x1024'],
) -> None:
    """Test valid combinations of aspect_ratio and size for OpenAI image generation."""
    tool = ImageGenerationTool(aspect_ratio=aspect_ratio, size=size)
    assert _resolve_openai_image_generation_size(tool) == expected


def test_openai_image_generation_tool_aspect_ratio_invalid() -> None:
    """Test that invalid aspect_ratio raises UserError."""
    tool = ImageGenerationTool(aspect_ratio='16:9')
    with pytest.raises(UserError, match='OpenAI image generation only supports `aspect_ratio` values'):
        _resolve_openai_image_generation_size(tool)


async def test_request_simple_success(allow_model_requests: None):
    c = completion_message(
        ChatCompletionMessage(content='world', role='assistant'),
    )
    mock_client = MockOpenAI.create_mock(c)
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m)

    result = await agent.run('hello')
    assert result.output == 'world'
    assert result.usage() == snapshot(RunUsage(requests=1))

    # reset the index so we get the same response again
    mock_client.index = 0  # type: ignore

    result = await agent.run('hello', message_history=result.new_messages())
    assert result.output == 'world'
    assert result.usage() == snapshot(RunUsage(requests=1))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='world')],
                model_name='gpt-4o-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='openai',
                provider_url='https://api.openai.com/v1',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='world')],
                model_name='gpt-4o-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='openai',
                provider_url='https://api.openai.com/v1',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )
    assert get_mock_chat_completion_kwargs(mock_client) == [
        {
            'messages': [{'content': 'hello', 'role': 'user'}],
            'model': 'gpt-4o',
            'extra_headers': {'User-Agent': IsStr(regex=r'pydantic-ai\/.*')},
            'extra_body': None,
        },
        {
            'messages': [
                {'content': 'hello', 'role': 'user'},
                {'content': 'world', 'role': 'assistant'},
                {'content': 'hello', 'role': 'user'},
            ],
            'model': 'gpt-4o',
            'extra_headers': {'User-Agent': IsStr(regex=r'pydantic-ai\/.*')},
            'extra_body': None,
        },
    ]


async def test_request_simple_usage(allow_model_requests: None):
    c = completion_message(
        ChatCompletionMessage(content='world', role='assistant'),
        usage=CompletionUsage(completion_tokens=1, prompt_tokens=2, total_tokens=3),
    )
    mock_client = MockOpenAI.create_mock(c)
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m)

    result = await agent.run('Hello')
    assert result.output == 'world'
    assert result.usage() == snapshot(
        RunUsage(
            requests=1,
            input_tokens=2,
            output_tokens=1,
        )
    )


async def test_response_with_created_timestamp_but_no_provider_details(allow_model_requests: None):
    class MinimalOpenAIChatModel(OpenAIChatModel):
        def _process_provider_details(self, response: chat.ChatCompletion) -> dict[str, Any] | None:
            return None

    c = completion_message(ChatCompletionMessage(content='world', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    m = MinimalOpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m)

    result = await agent.run('hello')
    assert result.output == 'world'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='world')],
                model_name='gpt-4o-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='openai',
                provider_url='https://api.openai.com/v1',
                provider_details={
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_chat_image_detail_vendor_metadata(allow_model_requests: None):
    c = completion_message(
        ChatCompletionMessage(content='done', role='assistant'),
    )
    mock_client = MockOpenAI.create_mock(c)
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(model)

    image_url = ImageUrl('https://example.com/image.png', vendor_metadata={'detail': 'high'})
    binary_image = BinaryContent(b'\x89PNG', media_type='image/png', vendor_metadata={'detail': 'high'})

    await agent.run(['Describe these inputs.', image_url, binary_image])

    request_kwargs = get_mock_chat_completion_kwargs(mock_client)
    image_parts = [
        item['image_url'] for item in request_kwargs[0]['messages'][0]['content'] if item['type'] == 'image_url'
    ]
    assert image_parts
    assert all(part['detail'] == 'high' for part in image_parts)


async def test_request_structured_response(allow_model_requests: None):
    c = completion_message(
        ChatCompletionMessage(
            content=None,
            role='assistant',
            tool_calls=[
                ChatCompletionMessageFunctionToolCall(
                    id='123',
                    function=Function(arguments='{"response": [1, 2, 123]}', name='final_result'),
                    type='function',
                )
            ],
        )
    )
    mock_client = MockOpenAI.create_mock(c)
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m, output_type=list[int])

    result = await agent.run('Hello')
    assert result.output == [1, 2, 123]
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"response": [1, 2, 123]}',
                        tool_call_id='123',
                    )
                ],
                model_name='gpt-4o-123',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
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
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )


async def test_request_tool_call(allow_model_requests: None):
    responses = [
        completion_message(
            ChatCompletionMessage(
                content=None,
                role='assistant',
                tool_calls=[
                    ChatCompletionMessageFunctionToolCall(
                        id='1',
                        function=Function(arguments='{"loc_name": "San Fransisco"}', name='get_location'),
                        type='function',
                    )
                ],
            ),
            usage=CompletionUsage(
                completion_tokens=1,
                prompt_tokens=2,
                total_tokens=3,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=1),
            ),
        ),
        completion_message(
            ChatCompletionMessage(
                content=None,
                role='assistant',
                tool_calls=[
                    ChatCompletionMessageFunctionToolCall(
                        id='2',
                        function=Function(arguments='{"loc_name": "London"}', name='get_location'),
                        type='function',
                    )
                ],
            ),
            usage=CompletionUsage(
                completion_tokens=2,
                prompt_tokens=3,
                total_tokens=6,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=2),
            ),
        ),
        completion_message(ChatCompletionMessage(content='final response', role='assistant')),
    ]
    mock_client = MockOpenAI.create_mock(responses)
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m, system_prompt='this is the system prompt')

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        if loc_name == 'London':
            return json.dumps({'lat': 51, 'lng': 0})
        else:
            raise ModelRetry('Wrong location, please try again')

    result = await agent.run('Hello')
    assert result.output == 'final response'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='this is the system prompt', timestamp=IsNow(tz=timezone.utc)),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_location',
                        args='{"loc_name": "San Fransisco"}',
                        tool_call_id='1',
                    )
                ],
                usage=RequestUsage(
                    input_tokens=2,
                    cache_read_tokens=1,
                    output_tokens=1,
                ),
                model_name='gpt-4o-123',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
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
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_location',
                        args='{"loc_name": "London"}',
                        tool_call_id='2',
                    )
                ],
                usage=RequestUsage(
                    input_tokens=3,
                    cache_read_tokens=2,
                    output_tokens=2,
                ),
                model_name='gpt-4o-123',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
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
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='final response')],
                model_name='gpt-4o-123',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )
    assert result.usage() == snapshot(
        RunUsage(requests=3, cache_read_tokens=3, input_tokens=5, output_tokens=3, tool_calls=1)
    )


FinishReason = Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call']


def chunk(delta: list[ChoiceDelta], finish_reason: FinishReason | None = None) -> chat.ChatCompletionChunk:
    return chat.ChatCompletionChunk(
        id='123',
        choices=[
            ChunkChoice(index=index, delta=delta, finish_reason=finish_reason) for index, delta in enumerate(delta)
        ],
        created=1704067200,  # 2024-01-01
        model='gpt-4o-123',
        object='chat.completion.chunk',
        usage=CompletionUsage(completion_tokens=1, prompt_tokens=2, total_tokens=3),
    )


def text_chunk(text: str, finish_reason: FinishReason | None = None) -> chat.ChatCompletionChunk:
    return chunk([ChoiceDelta(content=text, role='assistant')], finish_reason=finish_reason)


async def test_stream_text(allow_model_requests: None):
    stream = [text_chunk('hello '), text_chunk('world'), chunk([])]
    mock_client = MockOpenAI.create_mock_stream(stream)
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(['hello ', 'hello world'])
        assert result.is_complete
        assert result.usage() == snapshot(RunUsage(requests=1, input_tokens=6, output_tokens=3))


async def test_stream_text_finish_reason(allow_model_requests: None):
    first_chunk = text_chunk('hello ')
    # Test that we get the model name from a later chunk if it is not set on the first one, like on Azure OpenAI with content filter enabled.
    first_chunk.model = ''
    stream = [
        first_chunk,
        text_chunk('world'),
        text_chunk('.', finish_reason='stop'),
    ]
    mock_client = MockOpenAI.create_mock_stream(stream)
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(
            ['hello ', 'hello world', 'hello world.']
        )
        assert result.is_complete
        async for response, is_last in result.stream_responses(debounce_by=None):
            if is_last:
                assert response == snapshot(
                    ModelResponse(
                        parts=[TextPart(content='hello world.')],
                        usage=RequestUsage(input_tokens=6, output_tokens=3),
                        model_name='gpt-4o-123',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                        provider_url='https://api.openai.com/v1',
                        provider_details={
                            'finish_reason': 'stop',
                            'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                        },
                        provider_response_id='123',
                        finish_reason='stop',
                    )
                )


def struc_chunk(
    tool_name: str | None, tool_arguments: str | None, finish_reason: FinishReason | None = None
) -> chat.ChatCompletionChunk:
    return chunk(
        [
            ChoiceDelta(
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=0, function=ChoiceDeltaToolCallFunction(name=tool_name, arguments=tool_arguments)
                    )
                ]
            ),
        ],
        finish_reason=finish_reason,
    )


class MyTypedDict(TypedDict, total=False):
    first: str
    second: str


async def test_stream_structured(allow_model_requests: None):
    stream = [
        chunk([ChoiceDelta()]),
        chunk([ChoiceDelta(tool_calls=[])]),
        chunk([ChoiceDelta(tool_calls=[ChoiceDeltaToolCall(index=0, function=None)])]),
        chunk([ChoiceDelta(tool_calls=[ChoiceDeltaToolCall(index=0, function=None)])]),
        struc_chunk('final_result', None),
        chunk([ChoiceDelta(tool_calls=[ChoiceDeltaToolCall(index=0, function=None)])]),
        struc_chunk(None, '{"first": "One'),
        struc_chunk(None, '", "second": "Two"'),
        struc_chunk(None, '}'),
        chunk([]),
    ]
    mock_client = MockOpenAI.create_mock_stream(stream)
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m, output_type=MyTypedDict)

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
        assert result.usage() == snapshot(RunUsage(requests=1, input_tokens=20, output_tokens=10))
        # double check usage matches stream count
        assert result.usage().output_tokens == len(stream)


async def test_stream_structured_finish_reason(allow_model_requests: None):
    stream = [
        struc_chunk('final_result', None),
        struc_chunk(None, '{"first": "One'),
        struc_chunk(None, '", "second": "Two"'),
        struc_chunk(None, '}'),
        struc_chunk(None, None, finish_reason='stop'),
    ]
    mock_client = MockOpenAI.create_mock_stream(stream)
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m, output_type=MyTypedDict)

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


async def test_stream_native_output(allow_model_requests: None):
    stream = [
        chunk([]),
        text_chunk('{"first": "One'),
        text_chunk('", "second": "Two"'),
        text_chunk('}'),
        chunk([]),
    ]
    mock_client = MockOpenAI.create_mock_stream(stream)
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m, output_type=NativeOutput(MyTypedDict))

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


async def test_stream_tool_call_with_empty_text(allow_model_requests: None):
    stream = [
        chunk(
            [
                ChoiceDelta(
                    content='',  # Ollama will include an empty text delta even when it's going to call a tool
                    tool_calls=[
                        ChoiceDeltaToolCall(
                            index=0, function=ChoiceDeltaToolCallFunction(name='final_result', arguments=None)
                        )
                    ],
                ),
            ]
        ),
        struc_chunk(None, '{"first": "One'),
        struc_chunk(None, '", "second": "Two"'),
        struc_chunk(None, '}'),
        chunk([]),
    ]
    mock_client = MockOpenAI.create_mock_stream(stream)
    m = OpenAIChatModel('gpt-oss:20b', provider=OllamaProvider(openai_client=mock_client))
    agent = Agent(m, output_type=[str, MyTypedDict])

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_output(debounce_by=None)] == snapshot(
            [
                {'first': 'One'},
                {'first': 'One', 'second': 'Two'},
                {'first': 'One', 'second': 'Two'},
                {'first': 'One', 'second': 'Two'},
            ]
        )
    assert await result.get_output() == snapshot({'first': 'One', 'second': 'Two'})


async def test_stream_text_empty_think_tag_and_text_before_tool_call(allow_model_requests: None):
    # Ollama + Qwen3 will emit `<think>\n</think>\n\n` ahead of tool calls,
    # which we don't want to end up treating as a final result.
    stream = [
        text_chunk('<think>'),
        text_chunk('\n'),
        text_chunk('</think>'),
        text_chunk('\n\n'),
        struc_chunk('final_result', None),
        struc_chunk(None, '{"first": "One'),
        struc_chunk(None, '", "second": "Two"'),
        struc_chunk(None, '}'),
        chunk([]),
    ]
    mock_client = MockOpenAI.create_mock_stream(stream)
    m = OpenAIChatModel('qwen3', provider=OllamaProvider(openai_client=mock_client))
    agent = Agent(m, output_type=[str, MyTypedDict])

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_output(debounce_by=None)] == snapshot(
            [
                {},
                {'first': 'One'},
                {'first': 'One', 'second': 'Two'},
                {'first': 'One', 'second': 'Two'},
                {'first': 'One', 'second': 'Two'},
            ]
        )
    assert await result.get_output() == snapshot({'first': 'One', 'second': 'Two'})


async def test_no_delta(allow_model_requests: None):
    stream = [
        chunk([]),
        text_chunk('hello '),
        text_chunk('world'),
    ]
    mock_client = MockOpenAI.create_mock_stream(stream)
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(['hello ', 'hello world'])
        assert result.is_complete
        assert result.usage() == snapshot(RunUsage(requests=1, input_tokens=6, output_tokens=3))


def none_delta_chunk(finish_reason: FinishReason | None = None) -> chat.ChatCompletionChunk:
    choice = ChunkChoice(index=0, delta=ChoiceDelta())
    # When using Azure OpenAI and an async content filter is enabled, the openai SDK can return None deltas.
    choice.delta = None  # pyright: ignore[reportAttributeAccessIssue]
    return chat.ChatCompletionChunk(
        id='123',
        choices=[choice],
        created=1704067200,  # 2024-01-01
        model='gpt-4o-123',
        object='chat.completion.chunk',
        usage=CompletionUsage(completion_tokens=1, prompt_tokens=2, total_tokens=3),
    )


async def test_none_delta(allow_model_requests: None):
    stream = [
        none_delta_chunk(),
        text_chunk('hello '),
        text_chunk('world'),
    ]
    mock_client = MockOpenAI.create_mock_stream(stream)
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(['hello ', 'hello world'])
        assert result.is_complete
        assert result.usage() == snapshot(RunUsage(requests=1, input_tokens=6, output_tokens=3))


@pytest.mark.filterwarnings('ignore:Set the `system_prompt_role` in the `OpenAIModelProfile` instead.')
@pytest.mark.parametrize('system_prompt_role', ['system', 'developer', 'user', None])
async def test_system_prompt_role(
    allow_model_requests: None, system_prompt_role: OpenAISystemPromptRole | None
) -> None:
    """Testing the system prompt role for OpenAI models is properly set / inferred."""

    c = completion_message(ChatCompletionMessage(content='world', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    m = OpenAIChatModel(  # type: ignore[reportDeprecated]
        'gpt-4o', system_prompt_role=system_prompt_role, provider=OpenAIProvider(openai_client=mock_client)
    )
    assert m.system_prompt_role == system_prompt_role  # type: ignore[reportDeprecated]

    agent = Agent(m, system_prompt='some instructions')
    result = await agent.run('hello')
    assert result.output == 'world'

    assert get_mock_chat_completion_kwargs(mock_client) == [
        {
            'messages': [
                {'content': 'some instructions', 'role': system_prompt_role or 'system'},
                {'content': 'hello', 'role': 'user'},
            ],
            'model': 'gpt-4o',
            'extra_headers': {'User-Agent': IsStr(regex=r'pydantic-ai\/.*')},
            'extra_body': None,
        }
    ]


async def test_system_prompt_role_o1_mini(allow_model_requests: None, openai_api_key: str):
    model = OpenAIChatModel('o1-mini', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, system_prompt='You are a helpful assistant.')

    result = await agent.run("What's the capital of France?")
    assert result.output == snapshot('The capital of France is **Paris**.')


async def test_openai_pass_custom_system_prompt_role(allow_model_requests: None, openai_api_key: str):
    profile = ModelProfile(supports_tools=False)
    model = OpenAIChatModel(  # type: ignore[reportDeprecated]
        'o1-mini', profile=profile, provider=OpenAIProvider(api_key=openai_api_key), system_prompt_role='user'
    )
    profile = OpenAIModelProfile.from_profile(model.profile)
    assert profile.openai_system_prompt_role == 'user'
    assert profile.supports_tools is False


@pytest.mark.parametrize('system_prompt_role', ['system', 'developer'])
async def test_openai_o1_mini_system_role(
    allow_model_requests: None,
    system_prompt_role: Literal['system', 'developer'],
    openai_api_key: str,
) -> None:
    model = OpenAIChatModel(  # type: ignore[reportDeprecated]
        'o1-mini', provider=OpenAIProvider(api_key=openai_api_key), system_prompt_role=system_prompt_role
    )
    agent = Agent(model=model, system_prompt='You are a helpful assistant.')

    with pytest.raises(ModelHTTPError, match=r".*Unsupported value: 'messages\[0\]\.role' does not support.*"):
        await agent.run('Hello')


@pytest.mark.parametrize('parallel_tool_calls', [True, False])
async def test_parallel_tool_calls(allow_model_requests: None, parallel_tool_calls: bool) -> None:
    c = completion_message(
        ChatCompletionMessage(
            content=None,
            role='assistant',
            tool_calls=[
                ChatCompletionMessageFunctionToolCall(
                    id='123',
                    function=Function(arguments='{"response": [1, 2, 3]}', name='final_result'),
                    type='function',
                )
            ],
        )
    )
    mock_client = MockOpenAI.create_mock(c)
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m, output_type=list[int], model_settings=ModelSettings(parallel_tool_calls=parallel_tool_calls))

    await agent.run('Hello')
    assert get_mock_chat_completion_kwargs(mock_client)[0]['parallel_tool_calls'] == parallel_tool_calls


async def test_image_url_input(allow_model_requests: None):
    c = completion_message(ChatCompletionMessage(content='world', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m)

    result = await agent.run(
        [
            'hello',
            ImageUrl(url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg'),
        ]
    )
    assert result.output == 'world'
    assert get_mock_chat_completion_kwargs(mock_client) == snapshot(
        [
            {
                'model': 'gpt-4o',
                'messages': [
                    {
                        'role': 'user',
                        'content': [
                            {'text': 'hello', 'type': 'text'},
                            {
                                'image_url': {
                                    'url': 'https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg'
                                },
                                'type': 'image_url',
                            },
                        ],
                    }
                ],
                'extra_headers': {'User-Agent': IsStr(regex=r'pydantic-ai\/.*')},
                'extra_body': None,
            }
        ]
    )


async def test_image_url_input_force_download(allow_model_requests: None, openai_api_key: str):
    provider = OpenAIProvider(api_key=openai_api_key)
    m = OpenAIChatModel('gpt-4.1-nano', provider=provider)
    agent = Agent(m)

    result = await agent.run(
        [
            'What is this vegetable?',
            ImageUrl(
                force_download=True,
                url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg',
            ),
        ]
    )
    assert result.output == snapshot('This vegetable is a potato.')


async def test_image_url_input_force_download_response_api(allow_model_requests: None, openai_api_key: str):
    provider = OpenAIProvider(api_key=openai_api_key)
    m = OpenAIResponsesModel('gpt-4.1-nano', provider=provider)
    agent = Agent(m)

    result = await agent.run(
        [
            'What is this vegetable?',
            ImageUrl(
                force_download=True,
                url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg',
            ),
        ]
    )
    assert result.output == snapshot('This is a potato.')


async def test_openai_audio_url_input(allow_model_requests: None, openai_api_key: str):
    m = OpenAIChatModel('gpt-4o-audio-preview', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)

    result = await agent.run(['Hello', AudioUrl(url='https://cdn.openai.com/API/docs/audio/alloy.wav')])
    assert result.output == snapshot(
        'Yes, the phenomenon of the sun rising in the east and setting in the west is due to the rotation of the Earth. The Earth rotates on its axis from west to east, making the sun appear to rise on the eastern horizon and set in the west. This is a daily occurrence and has been a fundamental aspect of human observation and timekeeping throughout history.'
    )
    assert result.usage() == snapshot(
        RunUsage(
            input_tokens=81,
            output_tokens=72,
            input_audio_tokens=69,
            details={
                'accepted_prediction_tokens': 0,
                'audio_tokens': 0,
                'reasoning_tokens': 0,
                'rejected_prediction_tokens': 0,
                'text_tokens': 72,
            },
            requests=1,
        )
    )


async def test_document_url_input(allow_model_requests: None, openai_api_key: str):
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)

    document_url = DocumentUrl(url='https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf')

    result = await agent.run(['What is the main content on this document?', document_url])
    assert result.output == snapshot('The document contains the text "Dummy PDF file" on its single page.')


async def test_document_url_input_response_api(allow_model_requests: None, openai_api_key: str):
    """Test DocumentUrl with Responses API sends URL directly (default behavior)."""
    provider = OpenAIProvider(api_key=openai_api_key)
    m = OpenAIResponsesModel('gpt-4.1-nano', provider=provider)
    agent = Agent(m)

    document_url = DocumentUrl(url='https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf')

    result = await agent.run(['What is the main content on this document?', document_url])
    assert 'Dummy PDF' in result.output


async def test_document_url_input_force_download_response_api(allow_model_requests: None, openai_api_key: str):
    """Test DocumentUrl with force_download=True downloads and sends as file_data."""
    provider = OpenAIProvider(api_key=openai_api_key)
    m = OpenAIResponsesModel('gpt-4.1-nano', provider=provider)
    agent = Agent(m)

    document_url = DocumentUrl(
        url='https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf',
        force_download=True,
    )

    result = await agent.run(['What is the main content on this document?', document_url])
    assert 'Dummy PDF' in result.output


async def test_image_url_force_download_chat() -> None:
    """Test that force_download=True calls download_item for ImageUrl in OpenAIChatModel."""
    from unittest.mock import AsyncMock, patch

    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key='test-key'))

    with patch('pydantic_ai.models.openai.download_item', new_callable=AsyncMock) as mock_download:
        mock_download.return_value = {
            'data': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
            'content_type': 'image/png',
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


async def test_image_url_no_force_download_chat() -> None:
    """Test that force_download=False does not call download_item for ImageUrl in OpenAIChatModel."""
    from unittest.mock import AsyncMock, patch

    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key='test-key'))

    with patch('pydantic_ai.models.openai.download_item', new_callable=AsyncMock) as mock_download:
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


async def test_document_url_force_download_responses() -> None:
    """Test that force_download=True calls download_item for DocumentUrl in OpenAIResponsesModel."""
    from unittest.mock import AsyncMock, patch

    m = OpenAIResponsesModel('gpt-4.5-nano', provider=OpenAIProvider(api_key='test-key'))

    with patch('pydantic_ai.models.openai.download_item', new_callable=AsyncMock) as mock_download:
        mock_download.return_value = {
            'data': 'data:application/pdf;base64,JVBERi0xLjQK',
            'data_type': 'pdf',
        }

        messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'Test PDF',
                            DocumentUrl(
                                url='https://example.com/doc.pdf',
                                media_type='application/pdf',
                                force_download=True,
                            ),
                        ]
                    )
                ]
            )
        ]

        await m._map_messages(messages, {}, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage,reportArgumentType]

        mock_download.assert_called_once()
        assert mock_download.call_args[0][0].url == 'https://example.com/doc.pdf'


async def test_document_url_no_force_download_responses() -> None:
    """Test that force_download=False does not call download_item for DocumentUrl in OpenAIResponsesModel."""
    from unittest.mock import AsyncMock, patch

    m = OpenAIResponsesModel('gpt-4.5-nano', provider=OpenAIProvider(api_key='test-key'))

    with patch('pydantic_ai.models.openai.download_item', new_callable=AsyncMock) as mock_download:
        messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'Test document',
                            DocumentUrl(
                                url='https://example.com/doc.pdf',
                                media_type='application/pdf',
                                force_download=False,
                            ),
                        ]
                    )
                ]
            )
        ]

        await m._map_messages(messages, {}, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage,reportArgumentType]

        mock_download.assert_not_called()


async def test_audio_url_force_download_responses() -> None:
    """Test that force_download=True calls download_item for AudioUrl in OpenAIResponsesModel."""
    from unittest.mock import AsyncMock, patch

    m = OpenAIResponsesModel('gpt-4.5-nano', provider=OpenAIProvider(api_key='test-key'))

    with patch('pydantic_ai.models.openai.download_item', new_callable=AsyncMock) as mock_download:
        mock_download.return_value = {
            'data': 'data:audio/mp3;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2',
            'data_type': 'mp3',
        }

        messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'Test audio',
                            AudioUrl(
                                url='https://example.com/audio.mp3',
                                media_type='audio/mp3',
                                force_download=True,
                            ),
                        ]
                    )
                ]
            )
        ]

        await m._map_messages(messages, {}, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage,reportArgumentType]

        mock_download.assert_called_once()
        assert mock_download.call_args[0][0].url == 'https://example.com/audio.mp3'


@pytest.mark.vcr()
async def test_image_url_tool_response(allow_model_requests: None, openai_api_key: str):
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)

    @agent.tool_plain
    async def get_image() -> ImageUrl:
        return ImageUrl(url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg')

    result = await agent.run(['What food is in the image you can get from the get_image tool?'])
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=['What food is in the image you can get from the get_image tool?'],
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_image', args='{}', tool_call_id='call_4hrT4QP9jfojtK69vGiFCFjG')],
                usage=RequestUsage(
                    input_tokens=46,
                    output_tokens=11,
                    details={
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0,
                    },
                ),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'tool_calls',
                    'timestamp': datetime(2025, 4, 29, 21, 7, 59, tzinfo=timezone.utc),
                },
                provider_response_id='chatcmpl-BRmTHlrARTzAHK1na9s80xDlQGYPX',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_image',
                        content='See file bd38f5',
                        tool_call_id='call_4hrT4QP9jfojtK69vGiFCFjG',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content=[
                            'This is file bd38f5:',
                            ImageUrl(
                                url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg',
                                identifier='bd38f5',
                            ),
                        ],
                        timestamp=IsDatetime(),
                    ),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='The image shows a potato.')],
                usage=RequestUsage(
                    input_tokens=503,
                    output_tokens=8,
                    details={
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0,
                    },
                ),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2025, 4, 29, 21, 8, tzinfo=timezone.utc),
                },
                provider_response_id='chatcmpl-BRmTI0Y2zmkGw27kLarhsmiFQTGxR',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_image_as_binary_content_tool_response(
    allow_model_requests: None, image_content: BinaryContent, openai_api_key: str
):
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)

    @agent.tool_plain
    async def get_image() -> BinaryContent:
        return image_content

    result = await agent.run(['What fruit is in the image you can get from the get_image tool?'])
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=['What fruit is in the image you can get from the get_image tool?'],
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_image', args='{}', tool_call_id='call_1FnV4RIOyM7T9BxPHbSuUexJ')],
                usage=RequestUsage(
                    input_tokens=46,
                    output_tokens=10,
                    details={
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0,
                    },
                ),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={'finish_reason': 'tool_calls', 'timestamp': IsDatetime()},
                provider_response_id='chatcmpl-Cpwffm3QIHBYzhoYYZSF7Et1tiiqI',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_image',
                        content='See file 241a70',
                        tool_call_id='call_1FnV4RIOyM7T9BxPHbSuUexJ',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(content=['This is file 241a70:', image_content], timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='The fruit in the image is a kiwi.')],
                usage=RequestUsage(
                    input_tokens=847,
                    output_tokens=10,
                    details={
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0,
                    },
                ),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={'finish_reason': 'stop', 'timestamp': IsDatetime()},
                provider_response_id='chatcmpl-CpwfhFC1iUmDKeoxOTSN7KP8D11aq',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_image_as_binary_content_input(
    allow_model_requests: None, image_content: BinaryContent, openai_api_key: str
):
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)

    result = await agent.run(['What fruit is in the image?', image_content])
    assert result.output == snapshot('The fruit in the image is a kiwi.')


async def test_audio_as_binary_content_input(
    allow_model_requests: None, audio_content: BinaryContent, openai_api_key: str
):
    m = OpenAIChatModel('gpt-4o-audio-preview', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)

    result = await agent.run(['Whose name is mentioned in the audio?', audio_content])
    assert result.output == snapshot('The name mentioned in the audio is Marcelo.')
    assert result.usage() == snapshot(
        RunUsage(
            input_tokens=64,
            output_tokens=9,
            input_audio_tokens=44,
            details={
                'accepted_prediction_tokens': 0,
                'audio_tokens': 0,
                'reasoning_tokens': 0,
                'rejected_prediction_tokens': 0,
                'text_tokens': 9,
            },
            requests=1,
        )
    )


async def test_document_as_binary_content_input(
    allow_model_requests: None, document_content: BinaryContent, openai_api_key: str
):
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)

    result = await agent.run(['What is the main content on this document?', document_content])
    assert result.output == snapshot('The main content of the document is "Dummy PDF file."')


async def test_text_document_url_input(allow_model_requests: None, openai_api_key: str):
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)

    document_url = DocumentUrl(url='https://www.w3.org/TR/2003/REC-PNG-20031110/iso_8859-1.txt')

    result = await agent.run(['What is the main content on this document, in one sentence?', document_url])
    assert result.output == snapshot(
        'The document lists the graphical characters defined by ISO 8859-1 (1987) with their hexadecimal codes and descriptions.'
    )


async def test_text_document_as_binary_content_input(
    allow_model_requests: None, text_document_content: BinaryContent, openai_api_key: str
):
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)

    result = await agent.run(['What is the main content on this document?', text_document_content])
    assert result.output == snapshot(
        'The main content of the document is simply the text "Dummy TXT file." It does not appear to contain any other detailed information.'
    )


async def test_document_as_binary_content_input_with_tool(
    allow_model_requests: None, document_content: BinaryContent, openai_api_key: str
):
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)

    @agent.tool_plain
    async def get_upper_case(text: str) -> str:
        return text.upper()

    result = await agent.run(
        [
            'What is the main content on this document? Use the get_upper_case tool to get the upper case of the text.',
            document_content,
        ]
    )

    assert result.output == snapshot('The main content of the document is "DUMMY PDF FILE" in uppercase.')


def test_model_status_error(allow_model_requests: None) -> None:
    mock_client = MockOpenAI.create_mock(
        APIStatusError(
            'test error',
            response=httpx.Response(status_code=500, request=httpx.Request('POST', 'https://example.com/v1')),
            body={'error': 'test error'},
        )
    )
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m)
    with pytest.raises(ModelHTTPError) as exc_info:
        agent.run_sync('hello')
    assert str(exc_info.value) == snapshot("status_code: 500, model_name: gpt-4o, body: {'error': 'test error'}")


def test_model_connection_error(allow_model_requests: None) -> None:
    mock_client = MockOpenAI.create_mock(
        APIConnectionError(
            message='Connection to http://localhost:11434/v1 timed out',
            request=httpx.Request('POST', 'http://localhost:11434/v1'),
        )
    )
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m)
    with pytest.raises(ModelAPIError) as exc_info:
        agent.run_sync('hello')
    assert exc_info.value.model_name == 'gpt-4o'
    assert 'Connection to http://localhost:11434/v1 timed out' in str(exc_info.value.message)


def test_responses_model_connection_error(allow_model_requests: None) -> None:
    mock_client = MockOpenAIResponses.create_mock(
        APIConnectionError(
            message='Connection to http://localhost:11434/v1 timed out',
            request=httpx.Request('POST', 'http://localhost:11434/v1'),
        )
    )
    m = OpenAIResponsesModel('o3-mini', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m)
    with pytest.raises(ModelAPIError) as exc_info:
        agent.run_sync('hello')
    assert exc_info.value.model_name == 'o3-mini'
    assert 'Connection to http://localhost:11434/v1 timed out' in str(exc_info.value.message)


@pytest.mark.parametrize('model_name', ['o3-mini', 'gpt-4o-mini', 'gpt-4.5-preview'])
async def test_max_completion_tokens(allow_model_requests: None, model_name: str, openai_api_key: str):
    m = OpenAIChatModel(model_name, provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m, model_settings=ModelSettings(max_tokens=100))

    result = await agent.run('hello')
    assert result.output == IsStr()


async def test_multiple_agent_tool_calls(allow_model_requests: None, gemini_api_key: str, openai_api_key: str):
    gemini_model = GoogleModel('gemini-2.0-flash-exp', provider=GoogleProvider(api_key=gemini_api_key))
    openai_model = OpenAIChatModel('gpt-4o-mini', provider=OpenAIProvider(api_key=openai_api_key))

    agent = Agent(model=gemini_model)

    @agent.tool_plain
    async def get_capital(country: str) -> str:
        """Get the capital of a country.

        Args:
            country: The country name.
        """
        if country == 'France':
            return 'Paris'
        elif country == 'England':
            return 'London'
        else:
            raise ValueError(f'Country {country} not supported.')  # pragma: no cover

    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is Paris.\n')

    result = await agent.run(
        'What is the capital of England?', model=openai_model, message_history=result.all_messages()
    )
    assert result.output == snapshot('The capital of England is London.')


async def test_message_history_can_start_with_model_response(allow_model_requests: None, openai_api_key: str):
    """Test that an agent run with message_history starting with ModelResponse is executed correctly."""

    openai_model = OpenAIChatModel('gpt-4.1-mini', provider=OpenAIProvider(api_key=openai_api_key))

    message_history = [ModelResponse(parts=[TextPart('Where do you want to go today?')])]

    agent = Agent(model=openai_model)

    result = await agent.run('Answer in 5 words only. Who is Tux?', message_history=message_history)

    assert result.output == snapshot('Linux mascot, a penguin character.')
    assert result.all_messages() == snapshot(
        [
            ModelResponse(
                parts=[TextPart(content='Where do you want to go today?')],
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Answer in 5 words only. Who is Tux?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Linux mascot, a penguin character.')],
                usage=RequestUsage(
                    input_tokens=31,
                    output_tokens=8,
                    details={
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0,
                    },
                ),
                model_name='gpt-4.1-mini-2025-04-14',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2025, 11, 22, 10, 1, 40, tzinfo=timezone.utc),
                },
                provider_response_id='chatcmpl-Ceeiy4ivEE0hcL1EX5ZfLuW5xNUXB',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_extra_headers(allow_model_requests: None, openai_api_key: str):
    # This test doesn't do anything, it's just here to ensure that calls with `extra_headers` don't cause errors, including type.
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m, model_settings=OpenAIChatModelSettings(extra_headers={'Extra-Header-Key': 'Extra-Header-Value'}))
    await agent.run('hello')


async def test_user_id(allow_model_requests: None, openai_api_key: str):
    # This test doesn't do anything, it's just here to ensure that calls with `user` don't cause errors, including type.
    # Since we use VCR, creating tests with an `httpx.Transport` is not possible.
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m, model_settings=OpenAIChatModelSettings(openai_user='user_id'))
    await agent.run('hello')


@dataclass
class MyDefaultDc:
    x: int = 1


class MyEnum(Enum):
    a = 'a'
    b = 'b'


@dataclass
class MyRecursiveDc:
    field: MyRecursiveDc | None
    my_enum: MyEnum = Field(description='my enum')


@dataclass
class MyDefaultRecursiveDc:
    field: MyDefaultRecursiveDc | None = None


class MyModel(BaseModel):
    foo: str


class MyDc(BaseModel):
    foo: str


class MyOptionalDc(BaseModel):
    foo: str | None
    bar: str


class MyExtrasDc(BaseModel, extra='allow'):
    foo: str


class MyNormalTypedDict(TypedDict):
    foo: str


class MyOptionalTypedDict(TypedDict):
    foo: NotRequired[str]
    bar: str


class MyPartialTypedDict(TypedDict, total=False):
    foo: str


class MyExtrasModel(BaseModel, extra='allow'):
    pass


def strict_compatible_tool(x: int) -> str:
    return str(x)  # pragma: no cover


def tool_with_default(x: int = 1) -> str:
    return f'{x}'  # pragma: no cover


def tool_with_datetime(x: datetime) -> str:
    return f'{x}'  # pragma: no cover


def tool_with_url(x: AnyUrl) -> str:
    return f'{x}'  # pragma: no cover


def tool_with_recursion(x: MyRecursiveDc, y: MyDefaultRecursiveDc):
    return f'{x} {y}'  # pragma: no cover


def tool_with_model(x: MyModel) -> str:
    return f'{x}'  # pragma: no cover


def tool_with_dataclass(x: MyDc) -> str:
    return f'{x}'  # pragma: no cover


def tool_with_optional_dataclass(x: MyOptionalDc) -> str:
    return f'{x}'  # pragma: no cover


def tool_with_dataclass_with_extras(x: MyExtrasDc) -> str:
    return f'{x}'  # pragma: no cover


def tool_with_typed_dict(x: MyNormalTypedDict) -> str:
    return f'{x}'  # pragma: no cover


def tool_with_optional_typed_dict(x: MyOptionalTypedDict) -> str:
    return f'{x}'  # pragma: no cover


def tool_with_partial_typed_dict(x: MyPartialTypedDict) -> str:
    return f'{x}'  # pragma: no cover


def tool_with_model_with_extras(x: MyExtrasModel) -> str:
    return f'{x}'  # pragma: no cover


def tool_with_kwargs(x: int, **kwargs: Any) -> str:
    return f'{x} {kwargs}'  # pragma: no cover


def tool_with_typed_kwargs(x: int, **kwargs: int) -> str:
    return f'{x} {kwargs}'  # pragma: no cover


def tool_with_union(x: int | MyDefaultDc) -> str:
    return f'{x}'  # pragma: no cover


def tool_with_discriminated_union(
    x: Annotated[
        Annotated[int, Tag('int')] | Annotated[MyDefaultDc, Tag('MyDefaultDc')],
        Discriminator(lambda x: type(x).__name__),
    ],
) -> str:
    return f'{x}'  # pragma: no cover


def tool_with_lists(x: list[int], y: list[MyDefaultDc]) -> str:
    return f'{x} {y}'  # pragma: no cover


def tool_with_tuples(x: tuple[int], y: tuple[str] = ('abc',)) -> str:
    return f'{x} {y}'  # pragma: no cover


@pytest.mark.parametrize(
    'tool,tool_strict,expected_params,expected_strict',
    [
        (
            strict_compatible_tool,
            False,
            snapshot(
                {
                    'additionalProperties': False,
                    'properties': {'x': {'type': 'integer'}},
                    'required': ['x'],
                    'type': 'object',
                }
            ),
            snapshot(None),
        ),
        (
            tool_with_default,
            None,
            snapshot(
                {
                    'additionalProperties': False,
                    'properties': {'x': {'default': 1, 'type': 'integer'}},
                    'type': 'object',
                }
            ),
            snapshot(None),
        ),
        (
            tool_with_datetime,
            None,
            snapshot(
                {
                    'additionalProperties': False,
                    'properties': {'x': {'format': 'date-time', 'type': 'string'}},
                    'required': ['x'],
                    'type': 'object',
                }
            ),
            snapshot(True),
        ),
        (
            tool_with_url,
            None,
            snapshot(
                {
                    'additionalProperties': False,
                    'properties': {'x': {'format': 'uri', 'minLength': 1, 'type': 'string'}},
                    'required': ['x'],
                    'type': 'object',
                }
            ),
            snapshot(None),
        ),
        (
            tool_with_url,
            True,
            snapshot(
                {
                    'additionalProperties': False,
                    'properties': {'x': {'type': 'string', 'description': 'minLength=1, format=uri'}},
                    'required': ['x'],
                    'type': 'object',
                }
            ),
            snapshot(True),
        ),
        (
            tool_with_recursion,
            None,
            snapshot(
                {
                    '$defs': {
                        'MyDefaultRecursiveDc': {
                            'properties': {
                                'field': {
                                    'anyOf': [{'$ref': '#/$defs/MyDefaultRecursiveDc'}, {'type': 'null'}],
                                    'default': None,
                                }
                            },
                            'type': 'object',
                            'additionalProperties': False,
                        },
                        'MyEnum': {'enum': ['a', 'b'], 'type': 'string'},
                        'MyRecursiveDc': {
                            'properties': {
                                'field': {'anyOf': [{'$ref': '#/$defs/MyRecursiveDc'}, {'type': 'null'}]},
                                'my_enum': {'description': 'my enum', 'anyOf': [{'$ref': '#/$defs/MyEnum'}]},
                            },
                            'required': ['field', 'my_enum'],
                            'type': 'object',
                            'additionalProperties': False,
                        },
                    },
                    'additionalProperties': False,
                    'properties': {
                        'x': {'$ref': '#/$defs/MyRecursiveDc'},
                        'y': {'$ref': '#/$defs/MyDefaultRecursiveDc'},
                    },
                    'required': ['x', 'y'],
                    'type': 'object',
                }
            ),
            snapshot(None),
        ),
        (
            tool_with_recursion,
            True,
            snapshot(
                {
                    '$defs': {
                        'MyDefaultRecursiveDc': {
                            'properties': {
                                'field': {'anyOf': [{'$ref': '#/$defs/MyDefaultRecursiveDc'}, {'type': 'null'}]}
                            },
                            'type': 'object',
                            'additionalProperties': False,
                            'required': ['field'],
                        },
                        'MyEnum': {'enum': ['a', 'b'], 'type': 'string'},
                        'MyRecursiveDc': {
                            'properties': {
                                'field': {'anyOf': [{'$ref': '#/$defs/MyRecursiveDc'}, {'type': 'null'}]},
                                'my_enum': {'description': 'my enum', 'anyOf': [{'$ref': '#/$defs/MyEnum'}]},
                            },
                            'type': 'object',
                            'additionalProperties': False,
                            'required': ['field', 'my_enum'],
                        },
                    },
                    'additionalProperties': False,
                    'properties': {
                        'x': {'$ref': '#/$defs/MyRecursiveDc'},
                        'y': {'$ref': '#/$defs/MyDefaultRecursiveDc'},
                    },
                    'required': ['x', 'y'],
                    'type': 'object',
                }
            ),
            snapshot(True),
        ),
        (
            tool_with_model,
            None,
            snapshot(
                {
                    'additionalProperties': False,
                    'properties': {'foo': {'type': 'string'}},
                    'required': ['foo'],
                    'type': 'object',
                }
            ),
            snapshot(True),
        ),
        (
            tool_with_dataclass,
            None,
            snapshot(
                {
                    'additionalProperties': False,
                    'properties': {'foo': {'type': 'string'}},
                    'required': ['foo'],
                    'type': 'object',
                }
            ),
            snapshot(True),
        ),
        (
            tool_with_optional_dataclass,
            None,
            snapshot(
                {
                    'additionalProperties': False,
                    'properties': {'foo': {'anyOf': [{'type': 'string'}, {'type': 'null'}]}, 'bar': {'type': 'string'}},
                    'required': ['foo', 'bar'],
                    'type': 'object',
                }
            ),
            snapshot(True),
        ),
        (
            tool_with_dataclass_with_extras,
            None,
            snapshot(
                {
                    'additionalProperties': True,
                    'properties': {'foo': {'type': 'string'}},
                    'required': ['foo'],
                    'type': 'object',
                }
            ),
            snapshot(None),
        ),
        (
            tool_with_typed_dict,
            None,
            snapshot(
                {
                    'additionalProperties': False,
                    'properties': {'foo': {'type': 'string'}},
                    'required': ['foo'],
                    'type': 'object',
                }
            ),
            snapshot(True),
        ),
        (
            tool_with_optional_typed_dict,
            None,
            snapshot(
                {
                    'additionalProperties': False,
                    'properties': {'foo': {'type': 'string'}, 'bar': {'type': 'string'}},
                    'required': ['bar'],
                    'type': 'object',
                }
            ),
            snapshot(None),
        ),
        (
            tool_with_partial_typed_dict,
            None,
            snapshot(
                {
                    'additionalProperties': False,
                    'properties': {'foo': {'type': 'string'}},
                    'type': 'object',
                }
            ),
            snapshot(None),
        ),
        (
            tool_with_model_with_extras,
            None,
            snapshot(
                {
                    'additionalProperties': True,
                    'properties': {},
                    'type': 'object',
                }
            ),
            snapshot(None),
        ),
        (
            tool_with_model_with_extras,
            True,
            snapshot(
                {
                    'additionalProperties': False,
                    'properties': {},
                    'required': [],
                    'type': 'object',
                }
            ),
            snapshot(True),
        ),
        (
            tool_with_kwargs,
            None,
            snapshot(
                {
                    'additionalProperties': True,
                    'properties': {'x': {'type': 'integer'}},
                    'required': ['x'],
                    'type': 'object',
                }
            ),
            snapshot(None),
        ),
        (
            tool_with_kwargs,
            True,
            snapshot(
                {
                    'additionalProperties': False,
                    'properties': {'x': {'type': 'integer'}},
                    'required': ['x'],
                    'type': 'object',
                }
            ),
            snapshot(True),
        ),
        (
            tool_with_typed_kwargs,
            None,
            snapshot(
                {
                    'additionalProperties': {'type': 'integer'},
                    'properties': {'x': {'type': 'integer'}},
                    'required': ['x'],
                    'type': 'object',
                }
            ),
            snapshot(None),
        ),
        (
            tool_with_union,
            None,
            snapshot(
                {
                    '$defs': {
                        'MyDefaultDc': {
                            'properties': {'x': {'default': 1, 'type': 'integer'}},
                            'type': 'object',
                            'additionalProperties': False,
                        }
                    },
                    'additionalProperties': False,
                    'properties': {'x': {'anyOf': [{'type': 'integer'}, {'$ref': '#/$defs/MyDefaultDc'}]}},
                    'required': ['x'],
                    'type': 'object',
                }
            ),
            snapshot(None),
        ),
        (
            tool_with_union,
            True,
            snapshot(
                {
                    '$defs': {
                        'MyDefaultDc': {
                            'properties': {'x': {'type': 'integer'}},
                            'required': ['x'],
                            'type': 'object',
                            'additionalProperties': False,
                        }
                    },
                    'additionalProperties': False,
                    'properties': {'x': {'anyOf': [{'type': 'integer'}, {'$ref': '#/$defs/MyDefaultDc'}]}},
                    'required': ['x'],
                    'type': 'object',
                }
            ),
            snapshot(True),
        ),
        (
            tool_with_discriminated_union,
            None,
            snapshot(
                {
                    '$defs': {
                        'MyDefaultDc': {
                            'properties': {'x': {'default': 1, 'type': 'integer'}},
                            'type': 'object',
                            'additionalProperties': False,
                        }
                    },
                    'additionalProperties': False,
                    'properties': {'x': {'oneOf': [{'type': 'integer'}, {'$ref': '#/$defs/MyDefaultDc'}]}},
                    'required': ['x'],
                    'type': 'object',
                }
            ),
            snapshot(None),
        ),
        (
            tool_with_discriminated_union,
            True,
            snapshot(
                {
                    '$defs': {
                        'MyDefaultDc': {
                            'properties': {'x': {'type': 'integer'}},
                            'required': ['x'],
                            'type': 'object',
                            'additionalProperties': False,
                        }
                    },
                    'additionalProperties': False,
                    'properties': {'x': {'anyOf': [{'type': 'integer'}, {'$ref': '#/$defs/MyDefaultDc'}]}},
                    'required': ['x'],
                    'type': 'object',
                }
            ),
            snapshot(True),
        ),
        (
            tool_with_lists,
            None,
            snapshot(
                {
                    '$defs': {
                        'MyDefaultDc': {
                            'properties': {'x': {'default': 1, 'type': 'integer'}},
                            'type': 'object',
                            'additionalProperties': False,
                        }
                    },
                    'additionalProperties': False,
                    'properties': {
                        'x': {'items': {'type': 'integer'}, 'type': 'array'},
                        'y': {'items': {'$ref': '#/$defs/MyDefaultDc'}, 'type': 'array'},
                    },
                    'required': ['x', 'y'],
                    'type': 'object',
                }
            ),
            snapshot(None),
        ),
        (
            tool_with_lists,
            True,
            snapshot(
                {
                    '$defs': {
                        'MyDefaultDc': {
                            'properties': {'x': {'type': 'integer'}},
                            'required': ['x'],
                            'type': 'object',
                            'additionalProperties': False,
                        }
                    },
                    'additionalProperties': False,
                    'properties': {
                        'x': {'items': {'type': 'integer'}, 'type': 'array'},
                        'y': {'items': {'$ref': '#/$defs/MyDefaultDc'}, 'type': 'array'},
                    },
                    'required': ['x', 'y'],
                    'type': 'object',
                }
            ),
            snapshot(True),
        ),
        (
            tool_with_tuples,
            None,
            snapshot(
                {
                    'additionalProperties': False,
                    'properties': {
                        'x': {'maxItems': 1, 'minItems': 1, 'prefixItems': [{'type': 'integer'}], 'type': 'array'},
                        'y': {
                            'default': ['abc'],
                            'maxItems': 1,
                            'minItems': 1,
                            'prefixItems': [{'type': 'string'}],
                            'type': 'array',
                        },
                    },
                    'required': ['x'],
                    'type': 'object',
                }
            ),
            snapshot(None),
        ),
        (
            tool_with_tuples,
            True,
            snapshot(
                {
                    'additionalProperties': False,
                    'properties': {
                        'x': {'maxItems': 1, 'minItems': 1, 'prefixItems': [{'type': 'integer'}], 'type': 'array'},
                        'y': {'maxItems': 1, 'minItems': 1, 'prefixItems': [{'type': 'string'}], 'type': 'array'},
                    },
                    'required': ['x', 'y'],
                    'type': 'object',
                }
            ),
            snapshot(True),
        ),
        # (tool, None, snapshot({}), snapshot({})),
        # (tool, True, snapshot({}), snapshot({})),
    ],
)
async def test_strict_mode_cannot_infer_strict(
    allow_model_requests: None,
    tool: Callable[..., Any],
    tool_strict: bool | None,
    expected_params: dict[str, Any],
    expected_strict: bool | None,
):
    """Test that strict mode settings are properly passed to OpenAI and respect precedence rules."""
    # Create a mock completion for testing
    c = completion_message(ChatCompletionMessage(content='world', role='assistant'))

    async def assert_strict(expected_strict: bool | None, profile: ModelProfile | None = None):
        mock_client = MockOpenAI.create_mock(c)
        m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client), profile=profile)
        agent = Agent(m)

        agent.tool_plain(strict=tool_strict)(tool)

        await agent.run('hello')
        kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
        assert 'tools' in kwargs, kwargs

        assert kwargs['tools'][0]['function']['parameters'] == expected_params
        actual_strict = kwargs['tools'][0]['function'].get('strict')
        assert actual_strict == expected_strict
        if actual_strict is None:
            # If strict is included, it should be non-None
            assert 'strict' not in kwargs['tools'][0]['function']

    await assert_strict(expected_strict)

    # If the model profile says strict is not supported, we never pass strict
    await assert_strict(
        None,
        profile=OpenAIModelProfile(openai_supports_strict_tool_definition=False).update(
            openai_model_profile('test-model')
        ),
    )


def test_strict_schema():
    class Apple(BaseModel):
        kind: Literal['apple'] = 'apple'

    class Banana(BaseModel):
        kind: Literal['banana'] = 'banana'

    class MyModel(BaseModel):
        # We have all these different crazy fields to achieve coverage
        my_recursive: MyModel | None = None
        my_patterns: dict[Annotated[str, Field(pattern='^my-pattern$')], str]
        my_tuple: tuple[int]
        my_list: list[float]
        my_discriminated_union: Annotated[Apple | Banana, Discriminator('kind')]

    assert OpenAIJsonSchemaTransformer(MyModel.model_json_schema(), strict=True).walk() == snapshot(
        {
            '$defs': {
                'Apple': {
                    'additionalProperties': False,
                    'properties': {'kind': {'const': 'apple', 'type': 'string'}},
                    'required': ['kind'],
                    'type': 'object',
                },
                'Banana': {
                    'additionalProperties': False,
                    'properties': {'kind': {'const': 'banana', 'type': 'string'}},
                    'required': ['kind'],
                    'type': 'object',
                },
                'MyModel': {
                    'additionalProperties': False,
                    'properties': {
                        'my_discriminated_union': {'anyOf': [{'$ref': '#/$defs/Apple'}, {'$ref': '#/$defs/Banana'}]},
                        'my_list': {'items': {'type': 'number'}, 'type': 'array'},
                        'my_patterns': {
                            'additionalProperties': False,
                            'description': "patternProperties={'^my-pattern$': {'type': 'string'}}",
                            'type': 'object',
                            'properties': {},
                            'required': [],
                        },
                        'my_recursive': {'anyOf': [{'$ref': '#'}, {'type': 'null'}]},
                        'my_tuple': {
                            'maxItems': 1,
                            'minItems': 1,
                            'prefixItems': [{'type': 'integer'}],
                            'type': 'array',
                        },
                    },
                    'required': ['my_recursive', 'my_patterns', 'my_tuple', 'my_list', 'my_discriminated_union'],
                    'type': 'object',
                },
            },
            'properties': {
                'my_recursive': {'anyOf': [{'$ref': '#'}, {'type': 'null'}]},
                'my_patterns': {
                    'type': 'object',
                    'description': "patternProperties={'^my-pattern$': {'type': 'string'}}",
                    'additionalProperties': False,
                    'properties': {},
                    'required': [],
                },
                'my_tuple': {'maxItems': 1, 'minItems': 1, 'prefixItems': [{'type': 'integer'}], 'type': 'array'},
                'my_list': {'items': {'type': 'number'}, 'type': 'array'},
                'my_discriminated_union': {'anyOf': [{'$ref': '#/$defs/Apple'}, {'$ref': '#/$defs/Banana'}]},
            },
            'required': ['my_recursive', 'my_patterns', 'my_tuple', 'my_list', 'my_discriminated_union'],
            'type': 'object',
            'additionalProperties': False,
        }
    )


def test_native_output_strict_mode(allow_model_requests: None):
    class CityLocation(BaseModel):
        city: str
        country: str

    c = completion_message(
        ChatCompletionMessage(content='{"city": "Mexico City", "country": "Mexico"}', role='assistant'),
    )
    mock_client = MockOpenAI.create_mock(c)
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))

    # Explicit strict=True
    agent = Agent(model, output_type=NativeOutput(CityLocation, strict=True))

    agent.run_sync('What is the capital of Mexico?')
    assert get_mock_chat_completion_kwargs(mock_client)[-1]['response_format']['json_schema']['strict'] is True

    # Explicit strict=False
    agent = Agent(model, output_type=NativeOutput(CityLocation, strict=False))

    agent.run_sync('What is the capital of Mexico?')
    assert get_mock_chat_completion_kwargs(mock_client)[-1]['response_format']['json_schema']['strict'] is False

    # Strict-compatible
    agent = Agent(model, output_type=NativeOutput(CityLocation))

    agent.run_sync('What is the capital of Mexico?')
    assert get_mock_chat_completion_kwargs(mock_client)[-1]['response_format']['json_schema']['strict'] is True

    # Strict-incompatible
    CityLocation.model_config = ConfigDict(extra='allow')

    agent = Agent(model, output_type=NativeOutput(CityLocation))

    agent.run_sync('What is the capital of Mexico?')
    assert get_mock_chat_completion_kwargs(mock_client)[-1]['response_format']['json_schema']['strict'] is False


async def test_openai_instructions(allow_model_requests: None, openai_api_key: str):
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m, instructions='You are a helpful assistant.')

    result = await agent.run('What is the capital of France?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is the capital of France?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='The capital of France is Paris.')],
                usage=RequestUsage(
                    input_tokens=24,
                    output_tokens=8,
                    details={
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0,
                    },
                ),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2025, 4, 7, 16, 30, 56, tzinfo=timezone.utc),
                },
                provider_response_id='chatcmpl-BJjf61mLb9z5H45ClJzbx0UWKwjo1',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_model_without_system_prompt(allow_model_requests: None, openai_api_key: str):
    m = OpenAIChatModel('o3-mini', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m, system_prompt='You are a potato.')
    result = await agent.run()
    assert result.output == snapshot(
        "That's right—I am a potato! A spud of many talents, here to help you out. How can this humble potato be of service today?"
    )


async def test_openai_instructions_with_tool_calls_keep_instructions(allow_model_requests: None, openai_api_key: str):
    m = OpenAIChatModel('gpt-4.1-mini', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m, instructions='You are a helpful assistant.')

    @agent.tool_plain
    async def get_temperature(city: str) -> float:
        return 20.0

    result = await agent.run('What is the temperature in Tokyo?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is the temperature in Tokyo?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_temperature', args='{"city":"Tokyo"}', tool_call_id=IsStr())],
                usage=RequestUsage(
                    input_tokens=50,
                    output_tokens=15,
                    details={
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0,
                    },
                ),
                model_name='gpt-4.1-mini-2025-04-14',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'tool_calls',
                    'timestamp': datetime(2025, 4, 16, 13, 37, 14, tzinfo=timezone.utc),
                },
                provider_response_id='chatcmpl-BMxEwRA0p0gJ52oKS7806KAlfMhqq',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_temperature', content=20.0, tool_call_id=IsStr(), timestamp=IsDatetime()
                    )
                ],
                timestamp=IsDatetime(),
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='The temperature in Tokyo is currently 20.0 degrees Celsius.')],
                usage=RequestUsage(
                    input_tokens=75,
                    output_tokens=15,
                    details={
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0,
                    },
                ),
                model_name='gpt-4.1-mini-2025-04-14',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2025, 4, 16, 13, 37, 15, tzinfo=timezone.utc),
                },
                provider_response_id='chatcmpl-BMxEx6B8JEj6oDC45MOWKp0phg8UP',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_model_thinking_part(allow_model_requests: None, openai_api_key: str):
    provider = OpenAIProvider(api_key=openai_api_key)
    responses_model = OpenAIResponsesModel('o3-mini', provider=provider)
    settings = OpenAIResponsesModelSettings(openai_reasoning_effort='high', openai_reasoning_summary='detailed')
    agent = Agent(responses_model, model_settings=settings)

    result = await agent.run('How do I cross the street?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='How do I cross the street?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c1fa166e9c81979ff56b16882744f1093f57e27128848a',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    ThinkingPart(content=IsStr(), id='rs_68c1fa166e9c81979ff56b16882744f1093f57e27128848a'),
                    TextPart(content=IsStr(), id='msg_68c1fa1ec9448197b5c8f78a90999360093f57e27128848a'),
                ],
                usage=RequestUsage(input_tokens=13, output_tokens=1915, details={'reasoning_tokens': 1600}),
                model_name='o3-mini-2025-01-31',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 10, 22, 21, 57, tzinfo=timezone.utc),
                },
                provider_response_id='resp_68c1fa0523248197888681b898567bde093f57e27128848a',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    result = await agent.run(
        'Considering the way to cross the street, analogously, how do I cross the river?',
        model=OpenAIChatModel('o3-mini', provider=provider),
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
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content=IsStr())],
                usage=RequestUsage(
                    input_tokens=577,
                    output_tokens=2320,
                    details={
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 1792,
                        'rejected_prediction_tokens': 0,
                    },
                ),
                model_name='o3-mini-2025-01-31',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2025, 9, 10, 22, 22, 24, tzinfo=timezone.utc),
                },
                provider_response_id='chatcmpl-CENUmtwDD0HdvTUYL6lUeijDtxrZL',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_instructions_with_logprobs(allow_model_requests: None):
    # Create a mock response with logprobs
    c = completion_message(
        ChatCompletionMessage(content='world', role='assistant'),
        logprobs=ChoiceLogprobs(
            content=[
                ChatCompletionTokenLogprob(
                    token='world', logprob=-0.6931, top_logprobs=[], bytes=[119, 111, 114, 108, 100]
                )
            ],
        ),
    )

    mock_client = MockOpenAI.create_mock(c)
    m = OpenAIChatModel(
        'gpt-4o',
        provider=OpenAIProvider(openai_client=mock_client),
    )
    agent = Agent(m, instructions='You are a helpful assistant.')
    result = await agent.run(
        'What is the capital of Minas Gerais?',
        model_settings=OpenAIChatModelSettings(openai_logprobs=True),
    )
    messages = result.all_messages()
    response = cast(Any, messages[1])
    assert response.provider_details is not None
    assert response.provider_details['logprobs'] == [
        {
            'token': 'world',
            'logprob': -0.6931,
            'bytes': [119, 111, 114, 108, 100],
            'top_logprobs': [],
        }
    ]


async def test_openai_instructions_with_responses_logprobs(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel(
        'gpt-4o-mini',
        provider=OpenAIProvider(api_key=openai_api_key),
    )
    agent = Agent(m, instructions='You are a helpful assistant.')
    result = await agent.run(
        'What is the capital of Minas Gerais?',
        model_settings=OpenAIResponsesModelSettings(openai_logprobs=True),
    )
    messages = result.all_messages()
    response = cast(Any, messages[1])
    text_part = response.parts[0]
    assert hasattr(text_part, 'provider_details')
    assert text_part.provider_details is not None
    assert 'logprobs' in text_part.provider_details
    assert text_part.provider_details['logprobs'] == [
        {'token': 'The', 'logprob': -0.0, 'bytes': [84, 104, 101], 'top_logprobs': []},
        {'token': ' capital', 'logprob': 0.0, 'bytes': [32, 99, 97, 112, 105, 116, 97, 108], 'top_logprobs': []},
        {'token': ' of', 'logprob': 0.0, 'bytes': [32, 111, 102], 'top_logprobs': []},
        {'token': ' Minas', 'logprob': -0.0, 'bytes': [32, 77, 105, 110, 97, 115], 'top_logprobs': []},
        {'token': ' Gerais', 'logprob': -0.0, 'bytes': [32, 71, 101, 114, 97, 105, 115], 'top_logprobs': []},
        {'token': ' is', 'logprob': -5.2e-05, 'bytes': [32, 105, 115], 'top_logprobs': []},
        {'token': ' Belo', 'logprob': -4.3e-05, 'bytes': [32, 66, 101, 108, 111], 'top_logprobs': []},
        {
            'token': ' Horizonte',
            'logprob': -2.0e-06,
            'bytes': [32, 72, 111, 114, 105, 122, 111, 110, 116, 101],
            'top_logprobs': [],
        },
        {'token': '.', 'logprob': -0.0, 'bytes': [46], 'top_logprobs': []},
    ]


async def test_openai_web_search_tool_model_not_supported(allow_model_requests: None, openai_api_key: str):
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(
        m, instructions='You are a helpful assistant.', builtin_tools=[WebSearchTool(search_context_size='low')]
    )

    with pytest.raises(
        UserError,
        match=r"WebSearchTool is not supported with `OpenAIChatModel` and model 'gpt-4o'.*OpenAIResponsesModel",
    ):
        await agent.run('What day is today?')


async def test_openai_web_search_tool(allow_model_requests: None, openai_api_key: str):
    m = OpenAIChatModel('gpt-4o-search-preview', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(
        m, instructions='You are a helpful assistant.', builtin_tools=[WebSearchTool(search_context_size='low')]
    )

    result = await agent.run('What day is today?')
    assert result.output == snapshot('May 14, 2025, 8:51:29 AM ')


async def test_openai_web_search_tool_with_user_location(allow_model_requests: None, openai_api_key: str):
    m = OpenAIChatModel('gpt-4o-search-preview', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(
        m,
        instructions='You are a helpful assistant.',
        builtin_tools=[WebSearchTool(user_location={'city': 'Utrecht', 'country': 'NL'})],
    )

    result = await agent.run('What is the current temperature?')
    assert result.output == snapshot("""\
Het is momenteel zonnig in Utrecht met een temperatuur van 22°C.

## Weer voor Utrecht, Nederland:
Huidige omstandigheden: Zonnig, 72°F (22°C)

Dagvoorspelling:
* woensdag, mei 14: minimum: 48°F (9°C), maximum: 71°F (22°C), beschrijving: Afnemende bewolking
* donderdag, mei 15: minimum: 43°F (6°C), maximum: 67°F (20°C), beschrijving: Na een bewolkt begin keert de zon terug
* vrijdag, mei 16: minimum: 45°F (7°C), maximum: 64°F (18°C), beschrijving: Overwegend zonnig
* zaterdag, mei 17: minimum: 47°F (9°C), maximum: 68°F (20°C), beschrijving: Overwegend zonnig
* zondag, mei 18: minimum: 47°F (8°C), maximum: 68°F (20°C), beschrijving: Deels zonnig
* maandag, mei 19: minimum: 49°F (9°C), maximum: 70°F (21°C), beschrijving: Deels zonnig
* dinsdag, mei 20: minimum: 49°F (10°C), maximum: 72°F (22°C), beschrijving: Zonnig tot gedeeltelijk bewolkt
 \
""")


async def test_reasoning_model_with_temperature(allow_model_requests: None, openai_api_key: str):
    m = OpenAIChatModel('o3-mini', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m, model_settings=OpenAIChatModelSettings(temperature=0.5))
    result = await agent.run('What is the capital of Mexico?')
    assert result.output == snapshot(
        'The capital of Mexico is Mexico City. It is not only the seat of the federal government but also a major cultural, political, and economic center in the country.'
    )


def test_openai_model_profile():
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key='foobar'))
    assert isinstance(m.profile, OpenAIModelProfile)


def test_openai_model_profile_custom():
    m = OpenAIChatModel(
        'gpt-4o',
        provider=OpenAIProvider(api_key='foobar'),
        profile=ModelProfile(json_schema_transformer=InlineDefsJsonSchemaTransformer),
    )
    assert isinstance(m.profile, ModelProfile)
    assert m.profile.json_schema_transformer is InlineDefsJsonSchemaTransformer

    m = OpenAIChatModel(
        'gpt-4o',
        provider=OpenAIProvider(api_key='foobar'),
        profile=OpenAIModelProfile(openai_supports_strict_tool_definition=False),
    )
    assert isinstance(m.profile, OpenAIModelProfile)
    assert m.profile.openai_supports_strict_tool_definition is False


def test_openai_model_profile_function():
    def model_profile(model_name: str) -> ModelProfile:
        return ModelProfile(json_schema_transformer=InlineDefsJsonSchemaTransformer if model_name == 'gpt-4o' else None)

    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key='foobar'), profile=model_profile)
    assert isinstance(m.profile, ModelProfile)
    assert m.profile.json_schema_transformer is InlineDefsJsonSchemaTransformer

    m = OpenAIChatModel('gpt-4o-mini', provider=OpenAIProvider(api_key='foobar'), profile=model_profile)
    assert isinstance(m.profile, ModelProfile)
    assert m.profile.json_schema_transformer is None


def test_openai_model_profile_from_provider():
    class CustomProvider(OpenAIProvider):
        def model_profile(self, model_name: str) -> ModelProfile:
            return ModelProfile(
                json_schema_transformer=InlineDefsJsonSchemaTransformer if model_name == 'gpt-4o' else None
            )

    m = OpenAIChatModel('gpt-4o', provider=CustomProvider(api_key='foobar'))
    assert isinstance(m.profile, ModelProfile)
    assert m.profile.json_schema_transformer is InlineDefsJsonSchemaTransformer

    m = OpenAIChatModel('gpt-4o-mini', provider=CustomProvider(api_key='foobar'))
    assert isinstance(m.profile, ModelProfile)
    assert m.profile.json_schema_transformer is None


def test_model_profile_strict_not_supported():
    my_tool = ToolDefinition(
        name='my_tool',
        description='This is my tool',
        parameters_json_schema={'type': 'object', 'title': 'Result', 'properties': {'spam': {'type': 'number'}}},
        strict=True,
    )

    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key='foobar'))
    tool_param = m._map_tool_definition(my_tool)  # type: ignore[reportPrivateUsage]

    assert tool_param == snapshot(
        {
            'type': 'function',
            'function': {
                'name': 'my_tool',
                'description': 'This is my tool',
                'parameters': {'type': 'object', 'title': 'Result', 'properties': {'spam': {'type': 'number'}}},
                'strict': True,
            },
        }
    )

    # Some models don't support strict tool definitions
    m = OpenAIChatModel(
        'gpt-4o',
        provider=OpenAIProvider(api_key='foobar'),
        profile=OpenAIModelProfile(openai_supports_strict_tool_definition=False).update(openai_model_profile('gpt-4o')),
    )
    tool_param = m._map_tool_definition(my_tool)  # type: ignore[reportPrivateUsage]

    assert tool_param == snapshot(
        {
            'type': 'function',
            'function': {
                'name': 'my_tool',
                'description': 'This is my tool',
                'parameters': {'type': 'object', 'title': 'Result', 'properties': {'spam': {'type': 'number'}}},
            },
        }
    )


async def test_compatible_api_with_tool_calls_without_id(allow_model_requests: None, gemini_api_key: str):
    provider = OpenAIProvider(
        openai_client=AsyncOpenAI(
            base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
            api_key=gemini_api_key,
        )
    )

    model = OpenAIChatModel('gemini-2.5-pro-preview-05-06', provider=provider)

    agent = Agent(model)

    @agent.tool_plain
    def get_current_time() -> str:
        """Get the current time."""
        return 'Noon'

    response = await agent.run('What is the current time?')
    assert response.output == snapshot('The current time is Noon.')


def test_openai_response_timestamp_milliseconds(allow_model_requests: None):
    c = completion_message(
        ChatCompletionMessage(content='world', role='assistant'),
    )
    # Some models on OpenRouter return timestamps in milliseconds rather than seconds
    # https://github.com/pydantic/pydantic-ai/issues/1877
    c.created = 1748747268000

    mock_client = MockOpenAI.create_mock(c)
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m)

    result = agent.run_sync('Hello')
    response = cast(ModelResponse, result.all_messages()[-1])
    assert response.timestamp == IsNow(tz=timezone.utc)
    assert response.provider_details == snapshot(
        {'finish_reason': 'stop', 'timestamp': datetime(2025, 6, 1, 3, 7, 48, tzinfo=timezone.utc)}
    )


async def test_openai_tool_output(allow_model_requests: None, openai_api_key: str):
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))

    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent(m, output_type=ToolOutput(CityLocation))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run('What is the largest city in the user country?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in the user country?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_user_country', args='{}', tool_call_id=IsStr())],
                usage=RequestUsage(
                    input_tokens=68,
                    output_tokens=12,
                    details={
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0,
                    },
                ),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'tool_calls',
                    'timestamp': datetime(2025, 5, 1, 23, 36, 24, tzinfo=timezone.utc),
                },
                provider_response_id='chatcmpl-BSXk0dWkG4hfPt0lph4oFO35iT73I',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"city": "Mexico City", "country": "Mexico"}',
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(
                    input_tokens=89,
                    output_tokens=36,
                    details={
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0,
                    },
                ),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'tool_calls',
                    'timestamp': datetime(2025, 5, 1, 23, 36, 25, tzinfo=timezone.utc),
                },
                provider_response_id='chatcmpl-BSXk1xGHYzbhXgUkSutK08bdoNv5s',
                finish_reason='tool_call',
                run_id=IsStr(),
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
            ),
        ]
    )


async def test_openai_text_output_function(allow_model_requests: None, openai_api_key: str):
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))

    def upcase(text: str) -> str:
        return text.upper()

    agent = Agent(m, output_type=TextOutput(upcase))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run('What is the largest city in the user country?')
    assert result.output == snapshot('THE LARGEST CITY IN MEXICO IS MEXICO CITY.')

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in the user country?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='get_user_country', args='{}', tool_call_id='call_J1YabdC7G7kzEZNbbZopwenH')
                ],
                usage=RequestUsage(
                    input_tokens=42,
                    output_tokens=11,
                    details={
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0,
                    },
                ),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'tool_calls',
                    'timestamp': datetime(2025, 6, 9, 21, 20, 53, tzinfo=timezone.utc),
                },
                provider_response_id='chatcmpl-BgeDFS85bfHosRFEEAvq8reaCPCZ8',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id='call_J1YabdC7G7kzEZNbbZopwenH',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='The largest city in Mexico is Mexico City.')],
                usage=RequestUsage(
                    input_tokens=63,
                    output_tokens=10,
                    details={
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0,
                    },
                ),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2025, 6, 9, 21, 20, 54, tzinfo=timezone.utc),
                },
                provider_response_id='chatcmpl-BgeDGX9eDyVrEI56aP2vtIHahBzFH',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_native_output(allow_model_requests: None, openai_api_key: str):
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))

    class CityLocation(BaseModel):
        """A city and its country."""

        city: str
        country: str

    agent = Agent(m, output_type=NativeOutput(CityLocation))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run('What is the largest city in the user country?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in the user country?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='get_user_country', args='{}', tool_call_id='call_PkRGedQNRFUzJp2R7dO7avWR')
                ],
                usage=RequestUsage(
                    input_tokens=71,
                    output_tokens=12,
                    details={
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0,
                    },
                ),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'tool_calls',
                    'timestamp': datetime(2025, 5, 1, 23, 36, 22, tzinfo=timezone.utc),
                },
                provider_response_id='chatcmpl-BSXjyBwGuZrtuuSzNCeaWMpGv2MZ3',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id='call_PkRGedQNRFUzJp2R7dO7avWR',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"city":"Mexico City","country":"Mexico"}')],
                usage=RequestUsage(
                    input_tokens=92,
                    output_tokens=15,
                    details={
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0,
                    },
                ),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2025, 5, 1, 23, 36, 23, tzinfo=timezone.utc),
                },
                provider_response_id='chatcmpl-BSXjzYGu67dhTy5r8KmjJvQ4HhDVO',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_native_output_multiple(allow_model_requests: None, openai_api_key: str):
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))

    class CityLocation(BaseModel):
        city: str
        country: str

    class CountryLanguage(BaseModel):
        country: str
        language: str

    agent = Agent(m, output_type=NativeOutput([CityLocation, CountryLanguage]))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run('What is the largest city in the user country?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in the user country?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='get_user_country', args='{}', tool_call_id='call_SIttSeiOistt33Htj4oiHOOX')
                ],
                usage=RequestUsage(
                    input_tokens=160,
                    output_tokens=11,
                    details={
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0,
                    },
                ),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'tool_calls',
                    'timestamp': datetime(2025, 6, 9, 23, 21, 26, tzinfo=timezone.utc),
                },
                provider_response_id='chatcmpl-Bgg5utuCSXMQ38j0n2qgfdQKcR9VD',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id='call_SIttSeiOistt33Htj4oiHOOX',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='{"result":{"kind":"CityLocation","data":{"city":"Mexico City","country":"Mexico"}}}'
                    )
                ],
                usage=RequestUsage(
                    input_tokens=181,
                    output_tokens=25,
                    details={
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0,
                    },
                ),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2025, 6, 9, 23, 21, 27, tzinfo=timezone.utc),
                },
                provider_response_id='chatcmpl-Bgg5vrxUtCDlvgMreoxYxPaKxANmd',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_prompted_output(allow_model_requests: None, openai_api_key: str):
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))

    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent(m, output_type=PromptedOutput(CityLocation))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run('What is the largest city in the user country?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in the user country?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='get_user_country', args='{}', tool_call_id='call_s7oT9jaLAsEqTgvxZTmFh0wB')
                ],
                usage=RequestUsage(
                    input_tokens=109,
                    output_tokens=11,
                    details={
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0,
                    },
                ),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'tool_calls',
                    'timestamp': datetime(2025, 6, 10, 0, 21, 35, tzinfo=timezone.utc),
                },
                provider_response_id='chatcmpl-Bgh27PeOaFW6qmF04qC5uI2H9mviw',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id='call_s7oT9jaLAsEqTgvxZTmFh0wB',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"city":"Mexico City","country":"Mexico"}')],
                usage=RequestUsage(
                    input_tokens=130,
                    output_tokens=11,
                    details={
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0,
                    },
                ),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2025, 6, 10, 0, 21, 36, tzinfo=timezone.utc),
                },
                provider_response_id='chatcmpl-Bgh28advCSFhGHPnzUevVS6g6Uwg0',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_prompted_output_multiple(allow_model_requests: None, openai_api_key: str):
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))

    class CityLocation(BaseModel):
        city: str
        country: str

    class CountryLanguage(BaseModel):
        country: str
        language: str

    agent = Agent(m, output_type=PromptedOutput([CityLocation, CountryLanguage]))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run('What is the largest city in the user country?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in the user country?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='get_user_country', args='{}', tool_call_id='call_wJD14IyJ4KKVtjCrGyNCHO09')
                ],
                usage=RequestUsage(
                    input_tokens=273,
                    output_tokens=11,
                    details={
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0,
                    },
                ),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'tool_calls',
                    'timestamp': datetime(2025, 6, 10, 0, 21, 38, tzinfo=timezone.utc),
                },
                provider_response_id='chatcmpl-Bgh2AW2NXGgMc7iS639MJXNRgtatR',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id='call_wJD14IyJ4KKVtjCrGyNCHO09',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='{"result":{"kind":"CityLocation","data":{"city":"Mexico City","country":"Mexico"}}}'
                    )
                ],
                usage=RequestUsage(
                    input_tokens=294,
                    output_tokens=21,
                    details={
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0,
                    },
                ),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2025, 6, 10, 0, 21, 39, tzinfo=timezone.utc),
                },
                provider_response_id='chatcmpl-Bgh2BthuopRnSqCuUgMbBnOqgkDHC',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_valid_response(env: TestEnv, allow_model_requests: None):
    """VCR recording is of a valid response."""
    env.set('OPENAI_API_KEY', 'foobar')
    agent = Agent('openai:gpt-4o')

    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is Paris.')


async def test_invalid_response(allow_model_requests: None):
    """VCR recording is of an invalid JSON response."""
    m = OpenAIChatModel(
        'gpt-4o',
        provider=OpenAIProvider(
            api_key='foobar', base_url='https://demo-endpoints.pydantic.workers.dev/bin/content-type/application/json'
        ),
    )
    agent = Agent(m)

    with pytest.raises(UnexpectedModelBehavior) as exc_info:
        await agent.run('What is the capital of France?')
    assert exc_info.value.message.startswith(
        'Invalid response from openai chat completions endpoint: 4 validation errors for ChatCompletion'
    )


async def test_text_response(allow_model_requests: None):
    """VCR recording is of a text response."""
    m = OpenAIChatModel(
        'gpt-4o', provider=OpenAIProvider(api_key='foobar', base_url='https://demo-endpoints.pydantic.workers.dev/bin/')
    )
    agent = Agent(m)

    with pytest.raises(UnexpectedModelBehavior) as exc_info:
        await agent.run('What is the capital of France?')
    assert exc_info.value.message == snapshot(
        'Invalid response from openai chat completions endpoint, expected JSON data'
    )


async def test_process_response_no_created_timestamp(allow_model_requests: None):
    c = completion_message(
        ChatCompletionMessage(content='world', role='assistant'),
    )
    c.created = None  # type: ignore

    mock_client = MockOpenAI.create_mock(c)
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m)
    result = await agent.run('Hello')
    messages = result.all_messages()
    response_message = messages[1]
    assert isinstance(response_message, ModelResponse)
    assert response_message.timestamp == IsNow(tz=timezone.utc)


async def test_process_response_no_finish_reason(allow_model_requests: None):
    c = completion_message(
        ChatCompletionMessage(content='world', role='assistant'),
    )
    c.choices[0].finish_reason = None  # type: ignore

    mock_client = MockOpenAI.create_mock(c)
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m)
    result = await agent.run('Hello')
    messages = result.all_messages()
    response_message = messages[1]
    assert isinstance(response_message, ModelResponse)
    assert response_message.finish_reason == 'stop'


async def test_tool_choice_fallback(allow_model_requests: None) -> None:
    profile = OpenAIModelProfile(openai_supports_tool_choice_required=False).update(openai_model_profile('stub'))

    mock_client = MockOpenAI.create_mock(completion_message(ChatCompletionMessage(content='ok', role='assistant')))
    model = OpenAIChatModel('stub', provider=OpenAIProvider(openai_client=mock_client), profile=profile)

    params = ModelRequestParameters(function_tools=[ToolDefinition(name='x')], allow_text_output=False)

    await model._completions_create(  # pyright: ignore[reportPrivateUsage]
        messages=[],
        stream=False,
        model_settings={},
        model_request_parameters=params,
    )

    assert get_mock_chat_completion_kwargs(mock_client)[0]['tool_choice'] == 'auto'


async def test_tool_choice_fallback_response_api(allow_model_requests: None) -> None:
    """Ensure tool_choice falls back to 'auto' for Responses API when 'required' unsupported."""
    profile = OpenAIModelProfile(openai_supports_tool_choice_required=False).update(openai_model_profile('stub'))

    mock_client = MockOpenAIResponses.create_mock(response_message([]))
    model = OpenAIResponsesModel('openai/gpt-oss', provider=OpenAIProvider(openai_client=mock_client), profile=profile)

    params = ModelRequestParameters(function_tools=[ToolDefinition(name='x')], allow_text_output=False)

    await model._responses_create(  # pyright: ignore[reportPrivateUsage]
        messages=[],
        stream=False,
        model_settings={},
        model_request_parameters=params,
    )

    assert get_mock_responses_kwargs(mock_client)[0]['tool_choice'] == 'auto'


async def test_openai_model_settings_temperature_ignored_on_gpt_5(allow_model_requests: None, openai_api_key: str):
    m = OpenAIChatModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)

    result = await agent.run('What is the capital of France?', model_settings=ModelSettings(temperature=0.0))
    assert result.output == snapshot('Paris.')


async def test_openai_model_cerebras_provider(allow_model_requests: None, cerebras_api_key: str):
    m = OpenAIChatModel('llama3.3-70b', provider=CerebrasProvider(api_key=cerebras_api_key))
    agent = Agent(m)

    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is Paris.')


async def test_openai_model_cerebras_provider_qwen_3_coder(allow_model_requests: None, cerebras_api_key: str):
    class Location(TypedDict):
        city: str
        country: str

    m = OpenAIChatModel('qwen-3-coder-480b', provider=CerebrasProvider(api_key=cerebras_api_key))
    agent = Agent(m, output_type=Location)

    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot({'city': 'Paris', 'country': 'France'})


async def test_openai_model_cerebras_provider_harmony(allow_model_requests: None, cerebras_api_key: str):
    m = OpenAIChatModel('gpt-oss-120b', provider=CerebrasProvider(api_key=cerebras_api_key))
    agent = Agent(m)

    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is **Paris**.')


def test_deprecated_openai_model(openai_api_key: str):
    with pytest.warns(DeprecationWarning):
        from pydantic_ai.models.openai import OpenAIModel  # type: ignore[reportDeprecated]

        provider = OpenAIProvider(api_key=openai_api_key)
        OpenAIModel('gpt-4o', provider=provider)  # type: ignore[reportDeprecated]


async def test_cache_point_filtering(allow_model_requests: None):
    """Test that CachePoint is filtered out in OpenAI Chat Completions requests."""
    c = completion_message(ChatCompletionMessage(content='response', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    m = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))

    # Test the instance method directly to trigger line 864
    msg = await m._map_user_prompt(UserPromptPart(content=['text before', CachePoint(), 'text after']))  # pyright: ignore[reportPrivateUsage]

    # CachePoint should be filtered out, only text content should remain
    assert msg['role'] == 'user'
    assert len(msg['content']) == 2  # type: ignore[reportUnknownArgumentType]
    assert msg['content'][0]['text'] == 'text before'  # type: ignore[reportUnknownArgumentType]
    assert msg['content'][1]['text'] == 'text after'  # type: ignore[reportUnknownArgumentType]


async def test_cache_point_filtering_responses_model():
    """Test that CachePoint is filtered out in OpenAI Responses API requests."""
    # Test the static method directly to trigger line 1680
    msg = await OpenAIResponsesModel._map_user_prompt(  # pyright: ignore[reportPrivateUsage]
        UserPromptPart(content=['text before', CachePoint(), 'text after'])
    )

    # CachePoint should be filtered out, only text content should remain
    assert msg['role'] == 'user'
    assert len(msg['content']) == 2
    assert msg['content'][0]['text'] == 'text before'  # type: ignore[reportUnknownArgumentType]
    assert msg['content'][1]['text'] == 'text after'  # type: ignore[reportUnknownArgumentType]


async def test_openai_custom_reasoning_field_sending_back_in_thinking_tags(allow_model_requests: None):
    c = completion_message(
        ChatCompletionMessage.model_construct(content='response', reasoning_content='reasoning', role='assistant')
    )
    m = OpenAIChatModel(
        'foobar',
        provider=OpenAIProvider(openai_client=MockOpenAI.create_mock(c)),
        profile=OpenAIModelProfile(
            openai_chat_thinking_field='reasoning_content',
            openai_chat_send_back_thinking_parts='tags',
        ),
    )
    settings = ModelSettings()
    params = ModelRequestParameters()
    resp = await m.request(messages=[], model_settings=settings, model_request_parameters=params)
    assert m._map_model_response(resp) == snapshot(  # type: ignore[reportPrivateUsage]
        {
            'role': 'assistant',
            'content': """\
<think>
reasoning
</think>

response\
""",
        }
    )


async def test_openai_custom_reasoning_field_sending_back_in_custom_field(allow_model_requests: None):
    c = completion_message(
        ChatCompletionMessage.model_construct(content='response', reasoning_content='reasoning', role='assistant')
    )
    m = OpenAIChatModel(
        'foobar',
        provider=OpenAIProvider(openai_client=MockOpenAI.create_mock(c)),
        profile=OpenAIModelProfile(
            openai_chat_thinking_field='reasoning_content',
            openai_chat_send_back_thinking_parts='field',
        ),
    )
    settings = ModelSettings()
    params = ModelRequestParameters()
    resp = await m.request(messages=[], model_settings=settings, model_request_parameters=params)
    assert m._map_model_response(resp) == snapshot(  # type: ignore[reportPrivateUsage]
        {'role': 'assistant', 'reasoning_content': 'reasoning', 'content': 'response'}
    )


async def test_openai_custom_reasoning_field_not_sending(allow_model_requests: None):
    c = completion_message(
        ChatCompletionMessage.model_construct(content='response', reasoning_content='reasoning', role='assistant')
    )
    m = OpenAIChatModel(
        'foobar',
        provider=OpenAIProvider(openai_client=MockOpenAI.create_mock(c)),
        profile=OpenAIModelProfile(
            openai_chat_thinking_field='reasoning_content',
            openai_chat_send_back_thinking_parts=False,
        ),
    )
    settings = ModelSettings()
    params = ModelRequestParameters()
    resp = await m.request(messages=[], model_settings=settings, model_request_parameters=params)
    assert m._map_model_response(resp) == snapshot(  # type: ignore[reportPrivateUsage]
        {'role': 'assistant', 'content': 'response'}
    )


async def test_openai_reasoning_in_thinking_tags(allow_model_requests: None):
    c = completion_message(
        ChatCompletionMessage.model_construct(content='<think>reasoning</think>response', role='assistant')
    )
    m = OpenAIChatModel(
        'foobar',
        provider=OpenAIProvider(openai_client=MockOpenAI.create_mock(c)),
        profile=OpenAIModelProfile(openai_chat_send_back_thinking_parts='tags'),
    )
    settings = ModelSettings()
    params = ModelRequestParameters()
    resp = await m.request(messages=[], model_settings=settings, model_request_parameters=params)
    assert m._map_model_response(resp) == snapshot(  # type: ignore[reportPrivateUsage]
        {
            'role': 'assistant',
            'content': """\
<think>
reasoning
</think>

response\
""",
        }
    )


async def test_openai_chat_instructions_after_system_prompts(allow_model_requests: None):
    """Test that instructions are inserted after all system prompts in mapped messages."""
    mock_client = MockOpenAI.create_mock(completion_message(ChatCompletionMessage(content='ok', role='assistant')))
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))

    messages: list[ModelRequest | ModelResponse] = [
        ModelRequest(
            parts=[
                SystemPromptPart(content='System prompt 1'),
                SystemPromptPart(content='System prompt 2'),
                UserPromptPart(content='Hello'),
            ],
            instructions='Instructions content',
        ),
    ]

    openai_messages = await model._map_messages(messages, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]

    # Verify order: system1, system2, instructions, user
    assert len(openai_messages) == 4
    assert openai_messages == snapshot(
        [
            {'role': 'system', 'content': 'System prompt 1'},
            {'role': 'system', 'content': 'System prompt 2'},
            {'content': 'Instructions content', 'role': 'system'},
            {'role': 'user', 'content': 'Hello'},
        ]
    )


def test_openai_chat_audio_default_base64(allow_model_requests: None):
    c = completion_message(ChatCompletionMessage(content='success', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    model = OpenAIChatModel('gpt-4o-audio-preview', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(model)

    # BinaryContent
    audio_data = b'fake_audio_data'
    binary_audio = BinaryContent(audio_data, media_type='audio/wav')

    agent.run_sync(['Process this audio', binary_audio])

    request_kwargs = get_mock_chat_completion_kwargs(mock_client)
    messages = request_kwargs[0]['messages']
    user_message = messages[0]

    # Find the input_audio part
    audio_part = next(part for part in user_message['content'] if part['type'] == 'input_audio')

    # Expect raw base64
    expected_data = base64.b64encode(audio_data).decode('utf-8')
    assert audio_part['input_audio']['data'] == expected_data
    assert audio_part['input_audio']['format'] == 'wav'


def test_openai_chat_audio_uri_encoding(allow_model_requests: None):
    c = completion_message(ChatCompletionMessage(content='success', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)

    # Set profile to use URI encoding
    profile = OpenAIModelProfile(openai_chat_audio_input_encoding='uri')
    model = OpenAIChatModel('gpt-4o-audio-preview', provider=OpenAIProvider(openai_client=mock_client), profile=profile)
    agent = Agent(model)

    # BinaryContent
    audio_data = b'fake_audio_data'
    binary_audio = BinaryContent(audio_data, media_type='audio/wav')

    agent.run_sync(['Process this audio', binary_audio])

    request_kwargs = get_mock_chat_completion_kwargs(mock_client)
    messages = request_kwargs[0]['messages']
    user_message = messages[0]

    # Find the input_audio part
    audio_part = next(part for part in user_message['content'] if part['type'] == 'input_audio')

    # Expect Data URI
    expected_data = f'data:audio/wav;base64,{base64.b64encode(audio_data).decode("utf-8")}'
    assert audio_part['input_audio']['data'] == expected_data
    assert audio_part['input_audio']['format'] == 'wav'


async def test_openai_chat_audio_url_default_base64(allow_model_requests: None):
    c = completion_message(ChatCompletionMessage(content='success', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    model = OpenAIChatModel('gpt-4o-audio-preview', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(model)

    audio_url = AudioUrl('https://example.com/audio.mp3')

    # Mock download_item to return base64 data
    fake_base64_data = base64.b64encode(b'fake_downloaded_audio').decode('utf-8')

    with patch('pydantic_ai.models.openai.download_item') as mock_download:
        mock_download.return_value = {'data': fake_base64_data, 'data_type': 'mp3'}

        await agent.run(['Process this audio url', audio_url])

    request_kwargs = get_mock_chat_completion_kwargs(mock_client)
    messages = request_kwargs[0]['messages']
    user_message = messages[0]

    # Find the input_audio part
    audio_part = next(part for part in user_message['content'] if part['type'] == 'input_audio')

    # Expect raw base64 (which is what download_item returns in this mock)
    assert audio_part['input_audio']['data'] == fake_base64_data
    assert audio_part['input_audio']['format'] == 'mp3'


async def test_openai_chat_audio_url_uri_encoding(allow_model_requests: None):
    c = completion_message(ChatCompletionMessage(content='success', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)

    # Set profile to use URI encoding
    profile = OpenAIModelProfile(openai_chat_audio_input_encoding='uri')
    model = OpenAIChatModel('gpt-4o-audio-preview', provider=OpenAIProvider(openai_client=mock_client), profile=profile)
    agent = Agent(model)

    audio_url = AudioUrl('https://example.com/audio.mp3')

    # Mock download_item to return Data URI (since we're calling with data_format='base64_uri')
    fake_base64_data = base64.b64encode(b'fake_downloaded_audio').decode('utf-8')
    data_uri = f'data:audio/mpeg;base64,{fake_base64_data}'

    with patch('pydantic_ai.models.openai.download_item') as mock_download:
        mock_download.return_value = {'data': data_uri, 'data_type': 'mp3'}

        await agent.run(['Process this audio url', audio_url])

    request_kwargs = get_mock_chat_completion_kwargs(mock_client)
    messages = request_kwargs[0]['messages']
    user_message = messages[0]

    # Find the input_audio part
    audio_part = next(part for part in user_message['content'] if part['type'] == 'input_audio')

    # Expect Data URI with correct MIME type for mp3
    assert audio_part['input_audio']['data'] == data_uri
    assert audio_part['input_audio']['format'] == 'mp3'
