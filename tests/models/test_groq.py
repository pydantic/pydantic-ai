from __future__ import annotations as _annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from typing import Any, Literal, Union, cast
from unittest.mock import patch

import httpx
import pytest
from dirty_equals import IsListOrTuple
from inline_snapshot import snapshot
from typing_extensions import TypedDict

from pydantic_ai import Agent, ModelHTTPError, ModelRetry, UnexpectedModelBehavior
from pydantic_ai.messages import (
    BinaryContent,
    FinalResultEvent,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import Usage

from ..conftest import IsDatetime, IsInstance, IsNow, IsStr, raise_if_exception, try_import
from .mock_async_stream import MockAsyncStream

with try_import() as imports_successful:
    from groq import APIStatusError, AsyncGroq
    from groq.types import chat
    from groq.types.chat.chat_completion import Choice
    from groq.types.chat.chat_completion_chunk import (
        Choice as ChunkChoice,
        ChoiceDelta,
        ChoiceDeltaToolCall,
        ChoiceDeltaToolCallFunction,
    )
    from groq.types.chat.chat_completion_message import ChatCompletionMessage
    from groq.types.chat.chat_completion_message_tool_call import Function
    from groq.types.completion_usage import CompletionUsage

    from pydantic_ai.models.groq import GroqModel, GroqModelSettings
    from pydantic_ai.providers.groq import GroqProvider

    # note: we use Union here so that casting works with Python 3.9
    MockChatCompletion = Union[chat.ChatCompletion, Exception]
    MockChatCompletionChunk = Union[chat.ChatCompletionChunk, Exception]

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='groq not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


def test_init():
    m = GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(api_key='foobar'))
    assert m.client.api_key == 'foobar'
    assert m.model_name == 'llama-3.3-70b-versatile'
    assert m.system == 'groq'
    assert m.base_url == 'https://api.groq.com'


@dataclass
class MockGroq:
    completions: MockChatCompletion | Sequence[MockChatCompletion] | None = None
    stream: Sequence[MockChatCompletionChunk] | Sequence[Sequence[MockChatCompletionChunk]] | None = None
    index: int = 0

    @cached_property
    def chat(self) -> Any:
        chat_completions = type('Completions', (), {'create': self.chat_completions_create})
        return type('Chat', (), {'completions': chat_completions})

    @classmethod
    def create_mock(cls, completions: MockChatCompletion | Sequence[MockChatCompletion]) -> AsyncGroq:
        return cast(AsyncGroq, cls(completions=completions))

    @classmethod
    def create_mock_stream(
        cls,
        stream: Sequence[MockChatCompletionChunk] | Sequence[Sequence[MockChatCompletionChunk]],
    ) -> AsyncGroq:
        return cast(AsyncGroq, cls(stream=stream))

    async def chat_completions_create(
        self, *_args: Any, stream: bool = False, **_kwargs: Any
    ) -> chat.ChatCompletion | MockAsyncStream[MockChatCompletionChunk]:
        if stream:
            assert self.stream is not None, 'you can only used `stream=True` if `stream` is provided'
            if isinstance(self.stream[0], Sequence):
                response = MockAsyncStream(  # pragma: no cover
                    iter(cast(list[MockChatCompletionChunk], self.stream[self.index]))
                )
            else:
                response = MockAsyncStream(iter(cast(list[MockChatCompletionChunk], self.stream)))
        else:
            assert self.completions is not None, 'you can only used `stream=False` if `completions` are provided'
            if isinstance(self.completions, Sequence):
                raise_if_exception(self.completions[self.index])
                response = cast(chat.ChatCompletion, self.completions[self.index])
            else:
                raise_if_exception(self.completions)
                response = cast(chat.ChatCompletion, self.completions)
        self.index += 1
        return response


def completion_message(message: ChatCompletionMessage, *, usage: CompletionUsage | None = None) -> chat.ChatCompletion:
    return chat.ChatCompletion(
        id='123',
        choices=[Choice(finish_reason='stop', index=0, message=message)],
        created=1704067200,  # 2024-01-01
        model='llama-3.3-70b-versatile-123',
        object='chat.completion',
        usage=usage,
    )


async def test_request_simple_success(allow_model_requests: None):
    c = completion_message(ChatCompletionMessage(content='world', role='assistant'))
    mock_client = MockGroq.create_mock(c)
    m = GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(groq_client=mock_client))
    agent = Agent(m)

    result = await agent.run('hello')
    assert result.output == 'world'
    assert result.usage() == snapshot(Usage(requests=1))

    # reset the index so we get the same response again
    mock_client.index = 0  # type: ignore

    result = await agent.run('hello', message_history=result.new_messages())
    assert result.output == 'world'
    assert result.usage() == snapshot(Usage(requests=1))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[TextPart(content='world')],
                usage=Usage(requests=1),
                model_name='llama-3.3-70b-versatile-123',
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                vendor_id='123',
            ),
            ModelRequest(parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[TextPart(content='world')],
                usage=Usage(requests=1),
                model_name='llama-3.3-70b-versatile-123',
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                vendor_id='123',
            ),
        ]
    )


async def test_request_simple_usage(allow_model_requests: None):
    c = completion_message(
        ChatCompletionMessage(content='world', role='assistant'),
        usage=CompletionUsage(completion_tokens=1, prompt_tokens=2, total_tokens=3),
    )
    mock_client = MockGroq.create_mock(c)
    m = GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(groq_client=mock_client))
    agent = Agent(m)

    result = await agent.run('Hello')
    assert result.output == 'world'


async def test_request_structured_response(allow_model_requests: None):
    c = completion_message(
        ChatCompletionMessage(
            content=None,
            role='assistant',
            tool_calls=[
                chat.ChatCompletionMessageToolCall(
                    id='123',
                    function=Function(arguments='{"response": [1, 2, 123]}', name='final_result'),
                    type='function',
                )
            ],
        )
    )
    mock_client = MockGroq.create_mock(c)
    m = GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(groq_client=mock_client))
    agent = Agent(m, output_type=list[int])

    result = await agent.run('Hello')
    assert result.output == [1, 2, 123]
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"response": [1, 2, 123]}',
                        tool_call_id='123',
                    )
                ],
                usage=Usage(requests=1),
                model_name='llama-3.3-70b-versatile-123',
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
                vendor_id='123',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='123',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
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
                    chat.ChatCompletionMessageToolCall(
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
            ),
        ),
        completion_message(
            ChatCompletionMessage(
                content=None,
                role='assistant',
                tool_calls=[
                    chat.ChatCompletionMessageToolCall(
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
            ),
        ),
        completion_message(ChatCompletionMessage(content='final response', role='assistant')),
    ]
    mock_client = MockGroq.create_mock(responses)
    m = GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(groq_client=mock_client))
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
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_location',
                        args='{"loc_name": "San Fransisco"}',
                        tool_call_id='1',
                    )
                ],
                usage=Usage(requests=1, request_tokens=2, response_tokens=1, total_tokens=3),
                model_name='llama-3.3-70b-versatile-123',
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                vendor_id='123',
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        tool_name='get_location',
                        content='Wrong location, please try again',
                        tool_call_id='1',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_location',
                        args='{"loc_name": "London"}',
                        tool_call_id='2',
                    )
                ],
                usage=Usage(requests=1, request_tokens=3, response_tokens=2, total_tokens=6),
                model_name='llama-3.3-70b-versatile-123',
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                vendor_id='123',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_location',
                        content='{"lat": 51, "lng": 0}',
                        tool_call_id='2',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='final response')],
                usage=Usage(requests=1),
                model_name='llama-3.3-70b-versatile-123',
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                vendor_id='123',
            ),
        ]
    )


FinishReason = Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call']


def chunk(delta: list[ChoiceDelta], finish_reason: FinishReason | None = None) -> chat.ChatCompletionChunk:
    return chat.ChatCompletionChunk(
        id='x',
        choices=[
            ChunkChoice(index=index, delta=delta, finish_reason=finish_reason) for index, delta in enumerate(delta)
        ],
        created=1704067200,  # 2024-01-01
        x_groq=None,
        model='llama-3.3-70b-versatile',
        object='chat.completion.chunk',
        usage=CompletionUsage(completion_tokens=1, prompt_tokens=2, total_tokens=3),
    )


def text_chunk(text: str, finish_reason: FinishReason | None = None) -> chat.ChatCompletionChunk:
    return chunk([ChoiceDelta(content=text, role='assistant')], finish_reason=finish_reason)


async def test_stream_text(allow_model_requests: None):
    stream = text_chunk('hello '), text_chunk('world'), chunk([])
    mock_client = MockGroq.create_mock_stream(stream)
    m = GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(groq_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream(debounce_by=None)] == snapshot(['hello ', 'hello world', 'hello world'])
        assert result.is_complete


async def test_stream_text_finish_reason(allow_model_requests: None):
    stream = text_chunk('hello '), text_chunk('world'), text_chunk('.', finish_reason='stop')
    mock_client = MockGroq.create_mock_stream(stream)
    m = GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(groq_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream(debounce_by=None)] == snapshot(
            ['hello ', 'hello world', 'hello world.', 'hello world.']
        )
        assert result.is_complete


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
    stream = (
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
    )
    mock_client = MockGroq.create_mock_stream(stream)
    m = GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(groq_client=mock_client))
    agent = Agent(m, output_type=MyTypedDict)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [dict(c) async for c in result.stream(debounce_by=None)] == snapshot(
            [
                {},
                {'first': 'One'},
                {'first': 'One', 'second': 'Two'},
                {'first': 'One', 'second': 'Two'},
                {'first': 'One', 'second': 'Two'},
            ]
        )
        assert result.is_complete

    assert result.usage() == snapshot(Usage(requests=1))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"first": "One", "second": "Two"}',
                        tool_call_id=IsStr(),
                    )
                ],
                model_name='llama-3.3-70b-versatile',
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
            ),
        ]
    )


async def test_stream_structured_finish_reason(allow_model_requests: None):
    stream = (
        struc_chunk('final_result', None),
        struc_chunk(None, '{"first": "One'),
        struc_chunk(None, '", "second": "Two"'),
        struc_chunk(None, '}'),
        struc_chunk(None, None, finish_reason='stop'),
    )
    mock_client = MockGroq.create_mock_stream(stream)
    m = GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(groq_client=mock_client))
    agent = Agent(m, output_type=MyTypedDict)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [dict(c) async for c in result.stream(debounce_by=None)] == snapshot(
            [
                {'first': 'One'},
                {'first': 'One', 'second': 'Two'},
                {'first': 'One', 'second': 'Two'},
                {'first': 'One', 'second': 'Two'},
                {'first': 'One', 'second': 'Two'},
            ]
        )
        assert result.is_complete


async def test_no_content(allow_model_requests: None):
    stream = chunk([ChoiceDelta()]), chunk([ChoiceDelta()])
    mock_client = MockGroq.create_mock_stream(stream)
    m = GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(groq_client=mock_client))
    agent = Agent(m, output_type=MyTypedDict)

    with pytest.raises(UnexpectedModelBehavior, match='Received empty model response'):
        async with agent.run_stream(''):
            pass


async def test_no_delta(allow_model_requests: None):
    stream = chunk([]), text_chunk('hello '), text_chunk('world')
    mock_client = MockGroq.create_mock_stream(stream)
    m = GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(groq_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream(debounce_by=None)] == snapshot(['hello ', 'hello world', 'hello world'])
        assert result.is_complete


@pytest.mark.vcr()
async def test_extra_headers(allow_model_requests: None, groq_api_key: str):
    # This test doesn't do anything, it's just here to ensure that calls with `extra_headers` don't cause errors, including type.
    m = GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(api_key=groq_api_key))
    agent = Agent(m, model_settings=GroqModelSettings(extra_headers={'Extra-Header-Key': 'Extra-Header-Value'}))
    await agent.run('hello')


@pytest.mark.vcr()
async def test_image_url_input(allow_model_requests: None, groq_api_key: str):
    m = GroqModel('meta-llama/llama-4-scout-17b-16e-instruct', provider=GroqProvider(api_key=groq_api_key))
    agent = Agent(m)

    result = await agent.run(
        [
            'What is the name of this fruit?',
            ImageUrl(url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg'),
        ]
    )
    assert result.output == snapshot(
        'The fruit depicted in the image is a potato. Although commonly mistaken as a vegetable, potatoes are technically fruits because they are the edible, ripened ovary of a flower, containing seeds. However, in culinary and everyday contexts, potatoes are often referred to as a vegetable due to their savory flavor and uses in dishes. The botanical classification of a potato as a fruit comes from its origin as the tuberous part of the Solanum tuberosum plant, which produces flowers and subsequently the potato as a fruit that grows underground.'
    )


@pytest.mark.vcr()
async def test_image_as_binary_content_tool_response(
    allow_model_requests: None, groq_api_key: str, image_content: BinaryContent
):
    m = GroqModel('meta-llama/llama-4-scout-17b-16e-instruct', provider=GroqProvider(api_key=groq_api_key))
    agent = Agent(m)

    @agent.tool_plain
    async def get_image() -> BinaryContent:
        return image_content

    result = await agent.run(
        ['What fruit is in the image you can get from the get_image tool (without any arguments)?']
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'What fruit is in the image you can get from the get_image tool (without any arguments)?'
                        ],
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_image', args='{}', tool_call_id='call_wkpd')],
                usage=Usage(requests=1, request_tokens=192, response_tokens=8, total_tokens=200),
                model_name='meta-llama/llama-4-scout-17b-16e-instruct',
                timestamp=IsDatetime(),
                vendor_id='chatcmpl-3c327c89-e9f5-4aac-a5d5-190e6f6f25c9',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_image',
                        content='See file 1c8566',
                        tool_call_id='call_wkpd',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content=[
                            'This is file 1c8566:',
                            image_content,
                        ],
                        timestamp=IsDatetime(),
                    ),
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='The fruit in the image is a kiwi.')],
                usage=Usage(requests=1, request_tokens=2552, response_tokens=11, total_tokens=2563),
                model_name='meta-llama/llama-4-scout-17b-16e-instruct',
                timestamp=IsDatetime(),
                vendor_id='chatcmpl-82dfad42-6a28-4089-82c3-c8633f626c0d',
            ),
        ]
    )


@pytest.mark.parametrize('media_type', ['audio/wav', 'audio/mpeg'])
async def test_audio_as_binary_content_input(allow_model_requests: None, media_type: str):
    c = completion_message(ChatCompletionMessage(content='world', role='assistant'))
    mock_client = MockGroq.create_mock(c)
    m = GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(groq_client=mock_client))
    agent = Agent(m)

    base64_content = b'//uQZ'

    with pytest.raises(RuntimeError, match='Only images are supported for binary content in Groq.'):
        await agent.run(['hello', BinaryContent(data=base64_content, media_type=media_type)])


async def test_image_as_binary_content_input(
    allow_model_requests: None, groq_api_key: str, image_content: BinaryContent
) -> None:
    m = GroqModel('meta-llama/llama-4-scout-17b-16e-instruct', provider=GroqProvider(api_key=groq_api_key))
    agent = Agent(m)

    result = await agent.run(['What is the name of this fruit?', image_content])
    assert result.output == snapshot(
        'The fruit depicted in the image is a kiwi. The image shows a cross-section of a kiwi, revealing its characteristic green flesh and black seeds arranged in a radial pattern around a central white area. The fuzzy brown skin is visible on the edge of the slice.'
    )


def test_model_status_error(allow_model_requests: None) -> None:
    mock_client = MockGroq.create_mock(
        APIStatusError(
            'test error',
            response=httpx.Response(status_code=500, request=httpx.Request('POST', 'https://example.com/v1')),
            body={'error': 'test error'},
        )
    )
    m = GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(groq_client=mock_client))
    agent = Agent(m)
    with pytest.raises(ModelHTTPError) as exc_info:
        agent.run_sync('hello')
    assert str(exc_info.value) == snapshot(
        "status_code: 500, model_name: llama-3.3-70b-versatile, body: {'error': 'test error'}"
    )


async def test_init_with_provider():
    provider = GroqProvider(api_key='api-key')
    model = GroqModel('llama3-8b-8192', provider=provider)
    assert model.model_name == 'llama3-8b-8192'
    assert model.client == provider.client


async def test_init_with_provider_string():
    with patch.dict(os.environ, {'GROQ_API_KEY': 'env-api-key'}, clear=False):
        model = GroqModel('llama3-8b-8192', provider='groq')
        assert model.model_name == 'llama3-8b-8192'
        assert model.client is not None


async def test_groq_model_instructions(allow_model_requests: None, groq_api_key: str):
    m = GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(api_key=groq_api_key))
    agent = Agent(m, instructions='You are a helpful assistant.')

    result = await agent.run('What is the capital of France?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is the capital of France?', timestamp=IsDatetime())],
                instructions='You are a helpful assistant.',
            ),
            ModelResponse(
                parts=[TextPart(content='The capital of France is Paris.')],
                usage=Usage(requests=1, request_tokens=48, response_tokens=8, total_tokens=56),
                model_name='llama-3.3-70b-versatile',
                timestamp=IsDatetime(),
                vendor_id='chatcmpl-7586b6a9-fb4b-4ec7-86a0-59f0a77844cf',
            ),
        ]
    )


async def test_groq_model_thinking_part(allow_model_requests: None, groq_api_key: str):
    m = GroqModel('deepseek-r1-distill-llama-70b', provider=GroqProvider(api_key=groq_api_key))
    settings = GroqModelSettings(groq_reasoning_format='raw')
    agent = Agent(m, instructions='You are a chef.', model_settings=settings)

    result = await agent.run('I want a recipe to cook Uruguayan alfajores.')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='I want a recipe to cook Uruguayan alfajores.', timestamp=IsDatetime())],
                instructions='You are a chef.',
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content=IsStr()),
                    TextPart(
                        content="""\


To make Uruguayan alfajores, follow these organized steps for a delightful baking experience:

### Ingredients:
- 2 cups all-purpose flour
- 1 cup cornstarch
- 1 tsp baking powder
- 1/2 tsp salt
- 1 cup unsalted butter, softened
- 1 cup powdered sugar
- 1 egg
- 1 tsp vanilla extract
- Dulce de leche (store-bought or homemade)
- Powdered sugar for coating

### Instructions:

1. **Preheat Oven:**
   - Preheat your oven to 300°F (150°C). Line a baking sheet with parchment paper.

2. **Prepare Dough:**
   - In a bowl, whisk together flour, cornstarch, baking powder, and salt.
   - In another bowl, cream butter and powdered sugar until smooth. Add egg and vanilla, mixing well.
   - Gradually incorporate the dry ingredients into the wet mixture until a dough forms. Wrap and let rest for 30 minutes.

3. **Roll and Cut:**
   - Roll dough to 1/4 inch thickness. Cut into 2-inch circles using a cutter or glass.

4. **Bake:**
   - Place cookies on the prepared baking sheet, bake for 15-20 minutes until edges are lightly golden. Cool on the sheet for 5 minutes, then transfer to a wire rack to cool completely.

5. **Assemble Alfajores:**
   - Spread a layer of dulce de leche on one cookie half. Sandwich with another cookie. Handle gently to avoid breaking.

6. **Coat with Powdered Sugar:**
   - Roll each alfajor in powdered sugar, pressing gently to adhere.

7. **Optional Chocolate Coating:**
   - For a chocolate version, melt chocolate and dip alfajores, then chill to set.

8. **Storage:**
   - Store in an airtight container at room temperature for up to a week. Freeze for longer storage.

### Tips:
- Ensure butter is softened for smooth creaming.
- Check cookies after 15 minutes to avoid over-browning.
- Allow cookies to cool completely before handling.
- Homemade dulce de leche can be made by heating condensed milk until thickened.

Enjoy your traditional Uruguayan alfajores with a cup of coffee or tea!\
"""
                    ),
                ],
                usage=Usage(requests=1, request_tokens=21, response_tokens=1414, total_tokens=1435),
                model_name='deepseek-r1-distill-llama-70b',
                timestamp=IsDatetime(),
                vendor_id=IsStr(),
            ),
        ]
    )

    result = await agent.run(
        'Considering the Uruguayan recipe, how can I cook the Argentinian one?',
        message_history=result.all_messages(),
        model_settings=GroqModelSettings(groq_reasoning_format='parsed'),
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='I want a recipe to cook Uruguayan alfajores.', timestamp=IsDatetime())],
                instructions='You are a chef.',
            ),
            ModelResponse(
                parts=[IsInstance(ThinkingPart), IsInstance(TextPart)],
                usage=Usage(requests=1, request_tokens=21, response_tokens=1414, total_tokens=1435),
                model_name='deepseek-r1-distill-llama-70b',
                timestamp=IsDatetime(),
                vendor_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Considering the Uruguayan recipe, how can I cook the Argentinian one?',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions='You are a chef.',
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="""\
Alright, so I want to make Argentinian alfajores after successfully making the Uruguayan ones. I know that both countries have their own versions of alfajores, but I'm not entirely sure how they differ. Maybe the ingredients or the preparation steps are slightly different? I should probably start by researching what makes Argentinian alfajores unique compared to the Uruguayan ones.

First, I remember that in the Uruguayan recipe, the cookies were more delicate, and the filling was a generous amount of dulce de leche. The process involved making the dough from scratch, baking the cookies, and then assembling them with the filling and sometimes coating them in powdered sugar or chocolate.

For Argentinian alfajores, I think the cookies might have a different texture—perhaps they're crunchier or have a different flavor profile. Maybe the type of flour used is different, or there's an addition like cocoa powder for a chocolate version. Also, the filling might have variations, like adding caramel or nuts to the dulce de leche.

I should look up some authentic Argentinian recipes to see the differences. Maybe the method of making the dough is slightly different, or the baking time and temperature vary. I also wonder if Argentinian alfajores are typically coated in something else besides powdered sugar, like cinnamon or coconut flakes.

Another thing to consider is the size and shape of the cookies. Uruguayan alfajores might be smaller and rounder, while Argentinian ones could be larger or have a different shape. I should also check if there are any additional steps, like toasting the cookies or adding a layer of meringue.

I need to make sure I have all the necessary ingredients. If the Argentinian version requires something different, like a specific type of flour or a particular flavoring, I'll need to adjust my shopping list. Also, if the filling is different, I might need to prepare it in a different way or add extra ingredients.

I should also think about the assembly process. Maybe Argentinian alfajores are sandwiched with more filling or have a different way of sealing the cookies together. Perhaps they're rolled in coconut after being filled, or there's a step involving dipping them in chocolate.

It would be helpful to watch a video or read a detailed recipe from an Argentinian source to get a better understanding. That way, I can follow the traditional method and ensure that my alfajores turn out authentic. I should also consider any tips or tricks that Argentinian bakers use to make their alfajores special.

Finally, I need to plan the timing. Making alfajores can be a bit time-consuming, especially if you're making the dulce de leche from scratch. I should allocate enough time for preparing the dough, baking the cookies, and assembling the alfajores.

Overall, the key steps I think I'll need to follow are:

1. Research authentic Argentinian alfajores recipes to identify unique ingredients and steps.
2. Adjust the ingredient list based on the differences from the Uruguayan version.
3. Prepare the dough according to the Argentinian method, which might involve different mixing techniques or additional ingredients.
4. Bake the cookies, possibly at a different temperature or for a different duration.
5. Prepare the filling, which might include variations like caramel or nuts.
6. Assemble the alfajores, possibly adding extra coatings or layers.
7. Allow the alfajores to set before serving, to ensure the flavors meld together.

By carefully following these steps and paying attention to the unique aspects of Argentinian alfajores, I should be able to create a delicious and authentic batch that captures the essence of this traditional South American treat.
"""
                    ),
                    TextPart(
                        content="""\
To create authentic Argentinian alfajores, follow these organized steps, which highlight the unique characteristics and differences from the Uruguayan version:

### Ingredients:
- **For the Cookies:**
  - 2 cups all-purpose flour
  - 1/2 cup cornstarch
  - 1/4 cup unsweetened cocoa powder (optional for chocolate version)
  - 1 teaspoon baking powder
  - 1/2 teaspoon salt
  - 1 cup unsalted butter, softened
  - 1 cup powdered sugar
  - 1 egg
  - 1 teaspoon vanilla extract

- **For the Filling:**
  - Dulce de leche (store-bought or homemade)
  - Optional: caramel sauce, chopped nuts, or cinnamon for added flavor

- **For Coating:**
  - Powdered sugar
  - Optional: cinnamon, shredded coconut, or melted chocolate

### Instructions:

1. **Prepare the Dough:**
   - In a large bowl, whisk together the flour, cornstarch, cocoa powder (if using), baking powder, and salt.
   - In another bowl, cream the softened butter and powdered sugar until smooth. Add the egg and vanilla extract, mixing well.
   - Gradually incorporate the dry ingredients into the wet mixture until a dough forms. Wrap the dough in plastic wrap and let it rest for 30 minutes.

2. **Roll and Cut the Cookies:**
   - Roll the dough to about 1/4 inch thickness on a lightly floured surface.
   - Use a round cookie cutter or the rim of a glass to cut out circles of dough. Argentinian alfajores are often slightly larger than their Uruguayan counterparts.

3. **Bake the Cookies:**
   - Preheat the oven to 300°F (150°C). Line a baking sheet with parchment paper.
   - Place the cookie rounds on the prepared baking sheet, leaving about 1 inch of space between each cookie.
   - Bake for 15-20 minutes, or until the edges are lightly golden. Allow the cookies to cool on the baking sheet for 5 minutes before transferring them to a wire rack to cool completely.

4. **Prepare the Filling:**
   - Use store-bought dulce de leche or make your own by heating sweetened condensed milk until thickened.
   - Optional: Stir in caramel sauce or chopped nuts into the dulce de leche for added flavor.

5. **Assemble the Alfajores:**
   - Once the cookies are completely cool, spread a generous amount of dulce de leche on the flat side of one cookie.
   - Sandwich with another cookie, pressing gently to adhere. For an extra touch, roll the edges in chopped nuts or shredded coconut.

6. **Coat with Powdered Sugar or Chocolate:**
   - Roll each alfajor in powdered sugar, pressing gently to ensure it adheres.
   - Optional: Melt chocolate and dip the alfajores, then sprinkle with cinnamon or coconut before the chocolate sets.

7. **Allow to Set:**
   - Let the alfajores sit for about 30 minutes to allow the flavors to meld together.

8. **Serve and Enjoy:**
   - Serve with a cup of coffee or tea. Store any leftovers in an airtight container at room temperature for up to a week.

### Tips:
- **Dough Handling:** Ensure the butter is softened for a smooth dough, and don't overwork the dough to maintain the cookies' tender texture.
- **Baking:** Keep an eye on the cookies after 15 minutes to prevent over-browning.
- **Filling Variations:** Experiment with different fillings like caramel or nuts to create unique flavor profiles.
- **Coating Options:** Try different coatings such as cinnamon or coconut for a varied texture and taste.

By following these steps, you'll create authentic Argentinian alfajores that capture the rich flavors and traditions of this beloved South American treat.\
"""
                    ),
                ],
                usage=Usage(requests=1, request_tokens=524, response_tokens=1590, total_tokens=2114),
                model_name='deepseek-r1-distill-llama-70b',
                timestamp=IsDatetime(),
                vendor_id=IsStr(),
            ),
        ]
    )


async def test_groq_model_thinking_part_iter(allow_model_requests: None, groq_api_key: str):
    m = GroqModel('deepseek-r1-distill-llama-70b', provider=GroqProvider(api_key=groq_api_key))
    settings = GroqModelSettings(groq_reasoning_format='raw')
    agent = Agent(m, instructions='You are a chef.', model_settings=settings)

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='I want a recipe to cook Uruguayan alfajores.') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert event_parts == snapshot(
        IsListOrTuple(
            positions={
                0: PartStartEvent(index=0, part=ThinkingPart(content='')),
                1: PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='\n')),
                2: PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Okay')),
                3: PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
                4: PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
                5: PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' need')),
                6: PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
                7: PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' come')),
                8: PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' up')),
                9: PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
                589: PartStartEvent(index=1, part=TextPart(content='**')),
                590: FinalResultEvent(tool_name=None, tool_call_id=None),
                591: PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Ur')),
                592: PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ugu')),
                593: PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ayan')),
                594: PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Alf')),
                595: PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='aj')),
                596: PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ores')),
                597: PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Recipe')),
            },
            length=996,
        )
    )
