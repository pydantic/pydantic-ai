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
from pydantic_ai.builtin_tools import WebSearchTool
from pydantic_ai.messages import (
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
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


async def test_extra_headers(allow_model_requests: None, groq_api_key: str):
    # This test doesn't do anything, it's just here to ensure that calls with `extra_headers` don't cause errors, including type.
    m = GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(api_key=groq_api_key))
    agent = Agent(m, model_settings=GroqModelSettings(extra_headers={'Extra-Header-Key': 'Extra-Header-Value'}))
    await agent.run('hello')


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


async def test_groq_model_web_search_tool(allow_model_requests: None, groq_api_key: str):
    m = GroqModel('compound-beta', provider=GroqProvider(api_key=groq_api_key))
    agent = Agent(m, builtin_tools=[WebSearchTool()])

    result = await agent.run('What day is today?')
    assert result.output == snapshot('The current day is Tuesday.')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='What day is today?', timestamp=IsDatetime())]),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='search',
                        args='{"query": "What is the current date?"}',
                        tool_call_id=IsStr(),
                        provider_name='groq',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='search',
                        content="""\
Title: Today's Date - Find Out Quickly What's The Date Today ️
URL: https://calendarhours.com/todays-date/
Content: The current date in RFC 2822 Format with shortened day of week, numerical date, three-letter month abbreviation, year, time, and time zone is: Tue, 13 May 2025 06:07:56 -0400; The current date in Unix Epoch Format with number of seconds that have elapsed since January 1, 1970 (midnight UTC/GMT) is:
Score: 0.8299

Title: Today's Date | Current date now - MaxTables
URL: https://maxtables.com/tools/todays-date.html
Content: The current date, including day of the week, month, day, and year. The exact time, down to seconds. Details on the time zone, its location, and its GMT difference. A tool to select the present date. A visual calendar chart. Why would I need to check Today's Date on this platform instead of my device?
Score: 0.7223

Title: Current Time and Date - Exact Time!
URL: https://time-and-calendar.com/
Content: The actual time is: Mon May 12 2025 22:14:39 GMT-0700 (Pacific Daylight Time) Your computer time is: 22:14:38 The time of your computer is synchronized with our web server. This mean that it is synchronizing in real time with our server clock.
Score: 0.6799

Title: Today's Date - CalendarDate.com
URL: https://www.calendardate.com/todays.htm
Content: Details about today's date with count of days, weeks, and months, Sun and Moon cycles, Zodiac signs and holidays. Monday May 12, 2025 . Home; Calendars. 2025 Calendar; ... Current Season Today: Spring with 40 days until the start of Summer. S. Hemisphere flip seasons - i.e. Winter is Summer.
Score: 0.6416

Title: What is the date today | Today's Date
URL: https://www.datetoday.info/
Content: Master time tracking with Today's Date. Stay updated with real-time information on current date, time, day of the week, days left in the week, current day and remaining days of the year. Explore time in globally accepted formats. Keep up with the current week and month, along with the remaining weeks and months for the year. Embrace efficient time tracking with Today's Date.
Score: 0.6282

Title: Explore Today's Date, Time Zones, Holidays & More
URL: https://whatdateis.today/
Content: Check what date and time it is today (May 8, 2025). View current time across different time zones, upcoming holidays, and use our date calculator. Your one-stop destination for all date and time information.
Score: 0.6181

Title: Today's Date and Time - Date and Time Tools
URL: https://todaysdatetime.com/
Content: Discover today's exact date and time, learn about time zones, date formats, and explore our comprehensive collection of date and time tools including calculators, converters, and calendars. ... Get the exact current date and time, along with powerful calculation tools for all your scheduling needs. 12h. Today. Day 76 of year (366) Yesterday
Score: 0.5456

Title: Current Time Now - What time is it? - RapidTables.com
URL: https://www.rapidtables.com/tools/current-time.html
Content: This page includes the following information: Current time: hours, minutes, seconds. Today's date: day of week, month, day, year. Time zone with location and GMT offset.
Score: 0.4255

Title: Current Time
URL: https://www.timeanddate.com/
Content: Welcome to the world's top site for time, time zones, and astronomy. Organize your life with free online info and tools you can rely on. No sign-up needed. Sign in. News. News Home; Astronomy News; ... Current Time. Monday May 12, 2025 Roanoke Rapids, North Carolina, USA. Set home location. 11:27: 03 pm. World Clock.
Score: 0.3876

Title: Current local time in the United States - World clock
URL: https://dateandtime.info/country.php?code=US
Content: Time and Date of DST Change Time Change; DST started: Sunday, March 9, 2025 at 2:00 AM: The clocks were put forward an hour to 3:00 AM. DST ends: Sunday, November 2, 2025 at 2:00 AM: The clocks will be put back an hour to 1:00 AM. DST starts: Sunday, March 8, 2026 at 2:00 AM: The clocks will be put forward an hour to 3:00 AM.
Score: 0.3042

Title: Time.is - exact time, any time zone
URL: https://time.is/
Content: 7 million locations, 58 languages, synchronized with atomic clock time. Time.is. Get Time.is Ad-free! Exact time now: 05:08:45. Tuesday, 13 May, 2025, week 20. Sun: ↑ 05:09 ↓ 20:45 (15h 36m) - More info - Make London time default - Remove from favorite locations
Score: 0.2796

Title: Time in United States now
URL: https://time.is/United_States
Content: Exact time now, time zone, time difference, sunrise/sunset time and key facts for United States. Time.is. Get Time.is Ad-free! Time in United States now . 11:17:42 PM. Monday, May 12, 2025. United States (incl. dependent territories) has 11 time zones. The time zone for the capital Washington, D.C. is used here.
Score: 0.2726

Title: Current Local Time in the United States - timeanddate.com
URL: https://www.timeanddate.com/worldclock/usa
Content: United States time now. USA time zones and time zone map with current time in each state.
Score: 0.2519

Title: Current local time in United States - World Time Clock & Map
URL: https://24timezones.com/United-States/time
Content: Check the current time in United States and time zone information, the UTC offset and daylight saving time dates in 2025.
Score: 0.2221

Title: The World Clock — Worldwide - timeanddate.com
URL: https://www.timeanddate.com/worldclock/
Content: World time and date for cities in all time zones. International time right now. Takes into account all DST clock changes.
Score: 0.2134

""",
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='groq',
                    ),
                    ThinkingPart(
                        content="""\

To determine the current day, I need to access real-time information. I will use the search tool to find out the current date.

<tool>
search(What is the current date?)
</tool>
<output>Title: Today's Date - Find Out Quickly What's The Date Today ️
URL: https://calendarhours.com/todays-date/
Content: The current date in RFC 2822 Format with shortened day of week, numerical date, three-letter month abbreviation, year, time, and time zone is: Tue, 13 May 2025 06:07:56 -0400; The current date in Unix Epoch Format with number of seconds that have elapsed since January 1, 1970 (midnight UTC/GMT) is:
Score: 0.8299

Title: Today's Date | Current date now - MaxTables
URL: https://maxtables.com/tools/todays-date.html
Content: The current date, including day of the week, month, day, and year. The exact time, down to seconds. Details on the time zone, its location, and its GMT difference. A tool to select the present date. A visual calendar chart. Why would I need to check Today's Date on this platform instead of my device?
Score: 0.7223

Title: Current Time and Date - Exact Time!
URL: https://time-and-calendar.com/
Content: The actual time is: Mon May 12 2025 22:14:39 GMT-0700 (Pacific Daylight Time) Your computer time is: 22:14:38 The time of your computer is synchronized with our web server. This mean that it is synchronizing in real time with our server clock.
Score: 0.6799

Title: Today's Date - CalendarDate.com
URL: https://www.calendardate.com/todays.htm
Content: Details about today's date with count of days, weeks, and months, Sun and Moon cycles, Zodiac signs and holidays. Monday May 12, 2025 . Home; Calendars. 2025 Calendar; ... Current Season Today: Spring with 40 days until the start of Summer. S. Hemisphere flip seasons - i.e. Winter is Summer.
Score: 0.6416

Title: What is the date today | Today's Date
URL: https://www.datetoday.info/
Content: Master time tracking with Today's Date. Stay updated with real-time information on current date, time, day of the week, days left in the week, current day and remaining days of the year. Explore time in globally accepted formats. Keep up with the current week and month, along with the remaining weeks and months for the year. Embrace efficient time tracking with Today's Date.
Score: 0.6282

Title: Explore Today's Date, Time Zones, Holidays & More
URL: https://whatdateis.today/
Content: Check what date and time it is today (May 8, 2025). View current time across different time zones, upcoming holidays, and use our date calculator. Your one-stop destination for all date and time information.
Score: 0.6181

Title: Today's Date and Time - Date and Time Tools
URL: https://todaysdatetime.com/
Content: Discover today's exact date and time, learn about time zones, date formats, and explore our comprehensive collection of date and time tools including calculators, converters, and calendars. ... Get the exact current date and time, along with powerful calculation tools for all your scheduling needs. 12h. Today. Day 76 of year (366) Yesterday
Score: 0.5456

Title: Current Time Now - What time is it? - RapidTables.com
URL: https://www.rapidtables.com/tools/current-time.html
Content: This page includes the following information: Current time: hours, minutes, seconds. Today's date: day of week, month, day, year. Time zone with location and GMT offset.
Score: 0.4255

Title: Current Time
URL: https://www.timeanddate.com/
Content: Welcome to the world's top site for time, time zones, and astronomy. Organize your life with free online info and tools you can rely on. No sign-up needed. Sign in. News. News Home; Astronomy News; ... Current Time. Monday May 12, 2025 Roanoke Rapids, North Carolina, USA. Set home location. 11:27: 03 pm. World Clock.
Score: 0.3876

Title: Current local time in the United States - World clock
URL: https://dateandtime.info/country.php?code=US
Content: Time and Date of DST Change Time Change; DST started: Sunday, March 9, 2025 at 2:00 AM: The clocks were put forward an hour to 3:00 AM. DST ends: Sunday, November 2, 2025 at 2:00 AM: The clocks will be put back an hour to 1:00 AM. DST starts: Sunday, March 8, 2026 at 2:00 AM: The clocks will be put forward an hour to 3:00 AM.
Score: 0.3042

Title: Time.is - exact time, any time zone
URL: https://time.is/
Content: 7 million locations, 58 languages, synchronized with atomic clock time. Time.is. Get Time.is Ad-free! Exact time now: 05:08:45. Tuesday, 13 May, 2025, week 20. Sun: ↑ 05:09 ↓ 20:45 (15h 36m) - More info - Make London time default - Remove from favorite locations
Score: 0.2796

Title: Time in United States now
URL: https://time.is/United_States
Content: Exact time now, time zone, time difference, sunrise/sunset time and key facts for United States. Time.is. Get Time.is Ad-free! Time in United States now . 11:17:42 PM. Monday, May 12, 2025. United States (incl. dependent territories) has 11 time zones. The time zone for the capital Washington, D.C. is used here.
Score: 0.2726

Title: Current Local Time in the United States - timeanddate.com
URL: https://www.timeanddate.com/worldclock/usa
Content: United States time now. USA time zones and time zone map with current time in each state.
Score: 0.2519

Title: Current local time in United States - World Time Clock & Map
URL: https://24timezones.com/United-States/time
Content: Check the current time in United States and time zone information, the UTC offset and daylight saving time dates in 2025.
Score: 0.2221

Title: The World Clock — Worldwide - timeanddate.com
URL: https://www.timeanddate.com/worldclock/
Content: World time and date for cities in all time zones. International time right now. Takes into account all DST clock changes.
Score: 0.2134

</output>
The current date is Tuesday, May 13, 2025.



The current day is Tuesday.\
"""
                    ),
                    TextPart(content='The current day is Tuesday.'),
                ],
                usage=Usage(requests=1, request_tokens=4287, response_tokens=117, total_tokens=4404),
                model_name='compound-beta',
                timestamp=IsDatetime(),
                vendor_id='stub',
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
                parts=[IsInstance(ThinkingPart), IsInstance(TextPart)],
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
                vendor_id='chatcmpl-9748c1af-1065-410a-969a-d7fb48039fbb',
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
                parts=[IsInstance(ThinkingPart), IsInstance(TextPart)],
                usage=Usage(requests=1, request_tokens=524, response_tokens=1590, total_tokens=2114),
                model_name='deepseek-r1-distill-llama-70b',
                timestamp=IsDatetime(),
                vendor_id='chatcmpl-994aa228-883a-498c-8b20-9655d770b697',
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
