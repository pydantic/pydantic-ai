from __future__ import annotations as _annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from typing import Any, Literal, cast
from unittest.mock import patch

import httpx
import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel
from typing_extensions import TypedDict

from pydantic_ai import (
    Agent,
    BinaryContent,
    BinaryImage,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    FinalResultEvent,
    ImageUrl,
    ModelAPIError,
    ModelHTTPError,
    ModelRequest,
    ModelResponse,
    ModelRetry,
    PartDeltaEvent,
    PartEndEvent,
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
from pydantic_ai.builtin_tools import WebSearchTool
from pydantic_ai.messages import (
    BuiltinToolCallEvent,  # pyright: ignore[reportDeprecated]
    BuiltinToolResultEvent,  # pyright: ignore[reportDeprecated]
)
from pydantic_ai.output import NativeOutput, PromptedOutput
from pydantic_ai.usage import RequestUsage, RunUsage

from ..conftest import IsDatetime, IsInstance, IsStr, raise_if_exception, try_import
from .mock_async_stream import MockAsyncStream

with try_import() as imports_successful:
    from groq import APIConnectionError, APIStatusError, AsyncGroq
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

    MockChatCompletion = chat.ChatCompletion | Exception
    MockChatCompletionChunk = chat.ChatCompletionChunk | Exception

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='groq not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
    pytest.mark.filterwarnings(
        'ignore:`BuiltinToolCallEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolCallPart` instead.:DeprecationWarning'
    ),
    pytest.mark.filterwarnings(
        'ignore:`BuiltinToolResultEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolReturnPart` instead.:DeprecationWarning'
    ),
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
    base_url: str = 'https://api.groq.com'

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
    assert result.usage() == snapshot(RunUsage(requests=1))

    # reset the index so we get the same response again
    mock_client.index = 0  # type: ignore

    result = await agent.run('hello', message_history=result.new_messages())
    assert result.output == 'world'
    assert result.usage() == snapshot(RunUsage(requests=1))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='world')],
                model_name='llama-3.3-70b-versatile-123',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_url='https://api.groq.com',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='world')],
                model_name='llama-3.3-70b-versatile-123',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_url='https://api.groq.com',
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
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsDatetime())],
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
                model_name='llama-3.3-70b-versatile-123',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_url='https://api.groq.com',
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
                        timestamp=IsDatetime(),
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
                    SystemPromptPart(content='this is the system prompt', timestamp=IsDatetime()),
                    UserPromptPart(content='Hello', timestamp=IsDatetime()),
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
                usage=RequestUsage(input_tokens=2, output_tokens=1),
                model_name='llama-3.3-70b-versatile-123',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_url='https://api.groq.com',
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
                        tool_name='get_location',
                        content='Wrong location, please try again',
                        tool_call_id='1',
                        timestamp=IsDatetime(),
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
                usage=RequestUsage(input_tokens=3, output_tokens=2),
                model_name='llama-3.3-70b-versatile-123',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_url='https://api.groq.com',
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
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='final response')],
                model_name='llama-3.3-70b-versatile-123',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_url='https://api.groq.com',
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
        assert [c async for c in result.stream_output(debounce_by=None)] == snapshot(
            ['hello ', 'hello world', 'hello world']
        )
        assert result.is_complete


async def test_stream_text_finish_reason(allow_model_requests: None):
    stream = text_chunk('hello '), text_chunk('world'), text_chunk('.', finish_reason='stop')
    mock_client = MockGroq.create_mock_stream(stream)
    m = GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(groq_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_output(debounce_by=None)] == snapshot(
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

    assert result.usage() == snapshot(RunUsage(requests=1))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"first": "One", "second": "Two"}',
                        tool_call_id=IsStr(),
                    )
                ],
                model_name='llama-3.3-70b-versatile',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_url='https://api.groq.com',
                provider_details={'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)},
                provider_response_id='x',
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
        assert [dict(c) async for c in result.stream_output(debounce_by=None)] == snapshot(
            [
                {'first': 'One'},
                {'first': 'One', 'second': 'Two'},
                {'first': 'One', 'second': 'Two'},
                {'first': 'One', 'second': 'Two'},
            ]
        )
        assert result.is_complete


async def test_no_delta(allow_model_requests: None):
    stream = chunk([]), text_chunk('hello '), text_chunk('world')
    mock_client = MockGroq.create_mock_stream(stream)
    m = GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(groq_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_output(debounce_by=None)] == snapshot(
            ['hello ', 'hello world', 'hello world']
        )
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
    m = GroqModel('meta-llama/llama-4-maverick-17b-128e-instruct', provider=GroqProvider(api_key=groq_api_key))
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
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_image', args='{}', tool_call_id='arq6emmq6')],
                usage=RequestUsage(input_tokens=712, output_tokens=20),
                model_name='meta-llama/llama-4-maverick-17b-128e-instruct',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_url='https://api.groq.com',
                provider_details={'finish_reason': 'tool_calls', 'timestamp': IsDatetime()},
                provider_response_id='chatcmpl-31dace36-574a-42ee-a89f-154b2881e090',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_image',
                        content='See file 241a70',
                        tool_call_id='arq6emmq6',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(content=['This is file 241a70:', IsInstance(BinaryImage)], timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='The fruit in the image is a kiwi.')],
                usage=RequestUsage(input_tokens=1501, output_tokens=11),
                model_name='meta-llama/llama-4-maverick-17b-128e-instruct',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_url='https://api.groq.com',
                provider_details={'finish_reason': 'stop', 'timestamp': IsDatetime()},
                provider_response_id='chatcmpl-5644262c-ce2b-40af-9408-21690b4619a8',
                finish_reason='stop',
                run_id=IsStr(),
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
    assert result.output == snapshot('The fruit depicted in the image is a kiwi.')


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


def test_model_connection_error(allow_model_requests: None) -> None:
    mock_client = MockGroq.create_mock(
        APIConnectionError(
            message='Connection to https://api.groq.com timed out',
            request=httpx.Request('POST', 'https://api.groq.com/v1/chat/completions'),
        )
    )
    m = GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(groq_client=mock_client))
    agent = Agent(m)
    with pytest.raises(ModelAPIError) as exc_info:
        agent.run_sync('hello')
    assert exc_info.value.model_name == 'llama-3.3-70b-versatile'
    assert 'Connection to https://api.groq.com timed out' in str(exc_info.value.message)


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
                timestamp=IsDatetime(),
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='The capital of France is Paris.')],
                usage=RequestUsage(input_tokens=48, output_tokens=8),
                model_name='llama-3.3-70b-versatile',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_url='https://api.groq.com',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2025, 4, 7, 16, 32, 53, tzinfo=timezone.utc),
                },
                provider_response_id='chatcmpl-7586b6a9-fb4b-4ec7-86a0-59f0a77844cf',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_groq_model_web_search_tool(allow_model_requests: None, groq_api_key: str):
    m = GroqModel('compound-beta', provider=GroqProvider(api_key=groq_api_key))
    agent = Agent(m, builtin_tools=[WebSearchTool()])

    result = await agent.run('What is the weather in San Francisco today?')
    assert result.output == snapshot("""\
The weather in San Francisco today, September 17, 2025, is partly cloudy with a temperature of 17°C (62.6°F) and a wind speed of 7.8 mph (12.6 kph) from the west. The humidity is 94%, and there is a 0% chance of precipitation. The UV index is 6.8, and the feels-like temperature is also 17°C (62.6°F). \n\

Additionally, the forecast for the day indicates that it will be a comfortable day with a high of 75°F (24°C) and a low of 59°F (15°C). There is a slight chance of rain and thunderstorms in the Bay Area due to the remnants of Tropical Storm Mario, but it is not expected to significantly impact San Francisco.

It's worth noting that the weather in San Francisco can be quite variable, and the temperature can drop significantly at night, so it's a good idea to dress in layers. Overall, it should be a pleasant day in San Francisco, with plenty of sunshine and mild temperatures.\
""")
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the weather in San Francisco today?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="""\

To find the current weather in San Francisco, I will use the search tool to look up this information.

<tool>
search(What is the weather in San Francisco today?)
</tool>
<output>Title: Weather in San Francisco
URL: https://www.weatherapi.com/
Content: {'location': {'name': 'San Francisco', 'region': 'California', 'country': 'United States of America', 'lat': 37.775, 'lon': -122.4183, 'tz_id': 'America/Los_Angeles', 'localtime_epoch': 1758143975, 'localtime': '2025-09-17 14:19'}, 'current': {'last_updated_epoch': 1758143700, 'last_updated': '2025-09-17 14:15', 'temp_c': 17.0, 'temp_f': 62.6, 'is_day': 1, 'condition': {'text': 'Partly cloudy', 'icon': '//cdn.weatherapi.com/weather/64x64/day/116.png', 'code': 1003}, 'wind_mph': 7.8, 'wind_kph': 12.6, 'wind_degree': 264, 'wind_dir': 'W', 'pressure_mb': 1015.0, 'pressure_in': 29.96, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 94, 'cloud': 50, 'feelslike_c': 17.0, 'feelslike_f': 62.6, 'windchill_c': 17.7, 'windchill_f': 63.9, 'heatindex_c': 17.7, 'heatindex_f': 63.9, 'dewpoint_c': 15.3, 'dewpoint_f': 59.6, 'vis_km': 13.0, 'vis_miles': 8.0, 'uv': 6.8, 'gust_mph': 14.4, 'gust_kph': 23.1}}
Score: 0.9842

Title: Wednesday, September 17, 2025. San Francisco, CA - Weather ...
URL: https://weathershogun.com/weather/usa/ca/san-francisco/480/september/2025-09-17
Content: San Francisco, California Weather: Wednesday, September 17, 2025. Day 75°. Night 59°. Precipitation 0 %. Wind 8 mph. UV Index (0 - 11+) 11
Score: 0.9597

Title: Find cheap flights from Milan (MXP) to San Francisco (SFO)
URL: https://www.aa.com/en-it/flights-from-milan-to-san-francisco
Content: Weather in San Francisco. Weather Unit: Weather unit option Celsius Selected ... 17/09/2025. \u200b. Thursday. overcast clouds. 18°C. 18/09/2025. \u200b. Friday. few
Score: 0.9083

Title: San Francisco weather in September 2025 | Weather25.com
URL: https://www.weather25.com/north-america/usa/california/san-francisco?page=month&month=September
Content: The temperatures in San Francisco in September are comfortable with low of 57°F and and high up to 77°F. There is little to no rain in San Francisco during
Score: 0.8854

Title: Bay Area basks in unseasonable heat with thunderstorms and ...
URL: https://www.cbsnews.com/sanfrancisco/news/bay-area-hot-weather-thunderstorms-fire-danger-dry-lightning/
Content: Carlos E. Castañeda. September 17, 2025 / 11:20 AM PDT / CBS San Francisco. Wednesday morning First Alert weather forecast with Jessica Burch - 9/17/25
Score: 0.8625

Title: Area Forecast Discussion - National Weather Service
URL: https://forecast.weather.gov/product.php?site=mtr&issuedby=MTR&product=AFD
Content: 067 FXUS66 KMTR 171648 AFDMTR Area Forecast Discussion National Weather Service San Francisco CA 948 AM PDT Wed Sep 17 2025 ...New UPDATE, FIRE WEATHER.
Score: 0.8163

Title: Weather in San Francisco in September 2025
URL: https://world-weather.info/forecast/usa/san_francisco/september-2025/
Content: Detailed ⚡ San Francisco Weather Forecast for September 2025 – day/night 🌡️ temperatures, precipitations – World-Weather.info.
Score: 0.7421

Title: The Fantasy Forecast: September skies with Maye weather
URL: https://dailycampus.com/2025/09/17/the-fantasy-forecast-september-skies-with-maye-weather/
Content: ... Wednesday, September 17, 2025 ... The Arizona Cardinals quarterback will face the San Francisco 49ers this Sunday in a battle for the West.
Score: 0.7171

Title: 60-Day Extended Weather Forecast for San Francisco, San ...
URL: https://www.almanac.com/weather/longrange/CA/San%20Francisco%2C%20San%20Francisco%20County
Content: Almanac Logo Wednesday, September 17, 2025 · Almanac.com. Weather · Long-Range; California. Toggle navigation. Gardening. All Gardening · Planting Calendar
Score: 0.6857

Title: San Francisco weather in September 2025 | California
URL: https://www.weather2travel.com/california/san-francisco/september/
Content: Daytime temperatures usually reach 22°C in San Francisco in September, falling to 13°C at night. How sunny is it in San Francisco in September? There are
Score: 0.6799

Title: Weather Forecast for San Francisco for Wednesday 17 September
URL: https://www.metcheck.com/WEATHER/dayforecast.asp?location=San%20Francisco&locationID=1628582&lat=-25.23078&lon=-57.57218&dateFor=17/09/2025
Content: Time, Weather, Temp, Feels, RainRisk, Amount, Cloud, Dir, Speed, Gust, RH, UV. 0:00, 23 °c, 25 °c, 0%, 0.0mm, 0%, 6mph, 24mph, 73%, 0.
Score: 0.6581

Title: Weather in San Francisco, California for September 2025
URL: https://www.easeweather.com/north-america/united-states/california/city-and-county-of-san-francisco/san-francisco/september
Content: In general, the average temperature in San Francisco at the beginning of September is 70 °F. As the month progressed, temperatures tended to moderately fall,
Score: 0.6533

Title: San Francisco September 2025 Historical Weather Data (California ...
URL: https://weatherspark.com/h/m/557/2025/9/Historical-Weather-in-September-2025-in-San-Francisco-California-United-States
Content: This report shows the past weather for San Francisco, providing a weather history for September 2025. It features all historical weather data series we have
Score: 0.5855

Title: Weather Forecast for Batey San Francisco for Wednesday 17 ...
URL: https://www.metcheck.com/WEATHER/dayforecast.asp?location=Batey%20San%20Francisco&locationID=511664&lat=18.62123&lon=-68.63688&dateFor=17/09/2025
Content: Time, Weather, Temp, Feels, RainRisk, Amount, Cloud, Dir, Speed, Gust, RH, UV. 0:00, 25 °c, 27 °c, 88%, 0.1mm, 0%, 4mph, 19mph, 91%, 0.
Score: 0.3891

Title: Wednesday morning First Alert weather forecast with Jessica Burch
URL: https://www.youtube.com/watch?v=fzAVNg32R2M
Content: Wednesday morning First Alert weather forecast with Jessica Burch - 9/17/25. 742 views · 2 hours ago ...more. KPIX | CBS NEWS BAY AREA. 452K.
Score: 0.2947

Title: Monthly Weather Forecast for San Francisco, CA
URL: https://weather.com/weather/monthly/l/69bedc6a5b6e977993fb3e5344e3c06d8bc36a1fb6754c3ddfb5310a3c6d6c87
Content: Considerable cloudiness. Low 56F. Winds WSW at 10 to 15 mph. Record Low52°.
Score: 0.2857

Title: Rain and thunderstorms coming to Bay Area - SFGATE
URL: https://www.sfgate.com/bayarea/article/thunder-rain-tropical-storm-mario-21053020.php
Content: San Francisco could be hit with rain and lightning thanks to the remnants of Tropical Storm Mario.
Score: 0.2418

</output>


Based on the search results, the current weather in San Francisco is partly cloudy with a temperature of 17°C (62.6°F). \n\

The weather in San Francisco today is partly cloudy with a high of 17°C (62.6°F).\
"""
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'What is the weather in San Francisco today?'},
                        tool_call_id=IsStr(),
                        provider_name='groq',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={
                            'images': None,
                            'results': [
                                {
                                    'content': "{'location': {'name': 'San Francisco', 'region': 'California', 'country': 'United States of America', 'lat': 37.775, 'lon': -122.4183, 'tz_id': 'America/Los_Angeles', 'localtime_epoch': 1758143975, 'localtime': '2025-09-17 14:19'}, 'current': {'last_updated_epoch': 1758143700, 'last_updated': '2025-09-17 14:15', 'temp_c': 17.0, 'temp_f': 62.6, 'is_day': 1, 'condition': {'text': 'Partly cloudy', 'icon': '//cdn.weatherapi.com/weather/64x64/day/116.png', 'code': 1003}, 'wind_mph': 7.8, 'wind_kph': 12.6, 'wind_degree': 264, 'wind_dir': 'W', 'pressure_mb': 1015.0, 'pressure_in': 29.96, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 94, 'cloud': 50, 'feelslike_c': 17.0, 'feelslike_f': 62.6, 'windchill_c': 17.7, 'windchill_f': 63.9, 'heatindex_c': 17.7, 'heatindex_f': 63.9, 'dewpoint_c': 15.3, 'dewpoint_f': 59.6, 'vis_km': 13.0, 'vis_miles': 8.0, 'uv': 6.8, 'gust_mph': 14.4, 'gust_kph': 23.1}}",
                                    'score': 0.9842367,
                                    'title': 'Weather in San Francisco',
                                    'url': 'https://www.weatherapi.com/',
                                },
                                {
                                    'content': 'San Francisco, California Weather: Wednesday, September 17, 2025. Day 75°. Night 59°. Precipitation 0 %. Wind 8 mph. UV Index (0 - 11+) 11',
                                    'score': 0.95967525,
                                    'title': 'Wednesday, September 17, 2025. San Francisco, CA - Weather ...',
                                    'url': 'https://weathershogun.com/weather/usa/ca/san-francisco/480/september/2025-09-17',
                                },
                                {
                                    'content': 'Weather in San Francisco. Weather Unit: Weather unit option Celsius Selected ... 17/09/2025. \u200b. Thursday. overcast clouds. 18°C. 18/09/2025. \u200b. Friday. few',
                                    'score': 0.90830135,
                                    'title': 'Find cheap flights from Milan (MXP) to San Francisco (SFO)',
                                    'url': 'https://www.aa.com/en-it/flights-from-milan-to-san-francisco',
                                },
                                {
                                    'content': 'The temperatures in San Francisco in September are comfortable with low of 57°F and and high up to 77°F. There is little to no rain in San Francisco during',
                                    'score': 0.885404,
                                    'title': 'San Francisco weather in September 2025 | Weather25.com',
                                    'url': 'https://www.weather25.com/north-america/usa/california/san-francisco?page=month&month=September',
                                },
                                {
                                    'content': 'Carlos E. Castañeda. September 17, 2025 / 11:20 AM PDT / CBS San Francisco. Wednesday morning First Alert weather forecast with Jessica Burch - 9/17/25',
                                    'score': 0.8624794,
                                    'title': 'Bay Area basks in unseasonable heat with thunderstorms and ...',
                                    'url': 'https://www.cbsnews.com/sanfrancisco/news/bay-area-hot-weather-thunderstorms-fire-danger-dry-lightning/',
                                },
                                {
                                    'content': '067 FXUS66 KMTR 171648 AFDMTR Area Forecast Discussion National Weather Service San Francisco CA 948 AM PDT Wed Sep 17 2025 ...New UPDATE, FIRE WEATHER.',
                                    'score': 0.81630427,
                                    'title': 'Area Forecast Discussion - National Weather Service',
                                    'url': 'https://forecast.weather.gov/product.php?site=mtr&issuedby=MTR&product=AFD',
                                },
                                {
                                    'content': 'Detailed ⚡ San Francisco Weather Forecast for September 2025 – day/night 🌡️ temperatures, precipitations – World-Weather.info.',
                                    'score': 0.7420672,
                                    'title': 'Weather in San Francisco in September 2025',
                                    'url': 'https://world-weather.info/forecast/usa/san_francisco/september-2025/',
                                },
                                {
                                    'content': '... Wednesday, September 17, 2025 ... The Arizona Cardinals quarterback will face the San Francisco 49ers this Sunday in a battle for the West.',
                                    'score': 0.7171114,
                                    'title': 'The Fantasy Forecast: September skies with Maye weather',
                                    'url': 'https://dailycampus.com/2025/09/17/the-fantasy-forecast-september-skies-with-maye-weather/',
                                },
                                {
                                    'content': 'Almanac Logo Wednesday, September 17, 2025 · Almanac.com. Weather · Long-Range; California. Toggle navigation. Gardening. All Gardening · Planting Calendar',
                                    'score': 0.68571854,
                                    'title': '60-Day Extended Weather Forecast for San Francisco, San ...',
                                    'url': 'https://www.almanac.com/weather/longrange/CA/San%20Francisco%2C%20San%20Francisco%20County',
                                },
                                {
                                    'content': 'Daytime temperatures usually reach 22°C in San Francisco in September, falling to 13°C at night. How sunny is it in San Francisco in September? There are',
                                    'score': 0.67988104,
                                    'title': 'San Francisco weather in September 2025 | California',
                                    'url': 'https://www.weather2travel.com/california/san-francisco/september/',
                                },
                                {
                                    'content': 'Time, Weather, Temp, Feels, RainRisk, Amount, Cloud, Dir, Speed, Gust, RH, UV. 0:00, 23 °c, 25 °c, 0%, 0.0mm, 0%, 6mph, 24mph, 73%, 0.',
                                    'score': 0.6580885,
                                    'title': 'Weather Forecast for San Francisco for Wednesday 17 September',
                                    'url': 'https://www.metcheck.com/WEATHER/dayforecast.asp?location=San%20Francisco&locationID=1628582&lat=-25.23078&lon=-57.57218&dateFor=17/09/2025',
                                },
                                {
                                    'content': 'In general, the average temperature in San Francisco at the beginning of September is 70 °F. As the month progressed, temperatures tended to moderately fall,',
                                    'score': 0.6533265,
                                    'title': 'Weather in San Francisco, California for September 2025',
                                    'url': 'https://www.easeweather.com/north-america/united-states/california/city-and-county-of-san-francisco/san-francisco/september',
                                },
                                {
                                    'content': 'This report shows the past weather for San Francisco, providing a weather history for September 2025. It features all historical weather data series we have',
                                    'score': 0.5855047,
                                    'title': 'San Francisco September 2025 Historical Weather Data (California ...',
                                    'url': 'https://weatherspark.com/h/m/557/2025/9/Historical-Weather-in-September-2025-in-San-Francisco-California-United-States',
                                },
                                {
                                    'content': 'Time, Weather, Temp, Feels, RainRisk, Amount, Cloud, Dir, Speed, Gust, RH, UV. 0:00, 25 °c, 27 °c, 88%, 0.1mm, 0%, 4mph, 19mph, 91%, 0.',
                                    'score': 0.38908273,
                                    'title': 'Weather Forecast for Batey San Francisco for Wednesday 17 ...',
                                    'url': 'https://www.metcheck.com/WEATHER/dayforecast.asp?location=Batey%20San%20Francisco&locationID=511664&lat=18.62123&lon=-68.63688&dateFor=17/09/2025',
                                },
                                {
                                    'content': 'Wednesday morning First Alert weather forecast with Jessica Burch - 9/17/25. 742 views · 2 hours ago ...more. KPIX | CBS NEWS BAY AREA. 452K.',
                                    'score': 0.29469728,
                                    'title': 'Wednesday morning First Alert weather forecast with Jessica Burch',
                                    'url': 'https://www.youtube.com/watch?v=fzAVNg32R2M',
                                },
                                {
                                    'content': 'Considerable cloudiness. Low 56F. Winds WSW at 10 to 15 mph. Record Low52°.',
                                    'score': 0.28572106,
                                    'title': 'Monthly Weather Forecast for San Francisco, CA',
                                    'url': 'https://weather.com/weather/monthly/l/69bedc6a5b6e977993fb3e5344e3c06d8bc36a1fb6754c3ddfb5310a3c6d6c87',
                                },
                                {
                                    'content': 'San Francisco could be hit with rain and lightning thanks to the remnants of Tropical Storm Mario.',
                                    'score': 0.24180745,
                                    'title': 'Rain and thunderstorms coming to Bay Area - SFGATE',
                                    'url': 'https://www.sfgate.com/bayarea/article/thunder-rain-tropical-storm-mario-21053020.php',
                                },
                            ],
                        },
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='groq',
                    ),
                    TextPart(
                        content="""\
The weather in San Francisco today, September 17, 2025, is partly cloudy with a temperature of 17°C (62.6°F) and a wind speed of 7.8 mph (12.6 kph) from the west. The humidity is 94%, and there is a 0% chance of precipitation. The UV index is 6.8, and the feels-like temperature is also 17°C (62.6°F). \n\

Additionally, the forecast for the day indicates that it will be a comfortable day with a high of 75°F (24°C) and a low of 59°F (15°C). There is a slight chance of rain and thunderstorms in the Bay Area due to the remnants of Tropical Storm Mario, but it is not expected to significantly impact San Francisco.

It's worth noting that the weather in San Francisco can be quite variable, and the temperature can drop significantly at night, so it's a good idea to dress in layers. Overall, it should be a pleasant day in San Francisco, with plenty of sunshine and mild temperatures.\
"""
                    ),
                ],
                usage=RequestUsage(input_tokens=5296, output_tokens=387),
                model_name='groq/compound',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_url='https://api.groq.com',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2025, 9, 17, 21, 14, 13, tzinfo=timezone.utc),
                },
                provider_response_id='stub',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_groq_model_web_search_tool_stream(allow_model_requests: None, groq_api_key: str):
    m = GroqModel('compound-beta', provider=GroqProvider(api_key=groq_api_key))
    agent = Agent(m, builtin_tools=[WebSearchTool()])

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='What is the weather in San Francisco today?') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert agent_run.result is not None
    messages = agent_run.result.all_messages()
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the weather in San Francisco today?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="""\
<think>
To find the current weather in San Francisco, I will use the search tool to look up this information.

<tool>
search(What is the weather in San Francisco today?)
"""
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'What is the weather in San Francisco today?'},
                        tool_call_id=IsStr(),
                        provider_name='groq',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={
                            'images': None,
                            'results': [
                                {
                                    'content': "{'location': {'name': 'San Francisco', 'region': 'California', 'country': 'United States of America', 'lat': 37.775, 'lon': -122.4183, 'tz_id': 'America/Los_Angeles', 'localtime_epoch': 1758144075, 'localtime': '2025-09-17 14:21'}, 'current': {'last_updated_epoch': 1758143700, 'last_updated': '2025-09-17 14:15', 'temp_c': 17.4, 'temp_f': 63.3, 'is_day': 1, 'condition': {'text': 'Partly cloudy', 'icon': '//cdn.weatherapi.com/weather/64x64/day/116.png', 'code': 1003}, 'wind_mph': 7.8, 'wind_kph': 12.6, 'wind_degree': 264, 'wind_dir': 'W', 'pressure_mb': 1014.0, 'pressure_in': 29.95, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 94, 'cloud': 75, 'feelslike_c': 17.4, 'feelslike_f': 63.3, 'windchill_c': 17.7, 'windchill_f': 63.9, 'heatindex_c': 17.7, 'heatindex_f': 63.9, 'dewpoint_c': 15.3, 'dewpoint_f': 59.6, 'vis_km': 13.0, 'vis_miles': 8.0, 'uv': 6.8, 'gust_mph': 14.4, 'gust_kph': 23.1}}",
                                    'score': 0.9655062,
                                    'title': 'Weather in San Francisco',
                                    'url': 'https://www.weatherapi.com/',
                                },
                                {
                                    'content': "Today's Weather - San Francisco, CA. September 17, 2025 10:00 AM. Exploratorium. 61°. Feels Like 61°. Hi 69°F Lo 56°F. Mostly Sunny.",
                                    'score': 0.9512194,
                                    'title': 'San Francisco, CA | Weather Forecasts Now, Live Radar Maps ...',
                                    'url': 'https://www.weatherbug.com/weather-forecast/now/san-francisco-ca-94103',
                                },
                                {
                                    'content': "access_time 10:56 AM PDT on September 17, 2025 (GMT -7) | Updated 10 seconds ago. 76° | 59°. 74 °F. like 75°. icon. Sunny. N. 0. Today's temperature is forecast",
                                    'score': 0.92715925,
                                    'title': 'San Francisco, CA Weather Conditions | Weather Underground',
                                    'url': 'https://www.wunderground.com/weather/us/ca/san-francisco',
                                },
                                {
                                    'content': 'Weather in San Francisco, California, USA ; Sep 17, 2025 at 8:56 am · 10 mi · 29.98 "Hg · 87% · 57 °F',
                                    'score': 0.9224337,
                                    'title': 'Weather for San Francisco, California, USA - Time and Date',
                                    'url': 'https://www.timeanddate.com/weather/usa/san-francisco',
                                },
                                {
                                    'content': '... Current time: 01:50 2025/09/17. Current Weather; Forecast; Sun and Moon. partly cloudy, 16 °C. Wind speed 22 km/h. Humidity, 90 %. Air pressure, 1014 hPa.',
                                    'score': 0.91175514,
                                    'title': 'San Francisco - 14-Day Forecast: Temperature, Wind & Radar',
                                    'url': 'https://www.ventusky.com/san-francisco',
                                },
                                {
                                    'content': '723 FXUS66 KMTR 171146 AFDMTR Area Forecast Discussion National Weather Service San Francisco ... Issued at 406 AM PDT Wed Sep 17 2025 (Today and tonight)',
                                    'score': 0.8014549,
                                    'title': 'Bay Area forecast discussion - National Weather Service',
                                    'url': 'https://forecast.weather.gov/product.php?format=ci&glossary=1&issuedby=mtr&product=afd&site=mtr&version=1',
                                },
                                {
                                    'content': 'Detailed ⚡ San Francisco Weather Forecast for September 2025 – day/night 🌡️ temperatures, precipitations – World-Weather.info.',
                                    'score': 0.7646988,
                                    'title': 'Weather in San Francisco in September 2025',
                                    'url': 'https://world-weather.info/forecast/usa/san_francisco/september-2025/',
                                },
                                {
                                    'content': 'Full weather forecast for San Francisco in September 2025. Check the temperatures, chance of rain and more in San Francisco during September.',
                                    'score': 0.7192461,
                                    'title': 'San Francisco weather in September 2025 | Weather25.com',
                                    'url': 'https://www.weather25.com/north-america/usa/california/san-francisco?page=month&month=September',
                                },
                                {
                                    'content': '10-Day Weather Forecast ; Today. 9/17. 76° · Partly sunny ; Thu. 9/18. 68° · Rather cloudy ; Fri. 9/19. 73° · Partly sunny and pleasant ; Sat. 9/20. 71° · Mostly sunny',
                                    'score': 0.68318754,
                                    'title': 'San Francisco, CA Weather Forecast - AccuWeather',
                                    'url': 'https://www.accuweather.com/en/us/san-francisco/94103/weather-forecast/347629',
                                },
                                {
                                    'content': 'We have one more day of hot weather away from the coast today. A dense fog ... 2025 ABC, Inc., KGO-TV San Francisco. All Rights Reserved.',
                                    'score': 0.6164054,
                                    'title': 'AccuWeather Forecast: 1 more day of hot temperatures away from ...',
                                    'url': 'https://abc7news.com/post/weather-bay-area-forecast-temperatures/39468/',
                                },
                                {
                                    'content': 'Wednesday morning First Alert weather forecast with Jessica Burch - 9/17/25 ... National - Current Temperatures · National - First Alert Doppler. Latest',
                                    'score': 0.6010557,
                                    'title': 'San Francisco Bay Area weather and First Alert Weather forecasts',
                                    'url': 'https://www.cbsnews.com/sanfrancisco/weather/',
                                },
                                {
                                    'content': '10 Day Weather-San Francisco, CA. As of 2:31 pm PDT. Today. 67°/58°. 2%. Day. 67°. 2%. W 17 mph. Plentiful sunshine. High 67F. Winds W at 10 to 20 mph.',
                                    'score': 0.52290934,
                                    'title': '10-Day Weather Forecast for San Francisco, CA',
                                    'url': 'https://weather.com/weather/tenday/l/USCA0987:1:US',
                                },
                                {
                                    'content': '10 Day Weather-San Francisco, CA. As of 5:34 pm PDT. Tonight. --/58°. 18%. Night. 58°. 18%. W 15 mph. Partly cloudy early with increasing clouds overnight.',
                                    'score': 0.48221022,
                                    'title': '10-Day Weather Forecast for San Francisco, CA',
                                    'url': 'https://weather.com/weather/tenday/l/94112:4:US',
                                },
                                {
                                    'content': 'Night Sky · TodayHourly14 DaysPastClimate. Currently: 61 °F. Passing clouds. (Weather station: San Francisco International Airport, USA). See more current',
                                    'score': 0.42419788,
                                    'title': 'Past Weather in San Francisco, California, USA - Time and Date',
                                    'url': 'https://www.timeanddate.com/weather/usa/san-francisco/historic',
                                },
                                {
                                    'content': 'Considerable cloudiness. Low 56F. Winds WSW at 10 to 15 mph. Record Low52°.',
                                    'score': 0.327884,
                                    'title': 'Monthly Weather Forecast for San Francisco, CA',
                                    'url': 'https://weather.com/weather/monthly/l/69bedc6a5b6e977993fb3e5344e3c06d8bc36a1fb6754c3ddfb5310a3c6d6c87',
                                },
                                {
                                    'content': 'San Francisco Weather Forecasts. Weather Underground provides local & long-range weather ... Hourly Forecast for Today, Wednesday 09/17Hourly for Today, Wed 09/17.',
                                    'score': 0.26997215,
                                    'title': 'San Francisco, CA Hourly Weather Forecast - Weather Underground',
                                    'url': 'https://www.wunderground.com/hourly/us/ca/san-francisco',
                                },
                            ],
                        },
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='groq',
                    ),
                    ThinkingPart(content=IsStr()),
                    TextPart(
                        content='The weather in San Francisco today is partly cloudy with a temperature of 61°F (17°C) and high humidity. The current conditions include a wind speed of around 7-22 km/h and a humidity level of 90-94%.'
                    ),
                ],
                usage=RequestUsage(input_tokens=5003, output_tokens=359),
                model_name='groq/compound',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_url='https://api.groq.com',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2025, 9, 17, 21, 20, 46, tzinfo=timezone.utc),
                },
                provider_response_id='stub',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    assert event_parts == snapshot(
        [
            PartStartEvent(index=0, part=ThinkingPart(content='<th')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ink')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='>\n')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='To')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' find')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' current')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' weather')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' San')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Francisco')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' will')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' search')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' tool')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' look')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' up')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' this')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' information')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='<')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='tool')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='>\n')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='search')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='(')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='What')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' weather')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' San')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Francisco')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' today')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='?)\n')),
            PartEndEvent(
                index=0,
                part=ThinkingPart(
                    content="""\
<think>
To find the current weather in San Francisco, I will use the search tool to look up this information.

<tool>
search(What is the weather in San Francisco today?)
"""
                ),
                next_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=1,
                part=BuiltinToolCallPart(
                    tool_name='web_search',
                    args={'query': 'What is the weather in San Francisco today?'},
                    tool_call_id=IsStr(),
                    provider_name='groq',
                ),
                previous_part_kind='thinking',
            ),
            PartEndEvent(
                index=1,
                part=BuiltinToolCallPart(
                    tool_name='web_search',
                    args={'query': 'What is the weather in San Francisco today?'},
                    tool_call_id=IsStr(),
                    provider_name='groq',
                ),
                next_part_kind='builtin-tool-return',
            ),
            PartStartEvent(
                index=2,
                part=BuiltinToolReturnPart(
                    tool_name='web_search',
                    content={
                        'images': None,
                        'results': [
                            {
                                'content': "{'location': {'name': 'San Francisco', 'region': 'California', 'country': 'United States of America', 'lat': 37.775, 'lon': -122.4183, 'tz_id': 'America/Los_Angeles', 'localtime_epoch': 1758144075, 'localtime': '2025-09-17 14:21'}, 'current': {'last_updated_epoch': 1758143700, 'last_updated': '2025-09-17 14:15', 'temp_c': 17.4, 'temp_f': 63.3, 'is_day': 1, 'condition': {'text': 'Partly cloudy', 'icon': '//cdn.weatherapi.com/weather/64x64/day/116.png', 'code': 1003}, 'wind_mph': 7.8, 'wind_kph': 12.6, 'wind_degree': 264, 'wind_dir': 'W', 'pressure_mb': 1014.0, 'pressure_in': 29.95, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 94, 'cloud': 75, 'feelslike_c': 17.4, 'feelslike_f': 63.3, 'windchill_c': 17.7, 'windchill_f': 63.9, 'heatindex_c': 17.7, 'heatindex_f': 63.9, 'dewpoint_c': 15.3, 'dewpoint_f': 59.6, 'vis_km': 13.0, 'vis_miles': 8.0, 'uv': 6.8, 'gust_mph': 14.4, 'gust_kph': 23.1}}",
                                'score': 0.9655062,
                                'title': 'Weather in San Francisco',
                                'url': 'https://www.weatherapi.com/',
                            },
                            {
                                'content': "Today's Weather - San Francisco, CA. September 17, 2025 10:00 AM. Exploratorium. 61°. Feels Like 61°. Hi 69°F Lo 56°F. Mostly Sunny.",
                                'score': 0.9512194,
                                'title': 'San Francisco, CA | Weather Forecasts Now, Live Radar Maps ...',
                                'url': 'https://www.weatherbug.com/weather-forecast/now/san-francisco-ca-94103',
                            },
                            {
                                'content': "access_time 10:56 AM PDT on September 17, 2025 (GMT -7) | Updated 10 seconds ago. 76° | 59°. 74 °F. like 75°. icon. Sunny. N. 0. Today's temperature is forecast",
                                'score': 0.92715925,
                                'title': 'San Francisco, CA Weather Conditions | Weather Underground',
                                'url': 'https://www.wunderground.com/weather/us/ca/san-francisco',
                            },
                            {
                                'content': 'Weather in San Francisco, California, USA ; Sep 17, 2025 at 8:56 am · 10 mi · 29.98 "Hg · 87% · 57 °F',
                                'score': 0.9224337,
                                'title': 'Weather for San Francisco, California, USA - Time and Date',
                                'url': 'https://www.timeanddate.com/weather/usa/san-francisco',
                            },
                            {
                                'content': '... Current time: 01:50 2025/09/17. Current Weather; Forecast; Sun and Moon. partly cloudy, 16 °C. Wind speed 22 km/h. Humidity, 90 %. Air pressure, 1014 hPa.',
                                'score': 0.91175514,
                                'title': 'San Francisco - 14-Day Forecast: Temperature, Wind & Radar',
                                'url': 'https://www.ventusky.com/san-francisco',
                            },
                            {
                                'content': '723 FXUS66 KMTR 171146 AFDMTR Area Forecast Discussion National Weather Service San Francisco ... Issued at 406 AM PDT Wed Sep 17 2025 (Today and tonight)',
                                'score': 0.8014549,
                                'title': 'Bay Area forecast discussion - National Weather Service',
                                'url': 'https://forecast.weather.gov/product.php?format=ci&glossary=1&issuedby=mtr&product=afd&site=mtr&version=1',
                            },
                            {
                                'content': 'Detailed ⚡ San Francisco Weather Forecast for September 2025 – day/night 🌡️ temperatures, precipitations – World-Weather.info.',
                                'score': 0.7646988,
                                'title': 'Weather in San Francisco in September 2025',
                                'url': 'https://world-weather.info/forecast/usa/san_francisco/september-2025/',
                            },
                            {
                                'content': 'Full weather forecast for San Francisco in September 2025. Check the temperatures, chance of rain and more in San Francisco during September.',
                                'score': 0.7192461,
                                'title': 'San Francisco weather in September 2025 | Weather25.com',
                                'url': 'https://www.weather25.com/north-america/usa/california/san-francisco?page=month&month=September',
                            },
                            {
                                'content': '10-Day Weather Forecast ; Today. 9/17. 76° · Partly sunny ; Thu. 9/18. 68° · Rather cloudy ; Fri. 9/19. 73° · Partly sunny and pleasant ; Sat. 9/20. 71° · Mostly sunny',
                                'score': 0.68318754,
                                'title': 'San Francisco, CA Weather Forecast - AccuWeather',
                                'url': 'https://www.accuweather.com/en/us/san-francisco/94103/weather-forecast/347629',
                            },
                            {
                                'content': 'We have one more day of hot weather away from the coast today. A dense fog ... 2025 ABC, Inc., KGO-TV San Francisco. All Rights Reserved.',
                                'score': 0.6164054,
                                'title': 'AccuWeather Forecast: 1 more day of hot temperatures away from ...',
                                'url': 'https://abc7news.com/post/weather-bay-area-forecast-temperatures/39468/',
                            },
                            {
                                'content': 'Wednesday morning First Alert weather forecast with Jessica Burch - 9/17/25 ... National - Current Temperatures · National - First Alert Doppler. Latest',
                                'score': 0.6010557,
                                'title': 'San Francisco Bay Area weather and First Alert Weather forecasts',
                                'url': 'https://www.cbsnews.com/sanfrancisco/weather/',
                            },
                            {
                                'content': '10 Day Weather-San Francisco, CA. As of 2:31 pm PDT. Today. 67°/58°. 2%. Day. 67°. 2%. W 17 mph. Plentiful sunshine. High 67F. Winds W at 10 to 20 mph.',
                                'score': 0.52290934,
                                'title': '10-Day Weather Forecast for San Francisco, CA',
                                'url': 'https://weather.com/weather/tenday/l/USCA0987:1:US',
                            },
                            {
                                'content': '10 Day Weather-San Francisco, CA. As of 5:34 pm PDT. Tonight. --/58°. 18%. Night. 58°. 18%. W 15 mph. Partly cloudy early with increasing clouds overnight.',
                                'score': 0.48221022,
                                'title': '10-Day Weather Forecast for San Francisco, CA',
                                'url': 'https://weather.com/weather/tenday/l/94112:4:US',
                            },
                            {
                                'content': 'Night Sky · TodayHourly14 DaysPastClimate. Currently: 61 °F. Passing clouds. (Weather station: San Francisco International Airport, USA). See more current',
                                'score': 0.42419788,
                                'title': 'Past Weather in San Francisco, California, USA - Time and Date',
                                'url': 'https://www.timeanddate.com/weather/usa/san-francisco/historic',
                            },
                            {
                                'content': 'Considerable cloudiness. Low 56F. Winds WSW at 10 to 15 mph. Record Low52°.',
                                'score': 0.327884,
                                'title': 'Monthly Weather Forecast for San Francisco, CA',
                                'url': 'https://weather.com/weather/monthly/l/69bedc6a5b6e977993fb3e5344e3c06d8bc36a1fb6754c3ddfb5310a3c6d6c87',
                            },
                            {
                                'content': 'San Francisco Weather Forecasts. Weather Underground provides local & long-range weather ... Hourly Forecast for Today, Wednesday 09/17Hourly for Today, Wed 09/17.',
                                'score': 0.26997215,
                                'title': 'San Francisco, CA Hourly Weather Forecast - Weather Underground',
                                'url': 'https://www.wunderground.com/hourly/us/ca/san-francisco',
                            },
                        ],
                    },
                    tool_call_id=IsStr(),
                    timestamp=IsDatetime(),
                    provider_name='groq',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=3,
                part=ThinkingPart(
                    content="""\
</tool>
<output>Title: Weather in San Francisco
URL: https://www.weatherapi.com/
Content: {'location': {'name': 'San Francisco', 'region': 'California', 'country': 'United States of America', 'lat': 37.775, 'lon': -122.4183, 'tz_id': 'America/Los_Angeles', 'localtime_epoch': 1758144075, 'localtime': '2025-09-17 14:21'}, 'current': {'last_updated_epoch': 1758143700, 'last_updated': '2025-09-17 14:15', 'temp_c': 17.4, 'temp_f': 63.3, 'is_day': 1, 'condition': {'text': 'Partly cloudy', 'icon': '//cdn.weatherapi.com/weather/64x64/day/116.png', 'code': 1003}, 'wind_mph': 7.8, 'wind_kph': 12.6, 'wind_degree': 264, 'wind_dir': 'W', 'pressure_mb': 1014.0, 'pressure_in': 29.95, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 94, 'cloud': 75, 'feelslike_c': 17.4, 'feelslike_f': 63.3, 'windchill_c': 17.7, 'windchill_f': 63.9, 'heatindex_c': 17.7, 'heatindex_f': 63.9, 'dewpoint_c': 15.3, 'dewpoint_f': 59.6, 'vis_km': 13.0, 'vis_miles': 8.0, 'uv': 6.8, 'gust_mph': 14.4, 'gust_kph': 23.1}}
Score: 0.9655

Title: San Francisco, CA | Weather Forecasts Now, Live Radar Maps ...
URL: https://www.weatherbug.com/weather-forecast/now/san-francisco-ca-94103
Content: Today's Weather - San Francisco, CA. September 17, 2025 10:00 AM. Exploratorium. 61°. Feels Like 61°. Hi 69°F Lo 56°F. Mostly Sunny.
Score: 0.9512

Title: San Francisco, CA Weather Conditions | Weather Underground
URL: https://www.wunderground.com/weather/us/ca/san-francisco
Content: access_time 10:56 AM PDT on September 17, 2025 (GMT -7) | Updated 10 seconds ago. 76° | 59°. 74 °F. like 75°. icon. Sunny. N. 0. Today's temperature is forecast
Score: 0.9272

Title: Weather for San Francisco, California, USA - Time and Date
URL: https://www.timeanddate.com/weather/usa/san-francisco
Content: Weather in San Francisco, California, USA ; Sep 17, 2025 at 8:56 am · 10 mi · 29.98 "Hg · 87% · 57 °F
Score: 0.9224

Title: San Francisco - 14-Day Forecast: Temperature, Wind & Radar
URL: https://www.ventusky.com/san-francisco
Content: ... Current time: 01:50 2025/09/17. Current Weather; Forecast; Sun and Moon. partly cloudy, 16 °C. Wind speed 22 km/h. Humidity, 90 %. Air pressure, 1014 hPa.
Score: 0.9118

Title: Bay Area forecast discussion - National Weather Service
URL: https://forecast.weather.gov/product.php?format=ci&glossary=1&issuedby=mtr&product=afd&site=mtr&version=1
Content: 723 FXUS66 KMTR 171146 AFDMTR Area Forecast Discussion National Weather Service San Francisco ... Issued at 406 AM PDT Wed Sep 17 2025 (Today and tonight)
Score: 0.8015

Title: Weather in San Francisco in September 2025
URL: https://world-weather.info/forecast/usa/san_francisco/september-2025/
Content: Detailed ⚡ San Francisco Weather Forecast for September 2025 – day/night 🌡️ temperatures, precipitations – World-Weather.info.
Score: 0.7647

Title: San Francisco weather in September 2025 | Weather25.com
URL: https://www.weather25.com/north-america/usa/california/san-francisco?page=month&month=September
Content: Full weather forecast for San Francisco in September 2025. Check the temperatures, chance of rain and more in San Francisco during September.
Score: 0.7192

Title: San Francisco, CA Weather Forecast - AccuWeather
URL: https://www.accuweather.com/en/us/san-francisco/94103/weather-forecast/347629
Content: 10-Day Weather Forecast ; Today. 9/17. 76° · Partly sunny ; Thu. 9/18. 68° · Rather cloudy ; Fri. 9/19. 73° · Partly sunny and pleasant ; Sat. 9/20. 71° · Mostly sunny
Score: 0.6832

Title: AccuWeather Forecast: 1 more day of hot temperatures away from ...
URL: https://abc7news.com/post/weather-bay-area-forecast-temperatures/39468/
Content: We have one more day of hot weather away from the coast today. A dense fog ... 2025 ABC, Inc., KGO-TV San Francisco. All Rights Reserved.
Score: 0.6164

Title: San Francisco Bay Area weather and First Alert Weather forecasts
URL: https://www.cbsnews.com/sanfrancisco/weather/
Content: Wednesday morning First Alert weather forecast with Jessica Burch - 9/17/25 ... National - Current Temperatures · National - First Alert Doppler. Latest
Score: 0.6011

Title: 10-Day Weather Forecast for San Francisco, CA
URL: https://weather.com/weather/tenday/l/USCA0987:1:US
Content: 10 Day Weather-San Francisco, CA. As of 2:31 pm PDT. Today. 67°/58°. 2%. Day. 67°. 2%. W 17 mph. Plentiful sunshine. High 67F. Winds W at 10 to 20 mph.
Score: 0.5229

Title: 10-Day Weather Forecast for San Francisco, CA
URL: https://weather.com/weather/tenday/l/94112:4:US
Content: 10 Day Weather-San Francisco, CA. As of 5:34 pm PDT. Tonight. --/58°. 18%. Night. 58°. 18%. W 15 mph. Partly cloudy early with increasing clouds overnight.
Score: 0.4822

Title: Past Weather in San Francisco, California, USA - Time and Date
URL: https://www.timeanddate.com/weather/usa/san-francisco/historic
Content: Night Sky · TodayHourly14 DaysPastClimate. Currently: 61 °F. Passing clouds. (Weather station: San Francisco International Airport, USA). See more current
Score: 0.4242

Title: Monthly Weather Forecast for San Francisco, CA
URL: https://weather.com/weather/monthly/l/69bedc6a5b6e977993fb3e5344e3c06d8bc36a1fb6754c3ddfb5310a3c6d6c87
Content: Considerable cloudiness. Low 56F. Winds WSW at 10 to 15 mph. Record Low52°.
Score: 0.3279

Title: San Francisco, CA Hourly Weather Forecast - Weather Underground
URL: https://www.wunderground.com/hourly/us/ca/san-francisco
Content: San Francisco Weather Forecasts. Weather Underground provides local & long-range weather ... Hourly Forecast for Today, Wednesday 09/17Hourly for Today, Wed 09/17.
Score: 0.2700

</output>
"""
                ),
                previous_part_kind='builtin-tool-return',
            ),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta='</')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta='think')),
            PartDeltaEvent(
                index=3,
                delta=ThinkingPartDelta(
                    content_delta="""\
>

"""
                ),
            ),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta='Based')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' on')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' search')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' results')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' can')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' see')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' that')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' current')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' weather')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' San')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' Francisco')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' as')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' follows')),
            PartDeltaEvent(
                index=3,
                delta=ThinkingPartDelta(
                    content_delta="""\
:

"""
                ),
            ),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta='-')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' The')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' temperature')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' around')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta='61')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta='°F')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta='63')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta='°F')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' (')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta='17')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta='°C')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=').\n')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta='-')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' It')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' partly')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' cloudy')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' mostly')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' sunny')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta='-')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' The')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' humidity')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' around')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta='90')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta='-')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta='94')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta='%.\n')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta='-')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' The')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' wind')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' speed')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' around')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta='7')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta='-')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta='22')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' km')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta='/h')),
            PartDeltaEvent(
                index=3,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta='So')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' current')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' weather')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' San')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' Francisco')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' partly')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' cloudy')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' temperature')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta='61')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta='°F')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' (')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta='17')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta='°C')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=')')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' high')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' humidity')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(
                index=3,
                delta=ThinkingPartDelta(
                    content_delta="""\
 \n\

"""
                ),
            ),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta='Now')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' will')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' provide')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' final')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' answer')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' user')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(
                index=3,
                delta=ThinkingPartDelta(
                    content_delta="""\
 \n\

"""
                ),
            ),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta='The')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' weather')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' San')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' Francisco')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' today')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' partly')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' cloudy')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' temperature')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta='61')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta='°F')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' (')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta='17')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta='°C')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=')')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' high')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta=' humidity')),
            PartDeltaEvent(index=3, delta=ThinkingPartDelta(content_delta='.')),
            PartEndEvent(
                index=3,
                part=ThinkingPart(
                    content="""\
</tool>
<output>Title: Weather in San Francisco
URL: https://www.weatherapi.com/
Content: {'location': {'name': 'San Francisco', 'region': 'California', 'country': 'United States of America', 'lat': 37.775, 'lon': -122.4183, 'tz_id': 'America/Los_Angeles', 'localtime_epoch': 1758144075, 'localtime': '2025-09-17 14:21'}, 'current': {'last_updated_epoch': 1758143700, 'last_updated': '2025-09-17 14:15', 'temp_c': 17.4, 'temp_f': 63.3, 'is_day': 1, 'condition': {'text': 'Partly cloudy', 'icon': '//cdn.weatherapi.com/weather/64x64/day/116.png', 'code': 1003}, 'wind_mph': 7.8, 'wind_kph': 12.6, 'wind_degree': 264, 'wind_dir': 'W', 'pressure_mb': 1014.0, 'pressure_in': 29.95, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 94, 'cloud': 75, 'feelslike_c': 17.4, 'feelslike_f': 63.3, 'windchill_c': 17.7, 'windchill_f': 63.9, 'heatindex_c': 17.7, 'heatindex_f': 63.9, 'dewpoint_c': 15.3, 'dewpoint_f': 59.6, 'vis_km': 13.0, 'vis_miles': 8.0, 'uv': 6.8, 'gust_mph': 14.4, 'gust_kph': 23.1}}
Score: 0.9655

Title: San Francisco, CA | Weather Forecasts Now, Live Radar Maps ...
URL: https://www.weatherbug.com/weather-forecast/now/san-francisco-ca-94103
Content: Today's Weather - San Francisco, CA. September 17, 2025 10:00 AM. Exploratorium. 61°. Feels Like 61°. Hi 69°F Lo 56°F. Mostly Sunny.
Score: 0.9512

Title: San Francisco, CA Weather Conditions | Weather Underground
URL: https://www.wunderground.com/weather/us/ca/san-francisco
Content: access_time 10:56 AM PDT on September 17, 2025 (GMT -7) | Updated 10 seconds ago. 76° | 59°. 74 °F. like 75°. icon. Sunny. N. 0. Today's temperature is forecast
Score: 0.9272

Title: Weather for San Francisco, California, USA - Time and Date
URL: https://www.timeanddate.com/weather/usa/san-francisco
Content: Weather in San Francisco, California, USA ; Sep 17, 2025 at 8:56 am · 10 mi · 29.98 "Hg · 87% · 57 °F
Score: 0.9224

Title: San Francisco - 14-Day Forecast: Temperature, Wind & Radar
URL: https://www.ventusky.com/san-francisco
Content: ... Current time: 01:50 2025/09/17. Current Weather; Forecast; Sun and Moon. partly cloudy, 16 °C. Wind speed 22 km/h. Humidity, 90 %. Air pressure, 1014 hPa.
Score: 0.9118

Title: Bay Area forecast discussion - National Weather Service
URL: https://forecast.weather.gov/product.php?format=ci&glossary=1&issuedby=mtr&product=afd&site=mtr&version=1
Content: 723 FXUS66 KMTR 171146 AFDMTR Area Forecast Discussion National Weather Service San Francisco ... Issued at 406 AM PDT Wed Sep 17 2025 (Today and tonight)
Score: 0.8015

Title: Weather in San Francisco in September 2025
URL: https://world-weather.info/forecast/usa/san_francisco/september-2025/
Content: Detailed ⚡ San Francisco Weather Forecast for September 2025 – day/night 🌡️ temperatures, precipitations – World-Weather.info.
Score: 0.7647

Title: San Francisco weather in September 2025 | Weather25.com
URL: https://www.weather25.com/north-america/usa/california/san-francisco?page=month&month=September
Content: Full weather forecast for San Francisco in September 2025. Check the temperatures, chance of rain and more in San Francisco during September.
Score: 0.7192

Title: San Francisco, CA Weather Forecast - AccuWeather
URL: https://www.accuweather.com/en/us/san-francisco/94103/weather-forecast/347629
Content: 10-Day Weather Forecast ; Today. 9/17. 76° · Partly sunny ; Thu. 9/18. 68° · Rather cloudy ; Fri. 9/19. 73° · Partly sunny and pleasant ; Sat. 9/20. 71° · Mostly sunny
Score: 0.6832

Title: AccuWeather Forecast: 1 more day of hot temperatures away from ...
URL: https://abc7news.com/post/weather-bay-area-forecast-temperatures/39468/
Content: We have one more day of hot weather away from the coast today. A dense fog ... 2025 ABC, Inc., KGO-TV San Francisco. All Rights Reserved.
Score: 0.6164

Title: San Francisco Bay Area weather and First Alert Weather forecasts
URL: https://www.cbsnews.com/sanfrancisco/weather/
Content: Wednesday morning First Alert weather forecast with Jessica Burch - 9/17/25 ... National - Current Temperatures · National - First Alert Doppler. Latest
Score: 0.6011

Title: 10-Day Weather Forecast for San Francisco, CA
URL: https://weather.com/weather/tenday/l/USCA0987:1:US
Content: 10 Day Weather-San Francisco, CA. As of 2:31 pm PDT. Today. 67°/58°. 2%. Day. 67°. 2%. W 17 mph. Plentiful sunshine. High 67F. Winds W at 10 to 20 mph.
Score: 0.5229

Title: 10-Day Weather Forecast for San Francisco, CA
URL: https://weather.com/weather/tenday/l/94112:4:US
Content: 10 Day Weather-San Francisco, CA. As of 5:34 pm PDT. Tonight. --/58°. 18%. Night. 58°. 18%. W 15 mph. Partly cloudy early with increasing clouds overnight.
Score: 0.4822

Title: Past Weather in San Francisco, California, USA - Time and Date
URL: https://www.timeanddate.com/weather/usa/san-francisco/historic
Content: Night Sky · TodayHourly14 DaysPastClimate. Currently: 61 °F. Passing clouds. (Weather station: San Francisco International Airport, USA). See more current
Score: 0.4242

Title: Monthly Weather Forecast for San Francisco, CA
URL: https://weather.com/weather/monthly/l/69bedc6a5b6e977993fb3e5344e3c06d8bc36a1fb6754c3ddfb5310a3c6d6c87
Content: Considerable cloudiness. Low 56F. Winds WSW at 10 to 15 mph. Record Low52°.
Score: 0.3279

Title: San Francisco, CA Hourly Weather Forecast - Weather Underground
URL: https://www.wunderground.com/hourly/us/ca/san-francisco
Content: San Francisco Weather Forecasts. Weather Underground provides local & long-range weather ... Hourly Forecast for Today, Wednesday 09/17Hourly for Today, Wed 09/17.
Score: 0.2700

</output>
</think>

Based on the search results, I can see that the current weather in San Francisco is as follows:

- The temperature is around 61°F to 63°F (17°C).
- It is partly cloudy to mostly sunny.
- The humidity is around 90-94%.
- The wind speed is around 7-22 km/h.

So, the current weather in San Francisco is partly cloudy with a temperature of 61°F (17°C) and high humidity. \n\

Now, I will provide the final answer to the user. \n\

The weather in San Francisco today is partly cloudy with a temperature of 61°F (17°C) and high humidity.\
"""
                ),
                next_part_kind='text',
            ),
            PartStartEvent(index=4, part=TextPart(content='The'), previous_part_kind='thinking'),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' weather')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' San')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' Francisco')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' today')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' is')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' partly')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' cloudy')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' temperature')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' of')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='61')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='°F')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='17')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='°C')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=')')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' high')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' humidity')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' The')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' current')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' conditions')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' include')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' wind')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' speed')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' of')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' around')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='7')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='22')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' km')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='/h')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' humidity')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' level')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' of')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='90')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='94')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='%.')),
            PartEndEvent(
                index=4,
                part=TextPart(
                    content='The weather in San Francisco today is partly cloudy with a temperature of 61°F (17°C) and high humidity. The current conditions include a wind speed of around 7-22 km/h and a humidity level of 90-94%.'
                ),
            ),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='web_search',
                    args={'query': 'What is the weather in San Francisco today?'},
                    tool_call_id=IsStr(),
                    provider_name='groq',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='web_search',
                    content={
                        'images': None,
                        'results': [
                            {
                                'content': "{'location': {'name': 'San Francisco', 'region': 'California', 'country': 'United States of America', 'lat': 37.775, 'lon': -122.4183, 'tz_id': 'America/Los_Angeles', 'localtime_epoch': 1758144075, 'localtime': '2025-09-17 14:21'}, 'current': {'last_updated_epoch': 1758143700, 'last_updated': '2025-09-17 14:15', 'temp_c': 17.4, 'temp_f': 63.3, 'is_day': 1, 'condition': {'text': 'Partly cloudy', 'icon': '//cdn.weatherapi.com/weather/64x64/day/116.png', 'code': 1003}, 'wind_mph': 7.8, 'wind_kph': 12.6, 'wind_degree': 264, 'wind_dir': 'W', 'pressure_mb': 1014.0, 'pressure_in': 29.95, 'precip_mm': 0.0, 'precip_in': 0.0, 'humidity': 94, 'cloud': 75, 'feelslike_c': 17.4, 'feelslike_f': 63.3, 'windchill_c': 17.7, 'windchill_f': 63.9, 'heatindex_c': 17.7, 'heatindex_f': 63.9, 'dewpoint_c': 15.3, 'dewpoint_f': 59.6, 'vis_km': 13.0, 'vis_miles': 8.0, 'uv': 6.8, 'gust_mph': 14.4, 'gust_kph': 23.1}}",
                                'score': 0.9655062,
                                'title': 'Weather in San Francisco',
                                'url': 'https://www.weatherapi.com/',
                            },
                            {
                                'content': "Today's Weather - San Francisco, CA. September 17, 2025 10:00 AM. Exploratorium. 61°. Feels Like 61°. Hi 69°F Lo 56°F. Mostly Sunny.",
                                'score': 0.9512194,
                                'title': 'San Francisco, CA | Weather Forecasts Now, Live Radar Maps ...',
                                'url': 'https://www.weatherbug.com/weather-forecast/now/san-francisco-ca-94103',
                            },
                            {
                                'content': "access_time 10:56 AM PDT on September 17, 2025 (GMT -7) | Updated 10 seconds ago. 76° | 59°. 74 °F. like 75°. icon. Sunny. N. 0. Today's temperature is forecast",
                                'score': 0.92715925,
                                'title': 'San Francisco, CA Weather Conditions | Weather Underground',
                                'url': 'https://www.wunderground.com/weather/us/ca/san-francisco',
                            },
                            {
                                'content': 'Weather in San Francisco, California, USA ; Sep 17, 2025 at 8:56 am · 10 mi · 29.98 "Hg · 87% · 57 °F',
                                'score': 0.9224337,
                                'title': 'Weather for San Francisco, California, USA - Time and Date',
                                'url': 'https://www.timeanddate.com/weather/usa/san-francisco',
                            },
                            {
                                'content': '... Current time: 01:50 2025/09/17. Current Weather; Forecast; Sun and Moon. partly cloudy, 16 °C. Wind speed 22 km/h. Humidity, 90 %. Air pressure, 1014 hPa.',
                                'score': 0.91175514,
                                'title': 'San Francisco - 14-Day Forecast: Temperature, Wind & Radar',
                                'url': 'https://www.ventusky.com/san-francisco',
                            },
                            {
                                'content': '723 FXUS66 KMTR 171146 AFDMTR Area Forecast Discussion National Weather Service San Francisco ... Issued at 406 AM PDT Wed Sep 17 2025 (Today and tonight)',
                                'score': 0.8014549,
                                'title': 'Bay Area forecast discussion - National Weather Service',
                                'url': 'https://forecast.weather.gov/product.php?format=ci&glossary=1&issuedby=mtr&product=afd&site=mtr&version=1',
                            },
                            {
                                'content': 'Detailed ⚡ San Francisco Weather Forecast for September 2025 – day/night 🌡️ temperatures, precipitations – World-Weather.info.',
                                'score': 0.7646988,
                                'title': 'Weather in San Francisco in September 2025',
                                'url': 'https://world-weather.info/forecast/usa/san_francisco/september-2025/',
                            },
                            {
                                'content': 'Full weather forecast for San Francisco in September 2025. Check the temperatures, chance of rain and more in San Francisco during September.',
                                'score': 0.7192461,
                                'title': 'San Francisco weather in September 2025 | Weather25.com',
                                'url': 'https://www.weather25.com/north-america/usa/california/san-francisco?page=month&month=September',
                            },
                            {
                                'content': '10-Day Weather Forecast ; Today. 9/17. 76° · Partly sunny ; Thu. 9/18. 68° · Rather cloudy ; Fri. 9/19. 73° · Partly sunny and pleasant ; Sat. 9/20. 71° · Mostly sunny',
                                'score': 0.68318754,
                                'title': 'San Francisco, CA Weather Forecast - AccuWeather',
                                'url': 'https://www.accuweather.com/en/us/san-francisco/94103/weather-forecast/347629',
                            },
                            {
                                'content': 'We have one more day of hot weather away from the coast today. A dense fog ... 2025 ABC, Inc., KGO-TV San Francisco. All Rights Reserved.',
                                'score': 0.6164054,
                                'title': 'AccuWeather Forecast: 1 more day of hot temperatures away from ...',
                                'url': 'https://abc7news.com/post/weather-bay-area-forecast-temperatures/39468/',
                            },
                            {
                                'content': 'Wednesday morning First Alert weather forecast with Jessica Burch - 9/17/25 ... National - Current Temperatures · National - First Alert Doppler. Latest',
                                'score': 0.6010557,
                                'title': 'San Francisco Bay Area weather and First Alert Weather forecasts',
                                'url': 'https://www.cbsnews.com/sanfrancisco/weather/',
                            },
                            {
                                'content': '10 Day Weather-San Francisco, CA. As of 2:31 pm PDT. Today. 67°/58°. 2%. Day. 67°. 2%. W 17 mph. Plentiful sunshine. High 67F. Winds W at 10 to 20 mph.',
                                'score': 0.52290934,
                                'title': '10-Day Weather Forecast for San Francisco, CA',
                                'url': 'https://weather.com/weather/tenday/l/USCA0987:1:US',
                            },
                            {
                                'content': '10 Day Weather-San Francisco, CA. As of 5:34 pm PDT. Tonight. --/58°. 18%. Night. 58°. 18%. W 15 mph. Partly cloudy early with increasing clouds overnight.',
                                'score': 0.48221022,
                                'title': '10-Day Weather Forecast for San Francisco, CA',
                                'url': 'https://weather.com/weather/tenday/l/94112:4:US',
                            },
                            {
                                'content': 'Night Sky · TodayHourly14 DaysPastClimate. Currently: 61 °F. Passing clouds. (Weather station: San Francisco International Airport, USA). See more current',
                                'score': 0.42419788,
                                'title': 'Past Weather in San Francisco, California, USA - Time and Date',
                                'url': 'https://www.timeanddate.com/weather/usa/san-francisco/historic',
                            },
                            {
                                'content': 'Considerable cloudiness. Low 56F. Winds WSW at 10 to 15 mph. Record Low52°.',
                                'score': 0.327884,
                                'title': 'Monthly Weather Forecast for San Francisco, CA',
                                'url': 'https://weather.com/weather/monthly/l/69bedc6a5b6e977993fb3e5344e3c06d8bc36a1fb6754c3ddfb5310a3c6d6c87',
                            },
                            {
                                'content': 'San Francisco Weather Forecasts. Weather Underground provides local & long-range weather ... Hourly Forecast for Today, Wednesday 09/17Hourly for Today, Wed 09/17.',
                                'score': 0.26997215,
                                'title': 'San Francisco, CA Hourly Weather Forecast - Weather Underground',
                                'url': 'https://www.wunderground.com/hourly/us/ca/san-francisco',
                            },
                        ],
                    },
                    tool_call_id=IsStr(),
                    timestamp=IsDatetime(),
                    provider_name='groq',
                )
            ),
        ]
    )


async def test_groq_model_thinking_part(allow_model_requests: None, groq_api_key: str):
    m = GroqModel('qwen/qwen3-32b', provider=GroqProvider(api_key=groq_api_key))
    settings = GroqModelSettings(groq_reasoning_format='raw')
    agent = Agent(m, instructions='You are a chef.', model_settings=settings)

    result = await agent.run('I want a recipe to cook Uruguayan alfajores.')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='I want a recipe to cook Uruguayan alfajores.', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                instructions='You are a chef.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[IsInstance(ThinkingPart), IsInstance(TextPart)],
                usage=RequestUsage(input_tokens=31, output_tokens=1525),
                model_name='qwen/qwen3-32b',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_url='https://api.groq.com',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': IsDatetime(),
                },
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
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
                timestamp=IsDatetime(),
                instructions='You are a chef.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[IsInstance(ThinkingPart), IsInstance(TextPart)],
                usage=RequestUsage(input_tokens=31, output_tokens=1525),
                model_name='qwen/qwen3-32b',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_url='https://api.groq.com',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='chatcmpl-a2b14bc8-318b-4292-9c90-7eb1ea3eb2d3',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Considering the Uruguayan recipe, how can I cook the Argentinian one?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions='You are a chef.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[IsInstance(ThinkingPart), IsInstance(TextPart)],
                usage=RequestUsage(input_tokens=835, output_tokens=2006),
                model_name='qwen/qwen3-32b',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_url='https://api.groq.com',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='chatcmpl-22e14a71-3400-4399-a952-195bd9f630e0',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_groq_model_thinking_part_iter(allow_model_requests: None, groq_api_key: str):
    m = GroqModel('qwen/qwen3-32b', provider=GroqProvider(api_key=groq_api_key))
    settings = GroqModelSettings(groq_reasoning_format='raw')
    agent = Agent(m, instructions='You are a chef.', model_settings=settings)

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='I want a recipe to cook Uruguayan alfajores.') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    result = agent_run.result
    assert result is not None
    messages = result.all_messages()
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='I want a recipe to cook Uruguayan alfajores.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions='You are a chef.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="""\

Okay, I need to come up with a recipe for Uruguayan alfajores. Let me start by recalling what I know about them. From what I remember, alfajores are traditional South American cookies, often filled with dulce de leche. Uruguayan versions are typically made with a shortcrust pastry, filled with dulce de leche, and covered with chocolate. Wait, but in some countries like Argentina, they might be different, like with a cake-like base. So Uruguay's take might be different.

First, I should check the key components. Shortcrust dough is common. The filling is definitely dulce de leche. Then the coating is chocolate. Also, some recipes might use buttercream or other fillings, but the classic Uruguayan is dulce de leche for sure.

So the steps would be: making the dough, preparing the dulce de leche, assembling the cookies, coating in chocolate. Let me break it down.

For the dough: shortcrust usually requires flour, butter, sugar, maybe an egg. The exact measurements? Maybe 2 cups flour, 1 cup butter, 1/2 cup sugar, 1 egg. Knead until it forms a dough, then chill it. Then roll it out into a disc, maybe 20cm in diameter? Or a rectangle. Then cut into circles with a cookie cutter. Bake until golden, maybe at 180°C for about 10-12 minutes. Let them cool.

For the filling: dulce de leche. Some people make it from scratch by cooking sweetened condensed milk in a water bath for hours. But maybe the user might want to use store-bought if available. If they need to make it, the process is slow-cooked sweetened condensed milk. Alternatively, they can use caramelized sugar. Wait, no, in Uruguay, dulce de leche is the standard filling.

Then, after the dough is baked, each cookie is spread with dulce de leche, then topped with another cookie to make a sandwich. Then coat in rolled chocolate—could be dark, milk, or white chocolate. The coating is made by melting the chocolate and covering the cookies, then sprinkling with something like desiccated coconut, crushed nuts, or sometimes cinnamon. In some Uruguayan recipes, they might use a sugar coating instead of chocolate? Wait, maybe not. I think it's definitely chocolate coating with toppings.

Wait, maybe some variations include a powdered sugar coating instead of chocolate. But traditional Uruguayan alfajores are chocolate-covered. Let me confirm. Yes, in Uruguay, they are chocolate-dipped. So the coating is chocolate, melted and the cookies are dipped into it, then rolled in toppings. The toppings can vary: maybe they use coconut flakes, crushed nuts, sprinkles, or sometimes just keep it plain with a glaze.

Now, the chocolate coating: need to melt dark chocolate, maybe with a bit of vegetable oil or cocoa butter to make it easier. Or use baking chocolate. Then let the chocolate solidify.

So the steps are: make dough, shape and bake into discs, let cool, spread with dulce de leche, sandwich them, then coat each side in melted chocolate, then add toppings.

Let me check for any missing ingredients or steps. The dough might need some egg for binding. Also, the dough should not be too dry. Need to make sure the ratios are correct.

Another point: some recipes mention using 1/4 cup butter, but that might vary. Let me think of standard shortcrust ratios. For a basic shortcrust, the ratio is usually 2:1:1 (flour, fat, liquid). But here, the liquid would be egg. So maybe 2 cups flour, 1 cup butter, 1 egg (about 50g), and maybe a bit of water if needed. Sugar is definitely part of the dough for the desired texture.

Wait, in some Uruguayan recipes, the dough is a bit simpler. Let me think. Maybe using:

For dough:
- 2 cups all-purpose flour
- 1 cup (2 sticks) unsalted butter, cold
- 1 cup powdered sugar
- 1 egg yolk
- 1-2 tablespoons milk or water if needed

Mix the dry ingredients, cut in the butter until it resembles coarse crumbs. Gradually press the yolk into it, add milk if necessary to form a dough. Chill for about an hour, then roll out to 1/4 inch thickness, cut into 3-inch circles. Bake at 350°F (175°C) for 10-12 minutes. Let cool.

For dulce de leche filling: use store-bought if possible. If making from sweetened condensed milk, pour into oven-safe jars, wrap in foil, and bake at 300°F (150°C) for 2 hours, turning jars halfway.

Assemble by placing a dollop of dulce de leche on one cookie, then sandwich with another. Dip in melted dark chocolate (heated with oil or cocoa butter), then coat with coconut, cinnamon sugar, or sprinkles.

Wait, but in Uruguay, maybe they use a thicker layer of dulce de leche, and the chocolate is tempered properly for a good gloss. Also, the size can vary, but typically they're large cookies.

Another point: some recipes say to dust with powdered sugar before coating, but I don't remember that. Maybe for easier coating.

Also, the chocolate coating: sometimes they use a thin layer. Alternatively, they might use a thick glaze. Let me check. The Uruguayan version is often a solid chocolate shell, so maybe melted dark chocolate with a bit of oil, about 1 part oil to 7 parts chocolate.

Now, the toppings: sometimes they add cinnamon sugar, but the more common is coconut sprinkled over the chocolate. Maybe the classic is rolled in coconut, but other variations have nuts or sprinkles.

Putting it all together:

List the ingredients for each part.

Dough ingredients:
- All-purpose flour
- Unsalted butter
- Powdered sugar
- Egg yolk
- Optional: a pinch of salt

Dulce de leche:
- Sweetened condensed milk (if homemade)

Chocolate coating:
- Dark chocolate for melting, maybe 70% cacao
- Neutral oil (coconut oil or vegetable oil)

Toppings:
- Shredded coconut, crushed nuts, or desired toppings

Now, for the method:

1. Make the dough, chill, roll out, cut, bake.
2. Let cool completely.
3. Prepare dulce de leche.
4. Assemble by putting a dollop on one cookie half and pressing another half to sandwich.
5. Melt chocolate with oil.
6. Dip the assembled cookies into the chocolate, covering half for a classic look (like Argentina's) or fully coat them. Let set on parchment paper.
7. Once chocolate is set, sprinkle with toppings like coconut or cinnamon sugar before the chocolate cools completely.

Wait, in Uruguay, sometimes the chocolate covers the entire cookie, or half? Maybe it's just a coating on the outside. I think the standard is to fully cover the side with the filling (the "filled side") with chocolate. So the side with dulce de leche is dipped into the chocolate, then the toppings are applied. Or maybe both sides? I think usually one side is covered in chocolate, and the other side might have a sugared coating. But sometimes they're fully chocolate-coated.

Alternatively, in some recipes, you dip halfway. Let me verify. From what I recall in Uruguayan alfajores, they are often dipped halfway—half covered in chocolate. So the filling is on one side, and that side is covered in chocolate. So the steps would be:

After assembling the sandwiched cookies, take each one and gently dip the dulce de leche side into the melted chocolate, maybe swirling to coat, then place on parchment paper. While the chocolate is still wet, press into toppings like shredded coconut or cinnamon sugar. Let the chocolate set.

This is a common method. So the final shape is a cookie with chocolate half-coverage.

Now, making sure the recipe is clear and the steps are in order. Let me outline the steps properly, from dough to chocolate coating. Also, check if the user needs any clarification on certain steps, like how to make dulce de leche if they don't have it.

Possible questions: What is dulce de leche? How to make it if they don't have it. So maybe include a brief note on that. Also, can store-bought dulce de leche be used, yes. So the recipe should mention that the user can use either.

Possible alternative fillings? But the user want Uruguayan, so stick to dulce de leche.

Now, putting it all together in a structured way, making sure the measurements are precise. Let me check typical measurements for a similar recipe. For example, 2 cups flour, 1 cup butter, 1 cup sugar, 1 egg yolk. That seems like a good starting point.

Butter should be cold to ensure the dough stays flaky. Chilling the dough is important to prevent spreading while baking.

Also, the dough might be too dry, so adding egg yolk helps bind it. Maybe adding a bit of milk as needed. Let me think that egg adds moisture and helps in binding.

For the chocolate coating, maybe 150g of dark chocolate for a batch of say 12-16 alfajores. Depending on the size of the cookies. Let me check.

Now, let me outline the recipe step by step, ensuring that all these elements are covered. Include a note about dulce de leche, and maybe give an alternative if it's not available.

Wait, in some cases, if dulce de leche is not available, people use other fillings, but the answer should be based on the traditional Uruguayan method, so stick to dulce de leche.

Also, maybe mention that they can use a pin to remove excess chocolate if the coating becomes too thick.

Potential issues: dough too dry, cookies too flat. To prevent this, ensure the dough stays cold before and during rolling. Also, not overbaking them, so that they remain tender.

Another thing: when cutting the dough, maybe dust the rolling pin and surface to prevent sticking. Let the cookies cool on the baking sheet before handling to avoid breaking them.

Now, structure the answer with sections: ingredients, preparation for dough, filling, assembling, coating. Make it clear and easy to follow. Use proper measurements, and mention alternatives where possible.

Double-check the steps to ensure that everything is covered. For example, if the user needs to make dulce de leche, include the method. If they have it ready, just use it.

Also, the toppings: shredded coconut could be toasted or not, depending on preference. Mention that.

Okay, I think that's it. Now present the recipe in a clear, step-by-step format with ingredients listed. Make sure the instructions are concise and the user can follow without confusion.
"""
                    ),
                    TextPart(
                        content="""\
**Uruguayan Alfajores Recipe**  \n\
*Flaky, dulce de leche-filled cookies dipped in rich, melted chocolate*

---

### **Ingredients**  \n\
**Dough:**  \n\
- 2 cups (250g) all-purpose flour  \n\
- 1 cup (227g) unsalted butter, cold and cubed  \n\
- 1 cup (100g) powdered sugar  \n\
- 1 large egg yolk  \n\
- 1–2 tablespoons milk (as needed)  \n\

**Filling:**  \n\
- **Option 1** (Store-bought): 1 can dulce de leche  \n\
- **Option 2** (Homemade):  \n\
  - 1 can sweetened condensed milk (12 oz / 355 ml)  \n\

**Chocolate Coating:**  \n\
- 150g dark chocolate (70% cacao, chopped)  \n\
- 1–2 tablespoons coconut oil (to help melt the chocolate smoothly)  \n\

**Optional Toppings:**  \n\
- Shredded coconut, crushed nuts, or cinnamon-sugar mix  \n\

---

### **Instructions**  \n\

**1. Prepare the Dough:**  \n\
1. In a large bowl, mix flour and powdered sugar. Add cold butter and cut it in using a pastry cutter or your fingers until the mixture resembles coarse crumbs.  \n\
2. Add the egg yolk and stir until combined. If the dough is too dry, add milk 1 tablespoon at a time.  \n\
3. Knead gently into a ball (avoid overmixing). Wrap in plastic and refrigerate for 1 hour.  \n\

**2. Roll and Cut Cookies:**  \n\
1. On a lightly floured surface, roll the dough to ⅛-inch (3 mm) thickness.  \n\
2. Use a 3-inch round cookie cutter to cut out circles. Place them on a parchment-lined baking sheet.  \n\

**3. Bake the Cookies:**  \n\
1. Preheat oven to 350°F (175°C). Bake for 10–12 minutes or until golden and just firm.  \n\
2. Let cool completely on a wire rack.  \n\

**4. Make or Prepare Dulce de Leche:**  \n\
- **If using store-bought:** Scoop with a spoon.  \n\
- **If making homemade:**  \n\
  - Place sweetened condensed milk in an oven-safe jar. Wrap in aluminum foil and submerge in a water bath in a roasting pan.  \n\
  - Bake at 300°F (150°C) for 2–2½ hours, turning jars halfway. The milk will thicken and become richly caramelized.  \n\

**5. Assemble the Sandwiched Cookies:**  \n\
1. Place a dollop of dulce de leche on one cookie. Top with another cookie to form a sandwich. Repeat with all cookies.  \n\

**6. Melt the Chocolate Coating:**  \n\
1. Chop dark chocolate into small pieces. Combine in a heatproof bowl with coconut oil.  \n\
2. Microwave in 30-second intervals (stirring each time) until smooth and fully melted (alternatively, use a double boiler).  \n\

**7. Coat in Chocolate:**  \n\
1. Dipping: Hold the dulce de leche side of the cookie down and dip into the melted chocolate. Swirl gently to coat. Place coated cookies on parchment paper.  \n\
2. Toppings: While the chocolate is still wet, sprinkle with shredded coconut, cinnamon sugar, or crushed nuts.  \n\
3. Let sit until the chocolate hardens (about 15–20 minutes, or refrigerate for faster setting).  \n\

---

**Serving Suggestion:**  \n\
Store at room temperature in an airtight container for up to 1 week or freeze for 1 month.  \n\

**Pro Tips:**  \n\
- For a richer flavor, toast the shredded coconut before adding it to the chocolate.  \n\
- If the chocolate coating cracks, temper it by heating to ~90°F (32°C) first.  \n\

Enjoy your Uruguayan alfajores with coffee or tea! ☕🍪\
"""
                    ),
                ],
                usage=RequestUsage(input_tokens=31, output_tokens=3182),
                model_name='qwen/qwen3-32b',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_url='https://api.groq.com',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='chatcmpl-e5a2d9a2-0bff-47af-9796-de682d2c415e',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    assert event_parts == snapshot(
        [
            PartStartEvent(index=0, part=ThinkingPart(content='')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='\n')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Okay')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' need')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' come')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' up')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' recipe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Ur')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ugu')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ayan')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='aj')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ores')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Let')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' me')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' start')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' by')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' recalling')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' what')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' know')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' about')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' them')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' From')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' what')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' remember')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='aj')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ores')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' are')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' traditional')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' South')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' American')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookies')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' often')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' filled')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Ur')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ugu')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ayan')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' versions')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' are')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' typically')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' made')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' short')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='cr')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ust')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' pastry')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' filled')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' covered')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Wait')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' but')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' some')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' countries')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' like')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Argentina')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' might')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' be')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' different')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' like')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cake')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-like')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' base')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' So')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Uruguay')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'s")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' take')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' might')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' be')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' different')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='First')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' should')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' check')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' key')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' components')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Short')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='cr')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ust')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' common')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' The')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' filling')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' definitely')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Then')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coating')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Also')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' some')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' recipes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' might')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' butter')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='cream')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' other')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' fill')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ings')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' but')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' classic')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Ur')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ugu')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ayan')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sure')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='So')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' steps')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' would')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' be')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=':')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' making')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' preparing')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' assembling')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookies')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coating')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Let')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' me')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' break')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' down')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='For')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=':')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' short')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='cr')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ust')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' usually')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' requires')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' flour')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' butter')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' an')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' egg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' The')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' exact')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' measurements')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='?')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='2')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cups')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' flour')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='1')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cup')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' butter')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='1')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='/')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='2')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cup')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='1')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' egg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' K')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ne')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ad')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' until')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' forms')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' then')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chill')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Then')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' roll')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' out')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' into')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' disc')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='2')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='0')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='cm')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' diameter')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='?')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' rectangle')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Then')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cut')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' into')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' circles')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookie')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cutter')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Bake')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' until')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' golden')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' at')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='1')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='8')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='0')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='°C')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' about')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='1')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='0')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='1')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='2')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' minutes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Let')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' them')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cool')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='For')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' filling')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=':')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Some')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' people')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' make')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' from')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' scratch')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' by')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cooking')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sweet')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ened')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' condensed')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' milk')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' water')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' bath')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' hours')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' But')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' user')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' might')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' want')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' store')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-b')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ought')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' if')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' available')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' If')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' need')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' make')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' process')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' slow')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-co')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='oked')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sweet')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ened')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' condensed')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' milk')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Alternatively')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' can')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' caramel')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ized')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Wait')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' no')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Uruguay')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' standard')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' filling')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Then')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' after')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' baked')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' each')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookie')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' spread')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' then')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' topped')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' another')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookie')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' make')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sandwich')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Then')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coat')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' rolled')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='—')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='could')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' be')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dark')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' milk')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' white')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' The')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coating')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' made')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' by')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' melting')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' covering')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookies')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' then')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sprink')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ling')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' something')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' like')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' des')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='icc')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ated')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coconut')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' crushed')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' nuts')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sometimes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cinnamon')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' In')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' some')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Ur')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ugu')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ayan')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' recipes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' might')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coating')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' instead')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='?')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Wait')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' not')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' think')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'s")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' definitely')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coating')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' toppings')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Wait')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' some')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' variations')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' include')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' powdered')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coating')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' instead')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' But')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' traditional')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Ur')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ugu')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ayan')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='aj')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ores')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' are')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-covered')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Let')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' me')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' confirm')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Yes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Uruguay')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' are')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-d')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ipped')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' So')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coating')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' melted')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookies')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' are')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dipped')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' into')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' then')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' rolled')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' toppings')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' The')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' toppings')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' can')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' vary')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=':')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coconut')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' flakes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' crushed')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' nuts')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sprink')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='les')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sometimes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' just')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' keep')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' plain')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' gl')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='aze')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Now')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coating')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=':')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' need')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' melt')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dark')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' bit')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' vegetable')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' oil')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cocoa')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' butter')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' make')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' easier')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' baking')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Then')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' let')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' solid')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ify')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='So')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' steps')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' are')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=':')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' make')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' shape')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' bake')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' into')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' discs')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' let')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cool')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' spread')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sandwich')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' them')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' then')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coat')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' each')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' side')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' melted')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' then')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' add')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' toppings')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Let')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' me')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' check')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' any')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' missing')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ingredients')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' steps')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' The')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' might')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' need')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' some')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' egg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' binding')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Also')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' should')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' not')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' be')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' too')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dry')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Need')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' make')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sure')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ratios')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' are')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' correct')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Another')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' point')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=':')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' some')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' recipes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' mention')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' using')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='1')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='/')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='4')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cup')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' butter')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' but')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' might')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' vary')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Let')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' me')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' think')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' standard')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' short')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='cr')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ust')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ratios')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' For')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' basic')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' short')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='cr')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ust')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ratio')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' usually')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='2')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=':')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='1')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=':')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='1')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' (')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='fl')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='our')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' fat')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' liquid')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=').')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' But')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' here')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' liquid')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' would')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' be')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' egg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' So')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='2')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cups')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' flour')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='1')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cup')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' butter')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='1')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' egg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' (')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='about')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='5')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='0')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='g')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='),')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' bit')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' water')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' if')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' needed')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Sugar')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' definitely')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' part')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' desired')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' texture')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Wait')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' some')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Ur')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ugu')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ayan')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' recipes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' bit')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' simpler')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Let')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' me')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' think')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' using')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
:

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='For')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=':\n')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='2')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cups')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' all')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-purpose')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' flour')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='\n')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='1')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cup')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' (')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='2')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sticks')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=')')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' uns')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='alted')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' butter')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cold')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='\n')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='1')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cup')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' powdered')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='\n')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='1')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' egg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' y')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='olk')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='\n')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='1')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='2')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' tablespoons')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' milk')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' water')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' if')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' needed')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\


"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Mix')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dry')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ingredients')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cut')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' butter')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' until')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' resembles')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coarse')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' crumbs')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Grad')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ually')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' press')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' y')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='olk')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' into')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' add')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' milk')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' if')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' necessary')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' form')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Chill')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' about')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' an')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' hour')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' then')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' roll')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' out')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='1')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='/')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='4')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' inch')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' thickness')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cut')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' into')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='3')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-inch')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' circles')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Bake')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' at')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='3')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='5')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='0')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='°F')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' (')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='1')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='7')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='5')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='°C')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=')')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='1')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='0')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='1')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='2')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' minutes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Let')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cool')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='For')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' filling')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=':')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' store')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-b')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ought')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' if')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' possible')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' If')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' making')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' from')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sweet')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ened')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' condensed')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' milk')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' pour')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' into')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' oven')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-safe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' jars')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' wrap')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' foil')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' bake')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' at')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='3')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='0')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='0')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='°F')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' (')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='1')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='5')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='0')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='°C')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=')')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='2')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' hours')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' turning')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' jars')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' halfway')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='As')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='semble')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' by')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' placing')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dol')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='lop')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' on')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' one')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookie')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' then')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sandwich')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' another')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Dip')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' melted')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dark')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' (')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='he')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ated')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' oil')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cocoa')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' butter')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='),')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' then')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coat')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coconut')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cinnamon')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sprink')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='les')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Wait')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' but')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Uruguay')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' thicker')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' layer')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' tempered')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' properly')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' good')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' gloss')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Also')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' size')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' can')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' vary')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' but')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' typically')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'re")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' large')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookies')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Another')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' point')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=':')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' some')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' recipes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' say')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dust')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' powdered')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' before')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coating')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' but')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' don')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'t")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' remember')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' easier')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coating')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Also')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coating')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=':')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sometimes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' thin')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' layer')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Alternatively')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' might')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' thick')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' gl')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='aze')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Let')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' me')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' check')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' The')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Ur')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ugu')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ayan')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' version')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' often')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' solid')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' shell')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' so')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' melted')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dark')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' bit')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' oil')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' about')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='1')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' part')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' oil')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='7')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' parts')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Now')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' toppings')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=':')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sometimes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' add')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cinnamon')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' but')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' more')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' common')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coconut')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sprink')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='led')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' over')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' classic')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' rolled')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coconut')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' but')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' other')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' variations')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' have')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' nuts')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sprink')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='les')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Putting')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' all')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' together')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
:

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='List')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ingredients')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' each')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' part')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='D')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ough')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ingredients')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=':\n')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' All')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-purpose')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' flour')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='\n')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Un')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='salt')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ed')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' butter')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='\n')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Powder')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ed')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='\n')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Egg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' y')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='olk')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='\n')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Optional')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=':')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' pinch')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' salt')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\


"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='D')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=':\n')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Sweet')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ened')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' condensed')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' milk')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' (')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='if')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' homemade')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
)

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coating')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=':\n')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Dark')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' melting')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='7')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='0')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='%')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' c')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='acao')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='\n')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Neutral')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' oil')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' (')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='co')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='conut')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' oil')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' vegetable')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' oil')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
)

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='T')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='opp')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ings')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=':\n')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Sh')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='redd')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ed')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coconut')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' crushed')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' nuts')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' desired')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' toppings')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\


"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Now')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' method')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
:

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='1')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Make')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chill')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' roll')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' out')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cut')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' bake')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='2')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Let')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cool')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' completely')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='3')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Prepare')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='4')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' As')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='semble')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' by')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' putting')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dol')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='lop')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' on')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' one')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookie')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' half')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' pressing')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' another')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' half')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sandwich')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='5')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' M')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='elt')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' oil')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='6')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Dip')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' assembled')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookies')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' into')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' covering')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' half')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' classic')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' look')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' (')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='like')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Argentina')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'s")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=')')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' fully')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coat')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' them')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Let')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' set')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' on')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' parchment')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' paper')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='7')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Once')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' set')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sprinkle')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' toppings')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' like')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coconut')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cinnamon')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' before')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' co')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ols')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' completely')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Wait')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Uruguay')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sometimes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' covers')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' entire')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookie')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' half')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='?')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'s")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' just')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coating')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' on')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' outside')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' think')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' standard')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' fully')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cover')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' side')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' filling')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' (')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' "')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='filled')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' side')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='")')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' So')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' side')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dipped')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' into')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' then')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' toppings')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' are')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' applied')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' both')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sides')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='?')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' think')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' usually')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' one')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' side')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' covered')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' other')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' side')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' might')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' have')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sug')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ared')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coating')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' But')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sometimes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'re")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' fully')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-co')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ated')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Alternatively')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' some')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' recipes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' you')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dip')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' halfway')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Let')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' me')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' verify')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' From')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' what')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' recall')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Ur')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ugu')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ayan')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='aj')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ores')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' are')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' often')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dipped')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' halfway')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='—')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='half')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' covered')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' So')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' filling')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' on')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' one')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' side')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' side')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' covered')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' So')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' steps')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' would')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' be')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
:

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='After')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' assembling')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sandwich')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ed')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookies')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' take')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' each')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' one')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' gently')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dip')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' side')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' into')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' melted')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' swirling')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coat')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' then')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' place')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' on')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' parchment')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' paper')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' While')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' still')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' wet')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' press')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' into')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' toppings')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' like')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' shredded')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coconut')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cinnamon')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Let')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' set')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='This')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' common')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' method')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' So')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' final')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' shape')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookie')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' half')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='coverage')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Now')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' making')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sure')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' recipe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' clear')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' steps')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' are')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' order')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Let')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' me')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' outline')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' steps')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' properly')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' from')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coating')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Also')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' check')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' if')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' user')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' needs')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' any')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' clarification')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' on')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' certain')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' steps')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' like')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' how')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' make')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' if')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' don')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'t")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' have')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Possible')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' questions')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=':')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' What')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='?')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' How')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' make')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' if')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' don')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'t")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' have')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' So')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' include')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' brief')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' note')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' on')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Also')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' can')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' store')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-b')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ought')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' be')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' used')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' yes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' So')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' recipe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' should')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' mention')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' user')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' can')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' either')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Possible')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' alternative')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' fill')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ings')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='?')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' But')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' user')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' want')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Ur')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ugu')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ayan')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' so')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' stick')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Now')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' putting')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' all')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' together')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' structured')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' way')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' making')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sure')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' measurements')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' are')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' precise')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Let')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' me')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' check')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' typical')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' measurements')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' similar')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' recipe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' For')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' example')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='2')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cups')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' flour')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='1')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cup')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' butter')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='1')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cup')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='1')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' egg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' y')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='olk')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' That')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' seems')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' like')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' good')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' starting')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' point')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='But')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ter')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' should')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' be')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cold')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ensure')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' stays')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' fl')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='aky')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Ch')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='illing')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' important')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' prevent')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' spreading')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' while')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' baking')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Also')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' might')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' be')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' too')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dry')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' so')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' adding')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' egg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' y')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='olk')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' helps')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' bind')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' adding')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' bit')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' milk')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' as')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' needed')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Let')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' me')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' think')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' egg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' adds')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' moisture')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' helps')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' binding')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='For')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coating')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='1')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='5')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='0')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='g')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dark')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' batch')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' say')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='1')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='2')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='1')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='6')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='aj')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ores')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Depending')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' on')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' size')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookies')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Let')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' me')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' check')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Now')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' let')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' me')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' outline')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' recipe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' step')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' by')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' step')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ensuring')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' all')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' these')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' elements')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' are')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' covered')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Include')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' note')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' about')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' give')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' an')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' alternative')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' if')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'s")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' not')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' available')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Wait')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' some')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cases')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' if')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' not')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' available')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' people')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' other')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' fill')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ings')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' but')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' answer')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' should')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' be')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' based')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' on')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' traditional')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Ur')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ugu')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ayan')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' method')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' so')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' stick')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Also')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' mention')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' can')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' pin')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' remove')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' excess')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' if')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coating')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' becomes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' too')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' thick')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Potential')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' issues')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=':')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' too')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dry')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookies')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' too')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' flat')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' To')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' prevent')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' this')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ensure')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' stays')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cold')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' before')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' during')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' rolling')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Also')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' not')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' over')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='b')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='aking')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' them')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' so')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' remain')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' tender')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Another')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' thing')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=':')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' when')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cutting')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dust')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' rolling')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' pin')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' surface')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' prevent')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sticking')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Let')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookies')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cool')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' on')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' baking')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sheet')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' before')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' handling')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' avoid')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' breaking')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' them')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Now')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' structure')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' answer')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sections')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=':')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ingredients')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' preparation')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' filling')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' assembling')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coating')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Make')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' clear')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' easy')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' follow')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' proper')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' measurements')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' alternatives')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' where')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' possible')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Double')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-check')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' steps')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ensure')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' everything')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' covered')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' For')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' example')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' if')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' user')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' needs')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' make')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' include')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' method')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' If')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' have')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ready')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' just')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Also')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' toppings')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=':')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' shredded')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coconut')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' could')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' be')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' toasted')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' not')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' depending')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' on')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' preference')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Mention')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Okay')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' think')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'s")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Now')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' present')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' recipe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' clear')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' step')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-by')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-step')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' format')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ingredients')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' listed')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Make')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sure')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' instructions')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' are')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' concise')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' user')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' can')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' follow')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' without')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' confusion')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.\n')),
            PartEndEvent(
                index=0,
                part=ThinkingPart(
                    content="""\

Okay, I need to come up with a recipe for Uruguayan alfajores. Let me start by recalling what I know about them. From what I remember, alfajores are traditional South American cookies, often filled with dulce de leche. Uruguayan versions are typically made with a shortcrust pastry, filled with dulce de leche, and covered with chocolate. Wait, but in some countries like Argentina, they might be different, like with a cake-like base. So Uruguay's take might be different.

First, I should check the key components. Shortcrust dough is common. The filling is definitely dulce de leche. Then the coating is chocolate. Also, some recipes might use buttercream or other fillings, but the classic Uruguayan is dulce de leche for sure.

So the steps would be: making the dough, preparing the dulce de leche, assembling the cookies, coating in chocolate. Let me break it down.

For the dough: shortcrust usually requires flour, butter, sugar, maybe an egg. The exact measurements? Maybe 2 cups flour, 1 cup butter, 1/2 cup sugar, 1 egg. Knead until it forms a dough, then chill it. Then roll it out into a disc, maybe 20cm in diameter? Or a rectangle. Then cut into circles with a cookie cutter. Bake until golden, maybe at 180°C for about 10-12 minutes. Let them cool.

For the filling: dulce de leche. Some people make it from scratch by cooking sweetened condensed milk in a water bath for hours. But maybe the user might want to use store-bought if available. If they need to make it, the process is slow-cooked sweetened condensed milk. Alternatively, they can use caramelized sugar. Wait, no, in Uruguay, dulce de leche is the standard filling.

Then, after the dough is baked, each cookie is spread with dulce de leche, then topped with another cookie to make a sandwich. Then coat in rolled chocolate—could be dark, milk, or white chocolate. The coating is made by melting the chocolate and covering the cookies, then sprinkling with something like desiccated coconut, crushed nuts, or sometimes cinnamon. In some Uruguayan recipes, they might use a sugar coating instead of chocolate? Wait, maybe not. I think it's definitely chocolate coating with toppings.

Wait, maybe some variations include a powdered sugar coating instead of chocolate. But traditional Uruguayan alfajores are chocolate-covered. Let me confirm. Yes, in Uruguay, they are chocolate-dipped. So the coating is chocolate, melted and the cookies are dipped into it, then rolled in toppings. The toppings can vary: maybe they use coconut flakes, crushed nuts, sprinkles, or sometimes just keep it plain with a glaze.

Now, the chocolate coating: need to melt dark chocolate, maybe with a bit of vegetable oil or cocoa butter to make it easier. Or use baking chocolate. Then let the chocolate solidify.

So the steps are: make dough, shape and bake into discs, let cool, spread with dulce de leche, sandwich them, then coat each side in melted chocolate, then add toppings.

Let me check for any missing ingredients or steps. The dough might need some egg for binding. Also, the dough should not be too dry. Need to make sure the ratios are correct.

Another point: some recipes mention using 1/4 cup butter, but that might vary. Let me think of standard shortcrust ratios. For a basic shortcrust, the ratio is usually 2:1:1 (flour, fat, liquid). But here, the liquid would be egg. So maybe 2 cups flour, 1 cup butter, 1 egg (about 50g), and maybe a bit of water if needed. Sugar is definitely part of the dough for the desired texture.

Wait, in some Uruguayan recipes, the dough is a bit simpler. Let me think. Maybe using:

For dough:
- 2 cups all-purpose flour
- 1 cup (2 sticks) unsalted butter, cold
- 1 cup powdered sugar
- 1 egg yolk
- 1-2 tablespoons milk or water if needed

Mix the dry ingredients, cut in the butter until it resembles coarse crumbs. Gradually press the yolk into it, add milk if necessary to form a dough. Chill for about an hour, then roll out to 1/4 inch thickness, cut into 3-inch circles. Bake at 350°F (175°C) for 10-12 minutes. Let cool.

For dulce de leche filling: use store-bought if possible. If making from sweetened condensed milk, pour into oven-safe jars, wrap in foil, and bake at 300°F (150°C) for 2 hours, turning jars halfway.

Assemble by placing a dollop of dulce de leche on one cookie, then sandwich with another. Dip in melted dark chocolate (heated with oil or cocoa butter), then coat with coconut, cinnamon sugar, or sprinkles.

Wait, but in Uruguay, maybe they use a thicker layer of dulce de leche, and the chocolate is tempered properly for a good gloss. Also, the size can vary, but typically they're large cookies.

Another point: some recipes say to dust with powdered sugar before coating, but I don't remember that. Maybe for easier coating.

Also, the chocolate coating: sometimes they use a thin layer. Alternatively, they might use a thick glaze. Let me check. The Uruguayan version is often a solid chocolate shell, so maybe melted dark chocolate with a bit of oil, about 1 part oil to 7 parts chocolate.

Now, the toppings: sometimes they add cinnamon sugar, but the more common is coconut sprinkled over the chocolate. Maybe the classic is rolled in coconut, but other variations have nuts or sprinkles.

Putting it all together:

List the ingredients for each part.

Dough ingredients:
- All-purpose flour
- Unsalted butter
- Powdered sugar
- Egg yolk
- Optional: a pinch of salt

Dulce de leche:
- Sweetened condensed milk (if homemade)

Chocolate coating:
- Dark chocolate for melting, maybe 70% cacao
- Neutral oil (coconut oil or vegetable oil)

Toppings:
- Shredded coconut, crushed nuts, or desired toppings

Now, for the method:

1. Make the dough, chill, roll out, cut, bake.
2. Let cool completely.
3. Prepare dulce de leche.
4. Assemble by putting a dollop on one cookie half and pressing another half to sandwich.
5. Melt chocolate with oil.
6. Dip the assembled cookies into the chocolate, covering half for a classic look (like Argentina's) or fully coat them. Let set on parchment paper.
7. Once chocolate is set, sprinkle with toppings like coconut or cinnamon sugar before the chocolate cools completely.

Wait, in Uruguay, sometimes the chocolate covers the entire cookie, or half? Maybe it's just a coating on the outside. I think the standard is to fully cover the side with the filling (the "filled side") with chocolate. So the side with dulce de leche is dipped into the chocolate, then the toppings are applied. Or maybe both sides? I think usually one side is covered in chocolate, and the other side might have a sugared coating. But sometimes they're fully chocolate-coated.

Alternatively, in some recipes, you dip halfway. Let me verify. From what I recall in Uruguayan alfajores, they are often dipped halfway—half covered in chocolate. So the filling is on one side, and that side is covered in chocolate. So the steps would be:

After assembling the sandwiched cookies, take each one and gently dip the dulce de leche side into the melted chocolate, maybe swirling to coat, then place on parchment paper. While the chocolate is still wet, press into toppings like shredded coconut or cinnamon sugar. Let the chocolate set.

This is a common method. So the final shape is a cookie with chocolate half-coverage.

Now, making sure the recipe is clear and the steps are in order. Let me outline the steps properly, from dough to chocolate coating. Also, check if the user needs any clarification on certain steps, like how to make dulce de leche if they don't have it.

Possible questions: What is dulce de leche? How to make it if they don't have it. So maybe include a brief note on that. Also, can store-bought dulce de leche be used, yes. So the recipe should mention that the user can use either.

Possible alternative fillings? But the user want Uruguayan, so stick to dulce de leche.

Now, putting it all together in a structured way, making sure the measurements are precise. Let me check typical measurements for a similar recipe. For example, 2 cups flour, 1 cup butter, 1 cup sugar, 1 egg yolk. That seems like a good starting point.

Butter should be cold to ensure the dough stays flaky. Chilling the dough is important to prevent spreading while baking.

Also, the dough might be too dry, so adding egg yolk helps bind it. Maybe adding a bit of milk as needed. Let me think that egg adds moisture and helps in binding.

For the chocolate coating, maybe 150g of dark chocolate for a batch of say 12-16 alfajores. Depending on the size of the cookies. Let me check.

Now, let me outline the recipe step by step, ensuring that all these elements are covered. Include a note about dulce de leche, and maybe give an alternative if it's not available.

Wait, in some cases, if dulce de leche is not available, people use other fillings, but the answer should be based on the traditional Uruguayan method, so stick to dulce de leche.

Also, maybe mention that they can use a pin to remove excess chocolate if the coating becomes too thick.

Potential issues: dough too dry, cookies too flat. To prevent this, ensure the dough stays cold before and during rolling. Also, not overbaking them, so that they remain tender.

Another thing: when cutting the dough, maybe dust the rolling pin and surface to prevent sticking. Let the cookies cool on the baking sheet before handling to avoid breaking them.

Now, structure the answer with sections: ingredients, preparation for dough, filling, assembling, coating. Make it clear and easy to follow. Use proper measurements, and mention alternatives where possible.

Double-check the steps to ensure that everything is covered. For example, if the user needs to make dulce de leche, include the method. If they have it ready, just use it.

Also, the toppings: shredded coconut could be toasted or not, depending on preference. Mention that.

Okay, I think that's it. Now present the recipe in a clear, step-by-step format with ingredients listed. Make sure the instructions are concise and the user can follow without confusion.
"""
                ),
                next_part_kind='text',
            ),
            PartStartEvent(index=1, part=TextPart(content='**'), previous_part_kind='thinking'),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Ur')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ugu')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ayan')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Alf')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='aj')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ores')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Recipe')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='*')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Fl')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='aky')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ce')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' de')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' le')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='che')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-filled')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cookies')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dipped')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' rich')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' melted')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
*

"""
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
---

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='###')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Ingredients')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='D')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ough')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cups')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='5')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='0')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='g')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=')')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' all')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-purpose')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' flour')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cup')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='7')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='g')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=')')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' uns')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='alted')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' butter')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cold')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cub')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ed')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cup')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='0')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='0')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='g')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=')')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' powdered')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' large')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' egg')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' y')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='olk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='–')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' tablespoons')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' milk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='as')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' needed')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=')')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
  \n\

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='F')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='illing')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Option')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Store')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-b')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ought')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='):')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' can')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ce')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' de')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' le')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='che')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Option')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Hom')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='emade')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='):')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' can')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sweet')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ened')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' condensed')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' milk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' oz')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' /')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='3')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='5')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='5')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ml')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=')')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
  \n\

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Chocolate')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Co')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ating')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='5')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='0')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='g')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dark')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='7')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='0')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='%')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' c')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='acao')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' chopped')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=')')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='–')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' tablespoons')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' coconut')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' oil')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' help')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' melt')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' smoothly')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=')')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
  \n\

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Optional')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' T')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='opp')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ings')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Sh')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='redd')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ed')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' coconut')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' crushed')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' nuts')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cinnamon')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-s')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ugar')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' mix')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
  \n\

"""
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
---

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='###')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Instructions')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
  \n\

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Prepare')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Dough')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' In')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' large')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' bowl')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' mix')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' flour')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' powdered')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Add')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cold')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' butter')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cut')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' it')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' using')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' pastry')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cutter')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' your')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' fingers')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' until')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' mixture')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' resembles')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' coarse')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' crumbs')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Add')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' egg')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' y')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='olk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' stir')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' until')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' combined')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' If')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' is')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' too')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dry')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' add')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' milk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' tablespoon')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' at')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' time')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='3')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' K')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ne')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ad')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' gently')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' into')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ball')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='avoid')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' over')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='mix')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ing')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=').')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Wrap')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' plastic')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' refriger')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ate')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' hour')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
  \n\

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Roll')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Cut')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Cookies')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' On')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' lightly')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' fl')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='oured')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' surface')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' roll')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='⅛')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-inch')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='3')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' mm')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=')')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' thickness')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Use')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='3')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-inch')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' round')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cookie')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cutter')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cut')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' out')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' circles')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Place')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' them')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' on')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' parchment')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-lined')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' baking')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sheet')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
  \n\

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='3')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Bake')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Cookies')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Pre')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='heat')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' oven')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='3')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='5')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='0')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='°F')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='7')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='5')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='°C')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=').')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Bake')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='0')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='–')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' minutes')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' until')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' golden')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' just')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' firm')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Let')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cool')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' completely')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' on')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' wire')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' rack')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
  \n\

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='4')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Make')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Prepare')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Dul')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ce')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' de')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Le')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='che')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='If')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' using')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' store')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-b')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ought')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Sco')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='op')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' spoon')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='If')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' making')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' homemade')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Place')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sweet')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ened')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' condensed')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' milk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' an')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' oven')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-safe')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' jar')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Wrap')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' aluminum')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' foil')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sub')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='merge')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' water')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' bath')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ro')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='asting')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' pan')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Bake')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' at')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='3')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='0')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='0')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='°F')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='5')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='0')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='°C')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=')')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='–')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='½')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' hours')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' turning')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' jars')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' halfway')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' The')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' milk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' will')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' th')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='icken')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' become')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' rich')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ly')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' caramel')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ized')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
  \n\

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='5')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' As')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='semble')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Sandwich')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ed')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Cookies')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Place')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dol')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='lop')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' of')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ce')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' de')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' le')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='che')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' on')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' one')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cookie')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Top')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' another')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cookie')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' form')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sandwich')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Repeat')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' all')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cookies')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
  \n\

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='6')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' M')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='elt')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Chocolate')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Co')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ating')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Chop')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dark')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' into')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' small')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' pieces')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Combine')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' heat')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='proof')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' bowl')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' coconut')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' oil')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Microwave')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='3')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='0')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-second')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' intervals')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='st')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ir')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ring')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' each')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' time')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=')')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' until')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' smooth')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' fully')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' melted')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='altern')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='atively')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' use')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' double')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' boiler')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=').')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
  \n\

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='7')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Coat')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Chocolate')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' D')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ipping')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Hold')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ce')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' de')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' le')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='che')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' side')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' of')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cookie')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' down')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dip')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' into')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' melted')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Sw')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='irl')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' gently')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' coat')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Place')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' coated')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cookies')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' on')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' parchment')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' paper')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' T')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='opp')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ings')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' While')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' is')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' still')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' wet')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sprinkle')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' shredded')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' coconut')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cinnamon')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' crushed')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' nuts')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='3')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Let')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sit')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' until')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' hard')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ens')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='about')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='5')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='–')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='0')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' minutes')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' refriger')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ate')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' faster')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' setting')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=').')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
  \n\

"""
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
---

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='S')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='erving')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' S')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='uggestion')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Store')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' at')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' room')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' temperature')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' an')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='irt')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ight')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' container')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' up')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' week')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' freeze')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' month')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
  \n\

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Pro')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Tips')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' For')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' richer')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' flavor')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' toast')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' shredded')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' coconut')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' before')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' adding')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' it')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' If')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' coating')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cracks')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' temper')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' it')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' by')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' heating')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ~')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='9')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='0')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='°F')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='3')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='°C')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=')')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' first')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
  \n\

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Enjoy')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' your')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Ur')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ugu')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ayan')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='aj')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ores')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' coffee')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' tea')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='!')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ☕')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='🍪')),
            PartEndEvent(
                index=1,
                part=TextPart(
                    content="""\
**Uruguayan Alfajores Recipe**  \n\
*Flaky, dulce de leche-filled cookies dipped in rich, melted chocolate*

---

### **Ingredients**  \n\
**Dough:**  \n\
- 2 cups (250g) all-purpose flour  \n\
- 1 cup (227g) unsalted butter, cold and cubed  \n\
- 1 cup (100g) powdered sugar  \n\
- 1 large egg yolk  \n\
- 1–2 tablespoons milk (as needed)  \n\

**Filling:**  \n\
- **Option 1** (Store-bought): 1 can dulce de leche  \n\
- **Option 2** (Homemade):  \n\
  - 1 can sweetened condensed milk (12 oz / 355 ml)  \n\

**Chocolate Coating:**  \n\
- 150g dark chocolate (70% cacao, chopped)  \n\
- 1–2 tablespoons coconut oil (to help melt the chocolate smoothly)  \n\

**Optional Toppings:**  \n\
- Shredded coconut, crushed nuts, or cinnamon-sugar mix  \n\

---

### **Instructions**  \n\

**1. Prepare the Dough:**  \n\
1. In a large bowl, mix flour and powdered sugar. Add cold butter and cut it in using a pastry cutter or your fingers until the mixture resembles coarse crumbs.  \n\
2. Add the egg yolk and stir until combined. If the dough is too dry, add milk 1 tablespoon at a time.  \n\
3. Knead gently into a ball (avoid overmixing). Wrap in plastic and refrigerate for 1 hour.  \n\

**2. Roll and Cut Cookies:**  \n\
1. On a lightly floured surface, roll the dough to ⅛-inch (3 mm) thickness.  \n\
2. Use a 3-inch round cookie cutter to cut out circles. Place them on a parchment-lined baking sheet.  \n\

**3. Bake the Cookies:**  \n\
1. Preheat oven to 350°F (175°C). Bake for 10–12 minutes or until golden and just firm.  \n\
2. Let cool completely on a wire rack.  \n\

**4. Make or Prepare Dulce de Leche:**  \n\
- **If using store-bought:** Scoop with a spoon.  \n\
- **If making homemade:**  \n\
  - Place sweetened condensed milk in an oven-safe jar. Wrap in aluminum foil and submerge in a water bath in a roasting pan.  \n\
  - Bake at 300°F (150°C) for 2–2½ hours, turning jars halfway. The milk will thicken and become richly caramelized.  \n\

**5. Assemble the Sandwiched Cookies:**  \n\
1. Place a dollop of dulce de leche on one cookie. Top with another cookie to form a sandwich. Repeat with all cookies.  \n\

**6. Melt the Chocolate Coating:**  \n\
1. Chop dark chocolate into small pieces. Combine in a heatproof bowl with coconut oil.  \n\
2. Microwave in 30-second intervals (stirring each time) until smooth and fully melted (alternatively, use a double boiler).  \n\

**7. Coat in Chocolate:**  \n\
1. Dipping: Hold the dulce de leche side of the cookie down and dip into the melted chocolate. Swirl gently to coat. Place coated cookies on parchment paper.  \n\
2. Toppings: While the chocolate is still wet, sprinkle with shredded coconut, cinnamon sugar, or crushed nuts.  \n\
3. Let sit until the chocolate hardens (about 15–20 minutes, or refrigerate for faster setting).  \n\

---

**Serving Suggestion:**  \n\
Store at room temperature in an airtight container for up to 1 week or freeze for 1 month.  \n\

**Pro Tips:**  \n\
- For a richer flavor, toast the shredded coconut before adding it to the chocolate.  \n\
- If the chocolate coating cracks, temper it by heating to ~90°F (32°C) first.  \n\

Enjoy your Uruguayan alfajores with coffee or tea! ☕🍪\
"""
                ),
            ),
        ]
    )

    event_parts: list[Any] = []
    async with agent.iter(
        user_prompt='Considering the Uruguayan recipe, how can I cook the Argentinian one?',
        message_history=messages,
        model_settings=GroqModelSettings(groq_reasoning_format='parsed'),
    ) as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    result = agent_run.result
    assert result is not None
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Considering the Uruguayan recipe, how can I cook the Argentinian one?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions='You are a chef.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="""\
Okay, so the user wants to know how to make the Argentinian version of alfajores after I provided the Uruguayan recipe. First, I need to recall the differences between the two.

Uruguayan alfajores are typically sandwiched with dulce de leche and dipped in chocolate, as described. Argentinian ones might have variations. From what I remember, Argentinian alfajores can vary by region. Some might use similar methods but with different coatings or fillings. For example, I think some recipes use a layer of dulce de leche and are topped with chocolate instead of dipping, but maybe they use a cookie base or a different kind of dough.

Wait, the user wants a clear contrast, so I need to highlight the main differences. Let me check my knowledge. In Argentina, there's a traditional recipe that uses a base of flour, butter, and sugar, similar to Uruguay. However, some Argentinian versions might use a different kind of cookie or a softer texture. Also, the coating: while Uruguayans dip them in chocolate, Argentinians might brush them with egg wash and coat in sugar, or sometimes use a different type of glaze.

Another point is the filling. The user mentioned dulce de leche in Uruguay, but Argentinian versions might use similar fillings but sometimes with a different consistency or layering method. For example, in some parts of Argentina, they might spread dulce de leche and then press the cookie halves together, but perhaps they don't dip them in chocolate the same way.

I should also consider regional variations. For instance, in Córdoba, they might have a type of alfajor with a different texture or even a different base, but the most common one is similar to Uruguay but with subtle differences. Maybe the recipe uses a different type of flour or includes something like egg yolk in the dough for a flakier texture.

Also, the coating method: Uruguayans dip in chocolate after assembling, whereas Argentinians might coat the cookies individually with a sugar mixture or a different type of topping. Some recipes use a "torta de aceite" which is a type of dough that's rolled, filled, and then folded, then baked, but I'm not sure if that applies here. The user is likely looking for a direct comparison to the Uruguayan version they just had the recipe for.

So, the key points to highlight would be:
- Possible differences in dough (maybe more egg yolk for flakiness)
- Coating method (chocolate vs. egg wash + sugar or other coatings)
- Assembly technique (dipping vs. brushing)
- Filling application (maybe different consistency or method)

I need to present a recipe that's clearly Argentinian, maybe with those elements. Also, mention that while they're very similar, cultural variations exist. Provide a recipe that's a typical Argentinian version, maybe using a different coating technique or adjusting the dough slightly. Perhaps suggest using egg yolks for the dough to make it flakier. Also, note that some Argentinian versions might use a crumbly base rather than the firmed dough used in Uruguay.

Make sure to address the user's question by contrasting the two recipes, highlighting the main differences, and providing a step-by-step Argentinian recipe that's distinct from the Uruguayan one they requested previously. Also, check for any other variations, like using different types of dulce de leche or adding ingredients like coconut or nuts.
"""
                    ),
                    TextPart(
                        content="""\
Here's a traditional **Argentinian alfajores** recipe with subtle differences from the Uruguayan version. The main contrasts are in the **dough texture**, **coating method**, and **presentation**. These are softer, often brushed with egg wash and dusted with sugar, or coated in crumbled dough (like "alfajores con cobertura crujiente"). \n\

---

### **Argentinian Alfajores** (Traditional Version)  \n\
*Soft, crumbly dough sandwiched with dulce de leche, brushed with egg and sugar or coated in crumbled dough.*

---

### **Ingredients**  \n\
**Dough:**  \n\
- 2 cups (250g) all-purpose flour  \n\
- 1 cup (227g) unsalted butter, cold and cubed  \n\
- ½ cup (50g) powdered sugar  \n\
- 1–2 egg yolks (optional, for flakiness) *  \n\
- 1–2 tablespoons milk (as needed)  \n\

**Filling:**  \n\
- 1 can dulce de leche (see optional homemade recipe below)  \n\

**Coating Options:**  \n\
- **Option 1**: 1 egg yolk + 2 tablespoons sugar (for brushing)  \n\
- **Option 2**: 2–3 tablespoons crumbled dough scraps (see below)  \n\
- **Optional toppings**: Cinnamon sugar, crushed nuts, or shredded coconut  \n\

---

### **Instructions**  \n\

**1. Make the Dough (Same as Uruguayan, with optional tweak):**  \n\
1. In a bowl, mix flour and powdered sugar.  \n\
2. Add cold butter and cut it in until the mixture resembles coarse breadcrumbs.  \n\
3. Add **1–2 egg yolks** *(optional for flakier texture)* and mix. Add milk to bind the dough.  \n\
4. Cover and refrigerate for 1 hour.  \n\

**2. Roll and Cut the Dough:**  \n\
1. On a lightly floured surface, roll the dough to ⅛-inch (3 mm) thickness.  \n\
2. Use a 3–3.5-inch round cutter to cut circles. Reserve scraps for coating (Step 5).  \n\

**3. Bake:**  \n\
1. Preheat oven to 350°F (175°C).  \n\
2. Bake for 10–12 minutes until golden and slightly crisp. Cool slightly.  \n\

**4. Prepare the Dulce de Leche:**  \n\
- Use ½ spoonful of dulce de leche for each cookie half.  \n\

**5. Assemble the Sandwiches:**  \n\
1. Spread dulce de leche onto one cookie, then press another cookie on top. Set aside.  \n\

---

### **Coating Options**  \n\

**Option 1: Egg-Wash & Cinnamon Sugar**  \n\
1. Whisk 1 egg yolk with 2 tbsp sugar. Brush the bottom half of the cookie with this mixture.  \n\
2. Roll in cinnamon sugar while the coating is dry. Repeat for the top half if desired.  \n\

**Option 2: Crumbled Dough Coating**  \n\
1. Toss leftover dough scraps with a bit of flour and process in a food processor to create fine crumbs.  \n\
2. Dip one side of the assembled cookie in melted chocolate (50% cacao for richness) or spread with a thin layer of dulce de leche, then press into the crumbled dough.  \n\

---

### **Key Differences from Uruguayan Alfajores**  \n\
| Feature                  | **Uruguayan**                          | **Argentinian**                          |  \n\
|--------------------------|----------------------------------------|-------------------------------------------|  \n\
| **Dough**               | Dense, firmed dough                    | Flakier, optionally with egg yolks         |  \n\
| **Fillings**            | Sandwiched with dulce de leche         | Same, but sometimes with a thicker layer   |  \n\
| **Coating**             | Dipped in dark chocolate               | Brushed with egg/sugar or crusted with dough |  \n\
| **Presentation**        | Round, uniform shape                   | Varied shapes, sometimes with edges folded |  \n\

---

### **Optional DIY Dulce de Leche**  \n\
- **From sweetened condensed milk**: Place a can in a pot of simmering water for 3 hours.  \n\
- **From scratch**: Simmer **1 cup sugar + 1 cup heavy cream** until thickened, then stir in **2 cups whole milk** and cook 15 minutes.  \n\

---

### **Serving Suggestions**  \n\
- Store in an airtight container (up to 5 days).  \n\
- Argentinians often sell them in pairs with a stick for the dulce de leche to drizzle down.  \n\

Enjoy your Argentinian alfajores! 🇦🇷\
"""
                    ),
                ],
                usage=RequestUsage(input_tokens=903, output_tokens=1717),
                model_name='qwen/qwen3-32b',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_url='https://api.groq.com',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='chatcmpl-cc836043-bb1b-4994-8ea6-78f02cfc68ac',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    assert event_parts == snapshot(
        [
            PartStartEvent(index=0, part=ThinkingPart(content='Okay')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' so')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' user')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' wants')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' know')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' how')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' make')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Arg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='entin')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ian')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' version')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='aj')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ores')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' after')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' provided')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Ur')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ugu')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ayan')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' recipe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' First')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' need')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' recall')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' differences')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' between')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' two')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Ur')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ugu')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ayan')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='aj')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ores')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' are')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' typically')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sandwich')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ed')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dipped')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' as')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' described')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Arg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='entin')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ian')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ones')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' might')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' have')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' variations')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' From')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' what')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' remember')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Arg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='entin')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ian')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='aj')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ores')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' can')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' vary')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' by')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' region')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Some')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' might')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' similar')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' methods')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' but')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' different')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coatings')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' fill')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ings')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' For')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' example')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' think')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' some')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' recipes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' layer')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' are')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' topped')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' instead')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dipping')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' but')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookie')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' base')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' different')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' kind')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dough')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Wait')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' user')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' wants')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' clear')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' contrast')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' so')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' need')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' highlight')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' main')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' differences')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Let')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' me')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' check')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' my')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' knowledge')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' In')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Argentina')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' there')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'s")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' traditional')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' recipe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' uses')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' base')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' flour')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' butter')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' similar')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Uruguay')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' However')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' some')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Arg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='entin')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ian')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' versions')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' might')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' different')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' kind')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookie')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' softer')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' texture')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Also')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coating')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=':')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' while')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Uruguay')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ans')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dip')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' them')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Arg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='entin')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ians')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' might')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' brush')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' them')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' egg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' wash')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coat')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sometimes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' different')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' type')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' gl')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='aze')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Another')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' point')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' filling')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' The')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' user')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' mentioned')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Uruguay')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' but')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Arg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='entin')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ian')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' versions')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' might')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' similar')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' fill')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ings')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' but')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sometimes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' different')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' consistency')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' layer')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ing')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' method')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' For')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' example')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' some')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' parts')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Argentina')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' might')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' spread')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' then')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' press')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookie')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' halves')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' together')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' but')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' perhaps')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' don')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'t")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dip')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' them')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' same')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' way')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' should')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' also')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' consider')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' regional')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' variations')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' For')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' instance')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Có')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='rd')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='oba')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' might')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' have')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' type')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ajor')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' different')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' texture')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' even')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' different')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' base')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' but')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' most')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' common')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' one')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' similar')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Uruguay')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' but')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' subtle')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' differences')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' recipe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' uses')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' different')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' type')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' flour')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' includes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' something')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' like')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta=' egg'),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' y')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='olk')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' fl')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ak')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ier')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' texture')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Also')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coating')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' method')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=':')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Uruguay')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ans')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dip')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' after')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' assembling')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' whereas')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Arg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='entin')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ians')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' might')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coat')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookies')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' individually')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' mixture')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' different')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' type')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' topping')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Some')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' recipes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' "')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='t')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='orta')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ace')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ite')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='"')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' which')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' type')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'s")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' rolled')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' filled')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' then')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' folded')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' then')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' baked')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' but')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'m")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' not')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sure')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' if')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' applies')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' here')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' The')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' user')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' likely')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' looking')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' direct')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' comparison')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Ur')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ugu')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ayan')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' version')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' just')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' had')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' recipe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='So')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' key')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' points')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' highlight')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' would')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' be')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=':\n')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Possible')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' differences')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' (')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' more')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' egg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' y')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='olk')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' fl')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ak')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='iness')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=')\n')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Co')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ating')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' method')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' (')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ch')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' vs')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta='.'),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' egg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' wash')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' +')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' other')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coatings')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=')\n')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Assembly')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' technique')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' (')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='d')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ipping')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' vs')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' brushing')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=')\n')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' F')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='illing')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' application')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' (')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' different')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' consistency')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' method')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
)

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' need')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' present')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' recipe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'s")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' clearly')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Arg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='entin')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ian')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' those')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' elements')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Also')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' mention')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' while')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'re")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' very')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' similar')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cultural')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' variations')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' exist')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Provide')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' recipe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'s")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' typical')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Arg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='entin')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ian')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' version')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' using')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' different')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coating')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' technique')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' adjusting')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' slightly')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Perhaps')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' suggest')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' using')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' egg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' yol')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ks')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' make')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' fl')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ak')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ier')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Also')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' note')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' some')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Arg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='entin')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ian')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' versions')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' might')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cr')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='umb')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ly')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' base')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' rather')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' than')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' f')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='irmed')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' used')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Uruguay')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Make')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sure')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' address')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' user')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'s")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' question')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' by')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' contrasting')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' two')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' recipes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' highlighting')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' main')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' differences')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' providing')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' step')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-by')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-step')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Arg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='entin')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ian')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' recipe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'s")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' distinct')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' from')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Ur')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ugu')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ayan')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' one')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' requested')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' previously')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Also')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' check')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' any')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' other')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' variations')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' like')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' using')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' different')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' types')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' adding')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ingredients')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' like')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coconut')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' nuts')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.\n')),
            PartEndEvent(
                index=0,
                part=ThinkingPart(
                    content="""\
Okay, so the user wants to know how to make the Argentinian version of alfajores after I provided the Uruguayan recipe. First, I need to recall the differences between the two.

Uruguayan alfajores are typically sandwiched with dulce de leche and dipped in chocolate, as described. Argentinian ones might have variations. From what I remember, Argentinian alfajores can vary by region. Some might use similar methods but with different coatings or fillings. For example, I think some recipes use a layer of dulce de leche and are topped with chocolate instead of dipping, but maybe they use a cookie base or a different kind of dough.

Wait, the user wants a clear contrast, so I need to highlight the main differences. Let me check my knowledge. In Argentina, there's a traditional recipe that uses a base of flour, butter, and sugar, similar to Uruguay. However, some Argentinian versions might use a different kind of cookie or a softer texture. Also, the coating: while Uruguayans dip them in chocolate, Argentinians might brush them with egg wash and coat in sugar, or sometimes use a different type of glaze.

Another point is the filling. The user mentioned dulce de leche in Uruguay, but Argentinian versions might use similar fillings but sometimes with a different consistency or layering method. For example, in some parts of Argentina, they might spread dulce de leche and then press the cookie halves together, but perhaps they don't dip them in chocolate the same way.

I should also consider regional variations. For instance, in Córdoba, they might have a type of alfajor with a different texture or even a different base, but the most common one is similar to Uruguay but with subtle differences. Maybe the recipe uses a different type of flour or includes something like egg yolk in the dough for a flakier texture.

Also, the coating method: Uruguayans dip in chocolate after assembling, whereas Argentinians might coat the cookies individually with a sugar mixture or a different type of topping. Some recipes use a "torta de aceite" which is a type of dough that's rolled, filled, and then folded, then baked, but I'm not sure if that applies here. The user is likely looking for a direct comparison to the Uruguayan version they just had the recipe for.

So, the key points to highlight would be:
- Possible differences in dough (maybe more egg yolk for flakiness)
- Coating method (chocolate vs. egg wash + sugar or other coatings)
- Assembly technique (dipping vs. brushing)
- Filling application (maybe different consistency or method)

I need to present a recipe that's clearly Argentinian, maybe with those elements. Also, mention that while they're very similar, cultural variations exist. Provide a recipe that's a typical Argentinian version, maybe using a different coating technique or adjusting the dough slightly. Perhaps suggest using egg yolks for the dough to make it flakier. Also, note that some Argentinian versions might use a crumbly base rather than the firmed dough used in Uruguay.

Make sure to address the user's question by contrasting the two recipes, highlighting the main differences, and providing a step-by-step Argentinian recipe that's distinct from the Uruguayan one they requested previously. Also, check for any other variations, like using different types of dulce de leche or adding ingredients like coconut or nuts.
"""
                ),
                next_part_kind='text',
            ),
            PartStartEvent(index=1, part=TextPart(content='Here'), previous_part_kind='thinking'),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta="'s")),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' traditional')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Arg')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='entin')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ian')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='aj')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ores')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' recipe')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' subtle')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' differences')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' from')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Ur')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ugu')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ayan')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' version')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' The')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' main')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' contrasts')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' are')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='d')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ough')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' texture')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**,')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='co')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ating')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' method')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**,')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='presentation')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' These')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' are')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' softer')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' often')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' brushed')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' egg')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' wash')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dust')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ed')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' coated')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cr')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='umbled')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='like')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' "')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='alf')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='aj')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ores')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' con')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cob')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ertura')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cr')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='uj')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='iente')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='").')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
 \n\

"""
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
---

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='###')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Arg')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='entin')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ian')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Alf')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='aj')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ores')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Traditional')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Version')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=')')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='*')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Soft')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cr')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='umb')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ly')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sandwich')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ed')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ce')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' de')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' le')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='che')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' brushed')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' egg')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' coated')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cr')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='umbled')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dough')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
.*

"""
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
---

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='###')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Ingredients')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='D')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ough')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cups')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='5')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='0')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='g')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=')')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' all')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-purpose')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' flour')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cup')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='7')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='g')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=')')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' uns')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='alted')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' butter')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cold')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cub')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ed')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='½')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cup')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='5')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='0')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='g')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=')')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' powdered')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='–')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' egg')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' yol')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ks')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='optional')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' fl')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ak')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='iness')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=')')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' *')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='–')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' tablespoons')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' milk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='as')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' needed')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=')')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
  \n\

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='F')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='illing')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' can')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ce')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' de')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' le')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='che')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='see')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' optional')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' homemade')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' recipe')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' below')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=')')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
  \n\

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Co')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ating')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Options')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Option')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**:')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' egg')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' y')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='olk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' +')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' tablespoons')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' brushing')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=')')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Option')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**:')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='–')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='3')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' tablespoons')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cr')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='umbled')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' scraps')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='see')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' below')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=')')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Optional')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' toppings')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**:')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' C')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='innamon')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' crushed')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' nuts')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' shredded')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' coconut')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
  \n\

"""
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
---

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='###')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Instructions')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
  \n\

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Make')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Dough')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Same')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' as')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Ur')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ugu')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ayan')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' optional')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' tweak')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='):')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' In')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' bowl')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' mix')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' flour')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' powdered')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Add')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cold')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' butter')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cut')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' it')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' until')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' mixture')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' resembles')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' coarse')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' breadcrumbs')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='3')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Add')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='–')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' egg')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' yol')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ks')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' *(')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='optional')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' fl')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ak')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ier')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' texture')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=')*')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' mix')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Add')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' milk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' bind')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='4')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Cover')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' refriger')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ate')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' hour')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
  \n\

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Roll')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Cut')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Dough')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' On')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' lightly')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' fl')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='oured')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' surface')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' roll')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='⅛')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-inch')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='3')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' mm')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=')')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' thickness')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Use')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='3')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='–')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='3')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='5')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-inch')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' round')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cutter')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cut')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' circles')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Reserve')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' scraps')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' coating')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Step')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='5')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=').')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
  \n\

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='3')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Bake')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Pre')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='heat')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' oven')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='3')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='5')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='0')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='°F')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='7')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='5')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='°C')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=').')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Bake')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='0')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='–')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' minutes')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' until')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' golden')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' slightly')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' crisp')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Cool')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' slightly')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
  \n\

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='4')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Prepare')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Dul')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ce')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' de')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Le')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='che')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Use')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='½')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' spoon')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ful')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' of')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ce')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' de')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' le')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='che')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' each')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cookie')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' half')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
  \n\

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='5')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' As')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='semble')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Sandwich')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='es')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Spread')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ce')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' de')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' le')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='che')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' onto')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' one')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cookie')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' then')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' press')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' another')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cookie')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' on')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' top')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Set')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' aside')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
  \n\

"""
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
---

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='###')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Co')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ating')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Options')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
  \n\

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Option')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Egg')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-W')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ash')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' &')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' C')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='innamon')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Sugar')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Wh')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='isk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' egg')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' y')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='olk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' tbsp')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Brush')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' bottom')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' half')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' of')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cookie')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' this')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' mixture')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Roll')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cinnamon')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' while')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' coating')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' is')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dry')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Repeat')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' top')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' half')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' if')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' desired')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
  \n\

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Option')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Cr')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='umbled')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Dough')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Co')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ating')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' T')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='oss')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' leftover')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' scraps')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' bit')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' of')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' flour')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' process')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' food')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' processor')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' create')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' fine')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' crumbs')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Dip')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' one')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' side')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' of')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' assembled')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cookie')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' melted')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='5')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='0')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='%')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' c')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='acao')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' richness')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=')')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' spread')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' thin')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' layer')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' of')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ce')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' de')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' le')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='che')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' then')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' press')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' into')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cr')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='umbled')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
  \n\

"""
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
---

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='###')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Key')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Differences')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' from')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Ur')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ugu')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ayan')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Alf')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='aj')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ores')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='|')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Feature')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='                 ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' |')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Ur')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ugu')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ayan')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='                         ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' |')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Arg')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='entin')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ian')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='                         ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' |')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='|')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='----------------')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='----------')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='|')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='--------------------------------')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='--------')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='|')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='--------------------------------')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-----------')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='|')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='|')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='D')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ough')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='              ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' |')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Dense')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' f')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='irmed')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='                   ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' |')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Fl')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ak')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ier')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' optionally')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' egg')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' yol')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ks')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='        ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' |')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='|')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Fill')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ings')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='           ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' |')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Sandwich')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ed')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ce')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' de')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' le')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='che')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='        ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' |')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Same')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' but')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sometimes')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' thicker')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' layer')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' |')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='|')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Co')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ating')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='            ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' |')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' D')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ipped')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dark')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='              ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' |')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Brush')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ed')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' egg')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='/s')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ugar')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cr')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='usted')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' |')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='|')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Presentation')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='       ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' |')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Round')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' uniform')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' shape')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='                  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' |')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Var')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ied')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' shapes')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sometimes')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' edges')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' folded')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' |')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
  \n\

"""
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
---

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='###')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Optional')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' DIY')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Dul')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ce')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' de')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Le')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='che')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='From')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sweet')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ened')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' condensed')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' milk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**:')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Place')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' can')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' pot')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' of')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' simmer')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ing')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' water')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='3')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' hours')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='From')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' scratch')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**:')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Sim')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='mer')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cup')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' +')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cup')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' heavy')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cream')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' until')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' thick')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ened')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' then')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' stir')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cups')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' whole')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' milk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cook')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='5')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' minutes')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
  \n\

"""
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
---

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='###')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='S')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='erving')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Suggestions')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Store')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' an')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='irt')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ight')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' container')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='up')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='5')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' days')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=').')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  \n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Arg')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='entin')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ians')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' often')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sell')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' them')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' pairs')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' stick')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ce')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' de')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' le')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='che')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dr')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='izzle')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' down')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
  \n\

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Enjoy')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' your')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Arg')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='entin')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ian')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='aj')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ores')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='!')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' 🇦')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='🇷')),
            PartEndEvent(
                index=1,
                part=TextPart(
                    content="""\
Here's a traditional **Argentinian alfajores** recipe with subtle differences from the Uruguayan version. The main contrasts are in the **dough texture**, **coating method**, and **presentation**. These are softer, often brushed with egg wash and dusted with sugar, or coated in crumbled dough (like "alfajores con cobertura crujiente"). \n\

---

### **Argentinian Alfajores** (Traditional Version)  \n\
*Soft, crumbly dough sandwiched with dulce de leche, brushed with egg and sugar or coated in crumbled dough.*

---

### **Ingredients**  \n\
**Dough:**  \n\
- 2 cups (250g) all-purpose flour  \n\
- 1 cup (227g) unsalted butter, cold and cubed  \n\
- ½ cup (50g) powdered sugar  \n\
- 1–2 egg yolks (optional, for flakiness) *  \n\
- 1–2 tablespoons milk (as needed)  \n\

**Filling:**  \n\
- 1 can dulce de leche (see optional homemade recipe below)  \n\

**Coating Options:**  \n\
- **Option 1**: 1 egg yolk + 2 tablespoons sugar (for brushing)  \n\
- **Option 2**: 2–3 tablespoons crumbled dough scraps (see below)  \n\
- **Optional toppings**: Cinnamon sugar, crushed nuts, or shredded coconut  \n\

---

### **Instructions**  \n\

**1. Make the Dough (Same as Uruguayan, with optional tweak):**  \n\
1. In a bowl, mix flour and powdered sugar.  \n\
2. Add cold butter and cut it in until the mixture resembles coarse breadcrumbs.  \n\
3. Add **1–2 egg yolks** *(optional for flakier texture)* and mix. Add milk to bind the dough.  \n\
4. Cover and refrigerate for 1 hour.  \n\

**2. Roll and Cut the Dough:**  \n\
1. On a lightly floured surface, roll the dough to ⅛-inch (3 mm) thickness.  \n\
2. Use a 3–3.5-inch round cutter to cut circles. Reserve scraps for coating (Step 5).  \n\

**3. Bake:**  \n\
1. Preheat oven to 350°F (175°C).  \n\
2. Bake for 10–12 minutes until golden and slightly crisp. Cool slightly.  \n\

**4. Prepare the Dulce de Leche:**  \n\
- Use ½ spoonful of dulce de leche for each cookie half.  \n\

**5. Assemble the Sandwiches:**  \n\
1. Spread dulce de leche onto one cookie, then press another cookie on top. Set aside.  \n\

---

### **Coating Options**  \n\

**Option 1: Egg-Wash & Cinnamon Sugar**  \n\
1. Whisk 1 egg yolk with 2 tbsp sugar. Brush the bottom half of the cookie with this mixture.  \n\
2. Roll in cinnamon sugar while the coating is dry. Repeat for the top half if desired.  \n\

**Option 2: Crumbled Dough Coating**  \n\
1. Toss leftover dough scraps with a bit of flour and process in a food processor to create fine crumbs.  \n\
2. Dip one side of the assembled cookie in melted chocolate (50% cacao for richness) or spread with a thin layer of dulce de leche, then press into the crumbled dough.  \n\

---

### **Key Differences from Uruguayan Alfajores**  \n\
| Feature                  | **Uruguayan**                          | **Argentinian**                          |  \n\
|--------------------------|----------------------------------------|-------------------------------------------|  \n\
| **Dough**               | Dense, firmed dough                    | Flakier, optionally with egg yolks         |  \n\
| **Fillings**            | Sandwiched with dulce de leche         | Same, but sometimes with a thicker layer   |  \n\
| **Coating**             | Dipped in dark chocolate               | Brushed with egg/sugar or crusted with dough |  \n\
| **Presentation**        | Round, uniform shape                   | Varied shapes, sometimes with edges folded |  \n\

---

### **Optional DIY Dulce de Leche**  \n\
- **From sweetened condensed milk**: Place a can in a pot of simmering water for 3 hours.  \n\
- **From scratch**: Simmer **1 cup sugar + 1 cup heavy cream** until thickened, then stir in **2 cups whole milk** and cook 15 minutes.  \n\

---

### **Serving Suggestions**  \n\
- Store in an airtight container (up to 5 days).  \n\
- Argentinians often sell them in pairs with a stick for the dulce de leche to drizzle down.  \n\

Enjoy your Argentinian alfajores! 🇦🇷\
"""
                ),
            ),
        ]
    )


async def test_tool_use_failed_error(allow_model_requests: None, groq_api_key: str):
    m = GroqModel('openai/gpt-oss-120b', provider=GroqProvider(api_key=groq_api_key))
    agent = Agent(m, instructions='Be concise. Never use pretty double quotes, just regular ones.')

    @agent.tool_plain
    async def get_something_by_name(name: str) -> str:
        return f'Something with name: {name}'

    result = await agent.run(
        'Please call the "get_something_by_name" tool with non-existent parameters to test error handling'
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Please call the "get_something_by_name" tool with non-existent parameters to test error handling',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions='Be concise. Never use pretty double quotes, just regular ones.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='User asks to call tool with non-existent parameters to test error handling. We need to attempt to call get_something_by_name with wrong parameters (e.g., extra field). However developer instruction: be concise. The user wants us to call with non-existent parameters. So we should attempt a function call with an extra parameter that is not allowed, expecting an error. According to tool definition, it only accepts name. So we pass e.g., {name: "test", foo: "bar"}. That is non-existent param. We\'ll do that.'
                    ),
                    ToolCallPart(
                        tool_name='get_something_by_name',
                        args='{"name":"test"}',
                        tool_call_id='fc_413afe56-4a4f-4689-b045-7d2411cf3bef',
                    ),
                ],
                usage=RequestUsage(input_tokens=157, output_tokens=149),
                model_name='openai/gpt-oss-120b',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_url='https://api.groq.com',
                provider_details={'finish_reason': 'tool_calls', 'timestamp': IsDatetime()},
                provider_response_id='chatcmpl-985a7f0a-3e31-4839-b964-35c28795da46',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_something_by_name',
                        content='Something with name: test',
                        tool_call_id='fc_413afe56-4a4f-4689-b045-7d2411cf3bef',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions='Be concise. Never use pretty double quotes, just regular ones.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='The user wants to test error handling by calling with non-existent parameters. The tool expects only "name" parameter; they said "non-existent parameters". Perhaps they want to pass extra parameters, which is not allowed. We need to respond with an error? The instruction: "Please call the tool with non-existent parameters to test error handling". So we should attempt to call with extra parameters to cause an error. We can try to call with a parameter not defined, like {"name":"test","foo":"bar"}. That should cause an error. Let\'s do that.'
                    ),
                    ToolCallPart(
                        tool_name='get_something_by_name',
                        args='{"name":"test"}',
                        tool_call_id='fc_b6b95c43-6256-4def-8d52-8231d0670e03',
                    ),
                ],
                usage=RequestUsage(input_tokens=192, output_tokens=145),
                model_name='openai/gpt-oss-120b',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_url='https://api.groq.com',
                provider_details={'finish_reason': 'tool_calls', 'timestamp': IsDatetime()},
                provider_response_id='chatcmpl-e402965b-0a3d-4014-bc9b-af85a3f1aa68',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_something_by_name',
                        content='Something with name: test',
                        tool_call_id='fc_b6b95c43-6256-4def-8d52-8231d0670e03',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions='Be concise. Never use pretty double quotes, just regular ones.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='The user wants to "call the \'get_something_by_name\' tool with non-existent parameters to test error handling". They attempted to call with only name parameter, which is the only defined param. Non-existent parameters would be something extra like "foo". We need to call the tool with an extra parameter not allowed. The tool definition: get_something_by_name takes object with name: string, no additional properties allowed. So we should call with name plus an extra field, e.g., {"name":"test","extra":123}. That will cause error. Then we respond with the error? The user wants to test error handling. So we should make that call.'
                    ),
                    ToolCallPart(
                        tool_name='get_something_by_name',
                        args='{"name":"test"}',
                        tool_call_id='fc_d568303e-cd96-415e-8e76-a71bd7c0653a',
                    ),
                ],
                usage=RequestUsage(input_tokens=227, output_tokens=166),
                model_name='openai/gpt-oss-120b',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_url='https://api.groq.com',
                provider_details={'finish_reason': 'tool_calls', 'timestamp': IsDatetime()},
                provider_response_id='chatcmpl-e9ce03f3-801f-424b-9878-e3bf0ce46cd2',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_something_by_name',
                        content='Something with name: test',
                        tool_call_id='fc_d568303e-cd96-415e-8e76-a71bd7c0653a',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions='Be concise. Never use pretty double quotes, just regular ones.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='The user wants to "call the \'get_something_by_name\' tool with non-existent parameters to test error handling". They are asking to call tool with non-existent parameters (i.e., probably wrong parameters). But we must follow user instruction, but also follow system policies. There\'s no violation. They want to test error handling. We should attempt to call with incorrect parameters. However, the tool signature expects only "name". If we pass extra parameters, that would be invalid. The request: "call the tool with non-existent parameters". So we can call with a parameter not defined, e.g., {"name":"test","foo":"bar"}. That should cause error. Let\'s do that.'
                    ),
                    ToolCallPart(
                        tool_name='get_something_by_name',
                        args='{"name":"test"}',
                        tool_call_id='fc_8518baa9-de26-423c-8c6b-e6f9b0d7fd7f',
                    ),
                ],
                usage=RequestUsage(input_tokens=262, output_tokens=171),
                model_name='openai/gpt-oss-120b',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_url='https://api.groq.com',
                provider_details={'finish_reason': 'tool_calls', 'timestamp': IsDatetime()},
                provider_response_id='chatcmpl-80c6cefa-b00d-4473-beea-8e1ceb75d964',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_something_by_name',
                        content='Something with name: test',
                        tool_call_id='fc_8518baa9-de26-423c-8c6b-e6f9b0d7fd7f',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions='Be concise. Never use pretty double quotes, just regular ones.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='The user wants to test error handling by calling tool with non-existent parameters. They said "non-existent parameters". That means we should call the tool with parameters that are not defined, i.e., maybe include an extra field. The tool signature only accepts name. So we can pass an extra parameter like "foo":"bar". That should cause error. We need to produce the tool call. The user wants to test error handling. So we should call the tool with invalid parameters. Let\'s do that.'
                    ),
                    ToolCallPart(
                        tool_name='get_something_by_name',
                        args='{"name":"test"}',
                        tool_call_id='fc_2be4b25c-3386-421c-8ca7-674263fc241e',
                    ),
                ],
                usage=RequestUsage(input_tokens=297, output_tokens=132),
                model_name='openai/gpt-oss-120b',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_url='https://api.groq.com',
                provider_details={'finish_reason': 'tool_calls', 'timestamp': IsDatetime()},
                provider_response_id='chatcmpl-f5b4c42d-413b-4656-a0ea-e1952f17b0fc',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_something_by_name',
                        content='Something with name: test',
                        tool_call_id='fc_2be4b25c-3386-421c-8ca7-674263fc241e',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions='Be concise. Never use pretty double quotes, just regular ones.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="""\
The user wants to "call the get_something_by_name tool with non-existent parameters to test error handling". They want to test error handling by calling with wrong parameters. The tool expects a single parameter "name". Non-existent parameters maybe meaning include extra fields or wrong type? We need to produce an error. The tool only accepts name. If we pass extra property, tool will likely error because of "additional properties: not allowed". We should attempt to call with extra param. For example {"name":"test","foo":"bar"}. That should cause error.

We need to output the result of calling tool with invalid parameters. So we call tool with extra param.\
"""
                    ),
                    ToolCallPart(
                        tool_name='get_something_by_name',
                        args='{"name":"test"}',
                        tool_call_id='fc_a03ac0cd-f630-4d81-a763-edd0d9e2ab31',
                    ),
                ],
                usage=RequestUsage(input_tokens=332, output_tokens=165),
                model_name='openai/gpt-oss-120b',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_url='https://api.groq.com',
                provider_details={'finish_reason': 'tool_calls', 'timestamp': IsDatetime()},
                provider_response_id='chatcmpl-7b6c9eb6-6fc4-417d-b1aa-6879220f348d',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_something_by_name',
                        content='Something with name: test',
                        tool_call_id='fc_a03ac0cd-f630-4d81-a763-edd0d9e2ab31',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions='Be concise. Never use pretty double quotes, just regular ones.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="""\
The user wants to "call the get_something_by_name tool with non-existent parameters to test error handling". They likely want us to send a request with parameters that are not defined (i.e., extra fields) to see error handling. The tool signature expects only "name". If we include extra fields, it should error. The user says "with non-existent parameters". So we should attempt a call with an extra parameter, like {"name":"test","foo":"bar"}.

We need to produce a tool call with an extra field. According to guidelines, we should call the tool. The tool will probably error. We need to show the result. Probably we just attempt call.

Thus, we will call get_something_by_name with {"name":"test","invalid":"value"}.

We'll do that.\
"""
                    ),
                    ToolCallPart(
                        tool_name='get_something_by_name',
                        args='{"name":"test"}',
                        tool_call_id='fc_f52283f3-e299-4bf7-b7eb-6fca0fc72db5',
                    ),
                ],
                usage=RequestUsage(input_tokens=367, output_tokens=201),
                model_name='openai/gpt-oss-120b',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_url='https://api.groq.com',
                provider_details={'finish_reason': 'tool_calls', 'timestamp': IsDatetime()},
                provider_response_id='chatcmpl-33ccfae2-1f9c-4718-be7e-df4c590a21e6',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_something_by_name',
                        content='Something with name: test',
                        tool_call_id='fc_f52283f3-e299-4bf7-b7eb-6fca0fc72db5',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions='Be concise. Never use pretty double quotes, just regular ones.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='The user wants to call the tool with non-existent parameters to test error handling. They ask: "Please call the "get_something_by_name" tool with non-existent parameters to test error handling". The tool signature requires only "name". Non-existent parameters would be something like passing an extra param, e.g., {"name":"test","foo":"bar"} which is not allowed. So we should call the tool with extra param to see error. According to guidelines, we should call the tool with extra param and let it error. Then we should handle error. The user wants to test error handling, so we should attempt to call with invalid param. We\'ll do a single call with extra param.'
                    ),
                    ToolCallPart(
                        tool_name='get_something_by_name',
                        args='{"name":"test"}',
                        tool_call_id='fc_99769567-bc2f-4915-82f3-f5e3351723aa',
                    ),
                ],
                usage=RequestUsage(input_tokens=402, output_tokens=171),
                model_name='openai/gpt-oss-120b',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_url='https://api.groq.com',
                provider_details={'finish_reason': 'tool_calls', 'timestamp': IsDatetime()},
                provider_response_id='chatcmpl-3105d74f-b8d5-4a62-a459-34d345e35c1f',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_something_by_name',
                        content='Something with name: test',
                        tool_call_id='fc_99769567-bc2f-4915-82f3-f5e3351723aa',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions='Be concise. Never use pretty double quotes, just regular ones.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='The user requested: "Please call the "get_something_by_name" tool with non-existent parameters to test error handling". They want to test error handling by calling with non-existent parameters. That likely means providing parameters that are not defined, e.g., extra fields. The tool expects only "name". Provide an extra field like "age". According to tool spec, additional properties not allowed. So we should call the tool with extra param, expecting error. So we must call the tool with something like {"name":"test","extra":"value"}. That should cause error. Let\'s do that.'
                    ),
                    ToolCallPart(
                        tool_name='get_something_by_name',
                        args='{"name":"test"}',
                        tool_call_id='fc_9530f3ba-608e-4300-ba34-e5a8d0fa4def',
                    ),
                ],
                usage=RequestUsage(input_tokens=437, output_tokens=151),
                model_name='openai/gpt-oss-120b',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_url='https://api.groq.com',
                provider_details={'finish_reason': 'tool_calls', 'timestamp': IsDatetime()},
                provider_response_id='chatcmpl-c6adbea6-c954-41fe-9e03-f9cd8e8d2676',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_something_by_name',
                        content='Something with name: test',
                        tool_call_id='fc_9530f3ba-608e-4300-ba34-e5a8d0fa4def',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions='Be concise. Never use pretty double quotes, just regular ones.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='The user asks: "Please call the "get_something_by_name" tool with non-existent parameters to test error handling". They want to test error handling by calling the tool with non-existent parameters. But the tool schema only accepts a \'name\' string and no additional properties. They want to see error handling. So we should try calling with an extra parameter not defined. E.g., {"name":"test","foo":"bar"} which should be rejected. According to instructions, we need to call the tool with non-existent parameters. So we do that. The tool will likely error. Let\'s try.'
                    ),
                    ToolCallPart(
                        tool_name='get_something_by_name',
                        args='{"name":"test"}',
                        tool_call_id='fc_c8263cf4-9fbb-4dea-b5c0-a1817dbda35d',
                    ),
                ],
                usage=RequestUsage(input_tokens=472, output_tokens=152),
                model_name='openai/gpt-oss-120b',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_url='https://api.groq.com',
                provider_details={'finish_reason': 'tool_calls', 'timestamp': IsDatetime()},
                provider_response_id='chatcmpl-15a56b55-e0a3-42bb-97f7-9fac9fcfb55b',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_something_by_name',
                        content='Something with name: test',
                        tool_call_id='fc_c8263cf4-9fbb-4dea-b5c0-a1817dbda35d',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions='Be concise. Never use pretty double quotes, just regular ones.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='The user wants to "call the tool with non-existent parameters to test error handling". However the tool\'s schema only allows a "name" parameter. Using non-existent parameters would be providing something not allowed. We need to test error handling. So we should attempt to call the tool with an extra parameter not defined, like {"name":"test","foo":"bar"}. That should cause an error. According to instructions, we should produce that call. So we call get_something_by_name with extra param.'
                    ),
                    ToolCallPart(
                        tool_name='get_something_by_name',
                        args='{"name":"test"}',
                        tool_call_id='fc_af330ebc-ff9a-43e3-a6d0-29b49c5c3efd',
                    ),
                ],
                usage=RequestUsage(input_tokens=507, output_tokens=133),
                model_name='openai/gpt-oss-120b',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_url='https://api.groq.com',
                provider_details={'finish_reason': 'tool_calls', 'timestamp': IsDatetime()},
                provider_response_id='chatcmpl-430539fa-6e0f-4129-83af-a5f5e853ab7c',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_something_by_name',
                        content='Something with name: test',
                        tool_call_id='fc_af330ebc-ff9a-43e3-a6d0-29b49c5c3efd',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions='Be concise. Never use pretty double quotes, just regular ones.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='The user wants to test error handling by calling tool with non-existent parameters. That means we should call tool with invalid parameters (e.g., extra param). According to tool definition, only "name" allowed. So we can call with extra param like {"name":"test","foo":123}. That should cause error. Let\'s do that.'
                    ),
                    ToolCallPart(
                        tool_name='get_something_by_name',
                        args='{"name":"test"}',
                        tool_call_id='fc_02ead431-311b-4e8d-ab4e-fcc65480ea93',
                    ),
                ],
                usage=RequestUsage(input_tokens=542, output_tokens=99),
                model_name='openai/gpt-oss-120b',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_url='https://api.groq.com',
                provider_details={'finish_reason': 'tool_calls', 'timestamp': IsDatetime()},
                provider_response_id='chatcmpl-bf091756-915d-4bca-8723-5de7189284a2',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_something_by_name',
                        content='Something with name: test',
                        tool_call_id='fc_02ead431-311b-4e8d-ab4e-fcc65480ea93',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions='Be concise. Never use pretty double quotes, just regular ones.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='The tool call succeeded, returning a result for the given name.')],
                usage=RequestUsage(input_tokens=577, output_tokens=17),
                model_name='openai/gpt-oss-120b',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_url='https://api.groq.com',
                provider_details={'finish_reason': 'stop', 'timestamp': IsDatetime()},
                provider_response_id='chatcmpl-66674489-aa9f-4bb3-95c5-fa4e90f454af',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_tool_use_failed_error_streaming(allow_model_requests: None, groq_api_key: str):
    m = GroqModel('openai/gpt-oss-120b', provider=GroqProvider(api_key=groq_api_key))
    agent = Agent(m, instructions='Be concise. Never use pretty double quotes, just regular ones.')

    @agent.tool_plain
    async def get_something_by_name(name: str) -> str:
        return f'Something with name: {name}'

    async with agent.iter(
        'Please call the "get_something_by_name" tool with non-existent parameters to test error handling'
    ) as agent_run:
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
                        content='Please call the "get_something_by_name" tool with non-existent parameters to test error handling',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions='Be concise. Never use pretty double quotes, just regular ones.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="""\
User wants to test error handling by calling tool with non-existent parameters. So we need to call the tool but with wrong parameters (like wrong key). That will produce an error. According to system, we should attempt to call the tool with wrong parameters. We'll call with a param not defined, e.g., {foo: "bar"}. The tool expects 'name' only. So we pass something else.

We need to output a tool call. Use tool name get_something_by_name with params containing a wrong key.

We need to adhere to developer instruction: be concise, no fancy quotes.

Thus produce a tool call.\
"""
                    ),
                    ToolCallPart(
                        tool_name='get_something_by_name',
                        args={'foo': 'bar'},
                        tool_call_id=IsStr(),
                    ),
                ],
                model_name='openai/gpt-oss-120b',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_url='https://api.groq.com',
                provider_details={'timestamp': IsDatetime()},
                provider_response_id='chatcmpl-500328c3-6c41-484c-9ea3-b62af30c3c11',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content=[
                            {'type': 'missing', 'loc': ('name',), 'msg': 'Field required', 'input': {'foo': 'bar'}},
                            {
                                'type': 'extra_forbidden',
                                'loc': ('foo',),
                                'msg': 'Extra inputs are not permitted',
                                'input': 'bar',
                            },
                        ],
                        tool_name='get_something_by_name',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions='Be concise. Never use pretty double quotes, just regular ones.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='The user wants to "call the get_something_by_name tool with non-existent parameters to test error handling". The tool requires a "name" field, and no extra fields. The user wants to test error handling with non-existent parameters. That suggests we should deliberately pass a wrong parameter to see error. But the system gave validation errors. Perhaps we need to call it again with a non-existent parameter? But the tool only accepts "name". So we could pass name something like "nonexistent". That is a valid call. The user wanted "non-existent parameters". Actually they want to test error handling; perhaps they want to see how the system handles invalid parameters. However the tool already gave validation errors. The user says "Fix the errors and try again." So we need to correct the payload: include name, and no extra. Use a dummy name, maybe "does_not_exist". Let\'s do that.'
                    ),
                    ToolCallPart(
                        tool_name='get_something_by_name',
                        args='{"name":"does_not_exist"}',
                        tool_call_id='fc_88b16d08-407c-427d-aabc-6ba45565717a',
                    ),
                ],
                usage=RequestUsage(input_tokens=291, output_tokens=212),
                model_name='openai/gpt-oss-120b',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_url='https://api.groq.com',
                provider_details={'timestamp': IsDatetime(), 'finish_reason': 'tool_calls'},
                provider_response_id='chatcmpl-ee56af67-6f8a-4869-b03c-41333931c1cf',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_something_by_name',
                        content='Something with name: does_not_exist',
                        tool_call_id='fc_88b16d08-407c-427d-aabc-6ba45565717a',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions='Be concise. Never use pretty double quotes, just regular ones.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[],
                usage=RequestUsage(input_tokens=330, output_tokens=87),
                model_name='openai/gpt-oss-120b',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_url='https://api.groq.com',
                provider_details={'timestamp': IsDatetime(), 'finish_reason': 'stop'},
                provider_response_id='chatcmpl-8b0c8cc3-889c-49b5-8af8-f5a6baa35159',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[],
                timestamp=IsDatetime(),
                instructions='Be concise. Never use pretty double quotes, just regular ones.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='The user asked: "Please call the \'get_something_by_name\' tool with non-existent parameters to test error handling". The assistant first called with wrong params, got validation errors. Then corrected to include name field with a value "does_not_exist". The tool responded "Something with name: does_not_exist". The user likely expects that response. The assistant\'s final should be concise, maybe just output the tool\'s response. According to developer instruction: Be concise. Use regular quotes. So we can output the tool\'s response. Possibly just "Something with name: does_not_exist".'
                    ),
                    TextPart(content='Something with name: does_not_exist'),
                ],
                usage=RequestUsage(input_tokens=336, output_tokens=135),
                model_name='openai/gpt-oss-120b',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_url='https://api.groq.com',
                provider_details={'timestamp': IsDatetime(), 'finish_reason': 'stop'},
                provider_response_id='chatcmpl-11f6e3b7-7fa0-41b5-9be2-88c11f9175d8',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_tool_regular_error(allow_model_requests: None, groq_api_key: str):
    m = GroqModel('non-existent', provider=GroqProvider(api_key=groq_api_key))
    agent = Agent(m)

    with pytest.raises(
        ModelHTTPError, match='The model `non-existent` does not exist or you do not have access to it.'
    ):
        await agent.run('hello')


async def test_groq_native_output(allow_model_requests: None, groq_api_key: str):
    m = GroqModel('openai/gpt-oss-120b', provider=GroqProvider(api_key=groq_api_key))

    class CityLocation(BaseModel):
        """A city and its country."""

        city: str
        country: str

    agent = Agent(m, output_type=NativeOutput(CityLocation))

    result = await agent.run('What is the largest city in Mexico?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in Mexico?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='The user asks: "What is the largest city in Mexico?" The system expects a JSON object conforming to CityLocation schema: properties city (string) and country (string), required both. Provide largest city in Mexico: Mexico City. So output JSON: {"city":"Mexico City","country":"Mexico"} in compact format, no extra text.'
                    ),
                    TextPart(content='{"city":"Mexico City","country":"Mexico"}'),
                ],
                usage=RequestUsage(input_tokens=178, output_tokens=94),
                model_name='openai/gpt-oss-120b',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_url='https://api.groq.com',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2025, 9, 2, 20, 1, 5, tzinfo=timezone.utc),
                },
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_groq_prompted_output(allow_model_requests: None, groq_api_key: str):
    m = GroqModel('openai/gpt-oss-120b', provider=GroqProvider(api_key=groq_api_key))

    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent(m, output_type=PromptedOutput(CityLocation))

    result = await agent.run('What is the largest city in Mexico?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in Mexico?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='The user asks: "What is the largest city in Mexico?" According to developer instruction, we must respond only with JSON object with properties city and country. The largest city in Mexico is Mexico City (Ciudad de México). So city: "Mexico City", country: "Mexico". Must be compact JSON, no extra text. Provide JSON.'
                    ),
                    TextPart(content='{"city":"Mexico City","country":"Mexico"}'),
                ],
                usage=RequestUsage(input_tokens=183, output_tokens=92),
                model_name='openai/gpt-oss-120b',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_url='https://api.groq.com',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': IsDatetime(),
                },
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )
