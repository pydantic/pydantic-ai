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
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    FinalResultEvent,
    ImageUrl,
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
                parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='world')],
                model_name='llama-3.3-70b-versatile-123',
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                provider_name='groq',
                provider_details={'finish_reason': 'stop'},
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='world')],
                model_name='llama-3.3-70b-versatile-123',
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                provider_name='groq',
                provider_details={'finish_reason': 'stop'},
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
                parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
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
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
                provider_name='groq',
                provider_details={'finish_reason': 'stop'},
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
                    SystemPromptPart(content='this is the system prompt', timestamp=IsNow(tz=timezone.utc)),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
                ],
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
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                provider_name='groq',
                provider_details={'finish_reason': 'stop'},
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
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
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
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                provider_name='groq',
                provider_details={'finish_reason': 'stop'},
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
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='final response')],
                model_name='llama-3.3-70b-versatile-123',
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                provider_name='groq',
                provider_details={'finish_reason': 'stop'},
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
        assert [c async for c in result.stream_output(debounce_by=None)] == snapshot(['hello ', 'hello world'])
        assert result.is_complete


async def test_stream_text_finish_reason(allow_model_requests: None):
    stream = text_chunk('hello '), text_chunk('world'), text_chunk('.', finish_reason='stop')
    mock_client = MockGroq.create_mock_stream(stream)
    m = GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(groq_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_output(debounce_by=None)] == snapshot(
            ['hello ', 'hello world', 'hello world.']
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
            [{}, {'first': 'One'}, {'first': 'One', 'second': 'Two'}, {'first': 'One', 'second': 'Two'}]
        )
        assert result.is_complete

    assert result.usage() == snapshot(RunUsage(requests=1))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='', timestamp=IsNow(tz=timezone.utc))],
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
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                provider_name='groq',
                provider_response_id='x',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
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
            [{'first': 'One'}, {'first': 'One', 'second': 'Two'}, {'first': 'One', 'second': 'Two'}]
        )
        assert result.is_complete


async def test_no_delta(allow_model_requests: None):
    stream = chunk([]), text_chunk('hello '), text_chunk('world')
    mock_client = MockGroq.create_mock_stream(stream)
    m = GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(groq_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_output(debounce_by=None)] == snapshot(['hello ', 'hello world'])
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
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_image', args='{}', tool_call_id='call_wkpd')],
                usage=RequestUsage(input_tokens=192, output_tokens=8),
                model_name='meta-llama/llama-4-scout-17b-16e-instruct',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_details={'finish_reason': 'tool_calls'},
                provider_response_id='chatcmpl-3c327c89-e9f5-4aac-a5d5-190e6f6f25c9',
                finish_reason='tool_call',
                run_id=IsStr(),
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
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='The fruit in the image is a kiwi.')],
                usage=RequestUsage(input_tokens=2552, output_tokens=11),
                model_name='meta-llama/llama-4-scout-17b-16e-instruct',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_details={'finish_reason': 'stop'},
                provider_response_id='chatcmpl-82dfad42-6a28-4089-82c3-c8633f626c0d',
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
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='The capital of France is Paris.')],
                usage=RequestUsage(input_tokens=48, output_tokens=8),
                model_name='llama-3.3-70b-versatile',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_details={'finish_reason': 'stop'},
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
                provider_details={'finish_reason': 'stop'},
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
                provider_details={'finish_reason': 'stop'},
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
    m = GroqModel('deepseek-r1-distill-llama-70b', provider=GroqProvider(api_key=groq_api_key))
    settings = GroqModelSettings(groq_reasoning_format='raw')
    agent = Agent(m, instructions='You are a chef.', model_settings=settings)

    result = await agent.run('I want a recipe to cook Uruguayan alfajores.')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='I want a recipe to cook Uruguayan alfajores.', timestamp=IsDatetime())],
                instructions='You are a chef.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[IsInstance(ThinkingPart), IsInstance(TextPart)],
                usage=RequestUsage(input_tokens=21, output_tokens=1414),
                model_name='deepseek-r1-distill-llama-70b',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_details={'finish_reason': 'stop'},
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
                instructions='You are a chef.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[IsInstance(ThinkingPart), IsInstance(TextPart)],
                usage=RequestUsage(input_tokens=21, output_tokens=1414),
                model_name='deepseek-r1-distill-llama-70b',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_details={'finish_reason': 'stop'},
                provider_response_id='chatcmpl-9748c1af-1065-410a-969a-d7fb48039fbb',
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
                instructions='You are a chef.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[IsInstance(ThinkingPart), IsInstance(TextPart)],
                usage=RequestUsage(input_tokens=524, output_tokens=1590),
                model_name='deepseek-r1-distill-llama-70b',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_details={'finish_reason': 'stop'},
                provider_response_id='chatcmpl-994aa228-883a-498c-8b20-9655d770b697',
                finish_reason='stop',
                run_id=IsStr(),
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
                instructions='You are a chef.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="""\

Okay, so I want to make Uruguayan alfajores. I've heard they're a type of South American cookie sandwich with dulce de leche. I'm not entirely sure about the exact steps, but I can try to figure it out based on what I know.

First, I think alfajores are cookies, so I'll need to make the cookie part. From what I remember, the dough is probably made with flour, sugar, butter, eggs, vanilla, and maybe some baking powder or baking soda. I should look up a typical cookie dough recipe and adjust it for alfajores.

Once the dough is ready, I'll need to roll it out and cut into circles. I've seen people use a cookie cutter or even the rim of a glass. The thickness should be medium, not too thin to break easily.

Baking them in the oven, I suppose at around 350°F for about 10-15 minutes until they're lightly golden. I should keep an eye on them to make sure they don't burn.

After the cookies are baked and cooled, the next step is the dulce de leche filling. I can either make it from scratch or use store-bought. If I make it, I'll need to heat condensed milk until it thickens and turns golden. That might take some time, so I need to be patient and stir frequently to avoid burning.

Then, I'll sandwich two cookies together with the dulce de leche in the middle. I think pressing them gently is important so they stick together without breaking.

Finally, I've seen alfajores coated in powdered sugar. So, after assembling, I'll roll each sandwich in powdered sugar to coat them evenly. That should give them the classic look and extra sweetness.

Wait, I should make sure the cookies are completely cool before filling, otherwise the dulce de leche might melt or the cookies could become soggy. Also, maybe I can add a pinch of salt to balance the sweetness. Oh, and the vanilla extract is important for flavor.

I might have missed something, but this seems like a good start. I'll follow the steps, and if something doesn't turn out right, I can adjust next time.
"""
                    ),
                    TextPart(
                        content="""\
To make Uruguayan alfajores, follow these organized steps for a delightful cookie sandwich with dulce de leche:

### Ingredients:
- **For the Cookies:**
  - 2 cups all-purpose flour
  - 1 cup powdered sugar
  - 1/2 tsp baking powder
  - 1/4 tsp baking soda
  - 1/4 tsp salt
  - 1/2 cup unsalted butter, softened
  - 1 large egg
  - 1 egg yolk
  - 1 tsp vanilla extract

- **For the Filling:**
  - 1 can (14 oz) sweetened condensed milk (for dulce de leche)
  - Powdered sugar (for coating)

### Instructions:

1. **Prepare the Cookie Dough:**
   - In a large bowl, whisk together flour, powdered sugar, baking powder, baking soda, and salt.
   - Add softened butter and mix until the mixture resembles coarse crumbs.
   - In a separate bowl, whisk together egg, egg yolk, and vanilla extract. Pour into the dry mixture and mix until a dough forms.
   - Wrap dough in plastic wrap and refrigerate for 30 minutes.

2. **Roll and Cut Cookies:**
   - Roll out dough on a floured surface to about 1/4 inch thickness.
   - Cut into circles using a cookie cutter or glass rim.
   - Place cookies on a parchment-lined baking sheet, leaving space between each.

3. **Bake the Cookies:**
   - Preheat oven to 350°F (180°C).
   - Bake for 10-15 minutes until lightly golden. Allow to cool on the baking sheet for 5 minutes, then transfer to a wire rack to cool completely.

4. **Make Dulce de Leche:**
   - Pour sweetened condensed milk into a saucepan and heat over medium heat, stirring frequently, until thickened and golden (about 10-15 minutes).

5. **Assemble Alfajores:**
   - Spread a layer of dulce de leche on the flat side of one cookie. Sandwich with another cookie, pressing gently.
   - Roll each sandwich in powdered sugar to coat evenly.

6. **Serve:**
   - Enjoy your alfajores with a dusting of powdered sugar. Store in an airtight container.

### Tips:
- Ensure cookies are completely cool before filling to prevent sogginess.
- For an extra touch, add a pinch of salt to the dough for flavor balance.

Enjoy your homemade Uruguayan alfajores!\
"""
                    ),
                ],
                usage=RequestUsage(input_tokens=21, output_tokens=988),
                model_name='deepseek-r1-distill-llama-70b',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_details={'finish_reason': 'stop'},
                provider_response_id='chatcmpl-4ef92b12-fb9d-486f-8b98-af9b5ecac736',
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
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' so')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' want')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' make')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Ur')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ugu')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ayan')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='aj')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ores')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'ve")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' heard')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'re")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' type')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' South')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' American')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookie')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sandwich')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'m")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' not')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' entirely')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sure')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' about')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' exact')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' steps')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' but')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' can')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' try')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' figure')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' out')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' based')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' on')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' what')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' know')),
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
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' think')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='aj')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ores')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' are')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookies')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' so')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'ll")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' need')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' make')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookie')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' part')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' From')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' what')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' remember')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' probably')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' made')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' flour')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' butter')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' eggs')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' vanilla')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' some')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' baking')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' powder')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' baking')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' soda')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' should')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' look')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' up')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' typical')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookie')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' recipe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' adjust')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='aj')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ores')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Once')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ready')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'ll")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' need')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' roll')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' out')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cut')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' into')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' circles')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'ve")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' seen')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' people')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookie')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cutter')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' even')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' rim')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' glass')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' The')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' thickness')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' should')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' be')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' medium')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' not')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' too')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' thin')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' break')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' easily')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='B')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='aking')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' them')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' oven')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' suppose')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' at')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' around')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='350')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='°F')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' about')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='10')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='15')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' minutes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' until')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'re")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' lightly')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' golden')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' should')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' keep')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' an')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' eye')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' on')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' them')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' make')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sure')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' don')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'t")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' burn')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='After')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookies')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' are')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' baked')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cooled')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' next')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' step')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' filling')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' can')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' either')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' make')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' from')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' scratch')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' store')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-b')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ought')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' If')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' make')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'ll")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' need')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' heat')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' condensed')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' milk')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' until')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' thick')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ens')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' turns')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' golden')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' That')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' might')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' take')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' some')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' time')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' so')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' need')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' be')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' patient')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' stir')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' frequently')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' avoid')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' burning')),
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
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'ll")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sandwich')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' two')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookies')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' together')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' middle')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' think')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' pressing')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' them')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' gently')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' important')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' so')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' stick')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' together')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' without')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' breaking')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Finally')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'ve")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' seen')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='aj')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ores')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coated')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' powdered')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' So')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' after')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' assembling')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'ll")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' roll')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' each')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sandwich')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' powdered')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coat')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' them')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' evenly')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' That')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' should')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' give')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' them')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' classic')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' look')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' extra')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sweetness')),
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
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' should')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' make')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sure')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookies')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' are')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' completely')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cool')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' before')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' filling')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' otherwise')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' might')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' melt')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookies')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' could')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' become')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sog')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='gy')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Also')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' can')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' add')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' pinch')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' salt')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' balance')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sweetness')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Oh')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' vanilla')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' extract')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' important')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' flavor')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' might')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' have')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' missed')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' something')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' but')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' this')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' seems')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' like')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' good')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' start')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'ll")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' follow')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' steps')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' if')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' something')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'t")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' turn')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' out')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' right')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' can')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' adjust')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' next')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' time')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.\n')),
            PartEndEvent(
                index=0,
                part=ThinkingPart(
                    content="""\

Okay, so I want to make Uruguayan alfajores. I've heard they're a type of South American cookie sandwich with dulce de leche. I'm not entirely sure about the exact steps, but I can try to figure it out based on what I know.

First, I think alfajores are cookies, so I'll need to make the cookie part. From what I remember, the dough is probably made with flour, sugar, butter, eggs, vanilla, and maybe some baking powder or baking soda. I should look up a typical cookie dough recipe and adjust it for alfajores.

Once the dough is ready, I'll need to roll it out and cut into circles. I've seen people use a cookie cutter or even the rim of a glass. The thickness should be medium, not too thin to break easily.

Baking them in the oven, I suppose at around 350°F for about 10-15 minutes until they're lightly golden. I should keep an eye on them to make sure they don't burn.

After the cookies are baked and cooled, the next step is the dulce de leche filling. I can either make it from scratch or use store-bought. If I make it, I'll need to heat condensed milk until it thickens and turns golden. That might take some time, so I need to be patient and stir frequently to avoid burning.

Then, I'll sandwich two cookies together with the dulce de leche in the middle. I think pressing them gently is important so they stick together without breaking.

Finally, I've seen alfajores coated in powdered sugar. So, after assembling, I'll roll each sandwich in powdered sugar to coat them evenly. That should give them the classic look and extra sweetness.

Wait, I should make sure the cookies are completely cool before filling, otherwise the dulce de leche might melt or the cookies could become soggy. Also, maybe I can add a pinch of salt to balance the sweetness. Oh, and the vanilla extract is important for flavor.

I might have missed something, but this seems like a good start. I'll follow the steps, and if something doesn't turn out right, I can adjust next time.
"""
                ),
                next_part_kind='text',
            ),
            PartStartEvent(index=1, part=TextPart(content='To'), previous_part_kind='thinking'),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' make')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Ur')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ugu')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ayan')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='aj')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ores')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' follow')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' these')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' organized')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' steps')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' delightful')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cookie')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sandwich')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ce')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' de')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' le')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='che')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
:

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='###')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Ingredients')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='For')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Cookies')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cups')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' all')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-purpose')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' flour')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cup')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' powdered')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='/')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' tsp')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' baking')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' powder')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='/')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='4')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' tsp')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' baking')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' soda')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='/')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='4')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' tsp')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' salt')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='/')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cup')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' uns')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='alted')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' butter')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' softened')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' large')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' egg')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' egg')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' y')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='olk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' tsp')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' vanilla')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' extract')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\


"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='For')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' F')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='illing')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' can')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='14')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' oz')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=')')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sweet')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ened')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' condensed')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' milk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ce')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' de')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' le')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='che')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=')\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Powder')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ed')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' coating')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
)

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='###')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Instructions')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
:

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Prepare')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Cookie')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Dough')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' In')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' large')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' bowl')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' whisk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' together')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' flour')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' powdered')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' baking')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' powder')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' baking')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' soda')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' salt')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Add')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' softened')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' butter')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' mix')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' until')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' mixture')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' resembles')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' coarse')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' crumbs')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' In')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' separate')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' bowl')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' whisk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' together')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' egg')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' egg')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' y')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='olk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' vanilla')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' extract')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Pour')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' into')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dry')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' mixture')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' mix')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' until')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' forms')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Wrap')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' plastic')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' wrap')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' refriger')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ate')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='30')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' minutes')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Roll')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Cut')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Cookies')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Roll')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' out')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' on')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' fl')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='oured')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' surface')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' about')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='/')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='4')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' inch')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' thickness')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Cut')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' into')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' circles')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' using')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cookie')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cutter')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' glass')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' rim')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Place')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cookies')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' on')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' parchment')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-lined')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' baking')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sheet')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' leaving')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' space')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' between')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' each')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='3')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='B')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ake')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Cookies')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Pre')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='heat')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' oven')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='350')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='°F')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='180')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='°C')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=').\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Bake')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='10')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='15')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' minutes')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' until')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' lightly')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' golden')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Allow')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cool')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' on')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' baking')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sheet')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='5')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' minutes')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' then')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' transfer')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' wire')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' rack')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cool')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' completely')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='4')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Make')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Dul')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ce')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' de')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Le')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='che')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Pour')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sweet')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ened')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' condensed')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' milk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' into')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sauce')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='pan')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' heat')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' over')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' medium')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' heat')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' stirring')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' frequently')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' until')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' thick')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ened')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' golden')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='about')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='10')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='15')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' minutes')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
).

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='5')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='As')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='semble')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Alf')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='aj')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ores')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Spread')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' layer')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' of')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ce')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' de')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' le')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='che')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' on')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' flat')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' side')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' of')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' one')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cookie')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Sandwich')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' another')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cookie')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' pressing')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' gently')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Roll')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' each')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sandwich')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' powdered')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' coat')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' evenly')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='6')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Serve')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Enjoy')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' your')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='aj')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ores')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dust')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ing')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' of')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' powdered')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Store')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' an')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='irt')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ight')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' container')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='###')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Tips')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Ensure')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cookies')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' are')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' completely')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cool')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' before')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' filling')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' prevent')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' s')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ogg')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='iness')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' For')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' an')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' extra')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' touch')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' add')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' pinch')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' of')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' salt')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' flavor')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' balance')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Enjoy')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' your')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' homemade')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Ur')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ugu')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ayan')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='aj')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ores')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='!')),
            PartEndEvent(
                index=1,
                part=TextPart(
                    content="""\
To make Uruguayan alfajores, follow these organized steps for a delightful cookie sandwich with dulce de leche:

### Ingredients:
- **For the Cookies:**
  - 2 cups all-purpose flour
  - 1 cup powdered sugar
  - 1/2 tsp baking powder
  - 1/4 tsp baking soda
  - 1/4 tsp salt
  - 1/2 cup unsalted butter, softened
  - 1 large egg
  - 1 egg yolk
  - 1 tsp vanilla extract

- **For the Filling:**
  - 1 can (14 oz) sweetened condensed milk (for dulce de leche)
  - Powdered sugar (for coating)

### Instructions:

1. **Prepare the Cookie Dough:**
   - In a large bowl, whisk together flour, powdered sugar, baking powder, baking soda, and salt.
   - Add softened butter and mix until the mixture resembles coarse crumbs.
   - In a separate bowl, whisk together egg, egg yolk, and vanilla extract. Pour into the dry mixture and mix until a dough forms.
   - Wrap dough in plastic wrap and refrigerate for 30 minutes.

2. **Roll and Cut Cookies:**
   - Roll out dough on a floured surface to about 1/4 inch thickness.
   - Cut into circles using a cookie cutter or glass rim.
   - Place cookies on a parchment-lined baking sheet, leaving space between each.

3. **Bake the Cookies:**
   - Preheat oven to 350°F (180°C).
   - Bake for 10-15 minutes until lightly golden. Allow to cool on the baking sheet for 5 minutes, then transfer to a wire rack to cool completely.

4. **Make Dulce de Leche:**
   - Pour sweetened condensed milk into a saucepan and heat over medium heat, stirring frequently, until thickened and golden (about 10-15 minutes).

5. **Assemble Alfajores:**
   - Spread a layer of dulce de leche on the flat side of one cookie. Sandwich with another cookie, pressing gently.
   - Roll each sandwich in powdered sugar to coat evenly.

6. **Serve:**
   - Enjoy your alfajores with a dusting of powdered sugar. Store in an airtight container.

### Tips:
- Ensure cookies are completely cool before filling to prevent sogginess.
- For an extra touch, add a pinch of salt to the dough for flavor balance.

Enjoy your homemade Uruguayan alfajores!\
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
                instructions='You are a chef.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="""\
Alright, so I'm trying to figure out how to make Argentinian alfajores. I know that Uruguayan alfajores are these delicious cookie sandwiches filled with dulce de leche and coated in powdered sugar. But I heard that Argentinian alfajores are a bit different. I'm not exactly sure what makes them unique, so I need to look into that.

First, I think about what I know about Argentinian desserts. They have a rich tradition of sweet treats, and alfajores are definitely one of them. Maybe the difference lies in the type of cookies used or the filling. I recall that in some South American countries, alfajores can be more like a biscuit or even a cake-like cookie, whereas in others, they might be crisper.

I also remember that sometimes alfajores are coated in chocolate instead of just powdered sugar. That could be an Argentinian twist. I need to confirm that. Also, the filling might not just be dulce de leche; perhaps they use other ingredients like jam or chocolate ganache.

Another thing to consider is the texture of the cookies. Uruguayan alfajores have a softer, more delicate cookie, while Argentinian ones might be crunchier. Or maybe they use a different type of flour or baking technique. I should check recipes from both countries to see the differences in ingredients and preparation methods.

I also wonder about the history of alfajores in Argentina. They might have been influenced by European immigrants, especially from Spain or Italy, which could explain variations in the recipe. This cultural influence might contribute to differences in how the cookies are made and filled.

Additionally, I think about the assembly of the alfajores. In Uruguay, it's typically two cookies sandwiching the dulce de leche and then coated in powdered sugar. Maybe in Argentina, they add more layers or use a different coating, like cinnamon or cocoa powder mixed with sugar.

I also need to consider the availability of ingredients. Dulce de leche is a staple in many South American countries, but maybe in Argentina, they have a slightly different version of it or use it in combination with other fillings. Perhaps they also use nuts or other ingredients in the dough for added texture and flavor.

Another aspect is the baking process. The Uruguayan cookies might be baked until just set, while Argentinian ones could be baked longer for a crisper texture. Or perhaps they use a different leavening agent to achieve a lighter or denser cookie.

I also think about the size of the cookies. Are Argentinian alfajores larger or smaller than the Uruguayan ones? This could affect baking time and the overall appearance of the final product.

Furthermore, I recall that in some regions, alfajores are dipped in chocolate after being filled. This could be a distinguishing feature of the Argentinian version. The chocolate coating might be milk, dark, or even white chocolate, adding another layer of flavor to the cookies.

I also wonder about the storage and serving of Argentinian alfajores. Maybe they are best served fresh, or perhaps they can be stored for a few days like the Uruguayan ones. Understanding this can help in planning the baking and assembly process.

Lastly, I think about potential variations within Argentina itself. Different regions might have their own take on alfajores, so there could be multiple authentic Argentinian recipes. It would be helpful to find a classic or widely recognized version to ensure authenticity.

Overall, to cook Argentinian alfajores, I need to focus on the specific characteristics that distinguish them from their Uruguayan counterparts, whether it's the type of cookie, the filling, the coating, or the baking method. By identifying these differences, I can adapt the recipe accordingly to achieve an authentic Argentinian alfajor.
"""
                    ),
                    TextPart(
                        content="""\
To cook Argentinian alfajores, follow these steps, which highlight the unique characteristics that distinguish them from their Uruguayan counterparts:

### Ingredients:
- **For the Cookies:**
  - 2 cups all-purpose flour
  - 1 cup powdered sugar
  - 1/2 teaspoon baking powder
  - 1/4 teaspoon baking soda
  - 1/4 teaspoon salt
  - 1/2 cup unsalted butter, softened
  - 1 large egg
  - 1 egg yolk
  - 1 teaspoon vanilla extract

- **For the Filling:**
  - 1 can (14 oz) sweetened condensed milk (for dulce de leche)
  - Optional: jam or chocolate ganache

- **For the Coating:**
  - Powdered sugar
  - Optional: cinnamon or cocoa powder mixed with sugar
  - Optional: melted chocolate (milk, dark, or white)

### Instructions:

1. **Prepare the Cookie Dough:**
   - In a large bowl, whisk together flour, powdered sugar, baking powder, baking soda, and salt.
   - Add softened butter and mix until the mixture resembles coarse crumbs.
   - In a separate bowl, whisk together egg, egg yolk, and vanilla extract. Pour into the dry mixture and mix until a dough forms.
   - Wrap dough in plastic wrap and refrigerate for 30 minutes.

2. **Roll and Cut Cookies:**
   - Roll out dough on a floured surface to about 1/4 inch thickness.
   - Cut into circles using a cookie cutter or glass rim.
   - Place cookies on a parchment-lined baking sheet, leaving space between each.

3. **Bake the Cookies:**
   - Preheat oven to 350°F (180°C).
   - Bake for 15-20 minutes until golden. Argentinian cookies might be baked longer for a crisper texture.
   - Allow to cool on the baking sheet for 5 minutes, then transfer to a wire rack to cool completely.

4. **Make Dulce de Leche:**
   - Pour sweetened condensed milk into a saucepan and heat over medium heat, stirring frequently, until thickened and golden (about 10-15 minutes).

5. **Assemble Alfajores:**
   - Spread a layer of dulce de leche on the flat side of one cookie. For added flavor, a thin layer of jam or chocolate ganache can also be used.
   - Sandwich with another cookie, pressing gently.

6. **Coat the Alfajores:**
   - Roll each sandwich in powdered sugar to coat evenly.
   - For an Argentinian twist, dip the filled alfajores in melted chocolate (milk, dark, or white) for a chocolate coating.
   - Optionally, mix cinnamon or cocoa powder with powdered sugar for a different coating flavor.

7. **Serve:**
   - Enjoy your Argentinian alfajores with a dusting of powdered sugar or chocolate coating. Store in an airtight container for up to 5 days.

### Tips:
- Ensure cookies are completely cool before filling to prevent sogginess.
- For an extra touch, add a pinch of salt to the dough for flavor balance.
- Experiment with different fillings and coatings to explore various regional variations within Argentina.

By following these steps, you can create authentic Argentinian alfajores that showcase their unique characteristics, such as a crisper texture and optional chocolate coating.\
"""
                    ),
                ],
                usage=RequestUsage(input_tokens=573, output_tokens=1509),
                model_name='deepseek-r1-distill-llama-70b',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_details={'finish_reason': 'stop'},
                provider_response_id='chatcmpl-dd0af56b-f71d-4101-be2f-89efcf3f05ac',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    assert event_parts == snapshot(
        [
            PartStartEvent(index=0, part=ThinkingPart(content='Alright')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' so')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'m")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' trying')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' figure')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' out')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' how')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' make')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Arg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='entin')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ian')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='aj')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ores')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' know')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Ur')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ugu')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ayan')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='aj')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ores')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' are')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' these')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' delicious')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookie')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sandwiches')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' filled')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coated')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' powdered')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' But')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' heard')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Arg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='entin')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ian')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='aj')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ores')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' are')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' bit')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' different')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'m")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' not')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' exactly')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sure')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' what')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' makes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' them')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' unique')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' so')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' need')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' look')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' into')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that')),
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
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' think')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' about')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' what')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' know')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' about')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Arg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='entin')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ian')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' desserts')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' They')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' have')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' rich')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' tradition')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sweet')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' treats')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='aj')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ores')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' are')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' definitely')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' one')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' them')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' difference')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' lies')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' type')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookies')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' used')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' filling')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' recall')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' some')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' South')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' American')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' countries')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='aj')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ores')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' can')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' be')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' more')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' like')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' bisc')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='uit')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' even')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cake')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-like')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookie')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' whereas')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' others')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' might')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' be')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cr')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='isper')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' also')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' remember')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sometimes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='aj')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ores')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' are')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coated')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' instead')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' just')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' powdered')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' That')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' could')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' be')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' an')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Arg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='entin')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ian')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' twist')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' need')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' confirm')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Also')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' filling')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' might')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' not')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' just')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' be')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=';')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' perhaps')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' other')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ingredients')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' like')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' jam')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' gan')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ache')),
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
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' consider')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' texture')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookies')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Ur')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ugu')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ayan')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='aj')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ores')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' have')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' softer')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' more')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' delicate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookie')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' while')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Arg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='entin')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ian')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ones')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' might')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' be')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' crunch')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ier')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' different')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' type')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' flour')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' baking')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' technique')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' should')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' check')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' recipes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' from')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' both')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' countries')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' see')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' differences')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ingredients')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' preparation')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' methods')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' also')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' wonder')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' about')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' history')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='aj')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ores')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Argentina')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' They')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' might')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' have')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' been')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' influenced')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' by')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' European')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' immigrants')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' especially')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' from')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Spain')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Italy')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' which')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' could')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' explain')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' variations')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' recipe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' This')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cultural')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' influence')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' might')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' contribute')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' differences')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' how')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookies')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' are')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' made')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' filled')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Additionally')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' think')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' about')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' assembly')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='aj')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ores')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' In')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Uruguay')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'s")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' typically')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' two')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookies')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sandwich')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ing')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' then')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coated')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' powdered')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Argentina')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' add')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' more')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' layers')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' different')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coating')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' like')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cinnamon')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cocoa')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' powder')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' mixed')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sugar')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' also')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' need')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' consider')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' availability')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ingredients')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Dul')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ce')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' de')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='che')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' staple')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' many')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' South')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' American')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' countries')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' but')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Argentina')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' have')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' slightly')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' different')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' version')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' combination')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' other')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' fill')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ings')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Perhaps')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' also')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' nuts')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' other')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ingredients')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' added')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' texture')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' flavor')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Another')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' aspect')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' baking')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' process')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' The')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Ur')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ugu')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ayan')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookies')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' might')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' be')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' baked')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' until')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' just')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' set')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' while')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Arg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='entin')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ian')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ones')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' could')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' be')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' baked')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' longer')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cr')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='isper')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' texture')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' perhaps')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' different')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' le')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='aven')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ing')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' agent')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' achieve')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' lighter')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dens')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='er')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookie')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' also')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' think')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' about')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' size')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookies')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Are')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Arg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='entin')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ian')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='aj')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ores')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' larger')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' smaller')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' than')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Ur')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ugu')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ayan')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ones')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='?')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' This')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' could')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' affect')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' baking')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' time')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' overall')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' appearance')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' final')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' product')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Furthermore')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' recall')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' some')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' regions')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='aj')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ores')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' are')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dipped')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' after')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' being')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' filled')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' This')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' could')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' be')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' distinguishing')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' feature')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Arg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='entin')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ian')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' version')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' The')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coating')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' might')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' be')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' milk')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dark')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' even')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' white')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' adding')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' another')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' layer')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' flavor')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookies')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' also')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' wonder')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' about')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' storage')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' serving')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Arg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='entin')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ian')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='aj')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ores')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' are')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' best')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' served')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' fresh')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' perhaps')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' can')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' be')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' stored')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' few')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' days')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' like')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Ur')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ugu')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ayan')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ones')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Understanding')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' this')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' can')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' help')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' planning')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' baking')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' assembly')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' process')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Lastly')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' think')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' about')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' potential')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' variations')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' within')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Argentina')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' itself')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Different')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' regions')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' might')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' have')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' their')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' own')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' take')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' on')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='aj')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ores')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' so')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' there')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' could')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' be')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' multiple')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' authentic')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Arg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='entin')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ian')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' recipes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' It')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' would')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' be')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' helpful')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' find')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' classic')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' widely')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' recognized')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' version')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ensure')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' authenticity')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Overall')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cook')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Arg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='entin')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ian')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='aj')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ores')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' need')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' focus')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' on')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' specific')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' characteristics')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' distinguish')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' them')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' from')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' their')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Ur')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ugu')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ayan')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' counterparts')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' whether')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'s")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' type')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cookie')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' filling')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' coating')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' baking')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' method')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' By')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' identifying')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' these')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' differences')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' can')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' adapt')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' recipe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' accordingly')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' achieve')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' an')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' authentic')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Arg')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='entin')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ian')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ajor')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.\n')),
            PartEndEvent(
                index=0,
                part=ThinkingPart(
                    content="""\
Alright, so I'm trying to figure out how to make Argentinian alfajores. I know that Uruguayan alfajores are these delicious cookie sandwiches filled with dulce de leche and coated in powdered sugar. But I heard that Argentinian alfajores are a bit different. I'm not exactly sure what makes them unique, so I need to look into that.

First, I think about what I know about Argentinian desserts. They have a rich tradition of sweet treats, and alfajores are definitely one of them. Maybe the difference lies in the type of cookies used or the filling. I recall that in some South American countries, alfajores can be more like a biscuit or even a cake-like cookie, whereas in others, they might be crisper.

I also remember that sometimes alfajores are coated in chocolate instead of just powdered sugar. That could be an Argentinian twist. I need to confirm that. Also, the filling might not just be dulce de leche; perhaps they use other ingredients like jam or chocolate ganache.

Another thing to consider is the texture of the cookies. Uruguayan alfajores have a softer, more delicate cookie, while Argentinian ones might be crunchier. Or maybe they use a different type of flour or baking technique. I should check recipes from both countries to see the differences in ingredients and preparation methods.

I also wonder about the history of alfajores in Argentina. They might have been influenced by European immigrants, especially from Spain or Italy, which could explain variations in the recipe. This cultural influence might contribute to differences in how the cookies are made and filled.

Additionally, I think about the assembly of the alfajores. In Uruguay, it's typically two cookies sandwiching the dulce de leche and then coated in powdered sugar. Maybe in Argentina, they add more layers or use a different coating, like cinnamon or cocoa powder mixed with sugar.

I also need to consider the availability of ingredients. Dulce de leche is a staple in many South American countries, but maybe in Argentina, they have a slightly different version of it or use it in combination with other fillings. Perhaps they also use nuts or other ingredients in the dough for added texture and flavor.

Another aspect is the baking process. The Uruguayan cookies might be baked until just set, while Argentinian ones could be baked longer for a crisper texture. Or perhaps they use a different leavening agent to achieve a lighter or denser cookie.

I also think about the size of the cookies. Are Argentinian alfajores larger or smaller than the Uruguayan ones? This could affect baking time and the overall appearance of the final product.

Furthermore, I recall that in some regions, alfajores are dipped in chocolate after being filled. This could be a distinguishing feature of the Argentinian version. The chocolate coating might be milk, dark, or even white chocolate, adding another layer of flavor to the cookies.

I also wonder about the storage and serving of Argentinian alfajores. Maybe they are best served fresh, or perhaps they can be stored for a few days like the Uruguayan ones. Understanding this can help in planning the baking and assembly process.

Lastly, I think about potential variations within Argentina itself. Different regions might have their own take on alfajores, so there could be multiple authentic Argentinian recipes. It would be helpful to find a classic or widely recognized version to ensure authenticity.

Overall, to cook Argentinian alfajores, I need to focus on the specific characteristics that distinguish them from their Uruguayan counterparts, whether it's the type of cookie, the filling, the coating, or the baking method. By identifying these differences, I can adapt the recipe accordingly to achieve an authentic Argentinian alfajor.
"""
                ),
                next_part_kind='text',
            ),
            PartStartEvent(index=1, part=TextPart(content='To'), previous_part_kind='thinking'),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cook')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Arg')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='entin')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ian')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='aj')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ores')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' follow')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' these')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' steps')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' which')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' highlight')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' unique')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' characteristics')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' that')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' distinguish')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' them')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' from')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' their')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Ur')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ugu')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ayan')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' counterparts')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
:

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='###')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Ingredients')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='For')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Cookies')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cups')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' all')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-purpose')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' flour')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cup')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' powdered')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='/')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' teaspoon')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' baking')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' powder')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='/')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='4')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' teaspoon')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' baking')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' soda')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='/')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='4')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' teaspoon')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' salt')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='/')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cup')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' uns')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='alted')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' butter')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' softened')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' large')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' egg')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' egg')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' y')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='olk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' teaspoon')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' vanilla')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' extract')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\


"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='For')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' F')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='illing')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' can')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='14')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' oz')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=')')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sweet')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ened')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' condensed')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' milk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ce')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' de')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' le')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='che')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=')\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Optional')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' jam')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' gan')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ache')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\


"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='For')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Co')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ating')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Powder')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ed')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Optional')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cinnamon')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cocoa')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' powder')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' mixed')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Optional')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' melted')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='m')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ilk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dark')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' white')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
)

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='###')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Instructions')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
:

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Prepare')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Cookie')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Dough')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' In')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' large')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' bowl')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' whisk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' together')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' flour')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' powdered')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' baking')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' powder')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' baking')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' soda')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' salt')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Add')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' softened')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' butter')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' mix')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' until')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' mixture')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' resembles')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' coarse')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' crumbs')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' In')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' separate')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' bowl')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' whisk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' together')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' egg')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' egg')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' y')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='olk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' vanilla')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' extract')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Pour')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' into')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dry')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' mixture')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' mix')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' until')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' forms')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Wrap')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' plastic')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' wrap')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' refriger')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ate')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='30')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' minutes')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Roll')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Cut')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Cookies')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Roll')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' out')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' on')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' fl')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='oured')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' surface')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' about')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='/')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='4')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' inch')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' thickness')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Cut')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' into')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' circles')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' using')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cookie')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cutter')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' glass')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' rim')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Place')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cookies')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' on')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' parchment')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-lined')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' baking')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sheet')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' leaving')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' space')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' between')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' each')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='3')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='B')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ake')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Cookies')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Pre')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='heat')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' oven')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='350')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='°F')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='180')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='°C')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=').\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Bake')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='15')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='20')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' minutes')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' until')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' golden')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Arg')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='entin')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ian')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cookies')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' might')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' be')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' baked')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' longer')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cr')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='isper')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' texture')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Allow')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cool')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' on')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' baking')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sheet')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='5')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' minutes')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' then')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' transfer')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' wire')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' rack')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cool')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' completely')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='4')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Make')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Dul')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ce')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' de')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Le')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='che')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Pour')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sweet')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ened')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' condensed')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' milk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' into')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sauce')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='pan')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' heat')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' over')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' medium')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' heat')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' stirring')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' frequently')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' until')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' thick')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ened')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' golden')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='about')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='10')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='15')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' minutes')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
).

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='5')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='As')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='semble')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Alf')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='aj')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ores')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Spread')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' layer')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' of')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dul')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ce')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' de')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' le')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='che')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' on')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' flat')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' side')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' of')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' one')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cookie')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' For')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' added')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' flavor')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' thin')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' layer')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' of')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' jam')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' gan')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ache')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' can')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' also')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' be')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' used')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Sandwich')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' another')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cookie')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' pressing')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' gently')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='6')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Co')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='at')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Alf')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='aj')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ores')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Roll')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' each')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sandwich')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' powdered')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' coat')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' evenly')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' For')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' an')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Arg')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='entin')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ian')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' twist')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dip')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' filled')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='aj')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ores')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' melted')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='m')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ilk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dark')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' white')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=')')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' coating')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Optionally')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' mix')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cinnamon')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cocoa')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' powder')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' powdered')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' different')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' coating')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' flavor')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='7')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Serve')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Enjoy')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' your')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Arg')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='entin')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ian')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='aj')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ores')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dust')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ing')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' of')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' powdered')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sugar')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' coating')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Store')),
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
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='5')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' days')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='###')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Tips')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Ensure')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cookies')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' are')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' completely')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cool')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' before')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' filling')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' prevent')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' s')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ogg')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='iness')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' For')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' an')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' extra')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' touch')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' add')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' pinch')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' of')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' salt')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dough')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' flavor')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' balance')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Experiment')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' different')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' fill')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ings')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' coatings')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' explore')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' various')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' regional')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' variations')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' within')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Argentina')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='By')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' following')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' these')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' steps')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' you')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' can')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' create')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' authentic')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Arg')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='entin')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ian')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' alf')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='aj')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ores')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' that')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' showcase')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' their')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' unique')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' characteristics')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' such')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' as')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cr')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='isper')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' texture')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' optional')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' chocolate')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' coating')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartEndEvent(
                index=1,
                part=TextPart(
                    content="""\
To cook Argentinian alfajores, follow these steps, which highlight the unique characteristics that distinguish them from their Uruguayan counterparts:

### Ingredients:
- **For the Cookies:**
  - 2 cups all-purpose flour
  - 1 cup powdered sugar
  - 1/2 teaspoon baking powder
  - 1/4 teaspoon baking soda
  - 1/4 teaspoon salt
  - 1/2 cup unsalted butter, softened
  - 1 large egg
  - 1 egg yolk
  - 1 teaspoon vanilla extract

- **For the Filling:**
  - 1 can (14 oz) sweetened condensed milk (for dulce de leche)
  - Optional: jam or chocolate ganache

- **For the Coating:**
  - Powdered sugar
  - Optional: cinnamon or cocoa powder mixed with sugar
  - Optional: melted chocolate (milk, dark, or white)

### Instructions:

1. **Prepare the Cookie Dough:**
   - In a large bowl, whisk together flour, powdered sugar, baking powder, baking soda, and salt.
   - Add softened butter and mix until the mixture resembles coarse crumbs.
   - In a separate bowl, whisk together egg, egg yolk, and vanilla extract. Pour into the dry mixture and mix until a dough forms.
   - Wrap dough in plastic wrap and refrigerate for 30 minutes.

2. **Roll and Cut Cookies:**
   - Roll out dough on a floured surface to about 1/4 inch thickness.
   - Cut into circles using a cookie cutter or glass rim.
   - Place cookies on a parchment-lined baking sheet, leaving space between each.

3. **Bake the Cookies:**
   - Preheat oven to 350°F (180°C).
   - Bake for 15-20 minutes until golden. Argentinian cookies might be baked longer for a crisper texture.
   - Allow to cool on the baking sheet for 5 minutes, then transfer to a wire rack to cool completely.

4. **Make Dulce de Leche:**
   - Pour sweetened condensed milk into a saucepan and heat over medium heat, stirring frequently, until thickened and golden (about 10-15 minutes).

5. **Assemble Alfajores:**
   - Spread a layer of dulce de leche on the flat side of one cookie. For added flavor, a thin layer of jam or chocolate ganache can also be used.
   - Sandwich with another cookie, pressing gently.

6. **Coat the Alfajores:**
   - Roll each sandwich in powdered sugar to coat evenly.
   - For an Argentinian twist, dip the filled alfajores in melted chocolate (milk, dark, or white) for a chocolate coating.
   - Optionally, mix cinnamon or cocoa powder with powdered sugar for a different coating flavor.

7. **Serve:**
   - Enjoy your Argentinian alfajores with a dusting of powdered sugar or chocolate coating. Store in an airtight container for up to 5 days.

### Tips:
- Ensure cookies are completely cool before filling to prevent sogginess.
- For an extra touch, add a pinch of salt to the dough for flavor balance.
- Experiment with different fillings and coatings to explore various regional variations within Argentina.

By following these steps, you can create authentic Argentinian alfajores that showcase their unique characteristics, such as a crisper texture and optional chocolate coating.\
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
                instructions='Be concise. Never use pretty double quotes, just regular ones.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_something_by_name',
                        args={'invalid_param': 'test'},
                        tool_call_id=IsStr(),
                    )
                ],
                model_name='openai/gpt-oss-120b',
                timestamp=IsDatetime(),
                provider_name='groq',
                finish_reason='error',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content=[
                            {
                                'type': 'missing',
                                'loc': ('name',),
                                'msg': 'Field required',
                                'input': {'invalid_param': 'test'},
                            },
                            {
                                'type': 'extra_forbidden',
                                'loc': ('invalid_param',),
                                'msg': 'Extra inputs are not permitted',
                                'input': 'test',
                            },
                        ],
                        tool_name='get_something_by_name',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                instructions='Be concise. Never use pretty double quotes, just regular ones.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='We need to call with correct param name: name. Provide a non-existent name perhaps "nonexistent".'
                    ),
                    ToolCallPart(
                        tool_name='get_something_by_name',
                        args='{"name":"nonexistent"}',
                        tool_call_id=IsStr(),
                    ),
                ],
                usage=RequestUsage(input_tokens=283, output_tokens=49),
                model_name='openai/gpt-oss-120b',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_details={'finish_reason': 'tool_calls'},
                provider_response_id=IsStr(),
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_something_by_name',
                        content='Something with name: nonexistent',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                instructions='Be concise. Never use pretty double quotes, just regular ones.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='The user asked: "Please call the \'get_something_by_name\' tool with non-existent parameters to test error handling". They wanted to test error handling with non-existent parameters, but we corrected to proper parameters. The response from tool: "Something with name: nonexistent". Should we respond? Probably just output the result. Follow developer instruction: be concise, no fancy quotes. Use regular quotes only.'
                    ),
                    TextPart(content='Something with name: nonexistent'),
                ],
                usage=RequestUsage(input_tokens=319, output_tokens=96),
                model_name='openai/gpt-oss-120b',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_details={'finish_reason': 'stop'},
                provider_response_id=IsStr(),
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
                instructions='Be concise. Never use pretty double quotes, just regular ones.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="""\
The user requests to call the tool with non-existent parameters to test error handling. We need to call the function "get_something_by_name" with wrong parameters. The function expects a single argument object with "name". Non-existent parameters means we could provide a wrong key, or missing name. Let's provide an object with wrong key "nonexistent": "value". That should cause error. So we call the function with {"nonexistent": "test"}.

We need to output the call.\
"""
                    ),
                    ToolCallPart(
                        tool_name='get_something_by_name',
                        args={'nonexistent': 'test'},
                        tool_call_id=IsStr(),
                    ),
                ],
                model_name='openai/gpt-oss-120b',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_response_id='chatcmpl-4e0ca299-7515-490a-a98a-16d7664d4fba',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content=[
                            {
                                'type': 'missing',
                                'loc': ('name',),
                                'msg': 'Field required',
                                'input': {'nonexistent': 'test'},
                            },
                            {
                                'type': 'extra_forbidden',
                                'loc': ('nonexistent',),
                                'msg': 'Extra inputs are not permitted',
                                'input': 'test',
                            },
                        ],
                        tool_name='get_something_by_name',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                instructions='Be concise. Never use pretty double quotes, just regular ones.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content='We need to call with correct param: name. Use a placeholder name.'),
                    ToolCallPart(
                        tool_name='get_something_by_name',
                        args='{"name":"test_name"}',
                        tool_call_id=IsStr(),
                    ),
                ],
                usage=RequestUsage(input_tokens=283, output_tokens=43),
                model_name='openai/gpt-oss-120b',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_details={'finish_reason': 'tool_calls'},
                provider_response_id='chatcmpl-fffa1d41-1763-493a-9ced-083bd3f2d98b',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_something_by_name',
                        content='Something with name: test_name',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                instructions='Be concise. Never use pretty double quotes, just regular ones.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='The tool call succeeded with the name "test_name".')],
                usage=RequestUsage(input_tokens=320, output_tokens=15),
                model_name='openai/gpt-oss-120b',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_details={'finish_reason': 'stop'},
                provider_response_id='chatcmpl-fe6b5685-166f-4c71-9cd7-3d5a97301bf1',
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
                provider_details={'finish_reason': 'stop'},
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
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='We need to respond with JSON object with properties city and country. The question: "What is the largest city in Mexico?" The answer: City is Mexico City, country is Mexico. Must output compact JSON without any extra text or markdown. So {"city":"Mexico City","country":"Mexico"} Ensure valid JSON.'
                    ),
                    TextPart(content='{"city":"Mexico City","country":"Mexico"}'),
                ],
                usage=RequestUsage(input_tokens=177, output_tokens=87),
                model_name='openai/gpt-oss-120b',
                timestamp=IsDatetime(),
                provider_name='groq',
                provider_details={'finish_reason': 'stop'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )
