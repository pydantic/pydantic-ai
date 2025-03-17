from __future__ import annotations as _annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import cached_property
from typing import Any, Literal, Union, cast

import httpx
import pytest
from inline_snapshot import snapshot
from typing_extensions import TypedDict

from pydantic_ai import Agent, ModelHTTPError, ModelRetry, UnexpectedModelBehavior
from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.result import Usage
from pydantic_ai.settings import ModelSettings

from ..conftest import IsNow, TestEnv, raise_if_exception, try_import
from .mock_async_stream import MockAsyncStream

with try_import() as imports_successful:
    from openai import NOT_GIVEN, APIStatusError, AsyncOpenAI, OpenAIError
    from openai.types import chat
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_chunk import (
        Choice as ChunkChoice,
        ChoiceDelta,
        ChoiceDeltaToolCall,
        ChoiceDeltaToolCallFunction,
    )
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
    from openai.types.chat.chat_completion_message_tool_call import Function
    from openai.types.completion_usage import CompletionUsage, PromptTokensDetails

    from pydantic_ai.models.openai import OpenAIModel, OpenAISystemPromptRole
    from pydantic_ai.providers.openai import OpenAIProvider

    # note: we use Union here so that casting works with Python 3.9
    MockChatCompletion = Union[chat.ChatCompletion, Exception]
    MockChatCompletionChunk = Union[chat.ChatCompletionChunk, Exception]

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
]


def test_init():
    m = OpenAIModel('gpt-4o', provider=OpenAIProvider(api_key='foobar'))
    assert m.base_url == 'https://api.openai.com/v1/'
    assert m.client.api_key == 'foobar'
    assert m.model_name == 'gpt-4o'


def test_init_with_base_url():
    m = OpenAIModel('gpt-4o', provider=OpenAIProvider(base_url='https://example.com/v1', api_key='foobar'))
    assert str(m.client.base_url) == 'https://example.com/v1/'
    assert m.client.api_key == 'foobar'
    assert m.model_name == 'gpt-4o'


def test_init_with_no_api_key_will_still_setup_client():
    m = OpenAIModel('llama3.2', provider=OpenAIProvider(base_url='http://localhost:19434/v1'))
    assert str(m.client.base_url) == 'http://localhost:19434/v1/'


def test_init_with_non_openai_model():
    m = OpenAIModel('llama3.2-vision:latest', provider=OpenAIProvider(base_url='https://example.com/v1/'))
    assert m.model_name == 'llama3.2-vision:latest'


def test_init_of_openai_without_api_key_raises_error(env: TestEnv):
    env.remove('OPENAI_API_KEY')
    with pytest.raises(
        OpenAIError,
        match='^The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable$',
    ):
        OpenAIModel('gpt-4o')


@dataclass
class MockOpenAI:
    completions: MockChatCompletion | Sequence[MockChatCompletion] | None = None
    stream: Sequence[MockChatCompletionChunk] | Sequence[Sequence[MockChatCompletionChunk]] | None = None
    index: int = 0
    chat_completion_kwargs: list[dict[str, Any]] = field(default_factory=list)

    @cached_property
    def chat(self) -> Any:
        chat_completions = type('Completions', (), {'create': self.chat_completions_create})
        return type('Chat', (), {'completions': chat_completions})

    @classmethod
    def create_mock(cls, completions: MockChatCompletion | Sequence[MockChatCompletion]) -> AsyncOpenAI:
        return cast(AsyncOpenAI, cls(completions=completions))

    @classmethod
    def create_mock_stream(
        cls,
        stream: Sequence[MockChatCompletionChunk] | Sequence[Sequence[MockChatCompletionChunk]],
    ) -> AsyncOpenAI:
        return cast(AsyncOpenAI, cls(stream=stream))

    async def chat_completions_create(  # pragma: no cover
        self, *_args: Any, stream: bool = False, **kwargs: Any
    ) -> chat.ChatCompletion | MockAsyncStream[MockChatCompletionChunk]:
        self.chat_completion_kwargs.append({k: v for k, v in kwargs.items() if v is not NOT_GIVEN})

        if stream:
            assert self.stream is not None, 'you can only used `stream=True` if `stream` is provided'
            if isinstance(self.stream[0], Sequence):
                response = MockAsyncStream(iter(cast(list[MockChatCompletionChunk], self.stream[self.index])))
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


def get_mock_chat_completion_kwargs(async_open_ai: AsyncOpenAI) -> list[dict[str, Any]]:
    if isinstance(async_open_ai, MockOpenAI):
        return async_open_ai.chat_completion_kwargs
    else:  # pragma: no cover
        raise RuntimeError('Not a MockOpenAI instance')


def completion_message(message: ChatCompletionMessage, *, usage: CompletionUsage | None = None) -> chat.ChatCompletion:
    return chat.ChatCompletion(
        id='123',
        choices=[Choice(finish_reason='stop', index=0, message=message)],
        created=1704067200,  # 2024-01-01
        model='gpt-4o-123',
        object='chat.completion',
        usage=usage,
    )


async def test_request_simple_success(allow_model_requests: None):
    c = completion_message(ChatCompletionMessage(content='world', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    m = OpenAIModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m)

    result = await agent.run('hello')
    assert result.data == 'world'
    assert result.usage() == snapshot(Usage(requests=1))

    # reset the index so we get the same response again
    mock_client.index = 0  # type: ignore

    result = await agent.run('hello', message_history=result.new_messages())
    assert result.data == 'world'
    assert result.usage() == snapshot(Usage(requests=1))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[TextPart(content='world')],
                model_name='gpt-4o-123',
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
            ),
            ModelRequest(parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[TextPart(content='world')],
                model_name='gpt-4o-123',
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
            ),
        ]
    )
    assert get_mock_chat_completion_kwargs(mock_client) == [
        {'messages': [{'content': 'hello', 'role': 'user'}], 'model': 'gpt-4o', 'n': 1},
        {
            'messages': [
                {'content': 'hello', 'role': 'user'},
                {'content': 'world', 'role': 'assistant'},
                {'content': 'hello', 'role': 'user'},
            ],
            'model': 'gpt-4o',
            'n': 1,
        },
    ]


async def test_request_simple_usage(allow_model_requests: None):
    c = completion_message(
        ChatCompletionMessage(content='world', role='assistant'),
        usage=CompletionUsage(completion_tokens=1, prompt_tokens=2, total_tokens=3),
    )
    mock_client = MockOpenAI.create_mock(c)
    m = OpenAIModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m)

    result = await agent.run('Hello')
    assert result.data == 'world'
    assert result.usage() == snapshot(Usage(requests=1, request_tokens=2, response_tokens=1, total_tokens=3))


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
    mock_client = MockOpenAI.create_mock(c)
    m = OpenAIModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m, result_type=list[int])

    result = await agent.run('Hello')
    assert result.data == [1, 2, 123]
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
                model_name='gpt-4o-123',
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
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
                prompt_tokens_details=PromptTokensDetails(cached_tokens=1),
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
                prompt_tokens_details=PromptTokensDetails(cached_tokens=2),
            ),
        ),
        completion_message(ChatCompletionMessage(content='final response', role='assistant')),
    ]
    mock_client = MockOpenAI.create_mock(responses)
    m = OpenAIModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m, system_prompt='this is the system prompt')

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        if loc_name == 'London':
            return json.dumps({'lat': 51, 'lng': 0})
        else:
            raise ModelRetry('Wrong location, please try again')

    result = await agent.run('Hello')
    assert result.data == 'final response'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='this is the system prompt'),
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
                model_name='gpt-4o-123',
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Wrong location, please try again',
                        tool_name='get_location',
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
                model_name='gpt-4o-123',
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
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
                model_name='gpt-4o-123',
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ),
        ]
    )
    assert result.usage() == snapshot(
        Usage(
            requests=3,
            request_tokens=5,
            response_tokens=3,
            total_tokens=9,
            details={'cached_tokens': 3},
        )
    )


FinishReason = Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call']


def chunk(delta: list[ChoiceDelta], finish_reason: FinishReason | None = None) -> chat.ChatCompletionChunk:
    return chat.ChatCompletionChunk(
        id='x',
        choices=[
            ChunkChoice(index=index, delta=delta, finish_reason=finish_reason) for index, delta in enumerate(delta)
        ],
        created=1704067200,  # 2024-01-01
        model='gpt-4o',
        object='chat.completion.chunk',
        usage=CompletionUsage(completion_tokens=1, prompt_tokens=2, total_tokens=3),
    )


def text_chunk(text: str, finish_reason: FinishReason | None = None) -> chat.ChatCompletionChunk:
    return chunk([ChoiceDelta(content=text, role='assistant')], finish_reason=finish_reason)


async def test_stream_text(allow_model_requests: None):
    stream = [text_chunk('hello '), text_chunk('world'), chunk([])]
    mock_client = MockOpenAI.create_mock_stream(stream)
    m = OpenAIModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(['hello ', 'hello world'])
        assert result.is_complete
        assert result.usage() == snapshot(Usage(requests=1, request_tokens=6, response_tokens=3, total_tokens=9))


async def test_stream_text_finish_reason(allow_model_requests: None):
    stream = [
        text_chunk('hello '),
        text_chunk('world'),
        text_chunk('.', finish_reason='stop'),
    ]
    mock_client = MockOpenAI.create_mock_stream(stream)
    m = OpenAIModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(
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
    m = OpenAIModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m, result_type=MyTypedDict)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [dict(c) async for c in result.stream(debounce_by=None)] == snapshot(
            [
                {'first': 'One'},
                {'first': 'One', 'second': 'Two'},
                {'first': 'One', 'second': 'Two'},
                {'first': 'One', 'second': 'Two'},
            ]
        )
        assert result.is_complete
        assert result.usage() == snapshot(Usage(requests=1, request_tokens=20, response_tokens=10, total_tokens=30))
        # double check usage matches stream count
        assert result.usage().response_tokens == len(stream)


async def test_stream_structured_finish_reason(allow_model_requests: None):
    stream = [
        struc_chunk('final_result', None),
        struc_chunk(None, '{"first": "One'),
        struc_chunk(None, '", "second": "Two"'),
        struc_chunk(None, '}'),
        struc_chunk(None, None, finish_reason='stop'),
    ]
    mock_client = MockOpenAI.create_mock_stream(stream)
    m = OpenAIModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m, result_type=MyTypedDict)

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
    stream = [chunk([ChoiceDelta()]), chunk([ChoiceDelta()])]
    mock_client = MockOpenAI.create_mock_stream(stream)
    m = OpenAIModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m, result_type=MyTypedDict)

    with pytest.raises(UnexpectedModelBehavior, match='Received empty model response'):
        async with agent.run_stream(''):
            pass


async def test_no_delta(allow_model_requests: None):
    stream = [
        chunk([]),
        text_chunk('hello '),
        text_chunk('world'),
    ]
    mock_client = MockOpenAI.create_mock_stream(stream)
    m = OpenAIModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(['hello ', 'hello world'])
        assert result.is_complete
        assert result.usage() == snapshot(Usage(requests=1, request_tokens=6, response_tokens=3, total_tokens=9))


@pytest.mark.parametrize('system_prompt_role', ['system', 'developer', 'user', None])
async def test_system_prompt_role(
    allow_model_requests: None, system_prompt_role: OpenAISystemPromptRole | None
) -> None:
    """Testing the system prompt role for OpenAI models is properly set / inferred."""

    c = completion_message(ChatCompletionMessage(content='world', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    m = OpenAIModel('gpt-4o', system_prompt_role=system_prompt_role, provider=OpenAIProvider(openai_client=mock_client))
    assert m.system_prompt_role == system_prompt_role

    agent = Agent(m, system_prompt='some instructions')
    result = await agent.run('hello')
    assert result.data == 'world'

    assert get_mock_chat_completion_kwargs(mock_client) == [
        {
            'messages': [
                {'content': 'some instructions', 'role': system_prompt_role or 'system'},
                {'content': 'hello', 'role': 'user'},
            ],
            'model': 'gpt-4o',
            'n': 1,
        }
    ]


@pytest.mark.parametrize('system_prompt_role', ['system', 'developer'])
@pytest.mark.vcr
async def test_openai_o1_mini_system_role(
    allow_model_requests: None,
    system_prompt_role: Literal['system', 'developer'],
    openai_api_key: str,
) -> None:
    model = OpenAIModel(
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
                chat.ChatCompletionMessageToolCall(
                    id='123',
                    function=Function(arguments='{"response": [1, 2, 3]}', name='final_result'),
                    type='function',
                )
            ],
        )
    )
    mock_client = MockOpenAI.create_mock(c)
    m = OpenAIModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m, result_type=list[int], model_settings=ModelSettings(parallel_tool_calls=parallel_tool_calls))

    await agent.run('Hello')
    assert get_mock_chat_completion_kwargs(mock_client)[0]['parallel_tool_calls'] == parallel_tool_calls


async def test_image_url_input(allow_model_requests: None):
    c = completion_message(ChatCompletionMessage(content='world', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    m = OpenAIModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m)

    result = await agent.run(
        [
            'hello',
            ImageUrl(url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg'),
        ]
    )
    assert result.data == 'world'
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
                'n': 1,
            }
        ]
    )


@pytest.mark.vcr()
async def test_image_as_binary_content_input(
    allow_model_requests: None, image_content: BinaryContent, openai_api_key: str
):
    m = OpenAIModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)

    result = await agent.run(['What fruit is in the image?', image_content])
    assert result.data == snapshot('The fruit in the image is a kiwi.')


@pytest.mark.vcr()
async def test_audio_as_binary_content_input(
    allow_model_requests: None, audio_content: BinaryContent, openai_api_key: str
):
    m = OpenAIModel('gpt-4o-audio-preview', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(m)

    result = await agent.run(['Whose name is mentioned in the audio?', audio_content])
    assert result.data == snapshot('The name mentioned in the audio is Marcelo.')


def test_model_status_error(allow_model_requests: None) -> None:
    mock_client = MockOpenAI.create_mock(
        APIStatusError(
            'test error',
            response=httpx.Response(status_code=500, request=httpx.Request('POST', 'https://example.com/v1')),
            body={'error': 'test error'},
        )
    )
    m = OpenAIModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m)
    with pytest.raises(ModelHTTPError) as exc_info:
        agent.run_sync('hello')
    assert str(exc_info.value) == snapshot("status_code: 500, model_name: gpt-4o, body: {'error': 'test error'}")
