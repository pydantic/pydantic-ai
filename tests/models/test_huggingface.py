from __future__ import annotations as _annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import cached_property
from typing import Any, Literal, cast
from unittest.mock import Mock

import pytest
from inline_snapshot import snapshot
from typing_extensions import TypedDict

from pydantic_ai import (
    Agent,
    AudioUrl,
    BinaryContent,
    CachePoint,
    DocumentUrl,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    ModelRetry,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.result import RunUsage
from pydantic_ai.run import AgentRunResult, AgentRunResultEvent
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage

from ..conftest import IsDatetime, IsInstance, IsNow, IsStr, raise_if_exception, try_import
from .mock_async_stream import MockAsyncStream

with try_import() as imports_successful:
    import aiohttp
    from huggingface_hub import (
        AsyncInferenceClient,
        ChatCompletionInputMessage,
        ChatCompletionOutput,
        ChatCompletionOutputComplete,
        ChatCompletionOutputFunctionDefinition,
        ChatCompletionOutputMessage,
        ChatCompletionOutputToolCall,
        ChatCompletionOutputUsage,
        ChatCompletionStreamOutput,
        ChatCompletionStreamOutputChoice,
        ChatCompletionStreamOutputDelta,
        ChatCompletionStreamOutputDeltaToolCall,
        ChatCompletionStreamOutputFunction,
        ChatCompletionStreamOutputUsage,
    )
    from huggingface_hub.errors import HfHubHTTPError

    from pydantic_ai.models.huggingface import HuggingFaceModel
    from pydantic_ai.providers.huggingface import HuggingFaceProvider

    MockChatCompletion = ChatCompletionOutput | Exception
    MockStreamEvent = ChatCompletionStreamOutput | Exception

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='huggingface_hub not installed'),
    pytest.mark.anyio,
    pytest.mark.filterwarnings('ignore::ResourceWarning'),
]


@dataclass
class MockHuggingFace:
    completions: MockChatCompletion | Sequence[MockChatCompletion] | None = None
    stream: Sequence[MockStreamEvent] | Sequence[Sequence[MockStreamEvent]] | None = None
    index: int = 0
    chat_completion_kwargs: list[dict[str, Any]] = field(default_factory=list)
    model: str = 'https://api-inference.huggingface.co'

    @cached_property
    def chat(self) -> Any:
        completions = type('Completions', (), {'create': self.chat_completions_create})
        return type('Chat', (), {'completions': completions})

    @classmethod
    def create_mock(cls, completions: MockChatCompletion | Sequence[MockChatCompletion]) -> AsyncInferenceClient:
        return cast(AsyncInferenceClient, cls(completions=completions))

    @classmethod
    def create_stream_mock(
        cls, stream: Sequence[MockStreamEvent] | Sequence[Sequence[MockStreamEvent]]
    ) -> AsyncInferenceClient:
        return cast(AsyncInferenceClient, cls(stream=stream))

    async def chat_completions_create(
        self, *_args: Any, stream: bool = False, **kwargs: Any
    ) -> ChatCompletionOutput | MockAsyncStream[MockStreamEvent]:
        self.chat_completion_kwargs.append(kwargs)
        if stream or self.stream:
            assert self.stream is not None, 'you can only use `stream=True` if `stream` is provided'
            if isinstance(self.stream[0], Sequence):
                response = MockAsyncStream(iter(cast(list[MockStreamEvent], self.stream[self.index])))
            else:
                response = MockAsyncStream(iter(cast(list[MockStreamEvent], self.stream)))
        else:
            assert self.completions is not None, 'you can only use `stream=False` if `completions` are provided'
            if isinstance(self.completions, Sequence):
                raise_if_exception(self.completions[self.index])
                response = cast(ChatCompletionOutput, self.completions[self.index])
            else:
                raise_if_exception(self.completions)
                response = cast(ChatCompletionOutput, self.completions)
        self.index += 1
        return response


def get_mock_chat_completion_kwargs(hf_client: AsyncInferenceClient) -> list[dict[str, Any]]:
    if isinstance(hf_client, MockHuggingFace):
        return hf_client.chat_completion_kwargs
    else:  # pragma: no cover
        raise RuntimeError('Not a MockHuggingFace instance')


def completion_message(
    message: ChatCompletionInputMessage | ChatCompletionOutputMessage, *, usage: ChatCompletionOutputUsage | None = None
) -> ChatCompletionOutput:
    choices = [ChatCompletionOutputComplete(finish_reason='stop', index=0, message=message)]  # type:ignore
    return ChatCompletionOutput.parse_obj_as_instance(  # type: ignore
        {
            'id': '123',
            'choices': choices,
            'created': 1704067200,  # 2024-01-01
            'model': 'hf-model',
            'object': 'chat.completion',
            'usage': usage,
        }
    )


@pytest.mark.vcr()
async def test_simple_completion(allow_model_requests: None, huggingface_api_key: str):
    model = HuggingFaceModel(
        'Qwen/Qwen2.5-72B-Instruct',
        provider=HuggingFaceProvider(provider_name='nebius', api_key=huggingface_api_key),
    )
    agent = Agent(model)

    result = await agent.run('hello')
    assert (
        result.output
        == 'Hello! How can I assist you today? Feel free to ask me any questions or let me know if you need help with anything specific.'
    )
    messages = result.all_messages()
    request = messages[0]
    response = messages[1]
    assert request.parts[0].content == 'hello'  # type: ignore
    assert response == snapshot(
        ModelResponse(
            parts=[
                TextPart(
                    content='Hello! How can I assist you today? Feel free to ask me any questions or let me know if you need help with anything specific.'
                )
            ],
            usage=RequestUsage(input_tokens=30, output_tokens=29),
            model_name='Qwen/Qwen2.5-72B-Instruct-fast',
            timestamp=IsNow(tz=timezone.utc),
            provider_name='huggingface',
            provider_details={
                'finish_reason': 'stop',
                'timestamp': datetime(2025, 7, 8, 13, 42, 33, tzinfo=timezone.utc),
            },
            provider_response_id='chatcmpl-d445c0d473a84791af2acf356cc00df7',
            run_id=IsStr(),
        )
    )


@pytest.mark.vcr()
async def test_request_simple_usage(allow_model_requests: None, huggingface_api_key: str):
    model = HuggingFaceModel(
        'Qwen/Qwen2.5-72B-Instruct',
        provider=HuggingFaceProvider(provider_name='nebius', api_key=huggingface_api_key),
    )
    agent = Agent(model)

    result = await agent.run('Hello')
    assert (
        result.output
        == "Hello! It's great to meet you. How can I assist you today? Whether you have any questions, need some advice, or just want to chat, feel free to let me know!"
    )
    assert result.usage() == snapshot(RunUsage(requests=1, input_tokens=30, output_tokens=40))


async def test_request_structured_response(
    allow_model_requests: None,
):
    tool_call = ChatCompletionOutputToolCall.parse_obj_as_instance(  # type:ignore
        {
            'function': ChatCompletionOutputFunctionDefinition.parse_obj_as_instance(  # type:ignore
                {
                    'name': 'final_result',
                    'arguments': '{"response": [1, 2, 123]}',
                }
            ),
            'id': '123',
            'type': 'function',
        }
    )
    message = ChatCompletionOutputMessage.parse_obj_as_instance(  # type:ignore
        {
            'content': None,
            'role': 'assistant',
            'tool_calls': [tool_call],
        }
    )
    c = completion_message(message)

    mock_client = MockHuggingFace.create_mock(c)
    model = HuggingFaceModel(
        'Qwen/Qwen2.5-72B-Instruct',
        provider=HuggingFaceProvider(provider_name='nebius', hf_client=mock_client, api_key='x'),
    )
    agent = Agent(model, output_type=list[int])

    result = await agent.run('Hello')
    assert result.output == [1, 2, 123]
    messages = result.all_messages()
    assert messages[0].parts[0].content == 'Hello'  # type: ignore
    assert messages[1] == snapshot(
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='final_result',
                    args='{"response": [1, 2, 123]}',
                    tool_call_id='123',
                )
            ],
            model_name='hf-model',
            timestamp=IsNow(tz=timezone.utc),
            provider_name='huggingface',
            provider_url='https://api-inference.huggingface.co',
            provider_details={
                'finish_reason': 'stop',
                'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
            },
            provider_response_id='123',
            run_id=IsStr(),
        )
    )


async def test_stream_completion(allow_model_requests: None):
    stream = [text_chunk('hello '), text_chunk('world', finish_reason='stop')]
    mock_client = MockHuggingFace.create_stream_mock(stream)
    model = HuggingFaceModel('hf-model', provider=HuggingFaceProvider(hf_client=mock_client, api_key='x'))
    agent = Agent(model)

    async with agent.run_stream('') as result:
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(['hello ', 'hello world'])


async def test_multiple_stream_calls(allow_model_requests: None):
    stream = [
        [text_chunk('first '), text_chunk('call', finish_reason='stop')],
        [text_chunk('second '), text_chunk('call', finish_reason='stop')],
    ]
    mock_client = MockHuggingFace.create_stream_mock(stream)
    model = HuggingFaceModel('hf-model', provider=HuggingFaceProvider(hf_client=mock_client, api_key='x'))
    agent = Agent(model)

    async with agent.run_stream('first') as result:
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(['first ', 'first call'])

    async with agent.run_stream('second') as result:
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(['second ', 'second call'])


async def test_request_tool_call(allow_model_requests: None):
    tool_call_1 = ChatCompletionOutputToolCall.parse_obj_as_instance(  # type:ignore
        {
            'function': ChatCompletionOutputFunctionDefinition.parse_obj_as_instance(  # type:ignore
                {
                    'name': 'get_location',
                    'arguments': '{"loc_name": "San Fransisco"}',
                }
            ),
            'id': '1',
            'type': 'function',
        }
    )
    usage_1 = ChatCompletionOutputUsage.parse_obj_as_instance(  # type:ignore
        {
            'prompt_tokens': 1,
            'completion_tokens': 1,
            'total_tokens': 2,
        }
    )
    tool_call_2 = ChatCompletionOutputToolCall.parse_obj_as_instance(  # type:ignore
        {
            'function': ChatCompletionOutputFunctionDefinition.parse_obj_as_instance(  # type:ignore
                {
                    'name': 'get_location',
                    'arguments': '{"loc_name": "London"}',
                }
            ),
            'id': '2',
            'type': 'function',
        }
    )
    usage_2 = ChatCompletionOutputUsage.parse_obj_as_instance(  # type:ignore
        {
            'prompt_tokens': 2,
            'completion_tokens': 1,
            'total_tokens': 3,
        }
    )
    responses = [
        completion_message(
            ChatCompletionOutputMessage.parse_obj_as_instance(  # type:ignore
                {
                    'content': None,
                    'role': 'assistant',
                    'tool_calls': [tool_call_1],
                }
            ),
            usage=usage_1,
        ),
        completion_message(
            ChatCompletionOutputMessage.parse_obj_as_instance(  # type:ignore
                {
                    'content': None,
                    'role': 'assistant',
                    'tool_calls': [tool_call_2],
                }
            ),
            usage=usage_2,
        ),
        completion_message(
            ChatCompletionOutputMessage.parse_obj_as_instance(  # type:ignore
                {
                    'content': 'final response',
                    'role': 'assistant',
                }
            ),
        ),
    ]
    mock_client = MockHuggingFace.create_mock(responses)
    model = HuggingFaceModel('hf-model', provider=HuggingFaceProvider(hf_client=mock_client, api_key='x'))
    agent = Agent(model, system_prompt='this is the system prompt')

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
                usage=RequestUsage(input_tokens=1, output_tokens=1),
                model_name='hf-model',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='huggingface',
                provider_url='https://api-inference.huggingface.co',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='123',
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
                usage=RequestUsage(input_tokens=2, output_tokens=1),
                model_name='hf-model',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='huggingface',
                provider_url='https://api-inference.huggingface.co',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='123',
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
                model_name='hf-model',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='huggingface',
                provider_url='https://api-inference.huggingface.co',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='123',
                run_id=IsStr(),
            ),
        ]
    )


FinishReason = Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call']


def chunk(
    delta: list[ChatCompletionStreamOutputDelta], finish_reason: FinishReason | None = None
) -> ChatCompletionStreamOutput:
    return ChatCompletionStreamOutput.parse_obj_as_instance(  # type: ignore
        {
            'id': 'x',
            'choices': [
                ChatCompletionStreamOutputChoice(index=index, delta=delta, finish_reason=finish_reason)  # type: ignore
                for index, delta in enumerate(delta)
            ],
            'created': 1704067200,  # 2024-01-01
            'model': 'hf-model',
            'object': 'chat.completion.chunk',
            'usage': ChatCompletionStreamOutputUsage(completion_tokens=1, prompt_tokens=2, total_tokens=3),  # type: ignore
        }
    )


def text_chunk(text: str, finish_reason: FinishReason | None = None) -> ChatCompletionStreamOutput:
    return chunk([ChatCompletionStreamOutputDelta(content=text, role='assistant')], finish_reason=finish_reason)  # type: ignore


async def test_stream_text(allow_model_requests: None):
    stream = [text_chunk('hello '), text_chunk('world'), chunk([])]
    mock_client = MockHuggingFace.create_stream_mock(stream)
    m = HuggingFaceModel('hf-model', provider=HuggingFaceProvider(hf_client=mock_client, api_key='x'))
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(['hello ', 'hello world'])
        assert result.is_complete
        assert result.usage() == snapshot(RunUsage(requests=1, input_tokens=6, output_tokens=3))


async def test_stream_text_finish_reason(allow_model_requests: None):
    stream = [
        text_chunk('hello '),
        text_chunk('world'),
        text_chunk('.', finish_reason='stop'),
    ]
    mock_client = MockHuggingFace.create_stream_mock(stream)
    m = HuggingFaceModel('hf-model', provider=HuggingFaceProvider(hf_client=mock_client, api_key='x'))
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(
            ['hello ', 'hello world', 'hello world.']
        )
        assert result.is_complete


def struc_chunk(
    tool_name: str | None, tool_arguments: str | None, finish_reason: FinishReason | None = None
) -> ChatCompletionStreamOutput:
    return chunk(
        [
            ChatCompletionStreamOutputDelta.parse_obj_as_instance(  # type: ignore
                {
                    'role': 'assistant',
                    'tool_calls': [
                        ChatCompletionStreamOutputDeltaToolCall.parse_obj_as_instance(  # type: ignore
                            {
                                'index': 0,
                                'function': ChatCompletionStreamOutputFunction.parse_obj_as_instance(  # type: ignore
                                    {
                                        'name': tool_name,
                                        'arguments': tool_arguments,
                                    }
                                ),
                            }
                        )
                    ],
                }
            ),
        ],
        finish_reason=finish_reason,
    )


class MyTypedDict(TypedDict, total=False):
    first: str
    second: str


async def test_stream_structured(allow_model_requests: None):
    stream = [
        chunk([ChatCompletionStreamOutputDelta(role='assistant')]),  # type: ignore
        chunk([ChatCompletionStreamOutputDelta(role='assistant', tool_calls=[])]),  # type: ignore
        chunk(
            [
                ChatCompletionStreamOutputDelta(
                    role='assistant',  # type: ignore
                    tool_calls=[  # type: ignore
                        ChatCompletionStreamOutputDeltaToolCall(id='0', type='function', index=0, function=None)  # type: ignore
                    ],
                )
            ]
        ),
        chunk(
            [
                ChatCompletionStreamOutputDelta(
                    role='assistant',  # type: ignore
                    tool_calls=[  # type: ignore
                        ChatCompletionStreamOutputDeltaToolCall(id='0', type='function', index=0, function=None)  # type: ignore
                    ],
                )
            ]
        ),
        struc_chunk('final_result', None),
        chunk(
            [
                ChatCompletionStreamOutputDelta(
                    role='assistant',  # type: ignore
                    tool_calls=[  # type: ignore
                        ChatCompletionStreamOutputDeltaToolCall(id='0', type='function', index=0, function=None)  # type: ignore
                    ],
                )
            ]
        ),
        struc_chunk(None, '{"first": "One'),
        struc_chunk(None, '", "second": "Two"'),
        struc_chunk(None, '}'),
        chunk([]),
    ]
    mock_client = MockHuggingFace.create_stream_mock(stream)
    m = HuggingFaceModel('hf-model', provider=HuggingFaceProvider(hf_client=mock_client, api_key='x'))
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
    mock_client = MockHuggingFace.create_stream_mock(stream)
    m = HuggingFaceModel('hf-model', provider=HuggingFaceProvider(hf_client=mock_client, api_key='x'))
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
    stream = [
        chunk([]),
        text_chunk('hello '),
        text_chunk('world'),
    ]
    mock_client = MockHuggingFace.create_stream_mock(stream)
    m = HuggingFaceModel('hf-model', provider=HuggingFaceProvider(hf_client=mock_client, api_key='x'))
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(['hello ', 'hello world'])
        assert result.is_complete
        assert result.usage() == snapshot(RunUsage(requests=1, input_tokens=6, output_tokens=3))


@pytest.mark.vcr()
async def test_image_url_input(allow_model_requests: None, huggingface_api_key: str):
    m = HuggingFaceModel(
        'Qwen/Qwen2.5-VL-72B-Instruct',
        provider=HuggingFaceProvider(provider_name='nebius', api_key=huggingface_api_key),
    )
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
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Hello! How can I assist you with this image of a potato?')],
                usage=RequestUsage(input_tokens=269, output_tokens=15),
                model_name='Qwen/Qwen2.5-VL-72B-Instruct',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='huggingface',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2025, 7, 8, 14, 4, 39, tzinfo=timezone.utc),
                },
                provider_response_id='chatcmpl-49aa100effab4ca28514d5ccc00d7944',
                run_id=IsStr(),
            ),
        ]
    )


@pytest.mark.vcr()
async def test_image_as_binary_content_input(
    allow_model_requests: None, image_content: BinaryContent, huggingface_api_key: str
):
    m = HuggingFaceModel(
        'Qwen/Qwen2.5-VL-72B-Instruct',
        provider=HuggingFaceProvider(provider_name='nebius', api_key=huggingface_api_key),
    )
    agent = Agent(m)
    result = await agent.run(['What fruit is in the image?', image_content])
    assert result.output == snapshot(
        'The fruit in the image is a kiwi. It has been sliced in half, revealing its bright green flesh with small black seeds arranged in a circular pattern around a white center. The outer skin of the kiwi is fuzzy and brown.'
    )


def test_model_status_error(allow_model_requests: None) -> None:
    error = HfHubHTTPError(message='test_error', response=Mock(status_code=500, content={'error': 'test error'}))
    mock_client = MockHuggingFace.create_mock(error)
    m = HuggingFaceModel('not_a_model', provider=HuggingFaceProvider(hf_client=mock_client, api_key='x'))
    agent = Agent(m)
    with pytest.raises(ModelHTTPError) as exc_info:
        agent.run_sync('hello')
    assert str(exc_info.value) == snapshot("status_code: 500, model_name: not_a_model, body: {'error': 'test error'}")


@pytest.mark.vcr()
async def test_request_simple_success_with_vcr(allow_model_requests: None, huggingface_api_key: str):
    m = HuggingFaceModel(
        'Qwen/Qwen2.5-72B-Instruct', provider=HuggingFaceProvider(provider_name='nebius', api_key=huggingface_api_key)
    )
    agent = Agent(m)
    result = await agent.run('hello')
    assert result.output == snapshot(
        'Hello! How can I assist you today? Feel free to ask me any questions or let me know if you need help with anything specific.'
    )


@pytest.mark.vcr()
async def test_hf_model_instructions(allow_model_requests: None, huggingface_api_key: str):
    m = HuggingFaceModel(
        'Qwen/Qwen2.5-72B-Instruct', provider=HuggingFaceProvider(provider_name='nebius', api_key=huggingface_api_key)
    )

    def simple_instructions(ctx: RunContext):
        return 'You are a helpful assistant.'

    agent = Agent(m, instructions=simple_instructions)

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
                parts=[TextPart(content='Paris')],
                usage=RequestUsage(input_tokens=26, output_tokens=2),
                model_name='Qwen/Qwen2.5-72B-Instruct-fast',
                timestamp=IsDatetime(),
                provider_name='huggingface',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2025, 7, 2, 15, 39, 17, tzinfo=timezone.utc),
                },
                provider_response_id='chatcmpl-b3936940372c481b8d886e596dc75524',
                run_id=IsStr(),
            ),
        ]
    )


@pytest.mark.parametrize(
    'model_name', ['Qwen/Qwen2.5-72B-Instruct', 'deepseek-ai/DeepSeek-R1-0528', 'meta-llama/Llama-3.3-70B-Instruct']
)
@pytest.mark.vcr()
async def test_max_completion_tokens(allow_model_requests: None, model_name: str, huggingface_api_key: str):
    m = HuggingFaceModel(model_name, provider=HuggingFaceProvider(provider_name='nebius', api_key=huggingface_api_key))
    agent = Agent(m, model_settings=ModelSettings(max_tokens=100))

    result = await agent.run('hello')
    assert result.output == IsStr()


def test_system_property():
    model = HuggingFaceModel('some-model', provider=HuggingFaceProvider(hf_client=Mock(), api_key='x'))
    assert model.system == 'huggingface'


async def test_model_client_response_error(allow_model_requests: None) -> None:
    request_info = Mock(spec=aiohttp.RequestInfo)
    request_info.url = 'http://test.com'
    request_info.method = 'POST'
    request_info.headers = {}
    request_info.real_url = 'http://test.com'
    error = aiohttp.ClientResponseError(request_info, history=(), status=400, message='Bad Request')

    mock_client = MockHuggingFace.create_mock(error)
    m = HuggingFaceModel('not_a_model', provider=HuggingFaceProvider(hf_client=mock_client, api_key='x'))
    agent = Agent(m)
    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('hello')
    assert str(exc_info.value) == snapshot('status_code: 400, model_name: not_a_model, body: Bad Request')


async def test_process_response_no_created_timestamp(allow_model_requests: None):
    c = completion_message(
        ChatCompletionOutputMessage.parse_obj_as_instance({'content': 'response', 'role': 'assistant'}),  # type: ignore
    )
    c.created = None  # type: ignore

    mock_client = MockHuggingFace.create_mock(c)
    model = HuggingFaceModel(
        'test-model',
        provider=HuggingFaceProvider(hf_client=mock_client, api_key='x'),
    )
    agent = Agent(model)
    result = await agent.run('Hello')
    messages = result.all_messages()
    response_message = messages[1]
    assert isinstance(response_message, ModelResponse)
    assert response_message.timestamp == IsNow(tz=timezone.utc)


async def test_retry_prompt_without_tool_name(allow_model_requests: None):
    responses = [
        completion_message(
            ChatCompletionOutputMessage.parse_obj_as_instance({'content': 'invalid-response', 'role': 'assistant'})  # type: ignore
        ),
        completion_message(
            ChatCompletionOutputMessage.parse_obj_as_instance({'content': 'final-response', 'role': 'assistant'})  # type: ignore
        ),
    ]

    mock_client = MockHuggingFace.create_mock(responses)
    model = HuggingFaceModel(
        'test-model',
        provider=HuggingFaceProvider(hf_client=mock_client, api_key='x'),
    )
    agent = Agent(model)

    @agent.output_validator
    def response_validator(value: str) -> str:
        if value == 'invalid-response':
            raise ModelRetry('Response is invalid')
        return value

    result = await agent.run('Hello')
    assert result.output == 'final-response'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='invalid-response')],
                model_name='hf-model',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='huggingface',
                provider_url='https://api-inference.huggingface.co',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='123',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Response is invalid',
                        tool_name=None,
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='final-response')],
                model_name='hf-model',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='huggingface',
                provider_url='https://api-inference.huggingface.co',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='123',
                run_id=IsStr(),
            ),
        ]
    )
    kwargs = get_mock_chat_completion_kwargs(mock_client)[1]
    messages = kwargs['messages']
    assert {k: v for k, v in asdict(messages[-2]).items() if v is not None} == {
        'role': 'assistant',
        'content': 'invalid-response',
    }
    assert {k: v for k, v in asdict(messages[-1]).items() if v is not None} == {
        'role': 'user',
        'content': 'Validation feedback:\nResponse is invalid\n\nFix the errors and try again.',
    }


async def test_thinking_part_in_history(allow_model_requests: None):
    c = completion_message(ChatCompletionOutputMessage(content='response', role='assistant'))  # type: ignore
    mock_client = MockHuggingFace.create_mock(c)
    model = HuggingFaceModel('hf-model', provider=HuggingFaceProvider(hf_client=mock_client, api_key='x'))
    agent = Agent(model)
    messages = [
        ModelRequest(parts=[UserPromptPart(content='request')], timestamp=IsDatetime()),
        ModelResponse(
            parts=[
                TextPart(content='text 1'),
                ThinkingPart(content='let me do some thinking'),
                TextPart(content='text 2'),
            ],
            model_name='hf-model',
            timestamp=datetime.now(timezone.utc),
        ),
    ]

    await agent.run('another request', message_history=messages)

    kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    sent_messages = kwargs['messages']
    assert [{k: v for k, v in asdict(m).items() if v is not None} for m in sent_messages] == snapshot(
        [
            {'content': 'request', 'role': 'user'},
            {
                'content': """\
text 1

<think>
let me do some thinking
</think>

text 2\
""",
                'role': 'assistant',
            },
            {'content': 'another request', 'role': 'user'},
        ]
    )


@pytest.mark.parametrize(
    'content_item, error_message',
    [
        (AudioUrl(url='url'), 'AudioUrl is not supported for Hugging Face'),
        (DocumentUrl(url='url'), 'DocumentUrl is not supported for Hugging Face'),
        (VideoUrl(url='url'), 'VideoUrl is not supported for Hugging Face'),
    ],
)
async def test_unsupported_media_types(allow_model_requests: None, content_item: Any, error_message: str):
    model = HuggingFaceModel(
        'Qwen/Qwen2.5-VL-72B-Instruct',
        provider=HuggingFaceProvider(api_key='x'),
    )
    agent = Agent(model)

    with pytest.raises(NotImplementedError, match=error_message):
        await agent.run(['hello', content_item])


@pytest.mark.vcr()
async def test_hf_model_thinking_part(allow_model_requests: None, huggingface_api_key: str):
    m = HuggingFaceModel(
        'Qwen/Qwen3-235B-A22B', provider=HuggingFaceProvider(provider_name='nebius', api_key=huggingface_api_key)
    )
    agent = Agent(m)

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
                    IsInstance(ThinkingPart),
                    IsInstance(TextPart),
                ],
                usage=RequestUsage(input_tokens=15, output_tokens=1090),
                model_name='Qwen/Qwen3-235B-A22B',
                timestamp=IsDatetime(),
                provider_name='huggingface',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2025, 7, 9, 13, 17, 45, tzinfo=timezone.utc),
                },
                provider_response_id='chatcmpl-957db61fe60d4440bcfe1f11f2c5b4b9',
                run_id=IsStr(),
            ),
        ]
    )

    result = await agent.run(
        'Considering the way to cross the street, analogously, how do I cross the river?',
        model=HuggingFaceModel(
            'Qwen/Qwen3-235B-A22B', provider=HuggingFaceProvider(provider_name='nebius', api_key=huggingface_api_key)
        ),
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
                parts=[
                    IsInstance(ThinkingPart),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(input_tokens=691, output_tokens=1860),
                model_name='Qwen/Qwen3-235B-A22B',
                timestamp=IsDatetime(),
                provider_name='huggingface',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2025, 7, 9, 13, 18, 14, tzinfo=timezone.utc),
                },
                provider_response_id='chatcmpl-35fdec1307634f94a39f7e26f52e12a7',
                run_id=IsStr(),
            ),
        ]
    )


@pytest.mark.vcr()
async def test_hf_model_thinking_part_iter(allow_model_requests: None, huggingface_api_key: str):
    m = HuggingFaceModel(
        'Qwen/Qwen3-235B-A22B', provider=HuggingFaceProvider(provider_name='nebius', api_key=huggingface_api_key)
    )
    agent = Agent(m)

    result: AgentRunResult | None = None
    async for event in agent.run_stream_events(user_prompt='How do I cross the street?'):
        if isinstance(event, AgentRunResultEvent):
            result = event.result

    assert result is not None
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='How do I cross the street?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content=IsStr()),
                    TextPart(content=IsStr()),
                ],
                model_name='Qwen/Qwen3-235B-A22B',
                timestamp=IsDatetime(),
                provider_name='huggingface',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2025, 7, 23, 19, 58, 41, tzinfo=timezone.utc),
                },
                provider_response_id='chatcmpl-357f347a3f5d4897b36a128fb4e4cf7b',
                run_id=IsStr(),
            ),
        ]
    )


async def test_cache_point_filtering():
    """Test that CachePoint is filtered out in HuggingFace message mapping."""
    # Test the static method directly
    msg = await HuggingFaceModel._map_user_prompt(UserPromptPart(content=['text', CachePoint()]))  # pyright: ignore[reportPrivateUsage]

    # CachePoint should be filtered out
    assert msg['role'] == 'user'
    assert len(msg['content']) == 1  # pyright: ignore[reportUnknownArgumentType]
