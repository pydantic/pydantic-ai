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
    DocumentUrl,
    FinalResultEvent,
    ImageUrl,
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
    VideoUrl,
)
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.result import RunUsage
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
            timestamp=datetime(2025, 7, 8, 13, 42, 33, tzinfo=timezone.utc),
            provider_name='huggingface',
            provider_details={'finish_reason': 'stop'},
            provider_response_id='chatcmpl-d445c0d473a84791af2acf356cc00df7',
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
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            provider_name='huggingface',
            provider_details={'finish_reason': 'stop'},
            provider_response_id='123',
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
                usage=RequestUsage(input_tokens=1, output_tokens=1),
                model_name='hf-model',
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                provider_name='huggingface',
                provider_details={'finish_reason': 'stop'},
                provider_response_id='123',
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
                usage=RequestUsage(input_tokens=2, output_tokens=1),
                model_name='hf-model',
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                provider_name='huggingface',
                provider_details={'finish_reason': 'stop'},
                provider_response_id='123',
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
                model_name='hf-model',
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                provider_name='huggingface',
                provider_details={'finish_reason': 'stop'},
                provider_response_id='123',
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
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='Hello! How can I assist you with this image of a potato?')],
                usage=RequestUsage(input_tokens=269, output_tokens=15),
                model_name='Qwen/Qwen2.5-VL-72B-Instruct',
                timestamp=datetime(2025, 7, 8, 14, 4, 39, tzinfo=timezone.utc),
                provider_name='huggingface',
                provider_details={'finish_reason': 'stop'},
                provider_response_id='chatcmpl-49aa100effab4ca28514d5ccc00d7944',
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
                instructions='You are a helpful assistant.',
            ),
            ModelResponse(
                parts=[TextPart(content='Paris')],
                usage=RequestUsage(input_tokens=26, output_tokens=2),
                model_name='Qwen/Qwen2.5-72B-Instruct-fast',
                timestamp=IsDatetime(),
                provider_name='huggingface',
                provider_details={'finish_reason': 'stop'},
                provider_response_id='chatcmpl-b3936940372c481b8d886e596dc75524',
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
    error.response_error_payload = {'error': 'test error'}  # type: ignore

    mock_client = MockHuggingFace.create_mock(error)
    m = HuggingFaceModel('not_a_model', provider=HuggingFaceProvider(hf_client=mock_client, api_key='x'))
    agent = Agent(m)
    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('hello')
    assert str(exc_info.value) == snapshot("status_code: 400, model_name: not_a_model, body: {'error': 'test error'}")


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
            ModelRequest(parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[TextPart(content='invalid-response')],
                model_name='hf-model',
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                provider_name='huggingface',
                provider_details={'finish_reason': 'stop'},
                provider_response_id='123',
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Response is invalid',
                        tool_name=None,
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='final-response')],
                model_name='hf-model',
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                provider_name='huggingface',
                provider_details={'finish_reason': 'stop'},
                provider_response_id='123',
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
        ModelRequest(parts=[UserPromptPart(content='request')]),
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
            ModelRequest(parts=[UserPromptPart(content='How do I cross the street?', timestamp=IsDatetime())]),
            ModelResponse(
                parts=[
                    IsInstance(ThinkingPart),
                    IsInstance(TextPart),
                ],
                usage=RequestUsage(input_tokens=15, output_tokens=1090),
                model_name='Qwen/Qwen3-235B-A22B',
                timestamp=IsDatetime(),
                provider_name='huggingface',
                provider_details={'finish_reason': 'stop'},
                provider_response_id='chatcmpl-957db61fe60d4440bcfe1f11f2c5b4b9',
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
                ]
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
                provider_details={'finish_reason': 'stop'},
                provider_response_id='chatcmpl-35fdec1307634f94a39f7e26f52e12a7',
            ),
        ]
    )


@pytest.mark.vcr()
async def test_hf_model_thinking_part_iter(allow_model_requests: None, huggingface_api_key: str):
    m = HuggingFaceModel(
        'Qwen/Qwen3-235B-A22B', provider=HuggingFaceProvider(provider_name='nebius', api_key=huggingface_api_key)
    )
    agent = Agent(m)

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='How do I cross the street?') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert event_parts == snapshot(
        [
            PartStartEvent(index=0, part=ThinkingPart(content='')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='\n')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Okay')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' user')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' asking')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' how')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cross')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' street')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Let')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' me')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' think')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' about')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' basic')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' steps')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' First')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' should')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' check')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' traffic')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' need')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' emphasize')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' looking')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' both')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ways')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Wait')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' but')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' some')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' places')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' people')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' drive')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' on')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' right')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' left')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' side')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' so')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' mention')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' looking')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' left')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' then')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' right')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' right')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' then')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' left')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' depending')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' on')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' country')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Then')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' should')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' wait')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' safe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' gap')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' But')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' what')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' about')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' intersections')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' traffic')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' lights')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='?')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' They')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' need')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' signals')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' mention')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' pedestrian')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' crossings')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cross')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='walk')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='s')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Also')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' distractions')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' like')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' phones')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' can')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' be')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' dangerous')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Should')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' include')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' something')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' about')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' not')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' using')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' phones')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' while')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' crossing')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.

"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Oh')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' accessibility')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' some')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' people')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' might')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' use')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' wheel')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ch')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='airs')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' walkers')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' so')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cross')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='walk')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='s')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' should')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' be')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' accessible')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Also')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' if')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' there')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'s")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' curb')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' check')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ramps')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Oh')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' right')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' traffic')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' signals')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' pedestrians')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='—')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='like')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' walk')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='/d')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='on')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'t")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' walk')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' signs')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' tell')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' them')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' start')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' crossing')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' only')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' when')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' walk')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' signal')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' on')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' And')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' halfway')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' check')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' again')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='?')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Also')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' if')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' there')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'s")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' public')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' transport')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' like')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' buses')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' wait')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' them')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' stop')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' before')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' crossing')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' front')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' What')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' about')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' bicycles')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='?')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' In')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' some')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' places')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' bike')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' lanes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' are')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' present')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' so')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' watch')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cyclists')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' too')),
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
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' user')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' child')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='?')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' advice')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' needs')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' be')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' simple')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' But')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' since')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' I')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' don')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'t")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' know')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' age')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' better')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' cover')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' all')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ages')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Also')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' some')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' areas')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' drivers')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' might')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' not')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' always')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' obey')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' rules')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' so')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' stay')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' vigilant')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Alcohol')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' drugs')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' impair')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' judgment')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' so')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' avoid')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' crossing')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' under')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' their')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' influence')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' add')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' tips')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' about')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' visibility')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' like')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' at')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' night')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' wear')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' bright')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' clothes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Okay')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' organize')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' these')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' thoughts')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' step')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' by')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' step')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' starting')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' preparation')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' choosing')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' right')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' spot')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' checking')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' traffic')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' timing')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' signals')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' safe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' crossing')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' steps')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' additional')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' tips')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Make')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' sure')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' it')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'s")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' clear')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' covers')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' all')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' scenarios')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Check')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' any')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' missing')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' points')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' like')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' emergency')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' situations')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' what')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' do')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' if')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' stuck')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' mid')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='-cross')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='ing')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' H')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='mmm')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' maybe')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' mention')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' if')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' halfway')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' a')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' car')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' comes')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' keep')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' going')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' or')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' adjust')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' steps')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Also')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' if')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' unsure')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' don')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="'t")),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' hesitate')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' to')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' ask')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' help')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' Okay')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=',')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' structure')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' answer')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' these')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' points')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' in')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' mind')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='.\n')),
            PartEndEvent(
                index=0,
                part=ThinkingPart(
                    content="""\

Okay, the user is asking how to cross the street. Let me think about the basic steps. First, they should check for traffic. I need to emphasize looking both ways. Wait, but in some places, people drive on the right or left side, so maybe mention looking left then right, or right then left depending on the country. Then they should wait for a safe gap. But what about intersections with traffic lights? They need to use the signals. Maybe mention pedestrian crossings and crosswalks. Also, distractions like phones can be dangerous. Should include something about not using phones while crossing.

Oh, and for accessibility, some people might use wheelchairs or walkers, so crosswalks should be accessible. Also, if there's a curb, check for ramps. Oh, right, traffic signals for pedestrians—like walk/don't walk signs. Maybe tell them to start crossing only when the walk signal is on. And halfway, check again? Also, if there's public transport like buses, wait for them to stop before crossing in front. What about bicycles? In some places, bike lanes are present, so watch for cyclists too.

Wait, is the user a child? Maybe the advice needs to be simple. But since I don't know the age, better cover all ages. Also, in some areas, drivers might not always obey the rules, so stay vigilant. Alcohol or drugs impair judgment, so avoid crossing under their influence. Maybe add tips about visibility, like at night, wear bright clothes. Okay, organize these thoughts step by step, starting with preparation, choosing the right spot, checking traffic, timing with signals, safe crossing steps, and additional tips. Make sure it's clear and covers all scenarios. Check for any missing points like emergency situations or what to do if stuck mid-crossing. Hmmm, maybe mention if halfway and a car comes, keep going or adjust steps. Also, if unsure, don't hesitate to ask for help. Okay, structure the answer with these points in mind.
"""
                ),
                next_part_kind='text',
            ),
            PartStartEvent(index=1, part=TextPart(content='Cross'), previous_part_kind='thinking'),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ing')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' street')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' safely')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' requires')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' attent')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='iveness')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' awareness')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' of')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' your')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' surroundings')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Here')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta="'s")),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' step')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-by')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-step')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' guide')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' help')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ensure')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' your')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' safety')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
:

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
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='1')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Choose')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Right')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Spot')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Use')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cross')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='walk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' intersection')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**:')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Cross')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' at')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' marked')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cross')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='walk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='s')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' intersections')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' traffic')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' signals')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' where')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' pedestrians')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' commonly')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cross')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Avoid')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' blind')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' spots')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**:')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Do')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' not')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cross')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' near')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' curves')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' hills')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' obstacles')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='e')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.g')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.,')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' parked')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cars')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' bushes')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=')')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' that')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' block')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' drivers')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' views')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
.

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
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='2')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Check')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Traffic')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Stop')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' at')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' curb')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**:')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Face')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' street')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' scan')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' left')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' center')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' right')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' approaching')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' vehicles')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Look')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' listen')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**:')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Watch')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cars')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cyclists')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' motor')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='cycl')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ists')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Make')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' eye')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' contact')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' drivers')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' confirm')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' they')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' see')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' you')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Right')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-hand')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' vs')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' left')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-hand')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' traffic')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='    ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' In')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' countries')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' where')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' vehicles')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' drive')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' on')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='right')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='e')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.g')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.,')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' U')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.S')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.,')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' China')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='):')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Look')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='left')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**,')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' then')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' right')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' then')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' left')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' again')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='    ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' In')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' countries')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' where')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' vehicles')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' drive')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' on')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='left')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='e')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.g')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.,')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' U')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.K')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.,')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Japan')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='):')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Flip')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' this')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' process')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='—')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='look')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='right')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**,')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' then')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' left')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' then')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' right')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
.

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
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='3')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Use')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Traffic')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Signals')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Wait')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' pedestrian')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' signal')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**:')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' If')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' available')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' wait')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' until')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' "')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Walk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='"')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' signal')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' green')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' hand')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=')')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' appears')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Start')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' crossing')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' promptly')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**:')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Begin')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' when')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' signal')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' changes')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' If')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' it')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' still')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' flashing')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' finish')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' crossing')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' but')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' move')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' quickly')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='No')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' signal')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='?')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Wait')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' safe')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' gap')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' All')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' lanes')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' should')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' be')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' clear')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' of')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' approaching')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' vehicles')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' before')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' stepping')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' off')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' curb')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
.

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
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='4')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Cross')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Cal')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='m')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ly')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Predict')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ably')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Walk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='don')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' run')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=')**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=':')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Maintain')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' steady')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' pace')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Never')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' dart')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' into')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' street')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Keep')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' looking')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**:')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Gl')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ance')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' both')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ways')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' as')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' you')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cross')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Stay')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' alert')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' drivers')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' who')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' may')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' turn')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ignore')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' light')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Avoid')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' distractions')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**:')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Put')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' your')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' phone')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' down')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='—')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='don')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' text')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' wear')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' headphones')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' browse')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' while')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' crossing')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
.

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
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='5')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Stay')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Visible')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='W')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ear')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' bright')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='/')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='reflect')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ive')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' clothing')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**,')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' especially')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' at')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' night')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' low')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' visibility')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='rain')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' fog')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=').\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Use')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' sidewalks')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' shoulders')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' if')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' no')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cross')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='walk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='s')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' exist')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' your')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' area')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
.

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
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='6')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Additional')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Tips')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='If')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' stuck')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' mid')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='-way')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**:')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Stop')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' wait')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' signal')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' change')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' again')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Most')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' signals')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' allow')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' time')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' pause')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='B')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='uses')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='/tr')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ains')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**:')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Wait')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' until')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' vehicle')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' stops')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' entirely')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' before')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' crossing')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' in')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' front')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Use')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cross')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='walk')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='s')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' near')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' bus')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' stops')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='C')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ycl')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ists')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**:')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Watch')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' bike')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' lanes')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Some')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' drivers')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' may')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' not')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' notice')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cyclists')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' so')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' be')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' extra')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cautious')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Accessibility')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**:')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Use')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ramps')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' accessible')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' pedestrian')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' signals')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='APS')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=')')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' if')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' needed')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
.

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
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='7')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Stay')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Vig')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='il')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='ant')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Never')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' assume')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' driver')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' will')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' stop')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Always')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' confirm')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' their')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' intent')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' yield')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Avoid')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' crossing')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' under')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' the')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' influence')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' of')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' drugs')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='/al')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='cohol')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' which')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' slow')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' reaction')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' times')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='.\n')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='  ')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' If')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' unsure')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ask')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' local')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' or')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' wait')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' group')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' to')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' cross')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' together')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' added')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' safety')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
.

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
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='By')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' following')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' these')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' steps')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' you')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' minimize')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' risks')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' and')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' ensure')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' a')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' safer')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' crossing')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='!')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' 🚶')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='Stay')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' alert')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' stay')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' alive')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='!')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='**')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='')),
            PartEndEvent(
                index=1,
                part=TextPart(content=IsStr()),
            ),
        ]
    )
