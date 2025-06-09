from __future__ import annotations as _annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from typing import Any, Literal, Union, cast
from unittest.mock import Mock

import pytest
from huggingface_hub import (
    ChatCompletionStreamOutputChoice,
    ChatCompletionStreamOutputDelta,
    ChatCompletionStreamOutputDeltaToolCall,
    ChatCompletionStreamOutputFunction,
    ChatCompletionStreamOutputUsage,
)
from inline_snapshot import snapshot
from typing_extensions import TypedDict

from pydantic_ai import Agent, ModelRetry, UnexpectedModelBehavior
from pydantic_ai.exceptions import ModelHTTPError
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
from pydantic_ai.tools import RunContext

from ..conftest import IsDatetime, IsNow, raise_if_exception, try_import
from .mock_async_stream import MockAsyncStream

with try_import() as imports_successful:
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
    )
    from huggingface_hub.errors import HfHubHTTPError

    from pydantic_ai.models.huggingface import HuggingFaceModel
    from pydantic_ai.providers.huggingface import HuggingFaceProvider

    MockChatCompletion = Union[ChatCompletionOutput, Exception]
    MockStreamEvent = Union[ChatCompletionStreamOutput, Exception]

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='huggingface_hub not installed'),
    pytest.mark.anyio,
]


@dataclass
class MockHuggingFace:
    completions: MockChatCompletion | Sequence[MockChatCompletion] | None = None
    stream: Sequence[MockStreamEvent] | Sequence[Sequence[MockStreamEvent]] | None = None
    index: int = 0

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
        self, *_args: Any, stream: bool = False, **_kwargs: Any
    ) -> ChatCompletionOutput | MockAsyncStream[MockStreamEvent]:
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


async def test_simple_completion(allow_model_requests: None):
    c = completion_message(ChatCompletionInputMessage(content='world', role='assistant'))  # type:ignore
    mock_client = MockHuggingFace.create_mock(c)
    model = HuggingFaceModel(
        'Qwen/Qwen2.5-72B-Instruct', provider=HuggingFaceProvider(provider='nebius', hf_client=mock_client, api_key='x')
    )
    agent = Agent(model)

    result = await agent.run('hello')
    assert result.output == 'world'
    messages = result.all_messages()
    request = messages[0]
    response = messages[1]
    assert request.parts[0].content == 'hello'  # type: ignore
    assert response == ModelResponse(
        parts=[TextPart(content='world')],
        usage=Usage(requests=1),
        model_name='hf-model',
        timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
        vendor_id='123',
    )


async def test_request_simple_usage(allow_model_requests: None):
    c = completion_message(ChatCompletionInputMessage(content='world', role='assistant'))  # type:ignore
    mock_client = MockHuggingFace.create_mock(c)
    model = HuggingFaceModel(
        'Qwen/Qwen2.5-72B-Instruct', provider=HuggingFaceProvider(provider='nebius', hf_client=mock_client, api_key='x')
    )
    agent = Agent(model)

    result = await agent.run('Hello')
    assert result.output == 'world'
    assert result.usage() == snapshot(Usage(requests=1))


async def test_request_structured_response(allow_model_requests: None):
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
        'Qwen/Qwen2.5-72B-Instruct', provider=HuggingFaceProvider(provider='nebius', hf_client=mock_client, api_key='x')
    )
    agent = Agent(model, output_type=list[int])

    result = await agent.run('Hello')
    assert result.output == [1, 2, 123]
    messages = result.all_messages()
    assert messages[0].parts[0].content == 'Hello'  # type: ignore
    assert messages[1] == ModelResponse(
        parts=[
            ToolCallPart(
                tool_name='final_result',
                args='{"response": [1, 2, 123]}',
                tool_call_id='123',
            )
        ],
        usage=Usage(requests=1),
        model_name='hf-model',
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        vendor_id='123',
    )


async def test_stream_completion(allow_model_requests: None):
    stream = [text_chunk('hello '), text_chunk('world', finish_reason='stop')]
    mock_client = MockHuggingFace.create_stream_mock(stream)
    model = HuggingFaceModel('hf-model', provider=HuggingFaceProvider(hf_client=mock_client, api_key='x'))
    agent = Agent(model)

    async with agent.run_stream('') as result:
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(['hello ', 'hello world'])


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
                usage=Usage(requests=1, request_tokens=1, response_tokens=1, total_tokens=2),
                model_name='hf-model',
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                vendor_id='123',
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
                usage=Usage(requests=1, request_tokens=2, response_tokens=1, total_tokens=3),
                model_name='hf-model',
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
                model_name='hf-model',
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                vendor_id='123',
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
    m = HuggingFaceModel('hf-model', provider=HuggingFaceProvider(hf_client=mock_client))
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
    mock_client = MockHuggingFace.create_stream_mock(stream)
    m = HuggingFaceModel('hf-model', provider=HuggingFaceProvider(hf_client=mock_client))
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
    m = HuggingFaceModel('hf-model', provider=HuggingFaceProvider(hf_client=mock_client))
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
    mock_client = MockHuggingFace.create_stream_mock(stream)
    m = HuggingFaceModel('hf-model', provider=HuggingFaceProvider(hf_client=mock_client))
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
    stream = [
        chunk([ChatCompletionStreamOutputDelta(role='assistant')]),  # type: ignore
        chunk([ChatCompletionStreamOutputDelta(role='assistant')]),  # type: ignore
    ]
    mock_client = MockHuggingFace.create_stream_mock(stream)
    m = HuggingFaceModel('hf-model', provider=HuggingFaceProvider(hf_client=mock_client))
    agent = Agent(m, output_type=MyTypedDict)

    with pytest.raises(UnexpectedModelBehavior, match='Received empty model response'):
        async with agent.run_stream(''):
            pass


async def test_no_delta(allow_model_requests: None):
    stream = [
        chunk([]),
        text_chunk('hello '),
        text_chunk('world'),
    ]
    mock_client = MockHuggingFace.create_stream_mock(stream)
    m = HuggingFaceModel('hf-model', provider=HuggingFaceProvider(hf_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(['hello ', 'hello world'])
        assert result.is_complete
        assert result.usage() == snapshot(Usage(requests=1, request_tokens=6, response_tokens=3, total_tokens=9))


async def test_image_url_input(allow_model_requests: None):
    c = completion_message(ChatCompletionInputMessage(content='world', role='assistant'))  # type:ignore
    mock_client = MockHuggingFace.create_mock(c)
    m = HuggingFaceModel('hf-model', provider=HuggingFaceProvider(hf_client=mock_client))
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
                                url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg'
                            ),
                        ],
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='world')],
                usage=Usage(requests=1),
                model_name='hf-model',
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                vendor_id='123',
            ),
        ]
    )


async def test_image_as_binary_content_input(allow_model_requests: None):
    c = completion_message(ChatCompletionInputMessage(content='world', role='assistant'))  # type: ignore
    mock_client = MockHuggingFace.create_mock(c)
    m = HuggingFaceModel('hf-model', provider=HuggingFaceProvider(hf_client=mock_client))
    agent = Agent(m)

    base64_content = (
        b'/9j/4AAQSkZJRgABAQEAYABgAAD/4QBYRXhpZgAATU0AKgAAAAgAA1IBAAEAAAABAAAAPgIBAAEAAAABAAAARgMBAAEAAAABAAAA'
        b'WgAAAAAAAAAE'
    )

    result = await agent.run(['hello', BinaryContent(data=base64_content, media_type='image/jpeg')])
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=['hello', BinaryContent(data=base64_content, media_type='image/jpeg')],
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='world')],
                usage=Usage(requests=1),
                model_name='hf-model',
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                vendor_id='123',
            ),
        ]
    )


def test_model_status_error(allow_model_requests: None) -> None:
    error = HfHubHTTPError(message='test_error', response=Mock(status_code=500, content={'error': 'test error'}))
    mock_client = MockHuggingFace.create_mock(error)
    m = HuggingFaceModel('not_a_model', provider=HuggingFaceProvider(hf_client=mock_client))
    agent = Agent(m)
    with pytest.raises(ModelHTTPError) as exc_info:
        agent.run_sync('hello')
    assert str(exc_info.value) == snapshot("status_code: 500, model_name: not_a_model, body: {'error': 'test error'}")


@pytest.mark.vcr()
async def test_request_simple_success_with_vcr(allow_model_requests: None, huggingface_api_key: str):
    m = HuggingFaceModel(
        'Qwen/Qwen2.5-72B-Instruct', provider=HuggingFaceProvider(provider='nebius', api_key=huggingface_api_key)
    )
    agent = Agent(m)
    result = await agent.run('hello')
    assert result.output == snapshot(
        "Hello! It's great to meet you. How can I assist you today? Whether you have questions, need information, or just want to chat, I'm here to help!"
    )


@pytest.mark.vcr()
async def test_hf_model_instructions(allow_model_requests: None, huggingface_api_key: str):
    m = HuggingFaceModel(
        'Qwen/Qwen2.5-72B-Instruct', provider=HuggingFaceProvider(provider='nebius', api_key=huggingface_api_key)
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
                usage=Usage(requests=1, request_tokens=26, response_tokens=2, total_tokens=28),
                model_name='Qwen/Qwen2.5-72B-Instruct-fast',
                timestamp=IsDatetime(),
                vendor_id='chatcmpl-54246cfb4fa046e88a984020c4efab20',
            ),
        ]
    )
