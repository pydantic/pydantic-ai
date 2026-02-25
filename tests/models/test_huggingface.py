from __future__ import annotations as _annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import cached_property
from typing import Any, Literal, cast
from unittest.mock import Mock

import pytest
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
    TextContent,
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

from .._inline_snapshot import snapshot
from ..conftest import IsDatetime, IsInstance, IsNow, IsStr, raise_if_exception, try_import
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
    chat_completion_kwargs: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
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
        'deepseek-ai/DeepSeek-R1',
        provider=HuggingFaceProvider(provider_name='together', api_key=huggingface_api_key),
    )
    agent = Agent(model)

    result = await agent.run('hello')
    assert result.output == IsStr()
    messages = result.all_messages()
    request = messages[0]
    response = messages[1]
    assert request.parts[0].content == 'hello'  # type: ignore
    assert response == snapshot(
        ModelResponse(
            parts=[
                ThinkingPart(
                    content="""\

Hmm, the user just said "hello". That's a simple greeting, but I need to figure out how to respond appropriately.  \n\

First, I should acknowledge their greeting warmly since starting friendly sets a positive tone. Maybe "Hello!" with an exclamation to show enthusiasm.  \n\

I wonder if this is just a test message or if they have something specific in mind. The message is very short, so they might be:  \n\
- Checking if I'm active  \n\
- Unsure how to start  \n\
- Or just being polite before asking something else  \n\

Since they didn't follow up immediately, I'll keep it open-ended. Adding "How can I help you today?" invites them to share their actual need without pressure.  \n\

Also, no emojis yet--they didn't use any, so I'll match that tone unless they show preference later. Keeping it simple but friendly feels right here.
"""
                ),
                TextPart(
                    content="""\

Hello! 👋 How can I help you today?\
"""
                ),
            ],
            usage=RequestUsage(input_tokens=4, output_tokens=197),
            model_name='deepseek-ai/DeepSeek-R1',
            timestamp=IsDatetime(),
            provider_name='huggingface',
            provider_url='https://router.huggingface.co/together',
            provider_details={'finish_reason': 'stop', 'timestamp': IsDatetime()},
            provider_response_id='oV1mmQk-28Eivz-9c4b14712ea45a45',
            finish_reason='stop',
            run_id=IsStr(),
        )
    )


@pytest.mark.vcr()
async def test_request_simple_usage(allow_model_requests: None, huggingface_api_key: str):
    model = HuggingFaceModel(
        'deepseek-ai/DeepSeek-R1',
        provider=HuggingFaceProvider(provider_name='together', api_key=huggingface_api_key),
    )
    agent = Agent(model)

    result = await agent.run('Hello')
    assert result.output == IsStr()
    assert result.usage() == snapshot(RunUsage(input_tokens=4, output_tokens=258, requests=1))


@pytest.mark.vcr()
async def test_request_structured_response(allow_model_requests: None, huggingface_api_key: str):
    model = HuggingFaceModel(
        'deepseek-ai/DeepSeek-R1',
        provider=HuggingFaceProvider(provider_name='together', api_key=huggingface_api_key),
    )
    agent = Agent(model, output_type=list[int])

    result = await agent.run('What are the first three prime numbers? Return them as a list of integers.')
    assert result.output == snapshot([2, 3, 5])
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What are the first three prime numbers? Return them as a list of integers.',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"response":[2,3,5]}',
                        tool_call_id='call_7qxjvbuxpm6017n3jcq1uqwt',
                    )
                ],
                usage=RequestUsage(input_tokens=19, output_tokens=29),
                model_name='deepseek-ai/DeepSeek-R1',
                timestamp=IsDatetime(),
                provider_name='huggingface',
                provider_url='https://router.huggingface.co/together',
                provider_details={'finish_reason': 'tool_calls', 'timestamp': IsDatetime()},
                provider_response_id='oV1mqo1-28Eivz-9c4b14ce2f14c9b7',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='call_7qxjvbuxpm6017n3jcq1uqwt',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
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
                    'timestamp': IsDatetime(),
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
                usage=RequestUsage(input_tokens=2, output_tokens=1),
                model_name='hf-model',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='huggingface',
                provider_url='https://api-inference.huggingface.co',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': IsDatetime(),
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
                model_name='hf-model',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='huggingface',
                provider_url='https://api-inference.huggingface.co',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='123',
                finish_reason='stop',
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
                ChatCompletionStreamOutputChoice(index=index, delta=delta, finish_reason=finish_reason)
                for index, delta in enumerate(delta)
            ],
            'created': 1704067200,  # 2024-01-01
            'model': 'hf-model',
            'object': 'chat.completion.chunk',
            'usage': ChatCompletionStreamOutputUsage(completion_tokens=1, prompt_tokens=2, total_tokens=3),
        }
    )


def text_chunk(text: str, finish_reason: FinishReason | None = None) -> ChatCompletionStreamOutput:
    return chunk([ChatCompletionStreamOutputDelta(content=text, role='assistant')], finish_reason=finish_reason)


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
        chunk([ChatCompletionStreamOutputDelta(role='assistant')]),
        chunk([ChatCompletionStreamOutputDelta(role='assistant', tool_calls=[])]),
        chunk(
            [
                ChatCompletionStreamOutputDelta(
                    role='assistant',
                    tool_calls=[
                        ChatCompletionStreamOutputDeltaToolCall(id='0', type='function', index=0, function=None)  # type: ignore
                    ],
                )
            ]
        ),
        chunk(
            [
                ChatCompletionStreamOutputDelta(
                    role='assistant',
                    tool_calls=[
                        ChatCompletionStreamOutputDeltaToolCall(id='0', type='function', index=0, function=None)  # type: ignore
                    ],
                )
            ]
        ),
        struc_chunk('final_result', None),
        chunk(
            [
                ChatCompletionStreamOutputDelta(
                    role='assistant',
                    tool_calls=[
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
                                url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg'
                            ),
                        ],
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='Hello! How can I assist you with the image of the potato? Do you have any specific questions or need information about it?'
                    )
                ],
                usage=RequestUsage(input_tokens=269, output_tokens=27),
                model_name='Qwen/Qwen2.5-VL-72B-Instruct',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='huggingface',
                provider_url='https://router.huggingface.co/nebius',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='chatcmpl-d68e3c40c98e4d3f8ab4ff4cbf81c544',
                finish_reason='stop',
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
        'The fruit in the image is a kiwi. The distinctive green flesh, small black seeds arranged in a circular pattern, and the fuzzy brown skin are characteristic features of a kiwi.'
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
async def test_hf_model_instructions(allow_model_requests: None, huggingface_api_key: str):
    m = HuggingFaceModel(
        'deepseek-ai/DeepSeek-R1',
        provider=HuggingFaceProvider(provider_name='together', api_key=huggingface_api_key),
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
                parts=[
                    ThinkingPart(
                        content="""\

Okay, the user is asking about the capital of France. That's a straightforward geography question. \n\

Hmm, I recall that Paris is the capital--it's one of those basic facts everyone learns in school. The user might be a student doing homework, a traveler planning a trip, or just someone confirming a trivia detail. Since the question is simple, they probably want a quick, clear answer without extra fluff. \n\

But wait--should I add more context? Like how Paris is also France's cultural and economic hub? Nah, the query doesn't hint at needing depth. Over-explaining might annoy them. Keep it concise: "Paris" plus a tiny assurance ("That's correct!") to sound friendly. \n\

...Though if they follow up, I could mention the Seine River or the Eiffel Tower. For now, brevity wins.
"""
                    ),
                    TextPart(
                        content="""\

The capital of France is **Paris**.  \n\

That's correct! Paris is not only the political center but also the cultural, economic, and historical heart of France. 🇫🇷\
"""
                    ),
                ],
                usage=RequestUsage(input_tokens=16, output_tokens=216),
                model_name='deepseek-ai/DeepSeek-R1',
                timestamp=IsDatetime(),
                provider_name='huggingface',
                provider_url='https://router.huggingface.co/together',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='oV1mrRW-28Eivz-9c4b14db295620a5',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


@pytest.mark.vcr()
async def test_max_completion_tokens(allow_model_requests: None, huggingface_api_key: str):
    m = HuggingFaceModel(
        'deepseek-ai/DeepSeek-R1',
        provider=HuggingFaceProvider(provider_name='together', api_key=huggingface_api_key),
    )
    agent = Agent(m, model_settings=ModelSettings(max_tokens=100))

    result = await agent.run('hello')
    assert result.output == IsStr()
    assert result.usage().output_tokens is not None
    assert result.usage().output_tokens <= 100


def test_system_property():
    model = HuggingFaceModel('some-model', provider=HuggingFaceProvider(hf_client=Mock(), api_key='x'))
    assert model.system == 'huggingface'


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
                    'timestamp': IsDatetime(),
                },
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Response is invalid',
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
                    'timestamp': IsDatetime(),
                },
                provider_response_id='123',
                finish_reason='stop',
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
    c = completion_message(ChatCompletionOutputMessage(content='response', role='assistant'))
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
        'deepseek-ai/DeepSeek-R1',
        provider=HuggingFaceProvider(provider_name='together', api_key=huggingface_api_key),
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
                usage=RequestUsage(input_tokens=10, output_tokens=995),
                model_name='deepseek-ai/DeepSeek-R1',
                timestamp=IsDatetime(),
                provider_name='huggingface',
                provider_url='https://router.huggingface.co/together',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='oV1mwwj-28Eivz-9c4b154f3b427f82',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    result = await agent.run(
        'Considering the way to cross the street, analogously, how do I cross the river?',
        model=HuggingFaceModel(
            'deepseek-ai/DeepSeek-R1',
            provider=HuggingFaceProvider(provider_name='together', api_key=huggingface_api_key),
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
                usage=RequestUsage(input_tokens=32, output_tokens=1425),
                model_name='deepseek-ai/DeepSeek-R1',
                timestamp=IsDatetime(),
                provider_name='huggingface',
                provider_url='https://router.huggingface.co/together',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='oV1n6B7-zqrih-9c4b15fafffad6d3',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


@pytest.mark.vcr()
async def test_hf_model_thinking_part_iter(allow_model_requests: None, huggingface_api_key: str):
    m = HuggingFaceModel(
        'deepseek-ai/DeepSeek-R1',
        provider=HuggingFaceProvider(provider_name='together', api_key=huggingface_api_key),
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
                usage=RequestUsage(input_tokens=10, output_tokens=955),
                model_name='deepseek-ai/DeepSeek-R1',
                timestamp=IsDatetime(),
                provider_name='huggingface',
                provider_url='https://router.huggingface.co/together',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='oV1nHvx-28Eivz-9c4b16f37c27e605',
                finish_reason='stop',
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


async def test_map_user_prompt_with_text_content():
    """Test that UserPromptPart with text content is mapped correctly."""
    msg = await HuggingFaceModel._map_user_prompt(  # pyright: ignore[reportPrivateUsage]
        UserPromptPart(content=['hello', TextContent(content='there', metadata={'id': 'h01'})])
    )

    assert msg.content[0].text == snapshot('hello')  # pyright: ignore
    assert msg.content[1].text == snapshot('there')  # pyright: ignore
