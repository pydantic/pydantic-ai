from __future__ import annotations as _annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from typing import Any, cast

import httpx
import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel
from typing_extensions import TypedDict

from pydantic_ai import (
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.agent import Agent
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, ModelRetry
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.usage import RequestUsage

from ..conftest import IsDatetime, IsNow, IsStr, raise_if_exception, try_import
from .mock_async_stream import MockAsyncStream

with try_import() as imports_successful:
    from mistralai import (
        AssistantMessage as MistralAssistantMessage,
        ChatCompletionChoice as MistralChatCompletionChoice,
        CompletionChunk as MistralCompletionChunk,
        CompletionResponseStreamChoice as MistralCompletionResponseStreamChoice,
        CompletionResponseStreamChoiceFinishReason as MistralCompletionResponseStreamChoiceFinishReason,
        DeltaMessage as MistralDeltaMessage,
        FunctionCall as MistralFunctionCall,
        Mistral,
        TextChunk as MistralTextChunk,
        UsageInfo as MistralUsageInfo,
    )
    from mistralai.models import (
        ChatCompletionResponse as MistralChatCompletionResponse,
        CompletionEvent as MistralCompletionEvent,
        SDKError,
        ToolCall as MistralToolCall,
    )
    from mistralai.types.basemodel import Unset as MistralUnset

    from pydantic_ai.models.mistral import MistralModel, MistralStreamedResponse
    from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
    from pydantic_ai.providers.mistral import MistralProvider
    from pydantic_ai.providers.openai import OpenAIProvider

    MockChatCompletion = MistralChatCompletionResponse | Exception
    MockCompletionEvent = MistralCompletionEvent | Exception

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='mistral or openai not installed'),
    pytest.mark.anyio,
]


@dataclass
class MockSdkConfiguration:
    def get_server_details(self) -> tuple[str, ...]:
        return ('https://api.mistral.ai',)


@dataclass
class MockMistralAI:
    completions: MockChatCompletion | Sequence[MockChatCompletion] | None = None
    stream: Sequence[MockCompletionEvent] | Sequence[Sequence[MockCompletionEvent]] | None = None
    index: int = 0

    @cached_property
    def sdk_configuration(self) -> MockSdkConfiguration:
        return MockSdkConfiguration()

    @cached_property
    def chat(self) -> Any:
        if self.stream:
            return type(
                'Chat',
                (),
                {'stream_async': self.chat_completions_create, 'complete_async': self.chat_completions_create},
            )
        else:
            return type('Chat', (), {'complete_async': self.chat_completions_create})

    @classmethod
    def create_mock(cls, completions: MockChatCompletion | Sequence[MockChatCompletion]) -> Mistral:
        return cast(Mistral, cls(completions=completions))

    @classmethod
    def create_stream_mock(
        cls, completions_streams: Sequence[MockCompletionEvent] | Sequence[Sequence[MockCompletionEvent]]
    ) -> Mistral:
        return cast(Mistral, cls(stream=completions_streams))

    async def chat_completions_create(  # pragma: lax no cover
        self, *_args: Any, stream: bool = False, **_kwargs: Any
    ) -> MistralChatCompletionResponse | MockAsyncStream[MockCompletionEvent]:
        if stream or self.stream:
            assert self.stream is not None, 'you can only use `stream=True` if `stream` is provided'
            if isinstance(self.stream[0], list):
                response = MockAsyncStream(iter(cast(list[MockCompletionEvent], self.stream[self.index])))
            else:
                response = MockAsyncStream(iter(cast(list[MockCompletionEvent], self.stream)))
        else:
            assert self.completions is not None, 'you can only use `stream=False` if `completions` are provided'
            if isinstance(self.completions, Sequence):
                raise_if_exception(self.completions[self.index])
                response = cast(MistralChatCompletionResponse, self.completions[self.index])
            else:
                raise_if_exception(self.completions)
                response = cast(MistralChatCompletionResponse, self.completions)
        self.index += 1
        return response


def completion_message(
    message: MistralAssistantMessage, *, usage: MistralUsageInfo | None = None, with_created: bool = True
) -> MistralChatCompletionResponse:
    return MistralChatCompletionResponse(
        id='123',
        choices=[MistralChatCompletionChoice(finish_reason='stop', index=0, message=message)],
        created=1704067200 if with_created else 0,  # 2024-01-01
        model='mistral-large-123',
        object='chat.completion',
        usage=usage or MistralUsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=1),
    )


def chunk(
    delta: list[MistralDeltaMessage],
    finish_reason: MistralCompletionResponseStreamChoiceFinishReason | None = None,
    with_created: bool = True,
) -> MistralCompletionEvent:
    return MistralCompletionEvent(
        data=MistralCompletionChunk(
            id='x',
            choices=[
                MistralCompletionResponseStreamChoice(index=index, delta=delta, finish_reason=finish_reason)
                for index, delta in enumerate(delta)
            ],
            created=1704067200 if with_created else 0,  # 2024-01-01
            model='gpt-4',
            object='chat.completion.chunk',
            usage=MistralUsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=1),
        )
    )


def text_chunk(
    text: str, finish_reason: MistralCompletionResponseStreamChoiceFinishReason | None = None
) -> MistralCompletionEvent:
    return chunk([MistralDeltaMessage(content=text, role='assistant')], finish_reason=finish_reason)


def text_chunkk(
    text: str, finish_reason: MistralCompletionResponseStreamChoiceFinishReason | None = None
) -> MistralCompletionEvent:
    return chunk(
        [MistralDeltaMessage(content=[MistralTextChunk(text=text)], role='assistant')], finish_reason=finish_reason
    )


def func_chunk(
    tool_calls: list[MistralToolCall], finish_reason: MistralCompletionResponseStreamChoiceFinishReason | None = None
) -> MistralCompletionEvent:
    return chunk([MistralDeltaMessage(tool_calls=tool_calls, role='assistant')], finish_reason=finish_reason)


#####################
## Init
#####################


def test_init():
    m = MistralModel('mistral-large-latest', provider=MistralProvider(api_key='foobar'))
    assert m.model_name == 'mistral-large-latest'
    assert m.base_url == 'https://api.mistral.ai'


#####################
## Completion
#####################


async def test_multiple_completions(allow_model_requests: None):
    completions = [
        # First completion: created is "now" (simulate IsNow)
        completion_message(
            MistralAssistantMessage(content='world'),
            usage=MistralUsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=1),
            with_created=False,
        ),
        # Second completion: created is fixed 2024-01-01 00:00:00 UTC
        completion_message(MistralAssistantMessage(content='hello again')),
    ]
    mock_client = MockMistralAI.create_mock(completions)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model=model)

    result = await agent.run('hello')

    assert result.output == 'world'
    assert result.usage().input_tokens == 1
    assert result.usage().output_tokens == 1

    result = await agent.run('hello again', message_history=result.new_messages())
    assert result.output == 'hello again'
    assert result.usage().input_tokens == 1
    assert result.usage().output_tokens == 1
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='world')],
                usage=RequestUsage(input_tokens=1, output_tokens=1),
                model_name='mistral-large-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={'finish_reason': 'stop'},
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='hello again', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='hello again')],
                usage=RequestUsage(input_tokens=1, output_tokens=1),
                model_name='mistral-large-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
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


async def test_three_completions(allow_model_requests: None):
    completions = [
        completion_message(
            MistralAssistantMessage(content='world'),
            usage=MistralUsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=1),
        ),
        completion_message(MistralAssistantMessage(content='hello again')),
        completion_message(MistralAssistantMessage(content='final message')),
    ]
    mock_client = MockMistralAI.create_mock(completions)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model=model)

    result = await agent.run('hello')

    assert result.output == 'world'
    assert result.usage().input_tokens == 1
    assert result.usage().output_tokens == 1

    result = await agent.run('hello again', message_history=result.all_messages())
    assert result.output == 'hello again'
    assert result.usage().input_tokens == 1
    assert result.usage().output_tokens == 1

    result = await agent.run('final message', message_history=result.all_messages())
    assert result.output == 'final message'
    assert result.usage().input_tokens == 1
    assert result.usage().output_tokens == 1
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='world')],
                usage=RequestUsage(input_tokens=1, output_tokens=1),
                model_name='mistral-large-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='hello again', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='hello again')],
                usage=RequestUsage(input_tokens=1, output_tokens=1),
                model_name='mistral-large-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='final message', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='final message')],
                usage=RequestUsage(input_tokens=1, output_tokens=1),
                model_name='mistral-large-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
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


#####################
## Completion Stream
#####################


async def test_stream_text(allow_model_requests: None):
    stream = [
        text_chunk('hello '),
        text_chunk('world '),
        text_chunk('welcome '),
        text_chunkk('mistral'),
        chunk([]),
    ]
    mock_client = MockMistralAI.create_stream_mock(stream)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model=model)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(
            ['hello ', 'hello world ', 'hello world welcome ', 'hello world welcome mistral']
        )
        assert result.is_complete
        assert result.usage().input_tokens == 5
        assert result.usage().output_tokens == 5


async def test_stream_text_finish_reason(allow_model_requests: None):
    stream = [
        text_chunk('hello '),
        text_chunkk('world'),
        text_chunk('.', finish_reason='stop'),
    ]
    mock_client = MockMistralAI.create_stream_mock(stream)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model=model)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(
            ['hello ', 'hello world', 'hello world.']
        )
        assert result.is_complete


async def test_no_delta(allow_model_requests: None):
    stream = [
        chunk([], with_created=False),
        text_chunk('hello '),
        text_chunk('world'),
    ]
    mock_client = MockMistralAI.create_stream_mock(stream)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model=model)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(['hello ', 'hello world'])
        assert result.is_complete
        assert result.usage().input_tokens == 3
        assert result.usage().output_tokens == 3


#####################
## Completion Model Structured
#####################


async def test_request_native_with_arguments_dict_response(allow_model_requests: None):
    class CityLocation(BaseModel):
        city: str
        country: str

    completion = completion_message(
        MistralAssistantMessage(
            content=None,
            role='assistant',
            tool_calls=[
                MistralToolCall(
                    id='123',
                    function=MistralFunctionCall(arguments={'city': 'paris', 'country': 'france'}, name='final_result'),
                    type='function',
                )
            ],
        ),
        usage=MistralUsageInfo(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )
    mock_client = MockMistralAI.create_mock(completion)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model=model, output_type=CityLocation)

    result = await agent.run('User prompt value')

    assert result.output == CityLocation(city='paris', country='france')
    assert result.usage().input_tokens == 1
    assert result.usage().output_tokens == 2
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='User prompt value', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args={'city': 'paris', 'country': 'france'},
                        tool_call_id='123',
                    )
                ],
                usage=RequestUsage(input_tokens=1, output_tokens=2),
                model_name='mistral-large-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
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
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


async def test_request_native_with_arguments_str_response(allow_model_requests: None):
    class CityLocation(BaseModel):
        city: str
        country: str

    completion = completion_message(
        MistralAssistantMessage(
            content=None,
            role='assistant',
            tool_calls=[
                MistralToolCall(
                    id='123',
                    function=MistralFunctionCall(
                        arguments='{"city": "paris", "country": "france"}', name='final_result'
                    ),
                    type='function',
                )
            ],
        )
    )
    mock_client = MockMistralAI.create_mock(completion)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model=model, output_type=CityLocation)

    result = await agent.run('User prompt value')

    assert result.output == CityLocation(city='paris', country='france')
    assert result.usage().input_tokens == 1
    assert result.usage().output_tokens == 1
    assert result.usage().details == {}
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='User prompt value', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"city": "paris", "country": "france"}',
                        tool_call_id='123',
                    )
                ],
                usage=RequestUsage(input_tokens=1, output_tokens=1),
                model_name='mistral-large-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
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
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


async def test_request_output_type_with_arguments_str_response(allow_model_requests: None):
    completion = completion_message(
        MistralAssistantMessage(
            content=None,
            role='assistant',
            tool_calls=[
                MistralToolCall(
                    id='123',
                    function=MistralFunctionCall(arguments='{"response": 42}', name='final_result'),
                    type='function',
                )
            ],
        )
    )
    mock_client = MockMistralAI.create_mock(completion)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model=model, output_type=int, system_prompt='System prompt value')

    result = await agent.run('User prompt value')

    assert result.output == 42
    assert result.usage().input_tokens == 1
    assert result.usage().output_tokens == 1
    assert result.usage().details == {}
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='System prompt value', timestamp=IsNow(tz=timezone.utc)),
                    UserPromptPart(content='User prompt value', timestamp=IsNow(tz=timezone.utc)),
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"response": 42}',
                        tool_call_id='123',
                    )
                ],
                usage=RequestUsage(input_tokens=1, output_tokens=1),
                model_name='mistral-large-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
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
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


#####################
## Completion Model Structured Stream (JSON Mode)
#####################


async def test_stream_structured_with_all_type(allow_model_requests: None):
    class MyTypedDict(TypedDict, total=False):
        first: str
        second: int
        bool_value: bool
        nullable_value: int | None
        array_value: list[str]
        dict_value: dict[str, Any]
        dict_int_value: dict[str, int]
        dict_str_value: dict[int, str]

    stream = [
        text_chunk('{'),
        text_chunk('"first": "One'),
        text_chunk(
            '", "second": 2',
        ),
        text_chunk(
            ', "bool_value": true',
        ),
        text_chunk(
            ', "nullable_value": null',
        ),
        text_chunk(
            ', "array_value": ["A", "B", "C"]',
        ),
        text_chunk(
            ', "dict_value": {"A": "A", "B":"B"}',
        ),
        text_chunk(
            ', "dict_int_value": {"A": 1, "B":2}',
        ),
        text_chunk('}'),
        chunk([]),
    ]

    mock_client = MockMistralAI.create_stream_mock(stream)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model, output_type=MyTypedDict)

    async with agent.run_stream('User prompt value') as result:
        assert not result.is_complete
        v = [dict(c) async for c in result.stream_output(debounce_by=None)]
        assert v == snapshot(
            [
                {'first': 'One'},
                {'first': 'One', 'second': 2},
                {'first': 'One', 'second': 2, 'bool_value': True},
                {'first': 'One', 'second': 2, 'bool_value': True, 'nullable_value': None},
                {
                    'first': 'One',
                    'second': 2,
                    'bool_value': True,
                    'nullable_value': None,
                    'array_value': ['A', 'B', 'C'],
                },
                {
                    'first': 'One',
                    'second': 2,
                    'bool_value': True,
                    'nullable_value': None,
                    'array_value': ['A', 'B', 'C'],
                    'dict_value': {'A': 'A', 'B': 'B'},
                },
                {
                    'first': 'One',
                    'second': 2,
                    'bool_value': True,
                    'nullable_value': None,
                    'array_value': ['A', 'B', 'C'],
                    'dict_value': {'A': 'A', 'B': 'B'},
                    'dict_int_value': {'A': 1, 'B': 2},
                },
                {
                    'first': 'One',
                    'second': 2,
                    'bool_value': True,
                    'nullable_value': None,
                    'array_value': ['A', 'B', 'C'],
                    'dict_value': {'A': 'A', 'B': 'B'},
                    'dict_int_value': {'A': 1, 'B': 2},
                },
                {
                    'first': 'One',
                    'second': 2,
                    'bool_value': True,
                    'nullable_value': None,
                    'array_value': ['A', 'B', 'C'],
                    'dict_value': {'A': 'A', 'B': 'B'},
                    'dict_int_value': {'A': 1, 'B': 2},
                },
            ]
        )
        assert result.is_complete
        assert result.usage().input_tokens == 10
        assert result.usage().output_tokens == 10

        # double check usage matches stream count
        assert result.usage().output_tokens == len(stream)


async def test_stream_result_type_primitif_dict(allow_model_requests: None):
    """This test tests the primitif result with the pydantic ai format model response"""

    class MyTypedDict(TypedDict, total=False):
        first: str
        second: str

    stream = [
        text_chunk('{'),
        text_chunk('"'),
        text_chunk('f'),
        text_chunk('i'),
        text_chunk('r'),
        text_chunk('s'),
        text_chunk('t'),
        text_chunk('"'),
        text_chunk(':'),
        text_chunk(' '),
        text_chunk('"'),
        text_chunk('O'),
        text_chunk('n'),
        text_chunk('e'),
        text_chunk('"'),
        text_chunk(','),
        text_chunk(' '),
        text_chunk('"'),
        text_chunk('s'),
        text_chunk('e'),
        text_chunk('c'),
        text_chunk('o'),
        text_chunk('n'),
        text_chunk('d'),
        text_chunk('"'),
        text_chunk(':'),
        text_chunk(' '),
        text_chunk('"'),
        text_chunk('T'),
        text_chunk('w'),
        text_chunk('o'),
        text_chunk('"'),
        text_chunk('}'),
        chunk([]),
    ]

    mock_client = MockMistralAI.create_stream_mock(stream)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model=model, output_type=MyTypedDict)

    async with agent.run_stream('User prompt value') as result:
        assert not result.is_complete
        v = [c async for c in result.stream_output(debounce_by=None)]
        assert v == snapshot(
            [
                {'first': 'O'},
                {'first': 'On'},
                {'first': 'One'},
                {'first': 'One'},
                {'first': 'One'},
                {'first': 'One'},
                {'first': 'One'},
                {'first': 'One'},
                {'first': 'One'},
                {'first': 'One'},
                {'first': 'One'},
                {'first': 'One'},
                {'first': 'One'},
                {'first': 'One'},
                {'first': 'One'},
                {'first': 'One'},
                {'first': 'One', 'second': ''},
                {'first': 'One', 'second': 'T'},
                {'first': 'One', 'second': 'Tw'},
                {'first': 'One', 'second': 'Two'},
                {'first': 'One', 'second': 'Two'},
                {'first': 'One', 'second': 'Two'},
                {'first': 'One', 'second': 'Two'},
            ]
        )
        assert result.is_complete
        assert result.usage().input_tokens == 34
        assert result.usage().output_tokens == 34

        # double check usage matches stream count
        assert result.usage().output_tokens == len(stream)


async def test_stream_result_type_primitif_int(allow_model_requests: None):
    """This test tests the primitif result with the pydantic ai format model response"""

    stream = [
        # {'response':
        text_chunk('{'),
        text_chunk('"resp'),
        text_chunk('onse":'),
        text_chunk('1'),
        text_chunk('}'),
        chunk([]),
    ]

    mock_client = MockMistralAI.create_stream_mock(stream)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model=model, output_type=int)

    async with agent.run_stream('User prompt value') as result:
        assert not result.is_complete
        v = [c async for c in result.stream_output(debounce_by=None)]
        assert v == snapshot([1, 1, 1])
        assert result.is_complete
        assert result.usage().input_tokens == 6
        assert result.usage().output_tokens == 6

        # double check usage matches stream count
        assert result.usage().output_tokens == len(stream)


async def test_stream_result_type_primitif_array(allow_model_requests: None):
    """This test tests the primitif result with the pydantic ai format model response"""

    stream = [
        # {'response':
        text_chunk('{'),
        text_chunk('"resp'),
        text_chunk('onse":'),
        text_chunk('['),
        text_chunk('"'),
        text_chunk('f'),
        text_chunk('i'),
        text_chunk('r'),
        text_chunk('s'),
        text_chunk('t'),
        text_chunk('"'),
        text_chunk(','),
        text_chunk('"'),
        text_chunk('O'),
        text_chunk('n'),
        text_chunk('e'),
        text_chunk('"'),
        text_chunk(','),
        text_chunk('"'),
        text_chunk('s'),
        text_chunk('e'),
        text_chunk('c'),
        text_chunk('o'),
        text_chunk('n'),
        text_chunk('d'),
        text_chunk('"'),
        text_chunk(','),
        text_chunk('"'),
        text_chunk('T'),
        text_chunk('w'),
        text_chunk('o'),
        text_chunk('"'),
        text_chunk(']'),
        text_chunk('}'),
        chunk([]),
    ]

    mock_client = MockMistralAI.create_stream_mock(stream)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model, output_type=list[str])

    async with agent.run_stream('User prompt value') as result:
        assert not result.is_complete
        v = [c async for c in result.stream_output(debounce_by=None)]
        assert v == snapshot(
            [
                [''],
                ['f'],
                ['fi'],
                ['fir'],
                ['firs'],
                ['first'],
                ['first'],
                ['first'],
                ['first', ''],
                ['first', 'O'],
                ['first', 'On'],
                ['first', 'One'],
                ['first', 'One'],
                ['first', 'One'],
                ['first', 'One', ''],
                ['first', 'One', 's'],
                ['first', 'One', 'se'],
                ['first', 'One', 'sec'],
                ['first', 'One', 'seco'],
                ['first', 'One', 'secon'],
                ['first', 'One', 'second'],
                ['first', 'One', 'second'],
                ['first', 'One', 'second'],
                ['first', 'One', 'second', ''],
                ['first', 'One', 'second', 'T'],
                ['first', 'One', 'second', 'Tw'],
                ['first', 'One', 'second', 'Two'],
                ['first', 'One', 'second', 'Two'],
                ['first', 'One', 'second', 'Two'],
                ['first', 'One', 'second', 'Two'],
                ['first', 'One', 'second', 'Two'],
            ]
        )
        assert result.is_complete
        assert result.usage().input_tokens == 35
        assert result.usage().output_tokens == 35

        # double check usage matches stream count
        assert result.usage().output_tokens == len(stream)


async def test_stream_result_type_basemodel_with_default_params(allow_model_requests: None):
    class MyTypedBaseModel(BaseModel):
        first: str = ''  # Note: Default, set value.
        second: str = ''  # Note: Default, set value.

    stream = [
        text_chunk('{'),
        text_chunk('"'),
        text_chunk('f'),
        text_chunk('i'),
        text_chunk('r'),
        text_chunk('s'),
        text_chunk('t'),
        text_chunk('"'),
        text_chunk(':'),
        text_chunk(' '),
        text_chunk('"'),
        text_chunk('O'),
        text_chunk('n'),
        text_chunk('e'),
        text_chunk('"'),
        text_chunk(','),
        text_chunk(' '),
        text_chunk('"'),
        text_chunk('s'),
        text_chunk('e'),
        text_chunk('c'),
        text_chunk('o'),
        text_chunk('n'),
        text_chunk('d'),
        text_chunk('"'),
        text_chunk(':'),
        text_chunk(' '),
        text_chunk('"'),
        text_chunk('T'),
        text_chunk('w'),
        text_chunk('o'),
        text_chunk('"'),
        text_chunk('}'),
        chunk([]),
    ]

    mock_client = MockMistralAI.create_stream_mock(stream)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model=model, output_type=MyTypedBaseModel)

    async with agent.run_stream('User prompt value') as result:
        assert not result.is_complete
        v = [c async for c in result.stream_output(debounce_by=None)]
        assert v == snapshot(
            [
                MyTypedBaseModel(first='O', second=''),
                MyTypedBaseModel(first='On', second=''),
                MyTypedBaseModel(first='One', second=''),
                MyTypedBaseModel(first='One', second=''),
                MyTypedBaseModel(first='One', second=''),
                MyTypedBaseModel(first='One', second=''),
                MyTypedBaseModel(first='One', second=''),
                MyTypedBaseModel(first='One', second=''),
                MyTypedBaseModel(first='One', second=''),
                MyTypedBaseModel(first='One', second=''),
                MyTypedBaseModel(first='One', second=''),
                MyTypedBaseModel(first='One', second=''),
                MyTypedBaseModel(first='One', second=''),
                MyTypedBaseModel(first='One', second=''),
                MyTypedBaseModel(first='One', second=''),
                MyTypedBaseModel(first='One', second=''),
                MyTypedBaseModel(first='One', second=''),
                MyTypedBaseModel(first='One', second='T'),
                MyTypedBaseModel(first='One', second='Tw'),
                MyTypedBaseModel(first='One', second='Two'),
                MyTypedBaseModel(first='One', second='Two'),
                MyTypedBaseModel(first='One', second='Two'),
                MyTypedBaseModel(first='One', second='Two'),
            ]
        )
        assert result.is_complete
        assert result.usage().input_tokens == 34
        assert result.usage().output_tokens == 34

        # double check usage matches stream count
        assert result.usage().output_tokens == len(stream)


async def test_stream_result_type_basemodel_with_required_params(allow_model_requests: None):
    class MyTypedBaseModel(BaseModel):
        first: str  # Note: Required params
        second: str  # Note: Required params

    stream = [
        text_chunk('{'),
        text_chunk('"'),
        text_chunk('f'),
        text_chunk('i'),
        text_chunk('r'),
        text_chunk('s'),
        text_chunk('t'),
        text_chunk('"'),
        text_chunk(':'),
        text_chunk(' '),
        text_chunk('"'),
        text_chunk('O'),
        text_chunk('n'),
        text_chunk('e'),
        text_chunk('"'),
        text_chunk(','),
        text_chunk(' '),
        text_chunk('"'),
        text_chunk('s'),
        text_chunk('e'),
        text_chunk('c'),
        text_chunk('o'),
        text_chunk('n'),
        text_chunk('d'),
        text_chunk('"'),
        text_chunk(':'),
        text_chunk(' '),
        text_chunk('"'),
        text_chunk('T'),
        text_chunk('w'),
        text_chunk('o'),
        text_chunk('"'),
        text_chunk('}'),
        chunk([]),
    ]

    mock_client = MockMistralAI.create_stream_mock(stream)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model=model, output_type=MyTypedBaseModel)

    async with agent.run_stream('User prompt value') as result:
        assert not result.is_complete
        v = [c async for c in result.stream_output(debounce_by=None)]
        assert v == snapshot(
            [
                MyTypedBaseModel(first='One', second=''),
                MyTypedBaseModel(first='One', second='T'),
                MyTypedBaseModel(first='One', second='Tw'),
                MyTypedBaseModel(first='One', second='Two'),
                MyTypedBaseModel(first='One', second='Two'),
                MyTypedBaseModel(first='One', second='Two'),
                MyTypedBaseModel(first='One', second='Two'),
            ]
        )
        assert result.is_complete
        assert result.usage().input_tokens == 34
        assert result.usage().output_tokens == 34

        # double check cost matches stream count
        assert result.usage().output_tokens == len(stream)


#####################
## Completion Function call
#####################


async def test_request_tool_call(allow_model_requests: None):
    completion = [
        completion_message(
            MistralAssistantMessage(
                content=None,
                role='assistant',
                tool_calls=[
                    MistralToolCall(
                        id='1',
                        function=MistralFunctionCall(arguments='{"loc_name": "San Fransisco"}', name='get_location'),
                        type='function',
                    )
                ],
            ),
            usage=MistralUsageInfo(
                completion_tokens=1,
                prompt_tokens=2,
                total_tokens=3,
            ),
        ),
        completion_message(
            MistralAssistantMessage(
                content=None,
                role='assistant',
                tool_calls=[
                    MistralToolCall(
                        id='2',
                        function=MistralFunctionCall(arguments='{"loc_name": "London"}', name='get_location'),
                        type='function',
                    )
                ],
            ),
            usage=MistralUsageInfo(
                completion_tokens=2,
                prompt_tokens=3,
                total_tokens=6,
            ),
        ),
        completion_message(MistralAssistantMessage(content='final response', role='assistant')),
    ]
    mock_client = MockMistralAI.create_mock(completion)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model, system_prompt='this is the system prompt')

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        if loc_name == 'London':
            return json.dumps({'lat': 51, 'lng': 0})
        else:
            raise ModelRetry('Wrong location, please try again')

    result = await agent.run('Hello')

    assert result.output == 'final response'
    assert result.usage().input_tokens == 6
    assert result.usage().output_tokens == 4
    assert result.usage().total_tokens == 10
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='this is the system prompt', timestamp=IsNow(tz=timezone.utc)),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
                ],
                timestamp=IsNow(tz=timezone.utc),
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
                model_name='mistral-large-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
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
                timestamp=IsNow(tz=timezone.utc),
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
                model_name='mistral-large-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
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
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='final response')],
                usage=RequestUsage(input_tokens=1, output_tokens=1),
                model_name='mistral-large-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
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


async def test_request_tool_call_with_result_type(allow_model_requests: None):
    class MyTypedDict(TypedDict, total=False):
        lat: int
        lng: int

    completion = [
        completion_message(
            MistralAssistantMessage(
                content=None,
                role='assistant',
                tool_calls=[
                    MistralToolCall(
                        id='1',
                        function=MistralFunctionCall(arguments='{"loc_name": "San Fransisco"}', name='get_location'),
                        type='function',
                    )
                ],
            ),
            usage=MistralUsageInfo(
                completion_tokens=1,
                prompt_tokens=2,
                total_tokens=3,
            ),
        ),
        completion_message(
            MistralAssistantMessage(
                content=None,
                role='assistant',
                tool_calls=[
                    MistralToolCall(
                        id='2',
                        function=MistralFunctionCall(arguments='{"loc_name": "London"}', name='get_location'),
                        type='function',
                    )
                ],
            ),
            usage=MistralUsageInfo(
                completion_tokens=2,
                prompt_tokens=3,
                total_tokens=6,
            ),
        ),
        completion_message(
            MistralAssistantMessage(
                content=None,
                role='assistant',
                tool_calls=[
                    MistralToolCall(
                        id='1',
                        function=MistralFunctionCall(arguments='{"lat": 51, "lng": 0}', name='final_result'),
                        type='function',
                    )
                ],
            ),
            usage=MistralUsageInfo(
                completion_tokens=1,
                prompt_tokens=2,
                total_tokens=3,
            ),
        ),
    ]
    mock_client = MockMistralAI.create_mock(completion)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model, system_prompt='this is the system prompt', output_type=MyTypedDict)

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        if loc_name == 'London':
            return json.dumps({'lat': 51, 'lng': 0})
        else:
            raise ModelRetry('Wrong location, please try again')

    result = await agent.run('Hello')

    assert result.output == {'lat': 51, 'lng': 0}
    assert result.usage().input_tokens == 7
    assert result.usage().output_tokens == 4
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='this is the system prompt', timestamp=IsNow(tz=timezone.utc)),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
                ],
                timestamp=IsNow(tz=timezone.utc),
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
                model_name='mistral-large-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
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
                timestamp=IsNow(tz=timezone.utc),
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
                model_name='mistral-large-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
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
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"lat": 51, "lng": 0}',
                        tool_call_id='1',
                    )
                ],
                usage=RequestUsage(input_tokens=2, output_tokens=1),
                model_name='mistral-large-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
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
                        tool_call_id='1',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


#####################
## Completion Function call Stream
#####################


async def test_stream_tool_call_with_return_type(allow_model_requests: None):
    class MyTypedDict(TypedDict, total=False):
        won: bool

    completion = [
        [
            chunk(
                delta=[MistralDeltaMessage(role=MistralUnset(), content='', tool_calls=MistralUnset())],
                finish_reason='tool_calls',
            ),
            func_chunk(
                tool_calls=[
                    MistralToolCall(
                        id='1',
                        function=MistralFunctionCall(arguments='{"loc_name": "San Fransisco"}', name='get_location'),
                        type='function',
                    )
                ],
                finish_reason='tool_calls',
            ),
        ],
        [
            chunk(
                delta=[MistralDeltaMessage(role=MistralUnset(), content='', tool_calls=MistralUnset())],
                finish_reason='tool_calls',
            ),
            func_chunk(
                tool_calls=[
                    MistralToolCall(
                        id='1',
                        function=MistralFunctionCall(arguments='{"won": true}', name='final_result'),
                        type=None,
                    )
                ],
                finish_reason='tool_calls',
            ),
        ],
    ]

    mock_client = MockMistralAI.create_stream_mock(completion)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model, system_prompt='this is the system prompt', output_type=MyTypedDict)

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        return json.dumps({'lat': 51, 'lng': 0})

    async with agent.run_stream('User prompt value') as result:
        assert not result.is_complete
        v = [c async for c in result.stream_output(debounce_by=None)]
        assert v == snapshot([{'won': True}, {'won': True}])
        assert result.is_complete
        assert result.timestamp() == IsNow(tz=timezone.utc)
        assert result.usage().input_tokens == 4
        assert result.usage().output_tokens == 4

        # double check usage matches stream count
        assert result.usage().output_tokens == 4

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='this is the system prompt', timestamp=IsNow(tz=timezone.utc)),
                    UserPromptPart(content='User prompt value', timestamp=IsNow(tz=timezone.utc)),
                ],
                timestamp=IsNow(tz=timezone.utc),
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
                usage=RequestUsage(input_tokens=2, output_tokens=2),
                model_name='gpt-4',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'tool_calls',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='x',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_location',
                        content='{"lat": 51, "lng": 0}',
                        tool_call_id='1',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='final_result', args='{"won": true}', tool_call_id='1')],
                usage=RequestUsage(input_tokens=2, output_tokens=2),
                model_name='gpt-4',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'tool_calls',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='x',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='1',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )

    assert await result.get_output() == {'won': True}


async def test_stream_tool_call(allow_model_requests: None):
    completion = [
        [
            chunk(
                delta=[MistralDeltaMessage(role=MistralUnset(), content='', tool_calls=MistralUnset())],
                finish_reason='tool_calls',
            ),
            func_chunk(
                tool_calls=[
                    MistralToolCall(
                        id='1',
                        function=MistralFunctionCall(arguments='{"loc_name": "San Fransisco"}', name='get_location'),
                        type='function',
                    )
                ],
                finish_reason='tool_calls',
            ),
        ],
        [
            chunk(delta=[MistralDeltaMessage(role='assistant', content='', tool_calls=MistralUnset())]),
            chunk(delta=[MistralDeltaMessage(role=MistralUnset(), content='final ', tool_calls=MistralUnset())]),
            chunk(delta=[MistralDeltaMessage(role=MistralUnset(), content='response', tool_calls=MistralUnset())]),
            chunk(
                delta=[MistralDeltaMessage(role=MistralUnset(), content='', tool_calls=MistralUnset())],
                finish_reason='stop',
            ),
        ],
    ]

    mock_client = MockMistralAI.create_stream_mock(completion)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model, system_prompt='this is the system prompt')

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        return json.dumps({'lat': 51, 'lng': 0})

    async with agent.run_stream('User prompt value') as result:
        assert not result.is_complete
        v = [c async for c in result.stream_output(debounce_by=None)]
        assert v == snapshot(['final ', 'final response', 'final response'])
        assert result.is_complete
        assert result.timestamp() == IsNow(tz=timezone.utc)
        assert result.usage().input_tokens == 6
        assert result.usage().output_tokens == 6

        # double check usage matches stream count
        assert result.usage().output_tokens == 6

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='this is the system prompt', timestamp=IsNow(tz=timezone.utc)),
                    UserPromptPart(content='User prompt value', timestamp=IsNow(tz=timezone.utc)),
                ],
                timestamp=IsNow(tz=timezone.utc),
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
                usage=RequestUsage(input_tokens=2, output_tokens=2),
                model_name='gpt-4',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'tool_calls',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='x',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_location',
                        content='{"lat": 51, "lng": 0}',
                        tool_call_id='1',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='final response')],
                usage=RequestUsage(input_tokens=4, output_tokens=4),
                model_name='gpt-4',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='x',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_stream_tool_call_with_retry(allow_model_requests: None):
    completion = [
        [
            chunk(
                delta=[MistralDeltaMessage(role=MistralUnset(), content='', tool_calls=MistralUnset())],
                finish_reason='tool_calls',
            ),
            func_chunk(
                tool_calls=[
                    MistralToolCall(
                        id='1',
                        function=MistralFunctionCall(arguments='{"loc_name": "San Fransisco"}', name='get_location'),
                        type='function',
                    )
                ],
                finish_reason='tool_calls',
            ),
        ],
        [
            func_chunk(
                tool_calls=[
                    MistralToolCall(
                        id='2',
                        function=MistralFunctionCall(arguments='{"loc_name": "London"}', name='get_location'),
                        type='function',
                    )
                ],
                finish_reason='tool_calls',
            ),
        ],
        [
            chunk(delta=[MistralDeltaMessage(role='assistant', content='', tool_calls=MistralUnset())]),
            chunk(delta=[MistralDeltaMessage(role=MistralUnset(), content='final ', tool_calls=MistralUnset())]),
            chunk(delta=[MistralDeltaMessage(role=MistralUnset(), content='response', tool_calls=MistralUnset())]),
            chunk(
                delta=[MistralDeltaMessage(role=MistralUnset(), content='', tool_calls=MistralUnset())],
                finish_reason='stop',
            ),
        ],
    ]

    mock_client = MockMistralAI.create_stream_mock(completion)
    model = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(model, system_prompt='this is the system prompt')

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        if loc_name == 'London':
            return json.dumps({'lat': 51, 'lng': 0})
        else:
            raise ModelRetry('Wrong location, please try again')

    async with agent.run_stream('User prompt value') as result:
        assert not result.is_complete
        v = [c async for c in result.stream_text(debounce_by=None)]
        assert v == snapshot(['final ', 'final response'])
        assert result.is_complete
        assert result.timestamp() == IsNow(tz=timezone.utc)
        assert result.usage().input_tokens == 7
        assert result.usage().output_tokens == 7

        # double check usage matches stream count
        assert result.usage().output_tokens == 7

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='this is the system prompt', timestamp=IsNow(tz=timezone.utc)),
                    UserPromptPart(content='User prompt value', timestamp=IsNow(tz=timezone.utc)),
                ],
                timestamp=IsNow(tz=timezone.utc),
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
                usage=RequestUsage(input_tokens=2, output_tokens=2),
                model_name='gpt-4',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'tool_calls',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='x',
                finish_reason='tool_call',
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
                timestamp=IsNow(tz=timezone.utc),
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
                usage=RequestUsage(input_tokens=1, output_tokens=1),
                model_name='gpt-4',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'tool_calls',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='x',
                finish_reason='tool_call',
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
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='final response')],
                usage=RequestUsage(input_tokens=4, output_tokens=4),
                model_name='gpt-4',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
                },
                provider_response_id='x',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


#####################
## Test methods
#####################


def test_generate_user_output_format_complex(mistral_api_key: str):
    """
    Single test that includes properties exercising every branch
    in _get_python_type (anyOf, arrays, objects with additionalProperties, etc.).
    """
    schema = {
        'properties': {
            'prop_anyOf': {'anyOf': [{'type': 'string'}, {'type': 'integer'}]},
            'prop_no_type': {
                # no 'type' key
            },
            'prop_simple_string': {'type': 'string'},
            'prop_array_booleans': {'type': 'array', 'items': {'type': 'boolean'}},
            'prop_object_simple': {'type': 'object', 'additionalProperties': {'type': 'boolean'}},
            'prop_object_array': {
                'type': 'object',
                'additionalProperties': {'type': 'array', 'items': {'type': 'integer'}},
            },
            'prop_object_object': {'type': 'object', 'additionalProperties': {'type': 'object'}},
            'prop_object_unknown': {'type': 'object', 'additionalProperties': {'type': 'someUnknownType'}},
            'prop_unrecognized_type': {'type': 'customSomething'},
        }
    }
    m = MistralModel('', json_mode_schema_prompt='{schema}', provider=MistralProvider(api_key=mistral_api_key))
    result = m._generate_user_output_format([schema])  # pyright: ignore[reportPrivateUsage]
    assert result.content == (
        "{'prop_anyOf': 'Optional[str]', "
        "'prop_no_type': 'Any', "
        "'prop_simple_string': 'str', "
        "'prop_array_booleans': 'list[bool]', "
        "'prop_object_simple': 'dict[str, bool]', "
        "'prop_object_array': 'dict[str, list[int]]', "
        "'prop_object_object': 'dict[str, dict[str, Any]]', "
        "'prop_object_unknown': 'dict[str, Any]', "
        "'prop_unrecognized_type': 'Any'}"
    )


def test_generate_user_output_format_multiple(mistral_api_key: str):
    schema = {'properties': {'prop_anyOf': {'anyOf': [{'type': 'string'}, {'type': 'integer'}]}}}
    m = MistralModel('', json_mode_schema_prompt='{schema}', provider=MistralProvider(api_key=mistral_api_key))
    result = m._generate_user_output_format([schema, schema])  # pyright: ignore[reportPrivateUsage]
    assert result.content == "[{'prop_anyOf': 'Optional[str]'}, {'prop_anyOf': 'Optional[str]'}]"


@pytest.mark.parametrize(
    'desc, schema, data, expected',
    [
        (
            'Missing required parameter',
            {
                'required': ['name', 'age'],
                'properties': {
                    'name': {'type': 'string'},
                    'age': {'type': 'integer'},
                },
            },
            {'name': 'Alice'},  # Missing "age"
            False,
        ),
        (
            'Type mismatch (expected string, got int)',
            {'required': ['name'], 'properties': {'name': {'type': 'string'}}},
            {'name': 123},  # Should be a string, got int
            False,
        ),
        (
            'Array parameter check (param not a list)',
            {'required': ['tags'], 'properties': {'tags': {'type': 'array', 'items': {'type': 'string'}}}},
            {'tags': 'not a list'},  # Not a list
            False,
        ),
        (
            'Array item type mismatch',
            {'required': ['tags'], 'properties': {'tags': {'type': 'array', 'items': {'type': 'string'}}}},
            {'tags': ['ok', 123, 'still ok']},  # One item is int, not str
            False,
        ),
        (
            'Nested object fails',
            {
                'required': ['user'],
                'properties': {
                    'user': {
                        'type': 'object',
                        'required': ['id', 'profile'],
                        'properties': {
                            'id': {'type': 'integer'},
                            'profile': {
                                'type': 'object',
                                'required': ['address'],
                                'properties': {'address': {'type': 'string'}},
                            },
                        },
                    }
                },
            },
            {'user': {'id': 101, 'profile': {}}},  # Missing "address" in the nested profile
            False,
        ),
        (
            'All requirements met (success)',
            {
                'required': ['name', 'age', 'tags', 'user'],
                'properties': {
                    'name': {'type': 'string'},
                    'age': {'type': 'integer'},
                    'tags': {'type': 'array', 'items': {'type': 'string'}},
                    'user': {
                        'type': 'object',
                        'required': ['id', 'profile'],
                        'properties': {
                            'id': {'type': 'integer'},
                            'profile': {
                                'type': 'object',
                                'required': ['address'],
                                'properties': {'address': {'type': 'string'}},
                            },
                        },
                    },
                },
            },
            {
                'name': 'Alice',
                'age': 30,
                'tags': ['tag1', 'tag2'],
                'user': {'id': 101, 'profile': {'address': '123 Street'}},
            },
            True,
        ),
    ],
)
def test_validate_required_json_schema(desc: str, schema: dict[str, Any], data: dict[str, Any], expected: bool) -> None:
    result = MistralStreamedResponse._validate_required_json_schema(data, schema)  # pyright: ignore[reportPrivateUsage]
    assert result == expected, f'{desc} — expected {expected}, got {result}'


@pytest.mark.vcr()
async def test_image_as_binary_content_tool_response(
    allow_model_requests: None, mistral_api_key: str, image_content: BinaryContent
):
    m = MistralModel('pixtral-12b-latest', provider=MistralProvider(api_key=mistral_api_key))
    agent = Agent(m)

    @agent.tool_plain
    async def get_image() -> BinaryContent:
        return image_content

    result = await agent.run(['What fruit is in the image you can get from the get_image tool? Call the tool.'])
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=['What fruit is in the image you can get from the get_image tool? Call the tool.'],
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_image', args='{}', tool_call_id='FI5qQGzDE')],
                usage=RequestUsage(input_tokens=65, output_tokens=16),
                model_name='pixtral-12b-latest',
                timestamp=IsDatetime(),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={'finish_reason': 'tool_calls', 'timestamp': IsDatetime()},
                provider_response_id='20c656d7c70e4362858160d9d241ce92',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_image',
                        content='See file 241a70',
                        tool_call_id='FI5qQGzDE',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(content=['This is file 241a70:', image_content], timestamp=IsDatetime()),
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='The image shows a kiwi fruit that has been cut in half. Kiwis are small, oval-shaped fruits with a bright green flesh and tiny black seeds. They have a sweet and tangy flavor and are known for being rich in vitamin C and fiber.'
                    )
                ],
                usage=RequestUsage(input_tokens=1540, output_tokens=54),
                model_name='pixtral-12b-latest',
                timestamp=IsDatetime(),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={'finish_reason': 'stop', 'timestamp': IsDatetime()},
                provider_response_id='b9df7d6167a74543aed6c27557ab0a29',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_image_url_input(allow_model_requests: None):
    c = completion_message(MistralAssistantMessage(content='world', role='assistant'))
    mock_client = MockMistralAI.create_mock(c)
    m = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
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
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='world')],
                usage=RequestUsage(input_tokens=1, output_tokens=1),
                model_name='mistral-large-123',
                timestamp=IsDatetime(),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
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


async def test_image_as_binary_content_input(allow_model_requests: None):
    c = completion_message(MistralAssistantMessage(content='world', role='assistant'))
    mock_client = MockMistralAI.create_mock(c)
    m = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(m)

    # Fake image bytes for testing
    image_bytes = b'fake image data'

    result = await agent.run(['hello', BinaryContent(data=image_bytes, media_type='image/jpeg')])
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'hello',
                            BinaryContent(data=image_bytes, media_type='image/jpeg'),
                        ],
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='world')],
                usage=RequestUsage(input_tokens=1, output_tokens=1),
                model_name='mistral-large-123',
                timestamp=IsDatetime(),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
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


async def test_pdf_url_input(allow_model_requests: None):
    c = completion_message(MistralAssistantMessage(content='world', role='assistant'))
    mock_client = MockMistralAI.create_mock(c)
    m = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(m)

    result = await agent.run(
        [
            'hello',
            DocumentUrl(url='https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf'),
        ]
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'hello',
                            DocumentUrl(
                                url='https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf',
                                identifier='c6720d',
                            ),
                        ],
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='world')],
                usage=RequestUsage(input_tokens=1, output_tokens=1),
                model_name='mistral-large-123',
                timestamp=IsDatetime(),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
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


async def test_pdf_as_binary_content_input(allow_model_requests: None):
    c = completion_message(MistralAssistantMessage(content='world', role='assistant'))
    mock_client = MockMistralAI.create_mock(c)
    m = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(m)

    base64_content = b'%PDF-1.\rtrailer<</Root<</Pages<</Kids[<</MediaBox[0 0 3 3]>>>>>>>>>'

    result = await agent.run(['hello', BinaryContent(data=base64_content, media_type='application/pdf')])
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'hello',
                            BinaryContent(data=base64_content, media_type='application/pdf', identifier='b9d976'),
                        ],
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='world')],
                usage=RequestUsage(input_tokens=1, output_tokens=1),
                model_name='mistral-large-123',
                timestamp=IsDatetime(),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
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


async def test_txt_url_input(allow_model_requests: None):
    c = completion_message(MistralAssistantMessage(content='world', role='assistant'))
    mock_client = MockMistralAI.create_mock(c)
    m = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(m)

    with pytest.raises(RuntimeError, match='DocumentUrl other than PDF is not supported in Mistral.'):
        await agent.run(
            [
                'hello',
                DocumentUrl(url='https://examplefiles.org/files/documents/plaintext-example-file-download.txt'),
            ]
        )


async def test_audio_as_binary_content_input(allow_model_requests: None):
    c = completion_message(MistralAssistantMessage(content='world', role='assistant'))
    mock_client = MockMistralAI.create_mock(c)
    m = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(m)

    base64_content = b'//uQZ'

    with pytest.raises(RuntimeError, match='BinaryContent other than image or PDF is not supported in Mistral.'):
        await agent.run(['hello', BinaryContent(data=base64_content, media_type='audio/wav')])


async def test_video_url_input(allow_model_requests: None):
    c = completion_message(MistralAssistantMessage(content='world', role='assistant'))
    mock_client = MockMistralAI.create_mock(c)
    m = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(m)

    with pytest.raises(RuntimeError, match='VideoUrl is not supported in Mistral.'):
        await agent.run(['hello', VideoUrl(url='https://www.google.com')])


def test_model_status_error(allow_model_requests: None) -> None:
    response = httpx.Response(500, content=b'test error')
    mock_client = MockMistralAI.create_mock(SDKError('test error', raw_response=response))
    m = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(m)
    with pytest.raises(ModelHTTPError) as exc_info:
        agent.run_sync('hello')
    assert str(exc_info.value) == snapshot('status_code: 500, model_name: mistral-large-latest, body: test error')


def test_model_non_http_error(allow_model_requests: None) -> None:
    response = httpx.Response(300, content=b'redirect')
    mock_client = MockMistralAI.create_mock(SDKError('Connection error', raw_response=response))
    m = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(m)
    with pytest.raises(ModelAPIError) as exc_info:
        agent.run_sync('hello')
    assert exc_info.value.model_name == 'mistral-large-latest'


async def test_mistral_model_instructions(allow_model_requests: None, mistral_api_key: str):
    c = completion_message(MistralAssistantMessage(content='world', role='assistant'))
    mock_client = MockMistralAI.create_mock(c)
    m = MistralModel('mistral-large-latest', provider=MistralProvider(mistral_client=mock_client))
    agent = Agent(m, instructions='You are a helpful assistant.')

    result = await agent.run('hello')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                timestamp=IsNow(tz=timezone.utc),
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='world')],
                usage=RequestUsage(input_tokens=1, output_tokens=1),
                model_name='mistral-large-123',
                timestamp=IsDatetime(),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
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


@pytest.mark.vcr()
async def test_mistral_model_thinking_part(allow_model_requests: None, openai_api_key: str, mistral_api_key: str):
    openai_model = OpenAIResponsesModel('o3-mini', provider=OpenAIProvider(api_key=openai_api_key))
    settings = OpenAIResponsesModelSettings(openai_reasoning_effort='high', openai_reasoning_summary='detailed')
    agent = Agent(openai_model, model_settings=settings)

    result = await agent.run('How do I cross the street?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='How do I cross the street?', timestamp=IsDatetime())],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68bb645d50f48196a0c49fd603b87f4503498c8aa840cf12',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    ThinkingPart(content=IsStr(), id='rs_68bb645d50f48196a0c49fd603b87f4503498c8aa840cf12'),
                    ThinkingPart(content=IsStr(), id='rs_68bb645d50f48196a0c49fd603b87f4503498c8aa840cf12'),
                    TextPart(content=IsStr(), id='msg_68bb64663d1c8196b9c7e78e7018cc4103498c8aa840cf12'),
                ],
                usage=RequestUsage(input_tokens=13, output_tokens=1616, details={'reasoning_tokens': 1344}),
                model_name='o3-mini-2025-01-31',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 5, 22, 29, 38, tzinfo=timezone.utc),
                },
                provider_response_id='resp_68bb6452990081968f5aff503a55e3b903498c8aa840cf12',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    mistral_model = MistralModel('magistral-medium-latest', provider=MistralProvider(api_key=mistral_api_key))
    result = await agent.run(
        'Considering the way to cross the street, analogously, how do I cross the river?',
        model=mistral_model,
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
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content=IsStr()),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(input_tokens=664, output_tokens=747),
                model_name='magistral-medium-latest',
                timestamp=IsDatetime(),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2025, 9, 5, 22, 30, tzinfo=timezone.utc),
                },
                provider_response_id='9abe8b736bff46af8e979b52334a57cd',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


@pytest.mark.vcr()
async def test_mistral_model_thinking_part_iter(allow_model_requests: None, mistral_api_key: str):
    model = MistralModel('magistral-medium-latest', provider=MistralProvider(api_key=mistral_api_key))
    agent = Agent(model)

    async with agent.iter(user_prompt='How do I cross the street?') as agent_run:
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
                        content='How do I cross the street?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='Okay, the user is asking how to cross the street. I know that crossing the street safely involves a few key steps: first, look both ways to check for oncoming traffic; second, use a crosswalk if one is available; third, obey any traffic signals or signs that may be present; and finally, proceed with caution until you have safely reached the other side. Let me compile this information into a clear and concise response.'
                    ),
                    TextPart(
                        content="""\
To cross the street safely, follow these steps:

1. Look both ways to check for oncoming traffic.
2. Use a crosswalk if one is available.
3. Obey any traffic signals or signs that may be present.
4. Proceed with caution until you have safely reached the other side.

```markdown
To cross the street safely, follow these steps:

1. Look both ways to check for oncoming traffic.
2. Use a crosswalk if one is available.
3. Obey any traffic signals or signs that may be present.
4. Proceed with caution until you have safely reached the other side.
```

By following these steps, you can ensure a safe crossing.\
"""
                    ),
                ],
                usage=RequestUsage(input_tokens=10, output_tokens=232),
                model_name='magistral-medium-latest',
                timestamp=IsDatetime(),
                provider_name='mistral',
                provider_url='https://api.mistral.ai',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2025, 11, 28, 2, 19, 53, tzinfo=timezone.utc),
                },
                provider_response_id='9f9d90210f194076abeee223863eaaf0',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_image_url_force_download() -> None:
    """Test that force_download=True calls download_item for ImageUrl in MistralModel."""
    from unittest.mock import AsyncMock, patch

    m = MistralModel('mistral-large-2512', provider=MistralProvider(api_key='test-key'))

    with patch('pydantic_ai.models.mistral.download_item', new_callable=AsyncMock) as mock_download:
        mock_download.return_value = {
            'data': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
            'data_type': 'image/png',
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
        assert mock_download.call_args[1]['data_format'] == 'base64_uri'


async def test_image_url_no_force_download() -> None:
    """Test that force_download=False does not call download_item for ImageUrl in MistralModel."""
    from unittest.mock import AsyncMock, patch

    m = MistralModel('mistral-large-2512', provider=MistralProvider(api_key='test-key'))

    with patch('pydantic_ai.models.mistral.download_item', new_callable=AsyncMock) as mock_download:
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


async def test_document_url_force_download() -> None:
    """Test that force_download=True calls download_item for DocumentUrl PDF in MistralModel."""
    from unittest.mock import AsyncMock, patch

    m = MistralModel('mistral-large-2512', provider=MistralProvider(api_key='test-key'))

    with patch('pydantic_ai.models.mistral.download_item', new_callable=AsyncMock) as mock_download:
        mock_download.return_value = {
            'data': 'data:application/pdf;base64,JVBERi0xLjQKJdPr6eEKMSAwIG9iago8PC9UeXBlL',
            'data_type': 'application/pdf',
        }

        messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'Test PDF',
                            DocumentUrl(
                                url='https://example.com/document.pdf',
                                media_type='application/pdf',
                                force_download=True,
                            ),
                        ]
                    )
                ]
            )
        ]

        await m._map_messages(messages, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]

        mock_download.assert_called_once()
        assert mock_download.call_args[0][0].url == 'https://example.com/document.pdf'
        assert mock_download.call_args[1]['data_format'] == 'base64_uri'


async def test_document_url_no_force_download() -> None:
    """Test that force_download=False does not call download_item for DocumentUrl PDF in MistralModel."""
    from unittest.mock import AsyncMock, patch

    m = MistralModel('mistral-large-2512', provider=MistralProvider(api_key='test-key'))

    with patch('pydantic_ai.models.mistral.download_item', new_callable=AsyncMock) as mock_download:
        messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'Test PDF',
                            DocumentUrl(
                                url='https://example.com/document.pdf',
                                media_type='application/pdf',
                                force_download=False,
                            ),
                        ]
                    )
                ]
            )
        ]

        await m._map_messages(messages, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]

        mock_download.assert_not_called()
