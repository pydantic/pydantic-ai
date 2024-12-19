from __future__ import annotations as _annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from typing import Any, cast

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel
from typing_extensions import TypedDict

from pydantic_ai import _utils
from pydantic_ai.agent import Agent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import (
    ArgsDict,
    ArgsJson,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from ..conftest import IsNow, try_import

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
        ToolCall as MistralToolCall,
    )
    from mistralai.types.basemodel import Unset as MistralUnset

    from pydantic_ai.models.mistral import (
        MistralAgentModel,
        MistralModel,
        MistralStreamStructuredResponse,
    )

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='mistral not installed'),
    pytest.mark.anyio,
]


@dataclass
class MockAsyncStream:
    _iter: Iterator[MistralCompletionChunk]

    async def __anext__(self) -> MistralCompletionChunk:
        return _utils.sync_anext(self._iter)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args: Any):
        pass


@dataclass
class MockMistralAI:
    completions: MistralChatCompletionResponse | list[MistralChatCompletionResponse] | None = None
    stream: list[MistralCompletionEvent] | list[list[MistralCompletionEvent]] | None = None
    index = 0

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
    def create_mock(cls, completions: MistralChatCompletionResponse | list[MistralChatCompletionResponse]) -> Mistral:
        return cast(Mistral, cls(completions=completions))

    @classmethod
    def create_stream_mock(
        cls, completions_streams: list[MistralCompletionEvent] | list[list[MistralCompletionEvent]]
    ) -> Mistral:
        return cast(Mistral, cls(stream=completions_streams))

    async def chat_completions_create(  # pragma: no cover
        self, *_args: Any, stream: bool = False, **_kwargs: Any
    ) -> MistralChatCompletionResponse | MockAsyncStream | list[MistralChatCompletionResponse]:
        response: MistralChatCompletionResponse | MockAsyncStream | list[MistralChatCompletionResponse]
        if stream or self.stream:
            assert self.stream is not None, 'you can only used `stream=True` if `stream` is provided'
            if isinstance(self.stream[0], list):
                response = MockAsyncStream(iter(self.stream[self.index]))  # pyright: ignore[reportArgumentType]

            else:
                response = MockAsyncStream(iter(self.stream))  # pyright: ignore[reportArgumentType]

        else:
            assert self.completions is not None, 'you can only used `stream=False` if `completions` are provided'
            if isinstance(self.completions, list):
                response = self.completions[self.index]
            else:
                response = self.completions
        self.index += 1
        return response


def completion_message(
    message: MistralAssistantMessage, *, usage: MistralUsageInfo | None = None, with_created: bool = True
) -> MistralChatCompletionResponse:
    return MistralChatCompletionResponse(
        id='123',
        choices=[MistralChatCompletionChoice(finish_reason='stop', index=0, message=message)],
        created=1704067200 if with_created else None,  # 2024-01-01
        model='mistral-large-latest',
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
            created=1704067200 if with_created else None,  # 2024-01-01
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


def struc_chunk(
    tool_name: str,
    tool_arguments: str,
    finish_reason: MistralCompletionResponseStreamChoiceFinishReason | None = None,
) -> MistralCompletionEvent:
    return chunk(
        [
            MistralDeltaMessage(
                tool_calls=[
                    MistralToolCall(id='0', function=MistralFunctionCall(name=tool_name, arguments=tool_arguments))
                ]
            ),
        ],
        finish_reason=finish_reason,
    )


#####################
## Init
#####################


def test_init():
    m = MistralModel('mistral-large-latest', api_key='foobar')
    assert m.name() == 'mistral:mistral-large-latest'


#####################
## Completion
#####################


async def test_multiple_completions(allow_model_requests: None):
    # Given
    completions = [
        completion_message(
            MistralAssistantMessage(content='world'),
            usage=MistralUsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=1),
            with_created=False,
        ),
        completion_message(MistralAssistantMessage(content='hello again')),
    ]
    mock_client = MockMistralAI.create_mock(completions)
    model = MistralModel('mistral-large-latest', client=mock_client)
    agent = Agent(model=model)

    # When
    result = await agent.run('hello')

    # Then
    assert result.data == 'world'
    assert result.usage().request_tokens == 1
    assert result.usage().response_tokens == 1
    assert result.usage().total_tokens == 1

    result = await agent.run('hello again', message_history=result.new_messages())
    assert result.data == 'hello again'
    assert result.usage().request_tokens == 1
    assert result.usage().response_tokens == 1
    assert result.usage().total_tokens == 1
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse.from_text(content='world', timestamp=IsNow(tz=timezone.utc)),
            ModelRequest(parts=[UserPromptPart(content='hello again', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse.from_text(content='hello again', timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)),
        ]
    )


async def test_three_completions(allow_model_requests: None):
    # Given
    completions = [
        completion_message(
            MistralAssistantMessage(content='world'),
            usage=MistralUsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=1),
        ),
        completion_message(MistralAssistantMessage(content='hello again')),
        completion_message(MistralAssistantMessage(content='final message')),
    ]
    mock_client = MockMistralAI.create_mock(completions)
    model = MistralModel('mistral-large-latest', client=mock_client)
    agent = Agent(model=model)

    # When
    result = await agent.run('hello')

    # Them
    assert result.data == 'world'
    assert result.usage().request_tokens == 1
    assert result.usage().response_tokens == 1
    assert result.usage().total_tokens == 1

    result = await agent.run('hello again', message_history=result.all_messages())
    assert result.data == 'hello again'
    assert result.usage().request_tokens == 1
    assert result.usage().response_tokens == 1
    assert result.usage().total_tokens == 1

    result = await agent.run('final message', message_history=result.all_messages())
    assert result.data == 'final message'
    assert result.usage().request_tokens == 1
    assert result.usage().response_tokens == 1
    assert result.usage().total_tokens == 1
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse.from_text(content='world', timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)),
            ModelRequest(parts=[UserPromptPart(content='hello again', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse.from_text(content='hello again', timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)),
            ModelRequest(parts=[UserPromptPart(content='final message', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse.from_text(content='final message', timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)),
        ]
    )


#####################
## Completion Stream
#####################


async def test_stream_text(allow_model_requests: None):
    # Given
    stream = [text_chunk('hello '), text_chunk('world '), text_chunk('welcome '), text_chunkk('mistral'), chunk([])]
    mock_client = MockMistralAI.create_stream_mock(stream)
    model = MistralModel('mistral-large-latest', client=mock_client)
    agent = Agent(model=model)

    # When
    async with agent.run_stream('') as result:
        # Then
        assert not result.is_structured
        assert not result.is_complete
        assert [c async for c in result.stream(debounce_by=None)] == snapshot(
            ['hello ', 'hello world ', 'hello world welcome ', 'hello world welcome mistral']
        )
        assert result.is_complete
        assert result.usage().request_tokens == 5
        assert result.usage().response_tokens == 5
        assert result.usage().total_tokens == 5


async def test_stream_text_finish_reason(allow_model_requests: None):
    # Given
    stream = [text_chunk('hello '), text_chunkk('world'), text_chunk('.', finish_reason='stop')]
    mock_client = MockMistralAI.create_stream_mock(stream)
    model = MistralModel('mistral-large-latest', client=mock_client)
    agent = Agent(model=model)

    # When
    async with agent.run_stream('') as result:
        # Then
        assert not result.is_structured
        assert not result.is_complete
        assert [c async for c in result.stream(debounce_by=None)] == snapshot(['hello ', 'hello world', 'hello world.'])
        assert result.is_complete


async def test_no_delta(allow_model_requests: None):
    # Given
    stream = [chunk([], with_created=False), text_chunk('hello '), text_chunk('world')]
    mock_client = MockMistralAI.create_stream_mock(stream)
    model = MistralModel('mistral-large-latest', client=mock_client)
    agent = Agent(model=model)

    # When
    async with agent.run_stream('') as result:
        # Then
        assert not result.is_structured
        assert not result.is_complete
        assert [c async for c in result.stream(debounce_by=None)] == snapshot(['hello ', 'hello world'])
        assert result.is_complete
        assert result.usage().request_tokens == 3
        assert result.usage().response_tokens == 3
        assert result.usage().total_tokens == 3


#####################
## Completion Model Structured
#####################


async def test_request_model_structured_with_arguments_dict_response(allow_model_requests: None):
    class CityLocation(BaseModel):
        city: str
        country: str

    # Given
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
    model = MistralModel('mistral-large-latest', client=mock_client)
    agent = Agent(model=model, result_type=CityLocation)

    # When
    result = await agent.run('User prompt value')

    # Then
    assert result.data == CityLocation(city='paris', country='france')
    assert result.usage().request_tokens == 1
    assert result.usage().response_tokens == 2
    assert result.usage().total_tokens == 3
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='User prompt value', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args=ArgsDict(args_dict={'city': 'paris', 'country': 'france'}),
                        tool_call_id='123',
                    )
                ],
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
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


async def test_request_model_structured_with_arguments_str_response(allow_model_requests: None):
    class CityLocation(BaseModel):
        city: str
        country: str

    # Given
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
    model = MistralModel('mistral-large-latest', client=mock_client)
    agent = Agent(model=model, result_type=CityLocation)

    # When
    result = await agent.run('User prompt value')

    # Then
    assert result.data == CityLocation(city='paris', country='france')
    assert result.usage().request_tokens == 1
    assert result.usage().response_tokens == 1
    assert result.usage().total_tokens == 1
    assert result.usage().details is None
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='User prompt value', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args=ArgsJson(args_json='{"city": "paris", "country": "france"}'),
                        tool_call_id='123',
                    )
                ],
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
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


async def test_request_result_type_with_arguments_str_response(allow_model_requests: None):
    # Given
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
    model = MistralModel('mistral-large-latest', client=mock_client)
    agent = Agent(model=model, result_type=int, system_prompt='System prompt value')

    # When
    result = await agent.run('User prompt value')

    # Then
    assert result.data == 42
    assert result.usage().request_tokens == 1
    assert result.usage().response_tokens == 1
    assert result.usage().total_tokens == 1
    assert result.usage().details is None
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='System prompt value'),
                    UserPromptPart(content='User prompt value', timestamp=IsNow(tz=timezone.utc)),
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args=ArgsJson(args_json='{"response": 42}'),
                        tool_call_id='123',
                    )
                ],
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
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


#####################
## Completion Model Structured Stream (JSON Mode)
#####################


async def test_stream_structured_with_all_typd(allow_model_requests: None):
    class MyTypedDict(TypedDict, total=False):
        first: str
        second: int
        bool_value: bool
        nullable_value: int | None
        array_value: list[str]
        dict_value: dict[str, Any]
        dict_int_value: dict[str, int]
        dict_str_value: dict[int, str]

    # Given
    stream = [
        text_chunk('{'),
        text_chunk('"first": "One'),
        text_chunk(
            '", "second": 2',
        ),
        text_chunk(
            '", "bool_value": true',
        ),
        text_chunk(
            '", "nullable_value": null',
        ),
        text_chunk(
            '", "array_value": ["A", "B", "C"]',
        ),
        text_chunk(
            '", "dict_value": {"A": "A", "B":"B"}',
        ),
        text_chunk(
            '", "dict_int_value": {"A": 1, "B":2}',
        ),
        text_chunk('}'),
        chunk([]),
    ]

    mock_client = MockMistralAI.create_stream_mock(stream)
    model = MistralModel('mistral-large-latest', client=mock_client)
    agent = Agent(model, result_type=MyTypedDict)

    # When
    async with agent.run_stream('User prompt value') as result:
        # Then
        assert result.is_structured
        assert not result.is_complete
        v = [dict(c) async for c in result.stream(debounce_by=None)]
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
        assert result.usage().request_tokens == 10
        assert result.usage().response_tokens == 10
        assert result.usage().total_tokens == 10

        # double check usage matches stream count
        assert result.usage().response_tokens == len(stream)


async def test_stream_result_type_primitif_dict(allow_model_requests: None):
    """This test tests the primitif result with the pydantic ai format model response"""

    class MyTypedDict(TypedDict, total=False):
        first: str
        second: str

    # Given
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
    model = MistralModel('mistral-large-latest', client=mock_client)
    agent = Agent(model=model, result_type=MyTypedDict)

    # When
    async with agent.run_stream('User prompt value') as result:
        # Then
        assert result.is_structured
        assert not result.is_complete
        v = [c async for c in result.stream(debounce_by=None)]
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
                {'first': 'One', 'second': ''},
                {'first': 'One', 'second': ''},
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
        assert result.usage().request_tokens == 34
        assert result.usage().response_tokens == 34
        assert result.usage().total_tokens == 34

        # double check usage matches stream count
        assert result.usage().response_tokens == len(stream)


async def test_stream_result_type_primitif_int(allow_model_requests: None):
    """This test tests the primitif result with the pydantic ai format model response"""
    # Given
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
    model = MistralModel('mistral-large-latest', client=mock_client)
    agent = Agent(model=model, result_type=int)

    # When
    async with agent.run_stream('User prompt value') as result:
        # Then
        assert result.is_structured
        assert not result.is_complete
        v = [c async for c in result.stream(debounce_by=None)]
        assert v == snapshot([1, 1, 1])
        assert result.is_complete
        assert result.usage().request_tokens == 6
        assert result.usage().response_tokens == 6
        assert result.usage().total_tokens == 6

        # double check usage matches stream count
        assert result.usage().response_tokens == len(stream)


async def test_stream_result_type_primitif_array(allow_model_requests: None):
    """This test tests the primitif result with the pydantic ai format model response"""
    # Given
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
    model = MistralModel('mistral-large-latest', client=mock_client)
    agent = Agent(model, result_type=list[str])

    # When
    async with agent.run_stream('User prompt value') as result:
        # Then
        assert result.is_structured
        assert not result.is_complete
        v = [c async for c in result.stream(debounce_by=None)]
        assert v == snapshot(
            [
                ['f'],
                ['fi'],
                ['fir'],
                ['firs'],
                ['first'],
                ['first'],
                ['first'],
                ['first'],
                ['first', 'O'],
                ['first', 'On'],
                ['first', 'One'],
                ['first', 'One'],
                ['first', 'One'],
                ['first', 'One'],
                ['first', 'One', 's'],
                ['first', 'One', 'se'],
                ['first', 'One', 'sec'],
                ['first', 'One', 'seco'],
                ['first', 'One', 'secon'],
                ['first', 'One', 'second'],
                ['first', 'One', 'second'],
                ['first', 'One', 'second'],
                ['first', 'One', 'second'],
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
        assert result.usage().request_tokens == 35
        assert result.usage().response_tokens == 35
        assert result.usage().total_tokens == 35

        # double check usage matches stream count
        assert result.usage().response_tokens == len(stream)


async def test_stream_result_type_basemodel(allow_model_requests: None):
    class MyTypedBaseModel(BaseModel):
        first: str = ''  # Note: Don't forget to set default values
        second: str = ''

    # Given
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
    model = MistralModel('mistral-large-latest', client=mock_client)
    agent = Agent(model=model, result_type=MyTypedBaseModel)

    # When
    async with agent.run_stream('User prompt value') as result:
        # Then
        assert result.is_structured
        assert not result.is_complete
        v = [c async for c in result.stream(debounce_by=None)]
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
        assert result.usage().request_tokens == 34
        assert result.usage().response_tokens == 34
        assert result.usage().total_tokens == 34

        # double check usage matches stream count
        assert result.usage().response_tokens == len(stream)


#####################
## Completion Function call
#####################


async def test_request_tool_call(allow_model_requests: None):
    # Given
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
    model = MistralModel('mistral-large-latest', client=mock_client)
    agent = Agent(model, system_prompt='this is the system prompt')

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        if loc_name == 'London':
            return json.dumps({'lat': 51, 'lng': 0})
        else:
            raise ModelRetry('Wrong location, please try again')

    # When
    result = await agent.run('Hello')

    # Then
    assert result.data == 'final response'
    assert result.usage().request_tokens == 6
    assert result.usage().response_tokens == 4
    assert result.usage().total_tokens == 10
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
                        args=ArgsJson(args_json='{"loc_name": "San Fransisco"}'),
                        tool_call_id='1',
                    )
                ],
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
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
                        args=ArgsJson(args_json='{"loc_name": "London"}'),
                        tool_call_id='2',
                    )
                ],
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
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
            ModelResponse.from_text(
                content='final response', timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
            ),
        ]
    )


async def test_request_tool_call_with_result_type(allow_model_requests: None):
    class MyTypedDict(TypedDict, total=False):
        lat: int
        lng: int

    # Given
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
    model = MistralModel('mistral-large-latest', client=mock_client)
    agent = Agent(model, system_prompt='this is the system prompt', result_type=MyTypedDict)

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        if loc_name == 'London':
            return json.dumps({'lat': 51, 'lng': 0})
        else:
            raise ModelRetry('Wrong location, please try again')

    # When
    result = await agent.run('Hello')

    # Then
    assert result.data == {'lat': 51, 'lng': 0}
    assert result.usage().request_tokens == 7
    assert result.usage().response_tokens == 4
    assert result.usage().total_tokens == 12
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
                        args=ArgsJson(args_json='{"loc_name": "San Fransisco"}'),
                        tool_call_id='1',
                    )
                ],
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
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
                        args=ArgsJson(args_json='{"loc_name": "London"}'),
                        tool_call_id='2',
                    )
                ],
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
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
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args=ArgsJson(args_json='{"lat": 51, "lng": 0}'),
                        tool_call_id='1',
                    )
                ],
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='1',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
            ),
        ]
    )


#####################
## Completion Function call Stream
#####################


async def test_stream_tool_call_with_return_type(allow_model_requests: None):
    class MyTypedDict(TypedDict, total=False):
        won: bool

    # Given
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
    model = MistralModel('mistral-large-latest', client=mock_client)
    agent = Agent(model, system_prompt='this is the system prompt', result_type=MyTypedDict)

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        return json.dumps({'lat': 51, 'lng': 0})

    # When
    async with agent.run_stream('User prompt value') as result:
        # Then
        assert not result.is_complete
        v = [c async for c in result.stream(debounce_by=None)]
        assert v == snapshot([{'won': True}, {'won': True}])
        assert result.is_complete
        assert result.timestamp() == datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        assert result.usage().request_tokens == 4
        assert result.usage().response_tokens == 4
        assert result.usage().total_tokens == 4

        # double check usage matches stream count
        assert result.usage().response_tokens == 4

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='this is the system prompt'),
                    UserPromptPart(content='User prompt value', timestamp=IsNow(tz=timezone.utc)),
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_location',
                        args=ArgsJson(args_json='{"loc_name": "San Fransisco"}'),
                        tool_call_id='1',
                    )
                ],
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_location',
                        content='{"lat": 51, "lng": 0}',
                        tool_call_id='1',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='final_result', args=ArgsJson(args_json='{"won": true}'), tool_call_id='1')
                ],
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='1',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
            ),
        ]
    )

    assert await result.get_data() == {'won': True}


async def test_stream_tool_call(allow_model_requests: None):
    # Given
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
    model = MistralModel('mistral-large-latest', client=mock_client)
    agent = Agent(model, system_prompt='this is the system prompt')

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        return json.dumps({'lat': 51, 'lng': 0})

    # When
    async with agent.run_stream('User prompt value') as result:
        # Then
        assert not result.is_complete
        v = [c async for c in result.stream(debounce_by=None)]
        assert v == snapshot(['final ', 'final response'])
        assert result.is_complete
        assert result.timestamp() == datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        assert result.usage().request_tokens == 6
        assert result.usage().response_tokens == 6
        assert result.usage().total_tokens == 6

        # double check usage matches stream count
        assert result.usage().response_tokens == 6

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='this is the system prompt'),
                    UserPromptPart(content='User prompt value', timestamp=IsNow(tz=timezone.utc)),
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_location',
                        args=ArgsJson(args_json='{"loc_name": "San Fransisco"}'),
                        tool_call_id='1',
                    )
                ],
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_location',
                        content='{"lat": 51, "lng": 0}',
                        tool_call_id='1',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ]
            ),
            ModelResponse.from_text(content='final response', timestamp=IsNow(tz=timezone.utc)),
        ]
    )


async def test_stream_tool_call_with_retry(allow_model_requests: None):
    # Given
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
    model = MistralModel('mistral-large-latest', client=mock_client)
    agent = Agent(model, system_prompt='this is the system prompt')

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        if loc_name == 'London':
            return json.dumps({'lat': 51, 'lng': 0})
        else:
            raise ModelRetry('Wrong location, please try again')

    # When
    async with agent.run_stream('User prompt value') as result:
        # Then
        assert not result.is_complete
        v = [c async for c in result.stream(debounce_by=None)]
        assert v == snapshot(['final ', 'final response'])
        assert result.is_complete
        assert result.timestamp() == datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        assert result.usage().request_tokens == 7
        assert result.usage().response_tokens == 7
        assert result.usage().total_tokens == 7

        # double check usage matches stream count
        assert result.usage().response_tokens == 7

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='this is the system prompt'),
                    UserPromptPart(content='User prompt value', timestamp=IsNow(tz=timezone.utc)),
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_location',
                        args=ArgsJson(args_json='{"loc_name": "San Fransisco"}'),
                        tool_call_id='1',
                    )
                ],
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
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
                        args=ArgsJson(args_json='{"loc_name": "London"}'),
                        tool_call_id='2',
                    )
                ],
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
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
            ModelResponse.from_text(content='final response', timestamp=IsNow(tz=timezone.utc)),
        ]
    )


#####################
## Test methods
#####################


def test_generate_user_output_format_complex():
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
    mam = MistralAgentModel(Mistral(api_key=''), '', False, [], [], '{schema}')
    result = mam._generate_user_output_format([schema])  # pyright: ignore[reportPrivateUsage]
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


def test_generate_user_output_format_multiple():
    schema = {'properties': {'prop_anyOf': {'anyOf': [{'type': 'string'}, {'type': 'integer'}]}}}
    mam = MistralAgentModel(Mistral(api_key=''), '', False, [], [], '{schema}')
    result = mam._generate_user_output_format([schema, schema])  # pyright: ignore[reportPrivateUsage]
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
def test_validate_required_json_shema(desc: str, schema: dict[str, Any], data: dict[str, Any], expected: bool) -> None:
    result = MistralStreamStructuredResponse._validate_required_json_schema(data, schema)  # pyright: ignore[reportPrivateUsage]
    assert result == expected, f'{desc} — expected {expected}, got {result}'
