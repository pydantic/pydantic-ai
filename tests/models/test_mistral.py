from __future__ import annotations as _annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from typing import Any, cast

import pytest
from inline_snapshot import snapshot

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
        UsageInfo as MistralUsageInfo,
    )
    from mistralai.models import (
        ChatCompletionResponse as MistralChatCompletionResponse,
        CompletionEvent as MistralCompletionEvent,
        ToolCall as MistralToolCall,
    )
    from pydantic import BaseModel

    from pydantic_ai import _utils
    from pydantic_ai.agent import Agent
    from pydantic_ai.exceptions import ModelRetry
    from pydantic_ai.messages import (
        ArgsDict,
        ArgsJson,
        ModelStructuredResponse,
        ModelTextResponse,
        RetryPrompt,
        SystemPrompt,
        ToolCall,
        ToolReturn,
        UserPrompt,
    )
    from pydantic_ai.models.mistral import MistralModel

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
    stream: list[MistralCompletionEvent] | None = None
    index = 0
    sync: bool = False

    @cached_property
    def chat(self) -> Any:
        if self.sync:
            return type('Chat', (), {'complete': self.chat_completions_create})
        elif self.stream:
            return type('Chat', (), {'stream_async': self.chat_completions_create})
        else:
            return type('Chat', (), {'complete_async': self.chat_completions_create})

    @classmethod
    def create_sync_mock(cls, completions: MistralChatCompletionResponse) -> Mistral:
        return cast(Mistral, cls(completions=completions, sync=True))

    @classmethod
    def create_mock(cls, completions: MistralChatCompletionResponse | list[MistralChatCompletionResponse]) -> Mistral:
        return cast(Mistral, cls(completions=completions))

    @classmethod
    def create_stream_mock(cls, completions_streams: list[MistralCompletionEvent]) -> Mistral:
        return cast(Mistral, cls(stream=completions_streams))

    async def chat_completions_create(  # pragma: no cover
        self, *_args: Any, stream: bool = False, **_kwargs: Any
    ) -> MistralChatCompletionResponse | MockAsyncStream | list[MistralChatCompletionResponse]:
        if stream:
            assert self.stream is not None, 'you can only used `stream=True` if `stream` is provided'
            # noinspection PyUnresolvedReferences
            if isinstance(self.stream[0], list):
                response = MockAsyncStream(iter(self.stream[self.index]))  # type: ignore

            else:
                response = MockAsyncStream(iter(self.stream))  # type: ignore
        else:
            assert self.completions is not None, 'you can only used `stream=False` if `completions` are provided'
            if isinstance(self.completions, list):
                response = self.completions[self.index]
            else:
                response = self.completions
        self.index += 1
        return response


def completion_message(
    message: MistralAssistantMessage, *, usage: MistralUsageInfo | None = None
) -> MistralChatCompletionResponse:
    return MistralChatCompletionResponse(
        id='123',
        choices=[MistralChatCompletionChoice(finish_reason='stop', index=0, message=message)],
        created=1704067200,  # 2024-01-01
        model='mistral-large-latest',
        object='chat.completion',
        usage=usage or MistralUsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )


def chunk(
    delta: list[MistralDeltaMessage], finish_reason: MistralCompletionResponseStreamChoiceFinishReason | None = None
) -> MistralCompletionEvent:
    return MistralCompletionEvent(
        data=MistralCompletionChunk(
            id='x',
            choices=[
                MistralCompletionResponseStreamChoice(index=index, delta=delta, finish_reason=finish_reason)
                for index, delta in enumerate(delta)
            ],
            created=1704067200,  # 2024-01-01
            model='gpt-4',
            object='chat.completion.chunk',
            usage=MistralUsageInfo(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        )
    )


def text_chunk(
    text: str, finish_reason: MistralCompletionResponseStreamChoiceFinishReason | None = None
) -> MistralCompletionEvent:
    return chunk([MistralDeltaMessage(content=text, role='assistant')], finish_reason=finish_reason)


#####################
## Completion
#####################


async def test_multiple_completions(allow_model_requests: None):
    # Given
    completions = [
        completion_message(
            MistralAssistantMessage(content='world'),
            usage=MistralUsageInfo(prompt_tokens=1, completion_tokens=2, total_tokens=3),
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
    assert result.cost().request_tokens == 1
    assert result.cost().response_tokens == 2
    assert result.cost().total_tokens == 3

    result = await agent.run('hello again', message_history=result.new_messages())
    assert result.data == 'hello again'
    assert result.cost().request_tokens == 0
    assert result.all_messages() == snapshot(
        [
            UserPrompt(content='hello', timestamp=IsNow(tz=timezone.utc)),
            ModelTextResponse(content='world', timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)),
            UserPrompt(content='hello again', timestamp=IsNow(tz=timezone.utc)),
            ModelTextResponse(content='hello again', timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)),
        ]
    )


async def test_three_completions(allow_model_requests: None):
    # Given
    completions = [
        completion_message(
            MistralAssistantMessage(content='world'),
            usage=MistralUsageInfo(prompt_tokens=1, completion_tokens=2, total_tokens=3),
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
    assert result.cost().request_tokens == 1
    assert result.cost().response_tokens == 2
    assert result.cost().total_tokens == 3

    result = await agent.run('hello again', message_history=result.all_messages())
    assert result.data == 'hello again'
    assert result.cost().request_tokens == 0

    result = await agent.run('final message', message_history=result.all_messages())
    assert result.data == 'final message'
    assert result.cost().request_tokens == 0
    assert result.all_messages() == snapshot(
        [
            UserPrompt(content='hello', timestamp=IsNow(tz=timezone.utc)),
            ModelTextResponse(content='world', timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)),
            UserPrompt(content='hello again', timestamp=IsNow(tz=timezone.utc)),
            ModelTextResponse(content='hello again', timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)),
            UserPrompt(content='final message', timestamp=IsNow(tz=timezone.utc)),
            ModelTextResponse(content='final message', timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)),
        ]
    )


#####################
## Completion Stream
#####################


async def test_stream_text(allow_model_requests: None):
    # Given
    stream = [text_chunk('hello '), text_chunk('world '), text_chunk('welcome '), text_chunk('mistral'), chunk([])]
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
        assert result.cost().request_tokens == 1
        assert result.cost().response_tokens == 2
        assert result.cost().total_tokens == 3


async def test_stream_text_finish_reason(allow_model_requests: None):
    # Given
    stream = [text_chunk('hello '), text_chunk('world'), text_chunk('.', finish_reason='stop')]
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
    stream = [chunk([]), text_chunk('hello '), text_chunk('world')]
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
        assert result.cost().request_tokens == 1
        assert result.cost().response_tokens == 2
        assert result.cost().total_tokens == 3


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
    assert result.all_messages() == snapshot(
        [
            UserPrompt(content='User prompt value', timestamp=IsNow(tz=timezone.utc)),
            ModelStructuredResponse(
                calls=[
                    ToolCall(
                        tool_name='final_result',
                        args=ArgsDict(args_dict={'city': 'paris', 'country': 'france'}),
                        tool_call_id='123',
                    )
                ],
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ),
            ToolReturn(
                tool_name='final_result',
                content='Final result processed.',
                tool_call_id='123',
                timestamp=IsNow(tz=timezone.utc),
            ),
        ]
    )
    assert result.cost().request_tokens == 1
    assert result.cost().response_tokens == 2
    assert result.cost().total_tokens == 3


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
    assert result.all_messages() == snapshot(
        [
            UserPrompt(content='User prompt value', timestamp=IsNow(tz=timezone.utc)),
            ModelStructuredResponse(
                calls=[
                    ToolCall(
                        tool_name='final_result',
                        args=ArgsJson(args_json='{"city": "paris", "country": "france"}'),
                        tool_call_id='123',
                    )
                ],
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ),
            ToolReturn(
                tool_name='final_result',
                content='Final result processed.',
                tool_call_id='123',
                timestamp=IsNow(tz=timezone.utc),
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
    assert result.all_messages() == snapshot(
        [
            SystemPrompt(content='System prompt value'),
            UserPrompt(content='User prompt value', timestamp=IsNow(tz=timezone.utc)),
            ModelStructuredResponse(
                calls=[
                    ToolCall(
                        tool_name='final_result',
                        args=ArgsJson(args_json='{"response": 42}'),
                        tool_call_id='123',
                    )
                ],
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ),
            ToolReturn(
                tool_name='final_result',
                content='Final result processed.',
                tool_call_id='123',
                timestamp=IsNow(tz=timezone.utc),
            ),
        ]
    )


#####################
## Completion Model Structured Stream (Json Mode)
#####################

# TODO


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
    model = MistralModel('mistral-large-latest', client=mock_client)
    agent = Agent(model, system_prompt='this is the system prompt')

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
            SystemPrompt(content='this is the system prompt'),
            UserPrompt(content='Hello', timestamp=IsNow(tz=timezone.utc)),
            ModelStructuredResponse(
                calls=[
                    ToolCall(
                        tool_name='get_location',
                        args=ArgsJson(args_json='{"loc_name": "San Fransisco"}'),
                        tool_call_id='1',
                    )
                ],
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ),
            RetryPrompt(
                tool_name='get_location',
                content='Wrong location, please try again',
                tool_call_id='1',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelStructuredResponse(
                calls=[
                    ToolCall(
                        tool_name='get_location',
                        args=ArgsJson(args_json='{"loc_name": "London"}'),
                        tool_call_id='2',
                    )
                ],
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ),
            ToolReturn(
                tool_name='get_location',
                content='{"lat": 51, "lng": 0}',
                tool_call_id='2',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelTextResponse(content='final response', timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc)),
        ]
    )
    assert result.cost().request_tokens == 5
    assert result.cost().response_tokens == 3
    assert result.cost().total_tokens == 9


#####################
## Completion Function call Stream
#####################

# TODO
