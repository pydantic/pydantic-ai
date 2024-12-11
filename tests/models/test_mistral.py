from __future__ import annotations as _annotations

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
        AssistantMessage,
        ChatCompletionChoice,
        CompletionChunk,
        CompletionResponseStreamChoice,
        CompletionResponseStreamChoiceFinishReason,
        DeltaMessage,
        FunctionCall as MistralFunctionCall,
        Mistral,
        UsageInfo,
    )
    from mistralai.models import (
        ChatCompletionResponse as MistralChatCompletionResponse,
        CompletionEvent as MistralCompletionEvent,
        ToolCall as MistralToolCall,
    )
    from pydantic import BaseModel

    from pydantic_ai import _utils
    from pydantic_ai.agent import Agent
    from pydantic_ai.messages import (
        ArgsDict,
        ModelStructuredResponse,
        ModelTextResponse,
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
    _iter: Iterator[CompletionChunk]

    async def __anext__(self) -> CompletionChunk:
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


def completion_message(message: AssistantMessage, *, usage: UsageInfo | None = None) -> MistralChatCompletionResponse:
    return MistralChatCompletionResponse(
        id='123',
        choices=[ChatCompletionChoice(finish_reason='stop', index=0, message=message)],
        created=1704067200,  # 2024-01-01
        model='mistral-large-latest',
        object='chat.completion',
        usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )


def chunk(
    delta: list[DeltaMessage], finish_reason: CompletionResponseStreamChoiceFinishReason | None = None
) -> MistralCompletionEvent:
    return MistralCompletionEvent(
        data=CompletionChunk(
            id='x',
            choices=[
                CompletionResponseStreamChoice(index=index, delta=delta, finish_reason=finish_reason)
                for index, delta in enumerate(delta)
            ],
            created=1704067200,  # 2024-01-01
            model='gpt-4',
            object='chat.completion.chunk',
            usage=UsageInfo(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        )
    )


def text_chunk(
    text: str, finish_reason: CompletionResponseStreamChoiceFinishReason | None = None
) -> MistralCompletionEvent:
    return chunk([DeltaMessage(content=text, role='assistant')], finish_reason=finish_reason)


#####################
## Completion
#####################


async def test_multiple_completions(allow_model_requests: None):
    completions = [
        completion_message(AssistantMessage(content='world')),
        completion_message(AssistantMessage(content='hello again')),
    ]
    mock_client = MockMistralAI.create_mock(completions)
    m = MistralModel('mistral-large-latest', client=mock_client)
    agent = Agent(m)

    result = await agent.run('hello')
    assert result.data == 'world'
    assert result.cost().request_tokens == 0

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
    completions = [
        completion_message(AssistantMessage(content='world')),
        completion_message(AssistantMessage(content='hello again')),
        completion_message(AssistantMessage(content='final message')),
    ]
    mock_client = MockMistralAI.create_mock(completions)
    m = MistralModel('mistral-large-latest', client=mock_client)
    agent = Agent(m)

    result = await agent.run('hello')
    assert result.data == 'world'
    assert result.cost().request_tokens == 0

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
    stream = [text_chunk('hello '), text_chunk('world '), text_chunk('welcome '), text_chunk('mistral'), chunk([])]
    mock_client = MockMistralAI.create_stream_mock(stream)
    m = MistralModel('mistral-large-latest', client=mock_client)
    agent = Agent(model=m)

    async with agent.run_stream('') as result:
        assert not result.is_structured
        assert not result.is_complete
        assert [c async for c in result.stream(debounce_by=None)] == snapshot(
            ['hello ', 'hello world ', 'hello world welcome ', 'hello world welcome mistral']
        )
        assert result.is_complete
        assert result.cost().request_tokens == 1


async def test_stream_text_finish_reason(allow_model_requests: None):
    stream = [text_chunk('hello '), text_chunk('world'), text_chunk('.', finish_reason='stop')]
    mock_client = MockMistralAI.create_stream_mock(stream)
    m = MistralModel('mistral-large-latest', client=mock_client)
    agent = Agent(model=m)

    async with agent.run_stream('') as result:
        assert not result.is_structured
        assert not result.is_complete
        assert [c async for c in result.stream(debounce_by=None)] == snapshot(['hello ', 'hello world', 'hello world.'])
        assert result.is_complete


async def test_no_delta(allow_model_requests: None):
    stream = [chunk([]), text_chunk('hello '), text_chunk('world')]
    mock_client = MockMistralAI.create_stream_mock(stream)
    m = MistralModel('mistral-large-latest', client=mock_client)
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert not result.is_structured
        assert not result.is_complete
        assert [c async for c in result.stream(debounce_by=None)] == snapshot(['hello ', 'hello world'])
        assert result.is_complete
        assert result.cost().request_tokens == 1


#####################
## Completion Structured
#####################


async def test_request_structured_with_arguments_dict_response(allow_model_requests: None):
    class CityLocation(BaseModel):
        city: str
        country: str

    c = completion_message(
        AssistantMessage(
            content=None,
            role='assistant',
            tool_calls=[
                MistralToolCall(
                    id='123',
                    function=MistralFunctionCall(arguments={'city': 'paris', 'country': 'france'}, name='final_result'),
                    type='function',
                )
            ],
        )
    )
    mock_client = MockMistralAI.create_mock(c)
    m = MistralModel('mistral-large-latest', client=mock_client)
    agent = Agent(model=m, result_type=CityLocation)

    result = await agent.run('Hello')
    assert result.data == CityLocation(city='paris', country='france')
    assert result.all_messages() == snapshot(
        [
            UserPrompt(content='Hello', timestamp=IsNow(tz=timezone.utc)),
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
