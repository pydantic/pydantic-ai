from __future__ import annotations as _annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from typing import Any, cast


from inline_snapshot import snapshot
from pydantic_ai.agent import Agent
from pydantic_ai.messages import ModelTextResponse, UserPrompt
from pydantic_ai.models.mistral import MistralModel
import pytest

from pydantic_ai import _utils

from ..conftest import IsNow, try_import

with try_import() as imports_successful:
    from mistralai import CompletionResponseStreamChoice, CompletionResponseStreamChoiceFinishReason, DeltaMessage
    from mistralai import AssistantMessage, ChatCompletionChoice, CompletionChunk, Mistral, UsageInfo
    from mistralai.models import (
        ChatCompletionResponse as MistralChatCompletionResponse,
        CompletionEvent as MistralCompletionEvent,
    )

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

    @cached_property
    def chat(self) -> Any:
        if self.stream:
            return type('Chat', (), {'stream_async': self.chat_completions_create})
        else:
            return type('Chat', (), {'complete_async': self.chat_completions_create})

    @classmethod
    def create_mock(cls, completions: MistralChatCompletionResponse | list[MistralChatCompletionResponse]) -> Mistral:
        return cast(Mistral, cls(completions=completions))

    @classmethod
    def create_stream_mock(cls, completions_streams: list[MistralCompletionEvent]) -> Mistral:
        return cast(Mistral, cls(stream=completions_streams))


    async def chat_completions_create(  # pragma: no cover
        self, *_args: Any, stream: bool = False, **_kwargs: Any
    ) -> MistralChatCompletionResponse | MockAsyncStream:
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

def chunk(delta: list[DeltaMessage], finish_reason: CompletionResponseStreamChoiceFinishReason | None = None) -> MistralCompletionEvent:
    return MistralCompletionEvent(data=
    CompletionChunk(
        id='x',
        choices=[
            CompletionResponseStreamChoice(index=index, delta=delta, finish_reason=finish_reason) for index, delta in enumerate(delta)
        ],
        created=1704067200,  # 2024-01-01
        model='gpt-4',
        object='chat.completion.chunk',
        usage=UsageInfo(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    ))
    
def text_chunk(text: str, finish_reason: CompletionResponseStreamChoiceFinishReason | None = None) -> MistralCompletionEvent:
    return chunk([DeltaMessage(content=text, role='assistant')], finish_reason=finish_reason)


#####################
## No Streaming test
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

async def test_stream_text(allow_model_requests: None):
    stream = [text_chunk('hello '), text_chunk('world '), text_chunk('welcome '), text_chunk('mistral'), chunk([])]
    mock_client = MockMistralAI.create_stream_mock(stream)
    m = MistralModel('mistral-large-latest', client=mock_client)
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert not result.is_structured
        assert not result.is_complete
        assert [c async for c in result.stream(debounce_by=None)] == snapshot(['hello ', 'hello world ', 'hello world welcome ', 'hello world welcome mistral'])
        assert result.is_complete
        assert result.cost().request_tokens == 1
