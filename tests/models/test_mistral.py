from __future__ import annotations as _annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from typing import Any, cast

import pytest
from inline_snapshot import snapshot

from pydantic_ai import _utils
from pydantic_ai.agent import Agent
from pydantic_ai.messages import ModelTextResponse, UserPrompt
from pydantic_ai.models.mistral import MistralModel

from ..conftest import IsNow, try_import

with try_import() as imports_successful:
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
    stream: list[MistralCompletionEvent] | list[list[MistralCompletionEvent]] | None = None
    index = 0

    @cached_property
    def chat(self) -> Any:
        return type('Chat', (), {'complete_async': self.chat_completions_create})

    @classmethod
    def create_mock(cls, completions: MistralChatCompletionResponse | list[MistralChatCompletionResponse]) -> Mistral:
        return cast(Mistral, cls(completions=completions))

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


# TODO
# async def test_three_completions(allow_model_requests: None):
#     completions = [
#         completion_message(AssistantMessage(content='world')),
#         completion_message(AssistantMessage(content='hello again')),
#         completion_message(AssistantMessage(content='final message')),
#     ]
#     mock_client = MockMistralAI.create_mock(completions)
#     m = MistralModel('mistral-large-latest', client=mock_client)
#     agent = Agent(m)

#     result = await agent.run('hello')
#     assert result.data == 'world'
#     assert result.cost().request_tokens == 0

#     result = await agent.run('hello again', message_history=result.new_messages())
#     assert result.data == 'hello again'
#     assert result.cost().request_tokens == 0

#     result = await agent.run('final message', message_history=result.new_messages())
#     assert result.data == 'final message'
#     assert result.cost().request_tokens == 0
#     assert result.all_messages() == snapshot(
#         [
#             UserPrompt(content='hello', timestamp=IsNow(tz=timezone.utc)),
#             ModelTextResponse(content='world', timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)),
#             UserPrompt(content='hello again', timestamp=IsNow(tz=timezone.utc)),
#             ModelTextResponse(content='hello again', timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)),
#             UserPrompt(content='final message', timestamp=IsNow(tz=timezone.utc)),
#             ModelTextResponse(content='final message', timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)),
#         ]
#     )
