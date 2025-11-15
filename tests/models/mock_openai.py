from __future__ import annotations as _annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, cast

from ..conftest import raise_if_exception, try_import
from .mock_async_stream import MockAsyncStream

with try_import() as imports_successful:
    from openai import AsyncOpenAI
    from openai.types import chat, responses
    from openai.types.chat.chat_completion import Choice, ChoiceLogprobs
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
    from openai.types.completion_usage import CompletionUsage
    from openai.types.responses.response import ResponseUsage
    from openai.types.responses.response_output_item import ResponseOutputItem

    from pydantic_ai.models.openai import NOT_GIVEN, OMIT

    MockChatCompletion = chat.ChatCompletion | Exception
    MockChatCompletionChunk = chat.ChatCompletionChunk | Exception
    MockResponse = responses.Response | Exception
    MockResponseStreamEvent = responses.ResponseStreamEvent | Exception


@dataclass
class MockOpenAI:
    completions: MockChatCompletion | Sequence[MockChatCompletion] | None = None
    stream: Sequence[MockChatCompletionChunk] | Sequence[Sequence[MockChatCompletionChunk]] | None = None
    index: int = 0
    chat_completion_kwargs: list[dict[str, Any]] = field(default_factory=list)
    base_url: str = 'https://api.openai.com/v1'

    @cached_property
    def chat(self) -> Any:
        chat_completions = type('Completions', (), {'create': self.chat_completions_create})
        return type('Chat', (), {'completions': chat_completions})

    @classmethod
    def create_mock(cls, completions: MockChatCompletion | Sequence[MockChatCompletion]) -> AsyncOpenAI:
        return cast(AsyncOpenAI, cls(completions=completions))

    @classmethod
    def create_mock_stream(
        cls,
        stream: Sequence[MockChatCompletionChunk] | Sequence[Sequence[MockChatCompletionChunk]],
    ) -> AsyncOpenAI:
        return cast(AsyncOpenAI, cls(stream=stream))

    async def chat_completions_create(  # pragma: lax no cover
        self, *_args: Any, stream: bool = False, **kwargs: Any
    ) -> chat.ChatCompletion | MockAsyncStream[MockChatCompletionChunk]:
        self.chat_completion_kwargs.append({k: v for k, v in kwargs.items() if v not in (NOT_GIVEN, OMIT)})

        if stream:
            assert self.stream is not None, 'you can only used `stream=True` if `stream` is provided'
            if isinstance(self.stream[0], Sequence):
                response = MockAsyncStream(iter(cast(list[MockChatCompletionChunk], self.stream[self.index])))
            else:
                response = MockAsyncStream(iter(cast(list[MockChatCompletionChunk], self.stream)))
        else:
            assert self.completions is not None, 'you can only used `stream=False` if `completions` are provided'
            if isinstance(self.completions, Sequence):
                raise_if_exception(self.completions[self.index])
                response = cast(chat.ChatCompletion, self.completions[self.index])
            else:
                raise_if_exception(self.completions)
                response = cast(chat.ChatCompletion, self.completions)
        self.index += 1
        return response


def get_mock_chat_completion_kwargs(async_open_ai: AsyncOpenAI) -> list[dict[str, Any]]:
    if isinstance(async_open_ai, MockOpenAI):
        return async_open_ai.chat_completion_kwargs
    else:  # pragma: no cover
        raise RuntimeError('Not a MockOpenAI instance')


def completion_message(
    message: ChatCompletionMessage, *, usage: CompletionUsage | None = None, logprobs: ChoiceLogprobs | None = None
) -> chat.ChatCompletion:
    choices = [Choice(finish_reason='stop', index=0, message=message)]
    if logprobs:
        choices = [Choice(finish_reason='stop', index=0, message=message, logprobs=logprobs)]
    return chat.ChatCompletion(
        id='123',
        choices=choices,
        created=1704067200,  # 2024-01-01
        model='gpt-4o-123',
        object='chat.completion',
        usage=usage,
    )


@dataclass
class MockOpenAIResponses:
    response: MockResponse | Sequence[MockResponse] | None = None
    stream: Sequence[MockResponseStreamEvent] | Sequence[Sequence[MockResponseStreamEvent]] | None = None
    index: int = 0
    response_kwargs: list[dict[str, Any]] = field(default_factory=list)
    base_url: str = 'https://api.openai.com/v1'

    @cached_property
    def responses(self) -> Any:
        return type('Responses', (), {'create': self.responses_create})

    @classmethod
    def create_mock(cls, responses: MockResponse | Sequence[MockResponse]) -> AsyncOpenAI:
        return cast(AsyncOpenAI, cls(response=responses))

    @classmethod
    def create_mock_stream(
        cls,
        stream: Sequence[MockResponseStreamEvent] | Sequence[Sequence[MockResponseStreamEvent]],
    ) -> AsyncOpenAI:
        return cast(AsyncOpenAI, cls(stream=stream))  # pragma: lax no cover

    async def responses_create(  # pragma: lax no cover
        self, *_args: Any, stream: bool = False, **kwargs: Any
    ) -> responses.Response | MockAsyncStream[MockResponseStreamEvent]:
        self.response_kwargs.append({k: v for k, v in kwargs.items() if v is not NOT_GIVEN})

        if stream:
            assert self.stream is not None, 'you can only used `stream=True` if `stream` is provided'
            if isinstance(self.stream[0], Sequence):
                response = MockAsyncStream(iter(cast(list[MockResponseStreamEvent], self.stream[self.index])))
            else:
                response = MockAsyncStream(iter(cast(list[MockResponseStreamEvent], self.stream)))
        else:
            assert self.response is not None, 'you can only used `stream=False` if `response` are provided'
            if isinstance(self.response, Sequence):
                raise_if_exception(self.response[self.index])
                response = cast(responses.Response, self.response[self.index])
            else:
                raise_if_exception(self.response)
                response = cast(responses.Response, self.response)
        self.index += 1
        return response


def get_mock_responses_kwargs(async_open_ai: AsyncOpenAI) -> list[dict[str, Any]]:
    if isinstance(async_open_ai, MockOpenAIResponses):  # pragma: lax no cover
        return async_open_ai.response_kwargs
    else:  # pragma: no cover
        raise RuntimeError('Not a MockOpenAIResponses instance')


def response_message(
    output_items: Sequence[ResponseOutputItem], *, usage: ResponseUsage | None = None
) -> responses.Response:
    return responses.Response(
        id='123',
        model='gpt-4o-123',
        object='response',
        created_at=1704067200,  # 2024-01-01
        output=list(output_items),
        parallel_tool_calls=True,
        tool_choice='auto',
        tools=[],
        usage=usage,
    )
