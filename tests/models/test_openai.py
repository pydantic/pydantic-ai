from __future__ import annotations as _annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from typing import Any, Literal, cast

import pytest
from inline_snapshot import snapshot
from openai import AsyncOpenAI
from openai.types import chat
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import (
    Choice as ChunkChoice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.completion_usage import CompletionUsage, PromptTokensDetails
from typing_extensions import TypedDict

from pydantic_ai import Agent, ModelRetry, UnexpectedModelBehavior, _utils
from pydantic_ai.messages import (
    ArgsJson,
    ModelStructuredResponse,
    ModelTextResponse,
    RetryPrompt,
    SystemPrompt,
    ToolCall,
    ToolReturn,
    UserPrompt,
)
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.result import Cost
from tests.conftest import IsNow

pytestmark = pytest.mark.anyio


def test_init():
    m = OpenAIModel('gpt-4', api_key='foobar')
    assert m.client.api_key == 'foobar'
    assert m.name() == 'openai:gpt-4'


@dataclass
class MockAsyncStream:
    _iter: Iterator[chat.ChatCompletionChunk]

    async def __anext__(self) -> chat.ChatCompletionChunk:
        return _utils.sync_anext(self._iter)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args: Any):
        pass


@dataclass
class MockOpenAI:
    completions: chat.ChatCompletion | list[chat.ChatCompletion] | None = None
    stream: list[chat.ChatCompletionChunk] | list[list[chat.ChatCompletionChunk]] | None = None
    index = 0

    @cached_property
    def chat(self) -> Any:
        chat_completions = type('Completions', (), {'create': self.chat_completions_create})
        return type('Chat', (), {'completions': chat_completions})

    @classmethod
    def create_mock(cls, completions: chat.ChatCompletion | list[chat.ChatCompletion]) -> AsyncOpenAI:
        return cast(AsyncOpenAI, cls(completions=completions))

    @classmethod
    def create_mock_stream(
        cls, stream: Sequence[chat.ChatCompletionChunk] | Sequence[list[chat.ChatCompletionChunk]]
    ) -> AsyncOpenAI:
        return cast(AsyncOpenAI, cls(stream=list(stream)))  # pyright: ignore[reportArgumentType]

    async def chat_completions_create(
        self, *_args: Any, stream: bool = False, **_kwargs: Any
    ) -> chat.ChatCompletion | MockAsyncStream:
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


def completion_message(message: ChatCompletionMessage, *, usage: CompletionUsage | None = None) -> chat.ChatCompletion:
    return chat.ChatCompletion(
        id='123',
        choices=[Choice(finish_reason='stop', index=0, message=message)],
        created=1704067200,  # 2024-01-01
        model='gpt-4',
        object='chat.completion',
        usage=usage,
    )


async def test_request_simple_success(allow_model_requests: None):
    c = completion_message(ChatCompletionMessage(content='world', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    m = OpenAIModel('gpt-4', openai_client=mock_client)
    agent = Agent(m)

    result = await agent.run('hello')
    assert result.data == 'world'
    assert result.cost() == snapshot(Cost())

    # reset the index so we get the same response again
    mock_client.index = 0  # type: ignore

    result = await agent.run('hello', message_history=result.new_messages())
    assert result.data == 'world'
    assert result.cost() == snapshot(Cost())
    assert result.all_messages() == snapshot(
        [
            UserPrompt(content='hello', timestamp=IsNow(tz=timezone.utc)),
            ModelTextResponse(content='world', timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)),
            UserPrompt(content='hello', timestamp=IsNow(tz=timezone.utc)),
            ModelTextResponse(content='world', timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)),
        ]
    )


async def test_request_simple_usage(allow_model_requests: None):
    c = completion_message(
        ChatCompletionMessage(content='world', role='assistant'),
        usage=CompletionUsage(completion_tokens=1, prompt_tokens=2, total_tokens=3),
    )
    mock_client = MockOpenAI.create_mock(c)
    m = OpenAIModel('gpt-4', openai_client=mock_client)
    agent = Agent(m)

    result = await agent.run('Hello')
    assert result.data == 'world'
    assert result.cost() == snapshot(Cost(request_tokens=2, response_tokens=1, total_tokens=3))


async def test_request_structured_response(allow_model_requests: None):
    c = completion_message(
        ChatCompletionMessage(
            content=None,
            role='assistant',
            tool_calls=[
                chat.ChatCompletionMessageToolCall(
                    id='123',
                    function=Function(arguments='{"response": [1, 2, 123]}', name='final_result'),
                    type='function',
                )
            ],
        )
    )
    mock_client = MockOpenAI.create_mock(c)
    m = OpenAIModel('gpt-4', openai_client=mock_client)
    agent = Agent(m, result_type=list[int])

    result = await agent.run('Hello')
    assert result.data == [1, 2, 123]
    assert result.all_messages() == snapshot(
        [
            UserPrompt(content='Hello', timestamp=IsNow(tz=timezone.utc)),
            ModelStructuredResponse(
                calls=[
                    ToolCall(
                        tool_name='final_result',
                        args=ArgsJson(args_json='{"response": [1, 2, 123]}'),
                        tool_id='123',
                    )
                ],
                timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ),
        ]
    )


async def test_request_tool_call(allow_model_requests: None):
    responses = [
        completion_message(
            ChatCompletionMessage(
                content=None,
                role='assistant',
                tool_calls=[
                    chat.ChatCompletionMessageToolCall(
                        id='1',
                        function=Function(arguments='{"loc_name": "San Fransisco"}', name='get_location'),
                        type='function',
                    )
                ],
            ),
            usage=CompletionUsage(
                completion_tokens=1,
                prompt_tokens=2,
                total_tokens=3,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=1),
            ),
        ),
        completion_message(
            ChatCompletionMessage(
                content=None,
                role='assistant',
                tool_calls=[
                    chat.ChatCompletionMessageToolCall(
                        id='2',
                        function=Function(arguments='{"loc_name": "London"}', name='get_location'),
                        type='function',
                    )
                ],
            ),
            usage=CompletionUsage(
                completion_tokens=2,
                prompt_tokens=3,
                total_tokens=6,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=2),
            ),
        ),
        completion_message(ChatCompletionMessage(content='final response', role='assistant')),
    ]
    mock_client = MockOpenAI.create_mock(responses)
    m = OpenAIModel('gpt-4', openai_client=mock_client)
    agent = Agent(m, system_prompt='this is the system prompt')

    @agent.retriever_plain
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
                        tool_id='1',
                    )
                ],
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
            ),
            RetryPrompt(
                tool_name='get_location',
                content='Wrong location, please try again',
                tool_id='1',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelStructuredResponse(
                calls=[
                    ToolCall(
                        tool_name='get_location',
                        args=ArgsJson(args_json='{"loc_name": "London"}'),
                        tool_id='2',
                    )
                ],
                timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
            ),
            ToolReturn(
                tool_name='get_location',
                content='{"lat": 51, "lng": 0}',
                tool_id='2',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelTextResponse(content='final response', timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)),
        ]
    )
    assert result.cost() == snapshot(
        Cost(request_tokens=5, response_tokens=3, total_tokens=9, details={'cached_tokens': 3})
    )


FinishReason = Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call']


def chunk(delta: list[ChoiceDelta], finish_reason: FinishReason | None = None) -> chat.ChatCompletionChunk:
    return chat.ChatCompletionChunk(
        id='x',
        choices=[
            ChunkChoice(index=index, delta=delta, finish_reason=finish_reason) for index, delta in enumerate(delta)
        ],
        created=1704067200,  # 2024-01-01
        model='gpt-4',
        object='chat.completion.chunk',
        usage=CompletionUsage(completion_tokens=1, prompt_tokens=2, total_tokens=3),
    )


def text_chunk(text: str, finish_reason: FinishReason | None = None) -> chat.ChatCompletionChunk:
    return chunk([ChoiceDelta(content=text, role='assistant')], finish_reason=finish_reason)


async def test_stream_text(allow_model_requests: None):
    stream = text_chunk('hello '), text_chunk('world'), chunk([])
    mock_client = MockOpenAI.create_mock_stream(stream)
    m = OpenAIModel('gpt-4', openai_client=mock_client)
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert not result.is_structured
        assert not result.is_complete
        assert [c async for c in result.stream(debounce_by=None)] == snapshot(['hello ', 'hello world'])
        assert result.is_complete
        assert result.cost() == snapshot(Cost(request_tokens=6, response_tokens=3, total_tokens=9))


async def test_stream_text_finish_reason(allow_model_requests: None):
    stream = text_chunk('hello '), text_chunk('world'), text_chunk('.', finish_reason='stop')
    mock_client = MockOpenAI.create_mock_stream(stream)
    m = OpenAIModel('gpt-4', openai_client=mock_client)
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert not result.is_structured
        assert not result.is_complete
        assert [c async for c in result.stream(debounce_by=None)] == snapshot(['hello ', 'hello world', 'hello world.'])
        assert result.is_complete


def struc_chunk(
    tool_name: str | None, tool_arguments: str | None, finish_reason: FinishReason | None = None
) -> chat.ChatCompletionChunk:
    return chunk(
        [
            ChoiceDelta(
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=0, function=ChoiceDeltaToolCallFunction(name=tool_name, arguments=tool_arguments)
                    )
                ]
            ),
        ],
        finish_reason=finish_reason,
    )


class MyTypedDict(TypedDict, total=False):
    first: str
    second: str


async def test_stream_structured(allow_model_requests: None):
    stream = (
        chunk([ChoiceDelta()]),
        chunk([ChoiceDelta(tool_calls=[])]),
        chunk([ChoiceDelta(tool_calls=[ChoiceDeltaToolCall(index=0, function=None)])]),
        chunk([ChoiceDelta(tool_calls=[ChoiceDeltaToolCall(index=0, function=None)])]),
        struc_chunk('final_result', None),
        chunk([ChoiceDelta(tool_calls=[ChoiceDeltaToolCall(index=0, function=None)])]),
        struc_chunk(None, '{"first": "One'),
        struc_chunk(None, '", "second": "Two"'),
        struc_chunk(None, '}'),
        chunk([]),
    )
    mock_client = MockOpenAI.create_mock_stream(stream)
    m = OpenAIModel('gpt-4', openai_client=mock_client)
    agent = Agent(m, result_type=MyTypedDict)

    async with agent.run_stream('') as result:
        assert result.is_structured
        assert not result.is_complete
        assert [dict(c) async for c in result.stream(debounce_by=None)] == snapshot(
            [
                {'first': 'One'},
                {'first': 'One', 'second': 'Two'},
                {'first': 'One', 'second': 'Two'},
                {'first': 'One', 'second': 'Two'},
            ]
        )
        assert result.is_complete
        assert result.cost() == snapshot(Cost(request_tokens=20, response_tokens=10, total_tokens=30))
        # double check cost matches stream count
        assert result.cost().response_tokens == len(stream)


async def test_stream_structured_finish_reason(allow_model_requests: None):
    stream = (
        struc_chunk('final_result', None),
        struc_chunk(None, '{"first": "One'),
        struc_chunk(None, '", "second": "Two"'),
        struc_chunk(None, '}'),
        struc_chunk(None, None, finish_reason='stop'),
    )
    mock_client = MockOpenAI.create_mock_stream(stream)
    m = OpenAIModel('gpt-4', openai_client=mock_client)
    agent = Agent(m, result_type=MyTypedDict)

    async with agent.run_stream('') as result:
        assert result.is_structured
        assert not result.is_complete
        assert [dict(c) async for c in result.stream(debounce_by=None)] == snapshot(
            [
                {'first': 'One'},
                {'first': 'One', 'second': 'Two'},
                {'first': 'One', 'second': 'Two'},
                {'first': 'One', 'second': 'Two'},
            ]
        )
        assert result.is_complete


async def test_no_content(allow_model_requests: None):
    stream = chunk([ChoiceDelta()]), chunk([ChoiceDelta()])
    mock_client = MockOpenAI.create_mock_stream(stream)
    m = OpenAIModel('gpt-4', openai_client=mock_client)
    agent = Agent(m, result_type=MyTypedDict)

    with pytest.raises(UnexpectedModelBehavior, match='Streamed response ended without con'):
        async with agent.run_stream(''):
            pass
