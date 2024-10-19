from __future__ import annotations as _annotations

import datetime
import json
from typing import Any, cast

import pytest
from inline_snapshot import snapshot
from openai import AsyncOpenAI
from openai.types import chat
from openai.types.chat.chat_completion import ChatCompletionMessage, Choice  # pyright: ignore[reportPrivateImportUsage]
from openai.types.chat.chat_completion_message_tool_call import Function

from pydantic_ai import Agent, ModelRetry
from pydantic_ai.messages import (
    ArgsJson,
    LLMResponse,
    LLMToolCalls,
    SystemPrompt,
    ToolCall,
    ToolRetry,
    ToolReturn,
    UserPrompt,
)
from pydantic_ai.models.openai import OpenAIModel
from tests.conftest import IsNow

pytestmark = pytest.mark.anyio


def test_init():
    m = OpenAIModel('gpt-4', api_key='foobar')
    assert m.client.api_key == 'foobar'
    assert m.name() == 'openai:gpt-4'


class MockOpenAI:
    def __init__(self, completions: chat.ChatCompletion | list[chat.ChatCompletion]):
        self.completions = completions
        self.index = 0
        chat_completions = type('Completions', (), {'create': self.chat_completions_create})
        self.chat = type('Chat', (), {'completions': chat_completions})

    @classmethod
    def create_mock(cls, completions: chat.ChatCompletion | list[chat.ChatCompletion]) -> AsyncOpenAI:
        return cast(AsyncOpenAI, cls(completions))

    async def chat_completions_create(self, *_args: Any, **_kwargs: Any) -> chat.ChatCompletion:
        if isinstance(self.completions, list):
            completion = self.completions[self.index]
        else:
            completion = self.completions
        self.index += 1
        return completion


def completion_message(message: ChatCompletionMessage) -> chat.ChatCompletion:
    return chat.ChatCompletion(
        id='123',
        choices=[Choice(finish_reason='stop', index=0, message=message)],
        created=1704067200,
        model='gpt-4',
        object='chat.completion',
    )


async def test_request_simple_success():
    c = completion_message(ChatCompletionMessage(content='world', role='assistant'))
    mock_client = MockOpenAI.create_mock(c)
    m = OpenAIModel('gpt-4', openai_client=mock_client)
    agent = Agent(m, deps=None)

    result = await agent.run('Hello')
    assert result.response == 'world'


async def test_request_structured_response():
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
    agent = Agent(m, deps=None, result_type=list[int])

    result = await agent.run('Hello')
    assert result.response == [1, 2, 123]
    assert result.message_history == snapshot(
        [
            UserPrompt(content='Hello', timestamp=IsNow()),
            LLMToolCalls(
                calls=[
                    ToolCall(
                        tool_name='final_result',
                        args=ArgsJson(args_json='{"response": [1, 2, 123]}'),
                        tool_id='123',
                    )
                ],
                timestamp=datetime.datetime(2024, 1, 1),
            ),
        ]
    )


async def test_request_tool_call():
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
            )
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
            )
        ),
        completion_message(ChatCompletionMessage(content='final response', role='assistant')),
    ]
    mock_client = MockOpenAI.create_mock(responses)
    m = OpenAIModel('gpt-4', openai_client=mock_client)
    agent = Agent(m, deps=None, system_prompt='this is the system prompt')

    @agent.retriever_plain
    async def get_location(loc_name: str) -> str:
        if loc_name == 'London':
            return json.dumps({'lat': 51, 'lng': 0})
        else:
            raise ModelRetry('Wrong location, please try again')

    result = await agent.run('Hello')
    assert result.response == 'final response'
    assert result.message_history == snapshot(
        [
            SystemPrompt(content='this is the system prompt'),
            UserPrompt(content='Hello', timestamp=IsNow()),
            LLMToolCalls(
                calls=[
                    ToolCall(
                        tool_name='get_location',
                        args=ArgsJson(args_json='{"loc_name": "San Fransisco"}'),
                        tool_id='1',
                    )
                ],
                timestamp=datetime.datetime(2024, 1, 1, 0, 0),
            ),
            ToolRetry(
                tool_name='get_location', content='Wrong location, please try again', tool_id='1', timestamp=IsNow()
            ),
            LLMToolCalls(
                calls=[
                    ToolCall(
                        tool_name='get_location',
                        args=ArgsJson(args_json='{"loc_name": "London"}'),
                        tool_id='2',
                    )
                ],
                timestamp=datetime.datetime(2024, 1, 1, 0, 0),
            ),
            ToolReturn(
                tool_name='get_location',
                content='{"lat": 51, "lng": 0}',
                tool_id='2',
                timestamp=IsNow(),
            ),
            LLMResponse(content='final response', timestamp=datetime.datetime(2024, 1, 1, 0, 0)),
        ]
    )
