from __future__ import annotations as _annotations

import json
from dataclasses import dataclass
from datetime import timezone
from functools import cached_property
from typing import Any, cast

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, ModelRetry
from pydantic_ai.messages import (
    ArgsDict,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.result import Usage

from ..conftest import IsNow, try_import

with try_import() as imports_successful:
    from anthropic import AsyncAnthropic
    from anthropic.types import (
        ContentBlock,
        Message as AnthropicMessage,
        TextBlock,
        ToolUseBlock,
        Usage as AnthropicUsage,
    )

    from pydantic_ai.models.anthropic import AnthropicModel

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    pytest.mark.anyio,
]


def test_init():
    m = AnthropicModel('claude-3-5-haiku-latest', api_key='foobar')
    assert m.client.api_key == 'foobar'
    assert m.name() == 'claude-3-5-haiku-latest'


@dataclass
class MockAnthropic:
    messages_: AnthropicMessage | list[AnthropicMessage] | None = None
    index = 0

    @cached_property
    def messages(self) -> Any:
        return type('Messages', (), {'create': self.messages_create})

    @classmethod
    def create_mock(cls, messages_: AnthropicMessage | list[AnthropicMessage]) -> AsyncAnthropic:
        return cast(AsyncAnthropic, cls(messages_=messages_))

    async def messages_create(self, *_args: Any, **_kwargs: Any) -> AnthropicMessage:
        assert self.messages_ is not None, '`messages` must be provided'
        if isinstance(self.messages_, list):
            response = self.messages_[self.index]
        else:
            response = self.messages_
        self.index += 1
        return response


def completion_message(content: list[ContentBlock], usage: AnthropicUsage) -> AnthropicMessage:
    return AnthropicMessage(
        id='123',
        content=content,
        model='claude-3-5-haiku-latest',
        role='assistant',
        stop_reason='end_turn',
        type='message',
        usage=usage,
    )


async def test_sync_request_text_response(allow_model_requests: None):
    c = completion_message([TextBlock(text='world', type='text')], AnthropicUsage(input_tokens=5, output_tokens=10))
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-3-5-haiku-latest', anthropic_client=mock_client)
    agent = Agent(m)

    result = await agent.run('hello')
    assert result.data == 'world'
    assert result.usage() == snapshot(Usage(requests=1, request_tokens=5, response_tokens=10, total_tokens=15))

    # reset the index so we get the same response again
    mock_client.index = 0  # type: ignore

    result = await agent.run('hello', message_history=result.new_messages())
    assert result.data == 'world'
    assert result.usage() == snapshot(Usage(requests=1, request_tokens=5, response_tokens=10, total_tokens=15))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse.from_text(content='world', timestamp=IsNow(tz=timezone.utc)),
            ModelRequest(parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse.from_text(content='world', timestamp=IsNow(tz=timezone.utc)),
        ]
    )


async def test_async_request_text_response(allow_model_requests: None):
    c = completion_message(
        [TextBlock(text='world', type='text')],
        usage=AnthropicUsage(input_tokens=3, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-3-5-haiku-latest', anthropic_client=mock_client)
    agent = Agent(m)

    result = await agent.run('hello')
    assert result.data == 'world'
    assert result.usage() == snapshot(Usage(requests=1, request_tokens=3, response_tokens=5, total_tokens=8))


async def test_request_structured_response(allow_model_requests: None):
    c = completion_message(
        [ToolUseBlock(id='123', input={'response': [1, 2, 3]}, name='final_result', type='tool_use')],
        usage=AnthropicUsage(input_tokens=3, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-3-5-haiku-latest', anthropic_client=mock_client)
    agent = Agent(m, result_type=list[int])

    result = await agent.run('hello')
    assert result.data == [1, 2, 3]
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args=ArgsDict(args_dict={'response': [1, 2, 3]}),
                        tool_call_id='123',
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
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


async def test_request_tool_call(allow_model_requests: None):
    responses = [
        completion_message(
            [ToolUseBlock(id='1', input={'loc_name': 'San Francisco'}, name='get_location', type='tool_use')],
            usage=AnthropicUsage(input_tokens=2, output_tokens=1),
        ),
        completion_message(
            [ToolUseBlock(id='2', input={'loc_name': 'London'}, name='get_location', type='tool_use')],
            usage=AnthropicUsage(input_tokens=3, output_tokens=2),
        ),
        completion_message(
            [TextBlock(text='final response', type='text')],
            usage=AnthropicUsage(input_tokens=3, output_tokens=5),
        ),
    ]

    mock_client = MockAnthropic.create_mock(responses)
    m = AnthropicModel('claude-3-5-haiku-latest', anthropic_client=mock_client)
    agent = Agent(m, system_prompt='this is the system prompt')

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        if loc_name == 'London':
            return json.dumps({'lat': 51, 'lng': 0})
        else:
            raise ModelRetry('Wrong location, please try again')

    result = await agent.run('hello')
    assert result.data == 'final response'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='this is the system prompt'),
                    UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc)),
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_location',
                        args=ArgsDict(args_dict={'loc_name': 'San Francisco'}),
                        tool_call_id='1',
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
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
                        args=ArgsDict(args_dict={'loc_name': 'London'}),
                        tool_call_id='2',
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
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
