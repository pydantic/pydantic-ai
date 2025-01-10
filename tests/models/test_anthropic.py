from __future__ import annotations as _annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import timezone
from functools import cached_property
from typing import Any, TypeVar, cast

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
    from anthropic import AsyncAnthropic, AsyncStream
    from anthropic.types import (
        ContentBlock,
        InputJSONDelta,
        Message as AnthropicMessage,
        MessageDeltaUsage,
        RawContentBlockDeltaEvent,
        RawContentBlockStartEvent,
        RawContentBlockStopEvent,
        RawMessageDeltaEvent,
        RawMessageStartEvent,
        RawMessageStopEvent,
        RawMessageStreamEvent,
        TextBlock,
        ToolUseBlock,
        Usage as AnthropicUsage,
    )
    from anthropic.types.raw_message_delta_event import Delta

    from pydantic_ai.models.anthropic import AnthropicModel

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    pytest.mark.anyio,
]

# Type variable for generic AsyncStream
T = TypeVar('T')


def test_init():
    m = AnthropicModel('claude-3-5-haiku-latest', api_key='foobar')
    assert m.client.api_key == 'foobar'
    assert m.name() == 'anthropic:claude-3-5-haiku-latest'


if imports_successful():

    class MockAsyncStream(AsyncStream[T]):
        """Mock implementation of AsyncStream for testing."""

        def __init__(self, events: list[list[T]]):
            self.events = events
            self.stream_index = 0

        def __aiter__(self) -> AsyncIterator[T]:
            if self.stream_index >= len(self.events):
                raise StopAsyncIteration

            async def iterator() -> AsyncIterator[T]:
                current_stream = self.events[self.stream_index]
                for event in current_stream:
                    yield event
                self.stream_index += 1

            return iterator()

        async def __anext__(self) -> T:
            return await self._iterator.__anext__()

        async def __aenter__(self) -> MockAsyncStream[T]:
            return self

        async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            pass


@dataclass
class MockAnthropic:
    messages_: AnthropicMessage | list[AnthropicMessage] | AsyncStream[RawMessageStreamEvent] | None = None
    index = 0

    @cached_property
    def messages(self) -> Any:
        return type('Messages', (), {'create': self.messages_create})

    @classmethod
    def create_mock(
        cls, messages_: AnthropicMessage | list[AnthropicMessage] | AsyncStream[RawMessageStreamEvent]
    ) -> AsyncAnthropic:
        return cast(AsyncAnthropic, cls(messages_=messages_))

    async def messages_create(
        self, *_args: Any, stream: bool = False, **_kwargs: Any
    ) -> AnthropicMessage | AsyncStream[RawMessageStreamEvent]:
        assert self.messages_ is not None, '`messages` must be provided'
        if isinstance(self.messages_, AsyncStream):
            assert stream, 'stream must be True when using AsyncStream'
            return self.messages_
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


async def test_stream_structured(allow_model_requests: None):
    """Test streaming structured responses with Anthropic's API.

    This test simulates how Anthropic streams tool calls:
    1. Message start
    2. Tool block start with initial data
    3. Tool block delta with additional data
    4. Tool block stop
    5. Update usage
    6. Message stop
    """
    stream: list[RawMessageStreamEvent] = [
        RawMessageStartEvent(
            type='message_start',
            message=AnthropicMessage(
                id='msg_123',
                model='claude-3-5-haiku-latest',
                role='assistant',
                type='message',
                content=[],
                stop_reason=None,
                usage=AnthropicUsage(input_tokens=20, output_tokens=0),
            ),
        ),
        # Start tool block with initial data
        RawContentBlockStartEvent(
            type='content_block_start',
            index=0,
            content_block=ToolUseBlock(type='tool_use', id='tool_1', name='my_tool', input={'first': 'One'}),
        ),
        # Add more data through an incomplete JSON delta
        RawContentBlockDeltaEvent(
            type='content_block_delta',
            index=0,
            delta=InputJSONDelta(type='input_json_delta', partial_json='{"second":'),
        ),
        RawContentBlockDeltaEvent(
            type='content_block_delta',
            index=0,
            delta=InputJSONDelta(type='input_json_delta', partial_json='"Two"}'),
        ),
        # Mark tool block as complete
        RawContentBlockStopEvent(type='content_block_stop', index=0),
        # Update the top-level message with usage
        RawMessageDeltaEvent(
            type='message_delta',
            delta=Delta(
                stop_reason='end_turn',
            ),
            usage=MessageDeltaUsage(
                output_tokens=5,
            ),
        ),
        # Mark message as complete
        RawMessageStopEvent(type='message_stop'),
    ]

    done_stream: list[RawMessageStreamEvent] = [
        RawMessageStartEvent(
            type='message_start',
            message=AnthropicMessage(
                id='msg_123',
                model='claude-3-5-haiku-latest',
                role='assistant',
                type='message',
                content=[],
                stop_reason=None,
                usage=AnthropicUsage(input_tokens=0, output_tokens=0),
            ),
        ),
        # Text block with final data
        RawContentBlockStartEvent(
            type='content_block_start',
            index=0,
            content_block=TextBlock(type='text', text='FINAL_PAYLOAD'),
        ),
        RawContentBlockStopEvent(type='content_block_stop', index=0),
        RawMessageStopEvent(type='message_stop'),
    ]

    mock_client = MockAnthropic.create_mock(MockAsyncStream([stream, done_stream]))
    m = AnthropicModel('claude-3-5-haiku-latest', anthropic_client=mock_client)
    agent = Agent(m)

    tool_called = False

    @agent.tool_plain
    async def my_tool(first: str, second: str) -> int:
        nonlocal tool_called
        tool_called = True
        return len(first) + len(second)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        chunks = [c async for c in result.stream(debounce_by=None)]

        # The tool output doesn't echo any content to the stream, so we only get the final payload once when
        # the block starts and once when it ends.
        assert chunks == snapshot(
            [
                'FINAL_PAYLOAD',
                'FINAL_PAYLOAD',
            ]
        )
        assert result.is_complete
        assert result.usage() == snapshot(Usage(requests=2, request_tokens=20, response_tokens=5, total_tokens=25))
        assert tool_called
