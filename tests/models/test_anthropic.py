from __future__ import annotations as _annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timezone
from functools import cached_property
from typing import Any, Callable, TypeVar, Union, cast

import httpx
import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import Agent, ModelHTTPError, ModelRetry
from pydantic_ai.builtin_tools import CodeExecutionTool, WebSearchTool
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    DocumentUrl,
    FinalResultEvent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.output import NativeOutput, PromptedOutput, TextOutput, ToolOutput
from pydantic_ai.result import Usage
from pydantic_ai.settings import ModelSettings

from ..conftest import IsDatetime, IsInstance, IsNow, IsStr, TestEnv, raise_if_exception, try_import
from ..parts_from_messages import part_types_from_messages
from .mock_async_stream import MockAsyncStream

with try_import() as imports_successful:
    from anthropic import NOT_GIVEN, APIStatusError, AsyncAnthropic
    from anthropic.resources.beta import AsyncBeta
    from anthropic.types.beta import (
        BetaCodeExecutionResultBlock,
        BetaCodeExecutionToolResultBlock,
        BetaContentBlock,
        BetaInputJSONDelta,
        BetaMessage,
        BetaMessageDeltaUsage,
        BetaRawContentBlockDeltaEvent,
        BetaRawContentBlockStartEvent,
        BetaRawContentBlockStopEvent,
        BetaRawMessageDeltaEvent,
        BetaRawMessageStartEvent,
        BetaRawMessageStopEvent,
        BetaRawMessageStreamEvent,
        BetaServerToolUseBlock,
        BetaTextBlock,
        BetaToolUseBlock,
        BetaUsage,
        BetaWebSearchResultBlock,
        BetaWebSearchToolResultBlock,
    )
    from anthropic.types.beta.beta_raw_message_delta_event import Delta

    from pydantic_ai.models.anthropic import (
        AnthropicModel,
        AnthropicModelSettings,
        _map_usage,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.providers.anthropic import AnthropicProvider

    # note: we use Union here so that casting works with Python 3.9
    MockAnthropicMessage = Union[BetaMessage, Exception]
    MockRawMessageStreamEvent = Union[BetaRawMessageStreamEvent, Exception]

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]

# Type variable for generic AsyncStream
T = TypeVar('T')


def test_init():
    m = AnthropicModel('claude-3-5-haiku-latest', provider=AnthropicProvider(api_key='foobar'))
    assert m.client.api_key == 'foobar'
    assert m.model_name == 'claude-3-5-haiku-latest'
    assert m.system == 'anthropic'
    assert m.base_url == 'https://api.anthropic.com'


@dataclass
class MockAnthropic:
    messages_: MockAnthropicMessage | Sequence[MockAnthropicMessage] | None = None
    stream: Sequence[MockRawMessageStreamEvent] | Sequence[Sequence[MockRawMessageStreamEvent]] | None = None
    index = 0
    chat_completion_kwargs: list[dict[str, Any]] = field(default_factory=list)
    base_url: str | None = None

    @cached_property
    def beta(self) -> AsyncBeta:
        return cast(AsyncBeta, self)

    @cached_property
    def messages(self) -> Any:
        return type('Messages', (), {'create': self.messages_create})

    @classmethod
    def create_mock(cls, messages_: MockAnthropicMessage | Sequence[MockAnthropicMessage]) -> AsyncAnthropic:
        return cast(AsyncAnthropic, cls(messages_=messages_))

    @classmethod
    def create_stream_mock(
        cls, stream: Sequence[MockRawMessageStreamEvent] | Sequence[Sequence[MockRawMessageStreamEvent]]
    ) -> AsyncAnthropic:
        return cast(AsyncAnthropic, cls(stream=stream))

    async def messages_create(
        self, *_args: Any, stream: bool = False, **kwargs: Any
    ) -> BetaMessage | MockAsyncStream[MockRawMessageStreamEvent]:
        self.chat_completion_kwargs.append({k: v for k, v in kwargs.items() if v is not NOT_GIVEN})

        if stream:
            assert self.stream is not None, 'you can only use `stream=True` if `stream` is provided'
            if isinstance(self.stream[0], Sequence):
                response = MockAsyncStream(iter(cast(list[MockRawMessageStreamEvent], self.stream[self.index])))
            else:
                response = MockAsyncStream(  # pragma: no cover
                    iter(cast(list[MockRawMessageStreamEvent], self.stream))
                )
        else:
            assert self.messages_ is not None, '`messages` must be provided'
            if isinstance(self.messages_, Sequence):
                raise_if_exception(self.messages_[self.index])
                response = cast(BetaMessage, self.messages_[self.index])
            else:
                raise_if_exception(self.messages_)
                response = cast(BetaMessage, self.messages_)
        self.index += 1
        return response


def completion_message(content: list[BetaContentBlock], usage: BetaUsage) -> BetaMessage:
    return BetaMessage(
        id='123',
        content=content,
        model='claude-3-5-haiku-123',
        role='assistant',
        stop_reason='end_turn',
        type='message',
        usage=usage,
    )


async def test_sync_request_text_response(allow_model_requests: None):
    c = completion_message([BetaTextBlock(text='world', type='text')], BetaUsage(input_tokens=5, output_tokens=10))
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-3-5-haiku-latest', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    result = await agent.run('hello')
    assert result.output == 'world'
    assert result.usage() == snapshot(
        Usage(
            requests=1,
            request_tokens=5,
            response_tokens=10,
            total_tokens=15,
            details={'input_tokens': 5, 'output_tokens': 10},
        )
    )
    # reset the index so we get the same response again
    mock_client.index = 0  # type: ignore

    result = await agent.run('hello', message_history=result.new_messages())
    assert result.output == 'world'
    assert result.usage() == snapshot(
        Usage(
            requests=1,
            request_tokens=5,
            response_tokens=10,
            total_tokens=15,
            details={'input_tokens': 5, 'output_tokens': 10},
        )
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[TextPart(content='world')],
                usage=Usage(
                    requests=1,
                    request_tokens=5,
                    response_tokens=10,
                    total_tokens=15,
                    details={'input_tokens': 5, 'output_tokens': 10},
                ),
                model_name='claude-3-5-haiku-123',
                timestamp=IsNow(tz=timezone.utc),
                vendor_id='123',
            ),
            ModelRequest(parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[TextPart(content='world')],
                usage=Usage(
                    requests=1,
                    request_tokens=5,
                    response_tokens=10,
                    total_tokens=15,
                    details={'input_tokens': 5, 'output_tokens': 10},
                ),
                model_name='claude-3-5-haiku-123',
                timestamp=IsNow(tz=timezone.utc),
                vendor_id='123',
            ),
        ]
    )


async def test_async_request_prompt_caching(allow_model_requests: None):
    c = completion_message(
        [BetaTextBlock(text='world', type='text')],
        usage=BetaUsage(
            input_tokens=3,
            output_tokens=5,
            cache_creation_input_tokens=4,
            cache_read_input_tokens=6,
        ),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-3-5-haiku-latest', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    result = await agent.run('hello')
    assert result.output == 'world'
    assert result.usage() == snapshot(
        Usage(
            requests=1,
            request_tokens=13,
            response_tokens=5,
            total_tokens=18,
            details={
                'input_tokens': 3,
                'output_tokens': 5,
                'cache_creation_input_tokens': 4,
                'cache_read_input_tokens': 6,
            },
        )
    )


async def test_async_request_text_response(allow_model_requests: None):
    c = completion_message(
        [BetaTextBlock(text='world', type='text')],
        usage=BetaUsage(input_tokens=3, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-3-5-haiku-latest', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    result = await agent.run('hello')
    assert result.output == 'world'
    assert result.usage() == snapshot(
        Usage(
            requests=1,
            request_tokens=3,
            response_tokens=5,
            total_tokens=8,
            details={'input_tokens': 3, 'output_tokens': 5},
        )
    )


async def test_request_structured_response(allow_model_requests: None):
    c = completion_message(
        [BetaToolUseBlock(id='123', input={'response': [1, 2, 3]}, name='final_result', type='tool_use')],
        usage=BetaUsage(input_tokens=3, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-3-5-haiku-latest', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m, output_type=list[int])

    result = await agent.run('hello')
    assert result.output == [1, 2, 3]
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args={'response': [1, 2, 3]},
                        tool_call_id='123',
                    )
                ],
                usage=Usage(
                    requests=1,
                    request_tokens=3,
                    response_tokens=5,
                    total_tokens=8,
                    details={'input_tokens': 3, 'output_tokens': 5},
                ),
                model_name='claude-3-5-haiku-123',
                timestamp=IsNow(tz=timezone.utc),
                vendor_id='123',
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
            [BetaToolUseBlock(id='1', input={'loc_name': 'San Francisco'}, name='get_location', type='tool_use')],
            usage=BetaUsage(input_tokens=2, output_tokens=1),
        ),
        completion_message(
            [BetaToolUseBlock(id='2', input={'loc_name': 'London'}, name='get_location', type='tool_use')],
            usage=BetaUsage(input_tokens=3, output_tokens=2),
        ),
        completion_message(
            [BetaTextBlock(text='final response', type='text')],
            usage=BetaUsage(input_tokens=3, output_tokens=5),
        ),
    ]

    mock_client = MockAnthropic.create_mock(responses)
    m = AnthropicModel('claude-3-5-haiku-latest', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m, system_prompt='this is the system prompt')

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        if loc_name == 'London':
            return json.dumps({'lat': 51, 'lng': 0})
        else:
            raise ModelRetry('Wrong location, please try again')

    result = await agent.run('hello')
    assert result.output == 'final response'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='this is the system prompt', timestamp=IsNow(tz=timezone.utc)),
                    UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc)),
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_location',
                        args={'loc_name': 'San Francisco'},
                        tool_call_id='1',
                    )
                ],
                usage=Usage(
                    requests=1,
                    request_tokens=2,
                    response_tokens=1,
                    total_tokens=3,
                    details={'input_tokens': 2, 'output_tokens': 1},
                ),
                model_name='claude-3-5-haiku-123',
                timestamp=IsNow(tz=timezone.utc),
                vendor_id='123',
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
                        args={'loc_name': 'London'},
                        tool_call_id='2',
                    )
                ],
                usage=Usage(
                    requests=1,
                    request_tokens=3,
                    response_tokens=2,
                    total_tokens=5,
                    details={'input_tokens': 3, 'output_tokens': 2},
                ),
                model_name='claude-3-5-haiku-123',
                timestamp=IsNow(tz=timezone.utc),
                vendor_id='123',
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
                parts=[TextPart(content='final response')],
                usage=Usage(
                    requests=1,
                    request_tokens=3,
                    response_tokens=5,
                    total_tokens=8,
                    details={'input_tokens': 3, 'output_tokens': 5},
                ),
                model_name='claude-3-5-haiku-123',
                timestamp=IsNow(tz=timezone.utc),
                vendor_id='123',
            ),
        ]
    )


def get_mock_chat_completion_kwargs(async_anthropic: AsyncAnthropic) -> list[dict[str, Any]]:
    if isinstance(async_anthropic, MockAnthropic):
        return async_anthropic.chat_completion_kwargs
    else:  # pragma: no cover
        raise RuntimeError('Not a MockOpenAI instance')


@pytest.mark.parametrize('parallel_tool_calls', [True, False])
async def test_parallel_tool_calls(allow_model_requests: None, parallel_tool_calls: bool) -> None:
    responses = [
        completion_message(
            [BetaToolUseBlock(id='1', input={'loc_name': 'San Francisco'}, name='get_location', type='tool_use')],
            usage=BetaUsage(input_tokens=2, output_tokens=1),
        ),
        completion_message(
            [BetaTextBlock(text='final response', type='text')],
            usage=BetaUsage(input_tokens=3, output_tokens=5),
        ),
    ]

    mock_client = MockAnthropic.create_mock(responses)
    m = AnthropicModel('claude-3-5-haiku-latest', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m, model_settings=ModelSettings(parallel_tool_calls=parallel_tool_calls))

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        if loc_name == 'London':
            return json.dumps({'lat': 51, 'lng': 0})  # pragma: no cover
        else:
            raise ModelRetry('Wrong location, please try again')

    await agent.run('hello')
    assert get_mock_chat_completion_kwargs(mock_client)[0]['tool_choice']['disable_parallel_tool_use'] == (
        not parallel_tool_calls
    )


async def test_multiple_parallel_tool_calls(allow_model_requests: None):
    async def retrieve_entity_info(name: str) -> str:
        """Get the knowledge about the given entity."""
        data = {
            'alice': "alice is bob's wife",
            'bob': "bob is alice's husband",
            'charlie': "charlie is alice's son",
            'daisy': "daisy is bob's daughter and charlie's younger sister",
        }
        return data[name.lower()]

    system_prompt = """
    Use the `retrieve_entity_info` tool to get information about a specific person.
    If you need to use `retrieve_entity_info` to get information about multiple people, try
    to call them in parallel as much as possible.
    Think step by step and then provide a single most probable concise answer.
    """

    # If we don't provide some value for the API key, the anthropic SDK will raise an error.
    # However, we do want to use the environment variable if present when rewriting VCR cassettes.
    api_key = os.environ.get('ANTHROPIC_API_KEY', 'mock-value')
    agent = Agent(
        AnthropicModel('claude-3-5-haiku-latest', provider=AnthropicProvider(api_key=api_key)),
        system_prompt=system_prompt,
        tools=[retrieve_entity_info],
    )

    result = await agent.run('Alice, Bob, Charlie and Daisy are a family. Who is the youngest?')
    assert 'Daisy is the youngest' in result.output

    all_messages = result.all_messages()
    first_response = all_messages[1]
    second_request = all_messages[2]
    assert first_response.parts == snapshot(
        [
            TextPart(
                content="I'll help you find out who is the youngest by retrieving information about each family member.",
                part_kind='text',
            ),
            ToolCallPart(
                tool_name='retrieve_entity_info', args={'name': 'Alice'}, tool_call_id=IsStr(), part_kind='tool-call'
            ),
            ToolCallPart(
                tool_name='retrieve_entity_info', args={'name': 'Bob'}, tool_call_id=IsStr(), part_kind='tool-call'
            ),
            ToolCallPart(
                tool_name='retrieve_entity_info', args={'name': 'Charlie'}, tool_call_id=IsStr(), part_kind='tool-call'
            ),
            ToolCallPart(
                tool_name='retrieve_entity_info', args={'name': 'Daisy'}, tool_call_id=IsStr(), part_kind='tool-call'
            ),
        ]
    )
    assert second_request.parts == snapshot(
        [
            ToolReturnPart(
                tool_name='retrieve_entity_info',
                content="alice is bob's wife",
                tool_call_id=IsStr(),
                timestamp=IsDatetime(),
                part_kind='tool-return',
            ),
            ToolReturnPart(
                tool_name='retrieve_entity_info',
                content="bob is alice's husband",
                tool_call_id=IsStr(),
                timestamp=IsDatetime(),
                part_kind='tool-return',
            ),
            ToolReturnPart(
                tool_name='retrieve_entity_info',
                content="charlie is alice's son",
                tool_call_id=IsStr(),
                timestamp=IsDatetime(),
                part_kind='tool-return',
            ),
            ToolReturnPart(
                tool_name='retrieve_entity_info',
                content="daisy is bob's daughter and charlie's younger sister",
                tool_call_id=IsStr(),
                timestamp=IsDatetime(),
                part_kind='tool-return',
            ),
        ]
    )

    # Ensure the tool call IDs match between the tool calls and the tool returns
    tool_call_part_ids = [part.tool_call_id for part in first_response.parts if part.part_kind == 'tool-call']
    tool_return_part_ids = [part.tool_call_id for part in second_request.parts if part.part_kind == 'tool-return']
    assert len(set(tool_call_part_ids)) == 4  # ensure they are all unique
    assert tool_call_part_ids == tool_return_part_ids


async def test_anthropic_specific_metadata(allow_model_requests: None) -> None:
    c = completion_message([BetaTextBlock(text='world', type='text')], BetaUsage(input_tokens=5, output_tokens=10))
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-3-5-haiku-latest', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    result = await agent.run('hello', model_settings=AnthropicModelSettings(anthropic_metadata={'user_id': '123'}))
    assert result.output == 'world'
    assert get_mock_chat_completion_kwargs(mock_client)[0]['metadata']['user_id'] == '123'


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
    stream = [
        BetaRawMessageStartEvent(
            type='message_start',
            message=BetaMessage(
                id='msg_123',
                model='claude-3-5-haiku-latest',
                role='assistant',
                type='message',
                content=[],
                stop_reason=None,
                usage=BetaUsage(input_tokens=20, output_tokens=0),
            ),
        ),
        # Start tool block with initial data
        BetaRawContentBlockStartEvent(
            type='content_block_start',
            index=0,
            content_block=BetaToolUseBlock(type='tool_use', id='tool_1', name='my_tool', input={}),
        ),
        # Add more data through an incomplete JSON delta
        BetaRawContentBlockDeltaEvent(
            type='content_block_delta',
            index=0,
            delta=BetaInputJSONDelta(type='input_json_delta', partial_json='{"first": "One'),
        ),
        BetaRawContentBlockDeltaEvent(
            type='content_block_delta',
            index=0,
            delta=BetaInputJSONDelta(type='input_json_delta', partial_json='", "second": "Two"'),
        ),
        BetaRawContentBlockDeltaEvent(
            type='content_block_delta',
            index=0,
            delta=BetaInputJSONDelta(type='input_json_delta', partial_json='}'),
        ),
        # Mark tool block as complete
        BetaRawContentBlockStopEvent(type='content_block_stop', index=0),
        # Update the top-level message with usage
        BetaRawMessageDeltaEvent(
            type='message_delta',
            delta=Delta(stop_reason='end_turn'),
            usage=BetaMessageDeltaUsage(output_tokens=5),
        ),
        # Mark message as complete
        BetaRawMessageStopEvent(type='message_stop'),
    ]

    done_stream = [
        BetaRawMessageStartEvent(
            type='message_start',
            message=BetaMessage(
                id='msg_123',
                model='claude-3-5-haiku-latest',
                role='assistant',
                type='message',
                content=[],
                stop_reason=None,
                usage=BetaUsage(input_tokens=0, output_tokens=0),
            ),
        ),
        # Text block with final data
        BetaRawContentBlockStartEvent(
            type='content_block_start',
            index=0,
            content_block=BetaTextBlock(type='text', text='FINAL_PAYLOAD'),
        ),
        BetaRawContentBlockStopEvent(type='content_block_stop', index=0),
        BetaRawMessageStopEvent(type='message_stop'),
    ]

    mock_client = MockAnthropic.create_stream_mock([stream, done_stream])
    m = AnthropicModel('claude-3-5-haiku-latest', provider=AnthropicProvider(anthropic_client=mock_client))
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
        assert result.usage() == snapshot(
            Usage(
                requests=2,
                request_tokens=20,
                response_tokens=5,
                total_tokens=25,
                details={'input_tokens': 20, 'output_tokens': 5},
            )
        )
        assert tool_called


async def test_image_url_input(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-3-5-haiku-latest', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m)

    result = await agent.run(
        [
            'What is this vegetable?',
            ImageUrl(url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg'),
        ]
    )
    assert result.output == snapshot(
        "This is a potato. It's a yellow/golden-colored potato with a characteristic oblong, slightly irregular shape and a skin covered in small eyes or indentations. Potatoes are a starchy root vegetable that belongs to the nightshade family and are a staple food in many cuisines around the world. This particular potato looks like it's in good condition, with a smooth, unblemished skin and a uniform yellow color."
    )


async def test_extra_headers(allow_model_requests: None, anthropic_api_key: str):
    # This test doesn't do anything, it's just here to ensure that calls with `extra_headers` don't cause errors, including type.
    m = AnthropicModel('claude-3-5-haiku-latest', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(
        m,
        model_settings=AnthropicModelSettings(
            anthropic_metadata={'user_id': '123'}, extra_headers={'Extra-Header-Key': 'Extra-Header-Value'}
        ),
    )
    await agent.run('hello')


async def test_image_url_input_invalid_mime_type(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-3-5-haiku-latest', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m)

    result = await agent.run(
        [
            'What animal is this?',
            ImageUrl(
                url='https://lh3.googleusercontent.com/proxy/YngsuS8jQJysXxeucAgVBcSgIdwZlSQ-HvsNxGjHS0SrUKXI161bNKh6SOcMsNUGsnxoOrS3AYX--MT4T3S3SoCgSD1xKrtBwwItcgexaX_7W-qHo-VupmYgjjzWO-BuORLp9-pj8Kjr'
            ),
        ]
    )
    assert result.output == snapshot(
        'This is a Great Horned Owl (Bubo virginianus), a large and powerful owl species native to the Americas. The owl is shown sitting on a branch surrounded by yellow-green foliage, with its distinctive mottled gray-brown feathers and prominent ear tufts (often called "horns"). It has striking yellow eyes that are looking directly at the camera, giving it an intense and alert appearance. Great Horned Owls are known for their excellent camouflage, powerful talons, and nocturnal hunting habits. They are widespread across North and South America and are one of the most common and adaptable owl species.'
    )


async def test_image_as_binary_content_tool_response(
    allow_model_requests: None, anthropic_api_key: str, image_content: BinaryContent
):
    m = AnthropicModel('claude-3-5-sonnet-latest', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m)

    @agent.tool_plain
    async def get_image() -> BinaryContent:
        return image_content

    result = await agent.run(['What fruit is in the image you can get from the get_image tool?'])
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=['What fruit is in the image you can get from the get_image tool?'],
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(content='Let me get the image and check.'),
                    ToolCallPart(tool_name='get_image', args={}, tool_call_id='toolu_01YJiJ82nETV7aRdJr9f6Np7'),
                ],
                usage=Usage(
                    requests=1,
                    request_tokens=372,
                    response_tokens=45,
                    total_tokens=417,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 372,
                        'output_tokens': 45,
                    },
                ),
                model_name='claude-3-5-sonnet-20241022',
                timestamp=IsDatetime(),
                vendor_id='msg_01CC59GmUmYXKCV26rHfr32m',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_image',
                        content='See file 1c8566',
                        tool_call_id='toolu_01YJiJ82nETV7aRdJr9f6Np7',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content=[
                            'This is file 1c8566:',
                            image_content,
                        ],
                        timestamp=IsDatetime(),
                    ),
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="The image shows a kiwi fruit that has been cut in half, displaying its characteristic bright green flesh with small black seeds arranged in a circular pattern around a white center core. The fruit's thin, fuzzy brown skin is visible around the edges of the slice."
                    )
                ],
                usage=Usage(
                    requests=1,
                    request_tokens=2021,
                    response_tokens=57,
                    total_tokens=2078,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 2021,
                        'output_tokens': 57,
                    },
                ),
                model_name='claude-3-5-sonnet-20241022',
                timestamp=IsDatetime(),
                vendor_id='msg_014MJqSsWD1pUC23Vvi57LoY',
            ),
        ]
    )


@pytest.mark.parametrize('media_type', ('audio/wav', 'audio/mpeg'))
async def test_audio_as_binary_content_input(allow_model_requests: None, media_type: str):
    c = completion_message([BetaTextBlock(text='world', type='text')], BetaUsage(input_tokens=5, output_tokens=10))
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-3-5-haiku-latest', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    base64_content = b'//uQZ'

    with pytest.raises(RuntimeError, match='Only images and PDFs are supported for binary content'):
        await agent.run(['hello', BinaryContent(data=base64_content, media_type=media_type)])


def test_model_status_error(allow_model_requests: None) -> None:
    mock_client = MockAnthropic.create_mock(
        APIStatusError(
            'test error',
            response=httpx.Response(status_code=500, request=httpx.Request('POST', 'https://example.com/v1')),
            body={'error': 'test error'},
        )
    )
    m = AnthropicModel('claude-3-5-sonnet-latest', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)
    with pytest.raises(ModelHTTPError) as exc_info:
        agent.run_sync('hello')
    assert str(exc_info.value) == snapshot(
        "status_code: 500, model_name: claude-3-5-sonnet-latest, body: {'error': 'test error'}"
    )


async def test_document_binary_content_input(
    allow_model_requests: None, anthropic_api_key: str, document_content: BinaryContent
):
    m = AnthropicModel('claude-3-5-sonnet-latest', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m)

    result = await agent.run(['What is the main content on this document?', document_content])
    assert result.output == snapshot(
        'The document shows only the text "Dummy PDF file" at the top of what appears to be a blank white page.'
    )


async def test_document_url_input(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-3-5-sonnet-latest', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m)

    document_url = DocumentUrl(url='https://pdfobject.com/pdf/sample.pdf')

    result = await agent.run(['What is the main content on this document?', document_url])
    assert result.output == snapshot(
        'This document appears to be a sample PDF file that contains Lorem ipsum text, which is placeholder text commonly used in design and publishing. The document starts with "Sample PDF" and includes the line "This is a simple PDF file. Fun fun fun." followed by several paragraphs of Lorem ipsum text. Lorem ipsum is dummy text that has no meaningful content - it\'s typically used to demonstrate the visual form of a document or typeface without the distraction of meaningful content.'
    )


async def test_text_document_url_input(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-3-5-sonnet-latest', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m)

    text_document_url = DocumentUrl(url='https://example-files.online-convert.com/document/txt/example.txt')

    result = await agent.run(['What is the main content on this document?', text_document_url])
    assert result.output == snapshot("""\
This document is primarily about the use of placeholder names, specifically focusing on "John Doe" and its variants. The content explains how these placeholder names are used in legal contexts and other situations when a person's identity is unknown or needs to be withheld. The text describes:

1. Various placeholder names like:
- "John Doe" (for males)
- "Jane Doe" or "Jane Roe" (for females)
- "Jonnie Doe" and "Janie Doe" (for children)
- "Baby Doe" (for unidentified children)

2. The usage of these names in:
- Legal actions and cases
- Hospital settings for unidentified patients
- Forms and examples
- Popular culture

3. Regional variations, noting that while this practice is common in the United States and Canada, other English-speaking countries like the UK use alternatives such as "Joe Bloggs" or "John Smith."

The document appears to be a test file with example content sourced from Wikipedia, including licensing information and attribution details at the end.\
""")


def test_init_with_provider():
    provider = AnthropicProvider(api_key='api-key')
    model = AnthropicModel('claude-3-opus-latest', provider=provider)
    assert model.model_name == 'claude-3-opus-latest'
    assert model.client == provider.client


def test_init_with_provider_string(env: TestEnv):
    env.set('ANTHROPIC_API_KEY', 'env-api-key')
    model = AnthropicModel('claude-3-opus-latest', provider='anthropic')
    assert model.model_name == 'claude-3-opus-latest'
    assert model.client is not None


async def test_anthropic_model_instructions(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-3-opus-latest', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m)

    @agent.instructions
    def simple_instructions():
        return 'You are a helpful assistant.'

    result = await agent.run('What is the capital of France?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is the capital of France?', timestamp=IsDatetime())],
                instructions='You are a helpful assistant.',
            ),
            ModelResponse(
                parts=[TextPart(content='The capital of France is Paris.')],
                usage=Usage(
                    requests=1,
                    request_tokens=20,
                    response_tokens=10,
                    total_tokens=30,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 20,
                        'output_tokens': 10,
                    },
                ),
                model_name='claude-3-opus-20240229',
                timestamp=IsDatetime(),
                vendor_id='msg_01BznVNBje2zyfpCfNQCD5en',
            ),
        ]
    )


async def test_anthropic_model_thinking_part(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-3-7-sonnet-latest', provider=AnthropicProvider(api_key=anthropic_api_key))
    settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 1024})
    agent = Agent(m, model_settings=settings)

    result = await agent.run('How do I cross the street?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='How do I cross the street?', timestamp=IsDatetime())]),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="This is a basic question about pedestrian safety when crossing a street. I'll provide a clear, step-by-step explanation of how to safely cross a street. This is important safety information that applies to most places, though specific rules might vary slightly by country.",
                        signature='ErUBCkYIBBgCIkDdtS5sPfAhQSct3TDKHzeqm87m7bk/P0ecMKVxqofk9q15fEDVWXxuIzQIYZCLUfcJzFi4/IYnpQYrgP34x4pnEgzeA7mWRCy/f1bK+IYaDH5i0Q5hgZkqPeMdwSIwMzHMBPM4Xae4czWnzjHGLR9Xg7DN+sb+MXKFgdXY4bcaOKzhImS05aqIjqBs4CHyKh1dTzSnHd76MAHgM1qjBQ2eIZJJ7s5WGkRkbvWzTxgC',
                    ),
                    TextPart(content=IsStr()),
                ],
                usage=Usage(
                    requests=1,
                    request_tokens=42,
                    response_tokens=302,
                    total_tokens=344,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 42,
                        'output_tokens': 302,
                    },
                ),
                model_name='claude-3-7-sonnet-20250219',
                timestamp=IsDatetime(),
                vendor_id='msg_01FWiSVNCRHvHUYU21BRandY',
            ),
        ]
    )

    result = await agent.run(
        'Considering the way to cross the street, analogously, how do I cross the river?',
        message_history=result.all_messages(),
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='How do I cross the street?', timestamp=IsDatetime())]),
            ModelResponse(
                parts=[IsInstance(ThinkingPart), IsInstance(TextPart)],
                usage=Usage(
                    requests=1,
                    request_tokens=42,
                    response_tokens=302,
                    total_tokens=344,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 42,
                        'output_tokens': 302,
                    },
                ),
                model_name='claude-3-7-sonnet-20250219',
                timestamp=IsDatetime(),
                vendor_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Considering the way to cross the street, analogously, how do I cross the river?',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="""\
This is an interesting analogy question. The person is asking how to cross a river by comparing it to crossing a street. I should outline the key principles of river crossing that parallel street crossing safety, while adapting for the unique challenges of a river environment.

For crossing a river, I should consider:
1. Finding the right spot (like shallow areas, bridges, or ferry crossings)
2. Assessing the conditions (river speed, depth, width, obstacles)
3. Choosing the appropriate method based on the river conditions
4. Safety precautions specific to water crossings
5. The actual crossing technique

I'll structure this as a parallel to the street crossing guide, highlighting the similarities in approach while acknowledging the different hazards and considerations.\
""",
                        signature='ErUBCkYIBBgCIkCNPEqIUXmAqiaqIqaHEmtiTi5sG6jBLYWmyfr9ELAh9dLAPPiq0Bnp2YQFJB2kz0aWYC8pJW9ylay8cJPFOGdIEgwcoJGGceEVCihog7MaDBZNwmI8LweQANgdvCIwvYrhAAqUDGHfQUYWuVB3ay4ySnmnROCDhtjOe6zTA2N2NC+BCePcZQBGQh/tnuoXKh37QqP3KRrKdVU5j1x4vAtUNtxQhbh4ip5qU5J12xgC',
                    ),
                    TextPart(content=IsStr()),
                ],
                usage=Usage(
                    requests=1,
                    request_tokens=303,
                    response_tokens=486,
                    total_tokens=789,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 303,
                        'output_tokens': 486,
                    },
                ),
                model_name='claude-3-7-sonnet-20250219',
                timestamp=IsDatetime(),
                vendor_id=IsStr(),
            ),
        ]
    )


async def test_anthropic_model_thinking_part_stream(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-3-7-sonnet-latest', provider=AnthropicProvider(api_key=anthropic_api_key))
    settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 1024})
    agent = Agent(m, model_settings=settings)

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='How do I cross the street?') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert event_parts == snapshot(
        [
            PartStartEvent(index=0, part=ThinkingPart(content='', signature='')),
            PartDeltaEvent(index=0, delta=IsInstance(ThinkingPartDelta)),
            PartDeltaEvent(index=0, delta=IsInstance(ThinkingPartDelta)),
            PartDeltaEvent(index=0, delta=IsInstance(ThinkingPartDelta)),
            PartDeltaEvent(index=0, delta=IsInstance(ThinkingPartDelta)),
            PartDeltaEvent(index=0, delta=IsInstance(ThinkingPartDelta)),
            PartDeltaEvent(index=0, delta=IsInstance(ThinkingPartDelta)),
            PartDeltaEvent(index=0, delta=IsInstance(ThinkingPartDelta)),
            PartDeltaEvent(index=0, delta=IsInstance(ThinkingPartDelta)),
            PartDeltaEvent(index=0, delta=IsInstance(ThinkingPartDelta)),
            PartStartEvent(index=1, part=IsInstance(TextPart)),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\


1. **Fin\
"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='d a designated crossing point** if')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
 possible:
   -\
"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Crosswalks')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\

   - Pedestrian signals
   -\
"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Pedestrian bridges')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\

   - Inters\
"""
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
ections

2. **Before\
"""
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
 crossing:**
   - Stop\
"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' at the curb or edge of the road')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\

   - Look left,\
"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' right, then left again (')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='or right, left, right again')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' in countries with left')),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
            PartDeltaEvent(index=1, delta=IsInstance(TextPartDelta)),
        ]
    )


async def test_multiple_system_prompt_formatting(allow_model_requests: None):
    c = completion_message([BetaTextBlock(text='world', type='text')], BetaUsage(input_tokens=5, output_tokens=10))
    mock_client = MockAnthropic().create_mock(c)
    m = AnthropicModel('claude-3-5-haiku-latest', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m, system_prompt='this is the system prompt')

    @agent.system_prompt
    def system_prompt() -> str:
        return 'and this is another'

    await agent.run('hello')
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert 'system' in completion_kwargs
    assert completion_kwargs['system'] == 'this is the system prompt\n\nand this is another'


def anth_msg(usage: BetaUsage) -> BetaMessage:
    return BetaMessage(
        id='x',
        content=[],
        model='claude-3-7-sonnet-latest',
        role='assistant',
        type='message',
        usage=usage,
    )


@pytest.mark.parametrize(
    'message_callback,usage',
    [
        pytest.param(
            lambda: anth_msg(BetaUsage(input_tokens=1, output_tokens=1)),
            snapshot(
                Usage(
                    request_tokens=1, response_tokens=1, total_tokens=2, details={'input_tokens': 1, 'output_tokens': 1}
                )
            ),
            id='AnthropicMessage',
        ),
        pytest.param(
            lambda: anth_msg(
                BetaUsage(input_tokens=1, output_tokens=1, cache_creation_input_tokens=2, cache_read_input_tokens=3)
            ),
            snapshot(
                Usage(
                    request_tokens=6,
                    response_tokens=1,
                    total_tokens=7,
                    details={
                        'cache_creation_input_tokens': 2,
                        'cache_read_input_tokens': 3,
                        'input_tokens': 1,
                        'output_tokens': 1,
                    },
                )
            ),
            id='AnthropicMessage-cached',
        ),
        pytest.param(
            lambda: BetaRawMessageStartEvent(
                message=anth_msg(BetaUsage(input_tokens=1, output_tokens=1)), type='message_start'
            ),
            snapshot(
                Usage(
                    request_tokens=1, response_tokens=1, total_tokens=2, details={'input_tokens': 1, 'output_tokens': 1}
                )
            ),
            id='RawMessageStartEvent',
        ),
        pytest.param(
            lambda: BetaRawMessageDeltaEvent(
                delta=Delta(),
                usage=BetaMessageDeltaUsage(output_tokens=5),
                type='message_delta',
            ),
            snapshot(Usage(response_tokens=5, total_tokens=5, details={'output_tokens': 5})),
            id='RawMessageDeltaEvent',
        ),
        pytest.param(lambda: BetaRawMessageStopEvent(type='message_stop'), snapshot(Usage()), id='RawMessageStopEvent'),
    ],
)
def test_usage(message_callback: Callable[[], BetaMessage | BetaRawMessageStreamEvent], usage: Usage):
    assert _map_usage(message_callback()) == usage


async def test_anthropic_model_empty_message_on_history(allow_model_requests: None, anthropic_api_key: str):
    """The Anthropic API will error if you send an empty message on the history.

    Check <https://github.com/pydantic/pydantic-ai/pull/1027> for more details.
    """
    m = AnthropicModel('claude-3-5-sonnet-latest', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m, instructions='You are a helpful assistant.')

    result = await agent.run(
        'I need a potato!',
        message_history=[
            ModelRequest(parts=[], instructions='You are a helpful assistant.', kind='request'),
            ModelResponse(parts=[TextPart(content='Hello, how can I help you?')], kind='response'),
        ],
    )
    assert result.output == snapshot("""\
I can't physically give you a potato since I'm a computer program. However, I can:

1. Help you find recipes that use potatoes
2. Give you tips on how to select, store, or cook potatoes
3. Share information about different potato varieties
4. Provide guidance on growing potatoes

What specifically would you like to know about potatoes?\
""")


async def test_anthropic_web_search_tool(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-3-5-sonnet-latest', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m, builtin_tools=[WebSearchTool()])

    result = await agent.run('What day is today?')
    assert (
        result.all_messages()
        == snapshot(
            [
                ModelRequest(parts=[UserPromptPart(content='What day is today?', timestamp=IsDatetime())]),
                ModelResponse(
                    parts=[
                        TextPart(content="Let me search for current events to help establish today's date."),
                        BuiltinToolCallPart(
                            tool_name='web_search',
                            args={'query': 'current events news today May 26 2025'},
                            tool_call_id='srvtoolu_01MqVvTi9LWTrMRuZ2KttD3M',
                            provider_name='anthropic',
                        ),
                        BuiltinToolReturnPart(
                            tool_name='web_search_tool_result',
                            content=[
                                BetaWebSearchResultBlock(
                                    encrypted_content='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',
                                    title='News: U.S. and World News Headlines : NPR',
                                    type='web_search_result',
                                    url='https://www.npr.org/sections/news/',
                                ),
                                BetaWebSearchResultBlock(
                                    encrypted_content='ErwHCioIAxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDK2W0Qu0wNgI6mf5EhoMNCqr1gVeM/Fj3PSYIjBrgyGKmTOrJLgDCXcOvRtrbigKDeccd5oypMBGnMVhm6h3Ade/9+vNOwI3ByflKmwqvwZqUUdfJ6+k9ZDrmb7VM6ktRqsZ4Z++yOdyubDNbsyM6RdwYuNi+bS5ZUON+rMd8+ZrQYlGYqq7NF43o5klxpac+Dsgx3OlKbu6Hq6eiKOQ3rdPGYlUYKdDouAx6RjypXjhYkqErPrjlFZhNv2lO6cohI0QU66p6b7G8UMVyweqYZ+2QYTFfbwU5VdIAOiQW8PBgNwPC5LRnidfbiT04VY+cEsNW04zOq9coXs4NgFRw2WDCZDBPGTEJex0xv7vD0/D0YpBhfiawNJ8FgBTI6q0gXQ2+YwqelVaZ+BDpu2JeRABLXiXQAMIiBBiayofacvfgJZ4omPY1JRiJwX5IpbLFqLcNz2fWr8veYedwrDZV/lOjyn755WTp2i89GD4Pv53htWrDOH8/YJBQ9u5KA2DFz7zAtRLyPqvPz3YaLMr3ATFvs8m0igrllgC5uaWPWfO/28RU7QNnxyBLGNonF3dtz3Uu2naeNvxjRhqCtUOON5odOahtPrRs5qkjv/UrL2YzlnfsRL4Qb/qsGJE6YWScvLhjBaum29Whk2p6RtYJqzzSqDbk0jxKe/hNatl3s2JF1bAW4L7p9FnsK1v/G7AYSaIYl4RDLGuL1bFOKGKVlUZtohNMws+gvTCYKdhQzfurimTsNIpBP4Ci6aJ+/yACa22AXGhZQqyiOS7yxI6zj3vZdQGFBle1TjDpzveY2Nz/kuuTCPbGsWt5kd9v7BkWvkNacqZ70KijyIk5dVt3H0q4eavyNLU0gF4hSCPDHW7eeWXTmNs1YniKiaHrwmOOqXjw2PCQrZv0i7UQRjDmRQqx9NtuqzMup9DRPbQuZM23b8JwzqA0Qjyxc5pTlWRL9aU+U7ZKOD1OdBszAU54c5N9jOca8S2Plt4TGJcAv2Wy73Bex74GPlkHcKWO8TJYhrV4ZF2nMjssncQEKCltJaZg6TJpazpLKoQ1XmYmgzebbVMRc8RTDXk335AYKkN62xRnfrDd5T5wBhGbPNQeF7PGigtAK/SpSpTna/vmGOBul/cONWOFFKNdY+FtCAGd4AOo3s/N8QUnKR66TEv9ocuVep1UZxV2fJcqIuJukutfT9eWPcou6VImLUzRQMYAw==',
                                    page_age='4 days ago',
                                    title='The Biggest News Stories Of 2025',
                                    type='web_search_result',
                                    url='https://92q.com/playlist/the-biggest-news-stories-of-2025/',
                                ),
                                BetaWebSearchResultBlock(
                                    encrypted_content='EuYCCioIAxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDCmLeaYgzUtJ4Mi+fBoMpktwxWvlSdpeNRRlIjDckkEHzeMKWP+vkSJ+7ci91OlLvn1nTU4wG+0am9miZ+68Q8XjsyCBGekIPeSsgpkq6QFTAvqrNhd55GMbj8VQtB/7oV66lwp8PzaymgQlLzCnxBdZ6IRYyEd6XwFOPrWCwyjtlKbwRiM2NIaNsGcrraBVrDfsjCz20qsDPGNsQf587z/TD3zWUSelhjhf+T5nDCEXUkYM2+4MaGP5Ty57Khh3WQr5q6Q46m85jBBF+akWf1uKZEgjgFug1ufj/8TXEEAaKCVY9YeXTXfYH8DocKveCXH4Bp9TNbgx55UrL8NdXiwdtpI/zqY+8hM/SiaVeXXI/Rbmjg3HzFTLfrH4wSrl5awdKWuGwQy8nqRISZlwNVnwFY0e8uc08xgD',
                                    title='ABC News – Breaking News, Latest News and Videos',
                                    type='web_search_result',
                                    url='https://abcnews.go.com/',
                                ),
                                BetaWebSearchResultBlock(
                                    encrypted_content='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',
                                    title='Breaking News, Latest News and Videos | CNN',
                                    type='web_search_result',
                                    url='https://www.cnn.com/',
                                ),
                                BetaWebSearchResultBlock(
                                    encrypted_content='ErQCCioIAxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDGPat3RtBffd6jd9uxoMkx9uAfM4hgJRbrR4IjCBVNWqux+TsqDP0poLm+ss84SLrVR4rAcjrQSDPna9ZfR0OFhPjv2ko1ZVuBzeRE4qtwGOPV6my0G/y4nPEH8gNVc3y/8uZzh7O8CBrduzchEMd5RRXLlsC+bU/SjZ+5LBYGzAVwRCfVXIdaJ0/d8RYdJWHo3bvKc5Lu/WFPV6Po9gVHLOU5WVDsyzwmrvqzCYC0UhkUMa0yf5j7WTFaT+kgHZcFcbvYPG53USqNh0seahaaCC5fJRjRBTAvuyj4md+ppTjIXGZEp3rTMG3MTkv8t60MgPzn4ObLGEmBQIQrfES9G2BT1k7lUYAw==',  # codespell:ignore
                                    title='Philippines Top Stories: Politics, Environment, Education, Trending | Inquirer.net',
                                    type='web_search_result',
                                    url='https://newsinfo.inquirer.net',
                                ),
                                BetaWebSearchResultBlock(
                                    encrypted_content='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',
                                    title='Portal:Current events/May 2025 - Wikipedia',
                                    type='web_search_result',
                                    url='https://en.wikipedia.org/wiki/Portal:Current_events/May_2025',
                                ),
                                BetaWebSearchResultBlock(
                                    encrypted_content='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',
                                    title='Portal:Current events - Wikipedia',
                                    type='web_search_result',
                                    url='https://en.wikipedia.org/wiki/Portal:Current_events',
                                ),
                                BetaWebSearchResultBlock(
                                    encrypted_content='EsoJCioIAxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDAYjLeq1Kci5o2pi8BoMqJacE/46sh+pR01VIjDPMDUx0d3wj2zbUnowpxFwHvCCnCwLoPMxQsQm/dnvm0Fzfga5o6zXqDwWpXvUzhEqzQg/X9w2opL/m5o7bUw/TubgkYaR6l4t2n0oQlGgetbSj1gOEor/WWJ8bWXL3BZnS2xkIzwGrbLdlPDn/NoXICDQZ+P3IlA6B8CVwiijOnNq+x5nTpX0m794VFJIfO5SdupAiWqhWmtqt8XhcWs8W2gnhDDvNsBG6oH2ZsRenxt3n7a7eWo7yRk/KSHdBM9c+L5r3wu3Ul81DW9CuE6KqUdFjjePJfWKL8OvzfkvJjIqcRaqc/3RIRZbSPbimBiRMXtCBZeCYE1yeBs3xLQ7TJXgRM/9ScKromcFWckYpGXBYSGL8SiXXoBUD7pLsuP5FnRnZUkQLCHTLoId0/w4jVbuXmDh3oipIlGUQCSbp3FkogFB/CZFpKz4tY5E9WQ4pBkApGYgAeGgOOStiUW3pE9oCy5TRpCfilrg66RtJozGI+LWM/XYuuOwSK+f+/c6AaUJ7av+LCUSPFI6G1XErfHK/KeBSJp7ZVoRXn/f7yJXlZvybKQXdN6UtxqxRJbil9RnmmXsBc6cesWW/cHbz01V8tkaqcYdrtJdVM/LesICK77C/JYiA6PQsneeg5xdZDCUp7yUO9P/CHMBqhPB8geS9y4dG7UIdJrFbv43cGOiqoSBsBGCLCc7crptYYGydT6YBgKb+ktUJm14MfbF8lzKt6SVYpn8KWL2dyhsDbfi88h51fvZqDV6loTDpyHbMHeJoA4pIxLhkBIriQOLNnEIEwqTGy2XFy326bahzINKJVTY1mMq2v3O0Snl0DNcAZ1X/iHt393xPgdcSy6c2+sDRexvpU4grX1GGFD4E8kg1QP0fErasq17XzRVpnU7Kedk/ntU/X6zeI3aTEeyRNG7IPH67w6GyIF8XmgCh25H6bCBGN87N8hnPSVAy8/qIMcfZYaF1c8W/QB9n7HBWhQgdyZv3relj0Ur0xdRi2osqo+k2c0a9mmIVupbzpLAxfY7LiwU8Edsr+1WY62x1omk+b4XNiGnhHnrF4B2o+f89icgAVSqRo2ydqIUDnZUYewu8jjUg/j+WUI8yKqZHCgCRdkm4fDSOcK8faTeaITl1iI6XFbUicEWZzG87tFykNSv5fz+ueDbMj936cm97rPUhp/qMnS2uloAxmiWLcS2/oV605i97ccR9IlwB0tt259e9iCvltjxzcC6P95vbhLS94+xVNOG2fmQtzE8oyaREZBkwSjVHuJ3lDAxvHDRYY8F+lkuLE4AvLiye3CDAMXNyCrG+/xiQIBNUGs+1aV9edHMmwpCVs99Q+nHO1RBVPljY607Q6u06Wt4VHnY+45+IxzpHHWXxg3Jn8Lh1AuzFKEaRWaI7JDSCgJmYxjIwkUO7988PWjOmFLquOd6mQsQ6iVG/89zSwr019RlAQRDIbMimefHIYhLm4S/Y8TzPhLFXJ6FaxrPFAkkkp2LnLQUoNKlo2h64JaAerGAku2FEwn/vo3hsCXwILg6R4QYAw==',
                                    title='Current Affairs Today – Current Affairs – 2025-26 - GKToday',
                                    type='web_search_result',
                                    url='https://www.gktoday.in/current-affairs/',
                                ),
                                BetaWebSearchResultBlock(
                                    encrypted_content='Eo4ZCioIAxgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDPmBifgbGKjJUWA4LxoMysyEr2mcRqSQzmmjIjBCfvhhJTAOxHLRGv3ljzK5jUBicnJMjypAm7ZduBpPkX+mDQvdcj2CACyWCydaQ0gqkRjhs2EgE0MIO0YyXAzzkWltz0ki4S12d0f0bRdmI2W31SmMvd7jGXZyIQx60LTKRqqQa1GVeqfrM2855muFRsk6uji8F9tni8hjSdU4Pcd5WlC6f2dwWDJcfvt/HD2GCI1uPZkx4ha2BLYDV37uXfumDLk/tFPswHb3cbUgLc28rTb18pTi+HTE5f5r06/x2DPVVXYLylYuRPtr2WJ6l/2r/I59B+iwdTzI8sYRhSIRf47kt32Reo3w5esYpPsmGXFL0SOW57j5jtwBZWkJqkEc5wD10ObzxCDXrNfZ89KVkli/++RedncFZnqKcWkrLwctyW4eIBj0qiI4ZA81Wx6cnc09shOAflAw1EitPiOQ4HKoNkcFn9GNUfF1rBzblgVvjgO5t/zZpv53CnuZ9Aoo2nwF4pNrflPpmnd22gQdLpOmLTYrxygC/2vboGrNrS1HkxfvFKPib1DopDY/y9CECre1zHtdf6PNQxgvc+EGIncCnb8gTHFZxN1Mhyc1dmTDhitv/vawaqI8sZHx54tnP5l+KvWuXbegWPJETo6hMsbtMYHAmJIi5VFmhn6rq1zRuKYBFILEpHs3RPoybQzJtJoRYVRYA1E/vsBrdTfXD6jNDg6fz+88kc/menQlSALfAIhZhwxGz5eyhFqBVeYNfqKrJR+CHiUTAKN7t0R/nlsYo89V4SFMpvZakV9ywY6lqnu8mehn4c28OtFQ51wqtldG97kyQFNazwoXrayCNWo3xshZ5hqv1mSIAU2xGUD1UENXddR9bba6BmrU3zgroGPNbYkFUFVeyHAcHmw3oxy+18LW32bRY3Rgv6oZAXnZTELQDXMKGAjod49GlRKDwH+fEPPHu2FGgIATYdUErwDm/C4G0taall34pnQLXtT5+5H9bSmGzf/4f+4of7V2nRHgUPcAxh8Kg8W/RIdd50ZD3zhkDDkTXDYEoRB4EY69OXRtbUnn5rQo96S5zOQmMlbMQ7ik5kkHSKrLwxS8l6hm51UEXhosckj1BEuXMSsdvfhpXOHlgIOScK27Xhz5cIiGYfFO19GGJo0iDDTlOZypGJAqtyeuOy526cBr/0FlnRa9dGYCrAqVtEkb0NfcYRq6loOpU2gjAxs0bn4unbO/3cisywH9TKmdMydJ8WpO3VG3c/pICXFUs8etkT1H64uI2NfPazsdM99aaMsrTpoAr5b1yKGLP2w4NyRGtRA8n8wIXgrrLf7WSqXKJsN9x5v8ezSR72krIfSwXHvdAz3X2c/hcUyzgRVrTV6qssio63qc5ysdlXzkwhVpO8ChRdHebKROmYpU3EfWe++sHkMdYdO2IbOF9fB392Qt8H/FND/v5TAp6g/V9Jdo37lIbdbLdulNkaexrP1fgXl97sC8D+BHa5oF5IhYHxU328yF2pIr8RwD3eWuDvo6K7fC3Fh8DQOrT4dJNAihKKQok48GS+0J25yasYCLK6T7E67ZqETt1vRHHuJiSL26awGv0Qgc65IylcPcXJlddKk+nmTTMl6B5V5xxnGpxhhtXSmXReRgHOjxqrxsg1cBfDk8S16YzC6Qjg4fwR61ynDesgv9aaxabkcUHBqVAMh7qxWbEt0gicz45ciWa84fB7du53fuiRJA4CaIAhDWyH75OcYBthux+KUOpADOIlXJ0IBraFIcOTmDUrPInIAdSnmjFlUbGkbenWW0FGC08jY67UQfQUHQcIy3qyOKxu7SuFWo4wmFSI2WRKn9Ds/X4go99IXPHPcw8JrzFOcqUR0GXxDfwgxL1AyygyljWsj9PzC6HtSN008PAb5ve5X6PmpCGbH5bIR26WzUMCHJLBzUFv9vDmGbwDhKNmvPpkAi1apHxDY+Z0ZMvr87YH63SI6cI0wsxYvlpTaXSZI/4p6QzjCUbfQhaHNlS7/nMcgxMDzruRcp7h48gl2ViULjY5JCzXeadKJY8C/fxfPFW1qkzzpMwkZQyEboCd/q/GSo5Dt/2gh5Fe4oTAy78gBGHiVXjqp1RsBwGwRL2ReQ12Cq5bvpQMaDS8HCfpsukM6VMY2v/IS5luCxoeKUMkPzh/ATL3FFFXZ3Z+v5nCvr6QV6zol4XdFf8EsfKcH9LMDYWj35KpIhRif4/HUkysfaLJk8NRX+7ySlBQ6OZSA3QkCt0iwcWSaObK5D/eUWPLUpwReg1X6HJ7F4zo4iZh1h6RaThgclJeDwdkU+3QBKwa7XJn77HDQfEhpU0Jx6rTyvcdN/B2xAXJckjDDSaiv/CFYUOQKaMhXTgQyZ+/5JHSnOfcmnTePOUEj0Tge1iRQHb2fQU0kPpxA2va4dF8aBuJr/G1H772OvMUnfjTxWNFhbM1QZ4dO5hpBMvf6k4DgLMirSsCFrlc3FF+qpFEHkI3Ms0wb8w1llPq/chf0dzxTkWRA0ePN/1Nhkjf93MBYO1Er2hz5Pkgr2jxDmJ4R3cOtW/9vJIgTqUH5L4CvNAH3vhAfi0A4k+XQ4c5ML+4WGNsVApnPfdF+GoBRTrGWdkpjNfe6pSAeleQL9p/1gT7YFMCx6HkT3SfrEyO3ZYitkB/t/phzg/OJu6/n4HwQZuZNaZGQ5pd5yDL0TOXP5lz9ATAe6Qtp8VHUqZ6UyH9MDDZ22owsxuAbcHV7aJNCtcjOQWXv3hAElq5JaoZFJxr31yDdblQMZ4tswPhUUb1s2CUuv4oX30khUpeOBpk7PC8SeOVG1IRe1gSsHi0BiDzvZXDSSDSDxn7rHQKs/niUIAQqdMjbKK9H8X8KDb7h7IxhiqYuGSCt6UONFSv2aghhXEZIHmZTNymOPC1NLU6vPZEh26aTIstS+LIzP6HZjkgBgfXgHX4TvoDYIOsv/MDRO2cAJC6NwBj8BcPxXvsi1aqQeoQIT8U3CIyDwIUT3z0Dt0kmSnD3Sf+X2sK+iYc5Qkrc9f2M/VpcXr2WaF2n4yE/bti9dzlDWSpHSxus+ppAIF74N+bUCd1BVFyUYFAhNG1gMLA28ogL3dd8R5bsBFCrSHJWwOx55OzVgTN46peF2oKbEWxx8ngW+IpsEH4NbV9+jeFWL9tIDPz4TQqTndwpi3VZV4qXn8xUc2HjXDE42PvZYZnRt0LFWJpmj0F/XLpS0e3wLVuJmThY7Pf+8f5CYsN+7PCxElBqWYD2x5ngjN8g0nUv/xERjOuKOAb23ycsOQEgx+VkeqbayfAmnfROpOBzg/py9KzmhHNiwKESSKLm3BRey3SVqeUdmjwnWKjoLopgHmlE31kYbFSijjDYKmo+tgIkI0XAIqzHqpuUT7I6JOSfE2p74WqssiIYSi4gLQ9M41yf23lqb6U1Xs5hZeCDVHd3bgw7oBa2V71Vn2C3TGVW8zTC1HiBu3Ecxu1n57Hr3pgLJGAdl/Lj+Ay7G+E5+qXspAHWaiVTESMEmsr5klskSzovzqCp+A3NTBdPRwsKi8lZmQJ+H5nsNMt5g6PITF/WsS/pyvSNvlL4E79pYghythA12UmhMzkeHtg6zBta1Mq7C087Fihha6QrmOARa9khbpijLCKmjj8fydWmoQw5iCK2l8qwdOU1TkB++w8Vym3h25ai4j3X6ChkoAA5BQWivzFAycJ8PVfFs2WGGUNcNM649drxBpSNYzuJQpiLJeZS3RcyBWaeVHn0EqvmFnSYJB6I3loUw1aabJ2SWXrBU7SSGnSDsNQuE1M0JdN8NTT+KGARvjZISAYSCWHVdOzCWsj0I/2FcQHcv3Mv5nEUKp73tnK4KEiLKNuJ4oIvEndcOtqrmqGdl0sONVPiBvy8jOVw/VarOUpn+9OzNsEJ8LYV+dSos1qjc08b9AeH4RvDRk90KLMTfElM6e5Z526vj/IyCPWc8PEWMAT0Vaw2dSwL0AdsDn2yNH5Q7TS4CpWgzJHJHq3ph+J3E2Yuo1xXhVtdIPHorS+64+/lQ7rUCZ36sTmJj5eOLEJXhj5XnfeDQq1jU5keqBMiCUBkxNNlLCdkq34qWUcgmVfVskSh9Uq0ml5NhUFjKvHwxSfqZ3hlW8Z6a0PXzdYQDLi0EI2THYV1JTkOB2T9UC8N0pzRBesxLeXZTpfwLpUmI6rWtkwDIUh4HLo7UEZtX5s1kDVZqMcgRp5Ci4BYLVBgD',
                                    page_age='3 weeks ago',
                                    title='May 6, 2025: Top news events to look out for today - People Daily',
                                    type='web_search_result',
                                    url='https://peopledaily.digital/news/may-6-2025-top-news-events-to-look-out-for-today',
                                ),
                                BetaWebSearchResultBlock(
                                    encrypted_content='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',
                                    page_age='7 hours ago',
                                    title='26 May 2025 UPSC Current Affairs - Daily News Headlines',
                                    type='web_search_result',
                                    url='https://testbook.com/ias-preparation/upsc-current-affairs-for-26-may-2025',
                                ),
                            ],
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                            provider_name='anthropic',
                        ),
                        TextPart(
                            content="""\


Based on the search results, today is Monday, May 26, 2025. This is confirmed by several sources:

1. \
"""
                        ),
                        TextPart(content="It's Memorial Day today, May 26, 2025"),
                        TextPart(
                            content="""\


2. \
"""
                        ),
                        TextPart(
                            content='May 2025 is the fifth month of the current common year. The month began on a Thursday and will end on a Saturday after 31 days'
                        ),
                        TextPart(
                            content="""\


3. \
"""
                        ),
                        TextPart(
                            content="On May 26, 2025, there are significant developments happening, including India's launch of the Bharat Forecasting System to boost weather prediction and disaster preparedness"
                        ),
                    ],
                    usage=Usage(
                        requests=1,
                        request_tokens=16312,
                        response_tokens=258,
                        total_tokens=16570,
                        details={
                            'cache_creation_input_tokens': 0,
                            'cache_read_input_tokens': 0,
                            'input_tokens': 16312,
                            'output_tokens': 258,
                        },
                    ),
                    model_name='claude-3-5-sonnet-20241022',
                    timestamp=IsDatetime(),
                    vendor_id=IsStr(),
                ),
            ]
        )
    )


async def test_anthropic_code_execution_tool(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m, builtin_tools=[CodeExecutionTool()])

    result = await agent.run('How much is 3 * 12390?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='How much is 3 * 12390?', timestamp=IsDatetime())]),
            ModelResponse(
                parts=[
                    TextPart(content="I'll calculate 3 * 12390 for you."),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={
                            'code': """\
result = 3 * 12390
print(f"3 * 12390 = {result}")\
"""
                        },
                        tool_call_id=IsStr(),
                        provider_name='anthropic',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution_tool_result',
                        content=BetaCodeExecutionResultBlock(
                            content=[],
                            return_code=0,
                            stderr='',
                            stdout='3 * 12390 = 37170\n',
                            type='code_execution_result',
                        ),
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='anthropic',
                    ),
                    TextPart(content='The answer is **37,170**.'),
                ],
                usage=Usage(
                    requests=1,
                    request_tokens=1630,
                    response_tokens=105,
                    total_tokens=1735,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 1630,
                        'output_tokens': 105,
                    },
                ),
                model_name='claude-sonnet-4-20250514',
                timestamp=IsDatetime(),
                vendor_id=IsStr(),
            ),
        ]
    )


async def test_anthropic_server_tool_pass_history_to_another_provider(
    allow_model_requests: None, anthropic_api_key: str, openai_api_key: str
):
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

    openai_model = OpenAIResponsesModel('gpt-4.1', provider=OpenAIProvider(api_key=openai_api_key))
    anthropic_model = AnthropicModel('claude-3-5-sonnet-latest', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(anthropic_model, builtin_tools=[WebSearchTool()])

    result = await agent.run('What day is today?')
    assert result.output == snapshot("""\
Let me search for today's date.



Based on the search results, \n\

today is Monday, May 26, 2025 (Week 22)

. This is notably \n\

Memorial Day, which was originally known as Decoration Day

. \n\

The year 2025 is a regular year with 365 days

.\
""")
    result = await agent.run('What day is tomorrow?', model=openai_model, message_history=result.all_messages())
    assert result.new_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='What day is tomorrow?', timestamp=IsDatetime())]),
            ModelResponse(
                parts=[TextPart(content='Tomorrow will be **Tuesday, May 27, 2025**.')],
                usage=Usage(
                    request_tokens=410,
                    response_tokens=17,
                    total_tokens=427,
                    details={'reasoning_tokens': 0, 'cached_tokens': 0},
                ),
                model_name='gpt-4.1-2025-04-14',
                timestamp=IsDatetime(),
                vendor_id='resp_6834631faf2481918638284f62855ddf040b4e5d7e74f261',
            ),
        ]
    )


async def test_anthropic_server_tool_receive_history_from_another_provider(
    allow_model_requests: None, anthropic_api_key: str, gemini_api_key: str
):
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    google_model = GoogleModel('gemini-2.0-flash', provider=GoogleProvider(api_key=gemini_api_key))
    anthropic_model = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(builtin_tools=[CodeExecutionTool()])

    result = await agent.run('How much is 3 * 12390?', model=google_model)
    assert part_types_from_messages(result.all_messages()) == snapshot(
        [[UserPromptPart], [BuiltinToolCallPart, BuiltinToolReturnPart, TextPart]]
    )

    result = await agent.run('Multiplied by 12390', model=anthropic_model, message_history=result.all_messages())
    assert part_types_from_messages(result.all_messages()) == snapshot(
        [
            [UserPromptPart],
            [BuiltinToolCallPart, BuiltinToolReturnPart, TextPart],
            [UserPromptPart],
            [BuiltinToolCallPart, BuiltinToolReturnPart, TextPart],
        ]
    )


async def test_anthropic_empty_content_filtering(env: TestEnv):
    """Test the empty content filtering logic directly."""

    # Initialize model for all tests
    env.set('ANTHROPIC_API_KEY', 'test-key')
    model = AnthropicModel('claude-3-5-sonnet-latest', provider='anthropic')

    # Test _map_message with empty string in user prompt
    messages_empty_string: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='')], kind='request'),
    ]
    _, anthropic_messages = await model._map_message(messages_empty_string)  # type: ignore[attr-defined]
    assert anthropic_messages == snapshot([])  # Empty content should be filtered out

    # Test _map_message with list containing empty strings in user prompt
    messages_mixed_content: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content=['', 'Hello', '', 'World'])], kind='request'),
    ]
    _, anthropic_messages = await model._map_message(messages_mixed_content)  # type: ignore[attr-defined]
    assert anthropic_messages == snapshot(
        [{'role': 'user', 'content': [{'text': 'Hello', 'type': 'text'}, {'text': 'World', 'type': 'text'}]}]
    )

    # Test _map_message with empty assistant response
    messages: list[ModelMessage] = [
        ModelRequest(parts=[SystemPromptPart(content='You are helpful')], kind='request'),
        ModelResponse(parts=[TextPart(content='')], kind='response'),  # Empty response
        ModelRequest(parts=[UserPromptPart(content='Hello')], kind='request'),
    ]
    _, anthropic_messages = await model._map_message(messages)  # type: ignore[attr-defined]
    # The empty assistant message should be filtered out
    assert anthropic_messages == snapshot([{'role': 'user', 'content': [{'text': 'Hello', 'type': 'text'}]}])

    # Test with only empty assistant parts
    messages_resp: list[ModelMessage] = [
        ModelResponse(parts=[TextPart(content=''), TextPart(content='')], kind='response'),
    ]
    _, anthropic_messages = await model._map_message(messages_resp)  # type: ignore[attr-defined]
    assert len(anthropic_messages) == 0  # No messages should be added


async def test_anthropic_tool_output(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-3-5-sonnet-latest', provider=AnthropicProvider(api_key=anthropic_api_key))

    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent(m, output_type=ToolOutput(CityLocation))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run('What is the largest city in the user country?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in the user country?',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='get_user_country', args={}, tool_call_id='toolu_019pMboNVRg5jkw4PKkofQ6Y')
                ],
                usage=Usage(
                    requests=1,
                    request_tokens=445,
                    response_tokens=23,
                    total_tokens=468,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 445,
                        'output_tokens': 23,
                    },
                ),
                model_name='claude-3-5-sonnet-20241022',
                timestamp=IsDatetime(),
                vendor_id='msg_01EnfsDTixCmHjqvk9QarBj4',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id='toolu_019pMboNVRg5jkw4PKkofQ6Y',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args={'city': 'Mexico City', 'country': 'Mexico'},
                        tool_call_id='toolu_01V4d2H4EWp5LDM2aXaeyR6W',
                    )
                ],
                usage=Usage(
                    requests=1,
                    request_tokens=497,
                    response_tokens=56,
                    total_tokens=553,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 497,
                        'output_tokens': 56,
                    },
                ),
                model_name='claude-3-5-sonnet-20241022',
                timestamp=IsDatetime(),
                vendor_id='msg_01Hbm5BtKzfVtWs8Eb7rCNNx',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='toolu_01V4d2H4EWp5LDM2aXaeyR6W',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
        ]
    )


async def test_anthropic_text_output_function(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-3-5-sonnet-latest', provider=AnthropicProvider(api_key=anthropic_api_key))

    def upcase(text: str) -> str:
        return text.upper()

    agent = Agent(m, output_type=TextOutput(upcase))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run(
        'What is the largest city in the user country? Use the get_user_country tool and then your own world knowledge.'
    )
    assert result.output == snapshot(
        "BASED ON THE RESULT, YOU ARE LOCATED IN MEXICO. THE LARGEST CITY IN MEXICO IS MEXICO CITY (CIUDAD DE MÉXICO), WHICH IS ALSO THE NATION'S CAPITAL. MEXICO CITY HAS A POPULATION OF APPROXIMATELY 9.2 MILLION PEOPLE IN THE CITY PROPER, AND OVER 21 MILLION PEOPLE IN ITS METROPOLITAN AREA, MAKING IT ONE OF THE LARGEST URBAN AGGLOMERATIONS IN THE WORLD. IT IS BOTH THE POLITICAL AND ECONOMIC CENTER OF MEXICO, LOCATED IN THE VALLEY OF MEXICO IN THE CENTRAL PART OF THE COUNTRY."
    )

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in the user country? Use the get_user_country tool and then your own world knowledge.',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="I'll help you find the largest city in your country. Let me first check your country using the get_user_country tool."
                    ),
                    ToolCallPart(tool_name='get_user_country', args={}, tool_call_id='toolu_01EZuxfc6MsPsPgrAKQohw3e'),
                ],
                usage=Usage(
                    requests=1,
                    request_tokens=383,
                    response_tokens=66,
                    total_tokens=449,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 383,
                        'output_tokens': 66,
                    },
                ),
                model_name='claude-3-5-sonnet-20241022',
                timestamp=IsDatetime(),
                vendor_id='msg_014NE4yfV1Yz2vLAJzapxxef',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id='toolu_01EZuxfc6MsPsPgrAKQohw3e',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="Based on the result, you are located in Mexico. The largest city in Mexico is Mexico City (Ciudad de México), which is also the nation's capital. Mexico City has a population of approximately 9.2 million people in the city proper, and over 21 million people in its metropolitan area, making it one of the largest urban agglomerations in the world. It is both the political and economic center of Mexico, located in the Valley of Mexico in the central part of the country."
                    )
                ],
                usage=Usage(
                    requests=1,
                    request_tokens=461,
                    response_tokens=107,
                    total_tokens=568,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 461,
                        'output_tokens': 107,
                    },
                ),
                model_name='claude-3-5-sonnet-20241022',
                timestamp=IsDatetime(),
                vendor_id='msg_0193srwo7TCx49h97wDwc7K7',
            ),
        ]
    )


async def test_anthropic_prompted_output(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-3-5-sonnet-latest', provider=AnthropicProvider(api_key=anthropic_api_key))

    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent(m, output_type=PromptedOutput(CityLocation))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run(
        'What is the largest city in the user country? Use the get_user_country tool and then your own world knowledge.'
    )
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in the user country? Use the get_user_country tool and then your own world knowledge.',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions="""\
Always respond with a JSON object that's compatible with this schema:

{"properties": {"city": {"type": "string"}, "country": {"type": "string"}}, "required": ["city", "country"], "title": "CityLocation", "type": "object"}

Don't include any text or Markdown fencing before or after.\
""",
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='get_user_country', args={}, tool_call_id='toolu_017UryVwtsKsjonhFV3cgV3X')
                ],
                usage=Usage(
                    requests=1,
                    request_tokens=459,
                    response_tokens=38,
                    total_tokens=497,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 459,
                        'output_tokens': 38,
                    },
                ),
                model_name='claude-3-5-sonnet-20241022',
                timestamp=IsDatetime(),
                vendor_id='msg_014CpBKzioMqUyLWrMihpvsz',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id='toolu_017UryVwtsKsjonhFV3cgV3X',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions="""\
Always respond with a JSON object that's compatible with this schema:

{"properties": {"city": {"type": "string"}, "country": {"type": "string"}}, "required": ["city", "country"], "title": "CityLocation", "type": "object"}

Don't include any text or Markdown fencing before or after.\
""",
            ),
            ModelResponse(
                parts=[TextPart(content='{"city": "Mexico City", "country": "Mexico"}')],
                usage=Usage(
                    requests=1,
                    request_tokens=510,
                    response_tokens=17,
                    total_tokens=527,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 510,
                        'output_tokens': 17,
                    },
                ),
                model_name='claude-3-5-sonnet-20241022',
                timestamp=IsDatetime(),
                vendor_id='msg_014JeWCouH6DpdqzMTaBdkpJ',
            ),
        ]
    )


async def test_anthropic_prompted_output_multiple(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-3-5-sonnet-latest', provider=AnthropicProvider(api_key=anthropic_api_key))

    class CityLocation(BaseModel):
        city: str
        country: str

    class CountryLanguage(BaseModel):
        country: str
        language: str

    agent = Agent(m, output_type=PromptedOutput([CityLocation, CountryLanguage]))

    result = await agent.run('What is the largest city in Mexico?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in Mexico?',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions="""\
Always respond with a JSON object that's compatible with this schema:

{"type": "object", "properties": {"result": {"anyOf": [{"type": "object", "properties": {"kind": {"type": "string", "const": "CityLocation"}, "data": {"properties": {"city": {"type": "string"}, "country": {"type": "string"}}, "required": ["city", "country"], "type": "object"}}, "required": ["kind", "data"], "additionalProperties": false, "title": "CityLocation"}, {"type": "object", "properties": {"kind": {"type": "string", "const": "CountryLanguage"}, "data": {"properties": {"country": {"type": "string"}, "language": {"type": "string"}}, "required": ["country", "language"], "type": "object"}}, "required": ["kind", "data"], "additionalProperties": false, "title": "CountryLanguage"}]}}, "required": ["result"], "additionalProperties": false}

Don't include any text or Markdown fencing before or after.\
""",
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='{"result": {"kind": "CityLocation", "data": {"city": "Mexico City", "country": "Mexico"}}}'
                    )
                ],
                usage=Usage(
                    requests=1,
                    request_tokens=281,
                    response_tokens=31,
                    total_tokens=312,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 281,
                        'output_tokens': 31,
                    },
                ),
                model_name='claude-3-5-sonnet-20241022',
                timestamp=IsDatetime(),
                vendor_id='msg_013ttUi3HCcKt7PkJpoWs5FT',
            ),
        ]
    )


async def test_anthropic_native_output(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-3-5-sonnet-latest', provider=AnthropicProvider(api_key=anthropic_api_key))

    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent(m, output_type=NativeOutput(CityLocation))

    with pytest.raises(UserError, match='Native structured output is not supported by the model.'):
        await agent.run('What is the largest city in the user country?')


async def test_anthropic_tool_with_thinking(allow_model_requests: None, anthropic_api_key: str):
    """When using thinking with tool calls in Anthropic, we need to send the thinking part back to the provider.

    This tests the issue raised in https://github.com/pydantic/pydantic-ai/issues/2040.
    """
    m = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(api_key=anthropic_api_key))
    settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 3000})
    agent = Agent(m, model_settings=settings)

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run('What is the largest city in the user country?')
    assert result.output == snapshot("""\
Based on the information that you're in Mexico, the largest city in your country is **Mexico City** (Ciudad de México). \n\

Mexico City is not only the largest city in Mexico but also one of the largest metropolitan areas in the world, with a metropolitan population of over 21 million people. The city proper has a population of approximately 9 million people and serves as the capital and political, cultural, and economic center of Mexico.\
""")


async def test_anthropic_web_search_tool_pass_history_back(env: TestEnv, allow_model_requests: None):
    """Test passing web search tool history back to Anthropic."""
    # Create the first mock response with server tool blocks
    first_response = completion_message(
        [
            BetaTextBlock(text='Let me search for the current date.', type='text'),
            BetaServerToolUseBlock(
                id='server_tool_123', name='web_search', input={'query': 'current date today'}, type='server_tool_use'
            ),
            BetaWebSearchToolResultBlock(
                tool_use_id='server_tool_123',
                type='web_search_tool_result',
                content=[
                    BetaWebSearchResultBlock(
                        title='Current Date and Time',
                        url='https://example.com/date',
                        type='web_search_result',
                        encrypted_content='dummy_encrypted_content',
                    )
                ],
            ),
            BetaTextBlock(text='Today is January 2, 2025.', type='text'),
        ],
        BetaUsage(input_tokens=10, output_tokens=20),
    )

    # Create the second mock response that references the history
    second_response = completion_message(
        [BetaTextBlock(text='The web search result showed that today is January 2, 2025.', type='text')],
        BetaUsage(input_tokens=50, output_tokens=30),
    )

    mock_client = MockAnthropic.create_mock([first_response, second_response])
    m = AnthropicModel('claude-3-5-sonnet-latest', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m, builtin_tools=[WebSearchTool()])

    # First run to get server tool history
    result = await agent.run('What day is today?')

    # Verify we have server tool parts in the history
    server_tool_calls = [p for m in result.all_messages() for p in m.parts if isinstance(p, BuiltinToolCallPart)]
    server_tool_returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, BuiltinToolReturnPart)]
    assert len(server_tool_calls) == 1
    assert len(server_tool_returns) == 1
    assert server_tool_calls[0].tool_name == 'web_search'
    assert server_tool_returns[0].tool_name == 'web_search_tool_result'

    # Pass the history back to another Anthropic agent run
    agent2 = Agent(m)
    result2 = await agent2.run('What was the web search result?', message_history=result.all_messages())
    assert result2.output == 'The web search result showed that today is January 2, 2025.'


async def test_anthropic_code_execution_tool_pass_history_back(env: TestEnv, allow_model_requests: None):
    """Test passing code execution tool history back to Anthropic."""
    # Create the first mock response with server tool blocks
    first_response = completion_message(
        [
            BetaTextBlock(text='Let me calculate 2 + 2.', type='text'),
            BetaServerToolUseBlock(
                id='server_tool_456', name='code_execution', input={'code': 'print(2 + 2)'}, type='server_tool_use'
            ),
            BetaCodeExecutionToolResultBlock(
                tool_use_id='server_tool_456',
                type='code_execution_tool_result',
                content=BetaCodeExecutionResultBlock(
                    content=[],
                    return_code=0,
                    stderr='',
                    stdout='4\n',
                    type='code_execution_result',
                ),
            ),
            BetaTextBlock(text='The result is 4.', type='text'),
        ],
        BetaUsage(input_tokens=10, output_tokens=20),
    )

    # Create the second mock response that references the history
    second_response = completion_message(
        [BetaTextBlock(text='The code execution returned the result: 4', type='text')],
        BetaUsage(input_tokens=50, output_tokens=30),
    )

    mock_client = MockAnthropic.create_mock([first_response, second_response])
    m = AnthropicModel('claude-3-5-sonnet-latest', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m, builtin_tools=[CodeExecutionTool()])

    # First run to get server tool history
    result = await agent.run('What is 2 + 2?')

    # Verify we have server tool parts in the history
    server_tool_calls = [p for m in result.all_messages() for p in m.parts if isinstance(p, BuiltinToolCallPart)]
    server_tool_returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, BuiltinToolReturnPart)]
    assert len(server_tool_calls) == 1
    assert len(server_tool_returns) == 1
    assert server_tool_calls[0].tool_name == 'code_execution'
    assert server_tool_returns[0].tool_name == 'code_execution_tool_result'

    # Pass the history back to another Anthropic agent run
    agent2 = Agent(m)
    result2 = await agent2.run('What was the code execution result?', message_history=result.all_messages())
    assert result2.output == 'The code execution returned the result: 4'


async def test_anthropic_unsupported_server_tool_name_error(anthropic_api_key: str):
    """Test that unsupported server tool names raise an error."""

    model = AnthropicModel('claude-3-5-sonnet-latest', provider=AnthropicProvider(api_key=anthropic_api_key))

    # Create a message with an unsupported server tool name
    messages: list[ModelMessage] = [
        ModelResponse(
            parts=[
                BuiltinToolReturnPart(
                    tool_name='unsupported_tool',  # This should trigger the error
                    content='some content',
                    tool_call_id='test_id',
                    provider_name='anthropic',  # Need to set provider_name for validation
                )
            ]
        )
    ]

    # This should raise a ValueError
    with pytest.raises(ValueError, match='Unsupported tool name: unsupported_tool'):
        await model._map_message(messages)  # type: ignore[attr-defined]


async def test_anthropic_web_search_tool_stream(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m, instructions='You are a helpful assistant.', builtin_tools=[WebSearchTool()])

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='Give me the top 3 news in the world today.') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert event_parts.pop(0) == snapshot(
        PartStartEvent(index=0, part=TextPart(content="I'll search for the latest world"))
    )
    assert event_parts.pop(0) == snapshot(FinalResultEvent(tool_name=None, tool_call_id=None))
    assert ''.join(event.delta.content_delta for event in event_parts) == snapshot("""\
 news to get you the top 3 stories from today.Let me search for more specific and recent global news stories from today.Let me search for more specific global news stories from today.Based on my search results, here are the top 3 global news stories from today, July 16, 2025:

## 1. Trump's Ukraine Special Envoy Visits Kyiv

U.S. President Donald Trump's special envoy to Ukraine, retired Lt. Gen. Keith Kellogg, arrived in Kyiv on Monday. This high-profile diplomatic visit comes as tensions continue over the ongoing conflict, with President Trump threatening to punish Russia with heavy tariffs on countries that trade with Moscow if the Kremlin fails to reach a ceasefire deal with Ukraine, while promising Kyiv weapons.

## 2. EU Trade Ministers Meet Over U.S. Tariffs

European trade ministers are meeting in Brussels after U.S. President Donald Trump announced 30% tariffs on the European Union. The EU is America's biggest business partner and the world's largest trading block. The U.S. decision will have repercussions for governments, companies and consumers on both sides of the Atlantic.

## 3. Syria Violence Escalates

Clashes between Druze militias and Sunni Bedouin clans in Syria's Sweida province have killed more than 30 people and injured nearly 100. This represents a significant escalation in sectarian violence in the region.

Additional notable stories include Vietnam's plan to ban fossil-fuel motorcycles in the heart of Hanoi starting July 2026, aiming to cut air pollution and move toward cleaner transport, and ongoing restoration efforts for Copenhagen's Old Stock Exchange, which is taking shape 15 months after a fire destroyed more than half of the 400-year-old building.\
""")
