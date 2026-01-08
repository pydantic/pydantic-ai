from __future__ import annotations as _annotations

import json
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from functools import cached_property
from typing import Annotated, Any, TypeVar, cast

import httpx
import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel, Field

from pydantic_ai import (
    Agent,
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    CachePoint,
    DocumentUrl,
    FinalResultEvent,
    ImageUrl,
    ModelAPIError,
    ModelHTTPError,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelRetry,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
    UsageLimitExceeded,
    UserPromptPart,
)
from pydantic_ai.builtin_tools import CodeExecutionTool, MCPServerTool, MemoryTool, WebFetchTool, WebSearchTool
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    BuiltinToolCallEvent,  # pyright: ignore[reportDeprecated]
    BuiltinToolResultEvent,  # pyright: ignore[reportDeprecated]
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.output import NativeOutput, PromptedOutput, TextOutput, ToolOutput
from pydantic_ai.result import RunUsage
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage, UsageLimits

from ..conftest import IsDatetime, IsInstance, IsNow, IsStr, TestEnv, raise_if_exception, try_import
from ..parts_from_messages import part_types_from_messages
from .mock_async_stream import MockAsyncStream

with try_import() as imports_successful:
    from anthropic import NOT_GIVEN, APIConnectionError, APIStatusError, AsyncAnthropic
    from anthropic.lib.tools import BetaAbstractMemoryTool
    from anthropic.resources.beta import AsyncBeta
    from anthropic.types.beta import (
        BetaCodeExecutionResultBlock,
        BetaCodeExecutionToolResultBlock,
        BetaContentBlock,
        BetaDirectCaller,
        BetaInputJSONDelta,
        BetaMemoryTool20250818CreateCommand,
        BetaMemoryTool20250818DeleteCommand,
        BetaMemoryTool20250818InsertCommand,
        BetaMemoryTool20250818RenameCommand,
        BetaMemoryTool20250818StrReplaceCommand,
        BetaMemoryTool20250818ViewCommand,
        BetaMessage,
        BetaMessageDeltaUsage,
        BetaMessageTokensCount,
        BetaRawContentBlockDeltaEvent,
        BetaRawContentBlockStartEvent,
        BetaRawContentBlockStopEvent,
        BetaRawMessageDeltaEvent,
        BetaRawMessageStartEvent,
        BetaRawMessageStopEvent,
        BetaRawMessageStreamEvent,
        BetaServerToolUseBlock,
        BetaTextBlock,
        BetaTextDelta,
        BetaToolUseBlock,
        BetaUsage,
        BetaWebSearchResultBlock,
        BetaWebSearchToolResultBlock,
    )
    from anthropic.types.beta.beta_container import BetaContainer
    from anthropic.types.beta.beta_container_params import BetaContainerParams
    from anthropic.types.beta.beta_raw_message_delta_event import Delta

    from pydantic_ai.models.anthropic import (
        AnthropicModel,
        AnthropicModelSettings,
        _map_usage,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.openai import OpenAIProvider

    MockAnthropicMessage = BetaMessage | Exception
    MockRawMessageStreamEvent = BetaRawMessageStreamEvent | Exception

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
    pytest.mark.filterwarnings(
        'ignore:`BuiltinToolCallEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolCallPart` instead.:DeprecationWarning'
    ),
    pytest.mark.filterwarnings(
        'ignore:`BuiltinToolResultEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolReturnPart` instead.:DeprecationWarning'
    ),
]

# Type variable for generic AsyncStream
T = TypeVar('T')


def test_init():
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(api_key='foobar'))
    assert isinstance(m.client, AsyncAnthropic)
    assert m.client.api_key == 'foobar'
    assert m.model_name == 'claude-haiku-4-5'
    assert m.system == 'anthropic'
    assert m.base_url == 'https://api.anthropic.com'


@dataclass
class MockAnthropic:
    messages_: MockAnthropicMessage | Sequence[MockAnthropicMessage] | None = None
    stream: Sequence[MockRawMessageStreamEvent] | Sequence[Sequence[MockRawMessageStreamEvent]] | None = None
    index = 0
    chat_completion_kwargs: list[dict[str, Any]] = field(default_factory=list)
    base_url: str = 'https://api.anthropic.com'

    @cached_property
    def beta(self) -> AsyncBeta:
        return cast(AsyncBeta, self)

    @cached_property
    def messages(self) -> Any:
        return type('Messages', (), {'create': self.messages_create, 'count_tokens': self.messages_count_tokens})

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
                response = MockAsyncStream(iter(cast(list[MockRawMessageStreamEvent], self.stream)))
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

    async def messages_count_tokens(self, *_args: Any, **kwargs: Any) -> BetaMessageTokensCount:
        # check if we are configured to raise an exception
        if self.messages_ is not None:
            raise_if_exception(self.messages_ if not isinstance(self.messages_, Sequence) else self.messages_[0])

        # record the kwargs used
        self.chat_completion_kwargs.append({k: v for k, v in kwargs.items() if v is not NOT_GIVEN})

        return BetaMessageTokensCount(input_tokens=10)


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
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    result = await agent.run('hello')
    assert result.output == 'world'
    assert result.usage() == snapshot(
        RunUsage(
            requests=1,
            input_tokens=5,
            output_tokens=10,
            details={'input_tokens': 5, 'output_tokens': 10},
        )
    )
    # reset the index so we get the same response again
    mock_client.index = 0  # type: ignore

    result = await agent.run('hello', message_history=result.new_messages())
    assert result.output == 'world'
    assert result.usage() == snapshot(
        RunUsage(
            requests=1,
            input_tokens=5,
            output_tokens=10,
            details={'input_tokens': 5, 'output_tokens': 10},
        )
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='world')],
                usage=RequestUsage(input_tokens=5, output_tokens=10, details={'input_tokens': 5, 'output_tokens': 10}),
                model_name='claude-3-5-haiku-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='world')],
                usage=RequestUsage(input_tokens=5, output_tokens=10, details={'input_tokens': 5, 'output_tokens': 10}),
                model_name='claude-3-5-haiku-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
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
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    result = await agent.run('hello')
    assert result.output == 'world'
    assert result.usage() == snapshot(
        RunUsage(
            requests=1,
            input_tokens=13,
            cache_write_tokens=4,
            cache_read_tokens=6,
            output_tokens=5,
            details={
                'input_tokens': 3,
                'output_tokens': 5,
                'cache_creation_input_tokens': 4,
                'cache_read_input_tokens': 6,
            },
        )
    )
    last_message = result.all_messages()[-1]
    assert isinstance(last_message, ModelResponse)
    assert last_message.cost().total_price == snapshot(Decimal('0.00002688'))


async def test_cache_point_adds_cache_control(allow_model_requests: None):
    """Test that CachePoint correctly adds cache_control to content blocks.

    By default, CachePoint uses ttl='5m'. For non-Bedrock clients, the ttl field is included.
    """
    c = completion_message(
        [BetaTextBlock(text='response', type='text')],
        usage=BetaUsage(input_tokens=3, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    # Test with CachePoint after text content (default ttl='5m')
    await agent.run(['Some context to cache', CachePoint(), 'Now the question'])

    # Verify cache_control was added with default ttl='5m'
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    messages = completion_kwargs['messages']
    assert messages == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {
                        'text': 'Some context to cache',
                        'type': 'text',
                        'cache_control': {'type': 'ephemeral', 'ttl': '5m'},
                    },
                    {'text': 'Now the question', 'type': 'text'},
                ],
            }
        ]
    )


async def test_cache_point_multiple_markers(allow_model_requests: None):
    """Test multiple CachePoint markers in a single prompt."""
    c = completion_message(
        [BetaTextBlock(text='response', type='text')],
        usage=BetaUsage(input_tokens=3, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    await agent.run(['First chunk', CachePoint(), 'Second chunk', CachePoint(), 'Question'])

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    content = completion_kwargs['messages'][0]['content']

    # Default ttl='5m' for non-Bedrock clients
    assert content == snapshot(
        [
            {'text': 'First chunk', 'type': 'text', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}},
            {'text': 'Second chunk', 'type': 'text', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}},
            {'text': 'Question', 'type': 'text'},
        ]
    )


async def test_cache_point_as_first_content_raises_error(allow_model_requests: None):
    """Test that CachePoint as first content raises UserError."""
    c = completion_message(
        [BetaTextBlock(text='response', type='text')],
        usage=BetaUsage(input_tokens=3, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    with pytest.raises(
        UserError,
        match='CachePoint cannot be the first content in a user message - there must be previous content to attach the CachePoint to.',
    ):
        await agent.run([CachePoint(), 'This should fail'])


async def test_cache_point_with_image_content(allow_model_requests: None):
    """Test CachePoint works with image content."""
    c = completion_message(
        [BetaTextBlock(text='response', type='text')],
        usage=BetaUsage(input_tokens=3, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    await agent.run(
        [
            ImageUrl('https://example.com/image.jpg'),
            CachePoint(),
            'What is in this image?',
        ]
    )

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    content = completion_kwargs['messages'][0]['content']

    # Default ttl='5m' for non-Bedrock clients
    assert content == snapshot(
        [
            {
                'source': {'type': 'url', 'url': 'https://example.com/image.jpg'},
                'type': 'image',
                'cache_control': {'type': 'ephemeral', 'ttl': '5m'},
            },
            {'text': 'What is in this image?', 'type': 'text'},
        ]
    )


async def test_cache_point_in_otel_message_parts(allow_model_requests: None):
    """Test that CachePoint is handled correctly in otel message parts conversion."""
    from pydantic_ai.agent import InstrumentationSettings
    from pydantic_ai.messages import UserPromptPart

    # Create a UserPromptPart with CachePoint
    part = UserPromptPart(content=['text before', CachePoint(), 'text after'])

    # Convert to otel message parts
    settings = InstrumentationSettings(include_content=True)
    otel_parts = part.otel_message_parts(settings)

    # Should have 2 text parts, CachePoint is skipped
    assert otel_parts == snapshot(
        [{'type': 'text', 'content': 'text before'}, {'type': 'text', 'content': 'text after'}]
    )


def test_cache_control_unsupported_param_type():
    """Test that cache control raises error for unsupported param types."""
    from unittest.mock import MagicMock

    from pydantic_ai.exceptions import UserError
    from pydantic_ai.models.anthropic import AnthropicModel

    # Create a mock model instance
    mock_client = MagicMock()
    mock_client.__class__.__name__ = 'AsyncAnthropic'
    mock_client.base_url = 'https://api.anthropic.com'
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))

    # Create a list with an unsupported param type (thinking)
    params: list[dict[str, Any]] = [{'type': 'thinking', 'source': {'data': 'test'}}]

    with pytest.raises(UserError, match='Cache control not supported for param type: thinking'):
        m._add_cache_control_to_last_param(params)  # type: ignore[arg-type]  # Testing internal method


def test_build_cache_control_bedrock_omits_ttl():
    """Test that _build_cache_control automatically omits TTL for Bedrock clients."""
    from unittest.mock import MagicMock

    from anthropic import AsyncAnthropicBedrock

    # Create a mock client using spec=AsyncAnthropicBedrock for isinstance check
    mock_bedrock_client = MagicMock(spec=AsyncAnthropicBedrock)
    mock_bedrock_client.base_url = 'https://bedrock.amazonaws.com'

    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_bedrock_client))

    # Verify cache_control is built without TTL for Bedrock
    cache_control = m._build_cache_control('5m')  # pyright: ignore[reportPrivateUsage]
    assert cache_control == {'type': 'ephemeral'}  # No 'ttl' field

    cache_control_1h = m._build_cache_control('1h')  # pyright: ignore[reportPrivateUsage]
    assert cache_control_1h == {'type': 'ephemeral'}  # TTL still omitted


def test_build_cache_control_standard_client_includes_ttl():
    """Test that _build_cache_control includes TTL for standard Anthropic clients."""
    from unittest.mock import MagicMock

    # Create a mock client that looks like standard AsyncAnthropic
    mock_client = MagicMock()
    mock_client.__class__.__name__ = 'AsyncAnthropic'
    mock_client.base_url = 'https://api.anthropic.com'

    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))

    # Verify cache_control includes TTL for standard clients
    cache_control = m._build_cache_control('5m')  # pyright: ignore[reportPrivateUsage]
    assert cache_control == {'type': 'ephemeral', 'ttl': '5m'}

    cache_control_1h = m._build_cache_control('1h')  # pyright: ignore[reportPrivateUsage]
    assert cache_control_1h == {'type': 'ephemeral', 'ttl': '1h'}


async def test_cache_point_with_5m_ttl(allow_model_requests: None):
    """Test that CachePoint with explicit ttl='5m' includes the ttl field."""
    c = completion_message(
        [BetaTextBlock(text='response', type='text')],
        usage=BetaUsage(input_tokens=3, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    # Test with explicit CachePoint(ttl='5m')
    await agent.run(['Some context to cache', CachePoint(ttl='5m'), 'Now the question'])

    # Verify cache_control was added with 5m ttl
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    messages = completion_kwargs['messages']
    assert messages == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {
                        'text': 'Some context to cache',
                        'type': 'text',
                        'cache_control': {'type': 'ephemeral', 'ttl': '5m'},
                    },
                    {'text': 'Now the question', 'type': 'text'},
                ],
            }
        ]
    )


async def test_cache_point_with_1h_ttl(allow_model_requests: None):
    """Test that CachePoint with ttl='1h' correctly sets the TTL."""
    c = completion_message(
        [BetaTextBlock(text='response', type='text')],
        usage=BetaUsage(input_tokens=3, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    # Test with CachePoint(ttl='1h')
    await agent.run(['Some context to cache', CachePoint(ttl='1h'), 'Now the question'])

    # Verify cache_control was added with 1h ttl
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    messages = completion_kwargs['messages']
    assert messages == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {
                        'text': 'Some context to cache',
                        'type': 'text',
                        'cache_control': {'type': 'ephemeral', 'ttl': '1h'},
                    },
                    {'text': 'Now the question', 'type': 'text'},
                ],
            }
        ]
    )


async def test_anthropic_cache_tools(allow_model_requests: None):
    """Test that anthropic_cache_tool_definitions adds cache_control to last tool."""
    c = completion_message(
        [BetaTextBlock(text='Tool result', type='text')],
        usage=BetaUsage(input_tokens=10, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(
        m,
        system_prompt='Test system prompt',
        model_settings=AnthropicModelSettings(anthropic_cache_tool_definitions=True),
    )

    @agent.tool_plain
    def tool_one() -> str:  # pragma: no cover
        return 'one'

    @agent.tool_plain
    def tool_two() -> str:  # pragma: no cover
        return 'two'

    await agent.run('test prompt')

    # Verify cache_control was added to the last tool
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    tools = completion_kwargs['tools']
    has_strict_tools = any('strict' in tool for tool in tools)  # we only ever set strict: True
    assert has_strict_tools is False  # ensure strict is not set for haiku-4-5
    assert tools == snapshot(
        [
            {
                'name': 'tool_one',
                'description': '',
                'input_schema': {'additionalProperties': False, 'properties': {}, 'type': 'object'},
            },
            {
                'name': 'tool_two',
                'description': '',
                'input_schema': {'additionalProperties': False, 'properties': {}, 'type': 'object'},
                'cache_control': {'type': 'ephemeral', 'ttl': '5m'},
            },
        ]
    )


async def test_anthropic_cache_instructions(allow_model_requests: None):
    """Test that anthropic_cache_instructions adds cache_control to system prompt."""
    c = completion_message(
        [BetaTextBlock(text='Response', type='text')],
        usage=BetaUsage(input_tokens=10, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(
        m,
        system_prompt='This is a test system prompt with instructions.',
        model_settings=AnthropicModelSettings(anthropic_cache_instructions=True),
    )

    await agent.run('test prompt')

    # Verify system is a list with cache_control on last block
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    system = completion_kwargs['system']
    assert system == snapshot(
        [
            {
                'type': 'text',
                'text': 'This is a test system prompt with instructions.',
                'cache_control': {'type': 'ephemeral', 'ttl': '5m'},
            }
        ]
    )


async def test_anthropic_cache_tools_and_instructions(allow_model_requests: None):
    """Test that both cache settings work together."""
    c = completion_message(
        [BetaTextBlock(text='Response', type='text')],
        usage=BetaUsage(input_tokens=10, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(
        m,
        system_prompt='System instructions to cache.',
        model_settings=AnthropicModelSettings(
            anthropic_cache_tool_definitions=True,
            anthropic_cache_instructions=True,
        ),
    )

    @agent.tool_plain
    def my_tool(value: str) -> str:  # pragma: no cover
        return f'Result: {value}'

    await agent.run('test prompt')

    # Verify both have cache_control
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    tools = completion_kwargs['tools']
    system = completion_kwargs['system']
    has_strict_tools = any('strict' in tool for tool in tools)  # we only ever set strict: True
    assert has_strict_tools is False  # ensure strict is not set for haiku-4-5
    assert tools == snapshot(
        [
            {
                'name': 'my_tool',
                'description': '',
                'input_schema': {
                    'additionalProperties': False,
                    'properties': {'value': {'type': 'string'}},
                    'required': ['value'],
                    'type': 'object',
                },
                'cache_control': {'type': 'ephemeral', 'ttl': '5m'},
            }
        ]
    )
    assert system == snapshot(
        [{'type': 'text', 'text': 'System instructions to cache.', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}}]
    )


async def test_anthropic_cache_with_custom_ttl(allow_model_requests: None):
    """Test that cache settings support custom TTL values ('5m' or '1h')."""
    c = completion_message(
        [BetaTextBlock(text='Response', type='text')],
        usage=BetaUsage(input_tokens=10, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(
        m,
        system_prompt='System instructions to cache.',
        model_settings=AnthropicModelSettings(
            anthropic_cache_tool_definitions='1h',  # Custom 1h TTL
            anthropic_cache_instructions='5m',  # Explicit 5m TTL
        ),
    )

    @agent.tool_plain
    def my_tool(value: str) -> str:  # pragma: no cover
        return f'Result: {value}'

    await agent.run('test prompt')

    # Verify custom TTL values are applied
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    tools = completion_kwargs['tools']
    system = completion_kwargs['system']

    # Tool definitions should have 1h TTL
    assert tools[0]['cache_control'] == snapshot({'type': 'ephemeral', 'ttl': '1h'})
    # System instructions should have 5m TTL
    assert system[0]['cache_control'] == snapshot({'type': 'ephemeral', 'ttl': '5m'})


async def test_anthropic_incompatible_schema_disables_auto_strict(allow_model_requests: None):
    """Ensure strict mode is disabled when Anthropic cannot enforce the tool schema."""
    c = completion_message(
        [BetaTextBlock(text='Done', type='text')],
        usage=BetaUsage(input_tokens=8, output_tokens=3),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    @agent.tool_plain
    def constrained_tool(value: Annotated[str, Field(min_length=2)]) -> str:  # pragma: no cover
        return value

    await agent.run('hello')

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert 'strict' not in completion_kwargs['tools'][0]


async def test_beta_header_merge_builtin_tools_and_native_output(allow_model_requests: None):
    """Verify beta headers merge from custom headers, builtin tools, and native output."""
    c = completion_message(
        [BetaTextBlock(text='{"city": "Mexico City", "country": "Mexico"}', type='text')],
        BetaUsage(input_tokens=5, output_tokens=10),
    )
    mock_client = MockAnthropic.create_mock(c)

    class CityLocation(BaseModel):
        """A city and its country."""

        city: str
        country: str

    model = AnthropicModel(
        'claude-sonnet-4-5',
        provider=AnthropicProvider(anthropic_client=mock_client),
        settings=AnthropicModelSettings(extra_headers={'anthropic-beta': 'custom-feature-1, custom-feature-2'}),
    )

    agent = Agent(
        model,
        builtin_tools=[MemoryTool()],
        output_type=NativeOutput(CityLocation),
    )

    @agent.tool_plain
    def memory(**command: Any) -> Any:  # pragma: no cover
        return 'memory response'

    await agent.run('What is the capital of France?')

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    betas = completion_kwargs['betas']

    assert betas == snapshot(
        [
            'context-management-2025-06-27',
            'custom-feature-1',
            'custom-feature-2',
            'structured-outputs-2025-11-13',
        ]
    )


async def test_model_settings_reusable_with_beta_headers(allow_model_requests: None):
    """Verify that model_settings with extra_headers can be reused across multiple runs.

    This test ensures that the beta header extraction doesn't mutate the original model_settings,
    allowing the same settings to be used for multiple agent runs.
    """
    c = completion_message(
        [BetaTextBlock(text='Hello!', type='text')],
        BetaUsage(input_tokens=5, output_tokens=10),
    )
    mock_client = MockAnthropic.create_mock(c)

    model_settings = AnthropicModelSettings(extra_headers={'anthropic-beta': 'custom-feature-1, custom-feature-2'})

    model = AnthropicModel(
        'claude-sonnet-4-5',
        provider=AnthropicProvider(anthropic_client=mock_client),
        settings=model_settings,
    )

    agent = Agent(model)

    # First run
    await agent.run('Hello')

    # Verify the original model_settings is not mutated
    assert model_settings.get('extra_headers') == {'anthropic-beta': 'custom-feature-1, custom-feature-2'}

    # Second run should work with the same beta headers
    await agent.run('Hello again')

    # Verify again after second run
    assert model_settings.get('extra_headers') == {'anthropic-beta': 'custom-feature-1, custom-feature-2'}

    # Verify both runs had the correct betas
    all_kwargs = get_mock_chat_completion_kwargs(mock_client)
    assert len(all_kwargs) == 2
    for completion_kwargs in all_kwargs:
        betas = completion_kwargs['betas']
        assert 'custom-feature-1' in betas
        assert 'custom-feature-2' in betas


async def test_anthropic_mixed_strict_tool_run(allow_model_requests: None, anthropic_api_key: str):
    """Exercise both strict=True and strict=False tool definitions against the live API."""
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(
        m,
        system_prompt='Always call `country_source` first, then call `capital_lookup` with that result before replying.',
    )

    @agent.tool_plain(strict=True)
    async def country_source() -> str:
        return 'Japan'

    capital_called = {'value': False}

    @agent.tool_plain(strict=False)
    async def capital_lookup(country: str) -> str:
        capital_called['value'] = True
        if country == 'Japan':
            return 'Tokyo'
        return f'Unknown capital for {country}'  # pragma: no cover

    result = await agent.run('Use the registered tools and respond exactly as `Capital: <city>`.')
    assert capital_called['value'] is True
    assert result.output.startswith('Capital:')
    assert any(
        isinstance(part, ToolCallPart) and part.tool_name == 'capital_lookup'
        for message in result.all_messages()
        if isinstance(message, ModelResponse)
        for part in message.parts
    )


async def test_anthropic_cache_messages(allow_model_requests: None):
    """Test that anthropic_cache_messages caches only the last message."""
    c = completion_message(
        [BetaTextBlock(text='Response', type='text')],
        usage=BetaUsage(input_tokens=10, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(
        m,
        system_prompt='System instructions to cache.',
        model_settings=AnthropicModelSettings(
            anthropic_cache_messages=True,
        ),
    )

    await agent.run('User message')

    # Verify only last message has cache_control, not system
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    system = completion_kwargs['system']
    messages = completion_kwargs['messages']

    # System should NOT have cache_control (should be a plain string)
    assert system == snapshot('System instructions to cache.')

    # Last message content should have cache_control
    assert messages[-1]['content'][-1] == snapshot(
        {'type': 'text', 'text': 'User message', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}}
    )


async def test_anthropic_cache_messages_with_custom_ttl(allow_model_requests: None):
    """Test that anthropic_cache_messages supports custom TTL values."""
    c = completion_message(
        [BetaTextBlock(text='Response', type='text')],
        usage=BetaUsage(input_tokens=10, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(
        m,
        system_prompt='System instructions.',
        model_settings=AnthropicModelSettings(
            anthropic_cache_messages='1h',  # Custom 1h TTL
        ),
    )

    await agent.run('User message')

    # Verify use 1h TTL
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    messages = completion_kwargs['messages']

    assert messages[-1]['content'][-1]['cache_control'] == snapshot({'type': 'ephemeral', 'ttl': '1h'})


async def test_limit_cache_points_with_cache_messages(allow_model_requests: None):
    """Test that cache points are limited when using cache_messages + CachePoint markers."""
    c = completion_message(
        [BetaTextBlock(text='Response', type='text')],
        usage=BetaUsage(input_tokens=10, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(
        m,
        system_prompt='System instructions.',
        model_settings=AnthropicModelSettings(
            anthropic_cache_messages=True,  # Uses 1 cache point
        ),
    )

    # Add 4 CachePoint markers (total would be 5: 1 from cache_messages + 4 from markers)
    # Only 3 CachePoint markers should be kept (newest ones)
    await agent.run(
        [
            'Context 1',
            CachePoint(),  # Oldest, should be removed
            'Context 2',
            CachePoint(),  # Should be kept
            'Context 3',
            CachePoint(),  # Should be kept
            'Context 4',
            CachePoint(),  # Should be kept
            'Question',
        ]
    )

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    messages = completion_kwargs['messages']

    # Count cache_control occurrences in messages
    cache_count = 0
    for msg in messages:
        for block in msg['content']:
            if 'cache_control' in block:
                cache_count += 1

    # anthropic_cache_messages uses 1 cache point (last message only)
    # With 4 CachePoint markers, we'd have 5 total
    # Limit is 4, so 1 oldest CachePoint should be removed
    # Result: 3 cache points from CachePoint markers + 1 from cache_messages = 4 total
    assert cache_count == 4


async def test_limit_cache_points_all_settings(allow_model_requests: None):
    """Test cache point limiting with all cache settings enabled."""
    c = completion_message(
        [BetaTextBlock(text='Response', type='text')],
        usage=BetaUsage(input_tokens=10, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))

    agent = Agent(
        m,
        system_prompt='System instructions.',
        model_settings=AnthropicModelSettings(
            anthropic_cache_instructions=True,  # 1 cache point
            anthropic_cache_tool_definitions=True,  # 1 cache point
        ),
    )

    @agent.tool_plain
    def my_tool() -> str:  # pragma: no cover
        return 'result'

    # Add 3 CachePoint markers (total would be 5: 2 from settings + 3 from markers)
    # Only 2 CachePoint markers should be kept
    await agent.run(
        [
            'Context 1',
            CachePoint(),  # Oldest, should be removed
            'Context 2',
            CachePoint(),  # Should be kept
            'Context 3',
            CachePoint(),  # Should be kept
            'Question',
        ]
    )

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    messages = completion_kwargs['messages']

    # Count cache_control in messages (excluding system and tools)
    cache_count = 0
    for msg in messages:
        for block in msg['content']:
            if 'cache_control' in block:
                cache_count += 1

    # Should have exactly 2 cache points in messages
    # (4 total - 1 system - 1 tool = 2 available for messages)
    assert cache_count == 2


async def test_async_request_text_response(allow_model_requests: None):
    c = completion_message(
        [BetaTextBlock(text='world', type='text')],
        usage=BetaUsage(input_tokens=3, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    result = await agent.run('hello')
    assert result.output == 'world'
    assert result.usage() == snapshot(
        RunUsage(
            requests=1,
            input_tokens=3,
            output_tokens=5,
            details={'input_tokens': 3, 'output_tokens': 5},
        )
    )


async def test_request_structured_response(allow_model_requests: None):
    c = completion_message(
        [BetaToolUseBlock(id='123', input={'response': [1, 2, 3]}, name='final_result', type='tool_use')],
        usage=BetaUsage(input_tokens=3, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m, output_type=list[int])

    result = await agent.run('hello')
    assert result.output == [1, 2, 3]
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args={'response': [1, 2, 3]},
                        tool_call_id='123',
                    )
                ],
                usage=RequestUsage(input_tokens=3, output_tokens=5, details={'input_tokens': 3, 'output_tokens': 5}),
                model_name='claude-3-5-haiku-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='123',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
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
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
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
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_location',
                        args={'loc_name': 'San Francisco'},
                        tool_call_id='1',
                    )
                ],
                usage=RequestUsage(input_tokens=2, output_tokens=1, details={'input_tokens': 2, 'output_tokens': 1}),
                model_name='claude-3-5-haiku-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Wrong location, please try again',
                        tool_name='get_location',
                        tool_call_id='1',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_location',
                        args={'loc_name': 'London'},
                        tool_call_id='2',
                    )
                ],
                usage=RequestUsage(input_tokens=3, output_tokens=2, details={'input_tokens': 3, 'output_tokens': 2}),
                model_name='claude-3-5-haiku-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_location',
                        content='{"lat": 51, "lng": 0}',
                        tool_call_id='2',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='final response')],
                usage=RequestUsage(input_tokens=3, output_tokens=5, details={'input_tokens': 3, 'output_tokens': 5}),
                model_name='claude-3-5-haiku-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='123',
                finish_reason='stop',
                run_id=IsStr(),
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
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
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
    api_key = os.getenv('ANTHROPIC_API_KEY', 'mock-value')
    agent = Agent(
        AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(api_key=api_key)),
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
                content="I'll help you find out who is the youngest by retrieving information about each family member. I'll retrieve their entity information to compare their ages.",
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
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
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
                model='claude-3-5-haiku-123',
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
            usage=BetaMessageDeltaUsage(input_tokens=20, output_tokens=5),
        ),
        # Mark message as complete
        BetaRawMessageStopEvent(type='message_stop'),
    ]

    done_stream = [
        BetaRawMessageStartEvent(
            type='message_start',
            message=BetaMessage(
                id='msg_123',
                model='claude-3-5-haiku-123',
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
        BetaRawMessageDeltaEvent(
            type='message_delta',
            delta=Delta(stop_reason='end_turn'),
            usage=BetaMessageDeltaUsage(input_tokens=0, output_tokens=0),
        ),
        BetaRawMessageStopEvent(type='message_stop'),
    ]

    mock_client = MockAnthropic.create_stream_mock([stream, done_stream])
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    tool_called = False

    @agent.tool_plain
    async def my_tool(first: str, second: str) -> int:
        nonlocal tool_called
        tool_called = True
        return len(first) + len(second)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        chunks = [c async for c in result.stream_output(debounce_by=None)]

        # The tool output doesn't echo any content to the stream, so we only get the final payload once when
        # the block starts and once when it ends.
        assert chunks == snapshot(['FINAL_PAYLOAD', 'FINAL_PAYLOAD'])
        assert result.is_complete
        assert result.usage() == snapshot(
            RunUsage(
                requests=2,
                input_tokens=20,
                output_tokens=5,
                tool_calls=1,
                details={'input_tokens': 20, 'output_tokens': 5},
            )
        )
        assert tool_called
        async for response, is_last in result.stream_responses(debounce_by=None):
            if is_last:
                assert response == snapshot(
                    ModelResponse(
                        parts=[TextPart(content='FINAL_PAYLOAD')],
                        usage=RequestUsage(details={'input_tokens': 0, 'output_tokens': 0}),
                        model_name='claude-3-5-haiku-123',
                        timestamp=IsDatetime(),
                        provider_name='anthropic',
                        provider_url='https://api.anthropic.com',
                        provider_details={'finish_reason': 'end_turn'},
                        provider_response_id='msg_123',
                        finish_reason='stop',
                    )
                )


async def test_image_url_input(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m)

    result = await agent.run(
        [
            'What is this vegetable?',
            ImageUrl(url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg'),
        ]
    )
    assert result.output == snapshot(
        "This is a potato. It's a yellow/golden-colored potato with a smooth, slightly bumpy skin typical of many potato varieties. The potato appears to be a whole, unpeeled tuber with a classic oblong or oval shape. Potatoes are starchy root vegetables that are widely consumed around the world and can be prepared in many ways, such as boiling, baking, frying, or mashing."
    )


async def test_image_url_input_force_download(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m)

    result = await agent.run(
        [
            'What is this vegetable?',
            ImageUrl(
                url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg',
                force_download=True,
            ),
        ]
    )
    assert result.output == snapshot(
        """\
This is a **potato**, specifically a yellow or gold potato variety. You can identify it by its characteristic features:

- **Oval/round shape** with smooth skin
- **Golden-yellow color** with small dark spots or eyes
- **Starchy appearance** typical of potatoes

This appears to be a russet or similar yellow potato variety commonly used for cooking, baking, or making mashed potatoes.\
"""
    )


async def test_extra_headers(allow_model_requests: None, anthropic_api_key: str):
    # This test doesn't do anything, it's just here to ensure that calls with `extra_headers` don't cause errors, including type.
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(
        m,
        model_settings=AnthropicModelSettings(
            anthropic_metadata={'user_id': '123'}, extra_headers={'Extra-Header-Key': 'Extra-Header-Value'}
        ),
    )
    await agent.run('hello')


async def test_image_url_input_invalid_mime_type(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
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
        'This is a Great Horned Owl (Bubo virginianus), a large and powerful owl species native to the Americas. The image shows the owl perched on a log or branch, surrounded by soft yellow and green vegetation. The owl has distinctive ear tufts (the "horns"), large yellow eyes, and a mottled gray-brown plumage that provides excellent camouflage in woodland and grassland environments. Great Horned Owls are known for their impressive size, sharp talons, and nocturnal hunting habits. They are formidable predators that can hunt animals as large as skunks, rabbits, and even other birds of prey.'
    )


async def test_image_url_force_download() -> None:
    """Test that force_download=True calls download_item for ImageUrl."""
    from unittest.mock import AsyncMock, patch

    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='test-key'))

    with patch('pydantic_ai.models.anthropic.download_item', new_callable=AsyncMock) as mock_download:
        mock_download.return_value = {
            'data': b'\x89PNG\r\n\x1a\n fake image data',
            'content_type': 'image/png',
        }

        messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'Test image',
                            ImageUrl(
                                url='https://example.com/image.png',
                                media_type='image/png',
                                force_download=True,
                            ),
                        ]
                    )
                ]
            )
        ]

        await m._map_message(messages, ModelRequestParameters(), {})  # pyright: ignore[reportPrivateUsage,reportArgumentType]

        mock_download.assert_called_once()
        assert mock_download.call_args[0][0].url == 'https://example.com/image.png'


async def test_image_url_no_force_download() -> None:
    """Test that force_download=False does not call download_item for ImageUrl."""
    from unittest.mock import AsyncMock, patch

    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='test-key'))

    with patch('pydantic_ai.models.anthropic.download_item', new_callable=AsyncMock) as mock_download:
        messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'Test image',
                            ImageUrl(
                                url='https://example.com/image.png',
                                media_type='image/png',
                                force_download=False,
                            ),
                        ]
                    )
                ]
            )
        ]

        await m._map_message(messages, ModelRequestParameters(), {})  # pyright: ignore[reportPrivateUsage,reportArgumentType]

        mock_download.assert_not_called()


async def test_document_url_pdf_force_download() -> None:
    """Test that force_download=True calls download_item for DocumentUrl (PDF)."""
    from unittest.mock import AsyncMock, patch

    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='test-key'))

    with patch('pydantic_ai.models.anthropic.download_item', new_callable=AsyncMock) as mock_download:
        mock_download.return_value = {
            'data': b'%PDF-1.4 fake pdf data',
            'content_type': 'application/pdf',
        }

        messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'Test PDF',
                            DocumentUrl(
                                url='https://example.com/doc.pdf',
                                media_type='application/pdf',
                                force_download=True,
                            ),
                        ]
                    )
                ]
            )
        ]

        await m._map_message(messages, ModelRequestParameters(), {})  # pyright: ignore[reportPrivateUsage,reportArgumentType]

        mock_download.assert_called_once()
        assert mock_download.call_args[0][0].url == 'https://example.com/doc.pdf'


async def test_document_url_text_force_download() -> None:
    """Test that force_download=True calls download_item for DocumentUrl (text/plain)."""
    from unittest.mock import AsyncMock, patch

    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='test-key'))

    with patch('pydantic_ai.models.anthropic.download_item', new_callable=AsyncMock) as mock_download:
        mock_download.return_value = {
            'data': 'Sample text content',
            'content_type': 'text/plain',
        }

        messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'Test text file',
                            DocumentUrl(
                                url='https://example.com/doc.txt',
                                media_type='text/plain',
                                force_download=True,
                            ),
                        ]
                    )
                ]
            )
        ]

        await m._map_message(messages, ModelRequestParameters(), {})  # pyright: ignore[reportPrivateUsage,reportArgumentType]

        mock_download.assert_called_once()
        assert mock_download.call_args[0][0].url == 'https://example.com/doc.txt'


async def test_image_as_binary_content_tool_response(
    allow_model_requests: None, anthropic_api_key: str, image_content: BinaryContent
):
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
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
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(content="I'll get the image and identify the fruit in it."),
                    ToolCallPart(tool_name='get_image', args={}, tool_call_id='toolu_01W2SWpTnHpv1vZaLEknhfkj'),
                ],
                usage=RequestUsage(
                    input_tokens=555,
                    output_tokens=49,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 555,
                        'output_tokens': 49,
                    },
                ),
                model_name='claude-sonnet-4-5-20250929',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_01HQ5juE8oecrwBkoYMJi5fp',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_image',
                        content='See file 241a70',
                        tool_call_id='toolu_01W2SWpTnHpv1vZaLEknhfkj',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(content=['This is file 241a70:', image_content], timestamp=IsDatetime()),
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='The fruit in the image is a **kiwi** (also known as kiwifruit). The image shows a cross-section of the kiwi, revealing its distinctive bright green flesh, small black seeds arranged in a radial pattern around the pale center, and the brown fuzzy skin around the edge.'
                    )
                ],
                usage=RequestUsage(
                    input_tokens=1100,
                    output_tokens=68,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 1100,
                        'output_tokens': 68,
                    },
                ),
                model_name='claude-sonnet-4-5-20250929',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_015Cd8nysLLEjXi7JEm7A9DF',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


@pytest.mark.parametrize('media_type', ('audio/wav', 'audio/mpeg'))
async def test_audio_as_binary_content_input(allow_model_requests: None, media_type: str):
    c = completion_message([BetaTextBlock(text='world', type='text')], BetaUsage(input_tokens=5, output_tokens=10))
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    base64_content = b'//uQZ'

    with pytest.raises(RuntimeError, match='Unsupported binary content media type for Anthropic'):
        await agent.run(['hello', BinaryContent(data=base64_content, media_type=media_type)])


def test_model_status_error(allow_model_requests: None) -> None:
    mock_client = MockAnthropic.create_mock(
        APIStatusError(
            'test error',
            response=httpx.Response(status_code=500, request=httpx.Request('POST', 'https://example.com/v1')),
            body={'error': 'test error'},
        )
    )
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)
    with pytest.raises(ModelHTTPError) as exc_info:
        agent.run_sync('hello')
    assert str(exc_info.value) == snapshot(
        "status_code: 500, model_name: claude-sonnet-4-5, body: {'error': 'test error'}"
    )


def test_model_connection_error(allow_model_requests: None) -> None:
    mock_client = MockAnthropic.create_mock(
        APIConnectionError(
            message='Connection to https://api.anthropic.com timed out',
            request=httpx.Request('POST', 'https://api.anthropic.com/v1/messages'),
        )
    )
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)
    with pytest.raises(ModelAPIError) as exc_info:
        agent.run_sync('hello')
    assert exc_info.value.model_name == 'claude-sonnet-4-5'
    assert 'Connection to https://api.anthropic.com timed out' in str(exc_info.value.message)


async def test_count_tokens_connection_error(allow_model_requests: None) -> None:
    mock_client = MockAnthropic.create_mock(
        APIConnectionError(
            message='Connection to https://api.anthropic.com timed out',
            request=httpx.Request('POST', 'https://api.anthropic.com/v1/messages'),
        )
    )
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)
    with pytest.raises(ModelAPIError) as exc_info:
        await agent.run('hello', usage_limits=UsageLimits(input_tokens_limit=20, count_tokens_before_request=True))
    assert exc_info.value.model_name == 'claude-sonnet-4-5'
    assert 'Connection to https://api.anthropic.com timed out' in str(exc_info.value.message)


async def test_document_binary_content_input(
    allow_model_requests: None, anthropic_api_key: str, document_content: BinaryContent
):
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m)

    result = await agent.run(['What is the main content on this document?', document_content])
    assert result.output == snapshot(
        'The document simply contains the text "Dummy PDF file" at the top of what appears to be an otherwise blank page.'
    )


async def test_document_url_input(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m)

    document_url = DocumentUrl(url='https://pdfobject.com/pdf/sample.pdf')

    result = await agent.run(['What is the main content on this document?', document_url])
    assert result.output == snapshot(
        'This document appears to be a sample PDF file that mainly contains Lorem ipsum text, which is placeholder text commonly used in design and publishing. The document starts with "Sample PDF" as its title, followed by the line "This is a simple PDF file. Fun fun fun." The rest of the content consists of several paragraphs of Lorem ipsum text, which is Latin-looking but essentially meaningless text used to demonstrate the visual form of a document without the distraction of meaningful content.'
    )


async def test_text_document_url_input(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m)

    text_document_url = DocumentUrl(url='https://example-files.online-convert.com/document/txt/example.txt')

    result = await agent.run(['What is the main content on this document?', text_document_url])
    assert result.output == snapshot("""\
This document is a TXT test file that contains example content about the use of placeholder names like "John Doe," "Jane Doe," and their variants in legal and cultural contexts. The main content is divided into three main paragraphs explaining:

1. The use of "Doe" names as placeholders for unknown parties in legal actions
2. The use of "John Doe" as a reference to a typical male in various contexts
3. The use of variations like "Baby Doe" and numbered "John Doe"s in specific cases

The document also includes metadata about the file itself, including its purpose, type, and version, as well as attribution information indicating that the example content is from Wikipedia and is licensed under Attribution-ShareAlike 4.0.\
""")


async def test_text_document_as_binary_content_input(
    allow_model_requests: None, anthropic_api_key: str, text_document_content: BinaryContent
):
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m)

    result = await agent.run(['What does this text file say?', text_document_content])
    assert result.output == snapshot('The text file says "Dummy TXT file".')


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
                timestamp=IsNow(tz=timezone.utc),
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='The capital of France is Paris.')],
                usage=RequestUsage(
                    input_tokens=20,
                    output_tokens=10,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 20,
                        'output_tokens': 10,
                    },
                ),
                model_name='claude-3-opus-20240229',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_01Fg1JVgvCYUHWsxrj9GkpEv',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_anthropic_model_thinking_part(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 1024})
    agent = Agent(m, model_settings=settings)

    result = await agent.run('How do I cross the street?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='How do I cross the street?', timestamp=IsDatetime())],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="""\
This is a straightforward question about a common everyday task - crossing the street safely. I should provide clear, helpful instructions that emphasize safety.

The basic steps for crossing a street safely include:
1. Find a designated crossing area if possible (crosswalk, pedestrian crossing)
2. Look both ways before crossing
3. Make eye contact with drivers if possible
4. Follow traffic signals if present
5. Cross quickly but don't run
6. Continue to be aware of traffic while crossing

I'll provide this information in a clear, helpful way, emphasizing safety without being condescending.\
""",
                        signature='ErUBCkYIBhgCIkB9AyHADyBknnHL4dh+Yj3rg3javltU/bz1MLHKCQTEVZwvjis+DKTOFSYqZU0F2xasSofECVAmYmgtRf87AL52EgyXRs8lh+1HtZ0V+wAaDBo0eAabII+t1pdHzyIweFpD2l4j1eeUwN8UQOW+bxcN3mwu144OdOoUxmEKeOcU97wv+VF2pCsm07qcvucSKh1P/rZzWuYm7vxdnD4EVFHdBeewghoO0Ngc1MTNsxgC',
                        provider_name='anthropic',
                    ),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(
                    input_tokens=42,
                    output_tokens=363,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 42,
                        'output_tokens': 363,
                    },
                ),
                model_name='claude-sonnet-4-5-20250929',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_01BnZvs3naGorn93wjjCDwbd',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    result = await agent.run(
        'Considering the way to cross the street, analogously, how do I cross the river?',
        message_history=result.all_messages(),
    )
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Considering the way to cross the street, analogously, how do I cross the river?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="""\
The person is asking me to draw an analogy between crossing a street and crossing a river. I'll structure my response similarly to my street-crossing guidelines, but adapt it for river crossing, which has different safety considerations and methods.

For crossing a river, I should include:
1. Finding the right spot (bridges, shallow areas, ferry points)
2. Assessing safety (current speed, depth, obstacles)
3. Choosing the appropriate method (walking across shallow areas, using bridges, boats, etc.)
4. Safety precautions (life vests, ropes, etc.)
5. The actual crossing technique
6. What to do in emergencies

I'll keep the format similar to my street-crossing response for consistency.\
""",
                        signature='ErUBCkYIBhgCIkDvSvKCs5ePyYmR6zFw5i+jF7KEmortSIleqDa4gfa3pbuBclQt0TPdacouhdXFHdVSqR4qOAAAOpN7RQEUz2o6Egy9MPee6H8U4SW/G2QaDP/9ysoEvk+yNyVYZSIw+/+5wuRyc3oajwV3w0EdL9CIAXXd5thQH7DwAe3HTFvoJuF4oZ4fU+Kh6LRqxnEaKh3SSRqAH4UH/sD86duzg0jox4J/NH4C9iILVesEERgC',
                        provider_name='anthropic',
                    ),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(
                    input_tokens=291,
                    output_tokens=471,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 291,
                        'output_tokens': 471,
                    },
                ),
                model_name='claude-sonnet-4-5-20250929',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_anthropic_model_thinking_part_redacted(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-5-20250929', provider=AnthropicProvider(api_key=anthropic_api_key))
    settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 1024})
    agent = Agent(m, model_settings=settings)

    result = await agent.run(
        'ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB'
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='redacted_thinking',
                        signature=IsStr(),
                        provider_name='anthropic',
                    ),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(
                    input_tokens=92,
                    output_tokens=196,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 92,
                        'output_tokens': 196,
                    },
                ),
                model_name='claude-sonnet-4-5-20250929',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_01TbZ1ZKNMPq28AgBLyLX3c4',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    result = await agent.run(
        'What was that?',
        message_history=result.all_messages(),
    )
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What was that?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='redacted_thinking',
                        signature=IsStr(),
                        provider_name='anthropic',
                    ),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(
                    input_tokens=168,
                    output_tokens=232,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 168,
                        'output_tokens': 232,
                    },
                ),
                model_name='claude-sonnet-4-5-20250929',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_012oSSVsQdwoGH6b2fryM4fF',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_anthropic_model_thinking_part_redacted_stream(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-5-20250929', provider=AnthropicProvider(api_key=anthropic_api_key))
    settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 1024})
    agent = Agent(m, model_settings=settings)

    event_parts: list[Any] = []
    async with agent.iter(
        user_prompt='ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB'
    ) as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert agent_run.result is not None
    assert agent_run.result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='redacted_thinking',
                        signature=IsStr(),
                        provider_name='anthropic',
                    ),
                    ThinkingPart(
                        content='',
                        id='redacted_thinking',
                        signature=IsStr(),
                        provider_name='anthropic',
                    ),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(
                    input_tokens=92,
                    output_tokens=189,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 92,
                        'output_tokens': 189,
                    },
                ),
                model_name='claude-sonnet-4-5-20250929',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_018XZkwvj9asBiffg3fXt88s',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0,
                part=ThinkingPart(
                    content='',
                    id='redacted_thinking',
                    signature=IsStr(),
                    provider_name='anthropic',
                ),
            ),
            PartEndEvent(
                index=0,
                part=ThinkingPart(
                    content='',
                    id='redacted_thinking',
                    signature='EqkECkYIBxgCKkA8AZ4noDfV5VcOJe/p3JTRB6Xz5297mrWhl3MbHSXDKTMfuB/Z52U2teiWWTN0gg4eQ4bGS9TPilFX/xWTIq9HEgyOmstSPriNwyn1G7AaDC51r0hQ062qEd55IiIwYQj3Z3MSBBv0bSVdXi60LEHDvC7tzzmpQfw5Hb6R9rtyOz/6vC/xPw9/E1mUqfBqKpADO2HS2QlE/CnuzR901nZOn0TOw7kEXwH7kg30c85b9W7iKALgEejY9sELMBdPyIZNlTgKqNOKtY3R/aV5rGIRPTHh2Wh9Ijmqsf/TT7i//Z+InaYTo6f/fxF8R0vFXMRPOBME4XIscb05HcNhh4c9FDkpqQGYKaq31IR1NNwPWA0BsvdDz7SIo1nfx4H+X0qKKqqegKnQ3ynaXiD5ydT1C4U7fku4ftgF0LGwIk4PwXBE+4BP0DcKr1HV3cn7YSyNakBSDTvRJMKcXW6hl7X3w2a4//sxjC1Cjq0uzkIHkhzRWirN0OSXt+g3m6b1ex0wGmSyuO17Ak6kgVBpxwPugtrqsflG0oujFem44hecXJ9LQNssPf4RSlcydiG8EXp/XLGTe0YfHbe3kJagkowSH/Dm6ErXBiVs7249brncyY8WA+7MOoqIM82YIU095B9frCqDJDUWnN84VwOszRrcaywmpJXZO4aeQLMC1kXD5Wabu+O/00tD/X67EWkkWuR0AhDIXXjpot45vnBd4ewJ/hgB',
                    provider_name='anthropic',
                ),
                next_part_kind='thinking',
            ),
            PartStartEvent(
                index=1,
                part=ThinkingPart(
                    content='',
                    id='redacted_thinking',
                    signature='EtgBCkYIBxgCKkDQfGkwzflEJP5asG3oQfJXcTwJLoRznn8CmuczWCsJ36dv93X9H0NCeaJRbi5BrCA2DyMgFnRKRuzZx8VTv5axEgwkFmcHJk8BSiZMZRQaDDYv2KZPfbFgRa2QjyIwm47f5YYsSK9CT/oh/WWpU1HJJVHr8lrC6HG1ItRdtMvYQYmEGy+KhyfcIACfbssVKkDGv/NKqNMOAcu0bd66gJ2+R1R0PX11Jxn2Nd1JtZqkxx7vMT/PXtHDhm9jkDZ2k/6RjRRFuab/DBV3yRYdZ1J0GAE=',
                    provider_name='anthropic',
                ),
                previous_part_kind='thinking',
            ),
            PartEndEvent(
                index=1,
                part=ThinkingPart(
                    content='',
                    id='redacted_thinking',
                    signature='EtgBCkYIBxgCKkDQfGkwzflEJP5asG3oQfJXcTwJLoRznn8CmuczWCsJ36dv93X9H0NCeaJRbi5BrCA2DyMgFnRKRuzZx8VTv5axEgwkFmcHJk8BSiZMZRQaDDYv2KZPfbFgRa2QjyIwm47f5YYsSK9CT/oh/WWpU1HJJVHr8lrC6HG1ItRdtMvYQYmEGy+KhyfcIACfbssVKkDGv/NKqNMOAcu0bd66gJ2+R1R0PX11Jxn2Nd1JtZqkxx7vMT/PXtHDhm9jkDZ2k/6RjRRFuab/DBV3yRYdZ1J0GAE=',
                    provider_name='anthropic',
                ),
                next_part_kind='text',
            ),
            PartStartEvent(
                index=2, part=TextPart(content="I notice that you've sent what"), previous_part_kind='thinking'
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' appears to be some')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' kind of test string')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=" or command. I don't have")),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' any special "magic string"')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' triggers or backdoor commands')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' that would expose internal systems or')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' change my behavior.')),
            PartDeltaEvent(
                index=2,
                delta=TextPartDelta(
                    content_delta="""\


I'm Claude\
"""
                ),
            ),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=', an AI assistant create')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta='d by Anthropic to')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' be helpful, harmless')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=', and honest. How')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' can I assist you today with')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' a legitimate task or question?')),
            PartEndEvent(
                index=2,
                part=TextPart(
                    content="""\
I notice that you've sent what appears to be some kind of test string or command. I don't have any special "magic string" triggers or backdoor commands that would expose internal systems or change my behavior.

I'm Claude, an AI assistant created by Anthropic to be helpful, harmless, and honest. How can I assist you today with a legitimate task or question?\
"""
                ),
            ),
        ]
    )


async def test_anthropic_model_thinking_part_from_other_model(
    allow_model_requests: None, anthropic_api_key: str, openai_api_key: str
):
    provider = OpenAIProvider(api_key=openai_api_key)
    m = OpenAIResponsesModel('gpt-5', provider=provider)
    settings = OpenAIResponsesModelSettings(openai_reasoning_effort='high', openai_reasoning_summary='detailed')
    agent = Agent(m, system_prompt='You are a helpful assistant.', model_settings=settings)

    result = await agent.run('How do I cross the street?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(
                        content='You are a helpful assistant.',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content='How do I cross the street?',
                        timestamp=IsDatetime(),
                    ),
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c1fda7b4d481a1a65f48aef6a6b85e06da9901a3d98ab7',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    ThinkingPart(content=IsStr(), id='rs_68c1fda7b4d481a1a65f48aef6a6b85e06da9901a3d98ab7'),
                    ThinkingPart(content=IsStr(), id='rs_68c1fda7b4d481a1a65f48aef6a6b85e06da9901a3d98ab7'),
                    ThinkingPart(content=IsStr(), id='rs_68c1fda7b4d481a1a65f48aef6a6b85e06da9901a3d98ab7'),
                    ThinkingPart(content=IsStr(), id='rs_68c1fda7b4d481a1a65f48aef6a6b85e06da9901a3d98ab7'),
                    ThinkingPart(content=IsStr(), id='rs_68c1fda7b4d481a1a65f48aef6a6b85e06da9901a3d98ab7'),
                    TextPart(content=IsStr(), id='msg_68c1fdbecbf081a18085a084257a9aef06da9901a3d98ab7'),
                ],
                usage=RequestUsage(input_tokens=23, output_tokens=2211, details={'reasoning_tokens': 1920}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 10, 22, 37, 27, tzinfo=timezone.utc),
                },
                provider_response_id='resp_68c1fda6f11081a1b9fa80ae9122743506da9901a3d98ab7',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    result = await agent.run(
        'Considering the way to cross the street, analogously, how do I cross the river?',
        model=AnthropicModel(
            'claude-sonnet-4-0',
            provider=AnthropicProvider(api_key=anthropic_api_key),
            settings=AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 1024}),
        ),
        message_history=result.all_messages(),
    )
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Considering the way to cross the street, analogously, how do I cross the river?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        signature=IsStr(),
                        provider_name='anthropic',
                    ),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(
                    input_tokens=1343,
                    output_tokens=538,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 1343,
                        'output_tokens': 538,
                    },
                ),
                model_name='claude-sonnet-4-20250514',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_016e2w8nkCuArd5HFSfEwke7',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_anthropic_model_thinking_part_stream(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(api_key=anthropic_api_key))
    settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 1024})
    agent = Agent(m, model_settings=settings)

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='How do I cross the street?') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert agent_run.result is not None
    assert agent_run.result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='How do I cross the street?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        signature=IsStr(),
                        provider_name='anthropic',
                    ),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(
                    input_tokens=42,
                    output_tokens=419,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 42,
                        'output_tokens': 419,
                    },
                ),
                model_name='claude-sonnet-4-5-20250929',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_01PiJ6i3vjEZjHxojahi2YNc',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    assert event_parts == snapshot(
        [
            PartStartEvent(index=0, part=ThinkingPart(content='', signature='', provider_name='anthropic')),
            PartDeltaEvent(index=0, delta=IsInstance(ThinkingPartDelta)),
            PartDeltaEvent(index=0, delta=IsInstance(ThinkingPartDelta)),
            PartDeltaEvent(index=0, delta=IsInstance(ThinkingPartDelta)),
            PartDeltaEvent(index=0, delta=IsInstance(ThinkingPartDelta)),
            PartDeltaEvent(index=0, delta=IsInstance(ThinkingPartDelta)),
            PartDeltaEvent(index=0, delta=IsInstance(ThinkingPartDelta)),
            PartDeltaEvent(index=0, delta=IsInstance(ThinkingPartDelta)),
            PartDeltaEvent(index=0, delta=IsInstance(ThinkingPartDelta)),
            PartDeltaEvent(index=0, delta=IsInstance(ThinkingPartDelta)),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.)
2. Look\
""",
                    provider_name='anthropic',
                ),
            ),
            PartDeltaEvent(
                index=0, delta=ThinkingPartDelta(content_delta=' both ways (left-', provider_name='anthropic')
            ),
            PartDeltaEvent(
                index=0, delta=ThinkingPartDelta(content_delta='right-left in countries', provider_name='anthropic')
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta=' where cars drive on the right;', provider_name='anthropic'),
            ),
            PartDeltaEvent(
                index=0, delta=ThinkingPartDelta(content_delta=' right-left-right where', provider_name='anthropic')
            ),
            PartDeltaEvent(
                index=0, delta=ThinkingPartDelta(content_delta=' they drive on the left)', provider_name='anthropic')
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\

3. Wait for\
""",
                    provider_name='anthropic',
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta=' traffic to stop or for a clear', provider_name='anthropic'),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
 gap in traffic
4\
""",
                    provider_name='anthropic',
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta='. Make eye contact with drivers if', provider_name='anthropic'),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
 possible
5. Cross at\
""",
                    provider_name='anthropic',
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
 a steady pace without running
6. Continue\
""",
                    provider_name='anthropic',
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
 watching for traffic while crossing
7\
""",
                    provider_name='anthropic',
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta='. Use pedestrian signals where', provider_name='anthropic'),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
 available

I'll also mention\
""",
                    provider_name='anthropic',
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta=' some additional safety tips and considerations for', provider_name='anthropic'
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta=' different situations (busy streets, streets', provider_name='anthropic'
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta=' with traffic signals, etc.).', provider_name='anthropic'),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    signature_delta='ErUBCkYIBhgCIkA/Y+JwNMtmQyHcoo4/v2dpY6ruQifcu3pAzHbzIwpIrjIyaWaYdJOp9/0vUmBPj+LmqgiDSTktRcn0U75AlpXOEgwzVmYdHgDaZfeyBGcaDFSIZCHzzrZQkolJKCIwhMETosYLx+Dw/vKa83hht943z9R3/ViOqokT25JmMfaGOntuo+33Zxqf5rqUbkQ3Kh34rIqqnKaFSVr7Nn85z8OFN3Cwzz+HmXl2FgCXOxgC',
                    provider_name='anthropic',
                ),
            ),
            PartEndEvent(
                index=0,
                part=ThinkingPart(
                    content="""\
The question is asking about how to safely cross a street, which is a basic but important safety skill.

I should provide clear, step-by-step instructions for crossing a street safely:

1. Find a designated crossing point if possible (crosswalk, pedestrian crossing, etc.)
2. Look both ways (left-right-left in countries where cars drive on the right; right-left-right where they drive on the left)
3. Wait for traffic to stop or for a clear gap in traffic
4. Make eye contact with drivers if possible
5. Cross at a steady pace without running
6. Continue watching for traffic while crossing
7. Use pedestrian signals where available

I'll also mention some additional safety tips and considerations for different situations (busy streets, streets with traffic signals, etc.).\
""",
                    signature='ErUBCkYIBhgCIkA/Y+JwNMtmQyHcoo4/v2dpY6ruQifcu3pAzHbzIwpIrjIyaWaYdJOp9/0vUmBPj+LmqgiDSTktRcn0U75AlpXOEgwzVmYdHgDaZfeyBGcaDFSIZCHzzrZQkolJKCIwhMETosYLx+Dw/vKa83hht943z9R3/ViOqokT25JmMfaGOntuo+33Zxqf5rqUbkQ3Kh34rIqqnKaFSVr7Nn85z8OFN3Cwzz+HmXl2FgCXOxgC',
                    provider_name='anthropic',
                ),
                next_part_kind='text',
            ),
            PartStartEvent(
                index=1, part=TextPart(content='# How to Cross a Street Safely'), previous_part_kind='thinking'
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\


Follow these steps to cross a\
"""
                ),
            ),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
 street safely:

1\
"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='. **Find a proper')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' crossing point** - Use a crosswalk,')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' pedestrian crossing, or intersection')),
            PartDeltaEvent(
                index=1,
                delta=TextPartDelta(
                    content_delta="""\
 whenever possible.

2.\
"""
                ),
            ),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' **Stop at the curb** -')),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' Stand slightly back from the edge.')),
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
            PartEndEvent(
                index=1,
                part=TextPart(
                    content="""\
# How to Cross a Street Safely

Follow these steps to cross a street safely:

1. **Find a proper crossing point** - Use a crosswalk, pedestrian crossing, or intersection whenever possible.

2. **Stop at the curb** - Stand slightly back from the edge.

3. **Look both ways** - Look left, right, then left again (reverse in countries where cars drive on the left).

4. **Listen for traffic** - Remove headphones if you're wearing them.

5. **Wait for a gap** or for vehicles to stop completely.

6. **Make eye contact** with drivers to ensure they see you.

7. **Cross with purpose** - Walk at a steady pace without stopping or running.

8. **Continue watching** for traffic as you cross.

9. **Use signals** - Follow pedestrian crossing signals where available.

If there's a traffic light or pedestrian signal, only cross when indicated, and always check for turning vehicles even when you have the right of way.

Is there a specific situation or type of street crossing you're concerned about?\
"""
                ),
            ),
        ]
    )


async def test_multiple_system_prompt_formatting(allow_model_requests: None):
    c = completion_message([BetaTextBlock(text='world', type='text')], BetaUsage(input_tokens=5, output_tokens=10))
    mock_client = MockAnthropic().create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
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
        model='claude-sonnet-4-5',
        role='assistant',
        type='message',
        usage=usage,
    )


@pytest.mark.parametrize(
    'message_callback,usage',
    [
        pytest.param(
            lambda: anth_msg(BetaUsage(input_tokens=1, output_tokens=1)),
            snapshot(RequestUsage(input_tokens=1, output_tokens=1, details={'input_tokens': 1, 'output_tokens': 1})),
            id='AnthropicMessage',
        ),
        pytest.param(
            lambda: anth_msg(
                BetaUsage(input_tokens=1, output_tokens=1, cache_creation_input_tokens=2, cache_read_input_tokens=3)
            ),
            snapshot(
                RequestUsage(
                    input_tokens=6,
                    cache_write_tokens=2,
                    cache_read_tokens=3,
                    output_tokens=1,
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
            snapshot(RequestUsage(input_tokens=1, output_tokens=1, details={'input_tokens': 1, 'output_tokens': 1})),
            id='RawMessageStartEvent',
        ),
    ],
)
def test_usage(
    message_callback: Callable[[], BetaMessage | BetaRawMessageStartEvent | BetaRawMessageDeltaEvent], usage: RunUsage
):
    assert _map_usage(message_callback(), 'anthropic', '', 'unknown') == usage


def test_streaming_usage():
    start = BetaRawMessageStartEvent(message=anth_msg(BetaUsage(input_tokens=1, output_tokens=1)), type='message_start')
    initial_usage = _map_usage(start, 'anthropic', '', 'unknown')
    delta = BetaRawMessageDeltaEvent(delta=Delta(), usage=BetaMessageDeltaUsage(output_tokens=5), type='message_delta')
    final_usage = _map_usage(delta, 'anthropic', '', 'unknown', existing_usage=initial_usage)
    assert final_usage == snapshot(
        RequestUsage(input_tokens=1, output_tokens=5, details={'input_tokens': 1, 'output_tokens': 5})
    )


async def test_anthropic_model_empty_message_on_history(allow_model_requests: None, anthropic_api_key: str):
    """The Anthropic API will error if you send an empty message on the history.

    Check <https://github.com/pydantic/pydantic-ai/pull/1027> for more details.
    """
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m, instructions='You are a helpful assistant.')

    result = await agent.run(
        'I need a potato!',
        message_history=[
            ModelRequest(parts=[], instructions='You are a helpful assistant.', kind='request', timestamp=IsDatetime()),
            ModelResponse(parts=[TextPart(content='Hello, how can I help you?')], kind='response'),
        ],
    )
    assert result.output == snapshot("""\
I can't physically give you a potato since I'm a digital assistant. However, I can:

1. Help you find recipes that use potatoes
2. Give you tips on how to select, store, or prepare potatoes
3. Share information about different types of potatoes
4. Suggest where you might buy potatoes locally

What specific information about potatoes would be most helpful to you?\
""")


async def test_anthropic_web_search_tool(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(api_key=anthropic_api_key))
    settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 3000})
    agent = Agent(m, builtin_tools=[WebSearchTool()], model_settings=settings)

    result = await agent.run('What is the weather in San Francisco today?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is the weather in San Francisco today?', timestamp=IsDatetime())],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="""\
The user is asking about the weather in San Francisco today. This is asking for current, real-time information that would require a web search since weather conditions change frequently and I need up-to-date information. According to the guidelines, I should search for current conditions or recent events, and this clearly falls under that category.

I should search for San Francisco weather today to get the most current information.\
""",
                        signature='Et0ECkYIBxgCKkCXTXBKWJ3QYffHphenTDDE5jxo/vbyyvFuY7Gi5PGLYFdjxF0KQ4BGT7bGzB53hSRPgJtjUD975U7TZ4f9IheWEgy4pMKmvEJ0D9XDrxsaDDpjMZqhX/EnpJmjGyIwreKtd2Xj+RpguF1YI50dldiwk6qQNW2rK+xLwmWY5qF75b7WZrmOZ3endXYEQjBMKsQDmsnYnUODvD5Uh/yRIUgOp+6P5JrYjLabtsC3wfuIISLVe5QhC/3Ep7K/x55u97qy/DIhCAOz38x4YId37Pqq8XARrRq5CPwzxBzsMfPwpeV5eRHLQmasZxpOhivd1lMLC7B6D9EdpWefKWE+Ux1cMxpfaQj45cpMn93qLyCLGtNqnZJ2nPT7eoOtavZ9VvN5LsJOIWYEkxK+iq/6XYSJE5JlqBtDt9Y5P1QT/QnhFwfxjD/Cs3+RrGzKp2loEjmeYzNBwEfbY+pyKHJUS3bsxWyyi0d9Gc6Zfj4Xiuf/G0ninvXpSQheXi5gcvqIir6ZhcC40vHwvdVtJipSLkqMoPQcppCTOa2ATFyLKZIlug2OjoWIHrC5xnkCuKLXVMtHTF0mdrW0R/SgecnequYprzPeCc+Niqf4CVk62qtp+H06oWKQvHbP+s7kuAbdnhJjkcETiN8fP7+eLzKjRFAVnT0tixaNFjB6lWbg2ePyQDhqeVn6i/ULCzKyoY/hSIfZXUFwTCSDW42WvITFfPfWBBW+p6R/8peJ/KS2q0wHT2G3N4N7xFaNLOTXE0iPPtWsdqZw4cNQi9IUGKayqZ+/02tJYaEYAQ==',
                        provider_name='anthropic',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'San Francisco weather today'},
                        tool_call_id='srvtoolu_01EoSNE7k4dUJyGatASCV5qs',
                        provider_name='anthropic',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content=[
                            {
                                'encrypted_content': IsStr(),
                                'page_age': '6 days ago',
                                'title': 'San Francisco, CA Weather Forecast | AccuWeather',
                                'type': 'web_search_result',
                                'url': 'https://www.accuweather.com/en/us/san-francisco/94103/weather-forecast/347629',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': '6 days ago',
                                'title': '10-Day Weather Forecast for San Francisco, CA - The Weather Channel | weather.com',
                                'type': 'web_search_result',
                                'url': 'https://weather.com/weather/tenday/l/San+Francisco+CA+USCA0987:1:US',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'Weather Forecast and Conditions for San Francisco, CA - The Weather Channel | Weather.com',
                                'type': 'web_search_result',
                                'url': 'https://weather.com/weather/today/l/USCA0987:1:US',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'San Francisco, CA 10-Day Weather Forecast | Weather Underground',
                                'type': 'web_search_result',
                                'url': 'https://www.wunderground.com/forecast/us/ca/san-francisco',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': '1 week ago',
                                'title': 'National Weather Service',
                                'type': 'web_search_result',
                                'url': 'https://forecast.weather.gov/MapClick.php?lat=37.7771&lon=-122.4196',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': '1 week ago',
                                'title': 'San Francisco Bay Area weather forecast – NBC Bay Area',
                                'type': 'web_search_result',
                                'url': 'https://www.nbcbayarea.com/weather/',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'San Francisco, CA Current Weather - The Weather Network',
                                'type': 'web_search_result',
                                'url': 'https://www.theweathernetwork.com/en/city/us/california/san-francisco/current?_guid_iss_=1',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': '6 days ago',
                                'title': 'San Francisco, CA Weather Conditions | Weather Underground',
                                'type': 'web_search_result',
                                'url': 'https://www.wunderground.com/weather/us/ca/san-francisco',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'San Francisco, CA Hourly Weather Forecast | Weather Underground',
                                'type': 'web_search_result',
                                'url': 'https://www.wunderground.com/hourly/us/ca/san-francisco',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': '1 week ago',
                                'title': 'Live Doppler 7 | Bay Area Weather News - ABC7 San Francisco',
                                'type': 'web_search_result',
                                'url': 'https://abc7news.com/weather/',
                            },
                        ],
                        tool_call_id='srvtoolu_01EoSNE7k4dUJyGatASCV5qs',
                        timestamp=IsDatetime(),
                        provider_name='anthropic',
                    ),
                    TextPart(
                        content="""\
Based on the search results, here's the weather information for San Francisco today (September 16, 2025):

**Current Conditions:**
- \
"""
                    ),
                    TextPart(content='Temperature: 66°F with clear skies'),
                    TextPart(
                        content="""\

- \
"""
                    ),
                    TextPart(content='Wind: W at 3 mph with gusts up to 5 mph'),
                    TextPart(
                        content="""\

- \
"""
                    ),
                    TextPart(content='Air quality is poor and unhealthy for sensitive groups'),
                    TextPart(
                        content="""\


**Today's Forecast:**
- \
"""
                    ),
                    TextPart(content='High: 78°F with partly cloudy skies'),
                    TextPart(
                        content="""\

- \
"""
                    ),
                    TextPart(content='Winds W at 10 to 20 mph'),
                    TextPart(
                        content="""\

- \
"""
                    ),
                    TextPart(content='8% chance of precipitation'),
                    TextPart(
                        content="""\

- \
"""
                    ),
                    TextPart(
                        content='Some clouds in the morning will give way to mainly sunny skies for the afternoon'
                    ),
                    TextPart(
                        content="""\


**Tonight:**
- \
"""
                    ),
                    TextPart(content='Low: 57°F with clear to partly cloudy conditions'),
                    TextPart(
                        content="""\

- \
"""
                    ),
                    TextPart(content='Winds W at 10 to 20 mph'),
                    TextPart(
                        content="""\


Overall, it's a pleasant day in San Francisco with mild temperatures and mostly sunny conditions, though the air quality is poor, so sensitive individuals should limit outdoor activities.\
"""
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=8984,
                    output_tokens=520,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 8984,
                        'output_tokens': 520,
                    },
                ),
                model_name='claude-sonnet-4-20250514',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_0119wM5YxCLg3hwUWrxEQ9Y8',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    messages = result.all_messages()
    result = await agent.run(user_prompt='how about Mexico City?', message_history=messages)
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='how about Mexico City?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='The user is now asking about the weather in Mexico City today. I should search for current weather information for Mexico City.',
                        signature='EqgCCkYIBxgCKkAhyrWtc4MfwZtLCpH/f41h3xS0UBTKetW5LA6ADj/q/8G5GiD+31L8MWU5+8QbLKrdzKIr5RZTEmval6pjPCxwEgygcM1WHSKHKa3PiscaDDtaNmY6L04w/DaCFSIw4mjvUNimq2ShpHNyVrezsnnXaRyyt2Ei4Iik2sCgzARFHGyDNzerHS/aCxzMR8MFKo8BVo7IxMBObxJIn43oG4aHroTyH4tX0IB3HPE1L1O/RZ9HfrmCc/KJwvIc79klaolMdyFvc343GJbssZxF1YJ+8YgGJtrzsKaawjsNelJBqkNWdF/TFwY0G+zGS90yWmHp4hFylIib5OTYz1Dm8O066biiZps8EDkINIoiIfkslPdnP3FWiCl9g6+gSiJd+WwYAQ==',
                        provider_name='anthropic',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'Mexico City weather today'},
                        tool_call_id='srvtoolu_01SnV7n4h3ZQtz14JriSp4xa',
                        provider_name='anthropic',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content=[
                            {
                                'encrypted_content': IsStr(),
                                'page_age': '1 month ago',
                                'title': 'Weather Forecast and Conditions for Mexico City, Mexico - The Weather Channel | Weather.com',
                                'type': 'web_search_result',
                                'url': 'https://weather.com/weather/today/l/6121681b2c5df01145b9723d497c595c53ae08104787aa1c26bafdf2fb875c07',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'Mexico City, México City, Mexico Weather Forecast | AccuWeather',
                                'type': 'web_search_result',
                                'url': 'https://www.accuweather.com/en/mx/mexico-city/242560/weather-forecast/242560',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': 'August 12, 2025',
                                'title': 'Weather Forecast and Conditions for Cuauhtémoc, Mexico - The Weather Channel | Weather.com',
                                'type': 'web_search_result',
                                'url': 'https://weather.com/weather/today/l/Cuauht%C3%A9moc+Mexico?canonicalCityId=7164197a006f4e553a538a0b73c06757',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'Mexico City, CMX, MX Current Weather - The Weather Network',
                                'type': 'web_search_result',
                                'url': 'https://www.theweathernetwork.com/en/city/mx/ciudad-de-mexico/mexico-city/current?_guid_iss_=1',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'Mexico City, Mexico 10-Day Weather Forecast | Weather Underground',
                                'type': 'web_search_result',
                                'url': 'https://www.wunderground.com/forecast/mx/mexico-city',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': 'August 12, 2025',
                                'title': 'Mexico City, Mexico Weather Conditions | Weather Underground',
                                'type': 'web_search_result',
                                'url': 'https://www.wunderground.com/weather/mx/mexico-city',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': 'June 19, 2025',
                                'title': 'Weather for Mexico City, Ciudad de México, Mexico',
                                'type': 'web_search_result',
                                'url': 'https://www.timeanddate.com/weather/mexico/mexico-city',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': '10-Day Weather Forecast for Mexico City, Mexico - The Weather Channel | weather.com',
                                'type': 'web_search_result',
                                'url': 'https://weather.com/weather/tenday/l/6121681b2c5df01145b9723d497c595c53ae08104787aa1c26bafdf2fb875c07',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'Yr - Mexico City - Hourly weather forecast',
                                'type': 'web_search_result',
                                'url': 'https://www.yr.no/en/forecast/hourly-table/2-3530597/Mexico/Mexico%20City/Mexico%20City?i=0',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': '10-Day Weather Forecast for Cuauhtémoc, Mexico - The Weather Channel | weather.com',
                                'type': 'web_search_result',
                                'url': 'https://weather.com/weather/tenday/l/Cuauht%C3%A9moc+Mexico?canonicalCityId=7164197a006f4e553a538a0b73c06757',
                            },
                        ],
                        tool_call_id='srvtoolu_01SnV7n4h3ZQtz14JriSp4xa',
                        timestamp=IsDatetime(),
                        provider_name='anthropic',
                    ),
                    TextPart(
                        content="""\
Based on the search results, here's the weather information for Mexico City today (September 16, 2025):

**Current Conditions:**
- \
"""
                    ),
                    TextPart(content='Temperature: 59°F (15°C) with clouds and sun'),
                    TextPart(
                        content="""\

- \
"""
                    ),
                    TextPart(content='Wind: NNE at 6 mph with gusts up to 6 mph'),
                    TextPart(
                        content="""\

- \
"""
                    ),
                    TextPart(content='Air quality is poor and unhealthy for sensitive groups'),
                    TextPart(
                        content="""\


**Today's Forecast:**
- \
"""
                    ),
                    TextPart(content='High: 72°F (22°C) - mostly cloudy with a touch of rain this afternoon'),
                    TextPart(
                        content="""\

- \
"""
                    ),
                    TextPart(
                        content='High 73F with partly cloudy conditions early followed by scattered thunderstorms. Winds NNE at 10 to 15 mph, 70% chance of rain'
                    ),
                    TextPart(
                        content="""\

- \
"""
                    ),
                    TextPart(
                        content='Scattered thunderstorms developing during the afternoon. High near 75F with winds NNE at 10 to 15 mph and 70% chance of rain'
                    ),
                    TextPart(
                        content="""\


**Tonight:**
- \
"""
                    ),
                    TextPart(content='Low: 58°F with cloudy conditions and a couple of showers'),
                    TextPart(
                        content="""\

- \
"""
                    ),
                    TextPart(content='Cloudy overnight with low 57F and winds NNW at 10 to 15 mph'),
                    TextPart(
                        content="""\


Mexico City is experiencing typical rainy season weather with moderate temperatures, high humidity, and afternoon thunderstorms expected. Like San Francisco, the air quality is poor, so those with respiratory sensitivities should take precautions.\
"""
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=19859,
                    output_tokens=544,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 19859,
                        'output_tokens': 544,
                    },
                ),
                model_name='claude-sonnet-4-20250514',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_01Vatv9GeGaeqVHfSGhkU7mo',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_anthropic_model_web_search_tool_stream(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(api_key=anthropic_api_key))
    settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 3000})
    agent = Agent(m, builtin_tools=[WebSearchTool()], model_settings=settings)

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='What is the weather in San Francisco today?') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert agent_run.result is not None
    messages = agent_run.result.all_messages()
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the weather in San Francisco today?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="""\
The user is asking about the weather in San Francisco today. This is clearly a request for current, real-time information that changes daily, so I should use web search to get up-to-date weather information. According to the guidelines, today's date is September 16, 2025.

I should search for current weather in San Francisco. I'll include "today" in the search query to get the most current information.\
""",
                        signature='Er8ECkYIBxgCKkDp29haxwUos3j9hg3HNQI8e4jcFtinIsLxpzaQR/MhPnIpHkUpSNPatD/C2EVyiEGg2LIO1lhkU/P8XLgiyejFEgzinYyrRtGe03DeFEIaDL63CVUOAo1v/57lpSIw+msm1NHv1h+xLzkbu2YqlXPwjza0tVjwAj7RLUFwB1HpPbdv6hlityaMFb/SwKZZKqYDwbYu36cdPpUcpirpZaKZ/DITzfWJkX93BXmRl5au50mxAiFe9B8XxreADaofra5cmevEaaLH0b5Ze/IC0ja/cJdo9NoVlyHlqdXmex22CAkg0Y/HnsZr8MbnE6GyG9bOqAEhwb6YgKHMaMLDVmElbNSsD7luWtsbw5BDvRaqSSROzTxH4s0dqjUqJsoOBeUXuUqWHSl2KwQi8akELKUnvlDz15ZwFI1yVTHA5nSMFIhjB0jECs1g8PjFkAYTHkHddYR5/SLruy1ENpKU0xjc/hd/O41xnI3PxHBGDKv/hdeSVBKjJ0SDYIwXW96QS5vzlKxYGCqtibj2VxPzUlDITvhn1oO+cjCXClo1lE+ul//+nk7jk7fRkvl1/+pscYCpBoGKprA7CU1kpiggO9pAVUrpZM9vC2jF5/VVVYEoY3CyC+hrNpDWXTUdGdCTofhp2wdWVZzCmO7/+L8SUnlu64YYe9PWsRDuHRe8Lvl0M9EyBrhWnGWQkkk9b+O5uNU5xgE0sjbuGzgYswhwSd7Powb8XbtbW6h7lTbo1M2IQ3Ok0kdt0RAYAQ==',
                        provider_name='anthropic',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args='{"query": "San Francisco weather today"}',
                        tool_call_id='srvtoolu_01FYcUbzEaqqQh1WBRj1QX3h',
                        provider_name='anthropic',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content=[
                            {
                                'encrypted_content': IsStr(),
                                'page_age': '6 days ago',
                                'title': 'San Francisco, CA Weather Forecast | AccuWeather',
                                'type': 'web_search_result',
                                'url': 'https://www.accuweather.com/en/us/san-francisco/94103/weather-forecast/347629',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': '6 days ago',
                                'title': '10-Day Weather Forecast for San Francisco, CA - The Weather Channel | weather.com',
                                'type': 'web_search_result',
                                'url': 'https://weather.com/weather/tenday/l/San+Francisco+CA+USCA0987:1:US',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'Weather Forecast and Conditions for San Francisco, CA - The Weather Channel | Weather.com',
                                'type': 'web_search_result',
                                'url': 'https://weather.com/weather/today/l/USCA0987:1:US',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'San Francisco, CA 10-Day Weather Forecast | Weather Underground',
                                'type': 'web_search_result',
                                'url': 'https://www.wunderground.com/forecast/us/ca/san-francisco',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': '1 week ago',
                                'title': 'National Weather Service',
                                'type': 'web_search_result',
                                'url': 'https://forecast.weather.gov/MapClick.php?lat=37.7771&lon=-122.4196',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': '1 week ago',
                                'title': 'San Francisco Bay Area weather forecast – NBC Bay Area',
                                'type': 'web_search_result',
                                'url': 'https://www.nbcbayarea.com/weather/',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'San Francisco, CA Current Weather - The Weather Network',
                                'type': 'web_search_result',
                                'url': 'https://www.theweathernetwork.com/en/city/us/california/san-francisco/current?_guid_iss_=1',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': '6 days ago',
                                'title': 'San Francisco, CA Weather Conditions | Weather Underground',
                                'type': 'web_search_result',
                                'url': 'https://www.wunderground.com/weather/us/ca/san-francisco',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'San Francisco, CA Hourly Weather Forecast | Weather Underground',
                                'type': 'web_search_result',
                                'url': 'https://www.wunderground.com/hourly/us/ca/san-francisco',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': '1 week ago',
                                'title': 'Live Doppler 7 | Bay Area Weather News - ABC7 San Francisco',
                                'type': 'web_search_result',
                                'url': 'https://abc7news.com/weather/',
                            },
                        ],
                        tool_call_id='srvtoolu_01FYcUbzEaqqQh1WBRj1QX3h',
                        timestamp=IsDatetime(),
                        provider_name='anthropic',
                    ),
                    TextPart(
                        content='Based on the search results, I can see that the information is a bit dated (most results are from about 6 days to a week ago), but I can provide you with the available weather information for San Francisco. Let me search for more current information.'
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args='{"query": "San Francisco weather September 16 2025"}',
                        tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx',
                        provider_name='anthropic',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content=[
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'San Francisco weather in September 2025 | Weather25.com',
                                'type': 'web_search_result',
                                'url': 'https://www.weather25.com/north-america/usa/california/san-francisco?page=month&month=September',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'Weather in San Francisco in September 2025 (California) - detailed Weather Forecast for a month',
                                'type': 'web_search_result',
                                'url': 'https://world-weather.info/forecast/usa/san_francisco/september-2025/',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'San Francisco, CA Monthly Weather | AccuWeather',
                                'type': 'web_search_result',
                                'url': 'https://www.accuweather.com/en/us/san-francisco/94103/september-weather/347629',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'Weather San Francisco in September 2025: Temperature & Climate',
                                'type': 'web_search_result',
                                'url': 'https://en.climate-data.org/north-america/united-states-of-america/california/san-francisco-385/t/september-9/',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'San Francisco weather in September 2025 | California',
                                'type': 'web_search_result',
                                'url': 'https://www.weather2travel.com/california/san-francisco/september/',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'San Francisco, Weather for September, USA',
                                'type': 'web_search_result',
                                'url': 'https://www.holiday-weather.com/san_francisco/averages/september/',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'Monthly Weather Forecast for San Francisco, CA - weather.com',
                                'type': 'web_search_result',
                                'url': 'https://weather.com/weather/monthly/l/69bedc6a5b6e977993fb3e5344e3c06d8bc36a1fb6754c3ddfb5310a3c6d6c87',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': '3 weeks ago',
                                'title': 'September 2025 Weather - San Francisco',
                                'type': 'web_search_result',
                                'url': 'https://www.easeweather.com/north-america/united-states/california/city-and-county-of-san-francisco/san-francisco/september',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': None,
                                'title': 'San Francisco Weather in September | Thomas Cook',
                                'type': 'web_search_result',
                                'url': 'https://www.thomascook.com/holidays/weather/usa/california/san-francisco/september/',
                            },
                            {
                                'encrypted_content': IsStr(),
                                'page_age': '4 days ago',
                                'title': IsStr(),
                                'type': 'web_search_result',
                                'url': 'https://www.sfchronicle.com/weather-forecast/article/weather-forecast-san-francisco-21043269.php',
                            },
                        ],
                        tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx',
                        timestamp=IsDatetime(),
                        provider_name='anthropic',
                    ),
                    TextPart(
                        content="""\
Based on the search results, I can provide you with information about San Francisco's weather today (September 16, 2025):

According to AccuWeather's forecast, \
"""
                    ),
                    TextPart(content='today (September 16) shows a high of 76°F and low of 59°F'),
                    TextPart(
                        content="""\
 for San Francisco.

From the recent San Francisco Chronicle weather report, \
"""
                    ),
                    TextPart(content='average mid-September highs in San Francisco are around 70 degrees'),
                    TextPart(
                        content="""\
, so today's forecast of 76°F is slightly above the typical temperature for this time of year.

The general weather pattern for San Francisco in September includes:
- \
"""
                    ),
                    TextPart(
                        content='Daytime temperatures usually reach 22°C (72°F) in San Francisco in September, falling to 13°C (55°F) at night'
                    ),
                    TextPart(
                        content="""\

- \
"""
                    ),
                    TextPart(
                        content='There are normally 9 hours of bright sunshine each day in San Francisco in September'
                    ),
                    TextPart(
                        content="""\

- \
"""
                    ),
                    TextPart(
                        content='San Francisco experiences minimal rainfall in September, with an average precipitation of just 3mm. Typically, there are no rainy days during this month'
                    ),
                    TextPart(
                        content="""\


So for today, you can expect partly sunny to sunny skies with a high around 76°F (24°C) and a low around 59°F (15°C), with very little chance of rain. It's shaping up to be a pleasant day in San Francisco!\
"""
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=22397,
                    output_tokens=637,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 22397,
                        'output_tokens': 637,
                    },
                ),
                model_name='claude-sonnet-4-20250514',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_01QmxBSdEbD9ZeBWDVgFDoQ5',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    assert event_parts == snapshot(
        [
            PartStartEvent(index=0, part=ThinkingPart(content='', signature='', provider_name='anthropic')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta='The user is asking about the weather', provider_name='anthropic'
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta=' in San Francisco today. This is clearly a request', provider_name='anthropic'
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta=' for current, real-time information', provider_name='anthropic'),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta=' that changes daily, so I should use', provider_name='anthropic'
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta=' web search to get up-to-date weather', provider_name='anthropic'
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta=' information. According to the guidelines, today', provider_name='anthropic'
                ),
            ),
            PartDeltaEvent(
                index=0, delta=ThinkingPartDelta(content_delta="'s date is September 16, ", provider_name='anthropic')
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
2025.

I should search for current\
""",
                    provider_name='anthropic',
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta=' weather in San Francisco. I\'ll include "', provider_name='anthropic'
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta='today" in the search query to get the most current', provider_name='anthropic'
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' information.', provider_name='anthropic')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    signature_delta='Er8ECkYIBxgCKkDp29haxwUos3j9hg3HNQI8e4jcFtinIsLxpzaQR/MhPnIpHkUpSNPatD/C2EVyiEGg2LIO1lhkU/P8XLgiyejFEgzinYyrRtGe03DeFEIaDL63CVUOAo1v/57lpSIw+msm1NHv1h+xLzkbu2YqlXPwjza0tVjwAj7RLUFwB1HpPbdv6hlityaMFb/SwKZZKqYDwbYu36cdPpUcpirpZaKZ/DITzfWJkX93BXmRl5au50mxAiFe9B8XxreADaofra5cmevEaaLH0b5Ze/IC0ja/cJdo9NoVlyHlqdXmex22CAkg0Y/HnsZr8MbnE6GyG9bOqAEhwb6YgKHMaMLDVmElbNSsD7luWtsbw5BDvRaqSSROzTxH4s0dqjUqJsoOBeUXuUqWHSl2KwQi8akELKUnvlDz15ZwFI1yVTHA5nSMFIhjB0jECs1g8PjFkAYTHkHddYR5/SLruy1ENpKU0xjc/hd/O41xnI3PxHBGDKv/hdeSVBKjJ0SDYIwXW96QS5vzlKxYGCqtibj2VxPzUlDITvhn1oO+cjCXClo1lE+ul//+nk7jk7fRkvl1/+pscYCpBoGKprA7CU1kpiggO9pAVUrpZM9vC2jF5/VVVYEoY3CyC+hrNpDWXTUdGdCTofhp2wdWVZzCmO7/+L8SUnlu64YYe9PWsRDuHRe8Lvl0M9EyBrhWnGWQkkk9b+O5uNU5xgE0sjbuGzgYswhwSd7Powb8XbtbW6h7lTbo1M2IQ3Ok0kdt0RAYAQ==',
                    provider_name='anthropic',
                ),
            ),
            PartEndEvent(
                index=0,
                part=ThinkingPart(
                    content="""\
The user is asking about the weather in San Francisco today. This is clearly a request for current, real-time information that changes daily, so I should use web search to get up-to-date weather information. According to the guidelines, today's date is September 16, 2025.

I should search for current weather in San Francisco. I'll include "today" in the search query to get the most current information.\
""",
                    signature='Er8ECkYIBxgCKkDp29haxwUos3j9hg3HNQI8e4jcFtinIsLxpzaQR/MhPnIpHkUpSNPatD/C2EVyiEGg2LIO1lhkU/P8XLgiyejFEgzinYyrRtGe03DeFEIaDL63CVUOAo1v/57lpSIw+msm1NHv1h+xLzkbu2YqlXPwjza0tVjwAj7RLUFwB1HpPbdv6hlityaMFb/SwKZZKqYDwbYu36cdPpUcpirpZaKZ/DITzfWJkX93BXmRl5au50mxAiFe9B8XxreADaofra5cmevEaaLH0b5Ze/IC0ja/cJdo9NoVlyHlqdXmex22CAkg0Y/HnsZr8MbnE6GyG9bOqAEhwb6YgKHMaMLDVmElbNSsD7luWtsbw5BDvRaqSSROzTxH4s0dqjUqJsoOBeUXuUqWHSl2KwQi8akELKUnvlDz15ZwFI1yVTHA5nSMFIhjB0jECs1g8PjFkAYTHkHddYR5/SLruy1ENpKU0xjc/hd/O41xnI3PxHBGDKv/hdeSVBKjJ0SDYIwXW96QS5vzlKxYGCqtibj2VxPzUlDITvhn1oO+cjCXClo1lE+ul//+nk7jk7fRkvl1/+pscYCpBoGKprA7CU1kpiggO9pAVUrpZM9vC2jF5/VVVYEoY3CyC+hrNpDWXTUdGdCTofhp2wdWVZzCmO7/+L8SUnlu64YYe9PWsRDuHRe8Lvl0M9EyBrhWnGWQkkk9b+O5uNU5xgE0sjbuGzgYswhwSd7Powb8XbtbW6h7lTbo1M2IQ3Ok0kdt0RAYAQ==',
                    provider_name='anthropic',
                ),
                next_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=1,
                part=BuiltinToolCallPart(
                    tool_name='web_search', tool_call_id='srvtoolu_01FYcUbzEaqqQh1WBRj1QX3h', provider_name='anthropic'
                ),
                previous_part_kind='thinking',
            ),
            PartDeltaEvent(
                index=1, delta=ToolCallPartDelta(args_delta='', tool_call_id='srvtoolu_01FYcUbzEaqqQh1WBRj1QX3h')
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='{"query": ', tool_call_id='srvtoolu_01FYcUbzEaqqQh1WBRj1QX3h'),
            ),
            PartDeltaEvent(
                index=1, delta=ToolCallPartDelta(args_delta='"Sa', tool_call_id='srvtoolu_01FYcUbzEaqqQh1WBRj1QX3h')
            ),
            PartDeltaEvent(
                index=1, delta=ToolCallPartDelta(args_delta='n Fr', tool_call_id='srvtoolu_01FYcUbzEaqqQh1WBRj1QX3h')
            ),
            PartDeltaEvent(
                index=1, delta=ToolCallPartDelta(args_delta='anc', tool_call_id='srvtoolu_01FYcUbzEaqqQh1WBRj1QX3h')
            ),
            PartDeltaEvent(
                index=1, delta=ToolCallPartDelta(args_delta='isc', tool_call_id='srvtoolu_01FYcUbzEaqqQh1WBRj1QX3h')
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='o weather', tool_call_id='srvtoolu_01FYcUbzEaqqQh1WBRj1QX3h'),
            ),
            PartDeltaEvent(
                index=1, delta=ToolCallPartDelta(args_delta=' tod', tool_call_id='srvtoolu_01FYcUbzEaqqQh1WBRj1QX3h')
            ),
            PartDeltaEvent(
                index=1, delta=ToolCallPartDelta(args_delta='ay"}', tool_call_id='srvtoolu_01FYcUbzEaqqQh1WBRj1QX3h')
            ),
            PartEndEvent(
                index=1,
                part=BuiltinToolCallPart(
                    tool_name='web_search',
                    args='{"query": "San Francisco weather today"}',
                    tool_call_id='srvtoolu_01FYcUbzEaqqQh1WBRj1QX3h',
                    provider_name='anthropic',
                ),
                next_part_kind='builtin-tool-return',
            ),
            PartStartEvent(
                index=2,
                part=BuiltinToolReturnPart(
                    tool_name='web_search',
                    content=[
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '6 days ago',
                            'title': 'San Francisco, CA Weather Forecast | AccuWeather',
                            'type': 'web_search_result',
                            'url': 'https://www.accuweather.com/en/us/san-francisco/94103/weather-forecast/347629',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '6 days ago',
                            'title': '10-Day Weather Forecast for San Francisco, CA - The Weather Channel | weather.com',
                            'type': 'web_search_result',
                            'url': 'https://weather.com/weather/tenday/l/San+Francisco+CA+USCA0987:1:US',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'Weather Forecast and Conditions for San Francisco, CA - The Weather Channel | Weather.com',
                            'type': 'web_search_result',
                            'url': 'https://weather.com/weather/today/l/USCA0987:1:US',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'San Francisco, CA 10-Day Weather Forecast | Weather Underground',
                            'type': 'web_search_result',
                            'url': 'https://www.wunderground.com/forecast/us/ca/san-francisco',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '1 week ago',
                            'title': 'National Weather Service',
                            'type': 'web_search_result',
                            'url': 'https://forecast.weather.gov/MapClick.php?lat=37.7771&lon=-122.4196',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '1 week ago',
                            'title': 'San Francisco Bay Area weather forecast – NBC Bay Area',
                            'type': 'web_search_result',
                            'url': 'https://www.nbcbayarea.com/weather/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'San Francisco, CA Current Weather - The Weather Network',
                            'type': 'web_search_result',
                            'url': 'https://www.theweathernetwork.com/en/city/us/california/san-francisco/current?_guid_iss_=1',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '6 days ago',
                            'title': 'San Francisco, CA Weather Conditions | Weather Underground',
                            'type': 'web_search_result',
                            'url': 'https://www.wunderground.com/weather/us/ca/san-francisco',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'San Francisco, CA Hourly Weather Forecast | Weather Underground',
                            'type': 'web_search_result',
                            'url': 'https://www.wunderground.com/hourly/us/ca/san-francisco',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '1 week ago',
                            'title': 'Live Doppler 7 | Bay Area Weather News - ABC7 San Francisco',
                            'type': 'web_search_result',
                            'url': 'https://abc7news.com/weather/',
                        },
                    ],
                    tool_call_id='srvtoolu_01FYcUbzEaqqQh1WBRj1QX3h',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartStartEvent(index=3, part=TextPart(content='Base'), previous_part_kind='builtin-tool-return'),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='d on the search results, I can see')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' that the information is a bit date')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='d (most results are from about 6')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' days to a week ago), but I can provide')),
            PartDeltaEvent(
                index=3,
                delta=TextPartDelta(content_delta=' you with the available weather information for San Francisco.'),
            ),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' Let me search for more current')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' information.')),
            PartEndEvent(
                index=3,
                part=TextPart(
                    content='Based on the search results, I can see that the information is a bit dated (most results are from about 6 days to a week ago), but I can provide you with the available weather information for San Francisco. Let me search for more current information.'
                ),
                next_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=4,
                part=BuiltinToolCallPart(
                    tool_name='web_search', tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx', provider_name='anthropic'
                ),
                previous_part_kind='text',
            ),
            PartDeltaEvent(
                index=4, delta=ToolCallPartDelta(args_delta='', tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx')
            ),
            PartDeltaEvent(
                index=4, delta=ToolCallPartDelta(args_delta='{"', tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx')
            ),
            PartDeltaEvent(
                index=4, delta=ToolCallPartDelta(args_delta='quer', tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx')
            ),
            PartDeltaEvent(
                index=4, delta=ToolCallPartDelta(args_delta='y": ', tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx')
            ),
            PartDeltaEvent(
                index=4, delta=ToolCallPartDelta(args_delta='"San', tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx')
            ),
            PartDeltaEvent(
                index=4, delta=ToolCallPartDelta(args_delta=' Fra', tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx')
            ),
            PartDeltaEvent(
                index=4, delta=ToolCallPartDelta(args_delta='nci', tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx')
            ),
            PartDeltaEvent(
                index=4, delta=ToolCallPartDelta(args_delta='sco w', tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx')
            ),
            PartDeltaEvent(
                index=4,
                delta=ToolCallPartDelta(args_delta='eather S', tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx'),
            ),
            PartDeltaEvent(
                index=4, delta=ToolCallPartDelta(args_delta='ep', tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx')
            ),
            PartDeltaEvent(
                index=4,
                delta=ToolCallPartDelta(args_delta='tember 16 2', tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx'),
            ),
            PartDeltaEvent(
                index=4, delta=ToolCallPartDelta(args_delta='025"}', tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx')
            ),
            PartEndEvent(
                index=4,
                part=BuiltinToolCallPart(
                    tool_name='web_search',
                    args='{"query": "San Francisco weather September 16 2025"}',
                    tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx',
                    provider_name='anthropic',
                ),
                next_part_kind='builtin-tool-return',
            ),
            PartStartEvent(
                index=5,
                part=BuiltinToolReturnPart(
                    tool_name='web_search',
                    content=[
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'San Francisco weather in September 2025 | Weather25.com',
                            'type': 'web_search_result',
                            'url': 'https://www.weather25.com/north-america/usa/california/san-francisco?page=month&month=September',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'Weather in San Francisco in September 2025 (California) - detailed Weather Forecast for a month',
                            'type': 'web_search_result',
                            'url': 'https://world-weather.info/forecast/usa/san_francisco/september-2025/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'San Francisco, CA Monthly Weather | AccuWeather',
                            'type': 'web_search_result',
                            'url': 'https://www.accuweather.com/en/us/san-francisco/94103/september-weather/347629',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'Weather San Francisco in September 2025: Temperature & Climate',
                            'type': 'web_search_result',
                            'url': 'https://en.climate-data.org/north-america/united-states-of-america/california/san-francisco-385/t/september-9/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'San Francisco weather in September 2025 | California',
                            'type': 'web_search_result',
                            'url': 'https://www.weather2travel.com/california/san-francisco/september/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'San Francisco, Weather for September, USA',
                            'type': 'web_search_result',
                            'url': 'https://www.holiday-weather.com/san_francisco/averages/september/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'Monthly Weather Forecast for San Francisco, CA - weather.com',
                            'type': 'web_search_result',
                            'url': 'https://weather.com/weather/monthly/l/69bedc6a5b6e977993fb3e5344e3c06d8bc36a1fb6754c3ddfb5310a3c6d6c87',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '3 weeks ago',
                            'title': 'September 2025 Weather - San Francisco',
                            'type': 'web_search_result',
                            'url': 'https://www.easeweather.com/north-america/united-states/california/city-and-county-of-san-francisco/san-francisco/september',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'San Francisco Weather in September | Thomas Cook',
                            'type': 'web_search_result',
                            'url': 'https://www.thomascook.com/holidays/weather/usa/california/san-francisco/september/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '4 days ago',
                            'title': IsStr(),
                            'type': 'web_search_result',
                            'url': 'https://www.sfchronicle.com/weather-forecast/article/weather-forecast-san-francisco-21043269.php',
                        },
                    ],
                    tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartStartEvent(index=6, part=TextPart(content='Base'), previous_part_kind='builtin-tool-return'),
            PartDeltaEvent(
                index=6,
                delta=TextPartDelta(
                    content_delta="d on the search results, I can provide you with information about San Francisco's weather"
                ),
            ),
            PartDeltaEvent(
                index=6,
                delta=TextPartDelta(
                    content_delta="""\
 today (September 16, 2025):

According\
"""
                ),
            ),
            PartDeltaEvent(index=6, delta=TextPartDelta(content_delta=" to AccuWeather's forecast, ")),
            PartEndEvent(
                index=6,
                part=TextPart(
                    content="""\
Based on the search results, I can provide you with information about San Francisco's weather today (September 16, 2025):

According to AccuWeather's forecast, \
"""
                ),
                next_part_kind='text',
            ),
            PartStartEvent(
                index=7,
                part=TextPart(content='today (September 16) shows a high of 76°F and low of 59°F'),
                previous_part_kind='text',
            ),
            PartEndEvent(
                index=7,
                part=TextPart(content='today (September 16) shows a high of 76°F and low of 59°F'),
                next_part_kind='text',
            ),
            PartStartEvent(
                index=8,
                part=TextPart(
                    content="""\
 for San Francisco.

From the recent San\
"""
                ),
                previous_part_kind='text',
            ),
            PartDeltaEvent(index=8, delta=TextPartDelta(content_delta=' Francisco Chronicle weather report, ')),
            PartEndEvent(
                index=8,
                part=TextPart(
                    content="""\
 for San Francisco.

From the recent San Francisco Chronicle weather report, \
"""
                ),
                next_part_kind='text',
            ),
            PartStartEvent(
                index=9,
                part=TextPart(content='average mid-September highs in San Francisco are around 70 degrees'),
                previous_part_kind='text',
            ),
            PartEndEvent(
                index=9,
                part=TextPart(content='average mid-September highs in San Francisco are around 70 degrees'),
                next_part_kind='text',
            ),
            PartStartEvent(
                index=10, part=TextPart(content=", so today's forecast of 76°F is"), previous_part_kind='text'
            ),
            PartDeltaEvent(
                index=10,
                delta=TextPartDelta(
                    content_delta="""\
 slightly above the typical temperature for this time of year.

The\
"""
                ),
            ),
            PartDeltaEvent(
                index=10,
                delta=TextPartDelta(
                    content_delta="""\
 general weather pattern for San Francisco in September includes:
- \
"""
                ),
            ),
            PartEndEvent(
                index=10,
                part=TextPart(
                    content="""\
, so today's forecast of 76°F is slightly above the typical temperature for this time of year.

The general weather pattern for San Francisco in September includes:
- \
"""
                ),
                next_part_kind='text',
            ),
            PartStartEvent(
                index=11,
                part=TextPart(
                    content='Daytime temperatures usually reach 22°C (72°F) in San Francisco in September, falling to 13°C'
                ),
                previous_part_kind='text',
            ),
            PartDeltaEvent(index=11, delta=TextPartDelta(content_delta=' (55°F) at night')),
            PartEndEvent(
                index=11,
                part=TextPart(
                    content='Daytime temperatures usually reach 22°C (72°F) in San Francisco in September, falling to 13°C (55°F) at night'
                ),
                next_part_kind='text',
            ),
            PartStartEvent(
                index=12,
                part=TextPart(
                    content="""\

- \
"""
                ),
                previous_part_kind='text',
            ),
            PartEndEvent(
                index=12,
                part=TextPart(
                    content="""\

- \
"""
                ),
                next_part_kind='text',
            ),
            PartStartEvent(
                index=13,
                part=TextPart(content='There are normally 9 hours of bright sunshine each day in San Francisco in'),
                previous_part_kind='text',
            ),
            PartDeltaEvent(index=13, delta=TextPartDelta(content_delta=' September')),
            PartEndEvent(
                index=13,
                part=TextPart(
                    content='There are normally 9 hours of bright sunshine each day in San Francisco in September'
                ),
                next_part_kind='text',
            ),
            PartStartEvent(
                index=14,
                part=TextPart(
                    content="""\

- \
"""
                ),
                previous_part_kind='text',
            ),
            PartEndEvent(
                index=14,
                part=TextPart(
                    content="""\

- \
"""
                ),
                next_part_kind='text',
            ),
            PartStartEvent(
                index=15,
                part=TextPart(
                    content='San Francisco experiences minimal rainfall in September, with an average precipitation of just 3mm.'
                ),
                previous_part_kind='text',
            ),
            PartDeltaEvent(index=15, delta=TextPartDelta(content_delta=' Typically, there are no rainy days')),
            PartDeltaEvent(index=15, delta=TextPartDelta(content_delta=' during this month')),
            PartEndEvent(
                index=15,
                part=TextPart(
                    content='San Francisco experiences minimal rainfall in September, with an average precipitation of just 3mm. Typically, there are no rainy days during this month'
                ),
                next_part_kind='text',
            ),
            PartStartEvent(
                index=16,
                part=TextPart(
                    content="""\


So for today, you can expect partly sunny to sunny skies with a\
"""
                ),
                previous_part_kind='text',
            ),
            PartDeltaEvent(index=16, delta=TextPartDelta(content_delta=' high around 76°F (24°C)')),
            PartDeltaEvent(index=16, delta=TextPartDelta(content_delta=' and a low around 59°F (15°C),')),
            PartDeltaEvent(index=16, delta=TextPartDelta(content_delta=" with very little chance of rain. It's sh")),
            PartDeltaEvent(
                index=16, delta=TextPartDelta(content_delta='aping up to be a pleasant day in San Francisco!')
            ),
            PartEndEvent(
                index=16,
                part=TextPart(
                    content="""\


So for today, you can expect partly sunny to sunny skies with a high around 76°F (24°C) and a low around 59°F (15°C), with very little chance of rain. It's shaping up to be a pleasant day in San Francisco!\
"""
                ),
            ),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='web_search',
                    args='{"query": "San Francisco weather today"}',
                    tool_call_id='srvtoolu_01FYcUbzEaqqQh1WBRj1QX3h',
                    provider_name='anthropic',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='web_search',
                    content=[
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '6 days ago',
                            'title': 'San Francisco, CA Weather Forecast | AccuWeather',
                            'type': 'web_search_result',
                            'url': 'https://www.accuweather.com/en/us/san-francisco/94103/weather-forecast/347629',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '6 days ago',
                            'title': '10-Day Weather Forecast for San Francisco, CA - The Weather Channel | weather.com',
                            'type': 'web_search_result',
                            'url': 'https://weather.com/weather/tenday/l/San+Francisco+CA+USCA0987:1:US',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'Weather Forecast and Conditions for San Francisco, CA - The Weather Channel | Weather.com',
                            'type': 'web_search_result',
                            'url': 'https://weather.com/weather/today/l/USCA0987:1:US',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'San Francisco, CA 10-Day Weather Forecast | Weather Underground',
                            'type': 'web_search_result',
                            'url': 'https://www.wunderground.com/forecast/us/ca/san-francisco',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '1 week ago',
                            'title': 'National Weather Service',
                            'type': 'web_search_result',
                            'url': 'https://forecast.weather.gov/MapClick.php?lat=37.7771&lon=-122.4196',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '1 week ago',
                            'title': 'San Francisco Bay Area weather forecast – NBC Bay Area',
                            'type': 'web_search_result',
                            'url': 'https://www.nbcbayarea.com/weather/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'San Francisco, CA Current Weather - The Weather Network',
                            'type': 'web_search_result',
                            'url': 'https://www.theweathernetwork.com/en/city/us/california/san-francisco/current?_guid_iss_=1',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '6 days ago',
                            'title': 'San Francisco, CA Weather Conditions | Weather Underground',
                            'type': 'web_search_result',
                            'url': 'https://www.wunderground.com/weather/us/ca/san-francisco',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'San Francisco, CA Hourly Weather Forecast | Weather Underground',
                            'type': 'web_search_result',
                            'url': 'https://www.wunderground.com/hourly/us/ca/san-francisco',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '1 week ago',
                            'title': 'Live Doppler 7 | Bay Area Weather News - ABC7 San Francisco',
                            'type': 'web_search_result',
                            'url': 'https://abc7news.com/weather/',
                        },
                    ],
                    tool_call_id='srvtoolu_01FYcUbzEaqqQh1WBRj1QX3h',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                )
            ),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='web_search',
                    args='{"query": "San Francisco weather September 16 2025"}',
                    tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx',
                    provider_name='anthropic',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='web_search',
                    content=[
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'San Francisco weather in September 2025 | Weather25.com',
                            'type': 'web_search_result',
                            'url': 'https://www.weather25.com/north-america/usa/california/san-francisco?page=month&month=September',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'Weather in San Francisco in September 2025 (California) - detailed Weather Forecast for a month',
                            'type': 'web_search_result',
                            'url': 'https://world-weather.info/forecast/usa/san_francisco/september-2025/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'San Francisco, CA Monthly Weather | AccuWeather',
                            'type': 'web_search_result',
                            'url': 'https://www.accuweather.com/en/us/san-francisco/94103/september-weather/347629',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'Weather San Francisco in September 2025: Temperature & Climate',
                            'type': 'web_search_result',
                            'url': 'https://en.climate-data.org/north-america/united-states-of-america/california/san-francisco-385/t/september-9/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'San Francisco weather in September 2025 | California',
                            'type': 'web_search_result',
                            'url': 'https://www.weather2travel.com/california/san-francisco/september/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'San Francisco, Weather for September, USA',
                            'type': 'web_search_result',
                            'url': 'https://www.holiday-weather.com/san_francisco/averages/september/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'Monthly Weather Forecast for San Francisco, CA - weather.com',
                            'type': 'web_search_result',
                            'url': 'https://weather.com/weather/monthly/l/69bedc6a5b6e977993fb3e5344e3c06d8bc36a1fb6754c3ddfb5310a3c6d6c87',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '3 weeks ago',
                            'title': 'September 2025 Weather - San Francisco',
                            'type': 'web_search_result',
                            'url': 'https://www.easeweather.com/north-america/united-states/california/city-and-county-of-san-francisco/san-francisco/september',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'San Francisco Weather in September | Thomas Cook',
                            'type': 'web_search_result',
                            'url': 'https://www.thomascook.com/holidays/weather/usa/california/san-francisco/september/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '4 days ago',
                            'title': IsStr(),
                            'type': 'web_search_result',
                            'url': 'https://www.sfchronicle.com/weather-forecast/article/weather-forecast-san-francisco-21043269.php',
                        },
                    ],
                    tool_call_id='srvtoolu_01FDqc7ruGpVRoNuD5G6jkUx',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                )
            ),
        ]
    )


@pytest.mark.vcr()
async def test_anthropic_web_fetch_tool(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(api_key=anthropic_api_key))
    settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 3000})
    agent = Agent(m, builtin_tools=[WebFetchTool()], model_settings=settings)

    result = await agent.run(
        'What is the first sentence on the page https://ai.pydantic.dev? Reply with only the sentence.'
    )

    assert result.output == snapshot(
        'Pydantic AI is a Python agent framework designed to help you quickly, confidently, and painlessly build production grade applications and workflows with Generative AI.'
    )

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the first sentence on the page https://ai.pydantic.dev? Reply with only the sentence.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="""\
The user is asking me to fetch the content from https://ai.pydantic.dev and return only the first sentence on that page. I need to use the web_fetch tool to get the content from this URL, then identify the first sentence and return only that sentence.

Let me fetch the page first.\
""",
                        signature='EsIDCkYICRgCKkAKi/j4a8lGN12CjyS27ZXcPkXHGyTbn1vJENJz+AjinyTnsrynMEhidWT5IMNAs0TDgwSwPLNmgq4MsPkVekB8EgxetaK+Nhg8wUdhTEAaDMukODgr3JaYHZwVEiIwgKBckFLJ/C7wCD9oGCIECbqpaeEuWQ8BH3Hev6wpuc+66Wu7AJM1jGH60BpsUovnKqkCrHNq6b1SDT41cm2w7cyxZggrX6crzYh0fAkZ+VC6FBjy6mJikZtX6reKD+064KZ4F1oe4Qd40EBp/wHvD7oPV/fhGut1fzwl48ZgB8uzJb3tHr9MBjs4PVTsvKstpHKpOo6NLvCknQJ/0730OTENp/JOR6h6RUl6kMl5OrHTvsDEYpselUBPtLikm9p4t+d8CxqGm/B1kg1wN3FGJK31PD3veYIOO4hBirFPXWd+AiB1rZP++2QjToZ9lD2xqP/Q3vWEU+/Ryp6uzaRFWPVQkIr+mzpIaJsYuKDiyduxF4LD/hdMTV7IVDtconeQIPQJRhuO6nICBEuqb0uIotPDnCU6iI2l9OyEeKJM0RS6/NTNG8DZnvyVJ8gGKbtZKSHK6KKsdH0f7d+DGAE=',
                        provider_name='anthropic',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_fetch',
                        args={'url': 'https://ai.pydantic.dev'},
                        tool_call_id=IsStr(),
                        provider_name='anthropic',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_fetch',
                        content={
                            'content': {
                                'citations': None,
                                'source': {
                                    'data': IsStr(),
                                    'media_type': 'text/plain',
                                    'type': 'text',
                                },
                                'title': 'Pydantic AI',
                                'type': 'document',
                            },
                            'retrieved_at': IsStr(),
                            'type': 'web_fetch_result',
                            'url': 'https://ai.pydantic.dev',
                        },
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='anthropic',
                    ),
                    TextPart(
                        content='Pydantic AI is a Python agent framework designed to help you quickly, confidently, and painlessly build production grade applications and workflows with Generative AI.'
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=7262,
                    output_tokens=171,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 7262,
                        'output_tokens': 171,
                    },
                ),
                model_name='claude-sonnet-4-20250514',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    # Second run to test message replay (multi-turn conversation)
    result2 = await agent.run(
        'Based on the page you just fetched, what framework does it mention?',
        message_history=result.all_messages(),
    )

    assert 'Pydantic AI' in result2.output or 'pydantic' in result2.output.lower()
    assert result2.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the first sentence on the page https://ai.pydantic.dev? Reply with only the sentence.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="""\
The user is asking me to fetch the content from https://ai.pydantic.dev and return only the first sentence on that page. I need to use the web_fetch tool to get the content from this URL, then identify the first sentence and return only that sentence.

Let me fetch the page first.\
""",
                        signature='EsIDCkYICRgCKkAKi/j4a8lGN12CjyS27ZXcPkXHGyTbn1vJENJz+AjinyTnsrynMEhidWT5IMNAs0TDgwSwPLNmgq4MsPkVekB8EgxetaK+Nhg8wUdhTEAaDMukODgr3JaYHZwVEiIwgKBckFLJ/C7wCD9oGCIECbqpaeEuWQ8BH3Hev6wpuc+66Wu7AJM1jGH60BpsUovnKqkCrHNq6b1SDT41cm2w7cyxZggrX6crzYh0fAkZ+VC6FBjy6mJikZtX6reKD+064KZ4F1oe4Qd40EBp/wHvD7oPV/fhGut1fzwl48ZgB8uzJb3tHr9MBjs4PVTsvKstpHKpOo6NLvCknQJ/0730OTENp/JOR6h6RUl6kMl5OrHTvsDEYpselUBPtLikm9p4t+d8CxqGm/B1kg1wN3FGJK31PD3veYIOO4hBirFPXWd+AiB1rZP++2QjToZ9lD2xqP/Q3vWEU+/Ryp6uzaRFWPVQkIr+mzpIaJsYuKDiyduxF4LD/hdMTV7IVDtconeQIPQJRhuO6nICBEuqb0uIotPDnCU6iI2l9OyEeKJM0RS6/NTNG8DZnvyVJ8gGKbtZKSHK6KKsdH0f7d+DGAE=',
                        provider_name='anthropic',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_fetch',
                        args={'url': 'https://ai.pydantic.dev'},
                        tool_call_id=IsStr(),
                        provider_name='anthropic',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_fetch',
                        content={
                            'content': {
                                'citations': None,
                                'source': {
                                    'data': IsStr(),
                                    'media_type': 'text/plain',
                                    'type': 'text',
                                },
                                'title': 'Pydantic AI',
                                'type': 'document',
                            },
                            'retrieved_at': IsStr(),
                            'type': 'web_fetch_result',
                            'url': 'https://ai.pydantic.dev',
                        },
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='anthropic',
                    ),
                    TextPart(
                        content='Pydantic AI is a Python agent framework designed to help you quickly, confidently, and painlessly build production grade applications and workflows with Generative AI.'
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=7262,
                    output_tokens=171,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 7262,
                        'output_tokens': 171,
                    },
                ),
                model_name='claude-sonnet-4-20250514',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Based on the page you just fetched, what framework does it mention?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="""\
The user is asking about what framework is mentioned on the Pydantic AI page that I just fetched. Looking at the content, I can see several frameworks mentioned:

1. Pydantic AI itself - described as "a Python agent framework"
2. FastAPI - mentioned as having "revolutionized web development by offering an innovative and ergonomic design"
3. Various other frameworks/libraries mentioned like LangChain, LlamaIndex, AutoGPT, Transformers, CrewAI, Instructor
4. Pydantic Validation is mentioned as being used by many frameworks
5. OpenTelemetry is mentioned in relation to observability

But the most prominently featured framework that seems to be the main comparison point is FastAPI, as the page talks about bringing "that FastAPI feeling to GenAI app and agent development."\
""",
                        signature='ErIHCkYICRgCKkDZrwipmaxoEat4WffzPSjVzIuSQWM2sHE6FLC2wt5S2qiJN2MQh//EImuLE9I2ssZjTMxGXZV+esnf5ipnzbvnEgxfcXs2ax8vnLdroxMaDCpqvdPKpCP3Qi0txCIw55NdOjY30P3/yRL9RF8sPGioyitlzkhSpf+PuC3YXwz4N0hoy8zVY1MHecwc60vcKpkGxtZsfqmAuJwjeGRr/Ugxcxd69+0X/Y9pojMiklNHq9otW+ehDX0rR0EzfdN/2jNOs3bOrzfy9jmvYE5FU2c5e0JpMP3LH0LrFvZYkSh7RkbhYuHvrOqohlE3BhpflrszowmiozUk+aG4wSqx5Dtxo9W7jfeU4wduy6OyEFdIqdYdTMR8VVf9Qnd5bLX4rY09xcGQc4JcX2mFjdSR2WgEJM7p5lytlN5unH3selWBVPbCj7ogU8DbT9zhY3zkDW1dMt2vNbWNaY4gVrLwi42qBJvjC5eJTADckvXAt+MCT9AAe1kmH9NlsgBnRy13O4lhXv9SPNDfk2tU5Tdco4h/I/fXh+WuPe6/MKk+tJuoBQTGVQ5ryFmomsNiwhwtLbQ44fLVHhyqEKSEdo/107xvbzhjmY/MAzn1Pmc9rd+OhFsjUCvgqI8cWNc/E694eJqg3J2S+I6YRzG3d2tR7laUivf+J38c2XmwSyXfdRoJpyZ9TixubpPk04WSchdFlEkxPBGEWLDkWOVL1PG5ztY48di7EzM1tvAwiT1BOxl4WRZ78Ewc+C5BVHwT658rIrcKJXXI/zBMsoReQT9xsRhpozbb576wNXggJdZsd2ysQY0O6Pihz54emwigm+zPbO5n8HvlrGKf6dSsrwusUJ1BIY4wI6qjz7gweRryReDEvEzMT8Ul4mIrigRy4yL2w+03qAclz8oGwxinMvcu8vJzXg+uRm/WbOgyco4gTPQiN4NcXbzwhVtJlNWZYXCiiMb/i6IXuOzZmSjI7LqxLubD9RgOy/2890RLvVJQBBVnOowW8q+iE93CoVBr1l5D54opLS9fHYcM7ezV0Ul34qMu6K0uoBG0+aLVlZHKEecN2/VE4fh0zYEDaeqRZfNH2gnAGmokdmPtEHlp33pvJ0IFDAbxKq2CVFFdB+lCGlaLQuZ5v6Mhq4b6H8DjaGZqo/vcB/MK4pr/F1SRjLzSHyh7Ey4ogBYSOXWfaeXQiZZFoEfxIUG9PzofIA1CCFk+eZSG7bGY4wXe2Whhh5bs+cJ3duYI9SL+49WBABgB',
                        provider_name='anthropic',
                    ),
                    TextPart(
                        content="""\
Based on the page I fetched, the main framework it mentions and compares itself to is **FastAPI**. The page states that "FastAPI revolutionized web development by offering an innovative and ergonomic design" and that Pydantic AI was built with the aim "to bring that FastAPI feeling to GenAI app and agent development."

The page also mentions several other frameworks and libraries including:
- LangChain
- LlamaIndex  \n\
- AutoGPT
- Transformers
- CrewAI
- Instructor

It notes that "virtually every Python agent framework and LLM library" uses Pydantic Validation, which is the foundation that Pydantic AI builds upon.\
"""
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=6346,
                    output_tokens=354,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 6346,
                        'output_tokens': 354,
                    },
                ),
                model_name='claude-sonnet-4-20250514',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


@pytest.mark.vcr()
async def test_anthropic_web_fetch_tool_stream(
    allow_model_requests: None, anthropic_api_key: str
):  # pragma: lax no cover
    from pydantic_ai.messages import PartDeltaEvent, PartStartEvent

    m = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(api_key=anthropic_api_key))
    settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 3000})
    agent = Agent(m, builtin_tools=[WebFetchTool()], model_settings=settings)

    # Iterate through the stream to ensure streaming code paths are covered
    event_parts: list[Any] = []
    async with agent.iter(  # pragma: lax no cover
        user_prompt='What is the first sentence on the page https://ai.pydantic.dev? Reply with only the sentence.'
    ) as agent_run:
        async for node in agent_run:  # pragma: lax no cover
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):  # pragma: lax no cover
                async with node.stream(agent_run.ctx) as request_stream:  # pragma: lax no cover
                    async for event in request_stream:  # pragma: lax no cover
                        if (  # pragma: lax no cover
                            isinstance(event, PartStartEvent)
                            and isinstance(event.part, BuiltinToolCallPart | BuiltinToolReturnPart)
                        ) or isinstance(event, PartDeltaEvent):
                            event_parts.append(event)

    assert agent_run.result is not None
    assert agent_run.result.output == snapshot(
        'Pydantic AI is a Python agent framework designed to help you quickly, confidently, and painlessly build production grade applications and workflows with Generative AI.'
    )

    assert agent_run.result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the first sentence on the page https://ai.pydantic.dev? Reply with only the sentence.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='The user wants me to fetch the content from the URL https://ai.pydantic.dev and provide only the first sentence from that page. I need to use the web_fetch tool to get the content from this URL.',
                        signature='EusCCkYICRgCKkAG/7zhRcmUoiMtml5iZUXVv3nqupp8kgk0nrq9zOoklaXzVCnrb9kwLNWGETIcCaAnLd0cd0ESwjslkVKdV9n8EgxKKdu8LlEvh9VGIWIaDAJ2Ja2NEacp1Am6jSIwyNO36tV+Sj+q6dWf79U+3KOIa1khXbIYarpkIViCuYQaZwpJ4Vtedrd7dLWTY2d5KtIB9Pug5UPuvepSOjyhxLaohtGxmdvZN8crGwBdTJYF9GHSli/rzvkR6CpH+ixd8iSopwFcsJgQ3j68fr/yD7cHmZ06jU3LaESVEBwTHnlK0ABiYnGvD3SvX6PgImMSQxQ1ThARFTA7DePoWw+z5DI0L2vgSun2qTYHkmGxzaEskhNIBlK9r7wS3tVcO0Di4lD/rhYV61tklL2NBWJqvm7ZCtJTN09CzPFJy7HDkg7bSINVL4kuu9gTWEtb/o40tw1b+sO62UcfxQTVFQ4Cj8D8XFZbGAE=',
                        provider_name='anthropic',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_fetch',
                        args='{"url": "https://ai.pydantic.dev"}',
                        tool_call_id=IsStr(),
                        provider_name='anthropic',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_fetch',
                        content={
                            'content': {
                                'citations': None,
                                'source': {
                                    'data': IsStr(),
                                    'media_type': 'text/plain',
                                    'type': 'text',
                                },
                                'title': 'Pydantic AI',
                                'type': 'document',
                            },
                            'retrieved_at': IsStr(),
                            'type': 'web_fetch_result',
                            'url': 'https://ai.pydantic.dev',
                        },
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='anthropic',
                    ),
                    TextPart(
                        content='Pydantic AI is a Python agent framework designed to help you quickly, confidently, and painlessly build production grade applications and workflows with Generative AI.'
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=7244,
                    output_tokens=153,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 7244,
                        'output_tokens': 153,
                    },
                ),
                model_name='claude-sonnet-4-20250514',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )
    assert event_parts == snapshot(
        [
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='The user wants', provider_name='anthropic')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' me to fetch', provider_name='anthropic')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' the content', provider_name='anthropic')),
            PartDeltaEvent(
                index=0, delta=ThinkingPartDelta(content_delta=' from the URL https', provider_name='anthropic')
            ),
            PartDeltaEvent(
                index=0, delta=ThinkingPartDelta(content_delta='://ai.pydantic.dev', provider_name='anthropic')
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' and provide', provider_name='anthropic')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' only', provider_name='anthropic')),
            PartDeltaEvent(
                index=0, delta=ThinkingPartDelta(content_delta=' the first sentence from', provider_name='anthropic')
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' that page.', provider_name='anthropic')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta=' I need to use the web_fetch', provider_name='anthropic'),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' tool to', provider_name='anthropic')),
            PartDeltaEvent(
                index=0, delta=ThinkingPartDelta(content_delta=' get the content from', provider_name='anthropic')
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' this URL.', provider_name='anthropic')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    signature_delta='EusCCkYICRgCKkAG/7zhRcmUoiMtml5iZUXVv3nqupp8kgk0nrq9zOoklaXzVCnrb9kwLNWGETIcCaAnLd0cd0ESwjslkVKdV9n8EgxKKdu8LlEvh9VGIWIaDAJ2Ja2NEacp1Am6jSIwyNO36tV+Sj+q6dWf79U+3KOIa1khXbIYarpkIViCuYQaZwpJ4Vtedrd7dLWTY2d5KtIB9Pug5UPuvepSOjyhxLaohtGxmdvZN8crGwBdTJYF9GHSli/rzvkR6CpH+ixd8iSopwFcsJgQ3j68fr/yD7cHmZ06jU3LaESVEBwTHnlK0ABiYnGvD3SvX6PgImMSQxQ1ThARFTA7DePoWw+z5DI0L2vgSun2qTYHkmGxzaEskhNIBlK9r7wS3tVcO0Di4lD/rhYV61tklL2NBWJqvm7ZCtJTN09CzPFJy7HDkg7bSINVL4kuu9gTWEtb/o40tw1b+sO62UcfxQTVFQ4Cj8D8XFZbGAE=',
                    provider_name='anthropic',
                ),
            ),
            PartStartEvent(
                index=1,
                part=BuiltinToolCallPart(tool_name='web_fetch', tool_call_id=IsStr(), provider_name='anthropic'),
                previous_part_kind='thinking',
            ),
            PartDeltaEvent(
                index=1, delta=ToolCallPartDelta(args_delta='', tool_call_id='srvtoolu_018ADaxdJjyZ8HXtF3sTBPNk')
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='{"url": "', tool_call_id='srvtoolu_018ADaxdJjyZ8HXtF3sTBPNk'),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='https://ai', tool_call_id='srvtoolu_018ADaxdJjyZ8HXtF3sTBPNk'),
            ),
            PartDeltaEvent(
                index=1, delta=ToolCallPartDelta(args_delta='.p', tool_call_id='srvtoolu_018ADaxdJjyZ8HXtF3sTBPNk')
            ),
            PartDeltaEvent(
                index=1, delta=ToolCallPartDelta(args_delta='yd', tool_call_id='srvtoolu_018ADaxdJjyZ8HXtF3sTBPNk')
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='antic.dev"}', tool_call_id='srvtoolu_018ADaxdJjyZ8HXtF3sTBPNk'),
            ),
            PartStartEvent(
                index=2,
                part=BuiltinToolReturnPart(
                    tool_name='web_fetch',
                    content={
                        'content': {
                            'citations': None,
                            'source': {
                                'data': '''\
Pydantic AI
GenAI Agent Framework, the Pydantic way
Pydantic AI is a Python agent framework designed to help you quickly, confidently, and painlessly build production grade applications and workflows with Generative AI.
FastAPI revolutionized web development by offering an innovative and ergonomic design, built on the foundation of [Pydantic Validation](https://docs.pydantic.dev) and modern Python features like type hints.
Yet despite virtually every Python agent framework and LLM library using Pydantic Validation, when we began to use LLMs in [Pydantic Logfire](https://pydantic.dev/logfire), we couldn't find anything that gave us the same feeling.
We built Pydantic AI with one simple aim: to bring that FastAPI feeling to GenAI app and agent development.
Why use Pydantic AI
-
Built by the Pydantic Team:
[Pydantic Validation](https://docs.pydantic.dev/latest/)is the validation layer of the OpenAI SDK, the Google ADK, the Anthropic SDK, LangChain, LlamaIndex, AutoGPT, Transformers, CrewAI, Instructor and many more. Why use the derivative when you can go straight to the source? -
Model-agnostic: Supports virtually every
[model](models/overview/)and provider: OpenAI, Anthropic, Gemini, DeepSeek, Grok, Cohere, Mistral, and Perplexity; Azure AI Foundry, Amazon Bedrock, Google Vertex AI, Ollama, LiteLLM, Groq, OpenRouter, Together AI, Fireworks AI, Cerebras, Hugging Face, GitHub, Heroku, Vercel, Nebius, OVHcloud, and Outlines. If your favorite model or provider is not listed, you can easily implement a[custom model](models/overview/#custom-models). -
Seamless Observability: Tightly
[integrates](logfire/)with[Pydantic Logfire](https://pydantic.dev/logfire), our general-purpose OpenTelemetry observability platform, for real-time debugging, evals-based performance monitoring, and behavior, tracing, and cost tracking. If you already have an observability platform that supports OTel, you can[use that too](logfire/#alternative-observability-backends). -
Fully Type-safe: Designed to give your IDE or AI coding agent as much context as possible for auto-completion and
[type checking](agents/#static-type-checking), moving entire classes of errors from runtime to write-time for a bit of that Rust "if it compiles, it works" feel. -
Powerful Evals: Enables you to systematically test and
[evaluate](evals/)the performance and accuracy of the agentic systems you build, and monitor the performance over time in Pydantic Logfire. -
MCP, A2A, and UI: Integrates the
[Model Context Protocol](mcp/overview/),[Agent2Agent](a2a/), and various[UI event stream](ui/overview/)standards to give your agent access to external tools and data, let it interoperate with other agents, and build interactive applications with streaming event-based communication. -
Human-in-the-Loop Tool Approval: Easily lets you flag that certain tool calls
[require approval](deferred-tools/#human-in-the-loop-tool-approval)before they can proceed, possibly depending on tool call arguments, conversation history, or user preferences. -
Durable Execution: Enables you to build
[durable agents](durable_execution/overview/)that can preserve their progress across transient API failures and application errors or restarts, and handle long-running, asynchronous, and human-in-the-loop workflows with production-grade reliability. -
Streamed Outputs: Provides the ability to
[stream](output/#streamed-results)structured output continuously, with immediate validation, ensuring real time access to generated data. -
Graph Support: Provides a powerful way to define
[graphs](graph/)using type hints, for use in complex applications where standard control flow can degrade to spaghetti code.
Realistically though, no list is going to be as convincing as [giving it a try](#next-steps) and seeing how it makes you feel!
Sign up for our newsletter, The Pydantic Stack, with updates & tutorials on Pydantic AI, Logfire, and Pydantic:
Hello World Example
Here's a minimal example of Pydantic AI:
[Learn about Gateway](gateway)hello_world.py
from pydantic_ai import Agent
agent = Agent( # (1)!
'gateway/anthropic:claude-sonnet-4-0',
instructions='Be concise, reply with one sentence.', # (2)!
)
result = agent.run_sync('Where does "hello world" come from?') # (3)!
print(result.output)
"""
The first known use of "hello, world" was in a 1974 textbook about the C programming language.
"""
- We configure the agent to use
[Anthropic's Claude Sonnet 4.0](api/models/anthropic/)model, but you can also set the model when running the agent. - Register static
[instructions](agents/#instructions)using a keyword argument to the agent. [Run the agent](agents/#running-agents)synchronously, starting a conversation with the LLM.
from pydantic_ai import Agent
agent = Agent( # (1)!
'anthropic:claude-sonnet-4-0',
instructions='Be concise, reply with one sentence.', # (2)!
)
result = agent.run_sync('Where does "hello world" come from?') # (3)!
print(result.output)
"""
The first known use of "hello, world" was in a 1974 textbook about the C programming language.
"""
- We configure the agent to use
[Anthropic's Claude Sonnet 4.0](api/models/anthropic/)model, but you can also set the model when running the agent. - Register static
[instructions](agents/#instructions)using a keyword argument to the agent. [Run the agent](agents/#running-agents)synchronously, starting a conversation with the LLM.
(This example is complete, it can be run "as is", assuming you've [installed the pydantic_ai package](install/))
The exchange will be very short: Pydantic AI will send the instructions and the user prompt to the LLM, and the model will return a text response.
Not very interesting yet, but we can easily add [tools](tools/), [dynamic instructions](agents/#instructions), and [structured outputs](output/) to build more powerful agents.
Tools & Dependency Injection Example
Here is a concise example using Pydantic AI to build a support agent for a bank:
[Learn about Gateway](gateway)bank_support.py
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from bank_database import DatabaseConn
@dataclass
class SupportDependencies: # (3)!
customer_id: int
db: DatabaseConn # (12)!
class SupportOutput(BaseModel): # (13)!
support_advice: str = Field(description='Advice returned to the customer')
block_card: bool = Field(description="Whether to block the customer's card")
risk: int = Field(description='Risk level of query', ge=0, le=10)
support_agent = Agent( # (1)!
'gateway/openai:gpt-5', # (2)!
deps_type=SupportDependencies,
output_type=SupportOutput, # (9)!
instructions=( # (4)!
'You are a support agent in our bank, give the '
'customer support and judge the risk level of their query.'
),
)
@support_agent.instructions # (5)!
async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
return f"The customer's name is {customer_name!r}"
@support_agent.tool # (6)!
async def customer_balance(
ctx: RunContext[SupportDependencies], include_pending: bool
) -> float:
"""Returns the customer's current account balance.""" # (7)!
return await ctx.deps.db.customer_balance(
id=ctx.deps.customer_id,
include_pending=include_pending,
)
... # (11)!
async def main():
deps = SupportDependencies(customer_id=123, db=DatabaseConn())
result = await support_agent.run('What is my balance?', deps=deps) # (8)!
print(result.output) # (10)!
"""
support_advice='Hello John, your current account balance, including pending transactions, is $123.45.' block_card=False risk=1
"""
result = await support_agent.run('I just lost my card!', deps=deps)
print(result.output)
"""
support_advice="I'm sorry to hear that, John. We are temporarily blocking your card to prevent unauthorized transactions." block_card=True risk=8
"""
- This
[agent](agents/)will act as first-tier support in a bank. Agents are generic in the type of dependencies they accept and the type of output they return. In this case, the support agent has typeAgent[SupportDependencies, SupportOutput]
. - Here we configure the agent to use
[OpenAI's GPT-5 model](api/models/openai/), you can also set the model when running the agent. - The
SupportDependencies
dataclass is used to pass data, connections, and logic into the model that will be needed when running[instructions](agents/#instructions)and[tool](tools/)functions. Pydantic AI's system of dependency injection provides a[type-safe](agents/#static-type-checking)way to customise the behavior of your agents, and can be especially useful when running[unit tests](testing/)and evals. - Static
[instructions](agents/#instructions)can be registered with theto the agent.instructions
keyword argument - Dynamic
[instructions](agents/#instructions)can be registered with thedecorator, and can make use of dependency injection. Dependencies are carried via the@agent.instructions
argument, which is parameterized with theRunContext
deps_type
from above. If the type annotation here is wrong, static type checkers will catch it. - The
decorator let you register functions which the LLM may call while responding to a user. Again, dependencies are carried via@agent.tool
, any other arguments become the tool schema passed to the LLM. Pydantic is used to validate these arguments, and errors are passed back to the LLM so it can retry.RunContext
- The docstring of a tool is also passed to the LLM as the description of the tool. Parameter descriptions are
[extracted](tools/#function-tools-and-schema)from the docstring and added to the parameter schema sent to the LLM. [Run the agent](agents/#running-agents)asynchronously, conducting a conversation with the LLM until a final response is reached. Even in this fairly simple case, the agent will exchange multiple messages with the LLM as tools are called to retrieve an output.- The response from the agent will be guaranteed to be a
SupportOutput
. If validation fails[reflection](agents/#reflection-and-self-correction), the agent is prompted to try again. - The output will be validated with Pydantic to guarantee it is a
SupportOutput
, since the agent is generic, it'll also be typed as aSupportOutput
to aid with static type checking. - In a real use case, you'd add more tools and longer instructions to the agent to extend the context it's equipped with and support it can provide.
- This is a simple sketch of a database connection, used to keep the example short and readable. In reality, you'd be connecting to an external database (e.g. PostgreSQL) to get information about customers.
- This
[Pydantic](https://docs.pydantic.dev)model is used to constrain the structured data returned by the agent. From this simple definition, Pydantic builds the JSON Schema that tells the LLM how to return the data, and performs validation to guarantee the data is correct at the end of the run.
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from bank_database import DatabaseConn
@dataclass
class SupportDependencies: # (3)!
customer_id: int
db: DatabaseConn # (12)!
class SupportOutput(BaseModel): # (13)!
support_advice: str = Field(description='Advice returned to the customer')
block_card: bool = Field(description="Whether to block the customer's card")
risk: int = Field(description='Risk level of query', ge=0, le=10)
support_agent = Agent( # (1)!
'openai:gpt-5', # (2)!
deps_type=SupportDependencies,
output_type=SupportOutput, # (9)!
instructions=( # (4)!
'You are a support agent in our bank, give the '
'customer support and judge the risk level of their query.'
),
)
@support_agent.instructions # (5)!
async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
return f"The customer's name is {customer_name!r}"
@support_agent.tool # (6)!
async def customer_balance(
ctx: RunContext[SupportDependencies], include_pending: bool
) -> float:
"""Returns the customer's current account balance.""" # (7)!
return await ctx.deps.db.customer_balance(
id=ctx.deps.customer_id,
include_pending=include_pending,
)
... # (11)!
async def main():
deps = SupportDependencies(customer_id=123, db=DatabaseConn())
result = await support_agent.run('What is my balance?', deps=deps) # (8)!
print(result.output) # (10)!
"""
support_advice='Hello John, your current account balance, including pending transactions, is $123.45.' block_card=False risk=1
"""
result = await support_agent.run('I just lost my card!', deps=deps)
print(result.output)
"""
support_advice="I'm sorry to hear that, John. We are temporarily blocking your card to prevent unauthorized transactions." block_card=True risk=8
"""
- This
[agent](agents/)will act as first-tier support in a bank. Agents are generic in the type of dependencies they accept and the type of output they return. In this case, the support agent has typeAgent[SupportDependencies, SupportOutput]
. - Here we configure the agent to use
[OpenAI's GPT-5 model](api/models/openai/), you can also set the model when running the agent. - The
SupportDependencies
dataclass is used to pass data, connections, and logic into the model that will be needed when running[instructions](agents/#instructions)and[tool](tools/)functions. Pydantic AI's system of dependency injection provides a[type-safe](agents/#static-type-checking)way to customise the behavior of your agents, and can be especially useful when running[unit tests](testing/)and evals. - Static
[instructions](agents/#instructions)can be registered with theto the agent.instructions
keyword argument - Dynamic
[instructions](agents/#instructions)can be registered with thedecorator, and can make use of dependency injection. Dependencies are carried via the@agent.instructions
argument, which is parameterized with theRunContext
deps_type
from above. If the type annotation here is wrong, static type checkers will catch it. - The
decorator let you register functions which the LLM may call while responding to a user. Again, dependencies are carried via@agent.tool
, any other arguments become the tool schema passed to the LLM. Pydantic is used to validate these arguments, and errors are passed back to the LLM so it can retry.RunContext
- The docstring of a tool is also passed to the LLM as the description of the tool. Parameter descriptions are
[extracted](tools/#function-tools-and-schema)from the docstring and added to the parameter schema sent to the LLM. [Run the agent](agents/#running-agents)asynchronously, conducting a conversation with the LLM until a final response is reached. Even in this fairly simple case, the agent will exchange multiple messages with the LLM as tools are called to retrieve an output.- The response from the agent will be guaranteed to be a
SupportOutput
. If validation fails[reflection](agents/#reflection-and-self-correction), the agent is prompted to try again. - The output will be validated with Pydantic to guarantee it is a
SupportOutput
, since the agent is generic, it'll also be typed as aSupportOutput
to aid with static type checking. - In a real use case, you'd add more tools and longer instructions to the agent to extend the context it's equipped with and support it can provide.
- This is a simple sketch of a database connection, used to keep the example short and readable. In reality, you'd be connecting to an external database (e.g. PostgreSQL) to get information about customers.
- This
[Pydantic](https://docs.pydantic.dev)model is used to constrain the structured data returned by the agent. From this simple definition, Pydantic builds the JSON Schema that tells the LLM how to return the data, and performs validation to guarantee the data is correct at the end of the run.
Complete bank_support.py
example
The code included here is incomplete for the sake of brevity (the definition of DatabaseConn
is missing); you can find the complete bank_support.py
example [here](examples/bank-support/).
Instrumentation with Pydantic Logfire
Even a simple agent with just a handful of tools can result in a lot of back-and-forth with the LLM, making it nearly impossible to be confident of what's going on just from reading the code. To understand the flow of the above runs, we can watch the agent in action using Pydantic Logfire.
To do this, we need to [set up Logfire](logfire/#using-logfire), and add the following to our code:
[Learn about Gateway](gateway)bank_support_with_logfire.py
...
from pydantic_ai import Agent, RunContext
from bank_database import DatabaseConn
import logfire
logfire.configure() # (1)!
logfire.instrument_pydantic_ai() # (2)!
logfire.instrument_asyncpg() # (3)!
...
support_agent = Agent(
'gateway/openai:gpt-5',
deps_type=SupportDependencies,
output_type=SupportOutput,
system_prompt=(
'You are a support agent in our bank, give the '
'customer support and judge the risk level of their query.'
),
)
- Configure the Logfire SDK, this will fail if project is not set up.
- This will instrument all Pydantic AI agents used from here on out. If you want to instrument only a specific agent, you can pass the
to the agent.instrument=True
keyword argument - In our demo,
DatabaseConn
usesto connect to a PostgreSQL database, soasyncpg
is used to log the database queries.logfire.instrument_asyncpg()
...
from pydantic_ai import Agent, RunContext
from bank_database import DatabaseConn
import logfire
logfire.configure() # (1)!
logfire.instrument_pydantic_ai() # (2)!
logfire.instrument_asyncpg() # (3)!
...
support_agent = Agent(
'openai:gpt-5',
deps_type=SupportDependencies,
output_type=SupportOutput,
system_prompt=(
'You are a support agent in our bank, give the '
'customer support and judge the risk level of their query.'
),
)
- Configure the Logfire SDK, this will fail if project is not set up.
- This will instrument all Pydantic AI agents used from here on out. If you want to instrument only a specific agent, you can pass the
to the agent.instrument=True
keyword argument - In our demo,
DatabaseConn
usesto connect to a PostgreSQL database, soasyncpg
is used to log the database queries.logfire.instrument_asyncpg()
That's enough to get the following view of your agent in action:
See [Monitoring and Performance](logfire/) to learn more.
llms.txt
The Pydantic AI documentation is available in the [llms.txt](https://llmstxt.org/) format.
This format is defined in Markdown and suited for LLMs and AI coding assistants and agents.
Two formats are available:
: a file containing a brief description of the project, along with links to the different sections of the documentation. The structure of this file is described in detailsllms.txt
[here](https://llmstxt.org/#format).: Similar to thellms-full.txt
llms.txt
file, but every link content is included. Note that this file may be too large for some LLMs.
As of today, these files are not automatically leveraged by IDEs or coding agents, but they will use it if you provide a link or the full text.
Next Steps
To try Pydantic AI for yourself, [install it](install/) and follow the instructions [in the examples](examples/setup/).
Read the [docs](agents/) to learn more about building applications with Pydantic AI.
Read the [API Reference](api/agent/) to understand Pydantic AI's interface.
Join [ Slack](https://logfire.pydantic.dev/docs/join-slack/) or file an issue on [ GitHub](https://github.com/pydantic/pydantic-ai/issues) if you have any questions.\
''',
                                'media_type': 'text/plain',
                                'type': 'text',
                            },
                            'title': 'Pydantic AI',
                            'type': 'document',
                        },
                        'retrieved_at': IsStr(),
                        'type': 'web_fetch_result',
                        'url': 'https://ai.pydantic.dev',
                    },
                    tool_call_id=IsStr(),
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='ydantic AI is a')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' Python')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' agent')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' framework')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' designe')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='d to help')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' you quickly')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=',')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' confi')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='dently,')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' and pain')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='lessly build production')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' grade')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' applications')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' an')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='d workflows')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' with')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta=' Gener')),
            PartDeltaEvent(index=3, delta=TextPartDelta(content_delta='ative AI.')),
        ]
    )


async def test_anthropic_web_fetch_tool_message_replay():
    """Test that BuiltinToolCallPart and BuiltinToolReturnPart for WebFetchTool are correctly serialized."""
    from typing import cast

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    # Create a model instance
    m = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(api_key='test-key'))

    # Create message history with BuiltinToolCallPart and BuiltinToolReturnPart
    messages = [
        ModelRequest(parts=[UserPromptPart(content='Test')], timestamp=IsDatetime()),
        ModelResponse(
            parts=[
                BuiltinToolCallPart(
                    provider_name=m.system,
                    tool_name=WebFetchTool.kind,
                    args={'url': 'https://example.com'},
                    tool_call_id='test_id_1',
                ),
                BuiltinToolReturnPart(
                    provider_name=m.system,
                    tool_name=WebFetchTool.kind,
                    content={
                        'content': {'type': 'document'},
                        'type': 'web_fetch_result',
                        'url': 'https://example.com',
                        'retrieved_at': '2025-01-01T00:00:00Z',
                    },
                    tool_call_id='test_id_1',
                ),
            ],
            model_name='claude-sonnet-4-0',
        ),
    ]

    # Call _map_message to trigger serialization
    model_settings = {}
    model_request_parameters = ModelRequestParameters(
        function_tools=[],
        builtin_tools=[WebFetchTool()],
        output_tools=[],
    )

    system_prompt, anthropic_messages = await m._map_message(messages, model_request_parameters, model_settings)  # pyright: ignore[reportPrivateUsage,reportArgumentType]

    # Verify the messages were serialized correctly
    assert system_prompt is None or isinstance(system_prompt, (list | str))
    assert len(anthropic_messages) == 2
    assert anthropic_messages[1]['role'] == 'assistant'

    # Check that server_tool_use block is present
    content = anthropic_messages[1]['content']
    assert any(
        isinstance(item, dict) and item.get('type') == 'server_tool_use' and item.get('name') == 'web_fetch'
        for item in content
    )

    # Check that web_fetch_tool_result block is present and contains URL and retrieved_at
    web_fetch_result = next(
        item for item in content if isinstance(item, dict) and item.get('type') == 'web_fetch_tool_result'
    )
    assert 'content' in web_fetch_result
    result_content = web_fetch_result['content']
    assert isinstance(result_content, dict)  # Type narrowing for mypy
    assert result_content['type'] == 'web_fetch_result'  # type: ignore[typeddict-item]
    assert result_content['url'] == 'https://example.com'  # type: ignore[typeddict-item]
    # retrieved_at is optional - cast to avoid complex union type issues
    assert cast(dict, result_content).get('retrieved_at') == '2025-01-01T00:00:00Z'  # pyright: ignore[reportUnknownMemberType,reportMissingTypeArgument]
    assert 'content' in result_content  # The actual document content


async def test_anthropic_web_fetch_tool_with_parameters():
    """Test that WebFetchTool parameters are correctly passed to Anthropic API."""
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    # Create a model instance
    m = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(api_key='test-key'))

    # Create WebFetchTool with all parameters
    web_fetch_tool = WebFetchTool(
        max_uses=5,
        allowed_domains=['example.com', 'ai.pydantic.dev'],
        enable_citations=True,
        max_content_tokens=50000,
    )

    model_request_parameters = ModelRequestParameters(
        function_tools=[],
        builtin_tools=[web_fetch_tool],
        output_tools=[],
    )

    # Get tools from model
    tools, _, _ = m._add_builtin_tools([], model_request_parameters)  # pyright: ignore[reportPrivateUsage]

    # Find the web_fetch tool
    web_fetch_tool_param = next((t for t in tools if t.get('name') == 'web_fetch'), None)
    assert web_fetch_tool_param is not None

    # Verify all parameters are passed correctly
    assert web_fetch_tool_param.get('type') == 'web_fetch_20250910'
    assert web_fetch_tool_param.get('max_uses') == 5
    assert web_fetch_tool_param.get('allowed_domains') == ['example.com', 'ai.pydantic.dev']
    assert web_fetch_tool_param.get('blocked_domains') is None
    assert web_fetch_tool_param.get('citations') == {'enabled': True}
    assert web_fetch_tool_param.get('max_content_tokens') == 50000


async def test_anthropic_web_fetch_tool_domain_filtering():
    """Test that blocked_domains work and are mutually exclusive with allowed_domains."""
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    # Create a model instance
    m = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(api_key='test-key'))

    # Test with blocked_domains
    web_fetch_tool = WebFetchTool(blocked_domains=['private.example.com', 'internal.example.com'])

    model_request_parameters = ModelRequestParameters(
        function_tools=[],
        builtin_tools=[web_fetch_tool],
        output_tools=[],
    )

    # Get tools from model
    tools, _, _ = m._add_builtin_tools([], model_request_parameters)  # pyright: ignore[reportPrivateUsage]

    # Find the web_fetch tool
    web_fetch_tool_param = next((t for t in tools if t.get('name') == 'web_fetch'), None)
    assert web_fetch_tool_param is not None

    # Verify blocked_domains is passed correctly
    assert web_fetch_tool_param.get('blocked_domains') == ['private.example.com', 'internal.example.com']
    assert web_fetch_tool_param.get('allowed_domains') is None


@pytest.mark.vcr()
async def test_anthropic_mcp_servers(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(api_key=anthropic_api_key))
    settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 3000})
    agent = Agent(
        m,
        builtin_tools=[
            MCPServerTool(
                id='deepwiki',
                url='https://mcp.deepwiki.com/mcp',
            )
        ],
        model_settings=settings,
    )

    result = await agent.run('Can you tell me more about the pydantic/pydantic-ai repo? Keep your answer short')
    messages = result.all_messages()
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Can you tell me more about the pydantic/pydantic-ai repo? Keep your answer short',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='The user is asking about the pydantic/pydantic-ai repository and wants me to keep the answer short. I should use the deepwiki tools to get information about this repository. Let me start by asking a general question about what this repository is about.',
                        signature='EqUDCkYICBgCKkCTiLjx5Rzw9zXo4pFDhFAc9Ci1R+d2fpkiqw7IPt1PgxBankr7bhRfh2iQOFEUy7sYVtsBxvnHW8zfBRxH1j6lEgySvdOyObrcFdJX3qkaDMAMCdLHIevZ/mSx/SIwi917U34N5jLQH1yMoCx/k72klLG5v42vcwUTG4ngKDI69Ddaf0eeDpgg3tL5FHfvKowCnslWg3Pd3ITe+TLlzu+OVZhRKU9SEwDJbjV7ZF954Ls6XExAfjdXhrhvXDB+hz6fZFPGFEfXV7jwElFT5HcGPWy84xvlwzbklZ2zH3XViik0B5dMErMAKs6IVwqXo3s+0p9xtX5gCBuvLkalET2upNsmdKGJv7WQWoaLch5N07uvSgWkO8AkGuVtBgqZH+uRGlPfYlnAgifNHu00GSAVK3beeyZfpnSQ6LQKcH+wVmrOi/3UvzA5f1LvsXG32gQKUCxztATnlBaI+7GMs1IAloaRHBndyRoe8Lwv79zZe9u9gnF9WCgK3yQsAR5hGZXlBKiIWfnRrXQ7QmA2hVO+mhEOCnz7OQkMIEUlfxgB',
                        provider_name='anthropic',
                    ),
                    BuiltinToolCallPart(
                        tool_name='mcp_server:deepwiki',
                        args={
                            'action': 'call_tool',
                            'tool_name': 'ask_question',
                            'tool_args': {
                                'repoName': 'pydantic/pydantic-ai',
                                'question': 'What is pydantic-ai and what does this repository do?',
                            },
                        },
                        tool_call_id='mcptoolu_01SAss3KEwASziHZoMR6HcZU',
                        provider_name='anthropic',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_server:deepwiki',
                        content={
                            'content': [
                                {
                                    'citations': None,
                                    'text': IsStr(),
                                    'type': 'text',
                                }
                            ],
                            'is_error': False,
                        },
                        tool_call_id='mcptoolu_01SAss3KEwASziHZoMR6HcZU',
                        timestamp=IsDatetime(),
                        provider_name='anthropic',
                    ),
                    TextPart(
                        content="""\
**Pydantic AI** is a Python agent framework for building production-grade applications with Generative AI. It provides:

- **Type-safe agents** with compile-time validation using `Agent[Deps, Output]`
- **Model-agnostic design** supporting 15+ LLM providers (OpenAI, Anthropic, Google, etc.)
- **Structured outputs** with automatic Pydantic validation and self-correction
- **Built-in observability** via OpenTelemetry and Logfire integration
- **Production tooling** including evaluation framework, durable execution, and tool system

The repo is organized as a monorepo with core packages like `pydantic-ai-slim` (core framework), `pydantic-graph` (execution engine), and `pydantic-evals` (evaluation tools). It emphasizes developer ergonomics and type safety, similar to Pydantic and FastAPI.\
"""
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=2674,
                    output_tokens=373,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 2674,
                        'output_tokens': 373,
                    },
                ),
                model_name='claude-sonnet-4-20250514',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_01MYDjkvBDRaKsY6PDwQz3n6',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    result = await agent.run(
        'How about the pydantic repo in the same org?', message_history=messages
    )  # pragma: lax no cover
    messages = result.new_messages()  # pragma: lax no cover
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='How about the pydantic repo in the same org?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='The user is asking about the pydantic repo in the same org, so that would be pydantic/pydantic. I should ask about what this repository does and provide a short answer.',
                        signature='EtECCkYICBgCKkAkKy+K3Z/q4dGwZGr1MdsH8HLaULElUSaa/Y8A1L/Jp7y1AfJd1zrTL7Zfa2KoPr0HqO/AI/cJJreheuwcn/dWEgw0bPLie900a4h9wS0aDACnsdbr+adzpUyExiIwyuNjV82BVkK/kU+sMyrfbhgb6ob/DUgudJPaK5zR6cINAAGQnIy3iOXTwu3OUfPAKrgBzF9HD5HjiPSJdsxlkI0RA5Yjiol05/hR3fUB6WWrs0aouxIzlriJ6NzmzvqctkFJdRgAL9Mh06iK1A61PLyBWRdo1f5TBziFP1c6z7iQQzH9DdcaHvG8yLoaadbyTxMvTn2PtfEcSPjuZcLgv7QcF+HZXbDVjsHJW78OK2ta0M6/xuU1p4yG3qgoss3b0G6fAyvUVgVbb1wknkE/9W9gd2k/ZSh4P7F6AcvLTXQScTyMfWRtAWQqABgB',
                        provider_name='anthropic',
                    ),
                    BuiltinToolCallPart(
                        tool_name='mcp_server:deepwiki',
                        args={
                            'action': 'call_tool',
                            'tool_name': 'ask_question',
                            'tool_args': {
                                'repoName': 'pydantic/pydantic',
                                'question': 'What is Pydantic and what does this repository do?',
                            },
                        },
                        tool_call_id='mcptoolu_01A9RvAqDeoUnaMgQc6Nn75y',
                        provider_name='anthropic',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_server:deepwiki',
                        content={
                            'content': [
                                {
                                    'citations': None,
                                    'text': """\
Pydantic is a Python library for data validation, parsing, and serialization using type hints  . This repository, `pydantic/pydantic`, contains the source code for the Pydantic library itself, including its core validation logic, documentation, and continuous integration/continuous deployment (CI/CD) pipelines  .

## What is Pydantic

Pydantic is designed to ensure that data conforms to specified types and constraints at runtime . It leverages Python type hints to define data schemas and provides mechanisms for data conversion and validation . The library's core validation logic is implemented in Rust within a separate package called `pydantic-core`, which contributes to its performance .

Pydantic offers several user-facing APIs for validation:
*   `BaseModel`: Used for defining class-based models with fields, suitable for domain models, API schemas, and configuration .
*   `TypeAdapter`: Provides a flexible way to validate and serialize arbitrary Python types, including primitive types and dataclasses .
*   `@dataclass`: Enhances Python's built-in dataclasses with Pydantic's validation capabilities .
*   `@validate_call`: Used for validating function arguments and return values .

## What this Repository Does

The `pydantic/pydantic` repository serves as the development hub for the Pydantic library. Its primary functions include:

### Core Library Development
The repository contains the Python source code for the Pydantic library, including modules for `BaseModel` , `Field` definitions , configuration management , and type adapters . It also includes internal modules responsible for model construction and schema generation .

### Documentation
The repository hosts the documentation for Pydantic, which is built using MkDocs . The documentation covers installation instructions , core concepts like models , fields, and JSON Schema generation . It also includes information on contributing to the project .

### Continuous Integration and Deployment (CI/CD)
The repository utilizes GitHub Actions for its CI/CD pipeline . This pipeline includes:
*   **Linting**: Checks code quality and style .
*   **Testing**: Runs a comprehensive test suite across multiple operating systems and Python versions . This includes memory profiling tests, Mypy plugin tests, and type-checking integration tests   .
*   **Coverage**: Aggregates test coverage data and posts comments to pull requests .
*   **Release Process**: Automates publishing new versions to PyPI and sending release announcements .
*   **Third-Party Integration Testing**: Tests Pydantic's compatibility with other popular libraries like FastAPI, SQLModel, and Beanie .
*   **Dependency Management**: Uses `uv` for managing dependencies and includes workflows to check compatibility with various dependency versions  .
*   **Performance Benchmarking**: Utilizes CodSpeed to track and analyze performance .

## Versioning and Compatibility
Pydantic maintains strict version compatibility between the pure Python package (`pydantic`) and its Rust-based validation core (`pydantic-core`)  . A `SystemError` is raised if there's a mismatch in `pydantic-core` versions, ensuring a stable environment . The `version_info()` function provides detailed version information for Pydantic and its dependencies .

Notes:
The `CITATION.cff` file also provides a concise description of Pydantic as "the most widely used data validation library for Python" . The `README.md` and `docs/index.md` files reiterate this, emphasizing its speed and extensibility  .

Wiki pages you might want to explore:
- [Overview (pydantic/pydantic)](/wiki/pydantic/pydantic#1)
- [Development and Deployment (pydantic/pydantic)](/wiki/pydantic/pydantic#7)

View this search on DeepWiki: https://deepwiki.com/search/what-is-pydantic-and-what-does_dab96efa-752a-4688-a630-3f4658084a88
""",
                                    'type': 'text',
                                }
                            ],
                            'is_error': False,
                        },
                        tool_call_id='mcptoolu_01A9RvAqDeoUnaMgQc6Nn75y',
                        timestamp=IsDatetime(),
                        provider_name='anthropic',
                    ),
                    TextPart(
                        content="""\
**Pydantic** is Python's most widely used data validation library for parsing, validation, and serialization using type hints. The repository contains:

**Core Features:**
- **Data validation** with automatic type conversion and constraint checking
- **Multiple APIs**: `BaseModel` for class-based models, `TypeAdapter` for arbitrary types, `@dataclass` decorator, and `@validate_call` for functions
- **High performance** via Rust-based validation core (`pydantic-core`)
- **JSON Schema generation** and comprehensive serialization support

**Repository Contents:**
- Python source code for the main Pydantic library
- Comprehensive documentation built with MkDocs
- Extensive CI/CD pipeline with testing across multiple Python versions and OS
- Integration testing with popular libraries (FastAPI, SQLModel, etc.)
- Performance benchmarking and dependency compatibility checks

Pydantic ensures runtime data integrity through type hints and is foundational to many Python frameworks, especially in web APIs and data processing applications.\
"""
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=5262,
                    output_tokens=369,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 5262,
                        'output_tokens': 369,
                    },
                ),
                model_name='claude-sonnet-4-20250514',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_01DSGib8F7nNoYprfYSGp1sd',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_anthropic_mcp_servers_stream(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 3000})
    agent = Agent(
        m,
        builtin_tools=[
            MCPServerTool(
                id='deepwiki',
                url='https://mcp.deepwiki.com/mcp',
                allowed_tools=['ask_question'],
            )
        ],
        model_settings=settings,
    )

    event_parts: list[Any] = []
    async with agent.iter(
        user_prompt='Can you tell me more about the pydantic/pydantic-ai repo? Keep your answer short'
    ) as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        if (
                            isinstance(event, PartStartEvent)
                            and isinstance(event.part, BuiltinToolCallPart | BuiltinToolReturnPart)
                        ) or (isinstance(event, PartDeltaEvent) and isinstance(event.delta, ToolCallPartDelta)):
                            event_parts.append(event)

    assert agent_run.result is not None
    messages = agent_run.result.all_messages()
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Can you tell me more about the pydantic/pydantic-ai repo? Keep your answer short',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='The user is asking about the pydantic/pydantic-ai repository. They want a short answer about the repo. I should use the deepwiki_ask_question function to get information about this repository.',
                        signature='EuoCCkYICBgCKkDPqznnPHupi9rVXvaQQqrMprXof9wtQsCqw7Yw687UIk/FvF65omU22QO+CmIcYqTwhBfifPEp9A3/lM9C8cIcEgzGsjorcyNe2H0ZFf8aDCA4iLG6qgUL6fLhzCIwVWcg65CrvSFusXtMH18p+XiF+BUxT+rvnCFsnLbFsxtjGyKh1j4UW6V0Tk0O7+3sKtEBEzvxztXkMkeXkXRsQFJ00jTNhkUHu74sqnh6QxgV8wK2vlJRnBnes/oh7QdED0h/pZaUbxplYJiPFisWx/zTJQvOv29I46sM2CdY5ggGO1KWrEF/pognyod+jdCdb481XUET9T7nl/VMz/Og2QkyGf+5MvSecKQhujlS0VFhCgaYv68sl0Fv3hj2AkeE4vcYu3YdDaNDLXerbIaLCMkkn08NID/wKZTwtLSL+N6+kOi+4peGqXDNps8oa3mqIn7NAWFlwEUrFZd5kjtDkQ5dw/IYAQ==',
                        provider_name='anthropic',
                    ),
                    BuiltinToolCallPart(
                        tool_name='mcp_server:deepwiki',
                        args='{"action":"call_tool","tool_name":"ask_question","tool_args":{"repoName": "pydantic/pydantic-ai", "question": "What is this repository about? What are its main features and purpose?"}}',
                        tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1',
                        provider_name='anthropic',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_server:deepwiki',
                        content={
                            'content': [
                                {
                                    'citations': None,
                                    'text': IsStr(),
                                    'type': 'text',
                                }
                            ],
                            'is_error': False,
                        },
                        tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1',
                        timestamp=IsDatetime(),
                        provider_name='anthropic',
                    ),
                    TextPart(
                        content="""\
**Pydantic-AI** is a framework for building Generative AI applications with type safety. It provides:

- **Unified LLM interface** - Works with OpenAI, Anthropic, Google, Groq, Cohere, Mistral, AWS Bedrock, and more
- **Type-safe agents** - Uses Pydantic for validation and type checking throughout
- **Tool integration** - Easily add custom functions/tools agents can call
- **Graph-based execution** - Manages agent workflows as finite state machines
- **Multiple output formats** - Text, structured data, and multimodal content
- **Durable execution** - Integration with systems like DBOS and Temporal for fault tolerance
- **Streaming support** - Stream responses in real-time

It's designed to simplify building robust, production-ready AI agents while abstracting away provider-specific complexities.\
"""
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=3042,
                    output_tokens=354,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 3042,
                        'output_tokens': 354,
                    },
                ),
                model_name='claude-sonnet-4-5-20250929',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_01Xf6SmUVY1mDrSwFc5RsY3n',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=1,
                part=BuiltinToolCallPart(
                    tool_name='mcp_server:deepwiki',
                    tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1',
                    provider_name='anthropic',
                ),
                previous_part_kind='thinking',
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(
                    args_delta='{"action":"call_tool","tool_name":"ask_question","tool_args":',
                    tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1',
                ),
            ),
            PartDeltaEvent(
                index=1, delta=ToolCallPartDelta(args_delta='', tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1')
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='{"repoName"', tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1'),
            ),
            PartDeltaEvent(
                index=1, delta=ToolCallPartDelta(args_delta=': "', tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1')
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='pydantic', tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1'),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='/pydantic-ai', tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1'),
            ),
            PartDeltaEvent(
                index=1, delta=ToolCallPartDelta(args_delta='"', tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1')
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta=', "question', tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1'),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='": "What', tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1'),
            ),
            PartDeltaEvent(
                index=1, delta=ToolCallPartDelta(args_delta=' is ', tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1')
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='this repo', tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1'),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='sitory about', tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1'),
            ),
            PartDeltaEvent(
                index=1, delta=ToolCallPartDelta(args_delta='? Wha', tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1')
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='t are i', tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1'),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='ts main feat', tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1'),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='ure', tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1'),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='s and purpo', tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1'),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='se?"}', tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1'),
            ),
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='}', tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1'),
            ),
            PartStartEvent(
                index=2,
                part=BuiltinToolReturnPart(
                    tool_name='mcp_server:deepwiki',
                    content={
                        'content': [
                            {
                                'citations': None,
                                'text': """\
This repository, `pydantic/pydantic-ai`, is a GenAI Agent Framework that leverages Pydantic for building Generative AI applications. Its main purpose is to provide a unified and type-safe way to interact with various large language models (LLMs) from different providers, manage agent execution flows, and integrate with external tools and services. \n\

## Main Features and Purpose

The `pydantic-ai` repository offers several core features:

### 1. Agent System
The `Agent` class serves as the main orchestrator for managing interactions with LLMs and executing tasks.  Agents can be configured with generic types for dependency injection (`Agent[AgentDepsT, OutputDataT]`) and output validation, ensuring type safety throughout the application. \n\

Agents support various execution methods:
*   `agent.run()`: An asynchronous function that returns a completed `RunResult`. \n\
*   `agent.run_sync()`: A synchronous function that internally calls `run()` to return a completed `RunResult`. \n\
*   `agent.run_stream()`: An asynchronous context manager for streaming text and structured output. \n\
*   `agent.run_stream_events()`: Returns an asynchronous iterable of `AgentStreamEvent`s and a final `AgentRunResultEvent`. \n\
*   `agent.iter()`: A context manager that provides an asynchronous iterable over the nodes of the agent's underlying `Graph`, allowing for deeper control and insight into the execution flow. \n\

### 2. Model Integration
The framework provides a unified interface for integrating with various LLM providers, including OpenAI, Anthropic, Google, Groq, Cohere, Mistral, Bedrock, and HuggingFace.  Each model integration follows a consistent settings pattern with provider-specific prefixes (e.g., `google_*`, `anthropic_*`). \n\

Examples of supported models and their capabilities include:
*   `GoogleModel`: Integrates with Google's Gemini API, supporting both Gemini API (`google-gla`) and Vertex AI (`google-vertex`) providers.  It supports token counting, streaming, built-in tools like `WebSearchTool`, `WebFetchTool`, `CodeExecutionTool`, and native JSON schema output. \n\
*   `AnthropicModel`: Uses Anthropic's beta API for advanced features like "Thinking Blocks" and built-in tools. \n\
*   `GroqModel`: Offers high-speed inference and specialized reasoning support with configurable reasoning formats. \n\
*   `MistralModel`: Supports customizable JSON schema prompting and thinking support. \n\
*   `BedrockConverseModel`: Utilizes AWS Bedrock's Converse API for unified access to various foundation models like Claude, Titan, Llama, and Mistral. \n\
*   `CohereModel`: Integrates with Cohere's v2 API for chat completions, including thinking support and tool calling. \n\

The framework also supports multimodal inputs such as `AudioUrl`, `DocumentUrl`, `ImageUrl`, and `VideoUrl`, allowing agents to process and respond to diverse content types. \n\

### 3. Graph-based Execution
Pydantic AI uses `pydantic-graph` to manage the execution flow of agents, representing it as a finite state machine.  The execution typically flows through `UserPromptNode` → `ModelRequestNode` → `CallToolsNode`.  This allows for detailed tracking of message history and usage. \n\

### 4. Tool System
Function tools enable models to perform actions and retrieve additional information.  Tools can be registered using decorators like `@agent.tool` (for tools needing `RunContext` access) or `@agent.tool_plain` (for tools without `RunContext` access).  The framework also supports toolsets for managing collections of tools. \n\

Tools can return various types of output, including anything Pydantic can serialize to JSON, as well as multimodal content like `AudioUrl`, `VideoUrl`, `ImageUrl`, or `DocumentUrl`.  The `ToolReturn` object allows for separating the `return_value` (for the model), `content` (for additional context), and `metadata` (for application-specific use). \n\

Built-in tools like `WebFetchTool` allow agents to pull web content into their context. \n\

### 5. Output Handling
The framework supports various output types:
*   `TextOutput`: Plain text responses. \n\
*   `ToolOutput`: Structured data via tool calls. \n\
*   `NativeOutput`: Provider-specific structured output. \n\
*   `PromptedOutput`: Prompt-based structured extraction. \n\

### 6. Durable Execution
Pydantic AI integrates with durable execution systems like DBOS and Temporal.  This allows agents to maintain state and resume execution after failures or restarts, making them suitable for long-running or fault-tolerant applications. \n\

### 7. Multi-Agent Patterns and Integrations
The repository supports multi-agent applications and various integrations, including:
*   Pydantic Evals: For evaluating agent performance. \n\
*   Pydantic Graph: The underlying graph execution engine. \n\
*   Logfire: For debugging and monitoring. \n\
*   Agent-User Interaction (AG-UI) and Agent2Agent (A2A): For facilitating interactions between agents and users, and between agents themselves. \n\
*   Clai: A CLI tool. \n\

## Purpose

The overarching purpose of `pydantic-ai` is to simplify the development of robust and reliable Generative AI applications by providing a structured, type-safe, and extensible framework. It aims to abstract away the complexities of interacting with different LLM providers and managing agent workflows, allowing developers to focus on application logic. \n\

Notes:
The `CLAUDE.md` file provides guidance for Claude Code when working with the repository, outlining development commands and project architecture.  The `mkdocs.yml` file defines the structure and content of the project's documentation, further detailing the features and organization of the repository. \n\

Wiki pages you might want to explore:
- [Google, Anthropic and Other Providers (pydantic/pydantic-ai)](/wiki/pydantic/pydantic-ai#3.3)

View this search on DeepWiki: https://deepwiki.com/search/what-is-this-repository-about_5104a64d-2f5e-4461-80d8-eb0892242441
""",
                                'type': 'text',
                            }
                        ],
                        'is_error': False,
                    },
                    tool_call_id='mcptoolu_01FZmJ5UspaX5BB9uU339UT1',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                ),
                previous_part_kind='builtin-tool-call',
            ),
        ]
    )


async def test_anthropic_code_execution_tool(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(api_key=anthropic_api_key))
    settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 3000})
    agent = Agent(
        m,
        builtin_tools=[CodeExecutionTool()],
        model_settings=settings,
        instructions='Always use the code execution tool for math.',
    )

    result = await agent.run('How much is 3 * 12390?')
    messages = result.all_messages()
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='How much is 3 * 12390?', timestamp=IsDatetime())],
                timestamp=IsNow(tz=timezone.utc),
                instructions='Always use the code execution tool for math.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='The user is asking for a simple multiplication: 3 * 12390. This is a mathematical calculation, and according to my guidelines, I should always use the code execution tool for math. Even though this is a relatively simple calculation that could be done mentally, the instruction is clear that I should use the code execution tool for math.',
                        signature='EvsDCkYIBxgCKkCSFDXODoOrOHU14Yv7+TNxuR4sDsJKw9y9C1gGPIWqslF6apNZ1xwJ94E9KsQBfXlZ/ELoBSTj3YT0liwueN6kEgxrakXTN1a+YafcnckaDC2EYhQsezxdE/P7XSIwczAl/PquNGpiOLqC5DnYKvD2+F0JhBQsbLe1bQi/VR0XCQdd+4DZ5dBU5AmuDcntKuICIMg145F3vP8bFnTdUMOIQY0NASypKRnHj6owIkuqWJ+pwu6OdpDt2a+Lr7R1dw860hcPjEp65eg5nwtyi8bw1pzfQJmC48DoiQn/OYeiXMWeNv5HoKEK/lkikqVPcTnD03MytUsNGRqUBfDvr4bxNgxqeAENi5pZ21ySnjxhC879gN0G3uriEM8o4LXj/X2DotKO1lvIEL/2RQZGrFulDLq5I2FW51YBY3kzHerK7zwFgs3t39VLsy7Q3T6sLi4yh4BbFxF4RaSOCicTRbMYC8UO85uhArSSm/0EDDhX+kxIGJZ91F6Vv0vSS4qLy+55buZ8Jj4/P86t9YMxBeylQ/tUNGzhISqc1+CZeQ4aZKiRyQmlfkA6bcM42JAFQT/c0EbM2JmDsiSpkM8d021E9hqrr2eIhasaOo4vG5yUz7f9aSaRc/Muy02mckNxxxS7UshBCxr8veoMa0HYnB/rBNFeGAE=',
                        provider_name='anthropic',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={
                            'code': """\
result = 3 * 12390
print(f"3 * 12390 = {result}")\
"""
                        },
                        tool_call_id='srvtoolu_01Pc4vcD1JPUDcVhHaskFUfn',
                        provider_name='anthropic',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={
                            'content': [],
                            'return_code': 0,
                            'stderr': '',
                            'stdout': '3 * 12390 = 37170\n',
                            'type': 'code_execution_result',
                        },
                        tool_call_id='srvtoolu_01Pc4vcD1JPUDcVhHaskFUfn',
                        timestamp=IsDatetime(),
                        provider_name='anthropic',
                    ),
                    TextPart(content='3 * 12390 = 37170'),
                ],
                usage=RequestUsage(
                    input_tokens=1771,
                    output_tokens=171,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 1771,
                        'output_tokens': 171,
                    },
                ),
                model_name='claude-sonnet-4-20250514',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn', 'container_id': 'container_011CTCwceSoRxi8Pf16Fb7Tn'},
                provider_response_id='msg_018bVTPr9khzuds31rFDuqW4',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    result = await agent.run('How about 4 * 12390?')
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='How about 4 * 12390?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                instructions='Always use the code execution tool for math.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='The user is asking for a simple multiplication: 4 * 12390. This is a computational task that requires precise calculation, so I should use the code execution tool to get the accurate result.',
                        signature='EucCCkYIBxgCKkDrAwZF3dM/a2UiJFMD/+Z5mdZOkFXxJ1vmAg7GWzC2YUTBKtKvys1yFaWmkUuBSYBC/kaTPYVj28qa94V0Q/ngEgw+4333itH5QH/0B6gaDHxUZy/HGNpU04RbZiIwmQeS7P+gLHlV9b0tRYciwVbpjZl8WkrunyWyD5xXTC7bzv/tQKv8kMjxRsRGZZH1Ks4BDiNK1tuAlz4x5LDAsui8/8vBDY1c+NRtc6y0bOgxSXFXSemv2BHm7VokC7JG8+iCQEY9HIyFtyjLeJ93niDCszU8YHPtAa4o2Orw8K4Tc4Y18U/TqfgnZulkjkeONhDJP9uUk4Db4woJiLpAx13X8W5TriwqHWMRM2+D0coqTTWTovC/xbVFFZZmwyqaz/h6V6qqokyLpbqb+5B5kw/uQfybUv28h3GqxFyuD62zM9OPyMqbd2GrAPbSLE2JETkJsp6GzxVEh1vNI3DMgdQYAQ==',
                        provider_name='anthropic',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={
                            'code': """\
result = 4 * 12390
print(f"4 * 12390 = {result}")\
"""
                        },
                        tool_call_id='srvtoolu_017iCje5DPMZEdgBkxj1osgt',
                        provider_name='anthropic',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={
                            'content': [],
                            'return_code': 0,
                            'stderr': '',
                            'stdout': '4 * 12390 = 49560\n',
                            'type': 'code_execution_result',
                        },
                        tool_call_id='srvtoolu_017iCje5DPMZEdgBkxj1osgt',
                        timestamp=IsDatetime(),
                        provider_name='anthropic',
                    ),
                    TextPart(content='4 * 12390 = 49560'),
                ],
                usage=RequestUsage(
                    input_tokens=1741,
                    output_tokens=143,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 1741,
                        'output_tokens': 143,
                    },
                ),
                model_name='claude-sonnet-4-20250514',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn', 'container_id': 'container_011CTCwdXe48NC7LaX3rxQ4d'},
                provider_response_id='msg_01VngRFBcNddwrYQoKUmdePY',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_anthropic_code_execution_tool_stream(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(api_key=anthropic_api_key))
    settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 3000})
    agent = Agent(m, builtin_tools=[CodeExecutionTool()], model_settings=settings)

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='what is 65465-6544 * 65464-6+1.02255') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert agent_run.result is not None
    assert agent_run.result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='what is 65465-6544 * 65464-6+1.02255',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="""\
The user is asking me to calculate a mathematical expression: 65465-6544 * 65464-6+1.02255

This involves multiplication and subtraction operations, and I need to be careful about the order of operations (PEMDAS/BODMAS). Let me break this down:

65465-6544 * 65464-6+1.02255

Following order of operations:
1. First, multiplication: 6544 * 65464
2. Then left to right for addition and subtraction: 65465 - (result from step 1) - 6 + 1.02255

This is a computational task that requires precise calculations, so I should use the code_execution tool to get an accurate result.\
""",
                        signature='EucFCkYIBxgCKkCfcR3zTiKFcMLhP1aMZu4l0cfgiw3ukkSHOSX2qV1DEKtpe3pu1HpRvDz1mEw32e/wvHoS/AfpVYk3AFb8oAscEgxips//IwdGKRINkQoaDDc122APa5lQXEtsuiIw7RQW/ow7z+MOXL6D8pAl4Iz5V6VSbn2A37DxwRbzOYHSicZuvVrhZHLmn2WWwTZjKs4EYn4HNPF6+Y+9dITwGBWUz6WXsOnv/S1sp+WJLYD8vGMDG9DzTIdjQ9pMN/Bg6VB3hPTveXqxopBk+V7u1WaQC0NmkEmREv6Pdq9iHHEnuIhN0t7UrrNDxPwt/cmbilfa7QL8ofeeSorIRwvibXtG0aqNDu42r6JkatwttDSRIBSqIgKLkel8yPP9ksmOf4SRbNAbgijmq63s+EIkNHt2yjuTHV48pR1j1czHWcsoqJOHj6faeXge0OyGKuPqbBCzoqAjecNq0dRfHQUgXMWmeaJp1R6iWhKxyJV5Y2EwhA5WGH9xzc9h0TobIgGFGAk2OvzDPBO5qr+O85LbjNeHF3WfZciaj2lMIVsveklN9S8598m+R+D4/O8Sscebc2xoVf8qBDazJP5gVtuMoAKBcJuNVWeTR5snv2vs5BEejv6Q2gcb6rPa4ZxEmilhK1NTy9+dwoYvgLUm5o11PBXbI7uRv18tLwwer55Ult5Aq3JgG8Uj8FgBA4exLCw9LKUhzd+1lN0i19f2mDDuBORw5dPUBj2unzIb6sro/2SYm3MF2nmKhh5mm1F/v37ksOzJlTUPhbcs6aYrUJo5cM1H9AB8vpcNln38uWb4tuFgD5Wqy/0WFu60nsRsnInI5SPMN39wA4cx2eyrCfne32iw0Ov+VAdn0+D8FFzyVEEh7lrCQlJFoqoznxvpKh6NRhUzLmLpfEPOhFN/bZBHsj+3YJLT4JgRaYGTf6fMkZGCyIk60hIbqofwcuMFNqFYOK0nffOV8dz9ElisN/6cSJsYAQ==',
                        provider_name='anthropic',
                    ),
                    TextPart(
                        content="I'll calculate this mathematical expression for you. Let me break it down step by step following the order of operations."
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args='{"code": "# Calculate the expression: 65465-6544 * 65464-6+1.02255\\n# Following order of operations (PEMDAS/BODMAS)\\n\\nexpression = \\"65465-6544 * 65464-6+1.02255\\"\\nprint(f\\"Expression: {expression}\\")\\n\\n# Let\'s break it down step by step\\nstep1 = 6544 * 65464  # Multiplication first\\nprint(f\\"Step 1 - Multiplication: 6544 * 65464 = {step1}\\")\\n\\nstep2 = 65465 - step1  # First subtraction\\nprint(f\\"Step 2 - First subtraction: 65465 - {step1} = {step2}\\")\\n\\nstep3 = step2 - 6  # Second subtraction\\nprint(f\\"Step 3 - Second subtraction: {step2} - 6 = {step3}\\")\\n\\nfinal_result = step3 + 1.02255  # Final addition\\nprint(f\\"Step 4 - Final addition: {step3} + 1.02255 = {final_result}\\")\\n\\n# Let\'s also verify with direct calculation\\ndirect_result = 65465-6544 * 65464-6+1.02255\\nprint(f\\"\\\\nDirect calculation: {direct_result}\\")\\nprint(f\\"Results match: {final_result == direct_result}\\")"}',
                        tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG',
                        provider_name='anthropic',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={
                            'content': [],
                            'return_code': 0,
                            'stderr': '',
                            'stdout': """\
Expression: 65465-6544 * 65464-6+1.02255
Step 1 - Multiplication: 6544 * 65464 = 428396416
Step 2 - First subtraction: 65465 - 428396416 = -428330951
Step 3 - Second subtraction: -428330951 - 6 = -428330957
Step 4 - Final addition: -428330957 + 1.02255 = -428330955.97745

Direct calculation: -428330955.97745
Results match: True
""",
                            'type': 'code_execution_result',
                        },
                        tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG',
                        timestamp=IsDatetime(),
                        provider_name='anthropic',
                    ),
                    TextPart(
                        content="""\
The answer to **65465-6544 * 65464-6+1.02255** is **-428,330,955.97745**.

Here's how it breaks down following the order of operations:
1. First, multiplication: 6,544 × 65,464 = 428,396,416
2. Then left to right: 65,465 - 428,396,416 = -428,330,951
3. Continue: -428,330,951 - 6 = -428,330,957
4. Finally: -428,330,957 + 1.02255 = -428,330,955.97745\
"""
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=2316,
                    output_tokens=733,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 2316,
                        'output_tokens': 733,
                    },
                ),
                model_name='claude-sonnet-4-20250514',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_01TaPV5KLA8MsCPDuJNKPLF4',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    assert event_parts == snapshot(
        [
            PartStartEvent(index=0, part=ThinkingPart(content='', signature='', provider_name='anthropic')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta='The user is asking me to calculate', provider_name='anthropic'),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta=' a mathematical expression: 65465-6544 *', provider_name='anthropic'
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
 65464-6+1.02255

This\
""",
                    provider_name='anthropic',
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta=' involves multiplication and subtraction operations, and I need to be careful about the order of',
                    provider_name='anthropic',
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta=' operations (PEMDAS/BODMAS).', provider_name='anthropic'),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
 Let me break this down:

65\
""",
                    provider_name='anthropic',
                ),
            ),
            PartDeltaEvent(
                index=0, delta=ThinkingPartDelta(content_delta='465-6544 * 65464-6+1.02255', provider_name='anthropic')
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\


Following order of operations:
1. First, multiplication:\
""",
                    provider_name='anthropic',
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' 6544 * 65464', provider_name='anthropic')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\

2. Then left to right for\
""",
                    provider_name='anthropic',
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta=' addition and subtraction: 65465', provider_name='anthropic'),
            ),
            PartDeltaEvent(
                index=0, delta=ThinkingPartDelta(content_delta=' - (result from step 1)', provider_name='anthropic')
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
 - 6 + 1.02255

This\
""",
                    provider_name='anthropic',
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta=' is a computational task that requires precise', provider_name='anthropic'
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta=' calculations, so I should use the code_execution', provider_name='anthropic'
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(content_delta=' tool to get an accurate result.', provider_name='anthropic'),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    signature_delta='EucFCkYIBxgCKkCfcR3zTiKFcMLhP1aMZu4l0cfgiw3ukkSHOSX2qV1DEKtpe3pu1HpRvDz1mEw32e/wvHoS/AfpVYk3AFb8oAscEgxips//IwdGKRINkQoaDDc122APa5lQXEtsuiIw7RQW/ow7z+MOXL6D8pAl4Iz5V6VSbn2A37DxwRbzOYHSicZuvVrhZHLmn2WWwTZjKs4EYn4HNPF6+Y+9dITwGBWUz6WXsOnv/S1sp+WJLYD8vGMDG9DzTIdjQ9pMN/Bg6VB3hPTveXqxopBk+V7u1WaQC0NmkEmREv6Pdq9iHHEnuIhN0t7UrrNDxPwt/cmbilfa7QL8ofeeSorIRwvibXtG0aqNDu42r6JkatwttDSRIBSqIgKLkel8yPP9ksmOf4SRbNAbgijmq63s+EIkNHt2yjuTHV48pR1j1czHWcsoqJOHj6faeXge0OyGKuPqbBCzoqAjecNq0dRfHQUgXMWmeaJp1R6iWhKxyJV5Y2EwhA5WGH9xzc9h0TobIgGFGAk2OvzDPBO5qr+O85LbjNeHF3WfZciaj2lMIVsveklN9S8598m+R+D4/O8Sscebc2xoVf8qBDazJP5gVtuMoAKBcJuNVWeTR5snv2vs5BEejv6Q2gcb6rPa4ZxEmilhK1NTy9+dwoYvgLUm5o11PBXbI7uRv18tLwwer55Ult5Aq3JgG8Uj8FgBA4exLCw9LKUhzd+1lN0i19f2mDDuBORw5dPUBj2unzIb6sro/2SYm3MF2nmKhh5mm1F/v37ksOzJlTUPhbcs6aYrUJo5cM1H9AB8vpcNln38uWb4tuFgD5Wqy/0WFu60nsRsnInI5SPMN39wA4cx2eyrCfne32iw0Ov+VAdn0+D8FFzyVEEh7lrCQlJFoqoznxvpKh6NRhUzLmLpfEPOhFN/bZBHsj+3YJLT4JgRaYGTf6fMkZGCyIk60hIbqofwcuMFNqFYOK0nffOV8dz9ElisN/6cSJsYAQ==',
                    provider_name='anthropic',
                ),
            ),
            PartEndEvent(
                index=0,
                part=ThinkingPart(
                    content="""\
The user is asking me to calculate a mathematical expression: 65465-6544 * 65464-6+1.02255

This involves multiplication and subtraction operations, and I need to be careful about the order of operations (PEMDAS/BODMAS). Let me break this down:

65465-6544 * 65464-6+1.02255

Following order of operations:
1. First, multiplication: 6544 * 65464
2. Then left to right for addition and subtraction: 65465 - (result from step 1) - 6 + 1.02255

This is a computational task that requires precise calculations, so I should use the code_execution tool to get an accurate result.\
""",
                    signature='EucFCkYIBxgCKkCfcR3zTiKFcMLhP1aMZu4l0cfgiw3ukkSHOSX2qV1DEKtpe3pu1HpRvDz1mEw32e/wvHoS/AfpVYk3AFb8oAscEgxips//IwdGKRINkQoaDDc122APa5lQXEtsuiIw7RQW/ow7z+MOXL6D8pAl4Iz5V6VSbn2A37DxwRbzOYHSicZuvVrhZHLmn2WWwTZjKs4EYn4HNPF6+Y+9dITwGBWUz6WXsOnv/S1sp+WJLYD8vGMDG9DzTIdjQ9pMN/Bg6VB3hPTveXqxopBk+V7u1WaQC0NmkEmREv6Pdq9iHHEnuIhN0t7UrrNDxPwt/cmbilfa7QL8ofeeSorIRwvibXtG0aqNDu42r6JkatwttDSRIBSqIgKLkel8yPP9ksmOf4SRbNAbgijmq63s+EIkNHt2yjuTHV48pR1j1czHWcsoqJOHj6faeXge0OyGKuPqbBCzoqAjecNq0dRfHQUgXMWmeaJp1R6iWhKxyJV5Y2EwhA5WGH9xzc9h0TobIgGFGAk2OvzDPBO5qr+O85LbjNeHF3WfZciaj2lMIVsveklN9S8598m+R+D4/O8Sscebc2xoVf8qBDazJP5gVtuMoAKBcJuNVWeTR5snv2vs5BEejv6Q2gcb6rPa4ZxEmilhK1NTy9+dwoYvgLUm5o11PBXbI7uRv18tLwwer55Ult5Aq3JgG8Uj8FgBA4exLCw9LKUhzd+1lN0i19f2mDDuBORw5dPUBj2unzIb6sro/2SYm3MF2nmKhh5mm1F/v37ksOzJlTUPhbcs6aYrUJo5cM1H9AB8vpcNln38uWb4tuFgD5Wqy/0WFu60nsRsnInI5SPMN39wA4cx2eyrCfne32iw0Ov+VAdn0+D8FFzyVEEh7lrCQlJFoqoznxvpKh6NRhUzLmLpfEPOhFN/bZBHsj+3YJLT4JgRaYGTf6fMkZGCyIk60hIbqofwcuMFNqFYOK0nffOV8dz9ElisN/6cSJsYAQ==',
                    provider_name='anthropic',
                ),
                next_part_kind='text',
            ),
            PartStartEvent(
                index=1,
                part=TextPart(content="I'll calculate this mathematical expression for you. Let me break"),
                previous_part_kind='thinking',
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(
                index=1, delta=TextPartDelta(content_delta=' it down step by step following the order of operations.')
            ),
            PartEndEvent(
                index=1,
                part=TextPart(
                    content="I'll calculate this mathematical expression for you. Let me break it down step by step following the order of operations."
                ),
                next_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=2,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG',
                    provider_name='anthropic',
                ),
                previous_part_kind='text',
            ),
            PartDeltaEvent(
                index=2, delta=ToolCallPartDelta(args_delta='', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG')
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta='{"code": "# Calculate the expression: 65465-6544 * 65464-6+1.02255\\n# Following order of operations (PEMDAS/BODMAS',
                    tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG',
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta=')\\n\\nexpression = \\"65465-6544 ', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(args_delta='* 65464-6+1', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta='.02255\\"\\nprint(f\\"Expression: {expression',
                    tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG',
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta='}\\")\\n\\n# Let\'s break it down', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta=' step by step\\nstep1 = ', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(args_delta='6544 * 65464  ', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta='# Multiplication first\\nprint', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(args_delta='(f\\"Step 1 ', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta='- Multiplication: ', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(args_delta='6544 * 65464 ', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta='= {step1}\\")\\n\\nstep2', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta=' = 65465 - step1  ', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta='# First subtraction\\nprint(f\\"Step', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta=' 2 - First subtraction:', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(args_delta=' 65465 - {step1', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta='} = {step2}\\")\\n\\nstep', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta='3 = step2 - 6  # Second subtraction\\nprint',
                    tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG',
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta='(f\\"Step 3 - Second subtraction: {step2}',
                    tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG',
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(args_delta=' - 6 = {step3', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta='}\\")\\n\\nfinal_result = step3 + ', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta='1.02255  # Final addition\\nprint(f\\"Step ',
                    tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG',
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta='4 - Final addition: {step3', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(args_delta='} + 1.02255 ', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta='= {final_result}\\")\\n\\n#', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta=" Let's also verify with", tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta=' direct calculation\\ndirect_result = 65',
                    tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG',
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta='465-6544 * 65464-', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta='6+1.02255\\nprint(f\\"\\\\nDirect calculation:',
                    tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG',
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta=' {direct_result}\\")\\nprint', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(
                    args_delta='(f\\"Results match: {final_result == direct',
                    tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG',
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=ToolCallPartDelta(args_delta='_result}\\")', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG'),
            ),
            PartDeltaEvent(
                index=2, delta=ToolCallPartDelta(args_delta='"}', tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG')
            ),
            PartEndEvent(
                index=2,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args='{"code": "# Calculate the expression: 65465-6544 * 65464-6+1.02255\\n# Following order of operations (PEMDAS/BODMAS)\\n\\nexpression = \\"65465-6544 * 65464-6+1.02255\\"\\nprint(f\\"Expression: {expression}\\")\\n\\n# Let\'s break it down step by step\\nstep1 = 6544 * 65464  # Multiplication first\\nprint(f\\"Step 1 - Multiplication: 6544 * 65464 = {step1}\\")\\n\\nstep2 = 65465 - step1  # First subtraction\\nprint(f\\"Step 2 - First subtraction: 65465 - {step1} = {step2}\\")\\n\\nstep3 = step2 - 6  # Second subtraction\\nprint(f\\"Step 3 - Second subtraction: {step2} - 6 = {step3}\\")\\n\\nfinal_result = step3 + 1.02255  # Final addition\\nprint(f\\"Step 4 - Final addition: {step3} + 1.02255 = {final_result}\\")\\n\\n# Let\'s also verify with direct calculation\\ndirect_result = 65465-6544 * 65464-6+1.02255\\nprint(f\\"\\\\nDirect calculation: {direct_result}\\")\\nprint(f\\"Results match: {final_result == direct_result}\\")"}',
                    tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG',
                    provider_name='anthropic',
                ),
                next_part_kind='builtin-tool-return',
            ),
            PartStartEvent(
                index=3,
                part=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={
                        'content': [],
                        'return_code': 0,
                        'stderr': '',
                        'stdout': """\
Expression: 65465-6544 * 65464-6+1.02255
Step 1 - Multiplication: 6544 * 65464 = 428396416
Step 2 - First subtraction: 65465 - 428396416 = -428330951
Step 3 - Second subtraction: -428330951 - 6 = -428330957
Step 4 - Final addition: -428330957 + 1.02255 = -428330955.97745

Direct calculation: -428330955.97745
Results match: True
""",
                        'type': 'code_execution_result',
                    },
                    tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartStartEvent(index=4, part=TextPart(content='The answer to'), previous_part_kind='builtin-tool-return'),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' **65465-6544 * ')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='65464-6+1.02255** is **')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='-428,330,955.97745**.')),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(
                    content_delta="""\


Here's how it breaks down following the order of operations:
1. First\
"""
                ),
            ),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=', multiplication: 6,544 × 65,464 ')),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(
                    content_delta="""\
= 428,396,416
2. Then left\
"""
                ),
            ),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' to right: 65,465 - 428')),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(
                    content_delta="""\
,396,416 = -428,330,951
3\
"""
                ),
            ),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='. Continue: -428,330,951 -')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' 6 = -428,330')),
            PartDeltaEvent(
                index=4,
                delta=TextPartDelta(
                    content_delta="""\
,957
4. Finally: -428,330,957 + \
"""
                ),
            ),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='1.02255 = -428,330,955.97745')),
            PartEndEvent(
                index=4,
                part=TextPart(
                    content="""\
The answer to **65465-6544 * 65464-6+1.02255** is **-428,330,955.97745**.

Here's how it breaks down following the order of operations:
1. First, multiplication: 6,544 × 65,464 = 428,396,416
2. Then left to right: 65,465 - 428,396,416 = -428,330,951
3. Continue: -428,330,951 - 6 = -428,330,957
4. Finally: -428,330,957 + 1.02255 = -428,330,955.97745\
"""
                ),
            ),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args='{"code": "# Calculate the expression: 65465-6544 * 65464-6+1.02255\\n# Following order of operations (PEMDAS/BODMAS)\\n\\nexpression = \\"65465-6544 * 65464-6+1.02255\\"\\nprint(f\\"Expression: {expression}\\")\\n\\n# Let\'s break it down step by step\\nstep1 = 6544 * 65464  # Multiplication first\\nprint(f\\"Step 1 - Multiplication: 6544 * 65464 = {step1}\\")\\n\\nstep2 = 65465 - step1  # First subtraction\\nprint(f\\"Step 2 - First subtraction: 65465 - {step1} = {step2}\\")\\n\\nstep3 = step2 - 6  # Second subtraction\\nprint(f\\"Step 3 - Second subtraction: {step2} - 6 = {step3}\\")\\n\\nfinal_result = step3 + 1.02255  # Final addition\\nprint(f\\"Step 4 - Final addition: {step3} + 1.02255 = {final_result}\\")\\n\\n# Let\'s also verify with direct calculation\\ndirect_result = 65465-6544 * 65464-6+1.02255\\nprint(f\\"\\\\nDirect calculation: {direct_result}\\")\\nprint(f\\"Results match: {final_result == direct_result}\\")"}',
                    tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG',
                    provider_name='anthropic',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={
                        'content': [],
                        'return_code': 0,
                        'stderr': '',
                        'stdout': """\
Expression: 65465-6544 * 65464-6+1.02255
Step 1 - Multiplication: 6544 * 65464 = 428396416
Step 2 - First subtraction: 65465 - 428396416 = -428330951
Step 3 - Second subtraction: -428330951 - 6 = -428330957
Step 4 - Final addition: -428330957 + 1.02255 = -428330955.97745

Direct calculation: -428330955.97745
Results match: True
""",
                        'type': 'code_execution_result',
                    },
                    tool_call_id='srvtoolu_01MKwyo39KHRDr9Ubff5vWtG',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                )
            ),
        ]
    )


async def test_anthropic_server_tool_pass_history_to_another_provider(
    allow_model_requests: None, anthropic_api_key: str, openai_api_key: str
):
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

    openai_model = OpenAIResponsesModel('gpt-4.1', provider=OpenAIProvider(api_key=openai_api_key))
    anthropic_model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(anthropic_model, builtin_tools=[WebSearchTool()])

    result = await agent.run('What day is today?')
    assert result.output == snapshot('Today is November 19, 2025.')
    result = await agent.run('What day is tomorrow?', model=openai_model, message_history=result.all_messages())
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What day is tomorrow?', timestamp=IsDatetime())],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='Tomorrow is November 20, 2025.',
                        id='msg_0dcd74f01910b54500691e5596124081a087e8fa7b2ca19d5a',
                    )
                ],
                usage=RequestUsage(input_tokens=329, output_tokens=12, details={'reasoning_tokens': 0}),
                model_name='gpt-4.1-2025-04-14',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 11, 19, 23, 41, 8, tzinfo=timezone.utc),
                },
                provider_response_id='resp_0dcd74f01910b54500691e5594957481a0ac36dde76eca939f',
                finish_reason='stop',
                run_id=IsStr(),
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
    model = AnthropicModel('claude-sonnet-4-5', provider='anthropic')

    # Test _map_message with empty string in user prompt
    messages_empty_string: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='')], kind='request', timestamp=IsDatetime()),
    ]
    _, anthropic_messages = await model._map_message(messages_empty_string, ModelRequestParameters(), {})  # type: ignore[attr-defined]
    assert anthropic_messages == snapshot([])  # Empty content should be filtered out

    # Test _map_message with list containing empty strings in user prompt
    messages_mixed_content: list[ModelMessage] = [
        ModelRequest(
            parts=[UserPromptPart(content=['', 'Hello', '', 'World'])], kind='request', timestamp=IsDatetime()
        ),
    ]
    _, anthropic_messages = await model._map_message(messages_mixed_content, ModelRequestParameters(), {})  # type: ignore[attr-defined]
    assert anthropic_messages == snapshot(
        [{'role': 'user', 'content': [{'text': 'Hello', 'type': 'text'}, {'text': 'World', 'type': 'text'}]}]
    )

    # Test _map_message with empty assistant response
    messages: list[ModelMessage] = [
        ModelRequest(parts=[SystemPromptPart(content='You are helpful')], kind='request', timestamp=IsDatetime()),
        ModelResponse(parts=[TextPart(content='')], kind='response'),  # Empty response
        ModelRequest(parts=[UserPromptPart(content='Hello')], kind='request', timestamp=IsDatetime()),
    ]
    _, anthropic_messages = await model._map_message(messages, ModelRequestParameters(), {})  # type: ignore[attr-defined]
    # The empty assistant message should be filtered out
    assert anthropic_messages == snapshot([{'role': 'user', 'content': [{'text': 'Hello', 'type': 'text'}]}])

    # Test with only empty assistant parts
    messages_resp: list[ModelMessage] = [
        ModelResponse(parts=[TextPart(content=''), TextPart(content='')], kind='response'),
    ]
    _, anthropic_messages = await model._map_message(messages_resp, ModelRequestParameters(), {})  # type: ignore[attr-defined]
    assert len(anthropic_messages) == 0  # No messages should be added


async def test_anthropic_tool_output(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))

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
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='get_user_country', args={}, tool_call_id='toolu_01X9wcHKKAZD9tBC711xipPa')
                ],
                usage=RequestUsage(
                    input_tokens=445,
                    output_tokens=23,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 445,
                        'output_tokens': 23,
                    },
                ),
                model_name='claude-sonnet-4-5-20250929',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_012TXW181edhmR5JCsQRsBKx',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id='toolu_01X9wcHKKAZD9tBC711xipPa',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args={'city': 'Mexico City', 'country': 'Mexico'},
                        tool_call_id='toolu_01LZABsgreMefH2Go8D5PQbW',
                    )
                ],
                usage=RequestUsage(
                    input_tokens=497,
                    output_tokens=56,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 497,
                        'output_tokens': 56,
                    },
                ),
                model_name='claude-sonnet-4-5-20250929',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_01K4Fzcf1bhiyLzHpwLdrefj',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='toolu_01LZABsgreMefH2Go8D5PQbW',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


async def test_anthropic_text_output_function(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))

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
        'BASED ON THE RESULT, YOU ARE LOCATED IN MEXICO. THE LARGEST CITY IN MEXICO IS MEXICO CITY (CIUDAD DE MÉXICO), WHICH IS BOTH THE CAPITAL AND THE MOST POPULOUS CITY IN THE COUNTRY. WITH A POPULATION OF APPROXIMATELY 9.2 MILLION PEOPLE IN THE CITY PROPER AND OVER 21 MILLION PEOPLE IN ITS METROPOLITAN AREA, MEXICO CITY IS NOT ONLY THE LARGEST CITY IN MEXICO BUT ALSO ONE OF THE LARGEST CITIES IN THE WORLD.'
    )

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in the user country? Use the get_user_country tool and then your own world knowledge.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="I'll help find the largest city in your country. Let me first check your country using the get_user_country tool."
                    ),
                    ToolCallPart(tool_name='get_user_country', args={}, tool_call_id='toolu_01JJ8TequDsrEU2pv1QFRWAK'),
                ],
                usage=RequestUsage(
                    input_tokens=383,
                    output_tokens=65,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 383,
                        'output_tokens': 65,
                    },
                ),
                model_name='claude-sonnet-4-5-20250929',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_01MsqUB7ZyhjGkvepS1tCXp3',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id='toolu_01JJ8TequDsrEU2pv1QFRWAK',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='Based on the result, you are located in Mexico. The largest city in Mexico is Mexico City (Ciudad de México), which is both the capital and the most populous city in the country. With a population of approximately 9.2 million people in the city proper and over 21 million people in its metropolitan area, Mexico City is not only the largest city in Mexico but also one of the largest cities in the world.'
                    )
                ],
                usage=RequestUsage(
                    input_tokens=460,
                    output_tokens=91,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 460,
                        'output_tokens': 91,
                    },
                ),
                model_name='claude-sonnet-4-5-20250929',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_0142umg4diSckrDtV9vAmmPL',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


@pytest.mark.vcr()
async def test_anthropic_prompted_output(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))

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
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='get_user_country', args={}, tool_call_id='toolu_01ArHq5f2wxRpRF2PVQcKExM')
                ],
                usage=RequestUsage(
                    input_tokens=459,
                    output_tokens=38,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 459,
                        'output_tokens': 38,
                    },
                ),
                model_name='claude-sonnet-4-5-20250929',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_018YiNXULHGpoKoHkTt6GivG',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id='toolu_01ArHq5f2wxRpRF2PVQcKExM',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"city": "Mexico City", "country": "Mexico"}')],
                usage=RequestUsage(
                    input_tokens=510,
                    output_tokens=17,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 510,
                        'output_tokens': 17,
                    },
                ),
                model_name='claude-sonnet-4-5-20250929',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_01WiRVmLhCrJbJZRqmAWKv3X',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_anthropic_prompted_output_multiple(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))

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
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='{"result": {"kind": "CityLocation", "data": {"city": "Mexico City", "country": "Mexico"}}}'
                    )
                ],
                usage=RequestUsage(
                    input_tokens=265,
                    output_tokens=31,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 265,
                        'output_tokens': 31,
                    },
                ),
                model_name='claude-sonnet-4-5-20250929',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_01N2PwwVQo2aBtt6UFhMDtEX',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_anthropic_output_tool_with_thinking(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel(
        'claude-sonnet-4-0',
        provider=AnthropicProvider(api_key=anthropic_api_key),
        settings=AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 3000}),
    )

    agent = Agent(m, output_type=ToolOutput(int))

    with pytest.raises(
        UserError,
        match=re.escape(
            'Anthropic does not support thinking and output tools at the same time. Use `output_type=PromptedOutput(...)` instead.'
        ),
    ):
        await agent.run('What is 3 + 3?')

    # Will default to prompted output
    agent = Agent(m, output_type=int)

    result = await agent.run('What is 3 + 3?')
    assert result.output == snapshot(6)


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

Mexico City is not only the largest city in Mexico but also one of the largest metropolitan areas in the world. The city proper has a population of approximately 9.2 million people, while the greater Mexico City metropolitan area has over 21 million inhabitants, making it the most populous metropolitan area in North America.

Mexico City serves as the country's capital and is the political, economic, and cultural center of Mexico.\
""")


async def test_anthropic_web_search_tool_pass_history_back(env: TestEnv, allow_model_requests: None):
    """Test passing web search tool history back to Anthropic."""
    # Create the first mock response with server tool blocks
    content: list[BetaContentBlock] = []
    content.append(BetaTextBlock(text='Let me search for the current date.', type='text'))
    content.append(
        BetaServerToolUseBlock(
            id='server_tool_123',
            name='web_search',
            input={'query': 'current date today'},
            type='server_tool_use',
            caller=BetaDirectCaller(type='direct'),
        )
    )
    content.append(
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
    )
    content.append(BetaTextBlock(text='Today is January 2, 2025.', type='text'))
    first_response = completion_message(
        content,
        BetaUsage(input_tokens=10, output_tokens=20),
    )

    # Create the second mock response that references the history
    second_response = completion_message(
        [BetaTextBlock(text='The web search result showed that today is January 2, 2025.', type='text')],
        BetaUsage(input_tokens=50, output_tokens=30),
    )

    mock_client = MockAnthropic.create_mock([first_response, second_response])
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m, builtin_tools=[WebSearchTool()])

    # First run to get server tool history
    result = await agent.run('What day is today?')

    # Verify we have server tool parts in the history
    server_tool_calls = [p for m in result.all_messages() for p in m.parts if isinstance(p, BuiltinToolCallPart)]
    server_tool_returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, BuiltinToolReturnPart)]
    assert len(server_tool_calls) == 1
    assert len(server_tool_returns) == 1
    assert server_tool_calls[0].tool_name == 'web_search'
    assert server_tool_returns[0].tool_name == 'web_search'

    # Pass the history back to another Anthropic agent run
    agent2 = Agent(m)
    result2 = await agent2.run('What was the web search result?', message_history=result.all_messages())
    assert result2.output == 'The web search result showed that today is January 2, 2025.'


async def test_anthropic_code_execution_tool_pass_history_back(env: TestEnv, allow_model_requests: None):
    """Test passing code execution tool history back to Anthropic."""
    # Create the first mock response with server tool blocks
    content: list[BetaContentBlock] = []
    content.append(BetaTextBlock(text='Let me calculate 2 + 2.', type='text'))
    content.append(
        BetaServerToolUseBlock(
            id='server_tool_456',
            name='code_execution',
            input={'code': 'print(2 + 2)'},
            type='server_tool_use',
            caller=BetaDirectCaller(type='direct'),
        )
    )
    content.append(
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
    )
    content.append(BetaTextBlock(text='The result is 4.', type='text'))
    first_response = completion_message(
        content,
        BetaUsage(input_tokens=10, output_tokens=20),
    )

    # Create the second mock response that references the history
    second_response = completion_message(
        [BetaTextBlock(text='The code execution returned the result: 4', type='text')],
        BetaUsage(input_tokens=50, output_tokens=30),
    )

    mock_client = MockAnthropic.create_mock([first_response, second_response])
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m, builtin_tools=[CodeExecutionTool()])

    # First run to get server tool history
    result = await agent.run('What is 2 + 2?')

    # Verify we have server tool parts in the history
    server_tool_calls = [p for m in result.all_messages() for p in m.parts if isinstance(p, BuiltinToolCallPart)]
    server_tool_returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, BuiltinToolReturnPart)]
    assert len(server_tool_calls) == 1
    assert len(server_tool_returns) == 1
    assert server_tool_calls[0].tool_name == 'code_execution'
    assert server_tool_returns[0].tool_name == 'code_execution'

    # Pass the history back to another Anthropic agent run
    agent2 = Agent(m)
    result2 = await agent2.run('What was the code execution result?', message_history=result.all_messages())
    assert result2.output == 'The code execution returned the result: 4'


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

    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0,
                part=BuiltinToolCallPart(
                    tool_name='web_search', tool_call_id='srvtoolu_01NcU4XNwyxWK6a9tcJZ8wGY', provider_name='anthropic'
                ),
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='', tool_call_id='srvtoolu_01NcU4XNwyxWK6a9tcJZ8wGY')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='{"q', tool_call_id='srvtoolu_01NcU4XNwyxWK6a9tcJZ8wGY')
            ),
            PartDeltaEvent(
                index=0,
                delta=ToolCallPartDelta(args_delta='uery": "top', tool_call_id='srvtoolu_01NcU4XNwyxWK6a9tcJZ8wGY'),
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' w', tool_call_id='srvtoolu_01NcU4XNwyxWK6a9tcJZ8wGY')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='orld n', tool_call_id='srvtoolu_01NcU4XNwyxWK6a9tcJZ8wGY')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='ew', tool_call_id='srvtoolu_01NcU4XNwyxWK6a9tcJZ8wGY')
            ),
            PartDeltaEvent(
                index=0,
                delta=ToolCallPartDelta(args_delta='s today"}', tool_call_id='srvtoolu_01NcU4XNwyxWK6a9tcJZ8wGY'),
            ),
            PartEndEvent(
                index=0,
                part=BuiltinToolCallPart(
                    tool_name='web_search',
                    args='{"query": "top world news today"}',
                    tool_call_id='srvtoolu_01NcU4XNwyxWK6a9tcJZ8wGY',
                    provider_name='anthropic',
                ),
                next_part_kind='builtin-tool-return',
            ),
            PartStartEvent(
                index=1,
                part=BuiltinToolReturnPart(
                    tool_name='web_search',
                    content=[
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '4 hours ago',
                            'title': 'World news - breaking news, video, headlines and opinion | CNN',
                            'type': 'web_search_result',
                            'url': 'https://www.cnn.com/world',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '1 hour ago',
                            'title': 'Breaking News, World News and Video from Al Jazeera',
                            'type': 'web_search_result',
                            'url': 'https://www.aljazeera.com/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '1 hour ago',
                            'title': 'News: U.S. and World News Headlines : NPR',
                            'type': 'web_search_result',
                            'url': 'https://www.npr.org/sections/news/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '7 hours ago',
                            'title': 'NBC News - Breaking News & Top Stories - Latest World, US & Local News | NBC News',
                            'type': 'web_search_result',
                            'url': 'https://www.nbcnews.com/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '3 hours ago',
                            'title': 'Breaking News, Latest News and Videos | CNN',
                            'type': 'web_search_result',
                            'url': 'https://www.cnn.com/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '14 hours ago',
                            'title': "World news: Latest news, breaking news, today's news stories from around the world, updated daily from CBS News",
                            'type': 'web_search_result',
                            'url': 'https://www.cbsnews.com/world/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '4 hours ago',
                            'title': 'International News | Latest World News, Videos & Photos -ABC News - ABC News',
                            'type': 'web_search_result',
                            'url': 'https://abcnews.go.com/International',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '1 hour ago',
                            'title': 'Google News',
                            'type': 'web_search_result',
                            'url': 'https://news.google.com/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '2 days ago',
                            'title': 'World News Headlines - US News and World Report',
                            'type': 'web_search_result',
                            'url': 'https://www.usnews.com/news/world',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '2 hours ago',
                            'title': 'Fox News - Breaking News Updates | Latest News Headlines | Photos & News Videos',
                            'type': 'web_search_result',
                            'url': 'https://www.foxnews.com/',
                        },
                    ],
                    tool_call_id='srvtoolu_01NcU4XNwyxWK6a9tcJZ8wGY',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=2,
                part=TextPart(content='Let me search for more specific breaking'),
                previous_part_kind='builtin-tool-return',
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' news stories to get clearer headlines.')),
            PartEndEvent(
                index=2,
                part=TextPart(
                    content='Let me search for more specific breaking news stories to get clearer headlines.'
                ),
                next_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=3,
                part=BuiltinToolCallPart(
                    tool_name='web_search', tool_call_id='srvtoolu_01WiP3ZfXZXSykVQEL78XJ4T', provider_name='anthropic'
                ),
                previous_part_kind='text',
            ),
            PartDeltaEvent(
                index=3, delta=ToolCallPartDelta(args_delta='', tool_call_id='srvtoolu_01WiP3ZfXZXSykVQEL78XJ4T')
            ),
            PartDeltaEvent(
                index=3, delta=ToolCallPartDelta(args_delta='{"query', tool_call_id='srvtoolu_01WiP3ZfXZXSykVQEL78XJ4T')
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(args_delta='": "breaki', tool_call_id='srvtoolu_01WiP3ZfXZXSykVQEL78XJ4T'),
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(args_delta='ng news ', tool_call_id='srvtoolu_01WiP3ZfXZXSykVQEL78XJ4T'),
            ),
            PartDeltaEvent(
                index=3, delta=ToolCallPartDelta(args_delta='headl', tool_call_id='srvtoolu_01WiP3ZfXZXSykVQEL78XJ4T')
            ),
            PartDeltaEvent(
                index=3,
                delta=ToolCallPartDelta(args_delta='ines August ', tool_call_id='srvtoolu_01WiP3ZfXZXSykVQEL78XJ4T'),
            ),
            PartDeltaEvent(
                index=3, delta=ToolCallPartDelta(args_delta='14 2025', tool_call_id='srvtoolu_01WiP3ZfXZXSykVQEL78XJ4T')
            ),
            PartDeltaEvent(
                index=3, delta=ToolCallPartDelta(args_delta='"}', tool_call_id='srvtoolu_01WiP3ZfXZXSykVQEL78XJ4T')
            ),
            PartEndEvent(
                index=3,
                part=BuiltinToolCallPart(
                    tool_name='web_search',
                    args='{"query": "breaking news headlines August 14 2025"}',
                    tool_call_id='srvtoolu_01WiP3ZfXZXSykVQEL78XJ4T',
                    provider_name='anthropic',
                ),
                next_part_kind='builtin-tool-return',
            ),
            PartStartEvent(
                index=4,
                part=BuiltinToolReturnPart(
                    tool_name='web_search',
                    content=[
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'Breaking News, Latest News and Videos | CNN',
                            'type': 'web_search_result',
                            'url': 'https://edition.cnn.com/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'News: U.S. and World News Headlines : NPR',
                            'type': 'web_search_result',
                            'url': 'https://www.npr.org/sections/news/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'ABC News – Breaking News, Latest News and Videos',
                            'type': 'web_search_result',
                            'url': 'https://abcnews.go.com/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '4 hours ago',
                            'title': 'Newspaper headlines: Thursday, August 14, 2025 - Adomonline.com',
                            'type': 'web_search_result',
                            'url': 'https://www.adomonline.com/newspaper-headlines-thursday-august-14-2025/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'Global News - Breaking International News And Headlines | Inquirer.net',
                            'type': 'web_search_result',
                            'url': 'https://globalnation.inquirer.net',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'News – The White House',
                            'type': 'web_search_result',
                            'url': 'https://www.whitehouse.gov/news/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '1 hour ago',
                            'title': 'Latest News: Top News, Breaking News, LIVE News Headlines from India & World | Business Standard',
                            'type': 'web_search_result',
                            'url': 'https://www.business-standard.com/latest-news',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '10 hours ago',
                            'title': 'Ukraine News Today: Breaking Updates & Live Coverage - August 14, 2025 from Kyiv Post',
                            'type': 'web_search_result',
                            'url': 'https://www.kyivpost.com/thread/58085',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': 'July 14, 2025',
                            'title': '5 things to know for July 14: Immigration, Gaza, Epstein files, Kentucky shooting, Texas flooding | CNN',
                            'type': 'web_search_result',
                            'url': 'https://www.cnn.com/2025/07/14/us/5-things-to-know-for-july-14-immigration-gaza-epstein-files-kentucky-shooting-texas-flooding',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'Daily Show for July 14, 2025 | Democracy Now!',
                            'type': 'web_search_result',
                            'url': 'https://www.democracynow.org/shows/2025/7/14',
                        },
                    ],
                    tool_call_id='srvtoolu_01WiP3ZfXZXSykVQEL78XJ4T',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartStartEvent(index=5, part=TextPart(content='Base'), previous_part_kind='builtin-tool-return'),
            PartDeltaEvent(
                index=5, delta=TextPartDelta(content_delta='d on the search results, I can identify the top')
            ),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta=' 3 major news stories from aroun')),
            PartDeltaEvent(
                index=5,
                delta=TextPartDelta(
                    content_delta="""\
d the world today (August 14, 2025):

## Top\
"""
                ),
            ),
            PartDeltaEvent(
                index=5,
                delta=TextPartDelta(
                    content_delta="""\
 3 World News Stories Today

**\
"""
                ),
            ),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta='1. Trump-Putin Summit and Ukraine Crisis')),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta='**\n')),
            PartEndEvent(
                index=5,
                part=TextPart(
                    content="""\
Based on the search results, I can identify the top 3 major news stories from around the world today (August 14, 2025):

## Top 3 World News Stories Today

**1. Trump-Putin Summit and Ukraine Crisis**
"""
                ),
                next_part_kind='text',
            ),
            PartStartEvent(
                index=6,
                part=TextPart(
                    content='European leaders held a high-stakes meeting Wednesday with President Trump, Vice President Vance, Ukraine'
                ),
                previous_part_kind='text',
            ),
            PartDeltaEvent(index=6, delta=TextPartDelta(content_delta="'s Volodymyr Zel")),
            PartDeltaEvent(index=6, delta=TextPartDelta(content_delta="enskyy and NATO's chief ahea")),
            PartDeltaEvent(index=6, delta=TextPartDelta(content_delta="d of Friday's U.S.-")),
            PartDeltaEvent(index=6, delta=TextPartDelta(content_delta='Russia summit')),
            PartEndEvent(
                index=6,
                part=TextPart(
                    content="European leaders held a high-stakes meeting Wednesday with President Trump, Vice President Vance, Ukraine's Volodymyr Zelenskyy and NATO's chief ahead of Friday's U.S.-Russia summit"
                ),
                next_part_kind='text',
            ),
            PartStartEvent(index=7, part=TextPart(content='. '), previous_part_kind='text'),
            PartEndEvent(index=7, part=TextPart(content='. '), next_part_kind='text'),
            PartStartEvent(
                index=8,
                part=TextPart(content='The White House lowered its expectations surrounding'),
                previous_part_kind='text',
            ),
            PartDeltaEvent(index=8, delta=TextPartDelta(content_delta=' the Trump-Putin summit on Friday')),
            PartEndEvent(
                index=8,
                part=TextPart(
                    content='The White House lowered its expectations surrounding the Trump-Putin summit on Friday'
                ),
                next_part_kind='text',
            ),
            PartStartEvent(index=9, part=TextPart(content='. '), previous_part_kind='text'),
            PartEndEvent(index=9, part=TextPart(content='. '), next_part_kind='text'),
            PartStartEvent(
                index=10,
                part=TextPart(content='In a surprise move just days before the Trump-Putin summit'),
                previous_part_kind='text',
            ),
            PartDeltaEvent(index=10, delta=TextPartDelta(content_delta=', the White House swapped out pro')),
            PartDeltaEvent(index=10, delta=TextPartDelta(content_delta="-EU PM Tusk for Poland's new president –")),
            PartDeltaEvent(index=10, delta=TextPartDelta(content_delta=" a political ally who once opposed Ukraine's")),
            PartDeltaEvent(index=10, delta=TextPartDelta(content_delta=' NATO and EU bids')),
            PartEndEvent(
                index=10,
                part=TextPart(
                    content="In a surprise move just days before the Trump-Putin summit, the White House swapped out pro-EU PM Tusk for Poland's new president – a political ally who once opposed Ukraine's NATO and EU bids"
                ),
                next_part_kind='text',
            ),
            PartStartEvent(
                index=11,
                part=TextPart(
                    content="""\
.

**2. Trump's Federal Takeover of Washington D\
"""
                ),
                previous_part_kind='text',
            ),
            PartDeltaEvent(index=11, delta=TextPartDelta(content_delta='.C.**')),
            PartDeltaEvent(index=11, delta=TextPartDelta(content_delta='\n')),
            PartEndEvent(
                index=11,
                part=TextPart(
                    content="""\
.

**2. Trump's Federal Takeover of Washington D.C.**
"""
                ),
                next_part_kind='text',
            ),
            PartStartEvent(
                index=12,
                part=TextPart(
                    content="Federal law enforcement's presence in Washington, DC, continued to be felt Wednesday as President Donald Trump's tak"
                ),
                previous_part_kind='text',
            ),
            PartDeltaEvent(index=12, delta=TextPartDelta(content_delta="eover of the city's police entered its thir")),
            PartDeltaEvent(index=12, delta=TextPartDelta(content_delta='d night')),
            PartEndEvent(
                index=12,
                part=TextPart(
                    content="Federal law enforcement's presence in Washington, DC, continued to be felt Wednesday as President Donald Trump's takeover of the city's police entered its third night"
                ),
                next_part_kind='text',
            ),
            PartStartEvent(index=13, part=TextPart(content='. '), previous_part_kind='text'),
            PartEndEvent(index=13, part=TextPart(content='. '), next_part_kind='text'),
            PartStartEvent(
                index=14,
                part=TextPart(
                    content="National Guard troops arrived in Washington, D.C., following President Trump's deployment an"
                ),
                previous_part_kind='text',
            ),
            PartDeltaEvent(
                index=14, delta=TextPartDelta(content_delta='d federalization of local police to crack down on crime')
            ),
            PartDeltaEvent(index=14, delta=TextPartDelta(content_delta=" in the nation's capital")),
            PartEndEvent(
                index=14,
                part=TextPart(
                    content="National Guard troops arrived in Washington, D.C., following President Trump's deployment and federalization of local police to crack down on crime in the nation's capital"
                ),
                next_part_kind='text',
            ),
            PartStartEvent(index=15, part=TextPart(content='. '), previous_part_kind='text'),
            PartEndEvent(index=15, part=TextPart(content='. '), next_part_kind='text'),
            PartStartEvent(
                index=16,
                part=TextPart(content='Over 100 arrests made as National Guard rolls into DC under'),
                previous_part_kind='text',
            ),
            PartDeltaEvent(index=16, delta=TextPartDelta(content_delta=" Trump's federal takeover")),
            PartEndEvent(
                index=16,
                part=TextPart(
                    content="Over 100 arrests made as National Guard rolls into DC under Trump's federal takeover"
                ),
                next_part_kind='text',
            ),
            PartStartEvent(
                index=17,
                part=TextPart(
                    content="""\
.

**3. Air\
"""
                ),
                previous_part_kind='text',
            ),
            PartDeltaEvent(index=17, delta=TextPartDelta(content_delta=' Canada Flight Disruption')),
            PartDeltaEvent(index=17, delta=TextPartDelta(content_delta='**\n')),
            PartEndEvent(
                index=17,
                part=TextPart(
                    content="""\
.

**3. Air Canada Flight Disruption**
"""
                ),
                next_part_kind='text',
            ),
            PartStartEvent(
                index=18,
                part=TextPart(
                    content='Air Canada plans to lock out its flight attendants and cancel all flights starting this weekend'
                ),
                previous_part_kind='text',
            ),
            PartEndEvent(
                index=18,
                part=TextPart(
                    content='Air Canada plans to lock out its flight attendants and cancel all flights starting this weekend'
                ),
                next_part_kind='text',
            ),
            PartStartEvent(index=19, part=TextPart(content='. '), previous_part_kind='text'),
            PartEndEvent(index=19, part=TextPart(content='. '), next_part_kind='text'),
            PartStartEvent(
                index=20,
                part=TextPart(
                    content='Air Canada says it will begin cancelling flights starting Thursday to allow an orderly shutdown of operations'
                ),
                previous_part_kind='text',
            ),
            PartDeltaEvent(
                index=20,
                delta=TextPartDelta(
                    content_delta=" with a complete cessation of flights for the country's largest airline by"
                ),
            ),
            PartDeltaEvent(
                index=20, delta=TextPartDelta(content_delta=' Saturday as it faces a potential work stoppage by')
            ),
            PartDeltaEvent(index=20, delta=TextPartDelta(content_delta=' its flight attendants')),
            PartEndEvent(
                index=20,
                part=TextPart(
                    content="Air Canada says it will begin cancelling flights starting Thursday to allow an orderly shutdown of operations with a complete cessation of flights for the country's largest airline by Saturday as it faces a potential work stoppage by its flight attendants"
                ),
                next_part_kind='text',
            ),
            PartStartEvent(
                index=21,
                part=TextPart(
                    content="""\
.

These stories represent major international diplomatic developments, significant domestic policy\
"""
                ),
                previous_part_kind='text',
            ),
            PartDeltaEvent(index=21, delta=TextPartDelta(content_delta=' changes in the US, and major transportation')),
            PartDeltaEvent(index=21, delta=TextPartDelta(content_delta=' disruptions affecting North America.')),
            PartEndEvent(
                index=21,
                part=TextPart(
                    content="""\
.

These stories represent major international diplomatic developments, significant domestic policy changes in the US, and major transportation disruptions affecting North America.\
"""
                ),
            ),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='web_search',
                    args='{"query": "top world news today"}',
                    tool_call_id='srvtoolu_01NcU4XNwyxWK6a9tcJZ8wGY',
                    provider_name='anthropic',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='web_search',
                    content=[
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '4 hours ago',
                            'title': 'World news - breaking news, video, headlines and opinion | CNN',
                            'type': 'web_search_result',
                            'url': 'https://www.cnn.com/world',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '1 hour ago',
                            'title': 'Breaking News, World News and Video from Al Jazeera',
                            'type': 'web_search_result',
                            'url': 'https://www.aljazeera.com/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '1 hour ago',
                            'title': 'News: U.S. and World News Headlines : NPR',
                            'type': 'web_search_result',
                            'url': 'https://www.npr.org/sections/news/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '7 hours ago',
                            'title': 'NBC News - Breaking News & Top Stories - Latest World, US & Local News | NBC News',
                            'type': 'web_search_result',
                            'url': 'https://www.nbcnews.com/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '3 hours ago',
                            'title': 'Breaking News, Latest News and Videos | CNN',
                            'type': 'web_search_result',
                            'url': 'https://www.cnn.com/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '14 hours ago',
                            'title': "World news: Latest news, breaking news, today's news stories from around the world, updated daily from CBS News",
                            'type': 'web_search_result',
                            'url': 'https://www.cbsnews.com/world/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '4 hours ago',
                            'title': 'International News | Latest World News, Videos & Photos -ABC News - ABC News',
                            'type': 'web_search_result',
                            'url': 'https://abcnews.go.com/International',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '1 hour ago',
                            'title': 'Google News',
                            'type': 'web_search_result',
                            'url': 'https://news.google.com/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '2 days ago',
                            'title': 'World News Headlines - US News and World Report',
                            'type': 'web_search_result',
                            'url': 'https://www.usnews.com/news/world',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '2 hours ago',
                            'title': 'Fox News - Breaking News Updates | Latest News Headlines | Photos & News Videos',
                            'type': 'web_search_result',
                            'url': 'https://www.foxnews.com/',
                        },
                    ],
                    tool_call_id='srvtoolu_01NcU4XNwyxWK6a9tcJZ8wGY',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                )
            ),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='web_search',
                    args='{"query": "breaking news headlines August 14 2025"}',
                    tool_call_id='srvtoolu_01WiP3ZfXZXSykVQEL78XJ4T',
                    provider_name='anthropic',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='web_search',
                    content=[
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'Breaking News, Latest News and Videos | CNN',
                            'type': 'web_search_result',
                            'url': 'https://edition.cnn.com/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'News: U.S. and World News Headlines : NPR',
                            'type': 'web_search_result',
                            'url': 'https://www.npr.org/sections/news/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'ABC News – Breaking News, Latest News and Videos',
                            'type': 'web_search_result',
                            'url': 'https://abcnews.go.com/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '4 hours ago',
                            'title': 'Newspaper headlines: Thursday, August 14, 2025 - Adomonline.com',
                            'type': 'web_search_result',
                            'url': 'https://www.adomonline.com/newspaper-headlines-thursday-august-14-2025/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'Global News - Breaking International News And Headlines | Inquirer.net',
                            'type': 'web_search_result',
                            'url': 'https://globalnation.inquirer.net',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'News – The White House',
                            'type': 'web_search_result',
                            'url': 'https://www.whitehouse.gov/news/',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '1 hour ago',
                            'title': 'Latest News: Top News, Breaking News, LIVE News Headlines from India & World | Business Standard',
                            'type': 'web_search_result',
                            'url': 'https://www.business-standard.com/latest-news',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': '10 hours ago',
                            'title': 'Ukraine News Today: Breaking Updates & Live Coverage - August 14, 2025 from Kyiv Post',
                            'type': 'web_search_result',
                            'url': 'https://www.kyivpost.com/thread/58085',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': 'July 14, 2025',
                            'title': '5 things to know for July 14: Immigration, Gaza, Epstein files, Kentucky shooting, Texas flooding | CNN',
                            'type': 'web_search_result',
                            'url': 'https://www.cnn.com/2025/07/14/us/5-things-to-know-for-july-14-immigration-gaza-epstein-files-kentucky-shooting-texas-flooding',
                        },
                        {
                            'encrypted_content': IsStr(),
                            'page_age': None,
                            'title': 'Daily Show for July 14, 2025 | Democracy Now!',
                            'type': 'web_search_result',
                            'url': 'https://www.democracynow.org/shows/2025/7/14',
                        },
                    ],
                    tool_call_id='srvtoolu_01WiP3ZfXZXSykVQEL78XJ4T',
                    timestamp=IsDatetime(),
                    provider_name='anthropic',
                )
            ),
        ]
    )


async def test_anthropic_text_parts_ahead_of_built_in_tool_call(allow_model_requests: None, anthropic_api_key: str):
    # Verify that text parts ahead of the built-in tool call are not included in the output

    anthropic_model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(anthropic_model, builtin_tools=[WebSearchTool()], instructions='Be very concise.')

    result = await agent.run('Briefly mention 1 event that happened today in history?')
    assert result.output == snapshot("""\
Here's one significant historical event that occurred on September 17:

In 1939, Finnish runner Taisto Mäki made history by becoming the first person to run 10,000 meters in less than 30 minutes, completing the distance in 29 minutes and 52 seconds.\
""")

    async with agent.run_stream('Briefly mention 1 event that happened tomorrow in history?') as result:
        chunks = [c async for c in result.stream_output(debounce_by=None)]
        assert chunks == snapshot(
            [
                'Let',
                'Let me search for a significant',
                'Let me search for a significant historical event that occurred on',
                'Let me search for a significant historical event that occurred on September 18th.',
                '',
                '',
                '',
                '',
                '',
                '',
                '',
                '',
                '',
                '',
                '',
                '',
                'Here',
                "Here's one notable historical event that occurred on September",
                "Here's one notable historical event that occurred on September 18th: ",
                "Here's one notable historical event that occurred on September 18th: On September 18, 1793, President George Washington marke",
                "Here's one notable historical event that occurred on September 18th: On September 18, 1793, President George Washington marked the location for the Capitol Building",
                "Here's one notable historical event that occurred on September 18th: On September 18, 1793, President George Washington marked the location for the Capitol Building in Washington DC, and he",
                "Here's one notable historical event that occurred on September 18th: On September 18, 1793, President George Washington marked the location for the Capitol Building in Washington DC, and he would return periodically to oversee its",
                "Here's one notable historical event that occurred on September 18th: On September 18, 1793, President George Washington marked the location for the Capitol Building in Washington DC, and he would return periodically to oversee its construction personally",
                "Here's one notable historical event that occurred on September 18th: On September 18, 1793, President George Washington marked the location for the Capitol Building in Washington DC, and he would return periodically to oversee its construction personally.",
                "Here's one notable historical event that occurred on September 18th: On September 18, 1793, President George Washington marked the location for the Capitol Building in Washington DC, and he would return periodically to oversee its construction personally.",
            ]
        )

    assert await result.get_output() == snapshot(
        "Here's one notable historical event that occurred on September 18th: On September 18, 1793, President George Washington marked the location for the Capitol Building in Washington DC, and he would return periodically to oversee its construction personally."
    )

    async with agent.run_stream('Briefly mention 1 event that happened yesterday in history?') as result:
        chunks = [c async for c in result.stream_text(debounce_by=None)]
        assert chunks == snapshot(
            [
                'Let',
                'Let me search for a historical',
                'Let me search for a historical event that occurred on September',
                "Let me search for a historical event that occurred on September 16th (yesterday's date since",
                "Let me search for a historical event that occurred on September 16th (yesterday's date since today is September 17,",
                "Let me search for a historical event that occurred on September 16th (yesterday's date since today is September 17, 2025",
                "Let me search for a historical event that occurred on September 16th (yesterday's date since today is September 17, 2025).",
                """\
Let me search for a historical event that occurred on September 16th (yesterday's date since today is September 17, 2025).

""",
                """\
Let me search for a historical event that occurred on September 16th (yesterday's date since today is September 17, 2025).

Base\
""",
                """\
Let me search for a historical event that occurred on September 16th (yesterday's date since today is September 17, 2025).

Based on yesterday's date (\
""",
                """\
Let me search for a historical event that occurred on September 16th (yesterday's date since today is September 17, 2025).

Based on yesterday's date (September 16, 2025\
""",
                """\
Let me search for a historical event that occurred on September 16th (yesterday's date since today is September 17, 2025).

Based on yesterday's date (September 16, 2025), \
""",
                """\
Let me search for a historical event that occurred on September 16th (yesterday's date since today is September 17, 2025).

Based on yesterday's date (September 16, 2025), Asian markets rose higher as Federal Reserve rate cut hopes\
""",
                """\
Let me search for a historical event that occurred on September 16th (yesterday's date since today is September 17, 2025).

Based on yesterday's date (September 16, 2025), Asian markets rose higher as Federal Reserve rate cut hopes lifted global market sentiment\
""",
                """\
Let me search for a historical event that occurred on September 16th (yesterday's date since today is September 17, 2025).

Based on yesterday's date (September 16, 2025), Asian markets rose higher as Federal Reserve rate cut hopes lifted global market sentiment. Additionally, \
""",
                """\
Let me search for a historical event that occurred on September 16th (yesterday's date since today is September 17, 2025).

Based on yesterday's date (September 16, 2025), Asian markets rose higher as Federal Reserve rate cut hopes lifted global market sentiment. Additionally, there were severe rain and gales\
""",
                """\
Let me search for a historical event that occurred on September 16th (yesterday's date since today is September 17, 2025).

Based on yesterday's date (September 16, 2025), Asian markets rose higher as Federal Reserve rate cut hopes lifted global market sentiment. Additionally, there were severe rain and gales impacting parts\
""",
                """\
Let me search for a historical event that occurred on September 16th (yesterday's date since today is September 17, 2025).

Based on yesterday's date (September 16, 2025), Asian markets rose higher as Federal Reserve rate cut hopes lifted global market sentiment. Additionally, there were severe rain and gales impacting parts of New Zealand, an\
""",
                """\
Let me search for a historical event that occurred on September 16th (yesterday's date since today is September 17, 2025).

Based on yesterday's date (September 16, 2025), Asian markets rose higher as Federal Reserve rate cut hopes lifted global market sentiment. Additionally, there were severe rain and gales impacting parts of New Zealand, and a notable court case involving\
""",
                """\
Let me search for a historical event that occurred on September 16th (yesterday's date since today is September 17, 2025).

Based on yesterday's date (September 16, 2025), Asian markets rose higher as Federal Reserve rate cut hopes lifted global market sentiment. Additionally, there were severe rain and gales impacting parts of New Zealand, and a notable court case involving a British aristoc\
""",
                """\
Let me search for a historical event that occurred on September 16th (yesterday's date since today is September 17, 2025).

Based on yesterday's date (September 16, 2025), Asian markets rose higher as Federal Reserve rate cut hopes lifted global market sentiment. Additionally, there were severe rain and gales impacting parts of New Zealand, and a notable court case involving a British aristocrat\
""",
                """\
Let me search for a historical event that occurred on September 16th (yesterday's date since today is September 17, 2025).

Based on yesterday's date (September 16, 2025), Asian markets rose higher as Federal Reserve rate cut hopes lifted global market sentiment. Additionally, there were severe rain and gales impacting parts of New Zealand, and a notable court case involving a British aristocrat.\
""",
            ]
        )

    assert await result.get_output() == snapshot(
        "Based on yesterday's date (September 16, 2025), Asian markets rose higher as Federal Reserve rate cut hopes lifted global market sentiment. Additionally, there were severe rain and gales impacting parts of New Zealand, and a notable court case involving a British aristocrat."
    )

    async with agent.run_stream(
        'Briefly mention 1 event that happened the day after tomorrow in history?'
    ) as result:  # pragma: lax no cover
        chunks = [c async for c in result.stream_text(debounce_by=None, delta=True)]  # pragma: lax no cover
        assert chunks == snapshot(
            [
                'Let',
                ' me search for historical',
                ' events that occurred on',
                ' September 19th.',
                """\


""",
                'Here',
                "'s one significant historical event that occurred on September",
                ' 19th: ',
                'New Zealand made history by becoming the first self-governing nation to grant women the right',
                ' to vote in national elections. It',
                ' would take 27 more',
                ' years before American women gained the',
                ' same right.',
            ]
        )

    assert await result.get_output() == snapshot(
        "Here's one significant historical event that occurred on September 19th: New Zealand made history by becoming the first self-governing nation to grant women the right to vote in national elections. It would take 27 more years before American women gained the same right."
    )


async def test_anthropic_memory_tool(allow_model_requests: None, anthropic_api_key: str):
    anthropic_model = AnthropicModel(
        'claude-sonnet-4-5',
        provider=AnthropicProvider(api_key=anthropic_api_key),
        settings=AnthropicModelSettings(extra_headers={'anthropic-beta': 'context-1m-2025-08-07'}),
    )
    agent = Agent(anthropic_model, builtin_tools=[MemoryTool()])

    with pytest.raises(UserError, match="Built-in `MemoryTool` requires a 'memory' tool to be defined."):
        await agent.run('Where do I live?')

    class FakeMemoryTool(BetaAbstractMemoryTool):
        def view(self, command: BetaMemoryTool20250818ViewCommand) -> str:
            return 'The user lives in Mexico City.'

        def create(self, command: BetaMemoryTool20250818CreateCommand) -> str:
            return f'File created successfully at {command.path}'  # pragma: no cover

        def str_replace(self, command: BetaMemoryTool20250818StrReplaceCommand) -> str:
            return f'File {command.path} has been edited'  # pragma: no cover

        def insert(self, command: BetaMemoryTool20250818InsertCommand) -> str:
            return f'Text inserted at line {command.insert_line} in {command.path}'  # pragma: no cover

        def delete(self, command: BetaMemoryTool20250818DeleteCommand) -> str:
            return f'File deleted: {command.path}'  # pragma: no cover

        def rename(self, command: BetaMemoryTool20250818RenameCommand) -> str:
            return f'Renamed {command.old_path} to {command.new_path}'  # pragma: no cover

        def clear_all_memory(self) -> str:
            return 'All memory cleared'  # pragma: no cover

    fake_memory = FakeMemoryTool()

    @agent.tool_plain
    def memory(**command: Any) -> Any:
        return fake_memory.call(command)

    result = await agent.run('Where do I live?')
    assert result.output == snapshot("""\


According to my memory, you live in **Mexico City**.\
""")


async def test_anthropic_model_usage_limit_exceeded(
    allow_model_requests: None,
    anthropic_api_key: str,
):
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(model=model)

    with pytest.raises(
        UsageLimitExceeded,
        match='The next request would exceed the input_tokens_limit of 18 \\(input_tokens=19\\)',
    ):
        await agent.run(
            'The quick brown fox jumps over the lazydog.',
            usage_limits=UsageLimits(input_tokens_limit=18, count_tokens_before_request=True),
        )


async def test_anthropic_model_usage_limit_not_exceeded(
    allow_model_requests: None,
    anthropic_api_key: str,
):
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(model=model)

    result = await agent.run(
        'The quick brown fox jumps over the lazydog.',
        usage_limits=UsageLimits(input_tokens_limit=25, count_tokens_before_request=True),
    )
    assert result.output == snapshot(
        """\
I noticed a small typo in that famous pangram! It should be:

"The quick brown fox jumps over the **lazy dog**."

(There should be a space between "lazy" and "dog")

This sentence is often used for testing typewriters, fonts, and keyboards because it contains every letter of the English alphabet at least once.\
"""
    )


async def test_anthropic_count_tokens_with_mock(allow_model_requests: None):
    """Test that count_tokens is called on the mock client."""
    c = completion_message(
        [BetaTextBlock(text='hello world', type='text')], BetaUsage(input_tokens=5, output_tokens=10)
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    result = await agent.run('hello', usage_limits=UsageLimits(input_tokens_limit=20, count_tokens_before_request=True))
    assert result.output == 'hello world'
    assert len(mock_client.chat_completion_kwargs) == 2  # type: ignore
    count_tokens_kwargs = mock_client.chat_completion_kwargs[0]  # type: ignore
    assert 'model' in count_tokens_kwargs
    assert 'messages' in count_tokens_kwargs


async def test_anthropic_count_tokens_with_no_messages(allow_model_requests: None):
    """Test count_tokens when messages_ is None (no exception configured)."""
    mock_client = cast(AsyncAnthropic, MockAnthropic())
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))

    result = await m.count_tokens(
        [ModelRequest.user_text_prompt('hello')],
        None,
        ModelRequestParameters(),
    )

    assert result.input_tokens == 10


@pytest.mark.vcr()
async def test_anthropic_count_tokens_error(allow_model_requests: None, anthropic_api_key: str):
    """Test that errors convert to ModelHTTPError."""
    model_id = 'claude-does-not-exist'
    model = AnthropicModel(model_id, provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(model)

    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('hello', usage_limits=UsageLimits(input_tokens_limit=20, count_tokens_before_request=True))

    assert exc_info.value.status_code == 404
    assert exc_info.value.model_name == model_id


async def test_anthropic_bedrock_count_tokens_not_supported(env: TestEnv):
    """Test that AsyncAnthropicBedrock raises UserError for count_tokens."""
    from anthropic import AsyncAnthropicBedrock

    bedrock_client = AsyncAnthropicBedrock(
        aws_access_key='test-access-key',
        aws_secret_key='test-secret-key',
        aws_region='us-east-1',
    )
    provider = AnthropicProvider(anthropic_client=bedrock_client)
    model = AnthropicModel('anthropic.claude-3-5-sonnet-20241022-v2:0', provider=provider)
    agent = Agent(model)

    with pytest.raises(UserError, match='AsyncAnthropicBedrock client does not support `count_tokens` api.'):
        await agent.run('hello', usage_limits=UsageLimits(input_tokens_limit=20, count_tokens_before_request=True))


@pytest.mark.vcr()
async def test_anthropic_cache_messages_real_api(allow_model_requests: None, anthropic_api_key: str):
    """Test that anthropic_cache_messages setting adds cache_control and produces cache usage metrics.

    This test uses a cassette to verify the cache behavior without making real API calls in CI.
    When run with real API credentials, it demonstrates that:
    1. The first call with a long context creates a cache (cache_write_tokens > 0)
    2. Follow-up messages in the same conversation can read from that cache (cache_read_tokens > 0)
    """
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(
        m,
        system_prompt='You are a helpful assistant.',
        model_settings=AnthropicModelSettings(
            anthropic_cache_messages=True,
        ),
    )

    # First call with a longer message - this will cache the message content
    result1 = await agent.run('Please explain what Python is and its main use cases. ' * 100)
    usage1 = result1.usage()

    # With anthropic_cache_messages, the first call should write cache for the last message
    # (cache_write_tokens > 0 indicates that caching occurred)
    assert usage1.requests == 1
    assert usage1.cache_write_tokens > 0
    assert usage1.output_tokens > 0

    # Continue the conversation - this message appends to history
    # The previous cached message should still be in the request
    result2 = await agent.run('Can you summarize that in one sentence?', message_history=result1.all_messages())
    usage2 = result2.usage()

    # The second call should potentially read from cache if the previous message is still cached
    # (cache_read_tokens > 0 when cache hit occurs)
    # (cache_write_tokens > 0 as new message is added to cache)
    assert usage2.requests == 1
    assert usage2.cache_read_tokens > 0
    assert usage2.cache_write_tokens > 0
    assert usage2.output_tokens > 0


async def test_anthropic_container_setting_explicit(allow_model_requests: None):
    """Test that anthropic_container setting passes explicit container config to API."""
    c = completion_message([BetaTextBlock(text='world', type='text')], BetaUsage(input_tokens=5, output_tokens=10))
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    # Test with explicit container config
    await agent.run('hello', model_settings=AnthropicModelSettings(anthropic_container={'id': 'container_abc123'}))

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert completion_kwargs['container'] == BetaContainerParams(id='container_abc123')


async def test_anthropic_container_from_message_history(allow_model_requests: None):
    """Test that container_id from message history is passed to subsequent requests."""
    c = completion_message([BetaTextBlock(text='world', type='text')], BetaUsage(input_tokens=5, output_tokens=10))
    mock_client = MockAnthropic.create_mock([c, c])
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    # Create a message history with a container_id in provider_details
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
        ModelResponse(
            parts=[TextPart(content='world')],
            provider_name='anthropic',
            provider_details={'container_id': 'container_from_history'},
        ),
    ]

    # Run with the message history
    await agent.run('follow up', message_history=history)

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert completion_kwargs['container'] == BetaContainerParams(id='container_from_history')


async def test_anthropic_container_setting_false_ignores_history(allow_model_requests: None):
    """Test that anthropic_container=False ignores container_id from history."""
    c = completion_message([BetaTextBlock(text='world', type='text')], BetaUsage(input_tokens=5, output_tokens=10))
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    # Create a message history with a container_id
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
        ModelResponse(
            parts=[TextPart(content='world')],
            provider_name='anthropic',
            provider_details={'container_id': 'container_should_be_ignored'},
        ),
    ]

    # Run with anthropic_container=False to force fresh container
    await agent.run(
        'follow up', message_history=history, model_settings=AnthropicModelSettings(anthropic_container=False)
    )

    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    # When anthropic_container=False, container should be OMIT (filtered out before sending to API)
    from anthropic import omit as OMIT

    assert completion_kwargs.get('container') is OMIT


async def test_anthropic_container_id_from_stream_response(allow_model_requests: None):
    """Test that container_id is extracted from streamed response and stored in provider_details."""
    from datetime import datetime

    stream_events: list[BetaRawMessageStreamEvent] = [
        BetaRawMessageStartEvent(
            type='message_start',
            message=BetaMessage(
                id='msg_123',
                content=[],
                model='claude-3-5-haiku-123',
                role='assistant',
                stop_reason=None,
                type='message',
                usage=BetaUsage(input_tokens=5, output_tokens=0),
                container=BetaContainer(
                    id='container_from_stream',
                    expires_at=datetime(2025, 1, 1, 0, 0, 0),
                ),
            ),
        ),
        BetaRawContentBlockStartEvent(
            type='content_block_start',
            index=0,
            content_block=BetaTextBlock(text='', type='text'),
        ),
        BetaRawContentBlockDeltaEvent(
            type='content_block_delta',
            index=0,
            delta=BetaTextDelta(type='text_delta', text='hello'),
        ),
        BetaRawContentBlockStopEvent(type='content_block_stop', index=0),
        BetaRawMessageDeltaEvent(
            type='message_delta',
            delta=Delta(stop_reason='end_turn', stop_sequence=None),
            usage=BetaMessageDeltaUsage(output_tokens=5),
        ),
        BetaRawMessageStopEvent(type='message_stop'),
    ]

    mock_client = MockAnthropic.create_stream_mock(stream_events)
    m = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('hello') as result:
        response = await result.get_output()
        assert response == 'hello'

    # Check that container_id was captured in the response
    messages = result.all_messages()
    model_response = messages[-1]
    assert isinstance(model_response, ModelResponse)
    assert model_response.provider_details is not None
    assert model_response.provider_details.get('container_id') == 'container_from_stream'
    assert model_response.provider_details.get('finish_reason') == 'end_turn'


async def test_anthropic_system_prompts_and_instructions_ordering():
    """Test that instructions are appended after all system prompts in the system prompt string."""
    m = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='test-key'))

    messages: list[ModelRequest | ModelResponse] = [
        ModelRequest(
            parts=[
                SystemPromptPart(content='System prompt 1'),
                SystemPromptPart(content='System prompt 2'),
                UserPromptPart(content='Hello'),
            ],
            instructions='Instructions content',
        ),
    ]

    system_prompt, anthropic_messages = await m._map_message(messages, ModelRequestParameters(), {})  # pyright: ignore[reportPrivateUsage]

    # Verify system prompts and instructions are joined in order: system1, system2, instructions
    assert system_prompt == snapshot("""\
System prompt 1

System prompt 2

Instructions content\
""")
    # Verify user message is in anthropic_messages
    assert len(anthropic_messages) == 1
    assert anthropic_messages[0]['role'] == 'user'
