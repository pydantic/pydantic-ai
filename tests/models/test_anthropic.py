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
from pydantic_ai.result import RunUsage
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

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
            ModelRequest(parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[TextPart(content='world')],
                usage=RequestUsage(input_tokens=5, output_tokens=10, details={'input_tokens': 5, 'output_tokens': 10}),
                model_name='claude-3-5-haiku-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_request_id='123',
            ),
            ModelRequest(parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[TextPart(content='world')],
                usage=RequestUsage(input_tokens=5, output_tokens=10, details={'input_tokens': 5, 'output_tokens': 10}),
                model_name='claude-3-5-haiku-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_request_id='123',
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
                usage=RequestUsage(input_tokens=3, output_tokens=5, details={'input_tokens': 3, 'output_tokens': 5}),
                model_name='claude-3-5-haiku-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_request_id='123',
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
                usage=RequestUsage(input_tokens=2, output_tokens=1, details={'input_tokens': 2, 'output_tokens': 1}),
                model_name='claude-3-5-haiku-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_request_id='123',
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
                usage=RequestUsage(input_tokens=3, output_tokens=2, details={'input_tokens': 3, 'output_tokens': 2}),
                model_name='claude-3-5-haiku-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_request_id='123',
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
                usage=RequestUsage(input_tokens=3, output_tokens=5, details={'input_tokens': 3, 'output_tokens': 5}),
                model_name='claude-3-5-haiku-123',
                timestamp=IsNow(tz=timezone.utc),
                provider_request_id='123',
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
            RunUsage(
                requests=2,
                input_tokens=20,
                output_tokens=5,
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
        "This is a potato. It's a yellow/golden-colored potato with a smooth, slightly bumpy skin typical of many potato varieties. The potato appears to be a whole, unpeeled tuber with a classic oblong or oval shape. Potatoes are starchy root vegetables that are widely consumed around the world and can be prepared in many ways, such as boiling, baking, frying, or mashing."
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
        'This is a Great Horned Owl (Bubo virginianus), a large and powerful owl species native to the Americas. The image shows the owl perched on a log or branch, surrounded by soft yellow and green vegetation. The owl has distinctive ear tufts (the "horns"), large yellow eyes, and a mottled gray-brown plumage that provides excellent camouflage in woodland and grassland environments. Great Horned Owls are known for their impressive size, sharp talons, and nocturnal hunting habits. They are formidable predators that can hunt animals as large as skunks, rabbits, and even other birds of prey.'
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
                    TextPart(content='Let me get the image and check what fruit is shown.'),
                    ToolCallPart(tool_name='get_image', args={}, tool_call_id='toolu_01WALUz3dC75yywrmL6dF3Bc'),
                ],
                usage=RequestUsage(
                    input_tokens=372,
                    output_tokens=49,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 372,
                        'output_tokens': 49,
                    },
                ),
                model_name='claude-3-5-sonnet-20241022',
                timestamp=IsDatetime(),
                provider_request_id='msg_01Kwjzggomz7bv9og51qGFuH',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_image',
                        content='See file 1c8566',
                        tool_call_id='toolu_01WALUz3dC75yywrmL6dF3Bc',
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
                        content="The image shows a kiwi fruit that has been cut in half, displaying its characteristic bright green flesh with small black seeds arranged in a circular pattern around a white center core. The kiwi's flesh has the typical fuzzy brown skin visible around the edges. The image is a clean, well-lit close-up shot of the kiwi slice against a white background."
                    )
                ],
                usage=RequestUsage(
                    input_tokens=2025,
                    output_tokens=81,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 2025,
                        'output_tokens': 81,
                    },
                ),
                model_name='claude-3-5-sonnet-20241022',
                timestamp=IsDatetime(),
                provider_request_id='msg_015btMBYLTuDnMP7zAeuHQGi',
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
        'The document simply contains the text "Dummy PDF file" at the top of what appears to be an otherwise blank page.'
    )


async def test_document_url_input(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-3-5-sonnet-latest', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m)

    document_url = DocumentUrl(url='https://pdfobject.com/pdf/sample.pdf')

    result = await agent.run(['What is the main content on this document?', document_url])
    assert result.output == snapshot(
        'This document appears to be a sample PDF file that mainly contains Lorem ipsum text, which is placeholder text commonly used in design and publishing. The document starts with "Sample PDF" as its title, followed by the line "This is a simple PDF file. Fun fun fun." The rest of the content consists of several paragraphs of Lorem ipsum text, which is Latin-looking but essentially meaningless text used to demonstrate the visual form of a document without the distraction of meaningful content.'
    )


async def test_text_document_url_input(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-3-5-sonnet-latest', provider=AnthropicProvider(api_key=anthropic_api_key))
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
                provider_request_id='msg_01Fg1JVgvCYUHWsxrj9GkpEv',
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
                model_name='claude-3-7-sonnet-20250219',
                timestamp=IsDatetime(),
                provider_request_id='msg_01BnZvs3naGorn93wjjCDwbd',
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
                model_name='claude-3-7-sonnet-20250219',
                timestamp=IsDatetime(),
                provider_request_id=IsStr(),
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
                model_name='claude-3-7-sonnet-20250219',
                timestamp=IsDatetime(),
                provider_request_id=IsStr(),
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
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
.)
2. Look\
"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' both ways (left-')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='right-left in countries')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' where cars drive on the right;')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' right-left-right where')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' they drive on the left)')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\

3. Wait for\
"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' traffic to stop or for a clear')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
 gap in traffic
4\
"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='. Make eye contact with drivers if')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
 possible
5. Cross at\
"""
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
 a steady pace without running
6. Continue\
"""
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
 watching for traffic while crossing
7\
"""
                ),
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='. Use pedestrian signals where')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    content_delta="""\
 available

I'll also mention\
"""
                ),
            ),
            PartDeltaEvent(
                index=0, delta=ThinkingPartDelta(content_delta=' some additional safety tips and considerations for')
            ),
            PartDeltaEvent(
                index=0, delta=ThinkingPartDelta(content_delta=' different situations (busy streets, streets')
            ),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' with traffic signals, etc.).')),
            PartDeltaEvent(
                index=0,
                delta=ThinkingPartDelta(
                    signature_delta='ErUBCkYIBhgCIkA/Y+JwNMtmQyHcoo4/v2dpY6ruQifcu3pAzHbzIwpIrjIyaWaYdJOp9/0vUmBPj+LmqgiDSTktRcn0U75AlpXOEgwzVmYdHgDaZfeyBGcaDFSIZCHzzrZQkolJKCIwhMETosYLx+Dw/vKa83hht943z9R3/ViOqokT25JmMfaGOntuo+33Zxqf5rqUbkQ3Kh34rIqqnKaFSVr7Nn85z8OFN3Cwzz+HmXl2FgCXOxgC'
                ),
            ),
            PartStartEvent(index=1, part=IsInstance(TextPart)),
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
        pytest.param(
            lambda: BetaRawMessageDeltaEvent(
                delta=Delta(),
                usage=BetaMessageDeltaUsage(output_tokens=5),
                type='message_delta',
            ),
            snapshot(RequestUsage(output_tokens=5, details={'output_tokens': 5})),
            id='RawMessageDeltaEvent',
        ),
        pytest.param(
            lambda: BetaRawMessageStopEvent(type='message_stop'), snapshot(RequestUsage()), id='RawMessageStopEvent'
        ),
    ],
)
def test_usage(message_callback: Callable[[], BetaMessage | BetaRawMessageStreamEvent], usage: RunUsage):
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
I can't physically give you a potato since I'm a digital assistant. However, I can:

1. Help you find recipes that use potatoes
2. Give you tips on how to select, store, or prepare potatoes
3. Share information about different types of potatoes
4. Suggest where you might buy potatoes locally

What specific information about potatoes would be most helpful to you?\
""")


async def test_anthropic_web_search_tool(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-3-5-sonnet-latest', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m, builtin_tools=[WebSearchTool()])

    result = await agent.run('What day is today?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='What day is today?', timestamp=IsDatetime())]),
            ModelResponse(
                parts=[
                    TextPart(content="Let me search for current events to help establish today's date."),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'current events today August 14 2025'},
                        tool_call_id='srvtoolu_016Z2mQT8AFaH17TpZnmuj2Z',
                        provider_name='anthropic',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search_tool_result',
                        content=[
                            BetaWebSearchResultBlock(
                                encrypted_content='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',
                                title='Portal:Current events/August 2025 - Wikipedia',
                                type='web_search_result',
                                url='https://en.wikipedia.org/wiki/Portal:Current_events/August_2025',
                            ),
                            BetaWebSearchResultBlock(
                                encrypted_content='Eu8CCioIBhgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDI9+btUbgH5OLcLdUxoMG1wsVr7IuhSKMy3qIjBHk3pUXhm2b3ewsnrLIz6lTnViIULTjxK9hDivEVsoRww/fuVDQRZ8PbKaUO+dpk4q8gFrBL8hUB97ESFLZ4zGMOzedNRyEAAy3REe5WBSt2XXPIbvxMdlhxwDmnECJAlT/2tXFxrbSYFsrarrXfd6qGqb5ftE58fwKaISFh5dDwbZRT898FWEpWEjBhU+bXPy80gJOKZnQDva8eV0VYlFJnF+76Q5aIqcop0kHRzKD/KEj1LZj+k1WBrvKxNZU7RFdsr0mGx5apGPoczvdED5aQxkTIku+pea87u+coJg5kIsz0AM76P0sssYIcH8GmnlgMih5juw/P+g3GOYRiT67R9TfgPBff/6fpU8GlWZGn3Ap3nmUvrYp8LKUfOwR6sNyeuYxxgD',
                                title='August 14, 2025 Calendar with Holidays & Count Down - USA',
                                type='web_search_result',
                                url='https://www.wincalendar.com/Calendar/Date/August-14-2025',
                            ),
                            BetaWebSearchResultBlock(
                                encrypted_content='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',
                                title='Portal:Current events - Wikipedia',
                                type='web_search_result',
                                url='https://en.wikipedia.org/wiki/Portal:Current_events',
                            ),
                            BetaWebSearchResultBlock(
                                encrypted_content='EpsCCioIBhgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDM3/Olb0Z6OTNOAHHBoM8w2gzZQzodDjHZ5dIjDVb68LcKw9pkL8J9qZM/RCJ8Nwr9ntCnKALzYkJxFopMFT6Q0hKH9pK9Gw8K32o1gqngGFvSjOvyyznbW3kaNNQRMXXZtfg6l+GDJ8Vj+EbuSJkV+OCjRNUJCYTOeO0zlrDaspULGTCf5Y7Ifw1wCvpjwLE7I8oI+YJA9ZGhfHXxyDiP1jtu7Lm6TGT80vefH3yiNbYnuJnpogbkzYgAxHZtDWpq4kcGk7cci2gtBkBu4JskwW4XUio4gS1kJ4TGXLny6dTA72L+oe/EYBnyXiNRgD',
                                title='Portal:Current events/2025 August 14 - Wikipedia',
                                type='web_search_result',
                                url='https://en.wikipedia.org/wiki/Portal:Current_events/2025_August_14',
                            ),
                            BetaWebSearchResultBlock(
                                encrypted_content='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',
                                title='International News | Latest World News, Videos & Photos -ABC News - ABC News',
                                type='web_search_result',
                                url='https://abcnews.go.com/International',
                            ),
                            BetaWebSearchResultBlock(
                                encrypted_content='EqkHCioIBhgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDJhXQuVJyASY4nS14xoMPV7fmqhPOQvs53T2IjClShZH0ZjtMv3tcGa1ND/qwawzQZ3TWJQPdnRRlyky0zsjOz1hwoB4i/cw9vKMiuMqrAacvSreqm8Soq4R8zc3RlH+q99DrUCa6rb1U0zbH3YpGfeR+kZLIi55f45Xz0GGddBH4MiYjm2SvqBPXSHW51j0EsaqXXZ+fecTRZ6rF+HGwQl/sQbHSiL+uLqaEoraQMtQo3VxwM0CLFV7lbo3ntovUo//6oD1y5qCuJ0SyUWnkhgC9ERygHyTTnftbNO843fPiZ+ghIfxE7LhRbbKZXgRdsCXbjMvI6Y1lEhCAY6m8HLXEgHl/lLa7cFAEUsEhbBSf5e6zE3rDsY9XCdCf4yYZ0m5tWtkqfWA7z1tuUFE0WQDwEYJzhvpU9Kx7lBKC+T3gO5wddxy19g0NX0DV4D7uHl50G7J4eBdheUzXdlNo5PJkj5aWqZ5G1LpQMaJz6bTEVmp07TlA3c0/M2AY4AvRDiAARRI12+GRwKWBvkCkx4OchItfrZMx+APuv/ePFfrco4j0uddTdKSJZThRqLFP/nBbSsYUm3vewX1D+4BtJw5d/sUppt+CQGUtWcU+nKcFReHMIIFFzm6SjV3Q8LEjtx2BRYY2f53XXgtuV4CgqoI9MA+H9Gp3z9NNpQt4FuWfCFphOG+SzOIusldoE5rp6xuzedQpgR5AW4pV9sM5GgPLYlRogi6emLSaUOTx08gqCyuOOMJHu0KvSWBG5gi7t484ei9dLROA6R4Kbzj1WuiISYFBIMH65nEtnPWy51hXAAaOBuLQsT7gvYOTDN3bHCZ+O8p0FroaOz8q7D1fLWH/+yTHURa6PPfXV5POHUWjVDFdDRtdKBYMloBQ53azLx73Q9VH7ZGxE+g9336BCFXfl/nonNHbFNNo88mXy1+LVpRpsACQ5Txn17xGclRvfNHRK/7/7/hittleD7lbw8rBgUZaKWBCGq19QO2ovPr49IeW7xgCPXm6AgLDxcznI29dT2fzpsNbaufWrfIuNp/fZqjhieSGV24Iaq1xGqsNXOKnL0I86lpuMF8f+CO8zys1sm78S5PRgjnxj4jH/8eYOG/MwHR6zEYPQI5SYBpTqijc6sWznEe2ZsfZ0tyt1s03QzNiysRpWQx+wFxeQpg2CounVxImhgUMBgD',
                                title='ABC News – Breaking News, Latest News and Videos',
                                type='web_search_result',
                                url='https://abcnews.go.com/',
                            ),
                            BetaWebSearchResultBlock(
                                encrypted_content='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',
                                page_age='2 weeks ago',
                                title='40+ of the best NYC events in August 2025, including US Open, music fests and free shows',
                                type='web_search_result',
                                url='https://www.timeout.com/newyork/events-calendar/august-events-calendar',
                            ),
                            BetaWebSearchResultBlock(
                                encrypted_content='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',
                                title='Holidays for August 14th, 2025 | Checkiday.com',
                                type='web_search_result',
                                url='https://www.checkiday.com/8/14/2025',
                            ),
                            BetaWebSearchResultBlock(
                                encrypted_content='EocbCioIBhgCIiQ0NGFlNjc2Yy05NThmLTRkNjgtOTEwOC1lYWU5ZGU3YjM2NmISDP806GDOcftTy8XY9BoM1XIRbFy2H63nh+agIjCG9AfOfUL0r8f2fD32zkgylGjHIuo00IOYPPZRtkH6ieyqr80pwg/pQ4J7eQLcOZgqihr5IWNvV844glwPSig3Gn7v/sbeId3UJdyapWaJ6NhWnTjxqsX8JspWKovRcQxHcpaAhJRVwsfssRtJKIVgoq4eTJH0ypTOYbFXIs78UKsTJLhyZxgSW/XeCgzsm1pcozX/gz+UWVmoXLZ+D1R3TTfFcKwUoiR+aF+6imG3mn2SZnuKadS3BEdXULL1Gi/NY3w750FqBhZA8WEByjAMLjqp55TdCXqE2YGHot8cL/mn+2dWdv3y4ZQ45FCb7fmwAu2V7Gh7C1YKdGDin6EX9rJf3kEvlfH115Yg3ulonmERfpnfLbsTqUNa2YNhJPOcJ6vbF5payIOQOuMT/LgJ1nt+RsGGK8jtRJ5iIg/fTmbaeobAGRfjaFOFZfVP0HaDzD26fikt6s2fHzCQBIJMqdwPfoZ8a+/G6+BOrfcCRZE4eIGt9dfvbPDhmlbCC3kmwLvQFv4YM6HbH317mSnLjD9TOfNhD/VvNDajuuKYG6pF4ZLkzl0S0dL/5RV5QYRf00kiCfuc6tvaLXoXccQWUSXmxFX9dYEtR1CK6L6yWeflL6OYKRGMPY5/4ewnffw72kTnmwdwyOQlKVmxWWwAYC8Ted+xxUkm4Strl02EeQ6M5EyDsQduCOsW3Ju+iONOEoM8j5CXX9GuBcM3t6uyFWji9SJDNKYT+6OAls1AnJy+0LuOiEMpi79z2DC8PZX4TjjDvlfgEoJ5k4I0A0SQpYFDnnYBLvDeC3kRzo1H/I9LYrxGBXcHqvIgDqkeQWq+R5uAXtszPgY6uG7vjrUaReeBW+Q2H7Gy2tik+JkHQPunh1t1aJcBmYZ70ezW30wPpl4PiQYUj2lHpIS8y5xQ0hIUSGPkqDQEYF1AcltXoaBbPG2Z8oLN9k26O4etttgc3JtOP6PbzLcveHJ1knY7RutbHnZIIrT8gCnT8qFawvHXODEpNbFidGRcsxmsV94yR2BsBuhg/Krjp6HCEviuvE5vErJHS1iukJCRIEa+Et2KyKPvBv7qgQ7VzBC4aR91+VURdHITreN5j/AazYI5f9DywI7EMr+fcIoywufbcAbQX2dyf19VGuIOv/5Eo9KXPAgkL3eeDrjoOLMEd2gbu0AvagHjb+LUYTmxK2G02F2yj384RgVf9tMuQ9nqOaNCeZN0/s0knbNt5siTKR0muPPcZC4Nek9ZsTYw9bQFZ0mUf01lciGGjZ7euCK+D+W4cnYj2pNqJZjW4eHmrcFrM6j//nfBMt/7wo9kZk403iynOqwvJYew2Fzh7KT+e1Q+LFV5GA99iAw2yveUIuxonkCj9iWLTElT/Fy7f9FHXUQc3P1nGwwjKUVVgOJK3ajMjA9pqHcBUM43xlvGMZax6omiy8l4Hdmtt8Fx1xtaBlid/Z4qKLUxWsRRcdxjb4voxZIrp7lCVZQdFhLj/xgJByzpCYDY+LHv3xhrcn3kAUqaKVPqWZpnXD1kPRayITDEmJJyFo50yVB44iBqeLElIfuqzoB9ovPq+xNxh5squxsK8tisJfzuJEo2nqNNHJg8ZJZu/i/kBrCc4hSWs8m3ZzgaOVFQtsLNRIOMnAgvZWnZiA8HnbOtkg6PFeETr3P7MnKr/oGmAtsJ1K8gdTTngCrg1YdEQoCHftuTrB6s00wQYlljU6fj/kKKBcKanWAYArqhv146R1i0EJrSr9zOSktkKEZyhglx8pkbxjjrGQVTayjLKwuotgpT77vkptvALa++uBSTgLiwGQTMYYJBZUV6vV5WyVR1Iy0dUjePKBxt32wR+V0uS+uwhSEyTFJddLqTYCkkMMmdd6bq0BpOKKsid4vYD6z55G4+LMxJrr5J749bQRmnPPdu5tnB+GgQM6NxRD+Q2084udhDmB7PGihiR13qlH8wT1fo9GBN7NxRNb1kkLlM8TIwaMeYpVhkVaeBc1UEkDGoQnaosx7vnd1GL8pCkjQ8DLC+aiNxcgi/ZiK3UaGa65olpWdXpoZVmdR65UccRa3eEP3/16bv4vPdJZSFIDJq76g9edstmteoUsTdpgkzSVjvwwo+1gc4GFuuaZJtGtuWVYZu/Dao6k+WKpVo2rnf6aO5MLiJjP6Dw68aKMwvL9bhDKLI6XWYOcdpNa67q8oNqvKaX6rp4qKHsBdBNb+rSe6on4NHVeTINdU6K/fGx8mHQtU7tlSuBNfdj5ev0sYcc1+4ybjve49Oh/MoKvodmOPvlB31NrONb5B0VWk4F8TQ/3xpNHSAkeJnWoLQtbFrtazQBteYdBcE1aF/7iefOp2tJpUH0gjSrjFZkgDFkQHRbd3Zfu72xWqjZporPbQAkD8iM7485VWoDM7CLrXeVOPbMzJdsh4bChXy3282gbPcSqKKHI6Zwukyuz7xANFNuLWIyYugkpZgNhk5Waz6lZC2ou0i2hKVvDQr8jsTXUofXWDylgC71A2d6QX0IBY26kIT2kJ2R0ovZpNBPL1ZdINRvAfccWbpK4rEI+L92Pmb2N0GQSsbfzgOEEfeGYW9YdBAliQEzrIclByQ1juIuVPuHih3jd1e5Jo32wSqbsr7kKKjBGiNeSkEBwWBaZTJZq6808tyyuvIKVCNknDNFtV+yY5Yua7JPglVc4KQ8cz9baJEJ2Cqci5BXwokrpGSOSwAsJs6AYdH3xD4pnNfG4zzr6aEQ6Qm9L2AntKfvPkRg5g6tVWxSnFD68yCQldF/eIeIV+pnTV49jeO0xoCy2AVi2sbZ+MC5u30N49D/4Hg3szrC/bNLydAzw+qhtc4YByt6fYIXXHvz/Ks9BE54xKTqHJQH/4Pf0y5SA20TUftqM1Cd/5oeytjPm/nszaOAh1p9QbEV4IbaUIBoA7yX0C9gFZ8ubZifBALtGIB8k3SJDzdVzWB//M0LwO0ySTiIleeL5xwef8zyUyHl1nTLYE8+ke/JtKR1M5bWdW8S0nVcaO/HjzeJwKnleB/PC+0Gbf0se3n0VzzR3TRW/d7XLFCW5oD8Siyt1FHWdS3T0gt6uzrqYxoD6QQCKtKuPtePVKkLRBKyLQQErihb7wq3FN874Q2NONFKXiaBsyxvmFuYKCY4g/4Z8egIomR6wyLaTsnfrBIvYCatmm2jgWon6JUtttDBu/B5YNUCjkRJH32UI4gdLJh5cxd10kr2xcs6dkIx/jBOkSRu7IZWJsAMSjv5KuLf1fC/zuy3Wwon+7bfA+OCOKsyfwOPr73AWWQoX900CzTXVjlsB6zAeQinKBKixIXq+mRZ8qrg3TcA+bh5TqHlXV0gNxaTBrH7JqwRTBz13bxsp/IVu9Ns1jUXvDmmu9v6ucfIfT4oV70mPWjZSZo8PIpKx2xjEEqJmNOPP9boAsZnm5U4AFdFM6/x9qDIrfuhGEUi6cvMQV3eVoiMpec+IRLp2FyEQ0e//ixZ9cvFXNK2zl4xxBgrUqeb6CZ8Zhfb0iDWr0gXpdGNbF2n+Da8qPNT+9A9XpY5/YNZKhrZZRKefaEarn9Nh1ilyqnSUK5LQcJYY0kFM8ytMICE9hun3xCA+rWuj+Tu7Ji2CRANRmauEKePGUZVLNyLQAwxpOJVUxDEWUeKUypPfH7psq2BCkdL+S2iKM9IckK98GqCIL0gvfocrkJO3ngQWjgC+nmYYTC55ZRqXaPAkpBmr3qQmT7VOEATE5zAvOKBb5oI3HAbzQa/g1HGepwsErnLTaoUQ2Lo5IPuZykCVoiMSn/oa5ulHfhJpYK84+fz0ewe9p06TIdmlL10ABVZVryKrRFmHfu+fRgfqV84ZcA/2mArmRCW8Tt5y+8unYkMaap7qdmMir+r0j4WZoL7IqoniRehC+5uzIxyy3kqZ/BBjfvyK5OYkb12ShjpKsbI+oNxX0bSQvZYG8Ef4jmGgs4fIok+hb0Nlkwd3JVdqS4gri0YC2encvbxUPl1/FuH8+9SCd6U34x/Ns8XG66DMqwfO9AJeSLQeobMH66G0FkVjSc+BfQrt97VBbpgcjCSdIXouF+Yb5K8oA2MFE5kdOmeMEB30YACyBWahyWSeQv54xaAKvGuI5zPrEv+TSKjaA2V5fmWYk8Yg6J+SWZpRDDZq87eYdWAYMh9hqRPt5flt+FCo/YompUnbMqmhkwRm169VUPRy9qNOliDMsLD+dqTKWGwdCUZS42KWXH4lepjIEUI9zKF+gHuOom1hvwvzbYQMLg//6Zjxr+14toaT9rMqNtT3pKgCBWm875JQNNX8qz7nkINCiW+g3Ek7roPbClHix96g+5jiiHNHShDOvB3BwdIXOaNMXPGh02VWf+DqVABznHlqF2er3bKlQ3zsG3ORcxkP5KxGvxKSkHL6+ogYFgfRojSoc5xkLjmT+dmsrHPepi9Ptm3HpU9wa+n/Fa3GchAi75qoyrF+5mS6JZ3mRJoNPK7wA1hCaNG6xEdazmsiCMCI/1QgCb/NKxQ9/MucIOIdrdWXaL+2mBrBZz4hgD',
                                page_age='5 hours ago',
                                title='Current Affairs 14 August 2025',
                                type='web_search_result',
                                url='https://ssbcrackexams.com/current-affairs-14-august-2025/',
                            ),
                            BetaWebSearchResultBlock(
                                encrypted_content='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',
                                title='Monopoly Go Events Today Schedule for August 14, 2025',
                                type='web_search_result',
                                url='https://monopolygo.game/monopoly-go-events-today-schedule',
                            ),
                        ],
                        tool_call_id='srvtoolu_016Z2mQT8AFaH17TpZnmuj2Z',
                        timestamp=IsDatetime(),
                        provider_name='anthropic',
                    ),
                    TextPart(
                        content="""\
Based on the search results, today is Thursday, August 14, 2025. This is confirmed by multiple sources:

"""
                    ),
                    TextPart(content='Today is August 14, 2025 (Thursday)'),
                    TextPart(
                        content="""\


Several major events are happening today, including:

"""
                    ),
                    TextPart(
                        content='There is an ongoing situation in Gaza with Israeli forces launching a massive aerial bombardment of Gaza City'
                    ),
                    TextPart(
                        content="""\


"""
                    ),
                    TextPart(
                        content='U.S. President Donald Trump has announced he will be meeting Russian President Vladimir Putin in Alaska tomorrow (August 15) to discuss ending the war in Ukraine'
                    ),
                    TextPart(
                        content="""\


"""
                    ),
                    TextPart(
                        content="Mount Lewotobi Laki Laki in Indonesia is experiencing its second consecutive day of eruption, sending volcanic materials and ash up to 18 km into the sky. This is one of Indonesia's largest eruptions since 2010, though fortunately no casualties have been reported."
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=14923,
                    output_tokens=317,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 14923,
                        'output_tokens': 317,
                    },
                ),
                model_name='claude-3-5-sonnet-20241022',
                timestamp=IsDatetime(),
                provider_request_id='msg_01W2YfD2EF8BbAqLRr8ftH4W',
            ),
        ]
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
                    TextPart(content='3 * 12390 = 37,170'),
                ],
                usage=RequestUsage(
                    input_tokens=1630,
                    output_tokens=109,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 1630,
                        'output_tokens': 109,
                    },
                ),
                model_name='claude-sonnet-4-20250514',
                timestamp=IsDatetime(),
                provider_request_id='msg_01RJnbK7VMxvS2SyvtyJAQVU',
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
Let me search to find today's date.

Based on the search results, today is Thursday, August 14, 2025. Here are some additional details about the date:



It is the 226th day of the year 2025 in the Gregorian calendar, with 139 days remaining until the end of the year

.

Some interesting observances for today include:


It's being celebrated as:
- Color Book Day
- National Creamsicle Day
- National Financial Awareness Day
- National Navajo Code Talkers Day
- National Tattoo Removal Day
- National Wiffle Ball Day
- Social Security Day\
""")
    result = await agent.run('What day is tomorrow?', model=openai_model, message_history=result.all_messages())
    assert result.new_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='What day is tomorrow?', timestamp=IsDatetime())]),
            ModelResponse(
                parts=[TextPart(content='Tomorrow will be **Friday, August 15, 2025**.')],
                usage=RequestUsage(input_tokens=458, output_tokens=17, details={'reasoning_tokens': 0}),
                model_name='gpt-4.1-2025-04-14',
                timestamp=IsDatetime(),
                provider_request_id='resp_689dc4abe31c81968ed493d15d8810fe0afe80ec3d42722e',
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
                model_name='claude-3-5-sonnet-20241022',
                timestamp=IsDatetime(),
                provider_request_id='msg_012TXW181edhmR5JCsQRsBKx',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id='toolu_01X9wcHKKAZD9tBC711xipPa',
                        timestamp=IsDatetime(),
                    )
                ]
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
                model_name='claude-3-5-sonnet-20241022',
                timestamp=IsDatetime(),
                provider_request_id='msg_01K4Fzcf1bhiyLzHpwLdrefj',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='toolu_01LZABsgreMefH2Go8D5PQbW',
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
                ]
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
                model_name='claude-3-5-sonnet-20241022',
                timestamp=IsDatetime(),
                provider_request_id='msg_01MsqUB7ZyhjGkvepS1tCXp3',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id='toolu_01JJ8TequDsrEU2pv1QFRWAK',
                        timestamp=IsDatetime(),
                    )
                ]
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
                model_name='claude-3-5-sonnet-20241022',
                timestamp=IsDatetime(),
                provider_request_id='msg_0142umg4diSckrDtV9vAmmPL',
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
                model_name='claude-3-5-sonnet-20241022',
                timestamp=IsDatetime(),
                provider_request_id='msg_018YiNXULHGpoKoHkTt6GivG',
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
                instructions="""\
Always respond with a JSON object that's compatible with this schema:

{"properties": {"city": {"type": "string"}, "country": {"type": "string"}}, "required": ["city", "country"], "title": "CityLocation", "type": "object"}

Don't include any text or Markdown fencing before or after.\
""",
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
                model_name='claude-3-5-sonnet-20241022',
                timestamp=IsDatetime(),
                provider_request_id='msg_01WiRVmLhCrJbJZRqmAWKv3X',
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
                model_name='claude-3-5-sonnet-20241022',
                timestamp=IsDatetime(),
                provider_request_id='msg_01N2PwwVQo2aBtt6UFhMDtEX',
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

Mexico City is not only the largest city in Mexico but also one of the largest metropolitan areas in the world. The city proper has a population of approximately 9.2 million people, while the greater Mexico City metropolitan area has over 21 million inhabitants, making it the most populous metropolitan area in North America.

Mexico City serves as the country's capital and is the political, economic, and cultural center of Mexico.\
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
        PartStartEvent(index=0, part=TextPart(content='Let me search for more specific breaking'))
    )
    assert event_parts.pop(0) == snapshot(FinalResultEvent(tool_name=None, tool_call_id=None))
    assert ''.join(event.delta.content_delta for event in event_parts) == snapshot("""\
 news stories to get clearer headlines.Based on the search results, I can identify the top 3 major news stories from around the world today (August 14, 2025):

## Top 3 World News Stories Today

**1. Trump-Putin Summit and Ukraine Crisis**
European leaders held a high-stakes meeting Wednesday with President Trump, Vice President Vance, Ukraine's Volodymyr Zelenskyy and NATO's chief ahead of Friday's U.S.-Russia summit. The White House lowered its expectations surrounding the Trump-Putin summit on Friday. In a surprise move just days before the Trump-Putin summit, the White House swapped out pro-EU PM Tusk for Poland's new president – a political ally who once opposed Ukraine's NATO and EU bids.

**2. Trump's Federal Takeover of Washington D.C.**
Federal law enforcement's presence in Washington, DC, continued to be felt Wednesday as President Donald Trump's takeover of the city's police entered its third night. National Guard troops arrived in Washington, D.C., following President Trump's deployment and federalization of local police to crack down on crime in the nation's capital. Over 100 arrests made as National Guard rolls into DC under Trump's federal takeover.

**3. Air Canada Flight Disruption**
Air Canada plans to lock out its flight attendants and cancel all flights starting this weekend. Air Canada says it will begin cancelling flights starting Thursday to allow an orderly shutdown of operations with a complete cessation of flights for the country's largest airline by Saturday as it faces a potential work stoppage by its flight attendants.

These stories represent major international diplomatic developments, significant domestic policy changes in the US, and major transportation disruptions affecting North America.\
""")
