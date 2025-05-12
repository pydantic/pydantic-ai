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

from pydantic_ai import Agent, ModelHTTPError, ModelRetry
from pydantic_ai.messages import (
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.result import Usage
from pydantic_ai.settings import ModelSettings

from ..conftest import IsDatetime, IsNow, IsStr, TestEnv, raise_if_exception, try_import
from .mock_async_stream import MockAsyncStream

with try_import() as imports_successful:
    from anthropic import NOT_GIVEN, APIStatusError, AsyncAnthropic
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

    from pydantic_ai.models.anthropic import (
        AnthropicModel,
        AnthropicModelSettings,
        _map_usage,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.providers.anthropic import AnthropicProvider

    # note: we use Union here so that casting works with Python 3.9
    MockAnthropicMessage = Union[AnthropicMessage, Exception]
    MockRawMessageStreamEvent = Union[RawMessageStreamEvent, Exception]

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    pytest.mark.anyio,
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
    ) -> AnthropicMessage | MockAsyncStream[MockRawMessageStreamEvent]:
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
                response = cast(AnthropicMessage, self.messages_[self.index])
            else:
                raise_if_exception(self.messages_)
                response = cast(AnthropicMessage, self.messages_)
        self.index += 1
        return response


def completion_message(content: list[ContentBlock], usage: AnthropicUsage) -> AnthropicMessage:
    return AnthropicMessage(
        id='123',
        content=content,
        model='claude-3-5-haiku-123',
        role='assistant',
        stop_reason='end_turn',
        type='message',
        usage=usage,
    )


async def test_sync_request_text_response(allow_model_requests: None):
    c = completion_message([TextBlock(text='world', type='text')], AnthropicUsage(input_tokens=5, output_tokens=10))
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
            ),
        ]
    )


async def test_async_request_prompt_caching(allow_model_requests: None):
    c = completion_message(
        [TextBlock(text='world', type='text')],
        usage=AnthropicUsage(
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
        [TextBlock(text='world', type='text')],
        usage=AnthropicUsage(input_tokens=3, output_tokens=5),
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
        [ToolUseBlock(id='123', input={'response': [1, 2, 3]}, name='final_result', type='tool_use')],
        usage=AnthropicUsage(input_tokens=3, output_tokens=5),
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
            [ToolUseBlock(id='1', input={'loc_name': 'San Francisco'}, name='get_location', type='tool_use')],
            usage=AnthropicUsage(input_tokens=2, output_tokens=1),
        ),
        completion_message(
            [TextBlock(text='final response', type='text')],
            usage=AnthropicUsage(input_tokens=3, output_tokens=5),
        ),
    ]

    mock_client = MockAnthropic.create_mock(responses)
    m = AnthropicModel('claude-3-5-haiku-latest', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(m, model_settings=ModelSettings(parallel_tool_calls=parallel_tool_calls))

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        if loc_name == 'London':
            return json.dumps({'lat': 51, 'lng': 0})
        else:
            raise ModelRetry('Wrong location, please try again')

    await agent.run('hello')
    assert get_mock_chat_completion_kwargs(mock_client)[0]['tool_choice']['disable_parallel_tool_use'] == (
        not parallel_tool_calls
    )


@pytest.mark.vcr
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
    assert first_response.parts == [
        TextPart(
            content="I'll retrieve the information about each family member to determine their ages.",
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
    assert second_request.parts == [
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

    # Ensure the tool call IDs match between the tool calls and the tool returns
    tool_call_part_ids = [part.tool_call_id for part in first_response.parts if part.part_kind == 'tool-call']
    tool_return_part_ids = [part.tool_call_id for part in second_request.parts if part.part_kind == 'tool-return']
    assert len(set(tool_call_part_ids)) == 4  # ensure they are all unique
    assert tool_call_part_ids == tool_return_part_ids


async def test_anthropic_specific_metadata(allow_model_requests: None) -> None:
    c = completion_message([TextBlock(text='world', type='text')], AnthropicUsage(input_tokens=5, output_tokens=10))
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

    done_stream = [
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


@pytest.mark.vcr()
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
        "This is a potato. It's a yellow-brown, oblong-shaped potato with a smooth skin and some small eyes or blemishes visible on its surface. Potatoes are starchy root vegetables that are a staple food in many cuisines around the world. They can be prepared in numerous ways, such as boiling, baking, frying, or mashing, and are rich in carbohydrates and nutrients."
    )


@pytest.mark.vcr()
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


@pytest.mark.vcr()
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
        'This is a Great Horned Owl (Bubo virginianus), a large and powerful owl species. It has distinctive ear tufts (the "horns"), large yellow eyes, and a mottled gray-brown plumage that provides excellent camouflage. In this image, the owl is perched on a branch, surrounded by soft yellow and green vegetation, which creates a beautiful, slightly blurred background that highlights the owl\'s sharp features. Great Horned Owls are known for their adaptability, wide distribution across the Americas, and their status as powerful nocturnal predators.'
    )


@pytest.mark.vcr()
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
                    ToolCallPart(tool_name='get_image', args={}, tool_call_id='toolu_01VMGXdexE1Fy5xdWgoom9Te'),
                ],
                usage=Usage(
                    requests=1,
                    request_tokens=372,
                    response_tokens=49,
                    total_tokens=421,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 372,
                        'output_tokens': 49,
                    },
                ),
                model_name='claude-3-5-sonnet-20241022',
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_image',
                        content='See file 1c8566',
                        tool_call_id='toolu_01VMGXdexE1Fy5xdWgoom9Te',
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
                        content="The image shows a kiwi fruit that has been cut in half, displaying its characteristic bright green flesh with small black seeds arranged in a circular pattern around a white center core. The kiwi's fuzzy brown skin is visible around the edges of the slice."
                    )
                ],
                usage=Usage(
                    requests=1,
                    request_tokens=2025,
                    response_tokens=57,
                    total_tokens=2082,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 2025,
                        'output_tokens': 57,
                    },
                ),
                model_name='claude-3-5-sonnet-20241022',
                timestamp=IsDatetime(),
            ),
        ]
    )


@pytest.mark.parametrize('media_type', ('audio/wav', 'audio/mpeg'))
async def test_audio_as_binary_content_input(allow_model_requests: None, media_type: str):
    c = completion_message([TextBlock(text='world', type='text')], AnthropicUsage(input_tokens=5, output_tokens=10))
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


@pytest.mark.vcr()
async def test_document_binary_content_input(
    allow_model_requests: None, anthropic_api_key: str, document_content: BinaryContent
):
    m = AnthropicModel('claude-3-5-sonnet-latest', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m)

    result = await agent.run(['What is the main content on this document?', document_content])
    assert result.output == snapshot(
        'The document appears to be a simple PDF file with only the text "Dummy PDF file" displayed at the top. It appears to be mostly blank otherwise, likely serving as a template or placeholder document.'
    )


@pytest.mark.vcr()
async def test_document_url_input(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-3-5-sonnet-latest', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m)

    document_url = DocumentUrl(url='https://pdfobject.com/pdf/sample.pdf')

    result = await agent.run(['What is the main content on this document?', document_url])
    assert result.output == snapshot(
        'This document appears to be a sample PDF file that primarily contains Lorem ipsum text, which is placeholder text commonly used in design and publishing. The document begins with "Sample PDF" and states "This is a simple PDF file. Fun fun fun." followed by several paragraphs of Lorem ipsum text. The content doesn\'t convey any meaningful information as Lorem ipsum is essentially dummy text used to demonstrate the visual form of a document without the distraction of meaningful content.'
    )


@pytest.mark.vcr()
async def test_text_document_url_input(allow_model_requests: None, anthropic_api_key: str):
    m = AnthropicModel('claude-3-5-sonnet-latest', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(m)

    text_document_url = DocumentUrl(url='https://example-files.online-convert.com/document/txt/example.txt')

    result = await agent.run(['What is the main content on this document?', text_document_url])
    assert result.output == snapshot("""\
This document is a TXT test file that primarily contains information about the use of placeholder names, specifically focusing on "John Doe" and its variants. The main content explains how these placeholder names are used in legal contexts and popular culture, particularly in English-speaking countries. The text describes:

1. The various placeholder names used:
- "John Doe" for males
- "Jane Doe" or "Jane Roe" for females
- "Jonnie Doe" and "Janie Doe" for children
- "Baby Doe" for unknown children

2. The usage of these names in different English-speaking countries, noting that while common in the US and Canada, they're less used in the UK, where "Joe Bloggs" or "John Smith" are preferred.

3. How these names are used in legal contexts, forms, and popular culture.

The document is formatted as a test file with metadata including its purpose, file type, and version. It also includes attribution information indicating the content is from Wikipedia and is licensed under Attribution-ShareAlike 4.0.\
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


@pytest.mark.vcr()
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
            ),
        ]
    )


def anth_msg(usage: AnthropicUsage) -> AnthropicMessage:
    return AnthropicMessage(
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
            lambda: anth_msg(AnthropicUsage(input_tokens=1, output_tokens=1)),
            snapshot(
                Usage(
                    request_tokens=1, response_tokens=1, total_tokens=2, details={'input_tokens': 1, 'output_tokens': 1}
                )
            ),
            id='AnthropicMessage',
        ),
        pytest.param(
            lambda: anth_msg(
                AnthropicUsage(
                    input_tokens=1, output_tokens=1, cache_creation_input_tokens=2, cache_read_input_tokens=3
                )
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
            lambda: RawMessageStartEvent(
                message=anth_msg(AnthropicUsage(input_tokens=1, output_tokens=1)), type='message_start'
            ),
            snapshot(
                Usage(
                    request_tokens=1, response_tokens=1, total_tokens=2, details={'input_tokens': 1, 'output_tokens': 1}
                )
            ),
            id='RawMessageStartEvent',
        ),
        pytest.param(
            lambda: RawMessageDeltaEvent(
                delta=Delta(),
                usage=MessageDeltaUsage(output_tokens=5),
                type='message_delta',
            ),
            snapshot(Usage(response_tokens=5, total_tokens=5, details={'output_tokens': 5})),
            id='RawMessageDeltaEvent',
        ),
        pytest.param(lambda: RawMessageStopEvent(type='message_stop'), snapshot(Usage()), id='RawMessageStopEvent'),
    ],
)
def test_usage(message_callback: Callable[[], AnthropicMessage | RawMessageStreamEvent], usage: Usage):
    assert _map_usage(message_callback()) == usage
