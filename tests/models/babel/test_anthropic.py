from __future__ import annotations as _annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, cast

import httpx2
import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, ModelRequest, ModelResponse, TextPart, ThinkingPart, ToolCallPart
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import FinalResultEvent, InstructionPart, PartDeltaEvent, PartEndEvent, PartStartEvent
from pydantic_ai.models import ModelRequestParameters

from ...conftest import try_import
from ..mock_async_stream import MockAsyncStream

with try_import() as imports_successful:
    from anthropic import NOT_GIVEN, APIStatusError, AsyncAnthropic
    from anthropic.types.beta import BetaMessage, BetaRawMessageStartEvent, BetaRawMessageStreamEvent
    from pydantic import TypeAdapter

    from pydantic_ai.models.anthropic import AnthropicModelSettings
    from pydantic_ai.models.babel.anthropic import (
        BabelAnthropicModel,
        BabelAnthropicStreamedResponse,
        _cache_instructions_index,  # pyright: ignore[reportPrivateUsage]
        _pack_system,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.providers.anthropic import AnthropicProvider

    EVENT_ADAPTER: TypeAdapter[BetaRawMessageStreamEvent] = TypeAdapter(BetaRawMessageStreamEvent)

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic or llm-babel not installed'),
    pytest.mark.anyio,
]


@dataclass
class MockAnthropic:
    messages_: Sequence[BetaMessage] = ()
    stream: Sequence[Any] = ()
    index: int = 0
    kwargs: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    base_url: str = 'https://api.anthropic.com'

    @cached_property
    def beta(self) -> Any:
        return self

    @cached_property
    def messages(self) -> Any:
        return type('Messages', (), {'create': self.create})

    @classmethod
    def as_client(cls, messages_: Sequence[BetaMessage] = (), stream: Sequence[Any] = ()) -> AsyncAnthropic:
        return cast(AsyncAnthropic, cls(messages_=messages_, stream=stream))

    async def create(self, *_args: Any, stream: bool = False, **kwargs: Any) -> Any:
        self.kwargs.append({k: v for k, v in kwargs.items() if v is not NOT_GIVEN})
        if stream:
            return MockAsyncStream(iter(self.stream))
        message = self.messages_[self.index]
        self.index += 1
        return message


def message(content: list[dict[str, Any]], stop_reason: str = 'end_turn', **extra: Any) -> BetaMessage:
    return BetaMessage.model_validate(
        {
            'id': 'msg_1',
            'type': 'message',
            'role': 'assistant',
            'model': 'claude-sonnet-4-5',
            'content': content,
            'stop_reason': stop_reason,
            'stop_sequence': None,
            'usage': {
                'input_tokens': 10,
                'output_tokens': 4,
                'cache_read_input_tokens': 5,
                'cache_creation_input_tokens': 0,
            },
            **extra,
        }
    )


CONTAINER = {'id': 'container_1', 'expires_at': '2026-01-01T00:00:00Z'}
THINKING_DROPPED = {'type': 'thinking_dropped', 'path': 'messages.1.content.0', 'reason': 'model_binding_mismatch'}


def make_model(client: AsyncAnthropic) -> BabelAnthropicModel:
    return BabelAnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=client))


async def test_tool_loop_with_thinking(allow_model_requests: None):
    client = MockAnthropic.as_client(
        [
            message(
                [
                    {'type': 'thinking', 'thinking': 'Paris needs a lookup.', 'signature': 'SIG'},
                    {'type': 'tool_use', 'id': 'toolu_1', 'name': 'get_weather', 'input': {'city': 'Paris'}},
                ],
                'tool_use',
            ),
            message([{'type': 'text', 'text': 'It is sunny in Paris.'}]),
        ]
    )
    agent = Agent(make_model(client), system_prompt='You are a weather assistant.', instructions='Be terse.')

    @agent.tool_plain
    def get_weather(city: str) -> str:
        return f'{city}: sunny'

    result = await agent.run('What is the weather in Paris?')
    assert result.output == 'It is sunny in Paris.'
    response = result.all_messages()[1]
    assert isinstance(response, ModelResponse)
    assert response.parts == snapshot(
        [
            ThinkingPart(content='Paris needs a lookup.', signature='SIG', provider_name='anthropic'),
            ToolCallPart(tool_name='get_weather', args={'city': 'Paris'}, tool_call_id='toolu_1'),
        ]
    )
    # The cache reads count towards the input tokens, as they do for the native model.
    assert (response.usage.input_tokens, response.usage.cache_read_tokens, response.usage.output_tokens) == (15, 5, 4)
    assert response.model_name == 'claude-sonnet-4-5'
    assert response.provider_name == 'anthropic'
    assert response.provider_url == 'https://api.anthropic.com'
    assert response.provider_response_id == 'msg_1'
    assert response.finish_reason == 'tool_call'
    second_request = cast(MockAnthropic, client).kwargs[1]
    assert second_request['system'] == snapshot(
        [{'type': 'text', 'text': 'You are a weather assistant.'}, {'type': 'text', 'text': 'Be terse.'}]
    )
    # The thinking block replays with its signature, since it came from this provider.
    assert second_request['messages'] == snapshot(
        [
            {'role': 'user', 'content': [{'type': 'text', 'text': 'What is the weather in Paris?'}]},
            {
                'role': 'assistant',
                'content': [
                    {'type': 'thinking', 'thinking': 'Paris needs a lookup.', 'signature': 'SIG'},
                    {'type': 'tool_use', 'id': 'toolu_1', 'name': 'get_weather', 'input': {'city': 'Paris'}},
                ],
            },
            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'toolu_1', 'content': 'Paris: sunny'}]},
        ]
    )


async def test_response_details_the_next_request_depends_on(allow_model_requests: None):
    client = MockAnthropic.as_client(
        [
            message(
                [{'type': 'text', 'text': 'paused'}],
                'pause_turn',
                container=CONTAINER,
                input_transformations=[THINKING_DROPPED],
                stop_details={'type': 'refusal', 'explanation': 'no', 'category': 'general_harms'},
            )
        ]
    )
    response = await make_model(client).request([ModelRequest.user_text_prompt('hi')], None, ModelRequestParameters())
    # A paused server-side turn is suspended so the agent reissues it, with the container to reconnect to.
    assert response.state == 'suspended'
    assert response.provider_details == snapshot(
        {
            'finish_reason': 'pause_turn',
            'refusal': 'no',
            'refusal_category': 'general_harms',
            'container_id': 'container_1',
            'input_transformations': [THINKING_DROPPED],
        }
    )


async def test_plain_system_prompt_is_a_string(allow_model_requests: None):
    client = MockAnthropic.as_client([message([{'type': 'text', 'text': 'hi'}])])
    agent = Agent(make_model(client), system_prompt='be nice')
    await agent.run('hello')
    assert cast(MockAnthropic, client).kwargs[0]['system'] == 'be nice'


async def test_cache_instructions(allow_model_requests: None):
    client = MockAnthropic.as_client([message([{'type': 'text', 'text': 'hi'}])])
    agent = Agent(make_model(client), system_prompt='be nice', instructions='terse')
    await agent.run('hello', model_settings=AnthropicModelSettings(anthropic_cache_instructions='1h'))
    assert cast(MockAnthropic, client).kwargs[0]['system'] == snapshot(
        [
            {'type': 'text', 'text': 'be nice'},
            {'type': 'text', 'text': 'terse', 'cache_control': {'type': 'ephemeral', 'ttl': '1h'}},
        ]
    )


def test_pack_system():
    static = InstructionPart(content='static')
    dynamic = InstructionPart(content='dynamic', dynamic=True)
    prompt = [{'type': 'text', 'text': 'prompt'}]
    assert _pack_system(prompt, [], None) == 'prompt'
    assert _pack_system([], [], True) == []
    assert _pack_system(prompt, [], True) == snapshot(
        [{'type': 'text', 'text': 'prompt', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}}]
    )
    assert _pack_system([*prompt, {'text': 'static'}, {'text': 'dynamic'}], [static, dynamic], '1h') == snapshot(
        [
            {'type': 'text', 'text': 'prompt'},
            {'type': 'text', 'text': 'static', 'cache_control': {'type': 'ephemeral', 'ttl': '1h'}},
            {'type': 'text', 'text': 'dynamic'},
        ]
    )
    assert _pack_system([{'text': 'dynamic'}], [dynamic], True) == [{'type': 'text', 'text': 'dynamic'}]


def test_cache_instructions_index():
    static = InstructionPart(content='s')
    dynamic = InstructionPart(content='d', dynamic=True)
    assert _cache_instructions_index(True, 1, []) == 0
    assert _cache_instructions_index(False, 0, []) is None
    assert _cache_instructions_index(True, 3, [static, static]) == 2
    assert _cache_instructions_index(True, 3, [static, dynamic]) == 1
    assert _cache_instructions_index(False, 2, [static, dynamic]) == 0
    assert _cache_instructions_index(True, 2, [dynamic]) == 0
    assert _cache_instructions_index(False, 1, [dynamic]) is None


def stream_events() -> list[BetaRawMessageStreamEvent]:
    raw: list[dict[str, Any]] = [
        {
            'type': 'message_start',
            'message': {
                'id': 'msg_1',
                'type': 'message',
                'role': 'assistant',
                'model': 'claude-sonnet-4-5',
                'content': [],
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {
                    'input_tokens': 10,
                    'output_tokens': 0,
                    'cache_read_input_tokens': 5,
                    'cache_creation_input_tokens': 0,
                },
            },
        },
        {
            'type': 'content_block_start',
            'index': 0,
            'content_block': {'type': 'thinking', 'thinking': '', 'signature': ''},
        },
        {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'thinking_delta', 'thinking': 'hmm'}},
        {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'signature_delta', 'signature': 'SIG'}},
        {'type': 'content_block_stop', 'index': 0},
        {'type': 'content_block_start', 'index': 1, 'content_block': {'type': 'text', 'text': ''}},
        {'type': 'content_block_delta', 'index': 1, 'delta': {'type': 'text_delta', 'text': 'It is '}},
        {'type': 'content_block_delta', 'index': 1, 'delta': {'type': 'text_delta', 'text': 'sunny.'}},
        {'type': 'content_block_stop', 'index': 1},
        {
            'type': 'content_block_start',
            'index': 2,
            'content_block': {'type': 'tool_use', 'id': 'toolu_1', 'name': 'f', 'input': {}},
        },
        {'type': 'content_block_delta', 'index': 2, 'delta': {'type': 'input_json_delta', 'partial_json': '{"a": '}},
        {'type': 'content_block_delta', 'index': 2, 'delta': {'type': 'input_json_delta', 'partial_json': '1}'}},
        {'type': 'content_block_stop', 'index': 2},
        {
            'type': 'message_delta',
            'delta': {'stop_reason': 'tool_use', 'stop_sequence': None},
            'usage': {'output_tokens': 7},
        },
        {'type': 'message_stop'},
    ]
    return [EVENT_ADAPTER.validate_python(event) for event in raw]


async def test_stream(allow_model_requests: None):
    # A Bedrock-only chunk arrives as a start event with no message and must be skipped.
    bedrock_chunk = BetaRawMessageStartEvent.model_construct(type='message_start', message=None)
    client = MockAnthropic.as_client(stream=[*stream_events(), bedrock_chunk])
    model = make_model(client)
    async with model.request_stream([ModelRequest.user_text_prompt('hi')], None, ModelRequestParameters()) as response:
        assert isinstance(response, BabelAnthropicStreamedResponse)
        events = [event async for event in response]
    assert [type(event) for event in events] == snapshot(
        [
            PartStartEvent,
            PartDeltaEvent,
            PartEndEvent,
            PartStartEvent,
            FinalResultEvent,
            PartDeltaEvent,
            PartEndEvent,
            PartStartEvent,
            PartDeltaEvent,
            PartEndEvent,
        ]
    )
    assert response.get().parts == snapshot(
        [
            ThinkingPart(content='hmm', signature='SIG', provider_name='anthropic'),
            TextPart(content='It is sunny.'),
            ToolCallPart(tool_name='f', args='{"a":1}', tool_call_id='toolu_1'),
        ]
    )
    assert response.finish_reason == 'tool_call'
    assert response.provider_response_id == 'msg_1'
    assert (response.usage.input_tokens, response.usage.cache_read_tokens, response.usage.output_tokens) == (15, 5, 7)
    assert response.provider_details == {'finish_reason': 'tool_use'}
    assert response.state == 'complete'


async def test_stream_records_the_details_the_next_request_depends_on(allow_model_requests: None):
    start, *_rest = stream_events()
    raw_start = start.model_dump()
    raw_start['message']['container'] = CONTAINER
    # A delta may carry usage alone, before the one that stops the message.
    usage_only = {
        'type': 'message_delta',
        'delta': {'stop_reason': None, 'stop_sequence': None},
        'usage': {'output_tokens': 1},
    }
    delta = {
        'type': 'message_delta',
        'delta': {
            'stop_reason': 'pause_turn',
            'stop_sequence': None,
            'stop_details': {'type': 'refusal'},
            'container': {**CONTAINER, 'id': 'container_2'},
        },
        'usage': {'output_tokens': 3},
        'input_transformations': [THINKING_DROPPED],
    }
    events = [EVENT_ADAPTER.validate_python(raw) for raw in (raw_start, usage_only, delta)]
    model = make_model(MockAnthropic.as_client(stream=events))
    async with model.request_stream([ModelRequest.user_text_prompt('hi')], None, ModelRequestParameters()) as response:
        _ = [event async for event in response]
    assert response.state == 'suspended'
    # The delta's container supersedes the one announced at the start, as it does in the native model.
    assert response.provider_details == snapshot(
        {'container_id': 'container_2', 'finish_reason': 'pause_turn', 'input_transformations': [THINKING_DROPPED]}
    )


async def test_stream_api_error_is_mapped(allow_model_requests: None):
    error = APIStatusError(
        'boom',
        response=httpx2.Response(status_code=529, request=httpx2.Request('POST', 'https://example.com/v1')),
        body={'error': 'overloaded'},
    )
    model = make_model(MockAnthropic.as_client(stream=[*stream_events()[:3], error]))
    with pytest.raises(ModelHTTPError) as exc_info:
        async with model.request_stream(
            [ModelRequest.user_text_prompt('hi')], None, ModelRequestParameters()
        ) as response:
            _ = [event async for event in response]
    assert exc_info.value.status_code == 529
    assert exc_info.value.body == {'error': 'overloaded'}
