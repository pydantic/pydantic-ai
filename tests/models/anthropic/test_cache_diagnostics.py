"""Tests for Anthropic cache diagnostics (`anthropic_cache_diagnostics`).

The recorded tests pin the request field as the model built it (via `request_capture`) and the
diagnostics Anthropic reported, for a clean cache hit, a deliberate `tools_changed` miss (both
non-streaming and streaming) and baselines that must not be sent. The transport gating is tested
against each non-API client class with a mock HTTP transport, since those platforms reject the field.
"""

from __future__ import annotations as _annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import httpx2
import pytest

from pydantic_ai import Agent, ModelRequest, ModelResponse, TextPart
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.toolsets import FunctionToolset

from ..._inline_snapshot import snapshot
from ...conftest import RequestCapture, try_import

with try_import() as imports_successful:
    from anthropic import (
        AsyncAnthropic,
        AsyncAnthropicBedrock,
        AsyncAnthropicBedrockMantle,
        AsyncAnthropicFoundry,
        AsyncAnthropicVertex,
    )

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

if TYPE_CHECKING:
    ANTHROPIC_MODEL_FIXTURE = Callable[..., AnthropicModel]

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
]

# Longer than Claude Sonnet 4.5's 1024-token minimum cacheable prompt, so the second turn reads the cache.
_STABLE_PREFIX = 'Reference catalogue for the cache diagnostics test corpus.\n' + '\n'.join(
    f'Entry {i:04d}: shelf {i % 23}, aisle {i % 7}, volume {i}, catalogued under subject heading {i % 11}.'
    for i in range(160)
)


# An explicit `max_tokens` keeps non-streaming requests from being streamed behind the scenes.
_MAX_TOKENS = 256


def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f'Sunny in {city}'  # pragma: no cover


def get_time(city: str) -> str:
    """Get the local time in a city."""
    return f'Noon in {city}'  # pragma: no cover


def _diagnostics_agent(model: AnthropicModel) -> Agent[None, str]:
    settings = AnthropicModelSettings(max_tokens=_MAX_TOKENS, anthropic_cache=True, anthropic_cache_diagnostics=True)
    return Agent(model, instructions=_STABLE_PREFIX, model_settings=settings, tools=[get_weather])


def _last_response(messages: list[Any]) -> ModelResponse:
    response = messages[-1]
    assert isinstance(response, ModelResponse)
    return response


@pytest.mark.vcr
@pytest.mark.moves_cache_prefix(reason='adds a tool on purpose, to get a `tools_changed` diagnosis')
async def test_cache_diagnostics_hit_then_tools_changed(
    allow_model_requests: None, anthropic_model: ANTHROPIC_MODEL_FIXTURE, request_capture: RequestCapture
):
    """The first turn opts in, later turns name the previous response, and only a divergence is recorded."""
    agent = _diagnostics_agent(anthropic_model('claude-sonnet-4-5', capture=True))

    first = await agent.run('Reply with exactly: OK')
    second = await agent.run('Reply with exactly: AGAIN', message_history=first.all_messages())
    third = await agent.run(
        'Reply with exactly: DONE', message_history=second.all_messages(), toolsets=[FunctionToolset([get_time])]
    )

    first_response = _last_response(first.all_messages())
    second_response = _last_response(second.all_messages())
    third_response = _last_response(third.all_messages())
    assert [body['diagnostics'] for body in request_capture.bodies('/v1/messages')] == [
        {'previous_message_id': None},
        {'previous_message_id': first_response.provider_response_id},
        {'previous_message_id': second_response.provider_response_id},
    ]

    # Nothing to compare on the first turn, and no divergence on the second: Anthropic reports `null` for both.
    assert first_response.provider_details is not None
    assert 'cache_diagnostics' not in first_response.provider_details
    assert second_response.provider_details is not None
    assert 'cache_diagnostics' not in second_response.provider_details
    assert second.usage.cache_read_tokens > 0

    assert third_response.provider_details is not None
    assert third_response.provider_details['cache_diagnostics'] == snapshot(
        {'cache_miss_reason': {'cache_missed_input_tokens': 3713, 'type': 'tools_changed'}}
    )
    # The recorded shape survives message-history serialization.
    round_tripped = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(third.all_messages()))
    assert _last_response(round_tripped).provider_details == third_response.provider_details


@pytest.mark.vcr
@pytest.mark.moves_cache_prefix(reason='adds a tool on purpose, to get a `tools_changed` diagnosis')
async def test_cache_diagnostics_streamed(
    allow_model_requests: None, anthropic_model: ANTHROPIC_MODEL_FIXTURE, request_capture: RequestCapture
):
    """A streamed response reports diagnostics on `message_start`."""
    agent = _diagnostics_agent(anthropic_model('claude-sonnet-4-5', capture=True))

    first = await agent.run('Reply with exactly: OK')
    async with agent.run_stream(
        'Reply with exactly: DONE', message_history=first.all_messages(), toolsets=[FunctionToolset([get_time])]
    ) as streamed:
        await streamed.get_output()

    first_response = _last_response(first.all_messages())
    streamed_response = _last_response(streamed.all_messages())
    assert request_capture.body('/v1/messages', index=1)['diagnostics'] == {
        'previous_message_id': first_response.provider_response_id
    }
    assert streamed_response.provider_details is not None
    assert streamed_response.provider_details['cache_diagnostics'] == snapshot(
        {'cache_miss_reason': {'cache_missed_input_tokens': 3701, 'type': 'tools_changed'}}
    )


@pytest.mark.vcr
@pytest.mark.parametrize(
    ('provider_name', 'provider_response_id'),
    [
        pytest.param('openai', 'resp_0123456789abcdef0123456789abcdef', id='other-provider'),
        # An Anthropic-compatible endpoint (here OpenRouter) issues ids Anthropic would reject with a 400.
        pytest.param('anthropic', 'gen-1790722484-abcdefghijklmnopqrst', id='non-message-id'),
    ],
)
async def test_cache_diagnostics_skip_foreign_baseline(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
    request_capture: RequestCapture,
    provider_name: str,
    provider_response_id: str,
):
    """A previous response Anthropic can't compare against opts in without a baseline."""
    history = [
        ModelRequest.user_text_prompt('Say hi.'),
        ModelResponse(
            parts=[TextPart('Hi!')],
            model_name='some-model',
            provider_name=provider_name,
            provider_response_id=provider_response_id,
        ),
    ]
    agent = Agent(
        anthropic_model('claude-sonnet-4-5', capture=True),
        model_settings=AnthropicModelSettings(max_tokens=_MAX_TOKENS, anthropic_cache_diagnostics=True),
    )

    result = await agent.run('Reply with exactly: OK', message_history=history)

    assert request_capture.body('/v1/messages')['diagnostics'] == {'previous_message_id': None}
    assert result.output == 'OK'


_MESSAGE_JSON = {
    'id': 'msg_0123',
    'type': 'message',
    'role': 'assistant',
    'model': 'claude-sonnet-4-5',
    'content': [{'type': 'text', 'text': 'OK'}],
    'stop_reason': 'end_turn',
    'usage': {'input_tokens': 5, 'output_tokens': 1},
}


def _history(
    provider_name: str = 'anthropic', provider_response_id: str = 'msg_0122'
) -> list[ModelRequest | ModelResponse]:
    return [
        ModelRequest.user_text_prompt('Say hi.'),
        ModelResponse(
            parts=[TextPart('Hi!')],
            model_name='claude-sonnet-4-5',
            provider_name=provider_name,
            provider_response_id=provider_response_id,
        ),
    ]


@pytest.mark.parametrize(
    ('client_name', 'settings', 'expected'),
    [
        pytest.param('api', {}, None, id='api-default-off'),
        pytest.param('api', {'anthropic_cache_diagnostics': True}, {'previous_message_id': 'msg_0122'}, id='api'),
        pytest.param('bedrock', {'anthropic_cache_diagnostics': True}, None, id='bedrock'),
        pytest.param('mantle', {'anthropic_cache_diagnostics': True}, None, id='mantle'),
        pytest.param('vertex', {'anthropic_cache_diagnostics': True}, None, id='vertex'),
        pytest.param('foundry', {'anthropic_cache_diagnostics': True}, None, id='foundry'),
    ],
)
async def test_cache_diagnostics_request_field_by_client(
    allow_model_requests: None,
    client_name: str,
    settings: AnthropicModelSettings,
    expected: dict[str, str] | None,
):
    """The field is off by default and never sent to Bedrock, Vertex AI or Foundry.

    Mocked because these platforms can't be recorded here, and Bedrock rejects the field with a 400
    ("diagnostics: Extra inputs are not permitted") on both its InvokeModel and Messages APIs.
    """
    bodies: list[dict[str, Any]] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, json=_MESSAGE_JSON)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as http_client:
        clients = {
            'api': lambda: AsyncAnthropic(api_key='x', http_client=http_client),
            'bedrock': lambda: AsyncAnthropicBedrock(api_key='x', aws_region='us-east-1', http_client=http_client),
            'mantle': lambda: AsyncAnthropicBedrockMantle(api_key='x', aws_region='us-east-1', http_client=http_client),
            'vertex': lambda: AsyncAnthropicVertex(
                project_id='p', region='us-east5', access_token='x', http_client=http_client
            ),
            'foundry': lambda: AsyncAnthropicFoundry(
                api_key='x', base_url='https://example.com/anthropic', http_client=http_client
            ),
        }
        model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=clients[client_name]()))
        await Agent(model, model_settings={'max_tokens': _MAX_TOKENS, **settings}).run(
            'Reply with exactly: OK', message_history=_history()
        )

    [body] = bodies
    assert body.get('diagnostics') == expected


async def test_cache_diagnostics_recorded_while_pending(allow_model_requests: None):
    """A comparison that was still running is kept as `{'cache_miss_reason': None}`, distinct from no divergence."""

    def handle(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={**_MESSAGE_JSON, 'diagnostics': {'cache_miss_reason': None}})

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as http_client:
        client = AsyncAnthropic(api_key='x', http_client=http_client)
        model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=client))
        agent = Agent(
            model, model_settings=AnthropicModelSettings(max_tokens=_MAX_TOKENS, anthropic_cache_diagnostics=True)
        )
        result = await agent.run('Reply with exactly: OK', message_history=_history())

    assert _last_response(result.all_messages()).provider_details == snapshot(
        {'finish_reason': 'end_turn', 'cache_diagnostics': {'cache_miss_reason': None}}
    )


async def test_cache_diagnostics_skip_other_provider_with_message_id(allow_model_requests: None):
    """A `msg_` ID from another provider (here Bedrock Converse) isn't one the Claude API can look up."""
    bodies: list[dict[str, Any]] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, json=_MESSAGE_JSON)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as http_client:
        client = AsyncAnthropic(api_key='x', http_client=http_client)
        model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=client))
        agent = Agent(
            model, model_settings=AnthropicModelSettings(max_tokens=_MAX_TOKENS, anthropic_cache_diagnostics=True)
        )
        await agent.run('Reply with exactly: OK', message_history=_history('bedrock', 'msg_bdrk_0122'))

    [body] = bodies
    assert body['diagnostics'] == {'previous_message_id': None}
