"""Tests for Anthropic data residency (`anthropic_inference_geo`).

Mocked at the HTTP transport, so each test pins the request body as the SDK sent it and feeds back the `usage` shape
Anthropic answers with: `inference_geo` on a non-streamed message, and on `message_start` only for a stream (the
recorded streams in this suite carry it nowhere else). The Bedrock, Vertex AI and Foundry clients can't be recorded
here, and a mocked response is what lets those tests echo a geo the request didn't ask for.
"""

from __future__ import annotations as _annotations

import json
from typing import Any

import httpx2
import pytest

from pydantic_ai import Agent, ModelResponse

from ...conftest import try_import

with try_import() as imports_successful:
    from anthropic import (
        AsyncAnthropic,
        AsyncAnthropicBedrock,
        AsyncAnthropicBedrockMantle,
        AsyncAnthropicFoundry,
        AsyncAnthropicVertex,
    )

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.profiles.anthropic import AnthropicModelProfile
    from pydantic_ai.providers.anthropic import AnthropicProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
]

# An explicit `max_tokens` keeps non-streaming requests from being streamed behind the scenes.
_MAX_TOKENS = 64

_USAGE = {'input_tokens': 5, 'output_tokens': 1, 'inference_geo': 'us'}

_MESSAGE_JSON = {
    'id': 'msg_0123',
    'type': 'message',
    'role': 'assistant',
    'model': 'claude-sonnet-5-5',
    'content': [{'type': 'text', 'text': 'OK'}],
    'stop_reason': 'end_turn',
    'usage': _USAGE,
}


def _sse(event: str, data: dict[str, Any]) -> str:
    return f'event: {event}\ndata: {json.dumps(data)}\n\n'


_STREAM = ''.join(
    [
        _sse(
            'message_start',
            {'type': 'message_start', 'message': {**_MESSAGE_JSON, 'content': [], 'stop_reason': None}},
        ),
        _sse(
            'content_block_start',
            {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
        ),
        _sse(
            'content_block_delta',
            {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'OK'}},
        ),
        _sse('content_block_stop', {'type': 'content_block_stop', 'index': 0}),
        _sse(
            'message_delta',
            {'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': 1}},
        ),
        _sse('message_stop', {'type': 'message_stop'}),
    ]
).encode()


def _handler(bodies: list[dict[str, Any]]):
    def handle(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if body.get('stream'):
            return httpx2.Response(200, headers={'content-type': 'text/event-stream'}, content=_STREAM)
        return httpx2.Response(200, json=_MESSAGE_JSON)

    return handle


def _last_response(messages: list[Any]) -> ModelResponse:
    response = messages[-1]
    assert isinstance(response, ModelResponse)
    return response


@pytest.mark.parametrize('stream', [False, True], ids=['request', 'stream'])
@pytest.mark.parametrize(
    ('geo', 'provider_details'),
    [
        pytest.param('us', {'finish_reason': 'end_turn', 'inference_geo': 'us'}, id='set'),
        pytest.param(None, {'finish_reason': 'end_turn'}, id='unset'),
    ],
)
async def test_inference_geo_sent_and_reported(
    allow_model_requests: None,
    stream: bool,
    geo: str | None,
    provider_details: dict[str, Any],
):
    """The setting is sent as `inference_geo`, and where the request ran is reported only when the request set it.

    Every response from the Claude API reports a geo, so reporting it unasked would change `provider_details` for
    requests that don't use data residency.
    """
    bodies: list[dict[str, Any]] = []
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(_handler(bodies))) as http_client:
        model = AnthropicModel(
            'claude-sonnet-5-5',
            provider=AnthropicProvider(anthropic_client=AsyncAnthropic(api_key='x', http_client=http_client)),
        )
        settings = AnthropicModelSettings(max_tokens=_MAX_TOKENS)
        if geo is not None:
            settings['anthropic_inference_geo'] = geo
        agent = Agent(model, model_settings=settings)
        if stream:
            async with agent.run_stream('Reply with exactly: OK') as streamed:
                await streamed.get_output()
            messages = streamed.all_messages()
        else:
            messages = (await agent.run('Reply with exactly: OK')).all_messages()

    [body] = bodies
    assert body.get('stream', False) is stream
    assert body.get('inference_geo') == geo
    assert _last_response(messages).provider_details == provider_details


async def test_inference_geo_sent_to_models_without_data_residency(allow_model_requests: None):
    """The setting isn't gated by model: a model without data residency gets it and answers with a 400.

    Dropping it instead would quietly run a request the user asked to keep in one geography anywhere.
    """
    bodies: list[dict[str, Any]] = []
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(_handler(bodies))) as http_client:
        model = AnthropicModel(
            'claude-haiku-4-5',
            provider=AnthropicProvider(anthropic_client=AsyncAnthropic(api_key='x', http_client=http_client)),
        )
        await Agent(
            model, model_settings=AnthropicModelSettings(max_tokens=_MAX_TOKENS, anthropic_inference_geo='us')
        ).run('Reply with exactly: OK')

    [body] = bodies
    assert body['inference_geo'] == 'us'


@pytest.mark.parametrize(
    ('client_name', 'expected'),
    [
        pytest.param('api', 'us', id='api'),
        pytest.param('bedrock', None, id='bedrock'),
        pytest.param('mantle', None, id='mantle'),
        pytest.param('vertex', None, id='vertex'),
        pytest.param('foundry', None, id='foundry'),
    ],
)
async def test_inference_geo_request_field_by_client(
    allow_model_requests: None, client_name: str, expected: str | None
):
    """Bedrock, Vertex AI and Foundry decide the region by endpoint, so the setting is ignored there with a warning."""
    bodies: list[dict[str, Any]] = []
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(_handler(bodies))) as http_client:
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
        model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=clients[client_name]()))
        agent = Agent(
            model, model_settings=AnthropicModelSettings(max_tokens=_MAX_TOKENS, anthropic_inference_geo='us')
        )
        if expected is None:
            with pytest.warns(UserWarning, match='`anthropic_inference_geo` is not supported on this client'):
                messages = (await agent.run('Reply with exactly: OK')).all_messages()
        else:
            messages = (await agent.run('Reply with exactly: OK')).all_messages()

    [body] = bodies
    assert body.get('inference_geo') == expected
    # The mocked response always reports a geo; it's only surfaced when the request carried one.
    assert ('inference_geo' in (_last_response(messages).provider_details or {})) is (expected is not None)


@pytest.mark.parametrize('client_name', ['mantle', 'vertex', 'foundry'])
async def test_inference_geo_not_reported_when_ignored_streamed(allow_model_requests: None, client_name: str):
    """A streamed response on a client that ignored the setting doesn't report the geo its `message_start` echoes.

    The legacy Bedrock client streams AWS event-stream frames rather than SSE, so the mocked stream skips it; its
    non-streamed case is in `test_inference_geo_request_field_by_client`.
    """
    bodies: list[dict[str, Any]] = []
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(_handler(bodies))) as http_client:
        clients = {
            'mantle': lambda: AsyncAnthropicBedrockMantle(api_key='x', aws_region='us-east-1', http_client=http_client),
            'vertex': lambda: AsyncAnthropicVertex(
                project_id='p', region='us-east5', access_token='x', http_client=http_client
            ),
            'foundry': lambda: AsyncAnthropicFoundry(
                api_key='x', base_url='https://example.com/anthropic', http_client=http_client
            ),
        }
        model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(anthropic_client=clients[client_name]()))
        agent = Agent(
            model, model_settings=AnthropicModelSettings(max_tokens=_MAX_TOKENS, anthropic_inference_geo='us')
        )
        with pytest.warns(UserWarning, match='`anthropic_inference_geo` is not supported on this client'):
            async with agent.run_stream('Reply with exactly: OK') as streamed:
                await streamed.get_output()

    [body] = bodies
    assert body['stream'] is True
    assert 'inference_geo' not in body
    assert _last_response(streamed.all_messages()).provider_details == {'finish_reason': 'end_turn'}


async def test_inference_geo_profile_override(allow_model_requests: None):
    """A profile that sets the flag wins over the client-class default."""
    bodies: list[dict[str, Any]] = []
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(_handler(bodies))) as http_client:
        client = AsyncAnthropicFoundry(api_key='x', base_url='https://example.com/anthropic', http_client=http_client)
        model = AnthropicModel(
            'claude-sonnet-5-5',
            provider=AnthropicProvider(anthropic_client=client),
            profile=AnthropicModelProfile(anthropic_supports_inference_geo=True),
        )
        await Agent(
            model, model_settings=AnthropicModelSettings(max_tokens=_MAX_TOKENS, anthropic_inference_geo='us')
        ).run('Reply with exactly: OK')

    [body] = bodies
    assert body['inference_geo'] == 'us'
