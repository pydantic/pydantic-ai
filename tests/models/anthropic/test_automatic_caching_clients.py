"""Which Anthropic SDK clients get automatic prompt caching (the top-level `cache_control` parameter).

Mocked at the HTTP transport rather than recorded, because Bedrock and Vertex AI can't be recorded here.
Going through each real SDK client checks the parameter survives the client's own request rewriting.
"""

from __future__ import annotations as _annotations

import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from ...conftest import try_import

with try_import() as imports_successful:
    from anthropic import (
        AsyncAnthropic,
        AsyncAnthropicBedrock,
        AsyncAnthropicBedrockMantle,
        AsyncAnthropicFoundry,
        AsyncAnthropicVertex,
    )

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
]

_MESSAGE_JSON = {
    'id': 'msg_0123',
    'type': 'message',
    'role': 'assistant',
    'model': 'claude-sonnet-4-5',
    'content': [{'type': 'text', 'text': 'OK'}],
    'stop_reason': 'end_turn',
    'usage': {'input_tokens': 5, 'output_tokens': 1},
}

_CacheControl = dict[str, str]
_Breakpoints = tuple[_CacheControl | None, list[_CacheControl | None]]
"""The request-level `cache_control`, and the `cache_control` of each block in the user message."""


def _automatic(cache_control: _CacheControl) -> _Breakpoints:
    """The top-level `cache_control` parameter, with no breakpoint on the block."""
    return cache_control, [None]


def _last_block(cache_control: _CacheControl) -> _Breakpoints:
    """No top-level parameter, with the breakpoint on the last block of the conversation instead."""
    return None, [cache_control]


_EXPECTED_BREAKPOINTS: dict[str, Callable[[_CacheControl], _Breakpoints]] = {
    'api': _automatic,
    # The legacy Bedrock InvokeModel API rejects the top-level `cache_control` parameter with a 400.
    'bedrock': _last_block,
    'mantle': _automatic,
    'vertex': _automatic,
    'foundry': _automatic,
}


@pytest.mark.parametrize(
    'settings,ttl',
    [
        # Plain dicts: parameters are built at collection time, also on installs without `anthropic`.
        pytest.param({'max_tokens': 256, 'anthropic_cache': True}, '5m', id='anthropic_cache'),
        pytest.param({'max_tokens': 256, 'anthropic_cache': '1h'}, '1h', id='anthropic_cache-1h'),
        pytest.param({'max_tokens': 256, 'cache': True}, '5m', id='unified-cache'),
    ],
)
@pytest.mark.parametrize('client_name', list(_EXPECTED_BREAKPOINTS))
async def test_automatic_caching_by_client(
    allow_model_requests: None, client_name: str, settings: ModelSettings, ttl: str
):
    """Every client except the legacy Bedrock one sends the conversation breakpoint as automatic caching."""
    bodies: list[dict[str, Any]] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, json=_MESSAGE_JSON)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as http_client:
        clients: dict[str, Callable[[], Any]] = {
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
        await Agent(model, model_settings=settings).run('Hello')

    [body] = bodies
    [message] = body['messages']
    breakpoints = (body.get('cache_control'), [block.get('cache_control') for block in message['content']])
    assert breakpoints == _EXPECTED_BREAKPOINTS[client_name]({'type': 'ephemeral', 'ttl': ttl})
