"""Which Anthropic SDK clients and models get automatic prompt caching (the top-level `cache_control` parameter).

Mocked at the HTTP transport rather than recorded, because Bedrock and Vertex AI can't be recorded here.
Going through each real SDK client checks the parameter survives the client's own request rewriting.
"""

from __future__ import annotations as _annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx2
import pytest

from pydantic_ai import Agent
from pydantic_ai.profiles import ModelProfile
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
    from pydantic_ai.profiles.anthropic import AnthropicModelProfile
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


@dataclass(frozen=True)
class Case:
    id: str
    client: str
    expected: Callable[[_CacheControl], _Breakpoints]
    model_name: str = 'claude-sonnet-4-5'
    profile: AnthropicModelProfile | Callable[[ModelProfile], ModelProfile] | None = None


CASES = [
    Case('api', 'api', _automatic),
    Case('mantle', 'mantle', _automatic),
    Case('vertex', 'vertex', _automatic),
    Case('foundry', 'foundry', _automatic),
    # The legacy Bedrock InvokeModel API rejects the top-level `cache_control` parameter with a 400 on the
    # ARN-versioned models, but takes it on the models it serves from the Messages API infrastructure.
    Case('bedrock-sonnet-4-5', 'bedrock', _last_block, model_name='us.anthropic.claude-sonnet-4-5-20250929-v1:0'),
    Case('bedrock-opus-4-8', 'bedrock', _automatic, model_name='us.anthropic.claude-opus-4-8'),
    Case('bedrock-opus-5-5', 'bedrock', _automatic, model_name='global.anthropic.claude-opus-5-5'),
    # An explicit `supports_auto_cache` in either `profile=` form is the override, both ways.
    Case(
        'bedrock-profile-on',
        'bedrock',
        _automatic,
        model_name='us.anthropic.claude-sonnet-4-6',
        profile=AnthropicModelProfile(supports_auto_cache=True),
    ),
    Case(
        'bedrock-callable-profile-on',
        'bedrock',
        _automatic,
        model_name='us.anthropic.claude-sonnet-4-6',
        profile=lambda default: {**default, 'supports_auto_cache': True},
    ),
    # A callable that leaves the flag alone keeps the narrowed default.
    Case(
        'bedrock-callable-profile-untouched',
        'bedrock',
        _last_block,
        model_name='us.anthropic.claude-sonnet-4-6',
        profile=lambda default: default,
    ),
    Case('api-profile-off', 'api', _last_block, profile=AnthropicModelProfile(supports_auto_cache=False)),
    Case(
        'api-callable-profile-off',
        'api',
        _last_block,
        profile=lambda default: {**default, 'supports_auto_cache': False},
    ),
]


@pytest.mark.parametrize(
    'settings,ttl',
    [
        # Plain dicts: parameters are built at collection time, also on installs without `anthropic`.
        pytest.param({'max_tokens': 256, 'anthropic_cache': True}, '5m', id='anthropic_cache'),
        pytest.param({'max_tokens': 256, 'anthropic_cache': '1h'}, '1h', id='anthropic_cache-1h'),
        pytest.param({'max_tokens': 256, 'cache': True}, '5m', id='unified-cache'),
    ],
)
@pytest.mark.parametrize('case', [pytest.param(c, id=c.id) for c in CASES])
async def test_automatic_caching_by_client(allow_model_requests: None, case: Case, settings: ModelSettings, ttl: str):
    """The profile's `supports_auto_cache` decides between automatic caching and a breakpoint on the last block."""
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
        model = AnthropicModel(
            case.model_name,
            provider=AnthropicProvider(anthropic_client=clients[case.client]()),
            profile=case.profile,
        )
        await Agent(model, model_settings=settings).run('Hello')

    [body] = bodies
    [message] = body['messages']
    breakpoints = (body.get('cache_control'), [block.get('cache_control') for block in message['content']])
    assert breakpoints == case.expected({'type': 'ephemeral', 'ttl': ttl})
