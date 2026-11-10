"""Which Anthropic SDK clients and models get automatic prompt caching (the top-level `cache_control` parameter).

Mocked at the HTTP transport rather than recorded, because Bedrock and Vertex AI can't be recorded here.
Going through each real SDK client checks the parameter survives the client's own request rewriting.
"""

from __future__ import annotations as _annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import httpx2
import pytest

from pydantic_ai import Agent
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import ModelSettings

from ...conftest import try_import
from ..conftest import cache_breakpoints

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
    profile: ModelProfile | Callable[[ModelProfile], ModelProfile] | None = None


CASES = [
    Case('api', 'api', _automatic),
    Case('mantle', 'mantle', _automatic),
    Case('vertex', 'vertex', _automatic),
    Case('foundry', 'foundry', _automatic),
    # The legacy Bedrock InvokeModel API rejects the top-level `cache_control` parameter with a 400 on the
    # ARN-versioned models, but takes it on the models it serves from the Messages API infrastructure.
    Case('bedrock-sonnet-4-5', 'bedrock', _last_block, model_name='us.anthropic.claude-sonnet-4-5-20250929-v1:0'),
    Case('bedrock-opus-4-8', 'bedrock', _automatic, model_name='us.anthropic.claude-opus-4-8'),
    # An explicit `supports_auto_cache` in either `profile=` form is the override, both ways.
    Case(
        'bedrock-profile-on',
        'bedrock',
        _automatic,
        model_name='us.anthropic.claude-sonnet-4-6',
        profile=ModelProfile(supports_auto_cache=True),
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
    Case('api-profile-off', 'api', _last_block, profile=ModelProfile(supports_auto_cache=False)),
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


@pytest.mark.parametrize(
    'model_name,supports_auto_cache',
    [
        # Every documented family, with and without a cross-region geo prefix and point-release suffix.
        pytest.param('us.anthropic.claude-fable-5', True, id='fable-5'),
        pytest.param('global.anthropic.claude-fable-5-1', True, id='fable-5-1'),
        pytest.param('us.anthropic.claude-haiku-5-5', True, id='haiku-5-5'),
        pytest.param('us.anthropic.claude-opus-4-7', True, id='opus-4-7'),
        pytest.param('eu.anthropic.claude-opus-4-8', True, id='opus-4-8-eu'),
        pytest.param('anthropic.claude-opus-4-8', True, id='opus-4-8-no-geo'),
        pytest.param('us.anthropic.claude-opus-5', True, id='opus-5'),
        pytest.param('global.anthropic.claude-opus-5-5', True, id='opus-5-5'),
        pytest.param('us.anthropic.claude-sonnet-5', True, id='sonnet-5'),
        pytest.param('apac.anthropic.claude-sonnet-5-5', True, id='sonnet-5-5-apac'),
        # An inference profile ARN is classified by the profile ID it names.
        pytest.param(
            'arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-opus-4-8',
            True,
            id='opus-4-8-inference-profile-arn',
        ),
        pytest.param(
            'arn:aws:bedrock:us-east-1:123456789012:inference-profile/global.anthropic.claude-opus-4-6-v1',
            False,
            id='opus-4-6-inference-profile-arn',
        ),
        pytest.param(
            'arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-5-20250929-v1:0',
            False,
            id='sonnet-4-5-foundation-model-arn',
        ),
        # ARN-versioned models, verified live to reject top-level `cache_control` with a 400, plus older ones.
        pytest.param('us.anthropic.claude-opus-4-6-v1', False, id='opus-4-6'),
        pytest.param('global.anthropic.claude-opus-4-6-v1', False, id='opus-4-6-global'),
        pytest.param('us.anthropic.claude-opus-4-5-20251101-v1:0', False, id='opus-4-5'),
        pytest.param('us.anthropic.claude-sonnet-4-5-20250929-v1:0', False, id='sonnet-4-5'),
        pytest.param('anthropic.claude-opus-4-1-20250805-v1:0', False, id='opus-4-1'),
        # Accepted it live, but undocumented, so deliberately left out.
        pytest.param('us.anthropic.claude-sonnet-4-6', False, id='sonnet-4-6'),
        pytest.param('us.anthropic.claude-haiku-4-5-20251001-v1:0', False, id='haiku-4-5'),
    ],
)
def test_legacy_bedrock_automatic_caching_by_model(model_name: str, supports_auto_cache: bool):
    """On the legacy Bedrock client, `supports_auto_cache` follows the documented model list."""
    client = AsyncAnthropicBedrock(api_key='x', aws_region='us-east-1')
    model = AnthropicModel(model_name, provider=AnthropicProvider(anthropic_client=client))
    assert model.profile.get('supports_auto_cache') is supports_auto_cache


def _without_cache_control(value: Any) -> Any:
    """`value` with every `cache_control` key removed, which marks a breakpoint but isn't part of the cached prefix."""
    if isinstance(value, dict):
        items = cast(dict[str, Any], value).items()
        return {key: _without_cache_control(item) for key, item in items if key != 'cache_control'}
    if isinstance(value, list):
        return [_without_cache_control(item) for item in cast(list[Any], value)]
    return value


def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f'Sunny in {city}'  # pragma: no cover


@pytest.mark.parametrize(
    'client_name,model_name,expected_second_request',
    [
        # Automatic caching: the request-level parameter moves the breakpoint forward by itself.
        pytest.param('api', 'claude-sonnet-4-5', ({'type': 'ephemeral', 'ttl': '5m'}, list[str]()), id='automatic'),
        # The per-block fallback moves the breakpoint to the new last block.
        pytest.param(
            'bedrock',
            'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
            (None, ['messages[2].content[0]']),
            id='per-block-fallback',
        ),
    ],
)
async def test_anthropic_cache_keeps_the_prefix_stable(
    allow_model_requests: None,
    client_name: str,
    model_name: str,
    expected_second_request: tuple[dict[str, str] | None, list[str]],
):
    """The second turn re-sends the first request's cacheable prefix unchanged, so it can read the first turn's cache."""
    bodies: list[dict[str, Any]] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, json=_MESSAGE_JSON)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as http_client:
        client = (
            AsyncAnthropic(api_key='x', http_client=http_client)
            if client_name == 'api'
            else AsyncAnthropicBedrock(api_key='x', aws_region='us-east-1', http_client=http_client)
        )
        model = AnthropicModel(model_name, provider=AnthropicProvider(anthropic_client=client))
        agent = Agent(
            model,
            instructions='You are a weather assistant.',
            tools=[get_weather],
            model_settings=AnthropicModelSettings(max_tokens=256, anthropic_cache=True),
        )
        first = await agent.run('What is the weather like in Paris?')
        await agent.run('And in London?', message_history=first.all_messages())

    first_body, second_body = bodies
    for key in ('system', 'tools'):
        assert _without_cache_control(second_body[key]) == _without_cache_control(first_body[key])
    first_messages = first_body['messages']
    assert _without_cache_control(second_body['messages'][: len(first_messages)]) == _without_cache_control(
        first_messages
    )
    assert cache_breakpoints(second_body) == expected_second_request
