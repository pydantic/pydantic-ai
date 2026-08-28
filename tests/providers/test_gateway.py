import os
import re
from types import SimpleNamespace
from typing import Any, Literal, cast
from unittest.mock import patch
from urllib.parse import urlparse

import httpx
import httpx2
import pytest

from pydantic_ai import Agent, UserError
from pydantic_ai._warnings import PydanticAIDeprecationWarning

from .._inline_snapshot import raises, snapshot
from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from google.genai import Client

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers import Provider
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.bedrock import BedrockProvider
    from pydantic_ai.providers.gateway import (
        _set_google_ws_gateway_auth,  # pyright: ignore[reportPrivateUsage]
        gateway_provider,
        is_gateway_provider,
    )
    from pydantic_ai.providers.google_cloud import GoogleCloudProvider
    from pydantic_ai.providers.groq import GroqProvider
    from pydantic_ai.providers.openai import OpenAIProvider


if not imports_successful():
    pytest.skip('Providers not installed', allow_module_level=True)  # pragma: lax no cover

pytestmark = [pytest.mark.anyio, pytest.mark.vcr]

# Any URL works here — these tests exercise the explicit `PYDANTIC_AI_GATEWAY_BASE_URL` override path.
GATEWAY_BASE_URL = 'https://gateway.pydantic.dev/proxy'


@pytest.mark.parametrize(
    'provider_name, provider_cls, route',
    [
        # PAIG exposes a single canonical `openai` route; Chat vs Responses is selected by the
        # OpenAI SDK's sub-path (/chat/completions vs /responses), not by the URL prefix.
        ('openai', OpenAIProvider, 'openai'),
        ('openai-chat', OpenAIProvider, 'openai'),
        ('openai-responses', OpenAIProvider, 'openai'),
    ],
)
def test_init_with_base_url(
    provider_name: Literal['openai', 'openai-chat', 'openai-responses'], provider_cls: type[Provider[Any]], route: str
):
    provider = gateway_provider(provider_name, base_url='https://example.com/', api_key='foobar')
    assert isinstance(provider, provider_cls)
    assert provider.base_url == f'https://example.com/{route}/'
    assert provider.client.api_key == 'foobar'
    assert isinstance(provider.client._client, httpx2.AsyncClient)  # pyright: ignore[reportPrivateUsage]


def test_is_gateway_provider():
    assert is_gateway_provider(gateway_provider('openai', api_key='gw-key', base_url=GATEWAY_BASE_URL))
    assert not is_gateway_provider(OpenAIProvider(api_key='k'))


def test_is_gateway_provider_accepts_an_unhashable_provider():
    # A `Provider` is free to be a plain `@dataclass`, which sets `__hash__ = None`. Asking whether one
    # is a gateway provider must answer, not raise out of the set lookup.
    class UnhashableProvider(OpenAIProvider):
        __hash__ = None  # type: ignore[assignment]

    assert not is_gateway_provider(UnhashableProvider(api_key='k'))


def test_gateway_google_sets_static_ws_bearer_auth():
    # Unit (not VCR): the gateway's httpx request hook adds `Authorization: Bearer <key>` to REST calls,
    # but it can't reach the Gemini Live handshake — `google-genai` dials that WebSocket with the
    # `websockets` library, bypassing httpx. So `gateway_provider` sets the bearer as a *static* header on
    # the client's http options at build time (the SDK forwards those to both REST and the WS handshake).
    # This pins that the header lands on the client, since a cassette wouldn't exercise the WS dial.
    provider = gateway_provider('google', api_key='gw-key', base_url=GATEWAY_BASE_URL)
    headers = provider.client._api_client._http_options.headers  # pyright: ignore[reportPrivateUsage]
    assert headers is not None and headers['Authorization'] == 'Bearer gw-key'
    assert isinstance(provider.client._api_client._async_httpx_client, httpx2.AsyncClient)  # pyright: ignore[reportPrivateUsage]


async def test_gateway_google_preserves_caller_owned_httpx2_client():
    async with httpx2.AsyncClient() as http_client:
        provider = gateway_provider('google', http_client=http_client, api_key='gw-key', base_url=GATEWAY_BASE_URL)

        assert provider.client._api_client._async_httpx_client is http_client  # pyright: ignore[reportPrivateUsage]
        async with provider:
            pass
        assert not http_client.is_closed


async def test_gateway_google_recreates_owned_httpx2_client():
    provider = gateway_provider('google', api_key='gw-key', base_url=GATEWAY_BASE_URL)
    first_client = provider.client._api_client._async_httpx_client  # pyright: ignore[reportPrivateUsage]
    assert isinstance(first_client, httpx2.AsyncClient)

    async with provider:
        pass
    assert first_client.is_closed

    async with provider:
        second_client = provider.client._api_client._async_httpx_client  # pyright: ignore[reportPrivateUsage]
        assert isinstance(second_client, httpx2.AsyncClient)
        assert second_client is not first_client
        request = httpx2.Request('GET', provider.base_url)
        for hook in second_client.event_hooks['request']:
            await hook(request)
        assert request.headers['Authorization'] == 'Bearer gw-key'


def test_gateway_google_ws_bearer_auth_skips_unusable_clients():
    # The bearer is set by reaching into the SDK's private http options, so the helper has to tolerate a
    # client that doesn't have them (a fake or a future SDK layout) and must not clobber an
    # `Authorization` header the caller set deliberately. Both cases leave the client untouched.
    class _NoHttpOptions:
        pass

    _set_google_ws_gateway_auth(cast('Client', _NoHttpOptions()), 'gw-key')  # no attribute chain: no-op

    class _Client:
        def __init__(self, headers: dict[str, str]) -> None:
            self._api_client = SimpleNamespace(_http_options=SimpleNamespace(headers=headers))

    preset = {'Authorization': 'Bearer caller-supplied'}
    _set_google_ws_gateway_auth(cast('Client', _Client(preset)), 'gw-key')
    assert preset == {'Authorization': 'Bearer caller-supplied'}

    empty: dict[str, str] = {}
    _set_google_ws_gateway_auth(cast('Client', _Client(empty)), 'gw-key')
    assert empty == {'Authorization': 'Bearer gw-key'}


def test_init_gateway_without_api_key_raises_error(env: TestEnv):
    env.remove('PYDANTIC_AI_GATEWAY_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `PYDANTIC_AI_GATEWAY_API_KEY` environment variable or pass it via `gateway_provider(..., api_key=...)` to use the Pydantic AI Gateway provider.'
        ),
    ):
        gateway_provider('openai')


async def test_init_with_http_client():
    async with httpx2.AsyncClient() as http_client:
        provider = gateway_provider('openai', http_client=http_client, api_key='foobar', base_url=GATEWAY_BASE_URL)
        assert provider.client._client == http_client  # pyright: ignore[reportPrivateUsage]


async def test_init_with_http_client_preserves_existing_event_hooks():
    # Unit (not VCR): this checks local HTTPX hook merging by inspecting and invoking event hooks directly;
    # cassette playback would not exercise hook ordering or preservation.
    async def existing_request_hook(request: httpx2.Request) -> None:
        request.headers['X-Existing-Request-Hook'] = 'kept'

    async def existing_response_hook(response: httpx2.Response) -> None:
        response.headers['X-Existing-Response-Hook'] = 'kept'

    async with httpx2.AsyncClient(
        event_hooks={'request': [existing_request_hook], 'response': [existing_response_hook]}
    ) as http_client:
        provider = gateway_provider('openai', http_client=http_client, api_key='foobar', base_url=GATEWAY_BASE_URL)
        assert provider.client._client == http_client  # pyright: ignore[reportPrivateUsage]
        assert existing_request_hook in http_client.event_hooks['request']
        assert existing_response_hook in http_client.event_hooks['response']

        request = httpx2.Request('GET', provider.base_url)
        for hook in http_client.event_hooks['request']:
            await hook(request)

        assert request.headers['X-Existing-Request-Hook'] == 'kept'
        assert request.headers['Authorization'] == 'Bearer foobar'

        response = httpx2.Response(200, request=request)
        for hook in http_client.event_hooks['response']:
            await hook(response)

        assert response.headers['X-Existing-Response-Hook'] == 'kept'


async def test_init_with_http_client_replaces_existing_gateway_hook():
    # Unit (not VCR): this checks local HTTPX hook replacement by inspecting and invoking event hooks directly;
    # cassette playback would not exercise Gateway hook deduplication.
    async def existing_request_hook(request: httpx2.Request) -> None:
        request.headers['X-Existing-Request-Hook'] = 'kept'

    async with httpx2.AsyncClient(event_hooks={'request': [existing_request_hook]}) as http_client:
        first_provider = gateway_provider('openai', http_client=http_client, api_key='first', base_url=GATEWAY_BASE_URL)
        second_provider = gateway_provider(
            'openai', http_client=http_client, api_key='second', base_url=GATEWAY_BASE_URL
        )

        assert first_provider.client._client == http_client  # pyright: ignore[reportPrivateUsage]
        assert second_provider.client._client == http_client  # pyright: ignore[reportPrivateUsage]
        assert http_client.event_hooks['request'][0] == existing_request_hook
        assert len(http_client.event_hooks['request']) == 2

        request = httpx2.Request('GET', second_provider.base_url)
        for hook in http_client.event_hooks['request']:
            await hook(request)

        assert request.headers['X-Existing-Request-Hook'] == 'kept'
        assert request.headers['Authorization'] == 'Bearer second'


@pytest.mark.parametrize('provider_name', ['openai', 'google-cloud'])
async def test_gateway_provider_hooks_a_caller_owned_legacy_http_client(
    provider_name: Literal['openai', 'google-cloud'],
):
    # Unit (not VCR): the OpenAI and Google routes default to HTTPX2 but still accept the deprecated
    # `httpx.AsyncClient` through v2, and a missing auth hook on it would fail silently (every gateway
    # request 401s) rather than at construction. Invoking the hooks directly pins that they were installed;
    # cassette playback wouldn't exercise hook installation.
    async with httpx.AsyncClient() as http_client:
        with pytest.warns(PydanticAIDeprecationWarning, match=r'`httpx\.AsyncClient` support .* is deprecated'):
            provider = gateway_provider(
                provider_name, http_client=http_client, api_key='gw-key', base_url=GATEWAY_BASE_URL
            )

        request = httpx.Request('GET', provider.base_url)
        for hook in http_client.event_hooks['request']:
            await hook(request)
        assert request.headers['Authorization'] == 'Bearer gw-key'

        async with provider:
            pass
        assert not http_client.is_closed


async def test_non_openai_gateway_provider_recreates_owned_http_client():
    provider = gateway_provider('groq', api_key='foobar', base_url=GATEWAY_BASE_URL)

    async with provider:
        pass
    original_http_client = provider._own_http_client  # pyright: ignore[reportPrivateUsage]

    async with provider:
        pass

    assert provider._own_http_client is not original_http_client  # pyright: ignore[reportPrivateUsage]


async def test_non_openai_gateway_provider_preserves_custom_http_client():
    async with httpx.AsyncClient() as http_client:
        provider = gateway_provider('groq', http_client=http_client, api_key='foobar', base_url=GATEWAY_BASE_URL)

        async with provider:
            pass

        assert not http_client.is_closed


async def test_non_openai_gateway_provider_rejects_httpx2_client():
    async with httpx2.AsyncClient() as http_client:
        with pytest.raises(
            UserError,
            match=re.escape('`httpx2.AsyncClient` is only supported for OpenAI, Google and Anthropic Gateway routes.'),
        ):
            gateway_provider(  # pyright: ignore[reportCallIssue]
                'groq',
                http_client=http_client,  # pyright: ignore[reportArgumentType]
                api_key='foobar',
                base_url=GATEWAY_BASE_URL,
            )


async def test_anthropic_gateway_provider_rejects_legacy_httpx_client():
    async with httpx.AsyncClient() as http_client:
        with pytest.raises(UserError, match=re.escape('The Anthropic Gateway route requires an `httpx2.AsyncClient`.')):
            gateway_provider(  # pyright: ignore[reportCallIssue]
                'anthropic',
                http_client=http_client,  # pyright: ignore[reportArgumentType]
                api_key='foobar',
                base_url=GATEWAY_BASE_URL,
            )


@pytest.fixture
def gateway_api_key():
    return os.getenv('PYDANTIC_AI_GATEWAY_API_KEY', 'test-api-key')


@pytest.fixture(scope='module')
def vcr_config():
    return {
        'ignore_localhost': False,
        # Note: additional header filtering is done inside the serializer
        'filter_headers': ['authorization', 'x-api-key'],
        'decode_compressed_response': True,
    }


@patch.dict(
    os.environ, {'PYDANTIC_AI_GATEWAY_API_KEY': 'test-api-key', 'PYDANTIC_AI_GATEWAY_BASE_URL': GATEWAY_BASE_URL}
)
@pytest.mark.parametrize(
    'provider_name, provider_cls, route',
    [
        ('openai', OpenAIProvider, 'openai'),
        ('openai-chat', OpenAIProvider, 'openai'),
        ('openai-responses', OpenAIProvider, 'openai'),
        ('groq', GroqProvider, 'groq'),
        ('google', GoogleCloudProvider, 'google-vertex'),
        ('google-cloud', GoogleCloudProvider, 'google-vertex'),
        ('anthropic', AnthropicProvider, 'anthropic'),
        ('bedrock', BedrockProvider, 'bedrock'),
    ],
)
def test_gateway_provider(provider_name: str, provider_cls: type[Provider[Any]], route: str):
    provider = gateway_provider(provider_name)
    assert isinstance(provider, provider_cls)

    # Some providers add a trailing slash, others don't
    assert provider.base_url in (f'{GATEWAY_BASE_URL}/{route}/', f'{GATEWAY_BASE_URL}/{route}')


@patch.dict(
    os.environ, {'PYDANTIC_AI_GATEWAY_API_KEY': 'test-api-key', 'PYDANTIC_AI_GATEWAY_BASE_URL': GATEWAY_BASE_URL}
)
@pytest.mark.parametrize('removed_alias', ['foo', 'google-vertex', 'gemini'])
def test_gateway_provider_unknown(removed_alias: str):
    # `google-vertex` and `gemini` were removed in v2 alongside their bare-prefix counterparts —
    # `gateway/google-vertex:` and `gateway/gemini:` raise the same `UserError` as any other unknown alias.
    with pytest.raises(UserError, match=f'Unknown upstream provider: {removed_alias}'):
        gateway_provider(removed_alias)


async def test_gateway_provider_with_openai(allow_model_requests: None, gateway_api_key: str):
    provider = gateway_provider('openai-chat', api_key=gateway_api_key, base_url='http://localhost:8787')
    model = OpenAIChatModel('gpt-5', provider=provider)
    agent = Agent(model)

    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('Paris.')


async def test_gateway_provider_with_openai_responses(allow_model_requests: None, gateway_api_key: str):
    provider = gateway_provider('openai-responses', api_key=gateway_api_key, base_url='http://localhost:8787')
    model = OpenAIResponsesModel('gpt-5', provider=provider)
    agent = Agent(model)

    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('Paris.')


async def test_gateway_provider_with_groq(allow_model_requests: None, gateway_api_key: str):
    provider = gateway_provider('groq', api_key=gateway_api_key, base_url='http://localhost:8787')
    model = GroqModel('llama-3.3-70b-versatile', provider=provider)
    agent = Agent(model)

    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is Paris.')


async def test_gateway_provider_with_google_cloud(allow_model_requests: None, gateway_api_key: str):
    provider = gateway_provider('google-cloud', api_key=gateway_api_key, base_url='http://localhost:8787')
    model = GoogleModel('gemini-2.5-flash', provider=provider)
    agent = Agent(model)

    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is **Paris**.')


async def test_gateway_provider_with_anthropic(allow_model_requests: None, gateway_api_key: str):
    provider = gateway_provider('anthropic', api_key=gateway_api_key, base_url='http://localhost:8787')
    model = AnthropicModel('claude-sonnet-4-5', provider=provider)
    agent = Agent(model)

    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is Paris.')


async def test_gateway_provider_with_bedrock(allow_model_requests: None, gateway_api_key: str):
    provider = gateway_provider('bedrock', api_key=gateway_api_key, base_url='http://localhost:8787')
    model = BedrockConverseModel('amazon.nova-micro-v1:0', provider=provider)
    agent = Agent(model)

    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot(
        'The capital of France is Paris. Paris is not only the capital city but also the most populous city in France, and it is a major center for culture, commerce, fashion, and international diplomacy. The city is known for its historical landmarks, such as the Eiffel Tower, the Louvre Museum, Notre-Dame Cathedral, and the Champs-Élysées, among many other attractions.'
    )


@patch.dict(
    os.environ, {'PYDANTIC_AI_GATEWAY_API_KEY': 'test-api-key', 'PYDANTIC_AI_GATEWAY_BASE_URL': GATEWAY_BASE_URL}
)
async def test_model_provider_argument():
    model = OpenAIChatModel('gpt-5', provider='gateway')
    assert urlparse(model._provider.base_url).hostname == urlparse(GATEWAY_BASE_URL).hostname  # pyright: ignore[reportPrivateUsage]

    model = OpenAIResponsesModel('gpt-5', provider='gateway')
    assert urlparse(model._provider.base_url).hostname == urlparse(GATEWAY_BASE_URL).hostname  # pyright: ignore[reportPrivateUsage]

    model = GroqModel('llama-3.3-70b-versatile', provider='gateway')
    assert urlparse(model._provider.base_url).hostname == urlparse(GATEWAY_BASE_URL).hostname  # pyright: ignore[reportPrivateUsage]

    model = GoogleModel('gemini-1.5-flash', provider='gateway')
    assert urlparse(model._provider.base_url).hostname == urlparse(GATEWAY_BASE_URL).hostname  # pyright: ignore[reportPrivateUsage]

    model = AnthropicModel('claude-sonnet-4-5', provider='gateway')
    assert urlparse(model._provider.base_url).hostname == urlparse(GATEWAY_BASE_URL).hostname  # pyright: ignore[reportPrivateUsage]

    model = BedrockConverseModel('amazon.nova-micro-v1:0', provider='gateway')
    assert urlparse(model._provider.base_url).hostname == urlparse(GATEWAY_BASE_URL).hostname  # pyright: ignore[reportPrivateUsage]


async def test_gateway_provider_endpoint(gateway_api_key: str):
    provider = gateway_provider('openai', route='potato', api_key=gateway_api_key, base_url=GATEWAY_BASE_URL)
    assert provider.client.base_url.path.endswith('/potato/')


@pytest.mark.parametrize(
    'api_key, expected_base_url',
    [
        pytest.param('pylf_v1_us_abc123', 'gateway-us.pydantic.dev', id='us-region'),
        pytest.param('pylf_v1_eu_abc123', 'gateway-eu.pydantic.dev', id='eu-region'),
        pytest.param('pylf_v1_stagingus_abc123', 'gateway.pydantic.info', id='staging'),
        pytest.param('pylf_v1_ap_abc123', 'gateway-ap.pydantic.dev', id='any-region'),
    ],
)
def test_infer_base_url(api_key: str, expected_base_url: str):
    provider = gateway_provider('openai', api_key=api_key)
    assert urlparse(provider.base_url).netloc == expected_base_url


def test_infer_base_url_no_region():
    """An API key that doesn't encode a region used to fall back to a shared Gateway URL; that URL
    is dead, so it now raises instead of silently routing to a dead host."""
    with raises(
        snapshot(
            'UserError: Could not infer the Pydantic AI Gateway base URL: the API key does not encode a region. '
            'Generate a new key from the Pydantic AI Gateway, or set the `PYDANTIC_AI_GATEWAY_BASE_URL` '
            'environment variable explicitly.'
        )
    ):
        gateway_provider('openai', api_key='not-a-pylf-token')
