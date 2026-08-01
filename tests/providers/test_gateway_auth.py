from __future__ import annotations

from typing import Literal

import httpx
import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError

from ..conftest import try_import

with try_import() as imports_successful:
    from google.genai import Client as GoogleClient

    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers import Provider
    from pydantic_ai.providers.gateway import gateway_provider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='providers not installed'),
    pytest.mark.anyio,
]

GatewayProviderName = Literal['google-cloud', 'openai']


@pytest.mark.parametrize(
    'provider_order',
    [
        ('google-cloud', 'openai'),
        ('openai', 'google-cloud'),
    ],
)
async def test_shared_http_client_scopes_gateway_auth(
    allow_model_requests: None,
    provider_order: tuple[GatewayProviderName, GatewayProviderName],
):
    """Gateway auth is tested locally because VCR filters the credential headers under test.

    `MockTransport` captures the request produced by google-genai after HTTPX runs the real
    request hooks, including caller-provided default headers and all shared-client registrations.
    """
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith(':generateContent'):
            return httpx.Response(
                200,
                json={
                    'candidates': [{'content': {'parts': [{'text': 'OK'}], 'role': 'model'}, 'finishReason': 'STOP'}],
                    'modelVersion': 'gemini-2.5-flash',
                    'responseId': 'response-id',
                    'usageMetadata': {
                        'promptTokenCount': 1,
                        'candidatesTokenCount': 1,
                        'totalTokenCount': 2,
                    },
                },
            )
        return httpx.Response(200)

    google_provider: Provider[GoogleClient] | None = None
    async with httpx.AsyncClient(
        headers={'Authorization': 'Bearer unrelated'},
        transport=httpx.MockTransport(handler),
    ) as http_client:
        for provider_name in provider_order:
            if provider_name == 'google-cloud':
                google_provider = gateway_provider(
                    'google-cloud',
                    api_key='google-cloud-gateway-key',
                    base_url='https://example.com/proxy',
                    http_client=http_client,
                )
            else:
                gateway_provider(
                    'openai',
                    api_key='openai-gateway-key',
                    base_url='https://example.com/proxy',
                    http_client=http_client,
                )

        assert google_provider is not None
        model = GoogleModel('gemini-2.5-flash', provider=google_provider)
        result = await Agent(model).run('Reply only with OK.')
        await http_client.post('https://example.com/proxy/openai/chat/completions')
        await http_client.get('https://example.com/proxy/google-vertex-other')

    assert result.output == 'OK'
    assert len(requests) == 3

    google_request, openai_request, unrelated_request = requests
    assert google_request.headers['Authorization'] == 'Bearer google-cloud-gateway-key'
    assert 'X-Goog-Api-Key' not in google_request.headers
    assert openai_request.headers['Authorization'] == 'Bearer openai-gateway-key'
    assert unrelated_request.headers['Authorization'] == 'Bearer unrelated'


async def test_shared_http_client_uses_most_specific_gateway_route():
    """Unit (not VCR): overlapping custom route dispatch is local hook behavior."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200)

    async with httpx.AsyncClient(
        headers={'Authorization': 'Bearer unrelated'}, transport=httpx.MockTransport(handler)
    ) as http_client:
        gateway_provider(
            'openai',
            route='shared',
            api_key='parent-key',
            base_url='https://example.com/proxy',
            http_client=http_client,
        )
        gateway_provider(
            'google-cloud',
            route='shared/google',
            api_key='nested-key',
            base_url='https://example.com/proxy',
            http_client=http_client,
        )
        await http_client.post(
            'https://example.com/proxy/shared/v1/chat/completions',
            headers={'Authorization': 'Bearer unrelated'},
        )
        await http_client.post(
            'https://example.com/proxy/shared/google/v1/models/test:generateContent',
            headers={'X-Goog-Api-Key': 'nested-key', 'Authorization': 'Bearer unrelated'},
        )

    parent_request, nested_request = requests
    assert parent_request.headers['Authorization'] == 'Bearer parent-key'
    assert nested_request.headers['Authorization'] == 'Bearer nested-key'
    assert 'X-Goog-Api-Key' not in nested_request.headers


async def test_shared_http_client_rejects_ambiguous_gateway_auth():
    """Unit (not VCR): ambiguous local hook dispatch fails before making a request."""
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200))) as http_client:
        gateway_provider(
            'openai',
            api_key='first-key',
            base_url='https://example.com/proxy',
            http_client=http_client,
        )
        gateway_provider(
            'openai',
            api_key='second-key',
            base_url='https://example.com/proxy',
            http_client=http_client,
        )
        request = httpx.Request('POST', 'https://example.com/proxy/openai/v1/responses')

        with pytest.raises(UserError, match='Gateway authentication is ambiguous'):
            await http_client.send(request)
