"""This module implements the Pydantic AI Gateway provider."""

from __future__ import annotations as _annotations

import os
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal, overload

import httpx

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client

if TYPE_CHECKING:
    from botocore.client import BaseClient
    from google.genai import Client as GoogleClient
    from groq import AsyncGroq
    from openai import AsyncOpenAI

    from pydantic_ai.models.anthropic import AsyncAnthropicClient
    from pydantic_ai.providers import Provider

GATEWAY_BASE_URL = 'https://gateway.pydantic.dev/proxy'


@overload
def gateway_provider(
    upstream_provider: Literal['openai', 'openai-chat', 'openai-responses', 'chat', 'responses'],
    /,
    *,
    route: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> Provider[AsyncOpenAI]: ...


@overload
def gateway_provider(
    upstream_provider: Literal['groq'],
    /,
    *,
    route: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> Provider[AsyncGroq]: ...


@overload
def gateway_provider(
    upstream_provider: Literal['anthropic'],
    /,
    *,
    route: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> Provider[AsyncAnthropicClient]: ...


@overload
def gateway_provider(
    upstream_provider: Literal['bedrock', 'converse'],
    /,
    *,
    route: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Provider[BaseClient]: ...


@overload
def gateway_provider(
    upstream_provider: Literal['gemini', 'google-vertex'],
    /,
    *,
    route: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> Provider[GoogleClient]: ...


@overload
def gateway_provider(
    upstream_provider: str,
    /,
    *,
    route: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Provider[Any]: ...


UpstreamProvider = Literal[
    'openai',
    'groq',
    'anthropic',
    'bedrock',
    'google-vertex',
    # Those are only API formats, but we still support them for convenience.
    'openai-chat',
    'openai-responses',
    'chat',
    'responses',
    'converse',
    'gemini',
]


def gateway_provider(
    upstream_provider: UpstreamProvider | str,
    /,
    *,
    # Every provider
    route: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    # OpenAI, Groq, Anthropic & Gemini - Only Bedrock doesn't have an HTTPX client.
    http_client: httpx.AsyncClient | None = None,
) -> Provider[Any]:
    """Create a new Gateway provider.

    Args:
        upstream_provider: The upstream provider to use.
        route: The name of the provider or routing group to use to handle the request. If not provided, the default
            routing group for the API format will be used.
        api_key: The API key to use for authentication. If not provided, the `PYDANTIC_AI_GATEWAY_API_KEY`
            environment variable will be used if available.
        base_url: The base URL to use for the Gateway. If not provided, the `PYDANTIC_AI_GATEWAY_BASE_URL`
            environment variable will be used if available. Otherwise, defaults to `https://gateway.pydantic.dev/proxy`.
        http_client: The HTTP client to use for the Gateway.
    """
    api_key = api_key or os.getenv('PYDANTIC_AI_GATEWAY_API_KEY', os.getenv('PAIG_API_KEY'))
    if not api_key:
        raise UserError(
            'Set the `PYDANTIC_AI_GATEWAY_API_KEY` environment variable or pass it via `gateway_provider(..., api_key=...)`'
            ' to use the Pydantic AI Gateway provider.'
        )

    base_url = base_url or os.getenv('PYDANTIC_AI_GATEWAY_BASE_URL', os.getenv('PAIG_BASE_URL', GATEWAY_BASE_URL))
    http_client = http_client or cached_async_http_client(provider=f'gateway/{upstream_provider}')
    http_client.event_hooks = {'request': [_request_hook(api_key)]}

    if route is None:
        # Use the implied providerId as the default route.
        route = normalize_gateway_provider(upstream_provider)

    base_url = _merge_url_path(base_url, route)

    if upstream_provider in ('openai', 'openai-chat', 'openai-responses', 'chat', 'responses'):
        from .openai import OpenAIProvider

        return OpenAIProvider(api_key=api_key, base_url=base_url, http_client=http_client)
    elif upstream_provider == 'groq':
        from .groq import GroqProvider

        return GroqProvider(api_key=api_key, base_url=base_url, http_client=http_client)
    elif upstream_provider == 'anthropic':
        from anthropic import AsyncAnthropic

        from .anthropic import AnthropicProvider

        return AnthropicProvider(
            anthropic_client=AsyncAnthropic(auth_token=api_key, base_url=base_url, http_client=http_client)
        )
    elif upstream_provider in ('bedrock', 'converse'):
        from .bedrock import BedrockProvider

        return BedrockProvider(
            api_key=api_key,
            base_url=base_url,
            region_name='pydantic-ai-gateway',  # Fake region name to avoid NoRegionError
        )
    elif upstream_provider in ('google-vertex', 'gemini'):
        from .google import GoogleProvider

        return GoogleProvider(vertexai=True, api_key=api_key, base_url=base_url, http_client=http_client)
    else:
        raise UserError(f'Unknown upstream provider: {upstream_provider}')


def _request_hook(api_key: str) -> Callable[[httpx.Request], Awaitable[httpx.Request]]:
    """Request hook for the gateway provider.

    It adds the `"traceparent"` and `"Authorization"` headers to the request.
    """

    async def _hook(request: httpx.Request) -> httpx.Request:
        from opentelemetry.propagate import inject

        headers: dict[str, Any] = {}
        inject(headers)
        request.headers.update(headers)

        if 'Authorization' not in request.headers:
            request.headers['Authorization'] = f'Bearer {api_key}'

        return request

    return _hook


def _merge_url_path(base_url: str, path: str) -> str:
    """Merge a base URL and a path.

    Args:
        base_url: The base URL to merge.
        path: The path to merge.
    """
    return base_url.rstrip('/') + '/' + path.lstrip('/')


def normalize_gateway_provider(provider: str) -> str:
    """Normalize a gateway provider name.

    Args:
        provider: The provider name to normalize.
    """
    if provider in ('openai', 'openai-chat', 'chat'):
        return 'openai'
    elif provider in ('openai-responses', 'responses'):
        return 'openai-responses'
    elif provider in ('gemini', 'google-vertex'):
        return 'google-vertex'
    elif provider in ('bedrock', 'converse'):
        return 'bedrock'
    return provider
