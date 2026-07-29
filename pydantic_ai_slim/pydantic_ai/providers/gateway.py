"""This module implements the Pydantic AI Gateway provider."""

from __future__ import annotations as _annotations

import os
import re
from typing import TYPE_CHECKING, Any, Literal, overload

import httpx

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import create_async_http_client

if TYPE_CHECKING:
    from botocore.client import BaseClient
    from google.genai import Client as GoogleClient
    from groq import AsyncGroq
    from openai import AsyncOpenAI

    from pydantic_ai.providers import Provider
    from pydantic_ai.providers.anthropic import AsyncAnthropicClient


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
    upstream_provider: Literal['google', 'google-cloud'],
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


ModelProvider = Literal[
    'openai',
    'groq',
    'anthropic',
    'bedrock',
    'google',
    'google-cloud',
]


# These are only API flavors, we support them for convenience.
APIFlavor = Literal[
    'openai-chat',
    'openai-responses',
    'chat',
    'responses',
    'converse',
]

UpstreamProvider = ModelProvider | APIFlavor


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

    base_url = (
        base_url or os.getenv('PYDANTIC_AI_GATEWAY_BASE_URL', os.getenv('PAIG_BASE_URL')) or _infer_base_url(api_key)
    )

    canonical = normalize_gateway_provider(upstream_provider)
    if route is None:
        # Use the implied providerId as the default route.
        route = _gateway_route(canonical)

    base_url = _merge_url_path(base_url, route)

    # Bedrock uses the AWS SDK (botocore) rather than httpx, so skip http_client creation.
    if canonical == 'bedrock':
        from .bedrock import BedrockProvider

        return BedrockProvider(
            api_key=api_key,
            base_url=base_url,
            region_name='pydantic-ai-gateway',  # Fake region name to avoid NoRegionError
        )

    own_http_client = http_client is None
    http_client = http_client or create_async_http_client()
    _add_request_hook(http_client, _GatewayRequestHook(api_key))

    def _http_client_factory() -> httpx.AsyncClient:
        client = create_async_http_client()
        _add_request_hook(client, _GatewayRequestHook(api_key))
        return client

    def _with_http_client(provider: Provider[Any]) -> Provider[Any]:
        if own_http_client:
            provider._own_http_client = http_client  # pyright: ignore[reportPrivateUsage]
            provider._http_client_factory = _http_client_factory  # pyright: ignore[reportPrivateUsage]
        return provider

    if canonical in ('openai', 'openai-chat', 'openai-responses'):
        from .openai import OpenAIProvider

        return _with_http_client(OpenAIProvider(api_key=api_key, base_url=base_url, http_client=http_client))
    elif canonical == 'groq':
        from .groq import GroqProvider

        return _with_http_client(GroqProvider(api_key=api_key, base_url=base_url, http_client=http_client))
    elif canonical == 'anthropic':
        from anthropic import AsyncAnthropic

        from .anthropic import AnthropicProvider

        return _with_http_client(
            AnthropicProvider(
                anthropic_client=AsyncAnthropic(auth_token=api_key, base_url=base_url, http_client=http_client)
            )
        )
    elif canonical == 'google-cloud':
        # `gateway/google` is a convenience alias for `gateway/google-cloud` — the Gateway
        # server only exposes the Google Cloud (Vertex) route today, so both shorthands
        # land here via `normalize_gateway_provider`.
        from .google_cloud import GoogleCloudProvider

        return _with_http_client(GoogleCloudProvider(api_key=api_key, base_url=base_url, http_client=http_client))
    else:
        raise UserError(f'Unknown upstream provider: {upstream_provider}')


class _GatewayRequestHook:
    """Request hook for the gateway provider.

    It adds the `"traceparent"` and `"Authorization"` headers to the request. Implemented as a
    typed callable class (rather than a closure with a marker attribute) so that `_add_request_hook`
    can dedupe the gateway's own hook via `isinstance` on repeated calls with the same client.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def __call__(self, request: httpx.Request) -> httpx.Request:
        from opentelemetry.propagate import inject

        headers: dict[str, Any] = {}
        inject(headers)
        request.headers.update(headers)

        if 'Authorization' not in request.headers:
            request.headers['Authorization'] = f'Bearer {self._api_key}'

        return request


def _add_request_hook(http_client: httpx.AsyncClient, hook: _GatewayRequestHook) -> None:
    """Add a request hook without replacing caller-provided HTTPX hooks."""
    request_hooks = [
        existing_hook
        for existing_hook in http_client.event_hooks.get('request', [])
        if not isinstance(existing_hook, _GatewayRequestHook)
    ]
    request_hooks.append(hook)
    http_client.event_hooks['request'] = request_hooks


def _merge_url_path(base_url: str, path: str) -> str:
    """Merge a base URL and a path.

    Args:
        base_url: The base URL to merge.
        path: The path to merge.
    """
    return base_url.rstrip('/') + '/' + path.lstrip('/')


# Wire-value remaps for the PAIG URL route. Keyed by canonical class-lookup names
# (the output of `normalize_gateway_provider`); defaults to identity. Only providers
# whose Gateway wire value differs from the canonical name are listed.
# PAIG's canonical OpenAI route is `openai` (per the gateway's own 404 list of
# supported values). The Chat-vs-Responses API flavor is selected by the OpenAI
# SDK appending `/chat/completions` or `/responses` on top of the base URL, so all
# OpenAI flavors share the same wire route.
_GATEWAY_ROUTE_REMAP: dict[str, str] = {
    'openai-chat': 'openai',
    'openai-responses': 'openai',
    # Gateway team still uses the old name; flip this entry when they rename their side.
    'google-cloud': 'google-vertex',
}


def _gateway_route(provider: str) -> str:
    """Translate a canonical provider name into the Gateway URL route segment."""
    return _GATEWAY_ROUTE_REMAP.get(provider, provider)


# User-facing aliases resolved to canonical class-lookup names. `gateway/google` collapses
# onto `google-cloud` as a convenience — the Gateway server only exposes the Google Cloud
# (Vertex) route today, so both prefixes land on the same backend.
_GATEWAY_PROVIDER_ALIASES: dict[str, str] = {
    'chat': 'openai-chat',
    'responses': 'openai-responses',
    'converse': 'bedrock',
    'google': 'google-cloud',
}


def normalize_gateway_provider(provider: str) -> str:
    """Strip the `gateway/` prefix and resolve user-facing aliases to a canonical class-lookup name.

    Wire-value remapping for the Gateway URL belongs in `_gateway_route`.
    """
    provider = provider.removeprefix('gateway/')
    return _GATEWAY_PROVIDER_ALIASES.get(provider, provider)


_PYDANTIC_TOKEN_PATTERN = re.compile(r'^pylf_v(?P<version>[0-9]+)_(?P<region>[a-z]+)_[a-zA-Z0-9-_]+$')


def _infer_base_url(api_key: str) -> str:
    """Infer the Gateway base URL from the region encoded in the API key."""
    if match := _PYDANTIC_TOKEN_PATTERN.match(api_key):
        region = match.group('region')
        assert isinstance(region, str)

        if region.startswith('staging'):
            return 'https://gateway.pydantic.info/proxy'
        return f'https://gateway-{region}.pydantic.dev/proxy'

    raise UserError(
        'Could not infer the Pydantic AI Gateway base URL: the API key does not encode a region. '
        'Generate a new key from the Pydantic AI Gateway, or set the `PYDANTIC_AI_GATEWAY_BASE_URL` '
        'environment variable explicitly.'
    )
