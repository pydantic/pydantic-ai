"""This module implements the Pydantic AI Gateway provider."""

from __future__ import annotations as _annotations

import os
import re
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, overload

import httpx2
from typing_extensions import TypeVar

from pydantic_ai._http import AsyncHTTPClient, create_async_httpx2_client
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import create_async_http_client

if TYPE_CHECKING:
    import httpx
    from botocore.client import BaseClient
    from google.genai import Client as GoogleClient
    from groq import AsyncGroq
    from openai import AsyncOpenAI

    from pydantic_ai.providers import Provider
    from pydantic_ai.providers.anthropic import AsyncAnthropicClient


_gateway_providers: weakref.WeakSet[Provider[Any]] = weakref.WeakSet()

_ProviderT = TypeVar('_ProviderT', bound='Provider[Any]')
_HTTPClientT = TypeVar('_HTTPClientT', bound=AsyncHTTPClient)


@overload
def gateway_provider(
    upstream_provider: Literal['openai', 'openai-chat', 'openai-responses', 'chat', 'responses'],
    /,
    *,
    route: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    http_client: AsyncHTTPClient | None = None,
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
    http_client: httpx2.AsyncClient | None = None,
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
    http_client: AsyncHTTPClient | None = None,
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
    http_client: AsyncHTTPClient | None = None,
) -> Provider[Any]:
    """Create a new Gateway provider.

    Args:
        upstream_provider: The upstream provider to use.
        route: The name of the provider or routing group to use to handle the request. If not provided, the default
            routing group for the API format will be used.
        api_key: The API key to use for authentication. If not provided, the `PYDANTIC_AI_GATEWAY_API_KEY`
            environment variable will be used if available.
        base_url: The base URL to use for the Gateway. If not provided, the `PYDANTIC_AI_GATEWAY_BASE_URL`
            environment variable will be used if available. Otherwise, it is inferred from the API key's
            region, e.g. `https://gateway-us.pydantic.dev/proxy`.
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

        provider = BedrockProvider(
            api_key=api_key,
            base_url=base_url,
            region_name='pydantic-ai-gateway',  # Fake region name to avoid NoRegionError
        )
        _gateway_providers.add(provider)
        provider._model_id_namespace = f'gateway/{provider.name}'  # pyright: ignore[reportPrivateUsage]
        return provider

    if canonical in ('openai', 'openai-chat', 'openai-responses'):
        from .openai import OpenAIProvider

        return _build_gateway_provider(
            lambda client: OpenAIProvider(api_key=api_key, base_url=base_url, http_client=client),
            api_key=api_key,
            http_client=http_client,
            create_http_client=create_async_httpx2_client,
        )

    if canonical == 'google-cloud':
        # `gateway/google` is a convenience alias for `gateway/google-cloud` — the Gateway
        # server only exposes the Google Cloud (Vertex) route today.
        from .google_cloud import GoogleCloudProvider

        def build_google_provider(client: AsyncHTTPClient) -> GoogleCloudProvider:
            provider = GoogleCloudProvider(api_key=api_key, base_url=base_url, http_client=client)
            _set_google_ws_gateway_auth(provider.client, api_key)
            return provider

        return _build_gateway_provider(
            build_google_provider,
            api_key=api_key,
            http_client=http_client,
            create_http_client=create_async_httpx2_client,
        )

    if canonical == 'anthropic':
        from anthropic import AsyncAnthropic

        from .anthropic import AnthropicProvider

        # Narrower than the other routes: `anthropic>=1` rejects a legacy client outright, so the
        # route can't fall back to one the way the OpenAI and Google routes still do.
        if http_client is not None and not isinstance(http_client, httpx2.AsyncClient):
            raise UserError('The Anthropic Gateway route requires an `httpx2.AsyncClient`.')

        def build_anthropic_provider(client: httpx2.AsyncClient) -> AnthropicProvider:
            return AnthropicProvider(
                anthropic_client=AsyncAnthropic(auth_token=api_key, base_url=base_url, http_client=client)
            )

        return _build_gateway_provider(
            build_anthropic_provider,
            api_key=api_key,
            http_client=http_client,
            create_http_client=create_async_httpx2_client,
        )

    if isinstance(http_client, httpx2.AsyncClient):
        raise UserError('`httpx2.AsyncClient` is only supported for OpenAI, Google and Anthropic Gateway routes.')

    if canonical == 'groq':
        from .groq import GroqProvider

        return _build_gateway_provider(
            lambda client: GroqProvider(api_key=api_key, base_url=base_url, http_client=client),
            api_key=api_key,
            http_client=http_client,
            create_http_client=create_async_http_client,
        )
    else:
        raise UserError(f'Unknown upstream provider: {upstream_provider}')


def _build_gateway_provider(
    build_provider: Callable[[_HTTPClientT], _ProviderT],
    *,
    api_key: str,
    http_client: _HTTPClientT | None,
    create_http_client: Callable[[], _HTTPClientT],
) -> _ProviderT:
    """Build a provider on an HTTP client that carries the Gateway's auth hook.

    Shared by every Gateway route that speaks HTTP (all but Bedrock, which goes through botocore): a
    caller-provided client is hooked in place and stays the caller's to close, while a client we create is
    owned by the provider, which recreates it — hook included — when it is re-entered after being closed.
    """

    def create_hooked_http_client() -> _HTTPClientT:
        client = create_http_client()
        _add_request_hook(client, _GatewayRequestHook(api_key))
        return client

    own_http_client = http_client is None
    if http_client is None:
        http_client = create_hooked_http_client()
    else:
        _add_request_hook(http_client, _GatewayRequestHook(api_key))

    provider = build_provider(http_client)
    if own_http_client:
        provider._own_http_client = http_client  # pyright: ignore[reportPrivateUsage]
        provider._http_client_factory = create_hooked_http_client  # pyright: ignore[reportPrivateUsage]
    _gateway_providers.add(provider)
    provider._model_id_namespace = f'gateway/{provider.name}'  # pyright: ignore[reportPrivateUsage]
    return provider


def is_gateway_provider(provider: Provider[Any]) -> bool:
    """Whether `provider` routes requests through the Pydantic AI Gateway.

    True for any provider created by `gateway_provider(...)`, whether it reached the caller as the
    `gateway/<name>` string (resolved via `infer_provider`) or as a `gateway_provider(...)` instance.
    """
    try:
        return provider in _gateway_providers
    except TypeError:
        # A `Provider` is free to be an ordinary `@dataclass`, which sets `__hash__ = None` and makes
        # the set lookup raise rather than answer. It can't be one of ours either way — everything in
        # here was put there by `gateway_provider(...)` — so the answer is simply no.
        return False


def _set_google_ws_gateway_auth(client: GoogleClient, api_key: str) -> None:
    """Set the gateway bearer auth as a static header on the Google client so it reaches the Live WebSocket.

    The gateway authenticates on `Authorization: Bearer <key>`, which its `httpx` request hook adds to REST
    calls. That hook can't cover the Gemini Live handshake: `google-genai` dials the WebSocket with the
    `websockets` library, bypassing the provider's `httpx` client. The SDK forwards
    `client._api_client._http_options.headers` to *both* REST and the Live handshake, so setting the bearer
    there once — permanently — is what carries it onto the WebSocket. REST then carries it too, which is
    redundant with the httpx request hook but harmless: it's the same value, and the hook already leaves a
    pre-existing `Authorization` header untouched.

    Guarded with `getattr` chains: a custom/fake client without the SDK's private HTTP options simply skips
    this, and a pre-existing `Authorization` header is left in place.
    """
    raw_headers = getattr(getattr(getattr(client, '_api_client', None), '_http_options', None), 'headers', None)
    if not isinstance(raw_headers, dict) or 'Authorization' in raw_headers:
        return
    raw_headers['Authorization'] = f'Bearer {api_key}'


class _GatewayRequestHook:
    """Request hook for the gateway provider.

    It adds the `"traceparent"` and `"Authorization"` headers to the request. Implemented as a
    typed callable class (rather than a closure with a marker attribute) so that `_add_request_hook`
    can dedupe the gateway's own hook via `isinstance` on repeated calls with the same client.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    @overload
    async def __call__(self, request: httpx.Request) -> httpx.Request: ...

    @overload
    async def __call__(self, request: httpx2.Request) -> httpx2.Request: ...

    async def __call__(self, request: httpx.Request | httpx2.Request) -> httpx.Request | httpx2.Request:
        from opentelemetry.propagate import inject

        headers: dict[str, Any] = {}
        inject(headers)
        request.headers.update(headers)

        if 'Authorization' not in request.headers:
            request.headers['Authorization'] = f'Bearer {self._api_key}'

        return request


def _add_request_hook(http_client: AsyncHTTPClient, hook: _GatewayRequestHook) -> None:
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
