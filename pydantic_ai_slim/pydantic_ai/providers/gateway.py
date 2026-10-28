"""This module implements the Pydantic AI Gateway provider."""

from __future__ import annotations as _annotations

import importlib.util
import os
import re
import weakref
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cache
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


@dataclass(frozen=True)
class _GatewayConfig:
    """What a gateway provider was created with, so a provider for another route format can be created like it."""

    api_key: str
    base_url: str
    """The base URL without the route."""
    route: str
    http_client: AsyncHTTPClient | None
    """The caller's HTTP client, or `None` when the provider created its own."""


_gateway_providers: weakref.WeakKeyDictionary[Provider[Any], _GatewayConfig] = weakref.WeakKeyDictionary()

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
        route: The name of the provider or gateway endpoint to use to handle the request. If not provided, the default
            gateway endpoint for the API format will be used.
        api_key: The API key to use for authentication. If not provided, the `PYDANTIC_AI_GATEWAY_API_KEY`
            environment variable will be used if available.
        base_url: The base URL to use for the Gateway. If not provided, the `PYDANTIC_AI_GATEWAY_BASE_URL`
            environment variable will be used if available. Otherwise, it is inferred from the API key's
            region, e.g. `https://gateway-us.pydantic.dev/proxy`.
        http_client: The HTTP client to use for the Gateway.
    """
    return _create_gateway_provider(
        upstream_provider, route=route, api_key=api_key, base_url=base_url, http_client=http_client
    )


def _create_gateway_provider(
    upstream_provider: str,
    *,
    route: str | None,
    api_key: str | None,
    base_url: str | None,
    http_client: AsyncHTTPClient | None,
) -> Provider[Any]:
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

    config = _GatewayConfig(api_key=api_key, base_url=base_url, route=route, http_client=http_client)
    base_url = _merge_url_path(base_url, route)

    # Bedrock uses the AWS SDK (botocore) rather than httpx, so skip http_client creation.
    if canonical == 'bedrock':
        from .bedrock import BedrockProvider

        provider = BedrockProvider(
            api_key=api_key,
            base_url=base_url,
            region_name='pydantic-ai-gateway',  # Fake region name to avoid NoRegionError
        )
        _register_bedrock_fallback_headers(provider.client)
        _gateway_providers[provider] = config
        provider._model_id_namespace = f'gateway/{provider.name}'  # pyright: ignore[reportPrivateUsage]
        return provider

    if canonical in ('openai', 'openai-chat', 'openai-responses'):
        from .openai import OpenAIProvider

        return _build_gateway_provider(
            lambda client: OpenAIProvider(api_key=api_key, base_url=base_url, http_client=client),
            config,
            http_client=http_client,
            create_http_client=create_async_httpx2_client,
        )

    if canonical == 'google-cloud':
        # `gateway/google` is a convenience alias for `gateway/google-cloud` — the Gateway
        # server only exposes the Google Cloud (Vertex) route today.
        from .google import _create_google_httpx2_client  # pyright: ignore[reportPrivateUsage]
        from .google_cloud import GoogleCloudProvider

        def build_google_provider(client: AsyncHTTPClient) -> GoogleCloudProvider:
            provider = GoogleCloudProvider(api_key=api_key, base_url=base_url, http_client=client)
            _set_google_ws_gateway_auth(provider.client, api_key)
            return provider

        return _build_gateway_provider(
            build_google_provider,
            config,
            http_client=http_client,
            create_http_client=_create_google_httpx2_client,
        )

    if canonical == 'anthropic':
        from anthropic import AsyncAnthropic

        from .anthropic import AnthropicProvider

        # Narrower than the other routes: `anthropic>=1` rejects a legacy client outright, so the
        # route can't fall back to one the way the OpenAI and Google routes still do.
        if http_client is not None and not _accepts_http_client(canonical, http_client):
            raise UserError('The Anthropic Gateway route requires an `httpx2.AsyncClient`.')

        def build_anthropic_provider(client: httpx2.AsyncClient) -> AnthropicProvider:
            return AnthropicProvider(
                anthropic_client=AsyncAnthropic(auth_token=api_key, base_url=base_url, http_client=client)
            )

        return _build_gateway_provider(
            build_anthropic_provider,
            config,
            http_client=http_client,
            create_http_client=create_async_httpx2_client,
        )

    if http_client is not None and not _accepts_http_client(canonical, http_client):
        raise UserError('`httpx2.AsyncClient` is only supported for OpenAI, Google and Anthropic Gateway routes.')

    if canonical == 'groq':
        from .groq import GroqProvider

        return _build_gateway_provider(
            lambda client: GroqProvider(api_key=api_key, base_url=base_url, http_client=client),
            config,
            http_client=http_client,
            create_http_client=create_async_http_client,
        )
    else:
        raise UserError(f'Unknown upstream provider: {upstream_provider}')


def _build_gateway_provider(
    build_provider: Callable[[_HTTPClientT], _ProviderT],
    config: _GatewayConfig,
    *,
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
        _add_request_hook(client, _GatewayRequestHook(config.api_key))
        return client

    own_http_client = http_client is None
    if http_client is None:
        http_client = create_hooked_http_client()
    else:
        _add_request_hook(http_client, _GatewayRequestHook(config.api_key))

    provider = build_provider(http_client)
    if own_http_client:
        provider._own_http_client = http_client  # pyright: ignore[reportPrivateUsage]
        provider._http_client_factory = create_hooked_http_client  # pyright: ignore[reportPrivateUsage]
    _gateway_providers[provider] = config
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

        if fallback_headers := _fallback_headers.get():
            request.headers.update(fallback_headers)

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


# Gateway-directed fallback across API formats.
#
# When a Gateway Endpoint fails over to a member in another API format than the request's, the gateway can't forward
# the request itself. A client that says it can rebuild the request for that format gets a `424` naming the model to
# send it to instead, plus a state token. The client sends the rebuilt request to the same route with the token, so
# the gateway resumes the endpoint's chain where it left off. Clients that don't say so have those members skipped.

_FALLBACK_STATUS_CODE = 424
"""The status of the gateway's hand-back response."""

_FALLBACK_HEADER = 'x-pydantic-ai-fallback'
_FALLBACK_STATE_HEADER = 'x-pydantic-ai-fallback-state'
_FALLBACK_MODEL_RESPONSE_HEADER = 'pydantic-ai-gateway-fallback-model'
_FALLBACK_STATE_RESPONSE_HEADER = 'pydantic-ai-gateway-fallback-state'
_FALLBACK_PROTOCOL_VERSION = 1

# The gateway's own API format IDs, and the SDK that a provider for each format needs.
_FALLBACK_FORMAT_SDKS: dict[str, str] = {
    'openai': 'openai',
    'anthropic': 'anthropic',
    'gemini': 'google.genai',
    'bedrock-converse': 'boto3',
}

_fallback_headers: ContextVar[Mapping[str, str] | None] = ContextVar('_gateway_fallback_headers', default=None)


@cache
def _accepted_fallback_formats() -> str:
    """The API formats this install can build a model for: those whose SDK is installed."""
    return ','.join(name for name, module in _FALLBACK_FORMAT_SDKS.items() if _module_available(module))


def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:  # pragma: lax no cover
        # The parent package of a dotted name, like `google` for `google.genai`, isn't installed.
        return False


@contextmanager
def _fallback_headers_scope(state: str | None) -> Generator[None]:
    """Tell the gateway, on the requests sent in this scope, that the client handles a fallback across API formats.

    `state` is the token of the hand-back that sent the request here, if any.
    """
    headers = {_FALLBACK_HEADER: f'{_FALLBACK_PROTOCOL_VERSION}; formats={_accepted_fallback_formats()}'}
    if state is not None:
        headers[_FALLBACK_STATE_HEADER] = state
    token = _fallback_headers.set(headers)
    try:
        yield
    finally:
        _fallback_headers.reset(token)


def _fallback_target(headers: Mapping[str, str] | None) -> tuple[str, str | None] | None:  # pyright: ignore[reportUnusedFunction]
    """The model a gateway hand-back response names, and its state token, read from the response's headers.

    Only a `gateway/<provider>:<model>` string counts: the request goes back to the same gateway, with the same
    credentials.
    """
    if headers is None or not (model := headers.get(_FALLBACK_MODEL_RESPONSE_HEADER)):
        return None
    if not (model.startswith('gateway/') and ':' in model):
        return None
    return model, headers.get(_FALLBACK_STATE_RESPONSE_HEADER)


def _fallback_provider(provider: Provider[Any], upstream_provider: str) -> Provider[Any]:  # pyright: ignore[reportUnusedFunction]
    """A gateway provider for `upstream_provider` with the credentials, base URL and route gateway `provider` has.

    The caller's HTTP client is shared when the new provider's SDK accepts it. Each is created once per `provider` and
    API format, so the requests the gateway hands back from it reuse one client, as those to `provider` itself do.
    """
    canonical = normalize_gateway_provider(upstream_provider)
    created = _fallback_providers.setdefault(provider, {})
    if (fallback := created.get(canonical)) is None:
        config = _gateway_providers[provider]
        fallback = created[canonical] = _create_gateway_provider(
            canonical,
            route=config.route,
            api_key=config.api_key,
            base_url=config.base_url,
            http_client=(
                config.http_client
                if config.http_client is not None and _accepts_http_client(canonical, config.http_client)
                else None
            ),
        )
    return fallback


_fallback_providers: weakref.WeakKeyDictionary[Provider[Any], dict[str, Provider[Any]]] = weakref.WeakKeyDictionary()


def _accepts_http_client(canonical: str, http_client: AsyncHTTPClient) -> bool:
    """Whether the route for `canonical` takes `http_client`: the one place each route's client support is decided.

    Bedrock goes through botocore, so it takes none; Anthropic takes only an `httpx2.AsyncClient`; the OpenAI and Google
    routes take either; the rest take only a legacy `httpx.AsyncClient`.
    """
    if canonical == 'bedrock':
        return False
    if canonical == 'anthropic':
        return isinstance(http_client, httpx2.AsyncClient)
    if canonical in ('openai', 'openai-chat', 'openai-responses', 'google-cloud'):
        return True
    return not isinstance(http_client, httpx2.AsyncClient)


def _register_bedrock_fallback_headers(client: BaseClient) -> None:
    """Send the fallback headers on Converse requests too, which go through botocore rather than httpx."""
    for operation in ('Converse', 'ConverseStream'):
        client.meta.events.register(
            f'before-call.bedrock-runtime.{operation}',
            _inject_bedrock_fallback_headers,
            unique_id=f'pydantic-ai-gateway-fallback-{operation}',
        )


def _inject_bedrock_fallback_headers(params: dict[str, Any], **_: Any) -> None:
    if fallback_headers := _fallback_headers.get():
        params['headers'].update(fallback_headers)


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
