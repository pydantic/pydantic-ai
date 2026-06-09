"""This module implements the Pydantic AI Gateway provider."""

from __future__ import annotations as _annotations

import os
import re
import warnings
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal, overload

import httpx

from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import create_async_http_client
from pydantic_ai.providers._gateway_auth import (
    DEFAULT_CIMD_PATH,
    GatewayAuth,
    GatewayAuthConfig,
    GatewayCredentials,
    auto_credentials,
    gateway_bedrock_session,
)

if TYPE_CHECKING:
    from botocore.client import BaseClient
    from google.genai import Client as GoogleClient
    from groq import AsyncGroq
    from openai import AsyncOpenAI

    from pydantic_ai.models.anthropic import AsyncAnthropicClient
    from pydantic_ai.providers import Provider

# An `httpx.Auth` or a `GatewayCredentials`; either resolves a (refreshing) bearer token.
GatewayAuthArg = httpx.Auth | GatewayCredentials


@overload
def gateway_provider(
    upstream_provider: Literal['openai', 'openai-chat', 'openai-responses', 'chat', 'responses'],
    /,
    *,
    route: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    auth: GatewayAuthArg | None = None,
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
    auth: GatewayAuthArg | None = None,
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
    auth: GatewayAuthArg | None = None,
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
    auth: GatewayAuthArg | None = None,
) -> Provider[BaseClient]: ...


@overload
def gateway_provider(
    upstream_provider: Literal['gemini', 'google-cloud', 'google-vertex'],
    /,
    *,
    route: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    auth: GatewayAuthArg | None = None,
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
    auth: GatewayAuthArg | None = None,
) -> Provider[Any]: ...


ModelProvider = Literal[
    'openai',
    'groq',
    'anthropic',
    'bedrock',
    'google-cloud',
]


# These are only API flavors, we support them for convenience.
APIFlavor = Literal[
    'openai-chat',
    'openai-responses',
    'chat',
    'responses',
    'converse',
    'gemini',
]

UpstreamProvider = ModelProvider | APIFlavor

# Placeholder key handed to upstream SDKs when auth is supplied dynamically; the real
# bearer token is injected per-request by `GatewayAuth`, which overrides this.
_DYNAMIC_AUTH_PLACEHOLDER = 'gateway-oauth'


def gateway_provider(
    upstream_provider: UpstreamProvider | str,
    /,
    *,
    # Every provider
    route: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    auth: GatewayAuthArg | None = None,
    # OpenAI, Groq, Anthropic & Gemini - Only Bedrock doesn't have an HTTPX client.
    http_client: httpx.AsyncClient | None = None,
) -> Provider[Any]:
    """Create a new Gateway provider.

    Args:
        upstream_provider: The upstream provider to use.
        route: The name of the provider or routing group to use to handle the request. If not provided, the default
            routing group for the API format will be used.
        api_key: A static API key to use for authentication. If not provided, the `PYDANTIC_AI_GATEWAY_API_KEY`
            environment variable will be used if available. If neither a key nor `auth` is provided, the provider
            auto-detects credentials: it exchanges an ambient workload OIDC token (e.g. in CI) for a short-lived
            gateway token, or — when attached to a terminal — runs an interactive device-authorization login.
        base_url: The base URL to use for the Gateway. If not provided, the `PYDANTIC_AI_GATEWAY_BASE_URL`
            environment variable will be used if available. Otherwise, it is inferred from a `pylf_*` key's region.
        auth: An explicit `httpx.Auth` or [`GatewayCredentials`][pydantic_ai.providers._gateway_auth.GatewayCredentials]
            to authenticate with, mirroring the token-provider pattern of other SDKs. Takes precedence over `api_key`.
        http_client: The HTTP client to use for the Gateway.
    """
    api_key = api_key or os.getenv('PYDANTIC_AI_GATEWAY_API_KEY', os.getenv('PAIG_API_KEY'))

    # Resolve `base_url` first: a static `pylf_*` key encodes its region, dynamic auth needs it explicitly.
    base_url = base_url or os.getenv('PYDANTIC_AI_GATEWAY_BASE_URL', os.getenv('PAIG_BASE_URL'))
    if base_url is None:
        if api_key is None:
            raise UserError(
                'Set `base_url` or the `PYDANTIC_AI_GATEWAY_BASE_URL` environment variable when using the '
                'Pydantic AI Gateway without a `pylf_*` API key.'
            )
        base_url = _infer_base_url(api_key)

    # Resolve auth: explicit `auth` > static key > auto-detected (OIDC / device flow) > error.
    resolved_auth: httpx.Auth | None
    if auth is not None:
        resolved_auth = auth if isinstance(auth, httpx.Auth) else GatewayAuth(auth)
    elif api_key is not None:
        resolved_auth = None  # static bearer, set directly on the upstream client
    else:
        resolved_auth = GatewayAuth(_auto_credentials(base_url))

    if route is None:
        # Use the implied providerId as the default route.
        route = normalize_gateway_provider(upstream_provider)

    base_url = _merge_url_path(base_url, route)

    # Bedrock uses the AWS SDK (botocore) rather than httpx, so it can't use `GatewayAuth`.
    if upstream_provider in ('bedrock', 'converse'):
        from .bedrock import BedrockProvider

        if resolved_auth is not None:
            return _bedrock_dynamic_provider(auth, base_url)
        assert api_key is not None  # `resolved_auth is None` only when a static key is set
        return BedrockProvider(
            api_key=api_key,
            base_url=base_url,
            region_name='pydantic-ai-gateway',  # Fake region name to avoid NoRegionError
        )

    return _httpx_provider(upstream_provider, api_key, base_url, resolved_auth, http_client)


def _httpx_provider(
    upstream_provider: str,
    api_key: str | None,
    base_url: str,
    resolved_auth: httpx.Auth | None,
    http_client: httpx.AsyncClient | None,
) -> Provider[Any]:
    """Build an HTTPX-based gateway provider (everything except Bedrock).

    For dynamic auth, hand the SDK a placeholder key; `GatewayAuth` (set on the HTTP client)
    overrides the `Authorization` header per request and refreshes it.
    """
    client_api_key = api_key if api_key is not None else _DYNAMIC_AUTH_PLACEHOLDER

    own_http_client = http_client is None
    http_client = http_client or create_async_http_client()
    http_client.event_hooks = {'request': [_request_hook(client_api_key)]}
    if resolved_auth is not None:
        http_client.auth = resolved_auth

    def _http_client_factory() -> httpx.AsyncClient:
        client = create_async_http_client()
        client.event_hooks = {'request': [_request_hook(client_api_key)]}
        if resolved_auth is not None:
            client.auth = resolved_auth
        return client

    def _with_http_client(provider: Provider[Any]) -> Provider[Any]:
        if own_http_client:
            provider._own_http_client = http_client  # pyright: ignore[reportPrivateUsage]
            provider._http_client_factory = _http_client_factory  # pyright: ignore[reportPrivateUsage]
        return provider

    if upstream_provider in ('openai', 'openai-chat', 'openai-responses', 'chat', 'responses'):
        from .openai import OpenAIProvider

        return _with_http_client(OpenAIProvider(api_key=client_api_key, base_url=base_url, http_client=http_client))
    elif upstream_provider == 'groq':
        from .groq import GroqProvider

        return _with_http_client(GroqProvider(api_key=client_api_key, base_url=base_url, http_client=http_client))
    elif upstream_provider == 'anthropic':
        from anthropic import AsyncAnthropic

        from .anthropic import AnthropicProvider

        return _with_http_client(
            AnthropicProvider(
                anthropic_client=AsyncAnthropic(auth_token=client_api_key, base_url=base_url, http_client=http_client)
            )
        )
    elif upstream_provider in ('google-cloud', 'google-vertex', 'gemini'):
        from .google_cloud import GoogleCloudProvider

        return _with_http_client(
            GoogleCloudProvider(api_key=client_api_key, base_url=base_url, http_client=http_client)
        )
    else:
        raise UserError(f'Unknown upstream provider: {upstream_provider}')


def _bedrock_dynamic_provider(auth: GatewayAuthArg | None, base_url: str) -> Provider[Any]:
    """Build a Bedrock provider whose botocore client auto-refreshes its gateway token.

    Bedrock can't use `GatewayAuth` (botocore, not httpx); instead botocore refreshes the
    bearer token per request via a `DeferredRefreshableToken` — see `gateway_bedrock_session`.
    """
    if not isinstance(auth, GatewayCredentials):
        raise UserError(
            'Bedrock through the gateway needs a `GatewayCredentials` (or a static `api_key`); '
            'a bare `httpx.Auth` cannot drive botocore signing.'
        )

    import boto3
    from botocore.config import Config

    from .bedrock import BedrockProvider

    session = boto3.Session(botocore_session=gateway_bedrock_session(auth), region_name='pydantic-ai-gateway')
    client = session.client(  # type: ignore[reportUnknownMemberType]
        'bedrock-runtime',
        config=Config(signature_version='bearer'),
        endpoint_url=base_url,
    )
    return BedrockProvider(bedrock_client=client)


def _auto_credentials(base_url: str) -> GatewayCredentials:
    """Auto-detect gateway credentials: ambient OIDC in CI, else an interactive device flow."""
    auth_base_url = os.getenv('PYDANTIC_AI_GATEWAY_AUTH_URL')
    if auth_base_url is None:
        # TODO(gateway-auth): derive the authorization-server URL from the gateway/region instead of
        # requiring this env var (e.g. via protected-resource metadata on the gateway base URL).
        raise UserError(
            'Set the `PYDANTIC_AI_GATEWAY_AUTH_URL` environment variable (the OAuth authorization server) to '
            'authenticate to the Pydantic AI Gateway without a static API key.'
        )

    config = GatewayAuthConfig(
        auth_base_url=auth_base_url,
        resource=base_url,
        client_id=os.getenv('PYDANTIC_AI_GATEWAY_CLIENT_ID', f'{auth_base_url.rstrip("/")}{DEFAULT_CIMD_PATH}'),
        oidc_audience=os.getenv('PYDANTIC_AI_GATEWAY_OIDC_AUDIENCE'),
    )
    return auto_credentials(config)


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
    provider = provider.removeprefix('gateway/')

    if provider == 'google-vertex':
        warnings.warn(
            "The 'gateway/google-vertex:' prefix is deprecated and will be removed in v2.0. "
            "Use 'gateway/google-cloud:' instead.",
            PydanticAIDeprecationWarning,
            stacklevel=2,
        )
    elif provider == 'gemini':
        warnings.warn(
            "The 'gateway/gemini:' prefix is deprecated and will be removed in v2.0. "
            "Use 'gateway/google-cloud:' instead.",
            PydanticAIDeprecationWarning,
            stacklevel=2,
        )

    if provider in ('openai', 'openai-chat', 'chat'):
        return 'openai'
    elif provider in ('openai-responses', 'responses'):
        return 'openai-responses'
    elif provider in ('gemini', 'google-cloud', 'google-vertex'):
        # The Gateway API still expects `google-vertex` as the upstream-provider wire value.
        # When the Gateway team renames their side, flip this to `google-cloud`.
        return 'google-vertex'
    elif provider in ('bedrock', 'converse'):
        return 'bedrock'
    return provider


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
