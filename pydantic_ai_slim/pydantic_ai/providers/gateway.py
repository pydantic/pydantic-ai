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

    from pydantic_ai.models import Model
    from pydantic_ai.models.anthropic import AsyncAnthropicClient
    from pydantic_ai.providers import Provider

GATEWAY_BASE_URL = 'https://gateway.pydantic.dev/proxy'


@overload
def gateway_provider(
    api_type: Literal['chat', 'responses'],
    /,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> Provider[AsyncOpenAI]: ...


@overload
def gateway_provider(
    api_type: Literal['groq'],
    /,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> Provider[AsyncGroq]: ...


@overload
def gateway_provider(
    api_type: Literal['anthropic'],
    /,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> Provider[AsyncAnthropicClient]: ...


@overload
def gateway_provider(
    api_type: Literal['converse'],
    /,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Provider[BaseClient]: ...


@overload
def gateway_provider(
    api_type: Literal['gemini'],
    /,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> Provider[GoogleClient]: ...


@overload
def gateway_provider(
    api_type: str,
    /,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Provider[Any]: ...


APIType = Literal['chat', 'responses', 'gemini', 'converse', 'anthropic', 'groq']


def gateway_provider(
    api_type: APIType | str,
    /,
    *,
    # Every provider
    api_key: str | None = None,
    base_url: str | None = None,
    # OpenAI, Groq, Anthropic & Gemini - Only Bedrock doesn't have an HTTPX client.
    http_client: httpx.AsyncClient | None = None,
) -> Provider[Any]:
    """Create a new Gateway provider.

    Args:
        api_type: Determines the API type to use.
        api_key: The API key to use for authentication. If not provided, the `PYDANTIC_AI_GATEWAY_API_KEY`
            environment variable will be used if available.
        base_url: The base URL to use for the Gateway. If not provided, the `PYDANTIC_AI_GATEWAY_BASE_URL`
            environment variable will be used if available. Otherwise, defaults to `https://gateway.pydantic.dev/proxy`.
        http_client: The HTTP client to use for the Gateway.
    """
    api_key = api_key or os.getenv('PYDANTIC_AI_GATEWAY_API_KEY')
    if not api_key:
        raise UserError(
            'Set the `PYDANTIC_AI_GATEWAY_API_KEY` environment variable or pass it via `gateway_provider(..., api_key=...)`'
            ' to use the Pydantic AI Gateway provider.'
        )

    base_url = base_url or os.getenv('PYDANTIC_AI_GATEWAY_BASE_URL', GATEWAY_BASE_URL)
    http_client = http_client or cached_async_http_client(provider=f'gateway/{api_type}')
    http_client.event_hooks = {'request': [_request_hook(api_key)]}

    if api_type in ('chat', 'responses'):
        from .openai import OpenAIProvider

        return OpenAIProvider(api_key=api_key, base_url=_merge_url_path(base_url, api_type), http_client=http_client)
    elif api_type == 'groq':
        from .groq import GroqProvider

        return GroqProvider(api_key=api_key, base_url=_merge_url_path(base_url, 'groq'), http_client=http_client)
    elif api_type == 'anthropic':
        from anthropic import AsyncAnthropic

        from .anthropic import AnthropicProvider

        return AnthropicProvider(
            anthropic_client=AsyncAnthropic(
                auth_token=api_key,
                base_url=_merge_url_path(base_url, 'anthropic'),
                http_client=http_client,
            )
        )
    elif api_type in 'converse':
        from .bedrock import BedrockProvider

        return BedrockProvider(
            api_key=api_key,
            base_url=_merge_url_path(base_url, api_type),
            region_name='pydantic-ai-gateway',  # Fake region name to avoid NoRegionError
        )
    elif api_type in 'gemini':
        from .google import GoogleProvider

        return GoogleProvider(
            vertexai=True,
            api_key=api_key,
            base_url=_merge_url_path(base_url, 'gemini'),
            http_client=http_client,
        )
    else:
        raise UserError(f'Unknown API type: {api_type}')


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


def infer_gateway_model(api_type: APIType | str, *, model_name: str) -> Model:
    """Infer the model class for a given API type."""
    if api_type == 'chat':
        from pydantic_ai.models.openai import OpenAIChatModel

        return OpenAIChatModel(model_name=model_name, provider='gateway')
    elif api_type == 'groq':
        from pydantic_ai.models.groq import GroqModel

        return GroqModel(model_name=model_name, provider='gateway')
    elif api_type == 'responses':
        from pydantic_ai.models.openai import OpenAIResponsesModel

        return OpenAIResponsesModel(model_name=model_name, provider='gateway')
    elif api_type == 'gemini':
        from pydantic_ai.models.google import GoogleModel

        return GoogleModel(model_name=model_name, provider='gateway')
    elif api_type == 'converse':
        from pydantic_ai.models.bedrock import BedrockConverseModel

        return BedrockConverseModel(model_name=model_name, provider='gateway')
    elif api_type == 'anthropic':
        from pydantic_ai.models.anthropic import AnthropicModel

        return AnthropicModel(model_name=model_name, provider='gateway')
    else:
        raise ValueError(f'Unknown API type: {api_type}')  # pragma: no cover
