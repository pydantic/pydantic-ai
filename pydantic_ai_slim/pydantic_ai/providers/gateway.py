"""This module implements the Pydantic AI Gateway provider."""

from __future__ import annotations as _annotations

import os
from typing import TYPE_CHECKING, Any, Literal, overload

import httpx

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model, cached_async_http_client, get_user_agent

if TYPE_CHECKING:
    from botocore.client import BaseClient
    from google.genai import Client as GoogleClient
    from groq import AsyncGroq
    from openai import AsyncOpenAI

    from pydantic_ai.models.anthropic import AsyncAnthropicClient
    from pydantic_ai.providers import Provider


@overload
def gateway_provider(
    upstream_provider: Literal['openai', 'openai-chat', 'openai-responses'],
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> Provider[AsyncOpenAI]: ...


@overload
def gateway_provider(
    upstream_provider: Literal['groq'],
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> Provider[AsyncGroq]: ...


@overload
def gateway_provider(
    upstream_provider: Literal['google-vertex'],
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Provider[GoogleClient]: ...


@overload
def gateway_provider(
    upstream_provider: Literal['anthropic'],
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Provider[AsyncAnthropicClient]: ...


@overload
def gateway_provider(
    upstream_provider: Literal['bedrock'],
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Provider[BaseClient]: ...


UpstreamProvider = Literal['openai', 'openai-chat', 'openai-responses', 'groq', 'google-vertex', 'anthropic', 'bedrock']


def gateway_provider(
    upstream_provider: UpstreamProvider | str,
    *,
    # Every provider
    api_key: str | None = None,
    base_url: str | None = None,
    # OpenAI, Groq & Anthropic
    http_client: httpx.AsyncClient | None = None,
) -> Provider[Any]:
    """Create a new Gateway provider.

    Args:
        upstream_provider: The upstream provider to use.
        api_key: The API key to use for authentication. If not provided, the `PYDANTIC_AI_GATEWAY_API_KEY`
            environment variable will be used if available.
        base_url: The base URL to use for the Gateway. If not provided, the `PYDANTIC_AI_GATEWAY_BASE_URL`
            environment variable will be used if available. Otherwise, defaults to `https://gateway.pydantic.dev/proxy`.
        http_client: The HTTP client to use for the Gateway.
    """
    api_key = api_key or os.getenv('PYDANTIC_AI_GATEWAY_API_KEY')
    if not api_key:
        raise UserError(
            'Set the `PYDANTIC_AI_GATEWAY_API_KEY` environment variable or pass it via `gateway_provider(api_key=...)`'
            ' to use the Pydantic AI Gateway provider.'
        )

    base_url = base_url or os.getenv('PYDANTIC_AI_GATEWAY_BASE_URL', 'https://gateway.pydantic.dev/proxy')
    http_client = http_client or cached_async_http_client(provider=f'gateway-{upstream_provider}')
    http_client.event_hooks = {'request': [_request_hook]}

    if upstream_provider in ('openai', 'openai-chat'):
        from .openai import OpenAIProvider

        return OpenAIProvider(api_key=api_key, base_url=_merge_url_path(base_url, 'openai'), http_client=http_client)
    elif upstream_provider == 'openai-responses':
        from .openai import OpenAIProvider

        return OpenAIProvider(api_key=api_key, base_url=_merge_url_path(base_url, 'openai'), http_client=http_client)
    elif upstream_provider == 'groq':
        from .groq import GroqProvider

        return GroqProvider(api_key=api_key, base_url=_merge_url_path(base_url, 'groq'), http_client=http_client)
    elif upstream_provider == 'anthropic':
        from anthropic import AsyncAnthropic

        from .anthropic import AnthropicProvider

        return AnthropicProvider(
            anthropic_client=AsyncAnthropic(
                auth_token=api_key,
                base_url=_merge_url_path(base_url, 'anthropic'),
                http_client=http_client,
            )
        )
    elif upstream_provider == 'bedrock':
        from .bedrock import BedrockProvider

        return BedrockProvider(
            api_key=api_key,
            base_url=_merge_url_path(base_url, 'bedrock'),
            region_name='pydantic-ai-gateway',  # Fake region name to avoid NoRegionError
        )
    elif upstream_provider == 'google-vertex':
        from google.genai import Client as GoogleClient

        from .google import GoogleProvider

        return GoogleProvider(
            client=GoogleClient(
                vertexai=True,
                api_key='unset',
                http_options={
                    'base_url': _merge_url_path(base_url, 'google-vertex'),
                    'headers': {'User-Agent': get_user_agent(), 'Authorization': api_key},
                    # TODO(Marcelo): Until https://github.com/googleapis/python-genai/issues/1357 is solved.
                    'async_client_args': {
                        'transport': httpx.AsyncHTTPTransport(),
                        'event_hooks': {'request': [_request_hook]},
                    },
                },
            )
        )
    else:  # pragma: no cover
        raise UserError(f'Unknown provider: {upstream_provider}')


def infer_model(model_name: str) -> Model:
    """Infer the model class that will be used to make requests to the gateway.

    Args:
        model_name: The name of the model to infer. Must be in the format "provider/model_name".

    Returns:
        The model class that will be used to make requests to the gateway.
    """
    try:
        upstream_provider, model_name = model_name.split('/', 1)
    except ValueError:
        raise UserError(f'The model name "{model_name}" is not in the format "provider/model_name".')

    if upstream_provider in ('openai', 'openai-chat'):
        from pydantic_ai.models.openai import OpenAIChatModel

        return OpenAIChatModel(model_name, provider=gateway_provider('openai'))
    elif upstream_provider == 'openai-responses':
        from pydantic_ai.models.openai import OpenAIResponsesModel

        return OpenAIResponsesModel(model_name, provider=gateway_provider('openai'))
    elif upstream_provider == 'groq':
        from pydantic_ai.models.groq import GroqModel

        return GroqModel(model_name, provider=gateway_provider('groq'))
    elif upstream_provider == 'anthropic':
        from pydantic_ai.models.anthropic import AnthropicModel

        return AnthropicModel(model_name, provider=gateway_provider('anthropic'))
    elif upstream_provider == 'google-vertex':
        from pydantic_ai.models.google import GoogleModel

        return GoogleModel(model_name, provider=gateway_provider('google-vertex'))
    raise UserError(f'Unknown upstream provider: {upstream_provider}')


async def _request_hook(request: httpx.Request) -> httpx.Request:
    """Request hook for the gateway provider.

    It adds the `"traceparent"` header to the request.
    """
    from opentelemetry.propagate import inject

    headers: dict[str, Any] = {}
    inject(headers)
    request.headers.update(headers)

    return request


def _merge_url_path(base_url: str, path: str) -> str:
    """Merge a base URL and a path.

    Args:
        base_url: The base URL to merge.
        path: The path to merge.
    """
    return base_url.rstrip('/') + '/' + path.lstrip('/')
