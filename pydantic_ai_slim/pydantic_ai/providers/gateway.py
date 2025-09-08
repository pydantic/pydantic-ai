"""This module implements the Pydantic AI Gateway provider."""

from __future__ import annotations as _annotations

import os
from typing import TYPE_CHECKING, Any, Literal, overload
from urllib.parse import urljoin

import httpx

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import get_user_agent
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.providers import InterfaceClient, Provider

if TYPE_CHECKING:
    from google.auth.credentials import Credentials
    from google.genai import Client as GoogleClient
    from groq import AsyncGroq
    from openai import AsyncOpenAI

    from pydantic_ai.providers.google import VertexAILocation


class GatewayProvider(Provider[InterfaceClient]):
    """Provider to access the Pydantic AI Gateway."""

    @property
    def name(self) -> str: ...  # pragma: no cover

    @property
    def base_url(self) -> str: ...  # pragma: no cover

    @property
    def client(self) -> InterfaceClient: ...  # pragma: no cover

    def model_profile(self, model_name: str) -> ModelProfile | None: ...  # pragma: no cover

    @overload
    def __new__(
        cls,
        provider: Literal['openai', 'openai-chat', 'openai-responses'],
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> Provider[AsyncOpenAI]: ...

    @overload
    def __new__(
        cls,
        provider: Literal['groq'],
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> Provider[AsyncGroq]: ...

    @overload
    def __new__(
        cls,
        provider: Literal['google-vertex'],
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        credentials: Credentials | None = None,
        project: str | None = None,
        location: VertexAILocation | Literal['global'] | None = None,
    ) -> Provider[GoogleClient]: ...

    def __new__(
        cls,
        provider: Literal['openai', 'openai-chat', 'openai-responses', 'groq', 'google-vertex'] | str,
        *,
        # Every provider
        api_key: str | None = None,
        base_url: str | None = None,
        # OpenAI & Groq
        http_client: httpx.AsyncClient | None = None,
        # Google
        credentials: Credentials | None = None,
        project: str | None = None,
        location: VertexAILocation | Literal['global'] | None = None,
    ) -> Provider[Any]:
        api_key = api_key or os.getenv('PYDANTIC_AI_GATEWAY_API_KEY')
        if not api_key:
            raise UserError(
                'Set the `PYDANTIC_AI_GATEWAY_API_KEY` environment variable or pass it via `GatewayProvider(api_key=...)`'
                ' to use the Pydantic AI Gateway provider.'
            )

        base_url = base_url or 'http://localhost:8787/'
        http_client = http_client or httpx.AsyncClient()
        http_client.event_hooks = {'request': [_request_hook]}

        if provider in ('openai', 'openai-chat', 'openai-responses'):
            from .openai import OpenAIProvider

            return OpenAIProvider(api_key=api_key, base_url=urljoin(base_url, 'openai'), http_client=http_client)
        elif provider == 'groq':
            from .groq import GroqProvider

            return GroqProvider(api_key=api_key, base_url=urljoin(base_url, 'groq'), http_client=http_client)
        elif provider == 'google-vertex':
            from google.genai import Client as GoogleClient

            from .google import GoogleProvider

            return GoogleProvider(
                client=GoogleClient(
                    api_key=api_key,
                    http_options={
                        'base_url': urljoin(base_url, 'google'),
                        'headers': {'User-Agent': get_user_agent()},
                        'async_client_args': {
                            'transport': httpx.AsyncHTTPTransport(),
                            'event_hooks': {'request': [_request_hook]},
                        },
                    },
                )
            )
        else:  # pragma: no cover
            raise UserError(f'Unknown provider: {provider}')


async def _request_hook(request: httpx.Request) -> httpx.Request:
    from opentelemetry.propagate import inject

    headers: dict[str, Any] = {}
    inject(headers)
    request.headers.update(headers)

    return request
