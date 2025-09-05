"""This module implements the Pydantic AI Gateway provider."""

from __future__ import annotations as _annotations

import os
from typing import TYPE_CHECKING, Any, Literal, overload

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
    def name(self) -> str:
        # TODO(Marcelo): This is actually NEVER used, because the `provider` is never `GatewayProvider`, it's always
        # the actual provider class: `OpenAIProvider`, `GroqProvider` and `GoogleProvider`. Maybe it's fine?
        return 'gateway'

    @property
    def base_url(self) -> str: ...

    @property
    def client(self) -> InterfaceClient: ...

    def model_profile(self, model_name: str) -> ModelProfile | None: ...

    @overload
    def __new__(
        cls,
        provider: Literal['openai'],
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
        provider: Literal['openai', 'groq', 'google-vertex'] | str,
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
        api_key = api_key or os.getenv('PYDANTIC_AI_GATEWAY_PROVIDER')
        if not api_key:
            raise UserError(
                'Set the `PYDANTIC_AI_GATEWAY_API_KEY` environment variable or pass it via `GatewayProvider(api_key=...)`'
                'to use the Pydantic AI Gateway provider.'
            )

        # NOTE(Marcelo): What's the real URL?
        base_url = base_url or 'https://ai-gateway.pydantic.dev'

        if provider == 'openai':
            from .openai import OpenAIProvider

            return OpenAIProvider(api_key=api_key, base_url=base_url, http_client=http_client)
        elif provider == 'groq':
            from .groq import GroqProvider

            return GroqProvider(api_key=api_key, http_client=http_client)
        elif provider == 'google-vertex':
            from google.auth.api_key import Credentials as APIKeyCredentials
            from google.genai import Client as GoogleClient

            from .google import GoogleProvider

            credentials = APIKeyCredentials(token=api_key)
            return GoogleProvider(
                client=GoogleClient(
                    credentials=credentials,
                    location=location,
                    project=project,
                    http_options={
                        'base_url': base_url,
                        'headers': {'User-Agent': get_user_agent()},
                        'async_client_args': {'transport': httpx.AsyncHTTPTransport()},
                    },
                )
            )
        else:
            raise UserError(f'Unknown provider: {provider}')
