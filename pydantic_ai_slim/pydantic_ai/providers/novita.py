from __future__ import annotations as _annotations

import os
from typing import Literal, overload

import httpx

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles.openai import (
    OpenAIJsonSchemaTransformer,
    OpenAIModelProfile,
)
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the Novita provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


NovitaModelName = Literal[
    'moonshotai/kimi-k2.5',
    'zai-org/glm-5',
    'minimax/minimax-m2.5',
]
"""Known Novita model names.

See the Novita AI documentation for a full list of available models.
"""


class NovitaProvider(Provider[AsyncOpenAI]):
    """Provider for Novita AI API.

    Novita AI provides an OpenAI-compatible API endpoint.
    See https://novita.ai/docs for more details.
    """

    @property
    def name(self) -> str:
        return 'novita'

    @property
    def base_url(self) -> str:
        return 'https://api.novita.ai/openai'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        """Get the model profile for a Novita model."""
        # Novita models support structured output and function calling
        return OpenAIModelProfile(
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            supports_json_object_output=True,
        )

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, api_key: str) -> None: ...

    @overload
    def __init__(self, *, api_key: str, http_client: httpx.AsyncClient) -> None: ...

    @overload
    def __init__(self, *, http_client: httpx.AsyncClient) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI | None = None) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a new Novita provider.

        Args:
            api_key: The API key to use for authentication, if not provided, the `NOVITA_API_KEY`
                environment variable will be used if available.
            openai_client: An existing `AsyncOpenAI` client to use. If provided, `api_key` and
                `http_client` must be `None`.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        api_key = api_key or os.getenv('NOVITA_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `NOVITA_API_KEY` environment variable or pass it via `NovitaProvider(api_key=...)` '
                'to use the Novita provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
        else:
            http_client = cached_async_http_client(provider='novita')
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
