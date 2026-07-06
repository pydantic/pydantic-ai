from __future__ import annotations as _annotations

import os
from typing import overload

import httpx

from pydantic_ai import ModelProfile, __version__
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import create_async_http_client
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.profiles.perplexity import perplexity_model_profile
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the Perplexity provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


class PerplexityProvider(Provider[AsyncOpenAI]):
    """Provider for the Perplexity API.

    Perplexity accepts OpenAI-compatible chat completions requests at
    `https://api.perplexity.ai/chat/completions`, which is what
    [`PerplexityModel`][pydantic_ai.models.perplexity.PerplexityModel] talks to when paired with this provider.
    """

    @property
    def name(self) -> str:
        return 'perplexity'

    @property
    def base_url(self) -> str:
        return 'https://api.perplexity.ai'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        profile = perplexity_model_profile(model_name)

        # PerplexityProvider is always paired with PerplexityModel (an OpenAIChatModel), which used to
        # unconditionally apply the OpenAI JSON schema transformer, so we keep that behavior (mirrors `DeepSeekProvider`).
        return merge_profile(
            OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer),
            profile,
        )

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI) -> None: ...

    @overload
    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a new Perplexity provider.

        Args:
            api_key: The API key used for authentication. Falls back to the `PERPLEXITY_API_KEY` environment
                variable, then the `PPLX_API_KEY` alias used by Perplexity's own SDKs.
            openai_client: An existing `AsyncOpenAI` client to use. If provided, `api_key` and `http_client`
                must not be set.
            http_client: An existing `httpx.AsyncClient` to use for HTTP requests.
        """
        api_key = api_key or os.getenv('PERPLEXITY_API_KEY') or os.getenv('PPLX_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `PERPLEXITY_API_KEY` environment variable or pass it via `PerplexityProvider(api_key=...)`'
                ' to use the Perplexity provider.'
            )

        default_headers = {'X-Pplx-Integration': f'pydantic-ai/{__version__}'}

        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(
                base_url=self.base_url, api_key=api_key, http_client=http_client, default_headers=default_headers
            )
        else:
            http_client = create_async_http_client()
            self._own_http_client = http_client
            self._http_client_factory = create_async_http_client
            self._client = AsyncOpenAI(
                base_url=self.base_url, api_key=api_key, http_client=http_client, default_headers=default_headers
            )

    def _set_http_client(self, http_client: httpx.AsyncClient) -> None:
        self._client._client = http_client  # pyright: ignore[reportPrivateUsage]
