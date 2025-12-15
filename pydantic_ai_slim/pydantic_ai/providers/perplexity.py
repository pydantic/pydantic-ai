"""Perplexity AI provider."""

from __future__ import annotations as _annotations

import os
from typing import overload

import httpx
from openai import AsyncOpenAI

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles.openai import openai_model_profile
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the Perplexity provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


class PerplexityProvider(Provider[AsyncOpenAI]):
    """Perplexity AI provider.

    Perplexity's API is compatible with OpenAI's format, so we just use
    the OpenAI client with Perplexity's base URL.
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

    def model_profile(self, model_name: str) -> ModelProfile | None:
        """Get the model profile for a Perplexity model.

        Since Perplexity uses OpenAI's format, we just use the OpenAI profile.
        """
        return openai_model_profile(model_name)

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, api_key: str) -> None: ...

    @overload
    def __init__(self, *, api_key: str, http_client: httpx.AsyncClient) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI | None = None) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a Perplexity provider.

        Args:
            api_key: Your Perplexity API key. Falls back to PERPLEXITY_API_KEY env var.
            openai_client: Use an existing OpenAI client instead of creating one.
            http_client: Use a custom HTTP client instead of the default.
        """
        api_key = api_key or os.getenv('PERPLEXITY_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `PERPLEXITY_API_KEY` environment variable or pass it via '
                '`PerplexityProvider(api_key=...)` to use the Perplexity provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
        else:
            http_client = cached_async_http_client(provider='perplexity')
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
