from __future__ import annotations as _annotations

import os
from typing import TypeAlias, overload

import httpx

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.providers import Provider

try:
    from anthropic import AsyncAnthropic, AsyncAnthropicBedrock, AsyncAnthropicVertex
except ImportError as _import_error:
    raise ImportError(
        'Please install the `anthropic` package to use the Anthropic provider, '
        'you can use the `anthropic` optional group — `pip install "pydantic-ai-slim[anthropic]"`'
    ) from _import_error


AsyncAnthropicClient: TypeAlias = AsyncAnthropic | AsyncAnthropicBedrock | AsyncAnthropicVertex


class AnthropicProvider(Provider[AsyncAnthropicClient]):
    """Provider for Anthropic API."""

    @property
    def name(self) -> str:
        return 'anthropic'

    @property
    def base_url(self) -> str:
        return str(self._client.base_url)

    @property
    def client(self) -> AsyncAnthropicClient:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        return anthropic_model_profile(model_name)

    @overload
    def __init__(self, *, anthropic_client: AsyncAnthropicClient | None = None) -> None: ...

    @overload
    def __init__(
        self, *, api_key: str | None = None, base_url: str | None = None, http_client: httpx.AsyncClient | None = None
    ) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        anthropic_client: AsyncAnthropicClient | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a new Anthropic provider.

        Args:
            api_key: The API key to use for authentication, if not provided, the `ANTHROPIC_API_KEY` environment variable
                will be used if available.
            base_url: The base URL to use for the Anthropic API.
            anthropic_client: An existing [`AsyncAnthropic`](https://github.com/anthropics/anthropic-sdk-python)
                client to use. If provided, the `api_key` and `http_client` arguments will be ignored.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        if anthropic_client is not None:
            assert http_client is None, 'Cannot provide both `anthropic_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `anthropic_client` and `api_key`'
            self._client = anthropic_client
        else:
            api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise UserError(
                    'Set the `ANTHROPIC_API_KEY` environment variable or pass it via `AnthropicProvider(api_key=...)`'
                    'to use the Anthropic provider.'
                )
            if http_client is not None:
                self._client = AsyncAnthropic(api_key=api_key, base_url=base_url, http_client=http_client)
            else:
                http_client = cached_async_http_client(provider='anthropic')
                self._client = AsyncAnthropic(api_key=api_key, base_url=base_url, http_client=http_client)
