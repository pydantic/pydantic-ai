from __future__ import annotations as _annotations

import os
from typing import overload

import httpx

from pydantic_ai import ModelProfile
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles.openai import openai_model_profile
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the OpenAI provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


class OpenAIProvider(Provider[AsyncOpenAI]):
    """Provider for OpenAI API."""

    @property
    def name(self) -> str:
        return 'openai'

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        return openai_model_profile(model_name)

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI) -> None: ...

    @overload
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        openai_client: None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None: ...

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a new OpenAI provider.

        Args:
            base_url: The base url for the OpenAI requests. If not provided, the `OPENAI_BASE_URL` environment variable
                will be used if available. Otherwise, defaults to OpenAI's base url.
            api_key: The API key to use for authentication, if not provided, the `OPENAI_API_KEY` environment variable
                will be used if available.
            openai_client: An existing
                [`AsyncOpenAI`](https://github.com/openai/openai-python?tab=readme-ov-file#async-usage)
                client to use. If provided, `base_url`, `api_key`, and `http_client` must be `None`.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        # This is a workaround for the OpenAI client requiring an API key, whilst locally served,
        # openai compatible models do not always need an API key, but a placeholder (non-empty) key is required.
        if api_key is None and 'OPENAI_API_KEY' not in os.environ and base_url is not None and openai_client is None:
            api_key = 'api-key-not-set'

        if openai_client is not None:
            assert base_url is None, 'Cannot provide both `openai_client` and `base_url`'
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `openai_client` and `api_key`'
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
        else:
            http_client = cached_async_http_client(provider='openai')
            self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
