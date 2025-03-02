from __future__ import annotations as _annotations

import os
from typing import TypeVar

from httpx import AsyncClient as AsyncHTTPClient

from pydantic_ai.models import cached_async_http_client

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install `openai` to use the OpenAI provider, '
        "you can use the `openai` optional group — `pip install 'pydantic-ai-slim[openai]'`"
    ) from _import_error


from . import Provider

InterfaceClient = TypeVar('InterfaceClient')


class OpenAIProvider(Provider[AsyncOpenAI]):
    """Provider for OpenAI API."""

    @property
    def name(self) -> str:
        return 'openai'

    @property
    def base_url(self) -> str:
        return 'https://api.openai.com/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def __init__(
        self,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: AsyncHTTPClient | None = None,
    ) -> None:
        # This is a workaround for the OpenAI client requiring an API key, whilst locally served,
        # openai compatible models do not always need an API key, but a placeholder (non-empty) key is required.
        if api_key is None and 'OPENAI_API_KEY' not in os.environ and openai_client is None:
            api_key = 'api-key-not-set'

        if openai_client is not None:
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `openai_client` and `api_key`'
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
        else:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=cached_async_http_client())
