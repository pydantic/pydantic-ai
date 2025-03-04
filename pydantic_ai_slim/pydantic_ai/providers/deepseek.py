from __future__ import annotations as _annotations

import os

from httpx import AsyncClient as AsyncHTTPClient
from openai import AsyncOpenAI

from pydantic_ai.models import cached_async_http_client

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install `openai` to use the DeepSeek provider, '
        "you can use the `openai` optional group — `pip install 'pydantic-ai-slim[openai]'`"
    ) from _import_error

from . import Provider


class DeepSeekProvider(Provider[AsyncOpenAI]):
    """Provider for DeepSeek API."""

    @property
    def name(self) -> str:
        return 'deepseek'

    @property
    def base_url(self) -> str:
        return 'https://api.deepseek.com'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def __init__(
        self,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: AsyncHTTPClient | None = None,
    ) -> None:
        api_key = api_key or os.environ.get('DEEPSEEK_API_KEY')
        if api_key is None and openai_client is None:
            raise ValueError(
                'Set the `DEEPSEEK_API_KEY` environment variable or pass it via `DeepSeekProvider(api_key=...)`'
                'to use the DeepSeek provider.'
            )

        if openai_client is not None:
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `openai_client` and `api_key`'
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
        else:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=cached_async_http_client())
