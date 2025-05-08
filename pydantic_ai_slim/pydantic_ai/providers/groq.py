from __future__ import annotations as _annotations

import os
from typing import overload

from httpx import AsyncClient as AsyncHTTPClient

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.providers import Provider

try:
    from groq import AsyncGroq
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `groq` package to use the Groq provider, '
        'you can use the `groq` optional group — `pip install "pydantic-ai-slim[groq]"`'
    ) from _import_error


class GroqProvider(Provider[AsyncGroq]):
    """Provider for Groq API."""

    @property
    def name(self) -> str:
        return 'groq'

    @property
    def base_url(self) -> str:
        return os.environ.get('GROQ_BASE_URL', 'https://api.groq.com')

    @property
    def client(self) -> AsyncGroq:
        return self._client

    @overload
    def __init__(self, *, groq_client: AsyncGroq | None = None) -> None: ...

    @overload
    def __init__(self, *, api_key: str | None = None, http_client: AsyncHTTPClient | None = None) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        groq_client: AsyncGroq | None = None,
        http_client: AsyncHTTPClient | None = None,
    ) -> None:
        """Create a new Groq provider.

        Args:
            api_key: The API key to use for authentication, if not provided, the `GROQ_API_KEY` environment variable
                will be used if available.
            groq_client: An existing
                [`AsyncGroq`](https://github.com/groq/groq-python?tab=readme-ov-file#async-usage)
                client to use. If provided, `api_key` and `http_client` must be `None`.
            http_client: An existing `AsyncHTTPClient` to use for making HTTP requests.
        """
        if groq_client is not None:
            assert http_client is None, 'Cannot provide both `groq_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `groq_client` and `api_key`'
            self._client = groq_client
        else:
            api_key = api_key or os.environ.get('GROQ_API_KEY')

            if not api_key:
                raise UserError(
                    'Set the `GROQ_API_KEY` environment variable or pass it via `GroqProvider(api_key=...)`'
                    'to use the Groq provider.'
                )
            elif http_client is not None:
                self._client = AsyncGroq(base_url=self.base_url, api_key=api_key, http_client=http_client)
            else:
                http_client = cached_async_http_client(provider='groq')
                self._client = AsyncGroq(base_url=self.base_url, api_key=api_key, http_client=http_client)
