from __future__ import annotations as _annotations

import os
from typing import Any, overload

import httpx

from pydantic_ai.models import cached_async_http_client

try:
    from groq import AsyncGroq
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install `groq` to use the Groq provider, '
        "you can use the `groq` optional group — `pip install 'pydantic-ai-slim[groq]'`"
    ) from _import_error


from . import Provider


class GroqProvider(Provider[Any]):
    """Provider for Groq API."""

    @property
    def name(self) -> str:
        return 'groq'  # pragma: no cover

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def client(self) -> AsyncGroq:
        return self._client

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, api_key: str) -> None: ...

    @overload
    def __init__(self, *, api_key: str, http_client: httpx.AsyncClient) -> None: ...

    @overload
    def __init__(self, *, groq_client: AsyncGroq | None = None) -> None: ...

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        groq_client: AsyncGroq | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a new Groq provider.

        Args:
            base_url: The base url for the Groq requests. If not provided, the `GROQ_BASE_URL` environment variable
                will be used if available. Otherwise, defaults to Groq's base url.
            api_key: The API key to use for authentication, if not provided, the `GROQ_API_KEY` environment variable
                will be used if available.
            groq_client: An existing
                [`AsyncGroq`](https://github.com/groq/groq-python?tab=readme-ov-file#async-usage)
                client to use. If provided, `base_url`, `api_key`, and `http_client` must be `None`.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        # Use the environment variable GROQ_BASE_URL or default to the Groq API URL
        # According to docs, the base URL is https://api.groq.com/v1
        self._base_url = base_url or os.environ.get('GROQ_BASE_URL', 'https://api.groq.com/v1')
        api_key = api_key or os.environ.get('GROQ_API_KEY')

        if api_key is None and groq_client is None:
            raise ValueError(
                'Set the `GROQ_API_KEY` environment variable or pass it via `GroqProvider(api_key=...)`'
                'to use the Groq provider.'
            )

        if groq_client is not None:
            assert base_url is None, 'Cannot provide both `groq_client` and `base_url`'
            assert http_client is None, 'Cannot provide both `groq_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `groq_client` and `api_key`'
            self._client = groq_client
        elif http_client is not None:
            self._client = AsyncGroq(base_url=self.base_url, api_key=api_key, http_client=http_client)
        else:
            self._client = AsyncGroq(base_url=self.base_url, api_key=api_key, http_client=cached_async_http_client())
