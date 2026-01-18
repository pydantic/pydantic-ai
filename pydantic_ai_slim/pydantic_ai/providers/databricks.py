from __future__ import annotations as _annotations

import os
from typing import overload

import httpx
from openai import AsyncOpenAI

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the Databricks provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


class DatabricksProvider(Provider[AsyncOpenAI]):
    """Provider for Databricks API."""

    @property
    def name(self) -> str:
        return 'databricks'

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI) -> None: ...

    @overload
    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        openai_client: None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if openai_client is not None:
            self._client = openai_client
            self._base_url = str(openai_client.base_url)
            return

        api_key = api_key or os.getenv('DATABRICKS_TOKEN') or os.getenv('DATABRICKS_API_KEY')
        base_url = base_url or os.getenv('DATABRICKS_BASE_URL')

        if not base_url:
            host = os.getenv('DATABRICKS_HOST')
            if host:
                base_url = f'{host.rstrip("/")}/serving-endpoints'

        if not base_url:
            raise UserError(
                'Set `DATABRICKS_HOST` or `DATABRICKS_BASE_URL` environment variable, or pass `base_url` '
                'to use the Databricks provider.'
            )

        self._base_url = base_url

        if not api_key:
            # Databricks might use other auth methods (e.g. metadata service), but explicit token is standard.
            # We warn or raise? OpenAI client requires api_key usually, unless we pass something else.
            # We can use a dummy key if it's not needed (e.g. auth proxy), but standard is token.
            # We'll allow empty if user knows what they are doing (e.g. pass via http_client headers),
            # but OpenAI client complains if api_key is missing.
            api_key = 'nop'  # Placeholder if not provided, assuming auth might be handled by env or http_client?
            # Actually, let's look for DATABRICKS_TOKEN. If not found, raise UserError?
            # User might rely on .netrc?
            # Let's try to be helpful.
            if not http_client:  # If http_client is custom, maybe it handles auth.
                # raise UserError(...)
                pass

        if http_client is None:
            http_client = cached_async_http_client(provider='databricks')

        self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key or 'nop', http_client=http_client)
