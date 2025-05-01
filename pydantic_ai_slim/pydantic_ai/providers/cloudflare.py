from __future__ import annotations as _annotations

import os

from httpx import AsyncClient as AsyncHTTPClient

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.providers import Provider

try:
    from cloudflare import AsyncCloudflare
except ImportError as _import_error:
    raise ImportError(
        'Please install the `cloudflare` package to use the Cloudflare provider, '
        'you can use the `cloudflare` optional group — `pip install "pydantic-ai-slim[cloudflare]"`'
    ) from _import_error


class CloudflareProvider(Provider[AsyncCloudflare]):
    """Provider for Cloudflare Workers AI."""

    @property
    def name(self) -> str:
        return 'cloudflare'

    @property
    def base_url(self) -> str:
        return str(self._client.base_url)

    @property
    def client(self) -> AsyncCloudflare:
        return self._client

    def __init__(
        self,
        *,
        api_key: str | None = None,
        cloudflare_client: AsyncCloudflare | None = None,
        http_client: AsyncHTTPClient | None = None,
    ) -> None:
        """Create a new Cloudflare provider.

        Args:
                api_key: The API key to use for authentication, if not provided, the `CLOUDFLARE_API_KEY` env var is used.
                account_id: Cloudflare account ID, or set via `CLOUDFLARE_ACCOUNT_ID` env var.
                cloudflare_client: Pre-existing `AsyncCloudflare` client instance.
                http_client: Optional custom `httpx.AsyncClient`.
        """
        if cloudflare_client is not None:
            assert api_key is None, 'Cannot provide both `cloudflare_client` and `api_key`'
            assert http_client is None, 'Cannot provide both `cloudflare_client` and `http_client`'
            self._client = cloudflare_client
            self.account_id = '<from client>'  # replace with real extraction if possible
        else:
            api_key = api_key or os.getenv('CLOUDFLARE_API_KEY')

            if not api_key:
                raise UserError(
                    'Set the `CLOUDFLARE_API_KEY` environment variable or pass it via `CloudflareProvider(api_key=...)`'
                )

            http_client = http_client or cached_async_http_client(provider='cloudflare')
            self._client = AsyncCloudflare(api_key=api_key, http_client=http_client)
