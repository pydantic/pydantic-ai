from collections.abc import Mapping

import httpx
from openai import AsyncOpenAI

from pydantic_ai.models import create_async_http_client
from pydantic_ai.providers import Provider


class OpenAICompatibleProvider(Provider[AsyncOpenAI]):
    """Shared HTTP client lifecycle for providers backed by the OpenAI SDK."""

    def _get_http_client(self, http_client: httpx.AsyncClient | None) -> httpx.AsyncClient:
        if http_client is None:
            http_client = create_async_http_client()
            self._own_http_client = http_client
            self._http_client_factory = create_async_http_client
        return http_client

    def _create_openai_client(
        self,
        *,
        base_url: str | None,
        api_key: str | None,
        http_client: httpx.AsyncClient | None,
        default_headers: Mapping[str, str] | None = None,
    ) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=self._get_http_client(http_client),
            default_headers=default_headers,
        )

    def _set_http_client(self, http_client: httpx.AsyncClient) -> None:
        self._client._client = http_client  # pyright: ignore[reportPrivateUsage]
