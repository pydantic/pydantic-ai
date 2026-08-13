from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import httpx2
from openai import AsyncOpenAI

from pydantic_ai._http import create_httpx2_client, warn_if_legacy_httpx_client
from pydantic_ai.providers import Provider

if TYPE_CHECKING:
    import httpx

    OpenAIHTTPClient = httpx.AsyncClient | httpx2.AsyncClient
else:
    OpenAIHTTPClient = httpx2.AsyncClient


class OpenAICompatibleProvider(Provider[AsyncOpenAI]):
    """Shared HTTP client lifecycle for providers backed by the OpenAI SDK."""

    _own_http_client: OpenAIHTTPClient | None = None
    _http_client_factory: Callable[[], OpenAIHTTPClient] | None = None

    def _get_http_client(
        self, http_client: OpenAIHTTPClient | None, *, warning_stacklevel: int = 3
    ) -> OpenAIHTTPClient:
        if http_client is None:
            http_client = create_httpx2_client()
            self._own_http_client = http_client  # pyright: ignore[reportIncompatibleVariableOverride]
            self._http_client_factory = create_httpx2_client  # pyright: ignore[reportIncompatibleVariableOverride]
        else:
            warn_if_legacy_httpx_client(
                http_client, consumer='OpenAI-compatible providers', stacklevel=warning_stacklevel
            )
        return http_client

    def _create_openai_client(
        self,
        *,
        base_url: str | None,
        api_key: str | None,
        http_client: OpenAIHTTPClient | None,
        default_headers: Mapping[str, str] | None = None,
    ) -> AsyncOpenAI:
        http_client = self._get_http_client(http_client, warning_stacklevel=4)
        # OpenAI 3 keeps legacy HTTPX as a runtime-only escape hatch, outside its public type annotations.
        return AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=http_client,  # pyright: ignore[reportArgumentType]
            default_headers=default_headers,
        )

    # The generic Provider currently only knows the legacy HTTPX client type.
    def _set_http_client(self, http_client: OpenAIHTTPClient) -> None:
        self._client._client = http_client  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
