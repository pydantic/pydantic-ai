from collections.abc import Mapping

from openai import AsyncOpenAI

from pydantic_ai._http import (
    AsyncHTTPClient as AsyncHTTPClient,
    create_async_httpx2_client,
    warn_if_legacy_httpx_client,
)
from pydantic_ai.providers import Provider


class OpenAICompatibleProvider(Provider[AsyncOpenAI]):
    """Shared HTTP client lifecycle for providers backed by the OpenAI SDK."""

    def _get_http_client(
        self,
        http_client: AsyncHTTPClient | None,
        *,
        # Frames to skip so the warning lands on the user's `SomeProvider(...)` call: this method,
        # the provider `__init__` calling it, and the user's call site. Callers that add a frame in
        # between pass a higher value.
        warning_stacklevel: int = 3,
    ) -> AsyncHTTPClient:
        if http_client is None:
            http_client = create_async_httpx2_client()
            self._own_http_client = http_client
            self._http_client_factory = create_async_httpx2_client
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
        http_client: AsyncHTTPClient | None,
        default_headers: Mapping[str, str] | None = None,
    ) -> AsyncOpenAI:
        # One frame more than the default: this method sits between `_get_http_client` and the
        # provider `__init__`, and the warning still targets the user's `SomeProvider(...)` call.
        http_client = self._get_http_client(http_client, warning_stacklevel=4)
        # OpenAI 3 keeps legacy HTTPX as a runtime-only escape hatch, outside its public type annotations.
        return AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=http_client,  # pyright: ignore[reportArgumentType]
            default_headers=default_headers,
        )

    def _set_http_client(self, http_client: AsyncHTTPClient) -> None:
        self._client._client = http_client  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
