"""OpenAI protocol view over the native Cloudflare provider's HTTP lifecycle."""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING, Self

import httpx2
from openai import AsyncOpenAI

from ..profiles import ModelProfile
from ..profiles.cloudflare import cloudflare_model_profile
from . import Provider

if TYPE_CHECKING:
    from .cloudflare import CloudflareProvider


class CloudflareOpenAIProvider(Provider[AsyncOpenAI]):
    def __init__(self, provider: CloudflareProvider) -> None:
        self._provider = provider
        self._http_client: httpx2.AsyncClient | None = None

    @property
    def name(self) -> str:
        return self._provider.name

    @property
    def base_url(self) -> str:
        return f'{self._provider.base_url}/ai/v1'

    @property
    def client(self) -> AsyncOpenAI:
        self._provider._refresh_http_client()
        http_client = self._provider.client
        if self._http_client is not http_client:
            self._client = AsyncOpenAI(api_key=self._provider.api_key, base_url=self.base_url, http_client=http_client)
            self._http_client = http_client
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        return cloudflare_model_profile(model_name)

    def _refresh_http_client(self) -> None:
        self._provider._refresh_http_client()

    async def __aenter__(self) -> Self:
        await self._provider.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        return await self._provider.__aexit__(exc_type, exc_val, exc_tb)
