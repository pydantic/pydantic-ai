"""Azure OpenAI realtime support using the OpenAI GA protocol."""

from __future__ import annotations as _annotations

from dataclasses import InitVar, dataclass
from urllib.parse import urlencode, urlparse, urlunparse

from openai import AsyncOpenAI

from ..exceptions import UserError
from ..providers import Provider, infer_provider
from ..providers.azure import AzureProvider
from .openai import OpenAIRealtimeModel

__all__ = ('AzureRealtimeModel',)


@dataclass
class AzureRealtimeModel(OpenAIRealtimeModel):
    """Azure OpenAI realtime model using the OpenAI GA protocol.

    The existing [`AzureProvider`][pydantic_ai.providers.azure.AzureProvider] supplies the Azure
    resource endpoint and API key. The WebSocket transport does not use its OpenAI SDK client or
    `api_version`; it connects to the GA `/openai/v1/realtime` endpoint with an `api-key` header.
    """

    provider: InitVar[Provider[AsyncOpenAI] | str] = 'azure'

    def __post_init__(self, provider: Provider[AsyncOpenAI] | str) -> None:
        if isinstance(provider, str):
            provider = infer_provider(provider)
        if not isinstance(provider, AzureProvider):
            raise UserError("`AzureRealtimeModel` requires an `AzureProvider` or `provider='azure'`.")
        self._provider = provider

    @property
    def _azure_provider(self) -> AzureProvider:
        assert isinstance(self._provider, AzureProvider)
        return self._provider

    def _realtime_url(self) -> str:
        parsed = urlparse(self._azure_provider.azure_endpoint)
        return urlunparse(
            parsed._replace(scheme='wss', path='/openai/v1/realtime', query=urlencode({'model': self.model}))
        )

    async def _auth_headers(self) -> dict[str, str]:
        return {'api-key': self._azure_provider.api_key}
