"""Azure OpenAI realtime support using the OpenAI GA protocol."""

from __future__ import annotations as _annotations

import os
from dataclasses import InitVar, dataclass
from urllib.parse import urlencode, urlparse, urlunparse

from openai import AsyncOpenAI

from ..exceptions import UserError
from ..providers import Provider, infer_provider
from ..providers.azure import AzureProvider, _openai_compatible_v1_base_url  # pyright: ignore[reportPrivateUsage]
from .openai import OpenAIRealtimeModel

__all__ = ('AzureRealtimeModel',)


def _infer_azure_realtime_provider() -> Provider[AsyncOpenAI]:
    """Infer an [`AzureProvider`][pydantic_ai.providers.azure.AzureProvider] for realtime use from the environment.

    The realtime model speaks only the GA v1 protocol and never uses the provider's OpenAI SDK client
    or `api_version`. A bare resource `AZURE_OPENAI_ENDPOINT` (no `/openai/v1` path) without
    `OPENAI_API_VERSION` would make `AzureProvider` demand an `api_version` it doesn't need here, so
    the endpoint is normalized to its `/openai/v1` form instead — the realtime URLs are derived from
    the endpoint's host either way.
    """
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    if endpoint and not os.getenv('OPENAI_API_VERSION') and _openai_compatible_v1_base_url(endpoint) is None:
        return AzureProvider(azure_endpoint=endpoint.rstrip('/') + '/openai/v1')
    return infer_provider('azure')


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
            provider = _infer_azure_realtime_provider() if provider == 'azure' else infer_provider(provider)
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
