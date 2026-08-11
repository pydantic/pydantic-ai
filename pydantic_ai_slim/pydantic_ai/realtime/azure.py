"""Azure OpenAI realtime support using the OpenAI GA protocol."""

from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import ClassVar
from urllib.parse import urlencode, urlparse, urlunparse

from openai import AsyncOpenAI

from ..exceptions import UserError
from ..providers import Provider, infer_provider
from ..providers.azure import AzureProvider
from .openai import OpenAIRealtimeConnection, OpenAIRealtimeModel
from .profiles import RealtimeModelProfileSpec
from .settings import RealtimeModelSettings

__all__ = ('AzureRealtimeModel', 'AzureRealtimeConnection')


class AzureRealtimeConnection(OpenAIRealtimeConnection):
    """A live WebSocket connection to Azure OpenAI's realtime API.

    Reuses [`OpenAIRealtimeConnection`][pydantic_ai.realtime.openai.OpenAIRealtimeConnection] for the
    shared GA wire protocol, naming Azure as the vendor so a connection that drops or rejects content
    doesn't send someone debugging an Azure session to OpenAI's status page.
    """

    _provider_name = 'azure'
    _provider_label = 'Azure OpenAI Realtime'


@dataclass(init=False)
class AzureRealtimeModel(OpenAIRealtimeModel):
    """Azure OpenAI realtime model using the OpenAI GA protocol.

    The existing [`AzureProvider`][pydantic_ai.providers.azure.AzureProvider] supplies the Azure
    resource endpoint and API key. The WebSocket transport does not use its OpenAI SDK client or
    `api_version`; it connects to the GA `/openai/v1/realtime` endpoint with an `api-key` header.
    """

    _connection_type: ClassVar[type[OpenAIRealtimeConnection]] = AzureRealtimeConnection

    def __init__(
        self,
        model: str = 'gpt-realtime',
        *,
        provider: Provider[AsyncOpenAI] | str = 'azure',
        settings: RealtimeModelSettings | None = None,
        profile: RealtimeModelProfileSpec | None = None,
    ) -> None:
        """Create an Azure OpenAI realtime model.

        Args:
            model: The Azure *deployment* name, which is what the realtime URL and the profile lookup
                use. Azure deployments are conventionally named after their model; when yours isn't,
                `profile` is how to correct the facts inferred from the name.
            provider: The provider supplying the resource endpoint and API key. Defaults to `'azure'`.
            settings: [Model settings][pydantic_ai.realtime.RealtimeModelSettings] used as defaults
                for realtime sessions.
            profile: Optional override for the [realtime model profile][pydantic_ai.realtime.RealtimeModelProfile],
                merged over the provider's — a partial dict, or a callable taking the resolved profile
                and returning the one to use.
        """
        super().__init__(model, provider=provider, settings=settings, profile=profile)

    @staticmethod
    def _resolve_provider(provider: Provider[AsyncOpenAI] | str) -> AzureProvider:
        if isinstance(provider, str):
            provider = AzureProvider.for_realtime() if provider == 'azure' else infer_provider(provider)
        if not isinstance(provider, AzureProvider):
            raise UserError("`AzureRealtimeModel` requires an `AzureProvider` or `provider='azure'`.")
        return provider

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
