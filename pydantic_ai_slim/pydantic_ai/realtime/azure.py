"""Azure OpenAI realtime support using the OpenAI GA protocol."""

from __future__ import annotations as _annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Protocol
from urllib.parse import urlparse, urlunparse

from anyio.to_thread import run_sync
from openai import AsyncOpenAI

from ..exceptions import UserError
from ..providers import Provider, infer_provider
from ..providers.azure import AzureProvider
from ..tools import ToolDefinition
from ._openai_protocol import with_realtime_query
from ._openai_webrtc import relay_sdp_offer as _relay_sdp_offer
from .model import WebRTCAnswer
from .openai import OpenAIRealtimeConnection, OpenAIRealtimeModel, OpenAIRealtimeModelName
from .profiles import RealtimeModelProfileSpec
from .settings import RealtimeModelSettings

__all__ = ('AzureRealtimeModel', 'AzureRealtimeConnection', 'AzureTokenCredential')

LatestAzureRealtimeModelNames = Literal['gpt-realtime']
AzureRealtimeModelName = OpenAIRealtimeModelName

LatestAzureRealtimeTranscriptionModelNames = Literal['azure-speech', 'mai-transcribe']
AzureRealtimeTranscriptionModelName = str | LatestAzureRealtimeTranscriptionModelNames


class AzureRealtimeConnection(OpenAIRealtimeConnection):
    """A live WebSocket connection to Azure OpenAI's realtime API.

    Reuses [`OpenAIRealtimeConnection`][pydantic_ai.realtime.openai.OpenAIRealtimeConnection] for the
    shared GA wire protocol, naming Azure as the vendor so a connection that drops or rejects content
    doesn't send someone debugging an Azure session to OpenAI's status page.
    """

    _provider_name = 'azure'
    _provider_label = 'Azure OpenAI Realtime'


class _AccessToken(Protocol):
    token: str


class AzureTokenCredential(Protocol):
    """Structural type for a synchronous Microsoft Entra ID token credential."""

    def get_token(self, *scopes: str, **kwargs: Any) -> _AccessToken: ...


_ENTRA_SCOPE = 'https://ai.azure.com/.default'


@dataclass(init=False)
class AzureRealtimeModel(OpenAIRealtimeModel):
    """Azure OpenAI realtime model using the OpenAI GA protocol.

    The existing [`AzureProvider`][pydantic_ai.providers.azure.AzureProvider] supplies the Azure
    resource endpoint and API key. The WebSocket transport does not use its OpenAI SDK client or
    `api_version`; it connects to the GA `/openai/v1/realtime` endpoint with an `api-key` header.
    """

    _connection_type: ClassVar[type[OpenAIRealtimeConnection]] = AzureRealtimeConnection
    credential: AzureTokenCredential | None = None

    def __init__(
        self,
        model: AzureRealtimeModelName,
        *,
        provider: Provider[AsyncOpenAI] | str = 'azure',
        settings: RealtimeModelSettings | None = None,
        profile: RealtimeModelProfileSpec | None = None,
        credential: AzureTokenCredential | None = None,
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
            credential: Optional Microsoft Entra ID credential. When set, realtime requests use its
                bearer tokens instead of the resource API key.
        """
        if credential is not None and provider == 'azure':
            provider = AzureProvider.for_realtime(entra_authenticated=True)
        super().__init__(model, provider=provider, settings=settings, profile=profile)
        self.credential = credential

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

    def _realtime_ws_base(self) -> str:
        parsed = urlparse(self._azure_provider.azure_endpoint)
        return urlunparse(parsed._replace(scheme='wss', path='/openai/v1/realtime', query=''))

    def _realtime_url(self) -> str:
        return with_realtime_query(self._realtime_ws_base(), model=self.model)

    def _webrtc_http_base(self) -> str:
        parsed = urlparse(self._azure_provider.azure_endpoint)
        return urlunparse(parsed._replace(scheme='https', path='/openai/v1/', query=''))

    def _webrtc_calls_url(self) -> str:
        return self._webrtc_url('realtime/calls', webrtcfilter='on')

    async def answer_webrtc_offer(
        self,
        sdp_offer: str,
        *,
        instructions: str | None = None,
        tools: Sequence[ToolDefinition] | None = None,
        model_settings: RealtimeModelSettings | None = None,
    ) -> WebRTCAnswer:
        secret = await self.create_client_secret(instructions=instructions, tools=tools, model_settings=model_settings)
        return await _relay_sdp_offer(
            http_client=self._http_client,
            calls_url=self._webrtc_calls_url(),
            ephemeral_token=secret.value,
            provider_name=self.system,
            model_name=self.model_name,
            sdp_offer=sdp_offer,
        )

    async def _auth_headers(self) -> dict[str, str]:
        if (credential := self.credential) is not None:
            token = await run_sync(lambda: credential.get_token(_ENTRA_SCOPE))
            return {'Authorization': f'Bearer {token.token}'}
        return {'api-key': self._azure_provider.api_key}
