"""Azure OpenAI realtime support using the OpenAI GA protocol."""

from __future__ import annotations as _annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol
from urllib.parse import urlparse, urlunparse

from anyio.to_thread import run_sync
from openai import AsyncOpenAI

from ..exceptions import UserError
from ..providers import Provider, infer_provider
from ..providers.azure import AzureProvider
from ..tools import ToolDefinition
from ._base import RealtimeModelProfileSpec, RealtimeModelSettings, ReconnectPolicy, WebRTCAnswer
from ._openai_protocol import with_realtime_query
from ._openai_webrtc import relay_sdp_offer as _relay_sdp_offer
from .openai import OpenAIRealtimeConnection, OpenAIRealtimeModel

__all__ = ('AzureRealtimeModel', 'AzureRealtimeConnection', 'AzureTokenCredential')


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
    """Structural type for a Microsoft Entra ID credential, e.g. `azure.identity.DefaultAzureCredential`.

    Any object with a synchronous `get_token(*scopes) -> token` method (the `azure-core`
    `TokenCredential` interface) is accepted, so `AzureRealtimeModel` doesn't depend on `azure-identity`.
    """

    def get_token(self, *scopes: str, **kwargs: Any) -> _AccessToken: ...


# Microsoft Entra ID token scope for the Azure OpenAI data plane, per the Azure realtime WebRTC guide.
# Minting a realtime client secret (or relaying a WebRTC offer) with a `DefaultAzureCredential` token
# requires the caller to hold the "Cognitive Services User" role on the resource.
_ENTRA_SCOPE = 'https://ai.azure.com/.default'


@dataclass(init=False)
class AzureRealtimeModel(OpenAIRealtimeModel):
    """Azure OpenAI realtime model using the OpenAI GA protocol.

    The existing [`AzureProvider`][pydantic_ai.providers.azure.AzureProvider] supplies the Azure
    resource endpoint and API key. The WebSocket transport does not use its OpenAI SDK client or
    `api_version`; it connects to the GA `/openai/v1/realtime` endpoint with an `api-key` header.

    Pass a Microsoft Entra ID `credential` (e.g. `azure.identity.DefaultAzureCredential()`) to
    authenticate every request to the resource — the realtime WebSocket session *and* the browser
    WebRTC signaling calls — with a bearer token instead of the `api-key` (needed when the resource is
    locked to managed identity). For browser WebRTC the browser still only ever receives the short-lived
    ephemeral secret, never the Entra token or the API key.
    """

    _connection_type: ClassVar[type[OpenAIRealtimeConnection]] = AzureRealtimeConnection

    credential: AzureTokenCredential | None = None
    """Microsoft Entra ID credential; when set, every request to the resource uses a bearer token."""

    def __init__(
        self,
        model: str = 'gpt-realtime',
        *,
        provider: Provider[AsyncOpenAI] | str = 'azure',
        settings: RealtimeModelSettings | None = None,
        profile: RealtimeModelProfileSpec | None = None,
        reconnect: ReconnectPolicy | None = None,
        credential: AzureTokenCredential | None = None,
    ) -> None:
        """Create an Azure OpenAI realtime model.

        Args:
            model: The Azure *deployment* name, which is what the realtime URL and the profile lookup
                use. Azure deployments are conventionally named after their model; when yours isn't,
                `profile` is how to correct the facts inferred from the name.
            provider: The provider supplying the resource endpoint and API key. Defaults to `'azure'`.
            settings: Model settings used as defaults for realtime sessions.
            profile: Optional override for the [realtime model profile][pydantic_ai.realtime.RealtimeModelProfile],
                merged over the provider's — a partial dict, or a callable taking the resolved profile
                and returning the one to use.
            reconnect: Optional [`ReconnectPolicy`][pydantic_ai.realtime.ReconnectPolicy] to
                transparently recover from a dropped connection.
            credential: Optional Microsoft Entra ID credential (e.g. `azure.identity.DefaultAzureCredential()`).
                When set, the realtime WebSocket session *and* the browser WebRTC signaling calls
                authenticate with a bearer token instead of the resource `api-key`.
        """
        if credential is not None and provider == 'azure':
            # An Entra credential authenticates every request, so the resource key is never read — and a
            # resource locked to managed identity has none to give. Build the default provider knowing
            # that, rather than demanding `AZURE_OPENAI_API_KEY` for a value this model won't use.
            provider = AzureProvider.for_realtime(entra_authenticated=True)
        super().__init__(model, provider=provider, settings=settings, profile=profile, reconnect=reconnect)
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
        # Azure exposes the WebRTC signaling endpoints under the GA `/openai/v1/` path, regardless of the
        # `api_version`/path the provider's `base_url` carries. Deriving from `azure_endpoint` (rather than
        # inheriting the OpenAI behavior of appending to `base_url`) forces `/openai/v1/` here just as
        # `_realtime_ws_base` does for the WebSocket handshake — without it, signaling URLs silently drop
        # the `/v1` and Azure 404s. Always ends in `/` so callers can append `realtime/...`.
        parsed = urlparse(self._azure_provider.azure_endpoint)
        return urlunparse(parsed._replace(scheme='https', path='/openai/v1/', query=''))

    def _webrtc_calls_url(self) -> str:
        # `webrtcfilter=on` restricts the events forwarded to the browser data channel to a safe subset,
        # keeping the session instructions and tool traffic on the server's control connection only.
        return self._webrtc_url('realtime/calls', webrtcfilter='on')

    async def answer_webrtc_offer(
        self,
        sdp_offer: str,
        *,
        instructions: str | None = None,
        tools: Sequence[ToolDefinition] | None = None,
        model_settings: RealtimeModelSettings | None = None,
    ) -> WebRTCAnswer:
        # Azure's `/realtime/calls` rejects the resource api-key / Entra token with a 401 (`This operation
        # requires ephemeral tokens`). So — unlike OpenAI's single-step multipart relay — Azure negotiates
        # in two steps: mint a short-lived client secret server-side (with the api-key or Entra token),
        # which binds the session config, then relay the raw SDP offer authenticated with that secret.
        secret = await self.create_client_secret(instructions=instructions, tools=tools, model_settings=model_settings)
        return await _relay_sdp_offer(
            http_client=self._http_client,
            calls_url=self._webrtc_calls_url(),
            ephemeral_token=secret.value,
            provider_name=self.system,
            sdp_offer=sdp_offer,
        )

    async def _auth_headers(self) -> dict[str, str]:
        if (credential := self.credential) is not None:
            # `get_token` is synchronous (and may perform I/O), so run it off the event loop. The token is
            # cached by the credential, so this is cheap after the first call.
            token = await run_sync(lambda: credential.get_token(_ENTRA_SCOPE))
            return {'Authorization': f'Bearer {token.token}'}
        return {'api-key': self._azure_provider.api_key}
