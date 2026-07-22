"""Azure OpenAI realtime support using the OpenAI GA protocol."""

from __future__ import annotations as _annotations

from dataclasses import InitVar, dataclass, field
from typing import Any, Protocol
from urllib.parse import urlencode, urlparse, urlunparse

from anyio.to_thread import run_sync
from openai import AsyncOpenAI

from ..exceptions import UserError
from ..providers import Provider, infer_provider
from ..providers.azure import AzureProvider
from .openai import OpenAIRealtimeModel

__all__ = ('AzureRealtimeModel', 'AzureTokenCredential')


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


@dataclass
class AzureRealtimeModel(OpenAIRealtimeModel):
    """Azure OpenAI realtime model using the OpenAI GA protocol.

    The existing [`AzureProvider`][pydantic_ai.providers.azure.AzureProvider] supplies the Azure
    resource endpoint and API key. The WebSocket transport does not use its OpenAI SDK client or
    `api_version`; it connects to the GA `/openai/v1/realtime` endpoint with an `api-key` header.

    For browser WebRTC (minting client secrets and relaying SDP offers), pass a Microsoft Entra ID
    `credential` (e.g. `azure.identity.DefaultAzureCredential()`) to authenticate the signaling calls
    with a bearer token instead of the `api-key`; the browser only ever receives the short-lived
    ephemeral secret, never the Entra token or the API key.
    """

    provider: InitVar[Provider[AsyncOpenAI] | str] = 'azure'
    credential: AzureTokenCredential | None = field(default=None, kw_only=True)

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

    def _realtime_ws_base(self) -> str:
        parsed = urlparse(self._azure_provider.azure_endpoint)
        return urlunparse(parsed._replace(scheme='wss', path='/openai/v1/realtime', query=''))

    def _realtime_url(self) -> str:
        return f'{self._realtime_ws_base()}?{urlencode({"model": self.model})}'

    def _webrtc_calls_url(self) -> str:
        # `webrtcfilter=on` restricts the events forwarded to the browser data channel to a safe subset,
        # keeping the session instructions and tool traffic on the server's control connection only.
        return f'{self._webrtc_http_base()}realtime/calls?webrtcfilter=on'

    async def _auth_headers(self) -> dict[str, str]:
        if (credential := self.credential) is not None:
            # `get_token` is synchronous (and may perform I/O), so run it off the event loop. The token is
            # cached by the credential, so this is cheap after the first call.
            token = await run_sync(lambda: credential.get_token(_ENTRA_SCOPE))
            return {'Authorization': f'Bearer {token.token}'}
        return {'api-key': self._azure_provider.api_key}
