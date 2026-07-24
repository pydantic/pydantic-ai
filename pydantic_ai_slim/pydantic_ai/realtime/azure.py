"""Azure realtime support using the OpenAI GA or Azure AI Voice Live protocol."""

from __future__ import annotations as _annotations

from collections.abc import Sequence
from dataclasses import InitVar, dataclass, field
from typing import Any, Protocol, cast
from urllib.parse import urlencode, urlparse, urlunparse

from anyio.to_thread import run_sync
from openai import AsyncOpenAI

from ..exceptions import UserError
from ..providers import Provider, infer_provider
from ..providers.azure import AzureProvider
from ..tools import ToolDefinition
from ._base import RealtimeClientSecret, RealtimeCodecEvent, RealtimeModelSettings, Transcript, WebRTCAnswer
from ._openai_protocol import (
    SemanticVAD,
    ServerVAD,
    map_event as _map_openai_event,
    resolve_base_turn_detection,
    resolve_transcription_model,
    tool_choice_config,
    tool_def_to_openai,
    turn_detection_config,
)
from ._openai_webrtc import relay_sdp_offer as _relay_sdp_offer
from .openai import OpenAIRealtimeConnection, OpenAIRealtimeModel, OpenAIRealtimeModelSettings

__all__ = ('AzureRealtimeModel', 'AzureRealtimeModelSettings', 'AzureTokenCredential')


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


class AzureRealtimeModelSettings(OpenAIRealtimeModelSettings, total=False):
    """Settings specific to Azure realtime models.

    This inherits every [`OpenAIRealtimeModelSettings`][pydantic_ai.realtime.openai.OpenAIRealtimeModelSettings]
    field, but when [`azure_voice_live`][pydantic_ai.realtime.azure.AzureRealtimeModelSettings.azure_voice_live]
    is set the Voice Live session config is built from only the cross-protocol fields — `instructions`,
    `voice`, `turn_detection` (or `azure_voice_live_turn_detection`), `input_transcription_model`,
    `output_modality`, `max_tokens`, `tool_choice`, and tools. OpenAI-only fields that Voice Live's beta
    session schema doesn't accept (e.g. `openai_input_noise_reduction`, `openai_output_speed`,
    `openai_truncation`, `openai_turn_detection`, `thinking`, `parallel_tool_calls`) are **silently
    ignored** under Voice Live; they still apply on the GA path.
    """

    azure_voice_live: bool
    """Use the Azure AI Voice Live endpoint and beta session protocol instead of the GA endpoint.

    Voice Live is a distinct Azure resource; [`AzureProvider`][pydantic_ai.providers.azure.AzureProvider]
    reads its `AZURE_VOICELIVE_ENDPOINT` / `AZURE_VOICELIVE_API_KEY` / `AZURE_VOICELIVE_API_VERSION`
    credentials as a fallback to the `AZURE_OPENAI_*` variables.
    """
    azure_voice_live_turn_detection: ServerVAD | SemanticVAD
    """Voice Live server or semantic VAD config; only applies when `azure_voice_live=True`."""


def _map_voice_live_event(data: dict[str, Any]) -> RealtimeCodecEvent | None:
    """Map Voice Live's beta text events and delegate the remaining OpenAI-compatible events."""
    event_type = data.get('type')
    if event_type == 'response.text.delta':
        delta = data.get('delta')
        return Transcript(text=delta if isinstance(delta, str) else '', is_final=False, output_text=True)
    if event_type == 'response.text.done':
        text = data.get('text')
        return Transcript(text=text if isinstance(text, str) else '', is_final=True, output_text=True)
    return _map_openai_event(data)


class _AzureRealtimeConnection(OpenAIRealtimeConnection):
    """An Azure realtime connection supporting Voice Live's beta text events."""

    def _map_event(self, data: dict[str, Any]) -> RealtimeCodecEvent | None:
        return _map_voice_live_event(data)


@dataclass
class AzureRealtimeModel(OpenAIRealtimeModel):
    """Azure realtime model using the OpenAI GA protocol or Azure AI Voice Live.

    The existing [`AzureProvider`][pydantic_ai.providers.azure.AzureProvider] supplies the Azure
    resource endpoint and API key. The WebSocket transport does not use its OpenAI SDK client or
    `api_version`. By default it connects to the GA `/openai/v1/realtime` endpoint; set
    [`azure_voice_live`][pydantic_ai.realtime.azure.AzureRealtimeModelSettings.azure_voice_live]
    to connect to `/voice-live/realtime` with the Voice Live beta session protocol. Both use an
    `api-key` header.

    Pass a Microsoft Entra ID `credential` (e.g. `azure.identity.DefaultAzureCredential()`) to
    authenticate every request to the resource — the realtime WebSocket session *and* the browser
    WebRTC signaling calls — with a bearer token instead of the `api-key` (needed when the resource is
    locked to managed identity). For browser WebRTC the browser still only ever receives the short-lived
    ephemeral secret, never the Entra token or the API key.

    <!-- TODO(voice-live): Keep GA as the default until maintainers decide whether Voice Live should
    become the default Azure realtime path. -->
    <!-- TODO(voice-live): Auto-routing Voice-Live-exclusive model names requires an agreed model/path map. -->
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
        # Azure exposes the GA realtime WebSocket under `/openai/v1/realtime`, regardless of the
        # `api_version`/path the provider's `base_url` carries, so derive it from `azure_endpoint`.
        parsed = urlparse(self._azure_provider.azure_endpoint)
        return urlunparse(parsed._replace(scheme='wss', path='/openai/v1/realtime', query=''))

    def _realtime_url(self, model_settings: OpenAIRealtimeModelSettings | None = None) -> str:
        if model_settings and model_settings.get('azure_voice_live'):
            # Voice Live is a distinct resource with its own coherent endpoint/version (see
            # `AzureProvider.voice_live_*`); never the GA endpoint or a hard-coded version.
            parsed = urlparse(self._azure_provider.voice_live_endpoint)
            return urlunparse(
                parsed._replace(
                    scheme='wss',
                    path='/voice-live/realtime',
                    query=urlencode({'api-version': self._azure_provider.voice_live_api_version, 'model': self.model}),
                )
            )
        return f'{self._realtime_ws_base()}?{urlencode({"model": self.model})}'

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
        return f'{self._webrtc_http_base()}realtime/calls?webrtcfilter=on'

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

    async def create_client_secret(
        self,
        *,
        instructions: str | None = None,
        tools: Sequence[ToolDefinition] | None = None,
        model_settings: RealtimeModelSettings | None = None,
        expires_after_seconds: int | None = None,
    ) -> RealtimeClientSecret:
        settings = self._merge_model_settings(model_settings)
        if settings and cast('AzureRealtimeModelSettings', settings).get('azure_voice_live'):
            # Voice Live negotiates WebRTC over its WebSocket control channel, not the GA
            # `/realtime/client_secrets` + `/realtime/calls` path this inherits, so the GA signaling would
            # hit the wrong endpoint. Reject it (browser WebRTC support tracked in the linked issue) rather
            # than silently minting a GA secret for a Voice Live session. Guarding `create_client_secret`
            # also covers `answer_webrtc_offer`, which mints through it.
            raise NotImplementedError(
                'Browser WebRTC is not yet supported for Azure AI Voice Live (`azure_voice_live=True`): '
                'Voice Live negotiates WebRTC over its WebSocket control channel, which this model does not '
                'implement yet. Use a WebSocket session, or the GA Azure OpenAI realtime model for browser '
                'WebRTC. See https://github.com/pydantic/pydantic-ai/issues/6702.'
            )
        return await super().create_client_secret(
            instructions=instructions,
            tools=tools,
            model_settings=model_settings,
            expires_after_seconds=expires_after_seconds,
        )

    def _session_config(
        self,
        instructions: str,
        tools: list[ToolDefinition] | None,
        model_settings: OpenAIRealtimeModelSettings | None,
    ) -> dict[str, Any]:
        settings = cast('AzureRealtimeModelSettings', self._merge_model_settings(model_settings) or {})
        if not settings.get('azure_voice_live'):
            return super()._session_config(instructions, tools, settings)

        if 'azure_voice_live_turn_detection' in settings:
            turn_detection = settings['azure_voice_live_turn_detection']
        elif 'turn_detection' in settings:
            turn_detection = resolve_base_turn_detection(settings['turn_detection'])
        else:
            turn_detection = ServerVAD()
        auto_transcription_model = 'whisper-1' if self.model.startswith('gpt-realtime') else 'azure-speech'
        transcription_model = resolve_transcription_model(
            settings.get('input_transcription_model', 'auto'), default=auto_transcription_model
        )
        config: dict[str, Any] = {
            'instructions': instructions,
            'modalities': ['text'] if settings.get('output_modality') == 'text' else ['text', 'audio'],
            'input_audio_format': 'pcm16',
            'output_audio_format': 'pcm16',
            'input_audio_sampling_rate': 24000,
            'turn_detection': turn_detection_config(turn_detection),
        }
        if transcription_model is not None:
            config['input_audio_transcription'] = {'model': transcription_model}
        if voice := settings.get('voice'):
            config['voice'] = {'type': 'openai', 'name': voice}
        if tools:
            config['tools'] = [tool_def_to_openai(tool) for tool in tools]
        if (max_tokens := settings.get('max_tokens')) is not None:
            config['max_response_output_tokens'] = max_tokens
        if (tool_choice := tool_choice_config(settings.get('tool_choice'))) is not None:
            config['tool_choice'] = tool_choice
        return config

    def _connection_class(self, model_settings: OpenAIRealtimeModelSettings) -> type[OpenAIRealtimeConnection]:
        if model_settings.get('azure_voice_live'):
            return _AzureRealtimeConnection
        return OpenAIRealtimeConnection

    async def _auth_headers(self, model_settings: OpenAIRealtimeModelSettings | None = None) -> dict[str, str]:
        if (credential := self.credential) is not None:
            # `get_token` is synchronous (and may perform I/O), so run it off the event loop. The token is
            # cached by the credential, so this is cheap after the first call.
            token = await run_sync(lambda: credential.get_token(_ENTRA_SCOPE))
            return {'Authorization': f'Bearer {token.token}'}
        # A Voice Live session authenticates against the Voice Live resource, so use its coherent key.
        if model_settings and model_settings.get('azure_voice_live'):
            return {'api-key': self._azure_provider.voice_live_api_key}
        return {'api-key': self._azure_provider.api_key}
