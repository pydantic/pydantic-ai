"""Azure realtime support using the OpenAI GA or Azure AI Voice Live protocol."""

from __future__ import annotations as _annotations

from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, cast
from urllib.parse import urlencode, urlparse, urlunparse

from anyio.to_thread import run_sync
from openai import AsyncOpenAI
from openai.types.realtime.realtime_audio_config_output import VoiceID
from pydantic import BaseModel

from ..exceptions import UserError
from ..providers import Provider, infer_provider
from ..providers.azure import AzureProvider
from ..tools import ToolDefinition
from ._openai_protocol import (
    SemanticVAD,
    ServerVAD,
    map_event as _map_openai_event,
    resolve_base_turn_detection,
    resolve_transcription_model,
    tool_choice_config,
    tool_def_to_openai,
    turn_detection_config,
    with_realtime_query,
)
from ._openai_webrtc import relay_sdp_offer as _relay_sdp_offer
from ._utils import resolve_advertised_tools
from .codec import OutputTranscript, RealtimeCodecEvent
from .model import RealtimeClientSecret, WebRTCAnswer
from .openai import (
    OpenAIRealtimeConnection,
    OpenAIRealtimeModel,
    OpenAIRealtimeModelName,
    OpenAIRealtimeModelSettings,
)
from .profiles import RealtimeModelProfile, RealtimeModelProfileSpec, merge_realtime_profile
from .settings import RealtimeModelSettings

if TYPE_CHECKING:
    from ..messages import ModelMessage
    from ..models import ModelRequestParameters
    from .model import RealtimeProviderSession

__all__ = (
    'AzureRealtimeModel',
    'AzureRealtimeConnection',
    'AzureRealtimeModelProfile',
    'AzureRealtimeModelSettings',
    'AzureTokenCredential',
)

LatestAzureRealtimeModelNames = Literal['gpt-realtime']
AzureRealtimeModelName = OpenAIRealtimeModelName

LatestAzureRealtimeTranscriptionModelNames = Literal['azure-speech', 'mai-transcribe']
AzureRealtimeTranscriptionModelName = str | LatestAzureRealtimeTranscriptionModelNames

AzureRealtimeApi = Literal['azure_openai', 'voice_live']
"""An Azure realtime speech-to-speech API a model can be reached through: the Azure OpenAI GA realtime
API (`/openai/v1/realtime`) or [Azure AI Voice Live](https://learn.microsoft.com/azure/ai-services/speech-service/voice-live)
(`/voice-live/realtime`, selected with [`azure_voice_live=True`][pydantic_ai.realtime.azure.AzureRealtimeModelSettings.azure_voice_live])."""


class AzureRealtimeModelProfile(RealtimeModelProfile, total=False):
    """A [`RealtimeModelProfile`][pydantic_ai.realtime.RealtimeModelProfile] with the Azure-specific facts.

    Read via [`AzureRealtimeModel.profile`][pydantic_ai.realtime.azure.AzureRealtimeModel]. Pass a partial
    one as `profile=` to correct what's inferred from a deployment name that doesn't match its model.
    """

    azure_realtime_apis: frozenset[AzureRealtimeApi]
    """Which Azure realtime APIs serve this model, when only one does — the constraint that routes it.

    A model served only by Voice Live carries `{'voice_live'}` and routes there automatically; a GA-only
    model carries `{'azure_openai'}` and rejects
    [`azure_voice_live=True`][pydantic_ai.realtime.azure.AzureRealtimeModelSettings.azure_voice_live].
    Absent for a model served by *both* (e.g. `gpt-realtime`) or a name the table below doesn't recognize
    (e.g. a future `gpt-realtime-3`): either way it defaults to GA and reaches Voice Live only when
    `azure_voice_live=True` is set. Pass a `profile=` override to constrain a deployment named after
    something the table can't place."""


# Azure realtime models whose serving API is *constrained*, keyed by that API. Deliberately lists only
# the single-API models — a model served by both (`gpt-realtime`, `gpt-realtime-mini`, `gpt-realtime-1.5`)
# and any unrecognized name are left out, defaulting to GA and reaching Voice Live only via an explicit
# `azure_voice_live=True`. So a future model found to be Voice-Live-only is auto-routed simply by listing
# it here, while forward compatibility is preserved for names nobody has classified yet. Bases are matched
# at a `-`/`.` boundary (see `_name_matches`) so a version number like `gpt-realtime-2` doesn't swallow a
# dated `gpt-realtime-2025-…`; order matters, with the GA-only `-realtime` variants checked before the
# bare cascade names that also start with `gpt-4o`.
_AZURE_OPENAI: frozenset[AzureRealtimeApi] = frozenset({'azure_openai'})
_VOICE_LIVE: frozenset[AzureRealtimeApi] = frozenset({'voice_live'})
_AZURE_REALTIME_API_BASES: tuple[tuple[tuple[str, ...], frozenset[AzureRealtimeApi]], ...] = (
    # GA-only realtime models — not served by Voice Live, so `azure_voice_live=True` is rejected.
    (
        (
            'gpt-4o-realtime',
            'gpt-4o-mini-realtime',
            'gpt-realtime-2',
            'gpt-realtime-translate',
            'gpt-realtime-whisper',
            'gpt-live-transcribe',
        ),
        _AZURE_OPENAI,
    ),
    # Voice-Live-only models — auto-routed there. Native-audio models, then the cascade chat families
    # served via Azure speech-to-text and text-to-speech (`gpt-4o` also covers `gpt-4o-mini`; `gpt-5` and
    # `gpt-4.1` cover their point releases like `gpt-5.2`).
    (('phi4-mm-realtime', 'azure-realtime', 'gpt-4o', 'gpt-4.1', 'gpt-5', 'phi4-mini'), _VOICE_LIVE),
)


def _name_matches(model_name: str, base: str) -> bool:
    """Whether a deployment name is `base` or `base` followed by a `-`/`.` version or date boundary."""
    return model_name == base or (model_name.startswith(base) and model_name[len(base)] in '-.')


def _default_azure_realtime_apis(model_name: str) -> frozenset[AzureRealtimeApi] | None:
    """The API that serves a constrained Azure realtime model, or `None` for a both-API or unknown name."""
    return next(
        (apis for bases, apis in _AZURE_REALTIME_API_BASES if any(_name_matches(model_name, base) for base in bases)),
        None,
    )


def _route_voice_live(use_voice_live: bool, apis: frozenset[AzureRealtimeApi] | None, model_name: str) -> bool:
    """Resolve whether to use Voice Live, given the explicit setting and the model's serving APIs.

    A recognized Voice-Live-only model routes there automatically; a recognized GA-only model rejects the
    setting. An unrecognized model (`apis is None`) is unconstrained — the setting decides.
    """
    if apis is None:
        return use_voice_live
    if use_voice_live and 'voice_live' not in apis:
        raise UserError(
            f'`{model_name}` is served by the Azure OpenAI GA realtime API, not Azure AI Voice Live, so '
            '`azure_voice_live=True` cannot be used with it. Remove the setting, or choose a model that '
            'Voice Live serves.'
        )
    if 'azure_openai' not in apis:  # Voice-Live-only: route there whether or not the setting was passed.
        return True
    return use_voice_live


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


class AzureRealtimeModelSettings(OpenAIRealtimeModelSettings, total=False):
    """Settings specific to Azure realtime models.

    This inherits every [`OpenAIRealtimeModelSettings`][pydantic_ai.realtime.openai.OpenAIRealtimeModelSettings]
    field, but when [`azure_voice_live`][pydantic_ai.realtime.azure.AzureRealtimeModelSettings.azure_voice_live]
    is set the Voice Live session config is built from only the cross-protocol fields — `instructions`,
    `openai_voice` (by name), `turn_detection` (or `azure_voice_live_turn_detection`),
    `input_transcription_model`, `output_modality`, `max_tokens`, `tool_choice`, and tools. The
    inherited `openai_*` fields, plus `thinking` and `parallel_tool_calls`, are **silently ignored**
    under Voice Live; they still apply on the GA path.

    They fall into two groups, and only the first is settled:

    - `openai_output_speed`, `openai_turn_detection`, `thinking`, and `parallel_tool_calls` have no
      counterpart in Voice Live's beta session object (see the recorded `session.created` payload in
      `tests/realtime/cassettes/test_azure_voice_live_ws/`), so there is nothing to map them to.
      Voice Live's own turn detection is configured with `azure_voice_live_turn_detection`.
    - `openai_input_noise_reduction` and `openai_truncation` *do* have counterparts —
      `input_audio_noise_reduction` and `truncation_strategy` — but under Azure's own vocabulary, which
      the recording pins as `null` and so does not evidence. Mapping OpenAI's values onto them would be
      guessing at the accepted shape, so they stay unmapped until a recording proves it; a dedicated
      `azure_voice_live_*` setting is the natural home when it does.
    """

    azure_voice_live: bool
    """Use the Azure AI Voice Live endpoint and beta session protocol instead of the GA endpoint.

    Voice Live is a distinct Azure resource; [`AzureProvider`][pydantic_ai.providers.azure.AzureProvider]
    reads its `AZURE_VOICELIVE_ENDPOINT` / `AZURE_VOICELIVE_API_KEY` / `AZURE_VOICELIVE_API_VERSION`
    credentials as a fallback to the `AZURE_OPENAI_*` variables.
    """
    azure_voice_live_turn_detection: ServerVAD | SemanticVAD
    """Voice Live server or semantic VAD config; only applies when `azure_voice_live=True`."""


class _VoiceLiveSession(BaseModel):
    model: str | None = None


class _VoiceLiveSessionCreated(BaseModel):
    """The narrow slice of Voice Live's beta `session.created` frame the handshake reads."""

    session: _VoiceLiveSession


def _map_voice_live_event(data: dict[str, Any]) -> RealtimeCodecEvent | None:
    """Map Voice Live's beta text events and delegate the remaining OpenAI-compatible events."""
    event_type = data.get('type')
    if event_type in ('response.text.delta', 'response.text.done'):
        is_final = event_type == 'response.text.done'
        content = data.get('text' if is_final else 'delta')
        return OutputTranscript(
            text=content if isinstance(content, str) else '',
            is_final=is_final,
            # Carried through like the shared OpenAI mapper does: the session keys part identity off
            # `item_id`, so dropping it leaves the recorded `TextPart` without a provider id *and* stops
            # a second output item in the same response from finalizing the first — two replies in one
            # turn would accumulate into a single part.
            item_id=item_id if isinstance(item_id := data.get('item_id'), str) and item_id else None,
            output_text=True,
        )
    return _map_openai_event(data)


class _VoiceLiveRealtimeConnection(AzureRealtimeConnection):
    """An Azure realtime connection supporting Voice Live's beta text events.

    Subclasses the GA Azure connection rather than the OpenAI one so a Voice Live session that drops or
    rejects content names Azure too.
    """

    def _map_event(self, data: dict[str, Any]) -> RealtimeCodecEvent | None:
        return _map_voice_live_event(data)


@dataclass(init=False)
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

    A model served only by Voice Live (e.g. the cascade chat models like `gpt-5`, or `phi4-mm-realtime`)
    routes there automatically; a model served by both defaults to GA and needs `azure_voice_live=True`
    for Voice Live; a GA-only model rejects the setting. See
    [`AzureRealtimeModelProfile.azure_realtime_apis`][pydantic_ai.realtime.azure.AzureRealtimeModelProfile.azure_realtime_apis].
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
    def profile(self) -> RealtimeModelProfile:
        """The Azure realtime profile, with the model's serving APIs and minus what Voice Live can't do.

        Stamps [`azure_realtime_apis`][pydantic_ai.realtime.azure.AzureRealtimeModelProfile.azure_realtime_apis]
        for a recognized model (a `profile=` override wins), which routes between the GA API and Voice
        Live. And because Voice Live negotiates WebRTC over its own WebSocket control channel rather than
        the GA signaling endpoints this model inherits, a model configured for Voice Live reports no
        WebRTC support and the signaling methods refuse. Voice Live selected per session instead can't be
        seen from here, so those calls still refuse at the point of use.
        """
        profile = super().profile
        if (apis := _default_azure_realtime_apis(self.model)) is not None and 'azure_realtime_apis' not in profile:
            profile = merge_realtime_profile(AzureRealtimeModelProfile(azure_realtime_apis=apis), profile)
        # Voice Live has no GA-style browser WebRTC, so strip it whenever this model will use Voice Live —
        # forced by the setting, or because it's only served there and auto-routes.
        resolved_apis = cast('AzureRealtimeModelProfile', profile).get('azure_realtime_apis')
        forced = self.settings is not None and cast('AzureRealtimeModelSettings', self.settings).get('azure_voice_live')
        if forced or (resolved_apis is not None and 'azure_openai' not in resolved_apis):
            profile = merge_realtime_profile(profile, RealtimeModelProfile(supports_webrtc=False))
        return profile

    def _resolve_voice_live(self, model_settings: RealtimeModelSettings | None) -> bool:
        """Whether this session uses Azure AI Voice Live rather than the GA realtime API — the sole authority.

        `azure_voice_live` chooses the path for a model served by both; the model's recognized
        [`azure_realtime_apis`][pydantic_ai.realtime.azure.AzureRealtimeModelProfile.azure_realtime_apis]
        then auto-route a Voice-Live-only model and reject `azure_voice_live=True` on a GA-only one, rather
        than letting the wrong endpoint fail the handshake. An unrecognized model defaults to GA and
        reaches Voice Live only when `azure_voice_live=True` is set.
        """
        use_voice_live = bool(
            model_settings and cast('AzureRealtimeModelSettings', model_settings).get('azure_voice_live')
        )
        apis = cast('AzureRealtimeModelProfile', self.profile).get('azure_realtime_apis')
        return _route_voice_live(use_voice_live, apis, self.model)

    def _reject_webrtc_if_voice_live(self, settings: RealtimeModelSettings | None) -> None:
        """Reject browser WebRTC signaling for a Voice Live session, explicit or auto-routed.

        Voice Live negotiates WebRTC over its own WebSocket control channel, not the GA
        `/realtime/client_secrets` + `/realtime/calls` path this class inherits, so GA signaling would
        hit the wrong endpoint. Gate on `_resolve_voice_live` (the routing authority), not the raw
        `azure_voice_live` setting, so a Voice-Live-only model that auto-routes there without the setting
        (e.g. `gpt-5`) is rejected too rather than minting a GA secret for a Voice Live session. Shared by
        `create_client_secret` (which also covers `answer_webrtc_offer`) and `connect_webrtc`.
        """
        if self._resolve_voice_live(settings):
            raise UserError(
                'Browser WebRTC is not yet supported for Azure AI Voice Live: Voice Live negotiates WebRTC '
                'over its WebSocket control channel, which this model does not implement yet. Use a WebSocket '
                'session, or the GA Azure OpenAI realtime model for browser WebRTC. '
                'See https://github.com/pydantic/pydantic-ai/issues/6702.'
            )

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
        if self._resolve_voice_live(model_settings):
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

    async def create_client_secret(
        self,
        *,
        instructions: str | None = None,
        tools: Sequence[ToolDefinition] | None = None,
        model_settings: RealtimeModelSettings | None = None,
        expires_after_seconds: int | None = None,
    ) -> RealtimeClientSecret:
        self._reject_webrtc_if_voice_live(self._merge_model_settings(model_settings))
        return await super().create_client_secret(
            instructions=instructions,
            tools=tools,
            model_settings=model_settings,
            expires_after_seconds=expires_after_seconds,
        )

    @asynccontextmanager
    async def connect_webrtc(
        self,
        session: RealtimeProviderSession,
        *,
        messages: Sequence[ModelMessage],
        model_settings: RealtimeModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncGenerator[OpenAIRealtimeConnection]:
        self._reject_webrtc_if_voice_live(self._merge_model_settings(model_settings))
        async with super().connect_webrtc(
            session,
            messages=messages,
            model_settings=model_settings,
            model_request_parameters=model_request_parameters,
        ) as connection:
            yield connection

    def _session_config(
        self,
        instructions: str,
        tools: list[ToolDefinition] | None,
        *,
        model_settings: OpenAIRealtimeModelSettings | None,
    ) -> dict[str, Any]:
        settings = cast('AzureRealtimeModelSettings', self._merge_model_settings(model_settings) or {})
        if not self._resolve_voice_live(settings):
            return super()._session_config(instructions, tools, model_settings=settings)

        if 'azure_voice_live_turn_detection' in settings:
            turn_detection: ServerVAD | SemanticVAD | None = settings['azure_voice_live_turn_detection']
        elif 'turn_detection' in settings:
            turn_detection = resolve_base_turn_detection(settings['turn_detection'])
        else:
            turn_detection = ServerVAD(type='server_vad')
        auto_transcription_model = 'whisper-1' if self.model.startswith('gpt-realtime') else 'azure-speech'
        transcription_model = resolve_transcription_model(
            settings.get('input_transcription_model', 'auto'), default=auto_transcription_model
        )
        config: dict[str, Any] = {
            'instructions': instructions,
            'modalities': ['text'] if settings.get('output_modality') == 'text' else ['text', 'audio'],
            'input_audio_format': 'pcm16',
            'output_audio_format': 'pcm16',
            'input_audio_sampling_rate': self.profile.get('audio_input_sample_rate', 24000),
            'turn_detection': turn_detection_config(turn_detection),
        }
        if transcription_model is not None:
            config['input_audio_transcription'] = {'model': transcription_model}
        if voice := settings.get('openai_voice'):
            if isinstance(voice, VoiceID):
                # Voice Live's session schema addresses a voice by provider + *name*, with no place for
                # an OpenAI custom-voice id, so accepting one would silently drop it.
                raise UserError(
                    'Azure AI Voice Live does not accept an OpenAI custom `VoiceID`; set `openai_voice` '
                    'to a voice name instead.'
                )
            config['voice'] = {'type': 'openai', 'name': voice}
        advertised_tools, tool_choice = resolve_advertised_tools(list(tools or []), settings.get('tool_choice'))
        if advertised_tools:
            config['tools'] = [tool_def_to_openai(tool) for tool in advertised_tools]
        if (max_tokens := settings.get('max_tokens')) is not None:
            config['max_response_output_tokens'] = max_tokens
        if tool_choice is not None:
            config['tool_choice'] = tool_choice_config(tool_choice)
        return config

    def _connection_class(self, model_settings: OpenAIRealtimeModelSettings) -> type[OpenAIRealtimeConnection]:
        if self._resolve_voice_live(model_settings):
            return _VoiceLiveRealtimeConnection
        # The GA path: `_connection_type`, i.e. the Azure-labeled connection.
        return super()._connection_class(model_settings)

    def _session_model_name(self, created: dict[str, Any], model_settings: OpenAIRealtimeModelSettings) -> str | None:
        if self._resolve_voice_live(model_settings):
            # Voice Live's beta `session.created` has no GA `type` discriminator, so the SDK's
            # `SessionCreatedEvent` rejects it; validate the narrow slice actually read instead.
            return _VoiceLiveSessionCreated.model_validate(created).session.model
        return super()._session_model_name(created, model_settings)

    async def _auth_headers(self, model_settings: OpenAIRealtimeModelSettings | None = None) -> dict[str, str]:
        if (credential := self.credential) is not None:
            # `get_token` is synchronous (and may perform I/O), so run it off the event loop. The token is
            # cached by the credential, so this is cheap after the first call.
            token = await run_sync(lambda: credential.get_token(_ENTRA_SCOPE))
            return {'Authorization': f'Bearer {token.token}'}
        # A Voice Live session authenticates against the Voice Live resource, so use its coherent key.
        if self._resolve_voice_live(model_settings):
            return {'api-key': self._azure_provider.voice_live_api_key}
        return {'api-key': self._azure_provider.api_key}
