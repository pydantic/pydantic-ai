"""ElevenLabs realtime API provider for speech-to-speech sessions.

ElevenLabs' speech-to-speech API speaks the OpenAI Realtime protocol and works with the OpenAI SDK, so
[`ElevenLabsRealtimeModel`][pydantic_ai.realtime.elevenlabs.ElevenLabsRealtimeModel] is an
[`OpenAIRealtimeModel`][pydantic_ai.realtime.openai.OpenAIRealtimeModel] pointed at ElevenLabs: it reuses
the OpenAI codec (event mapping, session seeding, tool conversion, server-VAD config, reconnect-by-replay,
and the WebSocket connection itself), authenticated with the key and base URL of the shared
[`ElevenLabsProvider`][pydantic_ai.providers.elevenlabs.ElevenLabsProvider], and diverges only where
ElevenLabs does.

For a hosted ElevenLabs agent, use
[`ElevenLabsAgentsRealtimeModel`][pydantic_ai.realtime.elevenlabs_agents.ElevenLabsAgentsRealtimeModel] instead.

Requires the `websockets` package (the `realtime` optional group) and `openai` (the `openai` group, whose
SDK supplies the event types the shared OpenAI codec is built on):

    pip install "pydantic-ai-slim[elevenlabs-realtime]"
"""

# TODO(elevenlabs-launch): list ElevenLabs' divergences in the docstring above, as `xai.py`'s
# docstring lists xAI's (session shape, transcription, usage, reconnect, truncation, text output).

from __future__ import annotations as _annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn, cast

try:
    import websockets as websockets
    from openai.types.realtime import RealtimeResponseUsage

    from ..providers.openai import OpenAIProvider
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `websockets` and `openai` packages to use the ElevenLabs realtime model '
        '(`openai` supplies the event types of the OpenAI codec this provider reuses), you can use the '
        '`elevenlabs-realtime` optional group - `pip install "pydantic-ai-slim[elevenlabs-realtime]"`'
    ) from _import_error

from .._http import AsyncHTTPClient
from ..exceptions import UserError
from ..messages import ModelMessage
from ..models import ModelRequestParameters
from ..profiles.elevenlabs import elevenlabs_realtime_model_profile
from ..providers import Provider, infer_provider
from ..providers.elevenlabs import ElevenLabsProvider
from ..tools import ToolDefinition
from ..usage import RequestUsage
from ._openai_protocol import map_event as _map_openai_event, with_realtime_query
from .codec import RealtimeCodecEvent
from .model import RealtimeClientSecret, RealtimeModel, WebRTCAnswer
from .openai import OpenAIRealtimeConnection, OpenAIRealtimeModel, OpenAIRealtimeModelSettings
from .profiles import RealtimeModelProfile, RealtimeModelProfileSpec
from .settings import RealtimeModelSettings

if TYPE_CHECKING:
    from .model import RealtimeProviderSession

__all__ = (
    'ElevenLabsRealtimeModel',
    'ElevenLabsRealtimeModelSettings',
    'ElevenLabsRealtimeConnection',
    'map_event',
)

# --- Constants to fill from the API reference ---------------------------------------------------------------
# TODO(elevenlabs-launch): fill every constant in this block from ElevenLabs' API reference.

_OPENAI_BASE_PATH = 'v1'
"""Path under `ElevenLabsProvider.base_url` that the OpenAI SDK's `base_url` points at.

The realtime WebSocket is that URL plus `realtime`, with the HTTP scheme swapped for the WebSocket one, as on
OpenAI: `https://api.elevenlabs.io` → `https://api.elevenlabs.io/v1` → `wss://api.elevenlabs.io/v1/realtime`.
TODO(elevenlabs-launch): the real path. If the realtime API lives on a different *host* than the REST API, derive
the base URL from a host constant here instead of `ElevenLabsProvider.base_url`, and decide how data-residency
`base_url`s map onto it."""

_AUTH_HEADER = 'Authorization'
"""The handshake header carrying the API key.

TODO(elevenlabs-launch): `Authorization: Bearer <key>` is what the OpenAI SDK sends; the rest of the ElevenLabs API
uses `xi-api-key` (then `_AUTH_SCHEME = None`)."""

_AUTH_SCHEME: str | None = 'Bearer'
"""Prefix for the header value, e.g. `'Bearer'` for `Authorization: Bearer <key>`; `None` sends the bare key."""

_MODEL_QUERY_PARAM = True
"""Whether the model goes in the URL as `?model=`, as on OpenAI.

TODO(elevenlabs-launch): or in `session.update`'s `session.model`, or not at all."""

_AUTO_TRANSCRIPTION_MODEL = 'scribe_v2_realtime'
"""What `input_transcription_model='auto'` resolves to (see `resolve_transcription_model`).

TODO(elevenlabs-launch): the transcription model id the realtime API accepts, or whether transcription is always
on and takes no model at all."""

ElevenLabsRealtimeModelName = str
"""Possible ElevenLabs realtime model names.

TODO(elevenlabs-launch): `str | LatestElevenLabsRealtimeModelNames`, with a `Literal` of the published model ids, and
add them to `KnownRealtimeModelName` in `realtime/model.py` (a sync test pins that list)."""

# ------------------------------------------------------------------------------------------------------------


class ElevenLabsRealtimeModelSettings(OpenAIRealtimeModelSettings, total=False):
    """Settings specific to ElevenLabs realtime models.

    This inherits every [`OpenAIRealtimeModelSettings`][pydantic_ai.realtime.openai.OpenAIRealtimeModelSettings]
    field, since the session is configured with OpenAI's `session.update` shape. `openai_turn_detection`,
    `openai_input_noise_reduction`, `openai_output_speed`, `openai_truncation`, `openai_prune_tool_items` and
    `openai_keep_tool_items` are sent as on OpenAI. `openai_voice` is **ignored**: ElevenLabs voices are
    ElevenLabs voice IDs, set with `elevenlabs_voice`.
    """

    # TODO(elevenlabs-launch): for each inherited `openai_*` setting, confirm ElevenLabs honors it, and move any it
    # rejects or ignores to the ignored list in the docstring above (and drop it in `_session_config`), as
    # `AzureRealtimeModelSettings` does for Voice Live. In particular: semantic VAD in `openai_turn_detection`,
    # `conversation.item.delete` for `openai_prune_tool_items`, and `truncation`.
    # TODO(elevenlabs-launch): add only settings ElevenLabs documents, prefixed `elevenlabs_` (the Agents model uses
    # `elevenlabs_agents_`). Candidates: output format, language, TTS knobs (stability, similarity, style).

    elevenlabs_voice: str
    """Voice used for audio output: an ElevenLabs voice ID.

    TODO(elevenlabs-launch): voice ID vs. name, the default voice, and where it goes in `session.update`."""


def map_event(data: dict[str, Any]) -> RealtimeCodecEvent | None:
    """Map a raw ElevenLabs realtime event to a [`RealtimeCodecEvent`][pydantic_ai.realtime.codec.RealtimeCodecEvent].

    ElevenLabs speaks the OpenAI Realtime protocol, so events map via the OpenAI codec.
    """
    # TODO(elevenlabs-launch): the hook for every event that diverges from OpenAI's, as `xai.map_event` does:
    # - input transcription: incremental `.delta` (OpenAI) vs cumulative snapshots (xAI's `.updated`), and
    #   whether `.completed` is a whole snapshot (then `replace(event, cumulative=True)`);
    # - conversation lifecycle (`conversation.created`, `conversation.item.added`/`.created`), if it carries a
    #   conversation id worth exposing;
    # - error `type`s that end the session (xAI's `max_duration` → `recoverable=False`);
    # - any ElevenLabs-only events: map them to codec events, or drop them by returning `None`.
    return _map_openai_event(data)


class ElevenLabsRealtimeConnection(OpenAIRealtimeConnection):
    """A live WebSocket connection to the ElevenLabs realtime API.

    Reuses [`OpenAIRealtimeConnection`][pydantic_ai.realtime.openai.OpenAIRealtimeConnection] for the shared
    wire protocol, including reconnecting by replaying the local message history.
    """

    _provider_name = 'elevenlabs'
    _provider_label = 'ElevenLabs Realtime'
    # TODO(elevenlabs-launch): whether a `function_call_output` can be followed by an image item.
    _supports_tool_result_images = False

    # TODO(elevenlabs-launch): reconnect. The inherited `_attempt_reconnect` re-dials and replays the local
    # history into a fresh session, as on OpenAI and xAI. Override only if ElevenLabs resumes a conversation
    # server-side, or if a session-length error should re-dial right away as xAI's `max_duration` does (see
    # `XaiRealtimeConnection._decode_frame`).

    # TODO(elevenlabs-launch): push-to-talk. If, like xAI, a commit with `turn_detection: null` is answered
    # without a `response.create`, or a `response.create` after committed audio is dropped, port xAI's
    # `manual_turns` handling (`_send_event` / `_create_response` overrides) and add a simulator dialect rule.

    def _map_event(self, data: dict[str, Any]) -> RealtimeCodecEvent | None:
        return map_event(data)

    def _map_response_usage(self, usage: RealtimeResponseUsage | None) -> RequestUsage | None:
        # TODO(elevenlabs-launch): usage. Check `response.done`'s `usage` against OpenAI's shape: token buckets,
        # any provider-only fields in `model_extra` (xAI adds a *running-total* `billable_audio_seconds`), and
        # what genai-prices needs to price it (audio seconds? characters?). Put provider-only buckets in
        # `details`, and set the unit pricing knows when billing isn't per token, so `cost_limit` doesn't price
        # the session at a confident zero.
        return super()._map_response_usage(usage)


class _ElevenLabsOpenAIProvider(OpenAIProvider):
    """The OpenAI SDK pointed at ElevenLabs' OpenAI-compatible API, with an `ElevenLabsProvider`'s key and HTTP client.

    Named `'elevenlabs'` and resolving ElevenLabs' realtime profile, so the behavior
    [`OpenAIRealtimeModel`][pydantic_ai.realtime.openai.OpenAIRealtimeModel] derives from its provider
    (`system`, `profile`, the WebSocket URL) reports ElevenLabs rather than OpenAI.
    """

    def __init__(self, elevenlabs_provider: ElevenLabsProvider) -> None:
        super().__init__(
            base_url=f'{elevenlabs_provider.base_url}/{_OPENAI_BASE_PATH}',
            api_key=elevenlabs_provider.api_key,
            http_client=elevenlabs_provider.client,
        )
        self.elevenlabs_provider = elevenlabs_provider

    @property
    def name(self) -> str:
        return 'elevenlabs'

    @staticmethod
    def realtime_model_profile(model_name: str) -> RealtimeModelProfile:
        return elevenlabs_realtime_model_profile(model_name)


@dataclass(init=False)
class ElevenLabsRealtimeModel(OpenAIRealtimeModel):
    """ElevenLabs realtime API model, for ElevenLabs' speech-to-speech models.

    Pass `provider='elevenlabs'` (the default, which reads `ELEVENLABS_API_KEY`) or an
    [`ElevenLabsProvider`][pydantic_ai.providers.elevenlabs.ElevenLabsProvider] constructed with `api_key=`
    or a data-residency `base_url=`. The session is configured and driven through the OpenAI Realtime
    protocol; WebRTC is not supported.

    Args:
        model: The model name.
        provider: The provider to use for authentication and the base URL. Defaults to `'elevenlabs'`.
        settings: [Model settings][pydantic_ai.realtime.elevenlabs.ElevenLabsRealtimeModelSettings] used as
            defaults for realtime sessions. With a [`reconnect`][pydantic_ai.realtime.RealtimeModelSettings.reconnect]
            policy, a dropped connection is re-dialed and the local message history replayed into it.
        profile: Optional override for the [realtime model profile][pydantic_ai.realtime.RealtimeModelProfile],
            merged over the provider's — a partial dict, or a callable taking the resolved profile and
            returning the one to use.
    """

    # TODO(elevenlabs-launch): the model-name example in `Args` (e.g. "The model name, e.g. `<id>`").
    # TODO(elevenlabs-launch): WebRTC. If ElevenLabs serves OpenAI's `/realtime/calls`, `/realtime/client_secrets`
    # and the `?call_id=` sideband, delete the WebRTC overrides at the bottom of this class to inherit OpenAI's,
    # and set `supports_webrtc` in the profile.
    # TODO(elevenlabs-launch): the handshake. The inherited `connect` waits for `session.created`, sends
    # `session.update`, and waits for `session.updated`; frames in between fail the handshake. If a recording
    # shows ElevenLabs sending one there (xAI's `conversation.created`), override `connect` as `XaiRealtimeModel`
    # does to pass `on_unexpected_during_update`. Also check how a bad key or model name is refused (HTTP status
    # on the upgrade vs. an `error` frame vs. a close code), so it surfaces as `ModelHTTPError`/`ModelAPIError`.

    _connection_type: ClassVar[type[OpenAIRealtimeConnection]] = ElevenLabsRealtimeConnection
    _auto_transcription_model: ClassVar[str] = _AUTO_TRANSCRIPTION_MODEL

    def __init__(
        self,
        model: ElevenLabsRealtimeModelName,
        *,
        provider: Provider[AsyncHTTPClient] | str = 'elevenlabs',
        settings: RealtimeModelSettings | None = None,
        profile: RealtimeModelProfileSpec | None = None,
    ) -> None:
        if isinstance(provider, str):
            provider = infer_provider(provider)
        if not isinstance(provider, ElevenLabsProvider):
            # Reading `api_key` off a foreign provider would fail with an `AttributeError` instead of the real mistake.
            raise UserError(
                "`ElevenLabsRealtimeModel` requires an `ElevenLabsProvider` or `provider='elevenlabs'`; "
                f'got {provider.name!r}.'
            )
        super().__init__(model, provider=_ElevenLabsOpenAIProvider(provider), settings=settings, profile=profile)

    @property
    def _elevenlabs_provider(self) -> ElevenLabsProvider:
        return cast(_ElevenLabsOpenAIProvider, self._provider).elevenlabs_provider

    @property
    def base_url(self) -> str:
        return self._elevenlabs_provider.base_url

    def _realtime_url(self, model_settings: OpenAIRealtimeModelSettings | None = None) -> str:
        del model_settings
        return with_realtime_query(self._realtime_ws_base(), model=self.model if _MODEL_QUERY_PARAM else None)

    async def _auth_headers(self, model_settings: OpenAIRealtimeModelSettings | None = None) -> dict[str, str]:
        del model_settings
        api_key = self._elevenlabs_provider.api_key
        return {_AUTH_HEADER: f'{_AUTH_SCHEME} {api_key}' if _AUTH_SCHEME else api_key}

    def _session_config(
        self,
        instructions: str,
        tools: list[ToolDefinition] | None,
        *,
        model_settings: OpenAIRealtimeModelSettings | None,
    ) -> dict[str, Any]:
        # TODO(elevenlabs-launch): the `session.update` shape. This is OpenAI's GA shape (`type: 'realtime'`,
        # `voice`/`turn_detection` nested under `audio`); xAI instead puts `voice`/`turn_detection` at the session
        # top level and has no `type`/`output_modalities`. Check each field against their reference, including
        # the audio format names (`audio/pcm` + `rate`, or ElevenLabs' `pcm_16000`), and drop any the server
        # rejects or silently ignores (then document it in `ElevenLabsRealtimeModelSettings`).
        settings = cast('ElevenLabsRealtimeModelSettings', self._merge_model_settings(model_settings) or {})
        config = super()._session_config(instructions, tools, model_settings=settings)
        audio_output: dict[str, Any] = config['audio']['output']
        # `openai_voice` names an OpenAI voice, which ElevenLabs doesn't have; see `ElevenLabsRealtimeModelSettings`.
        audio_output.pop('voice', None)
        if voice := settings.get('elevenlabs_voice'):
            audio_output['voice'] = voice
        # TODO(elevenlabs-launch): reasoning. If a model takes `reasoning.effort`, set `supports_thinking` in the
        # profile; the inherited config then maps the unified `thinking` setting as OpenAI does.
        return config

    # WebRTC: the OpenAI signaling endpoints this class would inherit aren't known to exist on ElevenLabs.

    def connect_webrtc(
        self,
        session: RealtimeProviderSession,
        *,
        messages: Sequence[ModelMessage],
        model_settings: RealtimeModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> NoReturn:
        self._raise_unsupported_webrtc('connect_webrtc')

    async def create_client_secret(
        self,
        *,
        instructions: str | None = None,
        tools: Sequence[ToolDefinition] | None = None,
        model_settings: RealtimeModelSettings | None = None,
        expires_after_seconds: int | None = None,
    ) -> RealtimeClientSecret:
        self._raise_unsupported_webrtc('create_client_secret')

    async def answer_webrtc_offer(
        self,
        sdp_offer: str,
        *,
        instructions: str | None = None,
        tools: Sequence[ToolDefinition] | None = None,
        model_settings: RealtimeModelSettings | None = None,
        message_history: Sequence[ModelMessage] | None = None,
    ) -> WebRTCAnswer:
        self._raise_unsupported_webrtc('answer_webrtc_offer')

    def _check_hang_up(self, session: RealtimeProviderSession) -> None:
        RealtimeModel._check_hang_up(self, session)
