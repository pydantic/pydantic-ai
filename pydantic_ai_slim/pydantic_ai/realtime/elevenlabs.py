"""ElevenLabs realtime API provider for Eleven Realtime speech-to-speech sessions.

Connects to `wss://api.elevenlabs.io/v1/convai/realtime` over a WebSocket. Eleven Realtime speaks the OpenAI
Realtime protocol, so [`ElevenLabsRealtimeModel`][pydantic_ai.realtime.elevenlabs.ElevenLabsRealtimeModel] is an
[`OpenAIRealtimeModel`][pydantic_ai.realtime.openai.OpenAIRealtimeModel] pointed at ElevenLabs: it reuses the
OpenAI codec (event mapping, session seeding, tool conversion, reconnect-by-replay, and the WebSocket connection
itself), authenticated with the key and base URL of the shared
[`ElevenLabsProvider`][pydantic_ai.providers.elevenlabs.ElevenLabsProvider], and diverges only where ElevenLabs
does:

- authentication uses the `xi-api-key` header, and the model is chosen with `session.model` in `session.update`
  (the server ignores OpenAI's `?model=` query parameter);
- turn detection is semantic VAD only: the server rejects `server_vad`, so the cross-provider
  [`turn_detection`][pydantic_ai.realtime.RealtimeModelSettings.turn_detection] setting maps onto semantic VAD;
- input transcription is always on, even with `input_transcription_model=None`, and takes no model, only a
  language (see
  [`elevenlabs_language`][pydantic_ai.realtime.elevenlabs.ElevenLabsRealtimeModelSettings.elevenlabs_language]);
- voices are ElevenLabs voices, set with
  [`elevenlabs_voice`][pydantic_ai.realtime.elevenlabs.ElevenLabsRealtimeModelSettings.elevenlabs_voice];
- `response.done` carries no usage, so sessions report no token counts;
- `conversation.item.delete` is not implemented, so `openai_prune_tool_items` is ignored;
- images are dropped from user messages, and seeded history can only carry text;
- text output (`output_modalities: ['text']`) ignores audio input altogether, so `output_modality='text'` is
  unsupported;
- the server ends a session after `max_duration_seconds` (600 by default; see
  [`elevenlabs_max_duration_seconds`][pydantic_ai.realtime.elevenlabs.ElevenLabsRealtimeModelSettings.elevenlabs_max_duration_seconds])
  or after 60 seconds without a client message. With a
  [`reconnect`][pydantic_ai.realtime.RealtimeModelSettings.reconnect] policy, either close is re-dialed and the
  local message history replayed into a new session.

WebRTC is not supported, as ElevenLabs serves no `/realtime/calls` endpoint, but
[`create_client_secret`][pydantic_ai.realtime.openai.OpenAIRealtimeModel.create_client_secret] mints the
ephemeral credential a browser uses to open the WebSocket directly.

For a hosted ElevenLabs agent, use
[`ElevenLabsAgentsRealtimeModel`][pydantic_ai.realtime.elevenlabs_agents.ElevenLabsAgentsRealtimeModel] instead.

Requires the `websockets` package (the `realtime` optional group) and `openai` (the `openai` group, whose
SDK supplies the event types the shared OpenAI codec is built on):

    pip install "pydantic-ai-slim[elevenlabs-realtime]"
"""

from __future__ import annotations as _annotations

from collections.abc import Awaitable, Callable, Sequence, Set as AbstractSet
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NoReturn, cast

try:
    import websockets as websockets
    from websockets.asyncio.client import ClientConnection

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
from ._lifecycle import LifecycleEvent
from ._openai_lifecycle import OpenAILifecycle
from ._openai_protocol import AUTO_TRANSCRIPTION_MODEL, turn_detection_config, with_realtime_query
from .model import RealtimeModel, WebRTCAnswer
from .openai import OpenAIRealtimeConnection, OpenAIRealtimeModel, OpenAIRealtimeModelSettings
from .profiles import DEFAULT_AUDIO_SAMPLE_RATE, RealtimeModelProfile, RealtimeModelProfileSpec
from .settings import RealtimeModelSettings, ReconnectPolicy

if TYPE_CHECKING:
    from .model import RealtimeProviderSession

__all__ = (
    'ElevenLabsRealtimeModel',
    'ElevenLabsRealtimeModelName',
    'ElevenLabsRealtimeModelSettings',
    'ElevenLabsRealtimeConnection',
    'LatestElevenLabsRealtimeModelNames',
)

_OPENAI_BASE_PATH = 'v1/convai'
"""Path under `ElevenLabsProvider.base_url` that the OpenAI-protocol base URL points at.

The realtime WebSocket is that URL plus `realtime`, with the HTTP scheme swapped for the WebSocket one, and the
client-secret endpoint is that URL plus `realtime/client_secrets`, as on OpenAI:
`https://api.elevenlabs.io` → `https://api.elevenlabs.io/v1/convai` → `wss://api.elevenlabs.io/v1/convai/realtime`.
A data-residency `base_url` maps onto its own realtime host the same way."""

LatestElevenLabsRealtimeModelNames = Literal['eleven_realtime_v1_mini', 'eleven_realtime_v1_max']
"""The latest Eleven Realtime model names."""

ElevenLabsRealtimeModelName = str | LatestElevenLabsRealtimeModelNames
"""Possible Eleven Realtime model names.

`eleven_realtime_v1_mini` is optimized for price and speed; `eleven_realtime_v1_max` reasons before it responds,
for complex instructions and multi-step tool use, and needs a `max_tokens` of at least 1024."""


class ElevenLabsRealtimeModelSettings(OpenAIRealtimeModelSettings, total=False):
    """Settings specific to Eleven Realtime models.

    This inherits every [`OpenAIRealtimeModelSettings`][pydantic_ai.realtime.openai.OpenAIRealtimeModelSettings]
    field, since the session is configured with OpenAI's `session.update` shape. `openai_turn_detection` is sent
    as on OpenAI, except that ElevenLabs only supports semantic VAD: a `ServerVAD` value raises a
    [`UserError`][pydantic_ai.exceptions.UserError].

    The remaining inherited fields are **silently ignored**, because ElevenLabs accepts and then drops them:
    `openai_voice` (ElevenLabs voices are set with `elevenlabs_voice`), `openai_input_noise_reduction`,
    `openai_output_speed`, `openai_truncation`, and `openai_prune_tool_items` and `openai_keep_tool_items` (the
    server doesn't implement `conversation.item.delete`). So are the shared `parallel_tool_calls` and `thinking`
    settings. Input audio is always transcribed: `input_transcription_model` takes `'auto'` (the default) or
    `None`, which still leaves transcripts on, and any other value raises a `UserError`.
    """

    elevenlabs_voice: str
    """Voice used for audio output: an ElevenLabs voice ID (e.g. `'JBFqnCBsd6RMkjVDRZzb'`) or the name of a
    default voice. The server's default is `'brian'`."""

    elevenlabs_language: str
    """Language code of the conversation (e.g. `'en'`, `'es'`, `'pt-br'`), used for transcription, speech, and the
    model's reply language. The server's default is English."""

    elevenlabs_max_duration_seconds: int
    """Maximum wall-clock duration of a session, in seconds, from 60 to 7200. The server's default is 600.

    The server closes the WebSocket once it is exceeded. With a
    [`reconnect`][pydantic_ai.realtime.RealtimeModelSettings.reconnect] policy, the session carries on over a new
    connection, with the local message history replayed into it."""


class _ElevenLabsLifecycle(OpenAILifecycle):
    """The OpenAI lifecycle, except that a spoken turn joins the conversation when its audio is committed.

    OpenAI's VAD commits a turn as soon as it reports `speech_stopped`. ElevenLabs' semantic VAD can instead hear
    the user carry on after a pause, and report `speech_started` again for the same item, so a turn is only over
    once `input_audio_buffer.committed` says so.
    """

    def speech_stopped(self, data: dict[str, Any]) -> list[LifecycleEvent]:
        del data
        return []


class ElevenLabsRealtimeConnection(OpenAIRealtimeConnection):
    """A live WebSocket connection to the ElevenLabs realtime API.

    Reuses [`OpenAIRealtimeConnection`][pydantic_ai.realtime.openai.OpenAIRealtimeConnection] for the shared
    wire protocol, including reconnecting by replaying the local message history.
    """

    _provider_name = 'elevenlabs'
    _provider_label = 'ElevenLabs Realtime'
    # Images are dropped from conversation items, so there is nowhere for a tool result's images to go.
    _supports_tool_result_images = False

    def __init__(
        self,
        ws: ClientConnection,
        *,
        dial: Callable[[], Awaitable[ClientConnection]] | None = None,
        reconnect: ReconnectPolicy | None = None,
        input_transcription_enabled: bool = True,
        interrupts_response_on_speech: bool = False,
        model_name: str | None = None,
        model_name_getter: Callable[[], str | None] | None = None,
        observes_output_audio: bool = True,
        audio_output_sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE,
        prune_tool_items: bool = False,
        keep_tool_items: AbstractSet[str] = frozenset(),
    ) -> None:
        # ElevenLabs transcribes every spoken turn, so its transcripts arrive (and belong in history) even when
        # `input_transcription_model=None` asked for none.
        del input_transcription_enabled
        super().__init__(
            ws,
            dial=dial,
            reconnect=reconnect,
            interrupts_response_on_speech=interrupts_response_on_speech,
            model_name=model_name,
            model_name_getter=model_name_getter,
            observes_output_audio=observes_output_audio,
            audio_output_sample_rate=audio_output_sample_rate,
            prune_tool_items=prune_tool_items,
            keep_tool_items=keep_tool_items,
        )
        self._lifecycle = _ElevenLabsLifecycle()

    @property
    def _prunes_tool_calls(self) -> bool:
        # ElevenLabs answers `conversation.item.delete` with a `not_implemented` error: `openai_prune_tool_items`
        # is one of the settings it ignores.
        return False


class _ElevenLabsOpenAIProvider(OpenAIProvider):
    """The OpenAI SDK pointed at ElevenLabs' OpenAI-compatible API, with an `ElevenLabsProvider`'s key and HTTP client.

    Named `'elevenlabs'` and resolving ElevenLabs' realtime profile, so the behavior
    [`OpenAIRealtimeModel`][pydantic_ai.realtime.openai.OpenAIRealtimeModel] derives from its provider
    (`system`, `profile`, the WebSocket and client-secret URLs) reports ElevenLabs rather than OpenAI.
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


def _turn_detection_config(settings: ElevenLabsRealtimeModelSettings) -> dict[str, Any] | None:
    """The `turn_detection` payload: semantic VAD, the only kind ElevenLabs takes, or `None` for push-to-talk."""
    if 'openai_turn_detection' in settings:
        turn_detection = settings['openai_turn_detection']
        if turn_detection['type'] == 'server_vad':
            raise UserError(
                'Eleven Realtime only supports semantic VAD: set `openai_turn_detection` to '
                "`{'type': 'semantic_vad'}`, or use the cross-provider `turn_detection` setting."
            )
        config = turn_detection_config(turn_detection)
        assert config is not None
        if config.get('eagerness') == 'auto':
            # ElevenLabs takes `low`, `medium` or `high`; leaving it out keeps the server default, `medium`.
            del config['eagerness']
        return config
    base = settings.get('turn_detection', True)
    if base is False:
        return None
    config = {'type': 'semantic_vad'}
    if base is not True:
        # Semantic VAD's `eagerness` takes the same values as the cross-provider `sensitivity`.
        if (sensitivity := base.get('sensitivity')) is not None:
            config['eagerness'] = sensitivity
        for key in ('prefix_padding_ms', 'silence_duration_ms'):
            if (value := base.get(key)) is not None:
                config[key] = value
    return config


@dataclass(init=False)
class ElevenLabsRealtimeModel(OpenAIRealtimeModel):
    """ElevenLabs realtime API model, for the Eleven Realtime speech-to-speech models.

    Pass `provider='elevenlabs'` (the default, which reads `ELEVENLABS_API_KEY`) or an
    [`ElevenLabsProvider`][pydantic_ai.providers.elevenlabs.ElevenLabsProvider] constructed with `api_key=`
    or a data-residency `base_url=`. The session is configured and driven through the OpenAI Realtime
    protocol over a WebSocket; WebRTC is not supported.

    Args:
        model: The model name, `eleven_realtime_v1_mini` (optimized for price and speed) or
            `eleven_realtime_v1_max` (reasons before it responds).
        provider: The provider to use for authentication and the base URL. Defaults to `'elevenlabs'`.
        settings: [Model settings][pydantic_ai.realtime.elevenlabs.ElevenLabsRealtimeModelSettings] used as
            defaults for realtime sessions. With a [`reconnect`][pydantic_ai.realtime.RealtimeModelSettings.reconnect]
            policy, a dropped connection is re-dialed and the local message history replayed into it.
        profile: Optional override for the [realtime model profile][pydantic_ai.realtime.RealtimeModelProfile],
            merged over the provider's — a partial dict, or a callable taking the resolved profile and
            returning the one to use.
    """

    _connection_type: ClassVar[type[OpenAIRealtimeConnection]] = ElevenLabsRealtimeConnection

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
        # The model is chosen by `session.model` in `session.update`: the server ignores a `?model=` query.
        del model_settings
        return with_realtime_query(self._realtime_ws_base())

    async def _auth_headers(self, model_settings: OpenAIRealtimeModelSettings | None = None) -> dict[str, str]:
        # The API key goes in `xi-api-key`; `Authorization: Bearer` only takes a client secret.
        del model_settings
        return {'xi-api-key': self._elevenlabs_provider.api_key}

    def _session_model_name(self, created: dict[str, Any], model_settings: OpenAIRealtimeModelSettings) -> str | None:
        # `session.created` always reports the default model. The requested one is set by the `session.update`
        # that follows, which the server rejects, failing the handshake, if it names an unknown model.
        del created, model_settings
        return self.model

    def _session_config(
        self,
        instructions: str,
        tools: list[ToolDefinition] | None,
        *,
        model_settings: OpenAIRealtimeModelSettings | None,
    ) -> dict[str, Any]:
        settings = cast('ElevenLabsRealtimeModelSettings', self._merge_model_settings(model_settings) or {})
        transcription_model = settings.get('input_transcription_model', AUTO_TRANSCRIPTION_MODEL)
        if transcription_model not in (AUTO_TRANSCRIPTION_MODEL, None):
            raise UserError(
                'Eleven Realtime always transcribes input audio and takes no transcription model: leave '
                "`input_transcription_model` at `'auto'`, and set the language with `elevenlabs_language`; "
                f'got {transcription_model!r}.'
            )
        config = super()._session_config(instructions, tools, model_settings=settings)
        config['model'] = self.model
        audio_input: dict[str, Any] = config['audio']['input']
        audio_input['turn_detection'] = _turn_detection_config(settings)
        # Transcription takes no model, only the conversation's language, which also sets the reply language.
        audio_input.pop('transcription', None)
        audio_input.pop('noise_reduction', None)
        if (language := settings.get('elevenlabs_language')) is not None:
            audio_input['transcription'] = {'language': language}
        audio_output: dict[str, Any] = config['audio']['output']
        # Output audio is always 24 kHz, and its `format` takes no `rate`.
        audio_output['format'] = {'type': 'audio/pcm'}
        # `openai_voice` names an OpenAI voice, which ElevenLabs doesn't have; see `ElevenLabsRealtimeModelSettings`.
        audio_output.pop('voice', None)
        audio_output.pop('speed', None)
        if voice := settings.get('elevenlabs_voice'):
            audio_output['voice'] = voice
        for ignored in ('parallel_tool_calls', 'truncation'):
            config.pop(ignored, None)
        if (max_duration := settings.get('elevenlabs_max_duration_seconds')) is not None:
            config['max_duration_seconds'] = max_duration
        return config

    # WebRTC: ElevenLabs serves OpenAI's client-secret endpoint, which `create_client_secret` inherits, but not
    # `/realtime/calls`.

    def connect_webrtc(
        self,
        session: RealtimeProviderSession,
        *,
        messages: Sequence[ModelMessage],
        model_settings: RealtimeModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> NoReturn:
        self._raise_unsupported_webrtc('connect_webrtc')

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
