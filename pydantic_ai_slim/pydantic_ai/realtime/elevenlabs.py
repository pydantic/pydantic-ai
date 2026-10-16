"""ElevenLabs realtime API provider for speech-to-speech sessions.

ElevenLabs' realtime API is modelled on the OpenAI Realtime protocol, so, like
[`pydantic_ai.realtime.xai`][pydantic_ai.realtime.xai], this provider reuses the OpenAI codec — event
mapping, session seeding, tool conversion, server-VAD config, reconnect-by-replay, and the WebSocket
connection itself — and diverges only where ElevenLabs does.

Requires the `websockets` package (the `realtime` optional group) and `openai` (the `openai` group, whose
SDK supplies the event types the shared OpenAI codec is built on):

    pip install "pydantic-ai-slim[elevenlabs-realtime]"
"""

# TODO(elevenlabs-launch): replace the second paragraph of the docstring above with the list of divergences
# found on launch day, as `xai.py`'s docstring lists xAI's (session shape, transcription, usage, reconnect).

from __future__ import annotations as _annotations

from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, cast

try:
    import websockets as websockets
    from openai.types.realtime import RealtimeResponseUsage
    from pydantic import BaseModel
    from websockets.asyncio.client import ClientConnection
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `websockets` and `openai` packages to use the ElevenLabs realtime model '
        '(`openai` supplies the event types of the OpenAI codec this provider reuses), you can use the '
        '`elevenlabs-realtime` optional group - `pip install "pydantic-ai-slim[elevenlabs-realtime]"`'
    ) from _import_error

from .._http import AsyncHTTPClient
from .._instrumentation import get_instructions
from ..exceptions import UserError
from ..messages import ModelMessage
from ..models import ModelRequestParameters
from ..providers import Provider, infer_provider
from ..providers.elevenlabs import ElevenLabsProvider
from ..tools import ToolDefinition
from ..usage import RequestUsage
from ._openai_protocol import (
    ServerVAD,
    config_interrupts_response_on_speech,
    connect_openai_protocol,
    map_event as _map_openai_event,
    realtime_websocket_url,
    resolve_base_turn_detection,
    resolve_transcription_model,
    tool_choice_config,
    tool_def_to_openai,
    turn_detection_config,
)
from ._utils import inject_trace_context, resolve_advertised_tools
from .codec import RealtimeCodecEvent
from .model import RealtimeModel
from .openai import OpenAIRealtimeConnection
from .profiles import RealtimeModelProfileSpec
from .settings import RealtimeModelSettings

__all__ = (
    'ElevenLabsRealtimeModel',
    'ElevenLabsRealtimeModelSettings',
    'ElevenLabsRealtimeConnection',
    'map_event',
)

# --- Launch-day constants ----------------------------------------------------------------------------------
# TODO(elevenlabs-launch): fill every constant in this block from ElevenLabs' API reference.

_REALTIME_PATH = 'v1/realtime'
"""Path of the realtime WebSocket under the provider's `base_url` (`https://api.elevenlabs.io` → `wss://…`).

With the OpenAI SDK's convention this is its `base_url` (e.g. `https://api.elevenlabs.io/v1/`) plus `realtime`.
TODO(elevenlabs-launch): the real path. If the realtime API lives on a different *host* than the REST API
(e.g. a dedicated `wss://` host), derive the URL from a host constant here instead of `provider.base_url`,
and decide how data-residency `base_url`s map onto it."""

_AUTH_HEADER = 'Authorization'
"""The handshake header carrying the API key.

ElevenLabs says the speech-to-speech API works with the OpenAI SDK, which authenticates with
`Authorization: Bearer <key>`. TODO(elevenlabs-launch): confirm; the rest of the ElevenLabs API uses `xi-api-key`
(then `_AUTH_SCHEME = None`)."""

_AUTH_SCHEME: str | None = 'Bearer'
"""Prefix for the header value, e.g. `'Bearer'` for `Authorization: Bearer <key>`; `None` sends the bare key."""

_MODEL_QUERY_PARAM = True
"""Whether the model goes in the URL as `?model=`, as on OpenAI and xAI.

TODO(elevenlabs-launch): or in `session.update`'s `session.model`, or not at all."""

_AUTO_TRANSCRIPTION_MODEL = 'scribe_v2_realtime'
"""What `input_transcription_model='auto'` resolves to (see `resolve_transcription_model`).

TODO(elevenlabs-launch): the transcription model id the realtime API accepts (Scribe's realtime model is the
likely candidate), or whether transcription is always on and takes no model at all."""

ElevenLabsRealtimeModelName = str
"""Possible ElevenLabs realtime model names.

TODO(elevenlabs-launch): `str | LatestElevenLabsRealtimeModelNames`, with a `Literal` of the launch model ids,
and add them to `KnownRealtimeModelName` in `realtime/model.py` (a sync test pins that list)."""

# ------------------------------------------------------------------------------------------------------------


class _ElevenLabsSession(BaseModel):
    """Fields read from the `session.created` handshake payload."""

    # TODO(elevenlabs-launch): check the payload validates as OpenAI's SDK session type. Kept lenient (like
    # xAI's `_XaiSession`) until a recording shows its real shape.
    model: str | None = None


class _ElevenLabsSessionCreatedEvent(BaseModel):
    type: str
    session: _ElevenLabsSession


class ElevenLabsRealtimeModelSettings(RealtimeModelSettings, total=False):
    """Settings specific to ElevenLabs realtime models."""

    # TODO(elevenlabs-launch): add only settings ElevenLabs documents, prefixed `elevenlabs_` (the Agents model
    # from #7964 uses `elevenlabs_agents_`). Candidates: output format/sample rate, language, TTS knobs
    # (stability, similarity, speed), noise reduction, a semantic VAD.

    elevenlabs_voice: str
    """Voice used for audio output: an ElevenLabs voice ID.

    TODO(elevenlabs-launch): voice ID vs. name, the default voice, and where it goes in `session.update`."""

    elevenlabs_turn_detection: ServerVAD
    """ElevenLabs-specific server-VAD configuration, for an exact threshold, prefix padding, or silence duration.

    When present, this fully overrides the cross-provider `turn_detection` setting.

    TODO(elevenlabs-launch): which `ServerVAD` fields are honored, and whether `create_response`/`interrupt_response`
    are respected (xAI echoes `create_response: False` back but answers anyway)."""


def map_event(data: dict[str, Any]) -> RealtimeCodecEvent | None:
    """Map a raw ElevenLabs realtime event to a [`RealtimeCodecEvent`][pydantic_ai.realtime.codec.RealtimeCodecEvent].

    ElevenLabs models its API on the OpenAI Realtime protocol, so events map via the OpenAI codec.
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
    # server-side (and then check what it restores: xAI's resume dropped user text and function calls), or if
    # a session-length error should re-dial right away as xAI's `max_duration` does (see `xai._decode_frame`).

    # TODO(elevenlabs-launch): push-to-talk. If, like xAI, a commit with `turn_detection: null` is answered
    # without a `response.create`, or a `response.create` after committed audio is dropped, port xAI's
    # `manual_turns` handling (`_send_event` / `_create_response` overrides) and add a simulator dialect rule.

    def _map_event(self, data: dict[str, Any]) -> RealtimeCodecEvent | None:
        return map_event(data)

    def _map_response_usage(self, usage: RealtimeResponseUsage | None) -> RequestUsage | None:
        # TODO(elevenlabs-launch): usage. Check `response.done`'s `usage` against OpenAI's shape: token buckets,
        # any provider-only fields in `model_extra` (xAI adds `grok_tokens` and a *running-total*
        # `billable_audio_seconds`), and what genai-prices needs to price it (audio seconds? characters?).
        # Put provider-only buckets in `details`, and set the unit pricing knows when billing isn't per token,
        # so `cost_limit` doesn't price the session at a confident zero.
        return super()._map_response_usage(usage)


@dataclass(init=False)
class ElevenLabsRealtimeModel(RealtimeModel):
    """ElevenLabs realtime API model.

    Pass `provider='elevenlabs'` (the default, which reads `ELEVENLABS_API_KEY`) or an
    [`ElevenLabsProvider`][pydantic_ai.providers.elevenlabs.ElevenLabsProvider] constructed with `api_key=`
    or a data-residency `base_url=`.

    Args:
        model: The model name.
        provider: The provider to use for authentication and the base URL. Defaults to `'elevenlabs'`.
        settings: [Model settings][pydantic_ai.realtime.RealtimeModelSettings] used as defaults for
            realtime sessions. With a [`reconnect`][pydantic_ai.realtime.RealtimeModelSettings.reconnect]
            policy, a dropped connection is re-dialed and the local message history replayed into it.
        profile: Optional override for the [realtime model profile][pydantic_ai.realtime.RealtimeModelProfile],
            merged over the provider's — a partial dict, or a callable taking the resolved profile and
            returning the one to use.
    """

    # TODO(elevenlabs-launch): the model-name example in `Args` (e.g. "The model name, e.g. `<id>`").
    # TODO(elevenlabs-launch): decide whether to subclass `OpenAIRealtimeModel` instead (as `AzureRealtimeModel`
    # does), backed by an `OpenAIProvider(base_url=..., api_key=..., http_client=...)` built from the shared
    # `ElevenLabsProvider`. That brings WebRTC/client secrets/hang-up for free *if* ElevenLabs serves OpenAI's
    # `/realtime/calls` and `/realtime/client_secrets`. See scratch/LAUNCH_CHECKLIST.md, section 5.

    model: ElevenLabsRealtimeModelName
    _: KW_ONLY
    settings: RealtimeModelSettings | None = None
    _provider: ElevenLabsProvider = field(init=False, repr=False)

    # Written out rather than generated because `profile` has to be an init argument while
    # `RealtimeModel.profile` stays the *resolved* profile, exactly as on a standard `Model`.
    def __init__(
        self,
        model: ElevenLabsRealtimeModelName,
        *,
        provider: Provider[AsyncHTTPClient] | str = 'elevenlabs',
        settings: RealtimeModelSettings | None = None,
        profile: RealtimeModelProfileSpec | None = None,
    ) -> None:
        super().__init__(settings=settings, profile=profile)
        self.model = model
        if isinstance(provider, str):
            provider = infer_provider(provider)
        if not isinstance(provider, ElevenLabsProvider):
            # Reading `api_key` off a foreign provider would fail with an `AttributeError` instead of the real mistake.
            raise UserError(
                "`ElevenLabsRealtimeModel` requires an `ElevenLabsProvider` or `provider='elevenlabs'`; "
                f'got {provider.name!r}.'
            )
        self._provider = provider

    @property
    def model_name(self) -> ElevenLabsRealtimeModelName:
        return self.model

    @property
    def system(self) -> str:
        return 'elevenlabs'

    def _realtime_url(self) -> str:
        return realtime_websocket_url(
            self._provider.base_url, model=self.model if _MODEL_QUERY_PARAM else None, path=_REALTIME_PATH
        )

    def _auth_headers(self) -> dict[str, str]:
        api_key = self._provider.api_key
        return {_AUTH_HEADER: f'{_AUTH_SCHEME} {api_key}' if _AUTH_SCHEME else api_key}

    def _session_config(
        self,
        instructions: str,
        tools: list[ToolDefinition] | None,
        *,
        model_settings: ElevenLabsRealtimeModelSettings | None,
    ) -> dict[str, Any]:
        # TODO(elevenlabs-launch): the `session.update` shape. This is OpenAI's GA shape (`type: 'realtime'`,
        # `voice`/`turn_detection` nested under `audio`); xAI instead puts `voice`/`turn_detection` at the session
        # top level and has no `type`/`output_modalities`. Check each field against their reference, and drop
        # any the server rejects or silently ignores (then document it, as `docs/realtime/xai.md` does).
        model_settings = cast('ElevenLabsRealtimeModelSettings', self._merge_model_settings(model_settings) or {})
        if 'elevenlabs_turn_detection' in model_settings:
            turn_detection = model_settings['elevenlabs_turn_detection']
        elif 'turn_detection' in model_settings:
            turn_detection = resolve_base_turn_detection(model_settings['turn_detection'])
        else:
            turn_detection: ServerVAD | None = {'type': 'server_vad'}
        # `turn_detection` is always set: a dict enables VAD, `None` (explicit null) disables it.
        audio_input: dict[str, Any] = {
            # TODO(elevenlabs-launch): the audio format names (`audio/pcm` + `rate`, or ElevenLabs' `pcm_16000`).
            'format': {'type': 'audio/pcm', 'rate': self.profile.get('audio_input_sample_rate', 24000)},
            'turn_detection': turn_detection_config(turn_detection),
        }
        transcription_model = resolve_transcription_model(
            model_settings.get('input_transcription_model', 'auto'), default=_AUTO_TRANSCRIPTION_MODEL
        )
        if transcription_model is not None:
            audio_input['transcription'] = {'model': transcription_model}
        audio_output: dict[str, Any] = {
            'format': {'type': 'audio/pcm', 'rate': self.profile.get('audio_output_sample_rate', 24000)}
        }
        if voice := model_settings.get('elevenlabs_voice'):
            audio_output['voice'] = voice
        config: dict[str, Any] = {
            'type': 'realtime',
            'instructions': instructions,
            'output_modalities': [model_settings.get('output_modality', 'audio')],
            'audio': {'input': audio_input, 'output': audio_output},
        }
        advertised_tools, tool_choice = resolve_advertised_tools(tools, model_settings.get('tool_choice'))
        if advertised_tools:
            config['tools'] = [tool_def_to_openai(t) for t in advertised_tools]
        # TODO(elevenlabs-launch): `temperature`: GA OpenAI sessions have none, so it isn't forwarded there either.
        if (max_tokens := model_settings.get('max_tokens')) is not None:
            config['max_output_tokens'] = max_tokens
        if (parallel_tool_calls := model_settings.get('parallel_tool_calls')) is not None:
            config['parallel_tool_calls'] = parallel_tool_calls
        if tool_choice is not None:
            config['tool_choice'] = tool_choice_config(tool_choice)
        # TODO(elevenlabs-launch): reasoning. If a model takes `reasoning.effort`, set `supports_thinking` in the
        # profile and map the unified `thinking` setting here, as `XaiRealtimeModel._session_config` does.
        return config

    @asynccontextmanager
    async def connect(
        self,
        *,
        messages: Sequence[ModelMessage],
        model_settings: RealtimeModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncGenerator[ElevenLabsRealtimeConnection]:
        settings = cast('ElevenLabsRealtimeModelSettings', self._merge_model_settings(model_settings) or {})
        handshake_timeout = settings.get('handshake_timeout', 30.0)
        instructions = get_instructions(messages, model_request_parameters) or ''
        session_config = self._session_config(
            instructions=instructions, tools=model_request_parameters.function_tools, model_settings=settings
        )
        transcription_enabled = settings.get('input_transcription_model', 'auto') is not None
        url = self._realtime_url()

        async def dial_headers() -> dict[str, str]:
            headers = self._auth_headers()
            # The raw WebSocket bypasses the provider's HTTP client, so each handshake carries the trace context.
            inject_trace_context(headers)
            return headers

        def session_model(created: dict[str, Any]) -> str | None:
            return _ElevenLabsSessionCreatedEvent.model_validate(created).session.model

        def build_connection(
            ws: ClientConnection,
            dial: Callable[[], Awaitable[ClientConnection]],
            server_model: str | None,
            model_name_getter: Callable[[], str | None],
        ) -> ElevenLabsRealtimeConnection:
            return ElevenLabsRealtimeConnection(
                ws,
                dial=dial,
                reconnect=settings.get('reconnect'),
                input_transcription_enabled=transcription_enabled,
                interrupts_response_on_speech=config_interrupts_response_on_speech(session_config),
                model_name=server_model,
                model_name_getter=model_name_getter,
                audio_output_sample_rate=self.audio_output_sample_rate,
            )

        # TODO(elevenlabs-launch): the handshake. `connect_openai_protocol` waits for `session.created`, sends
        # `session.update`, and waits for `session.updated`; frames in between fail the handshake unless
        # `on_unexpected_during_update` takes them (xAI's `conversation.created` arrives there). Check the order
        # in a recording, and how a bad key or model name is refused (HTTP status on the upgrade vs. an `error`
        # frame vs. a close code) so it surfaces as `ModelHTTPError`/`ModelAPIError`.
        async with connect_openai_protocol(
            model_name=self.model,
            messages=messages,
            profile=self.profile,
            provider_name=self.system,
            session_config=session_config,
            handshake_timeout=handshake_timeout,
            dial_headers=dial_headers,
            dial_url=lambda: url,
            session_model=session_model,
            build_connection=build_connection,
        ) as connection:
            yield connection
