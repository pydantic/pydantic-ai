"""OpenAI Realtime API provider for speech-to-speech sessions.

Connects to `wss://api.openai.com/v1/realtime` over a WebSocket and maps the OpenAI event
protocol to the shared realtime event types.

Requires the `websockets` and `openai` packages, available via the `realtime` and `openai` optional
groups:

    pip install "pydantic-ai-slim[openai-realtime]"
"""

from __future__ import annotations as _annotations

import base64
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast
from urllib.parse import quote

from pydantic import BaseModel, FiniteFloat, TypeAdapter
from pydantic_core import to_json
from typing_extensions import TypeAliasType

try:
    import websockets
    from openai.types.realtime import (
        RealtimeResponseUsage,
        RealtimeSessionCreateRequest,
        SessionCreatedEvent,
    )
    from openai.types.realtime.conversation_item_input_audio_transcription_completed_event import (
        UsageTranscriptTextUsageDuration,
        UsageTranscriptTextUsageTokens,
        UsageTranscriptTextUsageTokensInputTokenDetails,
    )
    from openai.types.realtime.realtime_audio_config_output import VoiceID
    from websockets.asyncio.client import ClientConnection
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `websockets` package to use the OpenAI Realtime model, '
        'you can use the `openai-realtime` optional group - `pip install "pydantic-ai-slim[openai-realtime]"`'
    ) from _import_error

if TYPE_CHECKING:
    # Only needed for typing: the provider supplies the concrete client at runtime, so importing the
    # protocol helpers below (e.g. from the xAI realtime provider) doesn't require the `openai` package.
    from openai import AsyncOpenAI
    from openai.types.realtime.realtime_truncation_param import RealtimeTruncationParam

from .._http import AsyncHTTPClient
from .._instrumentation import get_instructions
from ..exceptions import UserError
from ..messages import (
    BinaryAudio,
    BinaryImage,
    ModelMessage,
    RealtimeOutputSpeechEndEvent,
    RealtimeOutputSpeechStartEvent,
    RealtimeSessionErrorEvent,
    RealtimeSessionReconnectEvent,
)
from ..models import ModelRequestParameters
from ..profiles.openai import OPENAI_REASONING_EFFORT_MAP
from ..providers import Provider, infer_provider
from ..tools import ToolDefinition
from ..usage import RequestUsage
from ._openai_protocol import (
    AUDIO_DELTA_TYPES,
    CONVERSATION_ITEM_CREATE_EVENT,
    CONVERSATION_ITEM_TRUNCATE_EVENT,
    INPUT_AUDIO_BUFFER_APPEND_EVENT,
    INPUT_AUDIO_BUFFER_CLEAR_EVENT,
    INPUT_AUDIO_BUFFER_COMMIT_EVENT,
    INPUT_TRANSCRIPT_DONE_TYPES,
    RESPONSE_CANCEL_EVENT,
    RESPONSE_CREATE_EVENT,
    RESPONSE_CREATED_EVENT_ADAPTER,
    RESPONSE_DONE_EVENT_ADAPTER,
    SESSION_UPDATE_EVENT,
    SESSION_UPDATED_EVENT,
    ProtocolResponseDoneEvent,
    RealtimeHandshakeError,
    SemanticVAD,
    ServerVAD,
    connect_openai_protocol,
    expect_event,
    loads_obj,
    map_connect_errors,
    map_event,
    realtime_websocket_url,
    resolve_base_turn_detection,
    resolve_transcription_model,
    response_finish_reason,
    seed_items,
    tool_choice_config,
    tool_def_to_openai,
    turn_detection_config,
    user_message_item,
    with_realtime_query,
)
from ._openai_webrtc import answer_webrtc_offer as _answer_webrtc_offer, mint_client_secret as _mint_client_secret
from ._utils import inject_trace_context, reconnect_with_backoff, require_pcm_audio, resolve_advertised_tools
from .codec import (
    AudioDelta,
    CancelResponse,
    ClearAudio,
    CommitAudio,
    CreateResponse,
    InputTranscript,
    RealtimeCodecEvent,
    RealtimeConnection,
    RealtimeInput,
    SessionUsage,
    ToolResult,
    TruncateOutput,
)
from .model import RealtimeClientSecret, RealtimeModel, RealtimeProviderSession, WebRTCAnswer
from .profiles import RealtimeModelProfileSpec
from .settings import RealtimeModelSettings, ReconnectPolicy

# `input_transcription_model='auto'` resolves to this — OpenAI's recommended realtime transcription model
# ("For the lowest-latency streaming transcription path, use gpt-realtime-whisper"; it's natively streaming
# and designed for realtime sessions, unlike the legacy `whisper-1`). Kept behind the `'auto'` sentinel
# (see `resolve_transcription_model`) so it can be bumped without changing the behavior of apps on `'auto'`.
_AUTO_TRANSCRIPTION_MODEL = 'gpt-realtime-whisper'

_OUTPUT_AUDIO_BUFFER_CLEAR_EVENT = 'output_audio_buffer.clear'
_OUTPUT_SPEECH_START_FRAME = 'output_audio_buffer.started'
_OUTPUT_SPEECH_END_FRAMES = frozenset({'output_audio_buffer.stopped', 'output_audio_buffer.cleared'})


class _SidebandContentPart(BaseModel):
    type: str


class _SidebandContentPartAdded(BaseModel):
    item_id: str
    content_index: int = 0
    part: _SidebandContentPart


class _SidebandSession(BaseModel):
    model: str | None = None


class _SidebandSessionUpdated(BaseModel):
    session: _SidebandSession


LatestOpenAIRealtimeModelNames = Literal['gpt-realtime', 'gpt-realtime-2.1', 'gpt-realtime-2.1-mini']
OpenAIRealtimeModelName = str | LatestOpenAIRealtimeModelNames

LatestOpenAIRealtimeTranscriptionModelNames = Literal[
    'whisper-1',
    'gpt-4o-transcribe',
    'gpt-4o-mini-transcribe',
    'gpt-realtime-whisper',
]
OpenAIRealtimeTranscriptionModelName = str | LatestOpenAIRealtimeTranscriptionModelNames

__all__ = (
    'OpenAIRealtimeModel',
    'OpenAIRealtimeModelSettings',
    'OpenAIRealtimeConnection',
    'KnownOpenAIRealtimeVoiceName',
    'ServerVAD',
    'SemanticVAD',
    'map_event',
)

KnownOpenAIRealtimeVoiceName = TypeAliasType(
    'KnownOpenAIRealtimeVoiceName',
    Literal[
        'alloy',
        'ash',
        'ballad',
        'cedar',
        'coral',
        'echo',
        'marin',
        'sage',
        'shimmer',
        'verse',
    ],
)
"""The prebuilt voices OpenAI's realtime API ships, mirroring the `openai` SDK's own `Voice` union.

The [`openai_voice`][pydantic_ai.realtime.openai.OpenAIRealtimeModelSettings.openai_voice] setting also
accepts any other string, so a voice OpenAI adds later works before this list catches up; a test pins
the list against the SDK so it doesn't silently fall behind.
"""


class OpenAIRealtimeModelSettings(RealtimeModelSettings, total=False):
    """Settings specific to OpenAI realtime models."""

    openai_voice: KnownOpenAIRealtimeVoiceName | str | VoiceID
    """Voice used for audio output, e.g. `alloy` or `VoiceID(id='voice_1234')`.

    The known prebuilt names provide autocomplete, while any string and the OpenAI SDK's custom
    `VoiceID` form (`openai.types.realtime.realtime_audio_config_output.VoiceID`) are also accepted.
    """

    openai_input_noise_reduction: Literal['near_field', 'far_field']
    """Noise reduction tuned for `near_field` (headset) or `far_field` (laptop/conference) microphones.

    Absent disables it.
    """
    openai_output_speed: float
    """Playback speed multiplier for generated audio (0.25-1.5)."""
    openai_turn_detection: ServerVAD | SemanticVAD
    """OpenAI-specific server or semantic VAD configuration.

    When present, this fully overrides the cross-provider `turn_detection` setting.
    """
    openai_truncation: RealtimeTruncationParam
    """How the session truncates conversation context once it exceeds the model's window.

    `'auto'` (the server default) drops the oldest turns; `'disabled'` keeps everything (and errors
    when the window is full); a `retention_ratio` truncation (`{'type': 'retention_ratio',
    'retention_ratio': 0.8}`) keeps a fixed fraction, holding the prompt-cached prefix stable across
    turns (cached audio is far cheaper). This is the OpenAI SDK's `truncation` shape, forwarded as-is.
    """


def _map_usage(usage: RealtimeResponseUsage | None) -> RequestUsage | None:
    """Map a `response.done` `usage` payload to a [`RequestUsage`][pydantic_ai.usage.RequestUsage].

    Mapped by hand rather than through [`RequestUsage.extract`][pydantic_ai.usage.RequestUsage.extract],
    which the standard OpenAI adapter uses: the realtime API reports per-modality buckets the Responses
    usage schema has no place for, so extraction recognizes only the totals and the typed fields would
    have to be re-set from the wire afterwards anyway. The typed fields below are nonetheless the same
    ones the standard adapter produces for the same concepts, pinned by
    `test_map_usage_matches_standard_openai_normalization`.
    """
    if usage is None or not usage.model_fields_set:
        return None
    usage = RealtimeResponseUsage.model_validate(usage.model_dump(warnings=False))
    inp = usage.input_token_details or None
    out = usage.output_token_details or None
    cached = inp.cached_tokens_details if inp is not None else None
    # `reasoning_tokens` is on the wire but isn't a field of the SDK model, so it arrives as an extra.
    # The standard adapter names the same concept `reasoning_tokens` in `details` and also sets the
    # typed `output_reasoning_tokens`; realtime set neither, so a reasoning turn reported none at all.
    reasoning_tokens = (out.model_extra or {}).get('reasoning_tokens') if out is not None else None
    details: dict[str, int] = {}
    for key, raw in (
        ('input_text_tokens', inp.text_tokens if inp is not None else None),
        ('input_image_tokens', inp.image_tokens if inp is not None else None),
        ('output_text_tokens', out.text_tokens if out is not None else None),
        # `audio_tokens` and `reasoning_tokens` are the names the standard adapter gives these same
        # two output buckets (it flattens `completion_tokens_details` into `details` wholesale), so
        # they carry over unprefixed; the input-side buckets, which the standard adapter doesn't
        # flatten, keep the `input_` prefix that tells them apart from their output counterparts.
        ('audio_tokens', out.audio_tokens if out is not None else None),
        ('reasoning_tokens', reasoning_tokens),
    ):
        if isinstance(raw, int) and not isinstance(raw, bool):
            details[key] = raw
    return RequestUsage(
        input_tokens=usage.input_tokens or 0,
        output_tokens=usage.output_tokens or 0,
        input_audio_tokens=inp.audio_tokens or 0 if inp is not None else 0,
        cache_read_tokens=inp.cached_tokens or 0 if inp is not None else 0,
        cache_audio_read_tokens=cached.audio_tokens or 0 if cached is not None else 0,
        output_audio_tokens=out.audio_tokens or 0 if out is not None else 0,
        # Left unset — not zeroed — when the provider doesn't report it, so a model that doesn't reason
        # is distinguishable from one that reasoned for free, exactly as `RequestUsage.extract` leaves it.
        **({'output_reasoning_tokens': details['reasoning_tokens']} if 'reasoning_tokens' in details else {}),
        details=details,
    )


RealtimeTranscriptionUsage = UsageTranscriptTextUsageTokens | UsageTranscriptTextUsageDuration


class _RealtimeTranscriptionTokenUsage(UsageTranscriptTextUsageTokens):
    """Token usage shape accepted by Azure/xAI, which omit OpenAI's required token fields."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    type: Literal['tokens'] = 'tokens'


RealtimeTranscriptionWireUsage = _RealtimeTranscriptionTokenUsage | UsageTranscriptTextUsageDuration
_TRANSCRIPTION_USAGE_ADAPTER: TypeAdapter[RealtimeTranscriptionWireUsage] = TypeAdapter(RealtimeTranscriptionWireUsage)
_FINITE_FLOAT_ADAPTER: TypeAdapter[float] = TypeAdapter(FiniteFloat)


def _validate_transcription_usage(usage: object) -> RealtimeTranscriptionUsage | None:
    if usage is None or usage == {}:
        return None
    return _TRANSCRIPTION_USAGE_ADAPTER.validate_python(usage)


def _map_transcription_usage(usage: RealtimeTranscriptionUsage | None) -> RequestUsage | None:
    """Map input-transcription usage into separate [`RequestUsage.details`][pydantic_ai.usage.RequestUsage.details]."""
    if usage is None or not usage.model_fields_set:
        return None
    details: dict[str, int] = {}
    if isinstance(usage, UsageTranscriptTextUsageTokens):
        # Branch on the variant, not `usage.type`: a protocol clone (xAI/Azure) may omit the `type`
        # discriminator, which `_validate_usage_shape` deliberately tolerates, and the SDK's lenient
        # `.construct` then builds this tokens variant with `type=None`. A type-less payload must be read
        # as tokens rather than reaching the duration branch, whose `usage.seconds` this variant lacks —
        # previously an `AttributeError` that escaped the recoverable path and tore the session down.
        token_details = usage.input_token_details or None
        if token_details is not None and not isinstance(token_details, UsageTranscriptTextUsageTokensInputTokenDetails):
            raise ValueError('`usage.input_token_details` must be an object')
        for key, raw in (
            ('input_transcription_tokens', usage.total_tokens),
            ('input_transcription_audio_tokens', token_details.audio_tokens if token_details is not None else None),
            ('input_transcription_text_tokens', token_details.text_tokens if token_details is not None else None),
        ):
            if isinstance(raw, int) and not isinstance(raw, bool):
                details[key] = raw
    elif (seconds := _FINITE_FLOAT_ADAPTER.validate_python(usage.seconds)) > 0:
        # `RunUsage.details` values are ints, so a fractional duration has to round. Sub-half-second
        # clips round to zero, which would drop billed transcription entirely — report the floor of
        # one second instead, so a short utterance is visible rather than free.
        details['input_transcription_seconds'] = max(1, round(seconds))
    return RequestUsage(details=details) if details else None


def _frame_error(error: ValueError) -> RealtimeSessionErrorEvent:
    """Report a malformed frame as a recoverable session error, rather than tearing the session down."""
    return RealtimeSessionErrorEvent(message=f'Failed to parse OpenAI realtime event: {error}', recoverable=True)


def _describe_close(ws: ClientConnection) -> str:
    """Describe a close that `websockets` reported by ending iteration instead of raising."""
    if (code := ws.close_code) is None:  # pragma: no cover
        # Only reachable if iteration ends while the connection is still open, which `websockets`
        # doesn't do; described generically rather than asserted so it can't mask a real close.
        return 'stream ended'
    reason = ws.close_reason
    return f'received {code} {reason}' if reason else f'received {code}'


class OpenAIRealtimeConnection(RealtimeConnection):
    """A live WebSocket connection to the OpenAI Realtime API."""

    _provider_name = 'openai'
    # How this provider names itself in error messages; protocol clones (xAI, Azure) override it so a
    # closed connection doesn't report the wrong vendor.
    _provider_label = 'OpenAI Realtime'
    _supports_tool_result_images = True
    # `OSError` covers the socket-level failures (reset, broken pipe) that `websockets` lets through.
    transport_errors = (websockets.WebSocketException, OSError)

    def __init__(
        self,
        ws: ClientConnection,
        *,
        dial: Callable[[], Awaitable[ClientConnection]] | None = None,
        reconnect: ReconnectPolicy | None = None,
        input_transcription_enabled: bool = True,
        model_name: str | None = None,
        model_name_getter: Callable[[], str | None] | None = None,
        observes_output_audio: bool = True,
    ) -> None:
        self._ws = ws
        self._model_name = model_name
        self._model_name_getter = model_name_getter
        # `dial` re-establishes a fully configured connection; with a `reconnect` policy it is used to
        # recover from a dropped WebSocket.
        self._dial = dial
        self._reconnect = reconnect
        self._restores_state_on_reconnect = False
        # Set by the session (see `set_message_history`) so a re-dial can replay the call. The API keeps no
        # state across sessions, so without it a reconnect resumes knowing nothing that was said.
        self._message_history: Callable[[], Sequence[ModelMessage]] | None = None
        self._input_transcription_enabled = input_transcription_enabled
        self._reconnects_used = 0
        self._observes_output_audio = observes_output_audio
        # The Realtime API rejects `response.create` while a response is already being generated.
        # We track that window and defer requests (e.g. a background tool result that lands while the
        # model is mid-answer) until the active response finishes, so the model still announces it.
        self._response_active = False
        self._active_response_id: str | None = None
        # Whether the active response has been confirmed by its `response.created`. A response solicited
        # (`response.create` sent) but not yet started is the one a mid-handshake drop would otherwise
        # lose: `_pending_response` only covers responses deferred behind an *active* one, so without this
        # a re-dial replays the finalized call but never re-asks for the answer the caller is waiting on.
        self._response_started = False
        self._pending_response = False
        self._cancel_sent = False
        # Id of a response we cancelled (barge-in): the server keeps streaming a few straggler deltas
        # before its `response.done`, and mapping them would surface speech the user already interrupted.
        # Drop frames carrying this id until its `response.done` closes it. `None` when nothing is being
        # cancelled, or when the cancelled response had no id (defensive/compat path — then not suppressed).
        self._cancelled_response_id: str | None = None
        # The current output audio item, tracked from output-audio deltas so a `TruncateOutput` can
        # name it. These are mutated by `__aiter__` and read by `send` from a separate task; under a
        # single cooperative event loop the plain reads/writes are safe and eventually consistent.
        self._current_item_id: str | None = None
        self._current_content_index = 0
        self._generated_audio_bytes = 0
        self._output_audio_playing = False
        self._output_speech_clear_sent = False

    @property
    def model_name(self) -> str | None:
        return self._model_name_getter() if self._model_name_getter is not None else self._model_name

    def set_message_history(self, message_history: Callable[[], Sequence[ModelMessage]]) -> None:
        self._message_history = message_history
        # A reconnect will now replay the call, so it restores state rather than starting blank.
        self._restores_state_on_reconnect = True

    def _map_response_usage(self, usage: RealtimeResponseUsage | None) -> RequestUsage | None:
        """Map a response's usage payload; a protocol clone overrides this to add buckets OpenAI lacks."""
        return _map_usage(usage)

    @property
    def message_history(self) -> Callable[[], Sequence[ModelMessage]] | None:
        """The call so far, when a session has offered it for replay on reconnect."""
        return self._message_history

    @property
    def reconnect_restores_in_flight_state(self) -> bool:
        # Local replay restores only the finalized turns; the response and tool calls in flight when
        # the socket dropped are gone, so the session settles them. (The xAI clone resumes natively and
        # overrides this back to `True`.)
        return False

    @property
    def input_transcription_enabled(self) -> bool:
        return self._input_transcription_enabled

    async def send(self, content: RealtimeInput) -> None:
        """Send content to the OpenAI Realtime API.

        Accepts `BinaryAudio` (raw PCM16, 24kHz, mono), a `str` text turn, `BinaryImage`,
        `ToolResult`, and the control verbs `CommitAudio`, `ClearAudio`, `CreateResponse`,
        `CancelResponse`, and `TruncateOutput`.
        """
        if isinstance(content, BinaryAudio):
            require_pcm_audio(content, provider_name=self._provider_label)
            await self._send_event(
                {
                    'type': INPUT_AUDIO_BUFFER_APPEND_EVENT,
                    'audio': base64.b64encode(content.data).decode('ascii'),
                }
            )
        elif isinstance(content, str):
            await self._send_event(
                {
                    'type': CONVERSATION_ITEM_CREATE_EVENT,
                    'item': {
                        'type': 'message',
                        'role': 'user',
                        'content': [{'type': 'input_text', 'text': content}],
                    },
                }
            )
            await self._request_response()
        elif isinstance(content, ToolResult):
            # Normalize any follow-up content (downloading and re-encoding media) before the first
            # frame goes out, so content this provider can't carry fails with nothing sent rather than
            # leaving the result on the wire without the material that explains it.
            item = (
                await user_message_item(
                    content.content,
                    provider_name=self._provider_name,
                    supports_images=self._supports_tool_result_images,
                )
                if content.content
                else None
            )
            await self._send_event(
                {
                    'type': CONVERSATION_ITEM_CREATE_EVENT,
                    'item': {
                        'type': 'function_call_output',
                        'call_id': content.tool_call_id,
                        'output': content.output,
                    },
                }
            )
            if item:
                await self._send_event({'type': CONVERSATION_ITEM_CREATE_EVENT, 'item': item})
            await self._request_response()
        elif isinstance(content, BinaryImage):
            # An image is added as conversation context (like a video frame), not a turn of its own,
            # so it doesn't trigger a response — drive that with audio (VAD) or `CreateResponse`.
            data_uri = content.data_uri
            await self._send_event(
                {
                    'type': CONVERSATION_ITEM_CREATE_EVENT,
                    'item': {
                        'type': 'message',
                        'role': 'user',
                        'content': [{'type': 'input_image', 'image_url': data_uri}],
                    },
                }
            )
        elif isinstance(content, CommitAudio):
            await self._send_event({'type': INPUT_AUDIO_BUFFER_COMMIT_EVENT})
        elif isinstance(content, ClearAudio):
            await self._send_event({'type': INPUT_AUDIO_BUFFER_CLEAR_EVENT})
        elif isinstance(content, CreateResponse):
            await self._request_response()
        elif isinstance(content, CancelResponse):
            # Only cancel when a response is actually active: with server VAD the provider may have
            # already cancelled on the user's barge-in, and a redundant cancel raises a session error.
            if self._response_active and not self._cancel_sent:
                await self._send_event({'type': RESPONSE_CANCEL_EVENT})
                self._cancel_sent = True
                # Suppress the cancelled response's trailing deltas until its `response.done` arrives.
                self._cancelled_response_id = self._active_response_id
                # The cancelled response's output item is gone; forget it so a following
                # `interrupt(played_ms=...)` before the next turn's first audio doesn't truncate a stale
                # item (server-initiated cancels clear it via `response.done`, but a client cancel doesn't).
                self._current_item_id = None
                self._generated_audio_bytes = 0
            if not self._observes_output_audio and self._output_audio_playing and not self._output_speech_clear_sent:
                await self._send_event({'type': _OUTPUT_AUDIO_BUFFER_CLEAR_EVENT})
                self._output_speech_clear_sent = True
        elif isinstance(content, TruncateOutput):
            # No current output item (e.g. the model wasn't speaking) → nothing to truncate.
            if self._current_item_id is not None:
                audio_end_ms = content.audio_end_ms
                # A WebRTC sideband connection does not receive output-audio deltas because media flows
                # directly between the browser and provider. Only clamp connections that observe those
                # deltas; otherwise the byte counter stays zero and every barge-in would truncate to zero.
                if self._observes_output_audio:
                    max_audio_end_ms = self._generated_audio_bytes * 1000 // 48_000
                    audio_end_ms = min(audio_end_ms, max_audio_end_ms)
                await self._send_event(
                    {
                        'type': CONVERSATION_ITEM_TRUNCATE_EVENT,
                        'item_id': self._current_item_id,
                        'content_index': self._current_content_index,
                        'audio_end_ms': audio_end_ms,
                    }
                )
        else:
            raise UserError(f'{self._provider_label} does not support {type(content).__name__} input.')

    async def _request_response(self) -> None:
        """Ask the model to respond now, or defer until the active response completes."""
        if self._response_active:
            self._pending_response = True
        else:
            self._response_active = True
            self._active_response_id = None
            await self._send_event({'type': RESPONSE_CREATE_EVENT})

    async def _send_event(self, event: dict[str, Any]) -> None:
        await self._ws.send(to_json(event).decode())

    def _map_event(self, data: dict[str, Any]) -> RealtimeCodecEvent | None:
        """Map a raw provider frame to a codec event.

        A hook so protocol clones (e.g. the xAI Grok Voice provider) can reuse the whole connection while
        overriding only how frames map to events. Defaults to the OpenAI [`map_event`][pydantic_ai.realtime.openai.map_event].
        """
        return map_event(data)

    async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
        while True:
            try:
                async for raw in self._ws:
                    if not isinstance(raw, str):
                        continue
                    try:
                        events = await self._decode_frame(raw)
                    except ValueError as e:
                        # A malformed frame (bad JSON or audio payload) shouldn't tear down the whole
                        # session; surface it as a recoverable error and keep reading.
                        yield _frame_error(e)
                        continue
                    for event in events:
                        yield event
                # `websockets` ends iteration silently on a *normal* close (1000/1001) and only raises
                # on an abnormal one, but a session the server hung up on is over either way: OpenAI
                # ends one that reaches its duration cap with `1001 Your session hit the maximum
                # duration of 60 minutes.`. Handle it like a drop so a reconnect policy still runs and,
                # without one, the consumer learns the conversation was cut off instead of seeing the
                # stream quietly end.
                if not self._observes_output_audio:
                    return
                closed = _describe_close(self._ws)
            except self.transport_errors as e:
                # `ConnectionClosed`, any other protocol error, or a socket-level `OSError` (reset,
                # broken pipe): all mean the link failed, so they all take the same reconnect path
                # instead of escaping the stream and bypassing the reconnect policy.
                closed = str(e)

            if self._reconnect is None or self._dial is None:
                # No reconnect policy: a closed connection is fatal. Surface it as a non-recoverable
                # error and end the stream cleanly, rather than raising.
                yield RealtimeSessionErrorEvent(
                    message=f'{self._provider_label} connection closed: {closed}', recoverable=False
                )
                return
            if await self._try_reconnect():
                yield RealtimeSessionReconnectEvent(state_restored=self._restores_state_on_reconnect)
                continue
            yield RealtimeSessionErrorEvent(
                message=f'{self._provider_label} connection closed; reconnect failed: {closed}', recoverable=False
            )
            return

    def _is_cancelled_straggler(self, event_type: str | None, data: dict[str, Any]) -> bool:
        """Whether this frame is a trailing delta from a response cancelled on barge-in (drop it).

        Its own `response.done` is excluded so the response still closes cleanly; frames for any other
        response carry a different `response_id`.
        """
        return (
            self._cancelled_response_id is not None
            and event_type != 'response.done'
            and data.get('response_id') == self._cancelled_response_id
        )

    def _close_cancelled_response(self, response_id: str | None) -> None:
        """Stop suppressing stragglers once a cancelled response's `response.done` arrives."""
        if response_id == self._cancelled_response_id:
            self._cancelled_response_id = None

    def _track_sideband_audio_part(self, data: dict[str, Any]) -> None:
        added = _SidebandContentPartAdded.model_validate(data)
        if added.part.type == 'audio':
            self._current_item_id = added.item_id
            self._current_content_index = added.content_index

    async def _decode_frame(self, raw: str) -> list[RealtimeCodecEvent]:  # noqa: C901
        """Parse one text frame into events, updating tracked response state.

        Raises `ValueError` (incl. `pydantic.ValidationError` / `binascii.Error`) on a malformed payload.
        """
        data = loads_obj(raw)
        event_type = data.get('type')
        if event_type == _OUTPUT_SPEECH_START_FRAME:
            if self._is_cancelled_straggler(event_type, data):
                # A start boundary for a response cancelled on barge-in: don't mark playback active or
                # report the model as speaking. Its matching stop/cleared boundary is still processed
                # below (it is never a straggler) so any playback state is cleaned up.
                return []
            self._output_audio_playing = True
            return [] if self._observes_output_audio else [RealtimeOutputSpeechStartEvent()]
        if event_type in _OUTPUT_SPEECH_END_FRAMES:
            was_playing, self._output_audio_playing = self._output_audio_playing, False
            self._output_speech_clear_sent = False
            if not self._response_active:
                # The response already closed; playback ending retires its output item (kept alive
                # past `response.done` by `_clear_active_response` for barge-in truncation).
                self._current_item_id = None
            return [] if self._observes_output_audio or not was_playing else [RealtimeOutputSpeechEndEvent()]
        # Drop trailing frames from a response we cancelled on barge-in (its audio/transcript deltas,
        # output-item events, etc.); its own `response.done` still passes through below to close the
        # response, emit usage, and clear the suppression.
        if self._is_cancelled_straggler(event_type, data):
            return []
        events: list[RealtimeCodecEvent] = []
        superseded = False
        if event_type == 'response.done':
            # Settled *before* the frame is mapped: mapping a malformed `response.done` raises
            # `ValueError`, which `__aiter__` surfaces as a recoverable frame error and keeps reading —
            # but the frame was still the only terminal its response will ever get, and bailing before
            # the state updates would leave `_response_active` held forever, queueing every later
            # `response.create` behind a response that already ended.
            done_events, superseded = await self._handle_response_done(data)
            events.extend(done_events)
        event = self._map_event(data)
        if event_type == 'response.created':
            created = RESPONSE_CREATED_EVENT_ADAPTER.validate_python(data)
            self._response_active = True
            self._response_started = True
            self._active_response_id = created.response.id or None
            if self._cancel_sent and self._cancelled_response_id is None:
                # A cancel raced ahead of this `response.created`: it was sent while the response the
                # client asked for had no server-assigned id yet, so the suppression id could not be
                # recorded. This frame names that response — backfill so its trailing deltas are
                # dropped instead of playing on after the barge-in.
                self._cancelled_response_id = self._active_response_id
        elif event_type == 'response.content_part.added' and not self._observes_output_audio:
            self._track_sideband_audio_part(data)
        elif event_type in AUDIO_DELTA_TYPES:
            # Track the speaking item so a later `TruncateOutput` can name it.
            if isinstance(event, AudioDelta) and event.item_id:
                content_index = TypeAdapter(int).validate_python(data.get('content_index', 0))
                item_changed = (event.item_id, content_index) != (
                    self._current_item_id,
                    self._current_content_index,
                )
                self._current_item_id = event.item_id
                self._current_content_index = content_index
                if item_changed:
                    self._generated_audio_bytes = len(event.data)
                else:
                    self._generated_audio_bytes += len(event.data)
        if event is not None and not (event_type == 'response.done' and superseded):
            events.append(event)
            if isinstance(event, InputTranscript) and event.is_final and event_type in INPUT_TRANSCRIPT_DONE_TYPES:
                # The transcript is already recorded, so a malformed `usage` payload costs the usage
                # event, not the user's words: report it as the same recoverable frame error `__aiter__`
                # would have raised rather than discarding the whole frame along with it.
                try:
                    usage = _validate_transcription_usage(data.get('usage'))
                except ValueError as e:
                    events.append(_frame_error(e))
                else:
                    if (asr := _map_transcription_usage(usage)) is not None:
                        events.append(SessionUsage(usage=asr, response_scoped=False))
        return events

    async def _handle_response_done(self, data: dict[str, Any]) -> tuple[list[RealtimeCodecEvent], bool]:
        """Update response state and emit usage for a `response.done`.

        Returns `(events, superseded)`. `superseded` is `True` when a *different* response is still
        active — a late/cancelled completion arriving after a new turn began — so the caller suppresses
        its `ResponseDone` (which would otherwise finalize the current response's output under this old
        boundary).
        """
        events: list[RealtimeCodecEvent] = []
        try:
            done = RESPONSE_DONE_EVENT_ADAPTER.validate_python(data)
        except ValueError:
            # A `response` payload of the wrong shape carries no id to reason about, so it is handled
            # exactly like a missing one below. The error itself is not swallowed: mapping the same
            # frame raises it again, and `__aiter__` reports it as a recoverable frame error.
            # A `response.done` with no usable `response` object is malformed, but it is still the only
            # terminal we will ever get for the response it was meant to close. Treating it as "no
            # information" leaves `_response_active` set forever, and every later `create_response()`
            # then queues behind a response that can never complete.
            self._clear_active_response()
            if self._pending_response:
                self._pending_response = False
                await self._request_response()
            return events, False
        response = done.response
        function_call_only = bool(response.output) and all(item.type == 'function_call' for item in response.output)
        finish_reason = (
            'tool_call' if response.status == 'completed' and function_call_only else response_finish_reason(response)
        )
        response_id = response.id
        # The cancelled response is now closed; stop suppressing its stragglers (its own usage still emits
        # below). A no-op for any other response.
        was_cancelled_response = isinstance(response_id, str) and response_id == self._cancelled_response_id
        self._close_cancelled_response(response_id)
        # OpenAI response events always carry an ID. Keep the ID-less fallback for compatible protocol
        # implementations and defensive unit inputs that predate response tracking. An *unknown* active
        # id matches too — the window between `response.create` and its `response.created`, which a
        # protocol clone that omits the id never leaves — since a done that can't be proven stale must
        # still settle the response, exactly as `superseded` below refuses to prove staleness from it.
        matches_active_response = not isinstance(response_id, str) or (
            self._response_active and self._active_response_id in (None, response_id)
        )
        # Superseded only when a *different, known* response is active — a late/cancelled completion after
        # a new turn began. When the active id is unknown (`None`, e.g. an id-less `response.created`), this
        # done can't be proven stale, so don't suppress its turn-completion.
        superseded = (
            isinstance(response_id, str)
            and self._active_response_id is not None
            and response_id != self._active_response_id
        )
        was_client_cancel = matches_active_response and self._cancel_sent
        if matches_active_response:
            self._clear_active_response()
        elif was_cancelled_response:
            # The cancel we sent has settled — its response just closed — but a newer response is
            # already active (`response.created` beat this late `response.done`), so
            # `_clear_active_response` must not run. Reset the flag directly: leaving it set would make
            # the `not self._cancel_sent` guard in `send` silently swallow every later `CancelResponse`,
            # leaving the new response uninterruptible.
            self._cancel_sent = False
        if matches_active_response and self._pending_response:
            self._pending_response = False
            # A *server*-cancelled response means the user barged in: a new turn is starting, so don't
            # replay the deferred response over it. After a *client* cancel (`interrupt()`), the caller
            # explicitly queued the next response behind the cancel, so send it now that the cancelled
            # response has closed.
            if response.status != 'cancelled' or was_client_cancel:
                self._response_active = True
                self._active_response_id = None
                await self._send_event({'type': RESPONSE_CREATE_EVENT})
        # Validated only now that all the response state above is settled: a malformed usage payload
        # raises `ValueError`, which `__aiter__` surfaces as a recoverable frame error and keeps reading
        # — but this `response.done` was still the terminal for its response, and bailing before the
        # state updates would leave `_response_active` held forever, queueing every later
        # `response.create` behind a response that already ended.
        # Emit usage for every response (including intermediate function-call-only ones) so the session
        # accounts for all tokens. Only the active response may replay a pending request; a late completion
        # for a superseded response must not change current state. OpenAI nests usage under
        # `response.usage`; xAI Grok Voice reports the same shape at the top level of the `response.done`
        # frame (its `response.usage` is empty), so fall back to it.
        frame_usage = done.usage if isinstance(done, ProtocolResponseDoneEvent) else None
        usage = self._map_response_usage(response.usage) or self._map_response_usage(frame_usage)
        if usage is not None:
            events.append(
                SessionUsage(
                    usage=usage,
                    provider_response_id=response_id or None,
                    finish_reason=finish_reason,
                )
            )
        elif matches_active_response and finish_reason == 'tool_call':
            events.append(
                SessionUsage(
                    usage=RequestUsage(),
                    provider_response_id=response_id or None,
                    finish_reason='tool_call',
                )
            )
        return events, superseded

    async def _try_reconnect(self) -> bool:
        """Re-dial with exponential backoff; return whether a new connection was established."""
        assert self._reconnect is not None and self._dial is not None
        if not await reconnect_with_backoff(
            self._reconnect, self._attempt_reconnect, reconnects_used=self._reconnects_used
        ):
            return False
        self._reconnects_used += 1
        return True

    async def _attempt_reconnect(self) -> bool:
        assert self._dial is not None
        try:
            self._ws = await self._dial()
            # A response the socket dropped before it could complete will never be released by its
            # `response.done`, so re-ask for it on the fresh socket (which has just replayed the call)
            # rather than leaving the session waiting for a turn that can never start. Two shapes qualify:
            # one deferred behind a now-dead active response (`_pending_response`), and one solicited but
            # not yet confirmed by its `response.created` (`_response_active` without `_response_started`)
            # — the answer the caller is waiting on, which `_pending_response` alone does not cover. That
            # second case is only ours to re-ask on a local-replay connection: one that restores in-flight
            # state (`reconnect_restores_in_flight_state`, e.g. the inherited xAI clone) has the server
            # resume that response itself, so re-asking would duplicate it. A response already streaming
            # when it dropped is *not* re-asked either: its partial reply is settled as interrupted,
            # matching the state-lost contract of staying quiet until the next input. Nor is one the
            # caller cancelled (`_cancel_sent`) before it started: re-asking would resurrect a response a
            # barge-in explicitly stopped. Sent inside this guard because the new socket can drop before
            # the frame reaches it, which has to consume an attempt like any other failed reconnect.
            replay_response = self._pending_response or (
                not self.reconnect_restores_in_flight_state
                and self._response_active
                and not self._response_started
                and not self._cancel_sent
            )
            self._clear_active_response()
            # A fresh socket also drops anything the old one was still holding for us.
            self._cancelled_response_id = None
            if replay_response:
                await self._request_response()
            # Cleared only once the replay is on the wire, so a send that failed above leaves the
            # request queued for the next attempt instead of losing it.
            self._pending_response = False
        except (websockets.WebSocketException, OSError, TimeoutError, RealtimeHandshakeError):
            # Expected dial/handshake failures: protocol/connection errors, network failures (DNS,
            # refused, reset), the handshake timeout, and a re-dial the server rejected with an `error`
            # frame or answered with a frame we couldn't parse — `map_connect_errors` only wraps the
            # *initial* dial, so `expect_event`'s `RealtimeHandshakeError` reaches this loop raw. A retry
            # may still succeed. Anything else is a bug in `dial()` and propagates rather than
            # masquerading as a failed reconnect.
            return False
        return True

    def _clear_active_response(self) -> None:
        """Forget the response currently being generated, so a new one can start."""
        self._response_active = False
        self._response_started = False
        self._active_response_id = None
        self._cancel_sent = False
        # On a sideband, the provider keeps playing this response's audio to the browser after
        # `response.done`, and a barge-in truncation during that tail still has to name the playing
        # item — so it is retired when playback ends (`output_audio_buffer.stopped`/`.cleared`) instead.
        if self._observes_output_audio or not self._output_audio_playing:
            self._current_item_id = None
        self._generated_audio_bytes = 0


@dataclass(init=False)
class OpenAIRealtimeModel(RealtimeModel):
    """OpenAI Realtime API model.

    Authentication and the base URL come from a
    [`Provider`][pydantic_ai.providers.Provider], mirroring [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel].
    Pass `provider='openai'` (the default) to read `OPENAI_API_KEY` / `OPENAI_BASE_URL` from the
    environment, or an [`OpenAIProvider`][pydantic_ai.providers.openai.OpenAIProvider] instance for a
    custom key or base URL. The realtime transport is opened separately with `websockets`, so the
    provider's `httpx` client is not used for the WebSocket connection. The realtime WebSocket URL is
    derived from the provider's base URL (e.g. `https://api.openai.com/v1/` →
    `wss://api.openai.com/v1/realtime`), so OpenAI-compatible endpoints that expose a realtime API
    work too.

    Args:
        model: The model name, e.g. `gpt-realtime` or `gpt-realtime-2.1-mini`.
        provider: The provider to use for authentication and the base URL. Defaults to `'openai'`.
            Azure OpenAI is not supported (its realtime endpoint uses a different URL and auth scheme).
        settings: [Model settings][pydantic_ai.realtime.RealtimeModelSettings] used as defaults for
            realtime sessions.
        profile: Optional override for the [realtime model profile][pydantic_ai.realtime.RealtimeModelProfile],
            merged over the provider's — a partial dict, or a callable taking the resolved profile and
            returning the one to use. Mirrors `profile=` on a standard
            [`Model`][pydantic_ai.models.Model], and is the escape hatch when a model name doesn't
            identify the model (e.g. an Azure deployment named something other than its model).
    """

    # The connection class `connect` yields; a protocol clone (Azure) overrides it to correct the
    # vendor a closed or rejecting connection names in its errors.
    _connection_type: ClassVar[type[OpenAIRealtimeConnection]] = OpenAIRealtimeConnection

    model: OpenAIRealtimeModelName
    _: KW_ONLY
    settings: RealtimeModelSettings | None = None
    _provider: Provider[AsyncOpenAI] = field(init=False, repr=False)

    # Written out rather than generated because `profile` has to be an init argument while
    # `RealtimeModel.profile` stays the *resolved* profile, exactly as on a standard `Model` — a
    # dataclass field of that name would shadow the property.
    def __init__(
        self,
        model: OpenAIRealtimeModelName,
        *,
        provider: Provider[AsyncOpenAI] | str = 'openai',
        settings: RealtimeModelSettings | None = None,
        profile: RealtimeModelProfileSpec | None = None,
    ) -> None:
        self.model = model
        self.settings = settings
        self._profile = profile
        self._provider = self._resolve_provider(provider)

    @staticmethod
    def _resolve_provider(provider: Provider[AsyncOpenAI] | str) -> Provider[AsyncOpenAI]:
        """Resolve the `provider=` argument; a protocol clone (Azure) overrides which providers it takes."""
        if isinstance(provider, str):
            provider = cast('Provider[AsyncOpenAI]', infer_provider(provider))
        if provider.name == 'azure':
            raise UserError(
                'Azure OpenAI is not supported through `OpenAIRealtimeModel`: its realtime endpoint uses a '
                'different URL and authentication scheme. Use `AzureRealtimeModel` (or the '
                '`azure:` prefix) for Azure OpenAI realtime.'
            )
        return provider

    @property
    def client(self) -> AsyncOpenAI:
        """The underlying [`AsyncOpenAI`](https://github.com/openai/openai-python) client from the provider."""
        return self._provider.client

    @property
    def model_name(self) -> OpenAIRealtimeModelName:
        return self.model

    @property
    def system(self) -> str:
        return self._provider.name

    def _session_config(
        self,
        instructions: str,
        tools: list[ToolDefinition] | None,
        *,
        model_settings: OpenAIRealtimeModelSettings | None,
    ) -> dict[str, Any]:
        model_settings = cast('OpenAIRealtimeModelSettings', self._merge_model_settings(model_settings) or {})
        if 'openai_turn_detection' in model_settings:
            turn_detection = model_settings['openai_turn_detection']
        elif 'turn_detection' in model_settings:
            turn_detection = resolve_base_turn_detection(model_settings['turn_detection'])
        else:
            turn_detection: ServerVAD | SemanticVAD | None = {'type': 'server_vad'}
        # `turn_detection` is always set: a dict enables VAD, `None` (explicit null) disables it.
        audio_input: dict[str, Any] = {
            'format': {'type': 'audio/pcm', 'rate': self.profile.get('audio_input_sample_rate', 24000)},
            'turn_detection': turn_detection_config(turn_detection),
        }
        transcription_model = resolve_transcription_model(
            model_settings.get('input_transcription_model', 'auto'), default=_AUTO_TRANSCRIPTION_MODEL
        )
        if transcription_model is not None:
            audio_input['transcription'] = {'model': transcription_model}
        if (noise_reduction := model_settings.get('openai_input_noise_reduction')) is not None:
            audio_input['noise_reduction'] = {'type': noise_reduction}
        audio_output: dict[str, Any] = {
            'format': {'type': 'audio/pcm', 'rate': self.profile.get('audio_output_sample_rate', 24000)}
        }
        if voice := model_settings.get('openai_voice'):
            audio_output['voice'] = voice.model_dump() if isinstance(voice, VoiceID) else voice
        if (output_speed := model_settings.get('openai_output_speed')) is not None:
            audio_output['speed'] = output_speed
        config: dict[str, Any] = {
            'type': 'realtime',
            'instructions': instructions,
            'output_modalities': [model_settings.get('output_modality', 'audio')],
            'audio': {'input': audio_input, 'output': audio_output},
        }
        advertised_tools, tool_choice = resolve_advertised_tools(tools, model_settings.get('tool_choice'))
        if advertised_tools:
            config['tools'] = [tool_def_to_openai(t) for t in advertised_tools]
        # Note: GA realtime sessions have no `temperature` field, so it is intentionally not forwarded.
        if (max_tokens := model_settings.get('max_tokens')) is not None:
            config['max_output_tokens'] = max_tokens
        if (parallel_tool_calls := model_settings.get('parallel_tool_calls')) is not None:
            config['parallel_tool_calls'] = parallel_tool_calls
        if tool_choice is not None:
            config['tool_choice'] = tool_choice_config(tool_choice)
        if (truncation := model_settings.get('openai_truncation')) is not None:
            # Already the OpenAI `truncation` wire shape (`'auto'`/`'disabled'`/retention-ratio dict).
            config['truncation'] = truncation
        if (thinking := model_settings.get('thinking')) is not None:
            if self.profile.get('supports_thinking', False):
                # `False` maps to `'none'`, which the realtime `reasoning.effort` doesn't accept — omit
                # it so a reasoning model falls back to its default rather than erroring.
                if (effort := OPENAI_REASONING_EFFORT_MAP[thinking]) != 'none':
                    config['reasoning'] = {'effort': effort}
        return config

    def _realtime_ws_base(self) -> str:
        return realtime_websocket_url(self._provider.base_url)

    def _realtime_url(self, model_settings: OpenAIRealtimeModelSettings | None = None) -> str:
        del model_settings  # only the Azure Voice Live override varies the URL on settings
        return with_realtime_query(self._realtime_ws_base(), model=self.model)

    def _sideband_url(self, call_id: str) -> str:
        return with_realtime_query(self._realtime_ws_base(), call_id=call_id)

    def _webrtc_http_base(self) -> str:
        # Split off the fragment first (like `realtime_websocket_url` / `with_realtime_query`), so the
        # trailing slash and the appended path land before it rather than inside the client-side part.
        base_url, _, fragment = self._provider.base_url.partition('#')
        base_url, separator, query = base_url.partition('?')
        base_url = base_url if base_url.endswith('/') else f'{base_url}/'
        base_url = f'{base_url}{separator}{query}'
        return f'{base_url}#{fragment}' if fragment else base_url

    def _webrtc_url(self, path: str, **params: str) -> str:
        base_url, _, fragment = self._webrtc_http_base().partition('#')
        base_url, _, query = base_url.partition('?')
        for name, value in params.items():
            param = f'{name}={quote(value, safe="")}'
            query = f'{query}&{param}' if query else param
        url = f'{base_url}{path}'
        url = f'{url}?{query}' if query else url
        return f'{url}#{fragment}' if fragment else url

    def _webrtc_calls_url(self) -> str:
        return self._webrtc_url('realtime/calls')

    def _webrtc_client_secrets_url(self) -> str:
        return self._webrtc_url('realtime/client_secrets')

    @property
    def _http_client(self) -> AsyncHTTPClient:
        return self._provider.client._client  # pyright: ignore[reportPrivateUsage]

    async def _webrtc_headers(self) -> dict[str, str]:
        headers = {
            key: value
            for key, value in self._provider.client.default_headers.items()
            if isinstance(value, str) and key.lower() not in ('accept', 'content-type', 'authorization', 'api-key')
        }
        headers.update(await self._auth_headers())
        return headers

    def _webrtc_session_config(
        self,
        instructions: str | None,
        tools: Sequence[ToolDefinition] | None,
        model_settings: RealtimeModelSettings | None,
    ) -> dict[str, Any]:
        settings = cast('OpenAIRealtimeModelSettings | None', model_settings)
        return {
            'model': self.model,
            **self._session_config(instructions or '', list(tools) if tools else None, model_settings=settings),
        }

    async def create_client_secret(
        self,
        *,
        instructions: str | None = None,
        tools: Sequence[ToolDefinition] | None = None,
        model_settings: RealtimeModelSettings | None = None,
        expires_after_seconds: int | None = None,
    ) -> RealtimeClientSecret:
        return await _mint_client_secret(
            http_client=self._http_client,
            client_secrets_url=self._webrtc_client_secrets_url(),
            headers=await self._webrtc_headers(),
            model_name=self.model_name,
            session_config=self._webrtc_session_config(instructions, tools, model_settings),
            expires_after_seconds=expires_after_seconds,
        )

    async def answer_webrtc_offer(
        self,
        sdp_offer: str,
        *,
        instructions: str | None = None,
        tools: Sequence[ToolDefinition] | None = None,
        model_settings: RealtimeModelSettings | None = None,
    ) -> WebRTCAnswer:
        return await _answer_webrtc_offer(
            http_client=self._http_client,
            calls_url=self._webrtc_calls_url(),
            headers=await self._webrtc_headers(),
            provider_name=self.system,
            model_name=self.model_name,
            sdp_offer=sdp_offer,
            session_config=self._webrtc_session_config(instructions, tools, model_settings),
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
        if session.provider_name != self.system:
            raise UserError(
                f'This WebRTC call was negotiated by provider {session.provider_name!r}, but this realtime '
                f'model connects through {self.system!r}. Answer the offer and attach the sideband with the '
                'same model/provider.'
            )
        settings = cast('OpenAIRealtimeModelSettings', self._merge_model_settings(model_settings) or {})
        handshake_timeout = settings.get('handshake_timeout', 30.0)
        instructions = get_instructions(messages, model_request_parameters) or ''
        session_config = self._session_config(
            instructions, model_request_parameters.function_tools, model_settings=settings
        )
        transcription_enabled = settings.get('input_transcription_model', 'auto') is not None
        seed = await seed_items(messages, profile=self.profile, provider_name=self.system)
        url = self._sideband_url(session.session_id)
        cm: AbstractAsyncContextManager[ClientConnection] | None = None
        server_model: str | None = None

        async def dial() -> ClientConnection:
            nonlocal cm, server_model
            if cm is not None:  # pragma: no branch
                previous, cm = cm, None
                await previous.__aexit__(None, None, None)
            headers = await self._auth_headers()
            inject_trace_context(headers)
            opening = websockets.connect(url, additional_headers=headers)
            ws = await opening.__aenter__()
            cm = opening
            # Existing WebRTC calls do not emit `session.created`; configure immediately and wait
            # for `session.updated` instead of using `connect_openai_protocol`'s new-session handshake.
            await ws.send(to_json({'type': SESSION_UPDATE_EVENT, 'session': session_config}).decode())
            updated = await expect_event(ws, SESSION_UPDATED_EVENT, timeout=handshake_timeout)
            server_model = _SidebandSessionUpdated.model_validate(updated).session.model
            return ws

        try:
            with map_connect_errors(self.model):
                ws = await dial()
                for item in seed:
                    await ws.send(to_json({'type': CONVERSATION_ITEM_CREATE_EVENT, 'item': item}).decode())
            yield self._connection_type(
                ws,
                dial=dial,
                reconnect=settings.get('reconnect'),
                input_transcription_enabled=transcription_enabled,
                model_name=server_model,
                model_name_getter=lambda: server_model,
                observes_output_audio=False,
            )
        finally:
            if cm is not None:  # pragma: no branch
                await cm.__aexit__(None, None, None)

    async def _auth_headers(self, model_settings: OpenAIRealtimeModelSettings | None = None) -> dict[str, str]:
        # `model_settings` lets a provider vary auth by session (e.g. Azure Voice Live uses a different
        # resource key); OpenAI's auth doesn't depend on it.
        del model_settings
        # The raw WebSocket handshake bypasses the SDK's request path, which is where `AsyncOpenAI`
        # resolves anything but a static key, so both dynamic forms are resolved the same way here.
        client = self._provider.client
        # A `workload_identity` client leaves `client.api_key` set to a placeholder string and
        # exchanges it for a real token per request; sending the placeholder would fail the handshake
        # with an opaque auth error.
        if (workload_identity := client._workload_identity_auth) is not None:  # pyright: ignore[reportPrivateUsage]
            return {'Authorization': f'Bearer {await workload_identity.get_token_async()}'}
        # An async `api_key` provider leaves `client.api_key` empty until resolved. The SDK's own
        # refresh is a no-op returning the static key when no provider is configured, so the handshake
        # stays byte-identical in that case.
        api_key = await client._refresh_api_key()  # pyright: ignore[reportPrivateUsage]
        return {'Authorization': f'Bearer {api_key}'}

    def _connection_class(self, model_settings: OpenAIRealtimeModelSettings) -> type[OpenAIRealtimeConnection]:
        """The connection class for a session, given its settings.

        Defers to [`_connection_type`][] — the declarative seam a protocol clone sets to correct the
        vendor its errors name — and exists on top of it for a provider whose connection varies by
        *session* rather than by model, as Azure's does for Voice Live.
        """
        del model_settings
        return self._connection_type

    def _session_model_name(self, created: dict[str, Any], model_settings: OpenAIRealtimeModelSettings) -> str | None:
        """The server-reported model name from the `session.created` handshake frame.

        Settings-aware because a provider's handshake shape can vary by *session*: Azure Voice Live's
        beta `session.created` doesn't carry the GA `type` discriminator this SDK model requires.
        """
        del model_settings
        session = SessionCreatedEvent.model_validate(created).session
        model = session.model if isinstance(session, RealtimeSessionCreateRequest) else None
        return model if isinstance(model, str) else None

    @asynccontextmanager
    async def connect(
        self,
        *,
        messages: Sequence[ModelMessage],
        model_settings: RealtimeModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncGenerator[OpenAIRealtimeConnection]:
        settings = cast('OpenAIRealtimeModelSettings', self._merge_model_settings(model_settings) or {})
        handshake_timeout = settings.get('handshake_timeout', 30.0)
        instructions = get_instructions(messages, model_request_parameters) or ''
        session_config = self._session_config(
            instructions=instructions, tools=model_request_parameters.function_tools, model_settings=settings
        )
        transcription_enabled = settings.get('input_transcription_model', 'auto') is not None

        async def dial_headers() -> dict[str, str]:
            headers = await self._auth_headers(settings)
            # The raw WebSocket bypasses the provider's `httpx` client, so every fresh handshake must
            # carry the current trace context as well as freshly resolved authentication.
            inject_trace_context(headers)
            return headers

        def session_model(created: dict[str, Any]) -> str | None:
            return self._session_model_name(created, settings)

        def build_connection(
            ws: ClientConnection,
            dial: Callable[[], Awaitable[ClientConnection]],
            server_model: str | None,
            model_name_getter: Callable[[], str | None],
        ) -> OpenAIRealtimeConnection:
            return self._connection_class(settings)(
                ws,
                dial=dial,
                reconnect=settings.get('reconnect'),
                input_transcription_enabled=transcription_enabled,
                model_name=server_model,
                model_name_getter=model_name_getter,
            )

        async with connect_openai_protocol(
            model_name=self.model,
            messages=messages,
            profile=self.profile,
            provider_name=self.system,
            session_config=session_config,
            handshake_timeout=handshake_timeout,
            dial_headers=dial_headers,
            dial_url=lambda: self._realtime_url(settings),
            session_model=session_model,
            build_connection=build_connection,
            replay_on_redial=True,
        ) as connection:
            yield connection
