"""OpenAI Realtime API provider for speech-to-speech sessions.

Connects to `wss://api.openai.com/v1/realtime` over a WebSocket and maps the OpenAI event
protocol to the shared realtime event types.

Requires the `websockets` and `openai` packages, available via the `realtime` and `openai` optional
groups:

    pip install "pydantic-ai-slim[openai-realtime]"
"""

from __future__ import annotations as _annotations

import base64
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Sequence, Set as AbstractSet
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
        InputAudioBufferTimeoutTriggered,
        RealtimeErrorEvent,
        RealtimeResponseUsage,
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
    RealtimeInputTranscriptionErrorEvent,
    RealtimeOutputSpeechEndEvent,
    RealtimeOutputSpeechStartEvent,
    RealtimeSessionErrorEvent,
    RealtimeSessionReconnectEvent,
    ToolCallPart,
)
from ..models import ModelRequestParameters
from ..models.openai import _map_api_errors as map_openai_api_errors  # pyright: ignore[reportPrivateUsage]
from ..profiles.openai import OPENAI_REASONING_EFFORT_MAP
from ..providers import Provider, infer_provider
from ..tools import ToolDefinition
from ..usage import RequestUsage
from ._lifecycle import InputId, LifecycleEvent, RealtimeFrame, ResponseStatus, StaleEvent
from ._openai_lifecycle import OpenAILifecycle, frame_response_id
from ._openai_protocol import (
    ACTIVE_RESPONSE_ERROR_CODE,
    ACTIVE_RESPONSE_RETRIES,
    AUDIO_DELTA_TYPES,
    CONVERSATION_ITEM_ADDED_EVENT_ADAPTER,
    CONVERSATION_ITEM_ADDED_EVENTS,
    CONVERSATION_ITEM_CREATE_EVENT,
    CONVERSATION_ITEM_DELETE_EVENT,
    CONVERSATION_ITEM_TRUNCATE_EVENT,
    INPUT_AUDIO_BUFFER_APPEND_EVENT,
    INPUT_AUDIO_BUFFER_CLEAR_EVENT,
    INPUT_AUDIO_BUFFER_COMMIT_EVENT,
    INPUT_AUDIO_BUFFER_TIMEOUT_TRIGGERED_EVENT,
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
    client_event_id,
    client_item_id,
    config_has_manual_turns,
    config_interrupts_response_on_speech,
    connect_openai_protocol,
    expect_event,
    is_user_message_item,
    loads_obj,
    map_connect_errors,
    map_event,
    openai_websocket_auth_headers,
    realtime_websocket_url,
    rejected_inputs,
    resolve_base_turn_detection,
    resolve_transcription_model,
    response_failed_error,
    response_finish_reason,
    response_provider_details,
    response_request_metadata,
    response_status_reason,
    seed_items,
    tool_choice_config,
    tool_content_item_id,
    tool_def_to_openai,
    turn_detection_config,
    user_message_item,
    with_realtime_query,
)
from ._openai_webrtc import (
    HANG_UP_MAX_RETRIES,
    HANG_UP_TIMEOUT,
    answer_webrtc_offer as _answer_webrtc_offer,
    ignore_ended_call,
    mint_client_secret as _mint_client_secret,
)
from ._utils import (
    DEFAULT_MAX_RECONNECTS,
    inject_trace_context,
    reconnect_with_backoff,
    require_pcm_audio,
    resolve_advertised_tools,
)
from .codec import (
    AudioDelta,
    CancelResponse,
    ClearAudio,
    CommitAudio,
    CreateResponse,
    InputRejected,
    InputTranscript,
    RealtimeCodecEvent,
    RealtimeConnection,
    RealtimeInput,
    SessionUsage,
    TextContext,
    ToolCall,
    ToolResult,
    TruncateOutput,
)
from .model import RealtimeClientSecret, RealtimeModel, RealtimeProviderSession, WebRTCAnswer
from .profiles import DEFAULT_AUDIO_SAMPLE_RATE, RealtimeModelProfileSpec
from .settings import RealtimeModelSettings, ReconnectPolicy

_OUTPUT_AUDIO_BUFFER_CLEAR_EVENT = 'output_audio_buffer.clear'
_MAX_TRACKED_OUTPUT_ITEMS = 32
"""How many recent output items a barge-in can still name for truncation: far more than can be queued for playback."""
_OUTPUT_SPEECH_START_FRAME = 'output_audio_buffer.started'
_OUTPUT_SPEECH_END_FRAMES = frozenset({'output_audio_buffer.stopped', 'output_audio_buffer.cleared'})


class _SidebandContentPart(BaseModel):
    type: str


class _SidebandContentPartAdded(BaseModel):
    item_id: str
    content_index: int = 0
    part: _SidebandContentPart


class _HandshakeSession(BaseModel):
    model: str | None = None


class _HandshakeSessionEvent(BaseModel):
    """The slice of a handshake's `session.created` or `session.updated` frame the connection reads.

    The SDK's session models close their enums (modalities, formats, turn detection, truncation, ...) to
    today's values, so a server-side default it doesn't know yet would fail the whole handshake.
    """

    session: _HandshakeSession


LatestOpenAIRealtimeModelNames = Literal['gpt-realtime', 'gpt-realtime-2.1', 'gpt-realtime-2.1-mini']
OpenAIRealtimeModelName = str | LatestOpenAIRealtimeModelNames

LatestOpenAIRealtimeTranscriptionModelNames = Literal[
    # OpenAI deprecated these three on 2026-08-26 and shuts them down on 2027-02-26, in favor of
    # `gpt-live-transcribe` and `gpt-transcribe`.
    'whisper-1',
    'gpt-4o-transcribe',
    'gpt-4o-mini-transcribe',
    'gpt-realtime-whisper',
    'gpt-live-transcribe',
    'gpt-transcribe',
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
    openai_prune_tool_items: bool
    """Whether to delete a tool round's items from the server conversation once the model is done with them.

    The OpenAI Realtime API bills every response for the whole conversation it keeps, so a tool-heavy agent pays
    for every earlier `function_call` and `function_call_output` item again on each later response. With this set,
    once a response completes without calling a tool, the session deletes the items of the calls the model has
    answered (each call with its output and any content the tool returned besides it, never one without the other),
    so later responses no longer read them. To keep some tools' items, name them in `openai_keep_tool_items`.
    [`all_messages()`][pydantic_ai.realtime.RealtimeSession.all_messages] keeps everything, and a reconnect's replay
    leaves the pruned items out. Off by default.
    """
    openai_keep_tool_items: AbstractSet[str]
    """The tools whose items `openai_prune_tool_items` leaves in the server conversation, for results the model may
    need to refer back to.

    Only applies with `openai_prune_tool_items` set to `True`; without it, nothing is pruned and this is ignored.
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
        output_reasoning_tokens=details.get('reasoning_tokens', 0),
        # Image input is priced at its own rate (and cached image input at another), so it has to reach
        # pricing under the meter names genai-prices reads; in `details` alone it was priced as text.
        **({'input_image_tokens': image} if (image := details.get('input_image_tokens')) else {}),
        **({'cache_image_read_tokens': cached_image} if (cached_image := _int_or_none(cached, 'image_tokens')) else {}),
        details=details,
    )


def _int_or_none(obj: object | None, name: str) -> int | None:
    value = getattr(obj, name, None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


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


@dataclass
class _DecodedFrame:
    """What one server frame makes: its codec events, and the lifecycle events around them."""

    codec: list[RealtimeCodecEvent] = field(default_factory=list[RealtimeCodecEvent])
    _: KW_ONLY
    before: list[LifecycleEvent] = field(default_factory=list[LifecycleEvent])
    """Lifecycle events that precede the codec events: the start of a response first seen here."""
    after: list[LifecycleEvent] = field(default_factory=list[LifecycleEvent])
    stale: bool = False
    """Whether the frame is about a response that has already ended, so its codec events repeat or trail
    that response's terminal and are left out of the lifecycle stream."""
    redial: str | None = None
    """Why the connection re-dials right after this frame, when the server ended the conversation on it in a way a
    new connection recovers from (xAI's `max_duration`): the socket is dropped rather than waited on to close."""

    @property
    def ends_session(self) -> bool:
        """Whether the frame reports the session over, with a non-recoverable error."""
        return any(isinstance(event, RealtimeSessionErrorEvent) and not event.recoverable for event in self.codec)

    def frame(self) -> RealtimeFrame:
        """The frame's events in order, its codec events marked stale if it is."""
        codec: list[RealtimeCodecEvent | StaleEvent] = (
            [StaleEvent(event) for event in self.codec] if self.stale else [*self.codec]
        )
        return [*self.before, *codec, *self.after]


@dataclass
class _ToolCallBatch:
    """The tool calls one response made, which get a single `response.create` once all have outputs."""

    unanswered: set[str] = field(default_factory=set[str])
    """Calls whose output hasn't been sent yet."""
    inputs: list[int] = field(default_factory=list[int])
    """The `send()` inputs that carried the outputs, which the `response.create` is made for."""
    done: bool = False
    """Whether the response is done, so no further call can join."""


@dataclass
class _ToolItems:
    """A tool call's items in the server conversation: what `openai_prune_tool_items` deletes together."""

    call: str
    output: str | None = None
    content: list[str] = field(default_factory=list[str])
    """User message items with content the tool returned besides its output."""


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
        interrupts_response_on_speech: bool = False,
        model_name: str | None = None,
        model_name_getter: Callable[[], str | None] | None = None,
        observes_output_audio: bool = True,
        audio_output_sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE,
        prune_tool_items: bool = False,
        keep_tool_items: AbstractSet[str] = frozenset(),
        manual_turns: bool = False,
    ) -> None:
        self._ws = ws
        # See `OpenAIRealtimeModelSettings.openai_prune_tool_items` and `openai_keep_tool_items`.
        self._prune_tool_items = prune_tool_items
        self._keep_tool_items = keep_tool_items
        # The items of each tool call in the server conversation, by call id, while pruning (see `_track_tool_item`).
        self._tool_items: dict[str, _ToolItems] = {}
        self._interrupts_response_on_speech = interrupts_response_on_speech
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
        self._gave_up = False
        # Set once the server reports the session over, e.g. xAI's `max_duration` error.
        self._session_ended = False
        self._observes_output_audio = observes_output_audio
        # Output audio is mono PCM16 at this rate: 2 bytes per sample, so a barge-in's truncation can be
        # clamped to the milliseconds of audio actually generated.
        self._output_audio_bytes_per_second = audio_output_sample_rate * 2
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
        # Every `send()` call is numbered (see `InputRejected.input_index`), and the frames a refusal can
        # take back carry the numbers of the inputs they serve in their `event_id` (`client_event_id`).
        # A response request covers the inputs that asked for it: the one that sent it, or every input
        # deferred behind an active response (`_pending_response`), which then share one request.
        self._inputs_received = 0
        self._deferred_response_inputs: list[int] = []
        self._response_request_inputs: tuple[int, ...] = ()
        # The inputs each of those requests answers: the ones that asked for it, except that a tool-call
        # batch's request is named after its last output (the session counts one reply for the batch)
        # while it answers all of them. Sent as the request's `metadata`, which the response echoes.
        self._deferred_response_answers: list[InputId] = []
        # Our request for a response that a response the provider started on its own (server VAD) overtook: still
        # to be refused (while that one is under way), or, on a protocol clone that cancels its own instead, started.
        self._request_in_flight: tuple[tuple[int, ...], tuple[InputId, ...]] | None = None
        # How many of our requests in a row the provider refused because a response of its own was under way.
        self._active_response_refusals = 0
        # Whether the user barged in on the response that overtook `_request_in_flight`: their speech takes the turn,
        # so if the provider refuses the request on that response's account, it isn't asked for again.
        self._in_flight_cut_off = False
        self._response_request_answers: tuple[InputId, ...] = ()
        # Which response, user turn, and input each frame is about (see `frames()`). Under server VAD (unless
        # `manual_turns` says turn detection is off), the provider commits the speech it hears on its own too.
        self._lifecycle = OpenAILifecycle(transcribes=input_transcription_enabled, server_vad=not manual_turns)
        # The calls each response made, by response id, and the response each call came from: a response's
        # tool results get a single `response.create`, once the response is done and every call has its
        # output. Asking after each one had the model answer before its sibling calls had results.
        self._tool_call_batches: dict[str, _ToolCallBatch] = {}
        self._tool_call_responses: dict[str, str] = {}
        self._cancel_sent = False
        # Id of a response we cancelled (barge-in): the server keeps streaming a few straggler deltas
        # before its `response.done`, and mapping them would surface speech the user already interrupted.
        # Drop frames carrying this id until its `response.done` closes it. `None` when nothing is being
        # cancelled, or when the cancelled response had no id (defensive/compat path — then not suppressed).
        self._cancelled_response_id: str | None = None
        # The current output audio item, tracked from output-audio deltas so a `TruncateOutput` can
        # name it. These are mutated by `frames` and read by `send` from a separate task; under a
        # single cooperative event loop the plain reads/writes are safe and eventually consistent.
        self._current_item_id: str | None = None
        self._current_content_index = 0
        self._generated_audio_bytes = 0
        # Recent output items, with their audio content index and generated audio bytes, so a barge-in
        # can truncate a reply the listener was still hearing after a newer one was generated. Bounded,
        # and cleared once none of them can be playing (a reconnect, a sideband's playback end).
        self._output_items: dict[str, tuple[int, int]] = {}
        self._output_audio_playing = False
        self._output_speech_clear_sent = False
        # Input items server VAD committed because its `idle_timeout_ms` ran out with nobody speaking. The
        # server commits the silent buffer and starts a follow-up response to it, but no user turn happened:
        # the lifecycle keeps the commit out of the user turns, and the connection drops the item's
        # transcription (empty, or failed). Tracked only while transcribing, until that transcription ends.
        self._idle_timeout_items: set[str] = set()

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

    def _history_items_sent(self, items: Sequence[dict[str, Any]]) -> None:
        """Seeded or replayed history went out on the socket: the items that join the conversation place no input."""
        self._lifecycle.foreign_messages_sent(sum(is_user_message_item(item) for item in items))

    @property
    def _can_reconnect(self) -> bool:
        return (
            not self._gave_up
            and not self._session_ended
            and self._dial is not None
            and self._reconnect is not None
            and self._reconnects_used < self._reconnect.get('max_reconnects', DEFAULT_MAX_RECONNECTS)
        )

    @property
    def _answers_tool_calls_per_response(self) -> bool:
        return True

    @property
    def reconnect_restores_in_flight_state(self) -> bool:
        # Local replay restores only the finalized turns; the response and tool calls in flight when
        # the socket dropped are gone, so the session settles them.
        return False

    @property
    def input_transcription_enabled(self) -> bool:
        return self._input_transcription_enabled

    @property
    def interrupts_response_on_speech(self) -> bool:
        return self._interrupts_response_on_speech

    async def send(self, content: RealtimeInput) -> None:
        """Send content to the OpenAI Realtime API.

        Accepts `BinaryAudio` (raw PCM16, 24kHz, mono), a `str` text turn, `TextContext` (text added
        without soliciting a reply), `BinaryImage`, `ToolResult`, and the control verbs `CommitAudio`, `ClearAudio`, `CreateResponse`,
        `CancelResponse`, and `TruncateOutput`.
        """
        input_index = self._inputs_received
        self._inputs_received += 1
        if isinstance(content, BinaryAudio):
            require_pcm_audio(content, provider_name=self._provider_label)
            await self._send_event(
                {
                    'type': INPUT_AUDIO_BUFFER_APPEND_EVENT,
                    'audio': base64.b64encode(content.data).decode('ascii'),
                }
            )
        elif isinstance(content, (str, TextContext)):
            text = content if isinstance(content, str) else content.text
            await self._send_text(text, respond=isinstance(content, str), input_index=input_index)
        elif isinstance(content, ToolResult):
            await self._send_tool_result(content, input_index=input_index)
        elif isinstance(content, BinaryImage):
            # An image is added as conversation context (like a video frame), not a turn of its own,
            # so it doesn't trigger a response — drive that with audio (VAD) or `CreateResponse`.
            data_uri = content.data_uri
            self._lifecycle.message_sent(input_index)
            await self._send_event(
                {
                    'type': CONVERSATION_ITEM_CREATE_EVENT,
                    'event_id': client_event_id('content', (input_index,)),
                    'item': {
                        'id': client_item_id(input_index),
                        'type': 'message',
                        'role': 'user',
                        'content': [{'type': 'input_image', 'image_url': data_uri}],
                    },
                }
            )
        elif isinstance(content, CommitAudio):
            # Under its own `event_id`, which an `input_audio_buffer_commit_empty` error refusing it echoes. (Not a
            # `client_event_id`: a refused commit takes back no input.)
            await self._send_event(
                {'type': INPUT_AUDIO_BUFFER_COMMIT_EVENT, 'event_id': f'pydantic_ai.commit.{input_index}'}
            )
        elif isinstance(content, ClearAudio):
            await self._send_event({'type': INPUT_AUDIO_BUFFER_CLEAR_EVENT})
        elif isinstance(content, CreateResponse):
            await self._solicit_response((input_index,))
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
            await self._truncate_output(content)
        else:
            raise UserError(f'{self._provider_label} does not support {type(content).__name__} input.')

    async def _truncate_output(self, content: TruncateOutput) -> None:
        if content.item_id is None:
            # No current output item (e.g. the model wasn't speaking) → nothing to truncate.
            if self._current_item_id is None:
                return
            item_id, content_index, generated = (
                self._current_item_id,
                self._current_content_index,
                self._generated_audio_bytes,
            )
        elif (item := self._output_items.get(content.item_id)) is not None:
            item_id, (content_index, generated) = content.item_id, item
        else:
            # An item this connection no longer tracks (e.g. from before a reconnect) can't be playing.
            return
        audio_end_ms = content.audio_end_ms
        # A WebRTC sideband connection does not receive output-audio deltas because media flows
        # directly between the browser and provider. Only clamp connections that observe those
        # deltas; otherwise the byte counter stays zero and every barge-in would truncate to zero.
        if self._observes_output_audio:
            audio_end_ms = min(audio_end_ms, generated * 1000 // self._output_audio_bytes_per_second)
        await self._send_event(
            {
                'type': CONVERSATION_ITEM_TRUNCATE_EVENT,
                'item_id': item_id,
                'content_index': content_index,
                'audio_end_ms': audio_end_ms,
            }
        )

    def _track_output_item(self, item_id: str, content_index: int, generated: int) -> None:
        self._output_items.pop(item_id, None)
        self._output_items[item_id] = (content_index, generated)
        if len(self._output_items) > _MAX_TRACKED_OUTPUT_ITEMS:
            del self._output_items[next(iter(self._output_items))]

    @property
    def _prunes_tool_calls(self) -> bool:
        return self._prune_tool_items

    def _track_tool_item(self, data: dict[str, Any]) -> None:
        """Remember the id of a tool call's item, or its output's, as it joins the server conversation, to prune it."""
        if not self._prunes_tool_calls:
            return
        # (The protocol's own item model: the SDK's union rejects the shapes clones send, as `item_added` notes.)
        try:
            item = CONVERSATION_ITEM_ADDED_EVENT_ADAPTER.validate_python(data).item
        except ValueError:
            return  # (malformed: the lifecycle reports it, as it reads the same frame)
        if item.call_id is None or item.id is None:
            return
        if item.type == 'function_call':
            if item.name not in self._keep_tool_items:
                self._tool_items[item.call_id] = _ToolItems(call=item.id)
        elif item.type == 'function_call_output' and (items := self._tool_items.get(item.call_id)) is not None:
            items.output = item.id

    async def _prune_tool_calls(self, calls: Sequence[ToolCallPart]) -> Sequence[str]:
        pruned: list[str] = []
        for call in calls:
            items = self._tool_items.get(call.tool_call_id)
            if items is None or items.output is None:
                # A kept tool's, or a call this conversation doesn't hold both the call and the output of.
                continue
            del self._tool_items[call.tool_call_id]
            for item_id in (items.call, items.output, *items.content):
                await self._send_event({'type': CONVERSATION_ITEM_DELETE_EVENT, 'item_id': item_id})
            pruned.append(call.tool_call_id)
        return pruned

    async def _send_tool_result(self, content: ToolResult, *, input_index: int) -> None:
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
        self._lifecycle.tool_output_sent(content.tool_call_id, input_index)
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
            if (items := self._tool_items.get(content.tool_call_id)) is not None:
                # Content the tool returned besides its output (an image, say) is pruned with the call it followed.
                item['id'] = item_id = tool_content_item_id(input_index)
                items.content.append(item_id)
            self._lifecycle.message_sent(None)
            await self._send_event({'type': CONVERSATION_ITEM_CREATE_EVENT, 'item': item})
        if (response_id := self._tool_call_responses.pop(content.tool_call_id, None)) is None:
            # A call this connection didn't see made: its result asks for a response of its own.
            await self._solicit_response((input_index,))
            return
        batch = self._tool_call_batches[response_id]
        batch.unanswered.discard(content.tool_call_id)
        batch.inputs.append(input_index)
        await self._answer_tool_call_batch_if_complete(response_id, solicited=True)

    async def _answer_tool_call_batch_if_complete(self, response_id: str, *, solicited: bool = False) -> None:
        """Ask for the response that answers `response_id`'s tool calls, once it's done and all have outputs.

        `solicited` is for a caller's `send()` of the last output, which is told when the request didn't go out
        (see `_solicit_response`).
        """
        batch = self._tool_call_batches[response_id]
        if batch.done and not batch.unanswered:
            del self._tool_call_batches[response_id]
            # One input names the request, as the batch gets one reply; the response answers every output.
            request = self._solicit_response if solicited else self._request_response
            await request((batch.inputs[-1],), answers=batch.inputs)

    async def _send_text(self, text: str, *, respond: bool, input_index: int) -> None:
        self._lifecycle.message_sent(input_index)
        await self._send_event(
            {
                'type': CONVERSATION_ITEM_CREATE_EVENT,
                'event_id': client_event_id('content', (input_index,)),
                'item': {
                    'id': client_item_id(input_index),
                    'type': 'message',
                    'role': 'user',
                    'content': [{'type': 'input_text', 'text': text}],
                },
            }
        )
        if respond:
            await self._solicit_response((input_index,))

    async def _solicit_response(
        self, input_indexes: Sequence[int], *, answers: Sequence[InputId] | None = None
    ) -> None:
        """Request a response for a caller's `send()`, which is told when the request didn't go out.

        A `response.create` that fails on a dead socket never reached the server, so no response is
        active. Left marked active, a reconnect would re-ask for a response the caller was told had
        failed. Only the caller's own request is taken back: one the receive loop sends for a deferred
        request has no caller to tell, so the reconnect re-asks for it as before.
        """
        ws = self._ws
        try:
            await self._request_response(input_indexes, answers=answers)
        except self.transport_errors:
            # Only a request that went straight out can fail here; a deferred one sends nothing yet. If
            # the link was replaced meanwhile, the active response is the new socket's, not this one.
            if self._ws is ws:
                self._response_active = False
            raise

    async def _request_response(
        self, input_indexes: Sequence[int], *, answers: Sequence[InputId] | None = None
    ) -> None:
        """Ask the model to respond now, or defer until the active response completes.

        `answers` are the inputs the response answers, when not just `input_indexes`.
        """
        answers = input_indexes if answers is None else answers
        if self._response_active:
            self._pending_response = True
            self._deferred_response_inputs.extend(input_indexes)
            self._deferred_response_answers.extend(answers)
        else:
            await self._create_response(input_indexes, answers)

    async def _create_response(self, input_indexes: Sequence[int], answers: Sequence[InputId]) -> None:
        """Send the `response.create` for `input_indexes`, making it the active response."""
        self._response_active = True
        self._active_response_id = None
        self._response_request_inputs = tuple(input_indexes)
        self._response_request_answers = tuple(answers)
        event: dict[str, Any] = {'type': RESPONSE_CREATE_EVENT}
        metadata = response_request_metadata(answers)
        if input_indexes:
            event_id = event['event_id'] = client_event_id('response', input_indexes)
            self._lifecycle.request_sent(event_id, self._response_request_answers, tagged=metadata is not None)
        if metadata is not None:
            event['response'] = {'metadata': metadata}
        await self._send_event(event)

    def _take_deferred_response_inputs(self) -> tuple[tuple[int, ...], tuple[InputId, ...]]:
        """Clear the deferred response request, returning the inputs it was made for and those it answers."""
        self._pending_response = False
        inputs = tuple(self._deferred_response_inputs)
        answers = tuple(self._deferred_response_answers)
        self._deferred_response_inputs.clear()
        self._deferred_response_answers.clear()
        return inputs, answers

    async def _send_event(self, event: dict[str, Any]) -> None:
        if event['type'] != INPUT_AUDIO_BUFFER_COMMIT_EVENT:
            await self._ws.send(to_json(event).decode())
            if event['type'] == INPUT_AUDIO_BUFFER_APPEND_EVENT:
                self._lifecycle.audio_appended()
            elif event['type'] == INPUT_AUDIO_BUFFER_CLEAR_EVENT:
                self._lifecycle.audio_clear_sent()
            return
        # Noted before it goes out, so nothing sent while it does is taken for ahead of it.
        commit = self._lifecycle.audio_commit_sent(event.get('event_id'))
        try:
            await self._ws.send(to_json(event).decode())
        except BaseException:
            self._lifecycle.audio_commit_failed(commit)
            raise

    @asynccontextmanager
    async def _session_scope(self) -> AsyncGenerator[None]:
        """Wrap the connection's whole lifetime, from the end of its handshake to its close.

        A hook for a protocol clone to own background work as long as the connection is open (Eleven Realtime's
        idle keep-alive); `connect()` enters it around its `yield`. A no-op here.
        """
        yield

    def _map_event(self, data: dict[str, Any]) -> RealtimeCodecEvent | None:
        """Map a raw provider frame to a codec event.

        A hook so protocol clones (e.g. the xAI Grok Voice provider) can reuse the whole connection while
        overriding only how frames map to events. Defaults to the OpenAI [`map_event`][pydantic_ai.realtime.openai.map_event].
        """
        return map_event(data)

    async def frames(self) -> AsyncIterator[RealtimeFrame]:
        while True:
            try:
                async for raw in self._ws:
                    if not isinstance(raw, str):
                        continue
                    try:
                        frame = await self._decode_frame(raw)
                    except ValueError as e:
                        # A malformed frame (bad JSON or audio payload) shouldn't tear down the whole
                        # session; surface it as a recoverable error and keep reading.
                        # The frame may have started a response before it failed: announce it all the same.
                        leading = [*self._lifecycle.take_leading(), *self._take_pending_lifecycle()]
                        yield [*leading, _frame_error(e)]
                        continue
                    # The server reporting the session over makes the close that follows final:
                    # re-dialing would only run into the same end.
                    self._session_ended = self._session_ended or frame.ends_session
                    yield frame.frame()
                    if frame.redial is not None:
                        # Whether or not the server closes the socket next, nothing more comes on it: the
                        # re-dial closes it.
                        closed = frame.redial
                        break
                else:
                    # `websockets` ends iteration silently on a *normal* close (1000/1001) and only raises
                    # on an abnormal one, but a session the server hung up on is over either way: OpenAI
                    # ends one that reaches its duration cap with `1001 Your session hit the maximum
                    # duration of 60 minutes.`. Handle it like a drop so a reconnect policy still runs and,
                    # without one, the consumer learns the conversation was cut off instead of seeing the
                    # stream quietly end.
                    if not self._observes_output_audio:
                        self._lifecycle.closed(self._unanswered_inputs())
                        yield [*self._take_pending_lifecycle()]
                        return
                    closed = _describe_close(self._ws)
            except self.transport_errors as e:
                # `ConnectionClosed`, any other protocol error, or a socket-level `OSError` (reset,
                # broken pipe): all mean the link failed, so they all take the same reconnect path
                # instead of escaping the stream and bypassing the reconnect policy.
                closed = str(e)

            reconnects = not self._session_ended and self._reconnect is not None and self._dial is not None
            if reconnects and await self._try_reconnect():
                reconnected = RealtimeSessionReconnectEvent(state_restored=self._restores_state_on_reconnect)
                yield [*self._take_pending_lifecycle(), reconnected]
                continue
            if reconnects:
                # Out of attempts: no reconnect is coming any more.
                self._gave_up = True
            self._lifecycle.closed(self._unanswered_inputs())
            settled: RealtimeFrame = [*self._take_pending_lifecycle()]
            if self._session_ended:
                # The server already said why the session ended; the close adds nothing to report.
                yield settled
                return
            # No reconnect policy, or the reconnect failed: a closed connection is fatal. Surface it as a
            # non-recoverable error and end the stream cleanly, rather than raising.
            reconnect_failed = '; reconnect failed' if reconnects else ''
            error = RealtimeSessionErrorEvent(
                message=f'{self._provider_label} connection closed{reconnect_failed}: {closed}', recoverable=False
            )
            yield [*settled, error]
            return

    def _stop_reading(self) -> Sequence[LifecycleEvent]:
        # A commit whose acknowledgement goes unread is still a turn.
        return self._lifecycle.unacknowledged_commits()

    def _unanswered_inputs(self) -> list[InputId]:
        """The inputs whose request for a response is still to be answered: sent, deferred, or not yet made."""
        unanswered = [*self._deferred_response_answers]
        if self._request_in_flight is not None:
            unanswered += self._request_in_flight[1]
        if self._response_active and not self._response_started:
            unanswered += self._response_request_answers
        unanswered += [input_ for batch in self._tool_call_batches.values() for input_ in batch.inputs]
        return unanswered

    def _take_pending_lifecycle(self) -> list[LifecycleEvent]:
        return self._lifecycle.take_pending()

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
            self._track_output_item(added.item_id, added.content_index, 0)

    async def _decode_frame(self, raw: str) -> _DecodedFrame:  # noqa: C901
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
                return _DecodedFrame()
            self._output_audio_playing = True
            return _DecodedFrame([] if self._observes_output_audio else [RealtimeOutputSpeechStartEvent()])
        if event_type in _OUTPUT_SPEECH_END_FRAMES:
            was_playing, self._output_audio_playing = self._output_audio_playing, False
            self._output_speech_clear_sent = False
            if not self._response_active:
                # The response already closed; playback ending retires its output item (kept alive
                # past `response.done` by `_clear_active_response` for barge-in truncation), and
                # every earlier one it could still name.
                self._current_item_id = None
                self._output_items.clear()
            return _DecodedFrame(
                [] if self._observes_output_audio or not was_playing else [RealtimeOutputSpeechEndEvent()]
            )
        # Drop trailing frames from a response we cancelled on barge-in (its audio/transcript deltas,
        # output-item events, etc.); its own `response.done` still passes through below to close the
        # response, emit usage, and clear the suppression.
        if self._is_cancelled_straggler(event_type, data):
            return _DecodedFrame()
        if event_type == INPUT_AUDIO_BUFFER_TIMEOUT_TRIGGERED_EVENT and self._input_transcription_enabled:
            self._idle_timeout_items.add(InputAudioBufferTimeoutTriggered.model_validate(data).item_id)
        # A frame about a response that has already ended repeats or trails its terminal.
        stale = self._lifecycle.is_ended(frame_response_id(event_type, data))
        if event_type in CONVERSATION_ITEM_ADDED_EVENTS:
            # (The beta protocol, which Azure's preview deployments speak, acknowledges items as created.)
            self._track_tool_item(data)
        self._lifecycle.before_frame(event_type, data)
        try:
            after = self._lifecycle.frame(event_type, data)
        except ValueError:
            # Only the lifecycle reads these frames, and the codec events never depended on them.
            after = []
        events: list[RealtimeCodecEvent] = []
        superseded = False
        asked_again = False
        if event_type == 'error':
            rejected = await self._handle_error(data)
            asked_again = rejected is None
            events.extend(rejected or ())
        if event_type == 'response.done':
            # Settled *before* the frame is mapped: mapping a malformed `response.done` raises
            # `ValueError`, which `frames` surfaces as a recoverable frame error and keeps reading —
            # but the frame was still the only terminal its response will ever get, and bailing before
            # the state updates would leave `_response_active` held forever, queueing every later
            # `response.create` behind a response that already ended.
            done_events, superseded = await self._handle_response_done(data)
            events.extend(done_events)
        event = None if asked_again else self._map_event(data)
        if (
            event_type == 'response.function_call_arguments.done'
            and isinstance(event, ToolCall)
            and isinstance(response_id := data.get('response_id'), str)
        ):
            self._tool_call_batches.setdefault(response_id, _ToolCallBatch()).unanswered.add(event.tool_call_id)
            self._tool_call_responses[event.tool_call_id] = response_id
        if event_type == 'response.created':
            created = RESPONSE_CREATED_EVENT_ADAPTER.validate_python(data)
            # The request this connection takes the response to be answering: its own, sent and not yet started.
            outstanding = (
                (client_event_id('response', self._response_request_inputs), self._response_request_answers)
                if self._response_active and not self._response_started and self._response_request_inputs
                else None
            )
            after.extend(self._lifecycle.response_created(created.response, outstanding=outstanding))
            if outstanding is not None and self._lifecycle.request_outstanding(outstanding[0]):
                # Not ours: one the provider started on its own while ours was on its way, which it refuses or
                # (xAI) starts once this one is cancelled.
                self._request_in_flight = (self._response_request_inputs, self._response_request_answers)
                self._in_flight_cut_off = False
            elif (in_flight := self._request_in_flight) is not None and not self._lifecycle.request_outstanding(
                client_event_id('response', in_flight[0])
            ):
                # Ours after all, started once the provider cancelled its own.
                self._request_in_flight = None
                self._in_flight_cut_off = False
                self._active_response_refusals = 0
            elif outstanding is not None:
                self._active_response_refusals = 0
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
                self._track_output_item(event.item_id, content_index, self._generated_audio_bytes)
        if event is not None and not (event_type == 'response.done' and superseded):
            if not (
                isinstance(event, InputTranscript | RealtimeInputTranscriptionErrorEvent)
                and event.item_id in self._idle_timeout_items
            ):
                events.append(event)
            elif isinstance(event, RealtimeInputTranscriptionErrorEvent) or event.is_final:
                # The item's transcription is over, whether it succeeded or failed.
                self._idle_timeout_items.discard(event.item_id)
            if isinstance(event, InputTranscript) and event.is_final and event_type in INPUT_TRANSCRIPT_DONE_TYPES:
                # The transcript is already recorded, so a malformed `usage` payload costs the usage
                # event, not the user's words: report it as the same recoverable frame error `frames`
                # would have raised rather than discarding the whole frame along with it.
                try:
                    usage = _validate_transcription_usage(data.get('usage'))
                except ValueError as e:
                    events.append(_frame_error(e))
                else:
                    if (asr := _map_transcription_usage(usage)) is not None:
                        events.append(SessionUsage(usage=asr, response_scoped=False))
        return _DecodedFrame(
            events, before=self._lifecycle.take_leading(), after=[*after, *self._take_pending_lifecycle()], stale=stale
        )

    async def _handle_error(self, data: dict[str, Any]) -> list[InputRejected] | None:
        """Report the inputs an `error` frame refused, and release a refused request for a response.

        The refused request can be the one this connection is waiting on to start, the only thing that
        would ever have cleared `_response_active`: left set, every later request would queue behind a
        response that never comes. Requests deferred behind it are sent now instead.

        A request refused because a response the provider started on its own was under way is asked for again
        once that one is over, up to `ACTIVE_RESPONSE_RETRIES` times in a row: `None` says the error is handled so,
        and reported to nobody.
        """
        error = RealtimeErrorEvent.model_validate(data).error
        rejected = rejected_inputs(error)
        if error.code == ACTIVE_RESPONSE_ERROR_CODE and await self._ask_again(error.event_id):
            return None
        if (in_flight := self._request_in_flight) is not None and error.event_id == client_event_id(
            'response', in_flight[0]
        ):
            # Refused for good: nothing asks for it again, not a reconnect either.
            self._request_in_flight = None
            self._in_flight_cut_off = False
        if (
            self._response_active
            and not self._response_started
            and self._response_request_inputs
            and error.event_id == client_event_id('response', self._response_request_inputs)
        ):
            self._clear_active_response()
            if self._pending_response:
                await self._create_response(*self._take_deferred_response_inputs())
        return rejected

    async def _ask_again(self, event_id: str | None) -> bool:
        """Ask again for a request of ours refused while another response was under way; `False` to give up on it.

        Asked once the response under way is over, which may be already: the refusal can come after its end.
        """
        in_flight = self._request_in_flight
        if in_flight is not None and event_id == client_event_id('response', in_flight[0]):
            # Refused as the response that overtook it was under way.
            self._request_in_flight = None
            if self._in_flight_cut_off:
                # The user barged in on that response: their speech takes the turn.
                self._in_flight_cut_off = False
                self._lifecycle.requests_dropped(in_flight[1])
                return True
            request = in_flight
        elif (
            self._response_active
            and not self._response_started
            and self._response_request_inputs
            and event_id == client_event_id('response', self._response_request_inputs)
        ):
            # Refused for a response of the provider's own that isn't reported yet: it is, next, and this one
            # waits behind it.
            request = (self._response_request_inputs, self._response_request_answers)
        else:
            return False
        self._active_response_refusals += 1
        if self._active_response_refusals > ACTIVE_RESPONSE_RETRIES:
            # The provider kept starting responses of its own: give up on this one, as on any other refusal.
            self._active_response_refusals = 0
            self._lifecycle.requests_refused(request[1])
            return False
        if request is not in_flight:
            self._response_request_inputs = self._response_request_answers = ()
        await self._request_response(request[0], answers=request[1])
        return True

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
            # frame raises it again, and `frames` reports it as a recoverable frame error.
            # A `response.done` with no usable `response` object is malformed, but it is still the only
            # terminal we will ever get for the response it was meant to close. Treating it as "no
            # information" leaves `_response_active` set forever, and every later `create_response()`
            # then queues behind a response that can never complete.
            self._lifecycle.response_unreadable(self._active_response_id)
            self._clear_active_response()
            if self._pending_response:
                inputs, answers = self._take_deferred_response_inputs()
                await self._request_response(inputs, answers=answers)
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
        status: ResponseStatus = (
            response.status if response.status in ('cancelled', 'failed', 'incomplete') else 'completed'
        )
        self._lifecycle.response_done(
            response_id or None,
            status=status,
            finish_reason=finish_reason,
            provider_details=response_provider_details(response),
        )
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
            deferred_inputs, deferred_answers = self._take_deferred_response_inputs()
            # A *server*-cancelled response means the user barged in: a new turn is starting, so don't
            # replay the deferred response over it. After a *client* cancel (`interrupt()`), the caller
            # explicitly queued the next response behind the cancel, so send it now that the cancelled
            # response has closed.
            if response.status != 'cancelled' or was_client_cancel:
                await self._create_response(deferred_inputs, deferred_answers)
            else:
                # The user's barge-in takes the turn: no response is requested for them.
                self._lifecycle.requests_dropped(deferred_answers)
        if matches_active_response and response_status_reason(response) == 'turn_detected':
            # The user barged in on it (server VAD's reason): likewise for the request that response overtook,
            # whether the provider refuses it on that response's account or the connection drops first. (xAI cancels
            # a response of its own for ours with another reason, and starts ours.)
            self._in_flight_cut_off = self._request_in_flight is not None
        if isinstance(response_id, str) and (batch := self._tool_call_batches.get(response_id)) is not None:
            # No more calls can join the response: its tool results are answered once all are in.
            batch.done = True
            if response.status == 'cancelled' and not was_client_cancel and not batch.unanswered:
                # As for a deferred request above: the user barged in, and the response their speech
                # starts answers the results already sent, rather than one talking over them.
                del self._tool_call_batches[response_id]
                self._lifecycle.requests_dropped(batch.inputs)
            else:
                await self._answer_tool_call_batch_if_complete(response_id)
        # Validated only now that all the response state above is settled: a malformed usage payload
        # raises `ValueError`, which `frames` surfaces as a recoverable frame error and keeps reading
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
        # The response's `provider_details` ride along too: a response that called a tool is recorded
        # from this usage rather than from its `ResponseDone` (suppressed for a function-call-only
        # response, and arriving after the response is already recorded otherwise), so without them a
        # tool-call response would lack the `status` every other response carries. Not for a superseded
        # response, though: the session may be recording the newer one when this usage lands, and the
        # older response's status (typically `cancelled`) doesn't describe it.
        provider_details = None if superseded else response_provider_details(response)
        if usage is not None:
            events.append(
                SessionUsage(
                    usage=usage,
                    provider_response_id=response_id or None,
                    finish_reason=finish_reason,
                    provider_details=provider_details,
                )
            )
        elif matches_active_response and finish_reason == 'tool_call':
            events.append(
                SessionUsage(
                    usage=RequestUsage(),
                    provider_response_id=response_id or None,
                    finish_reason='tool_call',
                    provider_details=provider_details,
                )
            )
        # Reported even for a superseded response: its failure is real, and it's the only report of it.
        if (error := response_failed_error(response)) is not None:
            events.append(error)
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
        # What the old socket hadn't acknowledged it never will; the new one acknowledges only its own sends.
        self._lifecycle.socket_replaced()
        # The new conversation's items have ids of their own, which its replay reports as it adds them.
        self._tool_items.clear()
        try:
            self._ws = await self._dial()
            # A response the socket dropped before it could complete will never be released by its
            # `response.done`, so re-ask for it on the fresh socket (which has just replayed the call)
            # rather than leaving the session waiting for a turn that can never start. Two shapes qualify:
            # one deferred behind a now-dead active response (`_pending_response`), and one solicited but
            # not yet confirmed by its `response.created` (`_response_active` without `_response_started`)
            # — the answer the caller is waiting on, which `_pending_response` alone does not cover. A
            # response already streaming when it dropped is *not* re-asked: its partial reply is settled as interrupted,
            # matching the state-lost contract of staying quiet until the next input. Nor is one the
            # caller cancelled (`_cancel_sent`) before it started: re-asking would resurrect a response a
            # barge-in explicitly stopped. Sent inside this guard because the new socket can drop before
            # the frame reaches it, which has to consume an attempt like any other failed reconnect.
            if (in_flight := self._request_in_flight) is not None:
                # Neither refused nor started on the old socket: asked for again like one deferred, unless the user
                # barged in on the response that overtook it (then their speech took the turn, and it is lost).
                self._request_in_flight = None
                if not self._in_flight_cut_off:
                    self._pending_response = True
                    self._deferred_response_inputs[:0] = in_flight[0]
                    self._deferred_response_answers[:0] = in_flight[1]
                self._in_flight_cut_off = False
            replay_response = self._pending_response or (
                self._response_active and not self._response_started and not self._cancel_sent
            )
            replay_inputs, replay_answers = (
                (tuple(self._deferred_response_inputs), tuple(self._deferred_response_answers))
                if self._pending_response
                else (self._response_request_inputs, self._response_request_answers)
            )
            # The requests the new socket will never answer: an unstarted request not asked for again (only
            # the deferred ones are, when there are any, and none the caller cancelled), and the tool calls
            # whose batches go below.
            lost_inputs: list[InputId] = []
            if self._response_active and not self._response_started:
                lost_inputs += self._response_request_answers
            lost_inputs += [input_ for batch in self._tool_call_batches.values() for input_ in batch.inputs]
            replayed = set(replay_answers) if replay_response else set[InputId]()
            lost_inputs = [input_ for input_ in dict.fromkeys(lost_inputs) if input_ not in replayed]
            self._clear_active_response()
            # A fresh socket also drops anything the old one was still holding for us, including the
            # output items a barge-in could have named. The calls in flight are gone too: the session settles
            # them rather than sending their results, so their batches go.
            self._cancelled_response_id = None
            self._output_items.clear()
            self._tool_call_batches.clear()
            self._tool_call_responses.clear()
            self._lifecycle.reconnected(lost_inputs=lost_inputs, asked_again=tuple(replayed))
            if replay_response:
                await self._create_response(replay_inputs, replay_answers)
            # Cleared only once the replay is on the wire, so a send that failed above leaves the
            # request queued for the next attempt instead of losing it.
            self._take_deferred_response_inputs()
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
        self._response_request_inputs = ()
        self._response_request_answers = ()
        self._cancel_sent = False
        # On a sideband, the provider keeps playing this response's audio to the browser after
        # `response.done`, and a barge-in truncation during that tail still has to name the playing
        # item — so it is retired when playback ends (`output_audio_buffer.stopped`/`.cleared`) instead.
        # Over a WebSocket the finished item stays reachable only by name, through `_output_items`: an
        # unnamed truncate means "the item being generated", and a finished reply may have been heard in full.
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
    # What `input_transcription_model='auto'` resolves to (see `resolve_transcription_model`): the model
    # OpenAI's realtime transcription guide says to start with, which streams transcript deltas as speech
    # arrives. Behind the sentinel so it can follow OpenAI's recommendation; pin an id to keep one.
    _auto_transcription_model: ClassVar[str] = 'gpt-live-transcribe'

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
        super().__init__(settings=settings, profile=profile)
        self.model = model
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
            model_settings.get('input_transcription_model', 'auto'), default=self._auto_transcription_model
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
        if (thinking := model_settings.get('thinking')) is not None and self.profile.get('supports_thinking', False):
            # `False` maps to `'none'`: the SDK's `RealtimeReasoningEffort` doesn't list it, but every
            # `gpt-realtime-2*` model accepts it and then reasons with zero tokens, whereas omitting
            # `reasoning` leaves the model reasoning at its default effort.
            config['reasoning'] = {'effort': OPENAI_REASONING_EFFORT_MAP[thinking]}
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
        message_history: Sequence[ModelMessage] | None = None,
    ) -> WebRTCAnswer:
        # The Realtime API's calls endpoint takes no conversation items (`session.input` is rejected), so
        # history is seeded by the sideband when it attaches, as `connect_webrtc` does.
        del message_history
        return await _answer_webrtc_offer(
            http_client=self._http_client,
            calls_url=self._webrtc_calls_url(),
            headers=await self._webrtc_headers(),
            provider_name=self.system,
            model_name=self.model_name,
            sdp_offer=sdp_offer,
            session_config=self._webrtc_session_config(instructions, tools, model_settings),
        )

    def _check_hang_up(self, session: RealtimeProviderSession) -> None:
        self._check_webrtc_session_provider(session)

    async def hang_up(self, session: RealtimeProviderSession) -> None:
        self._check_hang_up(session)
        client = self.client.with_options(timeout=HANG_UP_TIMEOUT, max_retries=HANG_UP_MAX_RETRIES)
        with ignore_ended_call('call_id_not_found'), map_openai_api_errors(self.model_name):
            await client.realtime.calls.hangup(session.session_id)

    @asynccontextmanager
    async def connect_webrtc(
        self,
        session: RealtimeProviderSession,
        *,
        messages: Sequence[ModelMessage],
        model_settings: RealtimeModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncGenerator[OpenAIRealtimeConnection]:
        self._check_webrtc_session_provider(session)
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
            server_model = _HandshakeSessionEvent.model_validate(updated).session.model
            return ws

        try:
            with map_connect_errors(self.model):
                ws = await dial()
                for item in seed:
                    await ws.send(to_json({'type': CONVERSATION_ITEM_CREATE_EVENT, 'item': item}).decode())
            connection = self._connection_type(
                ws,
                dial=dial,
                reconnect=settings.get('reconnect'),
                input_transcription_enabled=transcription_enabled,
                model_name=server_model,
                model_name_getter=lambda: server_model,
                observes_output_audio=False,
                prune_tool_items=settings.get('openai_prune_tool_items', False),
                keep_tool_items=settings.get('openai_keep_tool_items', frozenset()),
                manual_turns=config_has_manual_turns(session_config),
            )
            connection._history_items_sent(seed)  # pyright: ignore[reportPrivateUsage]
            yield connection
        finally:
            if cm is not None:  # pragma: no branch
                await cm.__aexit__(None, None, None)

    async def _auth_headers(self, model_settings: OpenAIRealtimeModelSettings | None = None) -> dict[str, str]:
        # `model_settings` lets a provider vary auth by session (e.g. Azure Voice Live uses a different
        # resource key); OpenAI's auth doesn't depend on it.
        del model_settings
        return await openai_websocket_auth_headers(self._provider.client)

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

        Settings-aware because a provider's handshake shape can vary by *session*, as Azure's does for
        Voice Live.
        """
        del model_settings
        return _HandshakeSessionEvent.model_validate(created).session.model

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
            return self._build_connection(
                ws,
                dial=dial,
                settings=settings,
                session_config=session_config,
                server_model=server_model,
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
        ) as connection:
            async with connection._session_scope():  # pyright: ignore[reportPrivateUsage]
                yield connection

    def _build_connection(
        self,
        ws: ClientConnection,
        *,
        dial: Callable[[], Awaitable[ClientConnection]],
        settings: OpenAIRealtimeModelSettings,
        session_config: dict[str, Any],
        server_model: str | None,
        model_name_getter: Callable[[], str | None],
    ) -> OpenAIRealtimeConnection:
        """Build the connection for a WebSocket session, once its handshake is done.

        A protocol clone overrides it to pass its own connection's arguments; see also `_connection_class`.
        """
        return self._connection_class(settings)(
            ws,
            dial=dial,
            reconnect=settings.get('reconnect'),
            input_transcription_enabled=settings.get('input_transcription_model', 'auto') is not None,
            interrupts_response_on_speech=config_interrupts_response_on_speech(session_config),
            model_name=server_model,
            model_name_getter=model_name_getter,
            audio_output_sample_rate=self.audio_output_sample_rate,
            prune_tool_items=settings.get('openai_prune_tool_items', False),
            keep_tool_items=settings.get('openai_keep_tool_items', frozenset()),
            manual_turns=config_has_manual_turns(session_config),
        )
