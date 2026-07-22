"""A realtime session that wraps a [`RealtimeConnection`][pydantic_ai.realtime.RealtimeConnection] with automatic tool execution."""

from __future__ import annotations as _annotations

import asyncio
import dataclasses
import io
import wave
from collections import deque
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import replace
from threading import Lock as ThreadLock
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

import pydantic_core
from anyio import Lock
from opentelemetry import context as otel_context
from opentelemetry.context import Context
from opentelemetry.trace import Span, SpanKind, StatusCode, set_span_in_context
from typing_extensions import assert_never

from .._enqueue import PendingMessage, PendingMessagePriority
from .._instrumentation import (
    model_metric_attributes,
    provider_attributes,
    response_attributes,
    response_price_calculation,
    safe_to_json,
    serialize_any,
)
from .._tool_execution import build_tool_return_part
from .._utils import cancel_and_drain
from ..exceptions import ApprovalRequired, CallDeferred, ToolRetryError, UsageLimitExceeded, UserError
from ..messages import (
    BinaryContent,
    DeferredToolRequestsEvent,
    DeferredToolResultsEvent,
    FinishReason,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RetryPromptPart,
    SpeechPart,
    SpeechPartDelta,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
)
from ..native_tools import SUPPORTED_NATIVE_TOOLS
from ..tool_manager import ToolManager
from ..usage import RequestUsage, RunUsage, UsageLimits
from ._base import (
    AudioDelta,
    AudioInput,
    AudioRetention,
    CancelResponse,
    ClearAudio,
    CommitAudio,
    ConversationCreated,
    ConversationItemCreated,
    CreateResponse,
    ImageInput,
    InputSpeechEndEvent,
    InputSpeechStartEvent,
    InputTranscript,
    InputTranscriptionFailedEvent,
    RealtimeCodecEvent,
    RealtimeConnection,
    RealtimeError,
    RealtimeEvent,
    RealtimeModelProfile,
    RealtimeSessionInput,
    ReconnectedEvent,
    SessionErrorEvent,
    SessionUsageEvent,
    TextInput,
    ToolCall,
    ToolCallCancelled,
    ToolResult,
    Transcript,
    TruncateOutput,
    TurnCompleteEvent,
    seed_pcm_audio,
)

if TYPE_CHECKING:
    from ..models.instrumented import InstrumentationSettings
    from ..tools import DeferredToolRequests, DeferredToolResults

# Realtime providers stream raw PCM audio, but retained history uses a WAV container so the sample
# format is self-describing and portable to classic model adapters. Live `SpeechPartDelta.audio_chunk`
# values remain raw PCM.
_WAV_MEDIA_TYPE = 'audio/wav'
# Recorded as the result of a tool call the model cancelled mid-flight (see `ToolCallCancelled`), so the
# call still has a matching return in history.
_CANCELLED_TOOL_RESULT = 'Tool call cancelled before it completed.'

# Fallback for a session created without a model's profile (e.g. directly, in tests): assume
# everything is supported so no guard fires. Real sessions receive `model.profile`. Native tools are
# validated up front by `Agent.realtime_session`, not the session, so this field is inert here.
_FULL_PROFILE = RealtimeModelProfile(
    supports_image_input=True,
    supports_manual_turn_control=True,
    supports_interruption=True,
    supports_output_truncation=True,
    supports_session_seeding=True,
    supported_native_tools=SUPPORTED_NATIVE_TOOLS,
    audio_input_sample_rate=24000,
    audio_output_sample_rate=24000,
)

# The `RealtimeEvent` variants that `_translate_event` handles: the full union minus `ToolCall` and
# `SessionUsageEvent`, which `_handle_pump_event` peels off first (they drive tool execution and usage
# accounting before delegating). Splitting the union lets `_translate_event` end in `assert_never`, so
# a new non-pump variant added to `RealtimeEvent` is caught at type-check time — either the call site
# (where the residual no longer fits this alias) or the `assert_never` flags it.
_TranslatableEvent: TypeAlias = (
    AudioDelta
    | Transcript
    | InputTranscript
    | TurnCompleteEvent
    | InputSpeechStartEvent
    | InputSpeechEndEvent
    | InputTranscriptionFailedEvent
    | ReconnectedEvent
    | PartStartEvent
    | PartEndEvent
    | SessionErrorEvent
)
_SettledToolResult: TypeAlias = tuple[ToolReturnPart | RetryPromptPart, str | Sequence[UserContent] | None]


def _as_event(item: object) -> RealtimeEvent:
    """Unwrap a queue item: re-raise a tool's exception, otherwise return the event."""
    if isinstance(item, Exception):
        raise item
    return cast('RealtimeEvent', item)


def _pcm_to_wav(data: bytes, sample_rate: int) -> bytes:
    """Wrap mono 16-bit PCM bytes in a WAV container at `sample_rate`."""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(data)
    return buffer.getvalue()


def _accumulate_transcript(accumulated: str, text: str) -> tuple[str, str]:
    """Fold a transcript event's `text` into the running transcript, returning `(new_accumulated, appended)`.

    Providers deliver transcripts two different ways: as incremental deltas (each event carries a new
    piece) or as a single final event carrying the full text. Both are handled by one rule: if `text`
    extends what we already have (the accumulated transcript is a prefix of it), it is a cumulative/full
    update and only the new suffix is appended; otherwise `text` is an incremental piece appended as-is.
    The second element is the newly appended text (empty when a final event merely repeats the deltas),
    suitable for a [`PartDeltaEvent`][pydantic_ai.messages.PartDeltaEvent].

    A cumulative/final snapshot can differ from the accumulated deltas by leading/trailing whitespace —
    OpenAI's input-audio-transcription deltas start with a leading space that the `.completed` snapshot
    trims — so the prefix check is applied to the stripped text too, adopting the snapshot as
    authoritative rather than concatenating a near-duplicate.
    """
    if accumulated and text.startswith(accumulated):
        return text, text[len(accumulated) :]
    stripped = accumulated.strip()
    if stripped and (stripped_text := text.strip()).startswith(stripped):
        return text, stripped_text[len(stripped) :]
    return accumulated + text, text


def _parse_tool_args(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse a tool call's raw JSON arguments.

    Returns `(args, None)` on success, or `(None, error_message)` when the payload is not a JSON
    object, so the caller can report the error back to the model rather than guessing.
    """
    if not raw:
        return {}, None
    try:
        parsed = pydantic_core.from_json(raw)
    except ValueError as e:
        return None, f'Error: could not parse tool arguments as JSON: {e}'
    if not isinstance(parsed, dict):
        return None, f'Error: expected tool arguments to be a JSON object, got {type(parsed).__name__}'
    return cast('dict[str, Any]', parsed), None


def _is_tool_result_request(message: ModelMessage) -> bool:
    """Whether a history request carries an inserted tool result and optional follow-up user content."""
    if not isinstance(message, ModelRequest) or not message.parts:
        return False
    return isinstance(message.parts[0], (ToolReturnPart, RetryPromptPart)) and all(
        isinstance(part, (ToolReturnPart, RetryPromptPart, UserPromptPart)) for part in message.parts
    )


def _pending_message_text(pending: PendingMessage) -> str:
    """Return the text a realtime session can deliver, or reject unsupported enqueue content."""
    if len(pending.messages) == 1 and isinstance(message := pending.messages[0], ModelRequest):
        if len(message.parts) == 1 and isinstance(part := message.parts[0], UserPromptPart):
            if isinstance(part.content, str):
                return part.content
    raise UserError(
        '`RunContext.enqueue()` in a realtime session currently supports one plain-text prompt per call. '
        'Multimodal content and prebuilt message or part sequences cannot be delivered over the live input channel.'
    )


class _RealtimePendingMessages(list[PendingMessage]):
    """A `RunContext.enqueue` queue that validates content and wakes the live session for `asap` delivery."""

    def __init__(self) -> None:
        super().__init__()
        self._on_asap: Callable[[], None] | None = None
        self._lock = ThreadLock()

    def bind(self, on_asap: Callable[[], None]) -> None:
        self._on_asap = on_asap

    def append(self, pending: PendingMessage) -> None:
        _pending_message_text(pending)
        with self._lock:
            super().append(pending)
        if pending.priority == 'asap' and self._on_asap is not None:
            self._on_asap()

    def pop_priority(self, priority: PendingMessagePriority) -> list[PendingMessage]:
        """Atomically remove and return all messages with `priority`."""
        with self._lock:
            selected = [pending for pending in self if pending.priority == priority]
            self[:] = [pending for pending in self if pending.priority != priority]
        return selected


class RealtimeSession:
    """Wraps a [`RealtimeConnection`][pydantic_ai.realtime.RealtimeConnection], building message history and auto-executing tools.

    The session translates the connection's low-level codec events into the shared message/part event
    vocabulary from [`pydantic_ai.messages`][pydantic_ai.messages] and accumulates ordinary
    [`ModelMessage`][pydantic_ai.messages.ModelMessage] history as the conversation proceeds, so a
    session can hand off to [`Agent.run`][pydantic_ai.agent.AbstractAgent.run] via
    [`all_messages`][pydantic_ai.realtime.RealtimeSession.all_messages]:

    - assistant speech becomes [`PartStartEvent`][pydantic_ai.messages.PartStartEvent] /
      [`PartDeltaEvent`][pydantic_ai.messages.PartDeltaEvent] / [`PartEndEvent`][pydantic_ai.messages.PartEndEvent]
      events carrying a [`SpeechPart`][pydantic_ai.messages.SpeechPart]
      (`speaker='assistant'`), finalized into a [`ModelResponse`][pydantic_ai.messages.ModelResponse]
      at the end of the turn;
    - user speech becomes the same part events with `speaker='user'`, finalized into a
      [`ModelRequest`][pydantic_ai.messages.ModelRequest];
    - a tool call becomes a [`ToolCallPart`][pydantic_ai.messages.ToolCallPart] (start/end) plus a
      [`FunctionToolCallEvent`][pydantic_ai.messages.FunctionToolCallEvent] when execution starts and a
      [`FunctionToolResultEvent`][pydantic_ai.messages.FunctionToolResultEvent] carrying a normalized
      [`ToolReturnPart`][pydantic_ai.messages.ToolReturnPart] or
      [`RetryPromptPart`][pydantic_ai.messages.RetryPromptPart] when it settles.

    Tools always run concurrently with the session. The session keeps streaming events while a tool
    runs, so the model can keep speaking and user speech keeps being processed, then sends the result
    back over the connection once it is ready. This mirrors how a person can keep talking while work
    happens.

    Tool outcomes use the same normalized history shapes as a classic agent run: retries become
    [`RetryPromptPart`][pydantic_ai.messages.RetryPromptPart]s, denials retain their `outcome`, and
    structured returns preserve `content`, `metadata`, and typed `tool_kind` identity. Realtime
    tool-output channels are string-only, so the structured part is rendered only when it is sent.
    OpenAI-protocol connections send additional user content as a follow-up conversation item; Gemini
    includes a text fallback in its tool response. If the provider cancels an in-flight call, the
    session records a synthetic interrupted return for valid history but does not send that abandoned
    result to the provider.

    History is accumulated in the order events are reported. Provider item IDs keep interleaved input
    transcripts associated with their correct user turns; providers without item IDs retain
    arrival-order association. Tool results are the exception: a tool's
    [`FunctionToolResultEvent`][pydantic_ai.messages.FunctionToolResultEvent] streams whenever the tool
    finishes (possibly after later turns), but in [`all_messages()`][pydantic_ai.realtime.RealtimeSession.all_messages]
    its result part is placed directly after the response carrying its call — request-response APIs
    require that adjacency, so the history stays valid for a handoff to a standard
    [`Agent.run`][pydantic_ai.agent.AbstractAgent.run].

    Images and video frames streamed with [`send`][pydantic_ai.realtime.RealtimeSession.send] are
    stored as ordinary user image turns by default. Set `retain_images_every_n` above `1` to sample
    high-rate frame streams and reduce history memory use.

    When constructing a session directly, use it as an async context manager. The context owns the
    receive pump, background tool tasks, and instrumentation spans; iteration only reads its event
    queue. [`Agent.realtime_session`][pydantic_ai.agent.Agent.realtime_session] enters the session
    before yielding it, so the usual agent API remains a single `async with` block.
    """

    def __init__(
        self,
        connection: RealtimeConnection,
        tool_manager: ToolManager[Any],
        *,
        instrumentation: InstrumentationSettings | None = None,
        model_name: str | None = None,
        provider_name: str | None = None,
        provider_url: str | None = None,
        agent_name: str | None = None,
        usage: RunUsage | None = None,
        usage_limits: UsageLimits | None = None,
        audio_retention: AudioRetention = 'transcript_only',
        retain_images_every_n: int = 1,
        message_history: Sequence[ModelMessage] | None = None,
        profile: RealtimeModelProfile | None = None,
        conversation_id: str | None = None,
        instructions: str | None = None,
        metadata: dict[str, Any] | None = None,
        agent_description: str | None = None,
        output_modality: Literal['audio', 'text'] = 'audio',
    ) -> None:
        self._connection = connection
        self._tool_manager = tool_manager
        self._tool_run_step = 0
        self._tool_manager_lock = Lock()
        self._instrumentation = instrumentation
        self._profile = profile if profile is not None else _FULL_PROFILE
        self._model_name = model_name
        self._provider_name = provider_name
        self._provider_url = provider_url
        self._agent_name = agent_name
        self._conversation_id = conversation_id
        self._instructions = instructions
        self._metadata = metadata
        self._agent_description = agent_description
        # The semconv `gen_ai.output.type` value for the session's configured output modality:
        # `'speech'` for spoken audio (the enum's term for voice output), `'text'` for text-only.
        self._otel_output_type = 'speech' if output_modality == 'audio' else 'text'
        self._usage_limits = usage_limits
        self._audio_retention = audio_retention
        self._retain_input = audio_retention in ('input_audio', 'all')
        self._retain_output = audio_retention in ('output_audio', 'all')
        if retain_images_every_n < 1:
            raise UserError('`retain_images_every_n` must be at least 1.')
        self._retain_images_every_n = retain_images_every_n
        self._sent_image_count = 0
        # Whether the connection transcribes the user's audio. When it doesn't, no `InputTranscript`
        # arrives to finalize a user turn, so retained input audio is finalized as an audio-only turn
        # instead (see `_finalize_audio_only_user`).
        self._input_transcription_enabled = connection.input_transcription_enabled
        if not self._input_transcription_enabled and not self._retain_input:
            # Neither transcription nor input-audio retention: the user's turns would be silently dropped
            # from history (nothing to build a user turn from), breaking the session's history/handoff
            # contract. Make the contradictory config an explicit error rather than a silent gap.
            raise UserError(
                "This realtime session can't capture the user's turns: input transcription is disabled "
                "and `audio_retention` doesn't retain input audio. Enable transcription in the model settings, "
                "or pass `audio_retention='input_audio'` or `'both'` to keep the "
                'raw audio instead.'
            )
        self.usage = usage if usage is not None else RunUsage()
        """Cumulative token usage and tool-call counts for the session, updated as events stream in.

        Pass `usage` to [`Agent.realtime_session`][pydantic_ai.agent.Agent.realtime_session] to accumulate
        into a shared [`RunUsage`][pydantic_ai.usage.RunUsage]; otherwise a fresh one is used.
        """

        # History: `_seeded` is the conversation the session was opened with (surfaced by
        # `all_messages` only); `_history` is what happened during this session (surfaced by both).
        self._seeded: list[ModelMessage] = list(message_history or [])
        self._history: list[ModelMessage] = []
        self._replayed_item_ids: set[str] = set()
        self._replayed_tool_call_ids: set[str] = set()

        # In-flight assistant response being assembled. Parts finalize into `_response_parts`, which
        # becomes a `ModelResponse` at the turn boundary (or when a tool call splits the turn).
        self._response_parts: list[ModelResponsePart] = []
        # Native tool parts reconstructed from a turn's provider metadata (web grounding / code
        # execution), buffered as they arrive mid-turn and prepended to the response at finalization so
        # history reads native-tool-activity-then-speech, mirroring the classic `GoogleModel` order.
        self._native_tool_parts: list[ModelResponsePart] = []
        # The `chat {model}` span for the response currently being assembled (see `_ensure_chat_span`).
        self._chat_span: Span | None = None
        self._pending_response_usage = RequestUsage()
        self._pending_provider_response_id: str | None = None
        self._pending_finish_reason: FinishReason | None = None
        self._response_finalized_before_terminal = False
        # User requests sent while a response is in flight are held until that response is finalized,
        # so the pump remains the sole writer for that portion of history and a caller cannot splice a
        # request between an assistant response's streamed parts.
        self._pending_sent_requests: list[ModelRequest] = []
        self._active_assistant: SpeechPart | TextPart | None = None
        self._active_assistant_index = 0
        self._assistant_transcript = ''
        self._output_audio = bytearray()

        # In-flight user request being assembled from input-transcript events.
        self._active_user: SpeechPart | None = None
        self._user_transcript = ''
        self._active_users_by_id: dict[str, SpeechPart] = {}
        self._user_transcripts_by_id: dict[str, str] = {}
        self._user_item_order: deque[str] = deque()
        self._finalized_users_by_id: dict[str, SpeechPart] = {}
        self._finalized_user_item_ids: set[str] = set()
        # Retained input audio (`audio_retention='input_audio'`/`'both'`). `_input_audio` is the rolling buffer
        # of audio sent since the last turn boundary; on providers that report a per-item speech-stopped
        # boundary, each segment is cut into `_input_audio_by_id` keyed by its input item id, so overlapping
        # turns whose transcripts finalize out of order still attach their own audio (not a later turn's).
        self._input_audio = bytearray()
        self._input_audio_by_id: dict[str, bytes] = {}

        # The session context is the single owner of the receive pump and background tool tasks.
        # Iteration starts the pump lazily, but never tears it down: an early `break` can abandon the
        # reader generator without affecting resource lifetime, and `__aexit__` still drains everything
        # before the connection and toolset close.
        self._queue: asyncio.Queue[RealtimeEvent | object] = asyncio.Queue()
        self._queue_changed = object()
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._pending_messages = _RealtimePendingMessages()
        self._pending_messages_lock = Lock()
        if self._tool_manager.ctx is not None:
            self._tool_manager.ctx.pending_messages = self._pending_messages
        # In-flight tool tasks keyed by tool call id, so a `ToolCallCancelled` can cancel the specific
        # calls the model abandoned (e.g. on barge-in) without touching the others.
        self._pending_tool_calls: dict[str, tuple[asyncio.Task[None], ToolCallPart]] = {}
        # OpenAI-protocol tool results can complete before the response's later `response.done` usage
        # finalizes the calling response. Hold their history requests until the call is present.
        self._pending_tool_returns: list[tuple[ToolCallPart, ModelRequest]] = []
        self._tool_calls_awaiting_usage: set[str] = set()
        self._asap_drain_deferred = False
        self._asap_drain_ready = False
        self._pump_task: asyncio.Task[None] | None = None
        self._pump_error: Exception | None = None
        self._pump_finished = False
        self._iterator_active = False
        self._stream_exhausted = False
        self._entered = False
        self._closed = False
        self._loop: asyncio.AbstractEventLoop | None = None

        # The session span is deliberately not made current in the owner's task. Child spans receive
        # this explicit context directly, or through the pump task's same-task attach/detach pair.
        self._session_span: Span | None = None
        self._session_span_context: Context | None = None
        self._session_span_attributes: dict[str, Any] | None = None

    async def __aenter__(self) -> RealtimeSession:
        if self._entered or self._closed:
            raise RuntimeError('This realtime session cannot be entered more than once.')
        self._entered = True
        self._loop = asyncio.get_running_loop()
        self._pending_messages.bind(self._notify_asap_pending_messages)

        settings = self._instrumentation
        if settings is not None:
            # The session is the realtime analog of an agent run, and the semconv operation-name
            # enum has no realtime/speech value (nor do other voice frameworks emit one), so use
            # `invoke_agent` like the classic agent-run span — backends render the session as an
            # agent invocation — with `gen_ai.output.type` (`speech`/`text`) capturing the modality.
            attributes: dict[str, Any] = {
                'gen_ai.operation.name': 'invoke_agent',
                'gen_ai.output.type': self._otel_output_type,
            }
            if self._model_name:
                attributes['gen_ai.request.model'] = self._model_name
            if self._agent_name:
                attributes['gen_ai.agent.name'] = self._agent_name
            if self._agent_description:
                attributes['gen_ai.agent.description'] = self._agent_description
            if self._conversation_id:
                # Match the classic agent-run span's key (see `capabilities/instrumentation.py`) so a
                # realtime session can be correlated with other runs sharing the conversation id.
                attributes['gen_ai.conversation.id'] = self._conversation_id
            span_name = f'realtime {self._model_name}' if self._model_name else 'realtime'
            parent_context = otel_context.get_current()
            span = settings.tracer.start_span(
                span_name,
                context=parent_context,
                attributes=attributes,
                kind=SpanKind.CLIENT,
            )
            self._session_span = span
            self._session_span_context = set_span_in_context(span, parent_context)
            self._session_span_attributes = attributes

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if not self._entered or self._closed:
            return
        self._closed = True

        tasks = [*self._background_tasks]
        if self._pump_task is not None:
            tasks.append(self._pump_task)
        if tasks:
            await cancel_and_drain(*tasks, msg='Realtime session exited')

        self._flush_pending_users()

        error = exc_value or self._pump_error
        if self._chat_span is not None:
            if error is not None:
                self._record_span_error(self._chat_span, error)
            self._chat_span.end()
            self._chat_span = None

        settings = self._instrumentation
        span = self._session_span
        attributes = self._session_span_attributes
        if settings is not None and span is not None and attributes is not None:
            if error is not None:
                self._record_span_error(span, error)
            self._finalize_span(settings, span)
            span.end()
        self._session_span = None
        self._session_span_context = None
        self._session_span_attributes = None
        self._loop = None

    @staticmethod
    def _record_span_error(span: Span, error: BaseException) -> None:
        if span.is_recording():
            span.record_exception(error, escaped=True)
            span.set_status(StatusCode.ERROR)

    def all_messages(self) -> list[ModelMessage]:
        """A snapshot of the seeded history plus messages recorded during this session.

        Returns a copy, so the result doesn't change as the session continues. Feed it into
        [`Agent.run(message_history=...)`][pydantic_ai.agent.AbstractAgent.run] to hand the
        conversation off to a standard agent run. Images streamed with `send()` are recorded according
        to `retain_images_every_n`.
        """
        return [*self._seeded, *self._history]

    def new_messages(self) -> list[ModelMessage]:
        """A snapshot of the messages created during this session (excluding the seeded history)."""
        return list(self._history)

    async def send(
        self, content: RealtimeSessionInput | str | BinaryContent | Sequence[RealtimeSessionInput | str | BinaryContent]
    ) -> None:
        """Feed content into the session.

        Accepts a precise [`RealtimeSessionInput`][pydantic_ai.realtime.RealtimeSessionInput], plain
        text as a `str`, image/audio [`BinaryContent`][pydantic_ai.messages.BinaryContent], or a
        sequence of these inputs, dispatched in order. Text and retained images are recorded in
        session history; audio is recorded later through its transcript and/or `audio_retention`.
        `retain_images_every_n=1` records every image, while larger values keep the first image and
        then one of every `N`. Profile-gated operations use the same guards as the dedicated control
        methods.

        [`ToolResult`][pydantic_ai.realtime.ToolResult] is deliberately excluded (`RealtimeSessionInput`
        is [`RealtimeInput`][pydantic_ai.realtime.RealtimeInput] minus `ToolResult`): the session sends
        tool results itself as each tool completes (see `_execute_tool`).
        """
        if isinstance(content, str):
            await self._connection.send(TextInput(text=content))
            self._record_sent_request(
                ModelRequest(parts=[UserPromptPart(content=content)], conversation_id=self._conversation_id)
            )
        elif isinstance(content, BinaryContent):
            if content.is_image:
                await self._send_image(content)
            elif content.media_type == _WAV_MEDIA_TYPE:
                # Retained `SpeechPart.audio` (from `audio_retention`) is a WAV container; unwrap it to
                # raw PCM — matching the seeding path — so a natural round-trip (retain a turn's audio,
                # then `send()` it back) doesn't stream the WAV header into the buffer as noise.
                await self.send_audio(
                    seed_pcm_audio(
                        content,
                        provider_name=self._provider_name or 'realtime',
                        sample_rate=self._profile.get('audio_input_sample_rate', 24000),
                    )
                )
            elif content.media_type == 'audio/pcm':
                await self.send_audio(content.data)
            else:
                raise UserError(
                    f'Unsupported binary media type {content.media_type!r} for `session.send()`. '
                    'Send an image, WAV audio, or raw PCM (`audio/pcm`); for a raw PCM byte stream use `send_audio()`.'
                )
        elif isinstance(content, AudioInput):
            await self.send_audio(content.data)
        elif isinstance(content, TextInput):
            await self._connection.send(content)
            self._record_sent_request(
                ModelRequest(parts=[UserPromptPart(content=content.text)], conversation_id=self._conversation_id)
            )
        elif isinstance(content, ImageInput):
            await self._send_image(BinaryContent(data=content.data, media_type=content.media_type))
        elif isinstance(content, (CommitAudio, ClearAudio, CreateResponse, TruncateOutput, CancelResponse)):
            await self._send_control(content)
        elif isinstance(content, (bytes, bytearray)):
            # `bytes` is a `Sequence[int]`, so guard it before the sequence branch below — otherwise it
            # iterates into a confusing per-byte error. Raw input audio goes through `send_audio()`.
            raise UserError('Raw audio bytes cannot be sent via `session.send()`; use `session.send_audio(...)`.')
        elif isinstance(content, Sequence):
            for item in content:
                await self.send(item)
        else:
            # Unreachable for a well-typed caller: `RealtimeSessionInput` is exhausted above and excludes
            # `ToolResult`. Guard the untyped-caller case (a `ToolResult` passed dynamically) with a clear
            # error, since the session sends tool results itself (see `_execute_tool`).
            raise UserError(
                'Tool results are sent automatically by the realtime session and cannot be sent via `session.send()`.'
            )

    async def _send_control(
        self, content: CommitAudio | ClearAudio | CreateResponse | TruncateOutput | CancelResponse
    ) -> None:
        """Dispatch a manual turn-control / interrupt verb to its session method."""
        if isinstance(content, CommitAudio):
            await self.commit_audio()
        elif isinstance(content, ClearAudio):
            await self.clear_audio()
        elif isinstance(content, CreateResponse):
            await self.create_response()
        elif isinstance(content, TruncateOutput):
            await self.interrupt(audio_end_ms=content.audio_end_ms)
        else:
            await self.interrupt()  # CancelResponse

    async def _send_image(self, content: BinaryContent) -> None:
        """Forward an image and retain it according to the session's sampling policy."""
        self._require_capability(self._profile.get('supports_image_input', False), 'send', 'image input')
        await self._connection.send(ImageInput(data=content.data, media_type=content.media_type))
        if self._sent_image_count % self._retain_images_every_n == 0:
            self._record_sent_request(
                ModelRequest(parts=[UserPromptPart(content=[content])], conversation_id=self._conversation_id)
            )
        self._sent_image_count += 1

    def _record_sent_request(self, request: ModelRequest) -> None:
        """Record a sent request without interleaving it with an in-flight assistant response."""
        response_in_flight = bool(
            self._active_assistant is not None
            or self._response_parts
            or self._native_tool_parts
            or self._pending_provider_response_id is not None
            or self._pending_finish_reason is not None
            or self._pending_response_usage != RequestUsage()
        )
        if response_in_flight:
            self._pending_sent_requests.append(request)
        else:
            self._history.append(request)

    async def send_audio(self, data: bytes) -> None:
        """Stream a chunk of audio to the model."""
        await self._connection.send(AudioInput(data=data))
        if self._retain_input:
            # Buffer the raw input so the finalized user turn can retain it. A per-item speech-stopped
            # boundary later cuts this into that turn's own segment (see `_segment_input_audio`); only the
            # exact split at the boundary is approximate (see `audio_retention`).
            self._input_audio.extend(data)

    async def commit_audio(self) -> None:
        """Commit buffered input audio as a user turn (manual turn-taking / push-to-talk)."""
        self._require_capability(
            self._profile.get('supports_manual_turn_control', False), 'commit_audio', 'manual turn-taking'
        )
        await self._connection.send(CommitAudio())

    async def clear_audio(self) -> None:
        """Discard buffered, uncommitted input audio."""
        self._require_capability(
            self._profile.get('supports_manual_turn_control', False), 'clear_audio', 'manual turn-taking'
        )
        await self._connection.send(ClearAudio())
        # Drop the locally retained copy too (with `audio_retention='input_audio'`/`'both'`), or the discarded
        # audio would still be attached to the next finalized user turn.
        self._input_audio.clear()

    async def create_response(self) -> None:
        """Ask the model to respond now (manual turn-taking, after `commit_audio`)."""
        self._require_capability(
            self._profile.get('supports_manual_turn_control', False), 'create_response', 'manual turn-taking'
        )
        await self._connection.send(CreateResponse())

    async def interrupt(self, *, audio_end_ms: int | None = None) -> None:
        """Barge-in: cancel the model's in-progress response, optionally truncating its audio first.

        This is server-side only — it stops generation and (when `audio_end_ms` is given) syncs the
        provider's transcript to what was actually heard. Flushing locally buffered playback is the
        caller's responsibility.

        Args:
            audio_end_ms: Milliseconds of the current output audio that were actually played. When
                given, the output item is truncated to this point before the response is cancelled.
        """
        self._require_capability(self._profile.get('supports_interruption', False), 'interrupt', 'interruption')
        # Truncate before cancelling: cancellation triggers `response.done`, which clears the tracked
        # output item, so a truncate sent afterwards could no-op.
        if audio_end_ms is not None:
            if not self._profile.get('supports_output_truncation', False):
                raise UserError(
                    'This realtime model does not support output truncation, so `interrupt(audio_end_ms=...)` '
                    'is unavailable. Call `interrupt()` without `audio_end_ms` to cancel without truncating.'
                )
            await self._connection.send(TruncateOutput(audio_end_ms=audio_end_ms))
        await self._connection.send(CancelResponse())

    def _require_capability(self, supported: bool, method: str, feature: str) -> None:
        """Raise a clear `UserError` before sending when the model doesn't support `method`."""
        if not supported:
            raise UserError(f'This realtime model does not support {feature}, so `session.{method}()` is unavailable.')

    # --- history assembly -------------------------------------------------------------------------

    def _ensure_active_assistant(self, *, output_text: bool = False, item_id: str | None = None) -> list[RealtimeEvent]:
        """Start an assistant output part if one isn't already in flight.

        `output_text` selects a plain [`TextPart`][pydantic_ai.messages.TextPart] (the model's
        `output_modalities=('text',)` responses) over the default
        [`SpeechPart`][pydantic_ai.messages.SpeechPart] (spoken audio and its transcript).
        """
        active = self._active_assistant
        events: list[RealtimeEvent] = []
        if active is not None:
            active_item_id = active.id
            item_changed = active_item_id is not None and item_id is not None and active_item_id != item_id
            modality_changed = output_text != isinstance(active, TextPart)
            if item_changed or modality_changed:
                events.extend(self._finalize_assistant_part())
                active = None
        if active is not None:
            if isinstance(self._active_assistant, SpeechPart) and item_id and self._active_assistant.id is None:
                self._active_assistant = replace(
                    self._active_assistant,
                    id=item_id,
                    provider_name=self._provider_name,
                )
            return events
        self._ensure_chat_span()
        part: SpeechPart | TextPart = (
            TextPart(
                content='',
                id=item_id,
                provider_name=self._provider_name if item_id else None,
            )
            if output_text
            else SpeechPart(
                speaker='assistant',
                transcript='',
                id=item_id,
                provider_name=self._provider_name if item_id else None,
            )
        )
        self._active_assistant = part
        self._active_assistant_index = len(self._response_parts)
        self._assistant_transcript = ''
        events.append(PartStartEvent(index=self._active_assistant_index, part=part))
        return events

    def _handle_assistant_transcript(
        self, text: str, *, output_text: bool = False, item_id: str | None = None
    ) -> list[RealtimeEvent]:
        events = self._ensure_active_assistant(output_text=output_text, item_id=item_id)
        active = self._active_assistant
        assert active is not None
        self._assistant_transcript, appended = _accumulate_transcript(self._assistant_transcript, text)
        if isinstance(active, TextPart):
            self._active_assistant = replace(active, content=self._assistant_transcript)
            delta: SpeechPartDelta | TextPartDelta = TextPartDelta(content_delta=appended)
        else:
            self._active_assistant = replace(active, transcript=self._assistant_transcript)
            delta = SpeechPartDelta(transcript_delta=appended)
        if appended:
            events.append(PartDeltaEvent(index=self._active_assistant_index, delta=delta))
        return events

    def _handle_assistant_audio(self, data: bytes, *, item_id: str | None = None) -> list[RealtimeEvent]:
        events = self._ensure_active_assistant(item_id=item_id)
        if self._retain_output:
            self._output_audio.extend(data)
        events.append(PartDeltaEvent(index=self._active_assistant_index, delta=SpeechPartDelta(audio_chunk=data)))
        return events

    def _finalize_assistant_part(self) -> list[RealtimeEvent]:
        """End the in-flight assistant part, appending it to the current response if it has content."""
        if self._active_assistant is None:
            return []
        part = self._active_assistant
        if isinstance(part, SpeechPart):
            if part.transcript == '':
                part = replace(part, transcript=None)
            if self._retain_output and self._output_audio:
                sample_rate = self._profile.get('audio_output_sample_rate', 24000)
                part = replace(
                    part,
                    audio=BinaryContent(
                        data=_pcm_to_wav(bytes(self._output_audio), sample_rate), media_type=_WAV_MEDIA_TYPE
                    ),
                )
        index = self._active_assistant_index
        self._active_assistant = None
        self._assistant_transcript = ''
        self._output_audio.clear()
        if part.has_content():
            self._response_parts.append(part)
        return [PartEndEvent(index=index, part=part)]

    def _finalize_response(
        self,
        *,
        provider_response_id: str | None = None,
        finish_reason: FinishReason | None = None,
        provider_details: dict[str, Any] | None = None,
        interrupted: bool = False,
        response_occurred: bool = False,
    ) -> None:
        """Finalize the current assistant response's parts into a `ModelResponse` in history."""
        response: ModelResponse | None = None
        response_recorded = False
        # The chat span's input is the history the response replied to, captured before we append it.
        input_messages = self.all_messages()
        # Native tool parts (web grounding / code execution) lead the response (call+return, then
        # speech), matching the classic `GoogleModel`, which prepends them ahead of the assistant's text.
        parts = [*self._native_tool_parts, *self._response_parts]
        # Parts prove a response happened. For an output-less response, only terminal/pending provider
        # metadata (or an interruption) does; a bare logical turn boundary must not invent a response.
        response_occurred = bool(
            response_occurred
            or parts
            or self._pending_provider_response_id is not None
            or self._pending_finish_reason is not None
            or self._pending_response_usage != RequestUsage()
            or self._chat_span is not None
        )
        if response_occurred:
            response = ModelResponse(
                parts=parts,
                usage=self._pending_response_usage,
                # Prefer the model the server reported actually serving the session (it can differ
                # from the requested id — xAI silently substitutes its default for unknown slugs),
                # mirroring how request-response models stamp the response's reported model.
                model_name=self._connection.model_name or self._model_name,
                provider_name=self._provider_name,
                provider_url=self._provider_url,
                provider_details=provider_details,
                provider_response_id=provider_response_id or self._pending_provider_response_id,
                finish_reason=finish_reason or self._pending_finish_reason,
                conversation_id=self._conversation_id,
                state='interrupted' if interrupted else 'complete',
            )
            self._history.append(response)
            self.usage.requests += 1
            response_recorded = True
            self._tool_run_step += 1
            for part in parts:
                if isinstance(part, ToolCallPart):
                    self._tool_calls_awaiting_usage.discard(part.tool_call_id)
            if self._pending_tool_returns:
                pending, self._pending_tool_returns = self._pending_tool_returns, []
                for call_part, request in pending:
                    self._insert_tool_return(call_part, request)
                if self._asap_drain_deferred:
                    self._asap_drain_ready = True
        if self._pending_sent_requests:
            self._history.extend(self._pending_sent_requests)
            self._pending_sent_requests = []
        self._end_chat_span(input_messages, response)
        self._response_parts = []
        self._native_tool_parts = []
        self._pending_response_usage = RequestUsage()
        self._pending_provider_response_id = None
        self._pending_finish_reason = None
        if response_recorded:
            self._check_request_limit()

    def _ensure_chat_span(self) -> None:
        """Open a `chat {model}` span for the assistant response now being assembled, if not already open.

        A realtime turn isn't a single request/response, so the honest lifetime of a `chat` span is one
        assistant `ModelResponse`: it opens when that response's first content arrives (the first
        assistant part or tool call) and closes in `_finalize_response`. Tool calls split a turn into
        multiple responses (mirroring a classic run), so each response gets its own span. The span is
        deliberately *not* entered as the current span: `execute_tool` spans run after the response is
        finalized and stay siblings under the session span, matching the classic agent-run tree.

        Attributes are limited to what a realtime session can report honestly. Omitted vs. the classic
        `chat` span (`open_model_request_span`): `model_request_parameters`,
        `gen_ai.tool.definitions`, and `gen_ai.request.*` settings (there are no per-turn request
        parameters or settings). Provider and server attributes, response metadata, usage, cost when
        pricing data is available, and per-response metrics reuse the classic instrumentation helpers.
        Added vs. the classic span: `gen_ai.output.type` (`speech`/`text`), the one semconv attribute
        specific to voice output.
        """
        settings = self._instrumentation
        if settings is None or self._chat_span is not None:
            return
        attributes: dict[str, Any] = {
            'gen_ai.operation.name': 'chat',
            'gen_ai.output.type': self._otel_output_type,
        }
        if self._model_name:
            attributes['gen_ai.request.model'] = self._model_name
        if self._provider_name:
            attributes.update(provider_attributes(self._provider_name, self._provider_url))
        name = f'chat {self._model_name}' if self._model_name else 'chat'
        context = self._session_span_context
        assert context is not None
        self._chat_span = settings.tracer.start_span(
            name,
            context=context,
            attributes=attributes,
            kind=SpanKind.CLIENT,
        )

    def _end_chat_span(self, input_messages: list[ModelMessage], response: ModelResponse | None) -> None:
        """Close the current `chat` span, attaching per-turn messages and usage from `response`."""
        settings = self._instrumentation
        span = self._chat_span
        if settings is None or span is None:
            return
        self._chat_span = None
        price_calculation = response_price_calculation(response) if response is not None else None
        if response is not None and span.is_recording():
            # Reuse the exact message → gen_ai serialization and response-attribute helpers the
            # instrumented model uses, so realtime `chat` spans can't drift from the classic path.
            settings.handle_messages(input_messages, response, span)
            span.set_attributes(
                response_attributes(response, response.model_name or self._model_name, price_calculation)
            )
        span.end()
        if response is not None:
            settings.record_metrics(
                response,
                price_calculation,
                model_metric_attributes(
                    self._provider_name,
                    self._model_name,
                    response.model_name or self._model_name,
                ),
            )

    def _handle_turn_complete(self, event: TurnCompleteEvent) -> list[RealtimeEvent]:
        # Turn boundary for a user turn that wasn't finalized earlier, so history reads user-then-assistant.
        # Gemini emits neither `InputSpeechEndEvent` nor a final (`is_final`) input transcript — it streams
        # only partial transcripts — so its user turn is finalized here: `_finalize_user` for a
        # transcript-driven turn, `_finalize_audio_only_user` for a retained-audio-only one. Both are no-ops
        # when the turn was already finalized (e.g. OpenAI's `is_final` transcript or `commit_audio`).
        events = self._finalize_user()
        events.extend(self._finalize_audio_only_user())
        events.extend(self._finalize_assistant_part())
        already_finalized = bool(
            self._response_finalized_before_terminal
            and not self._response_parts
            and not self._native_tool_parts
            and self._pending_provider_response_id is None
            and self._pending_finish_reason is None
            and self._pending_response_usage == RequestUsage()
        )
        self._response_finalized_before_terminal = False
        self._finalize_response(
            provider_response_id=event.provider_response_id,
            # An interrupted turn (barge-in) isn't an error and has no dedicated `FinishReason`; leave
            # it as the provider reported (usually unset) and let `state='interrupted'` carry the
            # meaning, matching a classic cancelled stream. A clean turn with no reported reason stops.
            finish_reason=event.finish_reason
            or (None if event.interrupted or event.provider_details is not None else 'stop'),
            provider_details=event.provider_details,
            interrupted=event.interrupted,
            response_occurred=bool(
                not already_finalized
                and (
                    event.provider_response_id is not None
                    or event.finish_reason is not None
                    or event.provider_details is not None
                    or event.interrupted
                )
            ),
        )
        events.append(event)
        return events

    def _handle_tool_call_part(self, call_part: ToolCallPart, *, response_usage_follows: bool) -> list[RealtimeEvent]:
        """Fold a tool call into the current response, deferring finalization when its usage follows.

        OpenAI-protocol providers report each call before the `response.done` frame carrying that
        response's usage, so finalization waits for the ensuing `SessionUsageEvent`. Gemini's tool-call
        frame has no per-response usage to wait for; it is finalized immediately with zero usage, while
        the later completed turn keeps the usage Gemini reports for that turn.
        """
        self._ensure_chat_span()
        events = self._finalize_assistant_part()
        index = len(self._response_parts)
        events.append(PartStartEvent(index=index, part=call_part))
        events.append(PartEndEvent(index=index, part=call_part))
        self._response_parts.append(call_part)
        if response_usage_follows:
            self._tool_calls_awaiting_usage.add(call_part.tool_call_id)
        else:
            self._finalize_response()
        return events

    def _complete_tool_call(
        self,
        call_part: ToolCallPart,
        result_part: ToolReturnPart | RetryPromptPart,
        content: str | Sequence[UserContent] | None = None,
    ) -> list[RealtimeEvent]:
        request_parts: list[ModelRequestPart] = [result_part]
        if content:
            request_parts.append(UserPromptPart(content=content))
        self._insert_tool_return(
            call_part,
            ModelRequest(parts=request_parts, conversation_id=self._conversation_id),
        )
        return [FunctionToolResultEvent(part=result_part, content=content)]

    def _insert_tool_return(self, call_part: ToolCallPart, request: ModelRequest) -> None:
        """Insert a tool-return request directly after the response carrying its call.

        A tool can finish after later turns, but request-response APIs demand call/return
        adjacency (OpenAI: a `tool` message must directly follow the assistant message with the call;
        Anthropic: the `tool_result` must open the next user message), so history keeps the canonical
        order even though the `FunctionToolResultEvent` streams in completion order.
        """
        for i in range(len(self._history) - 1, -1, -1):
            message = self._history[i]
            if isinstance(message, ModelResponse) and any(
                isinstance(part, ToolCallPart) and part.tool_call_id == call_part.tool_call_id for part in message.parts
            ):
                # Skip past returns already inserted for this response (parallel calls keep call order).
                insert_at = i + 1
                while insert_at < len(self._history) and _is_tool_result_request(self._history[insert_at]):
                    insert_at += 1
                self._history.insert(insert_at, request)
                return
        if call_part.tool_call_id in self._tool_calls_awaiting_usage:
            # OpenAI-protocol tool execution starts before `response.done` supplies usage and finalizes
            # the calling response. Preserve the streamed completion now and insert it once that lands.
            self._pending_tool_returns.append((call_part, request))
        else:
            # The calling response is otherwise finalized before execution begins, so this is an
            # invariant fallback: keep the history complete rather than dropping the tool result.
            self._history.append(request)  # pragma: no cover

    def _handle_input_transcript(self, text: str, is_final: bool, *, item_id: str | None = None) -> list[RealtimeEvent]:
        if item_id is not None:
            # Once an item is closed (finalized or its transcription failed), ignore any stray later event
            # for it — re-creating it would duplicate the turn or resurrect a discarded failed one.
            if item_id in self._finalized_user_item_ids:
                return []
            events: list[RealtimeEvent] = []
            if item_id not in self._active_users_by_id:
                part = SpeechPart(
                    speaker='user',
                    transcript='',
                    id=item_id,
                    provider_name=self._provider_name,
                )
                self._active_users_by_id[item_id] = part
                self._user_transcripts_by_id[item_id] = ''
                self._user_item_order.append(item_id)
                events.append(PartStartEvent(index=0, part=part))
            transcript, appended = _accumulate_transcript(self._user_transcripts_by_id[item_id], text)
            self._user_transcripts_by_id[item_id] = transcript
            self._active_users_by_id[item_id] = replace(self._active_users_by_id[item_id], transcript=transcript)
            if appended:
                events.append(PartDeltaEvent(index=0, delta=SpeechPartDelta(transcript_delta=appended)))
            if is_final:
                events.extend(self._finalize_user(item_id=item_id))
            return events

        events: list[RealtimeEvent] = []
        if self._active_user is None:
            part = SpeechPart(speaker='user', transcript='')
            self._active_user = part
            self._user_transcript = ''
            events.append(PartStartEvent(index=0, part=part))
        self._user_transcript, appended = _accumulate_transcript(self._user_transcript, text)
        assert self._active_user is not None
        self._active_user = replace(self._active_user, transcript=self._user_transcript)
        if appended:
            events.append(PartDeltaEvent(index=0, delta=SpeechPartDelta(transcript_delta=appended)))
        if is_final:
            events.extend(self._finalize_user())
        return events

    def _finalize_user(self, *, item_id: str | None = None) -> list[RealtimeEvent]:
        if item_id is None:
            if self._active_user is None:
                return []  # pragma: no cover
            part = self._active_user
            self._active_user = None
            self._user_transcript = ''
        else:
            part = self._active_users_by_id.pop(item_id)
            self._user_transcripts_by_id.pop(item_id)
            self._finalized_user_item_ids.add(item_id)
        # Strip surrounding whitespace at finalization: providers whose transcripts arrive as a cumulative
        # or final snapshot (OpenAI/xAI) already reconcile leading-space drift via `_accumulate_transcript`,
        # but a partial-only stream (Gemini) concatenates deltas verbatim and would otherwise keep the
        # leading space its first delta carries. Stripping here aligns the two; it's a no-op when already
        # reconciled, and an all-whitespace (or empty) transcript collapses to `None` (an audio-only turn).
        part = replace(part, transcript=(part.transcript or '').strip() or None)
        if self._retain_input:
            # Prefer this item's own segment (cut at its speech-stopped boundary); it's the correct audio
            # even if a later turn's transcript already finalized. Fall back to the rolling buffer for
            # id-less providers, manual push-to-talk, and boundary-less turns, where it holds this turn's
            # audio — and only clear the shared rolling buffer on that fallback, never when a segment was
            # used (a following turn's audio may already be accumulating there).
            segment = self._input_audio_by_id.pop(item_id, None) if item_id is not None else None
            if segment is None:
                segment = bytes(self._input_audio) if self._input_audio else None
                self._input_audio.clear()
            if segment:
                sample_rate = self._profile.get('audio_input_sample_rate', 24000)
                part = replace(
                    part,
                    audio=BinaryContent(data=_pcm_to_wav(segment, sample_rate), media_type=_WAV_MEDIA_TYPE),
                )
        if item_id is None:
            if part.has_content():
                self._history.append(ModelRequest(parts=[part], conversation_id=self._conversation_id))
        else:
            self._finalized_users_by_id[item_id] = part
            self._flush_finalized_user_prefix()
        return [PartEndEvent(index=0, part=part)]

    def _flush_finalized_user_prefix(self) -> None:
        """Append finalized user items in provider order, up to the first item still awaiting its final.

        Item-ID transcripts finalize in any order, but history must keep provider order (call/return
        adjacency etc.), so a finalized item waits in `_finalized_users_by_id` until every earlier item
        has resolved (finalized or discarded).
        """
        while self._user_item_order and self._user_item_order[0] in self._finalized_users_by_id:
            finalized_id = self._user_item_order.popleft()
            finalized = self._finalized_users_by_id.pop(finalized_id)
            if finalized.has_content():
                self._history.append(ModelRequest(parts=[finalized], conversation_id=self._conversation_id))

    def _segment_input_audio(self, item_id: str | None) -> None:
        """Cut the rolling input-audio buffer into `item_id`'s own segment at its speech-stopped boundary.

        Only applies with transcription enabled and input audio retained: the transcript arrives
        asynchronously (and possibly after a following turn's), so pinning the audio to the item now keeps
        it with the right user turn. `setdefault` makes it idempotent if the provider repeats the boundary
        (or also emits a `committed` one): the first segment for an id wins.
        """
        if self._input_transcription_enabled and self._retain_input and item_id and self._input_audio:
            self._input_audio_by_id.setdefault(item_id, bytes(self._input_audio))
            self._input_audio.clear()

    def _drop_input_audio_segment(self, item_id: str | None) -> None:
        """Discard a retained input-audio segment whose transcript will never arrive (e.g. on failure)."""
        if item_id is not None:
            self._input_audio_by_id.pop(item_id, None)

    def _discard_failed_user_item(self, item_id: str | None) -> None:
        """Drop all state for a user item whose transcription failed.

        A failed transcription never becomes a user turn (its partial text is unreliable), and — crucially —
        it must not sit at the head of `_user_item_order` blocking later finalized turns from reaching
        history until the session closes. Drop its retained audio, partial transcript, and ordering entry,
        mark it closed so stray late events are ignored, then flush any turns it was blocking.
        """
        self._drop_input_audio_segment(item_id)
        if item_id is None:
            return
        self._active_users_by_id.pop(item_id, None)
        self._user_transcripts_by_id.pop(item_id, None)
        self._finalized_users_by_id.pop(item_id, None)
        if item_id in self._user_item_order:
            self._user_item_order.remove(item_id)
        self._finalized_user_item_ids.add(item_id)
        self._flush_finalized_user_prefix()

    def _flush_pending_users(self) -> None:
        """Preserve transcript-bearing user items that never received an explicit final event."""
        if self._active_user is not None:
            self._finalize_user()
        for item_id in list(self._user_item_order):
            if item_id in self._active_users_by_id:
                self._finalize_user(item_id=item_id)
        while self._user_item_order:
            item_id = self._user_item_order.popleft()
            part = self._finalized_users_by_id.pop(item_id, None)
            if part is not None and part.has_content():
                self._history.append(ModelRequest(parts=[part], conversation_id=self._conversation_id))
        self._active_users_by_id.clear()
        self._user_transcripts_by_id.clear()
        self._finalized_users_by_id.clear()
        # Drop any input-audio segments whose transcript never arrived, so they can't leak across a
        # long-lived session (finalized items already popped their own segment above).
        self._input_audio_by_id.clear()

    def _finalize_audio_only_user(self) -> list[RealtimeEvent]:
        """Finalize a user turn from retained input audio when no transcript will arrive.

        With input transcription disabled but input audio retained (`audio_retention='input_audio'`/`'both'`),
        the user's turn produces no [`InputTranscript`][pydantic_ai.realtime.InputTranscript], so the
        transcript-driven `_finalize_user` never runs. This is called at each user-turn boundary (speech
        stopped / commit / turn complete) to finalize an audio-only user
        [`SpeechPart`][pydantic_ai.messages.SpeechPart] so the turn still lands in history.

        Gated on transcription being *off*: when it's on we wait for the transcript instead, so an
        (asynchronously delivered) transcript can never race this into a duplicate user turn. A no-op
        when there's an active transcript-driven user part, nothing retained, or transcription is on.
        """
        if self._input_transcription_enabled or not self._retain_input:
            return []
        if self._active_user is not None or not self._input_audio:
            return []
        part = SpeechPart(
            speaker='user',
            transcript=None,
            audio=BinaryContent(
                data=_pcm_to_wav(
                    bytes(self._input_audio),
                    self._profile.get('audio_input_sample_rate', 24000),
                ),
                media_type=_WAV_MEDIA_TYPE,
            ),
        )
        self._input_audio.clear()
        self._history.append(ModelRequest(parts=[part], conversation_id=self._conversation_id))
        # No deltas to stream (there's no transcript), so bracket the turn with just start/end so a
        # streaming consumer still sees the user turn boundary.
        return [PartStartEvent(index=0, part=part), PartEndEvent(index=0, part=part)]

    def _is_replayed_item(self, item_id: str | None, tool_call_id: str | None = None) -> bool:
        """Whether an xAI resumption replay already exists in local history."""
        return (item_id is not None and item_id in self._replayed_item_ids) or (
            tool_call_id is not None and tool_call_id in self._replayed_tool_call_ids
        )

    def _accept_item(self, item_id: str | None, tool_call_id: str | None = None) -> bool:
        """Return `False` for an xAI item that belongs to the resumption replay burst."""
        return not self._is_replayed_item(item_id, tool_call_id)

    def _handle_conversation_item(self, event: ConversationItemCreated) -> None:
        """Remember IDs assigned to xAI's replay burst so related events are suppressed."""
        if event.replayed:
            if event.item_id is not None:
                self._replayed_item_ids.add(event.item_id)
            if event.tool_call_id is not None:
                self._replayed_tool_call_ids.add(event.tool_call_id)

    def _translate_event(self, event: _TranslatableEvent) -> list[RealtimeEvent]:
        """Translate a low-level codec event into shared session events, building history as a side effect.

        Tool calls and usage are handled in `_handle_pump_event` (they interact with the queue and
        tool execution); everything else routes through here. `event` is typed as `_TranslatableEvent`
        (the pump-consumed variants narrowed out) so the final `assert_never` gives static exhaustiveness.
        """
        if isinstance(event, AudioDelta):
            if not self._accept_item(event.item_id):
                return []
            return self._handle_assistant_audio(event.data, item_id=event.item_id)
        if isinstance(event, Transcript):
            if not self._accept_item(event.item_id):
                return []
            # `is_final` doesn't end the part — the turn ends on `TurnCompleteEvent`; a final transcript just
            # carries the full text, which `_accumulate_transcript` reconciles against the deltas. Plain
            # text output (`output_text`) becomes a `TextPart`, an audio transcript a `SpeechPart`.
            return self._handle_assistant_transcript(event.text, output_text=event.output_text, item_id=event.item_id)
        if isinstance(event, InputTranscript):
            if not self._accept_item(event.item_id):
                return []
            return self._handle_input_transcript(event.text, event.is_final, item_id=event.item_id)
        if isinstance(event, InputSpeechEndEvent):
            # The user's speech segment ended (server VAD). With transcription enabled and input audio
            # retained, cut the rolling buffer into this item's own segment so a later out-of-order
            # transcript still attaches its own audio; with transcription off there's no lagging transcript,
            # so `_finalize_audio_only_user` consumes the rolling buffer synchronously here instead.
            self._segment_input_audio(event.item_id)
            return [*self._finalize_audio_only_user(), event]
        if isinstance(event, TurnCompleteEvent):
            return self._handle_turn_complete(event)
        if isinstance(event, PartStartEvent):
            # Providers emit native tool activity as ordinary part events. Buffer the started part for
            # the assistant response while streaming the same event to the caller.
            self._ensure_chat_span()
            self._native_tool_parts.append(event.part)
            return [event]
        if isinstance(event, PartEndEvent):
            return [event]
        if isinstance(event, InputTranscriptionFailedEvent):
            # This item's transcript won't arrive: discard its state (retained audio, any partial transcript,
            # ordering entry) so it never becomes a turn and doesn't block later turns, then surface the
            # failure.
            self._discard_failed_user_item(event.item_id)
            return [event]
        # The remaining control-plane events pass through unchanged. `assert_never` makes pyright flag
        # any new non-pump `RealtimeEvent` variant that isn't handled here.
        if isinstance(
            event,
            (
                InputSpeechStartEvent,
                ReconnectedEvent,
            ),
        ):
            return [event]
        if isinstance(event, SessionErrorEvent):
            if event.recoverable:
                # A recoverable error is mid-stream: the session keeps running, so surface the event to
                # the consumer (rather than swallowing it) for observability. Only a non-recoverable
                # error ends the session, by raising.
                return [event]
            raise RealtimeError(event.message)
        assert_never(event)

    # --- instrumentation --------------------------------------------------------------------------

    def _finalize_span(self, settings: InstrumentationSettings, span: Span) -> None:
        """Attach cumulative usage, run context, and conversation messages to the session span."""
        # Report cumulative usage under `gen_ai.aggregated_usage.*` (mirroring the classic agent-run
        # span) so backends that sum span attributes don't double-count it against the per-turn `chat`
        # spans, which carry each response's usage under `gen_ai.usage.*`. Shared with the classic span.
        attributes: dict[str, Any] = {
            **settings.aggregated_usage_attributes(self.usage),
            **settings.system_instructions_attributes(self._instructions),
        }
        schema_properties: dict[str, Any] = {}
        if 'gen_ai.system_instructions' in attributes:
            schema_properties['gen_ai.system_instructions'] = {'type': 'array'}
        # Mirror the classic agent-run span's end-of-run contract (the `Instrumentation`
        # capability's `_run_span_end_attributes`): the full conversation — seeded history included —
        # under `pydantic_ai.all_messages`, with `pydantic_ai.new_message_index` marking where this
        # session's messages begin. Emitted regardless of `include_content`: `otel_message_parts`
        # redacts part *content* when it is disabled, leaving the conversation structure. The
        # `logfire.json_schema` entry marks the attribute as a JSON array so the Logfire UI
        # deserializes and renders it as a conversation rather than as a string.
        if messages := self.all_messages():
            attributes['pydantic_ai.all_messages'] = safe_to_json(settings.messages_to_otel_messages(messages)).decode()
            if self._seeded:
                attributes['pydantic_ai.new_message_index'] = len(self._seeded)
            schema_properties['pydantic_ai.all_messages'] = {'type': 'array'}
        if self._metadata is not None:
            attributes['metadata'] = safe_to_json(serialize_any(self._metadata)).decode()
            schema_properties['metadata'] = {}
        if schema_properties:
            attributes['logfire.json_schema'] = pydantic_core.to_json(
                {'type': 'object', 'properties': schema_properties}
            ).decode()
        span.set_attributes(attributes)

    async def _execute_tool(
        self,
        call: ToolCall,
        call_part: ToolCallPart,
        validation_done: asyncio.Event,
    ) -> _SettledToolResult:
        # No `execute_tool` span is created here: the `execute_tool` span is owned by the
        # `Instrumentation` capability's `wrap_tool_execute` hook, which `Agent.realtime_session`
        # injects into the tool runner's `ToolManager` (mirroring a classic run). That capability
        # span is the single, canonical source of tool spans; the pump task runs inside the session
        # span's OTel context, so the capability's tool span nests under the session span as a sibling
        # of the `chat` spans. The session-level `realtime` span and per-response `chat` spans below
        # stay hand-managed for now — they move onto exchange-level capability hooks when those land.
        args, error = _parse_tool_args(call.args)
        if error is not None:
            await self._queue.put(FunctionToolCallEvent(part=call_part, args_valid=False))
            validation_done.set()
            result_part: ToolReturnPart | RetryPromptPart = RetryPromptPart(
                content=error,
                tool_name=call.tool_name,
                tool_call_id=call.tool_call_id,
            )
            user_content: str | Sequence[UserContent] | None = None
        else:
            assert args is not None
            async with self._tool_manager_lock:
                ctx = self._tool_manager.ctx
                if ctx is not None and ctx.run_step < self._tool_run_step:
                    self._tool_manager = await self._tool_manager.for_run_step(
                        replace(ctx, run_step=self._tool_run_step)
                    )
                # Pin the step-synchronized manager for this call: a concurrent tool task can swap
                # `self._tool_manager` (its own `for_run_step` advance) between here and the calls below,
                # so re-reading the attribute there could run against a different run-step's manager.
                tool_manager = self._tool_manager
            tool_call = ToolCallPart(tool_name=call.tool_name, args=args, tool_call_id=call.tool_call_id)

            async def on_validate(args_valid: bool) -> None:
                await self._queue.put(FunctionToolCallEvent(part=call_part, args_valid=args_valid))
                validation_done.set()

            async def on_inline_deferred(
                requests: DeferredToolRequests,
                results: DeferredToolResults,
            ) -> None:
                await self._queue.put(DeferredToolRequestsEvent(requests))
                await self._queue.put(DeferredToolResultsEvent(results))

            try:
                tool_result = await tool_manager.handle_call(
                    tool_call,
                    on_validate=on_validate,
                    on_inline_deferred=on_inline_deferred,
                )
            except ToolRetryError as e:
                result_part = e.tool_retry
                user_content = None
            except (ApprovalRequired, CallDeferred) as e:
                # `handle_call` already gave the `HandleDeferredToolCalls` capability handler the
                # chance to resolve the deferral inline (approve, deny, retry, or substitute a
                # result); reaching here means no handler resolved it. The graph's fallback — pausing
                # the run with a `DeferredToolRequests` output — has no realtime analog (a live
                # conversation can't wait for an out-of-band result, and the provider expects an
                # answer on the string-only tool channel), so answer with a deliberate explanation —
                # rather than a leaked exception repr — that the model can voice, keeping the
                # conversation flowing.
                reason = 'requires approval' if isinstance(e, ApprovalRequired) else 'runs externally'
                result_part = ToolReturnPart(
                    tool_name=call.tool_name,
                    content=(
                        f'Error: The {call.tool_name!r} tool {reason} and cannot be completed during a realtime session.'
                    ),
                    tool_call_id=call.tool_call_id,
                )
                user_content = None
            else:
                tool_def = tool_manager.get_tool_def(call.tool_name)
                result_part, user_content = build_tool_return_part(
                    tool_result,
                    call=tool_call,
                    tool_kind=tool_def.tool_kind if tool_def else None,
                )

        if isinstance(result_part, RetryPromptPart):
            output = result_part.model_response()
            wire_content: list[UserContent] = []
        else:
            output, wire_content = result_part.model_response_str_and_user_content()
        if isinstance(user_content, str):
            wire_content.append(user_content)
        elif user_content:
            wire_content.extend(user_content)
        if call.tool_call_id not in self._tool_calls_awaiting_usage:
            await self._drain_pending_messages('asap')
        await self._connection.send(
            ToolResult(
                tool_call_id=call.tool_call_id,
                output=output,
                content=wire_content or None,
            )
        )
        return result_part, user_content

    # --- streaming --------------------------------------------------------------------------------

    def _notify_asap_pending_messages(self) -> None:
        """Wake an `asap` drain from either an async tool or a sync-tool worker thread."""
        loop = self._loop
        if loop is not None and not self._closed:
            try:
                loop.call_soon_threadsafe(self._start_asap_pending_message_drain)
            except RuntimeError:
                pass

    def _start_asap_pending_message_drain(self) -> None:
        if self._closed:
            return
        task = asyncio.create_task(self._drain_pending_messages('asap'))
        self._background_tasks.add(task)
        task.add_done_callback(self._pending_message_task_done)

    def _pending_message_task_done(self, task: asyncio.Task[None]) -> None:
        self._background_tasks.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None:
            self._queue.put_nowait(error)
        self._queue.put_nowait(self._queue_changed)

    async def _drain_pending_messages(self, priority: PendingMessagePriority) -> None:
        """Deliver queued text prompts of `priority` and record them as normal user turns."""
        async with self._pending_messages_lock:
            if priority == 'asap' and self._tool_calls_awaiting_usage:
                self._asap_drain_deferred = True
                return
            selected = self._pending_messages.pop_priority(priority)
            for pending in selected:
                await self.send(_pending_message_text(pending))
            if priority == 'asap':
                self._asap_drain_deferred = False

    def _check_tool_call_limit(self) -> None:
        # Let `UsageLimitExceeded` propagate (caught by the pump and re-raised to the consumer), matching
        # how a regular `run`/`iter` surfaces a usage limit rather than wrapping it in another error.
        if self._usage_limits is None:
            return
        projected = dataclasses.replace(self.usage, tool_calls=self.usage.tool_calls + 1)
        self._usage_limits.check_before_tool_call(projected)

    def _check_token_limit(self) -> None:
        if self._usage_limits is None:
            return
        self._usage_limits.check_tokens(self.usage)

    def _check_request_limit(self) -> None:
        if self._usage_limits is None:
            return
        request_limit = self._usage_limits.request_limit
        if request_limit is not None and self.usage.requests > request_limit:
            raise UsageLimitExceeded(f'The next request would exceed the request_limit of {request_limit}')

    def _accumulate_response_usage(self, event: SessionUsageEvent) -> None:
        self._pending_response_usage = self._pending_response_usage + event.usage
        self._pending_provider_response_id = event.provider_response_id or self._pending_provider_response_id
        self._pending_finish_reason = event.finish_reason or self._pending_finish_reason
        if self._tool_calls_awaiting_usage:
            self._finalize_response(
                provider_response_id=event.provider_response_id,
                finish_reason=event.finish_reason,
            )
            # OpenAI emits this usage immediately before `response.done`; the response is complete
            # already, so that terminal must not append a second, empty `ModelResponse`.
            self._response_finalized_before_terminal = True

    async def _run_tool(self, call: ToolCall, call_part: ToolCallPart, validation_done: asyncio.Event) -> None:
        """Run a tool and feed its completion (or failure) back through the queue."""
        try:
            result_part, content = await self._execute_tool(call, call_part, validation_done)
        except Exception as e:
            # Surface the failure through the queue so the consumer re-raises it, instead of letting it
            # vanish into `__aexit__`'s cleanup-only drain and hang the session on a completion that
            # never arrives.
            await self._queue.put(e)
            return
        finally:
            validation_done.set()
            # Settled (completed, failed, or cancelled): no longer cancellable by `ToolCallCancelled`.
            self._pending_tool_calls.pop(call_part.tool_call_id, None)
        for event in self._complete_tool_call(call_part, result_part, content):
            await self._queue.put(event)
        if self._asap_drain_deferred and not self._tool_calls_awaiting_usage:
            await self._drain_pending_messages('asap')

    def _tool_task_done(self, task: asyncio.Task[None]) -> None:
        self._background_tasks.discard(task)
        # Surface any exception raised outside `_run_tool`'s own try/except — notably the post-`finally`
        # `asap` drain, whose `connection.send` can fail if the socket just dropped — mirroring
        # `_pending_message_task_done`. Otherwise it vanishes with only an "exception was never
        # retrieved" warning at GC, silently losing the enqueued message with no signal to the consumer.
        if not task.cancelled() and (error := task.exception()) is not None:
            self._queue.put_nowait(error)
        # Wake the queue reader so it can finish once both the pump and the last tool are done.
        self._queue.put_nowait(self._queue_changed)

    async def _handle_pump_event(
        self,
        event: RealtimeCodecEvent,
    ) -> bool:
        """Process one upstream event onto the queue; return `True` to stop the pump (a limit tripped)."""
        if isinstance(event, ConversationCreated):
            return False
        if isinstance(event, ConversationItemCreated):
            self._handle_conversation_item(event)
            return False
        if isinstance(event, ToolCall):
            if not self._accept_item(event.item_id, event.tool_call_id):
                return False
            self._check_tool_call_limit()
            self.usage.tool_calls += 1
            call_part = ToolCallPart(
                tool_name=event.tool_name,
                args=event.args,
                tool_call_id=event.tool_call_id,
                id=event.item_id,
                provider_name=self._provider_name if event.item_id is not None else None,
            )
            for out in self._handle_tool_call_part(
                call_part,
                response_usage_follows=event.response_usage_follows,
            ):
                await self._queue.put(out)
            validation_done = asyncio.Event()
            task = asyncio.create_task(self._run_tool(event, call_part, validation_done))
            self._background_tasks.add(task)
            self._pending_tool_calls[call_part.tool_call_id] = (task, call_part)
            task.add_done_callback(self._tool_task_done)
            await validation_done.wait()
            return False
        if isinstance(event, ToolCallCancelled):
            for tool_call_id in event.tool_call_ids:
                if (pending := self._pending_tool_calls.pop(tool_call_id, None)) is None:
                    continue
                task, call_part = pending
                task.cancel()
                # Record a cancelled result so the call still has a matching return in history (kept
                # valid for a handoff), and deliberately don't send a `ToolResult` back to the model —
                # it abandoned the call.
                cancelled_part = ToolReturnPart(
                    tool_name=call_part.tool_name,
                    content=_CANCELLED_TOOL_RESULT,
                    tool_call_id=call_part.tool_call_id,
                    outcome='interrupted',
                )
                for out in self._complete_tool_call(call_part, cancelled_part):
                    await self._queue.put(out)
            return False
        if isinstance(event, SessionUsageEvent):
            self.usage.incr(event.usage)
            self._check_token_limit()
            if event.response_scoped:
                self._accumulate_response_usage(event)
            if self._asap_drain_ready:
                self._asap_drain_ready = False
                await self._drain_pending_messages('asap')
            return False
        for out in self._translate_event(event):
            await self._queue.put(out)
        if isinstance(event, TurnCompleteEvent):
            await self._drain_pending_messages('when_idle')
        return False

    async def _pump(self, context: Context | None) -> None:
        """Drain the connection into the session queue under the explicit session-span context."""
        token = otel_context.attach(context) if context is not None else None
        try:
            async for event in self._connection:
                if await self._handle_pump_event(event):
                    return  # a usage limit tripped: stop reading the upstream
        except Exception as e:
            self._pump_error = e
        finally:
            self._pump_finished = True
            await self._queue.put(self._queue_changed)
            if token is not None:
                otel_context.detach(token)

    async def __aiter__(self) -> AsyncIterator[RealtimeEvent]:
        """Read translated events from the session queue without owning session resources."""
        if not self._entered:
            raise RuntimeError('Enter the realtime session with `async with` before iterating it.')
        if self._closed:
            raise RuntimeError('This realtime session is closed and cannot be iterated.')
        if self._iterator_active:
            raise RuntimeError('This realtime session is already being iterated.')
        if self._stream_exhausted:
            raise RuntimeError('This realtime session event stream has already ended.')

        self._iterator_active = True
        if self._pump_task is None:
            self._pump_task = asyncio.create_task(self._pump(self._session_span_context))
        try:
            while True:
                item = await self._queue.get()
                if item is self._queue_changed:
                    # A pump error takes priority over stuck background tools. Their cancellation and
                    # drain belong to `__aexit__`, which runs as this exception leaves the owner block.
                    if self._pump_error is not None:
                        self._stream_exhausted = True
                        raise self._pump_error
                    if self._pump_finished and not self._background_tasks:
                        self._stream_exhausted = True
                        return
                    continue
                yield _as_event(item)  # re-raises if a tool failed
        finally:
            self._iterator_active = False
