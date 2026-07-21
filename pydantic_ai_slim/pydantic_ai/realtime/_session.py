"""A realtime session that wraps a [`RealtimeConnection`][pydantic_ai.realtime.RealtimeConnection] with automatic tool execution."""

from __future__ import annotations as _annotations

import asyncio
import dataclasses
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

import pydantic_core
from anyio import Lock
from opentelemetry import context as otel_context
from opentelemetry.context import Context
from opentelemetry.trace import Span, SpanKind, StatusCode, set_span_in_context
from typing_extensions import assert_never

from .._instrumentation import response_attributes, safe_to_json
from .._utils import cancel_and_drain
from ..exceptions import ToolRetryError, UserError
from ..messages import (
    BinaryContent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    SpeechPart,
    SpeechPartDelta,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    UserPromptPart,
)
from ..native_tools import SUPPORTED_NATIVE_TOOLS
from ..tool_manager import ToolManager
from ..tools import ToolDenied
from ..usage import RequestUsage, RunUsage, UsageLimits
from ._base import (
    AudioDelta,
    AudioInput,
    AudioRetention,
    CancelResponse,
    ClearAudio,
    CommitAudio,
    CreateResponse,
    ImageInput,
    InputSpeechEndEvent,
    InputSpeechStartEvent,
    InputTranscript,
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
)

if TYPE_CHECKING:
    from ..models.instrumented import InstrumentationSettings

# Realtime providers stream raw PCM audio; there's no container to carry a richer media type, so
# retained audio is tagged as `audio/pcm`.
_PCM_MEDIA_TYPE = 'audio/pcm'
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
    | ReconnectedEvent
    | PartStartEvent
    | PartEndEvent
    | SessionErrorEvent
)


def _as_event(item: object) -> RealtimeEvent:
    """Unwrap a queue item: re-raise a tool's exception, otherwise return the event."""
    if isinstance(item, Exception):
        raise item
    return cast('RealtimeEvent', item)


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


def _is_tool_return_request(message: ModelMessage) -> bool:
    """Whether a history message is a request carrying only tool returns (an inserted tool result)."""
    return isinstance(message, ModelRequest) and all(isinstance(part, ToolReturnPart) for part in message.parts)


class RealtimeSession:
    """Wraps a [`RealtimeConnection`][pydantic_ai.realtime.RealtimeConnection], building message history and auto-executing tools.

    The session translates the connection's low-level codec events into the shared message/part event
    vocabulary from [`pydantic_ai.messages`][pydantic_ai.messages] and accumulates ordinary
    [`ModelMessage`][pydantic_ai.messages.ModelMessage] history as the conversation proceeds, so a
    session can hand off to [`Agent.run`][pydantic_ai.agent.AbstractAgent.run] via
    [`all_messages`][pydantic_ai.realtime.RealtimeSession.all_messages]:

    - assistant speech becomes [`PartStartEvent`][pydantic_ai.messages.PartStartEvent] /
      [`PartDeltaEvent`][pydantic_ai.messages.PartDeltaEvent] / [`PartEndEvent`][pydantic_ai.messages.PartEndEvent]
      events carrying an [`SpeechPart`][pydantic_ai.messages.SpeechPart]
      (`speaker='assistant'`), finalized into a [`ModelResponse`][pydantic_ai.messages.ModelResponse]
      at the end of the turn;
    - user speech becomes the same part events with `speaker='user'`, finalized into a
      [`ModelRequest`][pydantic_ai.messages.ModelRequest];
    - a tool call becomes a [`ToolCallPart`][pydantic_ai.messages.ToolCallPart] (start/end) plus a
      [`FunctionToolCallEvent`][pydantic_ai.messages.FunctionToolCallEvent] when execution starts and a
      [`FunctionToolResultEvent`][pydantic_ai.messages.FunctionToolResultEvent] carrying a
      [`ToolReturnPart`][pydantic_ai.messages.ToolReturnPart] when it completes.

    Tools always run concurrently with the session. The session keeps streaming events while a tool
    runs, so the model can keep speaking and user speech keeps being processed, then sends the result
    back over the connection once it is ready. This mirrors how a person can keep talking while work
    happens.

    A [`ToolReturnPart`][pydantic_ai.messages.ToolReturnPart] is always sent for every tool call, even
    when argument parsing or the tool itself fails, so the model never stalls waiting on a result.

    History is accumulated in the order events are reported, which is authoritative for turn-by-turn
    speech but approximate at the edges: user transcripts that a provider reports asynchronously are
    ordered as received. Tool results are the exception: a tool's
    [`FunctionToolResultEvent`][pydantic_ai.messages.FunctionToolResultEvent] streams whenever the tool
    finishes (possibly after later turns), but in [`all_messages()`][pydantic_ai.realtime.RealtimeSession.all_messages]
    its [`ToolReturnPart`][pydantic_ai.messages.ToolReturnPart] is placed directly after the response
    carrying its call — request-response APIs require that adjacency, so the history stays valid for a
    handoff to a standard [`Agent.run`][pydantic_ai.agent.AbstractAgent.run].

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
        agent_name: str | None = None,
        usage: RunUsage | None = None,
        usage_limits: UsageLimits | None = None,
        audio_retention: AudioRetention = 'transcript_only',
        message_history: Sequence[ModelMessage] | None = None,
        profile: RealtimeModelProfile | None = None,
        conversation_id: str | None = None,
        output_modality: Literal['audio', 'text'] = 'audio',
    ) -> None:
        self._connection = connection
        self._tool_manager = tool_manager
        self._tool_run_step = 0
        self._tool_manager_lock = Lock()
        self._instrumentation = instrumentation
        self._profile = profile if profile is not None else _FULL_PROFILE
        self._model_name = model_name
        self._agent_name = agent_name
        self._conversation_id = conversation_id
        # The semconv `gen_ai.output.type` value for the session's configured output modality:
        # `'speech'` for spoken audio (the enum's term for voice output), `'text'` for text-only.
        self._otel_output_type = 'speech' if output_modality == 'audio' else 'text'
        self._usage_limits = usage_limits
        self._audio_retention = audio_retention
        self._retain_input = audio_retention in ('input', 'both')
        self._retain_output = audio_retention in ('output', 'both')
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
                "or pass `audio_retention='input'` or `'both'` to keep the "
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
        self._active_assistant: SpeechPart | TextPart | None = None
        self._active_assistant_index = 0
        self._assistant_transcript = ''
        self._output_audio = bytearray()

        # In-flight user request being assembled from input-transcript events.
        self._active_user: SpeechPart | None = None
        self._user_transcript = ''
        self._input_audio = bytearray()

        # The session context is the single owner of the receive pump and background tool tasks.
        # Iteration starts the pump lazily, but never tears it down: an early `break` can abandon the
        # reader generator without affecting resource lifetime, and `__aexit__` still drains everything
        # before the connection and toolset close.
        self._queue: asyncio.Queue[RealtimeEvent | object] = asyncio.Queue()
        self._queue_changed = object()
        self._background_tasks: set[asyncio.Task[None]] = set()
        # In-flight tool tasks keyed by tool call id, so a `ToolCallCancelled` can cancel the specific
        # calls the model abandoned (e.g. on barge-in) without touching the others.
        self._pending_tool_calls: dict[str, tuple[asyncio.Task[None], ToolCallPart]] = {}
        self._pump_task: asyncio.Task[None] | None = None
        self._pump_error: Exception | None = None
        self._pump_finished = False
        self._iterator_active = False
        self._stream_exhausted = False
        self._entered = False
        self._closed = False

        # The session span is deliberately not made current in the owner's task. Child spans receive
        # this explicit context directly, or through the pump task's same-task attach/detach pair.
        self._session_span: Span | None = None
        self._session_span_context: Context | None = None
        self._session_span_attributes: dict[str, Any] | None = None

    async def __aenter__(self) -> RealtimeSession:
        if self._entered or self._closed:
            raise RuntimeError('This realtime session cannot be entered more than once.')
        self._entered = True

        settings = self._instrumentation
        if settings is not None:
            attributes: dict[str, Any] = {
                'gen_ai.operation.name': 'realtime',
                'gen_ai.output.type': self._otel_output_type,
            }
            if self._model_name:
                attributes['gen_ai.request.model'] = self._model_name
            if self._agent_name:
                attributes['gen_ai.agent.name'] = self._agent_name
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
            self._finalize_span(settings, span, attributes)
            span.end()
        self._session_span = None
        self._session_span_context = None
        self._session_span_attributes = None

    @staticmethod
    def _record_span_error(span: Span, error: BaseException) -> None:
        if span.is_recording():
            span.record_exception(error, escaped=True)
            span.set_status(StatusCode.ERROR)

    def all_messages(self) -> list[ModelMessage]:
        """A snapshot of the full conversation: the seeded history plus everything from this session.

        Returns a copy, so the result doesn't change as the session continues. Feed it into
        [`Agent.run(message_history=...)`][pydantic_ai.agent.AbstractAgent.run] to hand the
        conversation off to a standard agent run.
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
        sequence of these inputs, dispatched in order. All
        Inputs preserve the same history bookkeeping and model-profile guards as the dedicated
        control methods.

        [`ToolResult`][pydantic_ai.realtime.ToolResult] is deliberately excluded (`RealtimeSessionInput`
        is [`RealtimeInput`][pydantic_ai.realtime.RealtimeInput] minus `ToolResult`): the session sends
        tool results itself as each tool completes (see `_execute_tool`).
        """
        if isinstance(content, str):
            await self._connection.send(TextInput(text=content))
            self._history.append(ModelRequest(parts=[UserPromptPart(content=content)]))
        elif isinstance(content, BinaryContent):
            if content.is_image:
                self._require_capability(self._profile.get('supports_image_input', False), 'send', 'image input')
                await self._connection.send(ImageInput(data=content.data, mime_type=content.media_type))
            elif content.is_audio:
                await self.send_audio(content.data)
            else:
                raise UserError(
                    f'Unsupported binary media type {content.media_type!r} for `session.send()`. '
                    'Only image and audio content are supported.'
                )
        elif isinstance(content, AudioInput):
            await self.send_audio(content.data)
        elif isinstance(content, TextInput):
            await self._connection.send(content)
            self._history.append(ModelRequest(parts=[UserPromptPart(content=content.text)]))
        elif isinstance(content, ImageInput):
            self._require_capability(self._profile.get('supports_image_input', False), 'send', 'image input')
            await self._connection.send(content)
        elif isinstance(content, CommitAudio):
            await self.commit_audio()
        elif isinstance(content, ClearAudio):
            await self.clear_audio()
        elif isinstance(content, CreateResponse):
            await self.create_response()
        elif isinstance(content, TruncateOutput):
            await self.interrupt(audio_end_ms=content.audio_end_ms)
        elif isinstance(content, CancelResponse):
            await self.interrupt()
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

    async def send_audio(self, data: bytes) -> None:
        """Stream a chunk of audio to the model."""
        await self._connection.send(AudioInput(data=data))
        if self._retain_input:
            # Buffer the raw input so the finalized user turn can retain it. Alignment with the
            # transcript is approximate (see `audio_retention`).
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
        # Drop the locally retained copy too (with `audio_retention='input'`/`'both'`), or the discarded
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

    def _ensure_active_assistant(self, *, output_text: bool = False) -> list[RealtimeEvent]:
        """Start an assistant output part if one isn't already in flight.

        `output_text` selects a plain [`TextPart`][pydantic_ai.messages.TextPart] (the model's
        `output_modalities=('text',)` responses) over the default
        [`SpeechPart`][pydantic_ai.messages.SpeechPart] (spoken audio and its transcript).
        """
        if self._active_assistant is not None:
            return []
        self._ensure_chat_span()
        part: SpeechPart | TextPart = (
            TextPart(content='') if output_text else SpeechPart(speaker='assistant', transcript='')
        )
        self._active_assistant = part
        self._active_assistant_index = len(self._response_parts)
        self._assistant_transcript = ''
        return [PartStartEvent(index=self._active_assistant_index, part=part)]

    def _handle_assistant_transcript(self, text: str, *, output_text: bool = False) -> list[RealtimeEvent]:
        events = self._ensure_active_assistant(output_text=output_text)
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

    def _handle_assistant_audio(self, data: bytes) -> list[RealtimeEvent]:
        events = self._ensure_active_assistant()
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
                part = replace(part, audio=BinaryContent(data=bytes(self._output_audio), media_type=_PCM_MEDIA_TYPE))
        index = self._active_assistant_index
        self._active_assistant = None
        self._assistant_transcript = ''
        self._output_audio.clear()
        if part.has_content():
            self._response_parts.append(part)
        return [PartEndEvent(index=index, part=part)]

    def _finalize_response(self) -> None:
        """Finalize the current assistant response's parts into a `ModelResponse` in history."""
        response: ModelResponse | None = None
        # The chat span's input is the history the response replied to, captured before we append it.
        input_messages = self.all_messages()
        # Native tool parts (web grounding / code execution) lead the response (call+return, then
        # speech), matching the classic `GoogleModel`, which prepends them ahead of the assistant's text.
        parts = [*self._native_tool_parts, *self._response_parts]
        if parts:
            response = ModelResponse(
                parts=parts,
                usage=self._pending_response_usage,
                model_name=self._model_name,
            )
            self._history.append(response)
        self._end_chat_span(input_messages, response)
        self._response_parts = []
        self._native_tool_parts = []
        self._pending_response_usage = RequestUsage()

    def _ensure_chat_span(self) -> None:
        """Open a `chat {model}` span for the assistant response now being assembled, if not already open.

        A realtime turn isn't a single request/response, so the honest lifetime of a `chat` span is one
        assistant `ModelResponse`: it opens when that response's first content arrives (the first
        assistant part or tool call) and closes in `_finalize_response`. Tool calls split a turn into
        multiple responses (mirroring a classic run), so each response gets its own span. The span is
        deliberately *not* entered as the current span: `execute_tool` spans run after the response is
        finalized and stay siblings under the session span, matching the classic agent-run tree.

        Attributes are limited to what a realtime session can report honestly. Omitted vs. the classic
        `chat` span (`open_model_request_span`): `gen_ai.provider.name`/`gen_ai.system` and
        `server.address`/`server.port` (the session has only a model name, no provider/base URL);
        `model_request_parameters`, `gen_ai.tool.definitions`, and `gen_ai.request.*` settings (no
        per-turn request parameters or settings); `operation.cost` (cost needs the provider). The
        response-side `gen_ai.response.id`/`finish_reasons` are simply absent from realtime responses.
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
        if response is not None and span.is_recording():
            # Reuse the exact message → gen_ai serialization and response-attribute helpers the
            # instrumented model uses, so realtime `chat` spans can't drift from the classic path.
            settings.handle_messages(input_messages, response, span)
            span.set_attributes(response_attributes(response, response.model_name or self._model_name))
        span.end()

    def _handle_turn_complete(self, event: RealtimeEvent) -> list[RealtimeEvent]:
        # Turn boundary for a user turn that wasn't finalized earlier, so history reads user-then-assistant.
        # Gemini emits neither `InputSpeechEndEvent` nor a final (`is_final`) input transcript — it streams
        # only partial transcripts — so its user turn is finalized here: `_finalize_user` for a
        # transcript-driven turn, `_finalize_audio_only_user` for a retained-audio-only one. Both are no-ops
        # when the turn was already finalized (e.g. OpenAI's `is_final` transcript or `commit_audio`).
        events = self._finalize_user()
        events.extend(self._finalize_audio_only_user())
        events.extend(self._finalize_assistant_part())
        self._finalize_response()
        events.append(event)
        return events

    def _handle_tool_call_part(self, call_part: ToolCallPart) -> list[RealtimeEvent]:
        """Fold a tool call into the current response and close it out (its result follows in a request)."""
        self._ensure_chat_span()
        events = self._finalize_assistant_part()
        index = len(self._response_parts)
        events.append(PartStartEvent(index=index, part=call_part))
        events.append(PartEndEvent(index=index, part=call_part))
        self._response_parts.append(call_part)
        self._finalize_response()
        return events

    def _complete_tool_call(self, call_part: ToolCallPart, result: str) -> list[RealtimeEvent]:
        return_part = ToolReturnPart(tool_name=call_part.tool_name, content=result, tool_call_id=call_part.tool_call_id)
        self._insert_tool_return(call_part, ModelRequest(parts=[return_part]))
        return [FunctionToolResultEvent(part=return_part)]

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
                while insert_at < len(self._history) and _is_tool_return_request(self._history[insert_at]):
                    insert_at += 1
                self._history.insert(insert_at, request)
                return
        # The calling response is always in history (`_handle_tool_call_part` finalized it before the
        # tool started), so this is unreachable; append rather than drop the result if it ever isn't.
        self._history.append(request)  # pragma: no cover

    def _handle_input_transcript(self, text: str, is_final: bool) -> list[RealtimeEvent]:
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

    def _finalize_user(self) -> list[RealtimeEvent]:
        if self._active_user is None:
            return []  # pragma: no cover
        part = self._active_user
        if part.transcript == '':
            part = replace(part, transcript=None)
        if self._retain_input and self._input_audio:
            part = replace(part, audio=BinaryContent(data=bytes(self._input_audio), media_type=_PCM_MEDIA_TYPE))
        self._active_user = None
        self._user_transcript = ''
        self._input_audio.clear()
        if part.has_content():
            self._history.append(ModelRequest(parts=[part]))
        return [PartEndEvent(index=0, part=part)]

    def _finalize_audio_only_user(self) -> list[RealtimeEvent]:
        """Finalize a user turn from retained input audio when no transcript will arrive.

        With input transcription disabled but input audio retained (`audio_retention='input'`/`'both'`),
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
            audio=BinaryContent(data=bytes(self._input_audio), media_type=_PCM_MEDIA_TYPE),
        )
        self._input_audio.clear()
        self._history.append(ModelRequest(parts=[part]))
        # No deltas to stream (there's no transcript), so bracket the turn with just start/end so a
        # streaming consumer still sees the user turn boundary.
        return [PartStartEvent(index=0, part=part), PartEndEvent(index=0, part=part)]

    def _translate_event(self, event: _TranslatableEvent) -> list[RealtimeEvent]:
        """Translate a low-level codec event into shared session events, building history as a side effect.

        Tool calls and usage are handled in `_handle_pump_event` (they interact with the queue and
        tool execution); everything else routes through here. `event` is typed as `_TranslatableEvent`
        (the pump-consumed variants narrowed out) so the final `assert_never` gives static exhaustiveness.
        """
        if isinstance(event, AudioDelta):
            return self._handle_assistant_audio(event.data)
        if isinstance(event, Transcript):
            # `is_final` doesn't end the part — the turn ends on `TurnCompleteEvent`; a final transcript just
            # carries the full text, which `_accumulate_transcript` reconciles against the deltas. Plain
            # text output (`output_text`) becomes a `TextPart`, an audio transcript a `SpeechPart`.
            return self._handle_assistant_transcript(event.text, output_text=event.output_text)
        if isinstance(event, InputTranscript):
            return self._handle_input_transcript(event.text, event.is_final)
        if isinstance(event, InputSpeechEndEvent):
            # The user's speech segment ended (server VAD). Finalize an audio-only user turn from retained
            # input audio if transcription is off (a no-op otherwise), then pass the boundary event through.
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

    def _finalize_span(self, settings: InstrumentationSettings, span: Span, base_attributes: dict[str, Any]) -> None:
        """Attach cumulative usage and the conversation transcript (as gen_ai messages) to the session span."""
        # Report cumulative usage under `gen_ai.aggregated_usage.*` (mirroring the classic agent-run
        # span) so backends that sum span attributes don't double-count it against the per-turn `chat`
        # spans, which carry each response's usage under `gen_ai.usage.*`. Shared with the classic span.
        attributes: dict[str, Any] = dict(settings.aggregated_usage_attributes(self.usage))
        message_attributes: dict[str, str] = {}
        if settings.include_content:
            # Reuse the same message → gen_ai serialization the instrumented model uses. User/tool
            # requests land as input messages; assistant responses as output messages.
            requests: list[ModelMessage] = [m for m in self._history if isinstance(m, ModelRequest)]
            responses: list[ModelMessage] = [m for m in self._history if isinstance(m, ModelResponse)]
            if requests:
                message_attributes['gen_ai.input.messages'] = safe_to_json(
                    settings.messages_to_otel_messages(requests)
                ).decode()
            if responses:
                message_attributes['gen_ai.output.messages'] = safe_to_json(
                    settings.messages_to_otel_messages(responses)
                ).decode()
        if message_attributes:
            # `logfire.json_schema` marks the message attributes as JSON arrays so the Logfire UI
            # deserializes and renders them as a conversation rather than as strings — matching the
            # classic agent-run span and `InstrumentationSettings.handle_messages` on `chat` spans.
            attributes.update(message_attributes)
            attributes['logfire.json_schema'] = pydantic_core.to_json(
                {'type': 'object', 'properties': {key: {'type': 'array'} for key in message_attributes}}
            ).decode()
        span.set_attributes(attributes)
        for token_type, tokens in (('input', self.usage.input_tokens), ('output', self.usage.output_tokens)):
            if tokens:
                settings.tokens_histogram.record(tokens, {**base_attributes, 'gen_ai.token.type': token_type})

    async def _execute_tool(self, call: ToolCall) -> str:
        # No `execute_tool` span is created here: the `execute_tool` span is owned by the
        # `Instrumentation` capability's `wrap_tool_execute` hook, which `Agent.realtime_session`
        # injects into the tool runner's `ToolManager` (mirroring a classic run). That capability
        # span is the single, canonical source of tool spans; the pump task runs inside the session
        # span's OTel context, so the capability's tool span nests under the session span as a sibling
        # of the `chat` spans. The session-level `realtime` span and per-response `chat` spans below
        # stay hand-managed for now — they move onto exchange-level capability hooks when those land.
        args, error = _parse_tool_args(call.args)
        if error is not None:
            result = error
        else:
            assert args is not None
            async with self._tool_manager_lock:
                ctx = self._tool_manager.ctx
                if ctx is not None and ctx.run_step < self._tool_run_step:
                    self._tool_manager = await self._tool_manager.for_run_step(
                        replace(ctx, run_step=self._tool_run_step)
                    )
            tool_call = ToolCallPart(tool_name=call.tool_name, args=args, tool_call_id=call.tool_call_id)
            try:
                tool_result = await self._tool_manager.handle_call(tool_call)
            except ToolRetryError as e:
                result = e.tool_retry.model_response()
            except Exception as e:
                result = f'Error: {e}'
            else:
                if isinstance(tool_result, ToolDenied):
                    result = tool_result.message
                elif isinstance(tool_result, ToolReturn):
                    # A realtime tool result travels back over the provider's string-only tool-output
                    # channel, so — unlike the graph's `_call_tool`, which surfaces `ToolReturn.content`
                    # as extra model-facing content and keeps `metadata` — only the `return_value` is
                    # sent; `content` / `metadata` are dropped. A limitation of the provider channel.
                    result = str(tool_result.return_value)
                else:
                    result = str(tool_result)
        await self._connection.send(ToolResult(tool_call_id=call.tool_call_id, output=result))
        return result

    # --- streaming --------------------------------------------------------------------------------

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

    async def _run_tool(self, call: ToolCall, call_part: ToolCallPart) -> None:
        """Run a tool and feed its completion (or failure) back through the queue."""
        try:
            result = await self._execute_tool(call)
        except Exception as e:
            # Surface the failure through the queue so the consumer re-raises it, instead of letting it
            # vanish into `__aexit__`'s cleanup-only drain and hang the session on a completion that
            # never arrives.
            await self._queue.put(e)
            return
        finally:
            # Settled (completed, failed, or cancelled): no longer cancellable by `ToolCallCancelled`.
            self._pending_tool_calls.pop(call_part.tool_call_id, None)
        for event in self._complete_tool_call(call_part, result):
            await self._queue.put(event)

    def _tool_task_done(self, task: asyncio.Task[None]) -> None:
        self._background_tasks.discard(task)
        # Wake the queue reader so it can finish once both the pump and the last tool are done.
        self._queue.put_nowait(self._queue_changed)

    async def _handle_pump_event(
        self,
        event: RealtimeCodecEvent,
    ) -> bool:
        """Process one upstream event onto the queue; return `True` to stop the pump (a limit tripped)."""
        if isinstance(event, ToolCall):
            self._check_tool_call_limit()
            self.usage.tool_calls += 1
            call_part = ToolCallPart(tool_name=event.tool_name, args=event.args, tool_call_id=event.tool_call_id)
            for out in self._handle_tool_call_part(call_part):
                await self._queue.put(out)
            await self._queue.put(FunctionToolCallEvent(part=call_part))
            task = asyncio.create_task(self._run_tool(event, call_part))
            self._background_tasks.add(task)
            self._pending_tool_calls[call_part.tool_call_id] = (task, call_part)
            task.add_done_callback(self._tool_task_done)
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
                for out in self._complete_tool_call(call_part, _CANCELLED_TOOL_RESULT):
                    await self._queue.put(out)
            return False
        if isinstance(event, SessionUsageEvent):
            self.usage.incr(event.usage)
            self.usage.requests += 1
            self._pending_response_usage = self._pending_response_usage + event.usage
            self._check_token_limit()
            return False
        if isinstance(event, TurnCompleteEvent):
            self._tool_run_step += 1
        for out in self._translate_event(event):
            await self._queue.put(out)
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
