"""A realtime session that wraps a [`RealtimeConnection`][pydantic_ai.realtime.RealtimeConnection] with automatic tool execution."""

from __future__ import annotations as _annotations

import asyncio
import dataclasses
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import pydantic_core
from anyio import Lock
from opentelemetry.trace import Span, SpanKind
from typing_extensions import assert_never

from .._instrumentation import response_attributes, safe_to_json
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
    """
    if accumulated and text.startswith(accumulated):
        return text, text[len(accumulated) :]
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
                self._require_capability(self._profile['supports_image_input'], 'send', 'image input')
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
            self._require_capability(self._profile['supports_image_input'], 'send', 'image input')
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
        self._require_capability(self._profile['supports_manual_turn_control'], 'commit_audio', 'manual turn-taking')
        await self._connection.send(CommitAudio())

    async def clear_audio(self) -> None:
        """Discard buffered, uncommitted input audio."""
        self._require_capability(self._profile['supports_manual_turn_control'], 'clear_audio', 'manual turn-taking')
        await self._connection.send(ClearAudio())
        # Drop the locally retained copy too (with `audio_retention='input'`/`'both'`), or the discarded
        # audio would still be attached to the next finalized user turn.
        self._input_audio.clear()

    async def create_response(self) -> None:
        """Ask the model to respond now (manual turn-taking, after `commit_audio`)."""
        self._require_capability(self._profile['supports_manual_turn_control'], 'create_response', 'manual turn-taking')
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
        self._require_capability(self._profile['supports_interruption'], 'interrupt', 'interruption')
        # Truncate before cancelling: cancellation triggers `response.done`, which clears the tracked
        # output item, so a truncate sent afterwards could no-op.
        if audio_end_ms is not None:
            if not self._profile['supports_output_truncation']:
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
        """
        settings = self._instrumentation
        if settings is None or self._chat_span is not None:
            return
        attributes: dict[str, Any] = {'gen_ai.operation.name': 'chat'}
        if self._model_name:
            attributes['gen_ai.request.model'] = self._model_name
        name = f'chat {self._model_name}' if self._model_name else 'chat'
        self._chat_span = settings.tracer.start_span(name, attributes=attributes, kind=SpanKind.CLIENT)

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
        # Catch-all boundary for an audio-only user turn on providers that don't emit `InputSpeechEndEvent`
        # (e.g. Gemini): finalize it before the assistant response so history reads user-then-assistant.
        # A no-op when `InputSpeechEndEvent`/`commit_audio` already finalized it (nothing left retained).
        events = self._finalize_audio_only_user()
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
                return []
            raise RealtimeError(event.message)
        assert_never(event)

    # --- instrumentation --------------------------------------------------------------------------

    async def __aiter__(self) -> AsyncIterator[RealtimeEvent]:
        settings = self._instrumentation
        if settings is None:
            async for event in self._stream():
                yield event
            return
        # Open a session-level span; tool spans created in the pump task inherit it through the
        # OpenTelemetry context that `asyncio.create_task` copies.
        attributes: dict[str, Any] = {'gen_ai.operation.name': 'realtime'}
        if self._model_name:
            attributes['gen_ai.request.model'] = self._model_name
        if self._agent_name:
            attributes['gen_ai.agent.name'] = self._agent_name
        if self._conversation_id:
            # Match the classic agent-run span's key (see `capabilities/instrumentation.py`) so a
            # realtime session can be correlated with other runs sharing the conversation id.
            attributes['gen_ai.conversation.id'] = self._conversation_id
        span_name = f'realtime {self._model_name}' if self._model_name else 'realtime'
        with settings.tracer.start_as_current_span(span_name, attributes=attributes, kind=SpanKind.CLIENT) as span:
            try:
                async for event in self._stream():
                    yield event
            finally:
                # If the consumer broke/cancelled mid-turn, an assistant response was started but never
                # finalized, leaving its `chat` span open. End it here so it doesn't outlive the session
                # span as an unfinished child (no `ModelResponse` to attach — it was cut off).
                if self._chat_span is not None:
                    self._chat_span.end()
                    self._chat_span = None
                self._finalize_span(settings, span, attributes)

    def _finalize_span(self, settings: InstrumentationSettings, span: Span, base_attributes: dict[str, Any]) -> None:
        """Attach cumulative usage and the conversation transcript (as gen_ai messages) to the session span."""
        # Report cumulative usage under `gen_ai.aggregated_usage.*` (mirroring the classic agent-run
        # span) so backends that sum span attributes don't double-count it against the per-turn `chat`
        # spans, which carry each response's usage under `gen_ai.usage.*`. Shared with the classic span.
        span.set_attributes(settings.aggregated_usage_attributes(self.usage))
        if settings.include_content:
            # Reuse the same message → gen_ai serialization the instrumented model uses. User/tool
            # requests land as input messages; assistant responses as output messages.
            requests: list[ModelMessage] = [m for m in self._history if isinstance(m, ModelRequest)]
            responses: list[ModelMessage] = [m for m in self._history if isinstance(m, ModelResponse)]
            if requests:
                span.set_attribute(
                    'gen_ai.input.messages', safe_to_json(settings.messages_to_otel_messages(requests)).decode()
                )
            if responses:
                span.set_attribute(
                    'gen_ai.output.messages', safe_to_json(settings.messages_to_otel_messages(responses)).decode()
                )
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

    async def _run_tool(
        self, call: ToolCall, call_part: ToolCallPart, queue: asyncio.Queue[RealtimeEvent | object]
    ) -> None:
        """Run a tool and feed its completion (or failure) back through the queue."""
        try:
            result = await self._execute_tool(call)
        except Exception as e:
            # Surface the failure through the queue so the consumer re-raises it, instead of letting it
            # vanish into the final `gather(..., return_exceptions=True)` and hang the session on a
            # completion that never arrives.
            await queue.put(e)
            return
        for event in self._complete_tool_call(call_part, result):
            await queue.put(event)

    async def _handle_pump_event(
        self,
        event: RealtimeCodecEvent,
        queue: asyncio.Queue[RealtimeEvent | object],
        background: set[asyncio.Task[None]],
    ) -> bool:
        """Process one upstream event onto the queue; return `True` to stop the pump (a limit tripped)."""
        if isinstance(event, ToolCall):
            self._check_tool_call_limit()
            self.usage.tool_calls += 1
            call_part = ToolCallPart(tool_name=event.tool_name, args=event.args, tool_call_id=event.tool_call_id)
            for out in self._handle_tool_call_part(call_part):
                await queue.put(out)
            await queue.put(FunctionToolCallEvent(part=call_part))
            task = asyncio.create_task(self._run_tool(event, call_part, queue))
            background.add(task)
            task.add_done_callback(background.discard)
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
            await queue.put(out)
        return False

    async def _stream(self) -> AsyncIterator[RealtimeEvent]:
        # Both the upstream connection and finished tools feed a single queue, so a tool completion
        # wakes the consumer immediately instead of waiting for the next
        # provider event (which may never come while the model is idle).
        queue: asyncio.Queue[RealtimeEvent | object] = asyncio.Queue()
        closed = object()  # sentinel: the upstream connection has been fully drained
        background: set[asyncio.Task[None]] = set()
        pump_error: Exception | None = None

        async def pump() -> None:
            nonlocal pump_error
            try:
                async for event in self._connection:
                    if await self._handle_pump_event(event, queue, background):
                        return  # a usage limit tripped: stop reading the upstream
            except Exception as e:
                pump_error = e
            finally:
                await queue.put(closed)

        pump_task = asyncio.create_task(pump())
        try:
            while True:
                item = await queue.get()
                if item is closed:
                    break
                yield _as_event(item)  # re-raises if a tool failed
            # Upstream is done: wait for any in-flight tools, then flush their completions.
            if background:
                await asyncio.gather(*background, return_exceptions=True)
            while not queue.empty():
                yield _as_event(queue.get_nowait())
            if pump_error is not None:
                raise pump_error
        finally:
            pump_task.cancel()
            for task in list(background):
                task.cancel()
            await asyncio.gather(pump_task, *background, return_exceptions=True)
