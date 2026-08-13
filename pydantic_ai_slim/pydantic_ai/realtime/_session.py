"""A realtime session that wraps a [`RealtimeConnection`][pydantic_ai.realtime.codec.RealtimeConnection] with automatic tool execution."""

from __future__ import annotations as _annotations

import asyncio
import dataclasses
import io
import wave
from collections.abc import AsyncIterable, AsyncIterator, Callable, Sequence
from dataclasses import dataclass, replace
from threading import Lock as ThreadLock
from time import time_ns
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypeVar, cast, overload

from anyio import Lock
from opentelemetry import context as otel_context
from opentelemetry.context import Context
from typing_extensions import TypeAliasType, assert_never

from .. import _agent_graph
from .._enqueue import PendingMessage, PendingMessagePriority
from .._tool_execution import (
    _reject_unloaded_capability_reveals,  # pyright: ignore[reportPrivateUsage]
    build_tool_return_part,
    cancelled_sub_agent_return,
)
from .._utils import aclose_all, cancel_and_drain, dataclasses_no_defaults_repr, fill_run_metadata
from ..exceptions import ApprovalRequired, CallDeferred, RunCancelled, ToolFailedError, ToolRetryError, UserError
from ..messages import (
    INTERRUPTED_TOOL_RETURN_CONTENT,
    BinaryAudio,
    BinaryContent,
    BinaryImage,
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
    RealtimeInputSpeechEndEvent,
    RealtimeInputSpeechStartEvent,
    RealtimeInputTranscriptionErrorEvent,
    RealtimeOutputSpeechEndEvent,
    RealtimeOutputSpeechStartEvent,
    RealtimeResponseInterruptedEvent,
    RealtimeSessionErrorEvent,
    RealtimeSessionReconnectEvent,
    RealtimeTurnCompleteEvent,
    RetryPromptPart,
    SpeechPart,
    SpeechPartDelta,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
)
from ..native_tools import SUPPORTED_NATIVE_TOOLS
from ..run import AgentRunResult
from ..tool_manager import ToolManager
from ..usage import RequestUsage, RunUsage, UsageLimits
from ._instrumentation import (
    SessionInstrumentation,
)
from ._utils import seed_pcm_audio
from .codec import (
    AudioDelta,
    CancelResponse,
    ClearAudio,
    CommitAudio,
    ConversationCreated,
    ConversationItemCreated,
    CreateResponse,
    InputTranscript,
    OutputTranscript,
    RealtimeCodecEvent,
    RealtimeConnection,
    RealtimeInput,
    RealtimeSessionInput,
    ResponseDone,
    SessionUsage,
    ToolCall,
    ToolCallCancelled,
    ToolResult,
    TruncateOutput,
)
from .model import RealtimeError
from .profiles import DEFAULT_AUDIO_SAMPLE_RATE, RealtimeModelProfile
from .settings import AudioRetention, RealtimeModelSettings

if TYPE_CHECKING:
    from ..messages import AgentStreamEvent
    from ..models import ModelRequestParameters
    from ..models.instrumented import InstrumentationSettings
    from ..tools import DeferredToolRequests, DeferredToolResults
    from .model import RealtimeModel

# Session-level events (yielded by `RealtimeSession.__aiter__`).
#
# A session translates the low-level codec events into the shared message/part event vocabulary from
# `pydantic_ai.messages`: `AudioDelta`/`OutputTranscript`/`InputTranscript` become `PartStartEvent` /
# `PartDeltaEvent` / `PartEndEvent` for `SpeechPart`s, and `ToolCall` becomes a
# `ToolCallPart` part (start/end) plus `FunctionToolCallEvent` / `FunctionToolResultEvent` around its
# execution. Some control-plane events pass through unchanged.


RealtimeEvent = TypeAliasType(
    'RealtimeEvent',
    PartStartEvent
    | PartDeltaEvent
    | PartEndEvent
    | FunctionToolCallEvent
    | FunctionToolResultEvent
    | DeferredToolRequestsEvent
    | DeferredToolResultsEvent
    | RealtimeTurnCompleteEvent
    | RealtimeInputSpeechStartEvent
    | RealtimeResponseInterruptedEvent
    | RealtimeInputSpeechEndEvent
    | RealtimeOutputSpeechStartEvent
    | RealtimeOutputSpeechEndEvent
    | RealtimeInputTranscriptionErrorEvent
    | RealtimeSessionReconnectEvent
    | RealtimeSessionErrorEvent,
)
"""Union of events yielded by [`RealtimeSession`][pydantic_ai.realtime.RealtimeSession].

This is a strict subset of [`AgentStreamEvent`][pydantic_ai.messages.AgentStreamEvent].

Content is streamed as the shared [`PartStartEvent`][pydantic_ai.messages.PartStartEvent] /
[`PartDeltaEvent`][pydantic_ai.messages.PartDeltaEvent] / [`PartEndEvent`][pydantic_ai.messages.PartEndEvent]
events (carrying [`SpeechPart`][pydantic_ai.messages.SpeechPart]s and
[`ToolCallPart`][pydantic_ai.messages.ToolCallPart]s), tool execution as
[`FunctionToolCallEvent`][pydantic_ai.messages.FunctionToolCallEvent] /
[`FunctionToolResultEvent`][pydantic_ai.messages.FunctionToolResultEvent], inline deferred resolution
as [`DeferredToolRequestsEvent`][pydantic_ai.messages.DeferredToolRequestsEvent] /
[`DeferredToolResultsEvent`][pydantic_ai.messages.DeferredToolResultsEvent], and the rest as realtime
control-plane events.
"""


@dataclass(frozen=True, repr=False, kw_only=True)
class TranscriptUpdate:
    """One incremental transcript update, carrying everything needed to render it.

    Yielded by [`RealtimeSession.stream_transcripts(delta=True)`][pydantic_ai.realtime.RealtimeSession.stream_transcripts].
    A realtime session is duplex, so both speakers' transcripts stream at the same time and a caption
    UI needs to know not just *what* was said but *which* turn to put it in — otherwise two
    consecutive turns by the same speaker run together.
    """

    index: int
    """Identifies the turn this update belongs to, stable for the life of the session.

    Use it as the key for whatever you render a turn into: every update with the same `index` belongs
    to the same speech part.
    """

    speaker: Literal['user', 'assistant']
    """Who is speaking."""

    delta: str
    """The text this update added, when it added any.

    Empty when the provider *revised* the turn instead of extending it — speech recognition is
    revisable, and a correction can't be expressed as an addition. Render `transcript` and this never
    matters.
    """

    transcript: str
    """The full transcript of this turn so far.

    Render this, keyed on `index`, and captions are correct whatever the provider does: no
    accumulating, no special case for a revision, and a dropped update (if a consumer fell behind)
    self-corrects on the next one.
    """

    __repr__ = dataclasses_no_defaults_repr


@dataclass
class _UserTurn:
    part: SpeechPart
    transcript: str
    index: int
    finalized: bool = False


# Realtime providers stream raw PCM audio, but retained history uses a WAV container so the sample
# format is self-describing and portable to classic model adapters. Live `SpeechPartDelta.audio_chunk`
# values remain raw PCM.
_WAV_MEDIA_TYPE = 'audio/wav'
# Marks every span this session emits so the Logfire UI (and any consumer) can recognize realtime
# activity without parsing span names.

# Fallback for a session created without a model's profile (e.g. directly, in tests): assume
# everything is supported so no guard fires. Real sessions receive `model.profile`. Native tools are
# validated up front by `Agent.realtime`, not the session, so this field is inert here.
_FULL_PROFILE = RealtimeModelProfile(
    supports_image_input=True,
    supports_text_output=True,
    supports_manual_turn_control=True,
    supports_interruption=True,
    supports_output_truncation=True,
    supports_session_seeding=True,
    supported_native_tools=SUPPORTED_NATIVE_TOOLS,
    audio_input_sample_rate=DEFAULT_AUDIO_SAMPLE_RATE,
    audio_output_sample_rate=DEFAULT_AUDIO_SAMPLE_RATE,
)

# Audio chunks are kilobytes apiece, so a slow player is bounded tightly. Transcript items are short
# strings, and dropping one silently corrupts the text a user is reading rather than causing an
# audible glitch, so they get a far deeper window for the same trivial cost.
_AUDIO_TAP_SIZE = 32
_TRANSCRIPT_TAP_SIZE = 512
_TapItem = TypeVar('_TapItem')


def _put_tap(queue: asyncio.Queue[_TapItem], item: _TapItem) -> int:
    """Put without blocking the pump, retaining the newest bounded window.

    The queue is sized one over its data window so the completion sentinel always fits. Returns `1`
    when an older item was dropped, otherwise `0`.
    """
    dropped = 0
    if queue.qsize() >= queue.maxsize - 1:
        queue.get_nowait()
        dropped = 1
    queue.put_nowait(item)
    return dropped


# The `RealtimeEvent` variants that `_translate_event` handles: the full union minus `ToolCall` and
# `SessionUsage`, which `_handle_pump_event` peels off first (they drive tool execution and usage
# accounting before delegating). Splitting the union lets `_translate_event` end in `assert_never`, so
# a new non-pump variant added to `RealtimeEvent` is caught at type-check time — either the call site
# (where the residual no longer fits this alias) or the `assert_never` flags it.
_TranslatableEvent: TypeAlias = (
    AudioDelta
    | OutputTranscript
    | InputTranscript
    | ResponseDone
    | RealtimeInputSpeechStartEvent
    | RealtimeResponseInterruptedEvent
    | RealtimeInputSpeechEndEvent
    | RealtimeOutputSpeechStartEvent
    | RealtimeOutputSpeechEndEvent
    | RealtimeInputTranscriptionErrorEvent
    | RealtimeSessionReconnectEvent
    | PartStartEvent
    | PartEndEvent
    | RealtimeSessionErrorEvent
)
_SettledToolResult: TypeAlias = tuple[ToolReturnPart | RetryPromptPart, str | Sequence[UserContent] | None]


def _as_event(item: object) -> RealtimeEvent:
    """Unwrap a queue item: re-raise a tool's exception, otherwise return the event."""
    if isinstance(item, BaseException):
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


def _user_transcript_update(previous: str, text: str, *, cumulative: bool) -> tuple[str, SpeechPartDelta | None]:
    """Fold a user transcript event into the running text, returning it with the delta to emit.

    An incremental piece is accumulated by [`_accumulate_transcript`][pydantic_ai.realtime._session._accumulate_transcript]
    and surfaced as an appended delta. A cumulative snapshot is adopted wholesale, because a provider
    that sends snapshots may revise earlier words rather than only extend them: when it merely extends,
    the new suffix is still an appended delta (what a live transcript wants), but a revision can't be
    expressed by appending, so it goes out as a replacement instead. `None` when nothing changed.
    """

    def delta(transcript: str, added: str) -> SpeechPartDelta:
        return SpeechPartDelta(speaker='user', transcript_delta=added, transcript=transcript)

    if not cumulative:
        transcript, appended = _accumulate_transcript(previous, text)
        return transcript, delta(transcript, appended) if appended else None
    if text == previous:
        return previous, None
    if previous and text.startswith(previous):
        return text, delta(text, text[len(previous) :])
    stripped = previous.strip()
    if stripped and (stripped_text := text.strip()).startswith(stripped):
        return text, delta(text, stripped_text[len(stripped) :])
    if not previous:
        return text, delta(text, text)
    # A revision: nothing was *added*, so only the corrected whole is reported.
    return text, delta(text, '')


def _tool_result_call_id(message: ModelMessage) -> str | None:
    """The id of the call a history request holds the result for, or `None` if it holds no result.

    An inserted tool result leads its request and may be followed by user content.
    """
    if not isinstance(message, ModelRequest) or not message.parts:
        return None
    result = message.parts[0]
    if not isinstance(result, (ToolReturnPart, RetryPromptPart)) or not all(
        isinstance(part, (ToolReturnPart, RetryPromptPart, UserPromptPart)) for part in message.parts
    ):
        return None
    return result.tool_call_id


def _is_tool_result_request(message: ModelMessage) -> bool:
    """Whether a history request carries an inserted tool result and optional follow-up user content."""
    return _tool_result_call_id(message) is not None


def _build_session_tool_return(
    tool_result: Any, call_part: ToolCallPart, tool_manager: ToolManager[Any]
) -> tuple[ToolReturnPart | RetryPromptPart, str | Sequence[UserContent] | None]:
    """Translate a settled session tool result into its history part, rejecting mid-session reveals."""
    tool_def = tool_manager.get_tool_def(call_part.tool_name)
    result_part, user_content, tools_added = build_tool_return_part(
        tool_result,
        call=call_part,
        tool_kind=tool_def.tool_kind if tool_def else None,
    )
    if tools_added:
        # Mirrors the graph (`_tool_execution._call_tool`), which rejects only a name owned by an
        # unloaded capability and treats every other name — a typo, an already-visible tool — as a
        # silent no-op. Killing the session over any reveal at all was harsher than the graph *and*
        # unreachable-by-design harshness: a session refuses `defer_loading=True` tools and
        # tool-contributing deferred capabilities when it opens, so it holds nothing that a reveal
        # could have surfaced. Nothing is silently lost by dropping the request here.
        _reject_unloaded_capability_reveals(tools_added, tool_manager)
    return result_part, user_content


def _unsettled_call_return(call: ToolCallPart, error: ApprovalRequired | CallDeferred | RunCancelled) -> ToolReturnPart:
    """The failed return a session answers with when a tool call couldn't settle normally.

    Both cases are ones the graph resolves by ending or isolating the run, which a live conversation
    can't do — so each becomes a deliberate explanation the model can voice, marked `'failed'` rather
    than left at the default `'success'`. Recording a refusal as a successful return would be a
    misleading audit trail: `all_messages()` handed to `Agent.run` would read it as a tool that ran.
    """
    if isinstance(error, RunCancelled):
        # Exactly the graph path's settlement (this session's own cancellation arrives as
        # `CancelledError`, which `_run_tool` re-raises untouched) — shared so the two can't drift.
        return cancelled_sub_agent_return(call, error)
    else:
        # `handle_call` already gave the `HandleDeferredToolCalls` capability handler the chance to
        # resolve the deferral inline (approve, deny, retry, or substitute a result); reaching here
        # means no handler resolved it. The graph's fallback — pausing the run with a
        # `DeferredToolRequests` output — has no realtime analog (a live conversation can't wait for an
        # out-of-band result, and the provider expects an answer on the string-only tool channel).
        reason = 'requires approval' if isinstance(error, ApprovalRequired) else 'runs externally'
        content = f'Error: The {call.tool_name!r} tool {reason} and cannot be completed during a realtime session.'
    return ToolReturnPart(
        tool_name=call.tool_name,
        content=content,
        tool_call_id=call.tool_call_id,
        outcome='failed',
    )


def _is_user_speech_request(message: ModelMessage) -> bool:
    """Whether a history request is a transcribed (or audio-only) user speech turn."""
    if not isinstance(message, ModelRequest) or not message.parts:
        return False
    return all(isinstance(part, SpeechPart) and part.speaker == 'user' for part in message.parts)


def _pending_message_text(pending: PendingMessage) -> str:
    """Render enqueued messages down to the text a realtime session can deliver.

    Text-only parts join across messages, and a mid-conversation `SystemPromptPart` is delivered as
    `<system>`-tagged user text — the same degradation `Model.prepare_messages` applies for a
    standard run's non-leading system prompts. Anything else can't cross the live input channel.
    """
    error = UserError(
        '`RunContext.enqueue()` in a realtime session supports plain-text prompts and system-prompt '
        'parts only. Multimodal content and model responses cannot be delivered over the live input '
        'channel.'
    )
    texts: list[str] = []
    for message in pending.messages:
        if not isinstance(message, ModelRequest):
            raise error
        for part in message.parts:
            if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                texts.append(part.content)
            elif isinstance(part, SystemPromptPart):
                texts.append(f'<system>{part.content}</system>')
            else:
                raise error
    if not texts:
        raise error
    return '\n\n'.join(texts)


class _RealtimePendingMessages(list[PendingMessage]):
    """A `RunContext.enqueue` queue that validates content and wakes the live session for delivery."""

    def __init__(self) -> None:
        super().__init__()
        self._on_append: Callable[[PendingMessagePriority], None] | None = None
        self._lock = ThreadLock()

    def bind(self, on_append: Callable[[PendingMessagePriority], None]) -> None:
        self._on_append = on_append

    def append(self, pending: PendingMessage) -> None:
        _pending_message_text(pending)
        with self._lock:
            super().append(pending)
        if self._on_append is not None:
            self._on_append(pending.priority)

    def pop_priority(self, priority: PendingMessagePriority) -> list[PendingMessage]:
        """Atomically remove and return all messages with `priority`."""
        with self._lock:
            selected = [pending for pending in self if pending.priority == priority]
            self[:] = [pending for pending in self if pending.priority != priority]
        return selected


class RealtimeSession:
    """Wraps a [`RealtimeConnection`][pydantic_ai.realtime.codec.RealtimeConnection], building message history and auto-executing tools.

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
    high-rate frame streams, and `retain_images_max` (default `100`) to bound how many stay in
    history — the oldest retained image is evicted first, so a long-running stream can't grow memory
    without limit.

    When constructing a session directly, use it as an async context manager. The context owns the
    receive pump, background tool tasks, and instrumentation spans; iteration only reads its event
    queue. [`AgentRealtime.session`][pydantic_ai.agent.AgentRealtime.session] enters the session
    before yielding it, so the usual agent API remains a single `async with` block.
    """

    def __init__(
        self,
        connection: RealtimeConnection,
        *,
        model: RealtimeModel | None = None,
        tool_manager: ToolManager[Any],
        instrumentation: InstrumentationSettings | None = None,
        agent_name: str | None = None,
        usage: RunUsage | None = None,
        usage_limits: UsageLimits | None = None,
        audio_retention: AudioRetention = 'transcript_only',
        retain_images_every_n: int = 1,
        retain_images_max: int | None = 100,
        message_history: Sequence[ModelMessage] | None = None,
        profile: RealtimeModelProfile | None = None,
        owns_media: bool = True,
        conversation_id: str | None = None,
        run_id: str | None = None,
        instructions: str | None = None,
        metadata: dict[str, Any] | None = None,
        agent_description: str | None = None,
        output_modality: Literal['audio', 'text'] = 'audio',
        model_request_parameters: ModelRequestParameters | None = None,
        model_settings: RealtimeModelSettings | None = None,
        wrap_event_stream: Callable[[AsyncIterable[AgentStreamEvent]], AsyncIterable[AgentStreamEvent]] | None = None,
    ) -> None:
        self._connection = connection
        self._tool_manager = tool_manager
        self._tool_run_step = 0
        self._tool_manager_lock = Lock()
        self._instrumentation = instrumentation
        self._profile = profile if profile is not None else model.profile if model is not None else _FULL_PROFILE
        # Whether this session owns the audio transport. `False` for a WebRTC sideband session: the
        # browser exchanges audio with the provider directly, and this connection is only the control
        # plane, so the audio methods are unavailable and no audio bytes flow over it (transcripts still
        # build history). Set by the connect path when a `provider_session` is attached.
        self._owns_media = owns_media
        self._model_name = model.model_name if model is not None else None
        self._provider_name = model.system if model is not None else None
        self._provider_url = model.base_url if model is not None else None
        self._agent_name = agent_name
        self._conversation_id = conversation_id
        self._run_id = run_id
        # The request parameters and settings the session was opened with. Unlike a classic run — where
        # each model request can vary — a realtime session sends these once at connect, so they belong on
        # the session span (set once), not repeated on every per-turn `chat` span. Carrying
        # `model_request_parameters` is what makes the session's configured native tools inspectable.
        self._model_request_parameters = model_request_parameters
        self._model_settings = model_settings
        self._wrap_event_stream = wrap_event_stream
        self._instructions = instructions
        self._metadata = metadata
        self._agent_description = agent_description
        # All OTel span state and construction lives on the helper; the session hands it the static
        # session metadata once and delegates every span operation (see `realtime/_instrumentation.py`).
        self._session_instrumentation = SessionInstrumentation(
            instrumentation,
            agent_name=agent_name,
            agent_description=agent_description,
            model_name=self._model_name,
            provider_name=self._provider_name,
            provider_url=self._provider_url,
            conversation_id=conversation_id,
            run_id=run_id,
            instructions=instructions,
            metadata=metadata,
            model_request_parameters=model_request_parameters,
            model_settings=model_settings,
            output_type='speech' if output_modality == 'audio' else 'text',
        )
        self._usage_limits = usage_limits
        self._audio_retention = audio_retention
        self._retain_input = audio_retention in ('input_audio', 'all')
        self._retain_output = audio_retention in ('output_audio', 'all')
        if retain_images_every_n < 1:
            raise UserError('`retain_images_every_n` must be at least 1.')
        self._retain_images_every_n = retain_images_every_n
        if retain_images_max is not None and retain_images_max < 0:
            raise UserError('`retain_images_max` must be at least 0, or `None` for no limit.')
        self._retain_images_max = retain_images_max
        # Retained image requests in arrival order, so the cap can evict the oldest. Identity-based:
        # `_record_sent_request` stores these same objects in the pending list or the history.
        self._retained_image_requests: list[ModelRequest] = []
        self._sent_image_count = 0
        # Whether the connection transcribes the user's audio. When it doesn't, no `InputTranscript`
        # arrives to finalize a user turn, so its retained audio or content-less placeholder is finalized
        # at the turn boundary (see `_finalize_untranscribed_user`).
        self._input_transcription_enabled = connection.input_transcription_enabled
        self.usage = usage if usage is not None else RunUsage()
        """Cumulative token usage and tool-call counts for the session, updated as events stream in.

        Pass `usage` to [`Agent.realtime`][pydantic_ai.agent.Agent.realtime] to accumulate
        into a shared [`RunUsage`][pydantic_ai.usage.RunUsage]; otherwise a fresh one is used.
        """
        # `ToolManager` increments `tool_calls` on its context's usage as each call succeeds, and the
        # session is the single authority for `session.usage`, so the two have to be the same object.
        # A caller can pass a `usage` the manager wasn't built with, and rebinding the context properly
        # would mean `for_run_step`, which is async and can't run here.
        if (ctx := self._tool_manager.ctx) is not None:
            ctx.usage = self.usage

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
        # When the provider's VAD reported the user's current speech segment starting, in OTel's
        # nanosecond clock, so the `user speech` span can be backdated to it (see
        # `_record_user_speech_span`). `None` while nobody is speaking.
        self._user_speech_started_at: int | None = None
        # Set once the provider draws its first speech-end boundary. Gemini never does, so its
        # retained input is only ever consumed by a turn, never trimmed at one.
        self._provider_segments_input = False
        self._pending_response_usage = RequestUsage()
        self._response_limit_checked = False
        self._pending_response_requests = 0
        self._pending_provider_response_id: str | None = None
        self._pending_finish_reason: FinishReason | None = None
        self._pending_interrupted_at_ms: int | None = None
        self._response_finalized_before_terminal = False
        # User requests sent while a response is in flight are held until that response is finalized,
        # so the pump remains the sole writer for that portion of history and a caller cannot splice a
        # request between an assistant response's streamed parts.
        self._pending_sent_requests: list[ModelRequest] = []
        self._active_assistant: SpeechPart | TextPart | None = None
        self._active_assistant_item_id: str | None = None
        self._active_assistant_index = 0
        self._assistant_transcript = ''
        self._output_audio = bytearray()

        # A realtime session is duplex: the user's part and the model's assemble at the same time, and
        # each response starts a fresh `_response_parts` list. Numbering streamed parts by their
        # position in a message would therefore hand out index 0 to both sides at once, so a consumer
        # couldn't tell whose delta it holds. One monotonic counter per session keeps every streamed
        # part's index unique instead: in realtime, `index` identifies a part in the event stream, not
        # a slot in a message.
        self._next_part_index = 0
        # Connection-supplied native-tool part index -> the session index it was remapped to, so a
        # `PartEndEvent` closes the part its `PartStartEvent` opened. See `_remap_native_part_index`.
        self._native_part_indexes: dict[int, int] = {}

        # In-flight user request being assembled from input-transcript events.
        self._user_turn_active = False
        # Insertion order is provider item order. `None` is the single anonymous turn used by
        # providers that do not identify input transcript items.
        self._user_turns: dict[str | None, _UserTurn] = {}
        self._finalized_user_item_ids: set[str] = set()
        # Where in history each user turn belongs, remembered when the turn *starts* — see
        # `_open_user_turn_anchor`. `_pending_user_turn_anchor` holds the anchor of a turn that has begun
        # but whose transcript hasn't identified it yet; `_user_turn_anchors` keys them by item id (`None`
        # for id-less providers) once it has.
        self._pending_user_turn_anchor: tuple[ModelMessage | None] | None = None
        self._user_turn_anchors: dict[str | None, ModelMessage | None] = {}
        # Retained input audio (`audio_retention='input_audio'`/`'all'`). `_input_audio` is the rolling buffer
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
        self._tap_finished = object()
        self._audio_taps: set[asyncio.Queue[bytes | object]] = set()
        self._transcript_taps: set[asyncio.Queue[SpeechPart | object]] = set()
        self._transcript_delta_taps: set[asyncio.Queue[TranscriptUpdate | object]] = set()
        self._audio_tap_drops = 0
        self._transcript_tap_drops = 0
        # Transcript accumulated per streamed part index, so a `TranscriptUpdate` can carry the whole
        # turn so far and a renderer can replace rather than append.
        self._transcript_so_far: dict[int, str] = {}
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._pending_messages = _RealtimePendingMessages()
        self._pending_messages_lock = Lock()
        # Serializes everything we send. Tool results go out from background tasks while the caller may
        # be streaming audio or driving turn control, and a single operation can span several frames:
        # a tool result creates the item and then asks for a response, and `interrupt()` truncates
        # before it cancels. Without this, those sequences interleave — a barge-in's cancel can land on
        # a response that a tool result started in the gap, killing the wrong turn.
        self._send_lock = Lock()
        if self._tool_manager.ctx is not None:
            self._tool_manager.ctx.pending_messages = self._pending_messages
        # In-flight tool tasks keyed by tool call id, so a `ToolCallCancelled` can cancel the specific
        # calls the model abandoned (e.g. on barge-in) without touching the others.
        self._pending_tool_calls: dict[str, tuple[asyncio.Task[None], ToolCallPart]] = {}
        # Tool execution is gated inside each task so the receive pump remains free to deliver audio,
        # transcripts, and cancellations. A barrier snapshots every unfinished predecessor; an ordinary
        # call only waits for the latest barrier. Completion events are released from `_run_tool`'s
        # `finally`, including cancellation and failure paths.
        self._tool_completion_events: set[asyncio.Event] = set()
        self._last_tool_barrier: asyncio.Event | None = None
        # OpenAI-protocol tool results can complete before the response's later `response.done` usage
        # finalizes the calling response. Hold their history requests until the call is present.
        self._pending_tool_returns: list[tuple[ToolCallPart, ModelRequest]] = []
        self._tool_calls_awaiting_usage: set[str] = set()
        # `ToolManager` adds a call to `usage.tool_calls` only once it *succeeds*, so calls still
        # running aren't visible there. Reserved when a `ToolCall` passes `_check_tool_call_limit`,
        # released the moment `handle_call` settles (the same event-loop segment that records a
        # success), so a burst of parallel calls can't each clear a limit only one of them fits under.
        self._tool_calls_in_flight = 0
        # `parallel_ordered_events` buffers: call-order index -> the events that call produced.
        self._ordered_tool_events: dict[int, list[RealtimeEvent]] = {}
        self._next_tool_order_index = 0
        self._asap_drain_deferred = False
        self._asap_drain_ready = False
        self._pump_task: asyncio.Task[None] | None = None
        self._pump_error: Exception | None = None
        self._pump_finished = False
        self._iterator_active = False
        self._stream_exhausted = False
        # Whether the event stream was ever consumed, which decides whether a pump error still needs
        # somewhere to go when the session closes — see `close`.
        self._stream_consumed = False
        self._entered = False
        self._closed = False
        self._closing_error: BaseException | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        self._traceparent_value: str | None = None
        self._result: AgentRunResult[str] | None = None

    async def __aenter__(self) -> RealtimeSession:
        if self._entered or self._closed:
            raise UserError('This realtime session cannot be entered more than once.')
        self._entered = True
        self._loop = asyncio.get_running_loop()
        self._pending_messages.bind(self._notify_pending_messages)
        if self._profile.get('supports_session_seeding', False):
            # Offer the conversation for replay, so a provider that keeps no state across sessions can
            # carry the call through a reconnect instead of resuming with amnesia. Gated on seeding
            # support because that is the mechanism, and a no-op where the provider resumes natively.
            self._connection.set_message_history(self.all_messages)

        self._session_instrumentation.start_session_span()

        return self

    def _record_user_speech_span(self) -> None:
        """Consume the pending speech onset and record the spoken segment as a `user speech` span."""
        started_at, self._user_speech_started_at = self._user_speech_started_at, None
        self._session_instrumentation.record_user_speech(started_at)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._closing_error = exc_value
        await self.close()

    async def close(self) -> None:
        """Close the session and end its live stream views.

        This method is idempotent. Active [`stream_audio()`][pydantic_ai.realtime.RealtimeSession.stream_audio]
        and [`stream_transcripts()`][pydantic_ai.realtime.RealtimeSession.stream_transcripts] iterators
        finish cleanly, with any buffered items discarded. The surrounding model context owns the
        underlying connection, so it remains open until that context exits.

        Raises whatever ended the session — a provider hangup, an exceeded `usage_limits` — if the event
        stream was never iterated, since there was nowhere else for it to surface.
        """
        if not self._entered or self._closed:
            return
        self._closed = True
        self._finish_taps(discard_pending=True)
        if self._pump_task is not None:
            # Cancelled before state is settled below so the pump can't mutate it mid-settlement;
            # the task is awaited together with the rest afterwards.
            self._pump_task.cancel()
        if (early_error := self._closing_error or self._pump_error) is not None and (
            chat_span := self._session_instrumentation.chat_span
        ) is not None:
            # The reply this span covers is being torn down by a failure; record it now, before the
            # settlement below finalizes the interrupted response and ends the span cleanly.
            SessionInstrumentation.record_error(chat_span, early_error)
        self._flush_pending_users()
        if (
            self._pending_response_usage != RequestUsage()
            and self._active_assistant is None
            and not self._response_parts
        ):
            # Usage carried forward from an output-less turn boundary that no later response claimed.
            # Better an empty response holding it than silently dropping billed tokens — and, with no
            # reply in flight, nothing here was interrupted, so it isn't settled as such below.
            self._finalize_response(response_occurred=True)
        # Settle whatever the closing session still holds open, exactly as a reconnect settles state
        # the provider lost: open user turns land in history, a reply cut off mid-generation is
        # recorded as interrupted, and every still-running tool call gets a cancelled return. The
        # returned events are discarded — the stream is closing and has no consumer left.
        self._finalize_lost_state()
        tasks = [*self._background_tasks]
        if self._pump_task is not None:
            tasks.append(self._pump_task)
        if tasks:
            await cancel_and_drain(*tasks, msg='Realtime session exited')

        # Any open `chat` span was closed by the settlement above (an open span counts as a response
        # in flight), with the error — if any — already recorded on it before settlement.
        error = self._closing_error or self._pump_error
        # Closing mid-utterance is normal (the caller stopped listening), so the `speak` span is closed
        # rather than left open; it isn't an error even when the session ended on one.
        self._session_instrumentation.end_playback_span()
        # A session closed mid-sentence never learns how long that sentence was, so the pending onset
        # is dropped rather than turned into a span ending at teardown.
        self._user_speech_started_at = None

        self._traceparent_value = self._session_instrumentation.end_session_span(
            error,
            usage=self.usage,
            messages=self.all_messages(),
            new_message_index=len(self._seeded) if self._seeded else None,
            final_result=self._final_result_text(),
            audio_chunks_dropped=self._audio_tap_drops,
            transcript_items_dropped=self._transcript_tap_drops,
        )
        self._loop = None

        # A session that was never iterated has nowhere else to learn that it failed: the pump's error is
        # normally raised out of `__aiter__`, so a caller using only `send()` and the
        # `stream_audio()`/`stream_transcripts()` views would otherwise exit *cleanly* from a provider
        # hangup — or from an exceeded `usage_limits`, silently spending past a cost cap it asked for.
        # Not raised when the caller did consume the stream (it either saw the error or chose to stop
        # listening), nor over an exception already on its way out of the `async with` body.
        if self._closing_error is None and not self._stream_consumed:
            if self._pump_error is not None:
                raise self._pump_error
            # A failed tool (or background drain) surfaces through the queue rather than
            # `_pump_error`; with no consumer it would otherwise vanish here.
            while not self._queue.empty():
                item = self._queue.get_nowait()
                if isinstance(item, BaseException) and not isinstance(item, asyncio.CancelledError):
                    raise item

    @property
    def closed(self) -> bool:
        """Whether the session has been closed."""
        return self._closed

    @property
    def result(self) -> AgentRunResult[str] | None:
        """The final result once the session context has exited, otherwise `None`."""
        return self._result

    def _build_run_result(self, state: _agent_graph.GraphAgentState) -> AgentRunResult[str]:
        """Settle the session into the same result shape `Agent.run(output_type=str)` returns.

        The output is the model's final text — a session has no other kind of output — and
        `new_message_index` is the seeded/recorded boundary, so `result.new_messages()` and
        `new_messages()` agree. `_output_tool_name` stays `None`: plain text never came from a tool.
        """
        state.message_history = self.all_messages()
        return AgentRunResult(
            self._final_result_text() or '',
            None,
            state,
            len(self._seeded),
            self._traceparent_value,
        )

    @property
    def profile(self) -> RealtimeModelProfile:
        """What the connected model supports, as [`RealtimeModel.profile`][pydantic_ai.realtime.RealtimeModel.profile].

        Available here because the session is what a call actually holds: `agent.realtime()` accepts a
        model *name* and builds the model itself, leaving nothing else to read the profile from. The
        audio sample rates have their own dedicated properties —
        [`audio_input_sample_rate`][pydantic_ai.realtime.RealtimeSession.audio_input_sample_rate] and
        [`audio_output_sample_rate`][pydantic_ai.realtime.RealtimeSession.audio_output_sample_rate] —
        so most code never needs to read the profile directly.
        """
        return self._profile

    @property
    def audio_input_sample_rate(self) -> int:
        """The sample rate, in Hz, of the raw PCM audio this session expects.

        Resample the microphone to this rate before
        [`send_audio`][pydantic_ai.realtime.RealtimeSession.send_audio]: audio sent at the wrong rate
        is heard as a chipmunk (or slow-motion voice) rather than reported as an error.
        """
        return self._profile.get('audio_input_sample_rate', DEFAULT_AUDIO_SAMPLE_RATE)

    @property
    def audio_output_sample_rate(self) -> int:
        """The sample rate, in Hz, of the raw PCM audio [`stream_audio()`][pydantic_ai.realtime.RealtimeSession.stream_audio] yields.

        Play output at this rate; it can differ from
        [`audio_input_sample_rate`][pydantic_ai.realtime.RealtimeSession.audio_input_sample_rate]
        (Gemini Live, for example, listens at 16 kHz and speaks at 24 kHz).
        """
        return self._profile.get('audio_output_sample_rate', DEFAULT_AUDIO_SAMPLE_RATE)

    async def stream_audio(self) -> AsyncIterator[bytes]:
        """Stream model audio chunks ready for playback.

        The iterator contains only live model audio, in playback order. It never repeats retained
        audio from finalized speech parts. On a WebRTC sideband the browser owns the audio path, so
        this raises [`UserError`][pydantic_ai.exceptions.UserError]; consume the browser's remote media
        track instead.

        Each iterator has a 32-chunk buffer. If its consumer falls behind, the oldest chunk is
        dropped so audio playback cannot stall tool execution, turn tracking, or the main event
        stream. Closing the session discards buffered chunks and ends the iterator cleanly.
        """
        self._require_media_ownership('stream_audio')
        self._ensure_streamable()
        # The extra slot is reserved for the completion sentinel, so ending a full tap does not
        # discard one of its 32 data items or block the pump during teardown.
        queue: asyncio.Queue[bytes | object] = asyncio.Queue(maxsize=_AUDIO_TAP_SIZE + 1)
        self._audio_taps.add(queue)
        if self._pump_finished:
            queue.put_nowait(self._tap_finished)
        else:
            self._start_pump()
        try:
            while (item := await queue.get()) is not self._tap_finished:
                assert isinstance(item, bytes)
                yield item
        finally:
            self._audio_taps.discard(queue)

    @overload
    def stream_transcripts(self, *, delta: Literal[False] = False) -> AsyncIterator[SpeechPart]: ...

    @overload
    def stream_transcripts(self, *, delta: Literal[True]) -> AsyncIterator[TranscriptUpdate]: ...

    async def stream_transcripts(self, *, delta: bool = False) -> AsyncIterator[SpeechPart | TranscriptUpdate]:
        """Stream speech transcripts for both the user and assistant.

        By default, yields finalized [`SpeechPart`][pydantic_ai.messages.SpeechPart] instances — one
        per completed turn, carrying its `speaker` and full `transcript`.

        Pass `delta=True` for live captions, which yields
        [`TranscriptUpdate`][pydantic_ai.realtime.TranscriptUpdate]s carrying the new text, the turn's
        full transcript so far, the speaker, and an `index` identifying the turn. Both speakers
        stream at once, so that `index` is what lets a UI keep two turns apart instead of running
        them together. Empty updates and finalized parts without a transcript are omitted.

        Final transcripts and deltas are separate subscriptions, so one never crowds out the other.
        Each iterator buffers up to 512 items; if its consumer falls behind, the oldest is dropped,
        because captioning must not be able to stall tool execution, turn tracking, or the main event
        stream. Closing the session discards buffered items and ends the iterator cleanly.
        """
        self._ensure_streamable()
        # Each kind gets its own subscription, so a consumer's window is never spent on items it
        # would discard: sharing one queue let a burst of deltas evict the finalized part a
        # `delta=False` consumer was waiting for, and the two speakers' transcripts interleave.
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=_TRANSCRIPT_TAP_SIZE + 1)
        taps: set[Any] = self._transcript_delta_taps if delta else self._transcript_taps
        taps.add(queue)
        if self._pump_finished:
            queue.put_nowait(self._tap_finished)
        else:
            self._start_pump()
        try:
            while (item := await queue.get()) is not self._tap_finished:
                yield item
        finally:
            taps.discard(queue)

    def all_messages(self) -> list[ModelMessage]:
        """A snapshot of the seeded history plus messages recorded during this session.

        Returns a copy, so the result doesn't change as the session continues. Feed it into
        [`Agent.run(message_history=...)`][pydantic_ai.agent.AbstractAgent.run] to hand the
        conversation off to a standard agent run. Images streamed with `send()` are recorded according
        to `retain_images_every_n`, bounded by `retain_images_max`.
        """
        return [*self._seeded, *self._history]

    def new_messages(self) -> list[ModelMessage]:
        """A snapshot of the messages created during this session (excluding the seeded history)."""
        return list(self._history)

    def _new_request(self, parts: list[ModelRequestPart]) -> ModelRequest:
        """Create a request carrying the framework-managed session metadata."""
        request = ModelRequest(parts=parts)
        fill_run_metadata(request, run_id=self._run_id, conversation_id=self._conversation_id)
        return request

    async def send(self, content: RealtimeSessionInput | Sequence[RealtimeSessionInput]) -> None:
        """Feed content into the session.

        Accepts the shared message vocabulary: plain text as a `str`, image/audio
        [`BinaryContent`][pydantic_ai.messages.BinaryContent] (including
        [`BinaryImage`][pydantic_ai.messages.BinaryImage] and
        [`BinaryAudio`][pydantic_ai.messages.BinaryAudio]), or a sequence of these inputs, dispatched
        in order. Text and retained images are recorded in session history; audio is recorded later
        through its transcript and/or `audio_retention`. `retain_images_every_n=1` records every image,
        while larger values keep the first image and then one of every `N`; `retain_images_max` bounds
        how many stay recorded, evicting the oldest first. Sending an image is gated on the model
        profile's image-input support and raises `UserError` when it is unsupported.

        `send()` accepts session content only. Turn-control verbs (`CommitAudio`, `ClearAudio`,
        `CreateResponse`, `CancelResponse`, `TruncateOutput`) are driven through the dedicated methods
        (`commit_audio()`, `clear_audio()`, `create_response()`, `interrupt()`), and
        [`ToolResult`][pydantic_ai.realtime.codec.ToolResult] is sent by the session itself as each tool
        completes (see `_execute_tool`) — neither is accepted here.
        """
        if isinstance(content, str):
            self._reserve_response_request()
            request = self._new_request([UserPromptPart(content=content)])
            self._record_sent_request(request)
            try:
                await self._send_frame(content)
            except BaseException:
                self._pending_response_requests -= 1
                self._remove_sent_request(request)
                raise
        elif isinstance(content, BinaryContent):
            if content.is_image:
                await self._send_image(content)
            elif content.media_type == _WAV_MEDIA_TYPE:
                # Retained `SpeechPart.audio` (from `audio_retention`) is a WAV container; unwrap it to
                # raw PCM — matching the seeding path — so a natural round-trip (retain a turn's audio,
                # then `send()` it back) doesn't stream the WAV header into the buffer as noise.
                await self.send_audio(
                    seed_pcm_audio(
                        audio=content,
                        provider_name=self._provider_name or 'realtime',
                        sample_rate=self.audio_input_sample_rate,
                    )
                )
            elif content.media_type == 'audio/pcm':
                await self.send_audio(content.data)
            else:
                raise UserError(
                    f'Unsupported binary media type {content.media_type!r} for `session.send()`. '
                    'Send an image, WAV audio, or raw PCM (`audio/pcm`); for a raw PCM byte stream use `send_audio()`.'
                )
        elif isinstance(content, (bytes, bytearray)):
            # `bytes` is a `Sequence[int]`, so guard it before the sequence branch below — otherwise it
            # iterates into a confusing per-byte error. Raw input audio goes through `send_audio()`.
            raise UserError('Raw audio bytes cannot be sent via `session.send()`; use `session.send_audio(...)`.')
        elif isinstance(content, Sequence):
            for item in content:
                await self.send(item)
        else:
            assert_never(content)

    async def _send_image(self, content: BinaryContent) -> None:
        """Forward an image and retain it according to the session's sampling and cap policies."""
        self._require_capability('supports_image_input', method='send', feature='image input')
        request: ModelRequest | None = None
        if self._retain_images_max != 0 and self._sent_image_count % self._retain_images_every_n == 0:
            request = self._new_request([UserPromptPart(content=[content])])
            self._record_sent_request(request)
        try:
            # Callers guard on `is_image`, so the narrowed type only re-tags a plain `BinaryContent`.
            image = BinaryContent.narrow_type(content)
            assert isinstance(image, BinaryImage)
            await self._send_frame(image)
        except BaseException:
            # `None` when this image wasn't the one retained by the sampling policy: nothing recorded,
            # so nothing to take back.
            if request is not None:
                self._remove_sent_request(request)
            raise
        self._sent_image_count += 1
        if request is not None:
            self._retained_image_requests.append(request)
            # `retain_images_every_n` only slows history growth; the cap bounds it, so a long-running
            # frame stream can't grow the host's memory without limit. Providers hold their own
            # (server-side) image context; this only trims the local record.
            while self._retain_images_max is not None and len(self._retained_image_requests) > self._retain_images_max:
                self._remove_sent_request(self._retained_image_requests.pop(0))

    def _record_sent_request(self, request: ModelRequest) -> None:
        """Record a sent request without interleaving it with an in-flight assistant response."""
        if self._response_in_flight:
            self._pending_sent_requests.append(request)
        else:
            self._history.append(request)

    @property
    def _response_in_flight(self) -> bool:
        return bool(
            self._active_assistant is not None
            or self._response_parts
            or self._native_tool_parts
            or self._pending_provider_response_id is not None
            or self._pending_finish_reason is not None
            or self._pending_response_usage != RequestUsage()
            or self._session_instrumentation.chat_span is not None
        )

    def _remove_sent_request(self, request: ModelRequest) -> None:
        """Remove a recorded request: a failed network send takes it back, the image cap evicts it."""
        for messages in (self._pending_sent_requests, self._history):
            for index, message in enumerate(messages):
                if message is request:
                    messages.pop(index)
                    return

    async def send_audio(self, data: bytes) -> None:
        """Stream a chunk of mono PCM16 audio to the model.

        Resample it to
        [`audio_input_sample_rate`][pydantic_ai.realtime.RealtimeSession.audio_input_sample_rate]
        first (24 kHz on the OpenAI-protocol providers, 16 kHz on Gemini):
        raw bytes carry no rate, so the wrong one is heard as a chipmunk rather than reported.
        """
        self._require_media_ownership('send_audio')
        user_turn_was_active = self._user_turn_active
        if not user_turn_was_active:
            # Audio starting is the earliest sign of a user turn, and the only one on a provider that
            # reports no speech boundaries, so it's where the turn's place in history is reserved.
            self._open_user_turn_anchor()
        self._user_turn_active = True
        previous_length: int | None = None
        if self._retain_input:
            # Buffer the raw input so the finalized user turn can retain it. A per-item speech-stopped
            # boundary later cuts this into that turn's own segment (see `_segment_input_audio`); only the
            # exact split at the boundary is approximate (see `audio_retention`).
            previous_length = len(self._input_audio)
            self._input_audio.extend(data)
        try:
            await self._send_frame(BinaryAudio(data=data, media_type='audio/pcm'))
        except BaseException:
            self._user_turn_active = user_turn_was_active
            if previous_length is not None and len(self._input_audio) == previous_length + len(data):
                del self._input_audio[previous_length:]
            raise

    async def commit_audio(self) -> None:
        """Commit buffered input audio as a user turn (manual turn-taking / push-to-talk)."""
        self._require_media_ownership('commit_audio')
        self._require_capability('supports_manual_turn_control', method='commit_audio', feature='manual turn-taking')
        await self._send_frame(CommitAudio())
        self._user_turn_active = True
        for event in self._finalize_untranscribed_user():
            await self._queue.put(event)

    async def clear_audio(self) -> None:
        """Discard buffered, uncommitted input audio."""
        self._require_media_ownership('clear_audio')
        self._require_capability('supports_manual_turn_control', method='clear_audio', feature='manual turn-taking')
        await self._send_frame(ClearAudio())
        # Drop the locally retained copy too (with `audio_retention='input_audio'`/`'all'`), or the discarded
        # audio would still be attached to the next finalized user turn.
        self._input_audio.clear()
        self._user_turn_active = False

    async def create_response(self) -> None:
        """Ask the model to respond now (manual turn-taking, after `commit_audio`)."""
        self._require_capability('supports_manual_turn_control', method='create_response', feature='manual turn-taking')
        self._reserve_response_request()
        try:
            await self._send_frame(CreateResponse())
        except BaseException:
            self._pending_response_requests -= 1
            raise

    async def interrupt(self, *, played_ms: int | None = None) -> None:
        """Barge-in: cancel the model's in-progress response, optionally truncating its audio first.

        This is server-side only — it stops generation and (when `played_ms` is given) syncs the
        provider's transcript to what was actually heard. Flushing locally buffered playback is the
        caller's responsibility.

        Args:
            played_ms: Playback position in milliseconds from the start of the model's current audio
                output. When given, the model's output audio and its transcript are truncated to this
                point before the response is cancelled.
        """
        self._ensure_not_closed()
        self._require_capability('supports_interruption', method='interrupt', feature='interruption')
        if played_ms is not None and not self._profile.get('supports_output_truncation', False):
            raise UserError(
                'This realtime model does not support output truncation, so `interrupt(played_ms=...)` '
                'is unavailable. Call `interrupt()` without `played_ms` to cancel without truncating.'
            )
        # Truncate before cancelling: cancellation triggers `response.done`, which clears the tracked
        # output item, so a truncate sent afterwards could no-op. Both frames go out under one hold of
        # the send lock, so a tool result completing in between can't start a new response for the
        # cancel to hit instead.
        await self._send_frame(
            *([TruncateOutput(audio_end_ms=played_ms)] if played_ms is not None else []),
            CancelResponse(),
        )
        self._pending_interrupted_at_ms = played_ms
        # Mark the barge-in in the trace. When the caller supplied `played_ms` (the ms of output audio
        # actually played before truncating), record it so a reader can see how far the response got before
        # the user cut in; it's dropped when absent (a cancel without truncation).
        self._session_instrumentation.record_lifecycle('interrupt', played_ms=played_ms)

    async def _send_frame(self, *contents: RealtimeInput) -> None:
        """Send inputs to the provider as one group, serialized against every other outbound frame.

        A single input can expand to several protocol frames (a `ToolResult` creates the conversation
        item and then asks for a response), so the lock is what makes each input indivisible on the
        wire, not just ordered. Passing several inputs extends that indivisibility across them, for
        the cases where an interleaved frame would change what they mean.
        """
        self._ensure_not_closed()
        self._start_pump()
        async with self._send_lock:
            try:
                for content in contents:
                    await self._connection.send(content)
            except self._connection.transport_errors as e:
                # A send that fails because the link is gone is the same failure the receive side
                # reports; surface it as the same typed error rather than leaking a `websockets` or
                # provider-SDK exception from what looks like an ordinary method call.
                raise RealtimeError(
                    model_name=self._error_model_name,
                    message=f'Realtime connection failed while sending: {e}',
                ) from e

    @property
    def _error_model_name(self) -> str:
        """The model to attribute a provider failure to: the one asked for, else the one served."""
        return self._model_name or self._connection.model_name or 'unknown'

    def _ensure_not_closed(self) -> None:
        if self._closed:
            raise UserError('This realtime session is closed.')

    def _require_capability(self, capability: str, *, method: str, feature: str) -> None:
        """Raise a clear `UserError` before sending when the profile doesn't report `capability`."""
        if not self._profile.get(capability, False):
            raise UserError(f'This realtime model does not support {feature}, so `session.{method}()` is unavailable.')

    def _require_media_ownership(self, method: str) -> None:
        """Raise a clear `UserError` when an audio-transport method is used on a session that doesn't own media."""
        if not self._owns_media:
            raise UserError(
                f'This realtime session does not own the audio transport, so `session.{method}()` is unavailable. '
                'The browser exchanges audio with the provider directly over WebRTC; this sideband session only '
                'runs the control plane (instructions, tools, transcripts, history).'
            )

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
            active_item_id = self._active_assistant_item_id
            item_changed = active_item_id is not None and item_id is not None and active_item_id != item_id
            modality_changed = output_text != isinstance(active, TextPart)
            if item_changed or modality_changed:
                events.extend(self._finalize_assistant_part())
                active = None
        if active is not None:
            if item_id is not None and self._active_assistant_item_id is None:
                self._active_assistant_item_id = item_id
            return events
        self._ensure_chat_span()
        part: SpeechPart | TextPart = (
            TextPart(content='') if output_text else SpeechPart(speaker='assistant', transcript='')
        )
        self._active_assistant = part
        self._active_assistant_item_id = item_id
        self._active_assistant_index = self._take_part_index()
        self._assistant_transcript = ''
        events.append(PartStartEvent(index=self._active_assistant_index, part=part))
        return events

    def _take_part_index(self) -> int:
        """Claim the next session-unique index for a part about to start streaming."""
        index = self._next_part_index
        self._next_part_index += 1
        return index

    def _handle_native_part_event(self, event: PartStartEvent | PartEndEvent) -> list[RealtimeEvent]:
        """Forward a connection's native-tool part event under a session-unique index.

        Providers emit native tool activity (grounding, code execution) as ordinary part events,
        numbered from a counter of their own that knows nothing of the indexes the session has already
        handed out to the speech, text, and tool-call parts of the same turn. Since a repeated index
        *replaces* the part at it, forwarding those numbers verbatim would let a grounding part
        overwrite the model's answer for any consumer keyed on the index. Claim a session-unique index
        when the part starts and reuse it for the matching end. A started part is also buffered for the
        assistant response, which the session builds alongside the stream.
        """
        if isinstance(event, PartStartEvent):
            self._ensure_chat_span()
            self._native_tool_parts.append(event.part)
            index = self._native_part_indexes[event.index] = self._take_part_index()
        else:
            # An end with no recorded start closes nothing, so it gets an index of its own rather than
            # the connection's — which is not session-unique and could name a live part.
            index = self._native_part_indexes.pop(event.index, None)
            index = self._take_part_index() if index is None else index
        return [replace(event, index=index)]

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
            delta = SpeechPartDelta(
                speaker='assistant', transcript_delta=appended, transcript=self._assistant_transcript
            )
        if appended:
            events.append(PartDeltaEvent(index=self._active_assistant_index, delta=delta))
        return events

    def _handle_assistant_audio(self, data: bytes, *, item_id: str | None = None) -> list[RealtimeEvent]:
        events = self._ensure_active_assistant(item_id=item_id)
        if self._retain_output:
            self._output_audio.extend(data)
        events.append(
            PartDeltaEvent(
                index=self._active_assistant_index, delta=SpeechPartDelta(speaker='assistant', audio_chunk=data)
            )
        )
        return events

    def _finalize_assistant_part(self) -> list[RealtimeEvent]:
        """End the in-flight assistant part and append it to the current response."""
        if self._active_assistant is None:
            return []
        part = self._active_assistant
        if isinstance(part, SpeechPart):
            if part.transcript == '':
                part = replace(part, transcript=None)
            if self._retain_output and self._output_audio:
                sample_rate = self.audio_output_sample_rate
                part = replace(
                    part,
                    audio=BinaryContent(
                        data=_pcm_to_wav(bytes(self._output_audio), sample_rate), media_type=_WAV_MEDIA_TYPE
                    ),
                )
        index = self._active_assistant_index
        self._active_assistant = None
        self._active_assistant_item_id = None
        self._assistant_transcript = ''
        self._output_audio.clear()
        self._response_parts.append(part)
        return [PartEndEvent(index=index, part=part)]

    def _finalize_response(
        self,
        *,
        provider_response_id: str | None = None,
        finish_reason: FinishReason | None = None,
        provider_details: dict[str, Any] | None = None,
        interrupted: bool = False,
        interrupted_at_ms: int | None = None,
        response_occurred: bool = False,
    ) -> None:
        """Finalize the current assistant response's parts into a `ModelResponse` in history."""
        response: ModelResponse | None = None
        # The chat span's input is the history the response replied to, captured before we append it.
        input_messages = self.all_messages()
        # Native tool parts (web grounding / code execution) lead the response (call+return, then
        # speech), matching the classic `GoogleModel`, which prepends them ahead of the assistant's text.
        parts = [*self._native_tool_parts, *self._response_parts]
        if interrupted:
            for index in range(len(parts) - 1, -1, -1):
                if isinstance(part := parts[index], SpeechPart):
                    parts[index] = replace(part, interrupted_at_ms=interrupted_at_ms)
                    break
        # Parts prove a response happened. For an output-less response, only terminal/pending provider
        # metadata (or an interruption) does; a bare logical turn boundary must not invent a response.
        response_occurred = bool(
            response_occurred
            or parts
            or self._pending_provider_response_id is not None
            or self._pending_finish_reason is not None
            or self._pending_response_usage != RequestUsage()
            or self._session_instrumentation.chat_span is not None
        )
        reason = finish_reason or self._pending_finish_reason
        if (
            response_occurred
            and not parts
            and not interrupted
            and provider_details is None
            and reason in (None, 'stop')
            and not self._closed
        ):
            # A boundary that says nothing, in either sense: no output, and no provider detail or
            # abnormal finish reason explaining why. That isn't a response — it's one logical response
            # arriving in two frames. Gemini Live closes the turn when it takes a tool result and again
            # when it has finished speaking, and the first boundary carries only usage; recording it
            # would make one tool round five messages there and four everywhere else. Carry the metadata
            # onto the response that does say something, so a tool round has one shape on every
            # provider. A response truncated by `length` (or any other abnormal reason) is real
            # information about what happened and stays, empty or not; so does anything left pending
            # when the session ends, which the flush in `__aexit__` records rather than lose.
            self._pending_provider_response_id = provider_response_id or self._pending_provider_response_id
            self._pending_finish_reason = reason
            self._session_instrumentation.end_chat_span(input_messages, None)
            self._response_parts = []
            self._native_tool_parts = []
            self._response_limit_checked = False
            return
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
            fill_run_metadata(response, run_id=self._run_id, conversation_id=self._conversation_id)
            self._history.append(response)
            self.usage.requests += 1
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
        self._session_instrumentation.end_chat_span(input_messages, response)
        self._response_parts = []
        self._native_tool_parts = []
        self._pending_response_usage = RequestUsage()
        self._pending_provider_response_id = None
        self._pending_finish_reason = None
        self._response_limit_checked = False

    def _ensure_chat_span(self) -> None:
        """Begin assembling a response and open its `chat {model}` span if not already open.

        See `SessionInstrumentation.ensure_chat_span` for the span's shape and lifetime.
        """
        self._begin_response()
        self._session_instrumentation.ensure_chat_span()

    def _handle_turn_complete(self, event: ResponseDone) -> list[RealtimeEvent]:
        # Turn boundary for a user turn that wasn't finalized earlier, so history reads user-then-assistant.
        # Gemini emits neither `RealtimeInputSpeechEndEvent` nor a final (`is_final`) input transcript — it streams
        # only partial transcripts — so its user turn is finalized here: `_finalize_user` for a
        # transcript-driven turn, `_finalize_untranscribed_user` otherwise. Both are no-ops
        # when the turn was already finalized (e.g. OpenAI's `is_final` transcript or `commit_audio`).
        events = self._finalize_user()
        events.extend(self._finalize_untranscribed_user())
        if self._retain_input and self._provider_segments_input and self._user_speech_started_at is None:
            # A speech-stopped boundary may be processed while the caller is still sending the tail of
            # its input schedule. That tail accumulates after `_segment_input_audio` clears the rolling
            # buffer and must not become the prefix of the next user turn. The completed response is a
            # safe point to discard it unless the user has already started speaking again (barge-in).
            # Only for a provider that draws speech boundaries at all: Gemini never emits
            # `RealtimeInputSpeechEndEvent`, so its buffer holds the *next* utterance, not a spent tail.
            self._input_audio.clear()
        events.extend(self._finalize_assistant_part())
        already_finalized = bool(
            self._response_finalized_before_terminal
            and not self._response_parts
            and not self._native_tool_parts
            and self._pending_provider_response_id is None
            and self._pending_finish_reason is None
            and self._pending_response_usage == RequestUsage()
        )
        if not already_finalized:
            self._begin_response()
        # Whether the model will speak again: it always responds to a tool's result, so a response that
        # called one, that left one still running, or whose content was already recorded (making this the
        # trailing terminal of a tool-call response, with the answer still to come) is never the last of
        # the exchange. `RealtimeTurnCompleteEvent` waits for the one that is.
        #
        # Not the `_response_finalized_before_terminal` flag itself: the OpenAI protocol *suppresses* a
        # function-call-only `response.done`, so the flag would still be set when the answer's terminal
        # arrives and would swallow the turn boundary for the whole exchange.
        more_expected = bool(
            self._pending_tool_calls
            or already_finalized
            or any(isinstance(part, ToolCallPart) for part in self._response_parts)
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
            interrupted_at_ms=self._pending_interrupted_at_ms,
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
        self._pending_interrupted_at_ms = None
        if not more_expected:
            events.append(RealtimeTurnCompleteEvent())
            # Only the exchange boundary is marked: each response is already a `chat` span, so a marker
            # per response would say nothing the trace doesn't show, while the turn boundary — where the
            # model is actually done — has no span of its own.
            # An interrupted boundary says so in its display text: on providers that auto-respond per
            # VAD segment (OpenAI server VAD), a talking user cancels response after response, and a
            # bare "model turn complete" per cancellation reads as turns that never happened.
            self._session_instrumentation.record_lifecycle(
                'model turn complete',
                message='model turn complete (interrupted)' if event.interrupted else None,
                interrupted=event.interrupted or None,
            )
        return events

    def _handle_tool_call_part(self, call_part: ToolCallPart, *, response_usage_follows: bool) -> list[RealtimeEvent]:
        """Fold a tool call into the current response, deferring finalization when its usage follows.

        OpenAI-protocol providers report each call before the `response.done` frame carrying that
        response's usage, so finalization waits for the ensuing `SessionUsage`. Gemini's tool-call
        frame has no per-response usage to wait for; it is finalized immediately with zero usage, while
        the later completed turn keeps the usage Gemini reports for that turn.
        """
        self._ensure_chat_span()
        events = self._finalize_assistant_part()
        index = self._take_part_index()
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
        self._insert_tool_return(call_part, self._new_request(request_parts))
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
                call_order = {
                    part.tool_call_id: index
                    for index, part in enumerate(message.parts)
                    if isinstance(part, ToolCallPart)
                }
                position = call_order[call_part.tool_call_id]
                insert_at = i + 1
                while insert_at < len(self._history) and (
                    existing_id := _tool_result_call_id(self._history[insert_at])
                ):
                    # Parallel calls settle in whatever order they finish, so skip past a sibling's
                    # result only while it belongs *before* this one in the response's call order.
                    if call_order.get(existing_id, position) > position:
                        break
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
            self._history.append(request)

    def _handle_input_transcript(
        self, text: str, is_final: bool, *, item_id: str | None = None, cumulative: bool = False
    ) -> list[RealtimeEvent]:
        if item_id is not None:
            # Once an item is closed (finalized or its transcription failed), ignore any stray later event
            # for it — re-creating it would duplicate the turn or resurrect a discarded failed one.
            if item_id in self._finalized_user_item_ids:
                return []
            events: list[RealtimeEvent] = []
            if item_id not in self._user_turns:
                self._user_turn_active = True
                part = SpeechPart(speaker='user', transcript='')
                self._claim_user_turn_anchor(item_id)
                turn = self._user_turns[item_id] = _UserTurn(part, '', self._take_part_index())
                events.append(PartStartEvent(index=turn.index, part=part))
            turn = self._user_turns[item_id]
            transcript, delta = _user_transcript_update(turn.transcript, text, cumulative=cumulative)
            turn.transcript = transcript
            turn.part = replace(turn.part, transcript=transcript)
            if delta is not None:
                events.append(PartDeltaEvent(index=turn.index, delta=delta))
            if is_final:
                events.extend(self._finalize_user(item_id=item_id))
            return events

        events: list[RealtimeEvent] = []
        if None not in self._user_turns:
            self._user_turn_active = True
            part = SpeechPart(speaker='user', transcript='')
            self._claim_user_turn_anchor(None)
            turn = self._user_turns[None] = _UserTurn(part, '', self._take_part_index())
            events.append(PartStartEvent(index=turn.index, part=part))
        turn = self._user_turns[None]
        turn.transcript, delta = _user_transcript_update(turn.transcript, text, cumulative=cumulative)
        turn.part = replace(turn.part, transcript=turn.transcript)
        if delta is not None:
            events.append(PartDeltaEvent(index=turn.index, delta=delta))
        if is_final:
            events.extend(self._finalize_user())
        return events

    def _finalize_user(self, *, item_id: str | None = None) -> list[RealtimeEvent]:
        turn = self._user_turns.get(item_id)
        if turn is None or turn.finalized:
            return []
        part = turn.part
        index = turn.index
        if item_id is not None:
            self._finalized_user_item_ids.add(item_id)
        self._user_turn_active = any(not current.finalized for current in self._user_turns.values())
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
                sample_rate = self.audio_input_sample_rate
                part = replace(
                    part,
                    audio=BinaryContent(data=_pcm_to_wav(segment, sample_rate), media_type=_WAV_MEDIA_TYPE),
                )
        if item_id is None:
            self._record_user_request(None, self._new_request([part]))
            self._user_turns.pop(None)
        else:
            turn.part = part
            turn.finalized = True
            self._flush_finalized_user_prefix()
        return [PartEndEvent(index=index, part=part)]

    def _open_user_turn_anchor(self) -> None:
        """Remember where a starting user turn belongs in history, before its transcript arrives.

        Input transcription is asynchronous, and its final event can land after the response the speech
        prompted has already been recorded — Azure and Gemini Live do that routinely, OpenAI and xAI under
        load. Appending on finalize would then file the user's words *after* the answer they caused, and
        after the tool call and return in between; replaying that history reads as the model calling a tool
        unprompted. So the turn's position is taken when it starts (audio begins flowing, or the provider
        reports speech started), and `_record_user_request` inserts there however late the transcript is.
        """
        self._pending_user_turn_anchor = (self._history[-1] if self._history else None,)

    def _claim_user_turn_anchor(self, item_id: str | None) -> None:
        """Attach the starting turn's remembered position to the item the transcript identified it as."""
        anchor, self._pending_user_turn_anchor = self._pending_user_turn_anchor, None
        # No anchor when the first thing we ever hear about the turn is its transcript (text-only sessions
        # seeded with audio, or a provider that reports nothing before it); the turn starts here instead.
        self._user_turn_anchors[item_id] = (
            anchor[0] if anchor is not None else (self._history[-1] if self._history else None)
        )

    def _record_user_request(self, item_id: str | None, request: ModelRequest) -> None:
        """Record a finalized user turn at the position it held when it started."""
        # A turn that never opened an anchor has no earlier place to hold: only audio reserves one, so
        # a text turn simply belongs where it is finalized.
        if item_id not in self._user_turn_anchors:
            self._history.append(request)
            return
        anchor = self._user_turn_anchors.pop(item_id)
        insert_at = 0
        if anchor is not None:
            for index in range(len(self._history) - 1, -1, -1):
                if self._history[index] is anchor:
                    insert_at = index + 1
                    break
            else:  # pragma: no cover
                # An invariant fallback, like `_insert_tool_return`'s: nothing withdraws a message a user
                # turn has already anchored to, but keep history complete rather than losing the turn.
                self._history.append(request)
                return
        # Step over what already sits in the anchor's slot: an earlier response's tool returns, which must
        # stay adjacent to their call, and user turns that started at the same point, which were spoken first.
        while insert_at < len(self._history) and (
            _is_tool_result_request(self._history[insert_at]) or _is_user_speech_request(self._history[insert_at])
        ):
            insert_at += 1
        self._history.insert(insert_at, request)

    def _flush_finalized_user_prefix(self) -> None:
        """Record finalized user items in provider order, up to the first item still awaiting its final.

        Item-ID transcripts finalize in any order, but history must keep provider order (call/return
        adjacency etc.), so a finalized item waits in `_user_turns` until every earlier item has
        resolved (finalized or discarded).
        """
        while self._user_turns:
            finalized_id, turn = next(iter(self._user_turns.items()))
            if finalized_id is None or not turn.finalized:
                break
            self._user_turns.pop(finalized_id)
            self._record_user_request(finalized_id, self._new_request([turn.part]))

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

    def _finalize_failed_user_item(self, item_id: str | None) -> list[RealtimeEvent]:
        """Finalize a user item whose transcription failed without retaining unreliable partial text."""
        start_emitted = False
        if item_id is not None:
            # Same guard as `_handle_input_transcript`: once an item is closed, a stray duplicate or
            # late error event must not re-open it and record a second (blank) user turn.
            if item_id in self._finalized_user_item_ids:
                return []
            turn = self._user_turns.get(item_id)
            start_emitted = turn is not None
            part = replace(turn.part, transcript=None) if turn is not None else SpeechPart(speaker='user')
            index = turn.index if turn is not None else self._take_part_index()
            if turn is None:
                turn = self._user_turns[item_id] = _UserTurn(part, '', index)
            if item_id not in self._user_turn_anchors:
                # The failure is the first we hear of this item, so its turn is only placed now.
                self._claim_user_turn_anchor(item_id)
            self._finalized_user_item_ids.add(item_id)
        else:
            turn = self._user_turns.get(None)
            start_emitted = turn is not None
            part = replace(turn.part, transcript=None) if turn is not None else SpeechPart(speaker='user')
            index = turn.index if turn is not None else self._take_part_index()

        if self._retain_input:
            segment = self._input_audio_by_id.pop(item_id, None) if item_id is not None else None
            if segment is None:
                segment = bytes(self._input_audio) if self._input_audio else None
                self._input_audio.clear()
            if segment:
                part = replace(
                    part,
                    audio=BinaryContent(
                        data=_pcm_to_wav(segment, self.audio_input_sample_rate),
                        media_type=_WAV_MEDIA_TYPE,
                    ),
                )

        # Recompute like `_finalize_user`: with overlapping user items, one item's failure must not mark
        # the whole user side idle while another item is still active.
        self._user_turn_active = any(not current.finalized for current in self._user_turns.values())
        if item_id is None:
            self._record_user_request(None, self._new_request([part]))
            self._user_turns.pop(None, None)
        else:
            assert turn is not None
            turn.part = part
            turn.finalized = True
            self._flush_finalized_user_prefix()
        end = PartEndEvent(index=index, part=part)
        return [end] if start_emitted else [PartStartEvent(index=index, part=part), end]

    def _flush_pending_users(self) -> None:
        """Preserve transcript-bearing user items that never received an explicit final event."""
        if None in self._user_turns:
            self._finalize_user()
        for item_id, turn in list(self._user_turns.items()):
            if item_id is not None and not turn.finalized:
                self._finalize_user(item_id=item_id)
        # Finalizing the last blocked item flushed the whole finalized prefix, so nothing remains.
        assert not self._user_turns, 'every pending user turn should have been recorded'
        self._user_turn_anchors.clear()
        self._pending_user_turn_anchor = None
        # Drop any input-audio segments whose transcript never arrived, so they can't leak across a
        # long-lived session (finalized items already popped their own segment above).
        self._input_audio_by_id.clear()

    def _finalize_untranscribed_user(self) -> list[RealtimeEvent]:
        """Finalize a user turn when no transcript will arrive.

        This is called at each user-turn boundary when input transcription is disabled. It records retained
        input audio when available, or a content-less [`SpeechPart`][pydantic_ai.messages.SpeechPart]
        placeholder otherwise, so every turn remains represented in history.

        Gated on transcription being *off*: when it's on we wait for the transcript instead, so an
        (asynchronously delivered) transcript can never race this into a duplicate user turn. A no-op
        when there's an active transcript-driven user part, no user turn is active, or transcription is on.
        """
        if self._input_transcription_enabled:
            return []
        if None in self._user_turns or not self._user_turn_active:
            return []
        audio = None
        if self._input_audio:
            audio = BinaryContent(
                data=_pcm_to_wav(bytes(self._input_audio), self.audio_input_sample_rate),
                media_type=_WAV_MEDIA_TYPE,
            )
        part = SpeechPart(speaker='user', transcript=None, audio=audio)
        self._input_audio.clear()
        self._user_turn_active = False
        self._history.append(self._new_request([part]))
        # No deltas to stream (there's no transcript), so bracket the turn with just start/end so a
        # streaming consumer still sees the user turn boundary.
        index = self._take_part_index()
        return [PartStartEvent(index=index, part=part), PartEndEvent(index=index, part=part)]

    def _is_replayed_item(self, item_id: str | None, tool_call_id: str | None = None) -> bool:
        """Whether a provider-generic resumption replay already exists in local history.

        Only xAI currently emits `ConversationItemCreated(replayed=True)`.
        """
        return (item_id is not None and item_id in self._replayed_item_ids) or (
            tool_call_id is not None and tool_call_id in self._replayed_tool_call_ids
        )

    def _accept_item(self, item_id: str | None, tool_call_id: str | None = None) -> bool:
        """Return `False` for an item that belongs to a provider's resumption replay burst."""
        return not self._is_replayed_item(item_id, tool_call_id)

    def _handle_reconnected(self, event: RealtimeSessionReconnectEvent) -> list[RealtimeEvent]:
        """Settle any in-flight state the reconnect did not actually carry, and report restoration honestly.

        A connection that resumes in-flight state (native resumption on xAI Grok Voice; Gemini Live,
        which settles the cut turn in the connection itself) reports
        [`reconnect_restores_in_flight_state`][pydantic_ai.realtime.codec.RealtimeConnection.reconnect_restores_in_flight_state],
        so `state_restored=True` holds as reported and there is nothing more to settle. A connection we
        reconnect by replaying local history (OpenAI, Azure OpenAI) only restores *finalized* turns: the
        response and tool calls that were in flight when the socket dropped are gone, and the fresh
        server-side conversation knows nothing of them. Settle them here exactly as for a fully lost
        session — the partial reply as an interrupted response, running tool calls as cancelled returns —
        so the local history stays coherent and the turn ends (flushing anything queued behind it).
        Report `state_restored=False` whenever a response or tool call was actually in flight, so an app
        can branch on the flag the same way on every provider; a drop with nothing in flight loses
        nothing and stays `True`. Whether the settlement *emitted* events is not the test: an in-flight
        response carried only by pending provider metadata is finalized into history without any.
        """
        if event.state_restored and self._connection.reconnect_restores_in_flight_state:
            return [event]
        lost_in_flight = self._response_in_flight or bool(self._pending_tool_calls)
        events = self._finalize_lost_state()
        if lost_in_flight:
            event = replace(event, state_restored=False)
        if not event.state_restored:
            # The reconnect cut whatever was playing: the in-flight response (if any) is settled as
            # interrupted above, and even a finalized response's audio won't resume on the fresh
            # connection. End the `speak` span so it measures only what was actually audible, rather
            # than running until session close or being merged into the next utterance
            # (`start_playback_span` no-ops while a span is already open). No-op off a sideband.
            self._session_instrumentation.end_playback_span()
        return [*events, event]

    def _finalize_lost_state(self) -> list[RealtimeEvent]:
        """Settle everything still open into history: user turns, an in-flight response, running tools.

        Shared by the reconnect path (the provider lost this state) and `close()` (the session is
        ending with it still open). The partial reply is recorded as interrupted and every running
        tool call gets a cancelled return, so `all_messages()` stays a valid history for an
        `Agent.run(message_history=...)` handoff instead of dropping the tail of the conversation or
        ending on a dangling `ToolCallPart`.
        """
        events = self._finalize_user()
        for item_id, turn in list(self._user_turns.items()):
            if item_id is not None and not turn.finalized:
                events.extend(self._finalize_user(item_id=item_id))
        self._flush_pending_users()
        events.extend(self._finalize_untranscribed_user())
        self._input_audio.clear()

        if self._response_in_flight:
            events.extend(self._finalize_assistant_part())
            self._finalize_response(interrupted=True)
        for tool_call_id, (task, call_part) in list(self._pending_tool_calls.items()):
            self._pending_tool_calls.pop(tool_call_id, None)
            task.cancel()
            cancelled_part = ToolReturnPart(
                tool_name=call_part.tool_name,
                content=INTERRUPTED_TOOL_RETURN_CONTENT,
                tool_call_id=call_part.tool_call_id,
                outcome='interrupted',
            )
            events.extend(self._complete_tool_call(call_part, cancelled_part))
        return events

    def _handle_control_event(
        self,
        event: (
            RealtimeInputSpeechStartEvent
            | RealtimeSessionReconnectEvent
            | RealtimeOutputSpeechStartEvent
            | RealtimeOutputSpeechEndEvent
        ),
    ) -> list[RealtimeEvent]:
        if isinstance(event, RealtimeSessionReconnectEvent):
            return self._handle_reconnected(event)
        # The playback boundary brackets the `speak` span and is otherwise passed straight through.
        if isinstance(event, RealtimeOutputSpeechStartEvent):
            self._session_instrumentation.start_playback_span()
            return [event]
        if isinstance(event, RealtimeOutputSpeechEndEvent):
            self._session_instrumentation.end_playback_span()
            return [event]
        # A reported speech start is a turn boundary even mid-stream, so it re-anchors: with a continuously
        # open microphone the previous turn may not have finalized yet, leaving `_user_turn_active` set.
        self._open_user_turn_anchor()
        self._user_turn_active = True
        self._user_speech_started_at = time_ns()
        return [event]

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
            self._session_instrumentation.set_output_type('speech')
            return self._handle_assistant_audio(event.data, item_id=event.item_id)
        if isinstance(event, OutputTranscript):
            if not self._accept_item(event.item_id):
                return []
            self._session_instrumentation.set_output_type('text' if event.output_text else 'speech')
            # `is_final` doesn't end the part — the turn ends on `ResponseDone`; a final transcript just
            # carries the full text, which `_accumulate_transcript` reconciles against the deltas. Plain
            # text output (`output_text`) becomes a `TextPart`, an audio transcript a `SpeechPart`.
            return self._handle_assistant_transcript(event.text, output_text=event.output_text, item_id=event.item_id)
        if isinstance(event, InputTranscript):
            if not self._accept_item(event.item_id):
                return []
            return self._handle_input_transcript(
                event.text, event.is_final, item_id=event.item_id, cumulative=event.cumulative
            )
        if isinstance(event, RealtimeInputSpeechEndEvent):
            # The user's speech segment ended (server VAD). With transcription enabled and input audio
            # retained, cut the rolling buffer into this item's own segment so a later out-of-order
            # transcript still attaches its own audio; with transcription off there's no lagging transcript,
            # so `_finalize_untranscribed_user` consumes the rolling buffer synchronously here instead.
            self._provider_segments_input = True
            self._segment_input_audio(event.item_id)
            self._record_user_speech_span()
            return [*self._finalize_untranscribed_user(), event]
        if isinstance(event, ResponseDone):
            return self._handle_turn_complete(event)
        if isinstance(event, (PartStartEvent, PartEndEvent)):
            return self._handle_native_part_event(event)
        if isinstance(event, RealtimeResponseInterruptedEvent):
            return [event]
        if isinstance(event, RealtimeInputTranscriptionErrorEvent):
            return [*self._finalize_failed_user_item(event.item_id), event]
        # The remaining control-plane events pass through unchanged. `assert_never` makes pyright flag
        # any new non-pump `RealtimeEvent` variant that isn't handled here.
        if isinstance(
            event,
            (
                RealtimeInputSpeechStartEvent,
                RealtimeSessionReconnectEvent,
                RealtimeOutputSpeechStartEvent,
                RealtimeOutputSpeechEndEvent,
            ),
        ):
            return self._handle_control_event(event)
        if isinstance(event, RealtimeSessionErrorEvent):
            if event.recoverable:
                # A recoverable error is mid-stream: the session keeps running, so surface the event to
                # the consumer (rather than swallowing it) for observability. Only a non-recoverable
                # error ends the session, by raising.
                return [event]
            raise RealtimeError(model_name=self._error_model_name, message=event.message)
        assert_never(event)

    # --- instrumentation --------------------------------------------------------------------------

    def _final_result_text(self) -> str | None:
        """The most recent assistant reply's text, for the session span's `final_result`.

        Concatenates the text/transcript of the last `ModelResponse` that carries any, so a response
        that only made a tool call falls through to the spoken reply that followed it.
        """
        for message in reversed(self.all_messages()):
            if isinstance(message, ModelResponse) and (text := message.text):
                return text
        return None

    async def _execute_tool(
        self,
        call_part: ToolCallPart,
        *,
        validation_done: asyncio.Event,
        execution_prerequisites: tuple[asyncio.Event, ...],
        response_usage_follows: bool,
        run_step: int,
        reserved_budget: bool,
    ) -> _SettledToolResult:
        # No `execute_tool` span is created here: the `execute_tool` span is owned by the
        # `Instrumentation` capability's `wrap_tool_execute` hook, which `Agent.realtime`
        # injects into the tool runner's `ToolManager` (mirroring a classic run). That capability
        # span is the single, canonical source of tool spans; the pump task runs inside the session
        # span's OTel context, so the capability's tool span nests under the session span as a sibling
        # of the `chat` spans. The session-level `realtime` span and per-response `chat` spans below
        # stay hand-managed for now — they move onto exchange-level capability hooks when those land.
        async def on_validate(args_valid: bool) -> None:
            await self._queue.put(FunctionToolCallEvent(part=call_part, args_valid=args_valid))
            validation_done.set()
            for prerequisite in execution_prerequisites:
                await prerequisite.wait()

        async def on_inline_deferred(
            requests: DeferredToolRequests,
            results: DeferredToolResults,
        ) -> None:
            await self._queue.put(DeferredToolRequestsEvent(requests))
            await self._queue.put(DeferredToolResultsEvent(results))

        try:
            async with self._tool_manager_lock:
                ctx = self._tool_manager.ctx
                if ctx is not None:  # pragma: no branch
                    # `RunContext.messages` is a live view of the conversation in a classic run, because
                    # the graph builds each tool's context from the history so far. A session's context is
                    # built once at connect, so without this a tool would always see the seed (usually
                    # nothing) no matter how long the call has been going. Update in place, so contexts
                    # already handed out — `replace()` below keeps the same list object — see the update
                    # too.
                    ctx.messages[:] = self.all_messages()
                    # A run step here is one model turn: `_tool_run_step` increments when a
                    # `ModelResponse` is finalized, so this re-prepares the *local* manager once per
                    # turn — refreshing prepare hooks, availability filters, and retry state exactly
                    # as the graph does per model request. It never re-advertises tools mid-call: the
                    # tool list the provider sees is fixed at connect (`session.update` is sent
                    # once), so a prepare filter changing here only affects whether a call the model
                    # makes is accepted or settled as unknown-tool.
                    if ctx.run_step < run_step:
                        self._tool_manager = await self._tool_manager.for_run_step(replace(ctx, run_step=run_step))
                # Pin the step-synchronized manager for this call: a concurrent tool task can swap
                # `self._tool_manager` (its own `for_run_step` advance) between here and the calls below,
                # so re-reading the attribute there could run against a different run-step's manager.
                tool_manager = self._tool_manager

            tool_result = await tool_manager.handle_call(
                call_part,
                on_validate=on_validate,
                on_inline_deferred=on_inline_deferred,
            )
        except ToolRetryError as e:
            result_part = e.tool_retry
            user_content = None
        except ToolFailedError as e:
            # A tool that raised `ToolFailed` yields a `failed` result rather than a retry. Send it
            # back so the model sees the failure — error-key-wrapped below (via
            # `model_response_str_and_user_content`) for the string-only tool channel, since realtime
            # providers have no native failed-tool flag.
            result_part = e.tool_failed
            user_content = None
        except (ApprovalRequired, CallDeferred, RunCancelled) as e:
            result_part = _unsettled_call_return(call_part, e)
            user_content = None
        else:
            result_part, user_content = _build_session_tool_return(tool_result, call_part, tool_manager)
        finally:
            # The call has settled: on success `handle_call` has already recorded it on
            # `usage.tool_calls` in this same event-loop segment (so no limit check can observe the
            # reservation and the recorded call at once), and on failure or cancellation there is
            # nothing to count. Released only if it was claimed — a declaratively deferred call takes
            # no reservation, so decrementing here would drive the counter negative and under-project
            # every concurrent call.
            self._tool_calls_in_flight -= int(reserved_budget)

        if isinstance(result_part, RetryPromptPart):
            output = result_part.model_response()
            wire_content: list[UserContent] = []
        else:
            output, wire_content = result_part.model_response_str_and_user_content()
        if isinstance(user_content, str):
            wire_content.append(user_content)
        elif user_content:
            wire_content.extend(user_content)
        if not response_usage_follows:
            await self._drain_pending_messages('asap')
        self._reserve_response_request()
        try:
            await self._send_frame(
                ToolResult(
                    tool_call_id=call_part.tool_call_id,
                    output=output,
                    content=wire_content or None,
                )
            )
        except BaseException:
            self._pending_response_requests -= 1
            raise
        return result_part, user_content

    # --- streaming --------------------------------------------------------------------------------

    def _notify_pending_messages(self, priority: PendingMessagePriority) -> None:
        """Wake a pending-message drain from either an async tool or a sync-tool worker thread."""
        loop = self._loop
        if loop is not None and not self._closed:
            try:
                loop.call_soon_threadsafe(self._start_pending_message_drain, priority)
            except RuntimeError:
                pass

    def _start_pending_message_drain(self, priority: PendingMessagePriority) -> None:
        if self._closed:
            return
        task = asyncio.create_task(self._drain_pending_messages(priority))
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
            response_active = (
                self._active_assistant is not None
                or self._response_parts
                or self._native_tool_parts
                or self._tool_calls_awaiting_usage
                or (priority == 'when_idle' and self._response_limit_checked)
            )
            if response_active:
                if priority == 'asap':
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
        projected = dataclasses.replace(self.usage, tool_calls=self.usage.tool_calls + self._tool_calls_in_flight + 1)
        self._usage_limits.check_before_tool_call(projected)

    def _check_usage_limits(self) -> None:
        if self._usage_limits is None:
            return
        self._usage_limits.check_tokens(self.usage)
        self._usage_limits.check_cost(self.usage)

    def _reserve_response_request(self) -> None:
        """Claim the request budget for a response this session is about to solicit.

        `usage.requests` only counts responses already finalized, so a reservation covers the ones
        between: those solicited but not yet started, and the one in flight. Without them, sends
        issued back-to-back would each see the same count and oversubscribe the budget.
        """
        if self._usage_limits is not None:
            in_flight = 1 if self._response_limit_checked else 0
            projected = dataclasses.replace(
                self.usage, requests=self.usage.requests + self._pending_response_requests + in_flight
            )
            self._usage_limits.check_before_request(projected)
        self._pending_response_requests += 1

    def _begin_response(self) -> None:
        """Take the reservation for the response that's starting, or make the check now if it has none.

        With server-side voice activity detection the provider starts a response on its own, so there
        was no send to check ahead of: the check happens here instead, at the response's first event,
        before any of its content or usage is processed.
        """
        if self._response_limit_checked:
            return
        if self._pending_response_requests:
            self._pending_response_requests -= 1
        elif self._usage_limits is not None:
            self._usage_limits.check_before_request(self.usage)
        self._response_limit_checked = True

    def _accumulate_response_usage(self, event: SessionUsage) -> None:
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

    async def _handle_usage_event(self, event: SessionUsage) -> None:
        if event.response_scoped:
            self._begin_response()
        self.usage.incr(event.usage)
        self._check_usage_limits()
        if event.response_scoped:
            if self._usage_limits is not None:
                self._usage_limits.check_per_request_input_tokens(
                    (self._pending_response_usage + event.usage).input_tokens
                )
            self._accumulate_response_usage(event)
        if self._asap_drain_ready:
            self._asap_drain_ready = False
            await self._drain_pending_messages('asap')

    async def _run_tool(
        self,
        call_part: ToolCallPart,
        *,
        validation_done: asyncio.Event,
        execution_prerequisites: tuple[asyncio.Event, ...],
        completion: asyncio.Event,
        response_usage_follows: bool,
        run_step: int,
        reserved_budget: bool,
        order_index: int,
        ordered_events: bool,
    ) -> None:
        """Run a tool and feed its completion (or failure) back through the queue."""
        try:
            result_part, content = await self._execute_tool(
                call_part,
                validation_done=validation_done,
                execution_prerequisites=execution_prerequisites,
                response_usage_follows=response_usage_follows,
                run_step=run_step,
                reserved_budget=reserved_budget,
            )
        except asyncio.CancelledError:
            raise
        except BaseException as e:
            # Surface the failure through the queue so the consumer re-raises it, instead of letting it
            # vanish into `__aexit__`'s cleanup-only drain and hang the session on a completion that
            # never arrives.
            await self._queue.put(e)
            return
        finally:
            validation_done.set()
            completion.set()
            self._tool_completion_events.discard(completion)
            # Settled (completed, failed, or cancelled): no longer cancellable by `ToolCallCancelled`.
            self._pending_tool_calls.pop(call_part.tool_call_id, None)
        events = self._complete_tool_call(call_part, result_part, content)
        if ordered_events:
            # Held until nothing is still running, then released in call order — the graph waits for a
            # whole segment (`ALL_COMPLETED`) and replays it by index for the same reason. The release
            # is driven from `_tool_task_done`, which runs even for a tool that raised or was
            # cancelled, so a failed sibling can't strand the batch.
            self._ordered_tool_events[order_index] = events
        else:
            for event in events:
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
        self._release_ordered_tool_events()
        # Wake the queue reader so it can finish once both the pump and the last tool are done.
        self._queue.put_nowait(self._queue_changed)

    def _release_ordered_tool_events(self) -> None:
        """Emit `parallel_ordered_events` results in call order, once nothing is still running.

        Called from every tool task's done-callback, so the last one to settle — successful, failed, or
        cancelled — flushes the batch, and a tool that never produced events can't hold it back.
        """
        if self._tool_completion_events or not self._ordered_tool_events:
            return
        for order_index in sorted(self._ordered_tool_events):
            for event in self._ordered_tool_events[order_index]:
                self._queue.put_nowait(event)
        self._ordered_tool_events.clear()

    async def _handle_pump_event(
        self,
        event: RealtimeCodecEvent,
    ) -> bool:
        """Process one upstream event onto the queue; return `True` to stop the pump (a limit tripped)."""
        if isinstance(event, ToolCall):
            if self._accept_item(event.item_id, event.tool_call_id):
                await self._dispatch_tool_call(event)
            return False
        return await self._handle_non_tool_pump_event(event)

    async def _dispatch_tool_call(self, event: ToolCall) -> None:
        """Fold a tool call into the response and start its background task."""
        # A declaratively deferred call (`requires_approval=True`, an external tool) is resolved by
        # a handler or refused — it never reaches the tool body and so never increments
        # `usage.tool_calls`. The graph leaves those out of its own pre-check projection
        # (`_tool_execution` builds `function_indices` from `('function', 'unknown')`), so charging
        # the budget for one here could trip the limit over a call that costs nothing. An unknown
        # tool name *is* charged, exactly as `'unknown'` is projected there.
        tool_def = self._tool_manager.get_tool_def(event.tool_name)
        reserves_budget = tool_def is None or not tool_def.defer
        if reserves_budget:
            self._check_tool_call_limit()
            self._tool_calls_in_flight += 1
        # Captured at dispatch so every call from one response runs at the step in effect when the
        # response produced them, the way a graph run's whole batch shares the step advanced before
        # its request. Read at execution time instead, a call held behind a `sequential` barrier
        # could observe a later response's advance and land a step ahead of its own batch.
        tool_run_step = self._tool_run_step
        call_part = ToolCallPart(
            tool_name=event.tool_name,
            args=event.args,
            tool_call_id=event.tool_call_id,
        )
        for out in self._handle_tool_call_part(
            call_part,
            response_usage_follows=event.response_usage_follows,
        ):
            await self._queue.put(out)
        mode = self._tool_manager.get_parallel_execution_mode()
        is_barrier = mode == 'sequential' or self._tool_manager.is_sequential(call_part)
        # `parallel_ordered_events` keeps calls concurrent but hands their result events to the
        # consumer in the order the model asked for them. Claimed here, at dispatch, because that
        # is the order — the graph replays a segment's events by index for the same reason. Only
        # the events are held back: the provider still gets each result the moment it lands, and
        # history is already assembled in call order by `_insert_tool_return`, exactly as the graph
        # keeps `output_parts` in emission order whatever the mode.
        order_index = self._next_tool_order_index
        self._next_tool_order_index += 1
        completion = asyncio.Event()
        if is_barrier:
            execution_prerequisites = tuple(self._tool_completion_events)
            self._last_tool_barrier = completion
        elif self._last_tool_barrier is not None:
            execution_prerequisites = (self._last_tool_barrier,)
        else:
            execution_prerequisites = ()
        self._tool_completion_events.add(completion)
        validation_done = asyncio.Event()
        task = asyncio.create_task(
            self._run_tool(
                call_part,
                validation_done=validation_done,
                execution_prerequisites=execution_prerequisites,
                completion=completion,
                response_usage_follows=event.response_usage_follows,
                run_step=tool_run_step,
                reserved_budget=reserves_budget,
                order_index=order_index,
                ordered_events=mode == 'parallel_ordered_events',
            )
        )
        self._background_tasks.add(task)
        self._pending_tool_calls[call_part.tool_call_id] = (task, call_part)
        task.add_done_callback(self._tool_task_done)
        # Deliberate, bounded block: this waits for argument *validation* only — `on_validate` sets
        # the event before any execution prerequisite or the tool body runs, and `_run_tool`'s
        # `finally` sets it if validation itself raises — so `FunctionToolCallEvent` is enqueued
        # before the pump processes any later event, without the pump ever waiting on user tool code.
        await validation_done.wait()

    async def _handle_non_tool_pump_event(self, event: RealtimeCodecEvent) -> bool:
        """Process an upstream event other than a tool call; return `True` to stop the pump."""
        # `_handle_pump_event` routes every `ToolCall` to `_dispatch_tool_call`, so the remaining union
        # is what `_translate_event` accepts; asserted rather than re-tested so a future codec event
        # that slips past the dispatcher fails loudly instead of reaching the wrong translator.
        assert not isinstance(event, ToolCall)
        if isinstance(event, ConversationCreated):
            return False
        if isinstance(event, ConversationItemCreated):
            self._handle_conversation_item(event)
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
                    content=INTERRUPTED_TOOL_RETURN_CONTENT,
                    tool_call_id=call_part.tool_call_id,
                    outcome='interrupted',
                )
                for out in self._complete_tool_call(call_part, cancelled_part):
                    await self._queue.put(out)
            return False
        if isinstance(event, SessionUsage):
            await self._handle_usage_event(event)
            return False
        for out in self._translate_event(event):
            self._publish_taps(out)
            await self._queue.put(out)
        if isinstance(event, ResponseDone):
            await self._drain_pending_messages('asap')
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
            if not self._closed:
                self._finish_taps()
            await self._queue.put(self._queue_changed)
            if token is not None:
                otel_context.detach(token)

    def _ensure_streamable(self) -> None:
        if not self._entered:
            raise UserError('Enter the realtime session with `async with` before streaming it.')
        if self._closed:
            raise UserError('This realtime session is closed and cannot be streamed.')

    def _start_pump(self) -> None:
        # Only once the session owns its context: before `__aenter__` there is no session span to
        # attach the loop to, and teardown has nothing tracking the task.
        if self._entered and self._pump_task is None:
            self._pump_task = asyncio.create_task(self._pump(self._session_instrumentation.context))

    def _publish_taps(self, event: RealtimeEvent) -> None:
        if isinstance(event, PartDeltaEvent) and isinstance(delta := event.delta, SpeechPartDelta):
            if delta.audio_chunk:
                for queue in self._audio_taps:
                    self._audio_tap_drops += _put_tap(queue, delta.audio_chunk)
            if delta.transcript is not None and delta.speaker is not None:
                # Keyed on the running transcript, not on the added text: a revision adds nothing, and
                # gating on that would drop the very correction a caption UI needs.
                text = delta.transcript_delta or ''
                transcript = delta.transcript or self._transcript_so_far.get(event.index, '') + text
                self._transcript_so_far[event.index] = transcript
                if self._transcript_delta_taps:
                    update = TranscriptUpdate(
                        index=event.index, speaker=delta.speaker, delta=text, transcript=transcript
                    )
                    for queue in self._transcript_delta_taps:
                        self._transcript_tap_drops += _put_tap(queue, update)
        elif isinstance(event, PartEndEvent) and isinstance(part := event.part, SpeechPart):
            self._transcript_so_far.pop(event.index, None)
            if part.transcript:
                for queue in self._transcript_taps:
                    self._transcript_tap_drops += _put_tap(queue, part)

    def _finish_taps(self, *, discard_pending: bool = False) -> None:
        for queue in (*self._audio_taps, *self._transcript_taps, *self._transcript_delta_taps):
            if discard_pending:
                while not queue.empty():
                    queue.get_nowait()
            queue.put_nowait(self._tap_finished)

    async def __aiter__(self) -> AsyncIterator[RealtimeEvent]:
        """Read translated events from the session queue without owning session resources."""
        if not self._entered:
            raise UserError('Enter the realtime session with `async with` before iterating it.')
        if self._closed:
            raise UserError('This realtime session is closed and cannot be iterated.')
        if self._iterator_active:
            raise UserError('This realtime session is already being iterated.')
        if self._stream_exhausted:
            raise UserError('This realtime session event stream has already ended.')

        self._iterator_active = True
        self._stream_consumed = True
        self._start_pump()

        async def queue_events() -> AsyncIterator[RealtimeEvent]:
            while True:
                item = await self._queue.get()
                if item is self._queue_changed:
                    # A pump error takes priority over stuck background tools. Their cancellation and
                    # drain belong to `__aexit__`, which runs as this exception leaves the owner block.
                    if self._pump_error is not None:
                        self._stream_exhausted = True
                        raise self._pump_error
                    if self._pump_finished and not self._background_tasks and self._queue.empty():
                        self._stream_exhausted = True
                        return
                    continue
                yield _as_event(item)  # re-raises if a tool failed

        source = queue_events()
        stream: AsyncIterable[RealtimeEvent] = source
        if self._wrap_event_stream is not None:
            # The wrapper vocabulary is the `AgentStreamEvent` superset, but a wrapper applied to a
            # realtime stream must yield the `RealtimeEvent` subset exposed by the session.
            stream = cast('AsyncIterable[RealtimeEvent]', self._wrap_event_stream(source))
        stream_iterator = aiter(stream)
        try:
            async for event in stream_iterator:  # pragma: no branch
                yield event
        finally:
            try:
                await aclose_all((stream_iterator, stream, source))
            finally:
                self._iterator_active = False
