"""A realtime session that wraps a [`RealtimeConnection`][pydantic_ai.realtime.codec.RealtimeConnection] with automatic tool execution."""

from __future__ import annotations as _annotations

import asyncio
import dataclasses
import weakref
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator, Callable, Sequence
from copy import copy
from dataclasses import dataclass, field, replace
from functools import partial
from time import time_ns
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, Never, TypeAlias, TypeVar, assert_never, cast, overload

import anyio
from anyio import Lock
from opentelemetry import context as otel_context
from opentelemetry.context import Context
from typing_extensions import TypeAliasType

from .. import _agent_graph
from .._enqueue import EnqueueContent, PendingMessage, PendingMessagePriority, PendingMessageQueue
from .._instrumentation import annotate_tool_call_otel_metadata
from .._run_context import context_window_fraction
from .._tool_execution import (
    _reject_unloaded_capability_reveals,  # pyright: ignore[reportPrivateUsage]
    build_tool_return_part,
    cancelled_sub_agent_return,
)
from .._utils import aclose_all, cancel_and_drain, dataclasses_no_defaults_repr, fill_run_metadata
from ..conversation import Conversation
from ..exceptions import (
    ApprovalRequired,
    CallDeferred,
    RunCancelled,
    ToolFailedError,
    ToolRetryError,
    UsageLimitExceeded,
    UserError,
)
from ..messages import (
    INTERRUPTED_TOOL_RETURN_CONTENT,
    BinaryAudio,
    BinaryContent,
    BinaryImage,
    DeferredToolRequestsEvent,
    DeferredToolResultsEvent,
    EnqueuedMessagesEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
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
    ToolCallPart,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
)
from ..native_tools import SUPPORTED_NATIVE_TOOLS
from ..run import AgentRunResult
from ..tool_manager import ToolManager
from ..usage import RequestUsage, RunUsage, UsageLimits
from ._core import (
    MAX_CUTTABLE_OUTPUT_ITEMS,
    AudioCleared,
    AudioSent,
    Closed,
    CoreInput,
    Effect,
    ExchangeAbandoned,
    ExchangeOver,
    InputSent,
    InputWithdrawn,
    Interrupted,
    ReceiveEnded,
    ResponseClosed,
    ResponseOpened,
    ResponseOverdue,
    ResponseRefused,
    SessionCore,
    ToolCallRefused,
    ToolCallsPruned,
    ToolReturned,
    ToolRoundsOver,
    TranscriptOverdue,
    UsageCounted,
)
from ._instrumentation import (
    SessionInstrumentation,
)
from ._lifecycle import LifecycleEvent, ResponseStarted, StaleEvent
from ._retained_audio import RetainedAudioBudget
from ._utils import (
    seed_pcm_audio,
)
from .codec import (
    AudioDelta,
    CancelResponse,
    ClearAudio,
    CommitAudio,
    ConversationCreated,
    ConversationItemCreated,
    CreateResponse,
    InputRejected,
    InputTranscript,
    OutputTranscript,
    RealtimeCodecEvent,
    RealtimeConnection,
    RealtimeInput,
    RealtimeSessionInput,
    ResponseDone,
    SessionUsage,
    TextContext,
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
    from .model import RealtimeModel, RealtimeProviderSession

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
    | EnqueuedMessagesEvent
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
other shared message parts, including [`TextPart`][pydantic_ai.messages.TextPart],
[`ToolCallPart`][pydantic_ai.messages.ToolCallPart], and
[`NativeToolReturnPart`][pydantic_ai.messages.NativeToolReturnPart]), tool execution as
[`FunctionToolCallEvent`][pydantic_ai.messages.FunctionToolCallEvent] /
[`FunctionToolResultEvent`][pydantic_ai.messages.FunctionToolResultEvent], deferred calls and their resolution
as [`DeferredToolRequestsEvent`][pydantic_ai.messages.DeferredToolRequestsEvent] /
[`DeferredToolResultsEvent`][pydantic_ai.messages.DeferredToolResultsEvent], enqueued-message delivery
as [`EnqueuedMessagesEvent`][pydantic_ai.messages.EnqueuedMessagesEvent], and the rest as realtime
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


# Retained `SpeechPart.audio` is a WAV container, so the sample format is self-describing; live audio is raw PCM.
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

# Providers generate speech several times faster than it plays (OpenAI 6-9x, Gemini 3-4x), so a
# device-paced player routinely has most of a reply queued before it has heard the first seconds of it.
# The audio window is therefore measured in playback time rather than chunks (whose size varies from
# ~40 ms on Gemini to 400 ms on OpenAI) and sized to hold any realistic reply generated ahead of
# playback; drop-oldest is only a memory cap for a view that stopped being consumed. Five minutes of
# 24 kHz PCM16 is about 14 MB. Transcript items are short strings, and dropping one silently corrupts
# the text a user is reading, so they get a deep window for a trivial cost.
_AUDIO_TAP_SECONDS = 300
# How long a user turn that barged in waits for the response it interrupted to be recorded, so it can
# follow that response in history. The provider's terminal for a cut-off response normally arrives within
# a second of the user starting to speak; past this, the turn is recorded where history stands instead of
# staying out of `all_messages()` for as long as a provider keeps talking.
_BARGE_IN_TURN_HOLD_SECONDS = 5.0
# How long a spoken turn that joined the conversation may hold back history recorded after it while it waits for
# its transcript, with the session core: the turn is never inserted ahead of what was recorded since. Counted from
# when something after it is ready (normally the reply). Transcripts can arrive after the reply and the next turn
# (recorded: push-to-talk on gpt-realtime and gpt-realtime-2.1), so this is generous; past it, the turn is recorded
# with what it has, so a transcript that never comes can't hide the rest of the conversation for good.
_TRANSCRIPT_HOLD_SECONDS = 30.0
#: How long closing waits for the provider to end its session and report the usage it only reports then.
#: Not waited for at all when the session is closing because it was cancelled.
_END_SESSION_TIMEOUT = 2.0
# The byte budget alone would let a stream of tiny deltas queue millions of objects, so the window is
# also capped in chunks: five minutes at 10 ms apiece, well below any provider's real chunk size.
_AUDIO_TAP_MAX_CHUNKS = 30_000
_TRANSCRIPT_TAP_SIZE = 512
_SESSION_DELTA_QUEUE_SIZE = 512
# Structural events are some five per turn against one delta per audio frame, so they are not what
# makes an unread queue large — but they are never superseded the way deltas are, so without their own
# bound a session nothing iterates keeps every one of them for as long as it runs. The same 512 buys a
# late iterator around a hundred turns of structure, comfortably more history than the deltas it keeps.
_SESSION_STRUCTURAL_QUEUE_SIZE = 512
_TapItem = TypeVar('_TapItem')
_Tap = TypeVar('_Tap')


def _put_tap(queue: asyncio.Queue[_TapItem], item: _TapItem) -> _TapItem | None:
    """Put without blocking the pump, retaining the newest bounded window.

    The queue is sized one over its data window so the completion sentinel always fits. Returns the
    older item that was dropped to make room, or `None` when nothing was dropped.
    """
    dropped: _TapItem | None = None
    if queue.qsize() >= queue.maxsize - 1:
        dropped = queue.get_nowait()
    queue.put_nowait(item)
    return dropped


class _TapView(AsyncIterator[_TapItem]):
    """A `stream_audio()` / `stream_transcripts()` view whose subscription is registered before iteration.

    The generator discards its tap in `finally`, but an async generator that was never started has no
    frame to unwind, so `aclose()` would leave the tap registered for as long as the caller held the
    view. Wrapping it gives `aclose()` a hook, and a view that is simply abandoned is released when
    collected.
    """

    def __init__(
        self,
        iterator: AsyncGenerator[_TapItem, None],
        taps: set[_Tap],
        tap: _Tap,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self._iterator = iterator
        self._discard = weakref.finalize(self, taps.discard, tap)
        self._finish = weakref.finalize(self, on_close) if on_close is not None else None

    def __aiter__(self) -> AsyncIterator[_TapItem]:
        return self

    async def __anext__(self) -> _TapItem:
        return await self._iterator.__anext__()

    async def aclose(self) -> None:
        self._discard()
        if self._finish is not None:
            self._finish()
        await self._iterator.aclose()


@dataclass(eq=False)
class _AudioTap:
    """One `stream_audio()` subscription, with the accounting `interrupt(played_bytes=...)` needs.

    `subscribed_at_bytes` anchors the tap in the session's emitted-audio coordinates, and
    `dropped_bytes` counts audio removed before the consumer saw it (overflow and barge-in flushes),
    so a device-side playback position can be mapped back into those coordinates.
    """

    queue: asyncio.Queue[bytes | object]
    subscribed_at_bytes: int
    max_buffered_bytes: int
    """How much audio the queue may hold before the oldest chunks are dropped to make room."""
    buffered_bytes: int = 0
    """Audio currently queued for the consumer."""
    dropped_bytes: int = 0
    """Gaps the consumer has already moved past, which its playback position no longer accounts for."""
    pending_dropped_bytes: int = 0
    """Gaps that are still ahead of the consumer: dropped from the queue, but not yet stepped over.

    Drop-oldest discards the chunk that would have been delivered next, so a gap sits ahead of the
    playback position until the consumer takes the chunk beyond it — counting it before then would
    place the position further into the response than the device has actually reached.
    """
    played_bytes: int = 0
    """Chunks the consumer finished with — counted when it resumes the iterator for the next one."""
    progress: asyncio.Event = field(default_factory=asyncio.Event)
    """Set when playback advances or the view ends, waking `wait_for_playback()`."""
    ended: bool = False

    def put(self, chunk: bytes) -> int:
        """Queue a chunk without blocking the pump, dropping the oldest to stay within the byte and chunk budgets.

        The newest chunk is always kept, even on its own over budget. Returns how many chunks were
        dropped; their bytes become a gap ahead of the consumer.
        """
        dropped = 0
        while self.buffered_bytes and (
            self.buffered_bytes + len(chunk) > self.max_buffered_bytes or self.queue.qsize() >= _AUDIO_TAP_MAX_CHUNKS
        ):
            # The sentinel can't be dropped: it is only enqueued once the pump has finished, after
            # which nothing publishes.
            oldest = self.queue.get_nowait()
            assert isinstance(oldest, bytes)
            self.buffered_bytes -= len(oldest)
            self.pending_dropped_bytes += len(oldest)
            dropped += 1
        self.queue.put_nowait(chunk)
        self.buffered_bytes += len(chunk)
        return dropped

    def finish(self) -> None:
        self.ended = True
        self.progress.set()


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


def _chain_context(error: BaseException, context: BaseException) -> None:
    """Attach `context` at the end of `error`'s context chain, where a traceback shows it first.

    `error.__context__` is usually taken already (a mapped provider error carries the SDK's), so it can't
    just be set.
    """
    seen: set[int] = set()
    tail = error
    while tail.__context__ is not None:
        if tail is context or id(tail) in seen:
            return
        seen.add(id(tail))
        tail = tail.__context__
    if tail is not context:
        tail.__context__ = context


def _unsettled_call_return(call: ToolCallPart, error: BaseException) -> ToolReturnPart:
    """The return a session records when a tool call couldn't settle normally.

    These are cases the graph resolves by ending or isolating the run, which a live conversation
    can't do — so each gets a deliberate local settlement rather than leaving a dangling call.
    """
    if isinstance(error, RunCancelled):
        # Exactly the graph path's settlement (this session's own cancellation arrives as
        # `CancelledError`, which `_run_tool` re-raises untouched) — shared so the two can't drift.
        return cancelled_sub_agent_return(call, error)
    elif isinstance(error, (ApprovalRequired, CallDeferred)):
        # `handle_call` already gave the `HandleDeferredToolCalls` capability handler the chance to
        # resolve the deferral inline (approve, deny, retry, or substitute a result); reaching here
        # means no handler resolved it. The graph's fallback — pausing the run with a
        # `DeferredToolRequests` output — has no realtime analog (a live conversation can't wait for an
        # out-of-band result, and the provider expects an answer on the string-only tool channel).
        reason = 'requires approval' if isinstance(error, ApprovalRequired) else 'runs externally'
        content = f'Error: The {call.tool_name!r} tool {reason} and cannot be completed during a realtime session.'
    else:
        content = 'The tool raised an unhandled error.'
    return ToolReturnPart(
        tool_name=call.tool_name,
        content=content,
        tool_call_id=call.tool_call_id,
        tool_kind=call.tool_kind,
        outcome='failed',
    )


def _pending_message_text(pending: PendingMessage) -> str:
    """Render enqueued messages down to the text a realtime session can deliver.

    Text-only parts join across messages, and a mid-conversation `SystemPromptPart` is delivered as
    `<system>`-tagged user text — the same degradation `Model.prepare_messages` applies for a
    standard run's non-leading system prompts. Anything else can't cross the live input channel.
    """
    error = UserError(
        '`RealtimeSession.enqueue()` and `RunContext.enqueue()` in a realtime session support plain-text prompts and system-prompt '
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


class _RealtimePendingMessages(PendingMessageQueue):
    """A `RunContext.enqueue` queue that validates content and wakes the live session for delivery."""

    def __init__(self) -> None:
        super().__init__()
        self._on_append: Callable[[PendingMessagePriority], None] | None = None

    def bind(self, on_append: Callable[[PendingMessagePriority], None]) -> None:
        self._on_append = on_append

    def append(self, pending: PendingMessage) -> None:
        _pending_message_text(pending)
        super().append(pending)
        if self._on_append is not None:
            self._on_append(pending.priority)

    def has_priority(self, priority: PendingMessagePriority) -> bool:
        with self._lock:
            return any(pending.priority == priority for pending in self)


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
    without limit. Audio kept by `audio_retention` is bounded the same way: `retain_audio_max_seconds`
    (default `1800`, 30 minutes across both speakers) caps how much stays in memory, evicting the
    oldest retained audio first while keeping its transcript.

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
        handle_barge_in: bool = False,
        retain_images_every_n: int = 1,
        retain_images_max: int | None = 100,
        retain_audio_max_seconds: float | None = 1800,
        message_history: Sequence[ModelMessage] | None = None,
        profile: RealtimeModelProfile | None = None,
        owns_media: bool = True,
        provider_session: RealtimeProviderSession | None = None,
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
        self._responses_are_requests = self._profile.get('responses_are_requests', True)
        # The latest context-window fraction the provider reported, if it reports one at all.
        self._reported_context_window_used: float | None = None
        # Whether this session owns the audio transport. `False` for a WebRTC sideband session: the
        # browser exchanges audio with the provider directly, and this connection is only the control
        # plane, so the audio methods are unavailable and no audio bytes flow over it (transcripts still
        # build history). Set by the connect path when a `provider_session` is attached.
        self._owns_media = owns_media
        # The WebRTC call a sideband session runs, which `hang_up()` ends; `None` on a session that owns
        # its own connection, where closing it is what ends the call.
        self._model = model
        self._provider_session = provider_session
        self._hang_up_requested = False
        # The teardown's own failed hangup, which the caller that collects it from `close()` has been told.
        self._teardown_hang_up_error: Exception | None = None
        self._hung_up = False
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
        self._audio_budget = RetainedAudioBudget(
            retain_audio_max_seconds,
            input_sample_rate=self.audio_input_sample_rate,
            output_sample_rate=self.audio_output_sample_rate,
        )
        self._retain_input = audio_retention in ('input_audio', 'all') and self._audio_budget.retains_audio
        self._retain_output = audio_retention in ('output_audio', 'all') and self._audio_budget.retains_audio
        self._handle_barge_in = handle_barge_in
        if retain_images_every_n < 1:
            raise UserError('`retain_images_every_n` must be at least 1.')
        self._retain_images_every_n = retain_images_every_n
        if retain_images_max is not None and retain_images_max < 0:
            raise UserError('`retain_images_max` must be at least 0, or `None` for no limit.')
        self._retain_images_max = retain_images_max
        # Retained image requests in arrival order, so the cap can evict the oldest.
        self._retained_image_requests: list[ModelRequest] = []
        self._sent_image_count = 0
        # Every `send()` made on the connection is numbered, the same way the connection numbers the
        # inputs it receives, so an `InputRejected` can name the request whose content the provider
        # refused. Weak, so an image evicted from history by the retention cap isn't kept alive here.
        self._inputs_sent = 0
        self._input_requests: weakref.WeakValueDictionary[int, ModelRequest] = weakref.WeakValueDictionary()
        self.usage = usage if usage is not None else RunUsage()
        """Cumulative token usage and tool-call counts for the session, updated as events stream in.

        Pass `usage` to [`Agent.realtime`][pydantic_ai.agent.AbstractAgent.realtime] to accumulate
        into a shared [`RunUsage`][pydantic_ai.usage.RunUsage]; otherwise a fresh one is used.
        """
        # `ToolManager` increments `tool_calls` on its context's usage as each call succeeds, and the
        # session is the single authority for `session.usage`, so the two have to be the same object.
        # A caller can pass a `usage` the manager wasn't built with, and rebinding the context properly
        # would mean `for_run_step`, which is async and can't run here.
        if (ctx := self._tool_manager.ctx) is not None:
            ctx.usage = self.usage
        # What the session span reports: only the token usage this session's own responses recorded.
        # `self.usage` can start from a total carried in with `usage=` (or a `Conversation`), and a run
        # a tool starts with `usage=ctx.usage` accumulates into it too; reporting that on the span would
        # count both again against the spans they came from. Credited where `self.usage` records it.
        self._span_usage = RunUsage()

        # The conversation the session was opened with: the core's history starts from it.
        self._seeded: list[ModelMessage] = list(message_history or [])
        # When the provider's VAD reported the user's current speech segment starting, in OTel's
        # nanosecond clock, so the `user speech` span can be backdated to it (see
        # `_record_user_speech_span`). `None` while nobody is speaking.
        self._user_speech_started_at: int | None = None
        # Whether the provider's own turn detection has already cancelled the response now in
        # flight, because the user spoke over it. A client `CancelResponse` racing the server's own
        # can be applied to the *next* response and silence the reply to the barge-in, so an
        # interruption of a response the server is already stopping sends only the truncation.
        # Latched at speech onset rather than read off the clock: the pump can reach the speech
        # *end* before the app has even dequeued the start it is reacting to, and the fact this
        # guards is a property of the response, not of how long the user kept talking. Released
        # when the next response opens, and by a reconnect that discards the one it referred to.
        self._server_cancelled_the_response_on_speech = False
        self._exchange_progress = asyncio.Event()
        # Whether the frame being handled lost a response under way (a reconnect that didn't carry it).
        self._frame_lost_response = False
        # Answered calls of tool rounds that are over, which the connection may delete from the provider's
        # conversation once the frame is handled (see `RealtimeConnection._prune_tool_calls`).
        self._tool_rounds_over: list[ToolCallPart] = []
        # With the session core: the timer of each spoken turn waiting for its transcript (see
        # `_TRANSCRIPT_HOLD_SECONDS`), and of each response holding back user turns (see
        # `_BARGE_IN_TURN_HOLD_SECONDS`).
        self._transcript_watchdogs: dict[str, asyncio.TimerHandle] = {}
        self._barge_in_watchdogs: dict[str, asyncio.TimerHandle] = {}

        # The session context is the single owner of the receive pump and background tool tasks.
        # It starts the pump on entry and never tears it down before `close()`: an early `break` can
        # abandon the reader generator without affecting resource lifetime, and `__aexit__` still
        # drains everything before the connection and toolset close.
        self._queue: deque[RealtimeEvent | object] = deque()
        self._queue_event = asyncio.Event()
        self._queue_delta_count = 0
        self._queue_dropped_deltas = 0
        self._queue_structural_count = 0
        self._queue_dropped_structural = 0
        self._queue_changed = object()
        self._tap_finished = object()
        self._audio_taps: set[_AudioTap] = set()
        self._transcript_taps: set[asyncio.Queue[SpeechPart | object]] = set()
        self._transcript_delta_taps: set[asyncio.Queue[TranscriptUpdate | object]] = set()
        self._audio_tap_drops = 0
        self._transcript_tap_drops = 0
        # Emitted-audio coordinates for `interrupt(played_bytes=...)`: cumulative output audio bytes,
        # the offset and part index where the current audio part began, and the index of a part whose
        # remaining audio a barge-in erased (its stragglers are dropped as they arrive).
        self._emitted_audio_bytes = 0
        self._turn_audio_start_bytes = 0
        self._audio_part_index: int | None = None
        self._interrupted_audio_part_index: int | None = None
        # Where each recent assistant audio part begins in the emitted audio, with its provider output
        # item. Generation outruns playback, so the reply being heard is often not the last one
        # generated: a barge-in truncates the item actually playing (and later ones, never heard, at 0).
        self._audio_items: deque[tuple[int, str | None]] = deque(maxlen=MAX_CUTTABLE_OUTPUT_ITEMS)
        # Where the audio of the current connection began: a reconnect's fresh conversation holds the items
        # before it only as replayed text, so a barge-in can't truncate them there, and history doesn't cut them.
        self._connection_audio_start_bytes = 0
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
        # `ToolManager` adds a call to `usage.tool_calls` only once it *succeeds*, so calls still
        # running aren't visible there. Reserved when a `ToolCall` passes `_check_tool_call_limit`,
        # released the moment `handle_call` settles (the same event-loop segment that records a
        # success), so a burst of parallel calls can't each clear a limit only one of them fits under.
        self._tool_calls_in_flight = 0
        # `parallel_ordered_events` buffers: call-order index -> the events that call produced.
        self._ordered_tool_events: dict[int, list[RealtimeEvent]] = {}
        self._next_tool_order_index = 0
        # Whether an `asap` drain found a response under way, so it runs again once that response is over.
        self._asap_drain_deferred = False
        self._pump_task: asyncio.Task[None] | None = None
        self._pump_error: Exception | None = None
        self._pump_finished = False
        self._receive_ending = False
        self._iterator_active = False
        self._stream_exhausted = False
        self._parked_errors: list[BaseException] = []
        self._delivered_errors: list[BaseException] = []
        self._entered = False
        self._closed = False
        self._closing_error: BaseException | None = None
        self._teardown: asyncio.Task[None] | None = None
        self._close_error: BaseException | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        self._traceparent_value: str | None = None
        self._result: AgentRunResult[str] | None = None

        # The core, which keeps history, usage, and reply waits, from the lifecycle the connection reports.
        requested_model = self._model_name
        self._core = SessionCore(
            model_name=lambda: connection.model_name or requested_model,
            provider_name=self._provider_name,
            provider_url=self._provider_url,
            conversation_id=conversation_id,
            run_id=run_id,
            responses_are_requests=self._responses_are_requests,
            answers_tool_calls_per_response=connection._answers_tool_calls_per_response,  # pyright: ignore[reportPrivateUsage]
            tracks_tool_rounds=connection._prunes_tool_calls,  # pyright: ignore[reportPrivateUsage]
            input_transcription_enabled=connection.input_transcription_enabled,
            retain_input_audio=self._retain_input,
            retain_output_audio=self._retain_output,
            retain_audio_max_seconds=retain_audio_max_seconds,
            input_sample_rate=self.audio_input_sample_rate,
            output_sample_rate=self.audio_output_sample_rate,
            seeded=self._seeded,
            usage=self.usage,
            annotate_response=self._annotate_response,
        )

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
            self._connection.set_message_history(self._core.replayable_messages)

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

        This method is idempotent. Concurrent callers wait for the same teardown, which continues if
        a caller is cancelled; leaving the session context waits for it to finish. Active
        [`stream_audio()`][pydantic_ai.realtime.RealtimeSession.stream_audio] and
        [`stream_transcripts()`][pydantic_ai.realtime.RealtimeSession.stream_transcripts] iterators
        finish cleanly, with any buffered items discarded. The surrounding model context owns the
        underlying connection, so it remains open until that context exits.

        A tool can call this method through
        [`ctx.realtime_session`][pydantic_ai.tools.RunContext.realtime_session] to end the session. The
        calling tool does not resume, and its call is recorded as interrupted. On a WebRTC sideband this only
        detaches, and the browser's call stays up: use
        [`hang_up()`][pydantic_ai.realtime.RealtimeSession.hang_up] to end it.

        Raises a failure that hasn't been delivered yet: whatever ended the session (a provider hangup,
        an exceeded `usage_limits`), or a failed tool that nothing was iterating to receive. A failure
        already raised by event iteration or an outbound session method is not raised again.
        """
        if not self._entered:
            return
        if self._teardown is None:
            self._pending_messages.close()
            self._closed = True
            self._finish_taps(discard_pending=True)
            self._exchange_progress.set()
            # A session closed without ever sending, subscribing, or iterating never started one.
            # Cancelled before state is settled below so it can't mutate state mid-settlement; the
            # task is awaited together with the rest afterwards.
            if self._pump_task is not None:
                self._pump_task.cancel()
            if (early_error := self._closing_error or self._pump_error) is not None:
                # The replies these spans cover are being torn down by a failure; record it now, before the
                # settlement below records the interrupted responses and ends the spans cleanly.
                for chat_span in self._session_instrumentation.chat_spans.values():
                    self._session_instrumentation.record_error(chat_span, early_error)
            # Settle whatever the closing session still holds open: every still-running tool call gets an
            # interrupted return, and the core records open user turns, and a reply cut off mid-generation as
            # interrupted. What that streams is discarded: the stream is closing and has no consumer left.
            self._server_cancelled_the_response_on_speech = False
            self._stop_tool_calls(list(self._pending_tool_calls))
            for event in self._connection._stop_reading():  # pyright: ignore[reportPrivateUsage]
                self._apply_core(event)
            self._apply_core(Closed())
            self._end_chat_spans_left_open()
            self._teardown = asyncio.create_task(self._finish_teardown())
        elif asyncio.current_task() in self._background_tasks:
            # A tool closing the session is cancelled by the teardown, at the wait below of its own
            # `close()` call. One that swallows that cancellation and calls `close()` again would wait
            # for a teardown that is waiting for it, so there is nothing to wait for from this side.
            return

        # `asyncio.shield` lets a cancelled caller stop waiting while the teardown continues; the
        # shielded scope makes an outer *anyio* cancellation — level-triggered, re-raised at every
        # checkpoint — wait for the teardown instead of abandoning it (see `agent_docs/concurrency.md`).
        with anyio.CancelScope(shield=True):
            await asyncio.shield(self._teardown)
        if (error := self._close_error) is not None:
            self._close_error = None
            raise error

    async def hang_up(self) -> None:
        """End the call, then close the session.

        On a [WebRTC sideband](../realtime/deployment.md#browser-webrtc-server-sideband) session,
        [`close()`][pydantic_ai.realtime.RealtimeSession.close] only detaches this server from the call,
        and the browser stays connected to the provider (and billed) until it hangs up itself. This asks the
        provider to end the call for everyone. On a session that owns its connection, closing it already
        ends the call, so this is the same as `close()`. Either way, the session is closed afterwards, as
        by `close()`.

        A tool can call this method through
        [`ctx.realtime_session`][pydantic_ai.tools.RunContext.realtime_session], as it can `close()`: the
        call is still ended after the tool itself is cancelled by the teardown.

        Raises [`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError] or
        [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError] if the provider refuses to end the call,
        and [`UserError`][pydantic_ai.exceptions.UserError] if the model can't hang up WebRTC calls.
        """
        provider_session, model = self._provider_session, self._model
        if provider_session is None or model is None:
            await self.close()
            return
        # Refused before anything is torn down, so a model that can't hang up leaves the session as it was.
        model._check_hang_up(provider_session)  # pyright: ignore[reportPrivateUsage]
        # The call is ended from the teardown, once the pump has stopped: the provider closes the sideband
        # when the call ends, which the pump would otherwise report as a lost connection.
        self._hang_up_requested = True
        close_error: BaseException | None = None
        try:
            await self.close()
        except BaseException as e:
            close_error = e
            raise
        finally:
            teardown_done = self._teardown is None or self._teardown.done()
            told_of_failure = close_error is not None and close_error is self._teardown_hang_up_error
            if not self._hung_up and teardown_done and not told_of_failure:
                # The call is still up, and this caller hasn't been told why: the session was closed (or
                # never started) before this call, another caller collected the teardown's failed hangup,
                # or one is being retried. Even if closing raised something else, the call must still end.
                try:
                    await self._hang_up_call()
                except Exception as hang_up_error:
                    if close_error is not None:
                        _chain_context(hang_up_error, close_error)
                    raise

    async def _hang_up_call(self) -> None:
        provider_session, model = self._provider_session, self._model
        assert provider_session is not None and model is not None
        await model.hang_up(provider_session)
        # Only once it worked: a hangup that failed leaves the call up, so asking again tries again.
        self._hung_up = True

    async def _finish_teardown(self) -> None:
        # A session closed without ever sending, subscribing, or iterating started no pump, so there
        # may be nothing here but the background tasks.
        pump_tasks = (self._pump_task,) if self._pump_task is not None else ()
        await cancel_and_drain(*self._background_tasks, *pump_tasks, msg='Realtime session exited')
        # A WebRTC sideband doesn't own the provider session: ending it would end the browser's call. A
        # provider that already went away is the connection's to recognize, which it can do for certain.
        # A cancelled session closes promptly instead of waiting on the provider.
        if self._owns_media and not isinstance(self._closing_error, asyncio.CancelledError):
            await self._record_final_usage()
        hang_up_error: Exception | None = None
        if self._hang_up_requested and not self._hung_up:
            try:
                await self._hang_up_call()
            except Exception as e:
                hang_up_error = self._teardown_hang_up_error = e

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
            usage=self._span_usage,
            messages=self.all_messages(),
            new_message_index=len(self._seeded) if self._seeded else None,
            final_result=self._final_result_text(),
            audio_chunks_dropped=self._audio_tap_drops,
            transcript_items_dropped=self._transcript_tap_drops,
            queue_dropped_deltas=self._queue_dropped_deltas,
            queue_dropped_structural=self._queue_dropped_structural,
        )
        self._loop = None
        for watchdog in (*self._transcript_watchdogs.values(), *self._barge_in_watchdogs.values()):
            watchdog.cancel()
        self._transcript_watchdogs.clear()
        self._barge_in_watchdogs.clear()

        # Do not hide the caller's own exception, but make sure every receive-side failure has one
        # delivery point even when iteration stopped early or was never started. Stored rather than
        # raised: `close()` raises it once, for the first caller that finishes waiting on the
        # teardown. Recorded as delivered so no later path delivers it a second time.
        if self._closing_error is None and (error := self._first_undelivered_error()) is not None:
            self._delivered_errors.append(error)
            self._close_error = error
        if hang_up_error is not None:
            # A call left up is what matters most to the caller, so it wins; anything that ended the
            # session first stays attached to it.
            if self._close_error is not None:
                _chain_context(hang_up_error, self._close_error)
            self._close_error = hang_up_error

    async def _record_final_usage(self) -> None:
        """End the provider session, recording the usage it reports only as it ends.

        Runs once nothing reads the connection any more and before the session span reports its usage, so
        what a provider bills at the very end (GPT-Live's last seconds of audio) is counted. Bounded, and
        best-effort: what the provider reported before it stopped answering in time, or before the link
        failed, is recorded, and nothing more.
        """
        recorded = False
        try:
            with anyio.move_on_after(_END_SESSION_TIMEOUT):
                async for report in self._connection._end_session():  # pyright: ignore[reportPrivateUsage]
                    if report.context_window_used is not None:
                        self._reported_context_window_used = report.context_window_used
                    self.usage.incr(report.usage)  # usage-attribution: the session owns its spans
                    if report.response_scoped and not self._responses_are_requests:
                        # A request the model made (GPT-Live's backend finishing as the session closed), held to
                        # the request limits as one reported while the session ran is.
                        self.usage.requests += 1  # usage-attribution: the session owns its spans
                        self._park_exceeded_limit(partial(self._check_final_request, report.usage))
                    self._span_usage.incr(report.usage)  # usage-attribution: what the session span reports
                    recorded = True
        except self._connection.transport_errors:
            pass
        if recorded:
            # The session is closed, so an exceeded limit is parked for `close()` to raise.
            self._check_response_boundary_limits()

    def _queue_put(self, item: RealtimeEvent | object) -> None:
        """Append an item, bounding the queue while no session iterator is active."""
        if isinstance(item, PartDeltaEvent):
            self._queue_delta_count += 1
        elif self._is_structural(item):
            self._queue_structural_count += 1
        self._queue.append(item)
        if not self._iterator_active:
            self._trim_queue()
        self._queue_event.set()

    def _is_structural(self, item: RealtimeEvent | object) -> bool:
        """Whether an item is a droppable non-delta event.

        Excludes the parked exceptions `close()` sweeps for, and the wake-up sentinels whose arrival
        is what makes the iterator re-check for a pump error or for termination — neither is history
        a later reader can do without, and both are rare enough not to be what grows the queue.
        """
        return not isinstance(item, BaseException) and item is not self._queue_changed

    def _trim_queue(self) -> None:
        while self._queue_delta_count > _SESSION_DELTA_QUEUE_SIZE:
            # Deltas make up the bulk of a backed-up queue, so the oldest one is close to the head: the
            # scan and the deque deletion are linear only in what is kept ahead of it, which the
            # structural bound below caps in turn.
            oldest = next(index for index, queued in enumerate(self._queue) if isinstance(queued, PartDeltaEvent))
            del self._queue[oldest]
            self._queue_delta_count -= 1
            self._queue_dropped_deltas += 1
        while self._queue_structural_count > _SESSION_STRUCTURAL_QUEUE_SIZE:
            position = next(
                position
                for position, queued in enumerate(self._queue)
                if not isinstance(queued, PartDeltaEvent) and self._is_structural(queued)
            )
            evicted = self._queue[position]
            del self._queue[position]
            self._queue_structural_count -= 1
            self._queue_dropped_structural += 1
            if isinstance(evicted, PartStartEvent):
                # A part's start always precedes its deltas and its end, so this is the whole part
                # going: drop the rest of it rather than leave deltas a reader cannot attach to
                # anything. Part indexes are unique for the life of the session (the core's counter
                # only ever increments), except that a late transcript streams an ended part again
                # under its index (see `SessionCore._update_closed_part`): that restart is whole by itself.
                self._drop_queued_part(position, evicted.index)

    def _drop_queued_part(self, position: int, index: int) -> None:
        """Drop the deltas and end queued from `position` on for the part at `index`, up to a restart of it."""
        while position < len(self._queue):
            queued = self._queue[position]
            if isinstance(queued, PartStartEvent) and queued.index == index:
                break
            if not isinstance(queued, (PartDeltaEvent, PartEndEvent)) or queued.index != index:
                position += 1
                continue
            del self._queue[position]
            if isinstance(queued, PartDeltaEvent):
                self._queue_delta_count -= 1
                self._queue_dropped_deltas += 1
            else:
                self._queue_structural_count -= 1
                self._queue_dropped_structural += 1

    def _queue_get_nowait(self) -> RealtimeEvent | object:
        item = self._queue.popleft()
        if isinstance(item, PartDeltaEvent):
            self._queue_delta_count -= 1
        elif self._is_structural(item):
            self._queue_structural_count -= 1
        return item

    async def _queue_get(self) -> RealtimeEvent | object:
        while not self._queue:
            # Nothing can append between the clear and the wait: producers run on this event loop.
            self._queue_event.clear()
            await self._queue_event.wait()
        return self._queue_get_nowait()

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

    @property
    def context_window_used(self) -> float | None:
        """Fraction of the model's context window occupied, as of the latest report.

        When the provider reports it (OpenAI GPT-Live), this is the latest value it reported. Otherwise
        it is computed as for a standard run's
        [`RunContext.context_window_used`][pydantic_ai.tools.RunContext.context_window_used]: the latest
        response's [`total_tokens`][pydantic_ai.usage.RequestUsage.total_tokens] over the model's
        [`context_window`][pydantic_ai.realtime.RealtimeModelProfile.context_window].

        The value can go down as the session continues, when the provider compacts or truncates the
        conversation server-side. Returns `None` when the ratio cannot be calculated: when the context
        window or usage is unknown, before the first response, or when the model's response usage
        doesn't measure what the context holds (see
        [`response_usage_covers_context`][pydantic_ai.realtime.RealtimeModelProfile.response_usage_covers_context]).
        """
        if self._reported_context_window_used is not None:
            return self._reported_context_window_used
        if not self._profile.get('response_usage_covers_context', True):
            return None
        return context_window_fraction(self.all_messages(), self._profile.get('context_window'))

    def stream_audio(self) -> AsyncIterator[bytes]:
        """Stream model audio chunks ready for playback.

        The iterator contains only live model audio, in playback order. It never repeats retained
        audio from finalized speech parts. On a WebRTC sideband the browser owns the audio path, so
        this raises [`UserError`][pydantic_ai.exceptions.UserError]; consume the browser's remote media
        track instead.

        The subscription starts when this method is called, so audio the model produces between
        the call and the consumer's first iteration is buffered rather than missed. A view handed
        to `asyncio.create_task` therefore misses nothing while waiting for its first turn on the
        event loop. Models generate speech several times faster than it plays, so each iterator
        buffers up to five minutes of audio: a consumer that plays each chunk before pulling the next
        receives a long reply in full. Past that bound (a consumer that stopped iterating, or never
        started), the oldest chunk is dropped so audio playback cannot stall tool execution, turn
        tracking, or the main event stream; an unconsumed view keeps that bounded buffer until it
        is collected. A device-paced consumer also feeds
        [`played_audio_bytes`][pydantic_ai.realtime.RealtimeSession.played_audio_bytes], and on
        barge-in [`interrupt(played_bytes=...)`][pydantic_ai.realtime.RealtimeSession.interrupt]
        discards buffered chunks the user will never hear, so a playback loop can iterate one
        stream for the whole session. Closing the session discards buffered chunks and ends the
        iterator cleanly. If a tool fails while the event stream has no consumer, this view ends and
        the error is raised when the session closes.
        """
        self._require_media_ownership('stream_audio')
        self._ensure_streamable()
        # The queue itself is unbounded: `_AudioTap.put` bounds it by bytes, and the completion
        # sentinel always fits behind a full buffer without discarding audio or blocking teardown.
        queue: asyncio.Queue[bytes | object] = asyncio.Queue()
        tap = _AudioTap(
            queue=queue,
            subscribed_at_bytes=self._emitted_audio_bytes,
            max_buffered_bytes=_AUDIO_TAP_SECONDS * self.audio_output_sample_rate * 2,
        )
        self._audio_taps.add(tap)
        if self._pump_finished:
            queue.put_nowait(self._tap_finished)
        else:
            self._start_pump()

        async def iterate() -> AsyncGenerator[bytes, None]:
            try:
                while (item := await queue.get()) is not self._tap_finished:
                    assert isinstance(item, bytes)
                    tap.buffered_bytes -= len(item)
                    # Taking this chunk steps over every gap that opened before it, so those now sit
                    # behind the playback position and belong in the mapping.
                    tap.dropped_bytes += tap.pending_dropped_bytes
                    tap.pending_dropped_bytes = 0
                    yield item
                    # Resuming here means the consumer came back for the next chunk. Under the
                    # documented playback pattern — write each chunk to the device before pulling the
                    # next — that is the moment the previous chunk finished playing, which is what
                    # makes `played_audio_bytes` an accurate playback position with no caller counting.
                    tap.played_bytes += len(item)
                    tap.progress.set()
            finally:
                tap.finish()
                self._audio_taps.discard(tap)

        return _TapView(iterate(), self._audio_taps, tap, tap.finish)

    async def wait_for_playback(self) -> None:
        """Wait until the session's single audio view has accounted for all audio emitted so far.

        Call this after a reply finishes generating, before closing the session or opening the
        microphone, so buffered audio is not cut off. Playback advances with the same one-chunk lag
        as [`played_audio_bytes`][pydantic_ai.realtime.RealtimeSession.played_audio_bytes]: a chunk
        counts once the consumer requests the next one. Audio the view never plays counts as accounted for
        rather than played: discarded by a barge-in or by the view's buffer overflowing, or emitted before
        the view subscribed. The wait also ends if the view is closed or abandoned, or the session closes.
        Requires exactly one active
        [`stream_audio()`][pydantic_ai.realtime.RealtimeSession.stream_audio] iterator.
        """
        tap = self._single_audio_tap('`wait_for_playback()`', 'wait for playback from')
        while not self._closed and not tap.ended:
            # Clear before checking so that progress made between the check and the wait still wakes us.
            tap.progress.clear()
            playhead = tap.subscribed_at_bytes + tap.played_bytes + tap.dropped_bytes
            # Once the pump has finished, the only item left in the queue is the completion sentinel.
            buffer_empty = tap.queue.empty() or (self._pump_finished and tap.queue.qsize() == 1)
            if buffer_empty and playhead >= self._emitted_audio_bytes:
                return
            await tap.progress.wait()

    async def wait_for_reply(self) -> None:
        """Wait until the model has finished the reply it owes, if any.

        Returns once the exchange in progress reaches the boundary
        [`RealtimeTurnCompleteEvent`][pydantic_ai.realtime.RealtimeTurnCompleteEvent] marks — after a
        tool-calling turn, that is the answer that follows the tool results, not the response that
        called them. A reply is the response that answers what was sent, not one that was already under
        way when it reached the provider (see [The turn boundary](../realtime/events.md#the-turn-boundary)).
        Returns immediately when the model owes nothing, so a reply that finished between
        the [`send()`][pydantic_ai.realtime.RealtimeSession.send] and this call is not waited for twice
        over; it also returns if the session closes or ends (a response passed a usage limit), if the
        exchange fails (a tool raised, or its result would pass a usage limit) so the reply can no longer
        come, or if the provider refuses the request for the reply (reported as a
        [`RealtimeSessionErrorEvent`][pydantic_ai.realtime.RealtimeSessionErrorEvent]).

        This is the wait `async for event in session` would otherwise be written out to perform, and
        unlike that loop it can run while something else is iterating the session, so a caller that
        only wants the turn boundary does not have to take over the event stream to find it:

        ```python
        from pydantic_ai import Agent

        agent = Agent(instructions='You are a helpful voice assistant.')


        async def main():
            async with agent.realtime('openai:gpt-realtime').session() as session:
                await session.send('Say hello.')
                await session.wait_for_reply()
        ```

        The reply can show up in [`all_messages()`][pydantic_ai.realtime.RealtimeSession.all_messages] later than
        it ends (by up to 30 seconds), where history waits for the transcript of a spoken turn before it (see
        [History](../realtime/history.md#transcription-and-history-edge-cases)).

        Waiting for the model to stop *generating* is not the same as waiting for the speaker to stop
        *playing*: pair it with
        [`wait_for_playback()`][pydantic_ai.realtime.RealtimeSession.wait_for_playback] before opening
        the microphone, so the reply is not cut off and the model does not hear itself.
        """
        self._ensure_streamable()
        self._start_pump()
        if self._send_lock.locked():
            # A send under way may be asking for a reply (an image's `CreateResponse` goes out after the image):
            # that is owed at this moment too, once it's out, or nothing is, if it fails.
            async with self._send_lock:
                pass
        # What the model owes at this moment, followed to the end: a reply asked for later is not waited for.
        owed = self._core.wait_tokens()
        while True:
            # Cleared before the check, so a boundary reached between the check and the wait still
            # wakes us rather than leaving this parked until the turn after it.
            self._exchange_progress.clear()
            if self._closed or self._pump_finished or not self._core.still_owed(owed):
                return
            await self._exchange_progress.wait()

    def _single_audio_tap(self, method: str, purpose: str) -> _AudioTap:
        if len(self._audio_taps) != 1:
            raise UserError(
                f'{method} needs exactly one active `stream_audio()` iterator to {purpose}, '
                f'not {len(self._audio_taps)}; keep your own accounting instead.'
            )
        (tap,) = self._audio_taps
        return tap

    @property
    def played_audio_bytes(self) -> int:
        """The playback position of the session's single [`stream_audio()`][pydantic_ai.realtime.RealtimeSession.stream_audio] consumer, in raw PCM bytes.

        A chunk counts as played once the consumer comes back for the next one, so under the
        documented playback pattern — write each chunk to the device and wait until it is consumed
        before pulling the next — this is the position at the last chunk boundary the device
        reached, the same value a hand-rolled counter incremented after each write would report.
        The chunk being written is not counted until it is done, so barge-in attributes at most one
        chunk less than was really heard, never more. A consumer that instead buffers ahead reads
        too far; count actual device consumption yourself in that case. Made to be passed to
        [`interrupt(played_bytes=...)`][pydantic_ai.realtime.RealtimeSession.interrupt] on
        barge-in. Requires exactly one active `stream_audio()` iterator, like that call.
        """
        return self._single_audio_tap('`played_audio_bytes`', 'report the playback position of').played_bytes

    @overload
    def stream_transcripts(self, *, delta: Literal[False] = False) -> AsyncIterator[SpeechPart]: ...

    @overload
    def stream_transcripts(self, *, delta: Literal[True]) -> AsyncIterator[TranscriptUpdate]: ...

    def stream_transcripts(self, *, delta: bool = False) -> AsyncIterator[SpeechPart | TranscriptUpdate]:
        """Stream speech transcripts for both the user and assistant.

        By default, yields finalized [`SpeechPart`][pydantic_ai.messages.SpeechPart] instances — one
        per completed turn, carrying its `speaker` and full `transcript`. A model output item whose
        transcript grows after its part ended, possibly while a later item is still streaming, is
        yielded again each time it grows, with the transcript so far; `delta=True` carries the
        `index` to replace it by.

        Pass `delta=True` for live captions, which yields
        [`TranscriptUpdate`][pydantic_ai.realtime.TranscriptUpdate]s carrying the new text, the turn's
        full transcript so far, the speaker, and an `index` identifying the turn. Both speakers
        stream at once, so that `index` is what lets a UI keep two turns apart instead of running
        them together. Empty updates and finalized parts without a transcript are omitted.

        Final transcripts and deltas are separate subscriptions, so one never crowds out the other.
        The subscription starts when this method is called, so nothing said between the call and
        the consumer's first iteration is missed. Each iterator buffers up to 512 items; if its
        consumer falls behind (or never starts), the oldest is dropped, because captioning must not
        be able to stall tool execution, turn tracking, or the main event stream. Closing the
        session discards buffered items and ends the iterator cleanly; an unconsumed view otherwise
        keeps its bounded buffer until it is collected.
        If a tool fails while the event stream has no consumer, this view ends and the error is raised
        when the session closes.
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

        async def iterate() -> AsyncGenerator[SpeechPart | TranscriptUpdate, None]:
            try:
                while (item := await queue.get()) is not self._tap_finished:
                    yield item
            finally:
                taps.discard(queue)

        return _TapView(iterate(), taps, queue)

    def all_messages(self) -> list[ModelMessage]:
        """A snapshot of the seeded history plus messages recorded during this session.

        Returns a copy, so the result doesn't change as the session continues. Feed it into
        [`Agent.run(message_history=...)`][pydantic_ai.agent.AbstractAgent.run] to hand the
        conversation off to a standard agent run. Images streamed with `send()` are recorded according
        to `retain_images_every_n`, bounded by `retain_images_max`, and audio retained by
        `audio_retention` is bounded by `retain_audio_max_seconds`.
        """
        return self._core.all_messages()

    def new_messages(self) -> list[ModelMessage]:
        """A snapshot of the messages created during this session (excluding the seeded history)."""
        return self._core.new_messages()

    @property
    def conversation(self) -> Conversation:
        """This session's [`Conversation`][pydantic_ai.conversation.Conversation], ready to hand to a text run.

        The same bundle [`AgentRunResult.conversation`][pydantic_ai.agent.AgentRunResult.conversation]
        produces, so a spoken conversation can be continued by
        [`Agent.run`][pydantic_ai.agent.AbstractAgent.run] and handed back again, carrying its usage
        rather than restarting the budget each time it changes modality.

        A session opened without a `conversation_id` resolves one here the way
        [`Agent.realtime`][pydantic_ai.agent.AbstractAgent.realtime] does up front — continuing the
        conversation its seeded history belongs to, or starting a new one — and keeps it, so every
        bundle taken from the session and every message it records shares one identity to store the
        conversation under.
        """
        if self._conversation_id is None:
            self._conversation_id = _agent_graph.resolve_conversation_id(None, self._seeded)
            # Messages this session recorded before the id existed were stamped with none: the core stamps them
            # now, as they would have been had it been resolved up front, so the bundle agrees with its own
            # messages. The seeded history is the caller's, and is left as it was given.
            self._core.set_conversation_id(self._conversation_id)
            # The session span was opened before this id existed too, and has to agree for the
            # session to correlate with the text run that continues it.
            self._session_instrumentation.set_conversation_id(self._conversation_id)
        return Conversation(
            messages=self.all_messages(),
            usage=copy(self.usage),
            conversation_id=self._conversation_id,
        )

    def _new_request(self, parts: list[ModelRequestPart]) -> ModelRequest:
        """Create a request carrying the framework-managed session metadata."""
        request = ModelRequest(parts=parts)
        fill_run_metadata(request, run_id=self._run_id, conversation_id=self._conversation_id)
        return request

    async def send(
        self, content: RealtimeSessionInput | Sequence[RealtimeSessionInput], *, respond: bool | None = None
    ) -> None:
        """Feed content into the session.

        A `str` is a complete text turn that the model replies to. Do not follow `send('...')` with
        `create_response()`: that asks for two responses. An image is added as context without
        soliciting a response, while audio is streamed into the input buffer.

        Set `respond=False` to add text or an image as context without soliciting a response. Set
        `respond=True` to solicit a response to text or an image; this requires manual turn control
        for images. `respond=True` cannot be used with audio. For a sequence, an explicit `respond` value
        adds every item except the last as context and applies that value to the last item. With the
        default `respond=None`, each item keeps its usual behavior.

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
        self._ensure_can_send()
        if isinstance(content, str):
            await self._send_text(content, respond=respond is not False)
        elif isinstance(content, BinaryContent):
            if content.is_image:
                await self._send_image(content, respond=respond is True)
            else:
                if respond is True and content.media_type in (_WAV_MEDIA_TYPE, 'audio/pcm'):
                    # Audio never solicits a reply on its own (VAD or `commit_audio()` ends the turn), so
                    # `respond=False` is a no-op for it and only `respond=True` is a mistake.
                    raise UserError(
                        '`respond=True` cannot be used with audio sent via `session.send()`: a spoken turn is '
                        'ended by voice activity detection, or by `commit_audio()` and `create_response()`.'
                    )
                await self._send_audio_content(content)
        elif isinstance(content, (bytes, bytearray)):
            # `bytes` is a `Sequence[int]`, so guard it before the sequence branch below — otherwise it
            # iterates into a confusing per-byte error. Raw input audio goes through `send_audio()`.
            raise UserError('Raw audio bytes cannot be sent via `session.send()`; use `session.send_audio(...)`.')
        elif isinstance(content, Sequence):
            if respond is None:
                for item in content:
                    await self.send(item)
            else:
                for index, item in enumerate(content):
                    await self.send(item, respond=respond if index == len(content) - 1 else False)
        else:
            assert_never(content)

    async def _send_text(self, content: str, *, respond: bool) -> ModelRequest:
        request = self._new_request([UserPromptPart(content=content)])
        await self._send_frame(content if respond else TextContext(content), request=request)
        return request

    def enqueue(
        self,
        *content: EnqueueContent,
        priority: PendingMessagePriority = 'asap',
    ) -> str | None:
        """Enqueue content to be delivered to the model when the session is ready for it.

        Use this from code driving the session. For tools, use
        [`RunContext.enqueue`][pydantic_ai.tools.RunContext.enqueue]. Unlike
        [`send()`][pydantic_ai.realtime.RealtimeSession.send], which is delivered immediately even
        while the model is speaking, enqueued content waits for the appropriate turn boundary and
        triggers a model response when delivered.

        Args:
            *content: One or more [`EnqueueContent`][pydantic_ai.run.EnqueueContent] items. Realtime
                delivery supports text parts and
                [`SystemPromptPart`][pydantic_ai.messages.SystemPromptPart]s, which are joined into
                one user turn. System parts use the `<system>…</system>` convention to indicate that
                the text is not the person speaking; that marks where it came from, not that it should
                be handled silently. The model still gets a turn and decides whether to reply, call a
                tool, or move on. To add context without prompting a turn at all, use
                [`send(..., respond=False)`][pydantic_ai.realtime.RealtimeSession.send]. The delivered
                turn is recorded in history as a
                [`UserPromptPart`][pydantic_ai.messages.UserPromptPart]. Calling with no positional
                arguments is a no-op.
            priority: When to deliver:
                `'asap'` (default) — as soon as no reply is being generated; a reply already in
                    progress is allowed to finish first.
                `'when_idle'` — only once the model is idle: no response is in flight and all
                    `'asap'` items have been delivered.

        Returns:
            The `enqueue_id` of the queued message, or `None` when there was nothing to enqueue.

        Raises:
            UserError: If the session has not been entered or is closed, or the content cannot be
                delivered over the live input channel.
        """
        if not self._entered:
            raise UserError('Enter the realtime session with `async with` before enqueuing content.')
        self._ensure_can_send()
        pending = PendingMessage.from_content(*content, priority=priority)
        if pending is None:
            return None
        self._pending_messages.append(pending)
        return pending.enqueue_id

    async def _send_audio_content(self, content: BinaryContent) -> None:
        if content.media_type == _WAV_MEDIA_TYPE:
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

    async def _send_image(self, content: BinaryContent, *, respond: bool) -> None:
        """Forward an image and retain it according to the session's sampling and cap policies."""
        self._require_capability('supports_image_input', method='send', feature='image input')
        if self._profile.get('image_input_requires_response', False):
            # The model takes an image only to respond to it (GPT-Live hands it to the backend that
            # runs on it), so asking for a response is the one way to send one, not manual turn-taking.
            if not respond:
                raise UserError(
                    'This realtime model only takes an image to respond to it, so `session.send()` needs '
                    '`respond=True` for an image; it cannot add one as context alone.'
                )
        elif respond:
            self._require_capability('supports_manual_turn_control', method='send', feature='manual turn-taking')
        request: ModelRequest | None = None
        if self._retain_images_max != 0 and self._sent_image_count % self._retain_images_every_n == 0:
            request = self._new_request([UserPromptPart(content=[content])])
        # Callers guard on `is_image`, so the narrowed type only re-tags a plain `BinaryContent`.
        image = BinaryContent.narrow_type(content)
        assert isinstance(image, BinaryImage)
        if respond:
            await self._send_frame(image, CreateResponse(), request=request)
        else:
            await self._send_frame(image, request=request)
        self._sent_image_count += 1
        # The provider can refuse the image while the send is still in progress, and the refusal then
        # took the request out of history before it got here: retaining it anyway would let a request
        # that isn't in history count against the cap and evict one that is.
        if request is not None and any(sent is request for sent in self._input_requests.values()):
            self._retained_image_requests.append(request)
            # `retain_images_every_n` only slows history growth; the cap bounds it, so a long-running
            # frame stream can't grow the host's memory without limit. Providers hold their own
            # (server-side) image context; this only trims the local record.
            while self._retain_images_max is not None and len(self._retained_image_requests) > self._retain_images_max:
                self._evict_image(self._retained_image_requests.pop(0))

    def _evict_image(self, request: ModelRequest) -> None:
        """Drop the oldest retained image from history, to keep within `retain_images_max`."""
        evicted = [index for index, sent in self._input_requests.items() if sent is request]
        self._apply_core(InputWithdrawn(input_ids=tuple(evicted)))

    async def send_audio(self, data: bytes | AsyncIterable[bytes]) -> None:
        """Stream mono PCM16 audio to the model: a single chunk, or an async iterable of chunks.

        Given an async iterable — a microphone stream, a WebSocket receive loop — each chunk is
        forwarded as it arrives and the call returns when the iterable is exhausted, so a whole
        capture loop can be one task: `asyncio.create_task(session.send_audio(microphone))`. If the
        session is closed while consuming that iterable, the call returns cleanly at the next chunk
        the source yields. Cancel the task in application code if the source can stall indefinitely.
        Passing a single chunk to a session that is already closed still raises
        [`UserError`][pydantic_ai.exceptions.UserError].

        Resample the audio to
        [`audio_input_sample_rate`][pydantic_ai.realtime.RealtimeSession.audio_input_sample_rate]
        first (24 kHz on the OpenAI-protocol providers, 16 kHz on Gemini):
        raw bytes carry no rate, so the wrong one is heard as a chipmunk rather than reported.

        With a [`reconnect`][pydantic_ai.realtime.RealtimeModelSettings.reconnect] policy, a chunk sent
        while a dropped connection is being re-dialed is discarded rather than raised, so a microphone
        task survives the reconnect. That includes a one-shot clip passed as a single chunk: resend it
        after the
        [`RealtimeSessionReconnectEvent`][pydantic_ai.realtime.RealtimeSessionReconnectEvent] if it matters.
        """
        self._require_media_ownership('send_audio')
        self._ensure_can_send()
        if isinstance(data, AsyncIterable):
            async for chunk in data:
                if self._closed:
                    return
                await self.send_audio(chunk)
            return
        try:
            await self._send_frame(BinaryAudio(data=data, media_type='audio/pcm'))
        except BaseException as e:
            if (
                isinstance(e, RealtimeError)
                and isinstance(e.__cause__, self._connection.transport_errors)
                and self._connection._can_reconnect  # pyright: ignore[reportPrivateUsage]
                and not (self._pump_finished or self._receive_ending)
            ):
                # The link dropped and the connection's reconnect policy is replacing it, with receiving
                # still live to deliver it. A chunk of live audio is worthless once late, so it is dropped
                # rather than raised: the capture loop outlives the reconnect instead of dying on it. Once
                # the reconnect fails, the next chunk raises again.
                return
            raise

    async def commit_audio(self) -> None:
        """Commit buffered input audio as a user turn (manual turn-taking / push-to-talk)."""
        self._ensure_can_send()
        self._require_media_ownership('commit_audio')
        self._require_capability('supports_manual_turn_control', method='commit_audio', feature='manual turn-taking')
        await self._send_frame(CommitAudio())

    async def clear_audio(self) -> None:
        """Discard buffered, uncommitted input audio."""
        self._ensure_can_send()
        self._require_media_ownership('clear_audio')
        self._require_capability('supports_manual_turn_control', method='clear_audio', feature='manual turn-taking')
        await self._send_frame(ClearAudio())
        self._apply_core(AudioCleared())

    async def create_response(self) -> None:
        """Ask the model to respond now (manual turn-taking, after `commit_audio`).

        If a response is already in flight, the request is held until it completes and is dropped if
        the user barges in. Returning does not mean the model has started speaking.
        """
        self._ensure_can_send()
        self._require_capability('supports_manual_turn_control', method='create_response', feature='manual turn-taking')
        await self._send_frame(CreateResponse())

    @overload
    async def interrupt(self, *, played_ms: int | None = None) -> None: ...

    @overload
    async def interrupt(self, *, played_bytes: int) -> bool: ...

    async def interrupt(self, *, played_ms: int | None = None, played_bytes: int | None = None) -> bool | None:
        """Barge-in: cancel the model's in-progress response, optionally truncating its audio first.

        This is server-side only — it stops generation and (when a playback position is given) syncs
        the provider's transcript to what was actually heard. Generation outruns playback, so the reply
        being heard has often finished generating: with `played_bytes`, the session still truncates it,
        and history records it as interrupted where playback stopped (see
        [Barge-in](../realtime/turns.md#barge-in)).

        With `played_ms`, the caller owns all playback accounting: flushing locally buffered
        playback is the caller's responsibility, and deciding whether to interrupt at all is too.

        With `played_bytes`, the session owns the rest: it attributes the position to the current
        turn (adjusting for audio the stream dropped or that predates the subscription), discards
        buffered audio the user will never hear — chunks already queued, and the cancelled
        response's stragglers as they arrive — then truncates and cancels, returning `True`. The
        cancel is left out while the provider has a speech segment open on a model whose own turn
        detection stops the response being spoken over, where a client cancel racing it can be
        applied to the *next* response; an interruption raised outside a speech segment always
        cancels. On a
        model without output truncation the unheard audio is still flushed and the response
        cancelled; only the provider-side transcript sync is unavailable. When everything emitted
        was already played there is nothing to cut off: a reply that has not reached its first audio
        chunk is cancelled untruncated, since it is about to speak over the user, and otherwise
        nothing is sent and `False` is returned — reporting an interruption anyway would make the
        provider discard part of a completed reply. This form needs exactly one
        [`stream_audio()`][pydantic_ai.realtime.RealtimeSession.stream_audio] iterator, kept for the
        session's lifetime; with several (or none) the position is ambiguous and a
        [`UserError`][pydantic_ai.exceptions.UserError] is raised — keep your own accounting and
        pass `played_ms` instead.

        Args:
            played_ms: Playback position in milliseconds from the start of the model's current audio
                output. When given, the model's output audio and its transcript are truncated to this
                point before the response is cancelled.
            played_bytes: Playback position as the total raw PCM bytes actually played from the
                session's single `stream_audio()` iterator. Mutually exclusive with `played_ms`.
        """
        self._ensure_can_send()
        self._require_capability('supports_interruption', method='interrupt', feature='interruption')
        if played_bytes is not None:
            if played_ms is not None:
                raise UserError('`interrupt()` accepts either `played_ms` or `played_bytes`, not both.')
            return await self._interrupt_at_playback(
                played_bytes, cancel=not self._server_cancelled_the_response_on_speech
            )
        if played_ms is not None and not self._profile.get('supports_output_truncation', False):
            raise UserError(
                'This realtime model does not support output truncation, so `interrupt(played_ms=...)` '
                'is unavailable. Call `interrupt()` without `played_ms` to cancel without truncating.'
            )
        # Truncate before cancelling: cancellation triggers `response.done`, which clears the tracked
        # output item, so a truncate sent afterwards could no-op. Both frames go out under one hold of
        # the send lock, so a tool result completing in between can't start a new response for the
        # cancel to hit instead. The client cancel is skipped while the provider's own turn detection
        # is already cancelling the response spoken over — the same rule as the `played_bytes` path
        # and `handle_barge_in=True` — so it can't land on the *next* response instead.
        frames: list[TruncateOutput | CancelResponse] = []
        if played_ms is not None:
            frames.append(TruncateOutput(audio_end_ms=played_ms))
        if not self._server_cancelled_the_response_on_speech:
            frames.append(CancelResponse())
        # (Only a response still being generated records the position: a finished one keeps what history
        # recorded, and the position must not land on the next one instead.)
        if frames:
            try:
                await self._send_frame(*frames)
            except RealtimeError:
                # The link failed mid-send, so the provider may have applied the truncation already: either way the
                # listener heard only this much, so history records the cut too.
                self._apply_core(Interrupted(played_ms=played_ms))
                raise
        self._apply_core(Interrupted(played_ms=played_ms))
        # Mark the barge-in in the trace. When the caller supplied `played_ms` (the ms of output audio
        # actually played before truncating), record it so a reader can see how far the response got before
        # the user cut in; it's dropped when absent (a cancel without truncation).
        self._session_instrumentation.record_lifecycle('interrupt', played_ms=played_ms)

    async def _interrupt_at_playback(self, played_bytes: int, *, cancel: bool = True) -> bool:
        """Map a device playback position onto the current turn, flush unheard audio, and interrupt.

        `cancel=False` skips the client-side `CancelResponse` for a barge-in the provider's own VAD
        is already cancelling: see `_server_cancelled_the_response_on_speech`.
        """
        if played_bytes < 0:
            raise UserError(f'`played_bytes` must be a non-negative playback position, not {played_bytes}.')
        if len(self._audio_taps) != 1:
            raise UserError(
                '`interrupt(played_bytes=...)` needs exactly one active `stream_audio()` iterator to '
                f'attribute the playback position to, not {len(self._audio_taps)}; keep your own '
                'accounting and pass `played_ms` instead.'
            )
        (tap,) = self._audio_taps
        # The device consumed `played_bytes` of what this tap delivered; adding the gaps it has
        # already stepped over (overflow drops, earlier flushes) and what predates the subscription
        # places that position in the session's emitted-audio coordinates. (Flushed audio is erased
        # from the stream, so skipping over it is what keeps this mapping honest across barge-ins.)
        # Gaps still ahead of the consumer are deliberately excluded: see `pending_dropped_bytes`.
        playhead = tap.subscribed_at_bytes + played_bytes + tap.dropped_bytes
        if playhead >= self._emitted_audio_bytes:
            # Everything emitted has been heard, so there is nothing to cut off. A reply that has
            # opened a part but not reached its first audio chunk is still about to speak over the
            # user, though — barging in during the model's think time is ordinary — so cancel it
            # untruncated. Truncating is what would discard part of a reply the user did hear, and
            # once the part has audio the response may simply be over with its `ResponseDone` still
            # in flight, where a cancel can be applied to the *next* response instead.
            if cancel and (speaking := self._core.speaking_part()) is not None and self._audio_part_index != speaking:
                # Audio generated before the provider processed the cancel still arrives for this
                # part, so erase it the same way the unheard-audio path does: the reply was stopped
                # before it spoke, and none of it should reach the playback stream now.
                self._interrupted_audio_part_index = speaking
                await self._send_frame(CancelResponse())
                self._session_instrumentation.record_lifecycle('interrupt', played_ms=None)
                return True
            return False
        # Unheard audio exists: flush what's buffered, and erase the cancelled part's stragglers as
        # they arrive, so the stream carries nothing the barge-in already cut off.
        self._flush_tap(tap)
        self._interrupted_audio_part_index = self._audio_part_index
        if not self._profile.get('supports_output_truncation', False):
            # Best effort on a model without output truncation (xAI): the unheard audio is still
            # flushed and the response cancelled; only the provider-side transcript sync is
            # unavailable.
            if cancel:
                await self._send_frame(CancelResponse())
            self._session_instrumentation.record_lifecycle('interrupt', played_ms=None)
            return True
        # A playhead still inside a previous turn's audio means none of the current turn was heard.
        played_ms = self._bytes_to_ms(playhead - self._turn_audio_start_bytes)
        # Truncate before cancelling, under one hold of the send lock, for the same reasons as above.
        truncations = self._truncations_at(playhead)
        try:
            await self._send_frame(*truncations, *([CancelResponse()] if cancel else []))
        except RealtimeError:
            # As in `interrupt(played_ms=...)`: the truncations may have reached the provider before the link failed.
            self._apply_core(self._playback_interruption(played_ms, truncations))
            raise
        self._apply_core(self._playback_interruption(played_ms, truncations))
        self._session_instrumentation.record_lifecycle('interrupt', played_ms=played_ms)
        return True

    def _playback_interruption(self, played_ms: int, truncations: list[TruncateOutput]) -> Interrupted:
        """What a barge-in at a playback position cut off, for the core to record."""
        # (Only a response still being generated takes the position as a whole: a finished one is cut off by its
        # item's truncation instead, and the position must not land on the next response.)
        on_this_connection = {
            item_id for start, item_id in self._audio_items if start >= self._connection_audio_start_bytes
        }
        cuts = tuple(
            (cut.item_id, cut.audio_end_ms)
            for cut in truncations
            if cut.item_id is not None and cut.item_id in on_this_connection
        )
        return Interrupted(played_ms=played_ms, cuts=cuts)

    def _bytes_to_ms(self, audio_bytes: int) -> int:
        """Milliseconds of output audio in `audio_bytes`; a negative span (a position before it) is 0."""
        return max(0, audio_bytes) * 1000 // (self.audio_output_sample_rate * 2)

    def _truncations_at(self, playhead: int) -> list[TruncateOutput]:
        """Truncate every output item the listener didn't finish hearing at an emitted-audio playhead.

        The item the playhead is inside is cut where playback stopped, and items generated after it,
        never heard, at 0 — even when their replies have finished generating, since generation outruns
        playback. An item the provider gave no id can only be truncated as its current item, the last.
        """
        items = list(self._audio_items)
        ends = [start for start, _ in items[1:]] + [self._emitted_audio_bytes]
        cuts = [
            (item_id, self._bytes_to_ms(playhead - start))
            for (start, item_id), end in zip(items, ends)
            if end > playhead
        ]
        truncations = [
            TruncateOutput(audio_end_ms=played_ms, item_id=item_id) for item_id, played_ms in cuts if item_id
        ]
        if cuts[-1][0] is None:
            truncations.append(TruncateOutput(audio_end_ms=cuts[-1][1]))
        return truncations

    def _flush_tap(self, tap: _AudioTap) -> None:
        """Discard the tap's buffered chunks, counting them as dropped for position mapping."""
        # Nothing queued survives the flush, so the next chunk delivered lands beyond every gap that
        # was still ahead of the consumer as well.
        tap.dropped_bytes += tap.pending_dropped_bytes
        tap.pending_dropped_bytes = 0
        while not tap.queue.empty():
            item = tap.queue.get_nowait()
            if item is self._tap_finished:
                # A barge-in that lands after the pump finished still has buffered audio to discard,
                # but the completion sentinel behind it has to stay so the consumer's iterator ends.
                tap.queue.put_nowait(item)
                break
            assert isinstance(item, bytes)
            tap.dropped_bytes += len(item)
        tap.buffered_bytes = 0
        # Every change to the playback accounting wakes `wait_for_playback()`, so it re-checks against
        # the flushed queue rather than waiting for audio that will never come.
        tap.progress.set()

    async def _auto_barge_in(self, event: RealtimeEvent) -> None:
        """The local half of barge-in, run by the session itself under `handle_barge_in=True`.

        Only acts while exactly one `stream_audio()` iterator is subscribed — the device-paced
        playback setup whose position the session tracks; any other topology keeps the manual
        `interrupt()` flow. Provider differences are absorbed here: a speech start on a truncating
        model gets the full flush-attribute-truncate-cancel treatment; without output truncation
        (xAI) unheard audio is flushed and the response cancelled untruncated; and a
        provider-initiated interruption (Gemini, which reports no speech start) leaves only the
        local flush to do.
        """
        if len(self._audio_taps) != 1:
            return
        (tap,) = self._audio_taps
        if isinstance(event, RealtimeInputSpeechStartEvent):
            if self._profile.get('supports_interruption', False):
                await self._interrupt_at_playback(
                    tap.played_bytes, cancel=not self._server_cancelled_the_response_on_speech
                )
        elif isinstance(event, RealtimeResponseInterruptedEvent):
            self._flush_tap(tap)
            self._interrupted_audio_part_index = self._audio_part_index

    async def _send_frame(self, *contents: RealtimeInput, request: ModelRequest | None = None) -> None:
        """Send inputs to the provider as one group, serialized against every other outbound frame.

        A single input can expand to several protocol frames (a `ToolResult` creates the conversation
        item and then asks for a response), so the lock is what makes each input indivisible on the
        wire, not just ordered. Passing several inputs extends that indivisibility across them, for
        the cases where an interleaved frame would change what they mean.

        `request` is the history entry recording the first input, taken back if the provider later
        reports that input's content as refused (see `_handle_input_rejected`).

        An input that asks for a response (a text turn, a `CreateResponse`, a tool's result) is checked against
        `request_limit` first, under the lock: what the core counts ahead then includes every input sent before it.
        """
        self._ensure_not_closed()
        self._start_pump()
        # Only the closed check is re-taken here, not `_ensure_can_send`: a frame already waiting on
        # the lock when a background failure ends receiving is one already underway, not the "next
        # outbound method" that contract speaks of, and it costs a single frame nobody reads on a
        # session that is ending anyway. Re-taking the full guard under the lock would also raise that
        # parked failure out of `_send_tool_result` and the teardown `CancelResponse`, neither of
        # which is a caller that asked to send.
        async with self._send_lock:
            if replies := sum(self._asks_reply(content) for content in contents):
                self._check_request_limit(extra=replies)
            first_input = self._inputs_sent
            try:
                for position, content in enumerate(contents):
                    # Numbered before the call, and whether or not it raises, matching how
                    # `InputRejected.input_index` counts. Registered before the frame goes out, since
                    # the pump can read the refusal while a later input of this group is still sending.
                    input_index = self._inputs_sent
                    self._inputs_sent += 1
                    if position == 0 and request is not None:
                        self._input_requests[input_index] = request
                    self._apply_core(
                        InputSent(
                            input_id=input_index,
                            request=request if position == 0 else None,
                            solicits=isinstance(content, (str, CreateResponse, ToolResult)),
                            tool_call_id=content.tool_call_id if isinstance(content, ToolResult) else None,
                        )
                    )
                    if settled := await self._connection.send(content):
                        # What the send itself settled (where the input joined the conversation, say), applied before
                        # anything the connection reads next.
                        for event in settled:
                            self._core.apply(event)
                        self._emit_effects(self._core.take_effects())
                        self._core_advanced(self._core)
                    if isinstance(content, BinaryAudio):
                        # Under the lock, so the core takes audio in the order it goes out; only once it has
                        # gone out, as a turn the provider ended while it was on its way (which the pump can
                        # record meanwhile) was ended without it, and a chunk whose send fails is in no turn.
                        self._apply_core(AudioSent(data=bytes(content.data)))
            except BaseException as e:
                # The caller takes back what it sent, so the core does too.
                self._apply_core(InputWithdrawn(input_ids=tuple(range(first_input, self._inputs_sent))))
                if not isinstance(e, self._connection.transport_errors):
                    raise
                # A send that fails because the link is gone is the same failure the receive side
                # reports; surface it as the same typed error rather than leaking a `websockets` or
                # provider-SDK exception from what looks like an ordinary method call.
                raise RealtimeError(
                    model_name=self._error_model_name,
                    message=f'Realtime connection failed while sending: {e}',
                ) from e

    def _asks_reply(self, content: RealtimeInput) -> bool:
        """Whether sending `content` asks for a response the core doesn't count ahead yet (see `requests_ahead`)."""
        if isinstance(content, ToolResult):
            return self._core.result_asks_reply(content.tool_call_id)
        return isinstance(content, (str, CreateResponse))

    @property
    def _error_model_name(self) -> str:
        """The model to attribute a provider failure to: the one asked for, else the one served."""
        return self._model_name or self._connection.model_name or 'unknown'

    def _ensure_not_closed(self) -> None:
        if self._closed:
            raise UserError('This realtime session is closed.')

    def _ensure_can_send(self) -> None:
        self._ensure_not_closed()
        if (self._pump_finished or self._receive_ending) and (error := self._first_undelivered_error()) is not None:
            self._raise_delivered(error)
        # Started here rather than in `__aenter__` so a session driven by `send()` alone still notices
        # a receive-side failure, without the pump publishing to taps that do not exist yet. Entering
        # eagerly discarded provider output produced before the caller's first `stream_audio()` or
        # `stream_transcripts()`, which is the order every docs example uses.
        self._start_pump()

    def _error_was_delivered(self, error: BaseException) -> bool:
        return any(delivered is error for delivered in self._delivered_errors)

    def _raise_delivered(self, error: BaseException) -> Never:
        """Raise an error its callers have already checked is undelivered, recording the delivery."""
        self._delivered_errors.append(error)
        raise error

    def _first_undelivered_error(self) -> BaseException | None:
        if self._pump_error is not None and not self._error_was_delivered(self._pump_error):
            return self._pump_error
        for error in self._parked_errors:
            if not isinstance(error, asyncio.CancelledError) and not self._error_was_delivered(error):
                return error
        return None

    def _park_error(self, error: BaseException) -> None:
        """Park a background failure for iteration or close, ending receive-only views if nobody is iterating."""
        self._parked_errors.append(error)
        self._queue_put(error)
        # A tool that raised, or a usage limit tripped by the request its result would make, stops the model from
        # getting what it needs, so the exchange waiting on it is over: a caller in `wait_for_reply()` is done
        # waiting for it.
        self._apply_core(ExchangeAbandoned())
        if not self._iterator_active and self._pump_task is not None:
            self._receive_ending = True
            self._pump_task.cancel()

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

    def _check_response_boundary_limits(self) -> None:
        """Check the usage limits against a response that has just been recorded.

        A response settled by `close()` is the last thing the session will ever spend, but it is real
        spend: a caller who set a `cost_limit` and hung up mid-reply asked to be told it was passed, and
        the price of that reply is only known here. Skipping the check while closing let an over-budget
        teardown exit cleanly. Once receiving ended, the limit is parked rather than raised too.
        """
        if not (self._closed or self._pump_finished):
            self._check_usage_limits()
            return
        self._park_exceeded_limit(self._check_usage_limits)

    def _park_exceeded_limit(self, check: Callable[[], None]) -> None:
        """Run a usage limit check on a closed session, parking what it raises for `close()` to raise."""
        try:
            check()
        except UsageLimitExceeded as exceeded:
            # Parked rather than raised, so it reaches the caller through the same single delivery point
            # as every other receive-side failure — `close()` takes the first undelivered error, and the
            # caller's own exception still wins. Skipped once a limit has already been delivered: the
            # caller that saw it from iteration must not be told again when teardown re-checks the same
            # budget against the response it just settled. Nor when receiving stopped on a limit that is still
            # on its way to the caller.
            if not any(isinstance(error, UsageLimitExceeded) for error in (*self._delivered_errors, self._pump_error)):
                self._park_error(exceeded)

    def _check_final_request(self, usage: RequestUsage) -> None:
        """Check a request reported only as the session closed.

        Against `per_request_input_tokens_limit` and `request_limit`.
        """
        if self._usage_limits is not None:
            self._usage_limits.check_per_request_input_tokens(usage.input_tokens)
        self._check_reported_requests()

    def _check_reported_requests(self) -> None:
        """Check `request_limit` against the requests recorded, for a provider whose responses aren't requests.

        Its requests are only known once the model made them, so the one past the limit is recorded before it ends
        the session.
        """
        if (
            self._usage_limits is not None
            and (request_limit := self._usage_limits.request_limit) is not None
            and self.usage.requests > request_limit
        ):
            raise UsageLimitExceeded(
                f'Exceeded the request_limit of {request_limit} (`usage.requests`={self.usage.requests})'
            )

    def _complete_tool_call(
        self,
        call_part: ToolCallPart,
        result_part: ToolReturnPart | RetryPromptPart,
        content: str | Sequence[UserContent] | None = None,
    ) -> list[RealtimeEvent]:
        request_parts: list[ModelRequestPart] = [result_part]
        if content:
            request_parts.append(UserPromptPart(content=content))
        request = self._new_request(request_parts)
        self._apply_core(ToolReturned(tool_call_id=call_part.tool_call_id, request=request))
        return [FunctionToolResultEvent(part=result_part, content=content)]

    def _handle_reconnected(self, event: RealtimeSessionReconnectEvent) -> list[RealtimeEvent]:
        """Settle any in-flight state the reconnect did not actually carry, and report restoration honestly.

        A connection that resumes in-flight state (Gemini Live, which settles the cut turn in the
        connection itself) reports
        [`reconnect_restores_in_flight_state`][pydantic_ai.realtime.codec.RealtimeConnection.reconnect_restores_in_flight_state],
        so `state_restored=True` holds as reported and there is nothing more to settle. A connection we
        reconnect by replaying local history (OpenAI, Azure OpenAI, xAI) only restores *finalized* turns: the
        response and tool calls that were in flight when the socket dropped are gone, and the fresh
        server-side conversation knows nothing of them. The core records the partial reply as an interrupted
        response as the frame reporting the reconnect is applied; running tool calls get interrupted returns
        here, so the local history stays coherent.
        Report `state_restored=False` whenever a response or tool call was actually in flight, so an app
        can branch on the flag the same way on every provider; a drop with nothing in flight loses
        nothing and stays `True`.
        """
        if event.state_restored and self._connection.reconnect_restores_in_flight_state:
            return [event]
        # The fresh conversation holds what came before only as replayed text, which a barge-in can't truncate.
        self._connection_audio_start_bytes = self._emitted_audio_bytes
        # The response the provider was cancelling on speech onset went with the old connection, so interrupting
        # whatever comes next is the client's job again.
        self._server_cancelled_the_response_on_speech = False
        lost_in_flight = self._frame_lost_response or bool(self._pending_tool_calls)
        events = self._stop_tool_calls(list(self._pending_tool_calls))
        if lost_in_flight:
            event = replace(event, state_restored=False)
        if not event.state_restored:
            # The reconnect cut whatever was playing: the in-flight response (if any) is settled as
            # interrupted, and even a finalized response's audio won't resume on the fresh
            # connection. End the `speak` span so it measures only what was actually audible, rather
            # than running until session close or being merged into the next utterance
            # (`start_playback_span` no-ops while a span is already open). No-op off a sideband.
            self._session_instrumentation.end_playback_span()
        return [*events, event]

    def _stop_tool_calls(self, tool_call_ids: Sequence[str]) -> list[RealtimeEvent]:
        """Stop tool calls still running that the model no longer waits for, recording each as interrupted.

        The provider cancelled them (a barge-in abandons the turn), lost the conversation they were made in, or the
        session is closing. Each gets an interrupted return, so the call still has a matching return in history (kept
        valid for a handoff), and no result goes back to the model: with one of its calls settled this way, the core
        expects no reply to the calling response's results (see `SessionCore.requests_ahead`).
        """
        events: list[RealtimeEvent] = []
        for tool_call_id in tool_call_ids:
            if (pending := self._pending_tool_calls.pop(tool_call_id, None)) is None:
                continue
            task, call_part = pending
            task.cancel()
            cancelled_part = ToolReturnPart(
                tool_name=call_part.tool_name,
                content=INTERRUPTED_TOOL_RETURN_CONTENT,
                tool_call_id=call_part.tool_call_id,
                tool_kind=call_part.tool_kind,
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
        self._user_speech_started_at = time_ns()
        # Whatever response was in flight, the provider's own turn detection is cancelling it now.
        self._server_cancelled_the_response_on_speech = self._connection.interrupts_response_on_speech
        return [event]

    def _handle_input_rejected(self, event: InputRejected) -> None:
        """Take back what a send assumed, now that the provider refused it: the same rollback a failed send gets.

        (A refused request for a response is settled by the core, from the lifecycle's `ResponseRequestRefused`.)
        """
        if event.refused == 'content' and (request := self._input_requests.pop(event.input_index, None)) is not None:
            self._retained_image_requests = [kept for kept in self._retained_image_requests if kept is not request]

    def _translate_event(self, event: _TranslatableEvent) -> list[RealtimeEvent]:
        """The session events a codec event makes, beyond the parts the core streams for it (see `_emit_effects`).

        Tool calls and usage are handled in `_handle_pump_event` (they interact with the queue and
        tool execution); everything else routes through here. `event` is typed as `_TranslatableEvent`
        (the pump-consumed variants narrowed out) so the final `assert_never` gives static exhaustiveness.
        """
        if isinstance(event, AudioDelta):
            self._session_instrumentation.set_output_type('speech')
            return []
        if isinstance(event, OutputTranscript):
            self._session_instrumentation.set_output_type('text' if event.output_text else 'speech')
            return []
        if isinstance(event, (InputTranscript, ResponseDone, PartStartEvent, PartEndEvent)):
            # (What they make streams as the core applies them.)
            return []
        if isinstance(event, RealtimeInputSpeechEndEvent):
            self._record_user_speech_span()
            return [event]
        if isinstance(event, (RealtimeResponseInterruptedEvent, RealtimeInputTranscriptionErrorEvent)):
            return [event]
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

    async def _announce_deferred_requests(self, requests: DeferredToolRequests) -> None:
        """Surface a deferred call before a handler is asked to resolve it, as a run does.

        So a consumer can relay the request — e.g. to a person a handler is awaiting — while it's pending.
        """
        self._queue_put(DeferredToolRequestsEvent(requests))

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
            self._queue_put(FunctionToolCallEvent(part=call_part, args_valid=args_valid))
            validation_done.set()
            for prerequisite in execution_prerequisites:
                await prerequisite.wait()

        async def on_inline_deferred(requests: DeferredToolRequests, results: DeferredToolResults) -> None:
            self._queue_put(DeferredToolResultsEvent(results))

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
                    ctx.messages[:] = self._core.conversation_so_far()
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
                on_deferred_requests=self._announce_deferred_requests,
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
        if self._closed:
            # The session closed while the tool ran: it hung up with `close()` and then swallowed the
            # `CancelledError`. Its call already has an interrupted return in history and there is no
            # provider left to send to, so the result goes nowhere (`_run_tool` drops it too).
            return result_part, user_content
        if not response_usage_follows:
            await self._drain_pending_messages('asap')
        failed = isinstance(result_part, RetryPromptPart) or result_part.outcome == 'failed'
        await self._send_tool_result(call_part, output, wire_content, failed=failed)
        return result_part, user_content

    async def _send_tool_result(
        self, call_part: ToolCallPart, output: str, wire_content: list[UserContent], *, failed: bool = False
    ) -> None:
        result = ToolResult(
            tool_call_id=call_part.tool_call_id, output=output, content=wire_content or None, failed=failed
        )
        await self._send_frame(result)

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
            self._park_error(error)
        self._queue_put(self._queue_changed)

    async def _drain_pending_messages(self, priority: PendingMessagePriority) -> None:
        """Deliver queued text prompts of `priority` and record them as normal user turns."""
        async with self._pending_messages_lock:
            if self._core.response_under_way():
                if priority == 'asap':
                    self._asap_drain_deferred = True
                return
            if priority == 'when_idle' and self._pending_messages.has_priority('asap'):
                # `when_idle` follows every queued `asap` item: the `asap` drain delivers those and
                # starts a reply, and this priority is retried once that reply completes.
                return
            selected = self._pending_messages.pop_priority(priority)
            for pending in selected:
                request = await self._send_text(_pending_message_text(pending), respond=True)
                self._queue_put(EnqueuedMessagesEvent(enqueue_id=pending.enqueue_id, messages=(request,)))
            if priority == 'asap':
                self._asap_drain_deferred = False

    def _check_tool_call_limit(self) -> None:
        # Let `UsageLimitExceeded` propagate (caught by the pump and re-raised to the consumer), matching
        # how a regular `run`/`iter` surfaces a usage limit rather than wrapping it in another error.
        if self._usage_limits is None:
            return
        projected = dataclasses.replace(self.usage, tool_calls=self.usage.tool_calls + self._tool_calls_in_flight + 1)
        self._usage_limits.check_before_tool_call(projected)

    def _check_usage_limits(self, *, warn_if_cost_unavailable: bool = True) -> None:
        if self._usage_limits is None:
            return
        self._usage_limits.check_tokens(self.usage)
        self._usage_limits.check_cost(self.usage, warn_if_cost_unavailable=warn_if_cost_unavailable)

    def _check_request_limit(self, *, extra: int) -> None:
        """Check `request_limit` against `extra` more requests than those counted, under way and asked for."""
        if self._usage_limits is not None and self._responses_are_requests:
            self._usage_limits.check_before_request(self._usage_ahead(extra))

    def _usage_ahead(self, extra: int) -> RunUsage:
        """`usage` with the requests under way and asked for, and `extra` more, counted, as `check_before_request` takes it.

        That check counts the request it is made for on top of `usage.requests`, so the last of them is left to it.
        """
        return dataclasses.replace(self.usage, requests=self.usage.requests + self._core.requests_ahead() + extra - 1)

    def _check_unasked_response(self, response_id: str) -> None:
        """Check `request_limit` for a response that started unasked, before this session handles any of it.

        With server-side voice activity detection the provider starts a response on its own, so there was no send to
        check ahead of: the check happens as the frame that started it is handled. One that answers something the
        session sent was checked as that went out.
        """
        if (
            self._usage_limits is None
            or not self._responses_are_requests
            or not self._core.response_unasked(response_id)
        ):
            return
        try:
            # The core counts it already: as under way, or as a request if it ended in the frame that started it.
            self._check_request_limit(extra=0)
        except UsageLimitExceeded:
            request_limit = self._usage_limits.request_limit
            if request_limit is not None and self._usage_ahead(0).requests >= request_limit:
                # The request it is would exceed `request_limit`: not one this session records, or counts.
                self._apply_core(ResponseRefused(response_id=response_id))
            raise

    def _handle_usage_event(self, event: SessionUsage) -> None:
        """Check the usage limits against a usage report, which the core counted into `usage` as its frame was applied.

        (Its `context_window_used` was taken then too.)
        """
        self._span_usage.incr(event.usage)  # usage-attribution: what the session span reports
        # Response pricing happens once it is recorded, so cost is provisionally unavailable here.
        self._check_usage_limits(warn_if_cost_unavailable=False)
        if event.response_scoped and not self._responses_are_requests:
            self._check_reported_requests()

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
            # Whatever cancelled it settled the call (see `_stop_tool_calls`).
            raise
        except BaseException as e:
            self._complete_tool_call(call_part, _unsettled_call_return(call_part, e))
            # Surface the failure through the queue so the consumer re-raises it, instead of letting it
            # vanish into `__aexit__`'s cleanup-only drain and hang the session on a completion that
            # never arrives.
            self._park_error(e)
            return
        finally:
            validation_done.set()
            completion.set()
            self._tool_completion_events.discard(completion)
            # Settled (completed, failed, or cancelled): no longer cancellable by `ToolCallCancelled`.
            self._pending_tool_calls.pop(call_part.tool_call_id, None)
        if self._closed:
            # The session closed while this tool ran, so its call already has an interrupted return
            # in history. Reached by a tool that hung up with `close()` and then swallowed the
            # `CancelledError`: recording its result too would leave two returns for one call.
            return
        events = self._complete_tool_call(call_part, result_part, content)
        if ordered_events:
            # Held until nothing is still running, then released in call order — the graph waits for a
            # whole segment (`ALL_COMPLETED`) and replays it by index for the same reason. The release
            # is driven from `_tool_task_done`, which runs even for a tool that raised or was
            # cancelled, so a failed sibling can't strand the batch.
            self._ordered_tool_events[order_index] = events
        else:
            for event in events:
                self._queue_put(event)
        if self._asap_drain_deferred:
            await self._drain_pending_messages('asap')

    def _tool_task_done(self, task: asyncio.Task[None]) -> None:
        self._background_tasks.discard(task)
        # Surface any exception raised outside `_run_tool`'s own try/except — notably the post-`finally`
        # `asap` drain, whose `connection.send` can fail if the socket just dropped — mirroring
        # `_pending_message_task_done`. Otherwise it vanishes with only an "exception was never
        # retrieved" warning at GC, silently losing the enqueued message with no signal to the consumer.
        if not task.cancelled() and (error := task.exception()) is not None:
            self._park_error(error)
        self._release_ordered_tool_events()
        # Wake the queue reader so it can finish once both the pump and the last tool are done.
        self._queue_put(self._queue_changed)

    def _release_ordered_tool_events(self) -> None:
        """Emit `parallel_ordered_events` results in call order, once nothing is still running.

        Called from every tool task's done-callback, so the last one to settle — successful, failed, or
        cancelled — flushes the batch, and a tool that never produced events can't hold it back.
        """
        if self._tool_completion_events or not self._ordered_tool_events:
            return
        for order_index in sorted(self._ordered_tool_events):
            for event in self._ordered_tool_events[order_index]:
                self._queue_put(event)
        self._ordered_tool_events.clear()

    async def _handle_pump_event(
        self,
        event: RealtimeCodecEvent,
    ) -> bool:
        """Process one upstream event onto the queue; return `True` to stop the pump (a limit tripped)."""
        if isinstance(event, ToolCall):
            if self._core.take_call(event.tool_call_id):
                await self._dispatch_tool_call(event)
            return False
        return await self._handle_non_tool_pump_event(event)

    def _reserves_budget(self, tool_name: str) -> bool:
        """Whether a call counts toward `tool_calls_limit` while it runs.

        A declaratively deferred call (`requires_approval=True`, an external tool) is resolved by a handler or
        refused — it never reaches the tool body and so never increments `usage.tool_calls`. The graph leaves those
        out of its own pre-check projection (`_tool_execution` builds `function_indices` from
        `('function', 'unknown')`), so charging the budget for one here could trip the limit over a call that costs
        nothing. An unknown tool name *is* charged, exactly as `'unknown'` is projected there.
        """
        tool_def = self._tool_manager.get_tool_def(tool_name)
        return tool_def is None or not tool_def.defer

    def _check_tool_call(self, event: ToolCall) -> None:
        """Check `tool_calls_limit` for a call the session is to run, before anything of it streams.

        Tripped, the call is refused, so it stays out of history (or, if its response is recorded already, gets an
        interrupted return), and nothing shows it was made.
        """
        if not self._core.call_waiting(event.tool_call_id) or not self._reserves_budget(event.tool_name):
            return
        try:
            self._check_tool_call_limit()
        except UsageLimitExceeded:
            self._apply_core(ToolCallRefused(tool_call_id=event.tool_call_id))
            raise

    async def _dispatch_tool_call(self, event: ToolCall) -> None:
        """Start a tool call's background task (its limit was checked in `_check_tool_call`)."""
        reserves_budget = self._reserves_budget(event.tool_name)
        if reserves_budget:
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
        mode = self._tool_manager.get_parallel_execution_mode()
        is_barrier = mode == 'sequential' or self._tool_manager.is_sequential(call_part)
        # `parallel_ordered_events` keeps calls concurrent but hands their result events to the
        # consumer in the order the model asked for them. Claimed here, at dispatch, because that
        # is the order — the graph replays a segment's events by index for the same reason. Only
        # the events are held back: the provider still gets each result the moment it lands, and
        # the core keeps the results in history in call order, exactly as the graph keeps
        # `output_parts` in emission order whatever the mode.
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
        if isinstance(event, (ConversationCreated, ConversationItemCreated)):
            # (A resumed conversation's replay is the core's to recognize.)
            return False
        if isinstance(event, InputRejected):
            self._handle_input_rejected(event)
            return False
        if isinstance(event, ToolCallCancelled):
            # The model abandoned the calls (usually a barge-in): their siblings' results still go out, but no
            # reply comes for them, since the interruption starts the next one.
            for out in self._stop_tool_calls(event.tool_call_ids):
                self._queue_put(out)
            return False
        if isinstance(event, SessionUsage):
            self._handle_usage_event(event)
            return False
        for out in self._translate_event(event):
            self._publish_taps(out)
            if self._handle_barge_in:
                # Before the event reaches the consumer, so the app observes an already-handled
                # barge-in rather than racing the session to handle it.
                await self._auto_barge_in(out)
            self._queue_put(out)
        return False

    async def _pump(self, context: Context | None) -> None:
        """Drain the connection into the session queue under the explicit session-span context."""
        token = otel_context.attach(context) if context is not None else None
        core_events = self._core_events(self._core)
        try:
            response_done = response_closed = False
            async for effects, event in core_events:
                if isinstance(event, ToolCall):
                    self._check_tool_call(event)
                self._emit_effects(effects)
                response_closed = response_closed or any(isinstance(effect, ResponseClosed) for effect in effects)
                if event is not None:
                    response_done = response_done or isinstance(event, ResponseDone)
                    if await self._handle_pump_event(event):
                        return  # a usage limit tripped: stop reading the upstream
                    continue
                # The frame is handled. What the provider holds of tool rounds that are over can go, and queued
                # messages waiting for a reply to be over go out now.
                if self._tool_rounds_over:
                    await self._prune_tool_calls()
                if response_done:
                    await self._drain_pending_messages('asap')
                    await self._drain_pending_messages('when_idle')
                elif response_closed and self._asap_drain_deferred:
                    await self._drain_pending_messages('asap')
                response_done = response_closed = False
        except Exception as e:
            self._pump_error = e
        finally:
            # Closed here rather than whenever it is collected, so what it settles is settled before receiving ends.
            await core_events.aclose()
            self._pump_finished = True
            self._apply_core(ReceiveEnded())
            self._end_chat_spans_left_open()
            self._exchange_progress.set()
            if not self._closed:
                self._finish_taps()
            self._queue_put(self._queue_changed)
            if token is not None:
                otel_context.detach(token)

    def _apply_core(self, item: CoreInput) -> None:
        """Hand the core an event or a command, stream what it makes, and wake whoever waits on what it owes."""
        self._core.apply(item)
        if isinstance(item, AudioSent):
            # A microphone chunk: it neither settles anything owed nor changes what history holds back.
            return
        self._emit_effects(self._core.take_effects())
        self._core_advanced(self._core)

    def _emit_effects(self, effects: Sequence[Effect]) -> None:
        """Stream the parts the core made, and open and close `chat` spans and exchanges with its responses."""
        for effect in effects:
            if isinstance(effect, ResponseOpened):
                # A new response is not the one the provider cancelled on speech onset, so interrupting it is once
                # again the client's job.
                self._server_cancelled_the_response_on_speech = False
                self._session_instrumentation.ensure_chat_span(effect.response_id)
            elif isinstance(effect, ResponseClosed):
                self._response_closed(effect)
            elif isinstance(effect, ToolRoundsOver):
                self._tool_rounds_over.extend(effect.calls)
            elif isinstance(effect, UsageCounted):
                if self._usage_limits is not None:
                    self._usage_limits.check_per_request_input_tokens(effect.usage.input_tokens)
            elif isinstance(effect, ExchangeOver):
                self._queue_put(RealtimeTurnCompleteEvent())
                # Only the exchange boundary is marked: each response is already a `chat` span, so a marker
                # per response would say nothing the trace doesn't show, while the turn boundary — where the
                # model is actually done — has no span of its own.
                # An interrupted boundary says so in its display text: on providers that auto-respond per
                # VAD segment (OpenAI server VAD), a talking user cancels response after response, and a
                # bare "model turn complete" per cancellation reads as turns that never happened.
                self._session_instrumentation.record_lifecycle(
                    'model turn complete',
                    message='model turn complete (interrupted)' if effect.interrupted else None,
                    interrupted=effect.interrupted or None,
                )
            else:
                self._publish_taps(effect)
                self._queue_put(effect)

    async def _prune_tool_calls(self) -> None:
        """Let the connection delete the provider's copy of tool rounds that are over (`_prune_tool_calls` on it)."""
        calls, self._tool_rounds_over = self._tool_rounds_over, []
        async with self._send_lock:
            try:
                pruned = await self._connection._prune_tool_calls(calls)  # pyright: ignore[reportPrivateUsage]
            except self._connection.transport_errors:
                # The link dropped: nothing is taken as pruned, so a reconnect replays these calls, and they are
                # pruned from the new conversation once the next frame is handled.
                self._tool_rounds_over[:0] = calls
                return
        if pruned:
            self._apply_core(ToolCallsPruned(tool_call_ids=tuple(pruned)))
            self._session_instrumentation.record_lifecycle('prune tool items', tool_call_ids=list(pruned))

    def _end_chat_spans_left_open(self) -> None:
        """End the `chat` spans of responses that ended while their close wasn't streamed (receiving stopped mid-frame)."""
        for response_id in list(self._session_instrumentation.chat_spans):
            message, input_messages = self._core.recorded_response(response_id)
            self._session_instrumentation.end_chat_span(response_id, input_messages, message)

    def _response_closed(self, effect: ResponseClosed) -> None:
        self._frame_lost_response = self._frame_lost_response or effect.status == 'lost'
        self._session_instrumentation.end_chat_span(effect.response_id, effect.input_messages, effect.message)
        if effect.message is None:
            return
        # A run step here is one model response, as on a standard run (see `_execute_tool`).
        self._tool_run_step += 1
        self._check_response_boundary_limits()

    def _annotate_response(self, response: ModelResponse) -> None:
        """Copy the OTel metadata of the tools it calls onto its tool calls, as the instrumented `chat` span records."""
        instrumentation = self._session_instrumentation
        if instrumentation.settings is not None and instrumentation.model_request_parameters is not None:
            annotate_tool_call_otel_metadata(response, instrumentation.model_request_parameters)

    def _core_advanced(self, core: SessionCore) -> None:
        """Wake whoever waits on what the core owes.

        And bound how long a missing transcript, or a stalled response, holds history back.
        """
        self._exchange_progress.set()
        if (turn_id := core.transcript_holding_history()) is not None and turn_id not in self._transcript_watchdogs:
            self._transcript_watchdogs[turn_id] = asyncio.get_running_loop().call_later(
                _TRANSCRIPT_HOLD_SECONDS, self._transcript_overdue, turn_id
            )
        if (response_id := core.response_holding_turns()) is not None and response_id not in self._barge_in_watchdogs:
            self._barge_in_watchdogs[response_id] = asyncio.get_running_loop().call_later(
                _BARGE_IN_TURN_HOLD_SECONDS, self._response_overdue, response_id
            )

    def _response_overdue(self, response_id: str) -> None:
        del self._barge_in_watchdogs[response_id]
        self._apply_core(ResponseOverdue(response_id=response_id))

    def _transcript_overdue(self, turn_id: str) -> None:
        del self._transcript_watchdogs[turn_id]
        self._apply_core(TranscriptOverdue(turn_id=turn_id))

    async def _core_events(self, core: SessionCore) -> AsyncGenerator[tuple[list[Effect], RealtimeCodecEvent | None]]:
        """The connection's codec events, each with what the core made of it, feeding the core its frames on the way.

        Each codec event comes with the effects of applying it and the lifecycle events ahead of it in its frame. A
        frame ends with the effects of the lifecycle events after its last codec event, with no event.
        """
        frames = self._connection.frames()
        try:
            async for frame in frames:  # pragma: no branch
                # Applied a whole frame at a time, before this session handles any of it: a consumer reacting
                # to one of its events (and reading the history, say) must find the core as far along as this one.
                steps: list[tuple[list[Effect], RealtimeCodecEvent | None]] = []
                effects: list[Effect] = []
                for event in frame:
                    if not isinstance(event, StaleEvent):
                        core.apply(event)
                        effects.extend(core.take_effects())
                    report = event.event if isinstance(event, StaleEvent) else event
                    if isinstance(report, SessionUsage) and report.context_window_used is not None:
                        # Taken as the frame is applied, with the usage the core counts. (A stale report too: it is
                        # still the provider's latest reading of its context, though its usage isn't counted.)
                        self._reported_context_window_used = report.context_window_used
                    if not isinstance(report, LifecycleEvent):
                        steps.append((effects, report))
                        effects = []
                core.end_frame()
                steps.append((effects + core.take_effects(), None))
                self._frame_lost_response = False
                self._core_advanced(core)
                handled = 0
                try:
                    started = (event.response_id for event in reversed(frame) if isinstance(event, ResponseStarted))
                    if (response_id := next(started, None)) is not None:
                        self._check_unasked_response(response_id)
                    for step in steps:
                        handled += 1
                        yield step
                finally:
                    # Receiving stopped (a limit tripped, or the session closed) before this session got to the rest of
                    # the frame: a tool call in it never runs. It is refused, so it is left out of history, or given
                    # an interrupted return if its response is recorded already (Gemini ends a tool-call response in
                    # the message that makes the calls).
                    for _, event in steps[handled:]:
                        if isinstance(event, ToolCall):
                            self._apply_core(ToolCallRefused(tool_call_id=event.tool_call_id))
        finally:
            # Closed with this, rather than whenever it is collected, so a connection lets go of what it holds now.
            if isinstance(frames, AsyncGenerator):
                await frames.aclose()

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
                if event.index == self._interrupted_audio_part_index:
                    # A straggler from a barge-in-flushed part, generated before the provider
                    # processed the cancel: erased from the taps and their accounting alike, so it
                    # neither plays as stale audio nor skews the next playback-position mapping.
                    pass
                else:
                    if event.index != self._audio_part_index:
                        # First audio of a new part: when a playback position is attributed,
                        # everything before this offset belongs to earlier turns.
                        self._audio_part_index = event.index
                        self._turn_audio_start_bytes = self._emitted_audio_bytes
                        self._audio_items.append((self._emitted_audio_bytes, self._core.part_item(event.index)))
                    self._emitted_audio_bytes += len(delta.audio_chunk)
                    for tap in self._audio_taps:
                        self._audio_tap_drops += tap.put(delta.audio_chunk)
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
                        if _put_tap(queue, update) is not None:
                            self._transcript_tap_drops += 1
        elif isinstance(event, PartEndEvent) and isinstance(part := event.part, SpeechPart):
            self._transcript_so_far.pop(event.index, None)
            if part.transcript:
                for queue in self._transcript_taps:
                    if _put_tap(queue, part) is not None:
                        self._transcript_tap_drops += 1

    def _finish_taps(self, *, discard_pending: bool = False) -> None:
        if discard_pending:
            for tap in self._audio_taps:
                tap.finish()
        audio_queues = (tap.queue for tap in self._audio_taps)
        for queue in (*audio_queues, *self._transcript_taps, *self._transcript_delta_taps):
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
        self._start_pump()

        async def queue_events() -> AsyncIterator[RealtimeEvent]:
            while True:
                item = await self._queue_get()
                if item is self._queue_changed:
                    # A pump error takes priority over stuck background tools. Their cancellation and
                    # drain belong to `__aexit__`, which runs as this exception leaves the owner block.
                    if self._pump_error is not None:
                        self._stream_exhausted = True
                        if not self._error_was_delivered(self._pump_error):
                            self._raise_delivered(self._pump_error)
                    if self._pump_finished and not self._background_tasks and not self._queue:
                        self._stream_exhausted = True
                        return
                    continue
                if isinstance(item, BaseException):
                    if self._error_was_delivered(item):
                        continue
                    self._raise_delivered(item)
                yield cast('RealtimeEvent', item)

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
            # Nobody reads the queue past this point, so the bound applies again before the wrapper's
            # cleanup runs, however long that takes.
            self._iterator_active = False
            self._trim_queue()
            await aclose_all((stream_iterator, stream, source))
