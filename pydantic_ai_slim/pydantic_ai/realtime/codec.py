"""Low-level *codec* vocabulary for realtime providers.

Most users only need the session-level API in [`pydantic_ai.realtime`][pydantic_ai.realtime]
([`AgentRealtime.session`][pydantic_ai.agent.AgentRealtime.session], the events a session yields,
and the content passed to [`RealtimeSession.send`][pydantic_ai.realtime.RealtimeSession.send]). This
submodule holds the lower-level vocabulary used when *implementing* a realtime provider or consuming a
[`RealtimeConnection`][pydantic_ai.realtime.codec.RealtimeConnection] directly: the raw codec events a
connection yields to the session, the turn-control verbs and inputs a connection accepts, and the
model-profile merge helpers.
"""

from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import KW_ONLY, dataclass
from typing import Any, ClassVar, Literal

from typing_extensions import TypeAliasType

from .. import _utils
from ..messages import (
    BinaryAudio,
    BinaryContent,
    BinaryImage,
    FinishReason,
    ModelMessage,
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
    SystemPromptPart,
    ToolCallPart,
    UserContent,
)
from ..usage import RequestUsage
from ._lifecycle import LifecycleEvent, RealtimeFrame, StaleEvent
from .profiles import DEFAULT_AUDIO_SAMPLE_RATE, DEFAULT_REALTIME_PROFILE, merge_realtime_profile

# Input content types (fed into the connection via `send`). Session content reuses the shared message
# vocabulary — `str` for a text turn, `BinaryAudio`/`BinaryImage` from `pydantic_ai.messages` for
# audio and image frames — so only the wire-level `ToolResult` and the turn-control verbs are
# codec-specific types.


@dataclass(repr=False)
class ToolResult:
    """The result of a tool call, rendered for the wire and sent back to the model.

    Built and sent by [`RealtimeSession`][pydantic_ai.realtime.RealtimeSession] after it settles a
    call: the string-only realtime tool channel and the retry/failure error-key wrapping mean the
    session renders the [`ToolReturnPart`][pydantic_ai.messages.ToolReturnPart] it records in history
    down to this flat shape, so every provider sends exactly the same rendering.
    """

    tool_call_id: str
    """Identifier of the `ToolCall` this result answers."""
    _: KW_ONLY
    output: str
    """The tool's output, rendered as a string."""
    content: Sequence[UserContent] | None = None
    """Additional user content to send after the tool output when the provider supports it."""
    failed: bool = False
    """Whether the call failed: the tool failed, or the model is asked to retry it.

    `output` already says so, so it reaches every provider; one whose protocol also flags a failed call
    (ElevenLabs' `is_error`) sets the flag too, as a standard run does on a model that has one.
    """

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass
class TextContext:
    """A text item added to the conversation as context, without asking the model to respond (like an image)."""

    text: str


@dataclass
class CommitAudio:
    """Commit the buffered input audio as a user turn (manual turn-taking / push-to-talk).

    Only needed when automatic voice activity detection is disabled; with server-side VAD the
    provider commits audio and triggers a response automatically.
    """


@dataclass
class ClearAudio:
    """Discard any buffered, uncommitted input audio."""


@dataclass
class CreateResponse:
    """Ask the model to generate a response now (manual turn-taking, after `CommitAudio`)."""


@dataclass
class CancelResponse:
    """Cancel the model's in-progress response (maps to the provider's response-cancel)."""


@dataclass(repr=False)
class TruncateOutput:
    """Truncate the model's current audio output at `audio_end_ms`.

    After a barge-in the user only heard part of the model's audio. Truncating tells the provider how
    much was actually played, so its stored transcript matches and the conversation context stays
    consistent. The provider resolves which output item to truncate from its own state unless
    `item_id` names one.
    """

    audio_end_ms: int
    """Milliseconds of the output audio that were actually played before the interruption."""

    _: KW_ONLY
    item_id: str | None = None
    """The output item to truncate, when it isn't the provider's current one — a reply the listener was
    still hearing after a newer one was generated. `None` truncates the current output item."""

    __repr__ = _utils.dataclasses_no_defaults_repr


RealtimeSessionInput = TypeAliasType('RealtimeSessionInput', 'str | BinaryContent | SystemPromptPart')
"""The content types a caller feeds into [`RealtimeSession.send`][pydantic_ai.realtime.RealtimeSession.send].

Session content only, in the shared message vocabulary: a `str` is a complete text turn,
[`BinaryContent`][pydantic_ai.messages.BinaryContent] carries an image frame, WAV audio (unwrapped to
raw PCM before it is streamed, matching the history-seeding path), or a raw PCM chunk
(`media_type='audio/pcm'`), and a [`SystemPromptPart`][pydantic_ai.messages.SystemPromptPart] is a
system prompt added mid-session, delivered on the provider's channel for the operator's own text. The session normalizes these before forwarding them to the connection.
Turn-control verbs (`CommitAudio`, `ClearAudio`, `CreateResponse`, `CancelResponse`,
`TruncateOutput`) are connection-level vocabulary driven through the dedicated `RealtimeSession`
methods (`commit_audio()`, `clear_audio()`, `create_response()`, `interrupt()`), and
[`ToolResult`][pydantic_ai.realtime.codec.ToolResult] is sent by the session itself when a tool
completes — neither is accepted by `send()`.
"""

RealtimeInput = TypeAliasType(
    'RealtimeInput',
    'str | TextContext | BinaryAudio | BinaryImage | CommitAudio | ClearAudio | CreateResponse | CancelResponse | TruncateOutput | ToolResult',
)
"""Union of content types accepted by [`RealtimeConnection.send`][pydantic_ai.realtime.codec.RealtimeConnection.send].

The connection-level counterpart of [`RealtimeSessionInput`][pydantic_ai.realtime.RealtimeSessionInput],
already normalized: a `str` is a complete text turn that solicits a response,
[`TextContext`][pydantic_ai.realtime.codec.TextContext] is text added without soliciting a response, a
[`BinaryAudio`][pydantic_ai.messages.BinaryAudio] carries a raw mono PCM16 chunk at the model's
[`audio_input_sample_rate`][pydantic_ai.realtime.RealtimeSession.audio_input_sample_rate]
(`media_type='audio/pcm'`), and a [`BinaryImage`][pydantic_ai.messages.BinaryImage] an image frame.
The connection additionally accepts the turn-control verbs and
[`ToolResult`][pydantic_ai.realtime.codec.ToolResult], which
[`RealtimeSession`][pydantic_ai.realtime.RealtimeSession] sends on the caller's behalf.
"""


# Connection-level events (yielded by `RealtimeConnection.__aiter__`).


@dataclass(repr=False)
class AudioDelta:
    """A chunk of audio output from the model."""

    data: bytes
    """Raw PCM audio bytes. The sample rate is provider-specific."""
    _: KW_ONLY
    item_id: str | None = None
    """Provider item ID for the spoken output this chunk belongs to, when available."""
    response_id: str | None = None
    """Provider ID of the response this belongs to, when available.

    Lets the session name a reply that never gets the terminal that would otherwise carry its ID, such
    as one cut off by a dropped connection or by closing the session.
    """

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass(repr=False)
class OutputTranscript:
    """The model's textual output (partial or final): an audio transcript, or plain text output."""

    text: str
    """Transcript text. A partial event carries the incremental delta; a final event the full turn."""
    _: KW_ONLY
    is_final: bool = False
    """Whether this is the final transcript for the turn."""
    output_text: bool = False
    """Whether this is the model's plain text output (`output_modalities=('text',)`) rather than a
    transcription of spoken audio. Text output becomes a [`TextPart`][pydantic_ai.messages.TextPart];
    an audio transcript becomes a [`SpeechPart`][pydantic_ai.messages.SpeechPart]."""
    item_id: str | None = None
    """Provider item ID for the spoken output, when available."""
    response_id: str | None = None
    """Provider ID of the response this belongs to, when available.

    Lets the session name a reply that never gets the terminal that would otherwise carry its ID, such
    as one cut off by a dropped connection or by closing the session.
    """

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass(repr=False)
class InputTranscript:
    """A transcription of the user's audio input (partial or final).

    Providers with per-item IDs use `item_id` to associate interleaved transcripts with the correct
    user turn. Providers without them retain arrival-order association.
    """

    text: str
    """Transcript text."""
    _: KW_ONLY
    is_final: bool = False
    """Whether this is the final transcript for the user's turn."""
    item_id: str | None = None
    """Provider item ID for the user's turn, when available."""
    cumulative: bool = False
    """Whether `text` is the whole transcript so far rather than an incremental piece.

    Speech recognition is revisable, and a provider that streams cumulative snapshots may correct
    what it already transcribed instead of only extending it. Setting this lets the session adopt
    each snapshot as authoritative rather than guessing from prefixes whether the text appends;
    it surfaces the difference to callers as a
    [`SpeechPartDelta.transcript`][pydantic_ai.messages.SpeechPartDelta.transcript] carrying the
    corrected whole. Leave `False` for incremental deltas.
    """

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass(repr=False)
class ToolCall:
    """The model is requesting a tool call."""

    tool_call_id: str
    """Provider-assigned identifier for this call."""
    _: KW_ONLY
    tool_name: str
    """Name of the tool to invoke."""
    args: str
    """Raw JSON-encoded arguments. May be an empty string if the model sent no arguments."""
    response_usage_follows: bool = False
    """Whether per-response [`SessionUsage`][pydantic_ai.realtime.codec.SessionUsage] will follow
    this call before the provider's response is complete.

    OpenAI-protocol providers report calls before `response.done`, which carries usage; the session
    uses this signal to keep all calls and their usage on the same `ModelResponse`."""
    item_id: str | None = None
    """Provider conversation-item ID for this call, when available."""
    response_id: str | None = None
    """Provider ID of the response this belongs to, when available.

    Lets the session name a reply that never gets the terminal that would otherwise carry its ID, such
    as one cut off by a dropped connection or by closing the session.
    """
    response_continues: bool = False
    """Whether the model goes on with the same response while this call runs.

    Gemini Live's `NON_BLOCKING` calls do on the models that take `INTERRUPT` scheduling (the native-audio models
    and `gemini-3.8-live`): the model keeps talking ("this might take a moment") in the turn that made the call.
    An [`InferredLifecycle`][pydantic_ai.realtime.lifecycle.InferredLifecycle] then keeps that response going past
    the call, whatever `response_usage_follows` says, so what the model says before the call's result goes back is
    recorded with the call, ahead of the result. A message the call's tool enqueues `asap` goes out ahead of the
    result without waiting for that response to end, as the result ends it."""

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass(repr=False)
class ToolCallCancelled:
    """The model cancelled in-flight tool calls (e.g. the user barged in before they finished).

    Gemini Live sends this as `toolCallCancellation`; the session cancels the matching running tool
    tasks so their now-unwanted results are never sent back to the model.
    """

    tool_call_ids: list[str]
    """Identifiers of the [`ToolCall`][pydantic_ai.realtime.codec.ToolCall]s that were cancelled."""

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass(repr=False)
class ResponseDone:
    """The provider reported that its current response is done.

    This codec event is consumed by the session to finalize a
    [`ModelResponse`][pydantic_ai.messages.ModelResponse]. Providers do not necessarily report a
    terminal for every model response the session records in history.
    """

    _: KW_ONLY
    interrupted: bool = False
    """Whether the response ended because it was cancelled (e.g. the user barged in)."""

    provider_response_id: str | None = None
    """Provider-assigned ID for the completed response, when available."""

    finish_reason: FinishReason | None = None
    """Normalized reason the provider finished the response, when available."""

    provider_details: dict[str, Any] | None = None
    """Raw provider terminal status details retained on the finalized response, when available."""

    more_expected: bool = False
    """Whether the provider said this response is *not* the last of the exchange.

    The session works out for itself that a response calling a tool will be followed by another, but a
    model that reasons in the background can finish a response, keep working, and speak again with no
    tool call in between: `gemini-3.8-live-extended-thinking` speaks a filler ("Let me check those
    flights"), ends the turn, and only then issues the tool call. Left unsaid, that filler would
    synthesize a [`RealtimeTurnCompleteEvent`][pydantic_ai.messages.RealtimeTurnCompleteEvent] claiming
    the exchange was over while the model was still working on it.
    """

    event_kind: Literal['response_done'] = 'response_done'
    """Event type identifier, used as a discriminator."""

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass(repr=False)
class SessionUsage:
    """Usage reported by the provider for a model response or another run-level operation."""

    usage: RequestUsage
    """Normalized usage ready to accumulate into a `RunUsage`."""

    _: KW_ONLY
    provider_response_id: str | None = None
    """Provider-assigned ID for the response this usage belongs to, when available."""

    finish_reason: FinishReason | None = None
    """Normalized completion reason for the response this usage belongs to, when available."""

    provider_details: dict[str, Any] | None = None
    """Provider-specific details about how this usage was incurred, merged into the response's
    `ModelResponse.provider_details`. Only applies to response-scoped usage."""

    response_scoped: bool = True
    """Whether this usage belongs to a specific model response.

    `True`, the default, accumulates it both into the run total and the response's
    `ModelResponse.usage`. `False` is run-level only, e.g. input audio transcription usage,
    which is billed on a separate model/meter and is accumulated into the run's `RunUsage`
    but attributed to no `ModelResponse`.

    A response-scoped report read as the session closes, after its response was recorded, goes into the run total
    only: there's no response left to add it (or its `provider_details`) to. On a provider whose responses aren't
    requests (GPT-Live), it still counts as one, against `request_limit` too.
    """

    context_window_used: float | None = None
    """The fraction of the model's context window in use, when the provider reports it.

    A snapshot rather than an amount to accumulate: the session keeps the latest reported value and
    exposes it as [`RealtimeSession.context_window_used`][pydantic_ai.realtime.RealtimeSession.context_window_used].
    It can go down after the provider compacts or truncates the conversation. `None`, the default,
    means this report says nothing about the context window.
    """

    event_kind: Literal['session_usage'] = 'session_usage'
    """Event type identifier, used as a discriminator."""

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass(repr=False)
class ConversationCreated:
    """An OpenAI-protocol server assigned a conversation ID.

    This is a codec-level control event. Providers consume it during their handshake when possible;
    the session silently consumes any instance that reaches the live stream.
    """

    conversation_id: str
    """Provider-assigned conversation ID."""

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass(repr=False)
class ConversationItemCreated:
    """An OpenAI-protocol server reported a conversation item.

    A connection that resumes a provider-side conversation marks the items the provider replays into it
    with `replayed=True`. The session consumes those items and remembers their newly assigned IDs so any
    follow-on content or tool events aren't appended or executed again.
    """

    _: KW_ONLY
    item_id: str | None = None
    """Provider-assigned conversation-item ID, when present."""
    tool_call_id: str | None = None
    """Provider-assigned tool-call ID, for function call and result items."""
    replayed: bool = False
    """Whether the provider identified this item as part of a resumption replay."""

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass(repr=False)
class InputRejected:
    """The provider refused part of an input this connection was sent, so it had no effect.

    Yielded just ahead of the [`RealtimeSessionErrorEvent`][pydantic_ai.realtime.RealtimeSessionErrorEvent]
    that explains the refusal, and only when the provider's error identifies the frame it refused (the
    OpenAI protocol echoes the client `event_id`). A connection that can't tell which input an error was
    about yields the error alone. A reconnect whose new session no longer has an input (Gemini resumes
    from a handle that can predate a typed turn) yields it too, ahead of the
    [`RealtimeSessionReconnectEvent`][pydantic_ai.realtime.RealtimeSessionReconnectEvent]. The session
    uses it to take back what it assumed the input did: a
    refused request for a response releases the reply
    [`wait_for_reply()`][pydantic_ai.realtime.RealtimeSession.wait_for_reply] would otherwise wait for
    forever, and refused content is removed from history.
    """

    input_index: int
    """Zero-based position of the refused input among every [`send`][pydantic_ai.realtime.codec.RealtimeConnection.send] call made on this connection, including calls that raised."""
    _: KW_ONLY
    refused: Literal['content', 'response']
    """What was refused: `'content'` when the input never joined the conversation, `'response'` when the response it asked for will never come."""

    __repr__ = _utils.dataclasses_no_defaults_repr


RealtimeCodecEvent = TypeAliasType(
    'RealtimeCodecEvent',
    AudioDelta
    | OutputTranscript
    | InputTranscript
    | ToolCall
    | ToolCallCancelled
    | ResponseDone
    | RealtimeInputSpeechStartEvent
    | RealtimeResponseInterruptedEvent
    | RealtimeInputSpeechEndEvent
    | RealtimeOutputSpeechStartEvent
    | RealtimeOutputSpeechEndEvent
    | RealtimeInputTranscriptionErrorEvent
    | SessionUsage
    | RealtimeSessionReconnectEvent
    | ConversationCreated
    | ConversationItemCreated
    | InputRejected
    | PartStartEvent
    | PartEndEvent
    | RealtimeSessionErrorEvent,
)
"""Union of the low-level codec events yielded by [`RealtimeConnection`][pydantic_ai.realtime.codec.RealtimeConnection].

This is the provider-facing vocabulary: providers translate their wire protocol into these events, and
[`RealtimeSession`][pydantic_ai.realtime.RealtimeSession] translates them again into the shared
[`RealtimeEvent`][pydantic_ai.realtime.RealtimeEvent] vocabulary while building
[`ModelMessage`][pydantic_ai.messages.ModelMessage] history.
"""


class RealtimeConnection(ABC):
    """A live connection to a realtime model.

    Providers implement this to handle protocol-specific framing (WebSocket frames,
    HTTP/2 messages, etc.). Content is fed in via [`send`][pydantic_ai.realtime.codec.RealtimeConnection.send],
    and what the model does comes out of [`frames`][pydantic_ai.realtime.codec.RealtimeConnection.frames]: the
    codec events, with the [lifecycle events][pydantic_ai.realtime.lifecycle] that say which response, spoken
    turn, and input each is about. Iterating the connection yields the codec events alone.
    """

    transport_errors: ClassVar[tuple[type[Exception], ...]] = ()
    """The exception types this connection's transport raises when the link to the provider fails.

    A [`RealtimeSession`][pydantic_ai.realtime.RealtimeSession] maps these to
    [`RealtimeError`][pydantic_ai.realtime.RealtimeError] so a failed send surfaces as the same typed
    error as a failed receive, instead of leaking a `websockets` or provider-SDK exception the caller
    has no reason to expect from a model call. Leave empty if `send` already raises typed errors.

    The mapping covers the whole of [`send`][pydantic_ai.realtime.codec.RealtimeConnection.send], so
    anything it does *besides* writing to the transport — converting content, downloading media —
    must not raise these types, or a local failure would be reported as a lost connection. Do that
    work before the first frame goes out.
    """

    @abstractmethod
    async def send(self, content: RealtimeInput) -> Sequence[LifecycleEvent] | None:
        """Feed content into the session.

        Returns the [lifecycle events][pydantic_ai.realtime.lifecycle] the send itself settles, if any: where the
        input joined the conversation, say, for a provider that sends no acknowledgement to report it in. (An
        [`InferredLifecycle`][pydantic_ai.realtime.lifecycle.InferredLifecycle] has them in
        [`take_pending`][pydantic_ai.realtime.lifecycle.InferredLifecycle.take_pending].)

        Concrete connections accept provider-specific data and control inputs. OpenAI accepts audio,
        text, images, tool results, manual turn controls, cancellation, and truncation; Gemini accepts
        audio, text, images, and tool results. A high-level
        [`RealtimeSession`][pydantic_ai.realtime.RealtimeSession] checks profile-gated operations and
        raises [`UserError`][pydantic_ai.exceptions.UserError], as does a connection handed an input it
        can't send.
        """
        raise NotImplementedError

    @abstractmethod
    def frames(self) -> AsyncIterator[RealtimeFrame]:
        """Iterate over what the model does, a [`RealtimeFrame`][pydantic_ai.realtime.lifecycle.RealtimeFrame] at a time.

        Each frame holds the codec events one message from the provider makes, in order, with the
        [lifecycle events][pydantic_ai.realtime.lifecycle] about them, and ends once the connection is gone for
        good: by then every response it started has ended (as `'lost'` if nothing else ended it). A protocol
        that identifies nothing itself can have an
        [`InferredLifecycle`][pydantic_ai.realtime.lifecycle.InferredLifecycle] make its frames.
        """
        raise NotImplementedError

    async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
        """Iterate over the codec events received from the model, stale ones included (see `frames`)."""
        async for frame in self.frames():
            for event in frame:
                if isinstance(event, StaleEvent):
                    yield event.event
                elif not isinstance(event, LifecycleEvent):
                    yield event

    async def _end_session(self) -> AsyncIterator[SessionUsage]:
        """End the provider session, yielding the usage the provider reports only as it ends.

        Private: internal plumbing between the session and the built-in connections.

        The session iterates this once while closing, after
        it has stopped reading the connection and before it reports the session's usage, whenever the
        session owns the provider session (not on a WebRTC sideband, where ending it would end the browser's
        call). A provider that already ended the session, or went away, has nothing more to ask: yield what
        it reported that the session hasn't taken yet, if anything. Its usage counts toward the session, but
        is attributed to no response, as none is left to take it; a response-scoped report (a request the model
        made, on a provider whose responses aren't requests) also counts as that request.

        The session bounds how long it iterates, and stops at a transport error; everything yielded until
        then is recorded. It skips this when the session is closing because it was cancelled.

        The default yields nothing, for providers that report all usage as it happens or that end the
        session by closing the transport.
        """
        return
        yield  # pragma: no cover

    def _stop_reading(self) -> Sequence[LifecycleEvent]:
        """The lifecycle events of what the provider has, but the session won't read about, as it closes.

        Private: internal plumbing between the session and the built-in connections.

        The session calls this once while closing, after it has stopped reading the connection and before it settles
        its history. A provider that acknowledges what it was sent only later (the OpenAI protocol's audio commit) has
        the acknowledgements the session will now never read: say what they would have.

        The default returns none.
        """
        return ()

    @property
    def model_name(self) -> str | None:
        """The model id the server reported serving this session, when the provider reports one.

        Captured from the connect handshake (e.g. the OpenAI protocol's `session.created`). It can
        differ from the requested model id: xAI accepts any model slug and silently substitutes its
        current default, reporting the actually-served model only here. `None` when the provider
        doesn't report one (e.g. Gemini Live). The session stamps this on each
        [`ModelResponse.model_name`][pydantic_ai.messages.ModelResponse.model_name], mirroring how
        request-response models record the response's reported model rather than the requested one.
        """
        return None

    def set_message_history(self, message_history: Callable[[], Sequence[ModelMessage]]) -> None:
        """Tell the connection how to read the message history as it currently stands.

        A [`RealtimeSession`][pydantic_ai.realtime.RealtimeSession] calls this when it takes ownership of
        the connection, so a provider that loses server-side state on
        [reconnect][pydantic_ai.realtime.ReconnectPolicy] can replay the conversation into the new
        session instead of resuming with total amnesia. The session's history grows as the call goes on,
        hence a callable rather than a snapshot.

        A no-op by default: a provider that can't seed a session has nowhere to put it. Gemini Live, which
        resumes natively, replays only when a reconnect gets no resumption handle to resume from.
        """

    @property
    def input_transcription_enabled(self) -> bool:
        """Whether this connection will emit [`InputTranscript`][pydantic_ai.realtime.codec.InputTranscript] events for the user's audio.

        Providers that transcribe the user's input (the default) leave this `True`. When it is `False`,
        no transcript arrives, so [`RealtimeSession`][pydantic_ai.realtime.RealtimeSession] finalizes a
        user turn from retained input audio instead (see `audio_retention`). Defaults to `True` so a
        connection that doesn't override it never triggers the audio-only path (which would risk a
        duplicate turn if transcripts did arrive).
        """
        return True

    @property
    def interrupts_response_on_speech(self) -> bool:
        """Whether the configured turn detection makes the *server* cancel the active response when user speech starts.

        OpenAI-protocol server VAD with `interrupt_response` enabled (the default) cancels the
        in-progress response on speech onset by itself. A client cancel sent at that same moment can
        race the server's own cancellation and be applied to the *next* response instead, silencing
        the reply to the barge-in — so the session's automatic barge-in handling sends only the
        truncation when this is `True`. Defaults to `False`, which keeps the client-side cancel.
        """
        return False

    @property
    def _prunes_tool_calls(self) -> bool:
        """Whether this connection deletes tool rounds that are over (see `_prune_tool_calls`). `False` by default."""
        return False

    async def _prune_tool_calls(self, calls: Sequence[ToolCallPart]) -> Sequence[str]:
        """Delete the given calls of tool rounds that are over, with their results, from the provider's conversation.

        For a connection whose provider keeps the conversation and bills its input tokens on every response: what
        it deletes, the model no longer reads. The session calls this, under its send lock, with the answered calls
        of tool rounds once a response completes without calling a tool, if `_prunes_tool_calls`; history keeps them.
        Private until the realtime refactor settles. Returns the ids of the calls it asked the provider to delete.
        """
        # A connection whose `_prunes_tool_calls` is `True` overrides it.
        raise NotImplementedError

    @property
    def _can_reconnect(self) -> bool:
        """Whether this connection will still re-dial if its link drops.

        Private while send retries across a reconnect are being redesigned. `False` without a reconnect
        policy, once its `max_reconnects` budget is spent, and once a reconnect has failed for good.
        While it is `True`, a [`RealtimeSession`][pydantic_ai.realtime.RealtimeSession] drops an audio
        chunk that hits the dropped link instead of raising, so a microphone task survives the reconnect.
        """
        return False

    @property
    def _answers_tool_calls_per_response(self) -> bool:
        """Whether one reply answers all the tool results of a model response, rather than one per result.

        For the session's reply accounting only. The built-in connections make it so (Gemini Live answers
        a tool-call frame once; the OpenAI-protocol connection asks for one response per calling
        response; GPT-Live continues its delegated backend once per round of calls), and the session then
        counts one reply per tool-calling response against `request_limit`, and none once one of its calls
        settles without a result. Defaults to `False`, which counts one per result, as a connection that
        asks for a response after each one needs.
        """
        return False

    @property
    def _speech_spans_tool_calls(self) -> bool:
        """Whether the model's speech goes on through the tool calls it makes, rather than stopping at each one.

        Then a tool call doesn't end the speech part under way: the speech is recorded as one part, ahead of the calls
        made while it went on, as a standard run orders parts by where they start. Defaults to `False`: a call ends the
        part before it, and speech after it is a part of its own. Private until the realtime refactor settles.
        """
        return False

    def _tool_call_unanswered(self, tool_call_id: str) -> None:
        """The session settled this call without sending its result (a usage limit stopped it, or the send failed).

        For a connection that holds state until a round's results are all in. Does nothing by default.
        Private until the realtime refactor settles.
        """

    @property
    def reconnect_restores_in_flight_state(self) -> bool:
        """Whether a reconnect continues the response and tool calls that were in flight when the socket dropped.

        Otherwise it only brings back the finalized conversation. Gemini Live settles the cut turn in the
        connection before its
        [`RealtimeSessionReconnectEvent`][pydantic_ai.messages.RealtimeSessionReconnectEvent], so the
        [`RealtimeSession`][pydantic_ai.realtime.RealtimeSession] must not settle again and trusts
        `state_restored`. Local replay (OpenAI, Azure OpenAI, xAI) restores only finalized turns, so
        the session settles the interrupted turn itself and reports `state_restored=False`. Defaults to
        `True`; the OpenAI connection overrides it.
        """
        return True


__all__ = (
    # Connection ABC and the event/input unions it exchanges with a session.
    'RealtimeConnection',
    'RealtimeCodecEvent',
    'RealtimeInput',
    # Codec events a connection yields.
    'AudioDelta',
    'OutputTranscript',
    'InputTranscript',
    'ToolCall',
    'ToolResult',
    'ToolCallCancelled',
    'ResponseDone',
    'ConversationCreated',
    'ConversationItemCreated',
    'InputRejected',
    'SessionUsage',
    # Turn-control verbs a connection accepts.
    'CommitAudio',
    'ClearAudio',
    'CreateResponse',
    'TextContext',
    'CancelResponse',
    'TruncateOutput',
    # Model-profile helpers for provider implementations.
    'merge_realtime_profile',
    'DEFAULT_AUDIO_SAMPLE_RATE',
    'DEFAULT_REALTIME_PROFILE',
)
