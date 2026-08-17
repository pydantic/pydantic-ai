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
    UserContent,
)
from ..usage import RequestUsage
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
    session renders the [`ToolReturnPart`][pydantic_ai.messages.ToolReturnPart] or
    [`RetryPromptPart`][pydantic_ai.messages.RetryPromptPart] it records in history down to this flat
    shape, so every provider sends exactly the same rendering.
    """

    tool_call_id: str
    """Identifier of the `ToolCall` this result answers."""
    _: KW_ONLY
    output: str
    """The tool's output, rendered as a string."""
    content: Sequence[UserContent] | None = None
    """Additional user content to send after the tool output when the provider supports it."""

    __repr__ = _utils.dataclasses_no_defaults_repr


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
    consistent. The provider resolves which output item to truncate from its own state.
    """

    audio_end_ms: int
    """Milliseconds of the current output audio that were actually played before the interruption."""

    __repr__ = _utils.dataclasses_no_defaults_repr


RealtimeSessionInput = TypeAliasType('RealtimeSessionInput', 'str | BinaryContent')
"""The content types a caller feeds into [`RealtimeSession.send`][pydantic_ai.realtime.RealtimeSession.send].

Session content only, in the shared message vocabulary: a `str` is a complete text turn, and
[`BinaryContent`][pydantic_ai.messages.BinaryContent] carries an image frame, WAV audio (unwrapped to
raw PCM before it is streamed, matching the history-seeding path), or a raw PCM chunk
(`media_type='audio/pcm'`). The session normalizes these before forwarding them to the connection.
Turn-control verbs (`CommitAudio`, `ClearAudio`, `CreateResponse`, `CancelResponse`,
`TruncateOutput`) are connection-level vocabulary driven through the dedicated `RealtimeSession`
methods (`commit_audio()`, `clear_audio()`, `create_response()`, `interrupt()`), and
[`ToolResult`][pydantic_ai.realtime.codec.ToolResult] is sent by the session itself when a tool
completes — neither is accepted by `send()`.
"""

RealtimeInput = TypeAliasType(
    'RealtimeInput',
    'str | BinaryAudio | BinaryImage | CommitAudio | ClearAudio | CreateResponse | CancelResponse | TruncateOutput | ToolResult',
)
"""Union of content types accepted by [`RealtimeConnection.send`][pydantic_ai.realtime.codec.RealtimeConnection.send].

The connection-level counterpart of [`RealtimeSessionInput`][pydantic_ai.realtime.RealtimeSessionInput],
already normalized: a `str` is a complete text turn, a
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

    response_scoped: bool = True
    """Whether this usage belongs to a specific model response.

    `True`, the default, accumulates it both into the run total and the response's
    `ModelResponse.usage`. `False` is run-level only, e.g. input audio transcription usage,
    which is billed on a separate model/meter and is accumulated into the run's `RunUsage`
    but attributed to no `ModelResponse`.
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

    xAI uses `replayed=True` for item events emitted during the resume handshake. The session consumes
    those items and remembers their newly assigned IDs so any follow-on content or tool events aren't
    appended or executed again.
    """

    _: KW_ONLY
    item_id: str | None = None
    """Provider-assigned conversation-item ID, when present."""
    tool_call_id: str | None = None
    """Provider-assigned tool-call ID, for function call and result items."""
    replayed: bool = False
    """Whether the provider identified this item as part of a resumption replay."""

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
    HTTP/2 messages, etc.). Content is fed in via [`send`][pydantic_ai.realtime.codec.RealtimeConnection.send]
    and events are consumed by iterating the connection.
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
    async def send(self, content: RealtimeInput) -> None:
        """Feed content into the session.

        Concrete connections accept provider-specific data and control inputs. OpenAI accepts audio,
        text, images, tool results, manual turn controls, cancellation, and truncation; Gemini accepts
        audio, text, images, and tool results. A high-level
        [`RealtimeSession`][pydantic_ai.realtime.RealtimeSession] checks profile-gated operations and
        raises [`UserError`][pydantic_ai.exceptions.UserError], as does a connection handed an input it
        can't send.
        """
        raise NotImplementedError

    @abstractmethod
    def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
        """Iterate over events received from the model."""
        raise NotImplementedError

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

        A no-op by default: providers with native session resumption (Gemini Live, xAI) have nothing to
        replay, and one that can't seed a session at all has nowhere to put it.
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
    def reconnect_restores_in_flight_state(self) -> bool:
        """Whether a reconnect continues the response and tool calls that were in flight when the socket dropped.

        Otherwise it only brings back the finalized conversation. Native session resumption (xAI Grok Voice) restores the in-flight generation server-side, and
        Gemini Live settles the cut turn in the connection before its
        [`RealtimeSessionReconnectEvent`][pydantic_ai.messages.RealtimeSessionReconnectEvent], so in
        both the [`RealtimeSession`][pydantic_ai.realtime.RealtimeSession] must not settle again and
        trusts `state_restored`. Local replay (OpenAI, Azure OpenAI) restores only finalized turns, so
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
    'SessionUsage',
    # Turn-control verbs a connection accepts.
    'CommitAudio',
    'ClearAudio',
    'CreateResponse',
    'CancelResponse',
    'TruncateOutput',
    # Model-profile helpers for provider implementations.
    'merge_realtime_profile',
    'DEFAULT_AUDIO_SAMPLE_RATE',
    'DEFAULT_REALTIME_PROFILE',
)
