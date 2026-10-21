"""The realtime session's conversation state, over entities a connection identifies (lifecycle version 2).

A `SessionCore` takes everything that happens in a session, in the order it happens, through one
synchronous `apply()`: the events a version 2 connection yields (codec and lifecycle events alike), and
the commands the session issues for what the caller does (an input sent, a tool's result, an
interruption, the session closing). Nothing in it awaits, so each call is one atomic transition, and the
order of the calls is the order of the conversation.

It keeps state by id rather than in "current" slots:

- responses, keyed by the id their `ResponseStarted` gave them, each assembling its own parts and usage;
- spoken user turns, keyed by item id;
- inputs, keyed by their `InputId`, each placed in the conversation where the provider added it;
- reply obligations: every input that asked for a response, settled only by the response the connection
  says answers it (`ResponseStarted.answers`), or by the provider refusing or losing the request.

History is projected from those entities: a message is built once, when its entity is final, and
entities are ordered by where they joined the provider's conversation. A message is visible in
`all_messages()` once every entity placed before it is final, so a response still streaming holds back
what came after it, and nothing already visible ever moves. The one exception is a tool's result, which
sits directly after the response that called it, as request-response APIs require.

A recorded message is never edited in place, so a snapshot already handed out doesn't change. It is swapped
for an updated copy, in the same position, when the provider's copy changes after the fact: when retained
audio is evicted, and when a barge-in cuts off a response that had already ended (generation outruns
playback, so the reply the user speaks over has usually finished), whether the session cut it or the provider
reports what was heard.
"""

from __future__ import annotations as _annotations

from bisect import insort
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, TypeAlias, assert_never

from .._genai_prices import fill_response_cost
from .._utils import fill_run_metadata
from ..messages import (
    INTERRUPTED_TOOL_RETURN_CONTENT,
    BinaryContent,
    FinishReason,
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
    RetryPromptPart,
    SpeechPart,
    SpeechPartDelta,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolReturnPart,
)
from ..usage import RequestUsage, RunUsage
from ._lifecycle import (
    AnswersRevised,
    InputAdded,
    InputId,
    InputLost,
    LifecycleEvent,
    OutputItemCut,
    OutputItemDetails,
    ResponseEnded,
    ResponseRequestRefused,
    ResponseStarted,
    ResponseStatus,
    ToolCallsAbandoned,
    UserTurnClosed,
    UserTurnEnded,
    UserTurnStarted,
)
from ._retained_audio import RetainedAudioBudget
from ._utils import accumulate_transcript, fold_part_transcript, pcm_to_wav, user_transcript_update
from .codec import (
    AudioDelta,
    ConversationCreated,
    ConversationItemCreated,
    InputRejected,
    InputTranscript,
    OutputTranscript,
    RealtimeCodecEvent,
    ResponseDone,
    SessionUsage,
    ToolCall,
    ToolCallCancelled,
)

_WAV_MEDIA_TYPE = 'audio/wav'

MAX_CUTTABLE_OUTPUT_ITEMS = 32
"""How many recent assistant audio items a barge-in can still cut off: far more than can be queued for playback."""


# --- commands: what the caller did -------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class InputSent:
    """The session is sending an input (registered before its first frame goes out).

    `request` is the history entry recording it, when it records one, placed where the provider adds the
    input to its conversation. `solicits` is whether the input asks for a response, which makes it a
    reply obligation. `tool_call_id` names the call a tool result answers.
    """

    input_id: InputId
    request: ModelRequest | None = None
    solicits: bool = False
    tool_call_id: str | None = None


@dataclass(frozen=True, kw_only=True)
class InputWithdrawn:
    """Inputs the session took back: their send failed, or the image retention cap evicted their record."""

    input_ids: tuple[InputId, ...]


@dataclass(frozen=True, kw_only=True)
class AudioSent:
    """Input audio the caller streamed went out, retained for the spoken turn it belongs to.

    Reported once its send completed: a turn the provider ended while it was on its way was ended without it.
    """

    data: bytes


@dataclass(frozen=True, kw_only=True)
class AudioCleared:
    """The caller discarded the buffered input audio."""


@dataclass(frozen=True, kw_only=True)
class ToolReturned:
    """A tool call settled, and `request` records its result (a return, a retry prompt, or a failure)."""

    tool_call_id: str
    request: ModelRequest


@dataclass(frozen=True, kw_only=True)
class ToolCallRefused:
    """The session refused a tool call the model made, before running it.

    A usage limit tripped on it, or receiving stopped before the session got to it. It leaves the call out of
    history, as it never ran, unless its response is recorded already: then the call gets an interrupted return,
    as one the session was running when it closed does.
    """

    tool_call_id: str


@dataclass(frozen=True, kw_only=True)
class Interrupted:
    """The caller interrupted the response being spoken, having heard `played_ms` of it.

    `cuts` are the output items the session truncated on the provider, by item id, each with how much of its
    audio was heard: a response that already ended is recorded as cut off there too.
    """

    played_ms: int | None
    cuts: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True, kw_only=True)
class ExchangeAbandoned:
    """The session hit a failure that stops the model getting what it needs (a tool raised, a limit tripped).

    Nothing owed until now is waited for any more; what is asked for afterwards is.
    """


@dataclass(frozen=True, kw_only=True)
class TranscriptOverdue:
    """A spoken turn waited too long for its transcript: it is recorded with what it has.

    See `transcript_holding_history`.
    """

    turn_id: str


@dataclass(frozen=True, kw_only=True)
class ResponseRefused:
    """The session refused a response as the frame that started it is handled (the request it is would exceed a limit).

    It stops reading the connection, so the response is cut off: it isn't recorded, or counted as a request. It may
    have ended in that frame too.
    """

    response_id: str


@dataclass(frozen=True, kw_only=True)
class ResponseOverdue:
    """A response under way held back the user's turns too long: they are recorded ahead of it, where history stands.

    See `response_holding_turns`.
    """

    response_id: str


@dataclass(frozen=True, kw_only=True)
class ReceiveEnded:
    """The session stopped reading the connection: nothing more will be said, so nothing is owed any more."""


@dataclass(frozen=True, kw_only=True)
class ToolCallsPruned:
    """The connection deleted these tool calls, and their results, from the provider's conversation.

    History keeps them; a reconnect's replay leaves them out (see `replayable_messages`).
    """

    tool_call_ids: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class Closed:
    """The session is closing: everything still open is settled into history, and nothing is owed any more."""


Command: TypeAlias = (
    InputSent
    | InputWithdrawn
    | AudioSent
    | AudioCleared
    | ToolReturned
    | ToolCallRefused
    | Interrupted
    | ExchangeAbandoned
    | TranscriptOverdue
    | ResponseOverdue
    | ResponseRefused
    | ReceiveEnded
    | ToolCallsPruned
    | Closed
)
CoreInput: TypeAlias = RealtimeCodecEvent | LifecycleEvent | Command


# --- effects: what the session streams and instruments ---------------------------------------------


@dataclass(frozen=True)
class ResponseOpened:
    """A response started: the session opens its `chat` span."""

    response_id: str


@dataclass(frozen=True)
class ResponseClosed:
    """A response ended, however it ended: the session closes its `chat` span."""

    response_id: str
    status: ResponseStatus
    message: ModelResponse | None
    """What history records for it: `None` if nothing (the session refused it, or it was cut off before it said
    anything)."""
    input_messages: list[ModelMessage]
    """The conversation ahead of it, as it stood when it ended: what it replied to."""


@dataclass(frozen=True)
class UsageCounted:
    """A usage report was counted toward a response: `usage` is what that request has used so far.

    The per-request input-token limit is checked against it.
    """

    usage: RequestUsage


@dataclass(frozen=True)
class ToolRoundsOver:
    """A response completed without calling a tool: the tool rounds before it are over.

    `calls` are their calls whose results went out, which the provider answered: what the provider still holds of
    them in its conversation only costs input tokens from here on.
    """

    calls: tuple[ToolCallPart, ...]


@dataclass(frozen=True)
class ExchangeOver:
    """A response ended, and with it everything the model owed: what `wait_for_reply()` waits for is over."""

    interrupted: bool
    """Whether the response that ended it was cut off (the user barged in, or the caller interrupted it)."""


Effect: TypeAlias = (
    PartStartEvent
    | PartDeltaEvent
    | PartEndEvent
    | ResponseOpened
    | ResponseClosed
    | UsageCounted
    | ToolRoundsOver
    | ExchangeOver
)
"""What an `apply()` did that the session acts on, in order (see `SessionCore.take_effects`)."""


# --- entities --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Cut:
    """What the listener heard of one spoken output item: how long, and which words, as far as each is known."""

    heard_ms: int | None = None
    heard_transcript: str | None = None

    def merged(self, earlier: _Cut | None) -> _Cut:
        """This cut over an earlier one of the same item: the shorter hearing, and the latest words reported."""
        if earlier is None:
            return self
        heard_ms = min((ms for ms in (self.heard_ms, earlier.heard_ms) if ms is not None), default=None)
        transcript = self.heard_transcript if self.heard_transcript is not None else earlier.heard_transcript
        return _Cut(heard_ms=heard_ms, heard_transcript=transcript)


@dataclass(eq=False)
class _Response:
    id: str
    answers: tuple[InputId, ...]
    model_name: str | None
    """The model serving the session when the response started, which it is priced as."""
    provider_id: bool = True
    """Whether `id` is the provider's, recorded as the message's `provider_response_id`."""
    reported_id: str | None = None
    """The id its terminal named, for a response whose `id` is made up."""
    native_parts: list[ModelResponsePart] = field(default_factory=list[ModelResponsePart])
    """Native tool calls and returns (grounding, code execution), which lead the message, as on a standard run."""
    parts: list[ModelResponsePart] = field(default_factory=list[ModelResponsePart])
    open_part: SpeechPart | TextPart | None = None
    open_part_item: str | None = None
    open_part_index: int = 0
    """The stream index of `open_part` (see `SessionCore._take_index`)."""
    open_transcript: str = ''
    open_audio: bytearray = field(default_factory=bytearray)
    closed_parts: dict[str, tuple[int, SpeechPart | TextPart]] = field(
        default_factory=dict[str, tuple[int, SpeechPart | TextPart]]
    )
    """The stream index and part each item of the output closed, for a transcript of the item that arrives after it
    closed."""

    item_details: dict[str, dict[str, Any]] = field(default_factory=dict[str, dict[str, Any]])
    """What the provider said about each item of the output, for the part the item makes."""
    usage: RequestUsage = field(default_factory=RequestUsage)
    usage_details: dict[str, Any] | None = None
    usage_finish_reason: FinishReason | None = None
    interrupted_at_ms: int | None = None
    refused: bool = False
    """The session refused it as it started (a usage limit): whatever of it comes is no request it records."""
    status: ResponseStatus | None = None
    """How it ended; `None` while it is open."""
    message: ModelResponse | None = None
    """What history records for it, built once it ended (and `None` if it ended with nothing to record)."""
    tool_calls: list[str] = field(default_factory=list[str])
    placed_at: float = 0
    """Its position among the entities that joined the conversation (see `SessionCore._place`)."""


@dataclass(eq=False)
class _UserTurn:
    id: str
    transcript: str = ''
    transcribed: bool = False
    """Whether a transcript arrived at all (an item transcription never reports stays audio-only)."""
    audio: bytes | None = None
    ended: bool = False
    speaking: bool = False
    """Joined the conversation while the user was still saying it (xAI adds its item at speech start), so its
    audio, and with it the turn, ends at the speech end."""
    message: ModelRequest | None = None
    """Built once the turn is final: transcribed (or known never to be) and joined to the conversation."""
    placed_at: float = 0
    """Its position among the entities that joined the conversation (see `SessionCore._place`)."""
    index: int | None = None
    """Its stream index, once its part started streaming (at its first transcript, or once it is built)."""


@dataclass(eq=False)
class _Input:
    id: InputId
    request: ModelRequest


_Entry: TypeAlias = _Response | _UserTurn | _Input


def _has_audio(message: ModelMessage) -> bool:
    return any(isinstance(part, SpeechPart) and part.audio is not None for part in message.parts)


_Obligation = Literal['pending', 'answered', 'refused', 'lost', 'void']


@dataclass(frozen=True)
class Owed:
    """Something the model owes a waiter: a reply to an input, the end of a response, or the answer to a call."""

    kind: Literal['input', 'response', 'call']
    key: str
    epoch: int
    """Which exchange it was owed in; `ExchangeAbandoned` starts the next."""


class SessionCore:
    """The session's conversation state; see the module docstring."""

    def __init__(
        self,
        *,
        model_name: Callable[[], str | None],
        provider_name: str | None,
        provider_url: str | None,
        conversation_id: str | None,
        run_id: str | None,
        responses_are_requests: bool = True,
        answers_tool_calls_per_response: bool = False,
        tracks_tool_rounds: bool = False,
        input_transcription_enabled: bool = True,
        retain_input_audio: bool = False,
        retain_output_audio: bool = False,
        retain_audio_max_seconds: float | None = None,
        input_sample_rate: int = 24000,
        output_sample_rate: int = 24000,
        seeded: Sequence[ModelMessage] = (),
        usage: RunUsage | None = None,
        annotate_response: Callable[[ModelResponse], None] | None = None,
    ) -> None:
        self._model_name = model_name
        self._provider_name = provider_name
        self._provider_url = provider_url
        self._conversation_id = conversation_id
        self._run_id = run_id
        self._responses_are_requests = responses_are_requests
        self._answers_per_response = answers_tool_calls_per_response
        """Whether one reply answers all the tool results of a response, rather than one per result."""
        self._tracks_tool_rounds = tracks_tool_rounds
        """Whether to say when tool rounds are over (`ToolRoundsOver`), for a connection that prunes them."""
        self._transcription = input_transcription_enabled
        self._retain_input = retain_input_audio
        self._retain_output = retain_output_audio
        self._input_rate = input_sample_rate
        self._output_rate = output_sample_rate
        self._seeded = list(seeded)
        self._annotate_response = annotate_response
        """Adds what instrumentation knows about each response's parts (its tools' OTel metadata) before it's recorded."""
        self.usage = usage if usage is not None else RunUsage()
        """Token usage and requests, as the provider reported them for this session, added to what it held."""
        self.requests = 0
        """The requests counted into `usage` here."""

        self._placed: list[_Entry] = []
        """Entities in the order they joined the provider's conversation."""
        self._final_count = 0
        """How many entities lead `_placed` that are final, as far as `_held_from` looked: an entity that is final
        stays final, so it resumes from there rather than walk the whole conversation on every frame."""
        self._responses: dict[str, _Response] = {}
        self._open_responses: dict[str, _Response] = {}
        """The responses under way, in the order they started: those of `_responses` whose `status` is `None`."""
        self._turns: dict[str, _UserTurn] = {}
        self._replayed_items: set[str] = set()
        """Item ids of a resumed conversation's replay, whose content the core already has."""
        self._replayed_calls: set[str] = set()
        """Tool call ids of that replay (ids of another kind, so kept apart: an item id can recur as a call id)."""
        self._inputs: dict[InputId, _Input] = {}
        """The inputs history records (not audio chunks, say, or a bare request for a response), by id."""
        self._unplaced: dict[InputId, _Input] = {}
        """Those not in the conversation yet, in the order they were sent."""
        self._obligations: dict[InputId, _Obligation] = {}
        self._answered_by: dict[InputId, str] = {}
        self._continued_by: dict[str, str] = {}
        """The response each response the provider said the exchange continues past was carried on by."""
        self._call_response: dict[str, _Response] = {}
        """The response each tool call came from."""
        self._recorded_items: OrderedDict[str, tuple[_Response, int]] = OrderedDict()
        """The recorded response and part index of each recent output item that spoke, for a barge-in to cut off."""
        self._pending_cuts: dict[str, _Cut] = {}
        """Cuts of output items of responses still under way, applied once each is recorded."""
        self._returns: dict[str, ModelRequest] = {}
        self._result_input: dict[str, InputId] = {}
        """The input that carried each tool call's result to the provider."""
        self._result_call: dict[InputId, str] = {}
        """The tool call each result input answers."""
        self._undispatched: set[str] = set()
        """Calls the core accepted that the session hasn't taken to run yet (see `take_call`)."""
        self._unanswered_responses: set[str] = set()
        """Responses whose tool results get no reply: one of their calls settled without its result going out."""
        self._settled_calls: set[str] = set()
        """Calls that settled: their result is recorded (sent or not), or the provider cancelled or abandoned them."""
        self._abandoned_calls: set[str] = set()
        """Calls the provider abandoned: they run on, but no response answers their results (see `ToolCallsAbandoned`)."""
        self._input_audio = bytearray()
        # What `retain_audio_max_seconds` bounds, as `retain_images_max` bounds images: the audio built into history,
        # the audio of turns waiting to be recorded, and what is still coming in (`_input_audio`, `open_audio`).
        self._carried_usage: list[SessionUsage] = []
        """Usage reported between responses, naming none: the next response's."""
        self._audio_budget = RetainedAudioBudget(
            retain_audio_max_seconds, input_sample_rate=input_sample_rate, output_sample_rate=output_sample_rate
        )
        self._waiting_turn_audio: OrderedDict[str, _UserTurn] = OrderedDict()
        """Turns with audio cut for them but not recorded yet, oldest first."""
        self._waiting_turn_audio_bytes = 0
        self._open_audio: dict[str, _Response] = {}
        """Responses with output audio coming in for a part, oldest first."""
        self._open_audio_bytes = 0
        self._recorded_audio: deque[_Response | _UserTurn] = deque()
        """Recorded entities whose message still has audio, in the order they joined the conversation: what eviction
        takes audio from first, oldest first, without passing over anything that has none or isn't recorded yet."""
        self._placements = 0
        self._user_speaking = False
        self._speech_segmented = False
        """Whether the provider draws speech-end boundaries, so the retained audio is cut into turns at them."""
        self._closed = False
        self._receiving = True
        self._epoch = 0
        self._abandoned: set[tuple[str, str]] = set()
        """What was owed when an exchange was abandoned: no waiter waits for it again."""
        self._effects: list[Effect] = []
        self._exchange_ending: bool | None = None
        """A response ended in the frame being applied, and whether it was cut off (see `end_frame`)."""
        self._unrounded: dict[str, list[str]] = {}
        """The calls of each recorded response that no `ToolRoundsOver` named yet, while it has any (when tracking)."""
        self._pruned_calls: set[str] = set()
        """Calls the connection deleted from the provider's conversation (see `ToolCallsPruned`)."""
        self._next_index = 0
        """The stream index of the next part to start: one counter for the session, as both speakers' parts stream at
        once, so an index names a part in the event stream rather than a slot in a message."""
        self._dropped_native: set[int] = set()
        """Native parts the core dropped as they started, by the connection's index: their end is dropped too."""
        self._native_indexes: dict[int, int] = {}
        """The stream index of each native tool part under way, by the index the connection numbered it with."""
        self._part_items: OrderedDict[int, str | None] = OrderedDict()
        """The output item each recent assistant part streamed, by stream index, for a barge-in to name what it cuts."""

    # --- the one transition ------------------------------------------------------------------------

    def apply(self, item: CoreInput) -> None:  # noqa: C901
        """Apply one event or command. Synchronous, so it is one transition."""
        if isinstance(item, ResponseStarted):
            self._start_response(item)
        elif isinstance(item, ResponseEnded):
            self._end_response(item)
        elif isinstance(item, (AudioDelta, OutputTranscript)):
            self._response_content(item)
        elif isinstance(item, ToolCall):
            self._tool_call(item)
        elif isinstance(item, RealtimeSessionReconnectEvent):
            # The new connection hears none of what was being said on the old one: a turn still spoken is over,
            # with the audio sent for it.
            for turn_id in [turn_id for turn_id, turn in self._turns.items() if turn.speaking]:
                self._speech_ended(turn_id)
            self._user_speaking = False
        elif isinstance(item, ConversationItemCreated):
            if item.replayed:
                # A natively resumed conversation's replay of what history already has: nothing in it is new.
                if item.item_id is not None:
                    self._replayed_items.add(item.item_id)
                if item.tool_call_id is not None:
                    self._replayed_calls.add(item.tool_call_id)
        elif isinstance(item, OutputItemDetails):
            if (response := self._open_response(item.response_id)) is not None:
                response.item_details[item.item_id] = item.provider_details
        elif isinstance(item, OutputItemCut):
            self._cut_recorded_item(item.item_id, _Cut(heard_ms=item.heard_ms, heard_transcript=item.heard_transcript))
        elif isinstance(item, SessionUsage):
            self._usage(item)
        elif isinstance(item, UserTurnStarted):
            self._turn(item.turn_id)
        elif isinstance(item, UserTurnEnded):
            self._end_turn(item.turn_id, still_speaking=item.still_speaking, before=item.before_response)
        elif isinstance(item, UserTurnClosed):
            self._turn_closed(item.turn_id)
        elif isinstance(item, InputTranscript):
            self._input_transcript(item)
        elif isinstance(item, RealtimeInputTranscriptionErrorEvent):
            # One naming no item is the latest turn's, as a transcript is.
            turn = self._turns.get(item.item_id) if item.item_id is not None else self._latest_turn()
            if turn is not None and not turn.transcribed:  # (a stray one, after its transcript, changes nothing)
                self._transcribed(turn, failed=True)
        elif isinstance(item, RealtimeInputSpeechStartEvent):
            self._user_speaking = True
        elif isinstance(item, RealtimeInputSpeechEndEvent):
            self._speech_ended(item.item_id)
        elif isinstance(item, InputAdded):
            self._place_input(item.input_id)
        elif isinstance(item, AnswersRevised):
            self._revise_answers(item)
        elif isinstance(item, (ResponseRequestRefused, InputLost)):
            self._settle(item.input_ids, 'refused' if isinstance(item, ResponseRequestRefused) else 'lost')
        elif isinstance(item, InputRejected):
            # (Refusing the content of an input that records none, a bare request for a response, takes nothing back.)
            if item.refused == 'content' and item.input_index in self._inputs:
                self._withdraw((item.input_index,))
        elif isinstance(item, ToolCallCancelled):
            self._settled_calls.update(item.tool_call_ids)
            # (One the session hasn't taken yet still runs, to be stopped as the session reaches the cancellation.)
            for call_id in item.tool_call_ids:
                self._call_unanswered(call_id)
        elif isinstance(item, ToolCallsAbandoned):
            self._abandon_calls(item.tool_call_ids)
        elif isinstance(
            item,
            (
                RealtimeOutputSpeechStartEvent,
                RealtimeOutputSpeechEndEvent,
                RealtimeResponseInterruptedEvent,
                RealtimeSessionErrorEvent,
                ConversationCreated,
            ),
        ):
            pass
        elif isinstance(item, ResponseDone):
            # The lifecycle's `ResponseEnded` ends the response; its terminal can still name it, if nothing had.
            if item.provider_response_id and (response := self._open_response(None)) is not None:
                response.reported_id = response.reported_id or item.provider_response_id
        elif isinstance(item, (PartStartEvent, PartEndEvent)):
            self._native_part(item)
        elif isinstance(item, LifecycleEvent):
            raise TypeError(f'The session core has no handling for the lifecycle event {item!r}.')
        else:
            self._command(item)

    def _command(self, command: Command) -> None:
        if isinstance(command, InputSent):
            self._input_sent(command)
        elif isinstance(command, InputWithdrawn):
            self._withdraw(command.input_ids)
        elif isinstance(command, (AudioSent, AudioCleared)):
            self._input_audio_command(command)
        elif isinstance(command, (ToolReturned, ToolCallRefused, ToolCallsPruned)):
            self._tool_command(command)
        elif isinstance(command, Interrupted):
            # (A later cancel without a position doesn't erase where an earlier interruption cut the response.)
            if (response := self._speaking_response()) is not None and command.played_ms is not None:
                response.interrupted_at_ms = command.played_ms
            for item_id, heard_ms in command.cuts:
                self._cut_recorded_item(item_id, _Cut(heard_ms=heard_ms))
        elif isinstance(command, ExchangeAbandoned):
            self._abandoned.update((token.kind, token.key) for token in self.wait_tokens())
            self._epoch += 1
        elif isinstance(command, TranscriptOverdue):
            self._end_transcriptions([command.turn_id])
        elif isinstance(command, ResponseOverdue):
            self._release_held_turns(command.response_id)
        elif isinstance(command, ResponseRefused):
            self._refuse_response(command.response_id)
        elif isinstance(command, ReceiveEnded):
            self._receiving = False
            # Nothing more will be read: not the rest of a response under way, nor of a turn that joined.
            self._lose_open_responses()
            self._end_transcriptions([turn_id for turn_id, turn in self._turns.items() if turn.ended])
        elif isinstance(command, Closed):
            self._close()
        else:
            assert_never(command)

    def _tool_command(self, command: ToolReturned | ToolCallRefused | ToolCallsPruned) -> None:
        if isinstance(command, ToolReturned):
            self._tool_returned(command)
        elif isinstance(command, ToolCallRefused):
            self._refuse_call(command.tool_call_id)
        else:
            self._pruned_calls.update(command.tool_call_ids)

    def _input_audio_command(self, command: AudioSent | AudioCleared) -> None:
        if isinstance(command, AudioSent):
            if self._retain_input:
                self._input_audio.extend(command.data)
                self._bound_retained_audio()
        else:
            self._input_audio.clear()
            # A turn the user was saying that hadn't joined the conversation never will now.
            for turn_id in [turn_id for turn_id, turn in self._turns.items() if not turn.ended]:
                self._drop_turn(turn_id)

    def _input_sent(self, command: InputSent) -> None:
        if command.request is not None:
            self._inputs[command.input_id] = self._unplaced[command.input_id] = _Input(
                command.input_id, command.request
            )
        call_id = command.tool_call_id
        if command.solicits and not (call_id is not None and self._result_unanswered(call_id)):
            self._obligations[command.input_id] = 'pending'
        if call_id is not None:
            self._result_input[call_id] = command.input_id
            self._result_call[command.input_id] = call_id

    def _end_transcriptions(self, turn_ids: Iterable[str]) -> None:
        """Record turns that joined the conversation with the transcript they have, waiting for no more of it."""
        for turn_id in turn_ids:
            if (turn := self._turns.get(turn_id)) is not None and turn.ended:
                turn.speaking = False
                self._transcribed(turn, failed=False)

    # --- responses ----------------------------------------------------------------------------------

    def _start_response(self, event: ResponseStarted) -> None:
        if event.answers:
            # The provider handles a connection's frames in order, so everything sent before the request this
            # response answers was in its conversation before the response started, acknowledged or not.
            last = max(event.answers)
            for input_id in [input_id for input_id in self._unplaced if input_id <= last]:
                self._place_input(input_id)
        response = _Response(event.response_id, event.answers, self._model_name(), provider_id=event.provider_id)
        carried, self._carried_usage = self._carried_usage, []
        for usage in carried:
            self._add_usage(response, usage)
        self._responses[event.response_id] = self._open_responses[event.response_id] = response
        if event.continues is not None:
            self._continued_by[event.continues] = event.response_id
        self._place(response)
        for input_id in event.answers:
            if self._obligations.get(input_id) == 'pending':
                self._obligations[input_id] = 'answered'
                self._answered_by[input_id] = event.response_id
        self._effects.append(ResponseOpened(event.response_id))

    def _revise_answers(self, event: AnswersRevised) -> None:
        """The connection learned better which inputs a response under way answers (see `AnswersRevised`).

        Where the inputs are in history doesn't change: only which response settles their reply obligations.
        """
        response = self._responses[event.response_id]
        assert response.status is None or not event.claimed, f'{response.id!r} ended: it claims nothing more'
        for input_id in event.claimed:
            if self._obligations.get(input_id) == 'pending':
                self._obligations[input_id] = 'answered'
                self._answered_by[input_id] = response.id
                response.answers = (*response.answers, input_id)
        for input_id in event.released:
            if self._answered_by.get(input_id) == response.id and self._obligations.get(input_id) == 'answered':
                self._obligations[input_id] = 'pending'
                del self._answered_by[input_id]
                response.answers = tuple(answer for answer in response.answers if answer != input_id)

    def _response_content(self, event: AudioDelta | OutputTranscript) -> None:
        response = self._open_response(event.response_id)
        if response is None or (event.item_id is not None and event.item_id in self._replayed_items):
            return
        if isinstance(event, OutputTranscript) and self._update_closed_part(response, event):
            return
        output_text = isinstance(event, OutputTranscript) and event.output_text
        part = self._open_part(response, output_text=output_text, item_id=event.item_id)
        index = response.open_part_index
        if isinstance(event, AudioDelta):
            self._effects.append(
                PartDeltaEvent(index=index, delta=SpeechPartDelta(speaker='assistant', audio_chunk=event.data))
            )
            if self._retain_output:
                response.open_audio.extend(event.data)
                self._open_audio[response.id] = response
                self._open_audio_bytes += len(event.data)
                self._bound_retained_audio()
            return
        response.open_transcript, appended = accumulate_transcript(response.open_transcript, event.text)
        if isinstance(part, TextPart):
            response.open_part = replace(part, content=response.open_transcript)
            delta: SpeechPartDelta | TextPartDelta = TextPartDelta(content_delta=appended)
        else:
            response.open_part = replace(part, transcript=response.open_transcript)
            delta = SpeechPartDelta(speaker='assistant', transcript_delta=appended, transcript=response.open_transcript)
        if appended:
            self._effects.append(PartDeltaEvent(index=index, delta=delta))

    def _open_part(self, response: _Response, *, output_text: bool, item_id: str | None) -> SpeechPart | TextPart:
        part = response.open_part
        if part is not None:
            item_changed = (
                response.open_part_item is not None and item_id is not None and response.open_part_item != item_id
            )
            if item_changed or output_text != isinstance(part, TextPart):
                self._close_part(response)
                part = None
            elif item_id is not None and response.open_part_item is None:
                response.open_part_item = item_id
                part = response.open_part = self._with_item_details(response, part, item_id)
        if part is None:
            part = TextPart(content='') if output_text else SpeechPart(speaker='assistant', transcript='')
            part = response.open_part = self._with_item_details(response, part, item_id)
            response.open_part_item = item_id
            response.open_part_index = index = self._take_index()
            response.open_transcript = ''
            self._part_items[index] = item_id
            while len(self._part_items) > MAX_CUTTABLE_OUTPUT_ITEMS:
                self._part_items.popitem(last=False)
            self._effects.append(PartStartEvent(index=index, part=part))
        return part

    def _update_closed_part(self, response: _Response, event: OutputTranscript) -> bool:
        """Fold a transcript into the part its item already closed, rather than open a second part for the item.

        A provider can finish an item after a later item's content arrived, so the item's transcript comes late.
        """
        if event.item_id is None or event.item_id == response.open_part_item:
            return False
        closed = response.closed_parts.get(event.item_id)
        if closed is None or event.output_text != isinstance(closed[1], TextPart):
            return False
        stream_index, part = closed
        position = next(position for position, existing in enumerate(response.parts) if existing is part)
        updated: SpeechPart | TextPart
        restarted: SpeechPart | TextPart
        delta: SpeechPartDelta | TextPartDelta
        if isinstance(part, TextPart):
            updated, appended = fold_part_transcript(part, event.text)
            restarted, delta = part, TextPartDelta(content_delta=appended)
        else:
            updated, appended = fold_part_transcript(part, event.text)
            # Retained audio appears at a part's end, not at its start.
            restarted = replace(part, audio=None)
            delta = SpeechPartDelta(speaker='assistant', transcript_delta=appended, transcript=updated.transcript or '')
        response.parts[position] = updated
        response.closed_parts[event.item_id] = (stream_index, updated)
        if appended:
            # Streamed again under the index it ended with, which a `PartStartEvent` replaces, so the item still
            # streams as one part.
            self._effects.extend(
                [
                    PartStartEvent(index=stream_index, part=restarted),
                    PartDeltaEvent(index=stream_index, delta=delta),
                    PartEndEvent(index=stream_index, part=updated),
                ]
            )
        return True

    def _with_item_details(
        self, response: _Response, part: SpeechPart | TextPart, item_id: str | None
    ) -> SpeechPart | TextPart:
        if item_id is None or (details := response.item_details.get(item_id)) is None:
            return part
        return replace(part, provider_name=self._provider_name, provider_details=details)

    def _close_part(self, response: _Response) -> None:
        part = response.open_part
        if part is None:
            return
        if isinstance(part, SpeechPart):
            if part.transcript == '':
                part = replace(part, transcript=None)
            if self._retain_output and response.open_audio:
                wav = pcm_to_wav(bytes(response.open_audio), self._output_rate)
                audio = BinaryContent(data=wav, media_type=_WAV_MEDIA_TYPE)
                part = replace(part, audio=self._audio_budget.track(audio, len(response.open_audio), output=True))
        response.parts.append(part)
        if response.open_part_item is not None:
            response.closed_parts[response.open_part_item] = (response.open_part_index, part)
        self._effects.append(PartEndEvent(index=response.open_part_index, part=part))
        response.open_part = None
        response.open_part_item = None
        response.open_transcript = ''
        self._open_audio_bytes -= len(response.open_audio)
        self._open_audio.pop(response.id, None)
        response.open_audio.clear()

    def _tool_call(self, event: ToolCall) -> None:
        response = self._open_response(event.response_id)
        if (
            response is None
            or event.tool_call_id in self._call_response
            or event.tool_call_id in self._replayed_calls
            or (event.item_id is not None and event.item_id in self._replayed_items)
        ):
            # Repeated (a resumed conversation replaying it, say): the call is already recorded, once.
            return
        self._close_part(response)
        call = ToolCallPart(tool_name=event.tool_name, args=event.args, tool_call_id=event.tool_call_id)
        response.parts.append(call)
        index = self._take_index()
        self._effects.extend([PartStartEvent(index=index, part=call), PartEndEvent(index=index, part=call)])
        response.tool_calls.append(event.tool_call_id)
        self._call_response[event.tool_call_id] = response
        self._undispatched.add(event.tool_call_id)

    def _refuse_call(self, call_id: str) -> None:
        self._undispatched.discard(call_id)
        if (response := self._call_response.get(call_id)) is None or call_id in self._returns:
            return
        self._call_unanswered(call_id)
        if (message := response.message) is not None:
            # Recorded already (Gemini ends a tool-call response in the message that makes the calls, before the
            # session gets to them), so it stays, and gets the return a call that never finished gets.
            call = next(
                part for part in message.parts if isinstance(part, ToolCallPart) and part.tool_call_id == call_id
            )
            request = ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name=call.tool_name,
                        content=INTERRUPTED_TOOL_RETURN_CONTENT,
                        tool_call_id=call_id,
                        tool_kind=call.tool_kind,
                        outcome='interrupted',
                    )
                ]
            )
            fill_run_metadata(request, run_id=self._run_id, conversation_id=self._conversation_id)
            self._returns[call_id] = request
            self._settled_calls.add(call_id)
            return
        del self._call_response[call_id]
        response.tool_calls.remove(call_id)
        response.parts = [
            part for part in response.parts if not (isinstance(part, ToolCallPart) and part.tool_call_id == call_id)
        ]

    def _tool_returned(self, command: ToolReturned) -> None:
        self._returns[command.tool_call_id] = command.request
        self._settled_calls.add(command.tool_call_id)
        if command.tool_call_id not in self._result_input:
            # Settled without its result going out (cancelled, or the tool raised).
            self._call_unanswered(command.tool_call_id)

    def _call_unanswered(self, call_id: str) -> None:
        """A call settled without its result reaching the provider.

        A connection that answers a response's results with one reply, once it has all of them, gives none for that
        response's calls then: not for the results already out, nor for those still to go (see `_input_sent`).
        """
        if (
            not self._answers_per_response
            or (response := self._call_response.get(call_id)) is None
            or response.id in self._unanswered_responses
        ):
            return
        self._unanswered_responses.add(response.id)
        self._settle([self._result_input[call] for call in response.tool_calls if call in self._result_input], 'void')

    def _result_unanswered(self, call_id: str) -> bool:
        return call_id in self._abandoned_calls or (
            (response := self._call_response.get(call_id)) is not None and response.id in self._unanswered_responses
        )

    def _abandon_calls(self, call_ids: Sequence[str]) -> None:
        """The provider answers none of these calls' results: they run on, and nothing waits for them any more."""
        self._abandoned_calls.update(call_ids)
        self._settled_calls.update(call_ids)
        for call_id in call_ids:
            self._call_unanswered(call_id)
        self._settle([self._result_input[call_id] for call_id in call_ids if call_id in self._result_input], 'void')

    def _native_part(self, event: PartStartEvent | PartEndEvent) -> None:
        """A native tool part (grounding, code execution) the connection streamed, numbered by its own counter.

        It leads its response's message, as on a standard run, and streams under a stream index of its own: the
        connection's numbering knows nothing of the session's, and a repeated index replaces the part at it.
        """
        if isinstance(event, PartStartEvent):
            if (response := self._open_response(None)) is None:
                # No single response under way to lead: the part isn't recorded, so it doesn't stream, nor does its end.
                self._dropped_native.add(event.index)
                return
            self._dropped_native.discard(event.index)
            response.native_parts.append(event.part)
            index = self._native_indexes[event.index] = self._take_index()
        elif event.index in self._dropped_native:
            self._dropped_native.remove(event.index)
            return
        elif (index := self._native_indexes.pop(event.index, None)) is None:
            # An end with no start: it closes nothing, so it gets an index of its own.
            index = self._take_index()
        self._effects.append(replace(event, index=index))

    def _take_index(self) -> int:
        index = self._next_index
        self._next_index += 1
        return index

    def _usage(self, event: SessionUsage) -> None:
        self.usage.incr(event.usage)  # usage-attribution: the session owns its spans
        if not event.response_scoped:
            return
        if not self._responses_are_requests:
            self.usage.requests += 1  # usage-attribution: the session owns its spans; `wrap_run` opens none
            self.requests += 1
        if not event.provider_response_id:
            # Usage naming no response (Gemini's) is for the one under way, or, between responses, for the next:
            # a boundary that said nothing else (Vertex closing a tool call's turn before its answer) isn't one.
            response = self._open_response(None)
            if response is None:
                self._carried_usage.append(event)
                self._effects.append(UsageCounted(sum((usage.usage for usage in self._carried_usage), RequestUsage())))
                return
        elif (response := self._responses.get(event.provider_response_id)) is None:
            # Naming a response the lifecycle gave a made-up id (its content named none): the one under way.
            response = self._open_response(None)
            if response is None or response.provider_id:
                return
            response.reported_id = response.reported_id or event.provider_response_id
        elif response.status is not None:
            # Usage for a response already recorded counts toward the session, never toward its message.
            self._effects.append(UsageCounted(response.usage + event.usage))
            return
        self._add_usage(response, event)
        self._effects.append(UsageCounted(response.usage))

    @staticmethod
    def _add_usage(response: _Response, event: SessionUsage) -> None:
        response.usage = response.usage + event.usage
        response.usage_finish_reason = event.finish_reason or response.usage_finish_reason
        if event.provider_details:
            response.usage_details = {**(response.usage_details or {}), **event.provider_details}

    def _end_response(self, event: ResponseEnded) -> None:
        response = self._responses[event.response_id]
        self._close_part(response)
        response.status = event.status
        self._open_responses.pop(response.id, None)
        answered_calls_only = (
            bool(response.parts)
            and not response.native_parts
            and all(isinstance(part, ToolCallPart) for part in response.parts)
        )
        if event.status != 'lost' and not (event.status == 'completed' and answered_calls_only):
            # A reply is over, and nobody is speaking: audio sent since the last speech boundary is the tail
            # of the last turn, trailing its commit, not the start of the next one.
            if self._speech_segmented and not self._user_speaking:
                self._input_audio.clear()
        message = self._record_response(response, event)
        self._effects.append(ResponseClosed(response.id, event.status, message, self._history_before(response)))
        if event.status != 'lost' and not response.refused:
            self._exchange_ending = event.status == 'cancelled'
        if self._tracks_tool_rounds:
            if message is not None and response.tool_calls:
                self._unrounded[response.id] = list(response.tool_calls)
            elif event.status == 'completed' and message is not None:
                self._tool_rounds_over()

    def _tool_rounds_over(self) -> None:
        """The tool rounds before a response that called no tool are over: name their answered calls, once.

        A call whose result didn't go out (or wasn't answered) never will be: it isn't named, and isn't kept.
        """
        calls: list[ToolCallPart] = []
        for response_id, call_ids in list(self._unrounded.items()):
            message = self._responses[response_id].message
            waiting: list[str] = []
            for call_id in call_ids:
                result = self._result_input.get(call_id)
                outcome = self._obligations.get(result) if result is not None else None
                if outcome == 'answered' and message is not None:
                    calls.extend(
                        part
                        for part in message.parts
                        if isinstance(part, ToolCallPart) and part.tool_call_id == call_id
                    )
                elif outcome == 'pending' or (result is None and call_id not in self._settled_calls):
                    waiting.append(call_id)
            if waiting:
                self._unrounded[response_id] = waiting
            else:
                del self._unrounded[response_id]
        if calls:
            self._effects.append(ToolRoundsOver(tuple(calls)))

    def end_frame(self) -> None:
        """The frame applied is over: an exchange a response ended in it, with nothing more owed, is over.

        Taken at the end of the frame rather than as the response ends: the response that carries the exchange on
        (see `ResponseStarted.continues`) starts in the same frame.
        """
        if (interrupted := self._exchange_ending) is None:
            return
        self._exchange_ending = None
        if not self.wait_tokens():
            self._effects.append(ExchangeOver(interrupted=interrupted))

    def _record_response(self, response: _Response, event: ResponseEnded) -> ModelResponse | None:
        """Record a response that ended, returning what history records for it, if anything."""
        interrupted = event.status in ('cancelled', 'lost')
        if response.refused:
            # (Its usage counted toward the session as it came.)
            return None
        if (
            event.status == 'lost'
            and not response.parts
            and not response.native_parts
            and response.usage == RequestUsage()
        ):
            # Cut off before it said anything or was billed: there is nothing to record.
            return None
        parts = [*response.native_parts, *response.parts]
        for item_id, (_, part) in response.closed_parts.items():
            if isinstance(part, SpeechPart):
                self._recorded_items[item_id] = (response, next(i for i, p in enumerate(parts) if p is part))
                self._recorded_items.move_to_end(item_id)
        while len(self._recorded_items) > MAX_CUTTABLE_OUTPUT_ITEMS:
            self._recorded_items.popitem(last=False)
        pending_cuts = [
            (item_id, self._pending_cuts.pop(item_id))
            for item_id in response.closed_parts
            if item_id in self._pending_cuts
        ]
        # A response the listener cut off that ended before the provider processed the cancel was still cut off, as
        # one that had already ended is (see `_cut_recorded_item`).
        cut = response.interrupted_at_ms is not None and any(isinstance(part, SpeechPart) for part in parts)
        if interrupted or cut:
            for index in range(len(parts) - 1, -1, -1):
                if isinstance(part := parts[index], SpeechPart):
                    parts[index] = replace(part, interrupted_at_ms=response.interrupted_at_ms)
                    break
        finish_reason = event.finish_reason or response.usage_finish_reason
        if finish_reason is None and not interrupted and event.provider_details is None:
            finish_reason = 'stop'
        message = ModelResponse(
            parts=parts,
            usage=response.usage,
            model_name=response.model_name,
            provider_name=self._provider_name,
            provider_url=self._provider_url,
            provider_details={**(response.usage_details or {}), **(event.provider_details or {})} or None,
            provider_response_id=response.id if response.provider_id else response.reported_id,
            finish_reason=finish_reason,
            conversation_id=self._conversation_id,
            state='interrupted' if interrupted or cut else 'complete',
        )
        fill_run_metadata(message, run_id=self._run_id, conversation_id=self._conversation_id)
        if self._annotate_response is not None:
            self._annotate_response(message)
        provider_cost = message.usage.cost
        fill_response_cost(message)
        if provider_cost is None:
            self.usage.incr(RequestUsage(cost=message.usage.cost))  # usage-attribution: the session owns its spans
        self.usage.requests += int(self._responses_are_requests)  # usage-attribution: the session owns its spans
        self.requests += int(self._responses_are_requests)
        response.message = message
        # The message holds the parts now: the core keeps no other reference to their audio.
        response.parts = []
        response.native_parts = []
        response.closed_parts.clear()
        self._record_audio(response, message)
        for item_id, cut in pending_cuts:
            self._cut_recorded_item(item_id, cut)
        return response.message

    def _cut_recorded_item(self, item_id: str, cut: _Cut) -> None:
        """Record that the listener heard only part of a recorded response's output item.

        The response is marked interrupted, as one cut off while it was still being generated is. Its transcript
        becomes what the provider says was heard, with the original kept in the part's
        `provider_details['original_transcript']`, or stays what the provider said when it doesn't say (OpenAI
        doesn't). The message is swapped for a copy in the same position, so a snapshot of history already handed
        out doesn't change. An item cut off twice keeps the shorter hearing.
        """
        if (recorded := self._recorded_items.get(item_id)) is None:
            if any(
                item_id == response.open_part_item or item_id in response.closed_parts
                for response in self._open_responses.values()
            ):
                # Still being generated: cut once it is recorded, however it ends.
                self._pending_cuts[item_id] = cut.merged(self._pending_cuts.get(item_id))
            return
        response, index = recorded
        message = response.message
        assert message is not None
        part = message.parts[index]
        assert isinstance(part, SpeechPart)
        updated = part
        if cut.heard_ms is not None and (part.interrupted_at_ms is None or cut.heard_ms < part.interrupted_at_ms):
            updated = replace(updated, interrupted_at_ms=cut.heard_ms)
        if cut.heard_transcript is not None and (cut.heard_transcript or None) != part.transcript:
            details = dict(part.provider_details or {})
            details.setdefault('original_transcript', part.transcript)
            updated = replace(
                updated,
                transcript=cut.heard_transcript or None,
                provider_name=part.provider_name or self._provider_name,
                provider_details=details,
            )
        if updated is part and message.state == 'interrupted':
            return
        parts = list(message.parts)
        parts[index] = updated
        # The same audio object, so the retained-audio budget still tracks it, and eviction still finds it.
        response.message = replace(message, parts=parts, state='interrupted')

    def _open_response(self, response_id: str | None) -> _Response | None:
        """The open response content names; content naming none (Azure Voice Live's text) goes to the only one open."""
        if response_id is not None:
            response = self._responses.get(response_id)
            return response if response is not None and response.status is None else None
        if len(self._open_responses) != 1:
            return None
        return next(iter(self._open_responses.values()))

    def _speaking_response(self) -> _Response | None:
        return next(reversed(self._open_responses.values()), None)

    # --- spoken turns -------------------------------------------------------------------------------

    def _turn(self, turn_id: str) -> _UserTurn:
        if (turn := self._turns.get(turn_id)) is None:
            turn = self._turns[turn_id] = _UserTurn(turn_id)
        return turn

    def _latest_turn(self) -> _UserTurn | None:
        """The turn started last, while it isn't recorded yet."""
        turn = next(reversed(self._turns.values()), None)
        return turn if turn is not None and turn.message is None else None

    def _speech_ended(self, item_id: str | None) -> None:
        self._user_speaking = False
        self._speech_segmented = True
        if item_id is None or (turn := self._turns.get(item_id)) is None:
            return
        if self._retain_input and self._input_audio and turn.audio is None:
            self._cut_turn_audio(turn)
        if turn.speaking:
            turn.speaking = False
            self._turn_said(turn)

    def _end_turn(self, turn_id: str, *, still_speaking: bool, before: str | None = None) -> None:
        turn = self._turn(turn_id)
        if turn.ended:
            return
        turn.ended = True
        self._place(turn, before=self._responses.get(before) if before is not None else None)
        if still_speaking:
            turn.speaking = True
            return
        if turn.audio is None and self._retain_input and self._input_audio:
            # Committed by hand, with no speech end to cut it at: the whole buffer is this turn's.
            self._cut_turn_audio(turn)
        self._turn_said(turn)

    def _cut_turn_audio(self, turn: _UserTurn) -> None:
        """Give `turn` the input audio retained since the last cut: it waits with the turn until it is recorded."""
        turn.audio = bytes(self._input_audio)
        self._input_audio.clear()
        self._waiting_turn_audio[turn.id] = turn
        self._waiting_turn_audio_bytes += len(turn.audio)

    def _release_turn_audio(self, turn: _UserTurn) -> None:
        if self._waiting_turn_audio.pop(turn.id, None) is not None and turn.audio:
            self._waiting_turn_audio_bytes -= len(turn.audio)

    def _drop_turn(self, turn_id: str) -> None:
        turn = self._turns.pop(turn_id)
        self._release_turn_audio(turn)
        if turn.index is not None:
            # It started streaming, so its part ends, though it never joined the conversation.
            part = SpeechPart(speaker='user', transcript=turn.transcript.strip() or None)
            self._effects.append(PartEndEvent(index=turn.index, part=part))

    def _turn_said(self, turn: _UserTurn) -> None:
        """All of a joined turn's audio is in: it is final once transcribed, or at once if nothing transcribes it."""
        if not self._transcription:
            self._transcribed(turn, failed=True)
        elif turn.transcribed:
            self._build_turn(turn)

    def _turn_closed(self, turn_id: str) -> None:
        if (turn := self._turns.get(turn_id)) is None:
            return
        if turn.ended:
            # It joined the conversation, but no more of it is coming: it keeps what transcript it has.
            turn.speaking = False
            self._transcribed(turn, failed=False)
        else:
            self._drop_turn(turn_id)

    def _input_transcript(self, event: InputTranscript) -> None:
        # Only a spoken turn the connection reported transcribes into history: not, say, the empty audio item
        # an idle timeout commits to nudge the model. A transcript naming no item (Gemini's) is the latest turn's.
        turn = self._turns.get(event.item_id) if event.item_id is not None else self._latest_turn()
        if turn is None or turn.message is not None:
            return
        if turn.index is None:
            turn.index = self._take_index()
            self._effects.append(PartStartEvent(index=turn.index, part=SpeechPart(speaker='user', transcript='')))
        turn.transcript, delta = user_transcript_update(turn.transcript, event.text, cumulative=event.cumulative)
        if delta is not None:
            self._effects.append(PartDeltaEvent(index=turn.index, delta=delta))
        if event.is_final:
            self._transcribed(turn, failed=False)

    def _transcribed(self, turn: _UserTurn, *, failed: bool) -> None:
        if turn.message is not None:
            return
        if failed:
            turn.transcript = ''
        turn.transcribed = True
        if turn.ended and not turn.speaking:
            self._build_turn(turn)

    def _build_turn(self, turn: _UserTurn) -> None:
        self._release_turn_audio(turn)
        turn.message = self._turn_request(turn, retained=True)
        # The message holds the audio now (as WAV): the PCM goes, empty rather than `None` so nothing cuts it again.
        turn.audio = b''
        self._record_audio(turn, turn.message)
        part = turn.message.parts[0]
        assert isinstance(part, SpeechPart)
        if turn.index is None:
            # Nothing streamed for it (no transcript came): it starts and ends at once.
            turn.index = self._take_index()
            self._effects.append(PartStartEvent(index=turn.index, part=part))
        self._effects.append(PartEndEvent(index=turn.index, part=part))

    def _turn_request(self, turn: _UserTurn, *, retained: bool = False) -> ModelRequest:
        """The request `turn` records; counted against `retain_audio_max_seconds` when `retained` in history."""
        audio: BinaryContent | None = None
        if turn.audio:
            audio = BinaryContent(data=pcm_to_wav(turn.audio, self._input_rate), media_type=_WAV_MEDIA_TYPE)
            if retained:
                audio = self._audio_budget.track(audio, len(turn.audio), output=False)
        part = SpeechPart(speaker='user', transcript=turn.transcript.strip() or None, audio=audio)
        request = ModelRequest(parts=[part])
        fill_run_metadata(request, run_id=self._run_id, conversation_id=self._conversation_id)
        return request

    # --- inputs and obligations --------------------------------------------------------------------

    def _place_input(self, input_id: InputId) -> None:
        if (input_ := self._unplaced.pop(input_id, None)) is not None:
            self._place(input_)

    def _withdraw(self, input_ids: Sequence[InputId]) -> None:
        for input_id in input_ids:
            if (call_id := self._result_call.get(input_id)) is not None:
                # The provider is missing that result, so it won't answer the call's siblings either.
                self._call_unanswered(call_id)
            # Dropped altogether, so what it carried (an evicted image, say) isn't kept alive for nothing.
            if (input_ := self._inputs.pop(input_id, None)) is not None and self._unplaced.pop(input_id, None) is None:
                self._unplace(input_)
        self._settle(input_ids, 'void')

    def _settle(self, input_ids: Sequence[InputId], outcome: _Obligation) -> None:
        for input_id in input_ids:
            if self._obligations.get(input_id) == 'pending':
                self._obligations[input_id] = outcome

    # --- retained audio -------------------------------------------------------------------------------

    def _bound_retained_audio(self) -> None:
        """Evict the oldest retained audio until what the core retains fits `retain_audio_max_seconds`.

        As the session does: audio recorded in history goes first, oldest first, each part keeping its transcript;
        then the audio of turns waiting to be recorded, oldest first; and only then what is still coming in, which
        keeps its most recent audio. A part not recorded yet is only evicted once it is. Only the local copy is
        dropped: the provider received all of it.
        """
        budget = self._audio_budget
        excess = budget.excess(
            untracked_input=self._waiting_turn_audio_bytes + len(self._input_audio),
            untracked_output=self._open_audio_bytes,
        )
        if excess > 0:
            excess = self._evict_recorded_audio(excess)
        while excess > 0 and self._waiting_turn_audio:
            _, turn = self._waiting_turn_audio.popitem(last=False)
            evicted = len(turn.audio or b'')
            self._waiting_turn_audio_bytes -= evicted
            # Empty rather than `None`, so no later cut gives the turn audio again.
            turn.audio = b''
            excess -= budget.weight(evicted, output=False)
        excess = budget.trim(self._input_audio, excess, output=False)
        for response in list(self._open_audio.values()):
            if excess <= 0:
                break
            before = len(response.open_audio)
            excess = budget.trim(response.open_audio, excess, output=True)
            self._open_audio_bytes -= before - len(response.open_audio)
            if not response.open_audio:
                # Emptied: audio that comes in for it later is the newest, so it goes to the back.
                del self._open_audio[response.id]

    def _evict_recorded_audio(self, excess: int) -> int:
        """Strip the audio of recorded messages, oldest first, until `excess` is freed; the excess left."""
        budget = self._audio_budget
        while excess > 0 and self._recorded_audio:
            entry = self._recorded_audio[0]
            if isinstance(entry, _Response):
                assert entry.message is not None
                stripped, freed = budget.strip(entry.message, excess)
                assert isinstance(stripped, ModelResponse)
                entry.message = stripped
            else:
                assert entry.message is not None
                stripped, freed = budget.strip(entry.message, excess)
                assert isinstance(stripped, ModelRequest)
                entry.message = stripped
            excess -= freed
            if not freed or not _has_audio(stripped):
                self._recorded_audio.popleft()
        return excess

    def _place(self, entry: _Entry, *, before: _Response | None = None) -> None:
        """`entry` joins the conversation here, or ahead of the response `before` while it is still under way.

        A response under way holds back everything from it on, so nothing visible moves.
        """
        if before is not None and before.status is None:
            assert not isinstance(entry, _Input)
            entry.placed_at = before.placed_at - 0.5
            # (`before` is under way, so not final: it sits at or past `_final_count`, as does what goes in ahead of it.)
            self._placed.insert(self._placed.index(before), entry)
            return
        if not isinstance(entry, _Input):
            self._placements += 1
            entry.placed_at = self._placements
        self._placed.append(entry)

    def _unplace(self, entry: _Entry) -> None:
        """`entry` leaves the conversation (withdrawn, taken back, or moved)."""
        index = self._placed.index(entry)
        del self._placed[index]
        if index < self._final_count:
            self._final_count -= 1

    def _record_audio(self, entry: _Response | _UserTurn, message: ModelMessage) -> None:
        """`entry` is recorded as `message`: if that has audio, it is evicted in conversation order."""
        if not _has_audio(message):
            return
        queue = self._recorded_audio
        if not queue or queue[-1].placed_at < entry.placed_at:
            queue.append(entry)
        else:
            # Recorded after something that joined the conversation later (a turn whose transcript came late).
            insort(queue, entry, key=lambda recorded: recorded.placed_at)

    def _refuse_response(self, response_id: str) -> None:
        response = self._responses[response_id]
        if response.message is None:
            response.refused = True
        elif not response.tool_calls:
            # Over within the frame it started in: recorded already, so taken back. (Its usage counted toward the
            # session as it came; one that made tool calls stays, as its calls are refused one by one.)
            response.refused = True
            response.message = None
            self._unplace(response)
            for item_id in [item_id for item_id, (recorded, _) in self._recorded_items.items() if recorded is response]:
                del self._recorded_items[item_id]
            if response in self._recorded_audio:
                self._recorded_audio.remove(response)
            if self._responses_are_requests:
                self.usage.requests -= 1  # usage-attribution: the session owns its spans
                self.requests -= 1

    def _lose_open_responses(self) -> None:
        for response in list(self._open_responses.values()):
            self._end_response(ResponseEnded(response_id=response.id, status='lost'))

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._lose_open_responses()
        for turn in self._turns.values():
            if turn.message is None:
                if not turn.ended:
                    turn.ended = True
                    self._place(turn)
                turn.speaking = False
                self._transcribed(turn, failed=False)
        for input_id in list(self._unplaced):
            self._place_input(input_id)
        for input_id, outcome in self._obligations.items():
            if outcome == 'pending':
                self._obligations[input_id] = 'void'
        self._settled_calls.update(self._call_response)

    def set_conversation_id(self, conversation_id: str) -> None:
        """The conversation id the session resolved late: what the core records from now on, and has recorded, carries it."""
        self._conversation_id = conversation_id
        recorded: list[ModelMessage] = [*(input_.request for input_ in self._inputs.values()), *self._returns.values()]
        for entry in self._placed:
            if not isinstance(entry, _Input) and entry.message is not None:
                recorded.append(entry.message)
        for message in recorded:
            message.conversation_id = message.conversation_id or conversation_id

    # --- what the session reads ---------------------------------------------------------------------

    def all_messages(self) -> list[ModelMessage]:
        """The seeded history plus everything recorded this session, as far as it is final."""
        return [*self._seeded, *self._visible()]

    def new_messages(self) -> list[ModelMessage]:
        return list(self._visible())

    def _visible(self) -> Iterator[ModelMessage]:
        for entry in self._placed[: self._held_from()]:
            if isinstance(entry, _Response):
                if entry.message is not None:
                    yield entry.message
                    yield from self._returns_of(entry)
            elif isinstance(entry, _UserTurn):
                assert entry.message is not None
                yield entry.message
            else:
                yield entry.request

    def replayable_messages(self) -> list[ModelMessage]:
        """The provider's conversation as far as it went, for a reconnect to replay on a new connection.

        Unlike `all_messages()`, this holds nothing back: a spoken turn still waiting for its transcript is in it
        with what it has, and so is everything after it. A response under way isn't: the old connection takes it
        along. Inputs sent that the provider hadn't acknowledged yet come last, as sent: the old connection may
        never have delivered them, and the reply the reconnect asks for again answers them.
        """
        messages: list[ModelMessage] = list(self._seeded)
        for entry in self._placed:
            messages.extend(self._entry_messages(entry))
        messages.extend(input_.request for input_ in self._unplaced.values())
        if not self._pruned_calls:
            return messages
        # Tool calls the connection deleted from the provider's conversation stay deleted on the new one.
        return [kept for message in messages if (kept := self._without_pruned_calls(message)) is not None]

    def _without_pruned_calls(self, message: ModelMessage) -> ModelMessage | None:
        """`message` without the pruned tool calls it holds, or `None` if that leaves nothing.

        A pruned call's result goes whole, with the content that followed it (an image the tool returned, say),
        which the connection deleted with it.
        """
        pruned = self._pruned_calls

        def is_pruned(part: ModelRequestPart | ModelResponsePart) -> bool:
            return isinstance(part, (ToolCallPart, ToolReturnPart, RetryPromptPart)) and part.tool_call_id in pruned

        if not any(is_pruned(part) for part in message.parts):
            return message
        if isinstance(message, ModelRequest):
            return None
        parts = [part for part in message.parts if not is_pruned(part)]
        return replace(message, parts=parts) if parts else None

    def _held_from(self) -> int:
        """Where history is held back: the first entity not final yet (a response under way, a turn untranscribed)."""
        placed = self._placed
        while self._final_count < len(placed):
            entry = placed[self._final_count]
            if (isinstance(entry, _Response) and entry.status is None) or (
                isinstance(entry, _UserTurn) and entry.message is None
            ):
                break
            self._final_count += 1
        return self._final_count

    def transcript_holding_history(self) -> str | None:
        """The spoken turn waiting for its transcript that holds back a message recorded after it, if one does."""
        held_from = self._held_from()
        if held_from == len(self._placed):
            return None
        turn = self._placed[held_from]
        if not isinstance(turn, _UserTurn) or turn.speaking:
            # A response under way ends on its own, and a turn still spoken with the speech.
            return None
        later_final = any(
            (isinstance(entry, _Response) and entry.message is not None)
            or (isinstance(entry, _UserTurn) and entry.message is not None)
            or isinstance(entry, _Input)
            for entry in self._placed[held_from + 1 :]
        )
        return turn.id if later_final else None

    def response_holding_turns(self) -> str | None:
        """The response under way that holds back a user turn recorded after it, if one does.

        A turn the user spoke while the model answers follows that answer, once it ends. One that never does (the
        provider stalls) mustn't keep the turn out of history for good: see `ResponseOverdue`.
        """
        held_from = self._held_from()
        if held_from == len(self._placed):
            return None
        response = self._placed[held_from]
        if not isinstance(response, _Response):
            return None
        if any(isinstance(entry, _UserTurn) and entry.message is not None for entry in self._placed[held_from + 1 :]):
            return response.id
        return None

    def _release_held_turns(self, response_id: str) -> None:
        response = self._responses.get(response_id)
        if response is None or response.status is not None:
            return  # (it ended meanwhile, and the turns after it with it)
        index = self._placed.index(response)
        held = [
            entry for entry in self._placed[index + 1 :] if isinstance(entry, _UserTurn) and entry.message is not None
        ]
        for turn in held:
            self._unplace(turn)
            self._place(turn, before=response)

    def _returns_of(self, response: _Response) -> Iterator[ModelRequest]:
        for call_id in response.tool_calls:
            if (request := self._returns.get(call_id)) is not None:
                yield request

    def take_effects(self) -> list[Effect]:
        """What the `apply()` calls since the last take did that the session acts on, in order.

        Part events stream as they are, with session-unique indices; the rest open and close `chat` spans and end
        exchanges.
        """
        effects, self._effects = self._effects, []
        return effects

    def part_item(self, index: int) -> str | None:
        """The output item a recent assistant part streamed (by stream index), if the provider named it."""
        return self._part_items.get(index)

    def _history_before(self, response: _Response) -> list[ModelMessage]:
        """The conversation ahead of a response, as it stands: what it replied to.

        A spoken turn still waiting for its transcript is in it with what it has.
        """
        messages: list[ModelMessage] = list(self._seeded)
        for entry in self._placed[: self._placed.index(response)]:
            messages.extend(self._entry_messages(entry))
        return messages

    def _entry_messages(self, entry: _Entry) -> Iterator[ModelMessage]:
        if isinstance(entry, _Response):
            if entry.message is not None:
                yield entry.message
                yield from self._returns_of(entry)
        elif isinstance(entry, _UserTurn):
            yield entry.message or self._turn_request(entry)
        else:
            yield entry.request

    def conversation_so_far(self) -> list[ModelMessage]:
        """The conversation as far as it went, for a tool's `RunContext.messages`.

        Like `replayable_messages()`, with what a response under way has said and called so far: a tool runs while
        the response that called it can still be under way.
        """
        messages: list[ModelMessage] = list(self._seeded)
        for entry in self._placed:
            if isinstance(entry, _Response) and entry.status is None:
                parts = [*entry.native_parts, *entry.parts]
                if entry.open_part is not None:
                    parts.append(entry.open_part)
                message = ModelResponse(
                    parts=parts,
                    model_name=entry.model_name,
                    provider_name=self._provider_name,
                    provider_url=self._provider_url,
                    provider_response_id=entry.id if entry.provider_id else entry.reported_id,
                )
                fill_run_metadata(message, run_id=self._run_id, conversation_id=self._conversation_id)
                messages.append(message)
                messages.extend(self._returns_of(entry))
            else:
                messages.extend(self._entry_messages(entry))
        messages.extend(input_.request for input_ in self._unplaced.values())
        return messages

    def speaking_part(self) -> int | None:
        """The stream index of the part the latest response under way is streaming, if it is streaming one."""
        response = self._speaking_response()
        return response.open_part_index if response is not None and response.open_part is not None else None

    def response_under_way(self) -> bool:
        """Whether a response is under way that the session didn't refuse."""
        return any(not response.refused for response in self._open_responses.values())

    def call_waiting(self, call_id: str) -> bool:
        """Whether a tool call is waiting for the session to take it to run (see `take_call`)."""
        return call_id in self._undispatched

    def recorded_response(self, response_id: str) -> tuple[ModelResponse | None, list[ModelMessage]]:
        """What history records for a response that ended (if anything), and the conversation ahead of it."""
        response = self._responses[response_id]
        return response.message, self._history_before(response)

    def take_call(self, call_id: str) -> bool:
        """Whether the session is to run this tool call: the core accepted it, and nothing took it to run yet.

        A call the core didn't accept (a resumed conversation's replay of it, a repeat, one made outside a
        response) isn't run, as it isn't recorded.
        """
        if call_id not in self._undispatched:
            return False
        self._undispatched.remove(call_id)
        return True

    def requests_ahead(self) -> int:
        """The requests still to come that are under way or asked for, which `usage.requests` doesn't count yet.

        Each response under way that the session didn't refuse is one; so is each reply still owed for inputs that
        asked for one. The tool results of one response get a single reply on a connection that answers them
        together, and none once one of that response's calls settled without its result going out.
        """
        under_way = sum(1 for response in self._open_responses.values() if not response.refused)
        replies = {self._reply_for(input_id) for input_id, outcome in self._obligations.items() if outcome == 'pending'}
        return under_way + len(replies)

    def result_asks_reply(self, call_id: str) -> bool:
        """Whether sending this call's result asks for a reply `requests_ahead()` doesn't count yet."""
        if self._result_unanswered(call_id):
            return False
        if not self._answers_per_response or (response := self._call_response.get(call_id)) is None:
            return True
        return not any(
            outcome == 'pending' and self._reply_for(input_id) == response.id
            for input_id, outcome in self._obligations.items()
        )

    def _reply_for(self, input_id: InputId) -> InputId | str:
        """What answers an input that asked for a reply: its own reply, or the one for its calling response's results."""
        if (
            not self._answers_per_response
            or (call_id := self._result_call.get(input_id)) is None
            or (response := self._call_response.get(call_id)) is None
        ):
            return input_id
        return response.id

    def response_unasked(self, response_id: str) -> bool:
        """Whether a response answers nothing the session asked for (server-side voice activity detection started it)."""
        return not self._responses[response_id].answers

    def reply_outstanding(self) -> bool:
        """Whether anything the model owes is still to come: what `wait_for_reply()` waits for, taken now."""
        return bool(self.wait_tokens())

    def wait_tokens(self) -> frozenset[Owed]:
        """What the model owes right now, as tokens a waiter follows until they all resolve (see `still_owed`)."""
        epoch = self._epoch
        tokens = {Owed('input', str(input_id), epoch) for input_id, o in self._obligations.items() if o == 'pending'}
        tokens |= {Owed('response', response_id, epoch) for response_id in self._open_responses}
        tokens |= {
            Owed('call', call_id, epoch)
            for call_id, response in self._call_response.items()
            if call_id not in self._settled_calls and response.status in (None, 'completed')
        }
        owed = self.still_owed(frozenset(tokens))
        # An input a response under way (or the tool round it started, while a call or the input carrying its result
        # is owed) answers is followed from the input itself, so a waiter still waits for it if the response turns out
        # not to answer it after all (`AnswersRevised`).
        calls = [token.key for token in owed if token.kind == 'call'] + [
            call_id
            for token in owed
            if token.kind == 'input' and (call_id := self._result_call.get(int(token.key))) is not None
        ]
        revisable = {token.key for token in owed if token.kind == 'response'} | {
            response.id for call_id in calls if (response := self._call_response.get(call_id)) is not None
        }
        return owed | {
            Owed('input', str(input_id), epoch)
            for response_id in revisable
            for input_id in self._responses[response_id].answers
            if self._answered_by.get(input_id) == response_id
        }

    def still_owed(self, tokens: frozenset[Owed]) -> frozenset[Owed]:
        """Follow each token to what it led to, keeping only what is still to come.

        An answered input leads to the response that answers it; a finished response to its tool calls, which the
        model answers once it has their results, and to the response that carried it on (see
        `ResponseStarted.continues`); a call whose result went out to that result's input. A finished
        response can still be held back from history (see `transcript_holding_history`): that is not waited for.
        Once nothing more is read, nothing more can come, and what was owed when the exchange was abandoned
        (see `ExchangeAbandoned`) is owed no longer.
        """
        if not self._receiving or self._closed:
            return frozenset()
        owed: set[Owed] = set()
        pending = [
            token for token in tokens if token.epoch == self._epoch and (token.kind, token.key) not in self._abandoned
        ]
        seen: set[Owed] = set()
        while pending:
            token = pending.pop()
            if token in seen:
                continue
            seen.add(token)
            if token.kind == 'input':
                input_id = int(token.key)
                outcome = self._obligations.get(input_id, 'void')
                if outcome == 'pending':
                    owed.add(token)
                elif outcome == 'answered':
                    pending.append(replace(token, kind='response', key=self._answered_by[input_id]))
            elif token.kind == 'response':
                response = self._responses[token.key]
                if response.status is None:
                    owed.add(token)
                elif response.status == 'completed':
                    pending.extend(replace(token, kind='call', key=call_id) for call_id in response.tool_calls)
                    if (continuation := self._continued_by.get(token.key)) is not None:
                        pending.append(replace(token, key=continuation))
            elif (result := self._result_input.get(token.key)) is not None:
                pending.append(replace(token, kind='input', key=str(result)))
            elif token.key not in self._settled_calls:
                owed.add(token)
        return frozenset(owed)
