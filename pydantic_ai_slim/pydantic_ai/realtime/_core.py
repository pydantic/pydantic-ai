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

History is projected from those entities, never edited: a message is built once, when its entity is
final, and entities are ordered by where they joined the provider's conversation. A message is visible
in `all_messages()` once every entity placed before it is final, so a response still streaming holds back
what came after it, and nothing already visible ever moves. The one exception is a tool's result, which
sits directly after the response that called it, as request-response APIs require.
"""

from __future__ import annotations as _annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, TypeAlias

from typing_extensions import assert_never

from .._genai_prices import fill_response_cost
from .._utils import fill_run_metadata
from ..messages import (
    BinaryContent,
    FinishReason,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
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
    SpeechPart,
    TextPart,
    ToolCallPart,
)
from ..usage import RequestUsage, RunUsage
from ._lifecycle import (
    InputAdded,
    InputId,
    InputLost,
    LifecycleEvent,
    ResponseEnded,
    ResponseRequestRefused,
    ResponseStarted,
    ResponseStatus,
    UserTurnDiscarded,
    UserTurnEnded,
    UserTurnStarted,
)
from ._utils import accumulate_transcript, pcm_to_wav, user_transcript_update
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
    """Input audio the caller streamed, retained for the spoken turn it belongs to."""

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
    """The session refused a tool call the model made, before running it (a usage limit tripped on it).

    It leaves the call out of history, as it never ran.
    """

    tool_call_id: str


@dataclass(frozen=True, kw_only=True)
class Interrupted:
    """The caller interrupted the response being spoken, having heard `played_ms` of it."""

    played_ms: int | None


@dataclass(frozen=True, kw_only=True)
class ExchangeAbandoned:
    """The session hit a failure that stops the model getting what it needs (a tool raised, a limit tripped).

    Nothing owed until now is waited for any more; what is asked for afterwards is.
    """


@dataclass(frozen=True, kw_only=True)
class ReceiveEnded:
    """The session stopped reading the connection: nothing more will be said, so nothing is owed any more."""


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
    | ReceiveEnded
    | Closed
)
CoreInput: TypeAlias = RealtimeCodecEvent | LifecycleEvent | Command


# --- entities --------------------------------------------------------------------------------------


@dataclass(eq=False)
class _Response:
    id: str
    answers: tuple[InputId, ...]
    model_name: str | None
    """The model serving the session when the response started, which it is priced as."""
    parts: list[ModelResponsePart] = field(default_factory=list[ModelResponsePart])
    open_part: SpeechPart | TextPart | None = None
    open_part_item: str | None = None
    open_transcript: str = ''
    open_audio: bytearray = field(default_factory=bytearray)
    usage: RequestUsage = field(default_factory=RequestUsage)
    usage_details: dict[str, Any] | None = None
    usage_finish_reason: FinishReason | None = None
    interrupted_at_ms: int | None = None
    status: ResponseStatus | None = None
    """How it ended; `None` while it is open."""
    message: ModelResponse | None = None
    """What history records for it, built once it ended (and `None` if it ended with nothing to record)."""
    tool_calls: list[str] = field(default_factory=list[str])


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


@dataclass(eq=False)
class _Input:
    id: InputId
    request: ModelRequest
    withdrawn: bool = False


_Entry: TypeAlias = _Response | _UserTurn | _Input

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
        input_transcription_enabled: bool = True,
        retain_input_audio: bool = False,
        retain_output_audio: bool = False,
        input_sample_rate: int = 24000,
        output_sample_rate: int = 24000,
        seeded: Sequence[ModelMessage] = (),
    ) -> None:
        self._model_name = model_name
        self._provider_name = provider_name
        self._provider_url = provider_url
        self._conversation_id = conversation_id
        self._run_id = run_id
        self._responses_are_requests = responses_are_requests
        self._transcription = input_transcription_enabled
        self._retain_input = retain_input_audio
        self._retain_output = retain_output_audio
        self._input_rate = input_sample_rate
        self._output_rate = output_sample_rate
        self._seeded = list(seeded)
        self.usage = RunUsage()
        """Token usage and requests, as the provider reported them for this session."""

        self._placed: list[_Entry] = []
        """Entities in the order they joined the provider's conversation."""
        self._responses: dict[str, _Response] = {}
        self._turns: dict[str, _UserTurn] = {}
        self._inputs: dict[InputId, _Input] = {}
        """The inputs history records (not audio chunks, say, or a bare request for a response), by id."""
        self._unplaced: dict[InputId, _Input] = {}
        """Those not in the conversation yet, in the order they were sent."""
        self._obligations: dict[InputId, _Obligation] = {}
        self._answered_by: dict[InputId, str] = {}
        self._call_response: dict[str, _Response] = {}
        """The response each tool call came from."""
        self._returns: dict[str, ModelRequest] = {}
        self._result_input: dict[str, InputId] = {}
        """The input that carried each tool call's result to the provider."""
        self._settled_calls: set[str] = set()
        """Calls that settled: their result is recorded (sent or not), or the provider cancelled them."""
        self._input_audio = bytearray()
        self._user_speaking = False
        self._speaking_item: str | None = None
        self._speech_segmented = False
        """Whether the provider draws speech-end boundaries, so the retained audio is cut into turns at them."""
        self._closed = False
        self._receiving = True
        self._epoch = 0
        self._abandoned: set[tuple[str, str]] = set()
        """What was owed when an exchange was abandoned: no waiter waits for it again."""

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
        elif isinstance(item, SessionUsage):
            self._usage(item)
        elif isinstance(item, UserTurnStarted):
            self._turn(item.turn_id)
        elif isinstance(item, UserTurnEnded):
            self._end_turn(item.turn_id)
        elif isinstance(item, UserTurnDiscarded):
            self._discard_turn(item.turn_id)
        elif isinstance(item, InputTranscript):
            self._input_transcript(item)
        elif isinstance(item, RealtimeInputTranscriptionErrorEvent):
            if item.item_id is not None and (turn := self._turns.get(item.item_id)) is not None:
                self._transcribed(turn, failed=True)
        elif isinstance(item, RealtimeInputSpeechStartEvent):
            self._user_speaking = True
            self._speaking_item = item.item_id
        elif isinstance(item, RealtimeInputSpeechEndEvent):
            self._speech_ended(item.item_id)
        elif isinstance(item, InputAdded):
            self._place_input(item.input_id)
        elif isinstance(item, (ResponseRequestRefused, InputLost)):
            self._settle(item.input_ids, 'refused' if isinstance(item, ResponseRequestRefused) else 'lost')
        elif isinstance(item, InputRejected):
            if item.refused == 'content':
                self._withdraw((item.input_index,))
        elif isinstance(item, ToolCallCancelled):
            self._settled_calls.update(item.tool_call_ids)
        elif isinstance(
            item,
            (
                ResponseDone,  # the lifecycle's `ResponseEnded` says the same, once per response
                RealtimeOutputSpeechStartEvent,
                RealtimeOutputSpeechEndEvent,
                RealtimeResponseInterruptedEvent,
                RealtimeSessionReconnectEvent,
                RealtimeSessionErrorEvent,
                ConversationCreated,
                ConversationItemCreated,
                PartStartEvent,
                PartEndEvent,
            ),
        ):
            pass
        else:
            self._command(item)

    def _command(self, command: Command) -> None:
        if isinstance(command, InputSent):
            self._input_sent(command)
        elif isinstance(command, InputWithdrawn):
            self._withdraw(command.input_ids)
        elif isinstance(command, AudioSent):
            if self._retain_input:
                self._input_audio.extend(command.data)
        elif isinstance(command, AudioCleared):
            self._input_audio.clear()
        elif isinstance(command, ToolReturned):
            self._returns[command.tool_call_id] = command.request
            self._settled_calls.add(command.tool_call_id)
        elif isinstance(command, ToolCallRefused):
            self._refuse_call(command.tool_call_id)
        elif isinstance(command, Interrupted):
            if (response := self._speaking_response()) is not None:
                response.interrupted_at_ms = command.played_ms
        elif isinstance(command, ExchangeAbandoned):
            self._abandoned.update((token.kind, token.key) for token in self.wait_tokens())
            self._epoch += 1
        elif isinstance(command, ReceiveEnded):
            self._receive_ended()
        elif isinstance(command, Closed):
            self._close()
        else:
            assert_never(command)

    def _input_sent(self, command: InputSent) -> None:
        if command.request is not None:
            self._inputs[command.input_id] = self._unplaced[command.input_id] = _Input(
                command.input_id, command.request
            )
        if command.solicits:
            self._obligations[command.input_id] = 'pending'
        if command.tool_call_id is not None:
            self._result_input[command.tool_call_id] = command.input_id

    def _receive_ended(self) -> None:
        self._receiving = False
        for turn in self._turns.values():
            if turn.ended:
                # Nothing more of a turn that joined will be read: not the rest of it, nor its transcript.
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
        response = _Response(event.response_id, event.answers, self._model_name())
        self._responses[event.response_id] = response
        self._placed.append(response)
        for input_id in event.answers:
            if self._obligations.get(input_id) == 'pending':
                self._obligations[input_id] = 'answered'
                self._answered_by[input_id] = event.response_id

    def _response_content(self, event: AudioDelta | OutputTranscript) -> None:
        response = self._open_response(event.response_id)
        if response is None:
            return
        output_text = isinstance(event, OutputTranscript) and event.output_text
        part = self._open_part(response, output_text=output_text, item_id=event.item_id)
        if isinstance(event, AudioDelta):
            if self._retain_output:
                response.open_audio.extend(event.data)
            return
        response.open_transcript, _ = accumulate_transcript(response.open_transcript, event.text)
        response.open_part = (
            replace(part, content=response.open_transcript)
            if isinstance(part, TextPart)
            else replace(part, transcript=response.open_transcript)
        )

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
        if part is None:
            part = TextPart(content='') if output_text else SpeechPart(speaker='assistant', transcript='')
            response.open_part = part
            response.open_part_item = item_id
            response.open_transcript = ''
        return part

    def _close_part(self, response: _Response) -> None:
        part = response.open_part
        if part is None:
            return
        if isinstance(part, SpeechPart):
            if part.transcript == '':
                part = replace(part, transcript=None)
            if self._retain_output and response.open_audio:
                wav = pcm_to_wav(bytes(response.open_audio), self._output_rate)
                part = replace(part, audio=BinaryContent(data=wav, media_type=_WAV_MEDIA_TYPE))
        response.parts.append(part)
        response.open_part = None
        response.open_part_item = None
        response.open_transcript = ''
        response.open_audio.clear()

    def _tool_call(self, event: ToolCall) -> None:
        response = self._open_response(event.response_id)
        if response is None:
            return
        self._close_part(response)
        response.parts.append(ToolCallPart(tool_name=event.tool_name, args=event.args, tool_call_id=event.tool_call_id))
        response.tool_calls.append(event.tool_call_id)
        self._call_response[event.tool_call_id] = response

    def _refuse_call(self, call_id: str) -> None:
        if (response := self._call_response.get(call_id)) is None or response.status is not None:
            return
        del self._call_response[call_id]
        response.tool_calls.remove(call_id)
        response.parts = [
            part for part in response.parts if not (isinstance(part, ToolCallPart) and part.tool_call_id == call_id)
        ]

    def _usage(self, event: SessionUsage) -> None:
        self.usage.incr(event.usage)  # usage-attribution: the core's own tally
        if not event.response_scoped:
            return
        if not self._responses_are_requests:
            self.usage.requests += 1  # usage-attribution: the core's own tally
        response = self._responses.get(event.provider_response_id) if event.provider_response_id else None
        if response is None or response.status is not None:
            # Usage for a response already recorded counts toward the session, never toward its message.
            return
        response.usage = response.usage + event.usage
        response.usage_finish_reason = event.finish_reason or response.usage_finish_reason
        if event.provider_details:
            response.usage_details = {**(response.usage_details or {}), **event.provider_details}

    def _end_response(self, event: ResponseEnded) -> None:
        response = self._responses[event.response_id]
        self._close_part(response)
        response.status = event.status
        answered_calls_only = bool(response.parts) and all(isinstance(part, ToolCallPart) for part in response.parts)
        if event.status != 'lost' and not (event.status == 'completed' and answered_calls_only):
            # A reply is over, and nobody is speaking: audio sent since the last speech boundary is the tail
            # of the last turn, trailing its commit, not the start of the next one.
            if self._speech_segmented and not self._user_speaking:
                self._input_audio.clear()
        interrupted = event.status in ('cancelled', 'lost')
        if event.status == 'lost' and not response.parts and response.usage == RequestUsage():
            # Cut off before it said anything or was billed: there is nothing to record.
            return
        parts = list(response.parts)
        if interrupted:
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
            provider_response_id=response.id,
            finish_reason=finish_reason,
            conversation_id=self._conversation_id,
            state='interrupted' if interrupted else 'complete',
        )
        fill_run_metadata(message, run_id=self._run_id, conversation_id=self._conversation_id)
        provider_cost = message.usage.cost
        fill_response_cost(message)
        if provider_cost is None:
            self.usage.incr(RequestUsage(cost=message.usage.cost))  # usage-attribution: the core's own tally
        self.usage.requests += int(self._responses_are_requests)  # usage-attribution: the core's own tally
        response.message = message

    def _open_response(self, response_id: str | None) -> _Response | None:
        """The open response content names; content naming none (Azure Voice Live's text) goes to the only one open."""
        if response_id is not None:
            response = self._responses.get(response_id)
            return response if response is not None and response.status is None else None
        open_responses = [response for response in self._responses.values() if response.status is None]
        return open_responses[0] if len(open_responses) == 1 else None

    def _speaking_response(self) -> _Response | None:
        open_responses = [response for response in self._responses.values() if response.status is None]
        return open_responses[-1] if open_responses else None

    # --- spoken turns -------------------------------------------------------------------------------

    def _turn(self, turn_id: str) -> _UserTurn:
        if (turn := self._turns.get(turn_id)) is None:
            turn = self._turns[turn_id] = _UserTurn(turn_id)
        return turn

    def _speech_ended(self, item_id: str | None) -> None:
        self._user_speaking = False
        self._speech_segmented = True
        if item_id is None or (turn := self._turns.get(item_id)) is None:
            return
        if self._retain_input and self._input_audio:
            if turn.audio is None:
                turn.audio = bytes(self._input_audio)
                self._input_audio.clear()
        if turn.speaking:
            turn.speaking = False
            self._turn_said(turn)

    def _end_turn(self, turn_id: str) -> None:
        turn = self._turn(turn_id)
        if turn.ended:
            return
        turn.ended = True
        self._placed.append(turn)
        if self._user_speaking and turn_id == self._speaking_item:
            turn.speaking = True
            return
        if turn.audio is None and self._retain_input and self._input_audio:
            # Committed by hand, with no speech end to cut it at: the whole buffer is this turn's.
            turn.audio = bytes(self._input_audio)
            self._input_audio.clear()
        self._turn_said(turn)

    def _turn_said(self, turn: _UserTurn) -> None:
        """All of a joined turn's audio is in: it is final once transcribed, or at once if nothing transcribes it."""
        if not self._transcription:
            self._transcribed(turn, failed=True)
        elif turn.transcribed:
            self._build_turn(turn)

    def _discard_turn(self, turn_id: str) -> None:
        if (turn := self._turns.get(turn_id)) is None:
            return
        if turn.ended:
            # It joined the conversation, but no more of it is coming: it keeps what transcript it has.
            turn.speaking = False
            self._transcribed(turn, failed=False)
        else:
            del self._turns[turn_id]

    def _input_transcript(self, event: InputTranscript) -> None:
        # Only a spoken turn the connection reported transcribes into history: not, say, the empty audio item
        # an idle timeout commits to nudge the model.
        turn = self._turns.get(event.item_id) if event.item_id is not None else None
        if turn is None or turn.message is not None:
            return
        turn.transcript, _ = user_transcript_update(turn.transcript, event.text, cumulative=event.cumulative)
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
        audio = (
            BinaryContent(data=pcm_to_wav(turn.audio, self._input_rate), media_type=_WAV_MEDIA_TYPE)
            if turn.audio
            else None
        )
        part = SpeechPart(speaker='user', transcript=turn.transcript.strip() or None, audio=audio)
        request = ModelRequest(parts=[part])
        fill_run_metadata(request, run_id=self._run_id, conversation_id=self._conversation_id)
        turn.message = request

    # --- inputs and obligations --------------------------------------------------------------------

    def _place_input(self, input_id: InputId) -> None:
        if (input_ := self._unplaced.pop(input_id, None)) is not None:
            self._placed.append(input_)

    def _withdraw(self, input_ids: Sequence[InputId]) -> None:
        for input_id in input_ids:
            if (input_ := self._inputs.get(input_id)) is not None:
                input_.withdrawn = True
        self._settle(input_ids, 'void')

    def _settle(self, input_ids: Sequence[InputId], outcome: _Obligation) -> None:
        for input_id in input_ids:
            if self._obligations.get(input_id) == 'pending':
                self._obligations[input_id] = outcome

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for response in [response for response in self._responses.values() if response.status is None]:
            self._end_response(ResponseEnded(response_id=response.id, status='lost'))
        for turn in self._turns.values():
            if turn.message is None:
                if not turn.ended:
                    turn.ended = True
                    self._placed.append(turn)
                turn.speaking = False
                self._transcribed(turn, failed=False)
        for input_id in list(self._unplaced):
            self._place_input(input_id)
        for input_id, outcome in self._obligations.items():
            if outcome == 'pending':
                self._obligations[input_id] = 'void'
        self._settled_calls.update(self._call_response)

    # --- what the session reads ---------------------------------------------------------------------

    def all_messages(self) -> list[ModelMessage]:
        """The seeded history plus everything recorded this session, as far as it is final."""
        return [*self._seeded, *self._visible()]

    def new_messages(self) -> list[ModelMessage]:
        return list(self._visible())

    def _visible(self) -> Iterator[ModelMessage]:
        for entry in self._placed:
            if isinstance(entry, _Response):
                if entry.status is None:
                    return
                if entry.message is not None:
                    yield entry.message
                    yield from self._returns_of(entry)
            elif isinstance(entry, _UserTurn):
                if entry.message is None:
                    return
                yield entry.message
            elif not entry.withdrawn:
                yield entry.request

    def _returns_of(self, response: _Response) -> Iterator[ModelRequest]:
        for call_id in response.tool_calls:
            if (request := self._returns.get(call_id)) is not None:
                yield request

    def reply_outstanding(self) -> bool:
        """Whether anything the model owes is still to come: what `wait_for_reply()` waits for, taken now."""
        return bool(self.wait_tokens())

    def wait_tokens(self) -> frozenset[Owed]:
        """What the model owes right now, as tokens a waiter follows until they all resolve (see `still_owed`)."""
        epoch = self._epoch
        tokens = {Owed('input', str(input_id), epoch) for input_id, o in self._obligations.items() if o == 'pending'}
        tokens |= {
            Owed('response', response_id, epoch) for response_id, r in self._responses.items() if r.status is None
        }
        tokens |= {
            Owed('call', call_id, epoch)
            for call_id, response in self._call_response.items()
            if call_id not in self._settled_calls and response.status in (None, 'completed')
        }
        return self.still_owed(frozenset(tokens))

    def still_owed(self, tokens: frozenset[Owed]) -> frozenset[Owed]:
        """Follow each token to what it led to, keeping only what is still to come.

        An answered input leads to the response that answers it; a finished response to its tool calls, which
        the model answers once it has their results; a call whose result went out to that result's input.
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
            elif (result := self._result_input.get(token.key)) is not None:
                pending.append(replace(token, kind='input', key=str(result)))
            elif token.key not in self._settled_calls:
                owed.add(token)
        return frozenset(owed)
