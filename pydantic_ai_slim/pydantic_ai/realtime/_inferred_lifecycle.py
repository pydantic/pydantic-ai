"""`InferredLifecycle`: lifecycle events worked out for a protocol that identifies no responses, turns, or inputs."""

from __future__ import annotations as _annotations

from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, ClassVar

from ..messages import (
    BinaryContent,
    FinishReason,
    PartStartEvent,
    RealtimeInputSpeechEndEvent,
    RealtimeInputSpeechStartEvent,
    RealtimeInputTranscriptionErrorEvent,
    RealtimeResponseInterruptedEvent,
)
from ._lifecycle import (
    AnswersRevised,
    InputAdded,
    InputId,
    InputLost,
    LifecycleEvent,
    RealtimeFrame,
    ResponseEnded,
    ResponseRequestRefused,
    ResponseStarted,
    ResponseStatus,
    UserTurnClosed,
    UserTurnEnded,
    UserTurnStarted,
)
from .codec import (
    AudioDelta,
    CommitAudio,
    ConversationItemCreated,
    CreateResponse,
    InputRejected,
    InputTranscript,
    OutputTranscript,
    RealtimeCodecEvent,
    RealtimeInput,
    ResponseDone,
    SessionUsage,
    ToolCall,
    ToolCallCancelled,
    ToolResult,
)


@dataclass
class _Turn:
    """A spoken turn, from its first transcript until its transcript is whole (marked finished, or no more comes)."""

    id: str
    identified: bool = False
    """Named by the provider (its transcripts carry an item id), which then says when its transcript is final."""
    joined: bool = False
    words: bool = False
    """Some of its transcript came."""
    replied: bool = False
    """The reply to it is over, before any of its transcript came: what still comes is late, until the next turn."""


@dataclass
class _Uncommit:
    """What a commit changed, to undo should it never go out."""

    input_id: InputId
    pending: int
    audio_since_reply: bool
    unnamed_committed: bool
    spoken_unanswered: str | None
    joined: _Turn | None


class InferredLifecycle:
    """Works out a connection's lifecycle events from the order of what it sends and receives.

    For a protocol that identifies no responses, spoken turns, or inputs, as Gemini Live and OpenAI GPT-Live
    don't. The connection, in its [`send`][pydantic_ai.realtime.codec.RealtimeConnection.send], sends each input
    inside [`sending`][pydantic_ai.realtime.lifecycle.InferredLifecycle.sending] (which reports it, and reports it
    failed if sending it raises) and returns what the send settled
    ([`take_pending`][pydantic_ai.realtime.lifecycle.InferredLifecycle.take_pending]). It
    passes the codec events each provider message makes through
    [`frame_for`][pydantic_ai.realtime.lifecycle.InferredLifecycle.frame_for] to get the frame to yield, and reports a
    dropped connection ([`connection_lost`][pydantic_ai.realtime.lifecycle.InferredLifecycle.connection_lost]). A
    connection whose codec events are an async iterable of their own can have
    [`frames`][pydantic_ai.realtime.lifecycle.InferredLifecycle.frames] do the last two.

    What it works out:

    - A response starts with its first output (audio, a transcript, a native tool part, a tool call), and ends at
      its [`ResponseDone`][pydantic_ai.realtime.codec.ResponseDone]; a response that makes tool calls ends with them,
      or, when they are reported with `response_usage_follows`, at the usage report that follows them. Its id is
      made up, so it is recorded as no provider's (`ResponseStarted.provider_id=False`).
    - A response answers every input that asked for one (a `str` turn, a tool result, a `CreateResponse`) and wasn't
      answered when it started (`ResponseStarted.basis='inferred'`).
    - A spoken turn starts with its first input transcript, joins the conversation as the reply to it starts, and
      has its whole transcript once that reply ends, unless the transcript is marked final before. A turn the
      provider identifies (its transcripts carry an `item_id`) is named by that id, and waits for its final
      transcript. Without input transcription, audio sent since the last reply is a turn of its own, with no
      transcript, when the model next replies to speech.
    - An input joins the conversation when it is sent, unless a reply is under way or owed: it then reached the
      provider while the model was working on that reply, and follows it.
    - A dropped connection loses the response under way and every reply still owed.

    These are the defaults; a provider that departs from them gets a subclass overriding these extension points:

    - [`transcripts_lag_replies`][pydantic_ai.realtime.lifecycle.InferredLifecycle.transcripts_lag_replies]
      (`True`): whether the provider transcribes the user's words after the model started answering them.
    - [`audio_starts_response`][pydantic_ai.realtime.lifecycle.InferredLifecycle.audio_starts_response] (`True`):
      whether output audio is the model speaking, so a response starts with it.
    - [`untranscribed_audio_is_a_turn`][pydantic_ai.realtime.lifecycle.InferredLifecycle.untranscribed_audio_is_a_turn]
      (`True`): whether audio the model replies to before any transcript of it came is a spoken turn of its own.
    - [`inputs_cut_off_replies`][pydantic_ai.realtime.lifecycle.InferredLifecycle.inputs_cut_off_replies]
      (`False`): whether an input that reaches the provider mid-generation cuts the reply off, leaving it to the next.
    - [`takes_turn(codec)`][pydantic_ai.realtime.lifecycle.InferredLifecycle.takes_turn] (`False`): whether a
      provider message has the model take the turn, whatever its output shows.
    """

    transcripts_lag_replies: ClassVar[bool] = True
    """Whether the provider can transcribe the user's words after the model started answering them, as most do.

    A transcript that starts while a reply nobody asked for is under way is then the speech that reply answers,
    rather than speech over it. A subclass for a provider that transcribes as the user speaks (and so reports
    speech over a reply while the reply is still going) sets it to `False`.
    """

    audio_starts_response: ClassVar[bool] = True
    """Whether output audio is the model speaking, so a response starts with it.

    A subclass for a provider whose output is a continuous track, silent between replies, sets it to `False`, and
    says where the model takes the turn with `takes_turn` instead.
    """

    inputs_cut_off_replies: ClassVar[bool] = False
    """Whether an input that reaches the provider while it generates a reply cuts that reply off.

    The next reply then answers the input, so a reply cut off (`ResponseStarted` through a cancelled `ResponseEnded`)
    answers only the oldest input it was taken to answer, as that one started it, or none if it said nothing; and a
    reply whose tool calls the provider cancels answers only the oldest too. That is, unless the provider heard the
    user start speaking after those inputs went out ([`speech_heard()`][pydantic_ai.realtime.lifecycle.InferredLifecycle.speech_heard]):
    then the user may have cut the reply off, as ever. The rest are released
    ([`AnswersRevised.released`][pydantic_ai.realtime.lifecycle.AnswersRevised.released]) to the next reply. A
    subclass for a provider that cuts its reply off this way (Gemini Live's 3.x models do) sets it to `True`.
    """

    untranscribed_audio_is_a_turn: ClassVar[bool] = True
    """Whether audio the model replies to before any transcript of it came is a spoken turn of its own.

    A subclass for a provider that replies to speech only once it has transcribed it (a cascaded pipeline, say) sets
    it to `False`, so audio no transcript names, like a microphone streaming before a greeting, makes no turn.
    Without input transcription (`transcribes=False`), audio is the only sign of speech, and always makes a turn.
    """

    def __init__(self, *, transcribes: bool = True) -> None:
        """Track a connection's lifecycle.

        Args:
            transcribes: Whether the session has the user's speech transcribed: if not, audio alone makes a spoken
                turn. A setting of the session's (input transcription), not of the provider.
        """
        self._transcribes = transcribes
        self._inputs = 0
        self._pending: list[LifecycleEvent] = []
        """Events that come outside any message (an input sent), yielded ahead of the next one."""
        self._responses = 0
        self._open: str | None = None
        """The response under way."""
        self._open_named = False
        """Whether the provider named the response under way, by the `response_id` of its content."""
        self._open_answers: tuple[InputId, ...] = ()
        """The inputs the response under way answers, oldest first."""
        self._open_said = False
        """Whether the response under way said (or called) anything yet."""
        self._open_calls: list[str] = []
        """The tool calls the response under way made."""
        self._calling: dict[str, tuple[str, tuple[InputId, ...]]] = {}
        """The response that ended at each tool call (until the next response starts), and the inputs it answers: its
        reply goes on in the answer to the calls."""
        self._spoken_since = -1
        """The last input sent when the provider heard the user start speaking: only an input sent after it can have
        cut a reply off (see `inputs_cut_off_replies`). (Sending audio says nothing: a microphone streams silence too.)"""
        self._ended_by_usage: str | None = None
        """The response a usage report or its tool call just ended: its terminal, if one follows, ends nothing more."""
        self._cut_off = False
        """Whether the response under way was cut off (`RealtimeResponseInterruptedEvent`): more output is another."""
        self._deferred: ResponseDone | None = None
        """The terminal of a response the model said the exchange continues past (`interaction_status=IN_PROGRESS`):
        it stays open for the tool call it was stalling for, and ends at anything else."""
        self._calls_awaiting_usage = False
        """The open response made tool calls whose (empty) usage report closes it."""
        self._unanswered: list[InputId] = []
        """Inputs that asked for a reply no response has started yet, oldest first."""
        self._held: list[InputId] = []
        """Inputs sent while a reply was under way or owed, which join the conversation after that reply."""
        self._turns = 0
        self._turn: _Turn | None = None
        self._turn_reply: tuple[str, str] | None = None
        """The response replying to the joined turn, whose end closes the turn's transcript."""
        self._unprompted_reply: str | None = None
        """The response under way, if the model started it on its own with no transcript of speech to reply to yet."""
        self._audio_since_reply = False
        """Audio went out since the last reply started: without transcription, the next turn of speech."""
        self._continues: str | None = None
        """A response the model ended saying the exchange goes on, which the next response carries on from."""
        self._unnamed_committed = False
        """A turn joined the conversation, unnamed and without words yet (see `_commit_unnamed`)."""
        self._speaking = False
        """The provider heard the user start speaking, in a turn it didn't name, and not stop yet."""
        self._transcribed_while_speaking = False
        """Words came since the user started speaking: the speech is the turn they transcribe."""
        self._spoken_unanswered: str | None = None
        """A spoken turn that joined the conversation ahead of a reply that hasn't started yet."""
        self._uncommit: _Uncommit | None = None
        """How to take back the latest commit, while its send may still fail."""
        self._identified: set[str] = set()
        """The turns the provider named, by item id."""
        self._replayed_items: set[str] = set()
        self._replayed_calls: set[str] = set()
        """The items and tool calls a resumed conversation replays (`ConversationItemCreated.replayed`)."""

    # --- what the connection sends ----------------------------------------------------------------

    def input_sent(self, content: RealtimeInput) -> InputId:
        """An input is about to go out: note whether it asks for a reply, and where it joins the conversation.

        Call it for every `send()`, before anything in it can fail, as the session numbers its inputs that way.
        Returns the input's id, for `input_failed` and `input_unanswerable`.
        """
        input_id = self._inputs
        self._inputs += 1
        if isinstance(content, BinaryContent) and content.is_audio:
            self._audio_since_reply = True
            return input_id
        if isinstance(content, CommitAudio):
            # The audio sent since the last turn is one now: it joins the conversation as the commit arrives. (Undone
            # if the commit never goes out.)
            turn = self._turn
            self._uncommit = _Uncommit(
                input_id=input_id,
                pending=len(self._pending),
                audio_since_reply=self._audio_since_reply,
                unnamed_committed=self._unnamed_committed,
                spoken_unanswered=self._spoken_unanswered,
                joined=turn if turn is not None and not turn.joined else None,
            )
            self._commit_turn()
            return input_id
        if isinstance(content, CreateResponse):
            # A request for a reply, with nothing of its own to join the conversation.
            self._unanswered.append(input_id)
            return input_id
        self._announce_committed(self._pending)
        if self._reply_owed():
            # It reaches the provider while the model works on a reply, which it follows. (A response that answers
            # it places it ahead of itself, should that come first.)
            self._held.append(input_id)
        else:
            # Nothing is going on: it joins the conversation as it arrives.
            self._pending.append(InputAdded(input_id=input_id))
        if isinstance(content, (str, ToolResult)):
            self._unanswered.append(input_id)
        return input_id

    def _commit_turn(self) -> None:
        turn = self._turn
        if turn is not None and not turn.joined:
            self._audio_since_reply = False
            self._join(turn, self._pending)
        elif self._audio_since_reply:
            self._audio_since_reply = False
            self._commit_unnamed(self._pending)

    def _commit_unnamed(self, events: list[LifecycleEvent]) -> None:
        """A turn joined the conversation that has no name and no words yet.

        It's announced with its first transcript, which may name it, or once anything else joins the conversation.
        """
        self._announce_committed(events)
        self._unnamed_committed = True

    def _announce_committed(self, events: list[LifecycleEvent], item_id: str | None = None) -> _Turn | None:
        if not self._unnamed_committed:
            return None
        self._unnamed_committed = False
        if (turn := self._turn) is not None:
            self._end_turn(turn, events)
        turn = self._start_turn(item_id, events, transcribed=False)
        self._join(turn, events)
        return turn

    def input_failed(self, input_id: InputId) -> None:
        """The input never went out (its send raised): forget it."""
        if (uncommit := self._uncommit) is not None and uncommit.input_id == input_id:
            # A commit that never went out committed nothing.
            self._uncommit = None
            del self._pending[uncommit.pending :]
            self._audio_since_reply = uncommit.audio_since_reply
            self._unnamed_committed = uncommit.unnamed_committed
            self._spoken_unanswered = uncommit.spoken_unanswered
            if uncommit.joined is not None:
                uncommit.joined.joined = False
        self._unanswered = [unanswered for unanswered in self._unanswered if unanswered != input_id]
        self._held = [held for held in self._held if held != input_id]
        self._pending = [event for event in self._pending if event != InputAdded(input_id=input_id)]

    def input_unanswerable(self, input_id: InputId) -> None:
        """An input went (or was meant to go) out, but the provider will never reply to it."""
        if input_id in self._unanswered:
            self._unanswered.remove(input_id)
            # (Held behind the reply under way, it doesn't join the conversation after it either.)
            self._held = [held for held in self._held if held != input_id]
            self._pending.append(InputLost(input_ids=(input_id,)))

    @contextmanager
    def sending(self, content: RealtimeInput) -> Generator[InputId]:
        """Report an input a connection's `send()` is sending, around the sending itself: yields the input's id.

        It is [`input_sent`][pydantic_ai.realtime.lifecycle.InferredLifecycle.input_sent] going in, and
        [`input_failed`][pydantic_ai.realtime.lifecycle.InferredLifecycle.input_failed] if the block raises. The
        connection then returns [`take_pending()`][pydantic_ai.realtime.lifecycle.InferredLifecycle.take_pending]
        (see the example in [Implementing a connection](../realtime.md)).
        """
        input_id = self.input_sent(content)
        try:
            yield input_id
        except BaseException:
            self.input_failed(input_id)
            raise

    def take_pending(self) -> list[LifecycleEvent]:
        """The events the inputs sent since the last frame settled.

        Where each joined the conversation, or that it will go unanswered. A connection returns them from
        [`send()`][pydantic_ai.realtime.codec.RealtimeConnection.send], so the session applies them before anything it
        reads next, and puts them at the head of a frame of its own (like the one it reports a reconnect in).
        [`frame_for()`][pydantic_ai.realtime.lifecycle.InferredLifecycle.frame_for] takes any still waiting into the
        frame it makes.
        """
        pending, self._pending = self._pending, []
        # (A commit whose events are out can't be taken back any more.)
        self._uncommit = None
        return pending

    def _reply_owed(self) -> bool:
        turn = self._turn
        return self._open is not None or bool(self._unanswered) or (turn is not None and not turn.joined)

    # --- what the server says ---------------------------------------------------------------------

    def speech_heard(self, *, after: InputId | None = None) -> None:
        """The provider heard the user start speaking, which may cut a reply off (see `inputs_cut_off_replies`).

        A `RealtimeInputSpeechStartEvent` says so; a connection whose provider reports speech some other way (Gemini
        Live's `voice_activity`) calls this, or every reply cut off is taken for one an input cut off. `after` is the id
        of the last input sent before the user started speaking, when the provider says where in the audio that was:
        `-1` if they started before any input, and by default every input sent so far, as the report may come late.
        """
        self._spoken_since = max(self._spoken_since, self._inputs - 1 if after is None else after)

    def frame_for(self, codec: Sequence[RealtimeCodecEvent]) -> RealtimeFrame:
        """The frame to yield for the codec events one provider message makes, with the lifecycle events around them."""
        turn_taken = self.takes_turn(codec)
        frame: RealtimeFrame = [*self.take_pending()]
        if any(isinstance(event, RealtimeResponseInterruptedEvent) for event in codec):
            # The user cut in, and the message reports what they said ahead of the cut: that is new speech, not
            # more of the turn being answered, whose transcript is over.
            closed: list[LifecycleEvent] = []
            self._unprompted_reply = None
            self._close_turn(closed)
            frame.extend(closed)
        for event in codec:
            before, after = self._event(event, takes_turn=turn_taken)
            frame.extend(before)
            frame.append(event)
            frame.extend(after)
        if turn_taken:
            started: list[LifecycleEvent] = []
            self._ensure_response(started)
            frame.extend(started)
        return frame

    def takes_turn(self, codec: Sequence[RealtimeCodecEvent]) -> bool:
        """Whether the provider message `codec` makes has the model take the turn, whatever its output shows.

        An extension point for a subclass whose connection knows where the model takes the turn better than its
        output shows (see `audio_starts_response`): a response then starts with the message's audio, or after its
        events if it has none, if none is under way. `False` by default.
        """
        return False

    async def frames(self, events: AsyncIterable[RealtimeCodecEvent]) -> AsyncIterator[RealtimeFrame]:
        """Frames for a connection's codec events, one event to a frame, ending with what the end of `events` loses.

        For a connection whose codec events are an async iterable of their own, and end with the connection.
        """
        iterator = aiter(events)
        try:
            async for event in iterator:
                yield self.frame_for([event])
            lost: RealtimeFrame = [*self.connection_lost()]
            yield lost
        finally:
            if isinstance(iterator, AsyncGenerator):
                await iterator.aclose()

    def _event(
        self, event: RealtimeCodecEvent, *, takes_turn: bool = False
    ) -> tuple[list[LifecycleEvent], list[LifecycleEvent]]:
        before: list[LifecycleEvent] = []
        after: list[LifecycleEvent] = []
        if self._replayed(event):
            return before, after
        output = (
            isinstance(event, (PartStartEvent, ToolCall))
            or (isinstance(event, AudioDelta) and (self.audio_starts_response or takes_turn))
            or (isinstance(event, OutputTranscript) and bool(event.text))
        )
        if self._deferred is not None:
            self._settle_deferred(event, output, before)
        if output:
            self._output(event, before, after)
        elif isinstance(event, SessionUsage):
            # Usage reported between responses doesn't start one: it is the next response's (see `_response_done`).
            if event.response_scoped and self._calls_awaiting_usage:
                # A tool-call frame's report: the calls' response is over, and the answer comes once every result
                # is in.
                ended = self._open
                self._end(after, status='completed', finish_reason='tool_call')
                self._ended_by_usage = ended
        elif isinstance(event, (InputTranscript, RealtimeInputTranscriptionErrorEvent)):
            self._input_transcript(event, before)
        elif isinstance(event, RealtimeResponseInterruptedEvent):
            # The user (or a tool result) cut in: what was sent meanwhile follows what the model had said, and so
            # does the speech that cut in, a turn of its own.
            self._cut_off = self._open is not None
            self._close_turn(after)
            self._place_held(after)
        elif isinstance(event, RealtimeInputSpeechStartEvent):
            self._speech_started(event, before)
        elif isinstance(event, RealtimeInputSpeechEndEvent):
            self._speech_ended(event, after)
        elif isinstance(event, ResponseDone):
            self._response_done(event, before, after)
        elif isinstance(event, InputRejected):
            self._input_rejected(event, after)
        elif isinstance(event, ToolCallCancelled):
            self._calls_cancelled(event, before)
        # Everything else (the end of a native part, a cancelled call, a typed turn the connection reports lost to a
        # drop, which `connection_lost` lost already, ...) says nothing about the lifecycle.
        return before, after

    def _settle_deferred(self, event: RealtimeCodecEvent, output: bool, before: list[LifecycleEvent]) -> None:
        """Whether the filler the model said while stalling for a tool call is over, with this event."""
        if isinstance(event, ResponseDone) and event.interrupted:
            # The exchange was cut off before the tool call came (by a drop the connection closes this way, say):
            # the filler was a complete response, and the cut-off rest is one of its own.
            self._end_deferred(before)
        elif isinstance(event, (ToolCall, ResponseDone)):
            # The tool call the model was stalling for belongs to the filler's response, and so does the next
            # boundary (handled below).
            self._deferred = None
        elif output:
            # The stall didn't end in the tool call it was for: the filler was a response of its own, which
            # this one carries on.
            self._end_deferred(before)
        elif isinstance(event, RealtimeResponseInterruptedEvent):
            # The user cut in: the exchange is over, and the filler is ended by the terminal that follows.
            self._deferred = None
        # Anything else (the user's transcript, usage) leaves it held: the exchange isn't over.

    def _output(self, event: RealtimeCodecEvent, before: list[LifecycleEvent], after: list[LifecycleEvent]) -> None:
        """The model's output, of the response under way or of one it starts."""
        self._ended_by_usage = None
        named = event.response_id if isinstance(event, (AudioDelta, OutputTranscript, ToolCall)) else None
        if self._open is not None and self._cut_off:
            # More output after the response under way was cut off: a response of its own.
            self._end(before, status='cancelled')
        elif named is not None and self._open is not None and self._open_named and named != self._open:
            # Content of a response the provider names differently: the one under way is over.
            self._end(before, status='completed')
        self._ensure_response(before, named=named)
        self._open_said = True
        if isinstance(event, ToolCall):
            self._open_calls.append(event.tool_call_id)
            if event.response_usage_follows:
                self._calls_awaiting_usage = True
            elif not self._calls_awaiting_usage:
                # No usage report closes its response: the call does, as the answer comes once its result is in. (A
                # terminal that follows before anything else is said is this response's.)
                ended = self._open
                self._end(after, status='completed', finish_reason='tool_call')
                self._ended_by_usage = ended

    def _input_rejected(self, event: InputRejected, after: list[LifecycleEvent]) -> None:
        """The provider refused an input's content (it never joins the conversation), or its request for a reply."""
        input_id = event.input_index
        if event.refused == 'content':
            self._held = [held for held in self._held if held != input_id]
            self._pending = [pending for pending in self._pending if pending != InputAdded(input_id=input_id)]
        if input_id in self._unanswered:
            self._unanswered.remove(input_id)
            if event.refused == 'response':
                after.append(ResponseRequestRefused(input_ids=(input_id,)))

    def _replayed(self, event: RealtimeCodecEvent) -> bool:
        """Whether `event` is about an item a resumed conversation replays, which says nothing new."""
        if isinstance(event, ConversationItemCreated):
            if event.replayed:
                if event.item_id is not None:
                    self._replayed_items.add(event.item_id)
                if event.tool_call_id is not None:
                    self._replayed_calls.add(event.tool_call_id)
            return True
        if isinstance(event, ToolCall) and event.tool_call_id in self._replayed_calls:
            return True
        if isinstance(
            event,
            (
                AudioDelta,
                OutputTranscript,
                InputTranscript,
                ToolCall,
                RealtimeInputSpeechStartEvent,
                RealtimeInputSpeechEndEvent,
                RealtimeInputTranscriptionErrorEvent,
            ),
        ):
            return event.item_id is not None and event.item_id in self._replayed_items
        return False

    def _response_done(self, event: ResponseDone, before: list[LifecycleEvent], after: list[LifecycleEvent]) -> None:
        if self._open is None and self._ended_by_usage is not None:
            # The terminal of the response its usage report already ended (when it made tool calls).
            self._ended_by_usage = None
            return
        if self._open is None and not (
            event.interrupted or event.finish_reason or event.provider_details or self._unanswered
        ):
            # A boundary that says nothing, with nothing under way and no reply owed: no response. Vertex's
            # `gemini-live-2.5-flash` closes a tool call's turn before its answer this way, and Gemini 3.8 an
            # exchange a resumed session was stuck on; their usage is the next response's.
            return
        self._ensure_response(before, named=event.provider_response_id)
        if event.more_expected and not event.interrupted and event.provider_details is None:
            # The model said a filler while it stalls for a tool call: held open, as the call is its, but spoken
            # all the same, so what the user says next is a turn of its own.
            self._deferred = event
            self._close_turn(after)
            return
        self._end(
            after,
            status='cancelled' if event.interrupted else 'completed',
            finish_reason=event.finish_reason,
            provider_details=event.provider_details,
        )

    def _end_turn(self, turn: _Turn, events: list[LifecycleEvent]) -> None:
        """Another turn starts: `turn` is over, and joined the conversation, if it hadn't."""
        self._turn = None
        if not turn.joined:
            self._join(turn, events)
        elif turn.replied or not turn.identified:
            # No more words are coming: late words past its reply, or ones naming no turn, are the next turn's.
            # (A turn the provider named still gets words naming it, until it says its transcript is final.)
            events.append(UserTurnClosed(turn_id=turn.id))

    def _join(self, turn: _Turn, events: list[LifecycleEvent]) -> None:
        """`turn` joins the conversation here, ahead of the reply to it."""
        turn.joined = True
        self._spoken_unanswered = turn.id
        events.append(UserTurnEnded(turn_id=turn.id))

    def _speech_started(self, event: RealtimeInputSpeechStartEvent, before: list[LifecycleEvent]) -> None:
        """The provider heard the user start speaking: a spoken turn starts, if the provider names it.

        One it doesn't name starts with its first transcript, which may name it, or at its speech end.
        """
        self.speech_heard()
        self._announce_committed(before)
        turn = self._turn
        if turn is not None:
            if not turn.joined and (event.item_id is None or event.item_id == turn.id):
                return  # (the turn its words already started)
            self._end_turn(turn, before)
        self._speaking = True
        self._transcribed_while_speaking = False
        if event.item_id is not None and event.item_id not in self._identified:
            # Speech starting while the model answers is the user speaking over it, not the speech it answers.
            self._start_turn(event.item_id, before, transcribed=False)

    def _speech_ended(self, event: RealtimeInputSpeechEndEvent, after: list[LifecycleEvent]) -> None:
        """The provider heard the user stop: the turn joins the conversation, as the provider commits it now."""
        transcribed = self._speaking and self._transcribed_while_speaking
        self._speaking = self._transcribed_while_speaking = False
        if event.item_id is not None and event.item_id in self._identified:
            turn = self._turn
            if turn is None or turn.id != event.item_id or turn.joined:
                return  # (a turn that joined already)
        else:
            turn = self._turn
            if turn is None or turn.joined or (event.item_id is not None and turn.identified):
                # A turn of its own: the one before it is over.
                self._audio_since_reply = False
                if event.item_id is None:
                    if not transcribed:
                        self._commit_unnamed(after)
                    # (Otherwise the speech is the turn its words transcribed, which joined with the last of them.)
                    return
                if turn is not None:
                    self._end_turn(turn, after)
                turn = self._start_turn(event.item_id, after, transcribed=False)
        self._audio_since_reply = False
        self._join(turn, after)

    def _input_transcript(
        self, event: InputTranscript | RealtimeInputTranscriptionErrorEvent, before: list[LifecycleEvent]
    ) -> None:
        """Some of the user's words, or word that they can't be transcribed (which is the last there will be)."""
        final = isinstance(event, RealtimeInputTranscriptionErrorEvent) or event.is_final
        self._transcribed_while_speaking = self._speaking
        turn = self._turn
        late = turn is not None and turn.replied and not turn.words and event.item_id is None
        if not late and (event.item_id is None or event.item_id not in self._identified):
            # (Late words are the replied turn's, which came first, not those of a turn committed since.)
            self._announce_committed(before, event.item_id)
            turn = self._turn
        if event.item_id is not None and event.item_id in self._identified:
            # More of a turn the provider named, which needn't be the one being said now.
            if turn is not None and turn.id == event.item_id and final:
                if not turn.joined:
                    self._join(turn, before)
                self._turn = None
            return
        if turn is not None and (
            (event.item_id is not None and turn.id != event.item_id)
            or (turn.replied and turn.words and self._audio_since_reply)
        ):
            # The provider names a turn of its own, or the one before it said all it had: that one is over, and in
            # the conversation.
            self._end_turn(turn, before)
            turn = None
        if turn is None:
            turn = self._start_turn(event.item_id, before)
        if isinstance(event, InputTranscript) and event.text:
            turn.words = True
        if final:
            if not turn.joined:
                # All of it is transcribed: the provider has it.
                self._join(turn, before)
            self._turn = None

    def _start_turn(self, item_id: str | None, before: list[LifecycleEvent], *, transcribed: bool = True) -> _Turn:
        self._turns += 1
        turn = self._turn = _Turn(item_id or f'pydantic_ai_user_turn_{self._turns}', identified=item_id is not None)
        if item_id is not None:
            self._identified.add(item_id)
        before.append(UserTurnStarted(turn_id=turn.id))
        if transcribed and (reply := self._unprompted_reply) is not None:
            # The model started replying to speech before its transcript came (as most providers do): this is that
            # speech, which joined the conversation ahead of the reply.
            self._unprompted_reply = None
            turn.joined = True
            self._turn_reply = (turn.id, reply)
            before.append(UserTurnEnded(turn_id=turn.id, before_response=reply))
        return turn

    # --- responses ----------------------------------------------------------------------------------

    def _ensure_response(self, events: list[LifecycleEvent], *, named: str | None = None) -> None:
        """Start a response, if none is under way: its first output (or terminal) is all that announces it.

        `named` is the id the provider gave it, if its content carries one.
        """
        if self._open is not None:
            return
        self._announce_committed(events)
        answers = tuple(self._unanswered)
        self._unanswered.clear()
        self._held = [input_id for input_id in self._held if input_id not in answers]
        turn_id = self._join_turn(events, replying=not answers)
        if turn_id is None and not answers:
            # A spoken turn that joined the conversation before (at its speech end, say) and had no reply yet.
            turn_id = self._spoken_unanswered
        self._spoken_unanswered = None
        self._responses += 1
        self._open = named or f'pydantic_ai_response_{self._responses}'
        self._open_named = named is not None
        self._open_answers = answers
        self._open_calls = []
        self._calling = {}
        if turn_id is not None:
            self._turn_reply = (turn_id, self._open)
        elif not answers and self._transcribes and self.transcripts_lag_replies:
            self._unprompted_reply = self._open
        continues, self._continues = self._continues, None
        events.append(
            ResponseStarted(
                response_id=self._open,
                answers=answers,
                basis='inferred',
                user_turn_id=turn_id if not answers else None,
                continues=continues,
                provider_id=named is not None,
            )
        )

    def _join_turn(self, events: list[LifecycleEvent], *, replying: bool) -> str | None:
        """The spoken turn the starting response replies to joins the conversation ahead of it."""
        audio, self._audio_since_reply = self._audio_since_reply, False
        turn = self._turn
        if turn is not None and turn.replied:
            # The turn a reply already answered: this reply is to something said since.
            self._end_turn(turn, events)
            turn = None
        if turn is not None and self._transcribes:
            if not turn.joined:
                turn.joined = True
                events.append(UserTurnEnded(turn_id=turn.id))
            # (A turn that joined already, at its speech end or commit, is the one this replies to all the same.)
            return turn.id
        if (
            not (audio and replying)
            or (self._transcribes and (self._identified or not self.untranscribed_audio_is_a_turn))
            or self._spoken_unanswered
        ):
            # Nothing was said for the model to reply to (or the provider names its turns, and named none here), or
            # what was said joined the conversation already, at its last transcript, say.
            return None
        # The model replies to the audio sent since its last reply: that is a turn, which joins ahead of the reply,
        # and gets what transcript comes for it (transcripts naming no turn are the latest one's).
        turn = self._start_turn(None, events, transcribed=False)
        turn.joined = True
        events.append(UserTurnEnded(turn_id=turn.id))
        if not self._transcribes:
            self._turn = None
        return turn.id

    def _end(
        self,
        events: list[LifecycleEvent],
        *,
        status: ResponseStatus,
        finish_reason: FinishReason | None = None,
        provider_details: dict[str, Any] | None = None,
    ) -> None:
        """End the response under way; what was held back for it joins the conversation after it."""
        response_id = self._open
        assert response_id is not None
        self._open = None
        answers, self._open_answers = self._open_answers, ()
        said, self._open_said = self._open_said, False
        calls, self._open_calls = self._open_calls, []
        if status == 'cancelled':
            # Cut off: it answers what started it, if it said anything, and the next response answers the rest.
            self._release(events, response_id, answers if not said else answers[1:])
        elif finish_reason == 'tool_call' and self.inputs_cut_off_replies:
            # Its reply goes on in the answer to its calls, unless the provider cancels them (see `_calls_cancelled`).
            self._calling.update(dict.fromkeys(calls, (response_id, answers)))
        self._cut_off = False
        self._unprompted_reply = None
        self._deferred = None
        self._calls_awaiting_usage = False
        events.append(
            ResponseEnded(
                response_id=response_id, status=status, finish_reason=finish_reason, provider_details=provider_details
            )
        )
        if (reply := self._turn_reply) is not None and reply[1] == response_id and finish_reason != 'tool_call':
            # (A reply that only called tools goes on in the answer to their results, which the turn's words still
            # come before.)
            self._turn_reply = None
            self._close_turn(events, turn_id=reply[0])
        self._place_held(events)

    def _release(self, events: list[LifecycleEvent], response_id: str, released: tuple[InputId, ...]) -> None:
        """Inputs a response was taken to answer reached the provider after it started: the next response answers them.

        One that reaches the provider before a generation starts joins it, while one that comes during the generation
        cuts it off (as Gemini Live does). Which of those taken for its own came during it can't be told, but the
        oldest of them started it.
        """
        # Only those sent after the user last started speaking: speech after an input would cut the reply off as well,
        # so the reply may well have answered that input before it was.
        released = tuple(input_id for input_id in released if input_id > self._spoken_since)
        if self.inputs_cut_off_replies and released:
            events.append(AnswersRevised(response_id=response_id, released=released))
            self._unanswered[:0] = released

    def _calls_cancelled(self, event: ToolCallCancelled, events: list[LifecycleEvent]) -> None:
        """The provider cancelled tool calls: an input may have cut off the turn waiting on their results.

        That turn's reply would have gone on in the answer to the calls, so the response that made them answers
        only what started it, and the next response, which the provider starts for the input that cut in, the rest.
        """
        cancelled = next((self._calling[call_id] for call_id in event.tool_call_ids if call_id in self._calling), None)
        if cancelled is not None:
            self._calling = {call_id: calling for call_id, calling in self._calling.items() if calling != cancelled}
            response_id, answers = cancelled
            self._release(events, response_id, answers[1:])

    def _end_deferred(self, events: list[LifecycleEvent]) -> None:
        """End the held filler as the model carries the exchange on in a response of its own, starting now."""
        deferred, self._deferred = self._deferred, None
        assert deferred is not None
        held = self._open
        self._end(events, status='completed', finish_reason=deferred.finish_reason or 'stop')
        self._continues = held

    # --- user turns and inputs ----------------------------------------------------------------------

    def _close_turn(self, events: list[LifecycleEvent], *, turn_id: str | None = None) -> None:
        """The joined turn's reply (to `turn_id`, if given) is over, or was cut off: no more of its transcript comes."""
        turn = self._turn
        if turn is None or not turn.joined or (turn_id is not None and turn.id != turn_id):
            return
        # (A joined turn whose transcript was marked finished is done with already.)
        self._turn_reply = None
        if turn.identified:
            self._turn = None
        elif turn.words:
            # The provider doesn't say when it has transcribed all of it: what it has is all there is.
            self._turn = None
            events.append(UserTurnClosed(turn_id=turn.id))
        else:
            # None of its words came yet: what comes is late, and still this turn's, until the next turn starts.
            turn.replied = True

    def _place_held(self, events: list[LifecycleEvent]) -> None:
        held, self._held = self._held, []
        events.extend(InputAdded(input_id=input_id) for input_id in held)

    # --- the connection's own transitions ----------------------------------------------------------

    def connection_lost(self) -> list[LifecycleEvent]:
        """The connection dropped (and is being re-dialed, or is gone): nothing under way on it will finish.

        A re-dial never resumes a generation, even when it resumes the session, so the response under way is
        lost, and so is every reply still owed. A spoken turn the model was about to answer is in the
        conversation all the same, with what transcript it has.
        """
        events = self.take_pending()
        self._announce_committed(events)
        self._deferred = self._continues = None
        if self._open is not None:
            self._end(events, status='lost')
        if lost := tuple(self._unanswered):
            self._unanswered.clear()
            events.append(InputLost(input_ids=lost))
        # Ending the response closed the turn it replied to: one still here is the next, not joined yet.
        if (turn := self._turn) is not None:
            self._turn = None
            if not turn.joined:
                events.append(UserTurnEnded(turn_id=turn.id))
            # (Its transcript was still to come, or it would be over: what it has is all there will be.)
            events.append(UserTurnClosed(turn_id=turn.id))
        self._place_held(events)
        if self._transcribes:
            # What audio was sent is the turn its transcript names, or nothing. (Without transcripts, it is still
            # what the user said, for a commit or the next reply to make a turn of.)
            self._audio_since_reply = False
        return events
