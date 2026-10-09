"""Lifecycle events for a protocol that identifies no responses, turns, or inputs: which each message is about.

Gemini Live and OpenAI GPT-Live give their responses no ids, have no response-start frame and no item
acknowledgements, so their connections feed this tracker the inputs they send and the codec events each
server message makes, and yield the [lifecycle events](./_lifecycle.py) it works out from their order
alongside them:

- A response starts with its first output (audio, a transcript, a native tool part, a tool call), or where the
  connection says the model took the turn without one (GPT-Live delegating work), and ends at its terminal
  (`ResponseDone`), or at the usage report that closes a response's tool calls. Its id is made up
  (`ResponseStarted.provider_id=False`).
- A response answers every input that asked for one and hadn't been answered when it started (a typed turn, a
  tool result): the model replies to each on its own, in order (`ResponseStarted.basis='inferred'`).
- A spoken turn has no id and no speech frames either: it starts with its first input transcript, joins the
  conversation as the reply to it starts, and has its whole transcript once that reply ends, if the transcript
  isn't marked finished before. Without input transcription, audio streamed since the last reply is a turn of
  its own, with no transcript, when the model next replies to speech.
- An input joins the conversation when it is sent, unless a reply is under way or owed: then it reached the
  provider while the model was working on that reply, which it follows.
- A dropped connection loses the response under way and every reply still owed: neither protocol resumes a
  generation on a new connection.
"""

from __future__ import annotations as _annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..messages import BinaryContent, FinishReason, PartStartEvent, RealtimeResponseInterruptedEvent
from ._lifecycle import (
    InputAdded,
    InputId,
    InputLost,
    LifecycleEvent,
    ResponseEnded,
    ResponseStarted,
    ResponseStatus,
    TaggedEvent,
    UserTurnDiscarded,
    UserTurnEnded,
    UserTurnStarted,
)
from .codec import (
    AudioDelta,
    InputTranscript,
    OutputTranscript,
    RealtimeCodecEvent,
    RealtimeInput,
    ResponseDone,
    SessionUsage,
    ToolCall,
    ToolResult,
)


@dataclass
class _Turn:
    """A spoken turn, from its first transcript until its reply ended."""

    id: str
    joined: bool = False
    final: bool = False
    """Its transcript is whole: marked finished, or no more will come."""


class InferredLifecycle:
    """Turns a connection's sends and messages into lifecycle events, for a protocol that identifies nothing."""

    def __init__(
        self, *, transcribes: bool, transcripts_lag_replies: bool = False, audio_starts_response: bool = True
    ) -> None:
        """Track a connection's lifecycle.

        Args:
            transcribes: Whether the user's speech is transcribed: if not, audio alone makes a spoken turn.
            transcripts_lag_replies: Whether the provider can transcribe the user's words after the model started
                answering them (Gemini does): a transcript that starts while a reply nobody asked for is under way
                is then the speech that reply answers, rather than speech over it.
            audio_starts_response: Whether output audio is the model speaking. GPT-Live's output is a continuous
                track, silent between replies: its connection says where the model takes the turn instead
                (`message(takes_turn=True)`), and audio outside a reply is no response.
        """
        self._audio_starts_response = audio_starts_response
        self._transcribes = transcribes
        self._transcripts_lag_replies = transcripts_lag_replies
        self._pending: list[LifecycleEvent] = []
        """Events that come outside any message (an input sent), yielded ahead of the next one."""
        self._responses = 0
        self._open: str | None = None
        """The response under way."""
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
        self._turn_reply: str | None = None
        """The response replying to the joined turn, whose end closes the turn's transcript."""
        self._unprompted_reply: str | None = None
        """The response under way, if the model started it on its own with no transcript of speech to reply to yet."""
        self._audio_since_reply = False
        """Audio went out since the last reply started: without transcription, the next turn of speech."""
        self._continues: str | None = None
        """A response the model ended saying the exchange goes on, which the next response carries on from."""

    # --- what the connection sends ----------------------------------------------------------------

    def input_sent(self, input_id: InputId, content: RealtimeInput) -> None:
        """An input is about to go out: note whether it asks for a reply, and where it joins the conversation."""
        if isinstance(content, BinaryContent) and content.is_audio:
            self._audio_since_reply = True
            return
        if self._reply_owed():
            # It reaches the provider while the model works on a reply, which it follows. (A response that answers
            # it places it ahead of itself, should that come first.)
            self._held.append(input_id)
        else:
            # Nothing is going on: it joins the conversation as it arrives.
            self._pending.append(InputAdded(input_id=input_id))
        if isinstance(content, (str, ToolResult)):
            self._unanswered.append(input_id)

    def input_failed(self, input_id: InputId) -> None:
        """An input never went out: forget it."""
        self._unanswered = [unanswered for unanswered in self._unanswered if unanswered != input_id]
        self._held = [held for held in self._held if held != input_id]
        self._pending = [event for event in self._pending if event != InputAdded(input_id=input_id)]

    def input_unanswerable(self, input_id: InputId) -> None:
        """An input went (or was meant to go) out, but the provider will never reply to it."""
        if input_id in self._unanswered:
            self._unanswered.remove(input_id)
            self._pending.append(InputLost(input_ids=(input_id,)))

    def take_pending(self) -> list[LifecycleEvent]:
        pending, self._pending = self._pending, []
        return pending

    def _reply_owed(self) -> bool:
        turn = self._turn
        return self._open is not None or bool(self._unanswered) or (turn is not None and not turn.joined)

    # --- what the server says ---------------------------------------------------------------------

    def message(self, codec: Sequence[RealtimeCodecEvent], *, takes_turn: bool = False) -> list[TaggedEvent]:
        """The lifecycle events around the codec events one server message makes, in order.

        `takes_turn` says the message has the model take the turn even without output of its own (GPT-Live
        delegating work): a response starts after its events, if none is under way.
        """
        tagged: list[TaggedEvent] = [(event, False) for event in self.take_pending()]
        for event in codec:
            before, after = self._event(event)
            tagged.extend((lifecycle, False) for lifecycle in before)
            tagged.append((event, False))
            tagged.extend((lifecycle, False) for lifecycle in after)
        if takes_turn:
            started: list[LifecycleEvent] = []
            self._ensure_response(started)
            tagged.extend((lifecycle, False) for lifecycle in started)
        return tagged

    def _event(self, event: RealtimeCodecEvent) -> tuple[list[LifecycleEvent], list[LifecycleEvent]]:
        before: list[LifecycleEvent] = []
        after: list[LifecycleEvent] = []
        output = (
            isinstance(event, (PartStartEvent, ToolCall))
            or (isinstance(event, AudioDelta) and self._audio_starts_response)
            or (isinstance(event, OutputTranscript) and bool(event.text))
        )
        if self._deferred is not None:
            if isinstance(event, (ToolCall, ResponseDone)):
                # The tool call the model was stalling for belongs to the filler's response, and so does the next
                # boundary (handled below).
                self._deferred = None
            elif output:
                # The stall didn't end in the tool call it was for: the filler was a response of its own, which
                # this one carries on.
                self._end_deferred(before, continued=True)
            elif isinstance(event, RealtimeResponseInterruptedEvent):
                # The user cut in: the exchange is over.
                self._end_deferred(before, continued=False)
            # Anything else (the user's transcript, usage) leaves it held: the exchange isn't over.
        if output:
            self._ensure_response(before)
            if isinstance(event, ToolCall) and event.response_usage_follows:
                self._calls_awaiting_usage = True
        elif isinstance(event, SessionUsage):
            # Usage reported between responses doesn't start one: it is the next response's (see `_response_done`).
            if event.response_scoped and self._calls_awaiting_usage:
                # A tool-call frame's report: the calls' response is over, and the answer comes once every result
                # is in.
                self._end(after, status='completed', finish_reason='tool_call')
        elif isinstance(event, InputTranscript):
            self._input_transcript(event, before)
        elif isinstance(event, RealtimeResponseInterruptedEvent):
            # The user (or a tool result) cut in: what was sent meanwhile follows what the model had said, and so
            # does the speech that cut in, a turn of its own.
            self._close_turn(after)
            self._place_held(after)
        elif isinstance(event, ResponseDone):
            self._response_done(event, before, after)
        # Everything else (the end of a native part, a cancelled call, a typed turn the connection reports lost to a
        # drop, which `connection_lost` lost already, ...) says nothing about the lifecycle.
        return before, after

    def _response_done(self, event: ResponseDone, before: list[LifecycleEvent], after: list[LifecycleEvent]) -> None:
        if self._open is None and not (
            event.interrupted or event.finish_reason or event.provider_details or self._unanswered
        ):
            # A boundary that says nothing, with nothing under way and no reply owed: no response. Vertex's
            # `gemini-live-2.5-flash` closes a tool call's turn before its answer this way, and Gemini 3.8 an
            # exchange a resumed session was stuck on; their usage is the next response's.
            return
        self._ensure_response(before)
        if event.more_expected and not event.interrupted and event.provider_details is None:
            self._deferred = event
            return
        self._end(
            after,
            status='cancelled' if event.interrupted else 'completed',
            finish_reason=event.finish_reason,
            provider_details=event.provider_details,
        )

    def _input_transcript(self, event: InputTranscript, before: list[LifecycleEvent]) -> None:
        turn = self._turn
        if turn is None:
            self._turns += 1
            turn = self._turn = _Turn(f'pydantic_ai_user_turn_{self._turns}')
            before.append(UserTurnStarted(turn_id=turn.id))
            if (reply := self._unprompted_reply) is not None:
                # The model started replying to speech before its transcript came (as Gemini usually does): this
                # is that speech, which joined the conversation ahead of the reply.
                self._unprompted_reply = None
                turn.joined = True
                self._turn_reply = reply
                before.append(UserTurnEnded(turn_id=turn.id, before_response=reply))
        if event.is_final:
            turn.final = True
            if turn.joined:
                self._turn = None

    # --- responses ----------------------------------------------------------------------------------

    def _ensure_response(self, events: list[LifecycleEvent]) -> None:
        """Start a response, if none is under way: its first output (or usage) is all that announces it."""
        if self._open is not None:
            return
        answers = tuple(self._unanswered)
        self._unanswered.clear()
        self._held = [input_id for input_id in self._held if input_id not in answers]
        turn_id = self._join_turn(events, replying=not answers)
        self._responses += 1
        self._open = f'pydantic_ai_response_{self._responses}'
        if turn_id is not None:
            self._turn_reply = self._open
        elif not answers and self._transcribes and self._transcripts_lag_replies:
            self._unprompted_reply = self._open
        continues, self._continues = self._continues, None
        events.append(
            ResponseStarted(
                response_id=self._open,
                answers=answers,
                basis='inferred',
                user_turn_id=turn_id if not answers else None,
                continues=continues,
                provider_id=False,
            )
        )

    def _join_turn(self, events: list[LifecycleEvent], *, replying: bool) -> str | None:
        """The spoken turn the starting response replies to joins the conversation ahead of it."""
        if self._transcribes:
            turn = self._turn
            if turn is None or turn.joined:
                return None
            turn.joined = True
            events.append(UserTurnEnded(turn_id=turn.id))
            if turn.final:
                self._turn = None
            return turn.id
        if not (self._audio_since_reply and replying):
            return None
        self._audio_since_reply = False
        self._turns += 1
        turn_id = f'pydantic_ai_user_turn_{self._turns}'
        events.extend((UserTurnStarted(turn_id=turn_id), UserTurnEnded(turn_id=turn_id)))
        return turn_id

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
        self._unprompted_reply = None
        self._deferred = None
        self._calls_awaiting_usage = False
        events.append(
            ResponseEnded(
                response_id=response_id, status=status, finish_reason=finish_reason, provider_details=provider_details
            )
        )
        if self._turn_reply == response_id:
            self._close_turn(events)
        self._place_held(events)

    def _end_deferred(self, events: list[LifecycleEvent], *, continued: bool) -> None:
        deferred, self._deferred = self._deferred, None
        assert deferred is not None
        held = self._open
        self._end(events, status='completed', finish_reason=deferred.finish_reason or 'stop')
        if continued:
            # The exchange goes on past it, in the response starting now.
            self._continues = held

    # --- user turns and inputs ----------------------------------------------------------------------

    def _close_turn(self, events: list[LifecycleEvent]) -> None:
        """The joined turn's reply is over (or was cut off): no more of its transcript is coming."""
        turn = self._turn
        if turn is None or not turn.joined:
            return
        # (A joined turn whose transcript was marked finished is done with already.)
        self._turn = self._turn_reply = None
        events.append(UserTurnDiscarded(turn_id=turn.id))

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
        self._deferred = self._continues = None
        if self._open is not None:
            self._end(events, status='lost')
        if lost := tuple(self._unanswered):
            self._unanswered.clear()
            events.append(InputLost(input_ids=lost))
        # Ending the response closed the turn it replied to: one still here is the next, not joined yet.
        if (turn := self._turn) is not None:
            self._turn = None
            events.append(UserTurnEnded(turn_id=turn.id))
            if not turn.final:
                events.append(UserTurnDiscarded(turn_id=turn.id))
        self._place_held(events)
        self._audio_since_reply = False
        return events
