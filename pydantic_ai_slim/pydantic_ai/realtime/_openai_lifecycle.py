"""The OpenAI Realtime protocol's lifecycle events: which response, user turn, and input each frame is about.

`OpenAIRealtimeConnection` feeds this tracker the frames it reads and the inputs it sends, and yields the
[lifecycle events](./_lifecycle.py) it produces alongside the codec events. The protocol identifies almost
everything itself: responses have ids from `response.created` to `response.done`, spoken turns have item
ids from `input_audio_buffer.speech_started` to `.committed`, every item joins the conversation with a
`conversation.item.added`, and a `response.create`'s `metadata` comes back on the response it started.
What it doesn't identify, the tracker works out from the connection's own bookkeeping and says so
(`ResponseStarted.basis='inferred'`).

Shared by the Azure OpenAI and xAI dialects, which inherit the connection.
"""

from __future__ import annotations as _annotations

from collections import deque
from collections.abc import Sequence
from typing import Any

from openai.types.realtime import (
    InputAudioBufferCommittedEvent,
    InputAudioBufferSpeechStartedEvent,
    RealtimeErrorEvent,
)

from .._utils import is_str_dict
from ..messages import FinishReason
from ._lifecycle import (
    AnswersBasis,
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
from ._openai_protocol import (
    CONVERSATION_ITEM_ADDED_EVENT_ADAPTER,
    ProtocolResponse,
    client_item_input,
    is_user_message_item,
    rejected_inputs,
    response_metadata_answers,
)

_CONVERSATION_ITEM_ADDED_FRAMES = frozenset({'conversation.item.added', 'conversation.item.created'})


def frame_response_id(event_type: str | None, data: dict[str, Any]) -> str | None:
    """The id of the response a server frame is about, if it is about one."""
    if event_type in ('response.created', 'response.done'):
        response = data.get('response')
        response_id = response.get('id') if is_str_dict(response) else None
    elif isinstance(event_type, str) and event_type.startswith('response.'):
        response_id = data.get('response_id')
    else:
        return None
    return response_id if isinstance(response_id, str) and response_id else None


class OpenAILifecycle:
    """Turns an OpenAI-protocol connection's frames and sends into lifecycle events."""

    def __init__(self) -> None:
        self._leading: list[LifecycleEvent] = []
        """Events that precede the codec events of the frame being decoded: a response it starts."""
        self._pending: list[LifecycleEvent] = []
        """Events that follow them, or come outside any frame (a reconnect, a close)."""
        self._open: dict[str, None] = {}
        """Responses started and not yet ended, in start order."""
        self._ended: set[str] = set()
        self._synthetic_responses = 0
        self._current_synthetic: str | None = None
        """The id made up for a response the provider gave none, until its terminal."""
        self._metadata_echoed = False
        """Whether the server has echoed a `response.create`'s metadata: from then on, a response without
        ours in its metadata answers nothing of ours, whatever the connection had outstanding."""
        self._requests: dict[str, tuple[tuple[InputId, ...], bool]] = {}
        """Our `response.create`s neither started nor refused yet, by `event_id`: the inputs each answers, and
        whether it carried them as `metadata` (it can't when there are too many to fit)."""
        self._settled: set[InputId] = set()
        """Inputs already answered, refused, or lost. An inferred answer can be wrong (before the server has
        echoed any metadata, a response it started on its own looks like ours), and the response that really
        answers the input then settles nothing a second time."""
        self._speaking: dict[str, None] = {}
        """Spoken turns started and not yet committed or discarded."""
        self._unclaimed_turn: str | None = None
        """The latest committed spoken turn no response of the provider's own has answered yet."""
        self._ids_echoed = False
        """Whether the server has added an item under the id we chose for it: from then on, an item with any
        other id is none of ours, and the order items are added in no longer has to say which input they are."""
        self._messages: deque[InputId | None] = deque()
        """Our user message items awaiting their `conversation.item.added`, in the order they were sent.
        `None` is one that is no input of its own: seeded or replayed history, or a tool result's follow-up."""
        self._tool_outputs: dict[str, InputId] = {}
        """Our tool outputs awaiting their `conversation.item.added`, by call id."""
        self._carried_over: set[InputId] = set()
        """Inputs an old socket never acknowledged, which join the conversation once a reconnect succeeds."""
        self._committed: set[str] = set()
        """Spoken turns already committed, so a repeated commit doesn't make a second turn of one."""

    # --- what the connection sends ----------------------------------------------------------------

    def message_sent(self, input_id: InputId | None) -> None:
        """A user message item of ours is on its way: its `conversation.item.added` places the input."""
        self._messages.append(input_id)

    def foreign_messages_sent(self, count: int) -> None:
        """User message items that are no input of the session's (seeded or replayed history) are on their way."""
        self._messages.extend([None] * count)

    def tool_output_sent(self, call_id: str, input_id: InputId) -> None:
        self._tool_outputs[call_id] = input_id

    def request_sent(self, event_id: str, answers: tuple[InputId, ...], *, tagged: bool) -> None:
        """A `response.create` for `answers` is on its way under `event_id`, naming them in its metadata if `tagged`."""
        self._requests[event_id] = (answers, tagged)

    def take_leading(self) -> list[LifecycleEvent]:
        leading, self._leading = self._leading, []
        return leading

    def take_pending(self) -> list[LifecycleEvent]:
        pending, self._pending = self._pending, []
        return pending

    # --- what the server says ---------------------------------------------------------------------

    def is_ended(self, response_id: str | None) -> bool:
        return response_id is not None and response_id in self._ended

    def before_frame(self, event_type: str | None, data: dict[str, Any]) -> None:
        """Start the response a frame is the first to name, ahead of the frame's own events."""
        if event_type == 'response.created':
            return
        response_id = frame_response_id(event_type, data)
        if response_id is not None and response_id not in self._open and response_id not in self._ended:
            # A terminal echoes the request's metadata too, so a response first seen at its end is still known.
            response = data.get('response') if event_type == 'response.done' else None
            answers = response_metadata_answers(response.get('metadata')) if is_str_dict(response) else None
            self._leading.append(
                self._start(response_id, answers=self._settle(answers), basis='protocol')
                if answers is not None
                else self._start(response_id, answers=(), basis='inferred')
            )

    def response_created(
        self,
        response: ProtocolResponse,
        *,
        outstanding: tuple[str, tuple[InputId, ...]] | None,
    ) -> list[LifecycleEvent]:
        """A `response.created`: which inputs the response answers, from its metadata or what was outstanding.

        `outstanding` is the `event_id` of the connection's own unstarted `response.create` and the inputs it
        asked for, which the connection takes this response to be.
        """
        response_id = response.id or self._synthetic_id()
        if response_id in self._open or response_id in self._ended:
            return []
        answers = response_metadata_answers(response.metadata)
        basis: AnswersBasis = 'protocol'
        if answers is not None:
            self._metadata_echoed = True
            self._requests = {
                event_id: request for event_id, request in self._requests.items() if not set(request[0]) & set(answers)
            }
        elif outstanding is not None and not (
            # Once the server echoes metadata, a response without ours is not the one ours asked for, unless
            # ours carried none.
            self._metadata_echoed and self._requests.get(outstanding[0], ((), True))[1]
        ):
            event_id, answers = outstanding
            basis = 'inferred'
            self._requests.pop(event_id, None)
        else:
            answers = ()
        user_turn_id = None
        if not answers:
            # Started by the provider on its own: with server VAD, that is the reply to the spoken turn it
            # just committed.
            user_turn_id, self._unclaimed_turn = self._unclaimed_turn, None
        return [self._start(response_id, answers=self._settle(answers), basis=basis, user_turn_id=user_turn_id)]

    def response_done(
        self,
        response_id: str | None,
        *,
        status: ResponseStatus,
        finish_reason: FinishReason | None,
        provider_details: dict[str, Any] | None,
    ) -> None:
        """A `response.done`: the response ends, once. A repeat ends nothing."""
        if response_id is None:
            response_id = self._current_synthetic or self._synthetic_id()
            self._current_synthetic = None
        if response_id in self._ended:
            return
        if response_id not in self._open:
            self._pending.append(self._start(response_id, answers=(), basis='inferred'))
        self._pending.append(self._end(response_id, status, finish_reason, provider_details))

    def response_unreadable(self, active_response_id: str | None) -> None:
        """A `response.done` too malformed to name its response: it still ended the one being generated."""
        if active_response_id is not None and active_response_id in self._open:
            self._pending.append(self._end(active_response_id, 'lost', None, None))

    def frame(self, event_type: str | None, data: dict[str, Any]) -> list[LifecycleEvent]:
        """The events a frame about user turns, conversation items, or errors makes."""
        if event_type == 'input_audio_buffer.speech_started':
            return self.speech_started(data)
        if event_type == 'input_audio_buffer.committed':
            return self.audio_committed(data)
        if event_type == 'input_audio_buffer.cleared':
            return self.audio_cleared()
        if event_type in _CONVERSATION_ITEM_ADDED_FRAMES:
            return self.item_added(data)
        if event_type == 'error':
            return self.error(data)
        return []

    def speech_started(self, data: dict[str, Any]) -> list[LifecycleEvent]:
        item_id = InputAudioBufferSpeechStartedEvent.model_validate(data).item_id
        if not item_id or item_id in self._speaking:
            return []
        self._speaking[item_id] = None
        return [UserTurnStarted(turn_id=item_id)]

    def audio_committed(self, data: dict[str, Any]) -> list[LifecycleEvent]:
        """The input audio buffer was committed: the spoken turn joins the conversation here."""
        item_id = InputAudioBufferCommittedEvent.model_validate(data).item_id
        if item_id in self._committed:
            return []
        self._committed.add(item_id)
        events: list[LifecycleEvent] = []
        if item_id in self._speaking:
            del self._speaking[item_id]
        else:
            # Push-to-talk reports no speech start: the commit both starts and ends the turn.
            events.append(UserTurnStarted(turn_id=item_id))
        events.append(UserTurnEnded(turn_id=item_id))
        self._unclaimed_turn = item_id
        return events

    def audio_cleared(self) -> list[LifecycleEvent]:
        """The input audio buffer was cleared: a spoken turn under way never joins the conversation."""
        events: list[LifecycleEvent] = [UserTurnDiscarded(turn_id=turn_id) for turn_id in self._speaking]
        self._speaking.clear()
        return events

    def item_added(self, data: dict[str, Any]) -> list[LifecycleEvent]:
        """An item joined the conversation: if it is one of our inputs, that input is placed here."""
        item = CONVERSATION_ITEM_ADDED_EVENT_ADAPTER.validate_python(data).item
        if item.type == 'function_call_output' and item.call_id is not None:
            input_id = self._tool_outputs.pop(item.call_id, None)
        elif not is_user_message_item(item):
            # The model's output, or a spoken turn the server committed.
            return []
        elif (input_id := client_item_input(item.id)) is not None:
            if input_id not in self._messages:
                return []
            self._ids_echoed = True
            self._messages.remove(input_id)
        elif self._ids_echoed or not self._messages:
            # No item of ours: seeded or replayed history, or one a browser made on a sideband.
            return []
        else:
            # A server that doesn't keep our ids (or a recording made before we chose them): the server adds
            # items in the order it receives them.
            input_id = self._messages.popleft()
        return [] if input_id is None else [InputAdded(input_id=input_id)]

    def error(self, data: dict[str, Any]) -> list[LifecycleEvent]:
        """An `error` frame: a refused request for a response settles every input it was made for."""
        error = RealtimeErrorEvent.model_validate(data).error
        for rejected in rejected_inputs(error):
            if rejected.refused == 'content' and rejected.input_index in self._messages:
                # Refused content never joins the conversation, so no `conversation.item.added` will place it.
                self._messages.remove(rejected.input_index)
        if error.event_id is None or (request := self._requests.pop(error.event_id, None)) is None:
            return []
        return [ResponseRequestRefused(input_ids=answers)] if (answers := self._settle(request[0])) else []

    # --- the connection's own transitions ----------------------------------------------------------

    def requests_dropped(self, answers: Sequence[InputId]) -> None:
        """The connection dropped requests it was holding, so nothing will answer `answers`."""
        if unsettled := self._settle(answers):
            self._pending.append(InputLost(input_ids=unsettled))

    def socket_replaced(self) -> None:
        """A new socket is being dialed: what the old one hadn't acknowledged, it never will.

        Those inputs are in the conversation all the same: a replaying reconnect sends the history that holds
        them, and a resuming one carries on the conversation they were sent into.
        """
        placed = [input_id for input_id in self._messages if input_id is not None]
        placed += self._tool_outputs.values()
        # Placed once a reconnect succeeds, not before: a connection that never comes back placed nothing.
        self._carried_over.update(placed)
        self._messages.clear()
        self._tool_outputs.clear()

    def reconnected(self, *, restores_in_flight: bool, lost_inputs: Sequence[InputId]) -> None:
        """A reconnect succeeded: settle what it did not carry over, before it is reported."""
        self._pending.extend(InputAdded(input_id=input_id) for input_id in sorted(self._carried_over))
        self._carried_over.clear()
        self.requests_dropped(lost_inputs)
        if not restores_in_flight:
            self._requests.clear()
            self._lose_everything_open()

    def closed(self, unanswered: Sequence[InputId]) -> None:
        """The connection is gone for good: nothing still open will ever end on its own, or be answered."""
        self.requests_dropped(
            [*unanswered, *(input_id for answers, _ in self._requests.values() for input_id in answers)]
        )
        self._requests.clear()
        self._carried_over.clear()
        self._lose_everything_open()

    def _lose_everything_open(self) -> None:
        self._pending.extend(self._end(response_id, 'lost', None, None) for response_id in list(self._open))
        self._pending.extend(self.audio_cleared())
        self._current_synthetic = None

    # --- helpers ------------------------------------------------------------------------------------

    def _settle(self, input_ids: Sequence[InputId]) -> tuple[InputId, ...]:
        """Settle the inputs not settled yet, returning them."""
        unsettled = tuple(input_id for input_id in dict.fromkeys(input_ids) if input_id not in self._settled)
        self._settled.update(unsettled)
        return unsettled

    def _synthetic_id(self) -> str:
        self._synthetic_responses += 1
        self._current_synthetic = f'pydantic_ai_response_{self._synthetic_responses}'
        return self._current_synthetic

    def _start(
        self,
        response_id: str,
        *,
        answers: tuple[InputId, ...],
        basis: AnswersBasis,
        user_turn_id: str | None = None,
    ) -> ResponseStarted:
        self._open[response_id] = None
        return ResponseStarted(response_id=response_id, answers=answers, basis=basis, user_turn_id=user_turn_id)

    def _end(
        self,
        response_id: str,
        status: ResponseStatus,
        finish_reason: FinishReason | None,
        provider_details: dict[str, Any] | None,
    ) -> ResponseEnded:
        self._open.pop(response_id, None)
        self._ended.add(response_id)
        return ResponseEnded(
            response_id=response_id, status=status, finish_reason=finish_reason, provider_details=provider_details
        )
