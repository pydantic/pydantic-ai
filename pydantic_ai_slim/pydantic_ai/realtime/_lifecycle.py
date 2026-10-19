"""The lifecycle events a connection yields alongside the codec events (see `pydantic_ai.realtime.lifecycle`).

Defined here, apart from `InferredLifecycle`, so that `codec.py` can name them without importing the codec
vocabulary back.
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from typing_extensions import TypeAliasType

from ..messages import FinishReason

if TYPE_CHECKING:
    from .codec import RealtimeCodecEvent

InputId = int
"""An input's position among every `send()` call made on a connection, as in `InputRejected.input_index`."""


AnswersBasis = Literal['protocol', 'inferred']
"""Where a response's `answers` come from: the provider's own word, or the connection's bookkeeping."""


@dataclass(frozen=True, kw_only=True)
class LifecycleEvent:
    """What a connection says about the responses, spoken turns, and inputs its codec events are about.

    The base of the lifecycle events below, so `isinstance(event, LifecycleEvent)` tells one apart from a codec
    event in a [`RealtimeFrame`][pydantic_ai.realtime.lifecycle.RealtimeFrame].
    """


@dataclass(frozen=True, kw_only=True)
class ResponseStarted(LifecycleEvent):
    """A response began: exactly once per response, before any of its content."""

    response_id: str
    """The response's id: the provider's, or one the connection made up when the provider gives none."""
    answers: tuple[InputId, ...] = ()
    """The inputs whose request for a response this response satisfies.

    Empty for a response the provider started on its own, such as a server-VAD reply to a spoken turn.
    """
    basis: AnswersBasis = 'protocol'
    """Whether the provider said which inputs `answers` are (`'protocol'`), or the connection worked it out
    from the requests it had outstanding (`'inferred'`)."""
    user_turn_id: str | None = None
    """The spoken user turn a response the provider started on its own is replying to, when known."""
    continues: str | None = None
    """The response this one carries on from: one the provider ended while saying the exchange isn't over (Gemini's
    `interaction_status=IN_PROGRESS`), so whatever waits for that one's reply waits for this one too."""
    provider_id: bool = True
    """Whether `response_id` is the provider's own id, which history records, rather than one the connection made up
    for a protocol that gives its responses none."""


ResponseStatus = Literal['completed', 'cancelled', 'failed', 'incomplete', 'lost']
"""How a response ended. `'lost'` is a response the connection gave up on: its connection dropped (and the
reconnect did not carry it over) or closed before the provider reported it done."""


@dataclass(frozen=True, kw_only=True)
class ResponseEnded(LifecycleEvent):
    """A response ended: exactly once per started response, after all of its content and usage."""

    response_id: str
    status: ResponseStatus
    finish_reason: FinishReason | None = None
    """The normalized reason the provider finished the response, when it reported one."""
    provider_details: dict[str, Any] | None = None
    """The provider's raw terminal status details, when it reported any."""


@dataclass(frozen=True, kw_only=True)
class UserTurnStarted(LifecycleEvent):
    """The user began a spoken turn (the provider heard speech start, or a commit names a turn it hadn't)."""

    turn_id: str


@dataclass(frozen=True, kw_only=True)
class UserTurnEnded(LifecycleEvent):
    """A spoken turn joined the provider's conversation, at this point in its order."""

    turn_id: str
    still_speaking: bool = False
    """Whether it joined while the user is still saying it (xAI adds its item at speech start): its audio, and with
    it the turn, then ends with the speech."""
    before_response: str | None = None
    """The response it joined ahead of, when the connection learned of the turn only after that response, its reply,
    had started (Gemini transcribes the user's words after the model starts answering them)."""


@dataclass(frozen=True, kw_only=True)
class UserTurnClosed(LifecycleEvent):
    """No more audio or transcript is coming for a spoken turn.

    It was cleared, its connection lost, or its transcript is over. One that hadn't joined the conversation
    (`UserTurnEnded`) is dropped, as it never will; one that had keeps what it has.
    """

    turn_id: str


@dataclass(frozen=True, kw_only=True)
class InputAdded(LifecycleEvent):
    """An input joined the provider's conversation, at this point in its order."""

    input_id: InputId


@dataclass(frozen=True, kw_only=True)
class InputLost(LifecycleEvent):
    """The provider will never answer these inputs' request for a response.

    The connection dropped the request (a barge-in cut off the response it was waiting behind), or lost it
    with the connection. The inputs' content, if any, is unaffected.
    """

    input_ids: tuple[InputId, ...]


@dataclass(frozen=True, kw_only=True)
class ResponseRequestRefused(LifecycleEvent):
    """The provider refused the request for a response made for these inputs, so none will answer them."""

    input_ids: tuple[InputId, ...]


@dataclass(frozen=True, kw_only=True)
class OutputItemDetails(LifecycleEvent):
    """What the provider says about one item of a response's output, to keep on the part that item makes.

    Like OpenAI's `phase`, which tells an assistant message the model meant as commentary on its way to a tool
    call apart from its final answer.
    """

    response_id: str
    item_id: str
    provider_details: dict[str, Any]


@dataclass(frozen=True)
class StaleEvent:
    """A codec event about a response that has already ended: a repeated terminal, or content trailing it.

    The application still receives `event`, as it always has, but nothing applies it to the conversation: the
    response it is about is over. A connection that can't tell such events apart (most can't) never yields one.
    """

    event: RealtimeCodecEvent


RealtimeFrame = TypeAliasType('RealtimeFrame', 'list[RealtimeCodecEvent | LifecycleEvent | StaleEvent]')
"""What one message from the provider (or one transition of the connection's own, like a reconnect) makes.

The events are in order, and the session applies a frame whole before it streams any of it, so whatever reacts
to one of its events finds the conversation as far along as the frame takes it.
"""
