"""Identified lifecycle events: the second version of the contract between a connection and the session.

The codec vocabulary in `codec.py` tells the session *what* the model said, but not which response it
belongs to, when that response began or ended, or which of the session's inputs it answers. The session
used to infer all of that from the shapes of event sequences, which is where most of its ordering bugs
came from. Only the connection sees the frames that settle those questions, so under this contract it
states them outright, with an id on every entity:

- a response is bracketed by exactly one `ResponseStarted` and one `ResponseEnded`, and its content
  events (which carry its `response_id`) arrive only in between;
- `ResponseStarted.answers` names the inputs whose reply the response is, so a reply obligation is
  settled by the response the provider says answers it rather than by whichever starts next;
- a spoken user turn is bracketed by `UserTurnStarted` and `UserTurnEnded` (or `UserTurnDiscarded`);
- `InputAdded` marks where an input joined the provider's conversation, so history can follow the
  provider's order;
- `InputLost` and `ResponseRequestRefused` settle the inputs no response will ever answer.

An input is identified the way `InputRejected.input_index` identifies it: by its position among every
`send()` call made on the connection.

These types are internal. A connection opts in with `RealtimeConnection._lifecycle_version = 2` and
yields them, interleaved with the codec events, from `RealtimeConnection._lifecycle_events()`; its public
iterator keeps yielding the codec events alone, so anything iterating a connection directly sees exactly
the vocabulary it always has. They become public only once they are documented as a provider-facing API.
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from typing_extensions import TypeAliasType

from ..messages import FinishReason

if TYPE_CHECKING:
    from .codec import RealtimeCodecEvent

InputId = int
"""An input's position among every `send()` call made on a connection, as in `InputRejected.input_index`."""


AnswersBasis = Literal['protocol', 'inferred']
"""Where a response's `answers` come from: the provider's own word, or the connection's bookkeeping."""


@dataclass(frozen=True, kw_only=True)
class ResponseStarted:
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


ResponseStatus = Literal['completed', 'cancelled', 'failed', 'incomplete', 'lost']
"""How a response ended. `'lost'` is a response the connection gave up on: its connection dropped (and the
reconnect did not carry it over) or closed before the provider reported it done."""


@dataclass(frozen=True, kw_only=True)
class ResponseEnded:
    """A response ended: exactly once per started response, after all of its content and usage."""

    response_id: str
    status: ResponseStatus
    finish_reason: FinishReason | None = None
    """The normalized reason the provider finished the response, when it reported one."""
    provider_details: dict[str, Any] | None = None
    """The provider's raw terminal status details, when it reported any."""


@dataclass(frozen=True, kw_only=True)
class UserTurnStarted:
    """The user began a spoken turn (the provider heard speech start, or a commit names a turn it hadn't)."""

    turn_id: str


@dataclass(frozen=True, kw_only=True)
class UserTurnEnded:
    """A spoken turn joined the provider's conversation, at this point in its order."""

    turn_id: str


@dataclass(frozen=True, kw_only=True)
class UserTurnDiscarded:
    """A spoken turn gets no more audio: it was cleared, or its connection lost.

    One that hadn't joined the conversation (`UserTurnEnded`) never will; one that had stays, as it was.
    """

    turn_id: str


@dataclass(frozen=True, kw_only=True)
class InputAdded:
    """An input joined the provider's conversation, at this point in its order."""

    input_id: InputId


@dataclass(frozen=True, kw_only=True)
class InputLost:
    """The provider will never answer these inputs' request for a response.

    The connection dropped the request (a barge-in cut off the response it was waiting behind), or lost it
    with the connection. The inputs' content, if any, is unaffected.
    """

    input_ids: tuple[InputId, ...]


@dataclass(frozen=True, kw_only=True)
class ResponseRequestRefused:
    """The provider refused the request for a response made for these inputs, so none will answer them."""

    input_ids: tuple[InputId, ...]


LifecycleEvent = TypeAliasType(
    'LifecycleEvent',
    ResponseStarted
    | ResponseEnded
    | UserTurnStarted
    | UserTurnEnded
    | UserTurnDiscarded
    | InputAdded
    | InputLost
    | ResponseRequestRefused,
)
"""The events a connection on the second version of the lifecycle contract yields besides the codec events."""

LIFECYCLE_EVENT_TYPES = (
    ResponseStarted,
    ResponseEnded,
    UserTurnStarted,
    UserTurnEnded,
    UserTurnDiscarded,
    InputAdded,
    InputLost,
    ResponseRequestRefused,
)
"""The `LifecycleEvent` variants, for `isinstance` checks."""

TaggedEvent: TypeAlias = 'tuple[RealtimeCodecEvent | LifecycleEvent, bool]'
"""An event of a connection's `_tagged_frames()`, with whether it is stale."""
