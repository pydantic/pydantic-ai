"""The state a conversation carries from one run to the next."""

from __future__ import annotations as _annotations

import dataclasses
from collections.abc import Sequence
from copy import copy
from dataclasses import KW_ONLY

from . import messages as _messages, usage as _usage
from ._uuid import uuid7
from .exceptions import UserError

__all__ = ('Conversation',)


@dataclasses.dataclass
class Conversation:
    """Everything a conversation carries between runs, in one object.

    `Agent.run` and a realtime session each already accept these three values, one argument at a
    time. Keeping them together is what makes a conversation portable: a run's
    [`AgentRunResult.conversation`][pydantic_ai.agent.AgentRunResult.conversation] can seed a
    realtime session, and that session's `conversation` can seed the next text run, without the
    caller remembering which pieces travel.

    [`messages`][pydantic_ai.conversation.Conversation.messages] alone is often enough, and
    `message_history=` stays the way to pass just those. The other two are the ones quietly lost
    when a conversation is reassembled by hand — most of all
    [`usage`][pydantic_ai.conversation.Conversation.usage], since a conversation that doesn't carry
    it starts every turn's [`UsageLimits`][pydantic_ai.usage.UsageLimits] budget over from zero.

    It is a plain dataclass of serializable fields, so it round-trips through
    [Pydantic](../storage.md) like the rest of a message history.
    """

    _: KW_ONLY

    messages: list[_messages.ModelMessage] = dataclasses.field(default_factory=list[_messages.ModelMessage])
    """The conversation so far, in the form `message_history=` takes."""

    usage: _usage.RunUsage = dataclasses.field(default_factory=_usage.RunUsage)
    """Usage accumulated across every run in this conversation.

    Only some of this can be recovered from `messages` after the fact: the token counts and
    `requests` are recorded on each [`ModelResponse`][pydantic_ai.messages.ModelResponse], but
    [`tool_calls`][pydantic_ai.usage.RunUsage.tool_calls] is not, and a history that has since been
    trimmed no longer accounts for what the dropped turns cost. Carrying it keeps a conversation's
    spend true to what was actually spent.
    """

    conversation_id: str = dataclasses.field(default_factory=lambda: str(uuid7()))
    """The identifier every run in this conversation shares, and the key to store it under."""


def resolve_conversation(
    conversation: Conversation | None,
    *,
    message_history: Sequence[_messages.ModelMessage] | None,
    usage: _usage.RunUsage | None,
    conversation_id: str | None,
) -> tuple[Sequence[_messages.ModelMessage] | None, _usage.RunUsage | None, str | None]:
    """Resolve a `conversation` argument into the three arguments it stands in for.

    The usage is copied on the way out. A run accumulates into the `RunUsage` it is handed, so
    passing the conversation's own object would make running from a conversation change it —
    double-counting across two runs started from the same one, and corrupting it as a point to
    branch from. `copy` covers the mutable `details` mapping too, per `UsageBase.__copy__`.
    """
    if conversation is None:
        return message_history, usage, conversation_id

    if conflicts := [
        name
        for name, value in (
            ('message_history', message_history),
            ('usage', usage),
            ('conversation_id', conversation_id),
        )
        if value is not None
    ]:
        listed = ' and '.join(f'`{name}`' for name in conflicts)
        raise UserError(
            f'`conversation` already carries {listed}, so passing both is ambiguous. '
            f'Pass the conversation on its own, or pass its pieces yourself.'
        )

    return conversation.messages, copy(conversation.usage), conversation.conversation_id
