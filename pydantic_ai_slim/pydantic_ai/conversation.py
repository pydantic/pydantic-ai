"""The state a conversation carries from one run to the next."""

from __future__ import annotations as _annotations

import dataclasses
from dataclasses import KW_ONLY

import pydantic

from . import messages as _messages, usage as _usage
from ._deferred import DeferredToolRequests
from ._messages_serialization import MessageHistory as _MessageHistory
from ._uuid import uuid7

__all__ = ('Conversation', 'ConversationTypeAdapter')


@dataclasses.dataclass
class Conversation:
    """Everything a conversation carries from one run to the next, in one object.

    This is the unit to continue and to store a conversation by. A run's
    [`AgentRunResult.conversation`][pydantic_ai.agent.AgentRunResult.conversation] (or a realtime
    session's [`conversation`][pydantic_ai.realtime.RealtimeSession.conversation]) produces one, and
    every entry point that starts a run takes it back as `conversation=`. See
    [Carrying a conversation whole](../message-history.md#carrying-a-conversation-whole).

    Passing `message_history=` instead still works, but carries only the messages. What it drops is
    what a conversation reassembled by hand loses: the running
    [`usage`][pydantic_ai.conversation.Conversation.usage], without which every turn's
    [`UsageLimits`][pydantic_ai.usage.UsageLimits] budget starts over from zero; the
    [`conversation_id`][pydantic_ai.conversation.Conversation.conversation_id] that correlates its
    runs; and the [`deferred_tool_requests`][pydantic_ai.conversation.Conversation.deferred_tool_requests]
    a paused run is waiting on, which can't be recovered from the messages.

    It serializes like any Pydantic value — as a field on a model of your own, or with
    [`ConversationTypeAdapter`][pydantic_ai.conversation.ConversationTypeAdapter] — with the same
    fidelity as [`ModelMessagesTypeAdapter`][pydantic_ai.messages.ModelMessagesTypeAdapter]. See
    [Persistence](../persistence.md).
    """

    __pydantic_config__ = pydantic.ConfigDict(defer_build=True)

    _: KW_ONLY

    messages: _MessageHistory = dataclasses.field(default_factory=list[_messages.ModelMessage])
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

    deferred_tool_requests: DeferredToolRequests | None = None
    """The tool calls the conversation is waiting on, if its last run paused for them.

    Set when a run ends with [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests] as its
    output: calls that need [approval](../deferred-tools.md#human-in-the-loop-tool-approval) or
    [external execution](../deferred-tools.md#external-tool-execution). The messages show which calls
    are unanswered, but not which kind of answer each needs or the metadata it was deferred with, so
    the requests travel with the conversation rather than being rebuilt from it.

    Answer them with [`build_results`][pydantic_ai.tools.DeferredToolRequests.build_results] and
    pass the results to the next run as `deferred_tool_results=` alongside the conversation; see
    [Pausing a conversation for deferred tools](../deferred-tools.md#pausing-a-conversation).
    """


ConversationTypeAdapter: pydantic.TypeAdapter[Conversation] = pydantic.TypeAdapter(Conversation)
"""Pydantic [`TypeAdapter`][pydantic.type_adapter.TypeAdapter] for (de)serializing a [`Conversation`][pydantic_ai.conversation.Conversation].

The counterpart of [`ModelMessagesTypeAdapter`][pydantic_ai.messages.ModelMessagesTypeAdapter] for the
whole conversation rather than its messages alone.
"""
