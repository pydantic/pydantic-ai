"""Building a run's conversation from its state, knowing which of its messages its store doesn't have yet."""

from __future__ import annotations as _annotations

import dataclasses
from copy import copy
from typing import Any

from . import _agent_graph, messages as _messages
from ._conversation_marks import mark_stored
from .conversation import Conversation
from .tools import DeferredToolRequests

STAMPED_FIELDS = ('timestamp', 'instructions', 'run_id', 'conversation_id')
"""The fields a run fills in on a message it sends, when the message has them but they aren't set yet."""


def rewritten(message: _messages.ModelMessage, original: _messages.ModelMessage) -> bool:
    """Whether `message` differs from `original` other than by what a run fills in on a request it sends."""
    filled = {name: getattr(message, name, None) for name in STAMPED_FIELDS if getattr(original, name, False) is None}
    return message != dataclasses.replace(original, **filled)


def conversation_from_state(
    state: _agent_graph.GraphAgentState,
    new_message_index: int,
    *,
    messages: list[_messages.ModelMessage],
    deferred_tool_requests: DeferredToolRequests | None,
) -> Conversation:
    """The conversation a run's state holds, knowing which of its messages its store doesn't have yet."""
    conversation = Conversation(
        messages=messages,
        usage=copy(state.usage),
        conversation_id=state.conversation_id,
        deferred_tool_requests=deferred_tool_requests,
        revision=state.conversation_revision,
    )
    index = unsaved_start(state, new_message_index)
    if index is not None and state.conversation_unchecked:
        # Its store compares the messages before the index with the stored ones instead.
        conversation.new_message_index = index
    elif index is not None:
        mark_stored(conversation, conversation.messages[:index])
    return conversation


def unsaved_start(state: _agent_graph.GraphAgentState, new_message_index: int) -> int | None:
    if state.conversation_saved:
        return len(state.message_history)
    unsaved = state.conversation_unsaved
    if unsaved is None:
        return None
    # The unsaved messages the run was given come just before the ones it added, unless a capability rewrote
    # them, in which case the store compares the whole history instead. Filling in what a request it sends
    # lacks doesn't count as a rewrite.
    start = new_message_index - len(unsaved)
    carried = state.message_history[start:new_message_index] if start >= 0 else []
    if len(carried) != len(unsaved) or any(map(rewritten, carried, unsaved)):
        return None
    return start


def failed_run_conversation(
    state: _agent_graph.GraphAgentState, deps: _agent_graph.GraphAgentDeps[Any, Any]
) -> Conversation:
    """The conversation of a run that raised, with the messages it got to, so a store can record the failed turn."""
    return conversation_from_state(
        state, deps.new_message_index, messages=list(state.message_history), deferred_tool_requests=None
    )
