"""Building a run's conversation from the conversation it logged, knowing which of its messages its store has."""

from __future__ import annotations as _annotations

from copy import copy
from typing import Any

from . import _agent_graph
from ._conversation_marks import mark_stored
from .conversation import Conversation
from .tools import DeferredToolRequests


def conversation_from_state(
    state: _agent_graph.GraphAgentState, *, deferred_tool_requests: DeferredToolRequests | None
) -> Conversation:
    """The conversation a run's state logged, knowing which of its messages its store has.

    The usage and the deferred tool requests are copies, so the conversation can be carried and mutated without
    touching the run's own accounting or the conversation the run was given.
    """
    messages, stored, checked = state.conversation_log.conversation(state.message_history)
    conversation = Conversation(
        messages=messages,
        usage=copy(state.usage),
        conversation_id=state.conversation_id,
        deferred_tool_requests=None if deferred_tool_requests is None else _copy_requests(deferred_tool_requests),
        revision=state.conversation_revision,
    )
    if stored is not None and checked:
        mark_stored(conversation, messages[:stored])
    else:
        # Its store compares the messages before the index with the stored ones instead.
        conversation.new_message_index = stored
    return conversation


def _copy_requests(requests: DeferredToolRequests) -> DeferredToolRequests:
    return DeferredToolRequests(
        calls=list(requests.calls),
        approvals=list(requests.approvals),
        metadata=dict(requests.metadata),
        validated_args=dict(requests.validated_args),
        _kinds=dict(requests._kinds),  # pyright: ignore[reportPrivateUsage]
    )


def failed_run_conversation(
    state: _agent_graph.GraphAgentState, deps: _agent_graph.GraphAgentDeps[Any, Any]
) -> Conversation:
    """The conversation of a run that raised, with the messages it got to, so a store can record the failed turn.

    A run that failed before adding a message, say before it applied the results for the tool calls its
    conversation was waiting on, leaves those calls waiting.
    """
    added = len(state.message_history) > deps.new_message_index
    return conversation_from_state(
        state, deferred_tool_requests=None if added else state.conversation_deferred_tool_requests
    )
