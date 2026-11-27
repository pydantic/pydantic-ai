"""Building a run's conversation from its state, knowing which of its messages its store doesn't have yet."""

from __future__ import annotations as _annotations

import dataclasses
from collections.abc import Sequence
from copy import copy
from typing import Any

from . import _agent_graph, messages as _messages
from ._conversation_marks import mark_stored
from .conversation import Conversation
from .tools import DeferredToolRequests

STAMPED_FIELDS = ('timestamp', 'instructions', 'run_id', 'conversation_id')
"""The fields a run fills in on a message it sends, when the message has them but they aren't set yet."""


def rewritten(message: _messages.ModelMessage, original: _messages.ModelMessage) -> bool:
    """Whether `message` differs from `original` other than by what a run adds to a request it sends.

    A run fills in the fields the request lacks, and adds parts to it, like the system prompt it reinjects
    when the request opens the history it sends, or a `ModelSettingsDeltaPart` recording a settings change.
    Any part added around the request's own parts, by the run or a capability, is saved with the request as
    it was sent. Changing or dropping one of the request's own parts rewrites it.
    """
    filled = {name: getattr(message, name, None) for name in STAMPED_FIELDS if getattr(original, name, False) is None}
    if (
        isinstance(message, _messages.ModelRequest)
        and isinstance(original, _messages.ModelRequest)
        and _keeps_in_order(message.parts, original.parts)
    ):
        filled['parts'] = message.parts
    return message != dataclasses.replace(original, **filled)


def _keeps_in_order(parts: Sequence[object], kept: Sequence[object]) -> bool:
    """Whether `parts` holds each of `kept`, in the same order, with only other parts added around them."""
    remaining = iter(parts)
    return all(part in remaining for part in kept)


def conversation_from_state(
    state: _agent_graph.GraphAgentState,
    new_message_index: int,
    *,
    messages: list[_messages.ModelMessage],
    deferred_tool_requests: DeferredToolRequests | None,
) -> Conversation:
    """The conversation a run's state holds, knowing which of its messages its store doesn't have yet.

    The usage and the deferred tool requests are copies, so the conversation can be carried and mutated without
    touching the run's own accounting or the conversation the run was given.
    """
    messages, new_message_index = _as_given(state, *(full_history(state) or (messages, new_message_index)))
    conversation = Conversation(
        messages=list(messages),
        usage=copy(state.usage),
        conversation_id=state.conversation_id,
        deferred_tool_requests=None if deferred_tool_requests is None else _copy_requests(deferred_tool_requests),
        revision=state.conversation_revision,
    )
    index = unsaved_start(state, messages, new_message_index)
    if index is not None and state.conversation_unchecked:
        # Its store compares the messages before the index with the stored ones instead.
        conversation.new_message_index = index
    elif index is not None:
        mark_stored(conversation, conversation.messages[:index])
    return conversation


def _copy_requests(requests: DeferredToolRequests) -> DeferredToolRequests:
    return DeferredToolRequests(
        calls=list(requests.calls),
        approvals=list(requests.approvals),
        metadata=dict(requests.metadata),
        validated_args=dict(requests.validated_args),
        _kinds=dict(requests._kinds),  # pyright: ignore[reportPrivateUsage]
    )


def unsaved_start(
    state: _agent_graph.GraphAgentState, messages: list[_messages.ModelMessage], new_message_index: int
) -> int | None:
    if state.conversation_saved:
        return len(messages)
    unsaved = state.conversation_unsaved
    if unsaved is None:
        return None
    # The unsaved messages the run was given come just before the ones it added, unless the run changed
    # them, in which case the store compares the whole history instead. Filling in what a request it sends
    # lacks doesn't count as a change.
    start = new_message_index - len(unsaved)
    carried = messages[start:new_message_index] if start >= 0 else []
    if len(carried) != len(unsaved) or any(map(rewritten, carried, unsaved)):
        return None
    return start


def failed_run_conversation(
    state: _agent_graph.GraphAgentState, deps: _agent_graph.GraphAgentDeps[Any, Any]
) -> Conversation:
    """The conversation of a run that raised, with the messages it got to, so a store can record the failed turn.

    A run that failed before adding a message, say before it applied the results for the tool calls its
    conversation was waiting on, leaves those calls waiting.
    """
    added = len(state.message_history) > deps.new_message_index
    return conversation_from_state(
        state,
        deps.new_message_index,
        messages=list(state.message_history),
        deferred_tool_requests=None if added else state.conversation_deferred_tool_requests,
    )


def given_history(
    state: _agent_graph.GraphAgentState, messages: list[_messages.ModelMessage], new_message_index: int
) -> list[_messages.ModelMessage] | None:
    """The history the run was given, if cleaning it to send changed it and `messages` still starts with it as cleaned.

    Cleaning makes a history valid to send: it merges adjacent requests, like an output tool's result and the
    prompt appended after it, and drops a tool result whose call isn't in the history, as at the start of a
    window. A history the run changed otherwise, say in a hook, keeps the change instead. Filling in or adding
    to a request it sends, as `rewritten` allows, doesn't count, and neither does the new timestamp of the
    request a run resuming without a new prompt sends again. The unsaved requests that request merged get what
    the run filled in on it.
    """
    given, cleaned = state.appended_messages.given, state.appended_messages.cleaned
    if given is None or cleaned is None or new_message_index != len(cleaned) or len(messages) < len(cleaned):
        return None
    if any(map(rewritten, messages, [_unstamped(message) for message in cleaned])):
        return None
    sent = messages[len(cleaned) - 1] if cleaned else None
    if sent is None or sent is cleaned[-1] or not isinstance(sent, _messages.ModelRequest):
        return given
    unsaved = {id(message) for message in state.conversation_unsaved or ()}
    given = list(given)
    index = len(given) - 1
    while index >= 0 and isinstance(request := given[index], _messages.ModelRequest) and id(request) in unsaved:
        given[index] = dataclasses.replace(
            request,
            **{name: getattr(sent, name) for name in STAMPED_FIELDS if getattr(request, name) is None},
        )
        index -= 1
    return given


def _unstamped(message: _messages.ModelMessage) -> _messages.ModelMessage:
    """`message` without a timestamp, which a run resuming without a new prompt sets anew on the request it sends again."""
    return dataclasses.replace(message, timestamp=None) if isinstance(message, _messages.ModelRequest) else message


def _as_given(
    state: _agent_graph.GraphAgentState, messages: list[_messages.ModelMessage], new_message_index: int
) -> tuple[list[_messages.ModelMessage], int]:
    """`messages` and where the run's own start, with the history the run was given in place of it as cleaned."""
    given = given_history(state, messages, new_message_index)
    if given is None:
        return messages, new_message_index
    return [*given, *messages[new_message_index:]], len(given)


def full_history(state: _agent_graph.GraphAgentState) -> tuple[list[_messages.ModelMessage], int] | None:
    """The full history of a run whose history lost messages it was given or added, and where the ones it added start.

    For a run that ended, it's what `keep_full_history` kept. For one that raised, it's worked out from the run's
    record without changing it, as the run whose conversation a nested run's error carries may still be going.
    """
    appended = state.appended_messages
    if appended.full and appended.messages is not None:
        return appended.messages, appended.new_message_index
    return _merged_history(state)


def keep_full_history(state: _agent_graph.GraphAgentState) -> None:
    """Keep the full history of a run that just ended, if its history lost any of it, and nothing otherwise.

    Called once the run's last node and its hooks ran, so the result's `conversation` neither looks at the history
    again nor holds on to the record: an edit to the history after the run then reaches the conversation, as it
    would without a capability, unless one dropped or rewrote messages during the run.
    """
    appended = state.appended_messages
    full = _merged_history(state)
    if full is None:
        appended.messages = None
    else:
        appended.messages, appended.new_message_index = _as_given(state, *full)
        appended.full = True
        # The full history holds the history as it was given now.
        appended.given = appended.cleaned = None


def _merged_history(state: _agent_graph.GraphAgentState) -> tuple[list[_messages.ModelMessage], int] | None:
    """The history the run was given and every message it added, and where those start, if its history lost any of them.

    A capability can drop messages from the history or replace them with rewritten copies, say to keep what the
    model sees short. The full history then has the messages as the run was given or added them, and every other
    message the history holds, like a summary a capability put in place of the ones it dropped, right before the
    message it comes before in the history, or after the messages the run was given if that one is among them. A
    message a capability only adds, and filling in or adding to a message as `rewritten` allows, keeps the history
    whole, so this is `None`.
    """
    recorded = state.appended_messages.messages
    if recorded is None:
        return None
    merge = _HistoryMerge(recorded, given=state.appended_messages.new_message_index, history=state.message_history)
    for message in state.message_history:
        merge.place(message)
    merge.finish()
    return (merge.merged, merge.given_end) if merge.changed else None


@dataclasses.dataclass
class _HistoryMerge:
    """Merges a run's message history into the messages it was given and added, for `_merged_history`."""

    recorded: list[_messages.ModelMessage]
    given: int
    """How many of `recorded` the run was given."""
    history: dataclasses.InitVar[list[_messages.ModelMessage]]
    merged: list[_messages.ModelMessage] = dataclasses.field(default_factory=list[_messages.ModelMessage])
    given_end: int = 0
    """Where the messages the run was given end in `merged`."""
    changed: bool = False
    """Whether the history dropped or rewrote any of `recorded`."""
    _added: list[_messages.ModelMessage] = dataclasses.field(default_factory=list[_messages.ModelMessage])
    """Messages of the history not in `recorded`, waiting for the recorded message they come before."""
    _after_given: list[_messages.ModelMessage] = dataclasses.field(default_factory=list[_messages.ModelMessage])
    """Messages of the history that came before one the run was given, waiting for the last of those."""
    _start: int = 0
    _positions: dict[int, int] = dataclasses.field(default_factory=dict[int, int])
    _held: set[int] = dataclasses.field(default_factory=set[int])

    def __post_init__(self, history: list[_messages.ModelMessage]) -> None:
        self._positions = {id(message): index for index, message in enumerate(self.recorded)}
        self._held = {id(message) for message in history}

    def place(self, message: _messages.ModelMessage) -> None:
        """Merge in `message`, the next one of the history."""
        index = self._positions.get(id(message))
        if index is not None and index < self._start:
            # Already merged: the history moved it.
            self.changed = True
            return
        copied = False
        if index is None:
            index, copied = self._find_original(message)
        if index is None:
            self._added.append(message)
            return
        self.changed = self.changed or copied or index != self._start
        self.take(index)
        if id(message) not in self._positions and not copied:
            # A copy that only fills in or adds to the message stands for it as the history holds it.
            self.merged[-1] = message

    def finish(self) -> None:
        """Merge the recorded messages after the last one the history holds, then the added ones after it."""
        self.changed = self.changed or self._start < len(self.recorded)
        self.take(len(self.recorded))

    def take(self, index: int) -> None:
        """Merge the recorded messages the history dropped before `index`, the added ones waiting, then the one at `index`.

        Added messages that come before a message the run was given wait until the last of those is merged: those
        are stored already, and a store only adds after them, so a summary of some of them goes after all of them.
        """
        if index < self.given:
            self._merge_recorded(range(self._start, index))
            self._after_given += self._added
        else:
            given = max(self._start, self.given)
            self._merge_recorded(range(self._start, given))
            self.merged += self._after_given
            self._after_given = []
            self._merge_recorded(range(given, index))
            self.merged += self._added
        self._added = []
        self._merge_recorded(range(index, min(index + 1, len(self.recorded))))
        self._start = index + 1

    def _merge_recorded(self, indices: range) -> None:
        for index in indices:
            self.merged.append(self.recorded[index])
            if index < self.given:
                self.given_end = len(self.merged)

    def _find_original(self, message: _messages.ModelMessage) -> tuple[int | None, bool]:
        """The recorded message `message` is a copy of, among the ones the history dropped, and whether it rewrote it.

        A copy that only fills in or adds to a message, as `rewritten` allows, stands for it. So does one that keeps
        its timestamp and run, as `dataclasses.replace` does, as a rewrite of it, even with a `run_id` the run filled in.
        """
        for index in range(self._start, len(self.recorded)):
            original = self.recorded[index]
            if id(original) in self._held:
                break
            if type(original) is not type(message):
                continue
            if not rewritten(message, original):
                return index, False
            if (
                original.timestamp is not None
                and message.timestamp == original.timestamp
                and original.run_id in (None, message.run_id)
            ):
                return index, True
        return None, False
