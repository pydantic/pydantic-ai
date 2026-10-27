"""Where a conversation's stored messages end, kept beside `Conversation.new_message_index`."""

from __future__ import annotations as _annotations

from .conversation import Conversation

_STORED_TAIL = '_stored_tail'
"""Where `mark_stored` keeps the message just before the unsaved ones, outside the fields: not compared, serialized or copied."""


def mark_stored(conversation: Conversation, index: int) -> None:
    """Set `conversation.new_message_index`, remembering the message before it to check that it still marks the boundary."""
    conversation.new_message_index = index
    conversation.__dict__[_STORED_TAIL] = conversation.messages[index - 1] if index else None


def stored_boundary(conversation: Conversation) -> int | None:
    """`conversation.new_message_index`, if the messages before it are still the ones it was set after."""
    index = conversation.new_message_index
    if index is None or index == 0:
        return index
    if index > len(conversation.messages) or conversation.messages[index - 1] is not conversation.__dict__.get(
        _STORED_TAIL
    ):
        return None
    return index
