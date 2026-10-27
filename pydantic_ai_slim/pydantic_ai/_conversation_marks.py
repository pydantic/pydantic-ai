"""Where a conversation's stored messages end, kept beside `Conversation.new_message_index`."""

from __future__ import annotations as _annotations

from .conversation import Conversation

_STORED = '_stored_messages'
"""Where `mark_stored` keeps the messages before the unsaved ones, outside the fields: not compared, serialized or copied."""


def mark_stored(conversation: Conversation, index: int) -> None:
    """Set `conversation.new_message_index`, remembering the messages before it to check that it still marks the boundary."""
    conversation.new_message_index = index
    conversation.__dict__[_STORED] = tuple(conversation.messages[:index])


def stored_boundary(conversation: Conversation) -> int | None:
    """`conversation.new_message_index`, if the messages before it are still the ones it was set after."""
    index = conversation.new_message_index
    if index is None or index == 0:
        return index
    stored: tuple[object, ...] = conversation.__dict__.get(_STORED, ())
    if len(stored) != index or index > len(conversation.messages):
        return None
    if any(message is not kept for message, kept in zip(conversation.messages, stored)):
        return None
    return index
