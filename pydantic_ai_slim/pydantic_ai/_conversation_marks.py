"""Where a conversation's stored messages end, kept beside `Conversation.new_message_index`."""

from __future__ import annotations as _annotations

from collections.abc import Sequence

from .conversation import Conversation
from .messages import ModelMessage

_STORED = '_stored_messages'
"""Where `mark_stored` keeps the messages before the unsaved ones, outside the fields: not compared, serialized or copied."""


def mark_stored(conversation: Conversation, stored: Sequence[ModelMessage]) -> None:
    """Mark `stored`, the first messages of `conversation`, as the ones the store has, to check later that they still are."""
    conversation.new_message_index = len(stored)
    conversation.__dict__[_STORED] = tuple(stored)


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
