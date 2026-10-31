"""Turning conversation records and entries into the rows and documents the conversation stores keep."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from pydantic import TypeAdapter

from pydantic_ai.conversation_store import ConversationEntry, ConversationRecord
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_ai_harness.media import MediaStore, externalize_media, restore_media
from pydantic_ai_harness.step_persistence._messages import keyed_messages
from pydantic_ai_harness.step_persistence._store import _OBJECT_LIST_ADAPTER  # pyright: ignore[reportPrivateUsage]

RECORD_ADAPTER: TypeAdapter[ConversationRecord] = TypeAdapter(ConversationRecord)
_MESSAGE_ADAPTER: TypeAdapter[ModelMessage] = TypeAdapter(ModelMessage)


async def keyed_entry_messages(
    entries: Sequence[ConversationEntry], *, media_store: MediaStore | None, threshold_bytes: int
) -> list[tuple[str, str]]:
    """Each entry's message SHA-256 and JSON, with large parts externalized to `media_store`."""
    payload: object = ModelMessagesTypeAdapter.dump_python([entry.message for entry in entries], mode='json')
    if media_store is not None:
        payload = await externalize_media(payload, media_store=media_store, threshold_bytes=threshold_bytes)
    return keyed_messages(_OBJECT_LIST_ADAPTER.validate_python(payload))


async def load_message(text: str, *, media_store: MediaStore | None) -> ModelMessage:
    """The message stored as `text`, with externalized parts read back from `media_store`."""
    payload: object = json.loads(text)
    if media_store is not None:
        payload = await restore_media(payload, media_store=media_store)
    return _MESSAGE_ADAPTER.validate_python(payload)


def entry_positions(
    entries: Sequence[ConversationEntry], stored: Mapping[str, tuple[str, int]]
) -> list[tuple[str, int]]:
    """Each new entry's `root_id`, the first entry of its history, and `depth`, its distance from that entry.

    `stored` maps each stored entry the new ones continue from to its own `root_id` and `depth`.
    """
    known = dict(stored)
    positions: list[tuple[str, int]] = []
    for entry in entries:
        if entry.parent_id is None:
            position = (entry.id, 0)
        else:
            root_id, depth = known[entry.parent_id]
            position = (root_id, depth + 1)
        known[entry.id] = position
        positions.append(position)
    return positions


def missing_entries_error(head_id: str) -> LookupError:
    """Raised when a history's chain of entries is broken, which only a store modified by hand can be."""
    return LookupError(f'The history ending in entry {head_id!r} references entries or messages that are missing.')
