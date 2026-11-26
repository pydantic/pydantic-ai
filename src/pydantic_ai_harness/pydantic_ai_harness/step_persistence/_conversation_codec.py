"""Turning conversation records and entries into the rows and documents the conversation stores keep."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from typing import Any

from pydantic import TypeAdapter

from pydantic_ai.conversation_store import ConversationDetails, ConversationEntry, ConversationRecord
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_ai_harness.media import MediaStore, externalize_media, restore_media
from pydantic_ai_harness.step_persistence._messages import keyed_messages
from pydantic_ai_harness.step_persistence._store import _OBJECT_LIST_ADAPTER  # pyright: ignore[reportPrivateUsage]

_RECORD_ADAPTER: TypeAdapter[ConversationRecord] = TypeAdapter(ConversationRecord)
_METADATA_ADAPTER: TypeAdapter[dict[str, Any] | None] = TypeAdapter(dict[str, Any] | None)
_MESSAGE_ADAPTER: TypeAdapter[ModelMessage] = TypeAdapter(ModelMessage)


def dump_record(record: ConversationRecord) -> str:
    """The record's JSON, without the `title` and `metadata` the stores keep beside it.

    Keeping them apart lets a save replace the record without writing them, so it keeps what
    `update` set since the save read the record.
    """
    return _RECORD_ADAPTER.dump_json(record, exclude={'title', 'metadata'}).decode()


def dump_metadata(metadata: dict[str, Any] | None) -> str | None:
    return None if metadata is None else _METADATA_ADAPTER.dump_json(metadata).decode()


def load_record(record: object, title: object, metadata: object) -> ConversationRecord:
    """The record `dump_record` stored as `record`, with the `title` and `metadata` stored beside it."""
    return dataclasses.replace(
        _RECORD_ADAPTER.validate_json(str(record)),
        title=None if title is None else str(title),
        metadata=None if metadata is None else _METADATA_ADAPTER.validate_json(str(metadata)),
    )


def detail_columns(details: ConversationDetails) -> dict[str, str | None]:
    """The column, or field, each key of `details` sets or `write_details` compares, with its stored value."""
    columns: dict[str, str | None] = {}
    if 'title' in details:
        columns['title'] = details['title']
    if 'metadata' in details:
        columns['metadata'] = dump_metadata(details['metadata'])
    return columns


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


def missing_entries_error(head_id: str) -> LookupError:
    """Raised when a history's chain of entries is broken, which only a store modified by hand can be."""
    return LookupError(f'The history ending in entry {head_id!r} references entries or messages that are missing.')
