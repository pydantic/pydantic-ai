"""Content-addressed message storage shared by the durable step stores.

Every snapshot of a run, and every run continuing a conversation, repeats the history
before it, so storing each snapshot's messages inline makes a store grow with the
number of snapshots times the history length. The file, SQLite, Postgres, and Mongo
stores instead store each message once, keyed by the SHA-256 of its JSON, and write a
snapshot's messages as the ordered list of those keys: `{"message_refs": [...]}`.
A message that appears twice in a history is stored once and referenced twice.

Readers accept that shape and the inline JSON list written before it, so data written
by earlier releases stays readable. Hashing runs after media externalization, so a
stored message holds media URIs, not the payloads.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Mapping
from typing import Annotated

from pydantic import StringConstraints, TypeAdapter
from typing_extensions import TypedDict

from pydantic_ai_harness._warn import HarnessDeprecationWarning


class MessageRefs(TypedDict):
    """The stored form of a snapshot's messages: SHA-256 keys of its messages, in order."""

    message_refs: list[Annotated[str, StringConstraints(pattern=r'^[0-9a-f]{64}$')]]


_OBJECT_LIST_ADAPTER: TypeAdapter[list[object]] = TypeAdapter(list[object])
_MESSAGE_REFS_ADAPTER: TypeAdapter[MessageRefs] = TypeAdapter(MessageRefs)


def split_messages(messages_json: object) -> tuple[MessageRefs, dict[str, str]]:
    """Return a snapshot's references and the canonical JSON text of each distinct message."""
    refs: list[str] = []
    texts: dict[str, str] = {}
    for message in _OBJECT_LIST_ADAPTER.validate_python(messages_json):
        text = json.dumps(message, sort_keys=True, separators=(',', ':'))
        digest = hashlib.sha256(text.encode()).hexdigest()
        refs.append(digest)
        texts[digest] = text
    return {'message_refs': refs}, texts


def message_refs(stored: object) -> list[str] | None:
    """Return the references a stored snapshot holds, or `None` for an inline message list.

    Raises `ValueError` for anything that is neither.
    """
    if isinstance(stored, list):
        return None
    return _MESSAGE_REFS_ADAPTER.validate_python(stored)['message_refs']


def unknown_refs(refs: list[str], known: Mapping[str, str]) -> list[str]:
    """Return the distinct references in `refs` that `known` lacks, in first-use order."""
    return [digest for digest in dict.fromkeys(refs) if digest not in known]


def join_messages(refs: list[str], texts: Mapping[str, str]) -> list[object]:
    """Rebuild the JSON message list `refs` points at; raise `ValueError` if a message is missing."""
    if missing := unknown_refs(refs, texts):
        raise ValueError(f'snapshot references {len(missing)} stored message(s) that are missing')
    return [json.loads(texts[digest]) for digest in refs]


def warn_full_history_deprecated(owner: str) -> None:
    """Warn that `<owner>(deduplicate_messages=False)` is deprecated; called from the store's `__init__`."""
    warnings.warn(
        f'`{owner}(deduplicate_messages=False)` is deprecated: it stores the whole message history in every '
        'snapshot, so the store grows with the number of snapshots times the history length. Remove the argument '
        'to store each message once; full-history writes will be removed in a future release.',
        category=HarnessDeprecationWarning,
        stacklevel=3,
    )
