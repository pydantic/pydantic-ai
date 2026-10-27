"""Storing each snapshot message once, shared by the step stores."""

from __future__ import annotations

import hashlib
import json

from pydantic import TypeAdapter
from typing_extensions import TypedDict


def keyed_messages(messages: list[object]) -> list[tuple[str, str]]:
    """Each message's SHA-256 and the JSON it hashes, in history order."""
    keyed: list[tuple[str, str]] = []
    for message in messages:
        text = json.dumps(message, separators=(',', ':'))
        keyed.append((hashlib.sha256(text.encode()).hexdigest(), text))
    return keyed


class MessageRefs(TypedDict):
    """What a `PostgresStepStore` or `MongoStepStore` snapshot holds in place of its history."""

    message_refs: list[str]


_MESSAGE_REFS_ADAPTER: TypeAdapter[MessageRefs] = TypeAdapter(MessageRefs)


def message_refs(messages_text: str) -> list[str] | None:
    """The message keys a snapshot references, or `None` for one written with its history inline."""
    # Inline histories are JSON lists, so only a referencing snapshot starts with `{`.
    if not messages_text.startswith('{'):
        return None
    return _MESSAGE_REFS_ADAPTER.validate_json(messages_text)['message_refs']


def referenced_history_json(refs: list[str], stored: dict[str, str]) -> str:
    """The JSON list of the referenced messages; raises `ValueError` when one is missing."""
    missing = len({ref for ref in refs if ref not in stored})
    if missing:
        raise ValueError(f'snapshot references {missing} stored message(s) that are missing')
    return '[' + ','.join(stored[ref] for ref in refs) + ']'
