"""Forward compatibility for inbound AG-UI run input.

Our `ag-ui-protocol` floor is `>=0.1.10` and the policy (see `pydantic_ai/ui/AGENTS.md`) is that an
older install skips new functionality rather than erroring on it. AG-UI's models set `extra='allow'`,
so a *field* added to an existing type already parses and is ignored, but `Message` (discriminated on
`role`) and `InputContent` (discriminated on `type`) are tagged unions: a `role` or `type` the
installed models don't know is rejected outright, which fails validation for the whole request.
`ReasoningMessage` (0.1.11) and typed multimodal input content (0.1.15) both sit above the floor, so
a client that is merely newer than the server trips this.

This module reduces such a body to the items the installed models *can* dispatch, so the rest of the
run still parses. It deliberately removes nothing else: an item whose tag is known stays untouched
and keeps failing validation, so a genuinely malformed payload is still rejected rather than silently
reinterpreted. An unknown tag alone isn't enough either — an item only qualifies as new functionality
if it also satisfies the contract every member of its union shares, so a client bug can't ride in
under a tag we don't recognize.

The one tag that is unknown because the SDK *retired* it, `binary` on 1.0, is translated to its typed
replacement instead of skipped; a `binary` part with nothing to translate stays in and fails validation.
The translation rewrites the raw JSON rather than going through the SDK's deprecated `BinaryInputContent`
class, which 1.0 keeps importable for one release only.
"""

from __future__ import annotations

import json
from typing import get_args

from ag_ui.core import InputContent, Message
from pydantic import BaseModel, JsonValue

from ..._utils import get_union_args
from ._utils import media_part_type

__all__ = ['HAS_BINARY_INPUT_PART', 'adapt_unsupported_items']


def _known_tags(tagged_union: object, discriminator: str) -> frozenset[str]:
    """Discriminator values declared by the installed `ag-ui-protocol`'s members of a tagged union.

    Read off the union rather than hardcoded, so the known set tracks whatever version is installed —
    which is the whole point, since what counts as "new functionality" depends on the install.
    """
    members: tuple[type[BaseModel], ...] = get_union_args(tagged_union)
    return frozenset(
        tag
        for member in members
        for tag in get_args(member.model_fields[discriminator].annotation)
        if isinstance(tag, str)
    )


# The discriminator names themselves are AG-UI wire constants, stable across every version in range.
_KNOWN_MESSAGE_ROLES = _known_tags(Message, 'role')
_KNOWN_INPUT_CONTENT_TYPES = _known_tags(InputContent, 'type')

HAS_BINARY_INPUT_PART = 'binary' in _KNOWN_INPUT_CONTENT_TYPES
"""Whether the installed SDK still accepts the retired `binary` input part."""


def _unknown_tag(item: dict[str, JsonValue], discriminator: str, known: frozenset[str]) -> str | None:
    """A `"role='reasoning'"`-style label when `item`'s discriminator value is one the installed models don't know.

    `None` for an item that carries no string tag: that isn't new functionality, it's malformed, and
    validation should still report it.
    """
    tag = item.get(discriminator)
    if isinstance(tag, str) and tag not in known:
        return f'{discriminator}={tag!r}'
    return None


def _translate_binary_part(item: dict[str, JsonValue]) -> dict[str, JsonValue] | None:
    """The typed media part for a retired `binary` part, or `None` when it has no MIME type or payload.

    `url` wins over `data`, as it did in the legacy loader. A base64 data URI in `url` is how 0.x
    clients inlined bytes, so it becomes a data source, and its own media type wins over the declared
    one, as `BinaryContent.from_data_uri` did.
    """
    mime_type = item.get('mimeType', item.get('mime_type'))
    if not isinstance(mime_type, str):
        return None
    url = item.get('url')
    data = item.get('data')
    if isinstance(url, str) and url.startswith('data:') and ';base64,' in url:
        uri_mime_type, data = url.removeprefix('data:').split(';base64,', 1)
        mime_type, url = uri_mime_type or mime_type, None
    if isinstance(url, str) and url:
        source: dict[str, JsonValue] = {'type': 'url', 'value': url, 'mimeType': mime_type}
    elif isinstance(data, str) and data:
        source = {'type': 'data', 'value': data, 'mimeType': mime_type}
    else:
        return None
    return {'type': media_part_type(mime_type), 'source': source}


def adapt_unsupported_items(body: bytes) -> tuple[JsonValue, frozenset[str]] | None:
    """Re-read a rejected request, skipping unsupported tagged items and translating retired ones.

    Returns the adapted payload and labels for the skipped tags, or `None` when nothing was skipped
    or translated, in which case the caller should let the original `ValidationError` stand.
    """
    try:
        payload: JsonValue = json.loads(body)
    except (ValueError, RecursionError):
        # Re-reading the body is best effort on input that already failed validation, so every way
        # `json.loads` can reject it means there is nothing to skip and the caller's original
        # `ValidationError` (and the 422 it maps to) must stand. Invalid JSON and invalid UTF-8 both
        # arrive as `ValueError` subclasses — `UnicodeDecodeError` is not a `JSONDecodeError` — and
        # input nested past the interpreter's limit arrives as `RecursionError`.
        return None
    if not isinstance(payload, dict):
        return None
    messages = payload.get('messages')
    if not isinstance(messages, list):
        return None

    skipped: set[str] = set()
    translated = False
    kept_messages: list[JsonValue] = []
    for message in messages:
        if isinstance(message, dict):
            if (unknown_role := _unknown_tag(message, 'role', _KNOWN_MESSAGE_ROLES)) is not None:
                # A string `id` is the entire contract the `Message` union shares: it is the only
                # field every member requires in every version from our floor on, and `BaseMessage`
                # is not a common base (`ActivityMessage`, `ReasoningMessage` and `ToolMessage` don't
                # derive from it, and `ActivityMessage.content` is an object where
                # `BaseMessage.content` is a string). A message that fails it is malformed whatever
                # its role, so it stays in and keeps failing validation instead of being skipped.
                if isinstance(message.get('id'), str):
                    skipped.add(unknown_role)
                    continue
            elif isinstance(content := message.get('content'), list):
                # No such contract exists for content: the `InputContent` members share no field
                # beyond the discriminator, so a string `type` is all an unknown one can be held to.
                kept_content: list[JsonValue] = []
                for item in content:
                    if isinstance(item, dict) and (
                        (unknown_type := _unknown_tag(item, 'type', _KNOWN_INPUT_CONTENT_TYPES)) is not None
                    ):
                        if item.get('type') != 'binary':
                            skipped.add(unknown_type)
                            continue
                        if (media_part := _translate_binary_part(item)) is not None:
                            translated = True
                            kept_content.append(media_part)
                            continue
                        # A retired part with nothing to translate is malformed: it stays in so
                        # validation reports it, like any malformed item under a known tag.
                    kept_content.append(item)
                message['content'] = kept_content
        kept_messages.append(message)

    if not skipped and not translated:
        return None
    payload['messages'] = kept_messages
    return payload, frozenset(skipped)
