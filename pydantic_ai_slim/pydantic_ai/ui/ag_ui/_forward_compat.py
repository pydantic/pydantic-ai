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
"""

from __future__ import annotations

import json
from typing import get_args

from ag_ui.core import InputContent, Message
from pydantic import BaseModel, JsonValue

from ..._utils import get_union_args

__all__ = ['skip_unknown_tagged_items']


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


def _unknown_tag(item: dict[str, JsonValue], discriminator: str, known: frozenset[str]) -> str | None:
    """A `"role='reasoning'"`-style label when `item`'s discriminator value is one the installed models don't know.

    `None` for an item that carries no string tag: that isn't new functionality, it's malformed, and
    validation should still report it.
    """
    tag = item.get(discriminator)
    if isinstance(tag, str) and tag not in known:
        return f'{discriminator}={tag!r}'
    return None


def skip_unknown_tagged_items(body: bytes) -> tuple[JsonValue, frozenset[str]]:
    """Re-read a rejected AG-UI request body without the items this install can't dispatch.

    Returns the reduced payload and labels for the tags that were skipped. The payload is only
    meaningful when the label set is non-empty; an empty set means there was nothing to skip and the
    caller should let the original validation error stand.

    `messages[]` and a user message's list `content` are the only tagged-union lists in
    `RunAgentInput`. A body that isn't a JSON object, or whose `messages` isn't a list, is left for
    validation to reject.
    """
    try:
        payload: JsonValue = json.loads(body)
    except (ValueError, RecursionError):
        # Re-reading the body is best effort on input that already failed validation, so every way
        # `json.loads` can reject it means there is nothing to skip and the caller's original
        # `ValidationError` (and the 422 it maps to) must stand. Invalid JSON and invalid UTF-8 both
        # arrive as `ValueError` subclasses — `UnicodeDecodeError` is not a `JSONDecodeError` — and
        # input nested past the interpreter's limit arrives as `RecursionError`.
        return None, frozenset()
    if not isinstance(payload, dict):
        return None, frozenset()
    messages = payload.get('messages')
    if not isinstance(messages, list):
        return None, frozenset()

    skipped: set[str] = set()
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
                        skipped.add(unknown_type)
                        continue
                    kept_content.append(item)
                message['content'] = kept_content
        kept_messages.append(message)

    payload['messages'] = kept_messages
    return payload, frozenset(skipped)
