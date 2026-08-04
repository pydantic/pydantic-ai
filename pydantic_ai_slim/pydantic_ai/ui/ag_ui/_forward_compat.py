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
reinterpreted.
"""

from __future__ import annotations

import json
from typing import Any, TypeGuard, get_args

from ag_ui.core import InputContent, Message
from pydantic import BaseModel

from ..._utils import is_str_dict

__all__ = ['skip_unknown_tagged_items']


def _known_tags(tagged_union: object, discriminator: str) -> frozenset[str]:
    """Discriminator values declared by the installed `ag-ui-protocol`'s members of a tagged union.

    Read off the union rather than hardcoded, so the known set tracks whatever version is installed —
    which is the whole point, since what counts as "new functionality" depends on the install.
    """
    members: tuple[type[BaseModel], ...] = get_args(get_args(tagged_union)[0])
    return frozenset(
        tag
        for member in members
        for tag in get_args(member.model_fields[discriminator].annotation)
        if isinstance(tag, str)
    )


# The discriminator names themselves are AG-UI wire constants, stable across every version in range.
_KNOWN_MESSAGE_ROLES = _known_tags(Message, 'role')
_KNOWN_INPUT_CONTENT_TYPES = _known_tags(InputContent, 'type')


def _is_any_list(obj: Any) -> TypeGuard[list[Any]]:
    """Check if obj is a list, narrowing the type to `list[Any]`.

    The counterpart of `is_str_dict` for the arrays in a decoded JSON body: the items are arbitrary
    client input we only ever inspect through `is_str_dict`, so they type the same way.
    """
    return isinstance(obj, list)


def _unknown_tag(item: Any, discriminator: str, known: frozenset[str]) -> str | None:
    """A `"role='reasoning'"`-style label when `item`'s discriminator value is one the installed models don't know.

    `None` for anything else, including an item that isn't an object or carries no string tag: those
    aren't new functionality, they're malformed, and validation should still report them.
    """
    if not is_str_dict(item):
        return None
    tag = item.get(discriminator)
    if isinstance(tag, str) and tag not in known:
        return f'{discriminator}={tag!r}'
    return None


def skip_unknown_tagged_items(body: bytes) -> tuple[Any, frozenset[str]]:
    """Re-read a rejected AG-UI request body without the items this install can't dispatch.

    Returns the reduced payload and labels for the tags that were skipped. The payload is only
    meaningful when the label set is non-empty; an empty set means there was nothing to skip and the
    caller should let the original validation error stand.

    `messages[]` and a user message's list `content` are the only tagged-union lists in
    `RunAgentInput`. A body that isn't a JSON object, or whose `messages` isn't a list, is left for
    validation to reject.
    """
    try:
        payload = json.loads(body)
    except (ValueError, RecursionError):
        # Re-reading the body is best effort on input that already failed validation, so every way
        # `json.loads` can reject it means there is nothing to skip and the caller's original
        # `ValidationError` (and the 422 it maps to) must stand. Invalid JSON and invalid UTF-8 both
        # arrive as `ValueError` subclasses — `UnicodeDecodeError` is not a `JSONDecodeError` — and
        # input nested past the interpreter's limit arrives as `RecursionError`.
        return None, frozenset()
    if not is_str_dict(payload):
        return None, frozenset()
    messages = payload.get('messages')
    if not _is_any_list(messages):
        return None, frozenset()

    skipped: set[str] = set()
    kept_messages: list[Any] = []
    for message in messages:
        if (unknown_role := _unknown_tag(message, 'role', _KNOWN_MESSAGE_ROLES)) is not None:
            skipped.add(unknown_role)
            continue
        if is_str_dict(message) and _is_any_list(content := message.get('content')):
            kept_content: list[Any] = []
            for item in content:
                if (unknown_type := _unknown_tag(item, 'type', _KNOWN_INPUT_CONTENT_TYPES)) is not None:
                    skipped.add(unknown_type)
                    continue
                kept_content.append(item)
            message['content'] = kept_content
        kept_messages.append(message)

    payload['messages'] = kept_messages
    return payload, frozenset(skipped)
