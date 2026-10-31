"""Frames a capability writes for the user channel, marked so the lookalike check doesn't note them.

A model can't tell a frame the application wrote from text that only looks like one, such as a `<system>` tag a user
pasted or a web page contains. Rather than rewriting that text, a lookalike gets a note saying whose text it is
([`lookalike_frame_note`][pydantic_ai._framing.lookalike_frame_note]). A capability's own frame is a `TextContent` made
with `framework_frame_text`, which the check leaves alone.
"""

from __future__ import annotations as _annotations

import re
from typing import Any, Final, TypeGuard

from . import _utils
from ._framing import LookalikeSource
from .messages import (
    _PYDANTIC_AI_METADATA_KEY,  # pyright: ignore[reportPrivateUsage]
    TextContent,
)

FRAMEWORK_FRAME_KEY: Final = 'framework_frame'
"""The key under the reserved `__pydantic_ai__` namespace of a `TextContent`'s metadata that marks it as a frame the
framework or a capability wrote, such as a background tool's `<tool_result>`. See `framework_frame_text`."""

FRAMEWORK_FRAME_AUTHOR_KEY: Final = 'framework_frame_author'
"""Next to `FRAMEWORK_FRAME_KEY`: who wrote the frame's body, when it isn't the framework or a capability."""

_OWN_TAGS: Final = re.compile(r'\A\s*<[^<>]*>|</[^<>]*>\s*\Z')


def framework_frame_text(
    text: str, *, metadata: dict[str, Any] | None = None, author: LookalikeSource | None = None
) -> TextContent:
    """`text`, a frame the framework or a capability wrote for the user channel, as `TextContent` marked as such.

    `prepare_messages` doesn't take the frame's own tags for text that resembles one. `author` says who wrote its
    body, such as `'tool'` for a background tool's output: that body is still checked, and text in it that resembles
    a frame gets a note naming that author. Without one, the body is the framework's or the capability's own. The mark
    lives in `TextContent.metadata`, which the UI adapters drop from client-submitted history, so a client can't
    claim it.
    """
    namespace = (metadata or {}).get(_PYDANTIC_AI_METADATA_KEY)
    return TextContent(
        text,
        metadata={
            **(metadata or {}),
            _PYDANTIC_AI_METADATA_KEY: {
                **(namespace if _utils.is_str_dict(namespace) else {}),
                FRAMEWORK_FRAME_KEY: True,
                **({FRAMEWORK_FRAME_AUTHOR_KEY: author} if author is not None else {}),
            },
        },
    )


def is_framework_frame(item: object) -> TypeGuard[TextContent]:
    """Whether `item` is a `TextContent` marked by `framework_frame_text`."""
    if not isinstance(item, TextContent):
        return False
    # `TextContent.metadata` is whatever the application put there, not necessarily a dict.
    namespace = item.metadata.get(_PYDANTIC_AI_METADATA_KEY) if _utils.is_str_dict(item.metadata) else None
    return _utils.is_str_dict(namespace) and namespace.get(FRAMEWORK_FRAME_KEY) is True


def framework_frame_body(item: TextContent) -> tuple[LookalikeSource, str] | None:
    """Who wrote a marked frame's body and the body without the frame's own tags, or `None` when the body is its own."""
    namespace = item.metadata.get(_PYDANTIC_AI_METADATA_KEY) if _utils.is_str_dict(item.metadata) else None
    author = namespace.get(FRAMEWORK_FRAME_AUTHOR_KEY) if _utils.is_str_dict(namespace) else None
    if author not in ('user', 'tool', 'file'):
        return None
    return author, _OWN_TAGS.sub('', item.content)
