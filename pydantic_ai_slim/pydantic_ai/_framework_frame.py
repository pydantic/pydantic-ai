"""Frames a capability writes for the user channel, marked so the lookalike check doesn't note them.

A model can't tell a frame the application wrote from text that only looks like one, such as a `<system>` tag a user
pasted or a web page contains. Rather than rewriting that text, a lookalike gets a note saying whose text it is
([`lookalike_frame_note`][pydantic_ai._framing.lookalike_frame_note]). A capability's own frame is a `TextContent` made
with `framework_frame_text`, which the check leaves alone.
"""

from __future__ import annotations as _annotations

from typing import Any, Final, TypeGuard

from . import _utils
from .messages import (
    _PYDANTIC_AI_METADATA_KEY,  # pyright: ignore[reportPrivateUsage]
    TextContent,
)

FRAMEWORK_FRAME_KEY: Final = 'framework_frame'
"""The key under the reserved `__pydantic_ai__` namespace of a `TextContent`'s metadata that marks it as a frame the
framework or a capability wrote, such as a background tool's `<tool_result>`. See `framework_frame_text`."""


def framework_frame_text(text: str, *, metadata: dict[str, Any] | None = None) -> TextContent:
    """`text`, a frame the framework or a capability wrote for the user channel, as `TextContent` marked as such.

    `prepare_messages` doesn't take a marked frame for text that resembles one. The mark lives in
    `TextContent.metadata`, which the UI adapters drop from client-submitted history, so a client can't claim it.
    """
    namespace = (metadata or {}).get(_PYDANTIC_AI_METADATA_KEY)
    return TextContent(
        text,
        metadata={
            **(metadata or {}),
            _PYDANTIC_AI_METADATA_KEY: {
                **(namespace if _utils.is_str_dict(namespace) else {}),
                FRAMEWORK_FRAME_KEY: True,
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
