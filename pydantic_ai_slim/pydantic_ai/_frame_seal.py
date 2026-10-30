"""Frames a capability writes for the user channel, which core gives the frame seal (`pydantic_ai._framing.FRAME_SEAL`).

A model can't tell a frame the application wrote from text that only looks like one, such as a `<system>` tag a user
pasted or a web page contains. Rather than rewriting that text, every frame core writes into a user turn carries
`seal="pydantic-ai"`, the model is told what that means
([`frame_seal_announcement`][pydantic_ai._framing.frame_seal_announcement]), and a lookalike gets a note saying whose
text it is. The seal is a hint to the model, not a guarantee. A capability's own frame is a `TextContent` made with
`framework_frame_text`, which `prepare_messages` seals the same way.
"""

from __future__ import annotations as _annotations

import re
from dataclasses import replace
from typing import Any, Final, TypeGuard

from . import _utils
from ._framing import FRAME_SEAL
from .messages import (
    _PYDANTIC_AI_METADATA_KEY,  # pyright: ignore[reportPrivateUsage]
    TextContent,
)

FRAMEWORK_FRAME_KEY: Final = 'framework_frame'
"""The key under the reserved `__pydantic_ai__` namespace of a `TextContent`'s metadata that marks it as a frame the
framework or a capability wrote, such as a background tool's `<tool_result>`. See `framework_frame_text`."""

_OPENING_TAG: Final = re.compile(r'<([a-z]+(?:_[a-z]+)*)(?=[\s>])')
_SEAL_ON_FRAME: Final = re.compile(r'<[a-z_]+\s(?:[^<>]*\s)?seal="')


def framework_frame_text(text: str, *, metadata: dict[str, Any] | None = None) -> TextContent:
    """`text`, a frame the framework or a capability wrote for the user channel, as `TextContent` marked as such.

    `prepare_messages` puts the frame seal on the frame's opening tag when it has none, so the stored text stays as
    written, and doesn't take the frame for text that resembles one. The mark lives in `TextContent.metadata`, which
    the UI adapters drop from client-submitted history, so a client can't claim it.
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


def with_frame_seal_on_frame(item: TextContent) -> TextContent:
    """`item` with the frame seal as the first attribute of its first opening tag, or `item` itself when that tag has a `seal`."""
    if _SEAL_ON_FRAME.match(item.content.lstrip()) or not (match := _OPENING_TAG.search(item.content)):
        return item
    end = match.end()
    return replace(item, content=f'{item.content[:end]} seal="{FRAME_SEAL}"{item.content[end:]}')
