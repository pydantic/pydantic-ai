"""Private helpers for recognizing harness-authored user-channel content by its metadata."""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic_ai._framing import neutralize_tags
from pydantic_ai.messages import TextContent, UserContent

_HARNESS_TAG_OPENERS = re.compile(
    r'<(?=\s*(?:/\s*)?(?:task[-_]output|task_question|peer_message)(?=[\s/>]|$))', re.IGNORECASE
)


def neutralize_harness_tags(text: str) -> str:
    """Escape core and Harness data tags in untrusted content, including a frame's own closing tag."""
    return _HARNESS_TAG_OPENERS.sub('&lt;', neutralize_tags(text))


def tagged_text(content: str | Sequence[UserContent] | None, origin: str) -> TextContent | None:
    """The `TextContent` that is all of `content`, when its metadata names `origin`."""
    if isinstance(content, Sequence) and len(content) == 1 and isinstance(content[0], TextContent):
        text_content = content[0]
        metadata = text_content.metadata
        if isinstance(metadata, dict) and 'origin' in metadata and metadata['origin'] == origin:
            return text_content
    return None
