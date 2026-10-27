"""Private helpers for recognizing harness-authored user-channel content by its metadata."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic_ai.messages import TextContent, UserContent


def tagged_text(content: str | Sequence[UserContent] | None, origin: str) -> TextContent | None:
    """The `TextContent` that is all of `content`, when its metadata names `origin`."""
    if isinstance(content, Sequence) and len(content) == 1 and isinstance(content[0], TextContent):
        text_content = content[0]
        metadata = text_content.metadata
        if isinstance(metadata, dict) and 'origin' in metadata and metadata['origin'] == origin:
            return text_content
    return None
