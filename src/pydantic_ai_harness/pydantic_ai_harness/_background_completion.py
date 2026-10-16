"""Private helpers for background-tool completion provenance."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic_ai.messages import TextContent, UserContent

_BACKGROUND_RESULT_ORIGIN = 'pydantic-ai-harness.background_tools.result.v1'


def make_background_completion(text: str, task_id: str) -> TextContent:
    return TextContent(text, metadata={'origin': _BACKGROUND_RESULT_ORIGIN, 'tool_call_id': task_id})


def get_background_completion(content: str | Sequence[UserContent] | None) -> TextContent | None:
    if isinstance(content, Sequence) and len(content) == 1 and isinstance(content[0], TextContent):
        text_content = content[0]
        metadata = text_content.metadata
        if isinstance(metadata, dict) and 'origin' in metadata and metadata['origin'] == _BACKGROUND_RESULT_ORIGIN:
            return text_content
    return None
