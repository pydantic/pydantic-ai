"""Private helpers for background-tool completion provenance."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic_ai.messages import TextContent, UserContent
from pydantic_ai_harness._provenance import tagged_text

_BACKGROUND_RESULT_ORIGIN = 'pydantic-ai-harness.background_tools.result.v1'


def make_background_completion(text: str, task_id: str) -> TextContent:
    return TextContent(text, metadata={'origin': _BACKGROUND_RESULT_ORIGIN, 'tool_call_id': task_id})


def get_background_completion(content: str | Sequence[UserContent] | None) -> TextContent | None:
    return tagged_text(content, _BACKGROUND_RESULT_ORIGIN)
