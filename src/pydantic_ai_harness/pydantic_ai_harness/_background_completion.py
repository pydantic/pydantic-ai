"""Private helpers for background-tool completion provenance."""

from __future__ import annotations

import html
import re
from collections.abc import Sequence

from pydantic_ai.messages import TextContent, UserContent

_BACKGROUND_RESULT_ORIGIN = 'pydantic-ai-harness.background_tools.result.v1'

_TOOL_RESULT_CLOSING_TAG_OPENER = re.compile(r'<(?=\s*/\s*tool_result(?=[\s/>]))', re.IGNORECASE)
"""Matches only the `<` of a closing `tool_result` tag, in any case, spacing or with attributes, as `_framing` does."""


# TODO: switch to `pydantic_ai._framing.wrap()` once it lands (https://github.com/pydantic/pydantic-ai/pull/10547).
# This escaper is local because core's `_tool_result_provenance_tags` frames media only and never escapes a
# closing tag in a body; the background result frames text, so it neutralizes `</tool_result>` itself.
def tool_result_tags(tool_name: str, tool_call_id: str, *, failed: bool = False) -> tuple[str, str]:
    """The tags framing a background tool's output, which reaches the model on the user channel.

    They match the `<tool_result>` tags core puts around tool files it moves to the user channel, so the
    model can tell the tool's output from what the user said. A failed call is marked `outcome="failed"`,
    like [`ToolReturnPart.outcome`][pydantic_ai.messages.ToolReturnPart.outcome].
    """
    outcome = ' outcome="failed"' if failed else ''
    return (
        f'<tool_result tool_name="{html.escape(tool_name)}" tool_call_id="{html.escape(tool_call_id)}"{outcome}>',
        '</tool_result>',
    )


def escape_tool_result_body(text: str) -> str:
    """Neutralize closing `</tool_result>` tags in tool output, so it cannot end its own frame early."""
    return _TOOL_RESULT_CLOSING_TAG_OPENER.sub('&lt;', text)


def make_background_completion(text: str, task_id: str) -> TextContent:
    return TextContent(text, metadata={'origin': _BACKGROUND_RESULT_ORIGIN, 'tool_call_id': task_id})


def get_background_completion(content: str | Sequence[UserContent] | None) -> TextContent | None:
    if isinstance(content, Sequence) and len(content) == 1 and isinstance(content[0], TextContent):
        text_content = content[0]
        metadata = text_content.metadata
        if isinstance(metadata, dict) and 'origin' in metadata and metadata['origin'] == _BACKGROUND_RESULT_ORIGIN:
            return text_content
    return None
