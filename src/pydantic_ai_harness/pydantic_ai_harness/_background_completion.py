"""Private helpers for background-tool completion provenance."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic_ai._framing import TOOL_RESULT_TAG, close_tag, neutralize_tags, open_tag
from pydantic_ai.messages import TextContent, UserContent
from pydantic_ai_harness._provenance import tagged_text

_BACKGROUND_RESULT_ORIGIN = 'pydantic-ai-harness.background_tools.result.v1'


def tool_result_tags(tool_name: str, tool_call_id: str, *, failed: bool = False) -> tuple[str, str]:
    """The tags framing a background tool's output, which reaches the model on the user channel.

    They match the `<tool_result>` tags core puts around tool files it moves to the user channel, so the
    model can tell the tool's output from what the user said. A failed call is marked `outcome="failed"`,
    like [`ToolReturnPart.outcome`][pydantic_ai.messages.ToolReturnPart.outcome].
    """
    outcome = 'failed' if failed else None
    return (
        open_tag(TOOL_RESULT_TAG, tool_name=tool_name, tool_call_id=tool_call_id, outcome=outcome),
        close_tag(TOOL_RESULT_TAG),
    )


def escape_tool_result_body(text: str) -> str:
    """Escape every opening and closing framing tag in tool output, so it stays inside its `<tool_result>` frame.

    The output's files sit between the frame's tags as separate content, so the text can't go through
    `wrap()` in one piece; this escapes it the same way.
    """
    return neutralize_tags(text)


def make_background_completion(text: str, task_id: str) -> TextContent:
    return TextContent(text, metadata={'origin': _BACKGROUND_RESULT_ORIGIN, 'tool_call_id': task_id})


def get_background_completion(content: str | Sequence[UserContent] | None) -> TextContent | None:
    return tagged_text(content, _BACKGROUND_RESULT_ORIGIN)
