"""`BackgroundTools` results, as the capability part the run receives them as.

Kept outside `background_tools` so that `guardrails` can recognize a result without importing the capability.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic_ai.messages import DataCapabilityPart, TextContent, ToolResultSource, UserContent
from pydantic_ai_harness._provenance import tagged_text

_BACKGROUND_RESULT_ORIGIN = 'pydantic-ai-harness.background_tools.result.v1'
"""The `origin` in the metadata of the `TextContent` an earlier version carried a background call's result in."""


@dataclass(kw_only=True)
class BackgroundToolResultPart(
    DataCapabilityPart[str | Sequence[UserContent]],
    namespace='background_tools',
    name='tool_result',
    tag='tool_result',
    attributes=('tool_name', 'tool_call_id', 'outcome'),
):
    """A background tool call's result, delivered once the call finishes.

    Data on the user channel in a `<tool_result>` tag naming the call, the same tag Pydantic AI puts around tool files it
    sends on the user channel, with a note that the tool wrote it, not the user or the application. Its content is
    what the tool returned, as text followed by any files, or the message of a failed call.
    """

    tool_name: str
    """The name of the tool that ran in the background."""

    tool_call_id: str
    """The id of the call the "running in background" reply announced."""

    outcome: str | None = None
    """`'failed'` for a call that failed, like [`ToolReturnPart.outcome`][pydantic_ai.messages.ToolReturnPart.outcome]."""


def make_background_result(
    tool_name: str, tool_call_id: str, content: str | Sequence[UserContent], *, failed: bool = False
) -> BackgroundToolResultPart:
    """A background call's result, written by the tool."""
    return BackgroundToolResultPart(
        content=content,
        source=ToolResultSource(tool_name=tool_name, tool_call_id=tool_call_id),
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        outcome='failed' if failed else None,
    )


def get_background_completion(content: str | Sequence[UserContent] | None) -> TextContent | None:
    """The text of a background call's result an earlier version delivered as user text marked by metadata."""
    return tagged_text(content, _BACKGROUND_RESULT_ORIGIN)
