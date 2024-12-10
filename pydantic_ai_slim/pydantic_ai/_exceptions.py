from __future__ import annotations as _annotations

from typing import Any


class StopAgentRun(Exception):
    """Exception raised to stop the agent run and use the provided result as the output."""

    result: Any
    """The value to use as the result of the agent run."""
    tool_name: str | None
    """The name of the tool call, if available."""

    def __init__(self, result: Any, tool_name: str | None) -> None:
        self.result = result
        self.tool_name = tool_name
