"""Tool kinds of `WorkLog`'s tools, so other code can recognize them whatever they are named.

A leaf module: `ToolOutputLimits` exempts `search_work_log` and `read_work_log` by kind, and `WorkLog` orders itself
inside `ToolOutputLimits`, so neither package can import the kinds from the other without a cycle.
Each kind id is stored in every message history that records a call, so it cannot change later.
"""

from __future__ import annotations

from pydantic_ai import ToolKind

WORK_LOG_SEARCH_KIND = ToolKind('work_log', 'search_work_log')
"""The kind of `WorkLog`'s `search_work_log` tool, whatever it is named. Its returns are exempt from `ToolOutputLimits`."""

WORK_LOG_READ_KIND = ToolKind('work_log', 'read_work_log')
"""The kind of `WorkLog`'s `read_work_log` tool, whatever it is named. Its returns are exempt from `ToolOutputLimits`."""
