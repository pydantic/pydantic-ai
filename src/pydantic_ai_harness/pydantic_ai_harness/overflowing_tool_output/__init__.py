"""Overflow capability: reduce oversized tool returns at production time.

`OverflowingToolOutput` intercepts a tool return when it is produced and reduces it --
truncating, spilling to a queryable file, or summarizing -- so an oversized payload does
not persist in history and get re-sent on every later model request. Combine the three
modes through an ordered list of size `bands`.

Spilled payloads are read back on demand through the registered `read_tool_result` tool;
the `OverflowStore` protocol is the seam for a durable backend (the local-file default
ships for single-process runs).
"""

from pydantic_ai_harness.overflowing_tool_output._bands import (
    Action,
    Band,
    Passthrough,
    Spill,
    Summarize,
    SummarizeFunc,
    Truncate,
)
from pydantic_ai_harness.overflowing_tool_output._capability import READ_TOOL_NAME, OverflowingToolOutput
from pydantic_ai_harness.overflowing_tool_output._payload import TruncationStrategy
from pydantic_ai_harness.overflowing_tool_output._store import LocalFileStore, OverflowStore

__all__ = [
    'READ_TOOL_NAME',
    'Action',
    'Band',
    'LocalFileStore',
    'OverflowStore',
    'OverflowingToolOutput',
    'Passthrough',
    'Spill',
    'Summarize',
    'SummarizeFunc',
    'Truncate',
    'TruncationStrategy',
]
