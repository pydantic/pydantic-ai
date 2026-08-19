"""Tool output limits: reduce oversized tool returns at production time.

`ToolOutputLimits` intercepts a tool return when it is produced and reduces it --
truncating, spilling to a queryable file, or summarizing -- so an oversized payload does
not persist in history and get re-sent on every later model request. Combine the three
modes through an ordered list of size `bands`.

Spilled payloads are read back on demand through the registered `read_tool_result` tool;
the `OverflowStore` protocol is the seam for a durable backend (the local-file default
ships for single-process runs).
"""

from pydantic_ai_harness.tool_output_limits._bands import (
    Action,
    Band,
    Passthrough,
    Spill,
    Summarize,
    SummarizeFunc,
    Truncate,
)
from pydantic_ai_harness.tool_output_limits._capability import READ_TOOL_NAME, ToolOutputLimits
from pydantic_ai_harness.tool_output_limits._payload import (
    Serializer,
    TruncationStrategy,
    indented_json,
    json_lines,
)
from pydantic_ai_harness.tool_output_limits._store import LocalFileStore, OverflowStore

__all__ = [
    'READ_TOOL_NAME',
    'Action',
    'Band',
    'LocalFileStore',
    'OverflowStore',
    'Serializer',
    'ToolOutputLimits',
    'Passthrough',
    'Spill',
    'Summarize',
    'SummarizeFunc',
    'Truncate',
    'TruncationStrategy',
    'indented_json',
    'json_lines',
]
