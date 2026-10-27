"""Deprecated import location for `pydantic_ai_harness.tool_output_limits`.

This capability was renamed; importing from here still works but emits a
`DeprecationWarning`. Import from `pydantic_ai_harness.tool_output_limits` instead.
"""

from pydantic_ai_harness._warn import warn_module_renamed
from pydantic_ai_harness.tool_output_limits import (
    READ_TOOL_NAME,
    Action,
    Band,
    LocalFileStore,
    OverflowStore,
    Passthrough,
    Spill,
    Summarize,
    SummarizeFunc,
    ToolOutputLimits,
    Truncate,
    TruncationStrategy,
)

OverflowingToolOutput = ToolOutputLimits

warn_module_renamed('overflowing_tool_output', 'tool_output_limits')

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
