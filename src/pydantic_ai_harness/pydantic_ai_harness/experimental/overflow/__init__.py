"""Deprecated import location for `pydantic_ai_harness.tool_output_limits`.

This capability graduated out of `experimental`; importing from here still works but
emits a `DeprecationWarning`. Import from `pydantic_ai_harness.tool_output_limits` instead.
"""

from pydantic_ai_harness.experimental._warn import warn_moved
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

warn_moved('overflow', 'tool_output_limits')

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
