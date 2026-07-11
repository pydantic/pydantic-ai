"""Deprecated import location for `pydantic_ai_harness.overflowing_tool_output`.

This capability graduated out of `experimental`; importing from here still works but
emits a `DeprecationWarning`. Import from `pydantic_ai_harness.overflowing_tool_output` instead.
"""

from pydantic_ai_harness.experimental._warn import warn_moved
from pydantic_ai_harness.overflowing_tool_output import (
    READ_TOOL_NAME,
    Action,
    Band,
    LocalFileStore,
    OverflowingToolOutput,
    OverflowStore,
    Passthrough,
    Spill,
    Summarize,
    SummarizeFunc,
    Truncate,
    TruncationStrategy,
)

warn_moved('overflow', 'overflowing_tool_output')

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
