"""Deprecated import location for `pydantic_ai_harness.compaction`.

This capability graduated out of `experimental`; importing from here still works but
emits a `DeprecationWarning`. Import from `pydantic_ai_harness.compaction` instead.
"""

from pydantic_ai_harness.compaction import (
    ClampOversizedMessages,
    ClearToolResults,
    CompactionStrategy,
    DeduplicateFileReads,
    LimitWarner,
    SlidingWindow,
    SummarizingCompaction,
    TieredCompaction,
    WarningKind,
    estimate_token_count,
)
from pydantic_ai_harness.experimental._warn import warn_moved

warn_moved('compaction', 'compaction')

__all__ = [
    'ClampOversizedMessages',
    'ClearToolResults',
    'CompactionStrategy',
    'DeduplicateFileReads',
    'LimitWarner',
    'SlidingWindow',
    'SummarizingCompaction',
    'TieredCompaction',
    'WarningKind',
    'estimate_token_count',
]
