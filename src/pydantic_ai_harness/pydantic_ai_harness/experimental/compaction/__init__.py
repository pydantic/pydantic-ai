"""Deprecated import location for `pydantic_ai_harness.compaction`.

This capability graduated out of `experimental`; importing from here still works but
emits a `HarnessDeprecationWarning`. Import from `pydantic_ai_harness.compaction` instead.
"""

from pydantic_ai_harness.compaction import (
    ClampOversizedMessages,
    ClearToolResults,
    CompactionStrategy,
    DeduplicateFileReads,
    SlidingWindowCompaction,
    SummarizingCompaction,
    TieredCompaction,
    TranscriptHandleProvider,
    WarningKind,
    WarnNearLimits,
    estimate_token_count,
    is_pinned,
    pin,
    reinject_pinned,
)
from pydantic_ai_harness.experimental._warn import warn_moved

warn_moved('compaction', 'compaction')

LimitWarner = WarnNearLimits
SlidingWindow = SlidingWindowCompaction

__all__ = [
    'ClampOversizedMessages',
    'ClearToolResults',
    'CompactionStrategy',
    'DeduplicateFileReads',
    'LimitWarner',
    'SlidingWindow',
    'SummarizingCompaction',
    'TieredCompaction',
    'TranscriptHandleProvider',
    'WarningKind',
    'estimate_token_count',
    'is_pinned',
    'pin',
    'reinject_pinned',
]
