"""Compaction capabilities: keep an agent's conversation history within the context window.

Each capability lives in its own module; shared utilities (token estimation, the
`CompactionStrategy` protocol, tool-pair-safe cutoffs, in-place clearing) live in `_shared`.
"""

from pydantic_ai_harness._warn import warn_class_renamed
from pydantic_ai_harness.compaction._clamp_oversized_messages import ClampOversizedMessages
from pydantic_ai_harness.compaction._clear_tool_results import ClearToolResults
from pydantic_ai_harness.compaction._context_window import (
    DEFAULT_CONTEXT_WINDOW,
    resolve_context_window,
)
from pydantic_ai_harness.compaction._deduplicate_file_reads import DeduplicateFileReads
from pydantic_ai_harness.compaction._fallback_compaction import FallbackCompaction
from pydantic_ai_harness.compaction._manual import compact_now
from pydantic_ai_harness.compaction._pinning import is_pinned, pin, reinject_pinned
from pydantic_ai_harness.compaction._receipts import TranscriptHandleProvider
from pydantic_ai_harness.compaction._report_context_usage import ContextUsage, ReportContextUsage
from pydantic_ai_harness.compaction._shared import (
    CompactionStrategy,
    SupportsFocus,
    estimate_context_tokens,
    estimate_token_count,
)
from pydantic_ai_harness.compaction._sliding_window_compaction import SlidingWindowCompaction
from pydantic_ai_harness.compaction._summarizing_compaction import SummarizingCompaction
from pydantic_ai_harness.compaction._tiered_compaction import TieredCompaction
from pydantic_ai_harness.compaction._warn_near_limits import WarningKind, WarnNearLimits

__all__ = [
    'DEFAULT_CONTEXT_WINDOW',
    'ClampOversizedMessages',
    'ClearToolResults',
    'CompactionStrategy',
    'ContextUsage',
    'ReportContextUsage',
    'DeduplicateFileReads',
    'FallbackCompaction',
    'SlidingWindowCompaction',
    'SummarizingCompaction',
    'SupportsFocus',
    'TieredCompaction',
    'TranscriptHandleProvider',
    'WarnNearLimits',
    'WarningKind',
    'compact_now',
    'estimate_context_tokens',
    'estimate_token_count',
    'is_pinned',
    'pin',
    'reinject_pinned',
    'resolve_context_window',
]

_RENAMED = {
    'LimitWarner': WarnNearLimits,
    'SlidingWindow': SlidingWindowCompaction,
}


def __getattr__(name: str) -> object:
    renamed = _RENAMED.get(name)
    if renamed is not None:
        warn_class_renamed(name, renamed.__name__, 'pydantic_ai_harness.compaction')
        return renamed
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
