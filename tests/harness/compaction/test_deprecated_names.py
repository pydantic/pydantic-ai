"""Tests for the deprecated `SlidingWindow`, `LimitWarner` and `ContextUsageEvent` aliases."""

from __future__ import annotations

import re

import pytest

import pydantic_ai_harness.compaction as compaction
from pydantic_ai_harness import HarnessDeprecationWarning
from pydantic_ai_harness.compaction import ContextUsageMeasuredEvent, SlidingWindowCompaction, WarnNearLimits


def test_sliding_window_alias_warns_and_resolves() -> None:
    with pytest.warns(
        HarnessDeprecationWarning, match='renamed to `pydantic_ai_harness.compaction.SlidingWindowCompaction`'
    ):
        from pydantic_ai_harness.compaction import SlidingWindow  # importing is the assertion
    assert SlidingWindow is SlidingWindowCompaction


def test_limit_warner_alias_warns_and_resolves() -> None:
    with pytest.warns(HarnessDeprecationWarning, match='renamed to `pydantic_ai_harness.compaction.WarnNearLimits`'):
        from pydantic_ai_harness.compaction import LimitWarner  # importing is the assertion
    assert LimitWarner is WarnNearLimits


def test_context_usage_event_alias_warns_and_resolves() -> None:
    message = (
        '`pydantic_ai_harness.compaction.ContextUsageEvent` has been renamed to '
        '`pydantic_ai_harness.compaction.ContextUsageMeasuredEvent`. '
        'Update your imports; this deprecated alias will be removed in a future release.'
    )
    with pytest.warns(HarnessDeprecationWarning, match=re.escape(message)):
        from pydantic_ai_harness.compaction import ContextUsageEvent  # importing is the assertion
    assert ContextUsageEvent is ContextUsageMeasuredEvent
    assert ContextUsageMeasuredEvent.kind == 'report_context_usage.measured'


def test_unknown_attribute_raises() -> None:

    with pytest.raises(AttributeError, match='has no attribute'):
        _ = compaction.DoesNotExist
