"""Tests for the deprecated `SlidingWindow` and `LimitWarner` aliases."""

from __future__ import annotations

import pytest

import pydantic_ai_harness.compaction as compaction
from pydantic_ai_harness import HarnessDeprecationWarning
from pydantic_ai_harness.compaction import SlidingWindowCompaction, WarnNearLimits


def test_sliding_window_alias_warns_and_resolves() -> None:
    with pytest.warns(
        HarnessDeprecationWarning, match='renamed to `pydantic_ai_harness.compaction.SlidingWindowCompaction`'
    ):
        from pydantic_ai_harness.compaction import SlidingWindow  # noqa: PLC0415  # importing is the assertion
    assert SlidingWindow is SlidingWindowCompaction


def test_limit_warner_alias_warns_and_resolves() -> None:
    with pytest.warns(HarnessDeprecationWarning, match='renamed to `pydantic_ai_harness.compaction.WarnNearLimits`'):
        from pydantic_ai_harness.compaction import LimitWarner  # noqa: PLC0415  # importing is the assertion
    assert LimitWarner is WarnNearLimits


def test_unknown_attribute_raises() -> None:

    with pytest.raises(AttributeError, match='has no attribute'):
        _ = compaction.DoesNotExist
