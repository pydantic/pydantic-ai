"""Tests for shared nested-agent usage helpers."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest
from pydantic_ai.usage import UsageLimits

from pydantic_ai_harness._usage import reserved_usage_limits


def test_reserved_usage_limits_reserves_one_request_and_preserves_other_limits() -> None:
    limits = UsageLimits(
        cost_limit=Decimal('0.01'),
        request_limit=1,
        tool_calls_limit=2,
        input_tokens_limit=3,
        output_tokens_limit=4,
        total_tokens_limit=5,
        per_request_input_tokens_limit=6,
        count_tokens_before_request=True,
    )

    assert reserved_usage_limits(limits) == replace(limits, request_limit=0)


def test_reserved_usage_limits_clamps_zero_request_limit() -> None:
    limits = UsageLimits(request_limit=0)

    assert reserved_usage_limits(limits) == limits


@pytest.mark.parametrize('limits', [None, UsageLimits(request_limit=None)])
def test_reserved_usage_limits_preserves_absent_or_unbounded_limits(limits: UsageLimits | None) -> None:
    assert reserved_usage_limits(limits) is limits
