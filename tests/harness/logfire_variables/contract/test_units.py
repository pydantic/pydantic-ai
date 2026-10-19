"""The one canonical setting that carries a unit, and what converting it is allowed to do."""

from __future__ import annotations

import math

import pytest

from pydantic_ai_harness.logfire._contract import (
    MAX_TIMEOUT_MILLISECONDS,
    MAX_TIMEOUT_SECONDS,
    AgentConfig,
    apply_settings,
    is_representable_timeout,
    to_milliseconds,
)


@pytest.mark.parametrize(
    ('seconds', 'milliseconds'),
    [
        (0, 0),
        (1, 1000),
        (0.25, 250),
        (2.5, 2500),
        # Half up, which is what JavaScript's `Math.round` does for a non-negative value and what
        # Python's own `round` does not: it rounds halves to even, so this would be 2.
        (0.0025, 3),
        # A positive budget is never rounded down into "already expired".
        (0.0005, 1),
        (0.0001, 1),
        (MAX_TIMEOUT_SECONDS, MAX_TIMEOUT_MILLISECONDS),
    ],
)
def test_seconds_become_the_integer_milliseconds_a_timer_wants(seconds: float, milliseconds: int) -> None:
    assert to_milliseconds(seconds) == milliseconds


@pytest.mark.parametrize('seconds', [-1, -0.001, math.inf, -math.inf, math.nan, MAX_TIMEOUT_SECONDS + 1, 1e30])
def test_a_budget_a_request_cannot_be_given_is_refused_rather_than_clamped(seconds: float) -> None:
    # Clamping would turn "no real limit" into a deadline nobody published, and rounding a negative
    # one to zero would cancel the request before it was sent.
    assert not is_representable_timeout(seconds)
    with pytest.raises(ValueError, match='not a representable request budget'):
        to_milliseconds(seconds)


def test_an_unrepresentable_published_timeout_is_reported_and_left_out_of_the_patch() -> None:
    published = AgentConfig.model_validate({'settings': {'timeout': -5, 'temperature': 0.4}})
    applied = apply_settings(published)
    assert applied.settings == {'temperature': 0.4}
    assert [(i.reason, i.setting) for i in applied.issues] == [('unrepresentable-timeout', 'timeout')]
    assert 'request timeout of -5.0 seconds, which is not a budget a request can be' in applied.issues[0].message


def test_a_representable_timeout_is_applied_like_any_other_setting() -> None:
    published = AgentConfig.model_validate({'settings': {'timeout': 30}})
    assert apply_settings(published).settings == {'timeout': 30.0}


def test_an_integer_too_large_to_be_a_float_is_refused_rather_than_raised() -> None:
    # The bound has to be tested before `math.isfinite`, which raises `OverflowError` on exactly the
    # oversized value this is here to refuse -- so refusing it must not be the thing that crashes.
    assert not is_representable_timeout(10**400)
