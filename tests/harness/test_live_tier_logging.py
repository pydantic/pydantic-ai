from __future__ import annotations

import logging

import pytest

from .conftest import pytest_configure


@pytest.mark.parametrize(('markexpr', 'level'), [('daytona_live', logging.INFO), ('not slow', logging.NOTSET)])
def test_only_a_live_tier_captures_harness_info_logs(
    pytestconfig: pytest.Config, monkeypatch: pytest.MonkeyPatch, markexpr: str, level: int
) -> None:
    harness = logging.getLogger('pydantic_ai_harness')
    original = harness.level
    harness.setLevel(logging.NOTSET)
    monkeypatch.setattr(pytestconfig.option, 'markexpr', markexpr)
    try:
        pytest_configure(pytestconfig)
        assert harness.level == level
    finally:
        # `setLevel`, not a restored attribute, so the loggers' cached levels are cleared too.
        harness.setLevel(original)
