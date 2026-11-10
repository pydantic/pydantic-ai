"""The `pydantic-clai2` suite runs under the autouse blocking-call detector, like the rest of `tests/`."""

from __future__ import annotations

from tests import conftest


def test_the_clai2_suite_runs_under_the_blocking_call_detector(blockbuster_enabled: bool) -> None:
    # Code that blocks by design is exempted call by call in `tests/conftest.py`, never for the whole suite.
    assert blockbuster_enabled


def test_the_detector_scans_clai2() -> None:
    assert 'pydantic_clai2' in conftest._blockbuster_packages()  # pyright: ignore[reportPrivateUsage]
