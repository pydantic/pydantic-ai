"""The harness suite runs under the autouse blocking-call detector, like the rest of `tests/`."""

from __future__ import annotations


def test_the_harness_suite_runs_under_the_blocking_call_detector(blockbuster_enabled: bool) -> None:
    # Directories and modules that block by design turn it off at their own boundary, never for the whole suite.
    assert blockbuster_enabled
