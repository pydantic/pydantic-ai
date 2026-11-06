"""The `pydantic-clai2` suite runs under the autouse blocking-call detector, like the rest of `tests/`."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import pydantic_clai2
from tests import conftest


def test_the_clai2_suite_runs_under_the_blocking_call_detector(blockbuster_enabled: bool) -> None:
    # Code that blocks by design is exempted call by call in `tests/conftest.py`, never for the whole suite.
    assert blockbuster_enabled


async def test_blocking_calls_in_clai2_code_are_reported() -> None:
    if os.getenv('BLOCKBUSTER_ENABLED') == 'false':
        pytest.skip('BlockBuster is disabled in this CI lane')
    from blockbuster import BlockingError

    # Code compiled under a file name inside the package runs as one of its frames.
    probe = compile('os.stat(".")', str(Path(pydantic_clai2.__file__).parent / '_blocking_probe.py'), 'exec')
    bb = conftest._configure_blockbuster()  # pyright: ignore[reportPrivateUsage]
    bb.activate()
    try:
        with pytest.raises(BlockingError):
            exec(probe, {'os': os})
    finally:
        bb.deactivate()
