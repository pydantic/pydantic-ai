from __future__ import annotations

import subprocess

import pytest


@pytest.fixture(scope='package')
def started_node() -> None:
    """Start `node` once before the first test here that runs the gh-aw harness script.

    The first `node` start on a fresh CI runner can take several seconds. The per-test budget does not
    count shared fixture setup, so paying that cost here keeps it off whichever test each worker happens
    to run first. Package-scoped, so it covers every module that runs `node` (each requests it with
    `pytestmark`), not only the first one a worker collects.
    """
    # Bounded only against a hang: far longer than any cold start or runner stall.
    subprocess.run(['node', '-e', ''], check=True, timeout=300)
