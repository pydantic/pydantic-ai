"""Shared fixtures for DockerSandbox tests."""

from __future__ import annotations

import os

import pytest


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Gate the live tier, which starts real containers: it needs a Docker daemon.

    It runs only with `PYDANTIC_AI_HARNESS_DOCKER_LIVE=1`. Otherwise it skips, unless `DOCKER_REQUIRE_LIVE=1`
    (CI's live job sets it), where it fails instead: a live job that tested nothing must not pass as a skip.
    """
    if item.get_closest_marker('docker_live') is None:
        return
    if os.getenv('PYDANTIC_AI_HARNESS_DOCKER_LIVE') == '1':  # pragma: no cover - live tier
        return
    reason = 'requires PYDANTIC_AI_HARNESS_DOCKER_LIVE=1 with a Docker daemon'
    if os.getenv('DOCKER_REQUIRE_LIVE') == '1':  # pragma: no cover - live job
        pytest.fail(f'DOCKER_REQUIRE_LIVE=1 but the live tier cannot run: it {reason}.')
    pytest.skip(reason)
