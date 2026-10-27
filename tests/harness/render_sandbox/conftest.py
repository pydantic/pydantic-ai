"""Shared deterministic fixtures, and the gate for the live tier."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from ...conftest import try_import

with try_import():
    from .workspace_helpers import WorkspaceHarness, make_workspace_harness


@pytest.fixture
async def workspace_harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AsyncIterator[WorkspaceHarness]:
    harness = make_workspace_harness(monkeypatch, tmp_path)
    yield harness
    await harness.aclose()


@pytest.fixture
async def isolated_workspace_harness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> AsyncIterator[WorkspaceHarness]:
    harness = make_workspace_harness(monkeypatch, tmp_path, isolated_filesystems=True)
    # The conformance suite's commands, including the destroy-during-command rule, run under the shipped supervisor.
    harness.supervised = True
    yield harness
    await harness.aclose()


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Gate the live tier, before any live fixture can start a sandbox.

    It runs only with `PYDANTIC_AI_HARNESS_RENDER_LIVE=1`, `RENDER_API_KEY` and `RENDER_WORKSPACE_ID`.
    Otherwise it skips, unless `RENDER_REQUIRE_LIVE=1`, where missing credentials fail instead: a live
    job that tested nothing must not pass as a silent skip.
    """
    if item.get_closest_marker('render_live') is None:
        return
    if (
        os.getenv('PYDANTIC_AI_HARNESS_RENDER_LIVE') == '1'
        and os.getenv('RENDER_API_KEY')
        and os.getenv('RENDER_WORKSPACE_ID')
    ):  # pragma: no cover - live job
        return
    reason = 'requires PYDANTIC_AI_HARNESS_RENDER_LIVE=1, RENDER_API_KEY and RENDER_WORKSPACE_ID'
    if os.getenv('RENDER_REQUIRE_LIVE') == '1':  # pragma: no cover - live job
        pytest.fail(f'RENDER_REQUIRE_LIVE=1 but the live tier cannot run: it {reason}.')
    pytest.skip(reason)
