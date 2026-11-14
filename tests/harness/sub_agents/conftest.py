"""Shared fixtures for the sub-agents tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_agent_dirs(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the disk-loading convention roots at empty dirs for every test.

    `SubAgents` reads agent folders through a workspace only. Redirecting home and cwd,
    neither of which it may read, to fresh empty directories lets tests put definitions
    there and assert they are ignored, without touching the developer's real files."""
    home = tmp_path_factory.mktemp('home_root')
    monkeypatch.chdir(tmp_path_factory.mktemp('project_root'))

    def fake_home(cls: type[Path]) -> Path:
        return home

    monkeypatch.setattr(Path, 'home', classmethod(fake_home))
