"""Shared fixtures for the sub-agents tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_agent_dirs(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the disk-loading convention roots at empty dirs for every test.

    `SubAgents` auto-loads from `./.agents/agents/` and `~/.agents/agents/` (with
    a `.claude/` fallback) by default. Redirecting cwd and home to fresh empty
    directories keeps tests that build `SubAgents` from reading the developer's real
    agent files. The two roots are distinct so the default project + home pair does
    not resolve to one folder. Tests that exercise loading pass explicit folders, or
    populate the redirected home root themselves."""
    home = tmp_path_factory.mktemp('home_root')
    monkeypatch.chdir(tmp_path_factory.mktemp('project_root'))

    def fake_home(cls: type[Path]) -> Path:
        return home

    monkeypatch.setattr(Path, 'home', classmethod(fake_home))
