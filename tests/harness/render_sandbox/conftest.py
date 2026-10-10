"""Shared deterministic fixtures; no fixture opens a cloud resource."""

from pathlib import Path

import pytest

from ...conftest import try_import

with try_import():
    from .workspace_helpers import WorkspaceHarness, make_workspace_harness


@pytest.fixture
def workspace_harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> WorkspaceHarness:
    return make_workspace_harness(monkeypatch, tmp_path)


@pytest.fixture
def isolated_workspace_harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> WorkspaceHarness:
    return make_workspace_harness(monkeypatch, tmp_path, isolated_filesystems=True)
