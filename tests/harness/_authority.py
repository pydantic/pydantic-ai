"""Workspaces whose files are below `'operator'`, for tests of where that text can reach the model."""

from __future__ import annotations

from pathlib import Path

from pydantic_ai.workspaces import AuthorityWorkspace, LocalWorkspaceBackend, Workspace


def untrusted(path: Path) -> AuthorityWorkspace:
    """A workspace over `path` whose every file is at `'none'`, like a repository the operator doesn't control."""
    return AuthorityWorkspace(Workspace(LocalWorkspaceBackend(path)), authority='none')
