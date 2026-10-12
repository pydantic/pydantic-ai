"""Locate (not parse) a repo's coding-assistant CE assets."""

from __future__ import annotations

import posixpath
from collections import deque
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from pydantic_ai.workspaces import FileEntry, Workspace

# Directories are deduplicated by their resolved path, so symlink cycles and aliases are walked once;
# this bound backs that up on backends that cannot resolve links. Real skill trees are two levels deep.
_MAX_SKILL_DEPTH = 8


class AssetRoot(BaseModel):
    """Where CE assets live under a single root directory (e.g. `.claude`)."""

    root: str = Field(description='The root directory name, relative to the workspace, e.g. ".claude".')
    exists: bool = Field(description='Whether the root directory is present in the workspace.')
    skills: list[str] = Field(default_factory=list, description='Paths to SKILL.md files found under skills/.')
    agents: list[str] = Field(default_factory=list, description='Paths to agent .md files found under agents/.')
    settings: str | None = Field(default=None, description='Path to settings.json (hooks), if present.')


class AgentContextInventory(BaseModel):
    """A map of where a repo's CE assets live, for an orchestrator to inspect."""

    roots: list[AssetRoot] = Field(default_factory=list[AssetRoot], description='One entry per scanned root directory.')


async def scan_assets(workspace: Workspace, workspace_dir: Path, asset_roots: Sequence[str]) -> AgentContextInventory:
    """Scan `asset_roots` under `workspace_dir`, locating skills, agents, and hooks.

    This locates assets only; it does not open or parse SKILL.md, agent `.md`, or
    `settings.json` contents.
    """
    root_dir = await workspace.resolve(workspace_dir.as_posix())
    roots: list[AssetRoot] = []
    for name in asset_roots:
        directory = posixpath.normpath(posixpath.join(root_dir, name))
        try:
            entry = await workspace.stat(directory)
        except FileNotFoundError:
            roots.append(AssetRoot(root=name, exists=False))
            continue
        if not entry.is_dir:
            roots.append(AssetRoot(root=name, exists=False))
            continue

        skills = await _scan_skills(workspace, posixpath.join(directory, 'skills'), root_dir)
        agents = await _scan_agents(workspace, posixpath.join(directory, 'agents'), root_dir)
        settings_path = posixpath.join(directory, 'settings.json')
        settings_entry = await _stat(workspace, settings_path)
        settings = (
            _relative(settings_path, root_dir) if settings_entry is not None and not settings_entry.is_dir else None
        )
        skills.sort()
        agents.sort()
        roots.append(AssetRoot(root=name, exists=True, skills=skills, agents=agents, settings=settings))
    return AgentContextInventory(roots=roots)


async def _scan_skills(workspace: Workspace, skills_root: str, root_dir: str) -> list[str]:
    root = await _stat(workspace, skills_root)
    if root is None or not root.is_dir:
        return []

    found: list[str] = []
    seen: set[str] = set()
    pending = deque([(skills_root, 0)])
    while pending:
        directory, depth = pending.popleft()
        # Entries follow directory symlinks, so a link to `.` would otherwise multiply the walk at every level.
        real = await workspace.realpath(directory)
        if real in seen:
            continue
        seen.add(real)
        # Defensive race: the directory may disappear after `_stat`.
        try:
            entries = await workspace.list_dir(directory)
        except FileNotFoundError:  # pragma: no cover
            continue
        for entry in entries:
            if entry.is_dir:
                if depth < _MAX_SKILL_DEPTH:
                    pending.append((entry.path, depth + 1))
            elif entry.name == 'SKILL.md':
                found.append(_relative(entry.path, root_dir))
    return found


async def _scan_agents(workspace: Workspace, agents_root: str, root_dir: str) -> list[str]:
    root = await _stat(workspace, agents_root)
    if root is None or not root.is_dir:
        return []
    # Defensive race: the directory may disappear after `_stat`.
    try:
        entries = await workspace.list_dir(agents_root)
    except FileNotFoundError:  # pragma: no cover
        return []
    return [_relative(entry.path, root_dir) for entry in entries if not entry.is_dir and entry.name.endswith('.md')]


async def _stat(workspace: Workspace, path: str) -> FileEntry | None:
    try:
        return await workspace.stat(path)
    except FileNotFoundError:
        return None


def _relative(path: str, workspace: str) -> str:
    return posixpath.relpath(path, workspace)
