"""Locate (not parse) a repo's coding-assistant CE assets."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field

_ROOT_NOTES = {
    '.codex': 'Codex uses TOML config; assets are derived from the .claude/.agents setup.',
    '.grok': 'Grok setup is derived from the .claude/.agents setup.',
}


class AssetRoot(BaseModel):
    """Where CE assets live under a single root directory (e.g. `.claude`)."""

    root: str = Field(description='The root directory name, relative to the workspace, e.g. ".claude".')
    exists: bool = Field(description='Whether the root directory is present in the workspace.')
    skills: list[str] = Field(default_factory=list, description='Paths to SKILL.md files found under skills/.')
    agents: list[str] = Field(default_factory=list, description='Paths to agent .md files found under agents/.')
    settings: str | None = Field(default=None, description='Path to settings.json (hooks), if present.')
    notes: str | None = Field(default=None, description='Format or derivation notes for this root, if any.')


class AgentContextInventory(BaseModel):
    """A map of where a repo's CE assets live, for an orchestrator to read or translate."""

    roots: list[AssetRoot] = Field(default_factory=list[AssetRoot], description='One entry per scanned root directory.')


def _relposix(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace).as_posix()
    except ValueError:
        return path.as_posix()


def scan_assets(workspace_dir: Path, asset_roots: Sequence[str]) -> AgentContextInventory:
    """Scan `asset_roots` under `workspace_dir`, locating skills, agents, and hooks.

    This locates assets only; it does not open or parse SKILL.md, agent `.md`, or
    `settings.json` contents.
    """
    workspace = workspace_dir.resolve()
    roots: list[AssetRoot] = []
    for name in asset_roots:
        directory = workspace / name
        notes = _ROOT_NOTES.get(name)
        if not directory.is_dir():
            roots.append(AssetRoot(root=name, exists=False, notes=notes))
            continue
        skills = sorted(_relposix(p, workspace) for p in directory.glob('skills/**/SKILL.md') if p.is_file())
        agents = sorted(_relposix(p, workspace) for p in directory.glob('agents/*.md') if p.is_file())
        settings_path = directory / 'settings.json'
        settings = _relposix(settings_path, workspace) if settings_path.is_file() else None
        roots.append(AssetRoot(root=name, exists=True, skills=skills, agents=agents, settings=settings, notes=notes))
    return AgentContextInventory(roots=roots)
