"""Read-only repository identities for the session browser, without changing saved workspaces."""

import subprocess
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True, kw_only=True)
class ProjectIdentity:
    """A local repository key, display name, checkout root, and checkout label."""

    key: str
    name: str
    root: str
    """The checkout's top-level folder, or the workspace itself outside Git; unique per checkout."""
    checkout: str = ''
    missing: bool = False
    """The directory is gone, as with a deleted worktree, so its repository is unknown."""


def project_identity(workspace: str) -> ProjectIdentity:
    """Resolve existing Git checkouts; unavailable directories retain their workspace identity."""
    path = Path(workspace)
    fallback = ProjectIdentity(key=workspace, name=path.name or workspace, root=workspace)
    # Relative historical workspaces must not be interpreted against today's launch directory.
    if not path.is_absolute():
        return fallback
    try:
        if not path.exists():
            return replace(fallback, missing=True)
        root = _git_path(workspace=workspace, option='--show-toplevel')
        common = _git_path(workspace=workspace, option='--git-common-dir')
        common_path = Path(common).resolve()
        branch = subprocess.run(
            ['git', '-C', workspace, 'symbolic-ref', '--quiet', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return fallback
    return ProjectIdentity(
        key=str(common_path),
        name=common_path.parent.name if common_path.name == '.git' else common_path.name,
        root=root,
        checkout=branch or f'{Path(root).name} (detached)',
    )


def _git_path(*, workspace: str, option: str) -> str:
    # Read paths separately: Git's line delimiter is also legal inside a directory name.
    return subprocess.run(
        ['git', '-C', workspace, 'rev-parse', '--path-format=absolute', option],
        check=True,
        capture_output=True,
        text=True,
        timeout=2,
    ).stdout.removesuffix('\n')
