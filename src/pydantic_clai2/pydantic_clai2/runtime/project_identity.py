"""Read-only repository identities for the session browser, without changing saved workspaces."""

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, kw_only=True)
class ProjectIdentity:
    """A local repository key, display name, and checkout label."""

    key: str
    name: str
    checkout: str = ''


def project_identity(workspace: str) -> ProjectIdentity:
    """Resolve existing Git checkouts; unavailable directories retain their workspace identity."""
    path = Path(workspace)
    fallback = ProjectIdentity(key=workspace, name=path.name or workspace)
    # Relative historical workspaces must not be interpreted against today's launch directory.
    if not path.is_absolute():
        return fallback
    try:
        result = subprocess.run(
            ['git', '-C', workspace, 'rev-parse', '--path-format=absolute', '--show-toplevel', '--git-common-dir'],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        root, common = result.stdout.splitlines()
        common_path = Path(common).resolve()
        branch = subprocess.run(
            ['git', '-C', workspace, 'symbolic-ref', '--quiet', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError, ValueError):
        return fallback
    return ProjectIdentity(
        key=str(common_path),
        name=common_path.parent.name if common_path.name == '.git' else common_path.name,
        checkout=branch or f'{Path(root).name} (detached)',
    )
