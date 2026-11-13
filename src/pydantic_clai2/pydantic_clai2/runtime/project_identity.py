"""Read-only repository identities for the session browsers, without changing saved workspaces."""

import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePath

from pydantic_ai_harness.step_persistence.conversations import ConversationSummary, GitCheckout

MISSING = '\0missing'
"""The project key gathering removed folders no known repository claims; a NUL byte cannot occur in a real path."""


@dataclass(frozen=True, kw_only=True)
class ProjectIdentity:
    """A local repository key, display name, checkout root, and checkout label."""

    key: str
    name: str
    root: str
    """The checkout's top-level folder, or the workspace itself outside Git; unique per checkout."""
    checkout: str = ''
    missing: bool = False
    """The folder is gone, as with a removed worktree."""


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
            # Git for Windows emits UTF-8; POSIX Git preserves ref bytes for Python's locale decoder.
            encoding='utf-8' if sys.platform == 'win32' else None,
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


def git_checkout(identity: ProjectIdentity) -> GitCheckout | None:
    """What a session saves about its workspace's checkout; `None` outside Git or once the folder is gone."""
    # Outside Git, and when Git fails, the key is the workspace itself, as is the root.
    if identity.missing or identity.key == identity.root:
        return None
    return GitCheckout(repository=identity.key, name=identity.name, root=identity.root, branch=identity.checkout)


def worktree_roots(identity: ProjectIdentity) -> list[str]:
    """The checkouts of the identity's repository that Git lists, the main one first; none outside Git.

    A bare repository's own folder is not a checkout, and one deleted without `git worktree remove` is not listed.
    """
    if git_checkout(identity) is None:
        return []
    try:
        listing = subprocess.run(
            # `-z` ends each field with NUL: Git's line delimiter is also legal inside a directory name.
            ['git', '-C', identity.root, 'worktree', 'list', '--porcelain', '-z'],
            check=True,
            capture_output=True,
            text=True,
            # Git for Windows emits UTF-8; POSIX Git preserves path bytes for Python's locale decoder.
            encoding='utf-8' if sys.platform == 'win32' else None,
            timeout=2,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    # Each checkout is a run of fields, the first `worktree PATH`, ended by an empty field.
    records = [record.split('\0') for record in listing.split('\0\0') if record]
    return [
        fields[0].removeprefix('worktree ')
        for fields in records
        if not any(field == 'bare' or field.startswith('prunable') for field in fields)
    ]


def placements(
    entries: Sequence[ConversationSummary], *, identities: Mapping[str, ProjectIdentity]
) -> dict[str, ProjectIdentity]:
    """Where each workspace in `identities`, as `project_identity` found it, belongs in a browser.

    A folder that still exists keeps its identity. One that is gone takes the checkout saved with
    its newest session, else the innermost known checkout that held it or folder of a known
    repository's worktrees (`_holder`); the rest gather under `MISSING`, each folder its own checkout there.
    """
    saved: dict[str, ProjectIdentity] = {}
    for entry in entries:
        if entry.checkout is not None and entry.workspace not in saved:
            found = entry.checkout
            saved[entry.workspace] = ProjectIdentity(
                key=found.repository, name=found.name, root=found.root, checkout=found.branch, missing=True
            )
    live = [identity for identity in identities.values() if git_checkout(identity) is not None]
    # Checkouts that still exist win over ones saved with sessions, which may be removed. Paths compare
    # as paths: Git for Windows writes `/` where the saved workspace has `\`.
    roots = {PurePath(identity.root): identity for identity in [*saved.values(), *live]}
    return {
        workspace: _place(workspace, identity, saved.get(workspace), roots)
        for workspace, identity in identities.items()
    }


def _place(
    workspace: str,
    identity: ProjectIdentity,
    saved: ProjectIdentity | None,
    roots: Mapping[PurePath, ProjectIdentity],
) -> ProjectIdentity:
    if not identity.missing:
        return identity
    if saved is not None:
        return roots[PurePath(saved.root)]
    path = PurePath(workspace)
    if (found := roots.get(path)) is not None:
        return found
    # The innermost known folder decides: a checkout the workspace was in, or a folder of worktrees.
    for folder, child in zip(path.parents, (path, *path.parents)):
        if (found := roots.get(folder)) is not None:
            return found
        if (found := _holder(folder, roots)) is not None:
            # Its branch is unknown, so the removed checkout is labelled with its folder's name.
            return ProjectIdentity(key=found.key, name=found.name, root=str(child), checkout=child.name, missing=True)
    return ProjectIdentity(
        key=MISSING, name='Removed folders', root=workspace, checkout=path.name or workspace, missing=True
    )


def _holder(folder: PurePath, roots: Mapping[PurePath, ProjectIdentity]) -> ProjectIdentity | None:
    """The known repository whose worktrees `folder` holds, by where tools put them.

    CLAI's `--worktree` uses `ROOT/.worktrees/NAME`; herdr and similar tools `.../worktrees/REPO/NAME`,
    or a folder named after the repository beside its other worktrees.
    """
    if folder.name == '.worktrees' and folder.parent in roots:
        return roots[folder.parent]
    return next(
        (
            identity
            for identity in roots.values()
            if identity.name == folder.name
            and (folder.parent.name == 'worktrees' or PurePath(identity.root).parent == folder)
        ),
        None,
    )


def _git_path(*, workspace: str, option: str) -> str:
    # Read paths separately: Git's line delimiter is also legal inside a directory name.
    return subprocess.run(
        ['git', '-C', workspace, 'rev-parse', '--path-format=absolute', option],
        check=True,
        capture_output=True,
        text=True,
        # Git for Windows emits UTF-8; POSIX Git preserves path bytes for Python's locale decoder.
        encoding='utf-8' if sys.platform == 'win32' else None,
        timeout=2,
    ).stdout.removesuffix('\n')
