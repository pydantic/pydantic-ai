"""What every kind of workspace implements, and the `local-worktree` backend: a Git worktree on this machine."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote

from anyio import to_thread

from pydantic_clai2.builtin_plugins.workspaces.kinds import WorktreeKind
from pydantic_clai2.builtin_plugins.workspaces.registry import Workspace
from pydantic_clai2.runtime.worktrees import add_worktree, clone_or_fetch, commit_of, remove_worktree, repository_root

_SCP_URL = re.compile(r'[\w.-]+@[\w.-]+:')
"""Git's scp-like syntax, `git@github.com:owner/repo.git`."""


def is_url(repo: str) -> bool:
    """Whether `repo` is a URL to clone rather than a local path."""
    return '://' in repo or _SCP_URL.match(repo) is not None


def repo_slug(repo: str) -> str:
    """The repository's name in lowercase letters, digits, and hyphens, from the last part of its path or URL."""
    name = re.split(r'[/:\\]', repo.rstrip('/\\'))[-1].removesuffix('.git')
    return re.sub(r'[^a-z0-9]+', '-', unquote(name).lower()).strip('-')[:32] or 'repo'


@dataclass(frozen=True, kw_only=True)
class Checkout:
    """Where a new workspace's files are."""

    path: Path
    root: Path
    """The folder its checkout was made in."""
    branch: str


class Backend(Protocol):
    """What a kind of workspace does: make the files, and remove them again.

    The local Git worktree is the first. Its sessions run in this CLAI, in the checkout, as any other does.
    """

    async def create(self, *, workspace_id: str, kind: WorktreeKind, repo: str, ref: str) -> Checkout:
        """Make the workspace's files: `repo` checked out from `ref` on a new branch; `ValueError` when that fails."""
        ...

    async def destroy(self, workspace: Workspace) -> None:
        """Remove the workspace's files; `ValueError` when they are not where a workspace's files go."""
        ...


@dataclass(kw_only=True)
class LocalWorktrees:
    """`local-worktree`: a linked Git worktree of a local repository, or of a clone CLAI keeps under `clones`."""

    clones: Path

    async def create(self, *, workspace_id: str, kind: WorktreeKind, repo: str, ref: str) -> Checkout:
        remote = is_url(repo)
        if remote:
            digest = hashlib.sha256(repo.encode()).hexdigest()[:10]
            source = self.clones / f'{repo_slug(repo)}-{digest}'
            await to_thread.run_sync(lambda: clone_or_fetch(url=repo, path=source))
        else:
            source = await to_thread.run_sync(repository_root, Path(repo).expanduser().resolve())
        commit = await to_thread.run_sync(lambda: commit_of(repo=source, ref=ref, remote=remote))
        root = Path(kind.root).expanduser() if kind.root else source / '.worktrees'
        path = root / workspace_id
        if path.exists():
            raise ValueError(f'{path} already exists.')
        branch = f'{kind.branch_prefix}{workspace_id}'

        def check_out() -> None:
            root.mkdir(parents=True, exist_ok=True)
            add_worktree(repo=source, path=path, branch=branch, base=commit)

        await to_thread.run_sync(check_out)
        return Checkout(path=path.resolve(), root=root.resolve(), branch=branch)

    async def destroy(self, workspace: Workspace) -> None:
        path, root = workspace.path.resolve(), workspace.root.resolve()
        if path.parent != root or path.name != workspace.id:
            raise ValueError(f'Refusing to remove {path}: it is not the folder of workspace {workspace.id} in {root}.')
        if path.exists():
            await to_thread.run_sync(lambda: remove_worktree(path=path))
