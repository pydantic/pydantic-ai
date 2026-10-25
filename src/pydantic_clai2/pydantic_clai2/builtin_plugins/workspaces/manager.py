"""`Workspaces`: make, list, and destroy workspaces of the configured kinds, keeping the registry current.

This is what the `/workspaces` command and the web UI's orchestrator both call. A workspace's sessions run
in the CLAI that opens them, in its checkout, as any other session does.
"""

import secrets
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from anyio import to_thread

from pydantic_clai2.builtin_plugins.workspaces.backend import Backend, LocalWorktrees, is_url, repo_slug
from pydantic_clai2.builtin_plugins.workspaces.kinds import WorktreeKind
from pydantic_clai2.builtin_plugins.workspaces.registry import Registry, Workspace, workspaces_dir
from pydantic_clai2.plugins import LiveSessions


def new_id(repo: str) -> str:
    """A short, readable, unique ID: the repository's name and six random hex digits."""
    return f'{repo_slug(repo)}-{secrets.token_hex(3)}'


def works_in(directory: Path | str, folder: Path) -> bool:
    """Whether a session working in `directory` works in `folder` or below it."""
    return Path(directory).resolve().is_relative_to(folder)


async def close_sessions_in(live: LiveSessions, folder: Path) -> None:
    """Close each live session working in `folder` or below it; each opens again from its saved state.

    Raises `ValueError`, closing none, when the session asking is one of them, since closing it would cancel
    the asking itself, and `ValueError` for one only its own driver closes, such as the terminal's.
    """
    inside = [session for session in live.listing() if works_in(session.workspace, folder)]
    current = live.current()
    if current is not None and any(session.id == current.id for session in inside):
        raise ValueError(f'This session works in {folder}. Run this from a session outside it.')
    for session in inside:
        await live.close(session.id)


class Workspaces:
    """The workspaces in the registry, and the kinds new ones can be made of."""

    def __init__(self, *, kinds: Mapping[str, WorktreeKind], directory: Path | None = None) -> None:
        """`directory` holds the registry and clones; by default `workspaces` in CLAI's state directory."""
        directory = directory or workspaces_dir()
        self.kinds = dict(kinds)
        self.registry = Registry(directory / 'registry')
        self.backend: Backend = LocalWorktrees(clones=directory / 'clones')

    async def listing(self) -> list[Workspace]:
        """Every workspace, oldest first."""
        return await to_thread.run_sync(self.registry.all)

    async def get(self, workspace_id: str) -> Workspace:
        """One workspace; `LookupError` for an unknown ID."""
        return await to_thread.run_sync(self.registry.get, workspace_id)

    async def _save(self, workspace: Workspace) -> Workspace:
        await to_thread.run_sync(self.registry.save, workspace)
        return workspace

    async def create(self, *, kind: str, repo: str = '', ref: str = '', base: Path | None = None) -> Workspace:
        """Check out `repo` at `ref` as a new workspace of `kind`; the kind's own repository and ref fill in blanks.

        A relative local `repo` is in `base`, the asking session's directory, by default the current one.

        Raises `LookupError` for an unknown kind and `ValueError` when the checkout cannot be made.
        """
        options = self.kinds.get(kind)
        if options is None:
            raise LookupError(f'No workspace kind {kind!r}. Kinds: {", ".join(sorted(self.kinds)) or "none"}.')
        repo, ref = repo or options.repo, ref or options.ref
        if not repo:
            raise ValueError(f'Name a repository: kind {kind!r} has none of its own.')
        if base is not None and not is_url(repo):
            repo = str(base / Path(repo).expanduser())
        workspace_id = new_id(repo)
        checkout = await self.backend.create(workspace_id=workspace_id, kind=options, repo=repo, ref=ref)
        workspace = Workspace(
            id=workspace_id,
            kind=kind,
            backend=options.backend,
            repo=repo,
            ref=ref,
            branch=checkout.branch,
            path=checkout.path,
            root=checkout.root,
            created=datetime.now(UTC),
            teardown=options.teardown,
        )
        return await self._save(workspace)

    async def attach(self, workspace_id: str, conversation: str) -> Workspace:
        """Remember `conversation` as the session the orchestrator runs in the workspace."""
        return await self._save(replace(await self.get(workspace_id), conversation=conversation))

    async def destroy(self, workspace_id: str, *, live: LiveSessions | None = None) -> Workspace:
        """Remove the workspace's checkout unless its teardown keeps it, and forget it.

        Before a checkout is removed, each of `live`'s sessions working in it closes, so none is left in a folder
        that is gone (see `close_sessions_in`). Raises `ValueError`, forgetting nothing, when one of them cannot
        close from here, when the checkout is not where workspaces go, or when Git refuses.
        """
        workspace = await self.get(workspace_id)
        if workspace.teardown == 'remove':
            if live is not None:
                await close_sessions_in(live, workspace.path)
            await self.backend.destroy(workspace)
        await to_thread.run_sync(self.registry.forget, workspace_id)
        return workspace
