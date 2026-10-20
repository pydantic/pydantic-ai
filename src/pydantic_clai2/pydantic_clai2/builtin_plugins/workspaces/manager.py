"""`Workspaces`: make, start, stop, and destroy workspaces of the configured kinds, keeping the registry current.

This is what the `/workspaces` command and the web UI's orchestrator both call. No token leaves it
but through `token`, for a client that talks to a workspace's CLAI.
"""

import os
import secrets
import sys
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import anyio
from anyio import to_thread

from pydantic_clai2.builtin_plugins.workspaces.backend import Backend, LocalWorktrees, is_url, repo_slug
from pydantic_clai2.builtin_plugins.workspaces.kinds import WorktreeKind
from pydantic_clai2.builtin_plugins.workspaces.registry import Registry, Workspace, workspaces_dir
from pydantic_clai2.builtin_plugins.workspaces.sessions import Launcher, read_token


def _lock_file(descriptor: int) -> None:
    """Wait for an exclusive lock on the open file; Windows, which runs no workspace sessions, needs none."""
    if sys.platform != 'win32':
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_EX)


def new_id(repo: str) -> str:
    """A short, readable, unique ID: the repository's name and six random hex digits."""
    return f'{repo_slug(repo)}-{secrets.token_hex(3)}'


class Workspaces:
    """The workspaces in the registry, and the kinds new ones can be made of."""

    def __init__(
        self,
        *,
        kinds: Mapping[str, WorktreeKind],
        directory: Path | None = None,
        launcher: Launcher | None = None,
    ) -> None:
        """`directory` holds the registry and clones; by default `workspaces` in CLAI's state directory."""
        directory = directory or workspaces_dir()
        self.kinds = dict(kinds)
        self.registry = Registry(directory / 'registry')
        self.backend: Backend = LocalWorktrees(launcher=launcher or Launcher(), clones=directory / 'clones')
        self._lock = anyio.Lock()

    async def listing(self) -> list[Workspace]:
        """Every workspace, oldest first, with the status of its CLAI checked now."""
        return [await self._current(workspace) for workspace in await to_thread.run_sync(self.registry.all)]

    async def get(self, workspace_id: str) -> Workspace:
        """One workspace, with the status of its CLAI checked now; `LookupError` for an unknown ID."""
        return await self._current(await to_thread.run_sync(self.registry.get, workspace_id))

    async def _current(self, workspace: Workspace) -> Workspace:
        session = workspace.session
        if workspace.status != 'running' or (session is not None and self.backend.alive(session)):
            return workspace
        return await self._save(replace(workspace, status='stopped'))

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
            status='ready',
            created=datetime.now(UTC),
            teardown=options.teardown,
            model=options.model,
        )
        return await self._save(workspace)

    @asynccontextmanager
    async def _exclusive(self, workspace_id: str) -> AsyncGenerator[None]:
        """Hold the workspace's lock, so every CLAI process starts, stops, and destroys it one at a time."""
        async with self._lock:
            await self.get(workspace_id)  # An unknown ID has no folder to lock.
            descriptor = os.open(self.registry.folder(workspace_id) / '.lock', os.O_RDWR | os.O_CREAT, 0o600)
            try:
                await to_thread.run_sync(_lock_file, descriptor)
                yield
            finally:
                os.close(descriptor)  # Releases the lock.

    async def start(self, workspace_id: str) -> Workspace:
        """Start the workspace's CLAI unless it runs already; `RuntimeError` when it does not start."""
        async with self._exclusive(workspace_id):
            workspace = await self.get(workspace_id)
            if workspace.status == 'running':
                return workspace
            session = await self.backend.start(workspace, folder=self.registry.folder(workspace_id))
            return await self._save(replace(workspace, status='running', session=session))

    async def stop(self, workspace_id: str) -> Workspace:
        """Stop the workspace's CLAI, if one runs."""
        async with self._exclusive(workspace_id):
            return await self._stop(workspace_id)

    async def _stop(self, workspace_id: str) -> Workspace:
        workspace = await self.get(workspace_id)
        if workspace.session is None:
            return workspace
        await self.backend.stop(workspace.session)
        return await self._save(replace(workspace, status='stopped'))

    async def destroy(self, workspace_id: str) -> Workspace:
        """Stop the workspace's CLAI, remove its checkout unless its teardown keeps it, and forget it.

        Raises `ValueError`, forgetting nothing, when the checkout is not where workspaces go or Git refuses.
        """
        async with self._exclusive(workspace_id):
            workspace = await self._stop(workspace_id)
            if workspace.teardown == 'remove':
                await self.backend.destroy(workspace)
            await to_thread.run_sync(self.registry.forget, workspace_id)
            return workspace

    async def token(self, workspace_id: str) -> str:
        """The token of the workspace's running CLAI, for a client; `RuntimeError` when none runs."""
        workspace = await self.get(workspace_id)
        problem = RuntimeError(f'Workspace {workspace_id} has no CLAI running. Start it first.')
        if workspace.status != 'running' or workspace.session is None:
            raise problem
        try:
            return await to_thread.run_sync(read_token, workspace.session)
        except FileNotFoundError:  # Stopped meanwhile, by this CLAI or another.
            raise problem from None
