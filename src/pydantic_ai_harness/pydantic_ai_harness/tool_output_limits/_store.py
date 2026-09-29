"""Storage backends for spilled tool outputs.

`WorkspaceStore` is the default: it writes each payload to a file inside the run's workspace
(`ctx.workspace`), or a workspace of its own, so with a remote sandbox the spilled files live in
the sandbox, next to the files the agent's other tools see. `OverflowStore` is the narrow
protocol for any other backend (a blob store, a durable engine's storage): persist a payload
under a key, read it back by handle. `LocalFileStore` implements it on the host filesystem.
"""

from __future__ import annotations

import os
import posixpath
import re
import stat
import tempfile
import threading
import time
import warnings
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Protocol, runtime_checkable

import anyio.to_thread

from pydantic_ai.exceptions import UserError
from pydantic_ai.workspaces import Workspace, WorkspaceBackend
from pydantic_ai.workspaces.unavailable import NO_WORKSPACE
from pydantic_ai_harness._workspace import METADATA_DIR, is_unattached, metadata_dir, secondary_workspace


@runtime_checkable
class OverflowStore(Protocol):
    """Persist and retrieve spilled tool-output payloads outside the run's workspace.

    `write` takes a caller-chosen `key` and returns a `handle`. The handle is the only
    thing a later `read` needs, so it must be self-contained (a backend can encode the
    run, the call, and the retry into it). Implementations may return the key unchanged.
    """

    async def write(self, key: str, data: bytes) -> str:
        """Persist `data` under `key` and return a handle that `read` accepts."""
        ...  # pragma: no cover

    async def read(self, handle: str) -> bytes:
        """Return the payload previously stored for `handle`.

        Raise `FileNotFoundError` (or another `OSError`) when the handle is unknown.
        """
        ...  # pragma: no cover


_UNSAFE_SEGMENT = re.compile(r'[^A-Za-z0-9._-]+')


def _safe_segment(segment: str) -> str:
    """Make one path segment filesystem-safe without collapsing distinct keys.

    Empty or dot-only segments are replaced so a handle can never escape the root via
    `.`/`..`; the resolve-within-root check in `read` is the second line of defense.
    """
    cleaned = _UNSAFE_SEGMENT.sub('_', segment)
    if cleaned in ('', '.', '..'):
        return '_'
    return cleaned


def _segments(key: str) -> list[str]:
    return [_safe_segment(part) for part in key.split('/') if part] or ['_']


@dataclass
class WorkspaceStore:
    """The default store: each spilled payload is a file inside the run's workspace.

    The handle is the file's absolute workspace path. Files live under
    `.pydantic-ai-harness/tool-output` in the workspace's working directory, which holds a
    `.gitignore` so they stay out of version control, and the model can open them with
    `read_tool_result` or with `FileSystem`'s `read_file`, since they are inside the working
    directory.

    Files are not pruned: they go away with the sandbox, or when the directory is deleted for a
    local workspace. Set `workspace` to keep them in a workspace other than the run's.
    """

    workspace: WorkspaceBackend | None = None
    """A workspace to keep spills in instead of the run's, such as `LocalWorkspaceBackend('/var/spills')`.

    A backend, not the `LocalWorkspace` capability. It is used in-process only in this release: a
    durable engine does not route it through its workflow machinery.
    """

    _workspace: Workspace | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._workspace = secondary_workspace(self.workspace, 'WorkspaceStore')

    def bind(self, workspace: Workspace) -> OverflowStore:
        """Bind a run workspace to the common store interface."""
        return _BoundWorkspaceStore(self, workspace)

    def _target(self, workspace: Workspace) -> Workspace:
        workspace = self._workspace or workspace
        # With no workspace attached nothing can be stored: `UserError` lets callers warn or guide
        # the model, where core's placeholder would raise `WorkspaceUnavailableError` and end the run.
        if is_unattached(workspace):
            raise UserError(NO_WORKSPACE.reason)
        return workspace

    async def write(self, workspace: Workspace, key: str, data: bytes) -> str:
        """Write `data` and return its absolute workspace path as the handle.

        `workspace` is the run's; the store's own `workspace`, when set, is used instead.
        """
        workspace = self._target(workspace)
        path = posixpath.join(await metadata_dir(workspace, 'tool-output'), *_segments(key))
        await workspace.write_bytes(await _confine(workspace, path, key), data)
        return path

    async def read(self, workspace: Workspace, handle: str) -> bytes:
        """Read a payload back; a handle outside the store directory raises `PermissionError`."""
        workspace = self._target(workspace)
        directory = await metadata_dir(workspace, 'tool-output')
        path = posixpath.normpath(posixpath.join(directory, handle))
        if not path.startswith(directory.rstrip('/') + '/'):
            raise PermissionError(f'Handle {handle!r} is outside the store directory.')
        return await workspace.read_bytes(await _confine(workspace, path, handle))


@dataclass
class _BoundWorkspaceStore:
    store: WorkspaceStore
    workspace: Workspace

    async def write(self, key: str, data: bytes) -> str:
        return await self.store.write(self.workspace, key, data)

    async def read(self, handle: str) -> bytes:
        return await self.store.read(self.workspace, handle)


async def _confine(workspace: Workspace, path: str, name: str) -> str:
    """Return `path` with symlinks resolved, refusing one that leaves the store directory.

    The store directory is resolved from the working directory, so a symlink in place of the
    directory itself (or of `.pydantic-ai-harness`) is refused too, not only one inside it.
    """
    real_working_dir = await workspace.realpath(await workspace.working_dir())
    real_directory = posixpath.join(real_working_dir, METADATA_DIR, 'tool-output')
    real_path = await workspace.realpath(path)
    if not real_path.startswith(real_directory + '/'):
        raise PermissionError(f'Handle {name!r} is outside the store directory.')
    return real_path


def _default_root() -> Path:
    """Per-user root under the system temp dir, so users on one host never share it."""
    geteuid = getattr(os, 'geteuid', None)
    name = 'pyai_harness_overflow' if geteuid is None else f'pyai_harness_overflow-{geteuid()}'
    return Path(tempfile.gettempdir()) / name


@dataclass
class LocalFileStore:
    """`OverflowStore` that writes each payload to a file on the host running the agent.

    Pass it as `store=` to keep spills on this machine, for instance for a run with no workspace;
    with a workspace, the default `WorkspaceStore` keeps spills where the agent's other tools can
    see them.

    The handle equals the key: a relative `run_id/tool_call_id.retry` path under
    `base_dir`. The root is stable on purpose -- a later agent or run by the same user can
    read a spill a previous run produced, so the store is not isolated per instance.

    Security comes from three mechanisms: the default root is per user
    (`pyai_harness_overflow-<euid>` under the system temp dir); the root itself cannot be a
    symlink, and on POSIX it must be owned by the current user and is tightened to `0700` if
    group or other bits are set; and `read` resolves the target (following symlinks) and
    rejects anything that escapes the root via symlink, `..`, or an absolute path. Handle
    segments are also sanitized by `_safe_segment`. On Windows there is no uid, so the
    default root is `pyai_harness_overflow` and the ownership check is skipped.

    Files are kept after the run by default (a later `read_tool_result` may need them).
    Set `cleanup_after` to opt into age-based pruning; see that field.
    """

    base_dir: Path | None = None
    """Root directory for spilled files. Defaults to a per-user temp subdirectory.

    An explicit `base_dir` gets the same ownership check as the default root.
    """

    cleanup_after: timedelta | None = None
    """Opt-in TTL for spilled files. `None` (default) keeps files forever.

    When set, a `write` schedules a background prune (a daemon thread, off the hot path)
    that deletes files whose modification time is older than `cleanup_after`. Pruning is
    best-effort: any failure is caught and surfaced via `warnings.warn`, never propagated
    into the agent run. Modification time (`st_mtime`) is the age signal; last-read time
    (`st_atime`) is unreliable on `noatime`/`relatime` mounts and is not used.
    """

    _root: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._root = self.base_dir if self.base_dir is not None else _default_root()

    def _path(self, key: str) -> Path:
        return self._root.joinpath(*_segments(key))

    def _ensure_root(self) -> None:
        """Create the root and make sure only the current user can reach it.

        A root that already exists may have been created by someone else (the default
        lives in the shared temp dir), so ownership is checked rather than assumed. A root
        this call creates gets `0700` at creation, so there is no window before the `chmod`
        in which another user could plant an entry in it.
        """
        self._root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._check_root()

    def _check_root(self) -> None:
        """Refuse a symlinked root, then enforce POSIX ownership and permissions."""
        st = self._root.lstat()
        if stat.S_ISLNK(st.st_mode):
            raise PermissionError(f'Overflow store root {str(self._root)!r} is a symbolic link; refusing to use it.')
        geteuid = getattr(os, 'geteuid', None)
        if geteuid is None:
            return
        if st.st_uid != geteuid():
            raise PermissionError(
                f'Overflow store root {str(self._root)!r} is not owned by the current user; '
                'refusing to write spilled tool output there.'
            )
        if stat.S_IMODE(st.st_mode) & 0o077:
            self._root.chmod(0o700)

    async def write(self, key: str, data: bytes) -> str:
        # Disk I/O runs in a worker thread so it never blocks the event loop.
        await anyio.to_thread.run_sync(self._write, key, data)
        self._schedule_cleanup()
        return key

    def _write(self, key: str, data: bytes) -> None:
        self._ensure_root()
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    async def read(self, handle: str) -> bytes:
        return await anyio.to_thread.run_sync(self._read, handle)

    def _read(self, handle: str) -> bytes:
        self._check_root()
        target = self._path(handle).resolve()
        root = self._root.resolve()
        if not target.is_relative_to(root):
            raise PermissionError(f'Handle {handle!r} resolves outside the store root.')
        return target.read_bytes()

    # --- opt-in TTL pruning (non-blocking, non-erroring) ---

    def _schedule_cleanup(self) -> threading.Thread | None:
        """Fire a background prune when `cleanup_after` is set. Never blocks `write`."""
        if self.cleanup_after is None:
            return None
        thread = threading.Thread(target=self._run_prune, name='overflow-prune', daemon=True)
        thread.start()
        return thread

    def _run_prune(self) -> None:
        try:
            self._prune_sync()
        except Exception as exc:  # never let cleanup fail a run or block the hot path
            warnings.warn(f'LocalFileStore cleanup failed: {exc}', stacklevel=2)

    def _prune_sync(self) -> None:
        """Delete files older than `cleanup_after` (by `st_mtime`)."""
        assert self.cleanup_after is not None
        self._check_root()
        cutoff = time.time() - self.cleanup_after.total_seconds()
        for path in self._root.rglob('*'):
            if not path.is_file():
                continue
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:  # pragma: no cover - file vanished mid-prune
                continue
