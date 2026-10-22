"""Mounting storage, such as a bucket, at a folder in a workspace."""

from __future__ import annotations

import posixpath
from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from .protocol import SupportsFilesystem, WorkspaceBackend, WorkspaceError

__all__ = ('Mount', 'MountError', 'SupportsMounts')


class MountError(WorkspaceError):
    """A mount that shell commands were promised could not be made visible to them."""


@dataclass(frozen=True)
class Mount:
    """Storage that appears as a folder in a workspace.

    File tools reach it on every workspace. Shell commands see it too when the workspace's backend can mount it (see
    [`SupportsMounts`][pydantic_ai.workspaces.SupportsMounts]); otherwise the folder is reachable through file tools only.
    """

    path: str
    """Where the folder appears: relative to the working directory, or absolute."""

    storage: SupportsFilesystem
    """What the folder holds. Its paths start at `/`: `<path>/a/b` is `/a/b` in the storage."""

    _: KW_ONLY

    read_only: bool = False
    """Whether writes through this folder are refused."""

    description: str | None = None
    """What the folder is for, shown to the model."""

    shell: bool | None = None
    """Whether shell commands must see the folder.

    `None` lets them see it when the backend can mount it; `True` makes the workspace refuse a backend that can't;
    `False` keeps it to file tools.
    """

    def __post_init__(self) -> None:
        # `normpath` keeps a leading `//`, which POSIX leaves to the system; here it is `/`.
        path = posixpath.normpath(self.path).replace('//', '/', 1)
        if path in ('.', '/') or path == '..' or path.startswith('../'):
            raise ValueError(f'mount path must name a folder inside the workspace, got {self.path!r}')
        object.__setattr__(self, 'path', path)
        # A backend's paths are its whole environment (a local backend's `/` is the host's), not a folder's contents.
        if isinstance(self.storage, WorkspaceBackend) or not isinstance(self.storage, SupportsFilesystem):
            raise TypeError(
                f'mount storage must implement `SupportsFilesystem` and not be a workspace backend, '
                f'got {type(self.storage).__name__}'
            )


@runtime_checkable
class SupportsMounts(Protocol):
    """Optional on a backend: make mounts visible to its shell commands.

    Without it, every mount is reachable through file tools only.
    """

    def can_mount(self, mount: Mount) -> bool:
        """Whether this backend can show `mount` to commands. No I/O: it decides what the model is told."""
        ...

    async def ensure_mounts(self, mounts: Sequence[Mount], working_dir: str) -> datetime | None:
        """Make `mounts` (the backend's whole set) visible to commands, repairing lost ones and removing others.

        Relative mount paths are under `working_dir`. Return when to call again, such as before the keys a mount
        uses expire, or `None`. Raise [`MountError`][pydantic_ai.workspaces.MountError] when a mount can't be
        provided.
        """
        ...


def within(path: str, folder: str) -> bool:
    """Whether `path` is `folder` or inside it; both normalized, and both relative or both absolute."""
    return path == folder or path.startswith(folder.rstrip('/') + '/')


def check_mounts(paths: Sequence[str]) -> None:
    """Refuse mount paths that are the same folder or one inside another."""
    for index, path in enumerate(paths):
        for other in paths[index + 1 :]:
            if within(path, other) or within(other, path):
                raise ValueError(f'mounts {path!r} and {other!r} overlap')
