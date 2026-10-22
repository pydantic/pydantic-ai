"""Mounting a filesystem, such as a bucket, at a folder in a workspace."""

from __future__ import annotations

import posixpath
from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from .protocol import SupportsCommands, SupportsFilesystem, WorkspaceBackend, WorkspaceError

__all__ = ('Mount', 'MountError', 'SupportsMounts')


class MountError(WorkspaceError):
    """A mount could not be shown to commands: the sandbox can't mount it, or mounting it failed."""


@dataclass(frozen=True)
class Mount:
    """A filesystem shown as a folder in a workspace.

    ```python {test="skip" lint="skip"}
    from pydantic_ai.workspaces import Mount

    Mount('data', s3)  # file tools see `data/`
    Mount('data', s3, commands=True)  # the sandbox mounts it, so commands see `data/` too
    ```
    """

    path: str
    """Where the folder appears, relative to the working directory, such as `'data'`."""

    filesystem: WorkspaceBackend
    """What the folder shows: the backend's working directory, read on the first file operation. It must store files
    and run no commands, such as a bucket; a sandbox or a host directory can't be mounted."""

    _: KW_ONLY

    read_only: bool = False
    """Whether writes through this folder are refused."""

    description: str | None = None
    """What the folder is for, shown to the model."""

    commands: bool = False
    """Whether commands see the folder too.

    With `True`, the workspace's backend mounts the filesystem in the environment, and the workspace refuses a backend
    that runs no commands or can't mount it. With `False`, only file tools reach the folder, and the model is told so.
    """

    def __post_init__(self) -> None:
        path = posixpath.normpath(self.path)
        if posixpath.isabs(path) or path == '.' or path == '..' or path.startswith('../'):
            raise ValueError(
                f"a mount path is a folder inside the working directory, such as 'data', got {self.path!r}"
            )
        object.__setattr__(self, 'path', path)
        filesystem = self.filesystem
        # A backend that runs commands is an environment (a host's `/`, a sandbox), not a folder's contents. That
        # includes every `Workspace`, which brings mounts and policy of its own.
        if (
            not isinstance(filesystem, WorkspaceBackend)
            or not isinstance(filesystem, SupportsFilesystem)
            or isinstance(filesystem, SupportsCommands)
        ):
            raise TypeError(
                'a mount takes a workspace backend that stores files and runs no commands, such as a bucket, '
                f'got {type(filesystem).__name__}'
            )


@runtime_checkable
class SupportsMounts(Protocol):
    """Optional on a backend: mount filesystems in its environment, so commands see them.

    Only mounts with `commands=True` are passed here; file tools reach every mount without it.
    """

    def check_mount(self, mount: Mount) -> None:
        """Raise [`MountError`][pydantic_ai.workspaces.MountError] saying why `mount` can't be mounted. No I/O.

        Called when the workspace is built, so a mount that can't work is refused before the run. The message is
        one or more full sentences, ending with what to do.
        """
        ...

    async def ensure_mounts(self, mounts: Sequence[Mount]) -> datetime | None:
        """Mount `mounts` (the backend's whole set), repairing lost ones and removing others.

        Called before the workspace's first operation, with an empty set when nothing is to be mounted, so mounts an
        earlier run left can go. Mount paths are relative to the backend's working directory, which it needn't look up
        when there is nothing to do. Return when to call again, such as before the keys a mount
        uses expire, or `None`. Raise [`MountError`][pydantic_ai.workspaces.MountError] when a mount can't be
        provided.
        """
        ...


def within(path: str, folder: str) -> bool:
    """Whether `path` is `folder` or inside it; both normalized."""
    return path == folder or path.startswith(folder.rstrip('/') + '/')


def check_mounts(paths: Sequence[str]) -> None:
    """Refuse mount paths that are the same folder or one inside another."""
    for index, path in enumerate(paths):
        for other in paths[index + 1 :]:
            if within(path, other) or within(other, path):
                raise ValueError(f'mounts {path!r} and {other!r} overlap')
