"""Route one filesystem namespace across independently implemented filesystems."""

from __future__ import annotations as _annotations

import posixpath
from collections.abc import Mapping, Sequence

from .protocol import FileEntry, SupportsFilesystem, WorkspaceFileEntry

__all__ = ('CompositeFilesystem',)


class CompositeFilesystem(SupportsFilesystem):
    """Select a filesystem by longest matching mount prefix and delegate to it.

    Each child filesystem has its own namespace rooted at `/`: for example, a child mounted at
    `/skills` receives `/guide.md` when the composite receives `/skills/guide.md`. A child may be
    another `CompositeFilesystem`.

    Mount points and their ancestors appear as directories, including in directory listings. A
    more-specific mount shadows an entry with the same name in a parent filesystem.
    """

    def __init__(self, mounts: Mapping[str, SupportsFilesystem]) -> None:
        if not mounts:
            raise ValueError('CompositeFilesystem requires at least one mount.')

        for path in mounts:
            if _normalize(path) != path:
                raise ValueError(f'mount path must be a canonical absolute POSIX path, got {path!r}.')

        # More-specific mounts must be considered before their parents.
        self._mounts = tuple(sorted(mounts.items(), key=lambda item: len(item[0]), reverse=True))

    async def read_bytes(self, path: str) -> bytes:
        path = _normalize(path)
        if self._is_mount_directory(path):
            raise IsADirectoryError(path)
        _, filesystem, source_path = self._select(path)
        return await filesystem.read_bytes(source_path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        path = _normalize(path)
        if self._is_mount_directory(path):
            raise IsADirectoryError(path)
        _, filesystem, source_path = self._select(path)
        await filesystem.write_bytes(source_path, data)

    async def stat(self, path: str) -> WorkspaceFileEntry:
        path = _normalize(path)
        if self._is_mount_directory(path):
            return _directory_entry(path)

        mount_path, filesystem, source_path = self._select(path)
        return _rebase_entry(mount_path, await filesystem.stat(source_path))

    async def list_dir(self, path: str) -> Sequence[WorkspaceFileEntry]:
        path = _normalize(path)
        entries_by_name: dict[str, WorkspaceFileEntry] = {}

        try:
            mount_path, filesystem, source_path = self._select(path)
            entries = await filesystem.list_dir(source_path)
        except FileNotFoundError:
            if not self._is_mount_directory(path):
                raise
        else:
            for entry in entries:
                rebased = _rebase_entry(mount_path, entry)
                entries_by_name[rebased.name] = rebased

        # A mounted child shadows an entry with the same name in the selected parent filesystem.
        for name in self._mount_children(path):
            entries_by_name[name] = _directory_entry(posixpath.join(path, name))

        return tuple(entries_by_name[name] for name in sorted(entries_by_name))

    async def make_dir(self, path: str) -> None:
        path = _normalize(path)
        if self._is_mount_directory(path):
            return
        _, filesystem, source_path = self._select(path)
        await filesystem.make_dir(source_path)

    async def remove(self, path: str) -> None:
        path = _normalize(path)
        if self._is_mount_directory(path):
            raise PermissionError(f'Cannot remove composite mount or virtual directory {path!r}.')
        _, filesystem, source_path = self._select(path)
        await filesystem.remove(source_path)

    async def exists(self, path: str) -> bool:
        path = _normalize(path)
        if self._is_mount_directory(path):
            return True
        try:
            _, filesystem, source_path = self._select(path)
        except FileNotFoundError:
            return False
        return await filesystem.exists(source_path)

    def _select(self, path: str) -> tuple[str, SupportsFilesystem, str]:
        """Select the filesystem for `path` and rebase `path` into its root."""
        for mount_path, filesystem in self._mounts:
            prefix = mount_path.rstrip('/')
            if path == prefix or path.startswith(prefix + '/'):
                return mount_path, filesystem, path[len(prefix) :] or '/'
        raise FileNotFoundError(path)

    def _is_mount_directory(self, path: str) -> bool:
        prefix = path.rstrip('/')
        return any(mount == path or mount.startswith(prefix + '/') for mount, _ in self._mounts)

    def _mount_children(self, path: str) -> set[str]:
        prefix = path.rstrip('/') + '/'
        return {
            mount_path[len(prefix) :].split('/', 1)[0]
            for mount_path, _ in self._mounts
            if mount_path != path and mount_path.startswith(prefix)
        }


def _normalize(path: str) -> str:
    if not path.startswith('/'):
        raise ValueError(f'path must be absolute, got {path!r}.')
    return '/' + posixpath.normpath(path).lstrip('/')


def _rebase_entry(mount_path: str, entry: WorkspaceFileEntry) -> FileEntry:
    source_path = _normalize(entry.path)
    path = source_path if mount_path == '/' else mount_path + source_path
    return FileEntry(
        name='/' if path == '/' else posixpath.basename(path),
        path=path,
        is_dir=entry.is_dir,
        size=entry.size,
    )


def _directory_entry(path: str) -> FileEntry:
    return FileEntry(name='/' if path == '/' else posixpath.basename(path), path=path, is_dir=True, size=None)
