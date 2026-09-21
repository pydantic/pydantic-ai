"""A virtual workspace filesystem composed from independently rooted filesystems."""

from __future__ import annotations as _annotations

import posixpath
from collections.abc import Mapping, Sequence
from types import MappingProxyType

from .protocol import FileEntry, SupportsFilesystem, WorkspaceBackend, WorkspaceError, WorkspaceFileEntry, WorkspaceRef

__all__ = ('CompositeFilesystem',)


class CompositeFilesystem(WorkspaceBackend, SupportsFilesystem):
    """Route one POSIX namespace across filesystems mounted at absolute paths.

    Each mounted filesystem owns the behavior below its root: the composite strips the mount
    prefix, passes it an absolute path rooted at `/`, and rewrites returned metadata into the
    composite namespace. Longest-prefix routing gives nested mounts ordinary Unix-like shadowing.

    A bare composite is a filesystem-only workspace with `/` as its working directory. A
    command-capable backend may use a composite internally, but that backend is responsible for
    making every mount visible at the same path to its commands. The composite only routes file
    operations; it does not perform FUSE, volume, or bind mounts.
    """

    def __init__(self, mounts: Mapping[str, SupportsFilesystem]) -> None:
        if not mounts:
            raise ValueError('CompositeFilesystem requires at least one mount.')

        normalized_mounts: dict[str, SupportsFilesystem] = {}
        for path, filesystem in mounts.items():
            _validate_path(path, label='mount path')
            normalized_mounts[path] = filesystem

        self._mounts = MappingProxyType(dict(sorted(normalized_mounts.items())))
        self._routing_order = tuple(sorted(self._mounts, key=len, reverse=True))

    @property
    def mounts(self) -> Mapping[str, SupportsFilesystem]:
        """The read-only mapping of canonical mount paths to their filesystems."""
        return self._mounts

    @property
    def ref(self) -> WorkspaceRef | None:
        """A composite has no independently reconnectable identity."""
        return None

    async def working_dir(self) -> str:
        """Return the virtual root, the default for relative paths in a standalone composite."""
        return '/'

    async def read_bytes(self, path: str) -> bytes:
        path = _checked_operation_path(path)
        if self._is_mount_or_virtual_directory(path):
            raise IsADirectoryError(path)
        _, filesystem, source_path = self._route(path)
        try:
            return await filesystem.read_bytes(source_path)
        except FileNotFoundError as error:
            raise FileNotFoundError(path) from error

    async def write_bytes(self, path: str, data: bytes) -> None:
        path = _checked_operation_path(path)
        if self._is_mount_or_virtual_directory(path):
            raise IsADirectoryError(path)
        _, filesystem, source_path = self._route(path)
        try:
            await filesystem.write_bytes(source_path, data)
        except FileNotFoundError as error:
            raise FileNotFoundError(path) from error

    async def stat(self, path: str) -> WorkspaceFileEntry:
        path = _checked_operation_path(path)
        if self._is_mount_or_virtual_directory(path):
            return _directory_entry(path)

        mount_path, filesystem, source_path = self._route(path)
        try:
            entry = await filesystem.stat(source_path)
        except FileNotFoundError as error:
            raise FileNotFoundError(path) from error
        return self._rewrite_entry(mount_path, entry)

    async def list_dir(self, path: str) -> Sequence[WorkspaceFileEntry]:
        path = _checked_operation_path(path)
        virtual_children = self._virtual_children(path)
        entries_by_name: dict[str, WorkspaceFileEntry] = {}

        try:
            mount_path, filesystem, source_path = self._route(path)
        except FileNotFoundError:
            if not virtual_children and not self._is_mount_or_virtual_directory(path):
                raise
        else:
            try:
                entries = await filesystem.list_dir(source_path)
            except FileNotFoundError:
                if not virtual_children:
                    raise FileNotFoundError(path) from None
            else:
                for entry in entries:
                    rewritten = self._rewrite_entry(mount_path, entry)
                    entries_by_name[rewritten.name] = rewritten

        # A mount shadows an entry with the same name in its parent filesystem.
        for child in virtual_children:
            entry = _directory_entry(child)
            entries_by_name[entry.name] = entry
        return tuple(entries_by_name[name] for name in sorted(entries_by_name))

    async def make_dir(self, path: str) -> None:
        path = _checked_operation_path(path)
        if self._is_mount_or_virtual_directory(path):
            return
        _, filesystem, source_path = self._route(path)
        try:
            await filesystem.make_dir(source_path)
        except FileNotFoundError as error:
            raise FileNotFoundError(path) from error

    async def remove(self, path: str) -> None:
        path = _checked_operation_path(path)
        if self._is_mount_or_virtual_directory(path):
            raise PermissionError(f'Cannot remove composite mount or virtual directory {path!r}.')
        _, filesystem, source_path = self._route(path)
        try:
            await filesystem.remove(source_path)
        except FileNotFoundError as error:
            raise FileNotFoundError(path) from error

    async def exists(self, path: str) -> bool:
        path = _checked_operation_path(path)
        if self._is_mount_or_virtual_directory(path):
            return True
        try:
            _, filesystem, source_path = self._route(path)
        except FileNotFoundError:
            return False
        return await filesystem.exists(source_path)

    def _route(self, path: str) -> tuple[str, SupportsFilesystem, str]:
        path = _checked_operation_path(path)
        for mount_path in self._routing_order:
            if path == mount_path:
                return mount_path, self._mounts[mount_path], '/'
            prefix = '/' if mount_path == '/' else mount_path + '/'
            if path.startswith(prefix):
                relative = path[len(prefix) :]
                return mount_path, self._mounts[mount_path], '/' + relative
        raise FileNotFoundError(path)

    def _is_mount_or_virtual_directory(self, path: str) -> bool:
        if path in self._mounts:
            return True
        prefix = '/' if path == '/' else path + '/'
        return any(mount_path.startswith(prefix) for mount_path in self._mounts)

    def _virtual_children(self, path: str) -> tuple[str, ...]:
        prefix = '/' if path == '/' else path + '/'
        children: set[str] = set()
        for mount_path in self._mounts:
            if not mount_path.startswith(prefix) or mount_path == path:
                continue
            remainder = mount_path[len(prefix) :]
            child_name = remainder.split('/', 1)[0]
            children.add(posixpath.join(path, child_name))
        return tuple(sorted(children))

    @staticmethod
    def _rewrite_entry(mount_path: str, entry: WorkspaceFileEntry) -> WorkspaceFileEntry:
        source_path = _checked_operation_path(entry.path)
        relative = source_path.removeprefix('/')
        target_path = mount_path if not relative else posixpath.join(mount_path, relative)
        if mount_path != '/' and target_path != mount_path and not target_path.startswith(mount_path + '/'):
            raise WorkspaceError(f'Mounted filesystem returned path {source_path!r} outside its root.')
        return FileEntry(name=posixpath.basename(target_path), path=target_path, is_dir=entry.is_dir, size=entry.size)


def _validate_path(path: str, *, label: str) -> None:
    if not path.startswith('/') or posixpath.normpath(path) != path or path.startswith('//'):
        raise ValueError(f'{label} must be a canonical absolute POSIX path, got {path!r}.')


def _checked_operation_path(path: str) -> str:
    _validate_path(path, label='path')
    return path


def _directory_entry(path: str) -> FileEntry:
    return FileEntry(name='/' if path == '/' else posixpath.basename(path), path=path, is_dir=True, size=None)
