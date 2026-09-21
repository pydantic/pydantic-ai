"""A virtual workspace filesystem composed from independently backed mount points."""

from __future__ import annotations as _annotations

import posixpath
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .protocol import FileEntry, SupportsFilesystem, WorkspaceBackend, WorkspaceError, WorkspaceFileEntry, WorkspaceRef

__all__ = ('CompositeFilesystem', 'FilesystemMount')


@dataclass(frozen=True, kw_only=True)
class FilesystemMount:
    """A filesystem subtree exposed at a path in a [`CompositeFilesystem`][pydantic_ai.workspaces.CompositeFilesystem].

    `source_root` names the existing absolute directory in the source filesystem that becomes the
    mount's root. The caller keeps that directory available for the composite's lifetime. For an
    object-store-style filesystem this is normally `/`; for a workspace backend it can be that
    backend's working directory. Construction performs no I/O, so the composite does not verify it.
    """

    filesystem: SupportsFilesystem
    """The source filesystem."""
    source_root: str = '/'
    """Canonical absolute path in `filesystem` exposed at the mount point."""

    def __post_init__(self) -> None:
        _validate_path(self.source_root, label='source_root')


class CompositeFilesystem(WorkspaceBackend, SupportsFilesystem):
    """A filesystem-only workspace that routes absolute paths to non-overlapping mounts.

    The composite is deliberately not command-capable. It composes the logical filesystem
    namespace only; it does not make those files visible inside an unrelated command environment.
    A command-capable backend may delegate its native file operations to this class only when it
    also mounts the same sources at the same paths inside its command environment.

    Mount points may share virtual parents but may not overlap. For example, `/data` and `/skills`
    are valid, as are `/team/data` and `/team/skills`; `/data` and `/data/archive` are rejected.
    The composite synthesizes virtual parent directories and mount roots in `stat`, `exists`, and
    `list_dir` results.
    """

    def __init__(self, mounts: Mapping[str, FilesystemMount], *, working_dir: str = '/') -> None:
        if not mounts:
            raise ValueError('CompositeFilesystem requires at least one mount.')

        normalized_mounts: dict[str, FilesystemMount] = {}
        for path, mount in mounts.items():
            _validate_path(path, label='mount path')
            if path == '/':
                raise ValueError("The root path '/' cannot be a mount; it is the composite's virtual root.")
            for existing in normalized_mounts:
                if path.startswith(existing + '/') or existing.startswith(path + '/'):
                    raise ValueError(f'Mount paths must not overlap: {existing!r} and {path!r}.')
            normalized_mounts[path] = mount

        _validate_path(working_dir, label='working_dir')
        if not self._is_guaranteed_directory(working_dir, normalized_mounts):
            raise ValueError(
                f'working_dir {working_dir!r} must be the virtual root, a mount point, or a virtual parent.'
            )

        self._mounts = dict(sorted(normalized_mounts.items()))
        self._working_dir = working_dir

    @property
    def ref(self) -> WorkspaceRef | None:
        """Composite filesystems have no independently reconnectable identity."""
        return None

    async def working_dir(self) -> str:
        """Return the configured working directory in the composite namespace."""
        return self._working_dir

    async def read_bytes(self, path: str) -> bytes:
        _, mount, source_path = self._route(path)
        try:
            return await mount.filesystem.read_bytes(source_path)
        except FileNotFoundError as error:
            raise FileNotFoundError(path) from error

    async def write_bytes(self, path: str, data: bytes) -> None:
        _, mount, source_path = self._route(path)
        try:
            await mount.filesystem.write_bytes(source_path, data)
        except FileNotFoundError as error:
            raise FileNotFoundError(path) from error

    async def stat(self, path: str) -> WorkspaceFileEntry:
        path = _checked_operation_path(path)
        if path == '/' or self._is_virtual_directory(path) or path in self._mounts:
            return _directory_entry(path)
        mount_path, mount, source_path = self._route(path)
        try:
            entry = await mount.filesystem.stat(source_path)
        except FileNotFoundError as error:
            raise FileNotFoundError(path) from error
        return self._rewrite_entry(mount_path, mount, entry)

    async def list_dir(self, path: str) -> Sequence[WorkspaceFileEntry]:
        path = _checked_operation_path(path)
        virtual_children = self._virtual_children(path)
        if virtual_children:
            return tuple(_directory_entry(child) for child in virtual_children)
        if path == '/' or self._is_virtual_directory(path):
            return ()

        mount_path, mount, source_path = self._route(path)
        try:
            entries = await mount.filesystem.list_dir(source_path)
        except FileNotFoundError as error:
            raise FileNotFoundError(path) from error
        return tuple(self._rewrite_entry(mount_path, mount, entry) for entry in entries)

    async def make_dir(self, path: str) -> None:
        path = _checked_operation_path(path)
        if path == '/' or self._is_virtual_directory(path) or path in self._mounts:
            return
        _, mount, source_path = self._route(path)
        try:
            await mount.filesystem.make_dir(source_path)
        except FileNotFoundError as error:
            raise FileNotFoundError(path) from error

    async def remove(self, path: str) -> None:
        path = _checked_operation_path(path)
        if path == '/' or self._is_virtual_directory(path) or path in self._mounts:
            raise PermissionError(f'Cannot remove composite mount or virtual directory {path!r}.')
        _, mount, source_path = self._route(path)
        try:
            await mount.filesystem.remove(source_path)
        except FileNotFoundError as error:
            raise FileNotFoundError(path) from error

    async def exists(self, path: str) -> bool:
        path = _checked_operation_path(path)
        if path == '/' or self._is_virtual_directory(path) or path in self._mounts:
            return True
        try:
            _, mount, source_path = self._route(path)
        except FileNotFoundError:
            return False
        return await mount.filesystem.exists(source_path)

    @staticmethod
    def _is_guaranteed_directory(path: str, mounts: Mapping[str, FilesystemMount]) -> bool:
        if path == '/':
            return True
        return path in mounts or any(mount_path.startswith(path + '/') for mount_path in mounts)

    def _route(self, path: str) -> tuple[str, FilesystemMount, str]:
        path = _checked_operation_path(path)
        for mount_path, mount in self._mounts.items():
            if path == mount_path:
                return mount_path, mount, mount.source_root
            if path.startswith(mount_path + '/'):
                relative = path[len(mount_path) + 1 :]
                return mount_path, mount, posixpath.join(mount.source_root, relative)
        raise FileNotFoundError(path)

    def _is_virtual_directory(self, path: str) -> bool:
        return any(mount_path.startswith(path + '/') for mount_path in self._mounts)

    def _virtual_children(self, path: str) -> tuple[str, ...]:
        prefix = '/' if path == '/' else path + '/'
        children: set[str] = set()
        for mount_path in self._mounts:
            if not mount_path.startswith(prefix):
                continue
            remainder = mount_path[len(prefix) :]
            child_name = remainder.split('/', 1)[0]
            children.add(posixpath.join(path, child_name))
        return tuple(sorted(children))

    @staticmethod
    def _rewrite_entry(mount_path: str, mount: FilesystemMount, entry: WorkspaceFileEntry) -> WorkspaceFileEntry:
        source_path = _checked_operation_path(entry.path)
        if source_path == mount.source_root:
            relative = ''
        else:
            source_prefix = '/' if mount.source_root == '/' else mount.source_root + '/'
            if not source_path.startswith(source_prefix):
                raise WorkspaceError(
                    f'Mounted filesystem returned path {source_path!r} outside source root {mount.source_root!r}.'
                )
            relative = source_path[len(source_prefix) :]
        target_path = mount_path if not relative else posixpath.join(mount_path, relative)
        return FileEntry(name=posixpath.basename(target_path), path=target_path, is_dir=entry.is_dir, size=entry.size)


def _validate_path(path: str, *, label: str) -> None:
    if not path.startswith('/') or posixpath.normpath(path) != path or path.startswith('//'):
        raise ValueError(f'{label} must be a canonical absolute POSIX path, got {path!r}.')


def _checked_operation_path(path: str) -> str:
    _validate_path(path, label='path')
    return path


def _directory_entry(path: str) -> FileEntry:
    return FileEntry(name='/' if path == '/' else posixpath.basename(path), path=path, is_dir=True, size=None)
