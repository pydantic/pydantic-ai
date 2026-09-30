"""Answer sandboxed `pathlib` and `open()` calls from the run's workspace, for `CodeMode(workspace_files=True)`."""

from __future__ import annotations

import errno
import posixpath
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from stat import S_IFDIR, S_IFREG
from typing import TYPE_CHECKING

from pydantic_monty import (
    ExternalException,
    ExternalReturnValue,
    MontyFileHandle,
    MountDir,
    StatResult,
)
from pydantic_monty.os_access import path_from_arg

from pydantic_ai.workspaces import FileEntry, Workspace

if TYPE_CHECKING:
    from ._toolset import CodeModeMount

__all__ = ('WorkspaceFiles',)

_FILE_FUNCTIONS = frozenset(
    {
        'Path.exists',
        'Path.is_file',
        'Path.is_dir',
        'Path.is_symlink',
        'open',
        'Path.read_text',
        'Path.read_bytes',
        'Path.write_text',
        'Path.write_bytes',
        'Path.append_text',
        'Path.append_bytes',
        'Path.mkdir',
        'Path.unlink',
        'Path.rmdir',
        'Path.iterdir',
        'Path.stat',
        'Path.rename',
        'Path.resolve',
        'Path.absolute',
    }
)


@dataclass(kw_only=True)
class WorkspaceFiles:
    """Routes one run's sandbox file calls to its workspace; paths under a host `mount` are left to the mount."""

    workspace: Workspace
    mount: CodeModeMount | None = None

    async def answer(
        self, name: str, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> ExternalReturnValue | ExternalException | None:
        """Answer a sandbox OS call, or return `None` when it is not a workspace file call."""
        if name not in _FILE_FUNCTIONS or self._mounted(args[0]):
            return None
        try:
            value = await self._call(name, str(_path(args[0])), args[1:], kwargs)
        except Exception as exc:
            # Raised at the sandbox call site, as a failing nested tool call is.
            return ExternalException(exception=exc)
        return ExternalReturnValue(return_value=value)

    def _mounted(self, arg: object) -> bool:
        configured = self.mount
        mounts: Sequence[MountDir] = (
            [] if configured is None else configured if isinstance(configured, list) else [configured]
        )
        path = _path(arg)
        return path.is_absolute() and any(
            PurePosixPath(posixpath.normpath(path)).is_relative_to(mount.virtual_path) for mount in mounts
        )

    async def _call(self, name: str, path: str, args: tuple[object, ...], kwargs: dict[str, object]) -> object:  # noqa: C901
        workspace = self.workspace
        match name:
            case 'Path.exists':
                return await workspace.exists(path)
            case 'Path.is_file':
                entry = await self._stat_or_none(path)
                return entry is not None and not entry.is_dir
            case 'Path.is_dir':
                entry = await self._stat_or_none(path)
                return entry is not None and entry.is_dir
            case 'Path.is_symlink':
                parent, base = posixpath.split(await workspace.resolve(path))
                return bool(base) and await workspace.realpath(path) != posixpath.join(
                    await workspace.realpath(parent), base
                )
            case 'open':
                return await self._open(path, _str(args[0]))
            case 'Path.read_text':
                return (await workspace.read_bytes(path)).decode()
            case 'Path.read_bytes':
                return await workspace.read_bytes(path)
            case 'Path.write_text' | 'Path.write_bytes' | 'Path.append_text' | 'Path.append_bytes':
                data = args[0]
                encoded = data.encode() if isinstance(data, str) else _bytes(data)
                if name.startswith('Path.append'):
                    encoded = await self._read_or_empty(path) + encoded
                await workspace.write_bytes(path, encoded)
                return len(data) if isinstance(data, str) else len(encoded)
            case 'Path.mkdir':
                await self._mkdir(path, parents=kwargs.get('parents') is True, exist_ok=kwargs.get('exist_ok') is True)
                return None
            case 'Path.unlink':
                if (await workspace.stat(path)).is_dir:
                    raise IsADirectoryError(errno.EISDIR, 'Is a directory', path)
                await workspace.remove(path)
                return None
            case 'Path.rmdir':
                if not (await workspace.stat(path)).is_dir:
                    raise NotADirectoryError(errno.ENOTDIR, 'Not a directory', path)
                if await workspace.list_dir(path):
                    raise OSError(errno.ENOTEMPTY, 'Directory not empty', path)
                await workspace.remove(path)
                return None
            case 'Path.iterdir':
                return [PurePosixPath(path) / entry.name for entry in await workspace.list_dir(path)]
            case 'Path.stat':
                return await self._stat(path)
            case 'Path.resolve':
                return await workspace.realpath(path)
            case 'Path.absolute':
                return await workspace.resolve(path)
            case _:
                # `Path.rename`: the workspace API has no atomic move, and a copy-then-delete could half-apply.
                raise OSError(errno.ENOTSUP, f'`{name}` is not supported on the run workspace', path)

    async def _stat_or_none(self, path: str) -> FileEntry | None:
        try:
            return await self.workspace.stat(path)
        except OSError:
            return None

    async def _read_or_empty(self, path: str) -> bytes:
        try:
            return await self.workspace.read_bytes(path)
        except FileNotFoundError:
            return b''

    async def _stat(self, path: str) -> StatResult:
        entry = await self.workspace.stat(path)
        if entry.is_dir:
            mode, size = S_IFDIR | 0o755, 0
        else:
            mode = S_IFREG | 0o644
            size = entry.size if entry.size is not None else len(await self.workspace.read_bytes(path))
        # Workspaces report no owner, links, or times; fixed values keep replays deterministic.
        return StatResult(
            st_mode=mode,
            st_ino=0,
            st_dev=0,
            st_nlink=1,
            st_uid=0,
            st_gid=0,
            st_size=size,
            st_atime=0.0,
            st_mtime=0.0,
            st_ctime=0.0,
        )

    async def _open(self, path: str, mode: str) -> MontyFileHandle:
        # Built first so a malformed mode fails before any side effect, as `OSAccess.path_open` does.
        handle = MontyFileHandle(path, mode)
        entry = await self._stat_or_none(path)
        if entry is not None and entry.is_dir:
            raise IsADirectoryError(errno.EISDIR, 'Is a directory', path)
        match handle.mode[0]:
            case 'r':
                if entry is None:
                    raise FileNotFoundError(errno.ENOENT, 'No such file or directory', path)
            case 'w':
                await self.workspace.write_bytes(path, b'')
            case _:  # 'a': create if missing, keep existing content
                if entry is None:
                    await self.workspace.write_bytes(path, b'')
        return handle

    async def _mkdir(self, path: str, *, parents: bool, exist_ok: bool) -> None:
        workspace = self.workspace
        entry = await self._stat_or_none(path)
        if entry is not None:
            if entry.is_dir and exist_ok:
                return
            raise FileExistsError(errno.EEXIST, 'File exists', path)
        if not parents:
            parent = await self._stat_or_none(posixpath.dirname(await workspace.resolve(path)))
            if parent is None:
                raise FileNotFoundError(errno.ENOENT, 'No such file or directory', path)
            if not parent.is_dir:
                raise NotADirectoryError(errno.ENOTDIR, 'Not a directory', path)
        await workspace.make_dir(path)


def _path(arg: object) -> PurePosixPath:
    if isinstance(arg, PurePosixPath | MontyFileHandle):
        return path_from_arg(arg)
    raise TypeError(f'expected a path, got {type(arg).__name__}')  # pragma: no cover - Monty always sends a path


def _str(value: object) -> str:
    if isinstance(value, str):
        return value
    raise TypeError(f'expected a string, got {type(value).__name__}')  # pragma: no cover - Monty validates modes


def _bytes(value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    raise TypeError(f'expected bytes, got {type(value).__name__}')  # pragma: no cover - Monty validates the data
