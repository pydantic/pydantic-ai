"""`Workspace`, what tools and hooks get as `ctx.workspace`: a backend plus path resolution and text helpers."""

from __future__ import annotations as _annotations

import base64
import posixpath
import shlex
import uuid
from collections.abc import Mapping, Sequence
from typing import cast

import anyio
from typing_extensions import get_protocol_members

from pydantic_ai.exceptions import UserError

from .protocol import (
    CommandResult,
    FileEntry,
    SupportsCommands,
    SupportsFilesystem,
    SupportsRealpath,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    validate_timeout,
)
from .unavailable import UnavailableWorkspace

__all__ = ('Workspace', 'WrapperWorkspace')


_SHELL_READ_CHUNK_BYTES = 64 * 1024
_SHELL_READ_ATTEMPTS = 3
_SHELL_WRITE_CHUNK_CHARS = 64 * 1024
"""Maximum base64 characters embedded in one shell command.

Linux limits one `execve` argument to 128 KiB, independently of `ARG_MAX`. Leaving half of
that for quoting and the command template keeps fallback writes below the lower limit.
"""

_SHELL_MAX_SYMLINKS = 40
"""Symlinks `realpath` follows before giving up, as Linux does (`MAXSYMLINKS`), so a link loop ends."""

_SHELL_CLEANUP_TIMEOUT = 10
"""Maximum time spent removing an interrupted fallback write's temporary files."""

# Exit codes the shell filesystem's own path checks use to report a path-level failure: 100 plus
# the matching Linux errno, outside the small codes `sh`, `base64` and `find` return themselves
# (dash exits 2 on a failed redirection), so a utility failure is never mistaken for a missing path.
_SHELL_EXIT_NOT_FOUND = 102
_SHELL_EXIT_EXISTS = 117
_SHELL_EXIT_NOT_DIRECTORY = 120
_SHELL_EXIT_IS_DIRECTORY = 121
_SHELL_EXIT_NOT_REGULAR = 122
_SHELL_EXIT_PERMISSION = 113

# Inspect each existing ancestor before `mkdir -p`; shell utilities' diagnostic wording is not portable.
_SHELL_CHECK_PARENTS = (
    'while [ "$parent" != / ]; do '
    f'if test -e "$parent" && ! test -d "$parent"; then exit {_SHELL_EXIT_NOT_DIRECTORY}; fi; '
    f'if test -d "$parent"; then test -w "$parent" || exit {_SHELL_EXIT_PERMISSION}; break; fi; '
    'parent=${parent%/*}; [ -n "$parent" ] || parent=/; done; '
)


class _DamagedOutputError(WorkspaceError):
    pass


def _damaged(path: str) -> WorkspaceError:
    return _DamagedOutputError(f'shell filesystem returned damaged or incomplete output for {path!r}')


def _size(text: str, path: str) -> int:
    if not text.strip().isdigit():
        raise _damaged(path)
    return int(text)


def _decode(encoded: str, size: int, path: str) -> bytes:
    """Decode base64 command output (text-safe for any bytes), refusing anything but exactly `size` bytes.

    A truncated or damaged stream can still decode, and a file can shrink mid-read; never return a partial result.
    """
    try:
        data = base64.b64decode(encoded)
    except ValueError:
        raise _damaged(path) from None
    if len(data) != size:
        raise _damaged(path)
    return data


def _decode_sized(output: str, path: str) -> bytes:
    """Decode a size line followed by base64."""
    size, _, encoded = output.partition('\n')
    return _decode(encoded, _size(size, path), path)


_SHELL_TOOLS = ('base64', 'cp', 'dd', 'find', 'mkdir', 'mv', 'readlink', 'rm', 'wc')
"""The utilities `_ShellFilesystem` runs, beyond `sh` builtins."""


class _ShellFilesystem(SupportsFilesystem):
    """Derive filesystem operations from a backend's command-execution primitive.

    This is the portability floor for command-capable workspaces. Backends should implement
    `SupportsFilesystem` when their provider has a native API: native calls avoid the shell's
    utility assumptions and the base64 transfer overhead used here to preserve arbitrary bytes.

    It needs a POSIX `sh` with `test` and `printf`, plus `base64`, `cp`, `dd`, `mv`, `rm`, `mkdir`,
    `find` and `wc`, and `readlink` for `realpath`. A path under a directory the command cannot search reads as missing:
    `test -e` cannot tell a permission error from a missing path.
    """

    def __init__(self, backend: SupportsCommands):
        self._backend = backend

    async def read_bytes(self, path: str) -> bytes:
        # The size bounds the read, so a file growing meanwhile can't make it run on.
        size: int | None = None
        for _ in range(_SHELL_READ_ATTEMPTS):
            entry = await self.stat(path)
            if entry.is_dir:
                raise IsADirectoryError(path)
            assert entry.size is not None
            # A file replaced after its size was read (a job publishing its status, say) is read again
            # at its new size; an unchanged size means the output itself was damaged.
            if entry.size == size:
                break
            size = entry.size
            try:
                return await self._read_chunks(shlex.quote(path), size, path, 'read')
            except _DamagedOutputError:
                pass
        raise await self._damaged_or_missing_tools(path)

    async def _damaged_or_missing_tools(self, path: str) -> WorkspaceError:
        # A missing `dd` or `base64` looks like damaged output: a pipe reports only its last command's status.
        tools = ' '.join(_SHELL_TOOLS)
        probe = await self._backend.run(
            f'for tool in {tools}; do command -v "$tool" >/dev/null 2>&1 || printf "%s " "$tool"; done', shell=True
        )
        missing = probe.stdout.split()
        if not missing:
            return _damaged(path)
        names = ', '.join(f'`{tool}`' for tool in missing)
        return WorkspaceError(
            f'The workspace has no {names}, which its file operations need: it has no native filesystem, so they '
            f'run as shell commands. Install them in the image (coreutils and findutils provide them).'
        )

    async def _read_chunks(self, source: str, size: int, path: str, what: str) -> bytes:
        data = bytearray()
        # One command per chunk keeps each command's output under the backend's output limit.
        for index in range((size + _SHELL_READ_CHUNK_BYTES - 1) // _SHELL_READ_CHUNK_BYTES):
            try:
                result = await self._backend.run(
                    f'dd if={source} bs={_SHELL_READ_CHUNK_BYTES} skip={index} count=1 2>/dev/null | base64',
                    shell=True,
                )
            except WorkspaceOutputLimitError as error:
                raise WorkspaceError(f'shell filesystem {what} exceeded command output limit for {path!r}') from error
            await self._raise_for_error(result, path)
            data.extend(_decode(result.stdout, min(_SHELL_READ_CHUNK_BYTES, size - len(data)), path))
        return bytes(data)

    async def write_bytes(self, path: str, data: bytes) -> None:
        # Stage beside the resolved target so a failed transfer leaves it intact.
        destination = await self.realpath(path)
        parent = posixpath.dirname(destination)
        temporary_path = posixpath.join(parent, f'.pydantic-ai-{uuid.uuid4().hex}.tmp')
        decoded_path = f'{temporary_path}.decoded'
        quoted_parent = shlex.quote(parent)
        quoted_temporary = shlex.quote(temporary_path)
        quoted_decoded = shlex.quote(decoded_path)
        encoded = base64.b64encode(data).decode()
        chunks = [
            encoded[start : start + _SHELL_WRITE_CHUNK_CHARS]
            for start in range(0, len(encoded), _SHELL_WRITE_CHUNK_CHARS)
        ]
        try:
            for index, chunk in enumerate(chunks or ['']):
                start = (
                    f'parent={quoted_parent}; {_SHELL_CHECK_PARENTS}'
                    f'test -w {quoted_parent} || ! test -e {quoted_parent} || exit {_SHELL_EXIT_PERMISSION}; '
                    f'mkdir -p {quoted_parent} && '
                    if index == 0
                    else ''
                )
                redirect = '>' if index == 0 else '>>'
                result = await self._backend.run(
                    f"{start}printf '%s' {shlex.quote(chunk)} {redirect} {quoted_temporary}", shell=True
                )
                await self._raise_for_error(result, path)

            quoted_destination = shlex.quote(destination)
            # Copy first to preserve mode bits, and stop if that fails (an unreadable file, say) rather
            # than replace the file with a default mode; commit only after decoding succeeds.
            result = await self._backend.run(
                f'if test -d {quoted_destination}; then status={_SHELL_EXIT_IS_DIRECTORY}; '
                # `mv` would replace a FIFO or device with a regular file; refuse as the local backend does.
                f'elif test -e {quoted_destination} && ! test -f {quoted_destination}; '
                f'then status={_SHELL_EXIT_NOT_REGULAR}; '
                f'elif test -e {quoted_destination} && ! test -w {quoted_destination}; '
                f'then status={_SHELL_EXIT_PERMISSION}; else '
                f'if test -f {quoted_destination}; then cp {quoted_destination} {quoted_decoded}; fi && '
                f'base64 -d < {quoted_temporary} > {quoted_decoded} '
                f'&& mv -f {quoted_decoded} {quoted_destination}; '
                f'status=$?; fi; rm -f {quoted_temporary} {quoted_decoded}; exit $status',
                shell=True,
            )
            await self._raise_for_error(result, path)
        except BaseException:
            # AnyIO cancellation is level-triggered, so a plain cleanup await would immediately
            # be cancelled again and replace the original exception. Shield only this bounded,
            # best-effort removal; the interrupted operation still propagates unchanged.
            with anyio.move_on_after(_SHELL_CLEANUP_TIMEOUT, shield=True):
                try:
                    await self._backend.run(f'rm -f {quoted_temporary} {quoted_decoded}', shell=True)
                except Exception:
                    pass
            raise

    async def stat(self, path: str) -> FileEntry:
        quoted_path = shlex.quote(path)
        # Follows a symlink to its target, like the other operations.
        result = await self._backend.run(
            f"if test -d {quoted_path}; then printf 'directory\\n'; "
            f'elif test -f {quoted_path}; then '
            f'test -r {quoted_path} || exit {_SHELL_EXIT_PERMISSION}; wc -c < {quoted_path}; '
            f'elif test -e {quoted_path}; then exit {_SHELL_EXIT_NOT_REGULAR}; '
            f'else exit {_SHELL_EXIT_NOT_FOUND}; fi',
            shell=True,
        )
        await self._raise_for_error(result, path, missing=True)
        output = result.stdout.strip()
        name = posixpath.basename(posixpath.normpath(path))
        if output == 'directory':
            return FileEntry(name=name, path=path, is_dir=True, size=None)
        return FileEntry(name=name, path=path, is_dir=False, size=_size(output, path))

    async def list_dir(self, path: str) -> tuple[FileEntry, ...]:
        listing = await self._list_paths(path)
        # POSIX filenames are bytes; preserve undecodable names for a round trip via os.fsencode.
        entries = listing.decode(errors='surrogateescape').split('\0')
        # Each entry is `<d|-><path>`: whether it is a directory, following a symlink to its target.
        return tuple(
            FileEntry(name=posixpath.basename(entry[1:]), path=entry[1:], is_dir=entry[0] == 'd', size=None)
            for entry in sorted((entry for entry in entries if entry), key=lambda entry: entry[1:])
        )

    async def _list_paths(self, path: str) -> bytes:
        quoted_path = shlex.quote(path)
        # Keep the scratch file in the environment's temp directory (which needn't be /tmp).
        temporary_path = f'"${{TMPDIR:-/tmp}}/.pydantic-ai-{uuid.uuid4().hex}.list"'
        # `test` and `printf` rather than `find -printf`, which BusyBox and macOS lack.
        mark = ' -exec sh -c \'for f do test -d "$f" && d=d || d=-; printf "%s%s\\000" "$d" "$f"; done\' sh {} +'
        # Do not pipe `find` into `base64`: a POSIX shell reports only `base64`'s exit status and
        # could turn a failed traversal into a successful partial listing. The temporary file keeps
        # `find`'s status authoritative, and the trap removes it on every shell exit path.
        paged = False
        completed = False
        try:
            result = await self._backend.run(
                f'file={temporary_path}; trap \'rm -f "$file"\' EXIT HUP INT TERM; '
                f'if ! test -d {quoted_path}; then '
                f'test -e {quoted_path} && exit {_SHELL_EXIT_NOT_DIRECTORY}; exit {_SHELL_EXIT_NOT_FOUND}; fi; '
                f'find -H {quoted_path} -mindepth 1 -maxdepth 1{mark} > "$file" '
                f'&& size=$(wc -c < "$file") && printf "%s\\n" "$size" && '
                f'if [ "$size" -le {_SHELL_READ_CHUNK_BYTES} ]; then base64 < "$file"; '
                'else printf "PAGED\\n"; trap - EXIT HUP INT TERM; fi',
                shell=True,
            )
            completed = True
            size, separator, encoded = result.stdout.partition('\n')
            await self._raise_for_error(result, path, missing=True)
            if not separator or encoded.strip() != 'PAGED':
                return _decode_sized(result.stdout, path)
            paged = True
            return await self._read_chunks(temporary_path, _size(size, path), path, 'listing')
        finally:
            # A cancelled command may be killed before its EXIT trap runs. Paged listings also
            # keep the file alive across commands; shield only the bounded cleanup.
            with anyio.move_on_after(_SHELL_CLEANUP_TIMEOUT, shield=True):
                try:
                    if paged or not completed:
                        await self._backend.run(f'rm -f {temporary_path}', shell=True)
                except Exception:
                    pass

    async def make_dir(self, path: str) -> None:
        quoted_path = shlex.quote(path)
        # `mkdir -p` fails generically over an existing file; classify it like a native `mkdir`.
        result = await self._backend.run(
            f'if test -e {quoted_path} && ! test -d {quoted_path}; then exit {_SHELL_EXIT_EXISTS}; fi; '
            f'parent={shlex.quote(posixpath.dirname(path))}; {_SHELL_CHECK_PARENTS}'
            f'mkdir -p {quoted_path}',
            shell=True,
        )
        await self._raise_for_error(result, path)

    async def remove(self, path: str) -> None:
        root = await cast(WorkspaceBackend, self._backend).working_dir()
        # Refuse an ancestor before invoking `rm -rf`; never let removal of `.` destroy the environment.
        target = posixpath.join(await self.realpath(posixpath.dirname(path)), posixpath.basename(path))
        if root == target or root.startswith(target.rstrip('/') + '/'):
            raise ValueError('cannot remove the workspace root or its ancestor')
        quoted_path = shlex.quote(path)
        result = await self._backend.run(
            f'(test -e {quoted_path} || test -L {quoted_path}) && rm -rf {quoted_path}', shell=True
        )
        await self._raise_for_error(result, path, missing=True)

    async def exists(self, path: str) -> bool:
        result = await self._backend.run(f'test -e {shlex.quote(path)}', shell=True)
        # `test` exits 1 for "no"; anything else (a missing shell, a dropped SSH link) is a failure, not absence.
        if result.exit_code in (0, 1):
            return result.exit_code == 0
        raise WorkspaceError(
            result.stderr.strip() or f'could not check whether {path!r} exists (exit code {result.exit_code})'
        )

    async def realpath(self, path: str) -> str:
        # One round trip, resolving the way `os.path.realpath(strict=False)` does: walk the components
        # left to right, follow each symlink one level with `readlink` and walk its target in place of
        # the link, and let `..` climb the path resolved so far. A missing component is kept as
        # written and can still be climbed out of, after which symlinks resolve again. Past the link
        # limit, as in a loop, the rest is kept as written, like Python's loop handling. A link
        # `readlink` cannot read fails the call instead: kept as written, it would pass a root check
        # for a path that leads outside. The result is base64-encoded because command substitution
        # would drop a trailing newline from a name.
        result = await self._backend.run(
            f'rest={shlex.quote(path.lstrip("/"))}; resolved=; links=0; '
            'while [ -n "$rest" ]; do '
            'case "$rest" in */*) part="${rest%%/*}"; rest="${rest#*/}";; *) part="$rest"; rest=;; esac; '
            'case "$part" in ""|.) ;; ..) resolved="${resolved%/*}";; *) '
            f'if [ "$links" -lt {_SHELL_MAX_SYMLINKS} ] && [ -L "$resolved/$part" ]; then '
            'links=$((links + 1)); '
            'target=$(readlink -n -- "$resolved/$part" && printf x) || exit 1; target="${target%x}"; '
            'case "$target" in /*) resolved=;; esac; rest="$target/$rest"; '
            'else resolved="$resolved/$part"; fi;; esac; done; '
            'value="${resolved:-/}"; printf %s "$value" | wc -c; printf %s "$value" | base64',
            shell=True,
        )
        await self._raise_for_error(result, path)
        # Keep undecodable POSIX name bytes, as `list_dir` does, so its paths resolve too.
        return _decode_sized(result.stdout, path).decode(errors='surrogateescape')

    async def _raise_for_error(self, result: CommandResult, path: str, *, missing: bool = False) -> None:
        if result.exit_code == 0:
            return
        if result.exit_code == _SHELL_EXIT_NOT_FOUND:
            raise FileNotFoundError(path)
        if result.exit_code == _SHELL_EXIT_NOT_DIRECTORY:
            raise NotADirectoryError(path)
        if result.exit_code == _SHELL_EXIT_IS_DIRECTORY:
            raise IsADirectoryError(path)
        if result.exit_code == _SHELL_EXIT_EXISTS:
            raise FileExistsError(path)
        if result.exit_code == _SHELL_EXIT_NOT_REGULAR:
            raise OSError(f'not a regular file: {path!r}')
        if result.exit_code == _SHELL_EXIT_PERMISSION:
            raise PermissionError(path)
        if missing and not await self.exists(path):
            raise FileNotFoundError(path)
        message = result.stderr.strip() or f'shell filesystem operation failed for {path!r}'
        raise WorkspaceError(message)


class Workspace(WorkspaceBackend):
    """The workspace API tools and hooks use as `ctx.workspace`: the backend's operations, relative paths, and text."""

    def __init__(self, backend: WorkspaceBackend):
        self._backend = backend

    @property
    def backend(self) -> WorkspaceBackend:
        """The provider backend underneath every wrapper, for access to provider-specific functionality."""
        backend = self._backend
        return backend.backend if isinstance(backend, Workspace) else backend

    @property
    def read_only(self) -> bool:
        """Whether this workspace refuses commands and file changes, so tools can leave those out."""
        return isinstance(self._backend, Workspace) and self._backend.read_only

    def durable_policy(self) -> tuple[object, ...]:
        """Configuration a durable unit must rebuild with this wrapper.

        Empty unless the wrapper's own arguments change what the unit does. Two workspaces of the
        same class with different values are different policies: a run-level workspace that a unit
        would not rebuild is rejected, rather than replaced by the construction-time one.
        """
        return ()

    @property
    def attached(self) -> bool:
        """Whether this workspace reaches an environment.

        `False` for an [`UnavailableWorkspace`][pydantic_ai.workspaces.UnavailableWorkspace] and for a run
        with no workspace attached.
        """
        backend = self._backend
        if isinstance(backend, Workspace):
            return backend.attached
        return not isinstance(backend, UnavailableWorkspace)

    @property
    def ref(self) -> WorkspaceRef | None:
        """The environment's [`WorkspaceRef`][pydantic_ai.workspaces.WorkspaceRef], `None` until it exists."""
        return self._backend.ref

    @property
    def _filesystem(self) -> SupportsFilesystem:
        backend = self._backend
        if isinstance(backend, SupportsFilesystem):
            return backend
        if isinstance(backend, SupportsCommands):
            # The backend may provide native filesystem methods later.
            return _ShellFilesystem(backend)
        members = get_protocol_members(SupportsFilesystem)
        missing = sorted(name for name in members if not hasattr(backend, name))
        if len(missing) < len(members):
            # A backend that meant to implement it but missed a method would otherwise get no hint which.
            raise UserError(
                'This workspace does not support filesystem operations: its backend implements only part of '
                f'`SupportsFilesystem` and lacks {", ".join(f"`{name}`" for name in missing)}.'
            )
        raise UserError(
            'This workspace does not support filesystem operations. Attach a backend that implements '
            '`SupportsFilesystem` or `SupportsCommands`.'
        )

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        """Run a command in the working directory and wait for it.

        To start elsewhere, prefix a shell command with `cd <shlex-quoted dir> && `.
        There is no default `timeout`. Raises `UserError` if the backend can't run commands.
        """
        validate_timeout(timeout)
        if not isinstance(command, str) and not command:
            # Checked here so every backend reports it the same way, not as its SDK's own error.
            raise ValueError('command must not be empty')
        backend = self._backend
        if not isinstance(backend, SupportsCommands):
            raise UserError('This workspace does not support command execution.')
        return await backend.run(command, shell=shell, env=env, timeout=timeout)

    async def working_dir(self) -> str:
        """The default working directory, which relative paths resolve against."""
        return await self._backend.working_dir()

    async def resolve(self, path: str, *, base: str | None = None) -> str:
        """Join `path` onto `base` (default: the working directory) and normalize it as text.

        The only I/O is asking the backend for its working directory when `base` is omitted.

        Symlinks are not followed and `..` can escape `base`, so this confines nothing; use
        [`realpath`][pydantic_ai.workspaces.Workspace.realpath] to learn where a path actually leads.
        """
        if base is not None and not posixpath.isabs(base):
            raise ValueError(f'base must be an absolute path, got {base!r}')
        if posixpath.isabs(path):
            return posixpath.normpath(path)
        return posixpath.normpath(posixpath.join(base or await self.working_dir(), path))

    async def read_bytes(self, path: str) -> bytes:
        """Read a file's contents as bytes."""
        return await self._filesystem.read_bytes(await self.resolve(path))

    async def write_bytes(self, path: str, data: bytes) -> None:
        """Write bytes to a file, creating missing parents and replacing existing contents."""
        await self._filesystem.write_bytes(await self.resolve(path), data)

    async def stat(self, path: str) -> FileEntry:
        """Return metadata for a file or directory."""
        return await self._filesystem.stat(await self.resolve(path))

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        """List the entries of a directory (non-recursive)."""
        return await self._filesystem.list_dir(await self.resolve(path))

    async def make_dir(self, path: str) -> None:
        """Create a directory, including missing parents."""
        await self._filesystem.make_dir(await self.resolve(path))

    async def remove(self, path: str) -> None:
        """Remove a file, or a directory and its contents."""
        await self._filesystem.remove(await self.resolve(path))

    async def exists(self, path: str) -> bool:
        """Whether a file or directory exists at the path."""
        return await self._filesystem.exists(await self.resolve(path))

    async def realpath(self, path: str) -> str:
        """Follow every symlink in `path` in the environment, like `os.path.realpath(path, strict=False)`.

        Uses the backend's [`SupportsRealpath`][pydantic_ai.workspaces.SupportsRealpath], else `readlink` in
        its shell, raising [`WorkspaceError`][pydantic_ai.workspaces.WorkspaceError] on a link it cannot
        read, including when the shell has no `readlink`. Without either, it only normalizes the path as
        text: symlinks are not followed, so a path check built on it can be escaped through a link.

        `..` climbs from a symlink's target, as it does for commands. File methods open
        [`resolve(path)`][pydantic_ai.workspaces.Workspace.resolve], which collapses `..` as text, so
        `realpath(await ws.resolve(path))` names the file they open.
        """
        if not posixpath.isabs(path):
            # Joined, not normalized: `link/..` must climb from the link's target, not cancel out.
            path = posixpath.join(await self.working_dir(), path)
        backend = self._backend
        if isinstance(backend, SupportsRealpath):
            return await backend.realpath(path)
        if isinstance(backend, SupportsCommands):
            return await _ShellFilesystem(backend).realpath(path)
        return posixpath.normpath(path)

    async def read_text(self, path: str, *, encoding: str = 'utf-8') -> str:
        """Read a file as text; undecodable bytes raise `UnicodeDecodeError`."""
        return (await self.read_bytes(path)).decode(encoding)

    async def write_text(self, path: str, content: str, *, encoding: str = 'utf-8') -> None:
        """Write text to a file."""
        await self.write_bytes(path, content.encode(encoding))


class WrapperWorkspace(Workspace):
    """A workspace facade that composes another workspace."""

    _backend: Workspace

    def __init__(self, wrapped: Workspace):
        super().__init__(wrapped)

    @property
    def wrapped(self) -> Workspace:
        return self._backend


def workspace_layers(workspace: Workspace) -> list[type[object]]:
    """The policy wrappers around a workspace and its backend type, outermost first; plain `Workspace` layers are skipped."""
    layers: list[type[object]] = []
    layer: WorkspaceBackend = workspace
    while isinstance(layer, Workspace):
        if type(layer) is not Workspace:
            layers.append(type(layer))
        layer = layer._backend  # pyright: ignore[reportPrivateUsage]
    return [*layers, type(layer)]
