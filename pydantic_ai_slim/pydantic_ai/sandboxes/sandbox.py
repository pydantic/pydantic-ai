"""The user-facing sandbox API.

Sandbox backends implement the small
[`SandboxBackend`][pydantic_ai.sandboxes.SandboxBackend] protocol and typically also
[`SupportsFilesystem`][pydantic_ai.sandboxes.SupportsFilesystem]. The `Sandbox` object owns
model-facing semantics such as decoding and windowed file reads. Capabilities and user tools
consume it through [`RunContext.sandbox`][pydantic_ai.tools.RunContext.sandbox].
"""

from __future__ import annotations as _annotations

import base64
import posixpath
import shlex
import uuid
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

import anyio

from pydantic_ai.exceptions import UserError

if TYPE_CHECKING:
    from pydantic_ai.capabilities import AbstractCapability

from .protocol import (
    FileEntry,
    SandboxBackend,
    SandboxCommand,
    SandboxError,
    SandboxFileEntry,
    SandboxRef,
    SandboxResult,
    SandboxTimeoutError,
    SupportsFilesystem,
)

__all__ = ('FileWindow', 'Sandbox')

_SHELL_SLICE_TIMEOUT = 10
"""Deadline in seconds for the `sed` fast path in `read_file`.

The slice is an optimization, so a slow or wedged attempt (a FIFO path, a stalled mount)
falls back to the authoritative filesystem read instead of hanging the run without bound."""

_SHELL_WRITE_CHUNK_BYTES = 64 * 1024
"""Maximum base64 characters embedded in one shell command.

Linux limits one `execve` argument to 128 KiB, independently of `ARG_MAX`. Leaving half of
that for quoting and the command template keeps fallback writes below the lower limit.
"""

_SHELL_CLEANUP_TIMEOUT = 10
"""Maximum time spent removing an interrupted fallback write's temporary files."""


class _SandboxOperationDispatcher(Protocol):
    @property
    def ref(self) -> SandboxRef | None: ...

    @property
    def backend(self) -> SandboxBackend: ...

    def __call__(self, method: str, arguments: Mapping[str, Any]) -> Awaitable[Any]: ...


@dataclass(frozen=True, kw_only=True)
class FileWindow:
    """A line window of a sandbox file, as returned by [`Sandbox.read_file`][pydantic_ai.sandboxes.Sandbox.read_file]."""

    lines: tuple[str, ...]
    """The requested lines, without trailing newlines; a trailing `\r` (Windows line ending)
    is also stripped. For byte-exact access, use `read_bytes`.
    """
    start_line: int
    """1-based line number of `lines[0]` (the requested `offset`, even when `lines` is empty)."""
    has_more: bool
    """Whether the file has content after this window."""
    total_lines: int | None
    """Total number of lines in the file, when known (the read reached EOF); `None` otherwise."""

    @property
    def text(self) -> str:
        return '\n'.join(self.lines)


class _ShellFilesystem:
    """Derive filesystem operations from a backend's command-execution primitive.

    This is the portability floor for command-capable sandboxes. Backends should implement
    `SupportsFilesystem` when their provider has a native API: native calls avoid the shell's
    utility assumptions and the base64 transfer overhead used here to preserve arbitrary bytes.
    """

    def __init__(self, backend: SandboxBackend):
        self._backend = backend

    async def read_bytes(self, path: str) -> bytes:
        result = await self._backend.run(f'base64 < {shlex.quote(path)}', shell=True)
        await self._raise_for_error(result, path, missing=True)
        try:
            return base64.b64decode(result.stdout)
        except ValueError as error:
            raise SandboxError(f'shell filesystem returned invalid base64 while reading {path!r}') from error

    async def write_bytes(self, path: str, data: bytes) -> None:
        parent = posixpath.dirname(path)
        temporary_path = posixpath.join(parent, f'.pydantic-ai-{uuid.uuid4().hex}.tmp')
        decoded_path = f'{temporary_path}.decoded'
        quoted_parent = shlex.quote(parent)
        quoted_temporary = shlex.quote(temporary_path)
        quoted_decoded = shlex.quote(decoded_path)
        encoded = base64.b64encode(data).decode()
        chunks = [
            encoded[start : start + _SHELL_WRITE_CHUNK_BYTES]
            for start in range(0, len(encoded), _SHELL_WRITE_CHUNK_BYTES)
        ]
        try:
            for index, chunk in enumerate(chunks or ['']):
                redirect = '>' if index == 0 else '>>'
                result = await self._backend.run(
                    f"mkdir -p {quoted_parent} && printf '%s' {shlex.quote(chunk)} {redirect} {quoted_temporary}",
                    shell=True,
                )
                await self._raise_for_error(result, path)

            quoted_path = shlex.quote(path)
            # Decode beside the destination and rename into place so cancellation or a failed
            # decode never leaves a partially written file. Copying an existing regular file
            # first preserves its mode bits; a directory destination is deliberately rejected.
            result = await self._backend.run(
                f'{{ test -f {quoted_path} && cp {quoted_path} {quoted_decoded}; }}; '
                f'base64 -d < {quoted_temporary} > {quoted_decoded} '
                f'&& test ! -d {quoted_path} && mv -f {quoted_decoded} {quoted_path}; '
                f'status=$?; rm -f {quoted_temporary} {quoted_decoded}; exit $status',
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
        result = await self._backend.run(
            f"if test -d {quoted_path}; then printf 'directory\\n'; else wc -c < {quoted_path}; fi",
            shell=True,
        )
        await self._raise_for_error(result, path, missing=True)
        output = result.stdout.strip()
        if output == 'directory':
            return FileEntry(name=posixpath.basename(posixpath.normpath(path)), path=path, is_dir=True, size=None)
        try:
            size = int(output)
        except ValueError as error:
            raise SandboxError(f'shell filesystem returned an invalid size for {path!r}: {output!r}') from error
        return FileEntry(name=posixpath.basename(posixpath.normpath(path)), path=path, is_dir=False, size=size)

    async def list_dir(self, path: str) -> tuple[FileEntry, ...]:
        quoted_path = shlex.quote(path)
        result = await self._list_paths(quoted_path)
        await self._raise_for_error(result, path, missing=True)
        directory_result = await self._list_paths(quoted_path, directories_only=True)
        await self._raise_for_error(directory_result, path, missing=True)
        try:
            entries = base64.b64decode(result.stdout).decode().split('\0')
            directories = set(base64.b64decode(directory_result.stdout).decode().split('\0'))
        except (UnicodeDecodeError, ValueError) as error:
            raise SandboxError(f'shell filesystem returned an invalid directory listing for {path!r}') from error
        return tuple(
            FileEntry(
                name=posixpath.basename(entry_path),
                path=entry_path,
                is_dir=entry_path in directories,
                size=None,
            )
            for entry_path in sorted(entry for entry in entries if entry)
        )

    async def _list_paths(self, quoted_path: str, *, directories_only: bool = False) -> SandboxResult:
        temporary_path = f'/tmp/.pydantic-ai-{uuid.uuid4().hex}.list'
        quoted_temporary = shlex.quote(temporary_path)
        type_filter = ' -type d' if directories_only else ''
        # Do not pipe `find` into `base64`: a POSIX shell reports only `base64`'s exit status and
        # could turn a failed traversal into a successful partial listing. The temporary file keeps
        # `find`'s status authoritative, and the trap removes it on every shell exit path.
        return await self._backend.run(
            f'file={quoted_temporary}; trap \'rm -f "$file"\' EXIT HUP INT TERM; '
            f'test -d {quoted_path} && '
            f'find {quoted_path} -mindepth 1 -maxdepth 1{type_filter} -print0 > "$file" && base64 < "$file"',
            shell=True,
        )

    async def make_dir(self, path: str) -> None:
        result = await self._backend.run(f'mkdir -p {shlex.quote(path)}', shell=True)
        await self._raise_for_error(result, path)

    async def remove(self, path: str) -> None:
        quoted_path = shlex.quote(path)
        result = await self._backend.run(
            f'(test -e {quoted_path} || test -L {quoted_path}) && rm -rf {quoted_path}', shell=True
        )
        await self._raise_for_error(result, path, missing=True)

    async def exists(self, path: str) -> bool:
        result = await self._backend.run(f'test -e {shlex.quote(path)}', shell=True)
        return result.exit_code == 0

    async def _raise_for_error(self, result: SandboxResult, path: str, *, missing: bool = False) -> None:
        if result.exit_code == 0:
            return
        if missing and not await self.exists(path):
            raise FileNotFoundError(path)
        message = result.stderr.strip() or f'shell filesystem operation failed for {path!r}'
        raise SandboxError(message)


class Sandbox:
    """Rich sandbox interface exposed to tools and capabilities.

    `Sandbox` forwards the backend's required methods and adds filesystem access, path
    resolution, and uniform text and windowed-file helpers. Use
    [`backend`][pydantic_ai.sandboxes.Sandbox.backend] to reach provider-specific
    functionality.
    """

    def __init__(
        self,
        backend: SandboxBackend,
        *,
        _supplier_id: str | None = None,
        _supplier: AbstractCapability[Any] | None = None,
    ):
        self._backend = backend
        self._supplier_id = _supplier_id
        self._supplier = _supplier
        self._operation_dispatcher: _SandboxOperationDispatcher | None = None

    def _supplier_details(self) -> tuple[str | None, AbstractCapability[Any] | None]:
        """Return private routing metadata for durable execution."""
        return self._supplier_id, self._supplier

    @classmethod
    def wrap(cls, value: SandboxBackend) -> Sandbox:
        """Wrap `value`, returning an existing `Sandbox` unchanged."""
        return value if isinstance(value, Sandbox) else cls(value)

    def _install_operation_dispatcher(self, dispatcher: _SandboxOperationDispatcher) -> None:
        """Route user-facing methods without replacing this `Sandbox` object."""
        self._operation_dispatcher = dispatcher

    def _raw_backend(self) -> SandboxBackend:
        """Return the provider backend for framework-internal serialization and routing."""
        return self._backend

    def _replace_raw_backend(self, backend: SandboxBackend) -> None:
        """Reconnect the direct-use view after a durable operation learns its identity."""
        self._backend = backend

    @property
    def backend(self) -> SandboxBackend:
        """The wrapped backend, for access to provider-specific functionality."""
        if dispatcher := self._operation_dispatcher:
            return dispatcher.backend
        return self._backend

    @property
    def ref(self) -> SandboxRef | None:
        """Identity of the environment, once the backend has one.

        `None` until a backend built to create a fresh environment has run its first operation.
        """
        if dispatcher := self._operation_dispatcher:
            return dispatcher.ref
        return self._backend.ref

    @property
    def _filesystem(self) -> SupportsFilesystem:
        backend = self._backend
        if isinstance(backend, SupportsFilesystem):
            return backend
        # Do not cache this adapter: durable execution may reconnect and replace `_backend`.
        return _ShellFilesystem(backend)

    async def run(
        self,
        command: SandboxCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> SandboxResult:
        """Execute a command and wait for it to complete.

        Delegates to [`SandboxBackend.run`][pydantic_ai.sandboxes.SandboxBackend.run]; arguments
        and contracts are documented there.
        """
        # Checked here as well as in the backend: a relative cwd has no sandbox meaning, and the
        # wrapper is the seam every tool call goes through, so the error is the same whichever
        # backend is attached.
        if cwd is not None and not posixpath.isabs(cwd):
            raise ValueError(
                f'cwd must be an absolute POSIX path, got {cwd!r}; resolve relative paths with `sandbox.resolve()` first'
            )
        if dispatcher := self._operation_dispatcher:
            return cast(
                SandboxResult,
                await dispatcher(
                    'run', {'command': command, 'shell': shell, 'cwd': cwd, 'env': env, 'timeout': timeout}
                ),
            )
        return await self._backend.run(command, shell=shell, cwd=cwd, env=env, timeout=timeout)

    async def working_dir(self) -> str:
        """The sandbox's default working directory (absolute, filesystem-canonical POSIX path).

        The canonicality contract is documented on
        [`SandboxBackend.working_dir`][pydantic_ai.sandboxes.SandboxBackend.working_dir].
        """
        if dispatcher := self._operation_dispatcher:
            return cast(str, await dispatcher('working_dir', {}))
        return await self._backend.working_dir()

    async def resolve(self, path: str, *, base: str | None = None) -> str:
        """Resolve a possibly-relative path to an absolute POSIX path.

        Joins `path` onto `base` (default: [`working_dir`][pydantic_ai.sandboxes.Sandbox.working_dir])
        and normalizes it textually. This is a spelling convenience for model-supplied paths,
        **not** a confinement mechanism: `..` segments can escape `base` and symlinks are not
        inspected. Isolation is the sandbox's job, not this method's.
        """
        if base is not None and not posixpath.isabs(base):
            raise ValueError(f'base must be an absolute path, got {base!r}')
        if dispatcher := self._operation_dispatcher:
            return cast(str, await dispatcher('resolve', {'path': path, 'base': base}))
        if posixpath.isabs(path):
            return posixpath.normpath(path)
        return posixpath.normpath(posixpath.join(base or await self.working_dir(), path))

    async def read_bytes(self, path: str) -> bytes:
        """Read a file's contents as bytes."""
        if dispatcher := self._operation_dispatcher:
            return cast(bytes, await dispatcher('read_bytes', {'path': path}))
        return await self._filesystem.read_bytes(await self.resolve(path))

    async def write_bytes(self, path: str, data: bytes) -> None:
        """Write bytes to a file, creating missing parents and replacing existing contents."""
        if dispatcher := self._operation_dispatcher:
            await dispatcher('write_bytes', {'path': path, 'data': data})
            return
        await self._filesystem.write_bytes(await self.resolve(path), data)

    async def stat(self, path: str) -> SandboxFileEntry:
        """Return metadata for a file or directory."""
        if dispatcher := self._operation_dispatcher:
            return cast(SandboxFileEntry, await dispatcher('stat', {'path': path}))
        return await self._filesystem.stat(await self.resolve(path))

    async def list_dir(self, path: str) -> Sequence[SandboxFileEntry]:
        """List the entries of a directory (non-recursive)."""
        if dispatcher := self._operation_dispatcher:
            return cast(Sequence[SandboxFileEntry], await dispatcher('list_dir', {'path': path}))
        return await self._filesystem.list_dir(await self.resolve(path))

    async def make_dir(self, path: str) -> None:
        """Create a directory, including missing parents."""
        if dispatcher := self._operation_dispatcher:
            await dispatcher('make_dir', {'path': path})
            return
        await self._filesystem.make_dir(await self.resolve(path))

    async def remove(self, path: str) -> None:
        """Remove a file, or a directory and its contents."""
        if dispatcher := self._operation_dispatcher:
            await dispatcher('remove', {'path': path})
            return
        await self._filesystem.remove(await self.resolve(path))

    async def exists(self, path: str) -> bool:
        """Whether a file or directory exists at the path."""
        if dispatcher := self._operation_dispatcher:
            return cast(bool, await dispatcher('exists', {'path': path}))
        return await self._filesystem.exists(await self.resolve(path))

    async def read_text(self, path: str, *, encoding: str = 'utf-8') -> str:
        """Read text from `path`, resolving relative paths through the backend first.

        Decoding is strict: undecodable bytes raise `UnicodeDecodeError`. For a lossy,
        model-facing view use [`read_file`][pydantic_ai.sandboxes.Sandbox.read_file].
        """
        if dispatcher := self._operation_dispatcher:
            return cast(str, await dispatcher('read_text', {'path': path, 'encoding': encoding}))
        return (await self.read_bytes(path)).decode(encoding)

    async def write_text(self, path: str, content: str, *, encoding: str = 'utf-8') -> None:
        """Write text to `path`, resolving relative paths through the backend first."""
        if dispatcher := self._operation_dispatcher:
            await dispatcher('write_text', {'path': path, 'content': content, 'encoding': encoding})
            return
        await self.write_bytes(path, content.encode(encoding))

    async def read_file(self, path: str, *, offset: int = 1, limit: int | None = None) -> FileWindow:
        """Read a line window from `path`, resolving relative paths through the backend first.

        `offset` is the 1-based first line and `limit` is the maximum number of lines. When
        `limit` is `None`, the window extends through EOF. `limit` bounds returned lines, not
        bytes or characters: a single line may be arbitrarily large, and a backend without a
        usable in-sandbox `sed` command may transfer the whole file through its filesystem API
        before `Sandbox` applies the line window.

        This is a model-facing view: content is decoded as UTF-8 with U+FFFD replacement for
        undecodable bytes. Use [`read_text`][pydantic_ai.sandboxes.Sandbox.read_text] for
        strict decoding or `read_bytes` for exact bytes. Reading a special file that never
        ends (a FIFO, a device) blocks the way the underlying filesystem read does.
        """
        if offset < 1:
            raise ValueError('`offset` must be at least 1')
        if limit is not None and limit < 1:
            raise ValueError('`limit` must be at least 1')
        if dispatcher := self._operation_dispatcher:
            return cast(FileWindow, await dispatcher('read_file', {'path': path, 'offset': offset, 'limit': limit}))

        resolved_path = await self.resolve(path)
        filesystem: SupportsFilesystem
        if limit is not None:
            # Before the filesystem lookup: a backend with only `run()` can still serve
            # windowed reads through the slice, and command-capable remote backends avoid
            # transferring the whole file. This bounds line count, not byte size: one line
            # may still be arbitrarily large.
            window = await self._read_file_via_shell(resolved_path, offset, limit)
            if window is not None:
                return window

            await self._validate_bounded_read_path(resolved_path)
            filesystem = self._filesystem
        else:
            filesystem = self._filesystem

        data = await filesystem.read_bytes(resolved_path)
        return _window_from_data(data, offset, limit)

    async def _read_file_via_shell(self, path: str, offset: int, limit: int) -> FileWindow | None:
        """Slice a line window with `sed` inside the sandbox, so only the window crosses the wire.

        Returns `None` on failure (no usable `sed`, `run()` unsupported, or a slice that
        timed out), so the caller can fall back to the backend filesystem when available.
        `total_lines` is only reported when the slice provably reached EOF.
        """
        end = offset + limit  # one extra line, to learn whether more exist
        try:
            backend = self._backend
            # argv, never shell=True: the path is an argument, not shell-interpreted text.
            # `{end}q` stops `sed` at the window instead of scanning to EOF, and the timeout
            # bounds the optimization on paths that never finish.
            result = await backend.run(['sed', '-n', f'{offset},{end}p;{end}q', path], timeout=_SHELL_SLICE_TIMEOUT)
        except (NotImplementedError, OSError, SandboxTimeoutError, UserError):
            return None
        if result.exit_code != 0 or result.stderr:
            return None

        lines = list(_split_lines(result.stdout))
        if lines and lines[-1] == '':
            lines.pop()
        if not lines:
            # Empty output covers an empty file or an offset past EOF. The exact total is
            # unknown without scanning to EOF, which would defeat the bounded-read contract.
            await self._validate_bounded_read_path(path)
            return FileWindow(lines=(), start_line=offset, has_more=False, total_lines=None)
        if len(lines) > limit:
            return FileWindow(lines=tuple(lines[:limit]), start_line=offset, has_more=True, total_lines=None)
        return FileWindow(lines=tuple(lines), start_line=offset, has_more=False, total_lines=offset - 1 + len(lines))

    async def _validate_bounded_read_path(self, path: str) -> None:
        """Surface filesystem policy, missing-path, and directory errors without reading content."""
        try:
            entry = await (self._filesystem).stat(path)
        except NotImplementedError:
            return
        if entry.is_dir:
            raise IsADirectoryError(path)


def _window_from_data(data: bytes, offset: int, limit: int | None) -> FileWindow:
    text = data.decode('utf-8', errors='replace')
    lines = _split_lines(text)
    if lines[-1] == '':
        lines = lines[:-1]

    start = offset - 1
    end = None if limit is None else start + limit
    window = lines[start:end]
    return FileWindow(
        lines=window,
        start_line=offset,
        has_more=False if limit is None else start + limit < len(lines),
        total_lines=len(lines),
    )


def _split_lines(text: str) -> tuple[str, ...]:
    return tuple(line.removesuffix('\r') for line in text.split('\n'))


if TYPE_CHECKING:
    # Pins full structural conformance — signatures included — which `isinstance` cannot check.
    _conforms: SandboxBackend = Sandbox.__new__(Sandbox)
    _shell_filesystem_conforms: SupportsFilesystem = _ShellFilesystem.__new__(_ShellFilesystem)
