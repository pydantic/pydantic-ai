"""The user-facing sandbox facade.

Sandbox providers implement the small, frozen
[`SandboxBackend`][pydantic_ai.sandboxes.SandboxBackend] protocol. This facade owns
model-facing semantics such as decoding and windowed file reads, with a primitive-based
fallback and optional acceleration through extension protocols. Capabilities and user tools
consume it through [`RunContext.sandbox`][pydantic_ai.tools.RunContext.sandbox].
"""

from __future__ import annotations as _annotations

import base64
import posixpath
import shlex
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import anyio

from pydantic_ai.exceptions import UserError

from .protocol import (
    SandboxBackend,
    SandboxCommand,
    SandboxFileEntry,
    SandboxFilesystem,
    SandboxProcess,
    SandboxResult,
    SupportsFilesystem,
    SupportsReadBytesRange,
    SupportsStart,
)
from .references import SandboxRef

__all__ = ('FileWindow', 'Sandbox')

_READ_CHUNK_SIZE = 64 * 1024
_BASE64_WRITE_CHUNK_SIZE = 32 * 1024


@dataclass(frozen=True)
class FileWindow:
    """A line window of a sandbox file, as returned by [`Sandbox.read_file`][pydantic_ai.sandboxes.Sandbox.read_file]."""

    lines: tuple[str, ...]
    """The requested lines, without trailing newlines; a trailing `\r` (Windows line ending)
    is also stripped. For byte-exact access, use `fs.read_bytes`.
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


@dataclass(frozen=True)
class _ShellFileEntry:
    name: str
    path: str
    is_dir: bool
    size: int | None


class _ShellFilesystem:
    """POSIX-shell implementation of `SandboxFilesystem`."""

    def __init__(self, backend: SandboxBackend):
        self._backend = backend

    async def read_bytes(self, path: str) -> bytes:
        result = await self._backend.run(f'base64 < {shlex.quote(path)}', shell=True)
        await self._raise_for_error(result, path, missing=True)
        return base64.b64decode(result.stdout)

    async def read_bytes_range(self, path: str, start: int, end: int) -> bytes:
        quoted_path = shlex.quote(path)
        result = await self._backend.run(
            f'test -f {quoted_path} && tail -c +{start + 1} {quoted_path} | head -c {end - start} | base64',
            shell=True,
        )
        await self._raise_for_error(result, path, missing=True)
        data = base64.b64decode(result.stdout)
        if not data and end > start:
            # POSIX sh has no `pipefail`, so a failed `tail`/`head` can be hidden by
            # `base64` succeeding with empty input. Distinguish that from a real EOF.
            try:
                size = (await self.stat(path)).size
            except OSError as error:
                raise OSError(f'failed to read {path!r}') from error
            if size is not None and start < size:
                raise OSError(f'failed to read {path!r}')
        return data

    async def write_bytes(self, path: str, data: bytes) -> None:
        parent = posixpath.dirname(path)
        temporary_path = posixpath.join(parent, f'.pydantic-ai-{uuid.uuid4().hex}.tmp')
        quoted_parent = shlex.quote(parent)
        quoted_temporary_path = shlex.quote(temporary_path)
        result = await self._backend.run(f'mkdir -p {quoted_parent} && : > {quoted_temporary_path}', shell=True)
        await self._raise_for_error(result, path)

        encoded = base64.b64encode(data).decode()
        try:
            # Keep each command well below `ARG_MAX`; large payloads cannot be written in one invocation.
            for start in range(0, len(encoded), _BASE64_WRITE_CHUNK_SIZE):
                chunk = shlex.quote(encoded[start : start + _BASE64_WRITE_CHUNK_SIZE])
                result = await self._backend.run(
                    f"printf '%s' {chunk} >> {quoted_temporary_path}",
                    shell=True,
                )
                await self._raise_for_error(result, path)

            quoted_path = shlex.quote(path)
            result = await self._backend.run(
                f'base64 -d < {quoted_temporary_path} > {quoted_path}; '
                f'status=$?; rm -f {quoted_temporary_path}; exit $status',
                shell=True,
            )
            await self._raise_for_error(result, path)
        except BaseException:
            try:
                await self._backend.run(f'rm -f {quoted_temporary_path}', shell=True)
            except Exception:
                pass
            raise

    async def stat(self, path: str) -> _ShellFileEntry:
        quoted_path = shlex.quote(path)
        directory_result = await self._backend.run(f'test -d {quoted_path}', shell=True)
        if directory_result.exit_code == 0:
            return _ShellFileEntry(
                name=posixpath.basename(posixpath.normpath(path)),
                path=path,
                is_dir=True,
                size=None,
            )

        size_result = await self._backend.run(f'wc -c < {quoted_path}', shell=True)
        await self._raise_for_error(size_result, path, missing=True)
        return _ShellFileEntry(
            name=posixpath.basename(posixpath.normpath(path)),
            path=path,
            is_dir=False,
            size=int(size_result.stdout.strip()),
        )

    async def list_dir(self, path: str) -> tuple[_ShellFileEntry, ...]:
        quoted_path = shlex.quote(path)
        result = await self._backend.run(
            f'test -d {quoted_path} && find {quoted_path} -mindepth 1 -maxdepth 1 -print0',
            shell=True,
        )
        await self._raise_for_error(result, path, missing=True)
        directory_result = await self._backend.run(
            f'find {quoted_path} -mindepth 1 -maxdepth 1 -type d -print0',
            shell=True,
        )
        await self._raise_for_error(directory_result, path, missing=True)

        directories = {entry for entry in directory_result.stdout.split('\0') if entry}
        return tuple(
            _ShellFileEntry(
                name=posixpath.basename(entry_path),
                path=entry_path,
                is_dir=entry_path in directories,
                size=None,
            )
            for entry_path in sorted(entry for entry in result.stdout.split('\0') if entry)
        )

    async def make_dir(self, path: str) -> None:
        result = await self._backend.run(f'mkdir -p {shlex.quote(path)}', shell=True)
        await self._raise_for_error(result, path)

    async def remove(self, path: str) -> None:
        quoted_path = shlex.quote(path)
        result = await self._backend.run(
            f'(test -e {quoted_path} || test -L {quoted_path}) && rm -rf {quoted_path}',
            shell=True,
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
        raise OSError(message)


class _DeferredFilesystem:
    """Filesystem proxy that connects its sandbox before each operation."""

    def __init__(self, filesystem: Callable[[], Awaitable[SandboxFilesystem]]):
        self._get_filesystem = filesystem

    async def read_bytes(self, path: str) -> bytes:
        return await (await self._get_filesystem()).read_bytes(path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        await (await self._get_filesystem()).write_bytes(path, data)

    async def stat(self, path: str) -> SandboxFileEntry:
        return await (await self._get_filesystem()).stat(path)

    async def list_dir(self, path: str) -> Sequence[SandboxFileEntry]:
        return await (await self._get_filesystem()).list_dir(path)

    async def make_dir(self, path: str) -> None:
        await (await self._get_filesystem()).make_dir(path)

    async def remove(self, path: str) -> None:
        await (await self._get_filesystem()).remove(path)

    async def exists(self, path: str) -> bool:
        return await (await self._get_filesystem()).exists(path)


class Sandbox:
    """Rich sandbox interface exposed to tools and capabilities.

    The facade delegates the backend floor and adds filesystem access, path resolution, and
    uniform text and windowed-file helpers. Use
    [`backend`][pydantic_ai.sandboxes.Sandbox.backend] as an escape hatch for provider-specific
    functionality.
    """

    def __init__(self, backend: SandboxBackend):
        self._initialize(backend=backend, ref=None, resolver=None)

    def _initialize(
        self,
        *,
        backend: SandboxBackend | None,
        ref: SandboxRef | None,
        resolver: Callable[[SandboxRef], Awaitable[SandboxBackend]] | None,
    ) -> None:
        self._backend: SandboxBackend | None = backend
        self._ref = ref
        self._resolver = resolver
        self._connect_lock: anyio.Lock | None = anyio.Lock() if ref is not None else None
        self._shell_filesystem: _ShellFilesystem | None = None
        self._deferred_filesystem: _DeferredFilesystem | None = None

    @classmethod
    def wrap(cls, value: SandboxBackend) -> Sandbox:
        """Wrap `value`, returning an existing facade unchanged."""
        return value if isinstance(value, Sandbox) else cls(value)

    @classmethod
    def from_ref(
        cls,
        ref: SandboxRef,
        resolver: Callable[[SandboxRef], Awaitable[SandboxBackend]],
    ) -> Sandbox:
        """Create a facade that connects to `ref` on its first operation."""
        sandbox = cls.__new__(cls)
        sandbox._initialize(backend=None, ref=ref, resolver=resolver)
        return sandbox

    @property
    def _sandbox_ref(self) -> SandboxRef | None:
        """The deferred identity, for framework integrations that serialize sandbox state."""
        return self._ref

    @property
    def _live_backend(self) -> SandboxBackend | None:
        """The connected backend, without triggering connection."""
        return self._backend

    @property
    def backend(self) -> SandboxBackend:
        """The wrapped backend, for access to provider-specific functionality."""
        if self._backend is None:
            assert self._ref is not None
            raise UserError(
                f'Sandbox {self._ref.sandbox_id!r} for provider {self._ref.provider!r} has not connected yet. '
                'Call an async sandbox operation before accessing `sandbox.backend`.'
            )
        return self._backend

    @property
    def provider(self) -> str:
        if self._ref is not None:
            return self._ref.provider
        assert self._backend is not None
        return self._backend.provider

    @property
    def sandbox_id(self) -> str:
        if self._ref is not None:
            return self._ref.sandbox_id
        assert self._backend is not None
        return self._backend.sandbox_id

    @property
    def fs(self) -> SandboxFilesystem:
        if self._backend is not None:
            return self._filesystem_for_backend(self._backend)
        if self._deferred_filesystem is None:
            self._deferred_filesystem = _DeferredFilesystem(self._filesystem)
        return self._deferred_filesystem

    async def _ensure_backend(self) -> SandboxBackend:
        if self._backend is not None:
            return self._backend
        assert self._ref is not None
        assert self._resolver is not None
        assert self._connect_lock is not None
        async with self._connect_lock:
            if self._backend is None:
                self._backend = await self._resolver(self._ref)
        return self._backend

    def _filesystem_for_backend(self, backend: SandboxBackend) -> SandboxFilesystem:
        if isinstance(backend, SupportsFilesystem):
            return backend.fs
        if self._shell_filesystem is None:
            self._shell_filesystem = _ShellFilesystem(backend)
        return self._shell_filesystem

    async def _filesystem(self) -> SandboxFilesystem:
        return self._filesystem_for_backend(await self._ensure_backend())

    async def run(
        self,
        command: SandboxCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        output_limit: int | None = None,
    ) -> SandboxResult:
        """Execute a command and wait for it to complete.

        Delegates to [`SandboxBackend.run`][pydantic_ai.sandboxes.SandboxBackend.run]; arguments
        and contracts are documented there.
        """
        backend = await self._ensure_backend()
        return await backend.run(command, shell=shell, cwd=cwd, env=env, timeout=timeout, output_limit=output_limit)

    async def start(
        self,
        command: SandboxCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        output_limit: int | None = None,
    ) -> SandboxProcess:
        """Start a command without waiting, returning a handle to the running process.

        Requires a backend implementing [`SupportsStart`][pydantic_ai.sandboxes.SupportsStart];
        otherwise raises `NotImplementedError` — background the command over
        [`run()`][pydantic_ai.sandboxes.Sandbox.run] with `shell=True` instead.
        """
        backend = await self._ensure_backend()
        if isinstance(backend, SupportsStart):
            return await backend.start(
                command, shell=shell, cwd=cwd, env=env, timeout=timeout, output_limit=output_limit
            )
        raise NotImplementedError(
            'This sandbox backend does not implement `start()`; run the command with `shell=True` '
            'and shell backgrounding via `run()`, or use a backend that implements `SupportsStart`.'
        )

    async def working_dir(self) -> str:
        """The sandbox's default working directory (absolute POSIX path)."""
        return await (await self._ensure_backend()).working_dir()

    async def resolve(self, path: str, *, base: str | None = None) -> str:
        """Resolve a possibly-relative path to an absolute POSIX path.

        Joins `path` onto `base` (default: [`working_dir`][pydantic_ai.sandboxes.Sandbox.working_dir])
        and normalizes it textually. This is a spelling convenience for model-supplied paths,
        **not** a confinement mechanism: `..` segments can escape `base` and symlinks are not
        inspected. Isolation is the sandbox's job, not this method's.
        """
        if posixpath.isabs(path):
            return posixpath.normpath(path)
        return posixpath.normpath(posixpath.join(base or await self.working_dir(), path))

    async def read_text(self, path: str, *, encoding: str = 'utf-8') -> str:
        """Read text from `path`, resolving relative paths through the backend first."""
        resolved_path = await self.resolve(path)
        return (await self.fs.read_bytes(resolved_path)).decode(encoding)

    async def write_text(self, path: str, content: str, *, encoding: str = 'utf-8') -> None:
        """Write text to `path`, resolving relative paths through the backend first."""
        resolved_path = await self.resolve(path)
        await self.fs.write_bytes(resolved_path, content.encode(encoding))

    async def read_file(self, path: str, *, offset: int = 1, limit: int | None = None) -> FileWindow:
        """Read a line window from `path`, resolving relative paths through the backend first.

        `offset` is the 1-based first line and `limit` is the maximum number of lines. When
        `limit` is `None`, the window extends through EOF.
        """
        if offset < 1:
            raise ValueError('`offset` must be at least 1')
        if limit is not None and limit < 1:
            raise ValueError('`limit` must be at least 1')

        resolved_path = await self.resolve(path)
        filesystem = await self._filesystem()
        if limit is not None and isinstance(filesystem, SupportsReadBytesRange):
            return await self._read_file_range(filesystem, resolved_path, offset, limit)

        data = await filesystem.read_bytes(resolved_path)
        return _window_from_data(data, offset, limit)

    async def _read_file_range(
        self, filesystem: SupportsReadBytesRange, path: str, offset: int, limit: int
    ) -> FileWindow:
        data = bytearray()
        start = 0
        newline_count = 0
        # Newlines that terminate the window's lines; one further byte proves `has_more`.
        window_newlines = offset + limit - 1

        while True:
            chunk = await filesystem.read_bytes_range(path, start, start + _READ_CHUNK_SIZE)
            data.extend(chunk)
            newline_count += chunk.count(b'\n')
            if len(chunk) < _READ_CHUNK_SIZE:
                # EOF: totals are known, and the window may be incomplete — defer to the full logic.
                return _window_from_data(bytes(data), offset, limit)
            if newline_count >= window_newlines:
                window_end = _byte_after_newline(data, window_newlines)
                if len(data) > window_end:
                    line_start = _byte_after_newline(data, offset - 1)
                    text = bytes(data[line_start : window_end - 1]).decode('utf-8', errors='replace')
                    return FileWindow(_split_lines(text), offset, True, None)
            start += len(chunk)


def _byte_after_newline(data: bytearray, count: int) -> int:
    position = 0
    for _ in range(count):
        position = data.index(b'\n', position) + 1
    return position


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
    # Sandbox satisfies the SandboxBackend protocol structurally; this assignment makes the
    # type checker verify full conformance, including signatures.
    _conforms: SandboxBackend = Sandbox.__new__(Sandbox)
    _file_entry_conforms: SandboxFileEntry = _ShellFileEntry('', '', False, None)
    _filesystem_conforms: SandboxFilesystem = _ShellFilesystem.__new__(_ShellFilesystem)
    _range_conforms: SupportsReadBytesRange = _ShellFilesystem.__new__(_ShellFilesystem)
