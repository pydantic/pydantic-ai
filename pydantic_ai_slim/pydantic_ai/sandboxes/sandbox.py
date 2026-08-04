"""The user-facing sandbox facade.

Sandbox providers implement the small
[`SandboxBackend`][pydantic_ai.sandboxes.SandboxBackend] protocol and typically also
[`SupportsFilesystem`][pydantic_ai.sandboxes.SupportsFilesystem]. This facade owns
model-facing semantics such as decoding and windowed file reads, with optional acceleration
through extension protocols. Capabilities and user tools consume it through
[`RunContext.sandbox`][pydantic_ai.tools.RunContext.sandbox].
"""

from __future__ import annotations as _annotations

import functools
import posixpath
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import anyio

from pydantic_ai.exceptions import UserError

from ._policy import DefaultLocalSandbox
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
from .references import SandboxConnector, SandboxRef, connect_sandbox_ref

__all__ = ('FileWindow', 'Sandbox')

_READ_CHUNK_SIZE = 64 * 1024


@dataclass(frozen=True, kw_only=True)
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

    The facade forwards the backend's required methods and adds filesystem access, path
    resolution, and uniform text and windowed-file helpers. Use
    [`backend`][pydantic_ai.sandboxes.Sandbox.backend] to reach provider-specific
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
        self._deferred_filesystem: _DeferredFilesystem | None = None
        self._working_dir: str | None = None

    @functools.cached_property
    def _connect_lock(self) -> anyio.Lock:
        # `anyio.Lock` binds to the event loop on which it is first used.
        return anyio.Lock()

    @classmethod
    def wrap(cls, value: SandboxBackend) -> Sandbox:
        """Wrap `value`, returning an existing facade unchanged."""
        return value if isinstance(value, Sandbox) else cls(value)

    @classmethod
    def from_ref(
        cls,
        ref: SandboxRef,
        connectors: Sequence[SandboxConnector] | Callable[[], Sequence[SandboxConnector]],
    ) -> Sandbox:
        """Create a facade that connects to `ref` on its first operation, using a matching connector."""

        async def resolve(ref: SandboxRef) -> SandboxBackend:
            resolved = connectors() if callable(connectors) else connectors
            return await connect_sandbox_ref(ref, resolved)

        sandbox = cls.__new__(cls)
        sandbox._initialize(backend=None, ref=ref, resolver=resolve)
        return sandbox

    def durable_identity(self) -> SandboxRef | SandboxBackend | None:
        """The sandbox's identity for durable frameworks: its deferred [`SandboxRef`][pydantic_ai.sandboxes.SandboxRef], the explicitly attached backend, or `None` for the framework default."""
        if self._ref is not None:
            return self._ref
        if isinstance(self._backend, DefaultLocalSandbox):
            return None
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
        async with self._connect_lock:
            if self._backend is None:
                self._backend = await self._resolver(self._ref)
        return self._backend

    def _filesystem_for_backend(self, backend: SandboxBackend) -> SandboxFilesystem:
        if isinstance(backend, SupportsFilesystem):
            return backend.fs
        raise NotImplementedError(
            f'Sandbox backend {backend.provider!r} does not implement `SupportsFilesystem`; '
            'implement `fs` on the backend, or reach for files through `sandbox.run(...)` shell commands.'
        )

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
        if self._working_dir is None:
            self._working_dir = await (await self._ensure_backend()).working_dir()
        return self._working_dir

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
        window: list[str] = []
        buffer = bytearray()
        line_number = 0
        start = 0

        while True:
            chunk = await filesystem.read_bytes_range(path, start, start + _READ_CHUNK_SIZE)
            start += len(chunk)
            buffer.extend(chunk)
            at_eof = len(chunk) < _READ_CHUNK_SIZE

            position = 0
            while (newline := buffer.find(b'\n', position)) >= 0:
                line_number += 1
                if line_number >= offset and len(window) < limit:
                    window.append(_decode_line(buffer[position:newline]))
                position = newline + 1
            del buffer[:position]

            if at_eof:
                total_lines = line_number + (1 if buffer else 0)
                if buffer and total_lines >= offset and len(window) < limit:
                    window.append(_decode_line(buffer))
                return FileWindow(
                    lines=tuple(window),
                    start_line=offset,
                    has_more=offset - 1 + limit < total_lines,
                    total_lines=total_lines,
                )
            if len(window) == limit and (buffer or line_number >= offset + limit):
                return FileWindow(lines=tuple(window), start_line=offset, has_more=True, total_lines=None)


def _decode_line(data: bytes | bytearray) -> str:
    return data.decode('utf-8', errors='replace').removesuffix('\r')


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
