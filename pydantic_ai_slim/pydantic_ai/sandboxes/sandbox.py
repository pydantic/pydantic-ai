"""The user-facing sandbox API.

Sandbox backends implement the small
[`SandboxBackend`][pydantic_ai.sandboxes.SandboxBackend] protocol and typically also
[`SupportsFilesystem`][pydantic_ai.sandboxes.SupportsFilesystem]. The `Sandbox` object owns
model-facing semantics such as decoding and windowed file reads. Capabilities and user tools
consume it through [`RunContext.sandbox`][pydantic_ai.tools.RunContext.sandbox].
"""

from __future__ import annotations as _annotations

import functools
import posixpath
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import anyio

from pydantic_ai.exceptions import UserError

from ._connection import close_backend_connection
from .protocol import (
    SandboxBackend,
    SandboxCommand,
    SandboxFileEntry,
    SandboxFilesystem,
    SandboxProcess,
    SandboxResult,
    SupportsFilesystem,
    SupportsStart,
)
from .readonly import ReadOnlySandbox
from .references import SandboxRef

_SandboxResolver: TypeAlias = 'Callable[[SandboxRef], Awaitable[SandboxBackend]]'
"""Turns a serializable sandbox identity into a live backend — connect, never create.

Must raise (typically [`UserError`][pydantic_ai.exceptions.UserError]) when nothing recognizes
the reference.
"""

__all__ = ('FileWindow', 'Sandbox')

_SHELL_SLICE_TIMEOUT = 10
"""Deadline in seconds for the `sed` fast path in `read_file`.

The slice is an optimization, so a slow or wedged attempt (a FIFO path, a stalled mount)
falls back to the authoritative filesystem read instead of hanging the run without bound."""


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

    `Sandbox` forwards the backend's required methods and adds filesystem access, path
    resolution, and uniform text and windowed-file helpers. Use
    [`backend`][pydantic_ai.sandboxes.Sandbox.backend] to reach provider-specific
    functionality.
    """

    def __init__(self, backend: SandboxBackend):
        self._initialize(backend=backend, ref=None, capability_id=None, resolver=None, close_backend=False)

    def _initialize(
        self,
        *,
        backend: SandboxBackend | None,
        ref: SandboxRef | None,
        capability_id: str | None,
        resolver: Callable[[SandboxRef | None], Awaitable[SandboxBackend]] | None,
        close_backend: bool,
    ) -> None:
        self._backend: SandboxBackend | None = backend
        self._ref = ref
        self._capability_id = capability_id
        self._resolver = resolver
        self._close_backend = close_backend
        self._backend_closed = False
        self._deferred_filesystem: _DeferredFilesystem | None = None

    @functools.cached_property
    def _connect_lock(self) -> anyio.Lock:
        # `anyio.Lock` binds to the event loop on which it is first used.
        return anyio.Lock()

    @classmethod
    def wrap(cls, value: SandboxBackend) -> Sandbox:
        """Wrap `value`, returning an existing `Sandbox` unchanged."""
        return value if isinstance(value, Sandbox) else cls(value)

    @classmethod
    def _from_ref(cls, ref: SandboxRef, resolver: _SandboxResolver) -> Sandbox:
        """Create a `Sandbox` that connects to `ref` through `resolver` on its first operation."""
        sandbox = cls.__new__(cls)
        sandbox._initialize(
            backend=None,
            ref=ref,
            capability_id=ref.capability_id,
            resolver=lambda value: resolver(value if value is not None else ref),
            close_backend=True,
        )
        return sandbox

    @classmethod
    def _from_provider(
        cls,
        capability_id: str,
        resolver: Callable[[SandboxRef | None], Awaitable[SandboxBackend]],
    ) -> Sandbox:
        """Create a `Sandbox` that asks one capability for a backend connection on first use."""
        sandbox = cls.__new__(cls)
        sandbox._initialize(backend=None, ref=None, capability_id=capability_id, resolver=resolver, close_backend=True)
        return sandbox

    def _durable_identity(self) -> SandboxRef | SandboxBackend | None:
        """Identity for durable frameworks: a ref, backend, or `None` for provider-only resolution."""
        if self._ref is not None:
            return self._ref
        return self._backend

    def _durable_capability_id(self) -> str | None:
        """Return the capability that resolves a provider-only sandbox across a durable boundary."""
        return self._capability_id if self._backend is None and self._ref is None else None

    def _is_framework_default(self) -> bool:
        """Whether this `Sandbox` contains the framework's implicit unavailable placeholder."""
        from ._policy import is_default_sandbox_backend

        return is_default_sandbox_backend(self._backend)

    @property
    def backend(self) -> SandboxBackend:
        """The wrapped backend, for access to provider-specific functionality."""
        if self._backend is None:
            if self._ref is None:
                raise UserError(
                    'The capability-provided sandbox has not connected yet. '
                    'Call an async sandbox operation before accessing `sandbox.backend`.'
                )
            raise UserError(
                f'Sandbox {self._ref.sandbox_id!r} for provider {self._ref.provider!r} has not connected yet. '
                'Call an async sandbox operation before accessing `sandbox.backend`.'
            )
        return self._backend

    @property
    def provider(self) -> str:
        if self._ref is not None:
            return self._ref.provider
        if self._backend is None:
            raise UserError(
                'The capability-provided sandbox has not connected yet. '
                'Call an async sandbox operation before accessing `sandbox.provider`.'
            )
        return self._backend.provider

    @property
    def sandbox_id(self) -> str:
        if self._ref is not None:
            return self._ref.sandbox_id
        if self._backend is None:
            raise UserError(
                'The capability-provided sandbox has not connected yet. '
                'Call an async sandbox operation before accessing `sandbox.sandbox_id`.'
            )
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
        assert self._resolver is not None
        async with self._connect_lock:
            if self._backend is None:
                self._backend = await self._resolver(self._ref)
        return self._backend

    async def _close_connected_backend(self) -> None:
        """Detach the connection opened by this `Sandbox` without terminating the environment."""
        if not self._close_backend:
            return
        async with self._connect_lock:
            backend = self._backend
            if not self._backend_closed and backend is not None:
                await close_backend_connection(backend)
                self._backend_closed = True

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
    ) -> SandboxResult:
        """Execute a command and wait for it to complete.

        Delegates to [`SandboxBackend.run`][pydantic_ai.sandboxes.SandboxBackend.run]; arguments
        and contracts are documented there.
        """
        _require_absolute_cwd(cwd)
        backend = await self._ensure_backend()
        return await backend.run(command, shell=shell, cwd=cwd, env=env, timeout=timeout)

    async def start(
        self,
        command: SandboxCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> SandboxProcess:
        """Start a command without waiting, returning a handle to the running process.

        Requires a backend implementing [`SupportsStart`][pydantic_ai.sandboxes.SupportsStart];
        otherwise raises `NotImplementedError` — background the command over
        [`run()`][pydantic_ai.sandboxes.Sandbox.run] with `shell=True` instead.
        """
        _require_absolute_cwd(cwd)
        backend = await self._ensure_backend()
        if isinstance(backend, SupportsStart):
            return await backend.start(command, shell=shell, cwd=cwd, env=env, timeout=timeout)
        raise NotImplementedError(
            'This sandbox backend does not implement `start()`; run the command with `shell=True` '
            'and shell backgrounding via `run()`, or use a backend that implements `SupportsStart`.'
        )

    async def working_dir(self) -> str:
        """The sandbox's default working directory (absolute, filesystem-canonical POSIX path).

        The canonicality contract is documented on
        [`SandboxBackend.working_dir`][pydantic_ai.sandboxes.SandboxBackend.working_dir].
        """
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
        if base is not None and not posixpath.isabs(base):
            raise ValueError(f'base must be an absolute path, got {base!r}')
        return posixpath.normpath(posixpath.join(base or await self.working_dir(), path))

    async def read_text(self, path: str, *, encoding: str = 'utf-8') -> str:
        """Read text from `path`, resolving relative paths through the backend first.

        Decoding is strict: undecodable bytes raise `UnicodeDecodeError`. For a lossy,
        model-facing view use [`read_file`][pydantic_ai.sandboxes.Sandbox.read_file].
        """
        resolved_path = await self.resolve(path)
        return (await self.fs.read_bytes(resolved_path)).decode(encoding)

    async def write_text(self, path: str, content: str, *, encoding: str = 'utf-8') -> None:
        """Write text to `path`, resolving relative paths through the backend first."""
        resolved_path = await self.resolve(path)
        await self.fs.write_bytes(resolved_path, content.encode(encoding))

    async def read_file(self, path: str, *, offset: int = 1, limit: int | None = None) -> FileWindow:
        """Read a line window from `path`, resolving relative paths through the backend first.

        `offset` is the 1-based first line and `limit` is the maximum number of lines. When
        `limit` is `None`, the window extends through EOF. `limit` bounds returned lines, not
        bytes or characters: a single line may be arbitrarily large, and a backend without a
        usable in-sandbox `sed` command may transfer the whole file through its filesystem API
        before `Sandbox` applies the line window.

        This is a model-facing view: content is decoded as UTF-8 with U+FFFD replacement for
        undecodable bytes. Use [`read_text`][pydantic_ai.sandboxes.Sandbox.read_text] for
        strict decoding or `fs.read_bytes` for exact bytes. Reading a special file that never
        ends (a FIFO, a device) blocks the way the underlying filesystem read does.
        """
        if offset < 1:
            raise ValueError('`offset` must be at least 1')
        if limit is not None and limit < 1:
            raise ValueError('`limit` must be at least 1')

        resolved_path = await self.resolve(path)
        filesystem: SandboxFilesystem
        if limit is not None:
            # Before the filesystem lookup: a backend with only `run()` can still serve
            # windowed reads through the slice, and command-capable remote backends avoid
            # transferring the whole file. This bounds line count, not byte size: one line
            # may still be arbitrarily large.
            window = await self._read_file_via_shell(resolved_path, offset, limit)
            if window is not None:
                return window

            await self._validate_bounded_read_path(resolved_path)
            try:
                filesystem = await self._filesystem()
            except NotImplementedError as e:
                raise NotImplementedError(
                    'This sandbox could not perform a windowed file read. Provide a working `sed` '
                    'command through `run()`, or implement `SupportsFilesystem` on the backend.'
                ) from e
        else:
            filesystem = await self._filesystem()

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
            backend = await self._ensure_backend()
            # The policy wrapper blocks caller-supplied commands, but this argv is an
            # implementation-owned, non-mutating read. Running it on the wrapped backend keeps
            # windowed reads bounded without granting command execution through the wrapper.
            command_backend = backend._backend_for_internal_read() if isinstance(backend, ReadOnlySandbox) else backend  # pyright: ignore[reportPrivateUsage]
            # argv, never shell=True: the path is an argument, not shell-interpreted text.
            # `{end}q` stops `sed` at the window instead of scanning to EOF, and the timeout
            # bounds the optimization on paths that never finish.
            result = await command_backend.run(
                ['sed', '-n', f'{offset},{end}p;{end}q', path], timeout=_SHELL_SLICE_TIMEOUT
            )
        except Exception:
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
            entry = await (await self._filesystem()).stat(path)
        except NotImplementedError:
            return
        if entry.is_dir:
            raise IsADirectoryError(path)


def _require_absolute_cwd(cwd: str | None) -> None:
    # A relative cwd has no sandbox meaning: backends would resolve it against ambient state
    # (the host process's working directory for a local backend), outside the sandbox root.
    if cwd is not None and not posixpath.isabs(cwd):
        raise ValueError(
            f'cwd must be an absolute POSIX path, got {cwd!r}; resolve relative paths with `sandbox.resolve()` first'
        )


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
