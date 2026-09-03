"""The user-facing sandbox API.

Sandbox backends implement the small
[`SandboxBackend`][pydantic_ai.sandboxes.SandboxBackend] protocol and typically also
[`SupportsFilesystem`][pydantic_ai.sandboxes.SupportsFilesystem]. The `Sandbox` object owns
model-facing semantics such as decoding and windowed file reads. Capabilities and user tools
consume it through [`RunContext.sandbox`][pydantic_ai.tools.RunContext.sandbox].
"""

from __future__ import annotations as _annotations

import posixpath
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._connection import close_backend_connection
from .protocol import (
    SandboxBackend,
    SandboxCommand,
    SandboxFilesystem,
    SandboxResult,
    SupportsFilesystem,
)
from .references import SandboxRef

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


class Sandbox:
    """Rich sandbox interface exposed to tools and capabilities.

    `Sandbox` forwards the backend's required methods and adds filesystem access, path
    resolution, and uniform text and windowed-file helpers. Use
    [`backend`][pydantic_ai.sandboxes.Sandbox.backend] to reach provider-specific
    functionality.
    """

    def __init__(self, backend: SandboxBackend, *, ref: SandboxRef | None = None):
        self._backend = backend
        self._ref = ref
        self._backend_closed = False
        self._working_dir: str | None = None

    @classmethod
    def wrap(cls, value: SandboxBackend) -> Sandbox:
        """Wrap `value`, returning an existing `Sandbox` unchanged."""
        return value if isinstance(value, Sandbox) else cls(value)

    @property
    def backend(self) -> SandboxBackend:
        """The wrapped backend, for access to provider-specific functionality."""
        return self._backend

    @property
    def ref(self) -> SandboxRef | None:
        """The serializable identity used to obtain this backend, if it can be reconstructed."""
        return self._ref

    @property
    def sandbox_id(self) -> str:
        return self._ref.sandbox_id if self._ref is not None else self._backend.sandbox_id

    @property
    def fs(self) -> SandboxFilesystem:
        return self._filesystem_for_backend(self._backend)

    async def _close_connection(self) -> None:
        """Detach the connection opened by this `Sandbox` without terminating the environment."""
        if not self._backend_closed:
            await close_backend_connection(self._backend)
            self._backend_closed = True

    def _filesystem_for_backend(self, backend: SandboxBackend) -> SandboxFilesystem:
        if isinstance(backend, SupportsFilesystem):
            return backend.fs
        raise NotImplementedError(
            f'Sandbox backend {type(backend).__name__} does not implement `SupportsFilesystem`; '
            'implement `fs` on the backend, or reach for files through `sandbox.run(...)` shell commands.'
        )

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
        return await self._backend.run(command, shell=shell, cwd=cwd, env=env, timeout=timeout)

    async def working_dir(self) -> str:
        """The sandbox's default working directory (absolute, filesystem-canonical POSIX path).

        The value is cached after the first call because it never changes for an environment.
        The canonicality contract is documented on
        [`SandboxBackend.working_dir`][pydantic_ai.sandboxes.SandboxBackend.working_dir].
        """
        if self._working_dir is None:
            self._working_dir = await self._backend.working_dir()
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
                filesystem = self._filesystem_for_backend(self._backend)
            except NotImplementedError as e:
                raise NotImplementedError(
                    'This sandbox could not perform a windowed file read. Provide a working `sed` '
                    'command through `run()`, or implement `SupportsFilesystem` on the backend.'
                ) from e
        else:
            filesystem = self._filesystem_for_backend(self._backend)

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
            # argv, never shell=True: the path is an argument, not shell-interpreted text.
            # `{end}q` stops `sed` at the window instead of scanning to EOF, and the timeout
            # bounds the optimization on paths that never finish.
            result = await self._backend.run(
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
            entry = await self._filesystem_for_backend(self._backend).stat(path)
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
