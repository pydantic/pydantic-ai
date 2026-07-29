"""The user-facing sandbox facade.

Sandbox providers implement the small, frozen
[`SandboxBackend`][pydantic_ai.sandboxes.SandboxBackend] protocol. This facade owns
model-facing semantics such as decoding and windowed file reads, with a primitive-based
fallback and optional acceleration through extension protocols. Capabilities and user tools
consume it through [`RunContext.sandbox`][pydantic_ai.tools.RunContext.sandbox].
"""

from __future__ import annotations as _annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .protocol import (
    SandboxBackend,
    SandboxCommand,
    SandboxFilesystem,
    SandboxProcess,
    SandboxResult,
    SupportsReadBytesRange,
)

__all__ = ('FileWindow', 'Sandbox')

_READ_CHUNK_SIZE = 64 * 1024


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


class Sandbox:
    """Rich sandbox interface exposed to tools and capabilities.

    The facade delegates the full backend protocol and adds uniform text and windowed-file
    helpers. Use [`backend`][pydantic_ai.sandboxes.Sandbox.backend] as an escape hatch for
    provider-specific functionality.
    """

    def __init__(self, backend: SandboxBackend):
        self._backend = backend

    @classmethod
    def wrap(cls, value: SandboxBackend) -> Sandbox:
        """Wrap `value`, returning an existing facade unchanged."""
        return value if isinstance(value, Sandbox) else cls(value)

    @property
    def backend(self) -> SandboxBackend:
        """The wrapped backend, for access to provider-specific functionality."""
        return self._backend

    @property
    def provider(self) -> str:
        return self._backend.provider

    @property
    def sandbox_id(self) -> str:
        return self._backend.sandbox_id

    @property
    def fs(self) -> SandboxFilesystem:
        return self._backend.fs

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
        return await self._backend.run(
            command, shell=shell, cwd=cwd, env=env, timeout=timeout, output_limit=output_limit
        )

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
        return await self._backend.start(
            command, shell=shell, cwd=cwd, env=env, timeout=timeout, output_limit=output_limit
        )

    async def working_dir(self) -> str:
        return await self._backend.working_dir()

    async def resolve(self, path: str, *, base: str | None = None) -> str:
        return await self._backend.resolve(path, base=base)

    async def read_text(self, path: str, *, encoding: str = 'utf-8') -> str:
        """Read text from `path`, resolving relative paths through the backend first."""
        resolved_path = await self._backend.resolve(path)
        return (await self._backend.fs.read_bytes(resolved_path)).decode(encoding)

    async def write_text(self, path: str, content: str, *, encoding: str = 'utf-8') -> None:
        """Write text to `path`, resolving relative paths through the backend first."""
        resolved_path = await self._backend.resolve(path)
        await self._backend.fs.write_bytes(resolved_path, content.encode(encoding))

    async def read_file(self, path: str, *, offset: int = 1, limit: int | None = None) -> FileWindow:
        """Read a line window from `path`, resolving relative paths through the backend first.

        `offset` is the 1-based first line and `limit` is the maximum number of lines. When
        `limit` is `None`, the window extends through EOF.
        """
        if offset < 1:
            raise ValueError('`offset` must be at least 1')
        if limit is not None and limit < 1:
            raise ValueError('`limit` must be at least 1')

        resolved_path = await self._backend.resolve(path)
        filesystem = self._backend.fs
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
