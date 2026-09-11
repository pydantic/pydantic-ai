"""The user-facing workspace API.

Workspace backends implement the small
[`WorkspaceBackend`][pydantic_ai.workspaces.WorkspaceBackend] protocol and typically also
[`SupportsFilesystem`][pydantic_ai.workspaces.SupportsFilesystem]. The `Workspace` object owns
model-facing semantics such as decoding and windowed file reads. Capabilities and user tools
consume it through [`RunContext.workspace`][pydantic_ai.tools.RunContext.workspace].
"""

from __future__ import annotations as _annotations

import base64
import posixpath
import shlex
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import anyio

from pydantic_ai.exceptions import UserError

from .protocol import (
    FileEntry,
    SupportsFilesystem,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceError,
    WorkspaceFileEntry,
    WorkspaceRef,
    WorkspaceResult,
    WorkspaceTimeoutError,
)

__all__ = ('FileWindow', 'Workspace', 'WrapperWorkspace')

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

_DEFAULT_READ_LINES = 2000
"""Line cap applied by `read_file` when the caller passes no `limit`.

Claude Code, Pi, OpenCode, and Gemini CLI all default to 2000 lines. Pass `limit=None` to drop
the line cap (the byte cap still applies unless `max_bytes=None` too)."""

_DEFAULT_READ_BYTES = 50 * 1024
"""Byte cap applied by `read_file` when the caller passes no `max_bytes`.

Pi and OpenCode both default to 50 KiB and take whichever of the line and byte caps hits first.
Pass `max_bytes=None` to drop the byte cap. `read_text` and `read_bytes` are uncapped."""

_BINARY_SNIFF_BYTES = 8192
"""Bytes sampled from a file's head to classify it as binary before a window crosses the wire."""


@dataclass(frozen=True, kw_only=True)
class FileWindow:
    """A line window of a workspace file, as returned by [`Workspace.read_file`][pydantic_ai.workspaces.Workspace.read_file].

    Always inspect [`truncated`][pydantic_ai.workspaces.FileWindow.truncated] before treating
    `lines` as the whole file. [`text`][pydantic_ai.workspaces.FileWindow.text] includes a
    truncation notice when the window was cut, so stringifying the result cannot hide a cap.
    """

    lines: tuple[str, ...]
    """Complete lines in this window, without trailing newlines; a trailing `\r` (Windows
    line ending) is also stripped. Lines are never mid-line truncated and never carry a
    truncation marker in the content itself. For byte-exact access, use `read_bytes`.
    """
    start_line: int
    """1-based line number of `lines[0]` (the requested `offset`, even when `lines` is empty)."""
    has_more: bool
    """Whether the file has content after this window."""
    total_lines: int | None
    """Total number of lines in the file, when known (the read reached EOF, or the whole
    file was already in memory); `None` otherwise."""
    truncated_by: Literal['lines', 'bytes'] | None = None
    """Which cap cut this window: `'lines'`, `'bytes'`, or `None` when the window is complete."""
    first_line_exceeds_limit: bool = False
    """Whether the first requested line is longer than `max_bytes` by itself.

    When `True`, `lines` is empty: a partial line is not returned as if it were complete.
    """
    binary: bool = False
    """Whether the file is binary. When `True`, `lines` is empty and `text` is a size marker,
    not content: binary bytes are not decoded into the model context. Use `read_bytes` for the
    raw bytes."""
    byte_size: int | None = None
    """The file's size in bytes when known; reported for a binary file so the marker can name it."""

    @property
    def truncated(self) -> bool:
        """Whether this window is not the complete requested view.

        True when `has_more` or `first_line_exceeds_limit`. Check this before treating `lines`
        or `text` as the whole file.
        """
        return self.has_more or self.first_line_exceeds_limit

    @property
    def remaining_lines(self) -> int | None:
        """Lines after this window, when `total_lines` is known; `None` otherwise."""
        if self.total_lines is None:
            return None
        consumed = self.start_line - 1 + len(self.lines)
        return max(self.total_lines - consumed, 0)

    @property
    def end_line(self) -> int | None:
        """1-based line number of `lines[-1]`, or `None` when `lines` is empty."""
        if not self.lines:
            return None
        return self.start_line + len(self.lines) - 1

    @property
    def text(self) -> str:
        if self.binary:
            size = 'unknown size' if self.byte_size is None else f'{self.byte_size} bytes'
            return f'[Binary file ({size}). Use a binary-aware tool to inspect it.]'
        if self.first_line_exceeds_limit:
            return _truncation_notice(self)
        body = '\n'.join(self.lines)
        if not self.truncated:
            return body
        notice = _truncation_notice(self)
        return f'{body}\n\n{notice}' if body else notice


class _ShellFilesystem(SupportsFilesystem):
    """Derive filesystem operations from a backend's command-execution primitive.

    This is the portability floor for command-capable workspaces. Backends should implement
    `SupportsFilesystem` when their provider has a native API: native calls avoid the shell's
    utility assumptions and the base64 transfer overhead used here to preserve arbitrary bytes.
    """

    def __init__(self, backend: WorkspaceBackend):
        self._backend = backend

    async def read_bytes(self, path: str) -> bytes:
        result = await self._backend.run(f'base64 < {shlex.quote(path)}', shell=True)
        await self._raise_for_error(result, path, missing=True)
        try:
            return base64.b64decode(result.stdout)
        except ValueError as error:
            raise WorkspaceError(f'shell filesystem returned invalid base64 while reading {path!r}') from error

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
            raise WorkspaceError(f'shell filesystem returned an invalid size for {path!r}: {output!r}') from error
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
            raise WorkspaceError(f'shell filesystem returned an invalid directory listing for {path!r}') from error
        return tuple(
            FileEntry(
                name=posixpath.basename(entry_path),
                path=entry_path,
                is_dir=entry_path in directories,
                size=None,
            )
            for entry_path in sorted(entry for entry in entries if entry)
        )

    async def _list_paths(self, quoted_path: str, *, directories_only: bool = False) -> WorkspaceResult:
        temporary_path = f'/tmp/.pydantic-ai-{uuid.uuid4().hex}.list'
        quoted_temporary = shlex.quote(temporary_path)
        type_filter = r' -exec test -d {} \;' if directories_only else ''
        # Do not pipe `find` into `base64`: a POSIX shell reports only `base64`'s exit status and
        # could turn a failed traversal into a successful partial listing. The temporary file keeps
        # `find`'s status authoritative, and the trap removes it on every shell exit path.
        return await self._backend.run(
            f'file={quoted_temporary}; trap \'rm -f "$file"\' EXIT HUP INT TERM; '
            f'test -d {quoted_path} && '
            f'find -H {quoted_path} -mindepth 1 -maxdepth 1{type_filter} -print0 > "$file" && base64 < "$file"',
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

    async def _raise_for_error(self, result: WorkspaceResult, path: str, *, missing: bool = False) -> None:
        if result.exit_code == 0:
            return
        if missing and not await self.exists(path):
            raise FileNotFoundError(path)
        message = result.stderr.strip() or f'shell filesystem operation failed for {path!r}'
        raise WorkspaceError(message)


class Workspace(WorkspaceBackend):
    """Rich workspace interface exposed to tools and capabilities.

    `Workspace` forwards the backend's required methods and adds filesystem access, path
    resolution, and uniform text and windowed-file helpers. Use
    [`backend`][pydantic_ai.workspaces.Workspace.backend] to reach provider-specific
    functionality.
    """

    def __init__(
        self,
        backend: WorkspaceBackend,
    ):
        self._backend = backend

    @property
    def backend(self) -> WorkspaceBackend:
        """The wrapped backend, for access to provider-specific functionality."""
        return self._backend

    @property
    def ref(self) -> WorkspaceRef | None:
        """Identity of the environment when the backend has a reconnectable identity; otherwise `None`."""
        return self._backend.ref

    @property
    def _filesystem(self) -> SupportsFilesystem:
        backend = self._backend
        if isinstance(backend, SupportsFilesystem):
            return backend
        # Do not cache this adapter: the backend may provide native filesystem methods later.
        return _ShellFilesystem(backend)

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> WorkspaceResult:
        """Execute a command and wait for it to complete.

        Delegates to [`WorkspaceBackend.run`][pydantic_ai.workspaces.WorkspaceBackend.run]; arguments
        and contracts are documented there.
        """
        # Checked here as well as in the backend: a relative cwd has no workspace meaning, and the
        # wrapper is the seam every tool call goes through, so the error is the same whichever
        # backend is attached.
        if cwd is not None and not posixpath.isabs(cwd):
            raise ValueError(
                f'cwd must be an absolute POSIX path, got {cwd!r}; resolve relative paths with `workspace.resolve()` first'
            )
        return await self._backend.run(command, shell=shell, cwd=cwd, env=env, timeout=timeout)

    async def working_dir(self) -> str:
        """The workspace's default working directory (absolute, filesystem-canonical POSIX path).

        The canonicality contract is documented on
        [`WorkspaceBackend.working_dir`][pydantic_ai.workspaces.WorkspaceBackend.working_dir].
        """
        return await self._backend.working_dir()

    async def resolve(self, path: str, *, base: str | None = None) -> str:
        """Resolve a possibly-relative path to an absolute POSIX path.

        Joins `path` onto `base` (default: [`working_dir`][pydantic_ai.workspaces.Workspace.working_dir])
        and normalizes it textually. This is a spelling convenience for model-supplied paths,
        **not** a confinement mechanism: `..` segments can escape `base` and symlinks are not
        inspected. Isolation is the workspace's job, not this method's.
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

    async def stat(self, path: str) -> WorkspaceFileEntry:
        """Return metadata for a file or directory."""
        return await self._filesystem.stat(await self.resolve(path))

    async def list_dir(self, path: str) -> Sequence[WorkspaceFileEntry]:
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

    async def read_text(self, path: str, *, encoding: str = 'utf-8') -> str:
        """Read text from `path`, resolving relative paths through the backend first.

        Decoding is strict: undecodable bytes raise `UnicodeDecodeError`. For a lossy,
        model-facing view use [`read_file`][pydantic_ai.workspaces.Workspace.read_file].
        """
        return (await self.read_bytes(path)).decode(encoding)

    async def write_text(self, path: str, content: str, *, encoding: str = 'utf-8') -> None:
        """Write text to `path`, resolving relative paths through the backend first."""
        await self.write_bytes(path, content.encode(encoding))

    async def read_file(
        self,
        path: str,
        *,
        offset: int = 1,
        limit: int | None = _DEFAULT_READ_LINES,
        max_bytes: int | None = _DEFAULT_READ_BYTES,
    ) -> FileWindow:
        """Read a line window from `path`, capped for a model to read safely.

        `offset` is the 1-based first line. `limit` defaults to 2000 lines; `max_bytes` defaults
        to 50 KiB. Both caps apply at once and the window stops at whichever hits first. Pass
        `limit=None` and `max_bytes=None` together to read through end of file.

        The returned [`FileWindow`][pydantic_ai.workspaces.FileWindow] is structured so a cap
        cannot be missed: `truncated` is true when the window is incomplete, `truncated_by`
        names the cap that fired, and `text` includes a continuation notice. A single line
        longer than `max_bytes` yields an empty window with `first_line_exceeds_limit=True`
        rather than a partial line presented as complete.

        A file whose head contains a NUL byte is treated as binary: the returned window has
        `binary=True`, empty `lines`, and a `text` that names the size instead of decoding the
        bytes. Nothing large crosses the wire to make that decision.

        This is a model-facing view: text content is decoded as UTF-8 with U+FFFD replacement for
        undecodable bytes. Use [`read_text`][pydantic_ai.workspaces.Workspace.read_text] for strict
        decoding or [`read_bytes`][pydantic_ai.workspaces.Workspace.read_bytes] for exact, uncapped
        bytes. Reading a special file that never ends (a FIFO, a device) blocks the way the
        underlying filesystem read does.
        """
        if offset < 1:
            raise ValueError('`offset` must be at least 1')
        if limit is not None and limit < 1:
            raise ValueError('`limit` must be at least 1')
        if max_bytes is not None and max_bytes < 1:
            raise ValueError('`max_bytes` must be at least 1')
        if limit is not None or max_bytes is not None:
            # Bounded path: classify the file and slice the window inside the workspace, so a
            # binary or oversized file never crosses the wire in full. Returns `None` when the
            # shell utilities are unavailable, so the authoritative read below serves the window.
            window = await self._read_file_via_shell(path, offset, limit, max_bytes)
            if window is not None:
                return window

        data = await self.read_bytes(path)
        if _is_binary(data):
            return FileWindow(
                lines=(), start_line=offset, has_more=False, total_lines=None, binary=True, byte_size=len(data)
            )
        return _window_from_data(data, offset, limit, max_bytes)

    async def _read_file_via_shell(
        self, path: str, offset: int, limit: int | None, max_bytes: int | None
    ) -> FileWindow | None:
        """Classify and slice a file inside the workspace, so only a bounded amount crosses the wire.

        Returns `None` on failure (no usable `head`/`sed`, `run()` unsupported, or a slice that
        timed out), so the caller can fall back to the backend filesystem when available.
        `total_lines` is only reported when the slice provably reached EOF.
        """
        resolved_path = await self.resolve(path)
        is_binary = await self._sniff_is_binary(resolved_path)
        if is_binary is None:
            # The shell utilities are unavailable; the authoritative read classifies from bytes.
            return None
        if is_binary:
            return FileWindow(
                lines=(),
                start_line=offset,
                has_more=False,
                total_lines=None,
                binary=True,
                byte_size=await self._safe_size(resolved_path),
            )
        if limit is not None:
            end = offset + limit  # one extra line, to learn whether more exist
            sed_expr = f'{offset},{end}p;{end}q'
        else:
            sed_expr = f'{offset},$p'
        # `head -c` caps the bytes that cross the wire, so a window whose lines are individually
        # huge cannot drag the whole file across. The sed expression holds only integers and the
        # path is quoted, so `shell=True` is safe here. The timeout bounds the optimization on
        # paths that never finish.
        command = f'sed -n {shlex.quote(sed_expr)} {shlex.quote(resolved_path)}'
        if max_bytes is not None:
            command = f'{command} | head -c {max_bytes}'
        try:
            result = await self.run(command, shell=True, timeout=_SHELL_SLICE_TIMEOUT)
        except (NotImplementedError, OSError, WorkspaceTimeoutError, UserError):
            return None
        if result.exit_code != 0 or result.stderr:
            return None

        byte_capped = max_bytes is not None and len(result.stdout.encode('utf-8')) >= max_bytes
        lines = list(_split_lines(result.stdout))
        if lines and lines[-1] == '':
            lines.pop()
        if byte_capped and lines:
            # The ceiling cut the final line mid-way; drop it so no partial line is shown as
            # complete. If that was the only line, the window is empty and
            # `first_line_exceeds_limit` tells the caller to use a byte-range read.
            lines.pop()
        if not lines:
            await self._validate_bounded_read_path(resolved_path)
            if byte_capped:
                return FileWindow(
                    lines=(),
                    start_line=offset,
                    has_more=True,
                    total_lines=None,
                    truncated_by='bytes',
                    first_line_exceeds_limit=True,
                )
            # Empty output covers an empty file or an offset past EOF. The exact total is
            # unknown without scanning to EOF, which would defeat the bounded-read contract.
            return FileWindow(lines=(), start_line=offset, has_more=False, total_lines=None)
        line_capped = limit is not None and len(lines) > limit
        if line_capped:
            lines = lines[:limit]
        if byte_capped or line_capped:
            return FileWindow(
                lines=tuple(lines),
                start_line=offset,
                has_more=True,
                total_lines=None,
                truncated_by='bytes' if byte_capped and not line_capped else 'lines',
            )
        return FileWindow(
            lines=tuple(lines),
            start_line=offset,
            has_more=False,
            total_lines=offset - 1 + len(lines),
        )

    async def _sniff_is_binary(self, resolved_path: str) -> bool | None:
        """Classify a file as binary from a bounded prefix, without transferring it.

        Returns `True` for binary, `False` for text, or `None` when the shell utilities are
        unavailable, so the caller can fall back to the authoritative filesystem read (which
        classifies from the bytes it already holds).
        """
        try:
            result = await self.run(
                ['head', '-c', str(_BINARY_SNIFF_BYTES), resolved_path], timeout=_SHELL_SLICE_TIMEOUT
            )
        except (NotImplementedError, OSError, WorkspaceTimeoutError, UserError):
            return None
        if result.exit_code != 0 or result.stderr:
            return None
        # `run` returns text decoded with U+FFFD replacement; a NUL code point marks binary
        # content, the same heuristic Git uses to tell text from binary.
        return '\x00' in result.stdout

    async def _safe_size(self, resolved_path: str) -> int | None:
        """The file's byte size when the backend can stat it, else `None`."""
        try:
            return (await self.stat(resolved_path)).size
        except NotImplementedError:
            return None

    async def _validate_bounded_read_path(self, path: str) -> None:
        """Surface filesystem policy, missing-path, and directory errors without reading content."""
        try:
            entry = await self.stat(path)
        except NotImplementedError:
            return
        if entry.is_dir:
            raise IsADirectoryError(path)


class WrapperWorkspace(Workspace):
    """A workspace facade that composes another workspace."""

    _backend: Workspace

    def __init__(self, wrapped: Workspace):
        super().__init__(wrapped)

    @property
    def wrapped(self) -> Workspace:
        return self._backend

    async def _read_file_via_shell(
        self, path: str, offset: int, limit: int | None, max_bytes: int | None
    ) -> FileWindow | None:
        return None


def _window_from_data(
    data: bytes, offset: int, limit: int | None, max_bytes: int | None
) -> FileWindow:
    text = data.decode('utf-8', errors='replace')
    lines = _split_lines(text)
    if lines[-1] == '':
        lines = lines[:-1]

    start = offset - 1
    remaining = lines[start:]
    line_capped = limit is not None and len(remaining) > limit
    candidates = remaining if limit is None else remaining[:limit]

    selected: list[str] = []
    size = 0
    byte_capped = False
    first_line_exceeds = False
    for line in candidates:
        encoded_size = len(line.encode('utf-8'))
        extra = encoded_size if not selected else encoded_size + 1
        if max_bytes is not None and size + extra > max_bytes:
            byte_capped = True
            if not selected:
                first_line_exceeds = True
            break
        selected.append(line)
        size += extra

    has_more = line_capped or byte_capped or first_line_exceeds
    truncated_by: Literal['lines', 'bytes'] | None = None
    if first_line_exceeds or (byte_capped and not line_capped):
        truncated_by = 'bytes'
    elif line_capped:
        truncated_by = 'lines'
    return FileWindow(
        lines=tuple(selected),
        start_line=offset,
        has_more=has_more,
        total_lines=len(lines),
        truncated_by=truncated_by,
        first_line_exceeds_limit=first_line_exceeds,
    )


def _split_lines(text: str) -> tuple[str, ...]:
    return tuple(line.removesuffix('\r') for line in text.split('\n'))


def _is_binary(data: bytes) -> bool:
    """Classify bytes as binary by a NUL byte in the sampled head.

    Text files do not contain NUL, and common binary formats (images, executables, compiled
    objects) carry one within their first bytes. This is the heuristic Git uses.
    """
    return b'\x00' in data[:_BINARY_SNIFF_BYTES]


def _truncation_notice(window: FileWindow) -> str:
    """A continuation notice that makes a capped window obvious in `FileWindow.text`."""
    if window.first_line_exceeds_limit:
        return (
            f'[truncated: line {window.start_line} exceeds the byte limit; no content returned. '
            f'Pass a higher max_bytes, max_bytes=None, or read a byte slice via the shell.]'
        )
    end = window.end_line
    shown = f'lines {window.start_line}-{end}' if end is not None else f'line {window.start_line}'
    if window.total_lines is not None:
        shown = f'{shown} of {window.total_lines}'
        remaining = window.remaining_lines
        remaining_note = f'; {remaining} line{"" if remaining == 1 else "s"} remaining' if remaining else ''
    else:
        remaining_note = ''
    if window.truncated_by == 'bytes':
        cap = 'byte limit'
        next_step = 'Use offset={} to continue, or pass max_bytes=None to raise the byte cap.'
    else:
        cap = 'line limit'
        next_step = 'Use offset={} to continue, or pass limit=None to raise the line cap.'
    next_offset = (end + 1) if end is not None else window.start_line
    return f'[truncated: showing {shown}{remaining_note} ({cap}). {next_step.format(next_offset)}]'
