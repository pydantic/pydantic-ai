"""Backend protocols for the environments an agent run works in.

A backend implements [`WorkspaceBackend`][pydantic_ai.workspaces.WorkspaceBackend] plus
[`SupportsCommands`][pydantic_ai.workspaces.SupportsCommands],
[`SupportsFilesystem`][pydantic_ai.workspaces.SupportsFilesystem], or both, and optionally
[`SupportsRealpath`][pydantic_ai.workspaces.SupportsRealpath]. Every backend must:

- run commands and file operations against one filesystem, when it supports both;
- report the real exit code: a non-zero exit is a result, never an exception;
- create an environment on its first operation when built without a ref, and report its ref from
  then on; when built with a ref, attach to it, raising
  [`WorkspaceUnavailableError`][pydantic_ai.workspaces.WorkspaceUnavailableError] if it is gone;
- raise `WorkspaceUnavailableError` for a dead environment,
  [`WorkspaceTimeoutError`][pydantic_ai.workspaces.WorkspaceTimeoutError] for a timeout, the builtin
  file errors for path-level failures, and `TypeError`/`ValueError` for invalid arguments, and let
  anything else (a provider SDK's transient errors) propagate so durable engines retry it.
"""

from __future__ import annotations as _annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

from pydantic_ai.messages import WorkspaceRef

# These protocols are frozen once released: conformance is structural, so adding a member
# would silently break every existing backend. New operations go on concrete types or on new
# optional `Supports*` protocols. Data carriers declare read-only properties so that plain
# attributes, frozen dataclass fields, and properties all conform.
__all__ = (
    'CommandResult',
    'FileEntry',
    'WorkspaceBackend',
    'WorkspaceCommand',
    'WorkspaceError',
    'WorkspaceOutputLimitError',
    'WorkspaceFileEntry',
    'WorkspaceRef',
    'WorkspaceResult',
    'WorkspaceReadOnlyError',
    'WorkspaceTimeoutError',
    'WorkspaceUnavailableError',
    'SupportsCommands',
    'SupportsFilesystem',
    'SupportsRealpath',
)


def validate_timeout(timeout: float | None) -> None:
    if timeout is not None and (not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError('timeout must be a positive finite number or None')


WorkspaceCommand: TypeAlias = str | Sequence[str]
"""An argv sequence (`['python', '-c', 'print(1)']`), or a shell string with `shell=True`."""


class WorkspaceError(RuntimeError):
    """The workspace layer deliberately failed an operation."""


class WorkspaceOutputLimitError(WorkspaceError):
    """A command exceeded its output cap; `stdout` and `stderr` hold their captured beginnings."""

    def __init__(self, message: str, *, limit: int, stdout: str = '', stderr: str = '') -> None:
        super().__init__(message)
        self.limit = limit
        self.stdout = stdout
        self.stderr = stderr


class WorkspaceUnavailableError(WorkspaceError):
    """The environment is gone or unusable (terminated, expired, not found), so retrying can't succeed."""


class WorkspaceTimeoutError(WorkspaceError, TimeoutError):
    """A command exceeded its `timeout=`; `stdout`/`stderr` hold partial output, like `subprocess.TimeoutExpired`."""

    def __init__(self, message: str, *, stdout: str = '', stderr: str = '') -> None:
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr


class WorkspaceReadOnlyError(WorkspaceError, PermissionError):
    """A mutation was refused because the workspace is read-only.

    Raised by [`ReadOnlyWorkspace`][pydantic_ai.workspaces.ReadOnlyWorkspace] and wrappers like it.
    """


class WorkspaceResult(Protocol):
    """The result of a completed command; backends may return their own objects with these fields."""

    @property
    def exit_code(self) -> int:
        """The real exit code of the process. Non-zero is a normal result, not an error."""
        ...

    @property
    def stdout(self) -> str:
        """Captured standard output."""
        ...

    @property
    def stderr(self) -> str:
        """Captured standard error."""
        ...


@dataclass(frozen=True, kw_only=True)
class CommandResult:
    """A [`WorkspaceResult`][pydantic_ai.workspaces.WorkspaceResult] any backend can return."""

    exit_code: int
    stdout: str
    stderr: str


class WorkspaceFileEntry(Protocol):
    """Metadata about a file or directory; backends may return their own objects with these fields."""

    @property
    def name(self) -> str:
        """Base name of the entry."""
        ...

    @property
    def path(self) -> str:
        """Absolute POSIX path of the entry inside the workspace."""
        ...

    @property
    def is_dir(self) -> bool:
        """Whether the entry is a directory, following a symlink to its target."""
        ...

    @property
    def size(self) -> int | None:
        """Size in bytes, or `None` when the backend doesn't report one (e.g. for directories)."""
        ...


@dataclass(frozen=True, kw_only=True)
class FileEntry:
    """A [`WorkspaceFileEntry`][pydantic_ai.workspaces.WorkspaceFileEntry] any backend can return."""

    name: str
    path: str
    is_dir: bool
    size: int | None


@runtime_checkable
class SupportsCommands(Protocol):
    """Optional command execution.

    Without it, [`Workspace.run`][pydantic_ai.workspaces.Workspace.run] raises `UserError`.
    """

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> WorkspaceResult:
        """Execute a command with stdin at EOF, returning complete output or raising an error.

        Undecodable stdout/stderr bytes are replaced with U+FFFD, never dropped.
        A missing argv program exits 127. A missing `cwd` raises `FileNotFoundError`.
        On timeout or cancellation, stop the foreground process tree on a best-effort basis;
        background jobs may continue if they detach.

        Args:
            command: An argv sequence, or a shell string with `shell=True`; a mismatch raises `TypeError`.
            shell: Whether to interpret `command` with the workspace's shell.
            cwd: Absolute working directory, defaulting to `working_dir()`; a relative one raises `ValueError`.
            env: Extra environment variables, layered over the backend's own.
            timeout: A positive finite number of seconds before
                [`WorkspaceTimeoutError`][pydantic_ai.workspaces.WorkspaceTimeoutError]; no timeout by default.
                Invalid values raise `ValueError`.
        """
        ...


@runtime_checkable
class SupportsFilesystem(Protocol):
    """Optional native file access; without it, [`Workspace`][pydantic_ai.workspaces.Workspace] uses the shell.

    Paths are absolute POSIX paths. A missing path raises `FileNotFoundError` (except in `exists`),
    and reading a directory raises `IsADirectoryError`.
    """

    async def read_bytes(self, path: str) -> bytes:
        """Read a file's contents as bytes."""
        ...

    async def write_bytes(self, path: str, data: bytes) -> None:
        """Write bytes to a file, creating missing parents and writing through an existing symlink."""
        ...

    async def stat(self, path: str) -> WorkspaceFileEntry:
        """Return metadata for a file or directory."""
        ...

    async def list_dir(self, path: str) -> Sequence[WorkspaceFileEntry]:
        """List the entries of a directory (non-recursive)."""
        ...

    async def make_dir(self, path: str) -> None:
        """Create a directory, including missing parents (`mkdir -p` semantics)."""
        ...

    async def remove(self, path: str) -> None:
        """Remove a file, or a directory and its contents."""
        ...

    async def exists(self, path: str) -> bool:
        """Whether a file or directory exists at the path."""
        ...


@runtime_checkable
class SupportsRealpath(Protocol):
    """Optional native symlink resolution.

    Without it, [`Workspace.realpath`][pydantic_ai.workspaces.Workspace.realpath] uses the backend's shell.
    """

    async def realpath(self, path: str) -> str:
        """Resolve every symlink in an absolute POSIX path, like `os.path.realpath(path, strict=False)`."""
        ...


@runtime_checkable
class WorkspaceBackend(Protocol):
    """The environment an agent run works in; any object with these members conforms.

    Built without I/O from configuration and an optional [`WorkspaceRef`][pydantic_ai.workspaces.WorkspaceRef];
    the first operation creates the environment, or attaches to the one the ref names. Nothing tears it down.
    """

    @property
    def ref(self) -> WorkspaceRef | None:
        """The environment's identity: `None` until a fresh one is created, then set for good."""
        ...

    async def working_dir(self) -> str:
        """The default working directory: absolute, symlinks resolved, no `.`/`..` segments."""
        ...
