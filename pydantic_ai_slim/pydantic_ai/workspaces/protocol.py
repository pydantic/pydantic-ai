"""Backend protocols for the environments an agent run works in.

A backend implements [`WorkspaceBackend`][pydantic_ai.workspaces.WorkspaceBackend] plus
[`SupportsCommands`][pydantic_ai.workspaces.SupportsCommands],
[`SupportsFilesystem`][pydantic_ai.workspaces.SupportsFilesystem], or both, and optionally
[`SupportsRealpath`][pydantic_ai.workspaces.SupportsRealpath]. Every backend must:

- run commands and file operations against one filesystem, when it supports both;
- report the real exit code: a non-zero exit is a result, never an exception;
- do no I/O when built; create an environment on its first operation when built without a ref, or
  attach to the one the ref names, raising
  [`WorkspaceUnavailableError`][pydantic_ai.workspaces.WorkspaceUnavailableError] if it is gone
  rather than creating a replacement;
- record the ref as soon as a create returns, before any setup that can fail, and never change it;
- create at most one environment when first operations run concurrently: the backend owns this
  lock, since `Workspace` holds none;
- keep the ref of an environment it created even when the caller is cancelled mid-create;
- never destroy the environment when a run ends: whoever holds the ref decides when to delete it;
- raise `WorkspaceUnavailableError` for a dead environment, including when it is destroyed
  during a command; a command killed by a signal in a live environment instead returns its exit code,
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
# optional `Supports*` protocols. New fields on `CommandResult` and `FileEntry` must carry
# defaults so backends keep constructing them.
__all__ = (
    'CommandResult',
    'FileEntry',
    'WorkspaceBackend',
    'WorkspaceCommand',
    'WorkspaceError',
    'WorkspaceOutputLimitError',
    'WorkspaceRef',
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


@dataclass(frozen=True, kw_only=True)
class CommandResult:
    """The result of a completed command."""

    exit_code: int
    """The real exit code of the process. Non-zero is a normal result, not an error."""
    stdout: str
    """Captured standard output."""
    stderr: str
    """Captured standard error."""


@dataclass(frozen=True, kw_only=True)
class FileEntry:
    """Metadata about a file or directory."""

    name: str
    """Base name of the entry."""
    path: str
    """Absolute POSIX path of the entry inside the workspace."""
    is_dir: bool
    """Whether the entry is a directory, following a symlink to its target."""
    size: int | None
    """Size in bytes for a regular file when known; `None` is allowed (the shell fallback can measure it)."""


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
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        """Execute a command with stdin at EOF, returning complete output or raising an error.

        Undecodable stdout/stderr bytes are replaced with U+FFFD, never dropped.
        The command starts in `working_dir()`. A missing argv program exits 127.
        If the environment is destroyed while the command runs, raise `WorkspaceUnavailableError`;
        a command killed by a signal in a live environment returns its exit code.
        On timeout or cancellation, stop the foreground process tree on a best-effort basis;
        background jobs may continue if they detach. Return when the direct command exits,
        after at most a short output-drain grace even if a background child keeps stdout open.

        Args:
            command: An argv sequence, or a shell string with `shell=True`; a mismatch raises `TypeError`.
                [`Workspace.run`][pydantic_ai.workspaces.Workspace.run] rejects an empty argv with `ValueError`.
            shell: Whether to interpret `command` with the workspace's shell.
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

    async def stat(self, path: str) -> FileEntry:
        """Return metadata for a file or directory."""
        ...

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        """List the entries of a directory (non-recursive)."""
        ...

    async def make_dir(self, path: str) -> None:
        """Create a directory, including missing parents (`mkdir -p` semantics)."""
        ...

    async def remove(self, path: str) -> None:
        """Remove a file, or a directory and its contents.

        Refuses the working directory and its ancestors with `ValueError`; a symlink is removed itself.
        """
        ...

    async def exists(self, path: str) -> bool:
        """Whether a file or directory exists at the path."""
        ...


@runtime_checkable
class SupportsRealpath(Protocol):
    """Native symlink resolution for path boundaries such as harness FileSystem's `root_dir`.

    Without it, `Workspace.realpath` uses the backend's shell, and on a backend without commands it
    only normalizes the path as text, so path checks cannot see through symlinks. Implement it on a
    filesystem-only backend whose storage can hold symlinks.
    """

    async def realpath(self, path: str) -> str:
        """Resolve inside the environment, like `os.path.realpath(path, strict=False)`.

        Relative link targets start in the link's directory; `..` after a link climbs from its
        target. Keep missing components as written. A loop must not hang; a returned path through a loop
        must stay inside the directory holding it (or raise `OSError`). Return an absolute, normalized path.
        """
        ...


@runtime_checkable
class WorkspaceBackend(Protocol):
    """The environment an agent run works in; any object with these members conforms.

    Built without I/O from configuration and an optional [`WorkspaceRef`][pydantic_ai.workspaces.WorkspaceRef];
    the first operation creates the environment, or attaches to the one the ref names. The backend never
    tears it down; whoever holds the ref does. See the module docstring for the full contract.
    """

    @property
    def ref(self) -> WorkspaceRef | None:
        """The environment's identity: `None` until a fresh one is created, then set for good."""
        ...

    async def working_dir(self) -> str:
        """The stable working directory for this environment: absolute, symlinks resolved, no `.`/`..` segments."""
        ...
