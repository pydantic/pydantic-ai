"""Structural backend protocols for execution environments attached to an agent run.

A *workspace* is an environment — a subprocess jail, a container, a microVM, a remote worker,
or a virtual filesystem — that an agent can work in. Backends implement the small
[`WorkspaceBackend`][pydantic_ai.workspaces.WorkspaceBackend] identity and working-directory
protocol, plus [`SupportsCommands`][pydantic_ai.workspaces.SupportsCommands],
[`SupportsFilesystem`][pydantic_ai.workspaces.SupportsFilesystem], or both. This lets a backend
expose exactly what its platform supports, including files without a shell. A backend that can
resolve symlinks natively also implements the optional
[`SupportsRealpath`][pydantic_ai.workspaces.SupportsRealpath].
Tools and capabilities use the
read-only [`RunContext.workspace`][pydantic_ai.tools.RunContext.workspace] object; identity and
lifecycle are covered in the [workspace documentation](../workspace.md).

Contracts every implementation must honor (the rest are on the relevant members):

- **One environment.** When a backend supports both commands and native filesystem methods, they
  operate on the same filesystem: a file written through either is visible to the other. Consumers
  (including [`Workspace`][pydantic_ai.workspaces.Workspace]) rely on this to serve file operations
  through whichever of the two paths is cheaper.
- **Results are honest.** `exit_code` is the real process exit code; a non-zero exit is a
  normal result, not an exception. Infrastructure failures raise; they are never disguised as
  fake exit codes or empty output.
- **A ref names an environment that exists.** A backend built without a
  [`WorkspaceRef`][pydantic_ai.workspaces.WorkspaceRef] reports `ref` as `None`, creates a fresh
  environment on its first operation, and sets `ref` to that environment's identity as soon as the
  creation call returns. A backend built with a ref is bound to that environment: its first
  operation attaches, and if the environment is gone the operation raises
  [`WorkspaceUnavailableError`][pydantic_ai.workspaces.WorkspaceUnavailableError] instead of
  creating a replacement. A ref is never a label assigned ahead of the environment it names.
- **Failures are typed.** The exception a backend raises tells consumers whether the environment
  is still usable:
    - [`WorkspaceUnavailableError`][pydantic_ai.workspaces.WorkspaceUnavailableError] when the
      environment is gone or cannot be reached from this process, including a ref that cannot be
      attached. Retrying cannot succeed, and the run cannot continue in this environment.
    - [`WorkspaceTimeoutError`][pydantic_ai.workspaces.WorkspaceTimeoutError] when a command
      exceeds its `timeout=`.
    - The builtin file errors (`FileNotFoundError`, `IsADirectoryError`, `NotADirectoryError`,
      `PermissionError`, `FileExistsError`) for a path-level failure, translated from the
      platform's own exceptions; the environment itself is fine.
    - [`WorkspaceReadOnlyError`][pydantic_ai.workspaces.WorkspaceReadOnlyError] when a mutation is
      refused because the workspace is read-only; the environment itself is fine.
    - [`WorkspaceError`][pydantic_ai.workspaces.WorkspaceError] for any other failure the
      workspace layer refuses deliberately, such as output exceeding a limit.
    - `TypeError` and `ValueError` for invalid arguments, such as a relative `cwd`.

  Anything else, such as a provider SDK's own connection or rate-limit error, propagates as is
  and is treated as a transient infrastructure failure that durable engines retry. A backend whose
  platform reports a dead environment and a failed operation with the same exception should
  probe, for example with `working_dir()`, and raise `WorkspaceUnavailableError` when the
  environment is gone. [`UserError`][pydantic_ai.exceptions.UserError] is reserved for the facade
  and policy wrappers, such as an unattached workspace.
"""

from __future__ import annotations as _annotations

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

WorkspaceCommand: TypeAlias = str | Sequence[str]
"""A command to execute in a workspace.

Either an argv sequence (`['python', '-c', 'print(1)']`), or — with `shell=True` — a shell
string (`'echo $HOME | wc -c'`). Passing a `str` without `shell=True` is invalid, and so is
an argv sequence with `shell=True`: implementations must reject either mismatch with a
`TypeError`, forcing callers to be explicit about shell interpretation.
"""


class WorkspaceError(RuntimeError):
    """The workspace layer deliberately failed an operation.

    Callers should catch specific subclasses before this base class.
    """


class WorkspaceUnavailableError(WorkspaceError):
    """The workspace environment is gone or permanently unusable from this process.

    Backends raise this (or a subclass) when the environment was terminated, expired at its
    platform-side lifetime, cannot be found, or rejected the process's credentials — any
    failure where retrying the same operation cannot succeed. In particular, a backend built with
    a [`WorkspaceRef`][pydantic_ai.workspaces.WorkspaceRef] whose environment no longer exists
    raises it from the first operation rather than creating a replacement. Consumers use it to
    stop using the workspace instead of retrying: a tool cannot recover from it, so it ends the
    agent run. Other exceptions from a backend may be transient.
    """


class WorkspaceTimeoutError(WorkspaceError, TimeoutError):
    """A command exceeded the `timeout=` it was started with.

    `stdout` and `stderr` carry any captured output available when the error is raised (empty when
    the backend cannot recover it). Whether the command and its descendants are terminated is backend-specific.
    `timeout` is the deadline that was enforced, which may be coarser than requested (e.g. platforms
    that take whole seconds).
    """

    def __init__(self, message: str, *, stdout: str = '', stderr: str = '', timeout: float | None = None) -> None:
        super().__init__(message)
        self.stdout = stdout
        """Captured standard output, when available."""
        self.stderr = stderr
        """Captured standard error, when available."""
        self.timeout = timeout
        """The deadline that was enforced, in seconds."""


class WorkspaceReadOnlyError(WorkspaceError, PermissionError):
    """A mutation was refused because the workspace is read-only.

    Raised by [`ReadOnlyWorkspace`][pydantic_ai.workspaces.ReadOnlyWorkspace] and by any wrapper
    that enforces the same policy. Tool providers catch it with the other `WorkspaceError`s and
    report the refusal to the model rather than ending the run.
    """


class WorkspaceResult(Protocol):
    """The result of a completed command execution.

    Backends return richer native result objects with these fields. Requiring `CommandResult`
    would make them import Pydantic AI or wrap every result; the protocol keeps those objects
    unwrapped.
    """

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
    """Concrete [`WorkspaceResult`][pydantic_ai.workspaces.WorkspaceResult] carrier used by the built-in backends.

    Third-party backends may reuse it instead of declaring their own carrier.
    """

    exit_code: int
    stdout: str
    stderr: str


class WorkspaceFileEntry(Protocol):
    """Metadata about a file or directory inside the workspace.

    Structural, like [`WorkspaceResult`][pydantic_ai.workspaces.WorkspaceResult]: implementations
    return their native entry types.
    """

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
    """Concrete `WorkspaceFileEntry` carrier used by the built-in filesystems.

    Third-party backends may reuse it instead of declaring their own carrier.
    """

    name: str
    path: str
    is_dir: bool
    size: int | None


@runtime_checkable
class SupportsCommands(Protocol):
    """Optional command execution implemented by a workspace backend.

    Filesystem-only backends do not need to provide this protocol. The
    [`Workspace`][pydantic_ai.workspaces.Workspace] facade raises `UserError` when `run` is called
    without it and only derives filesystem operations through a shell when it is available.
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
        """Execute a command and wait for it to complete.

        When the awaiting task is cancelled, implementations must not knowingly leave the command
        running in the workspace; a backend whose platform offers no way to stop a running command
        must document that limitation.

        Args:
            command: An argv sequence, or a shell string with `shell=True`.
            shell: Whether to interpret `command` with the workspace's shell.
            cwd: Absolute working directory for the command; defaults to the workspace's
                [`working_dir`][pydantic_ai.workspaces.WorkspaceBackend.working_dir].
                Implementations must reject a relative path with `ValueError`: resolving it
                against ambient state (such as a local backend's host process working
                directory) would silently escape the workspace root.
            env: Extra environment variables for the command, layered over the backend's own.
            timeout: Deadline in seconds, measured from this call. On expiry a
                [`WorkspaceTimeoutError`][pydantic_ai.workspaces.WorkspaceTimeoutError] is raised;
                whether the command is terminated is backend-specific.
        """
        ...


@runtime_checkable
class SupportsFilesystem(Protocol):
    """Optional native file access implemented directly by a workspace backend.

    The methods are flat on the backend rather than hidden behind a separate `.fs` object.
    [`Workspace`][pydantic_ai.workspaces.Workspace] prefers these native methods and derives the same
    operations from [`SupportsCommands.run`][pydantic_ai.workspaces.SupportsCommands.run] when they
    are absent.

    All paths are absolute POSIX paths; use
    [`Workspace.resolve`][pydantic_ai.workspaces.Workspace.resolve] to turn model-supplied relative
    paths into absolute ones first. The filesystem API is bytes-only: decoding policy lives in
    the [`Workspace`][pydantic_ai.workspaces.Workspace] text helpers.

    Operations that require an existing path raise the builtin `FileNotFoundError` when it is
    missing; `exists` returns `False`. Backends translate their SDK's own missing-file exception.
    Reading a directory with `read_bytes` raises the builtin `IsADirectoryError`.
    """

    async def read_bytes(self, path: str) -> bytes:
        """Read a file's contents as bytes."""
        ...

    async def write_bytes(self, path: str, data: bytes) -> None:
        """Write bytes to a file, creating missing parent directories and replacing existing contents."""
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
    """Optional native symlink resolution implemented by a workspace backend.

    [`Workspace.realpath`][pydantic_ai.workspaces.Workspace.realpath] prefers this method and
    derives the same answer through [`SupportsCommands.run`][pydantic_ai.workspaces.SupportsCommands.run]
    when it is absent.
    """

    async def realpath(self, path: str) -> str:
        """Resolve every symlink in an absolute POSIX path.

        Returns the path with every symlink in its existing components resolved and `.`/`..`
        segments normalized. Components that don't exist are kept as written, like
        `os.path.realpath(path, strict=False)`.
        """
        ...


@runtime_checkable
class WorkspaceBackend(Protocol):
    """Backend for an execution environment attached to an agent run.

    Structural protocol: any object with these members conforms — no registration or base
    class required. See the [module doc string][pydantic_ai.workspaces] for the contracts
    implementations must honor, and the [workspace documentation](../workspace.md) for lifecycle
    rules: this protocol has no create, connect or destroy member. A backend is built from
    configuration plus an optional [`WorkspaceRef`][pydantic_ai.workspaces.WorkspaceRef] and does no
    I/O until its first operation: without a ref that operation creates a fresh environment, whose
    identity the backend reports from then on; with a ref it attaches to the named environment,
    raising [`WorkspaceUnavailableError`][pydantic_ai.workspaces.WorkspaceUnavailableError] rather
    than creating another when it is gone. Pydantic AI does not automatically provision or tear
    down an environment at run boundaries.
    """

    @property
    def ref(self) -> WorkspaceRef | None:
        """Identity of the environment this backend is bound to, or `None` while there is none.

        A ref exists only once an environment does. A backend that creates a fresh environment
        reports `None` until the creation call returns, and must set its ref then, so a consumer
        that reads a ref after an operation can hand it to another process to attach with. A backend
        built to attach to an existing environment reports the ref it was given straight away, and a
        backend whose environment is its configuration, such as
        [`LocalWorkspaceBackend`][pydantic_ai.workspaces.LocalWorkspaceBackend] and its directory,
        reports it from construction; in both cases the first operation verifies the environment
        and raises [`WorkspaceUnavailableError`][pydantic_ai.workspaces.WorkspaceUnavailableError]
        if it is gone. Every ref names its `provider`, so an identity always says where it came from.
        """
        ...

    async def working_dir(self) -> str:
        """The workspace's default working directory (absolute POSIX path).

        The path must be filesystem-canonical: symlinks resolved and no `.`/`..` segments.
        Only the backend can resolve paths inside its own environment, and consumers join
        model-supplied relative paths onto this value textually — a non-canonical spelling
        (e.g. one containing `symlink/..`) can make command and filesystem operations disagree
        about the same relative path.
        """
        ...
