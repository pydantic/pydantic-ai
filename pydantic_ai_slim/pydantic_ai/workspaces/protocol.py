"""Structural backend protocols for execution environments attached to an agent run.

A *workspace* is an environment — a subprocess jail, a container, a microVM, a remote worker —
that an agent run can execute commands in and read/write files of. Backends implement the
small [`WorkspaceBackend`][pydantic_ai.workspaces.WorkspaceBackend] protocol (command execution and
working-directory reporting); native filesystem access is the optional, flat
`SupportsFilesystem` protocol, so a backend implements exactly the parts its platform supports.
Tools and capabilities use the
read-only [`RunContext.workspace`][pydantic_ai.tools.RunContext.workspace] object; identity and
lifecycle are covered in the [workspace documentation](../workspace.md).

Contracts every implementation must honor (the rest are on the relevant members):

- **One environment.** `run` and native filesystem methods operate on the same filesystem: a file
  written through either is visible to the other. Consumers (including
  [`Workspace`][pydantic_ai.workspaces.Workspace]) rely on this to serve file operations
  through whichever of the two paths is cheaper.
- **Results are honest.** `exit_code` is the real process exit code; a non-zero exit is a
  normal result, not an exception. Infrastructure failures raise; they are never disguised as
  fake exit codes or empty output.
"""

from __future__ import annotations as _annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

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
    'WorkspaceTimeoutError',
    'WorkspaceUnavailableError',
    'SupportsFilesystem',
)

WorkspaceCommand: TypeAlias = str | Sequence[str]
"""A command to execute in a workspace.

Either an argv sequence (`['python', '-c', 'print(1)']`), or — with `shell=True` — a shell
string (`'echo $HOME | wc -c'`). Passing a `str` without `shell=True` is invalid, and so is
an argv sequence with `shell=True`: implementations must reject either mismatch with a
`TypeError`, forcing callers to be explicit about shell interpretation.
"""


@dataclass(frozen=True, kw_only=True)
class WorkspaceRef:
    """Serializable identity of a workspace environment, as the backend spells it.

    The string is whatever that backend needs to find its environment again: a provider-issued
    id for Modal, Daytona or E2B, a caller-chosen name for platforms that cannot reattach by id.
    Pydantic AI never interprets it, and it must never carry credentials.
    """

    workspace_id: str
    """The backend's own identifier for the environment."""


class WorkspaceError(RuntimeError):
    """The workspace layer deliberately failed an operation.

    Callers should catch specific subclasses before this base class.
    """


class WorkspaceUnavailableError(WorkspaceError):
    """The workspace environment is gone or permanently unusable from this process.

    Backends raise this (or a subclass) when the environment was terminated, expired at its
    platform-side lifetime, cannot be found, or rejected the process's credentials — any
    failure where retrying the same operation cannot succeed. Consumers use it to stop using
    the workspace instead of retrying; other exceptions from a backend may be transient.
    """


class WorkspaceTimeoutError(WorkspaceError, TimeoutError):
    """A command exceeded the `timeout=` it was started with and was killed.

    `stdout` and `stderr` carry any output the command produced before the kill (empty when
    the backend cannot recover it); `timeout` is the deadline that was enforced, which may be
    coarser than requested (e.g. platforms that take whole seconds).
    """

    def __init__(self, message: str, *, stdout: str = '', stderr: str = '', timeout: float | None = None) -> None:
        super().__init__(message)
        self.stdout = stdout
        """Standard output produced before the command was killed."""
        self.stderr = stderr
        """Standard error produced before the command was killed."""
        self.timeout = timeout
        """The deadline that was enforced, in seconds."""


class WorkspaceResult(Protocol):
    """The result of a completed command execution.

    Backends return richer native result objects with these fields. Requiring `CommandResult`
    would make them import Pydantic AI or wrap every result; the protocol keeps those objects
    unwrapped and exposes the minimum read by `Workspace._read_file_via_shell`.
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
        """Whether the entry is a directory."""
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
class SupportsFilesystem(Protocol):
    """Optional native file access implemented directly by a workspace backend.

    The methods are flat on the backend rather than hidden behind a separate `.fs` object.
    [`Workspace`][pydantic_ai.workspaces.Workspace] prefers these native methods and derives the same
    operations from [`WorkspaceBackend.run`][pydantic_ai.workspaces.WorkspaceBackend.run] when they
    are absent.

    All paths are absolute POSIX paths; use
    [`Workspace.resolve`][pydantic_ai.workspaces.Workspace.resolve] to turn model-supplied relative
    paths into absolute ones first. The filesystem API is bytes-only: decoding policy lives in
    the [`Workspace`][pydantic_ai.workspaces.Workspace] text helpers.

    Operations that require an existing path raise the builtin `FileNotFoundError` when it is
    missing; `exists` returns `False`. Backends translate their SDK's own missing-file exception.
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
class WorkspaceBackend(Protocol):
    """Backend for an isolated execution environment attached to an agent run.

    Structural protocol: any object with these members conforms — no registration or base
    class required. See the [module doc string][pydantic_ai.workspaces] for the contracts
    implementations must honor, and the [workspace documentation](../workspace.md) for lifecycle
    rules: this protocol has no create, connect or destroy member. A backend is built from
    configuration plus an optional [`WorkspaceRef`][pydantic_ai.workspaces.WorkspaceRef] and does no
    I/O until its first operation, which creates or attaches as needed. Pydantic AI never starts
    or stops an environment.
    """

    @property
    def ref(self) -> WorkspaceRef | None:
        """Identity of the environment this backend is bound to, or `None` before it has one.

        A backend built to attach to an existing environment reports its ref straight away. One
        built to create a fresh environment reports `None` until its first operation has run,
        because only the provider can say what the new environment is called. Once an operation
        has succeeded, this must not be `None`.
        """
        ...

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
            env: Extra environment variables for the command.
            timeout: Deadline in seconds, measured from this call. On expiry the command is killed
                and a [`WorkspaceTimeoutError`][pydantic_ai.workspaces.WorkspaceTimeoutError] is raised.
        """
        ...

    async def working_dir(self) -> str:
        """The workspace's default working directory (absolute POSIX path).

        The path must be filesystem-canonical: symlinks resolved and no `.`/`..` segments.
        Only the backend can resolve paths inside its own environment, and consumers join
        model-supplied relative paths onto this value textually — a non-canonical spelling
        (e.g. one containing `symlink/..`) makes `run` (which resolves paths like the kernel)
        and filesystem operations (which use the spelling) disagree about the same relative path.
        """
        ...
