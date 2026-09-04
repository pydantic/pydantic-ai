"""Structural backend protocols for execution environments attached to an agent run.

A *sandbox* is an environment — a subprocess jail, a container, a microVM, a remote worker —
that an agent run can execute commands in and read/write files of. Backends implement the
small [`SandboxBackend`][pydantic_ai.sandboxes.SandboxBackend] protocol (command execution and
working-directory reporting); filesystem access is the optional `SupportsFilesystem` protocol,
so a backend implements exactly the parts its platform supports. Tools and capabilities use the
read-only [`RunContext.sandbox`][pydantic_ai.tools.RunContext.sandbox] object; identity and
lifecycle are covered in the [sandbox documentation](../sandbox.md).

Contracts every implementation must honor (the rest are on the relevant members):

- **One environment.** `run` executes against the same filesystem that `fs` exposes: a file
  written through either is visible to the other. Consumers (including
  [`Sandbox`][pydantic_ai.sandboxes.Sandbox]) rely on this to serve file operations
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
    'SandboxBackend',
    'SandboxCommand',
    'SandboxError',
    'SandboxFileEntry',
    'SandboxFilesystem',
    'SandboxRef',
    'SandboxResult',
    'SandboxTimeoutError',
    'SandboxUnavailableError',
    'SupportsFilesystem',
)

SandboxCommand: TypeAlias = str | Sequence[str]
"""A command to execute in a sandbox.

Either an argv sequence (`['python', '-c', 'print(1)']`), or — with `shell=True` — a shell
string (`'echo $HOME | wc -c'`). Passing a `str` without `shell=True` is invalid, and so is
an argv sequence with `shell=True`: implementations must reject either mismatch with a
`TypeError`, forcing callers to be explicit about shell interpretation.
"""


@dataclass(frozen=True, kw_only=True)
class SandboxRef:
    """Serializable identity of a sandbox environment, as the backend spells it.

    The string is whatever that backend needs to find its environment again: a provider-issued
    id for Modal, Daytona or E2B, a caller-chosen name for platforms that cannot reattach by id.
    Pydantic AI never interprets it, and it must never carry credentials.
    """

    sandbox_id: str
    """The backend's own identifier for the environment."""


class SandboxError(RuntimeError):
    """The sandbox layer deliberately failed an operation.

    Callers should catch specific subclasses before this base class.
    """


class SandboxUnavailableError(SandboxError):
    """The sandbox environment is gone or permanently unusable from this process.

    Backends raise this (or a subclass) when the environment was terminated, expired at its
    platform-side lifetime, cannot be found, or rejected the process's credentials — any
    failure where retrying the same operation cannot succeed. Consumers use it to stop using
    the sandbox instead of retrying; other exceptions from a backend may be transient.
    """


class SandboxTimeoutError(SandboxError, TimeoutError):
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


class SandboxResult(Protocol):
    """The result of a completed command execution.

    Backends return richer native result objects with these fields. Requiring `CommandResult`
    would make them import Pydantic AI or wrap every result; the protocol keeps those objects
    unwrapped and exposes the minimum read by `Sandbox._read_file_via_shell`.
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
    """Concrete [`SandboxResult`][pydantic_ai.sandboxes.SandboxResult] carrier used by the built-in backends.

    Third-party backends may reuse it instead of declaring their own carrier.
    """

    exit_code: int
    stdout: str
    stderr: str


class SandboxFileEntry(Protocol):
    """Metadata about a file or directory inside the sandbox.

    Structural, like [`SandboxResult`][pydantic_ai.sandboxes.SandboxResult]: implementations
    return their native entry types.
    """

    @property
    def name(self) -> str:
        """Base name of the entry."""
        ...

    @property
    def path(self) -> str:
        """Absolute POSIX path of the entry inside the sandbox."""
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
    """Concrete `SandboxFileEntry` carrier used by the built-in filesystems.

    Third-party backends may reuse it instead of declaring their own carrier.
    """

    name: str
    path: str
    is_dir: bool
    size: int | None


class SandboxFilesystem(Protocol):
    """File access inside a sandbox.

    All paths are absolute POSIX paths; use
    [`Sandbox.resolve`][pydantic_ai.sandboxes.Sandbox.resolve] to turn model-supplied
    relative paths into absolute ones first. The filesystem API is bytes-only: decoding policy
    lives in the [`Sandbox`][pydantic_ai.sandboxes.Sandbox] text helpers.

    Operations on a path that does not exist raise the builtin `FileNotFoundError`: backends
    translate their SDK's own missing-file exception.
    """

    async def read_bytes(self, path: str) -> bytes:
        """Read a file's contents as bytes."""
        ...

    async def write_bytes(self, path: str, data: bytes) -> None:
        """Write bytes to a file, creating missing parent directories and replacing existing contents."""
        ...

    async def stat(self, path: str) -> SandboxFileEntry:
        """Return metadata for a file or directory."""
        ...

    async def list_dir(self, path: str) -> Sequence[SandboxFileEntry]:
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
class SupportsFilesystem(Protocol):
    """Optional native filesystem access for a sandbox backend.

    Checked via `isinstance` by [`Sandbox`][pydantic_ai.sandboxes.Sandbox]: its file methods
    (`read_bytes`, `write_bytes`, `stat`, `list_dir`, `make_dir`, `remove`, `exists`) raise
    `NotImplementedError` when the backend does not implement this.
    """

    @property
    def fs(self) -> SandboxFilesystem:
        """Native file access inside the sandbox."""
        ...


@runtime_checkable
class SandboxBackend(Protocol):
    """Backend for an isolated execution environment attached to an agent run.

    Structural protocol: any object with these members conforms — no registration or base
    class required. See the [module doc string][pydantic_ai.sandboxes] for the contracts
    implementations must honor, and the [sandbox documentation](../sandbox.md) for lifecycle
    rules: this protocol has no create, connect or destroy member. A backend is built from
    configuration plus an optional [`SandboxRef`][pydantic_ai.sandboxes.SandboxRef] and does no
    I/O until its first operation, which creates or attaches as needed. Pydantic AI never starts
    or stops an environment.
    """

    @property
    def ref(self) -> SandboxRef | None:
        """Identity of the environment this backend is bound to, or `None` before it has one.

        A backend built to attach to an existing environment reports its ref straight away. One
        built to create a fresh environment reports `None` until its first operation has run,
        because only the provider can say what the new environment is called. Once an operation
        has succeeded, this must not be `None`.
        """
        ...

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

        When the awaiting task is cancelled, implementations must not knowingly leave the command
        running in the sandbox; a backend whose platform offers no way to stop a running command
        must document that limitation.

        Args:
            command: An argv sequence, or a shell string with `shell=True`.
            shell: Whether to interpret `command` with the sandbox's shell.
            cwd: Absolute working directory for the command; defaults to the sandbox's
                [`working_dir`][pydantic_ai.sandboxes.SandboxBackend.working_dir].
                Implementations must reject a relative path with `ValueError`: resolving it
                against ambient state (such as a local backend's host process working
                directory) would silently escape the sandbox root.
            env: Extra environment variables for the command.
            timeout: Deadline in seconds, measured from this call. On expiry the command is killed
                and a [`SandboxTimeoutError`][pydantic_ai.sandboxes.SandboxTimeoutError] is raised.
        """
        ...

    async def working_dir(self) -> str:
        """The sandbox's default working directory (absolute POSIX path).

        The path must be filesystem-canonical: symlinks resolved and no `.`/`..` segments.
        Only the backend can resolve paths inside its own environment, and consumers join
        model-supplied relative paths onto this value textually — a non-canonical spelling
        (e.g. one containing `symlink/..`) makes `run` (which resolves paths like the kernel)
        and `fs` (which operates on the spelling) disagree about the same relative path.
        """
        ...
