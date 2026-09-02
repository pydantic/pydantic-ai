"""Structural backend protocols for execution environments attached to an agent run.

A *sandbox* is an environment — a subprocess jail, a container, a microVM, a remote worker —
that an agent run can execute commands in and read/write files of. Backends implement the
small [`SandboxBackend`][pydantic_ai.sandboxes.SandboxBackend] protocol (command execution and
working-directory reporting); each additional capability (filesystem, background processes,
streaming) is a separate optional `Supports*` protocol, so a backend implements exactly the
parts its platform supports. Tools and capabilities use the read-only
[`RunContext.sandbox`][pydantic_ai.tools.RunContext.sandbox] object; identity and lifecycle
are covered in the [sandbox documentation](../sandbox.md).

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

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, runtime_checkable

# These protocols are frozen once released: conformance is structural, so adding a member
# would silently break every existing backend. New operations go on concrete types or on new
# optional `Supports*` protocols. Data carriers declare read-only properties so that plain
# attributes, frozen dataclass fields, and properties all conform.
__all__ = (
    'CommandResult',
    'FileEntry',
    'SandboxBackend',
    'SandboxCommand',
    'SandboxFileEntry',
    'SandboxFilesystem',
    'SandboxOutputChunk',
    'SandboxProcess',
    'SandboxResult',
    'SandboxTimeoutError',
    'SandboxUnavailableError',
    'SupportsFilesystem',
    'SupportsStart',
    'SupportsStream',
)

SandboxCommand: TypeAlias = str | Sequence[str]
"""A command to execute in a sandbox.

Either an argv sequence (`['python', '-c', 'print(1)']`), or — with `shell=True` — a shell
string (`'echo $HOME | wc -c'`). Passing a `str` without `shell=True` is invalid, and so is
an argv sequence with `shell=True`: implementations must reject either mismatch with a
`TypeError`, forcing callers to be explicit about shell interpretation.
"""


class SandboxUnavailableError(RuntimeError):
    """The sandbox environment is gone or permanently unusable from this process.

    Backends raise this (or a subclass) when the environment was terminated, expired at its
    platform-side lifetime, cannot be found, or rejected the process's credentials — any
    failure where retrying the same operation cannot succeed. Consumers use it to stop using
    the sandbox instead of retrying; other exceptions from a backend may be transient.
    """


class SandboxTimeoutError(TimeoutError):
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

    A protocol rather than a concrete class: implementations return their native result
    objects unwrapped, and richer provider fields survive for callers that know the concrete
    type.
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


class SandboxOutputChunk(Protocol):
    """A chunk of live output from a started process.

    Structural, like [`SandboxResult`][pydantic_ai.sandboxes.SandboxResult]: implementations
    yield their native chunk types.
    """

    @property
    def stream(self) -> Literal['stdout', 'stderr']:
        """Which stream the chunk belongs to."""
        ...

    @property
    def data(self) -> str:
        """The chunk's text."""
        ...


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


class SandboxProcess(Protocol):
    """A started command inside a sandbox.

    Returned by [`Sandbox.start`][pydantic_ai.sandboxes.Sandbox.start]. `wait()` must be safe to
    call more than once (and concurrently), returning the same result each time.
    """

    @property
    def pid(self) -> int | None:
        """Process ID inside the sandbox, if the backend reports one."""
        ...

    async def wait(self) -> SandboxResult:
        """Wait for the process to complete and return its result.

        If the process was started with `timeout=` and the deadline passes, the command is killed
        and a [`SandboxTimeoutError`][pydantic_ai.sandboxes.SandboxTimeoutError] is raised.
        """
        ...

    async def kill(self) -> None:
        """Terminate the process.

        Implementations that cannot kill must raise `NotImplementedError` naming the
        alternative (typically: start the command with `timeout=`).
        """
        ...


@runtime_checkable
class SupportsStream(Protocol):
    """Optional live-output support for a sandbox process.

    Checked via `isinstance` against the process returned by
    [`Sandbox.start`][pydantic_ai.sandboxes.Sandbox.start].
    """

    def stream(self) -> AsyncIterator[SandboxOutputChunk]:
        """Iterate over the process's output as it is produced.

        Consuming or skipping the stream never changes
        [`wait()`][pydantic_ai.sandboxes.SandboxProcess.wait]: it returns the complete result
        either way. The stream is single-consumer: callers must not assume that a second or
        concurrent `stream()` call is supported.
        """
        ...


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

    Checked via `isinstance` by [`Sandbox`][pydantic_ai.sandboxes.Sandbox];
    [`Sandbox.fs`][pydantic_ai.sandboxes.Sandbox.fs] raises `NotImplementedError` when the
    backend does not implement this.
    """

    @property
    def fs(self) -> SandboxFilesystem:
        """Native file access inside the sandbox."""
        ...


@runtime_checkable
class SupportsStart(Protocol):
    """Optional native background-process support for a sandbox backend.

    Checked via `isinstance` by [`Sandbox`][pydantic_ai.sandboxes.Sandbox].
    """

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

        When the returned process implements
        [`SupportsStream`][pydantic_ai.sandboxes.SupportsStream], prefer `start()` + `stream()` +
        [`wait()`][pydantic_ai.sandboxes.SandboxProcess.wait] over
        [`run()`][pydantic_ai.sandboxes.SandboxBackend.run] when output produced before a timeout or
        kill matters. Arguments as for `run()`.

        The `timeout=` deadline runs from `start()`, not from the first `wait()`.
        """
        ...


@runtime_checkable
class SandboxBackend(Protocol):
    """Backend for an isolated execution environment attached to an agent run.

    Structural protocol: any object with these members conforms — no registration or base
    class required. See the [module doc string][pydantic_ai.sandboxes] for the contracts
    implementations must honor, and the [sandbox documentation](../sandbox.md) for lifecycle
    rules: this protocol deliberately contains no create/destroy/connect surface, because the
    supplier of a sandbox always owns its lifecycle.

    `isinstance` checks are shallow (member presence, not signatures), so full conformance is
    the type checker's job: verify it statically, e.g.
    `sandbox: SandboxBackend = MySandbox(...)`.
    """

    @property
    def provider(self) -> str:
        """Short identifier of the backing implementation (e.g. `'docker'`, `'local'`)."""
        ...

    @property
    def sandbox_id(self) -> str:
        """The implementation's stable identifier for this sandbox, carried by [`SandboxRef`][pydantic_ai.sandboxes.SandboxRef]."""
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
