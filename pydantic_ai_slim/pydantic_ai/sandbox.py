"""A structural protocol for sandboxes: isolated execution environments attached to an agent run.

A *sandbox* is an environment — a subprocess jail, a container, a microVM, a remote worker —
that an agent run can execute commands in and read/write files of. Pydantic AI defines the
protocol only: any object that structurally satisfies [`Sandbox`][pydantic_ai.sandbox.Sandbox]
can be attached to a run via the `sandbox=` argument to the run methods. Tools and
capabilities then reach it through the read-only
[`RunContext.sandbox`][pydantic_ai.tools.RunContext.sandbox] field.

This is a *usage* interface: creating, destroying, sharing, and reconnecting sandboxes is the
business of whoever supplies one, never of Pydantic AI or of code that merely uses the sandbox.
See the [sandbox documentation](../sandbox.md) for the lifecycle rules and how sandboxes
interact with durable execution.

The protocol is deliberately a floor, not a ceiling: implementations are expected to offer
richer surfaces (reconnection, snapshotting, streaming limits) on their concrete types, and
code written against the protocol must only rely on what is documented here. The floor is
also frozen once released: because conformance is structural, adding a member to any protocol
in this module silently breaks every existing implementation. New operations must arrive on
concrete types or as new, separate protocols, never as members of these.

Every type in this module — including the plain data carriers
[`SandboxResult`][pydantic_ai.sandbox.SandboxResult],
[`SandboxOutputChunk`][pydantic_ai.sandbox.SandboxOutputChunk], and
[`SandboxFileEntry`][pydantic_ai.sandbox.SandboxFileEntry] — is a protocol rather than a
concrete class for the same reason: a sandbox library's existing native types conform as-is,
with no pydantic-ai dependency or adapter layer. The carriers declare their members as
*read-only properties* deliberately: a bare annotated protocol member demands a settable
attribute, which frozen dataclass fields and properties fail to satisfy — declared read-only,
plain attributes, dataclass fields (frozen included), and properties all conform.

Contracts every implementation must honor:

- **Optional operations raise `NotImplementedError`.** Not every backend can stream output,
  kill a process, or bound retained output. Implementations that can't must raise the builtin
  `NotImplementedError` (from the call itself, not lazily) naming an alternative, and callers
  must treat it as "use the documented fallback" — `wait()` instead of `stream()`, `timeout=`
  instead of `kill()`, bounding output in-command instead of `output_limit=`. Never fake success.
- **`timeout=` is a kill guarantee.** When the deadline passes, the implementation must
  terminate the command and raise an exception that derives from the builtin `TimeoutError`.
  Cancelling the awaiting task is *not* required to kill the remote command — `timeout=` and
  `kill()` are the only guaranteed-termination paths.
- **Results are honest.** `exit_code` is the real process exit code; a non-zero exit is a
  normal result, not an exception. Infrastructure failures raise; they are never disguised as
  fake exit codes or empty output.
- **Command/shell mismatches raise `TypeError`.** A `str` command without `shell=True`, or an
  argv sequence with it, must be rejected with a `TypeError` — never shell-interpreted or
  joined by guesswork. Since `str` is itself a `Sequence[str]`, the type checker cannot catch
  the mismatch; this runtime rejection is what keeps it from becoming an injection vector.
- **The protocol is not a security boundary.** Isolation comes from the environment the
  implementation provides; [`resolve`][pydantic_ai.sandbox.Sandbox.resolve] is a textual
  convenience that does not confine paths.
"""

from __future__ import annotations as _annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Literal, Protocol, TypeAlias, runtime_checkable

__all__ = (
    'Sandbox',
    'SandboxCommand',
    'SandboxFileEntry',
    'SandboxFilesystem',
    'SandboxOutputChunk',
    'SandboxProcess',
    'SandboxResult',
)

SandboxCommand: TypeAlias = str | Sequence[str]
"""A command to execute in a sandbox.

Either an argv sequence (`['python', '-c', 'print(1)']`), or — with `shell=True` — a shell
string (`'echo $HOME | wc -c'`). Passing a `str` without `shell=True` is invalid, and so is
an argv sequence with `shell=True`: implementations must reject either mismatch with a
`TypeError`, forcing callers to be explicit about shell interpretation.
"""


class SandboxResult(Protocol):
    """The result of a completed command execution.

    A protocol rather than a concrete class so implementations return their native result
    objects unwrapped: any object carrying these attributes conforms, and richer provider
    fields survive for callers that know the concrete type.
    """

    @property
    def exit_code(self) -> int:
        """The real exit code of the process. Non-zero is a normal result, not an error."""
        ...

    @property
    def stdout(self) -> str:
        """Captured standard output (possibly bounded by `output_limit=`)."""
        ...

    @property
    def stderr(self) -> str:
        """Captured standard error (possibly bounded by `output_limit=`)."""
        ...

    @property
    def stdout_dropped(self) -> int:
        """Number of stdout characters dropped due to `output_limit=`; `0` when nothing was dropped."""
        ...

    @property
    def stderr_dropped(self) -> int:
        """Number of stderr characters dropped due to `output_limit=`; `0` when nothing was dropped."""
        ...


class SandboxOutputChunk(Protocol):
    """A chunk of live output from a started process.

    Structural, like [`SandboxResult`][pydantic_ai.sandbox.SandboxResult]: implementations
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

    Structural, like [`SandboxResult`][pydantic_ai.sandbox.SandboxResult]: implementations
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


class SandboxProcess(Protocol):
    """A started command inside a sandbox.

    Returned by [`Sandbox.start`][pydantic_ai.sandbox.Sandbox.start]. `wait()` must be safe to
    call more than once (and concurrently), returning the same result each time.
    """

    @property
    def pid(self) -> int | None:
        """Process ID inside the sandbox, if the backend reports one."""
        ...

    async def wait(self) -> SandboxResult:
        """Wait for the process to complete and return its result.

        If the process was started with `timeout=` and the deadline passes, this raises an
        exception deriving from the builtin `TimeoutError` after the process has been killed.
        """
        ...

    def stream(self) -> AsyncIterator[SandboxOutputChunk]:
        """Iterate over the process's output as it is produced.

        Implementations that cannot stream must raise `NotImplementedError` from this call
        (not from the first iteration); callers fall back to `wait()`.
        """
        ...

    async def kill(self) -> None:
        """Terminate the process.

        Implementations that cannot kill must raise `NotImplementedError` naming the
        alternative (typically: start the command with `timeout=`).
        """
        ...


class SandboxFilesystem(Protocol):
    """File access inside a sandbox.

    All paths are absolute POSIX paths; use
    [`Sandbox.resolve`][pydantic_ai.sandbox.Sandbox.resolve] to turn model-supplied relative
    paths into absolute ones first.
    """

    async def read_bytes(self, path: str) -> bytes:
        """Read a file's contents as bytes."""
        ...

    async def write_bytes(self, path: str, data: bytes) -> None:
        """Write bytes to a file, creating missing parent directories and replacing existing contents."""
        ...

    async def read_text(self, path: str, encoding: str = 'utf-8') -> str:
        """Read a file's contents as text."""
        ...

    async def write_text(self, path: str, content: str, encoding: str = 'utf-8') -> None:
        """Write text to a file, creating missing parent directories and replacing existing contents."""
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
class Sandbox(Protocol):
    """An isolated execution environment attached to an agent run.

    Structural protocol: any object with these members conforms — no registration or base
    class required. See the [module docstring][pydantic_ai.sandbox] for the contracts
    implementations must honor, and the [sandbox documentation](../sandbox.md) for lifecycle
    rules: this protocol deliberately contains no create/destroy/connect surface, because the
    supplier of a sandbox always owns its lifecycle.

    This is the module's only `@runtime_checkable` protocol, because it is the boundary
    object handed to `sandbox=`. `isinstance` checks are shallow (member presence, not
    signatures), so full conformance is the type checker's job: verify it statically, e.g.
    `sandbox: Sandbox = MySandbox(...)`.
    """

    @property
    def provider(self) -> str:
        """Short identifier of the backing implementation (e.g. `'docker'`, `'local'`).

        Identity and observability only: together with `sandbox_id` it names the environment
        in logs and serialized references, but the pair is *not* a self-contained reconnection
        token — reconnection (if the implementation supports it at all) usually needs
        implementation-specific configuration held by the caller.
        """
        ...

    @property
    def sandbox_id(self) -> str:
        """The implementation's stable identifier for this sandbox, unique per provider.

        Together with `provider` this is the durable reference: it is what a durable-execution
        workflow should carry across serialization boundaries (on `deps` or `metadata`) so
        activity-side code can re-open the environment using the implementation's own
        reconnection surface. On its own it is not globally unique.
        """
        ...

    @property
    def fs(self) -> SandboxFilesystem:
        """File access inside the sandbox."""
        ...

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
        """Execute a command and wait for it to complete.

        Args:
            command: An argv sequence, or a shell string with `shell=True`.
            shell: Whether to interpret `command` with the sandbox's shell.
            cwd: Absolute working directory for the command; defaults to the sandbox's
                [`working_dir`][pydantic_ai.sandbox.Sandbox.working_dir].
            env: Extra environment variables for the command.
            timeout: Deadline in seconds. On expiry the command is killed and an exception
                deriving from `TimeoutError` is raised.
            output_limit: Maximum number of output characters to retain in total across both
                streams (oldest dropped first, reported via `stdout_dropped`/`stderr_dropped`).
                Implementations that cannot bound output raise `NotImplementedError` when this
                is set.
        """
        ...

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
        """Start a command without waiting, returning a handle to the running process.

        Prefer `start()` + [`stream()`][pydantic_ai.sandbox.SandboxProcess.stream] +
        [`wait()`][pydantic_ai.sandbox.SandboxProcess.wait] over
        [`run()`][pydantic_ai.sandbox.Sandbox.run] when output produced before a timeout or
        kill matters. Arguments as for `run()`.
        """
        ...

    async def working_dir(self) -> str:
        """The sandbox's default working directory (absolute POSIX path)."""
        ...

    async def resolve(self, path: str, *, base: str | None = None) -> str:
        """Resolve a possibly-relative path to an absolute POSIX path.

        Joins `path` onto `base` (default: [`working_dir`][pydantic_ai.sandbox.Sandbox.working_dir])
        and normalizes it textually. This is a spelling convenience for model-supplied paths,
        **not** a confinement mechanism: `..` segments can escape `base` and symlinks are not
        inspected. Isolation is the sandbox's job, not this method's.
        """
        ...
