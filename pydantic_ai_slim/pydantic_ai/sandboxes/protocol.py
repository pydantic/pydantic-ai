"""Structural backend protocols for execution environments attached to an agent run.

A *sandbox* is an environment — a subprocess jail, a container, a microVM, a remote worker —
that an agent run can execute commands in and read/write files of. Providers implement the small
[`SandboxBackend`][pydantic_ai.sandboxes.SandboxBackend] protocol defined here. Its floor is
command execution and working-directory reporting; each additional capability
(filesystem, background processes, byte-range reads) is a separate `Supports*` opt-in protocol,
letting providers ship in pieces without inheriting placeholder behavior. Tools and capabilities
receive the facade through the read-only
[`RunContext.sandbox`][pydantic_ai.tools.RunContext.sandbox] field.

Sandbox identity, reconnection, and lifecycle are documented in the
[`references.py` module][pydantic_ai.sandboxes.references] and the
[sandbox documentation](../sandbox.md).

The backend protocol is deliberately a floor, not a ceiling: implementations are expected to offer
richer surfaces (reconnection, snapshotting, streaming limits) on their concrete types, and
code written against the protocol must only rely on what is documented here. The floor is
also frozen once released: because conformance is structural, adding a member to any protocol
in this module silently breaks every existing implementation. New operations must arrive on
concrete types or as new, separate protocols, such as
[`SupportsFilesystem`][pydantic_ai.sandboxes.SupportsFilesystem],
[`SupportsStart`][pydantic_ai.sandboxes.SupportsStart], and
[`SupportsReadBytesRange`][pydantic_ai.sandboxes.SupportsReadBytesRange], never as members of the
floor.

Every public type in this module — including the plain data carriers
[`SandboxResult`][pydantic_ai.sandboxes.SandboxResult],
[`SandboxOutputChunk`][pydantic_ai.sandboxes.SandboxOutputChunk], and
[`SandboxFileEntry`][pydantic_ai.sandboxes.SandboxFileEntry] — is a protocol rather than a
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
  implementation provides; [`Sandbox.resolve`][pydantic_ai.sandboxes.Sandbox.resolve] is a textual
  convenience that does not confine paths.
"""

from __future__ import annotations as _annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, runtime_checkable

__all__ = (
    'SandboxBackend',
    'SandboxCommand',
    'SandboxFileEntry',
    'SandboxFilesystem',
    'SandboxOutputChunk',
    'SandboxProcess',
    'SandboxResult',
    'SupportsFilesystem',
    'SupportsReadBytesRange',
    'SupportsStart',
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
    fields survive for callers that know the concrete type. "Native" here means the sandbox
    library's own result type, not a raw provider SDK result: no provider bounds output
    server-side, so `output_limit=` and the `*_dropped` counts can only be implemented in the
    library layer, and raw SDK results conform only once that layer adds them.
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


@dataclass(frozen=True)
class FileEntry:
    """The framework's own concrete `SandboxFileEntry` carrier, deliberately unexported.

    Returned by the built-in filesystems (`LocalSandbox` and the `Sandbox` facade's shell
    fallback); third-party backends return their own native entry types instead.
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
    [`Sandbox.resolve`][pydantic_ai.sandboxes.Sandbox.resolve] to turn model-supplied
    relative paths into absolute ones first. The backend SPI is bytes-only: decoding policy lives
    in the [`Sandbox`][pydantic_ai.sandboxes.Sandbox] facade's text helpers.
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
class SupportsReadBytesRange(Protocol):
    """Optional fast path for reading a byte range of a file.

    Checked (via `isinstance`) against a backend's `fs` object by the
    [`Sandbox`][pydantic_ai.sandboxes.Sandbox] facade: when present, windowed reads fetch only
    the bytes they need instead of the whole file. This check is shallow: the presence of a
    method with this name activates the fast path, so implementations must match this
    signature and contract exactly. Pin conformance statically (for example,
    `_conforms: SupportsReadBytesRange = MyFs()`) as `local.py` does.
    """

    async def read_bytes_range(self, path: str, start: int, end: int) -> bytes:
        """Read bytes `[start, end)` of a file (absolute POSIX path).

        Returns fewer bytes than requested when the range extends past EOF, and `b''` when
        `start` is at or past EOF. Implementations must not raise for out-of-range reads on an
        existing file.
        """
        ...


@runtime_checkable
class SupportsFilesystem(Protocol):
    """Optional native filesystem access for a sandbox backend.

    Checked via `isinstance` by the [`Sandbox`][pydantic_ai.sandboxes.Sandbox] facade;
    [`Sandbox.fs`][pydantic_ai.sandboxes.Sandbox.fs] raises `NotImplementedError` when the
    backend does not implement this. The runtime check is shallow, so implementations must
    match this contract exactly and pin conformance statically.
    """

    @property
    def fs(self) -> SandboxFilesystem:
        """Native file access inside the sandbox."""
        ...


@runtime_checkable
class SupportsStart(Protocol):
    """Optional native background-process support for a sandbox backend.

    Checked via `isinstance` by the [`Sandbox`][pydantic_ai.sandboxes.Sandbox] facade. The runtime
    check is shallow, so implementations must match this contract exactly and pin conformance
    statically.
    """

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

        Prefer `start()` + [`stream()`][pydantic_ai.sandboxes.SandboxProcess.stream] +
        [`wait()`][pydantic_ai.sandboxes.SandboxProcess.wait] over
        [`run()`][pydantic_ai.sandboxes.SandboxBackend.run] when output produced before a timeout or
        kill matters. Arguments as for `run()`.
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
        """Short identifier of the backing implementation (e.g. `'docker'`, `'local'`).

        Together with `sandbox_id`, this is the identity consumed by a
        [`SandboxConnector`][pydantic_ai.sandboxes.SandboxConnector]. Credentials and other
        worker-side configuration stay on the connector rather than in the identity.

        The name is `provider` — not `provider_name` — by contract: conformance is
        structural, and sandbox libraries already expose `provider` on their native types, so
        the member name is shared cross-repo surface that reads as a compact identity pair
        with `sandbox_id`. Renaming it would silently unconform every existing implementation.
        """
        ...

    @property
    def sandbox_id(self) -> str:
        """The implementation's stable identifier for this sandbox, unique per provider.

        Together with `provider`, this is the durable identity carried by
        [`SandboxRef`][pydantic_ai.sandboxes.SandboxRef]. On its own it is not globally unique.
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
        output_limit: int | None = None,
    ) -> SandboxResult:
        """Execute a command and wait for it to complete.

        Args:
            command: An argv sequence, or a shell string with `shell=True`.
            shell: Whether to interpret `command` with the sandbox's shell.
            cwd: Absolute working directory for the command; defaults to the sandbox's
                [`working_dir`][pydantic_ai.sandboxes.SandboxBackend.working_dir].
            env: Extra environment variables for the command.
            timeout: Deadline in seconds. On expiry the command is killed and an exception
                deriving from `TimeoutError` is raised.
            output_limit: Maximum number of output characters to retain in total across both
                streams (oldest dropped first, reported via `stdout_dropped`/`stderr_dropped`).
                Implementations that cannot bound output raise `NotImplementedError` when this
                is set.
        """
        ...

    async def working_dir(self) -> str:
        """The sandbox's default working directory (absolute POSIX path)."""
        ...
