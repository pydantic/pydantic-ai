"""[Modal](https://modal.com) cloud sandboxes as a [sandbox backend][pydantic_ai.sandboxes.SandboxBackend].

Needs the `modal` optional dependency group (`pip install "pydantic-ai-slim[modal]"`). Import
[`ModalSandbox`][pydantic_ai.sandboxes.modal.ModalSandbox] from this module directly:
`pydantic_ai.sandboxes` deliberately doesn't re-export it, so importing the package never
pulls in the Modal SDK.
"""

from __future__ import annotations as _annotations

import asyncio
import itertools
import math
import posixpath
import time
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import anyio
from typing_extensions import Self

from .protocol import FileEntry, SandboxCommand

try:
    from modal import App, Client, Image, Sandbox
    from modal.container_process import ContainerProcess
    from modal.exception import ExecutionError, SandboxFilesystemNotFoundError
    from modal.types import FileInfo, FileType
except ImportError as _import_error:
    raise ImportError(
        'Please install `modal` to use the Modal sandbox, '
        'you can use the `modal` optional group — `pip install "pydantic-ai-slim[modal]"`'
    ) from _import_error

if TYPE_CHECKING:
    from .protocol import SandboxBackend, SandboxProcess, SupportsFilesystem, SupportsStart, SupportsStream

__all__ = ('ModalSandbox',)

_Stream = Literal['stdout', 'stderr']

# What Modal's client reports when its own copy of a command's deadline ends the wait.
_CLIENT_DEADLINE_EXIT = -1
# The exit status of a process killed by SIGKILL (128 + 9): what the *server* side of the same
# deadline looks like when its kill lands before the client's deadline fires.
_SIGKILL_EXIT = 137


@dataclass(frozen=True)
class _ModalResult:
    exit_code: int
    stdout: str
    stderr: str
    stdout_dropped: int = 0
    stderr_dropped: int = 0


@dataclass(frozen=True)
class _ModalOutputChunk:
    stream: _Stream
    data: str


class _ModalProcess:
    """A command running inside a Modal sandbox, as returned by `ModalSandbox.start()`."""

    def __init__(self, process: ContainerProcess[str], *, timeout: float | None, deadline: int | None):
        self._process = process
        self._timeout = timeout
        self._deadline = deadline
        self._started = time.monotonic()
        self._output: dict[_Stream, str] = {'stdout': '', 'stderr': ''}
        self._lock = anyio.Lock()
        self._outcome: _ModalResult | Exception | None = None

    @property
    def pid(self) -> int | None:
        # Modal identifies a command by an exec id of its own and never reports the container's
        # OS process id, so there is no honest number to give here.
        return None

    async def stream(self) -> AsyncGenerator[_ModalOutputChunk]:
        """Iterate over the command's output as Modal produces it.

        Chunks from the two streams are interleaved in arrival order. Everything Modal has
        already delivered is also kept, so a later
        [`wait()`][pydantic_ai.sandboxes.SandboxProcess.wait] still reports it — whether this
        iterator ran to completion or was abandoned part way. A read still in flight when the
        iterator is abandoned is cancelled, so output produced after that point is not. It is an
        async generator rather than a bare iterator so that a consumer stopping early can close
        it — and cancel that read — at a point of its own choosing.
        """
        iterators: dict[_Stream, AsyncIterator[str]] = {
            'stdout': aiter(self._process.stdout),
            'stderr': aiter(self._process.stderr),
        }
        arrival = itertools.count()

        async def read_one(name: _Stream) -> tuple[int, _Stream, str | None]:
            chunk = await anext(iterators[name], None)
            # Stamped where the chunk lands rather than where it is collected: a wake-up that
            # finds both streams ready hands them back as an unordered set, and this is what
            # keeps the merge in the order Modal produced the output.
            return next(arrival), name, chunk

        # One read in flight per stream, re-armed as each is consumed.
        pending = {asyncio.ensure_future(read_one(name)) for name in iterators}
        try:
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                arrived = [task.result() for task in sorted(done, key=lambda finished: finished.result()[0])]
                # Recorded before any of it is handed out, because a delivered chunk has already
                # moved the SDK's stream position: a consumer that stops iterating half way must
                # not cost `wait()` output that the sandbox can no longer be asked for again.
                for _, name, chunk in arrived:
                    if chunk is not None:
                        self._output[name] += chunk
                for _, name, chunk in arrived:
                    if chunk is None:  # that stream reached EOF and is not re-armed
                        continue
                    # Re-armed before the chunk is handed over, so the next read is already in
                    # flight while the consumer works through this one.
                    pending.add(asyncio.ensure_future(read_one(name)))
                    yield _ModalOutputChunk(stream=name, data=chunk)
        finally:
            for task in pending:
                task.cancel()

    async def wait(self) -> _ModalResult:
        # The timeout verdict below can only be reached once, so the first call's verdict is the
        # process's verdict: caching it is what makes repeated and concurrent `wait()`s agree,
        # as the protocol requires.
        async with self._lock:
            if self._outcome is None:
                try:
                    self._outcome = await self._settle()
                except Exception as error:
                    self._outcome = error
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome

    async def _settle(self) -> _ModalResult:
        # Modal's own SDK reads both streams alongside the wait; a stream already drained by
        # `stream()` reports itself at EOF and contributes nothing a second time.
        stdout, stderr, exit_code = await asyncio.gather(
            self._process.stdout.read.aio(),
            self._process.stderr.read.aio(),
            self._process.wait.aio(),
        )
        self._output['stdout'] += stdout
        self._output['stderr'] += stderr
        if self._timed_out(exit_code):
            raise TimeoutError(f'command timed out after {self._timeout} seconds and was killed')
        return _ModalResult(exit_code=exit_code, stdout=self._output['stdout'], stderr=self._output['stderr'])

    def _timed_out(self, exit_code: int) -> bool:
        if self._deadline is None:
            return False
        if exit_code == _CLIENT_DEADLINE_EXIT:
            return True
        # A command can exit 137 on its own account (an OOM kill, a `kill -9` it asked for), so
        # that exit only means "the deadline killed it" once the deadline window has elapsed.
        return exit_code == _SIGKILL_EXIT and time.monotonic() - self._started >= self._deadline

    async def kill(self) -> None:
        raise NotImplementedError(
            'Modal exposes no way to kill an individual command; start it with `timeout=` so the '
            'platform kills it at the deadline, or terminate the whole sandbox.'
        )


class _ModalFilesystem:
    def __init__(self, sandbox: Sandbox):
        self._sandbox = sandbox

    async def read_bytes(self, path: str) -> bytes:
        return await self._sandbox.filesystem.read_bytes.aio(path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        # Modal takes the data first, creates missing parents, and replaces existing contents.
        await self._sandbox.filesystem.write_bytes.aio(data, path)

    async def stat(self, path: str) -> FileEntry:
        return _file_entry(await self._sandbox.filesystem.stat.aio(path), path)

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        entries = await self._sandbox.filesystem.list_files.aio(path)
        # A listing names its entries relative to the directory that was listed, so the absolute
        # path the protocol promises is rebuilt from the directory we asked about.
        return [_file_entry(entry, posixpath.join(path, entry.name)) for entry in entries]

    async def make_dir(self, path: str) -> None:
        # Modal's default `create_parents=True` is `mkdir -p`: missing parents are created and an
        # existing directory is not an error.
        await self._sandbox.filesystem.make_directory.aio(path)

    async def remove(self, path: str) -> None:
        # `recursive=True` is what lets this remove a non-empty directory; on a file it changes
        # nothing, so one call covers both halves of the protocol's `remove`.
        await self._sandbox.filesystem.remove.aio(path, recursive=True)

    async def exists(self, path: str) -> bool:
        # Modal has no existence check of its own, and `stat` is the cheapest question that
        # answers one. Only "not found" is an answer: every other failure is still a failure.
        try:
            await self._sandbox.filesystem.stat.aio(path)
        except SandboxFilesystemNotFoundError:
            return False
        return True


def _file_entry(entry: FileInfo, path: str) -> FileEntry:
    is_dir = entry.type is FileType.DIRECTORY
    # A directory's reported size is an implementation detail of the underlying filesystem
    # rather than a content length, so report none for it, like the other built-in backends.
    return FileEntry(name=entry.name, path=path, is_dir=is_dir, size=None if is_dir else entry.size)


def _command_argv(command: SandboxCommand, shell: bool) -> Sequence[str]:
    if shell:
        if not isinstance(command, str):
            raise TypeError('an argv sequence cannot be combined with shell=True; pass a single command string')
        # Modal executes argv and never a shell string, so shell interpretation is requested
        # explicitly — the mirror image of `E2BSandbox`, which has to quote argv back into a
        # string. `/bin/sh` rather than bash: it is the one shell every sandbox image carries.
        return ['/bin/sh', '-c', command]
    if isinstance(command, str):
        raise TypeError('a string command requires shell=True; pass an argv sequence otherwise')
    return command


class ModalSandbox:
    """[`SandboxBackend`][pydantic_ai.sandboxes.SandboxBackend] over a [Modal](https://modal.com) sandbox.

    Commands and file operations run inside a Modal container, so — unlike
    [`LocalSandbox`][pydantic_ai.sandboxes.LocalSandbox] — the host is never exposed. Create one
    with [`create()`][pydantic_ai.sandboxes.modal.ModalSandbox.create], or attach to an existing
    environment with [`connect()`][pydantic_ai.sandboxes.modal.ModalSandbox.connect].

    Both process opt-ins are implemented, which is where this backend differs from
    [`E2BSandbox`][pydantic_ai.sandboxes.e2b.E2BSandbox]: alongside background processes
    ([`SupportsStart`][pydantic_ai.sandboxes.SupportsStart]), Modal's SDK exposes a command's
    output as async-iterable streams that it keeps drained on its own, so live output
    ([`SupportsStream`][pydantic_ai.sandboxes.SupportsStream]) needs no bridging and is offered
    too. What Modal has no API for is killing a single command:
    [`kill()`][pydantic_ai.sandboxes.SandboxProcess.kill] raises `NotImplementedError`, and
    `timeout=` — which Modal enforces itself, killing the command at the deadline — is how a
    command is bounded.

    `output_limit=` is not implemented: Modal always delivers the full output, and dropping
    characters after the fact would misreport what the command produced. Bound it in-command
    instead, e.g. `| tail -c 10000`.

    Deliberately no base class: it conforms to the protocol structurally, like any third-party
    backend would.

    Args:
        sandbox: A live `modal.Sandbox`. Prefer `create()` or `connect()`; pass a sandbox here
            when you need a Modal option those two don't expose, and remember that whoever
            created it owns terminating it.
    """

    provider = 'modal'

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox
        """The underlying `modal.Sandbox`, for provider-specific functionality."""
        self.fs = _ModalFilesystem(sandbox)
        self._working_dir: str | None = None

    @property
    def sandbox_id(self) -> str:
        return self.sandbox.object_id

    @classmethod
    @asynccontextmanager
    async def create(
        cls,
        *,
        app: App | str = 'pydantic-ai-sandbox',
        image: Image | None = None,
        timeout: int = 300,
        workdir: str | None = None,
        env: Mapping[str, str] | None = None,
        cpu: float | tuple[float, float] | None = None,
        memory: int | tuple[int, int] | None = None,
        client: Client | None = None,
    ) -> AsyncGenerator[Self]:
        """Provision a fresh Modal sandbox for the duration of the block, then terminate it.

        The supplier of a sandbox owns its lifecycle, so this is an async context manager: the
        environment is terminated when the block ends, including on failure and on cancellation.
        Modal's own `timeout` remains the backstop for the block that never ends.

        ```python {test="skip"}
        from pydantic_ai.sandboxes.modal import ModalSandbox


        async def main() -> None:
            async with ModalSandbox.create(timeout=600) as sandbox:
                result = await sandbox.run(['python', '-c', 'print(1 + 1)'])
                print(result.stdout)
        ```

        Args:
            app: The Modal app the sandbox belongs to — every sandbox needs one. A name (the
                default) is looked up and created if it doesn't exist yet; pass a `modal.App`
                to reuse one you already hold.
            image: The image the sandbox runs; defaults to Modal's own default image.
            timeout: How long Modal keeps the sandbox alive, in seconds (Modal's own default).
            workdir: Absolute directory commands start in; defaults to the image's.
            env: Environment variables set for the whole sandbox.
            cpu: Reserved cores, or a `(request, limit)` pair.
            memory: Reserved MiB, or a `(request, limit)` pair.
            client: Modal client to use; defaults to the ambient Modal credentials.
        """
        resolved_app = app if isinstance(app, App) else await App.lookup.aio(app, create_if_missing=True, client=client)
        # The ignore is for unparameterized `os.PathLike` mount keys in the SDK's own signature,
        # not for anything this call passes.
        sandbox = await Sandbox.create.aio(  # pyright: ignore[reportUnknownMemberType]
            app=resolved_app,
            image=image,
            timeout=timeout,
            workdir=workdir,
            env=dict(env) if env is not None else None,
            cpu=cpu,
            memory=memory,
            client=client,
        )
        try:
            yield cls(sandbox)
        finally:
            # Shielded because teardown can run inside an already-cancelled scope, where every
            # await re-raises: an unshielded terminate would leak a running (and billed) sandbox.
            with anyio.CancelScope(shield=True):
                await sandbox.terminate.aio()

    @classmethod
    async def connect(cls, sandbox_id: str, *, client: Client | None = None) -> Self:
        """Attach to a Modal sandbox that already exists, without taking over its lifecycle.

        This is the building block for a
        [`SandboxResolver`][pydantic_ai.sandboxes.SandboxResolver] — a capability's
        [`get_sandbox`][pydantic_ai.capabilities.AbstractCapability.get_sandbox] hook turning a
        [`SandboxRef`][pydantic_ai.sandboxes.SandboxRef] back into a live backend, wherever the
        work actually runs:

        ```python {test="skip"}
        from typing import Any

        from pydantic_ai import RunContext
        from pydantic_ai.capabilities import AbstractCapability
        from pydantic_ai.sandboxes import SandboxBackend, SandboxRef
        from pydantic_ai.sandboxes.modal import ModalSandbox


        class ModalCapability(AbstractCapability[Any]):
            async def get_sandbox(
                self, ctx: RunContext[Any], ref: SandboxRef
            ) -> SandboxBackend | None:
                if ref.provider != 'modal':
                    return None
                return await ModalSandbox.connect(ref.sandbox_id)
        ```

        Connecting never creates: `modal.exception.NotFoundError` is raised if the environment is
        gone, rather than an empty replacement being silently swapped in.

        Args:
            sandbox_id: The `sandbox_id` of the environment to attach to.
            client: Modal client to use; defaults to the ambient Modal credentials.
        """
        return cls(await Sandbox.from_id.aio(sandbox_id, client=client))

    async def working_dir(self) -> str:
        # Modal exposes no API for a running sandbox's working directory — it is the image's
        # unless `create(workdir=...)` overrode it — so ask the environment itself. It cannot
        # change, so one answer serves the sandbox's whole life.
        if self._working_dir is None:
            result = await self.run(['pwd'])
            if result.exit_code != 0:
                raise ExecutionError(
                    f'Could not determine the working directory of sandbox {self.sandbox_id!r}: '
                    f'`pwd` exited {result.exit_code}: {result.stderr}'
                )
            self._working_dir = result.stdout.strip()
        return self._working_dir

    async def run(
        self,
        command: SandboxCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        output_limit: int | None = None,
    ) -> _ModalResult:
        process = await self.start(command, shell=shell, cwd=cwd, env=env, timeout=timeout, output_limit=output_limit)
        return await process.wait()

    async def start(
        self,
        command: SandboxCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        output_limit: int | None = None,
    ) -> _ModalProcess:
        if output_limit is not None:
            raise NotImplementedError('ModalSandbox does not bound output; bound it in-command, e.g. `| tail -c`.')
        # Modal takes whole seconds and reads a missing deadline as "run until the sandbox dies",
        # so a sub-second deadline rounds up rather than silently becoming unbounded.
        deadline = None if timeout is None else max(1, math.ceil(timeout))
        process = await self.sandbox.exec.aio(
            *_command_argv(command, shell),
            # Modal's deadline really does kill the command, so — unlike E2B's — the platform
            # owns it, and the process handle only has to recognize the kill it performed.
            timeout=deadline,
            # `workdir=None` leaves the command in the sandbox's own default directory.
            workdir=cwd,
            # `env` is applied on top of the sandbox's environment, not in place of it.
            env=dict(env) if env is not None else None,
            text=True,
        )
        return _ModalProcess(process, timeout=timeout, deadline=deadline)


if TYPE_CHECKING:
    # Pins full structural conformance — signatures included — which `isinstance` cannot check.
    # `__new__` rather than a call, because neither SDK object can be constructed without a
    # live sandbox behind it; this block never runs.
    _sandbox = Sandbox.__new__(Sandbox)
    _process = ContainerProcess[str].__new__(ContainerProcess[str])
    _backend_conforms: SandboxBackend = ModalSandbox(_sandbox)
    _filesystem_backend_conforms: SupportsFilesystem = ModalSandbox(_sandbox)
    _start_conforms: SupportsStart = ModalSandbox(_sandbox)
    _process_conforms: SandboxProcess = _ModalProcess(_process, timeout=None, deadline=None)
    _stream_conforms: SupportsStream = _ModalProcess(_process, timeout=None, deadline=None)
