"""[E2B](https://e2b.dev) cloud sandboxes as a [sandbox backend][pydantic_ai.sandboxes.SandboxBackend].

Needs the `e2b` optional dependency group (`pip install "pydantic-ai-slim[e2b]"`). Import
[`E2BSandbox`][pydantic_ai.sandboxes.e2b.E2BSandbox] from this module directly:
`pydantic_ai.sandboxes` deliberately doesn't re-export it, so importing the package never
pulls in the E2B SDK.
"""

from __future__ import annotations as _annotations

import asyncio
import math
import posixpath
import shlex
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING

import anyio
from typing_extensions import Self, Unpack

from ._lifecycle import destroy_quietly, guarded_create
from .protocol import FileEntry, SandboxCommand

try:
    from e2b import (
        ApiParams,
        AsyncCommandHandle,
        AsyncSandbox,
        CommandExitException,
        CommandResult,
        EntryInfo,
        FileType,
        SandboxException,
    )
except ImportError as _import_error:
    raise ImportError(
        'Please install `e2b` to use the E2B sandbox, '
        'you can use the `e2b` optional group — `pip install "pydantic-ai-slim[e2b]"`'
    ) from _import_error

if TYPE_CHECKING:
    from .protocol import SandboxBackend, SandboxProcess, SupportsFilesystem, SupportsStart

__all__ = ('E2BSandbox',)

# Bounds the best-effort kill of a command whose wait is over: a sandbox that has stopped
# answering must not hang the caller that is walking away from it.
_TEARDOWN_TIMEOUT = 30.0


@dataclass(frozen=True)
class _E2BResult:
    exit_code: int
    stdout: str
    stderr: str
    stdout_dropped: int = 0
    stderr_dropped: int = 0


class _E2BProcess:
    """A command running inside an E2B sandbox, as returned by `E2BSandbox.start()`."""

    def __init__(self, handle: AsyncCommandHandle, *, timeout: float | None):
        self._handle = handle
        self._timeout = timeout
        # Stamped here rather than at the first `wait()`, so `start(timeout=5)` bounds the
        # command's life and not the length of whichever wait happens to come first.
        self._deadline = math.inf if timeout is None else anyio.current_time() + timeout
        self._lock = anyio.Lock()
        self._pump: asyncio.Future[CommandResult] | None = None
        self._abandoned = False
        self._outcome: _E2BResult | Exception | None = None

    @property
    def pid(self) -> int:
        return self._handle.pid

    async def wait(self) -> _E2BResult:
        # The deadline below fires once, so the first call's verdict is the process's verdict:
        # caching it is what makes repeated and concurrent `wait()`s agree, as the protocol requires.
        async with self._lock:
            if self._outcome is None:
                try:
                    self._outcome = await self._settle()
                except Exception as error:
                    self._outcome = error
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome

    async def _settle(self) -> _E2BResult:
        with anyio.CancelScope(deadline=self._deadline):
            try:
                result = await self._await_pump()
            except CommandExitException as exit_error:
                # A non-zero exit is a normal result, not an error; the SDK reports it by raising.
                return _E2BResult(exit_code=exit_error.exit_code, stdout=exit_error.stdout, stderr=exit_error.stderr)
            return _E2BResult(exit_code=result.exit_code, stdout=result.stdout, stderr=result.stderr)
        # Only the deadline reaches here: a cancel scope absorbs the cancellation it raised
        # itself and lets an outer one through, which is what keeps a caller's cancellation — or
        # a transport `TimeoutError` from the SDK — from being mistaken for the deadline.
        kill_failure = await self.abandon()
        # E2B's own command deadline only tears the event stream down and leaves the command
        # running in the sandbox, so the kill the timeout contract promises is ours to perform.
        # It is reported either way: a kill that failed cannot turn the timeout into a success.
        raise TimeoutError(f'command timed out after {self._timeout} seconds and was killed') from kill_failure

    async def _await_pump(self) -> CommandResult:
        """Wait for the SDK's event pump without handing it this caller's cancellation.

        `AsyncCommandHandle.wait()` awaits a shared `asyncio.Task` — the pump collecting the
        command's output and exit code — and awaiting a task cancels it along with its awaiter.
        One cancelled `wait()` would therefore kill the pump for good, and every later `wait()`
        on the same process would raise `CancelledError` instead of the process's result.
        """
        if self._pump is None:
            self._pump = asyncio.ensure_future(self._handle.wait())
            # A wait this caller walked away from still completes; retrieving its outcome keeps
            # the loop from reporting it as an exception nobody ever looked at.
            self._pump.add_done_callback(_retrieve_outcome)
        return await asyncio.shield(self._pump)

    async def abandon(self) -> Exception | None:
        """Best-effort teardown of a command whose wait will never finish.

        Returns the kill's failure rather than raising it: the verdict that brought us here —
        a deadline, a cancelled `run()` — is the one the caller has to see. Called at most once
        per command, so a `run()` whose own deadline already dealt with it doesn't kill twice.
        """
        if self._abandoned:
            return None
        self._abandoned = True
        failure: Exception | None = None
        # Shielded and bounded, because this runs on cancellation paths, where an unshielded
        # await re-raises before the command is dealt with.
        with anyio.CancelScope(shield=True), anyio.move_on_after(_TEARDOWN_TIMEOUT):
            try:
                await self._handle.kill()
            except Exception as error:
                failure = error
            # The kill stops the command; this closes the event stream the SDK keeps open for
            # it, which is the SDK's own teardown surface for a handle nobody will wait on.
            with suppress(Exception):
                await self._handle.disconnect()
        return failure

    async def kill(self) -> None:
        await self._handle.kill()


def _retrieve_outcome(wait: asyncio.Future[CommandResult]) -> None:
    if not wait.cancelled():
        wait.exception()


class _E2BFilesystem:
    def __init__(self, sandbox: AsyncSandbox):
        self._sandbox = sandbox

    async def read_bytes(self, path: str) -> bytes:
        # `format='bytes'` is the byte-exact read; it yields a `bytearray`, the protocol wants `bytes`.
        return bytes(await self._sandbox.files.read(path, format='bytes'))

    async def write_bytes(self, path: str, data: bytes) -> None:
        # E2B's `write` already creates missing parents and replaces existing contents.
        # The ignore is for an unparameterized `IO` in the SDK's own signature, not for our call.
        await self._sandbox.files.write(path, data)  # pyright: ignore[reportUnknownMemberType]

    async def stat(self, path: str) -> FileEntry:
        return _file_entry(await self._sandbox.files.get_info(path))

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        return [_file_entry(entry) for entry in await self._sandbox.files.list(path)]

    async def make_dir(self, path: str) -> None:
        # Creates parents, and returns `False` rather than raising when the directory
        # already exists: `mkdir -p` semantics, so the return value carries nothing we need.
        await self._sandbox.files.make_dir(path)

    async def remove(self, path: str) -> None:
        await self._sandbox.files.remove(path)

    async def exists(self, path: str) -> bool:
        return await self._sandbox.files.exists(path)


def _file_entry(entry: EntryInfo) -> FileEntry:
    is_dir = entry.type is FileType.DIR
    # A directory's reported size is an implementation detail of the underlying filesystem
    # rather than a content length, so report none for it, like the other built-in backends.
    return FileEntry(name=entry.name, path=entry.path, is_dir=is_dir, size=None if is_dir else entry.size)


def _command_text(command: SandboxCommand, shell: bool) -> str:
    if shell:
        if not isinstance(command, str):
            raise TypeError('an argv sequence cannot be combined with shell=True; pass a single command string')
        return command
    if isinstance(command, str):
        raise TypeError('a string command requires shell=True; pass an argv sequence otherwise')
    if not command:
        # An empty argv quotes back into an empty shell string, which bash would run as a
        # successful command that did nothing; `LocalSandbox` rejects the same thing, because
        # there is no program in an empty argv to report an exit code for.
        raise TypeError('a command needs at least the program to run; the argv sequence is empty')
    # E2B only executes shell strings (every command runs as `/bin/bash -l -c <string>`), so argv
    # form is quoted back into one. `shlex.join` makes each element a single literal word, which
    # is what argv execution means: no element can be split, expanded, or read as an operator.
    return shlex.join(command)


class E2BSandbox:
    """[`SandboxBackend`][pydantic_ai.sandboxes.SandboxBackend] over an [E2B](https://e2b.dev) cloud sandbox.

    Commands and file operations run inside E2B's Firecracker microVM, so — unlike
    [`LocalSandbox`][pydantic_ai.sandboxes.LocalSandbox] — the host is never exposed. Create one
    with [`create()`][pydantic_ai.sandboxes.e2b.E2BSandbox.create], or attach to an existing
    environment with [`connect()`][pydantic_ai.sandboxes.e2b.E2BSandbox.connect].

    Background processes are supported natively
    ([`SupportsStart`][pydantic_ai.sandboxes.SupportsStart]), but live output
    ([`SupportsStream`][pydantic_ai.sandboxes.SupportsStream]) is not: the E2B async SDK
    delivers output through `on_stdout`/`on_stderr` callbacks that its own event pump awaits, so
    turning them into an async iterator needs a queue that either grows without bound or stalls
    the pump — and a stalled pump never completes `wait()`. Poll the process with
    `wait()` instead, or tee the output to a file inside the sandbox and read it.

    `output_limit=` is not implemented either: E2B always delivers the full output, and dropping
    characters after the fact would misreport what the command produced. Bound it in-command
    instead, e.g. `| tail -c 10000`.

    `timeout=` is enforced here rather than by E2B, whose own command deadline only drops the
    event stream and leaves the command running. It runs from `start()`, so it bounds the
    command's life rather than the length of whichever wait happens to come first, and the kill
    it promises is performed even when the caller is being cancelled at the same moment. A
    [`run()`][pydantic_ai.sandboxes.SandboxBackend.run] that is cancelled kills the command it
    started, for the same reason; a cancelled
    [`wait()`][pydantic_ai.sandboxes.SandboxProcess.wait] on a process from
    [`start()`][pydantic_ai.sandboxes.SupportsStart.start] does not, because that process is the
    caller's, and it stays waitable afterwards.

    Deliberately no base class: it conforms to the protocol structurally, like any third-party
    backend would.

    Args:
        sandbox: A live E2B `AsyncSandbox`. Prefer `create()` or `connect()`; pass a sandbox
            here only when you already hold one, and remember that whoever created it owns
            killing it.
    """

    provider = 'e2b'

    def __init__(self, sandbox: AsyncSandbox):
        self.sandbox = sandbox
        """The underlying E2B `AsyncSandbox`, for provider-specific functionality."""
        self.fs = _E2BFilesystem(sandbox)
        self._working_dir: str | None = None

    @property
    def sandbox_id(self) -> str:
        return self.sandbox.sandbox_id

    @classmethod
    @asynccontextmanager
    async def create(
        cls,
        template: str | None = None,
        *,
        timeout: int | None = None,
        envs: Mapping[str, str] | None = None,
        metadata: Mapping[str, str] | None = None,
        **api_params: Unpack[ApiParams],
    ) -> AsyncGenerator[Self]:
        """Provision a fresh E2B sandbox for the duration of the block, then kill it.

        The supplier of a sandbox owns its lifecycle, so this is an async context manager: the
        environment is killed when the block ends, including on failure and on cancellation.
        E2B's own `timeout` remains the backstop for the block that never ends.

        ```python {test="skip"}
        from pydantic_ai.sandboxes.e2b import E2BSandbox


        async def main() -> None:
            async with E2BSandbox.create(timeout=600) as sandbox:
                result = await sandbox.run(['python', '-c', 'print(1 + 1)'])
                print(result.stdout)
        ```

        Args:
            template: E2B sandbox template name or ID; defaults to E2B's `base` template.
            timeout: How long E2B keeps the sandbox alive, in seconds (E2B's default is 300).
            envs: Environment variables set for the whole sandbox.
            metadata: Custom metadata to tag the sandbox with, e.g. the run it belongs to.
            api_params: E2B connection options such as `api_key` (which defaults to the
                `E2B_API_KEY` environment variable) and `domain`.
        """
        # Guarded because a caller cancelled while the create is in flight would otherwise leave
        # a running (and billed) sandbox behind that nobody holds a handle to.
        sandbox = await guarded_create(
            AsyncSandbox.create(
                template=template,
                timeout=timeout,
                envs=dict(envs) if envs is not None else None,
                metadata=dict(metadata) if metadata is not None else None,
                **api_params,
            ),
            lambda created: created.kill(),
        )
        try:
            yield cls(sandbox)
        finally:
            await destroy_quietly(sandbox.kill())

    @classmethod
    async def connect(cls, sandbox_id: str, *, timeout: int | None = None, **api_params: Unpack[ApiParams]) -> Self:
        """Attach to an E2B sandbox that already exists, without taking over its destruction.

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
        from pydantic_ai.sandboxes.e2b import E2BSandbox


        class E2BCapability(AbstractCapability[Any]):
            async def get_sandbox(
                self, ctx: RunContext[Any], ref: SandboxRef
            ) -> SandboxBackend | None:
                if ref.provider != 'e2b':
                    return None
                return await E2BSandbox.connect(ref.sandbox_id)
        ```

        Connecting never creates: `e2b.SandboxNotFoundException` is raised if the environment is
        gone, rather than an empty replacement being silently swapped in. It is not passive
        either, and there is no way to ask E2B for a look that is: a paused sandbox is resumed,
        and the sandbox's own keep-alive timeout is set to `timeout` — E2B's own default of 300
        seconds when it is omitted — for a sandbox whose remaining time is shorter than that.

        Args:
            sandbox_id: The `sandbox_id` of the environment to attach to.
            timeout: How long E2B keeps the sandbox alive from now, in seconds; a running
                sandbox's timeout is only ever extended, never cut short. E2B applies its own
                default of 300 seconds when this is omitted.
            api_params: E2B connection options such as `api_key` (which defaults to the
                `E2B_API_KEY` environment variable) and `domain`.
        """
        return cls(await AsyncSandbox.connect(sandbox_id, timeout=timeout, **api_params))

    async def working_dir(self) -> str:
        # E2B exposes no API for the template's default directory, and a command started without
        # `cwd` lands in exactly that directory, so ask the environment itself. It cannot change.
        if self._working_dir is None:
            result = await self.run(['pwd'])
            if result.exit_code != 0:
                raise SandboxException(
                    f'Could not determine the working directory of sandbox {self.sandbox_id!r}: '
                    f'`pwd` exited {result.exit_code}: {result.stderr}'
                )
            # Every command runs under `bash -l`, so a template whose login shell prints a banner
            # puts it on stdout ahead of the answer: the path is the last line, and only an
            # absolute one is an answer. Caching whatever else the shell printed would hand every
            # later `resolve()` a working directory that is not one.
            working_dir = result.stdout.strip().rpartition('\n')[2].strip()
            if not posixpath.isabs(working_dir):
                raise SandboxException(
                    f'Could not determine the working directory of sandbox {self.sandbox_id!r}: '
                    f'`pwd` printed {result.stdout!r}'
                )
            self._working_dir = working_dir
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
    ) -> _E2BResult:
        process = await self.start(command, shell=shell, cwd=cwd, env=env, timeout=timeout, output_limit=output_limit)
        try:
            return await process.wait()
        except BaseException:
            # `run()` owns the command it started, so a caller that walks away from it — a
            # cancellation, an SDK failure mid-wait — must not leave it running in the sandbox.
            # (`start()` hands the process to the caller instead, so `wait()` never kills.)
            await process.abandon()
            raise

    async def start(
        self,
        command: SandboxCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        output_limit: int | None = None,
    ) -> _E2BProcess:
        if output_limit is not None:
            raise NotImplementedError('E2BSandbox does not bound output; bound it in-command, e.g. `| tail -c`.')
        # Guarded like `create()`: cancelling the caller of an in-flight start would otherwise
        # leave a command running in the sandbox that nobody holds a handle to.
        handle = await guarded_create(
            self.sandbox.commands.run(
                _command_text(command, shell),
                background=True,
                # `envs` are applied on top of the sandbox's environment, not in place of it.
                envs=dict(env) if env is not None else None,
                # `cwd=None` leaves the command in the sandbox's own default directory.
                cwd=cwd,
                # `0` disables E2B's command deadline (which would otherwise default to 60
                # seconds). The deadline is the process handle's to own, because E2B's merely
                # drops the event stream while the command keeps running, which the timeout
                # contract forbids.
                timeout=0,
            ),
            lambda started: started.kill(),
        )
        return _E2BProcess(handle, timeout=timeout)


if TYPE_CHECKING:
    # Pins full structural conformance — signatures included — which `isinstance` cannot check.
    # `__new__` rather than a call, because neither SDK object can be constructed without a
    # live sandbox behind it; this block never runs.
    _sandbox = AsyncSandbox.__new__(AsyncSandbox)
    _handle = AsyncCommandHandle.__new__(AsyncCommandHandle)
    _backend_conforms: SandboxBackend = E2BSandbox(_sandbox)
    _filesystem_backend_conforms: SupportsFilesystem = E2BSandbox(_sandbox)
    _start_conforms: SupportsStart = E2BSandbox(_sandbox)
    _process_conforms: SandboxProcess = _E2BProcess(_handle, timeout=None)
