"""The default local implementation of the [sandbox backend protocol][pydantic_ai.sandboxes.SandboxBackend].

[`LocalSandbox`][pydantic_ai.sandboxes.LocalSandbox] runs commands as plain host subprocesses —
it **isolates nothing** — and doubles as the reference implementation of the protocol.
"""

from __future__ import annotations as _annotations

import asyncio
import os
import shutil
import signal
import tempfile
import time
import uuid
from collections.abc import Awaitable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

import anyio
from typing_extensions import Self

from pydantic_ai._utils import cancel_and_drain, run_in_executor

from .protocol import CommandResult, FileEntry, SandboxCommand, SandboxError, SandboxRef, SandboxTimeoutError

if TYPE_CHECKING:
    from .protocol import SandboxBackend, SupportsFilesystem

__all__ = ('LocalSandbox',)

_MAX_CAPTURE_BYTES = 10 * 1024 * 1024
"""Ceiling on the combined stdout and stderr a single command may produce."""

_MAX_CAPTURE_MIB = _MAX_CAPTURE_BYTES // (1024 * 1024)
"""`_MAX_CAPTURE_BYTES` in MiB, for the error message."""

_READ_CHUNK_BYTES = 64 * 1024
"""Bytes requested per pipe read.

Matches asyncio's own `StreamReader` buffer limit (`asyncio.streams._DEFAULT_LIMIT`, 2**16), so a
read never asks for more than the reader can hold in one go, and one chunk can overshoot the
capture ceiling by at most this much."""

_CHILD_POLL_INTERVAL = 0.01
"""How often to re-check whether the direct child has exited.

Exit is polled rather than awaited (see `_wait_for_direct_child`), so this is the granularity of
a timeout and the idle cost of a long-running command: 100 wake-ups a second."""

_OUTPUT_DRAIN_GRACE = 2.0
"""How long to keep reading a command's pipes after the direct child has exited."""


class _LocalFilesystem:
    @staticmethod
    def _path(path: str) -> Path:
        target = Path(path)
        if not target.is_absolute():
            raise ValueError(f'path must be absolute, got {path!r}')
        return target

    # Every method here wraps a blocking syscall, so the work goes to a thread through the
    # framework's `run_in_executor`. The syscalls themselves have no async form; `anyio.Path`
    # would only move the same offload behind a different helper.
    async def read_bytes(self, path: str) -> bytes:
        return await run_in_executor(self._path(path).read_bytes)

    async def write_bytes(self, path: str, data: bytes) -> None:
        def write() -> None:
            target = self._path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

        await run_in_executor(write)

    async def stat(self, path: str) -> FileEntry:
        def stat() -> FileEntry:
            target = self._path(path)
            size = target.stat().st_size
            is_dir = target.is_dir()
            return FileEntry(name=target.name, path=path, is_dir=is_dir, size=None if is_dir else size)

        return await run_in_executor(stat)

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        def list_entries() -> list[FileEntry]:
            entries: list[FileEntry] = []
            # `os.scandir`, not `Path.iterdir`: each `DirEntry` carries the type and stat data the
            # directory read already returned, so an ordinary entry costs one syscall instead of
            # the three that `iterdir` plus `is_dir` plus `stat` make.
            with os.scandir(self._path(path)) as scan:
                children = sorted(scan, key=lambda child: child.path)
            for child in children:
                is_dir = child.is_dir()
                try:
                    # stat, not lstat: a symlinked file reports its target's size, matching `stat()`.
                    size = None if is_dir else child.stat().st_size
                except OSError:
                    # A broken symlink in the directory must not fail the whole listing.
                    size = None
                entries.append(FileEntry(name=child.name, path=child.path, is_dir=is_dir, size=size))
            return entries

        return await run_in_executor(list_entries)

    async def make_dir(self, path: str) -> None:
        await run_in_executor(lambda: self._path(path).mkdir(parents=True, exist_ok=True))

    async def remove(self, path: str) -> None:
        def remove() -> None:
            target = self._path(path)
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()  # files and symlinks (even to directories) unlink

        await run_in_executor(remove)

    async def exists(self, path: str) -> bool:
        return await run_in_executor(self._path(path).exists)


class LocalSandbox:
    """[`SandboxBackend`][pydantic_ai.sandboxes.SandboxBackend] over host subprocesses and the host filesystem.

    Isolates nothing: commands run as host subprocesses with the host process's privileges.
    It is never attached by default — runs without a sandbox get
    [`UnavailableSandbox`][pydantic_ai.sandboxes.UnavailableSandbox] — so attaching it is an
    explicit opt-in for trusted workloads, tests, and development. POSIX-only: construction
    raises `NotImplementedError` elsewhere, where the timeout contract (kill the whole process
    group at the deadline) can't be honored.

    Commands receive only `PATH`, `HOME`, `LANG`, and `TMPDIR` from the parent when present, plus
    variables explicitly supplied through `env`. This prevents framework credentials from being
    inherited, but is a leak fix rather than an isolation boundary.

    Deliberately no base class: it conforms to the protocol structurally, like any
    third-party backend would. It is also the in-tree worked example of the lazy pattern every
    backend follows — see [`root`][pydantic_ai.sandboxes.LocalSandbox.root].

    Args:
        root: The working directory commands run in and relative paths resolve against; must
            be an absolute path (a relative one would silently depend on the host process's
            working directory). Defaults to a fresh temporary directory, created on first use
            and removed again when the sandbox is used as an async context manager — pass a
            `root` of your own to keep the files a run produces. A caller-supplied `root` is
            never removed, and is canonicalized (symlinks resolved) on first use, so
            [`working_dir()`][pydantic_ai.sandboxes.SandboxBackend.working_dir] reports the
            directory commands actually run in.
    """

    def __init__(self, root: str | Path | None = None):
        if os.name != 'posix':
            raise NotImplementedError(
                'LocalSandbox only supports POSIX platforms: its timeout contract kills the whole '
                'process group. On other platforms, attach a container- or VM-based sandbox instead.'
            )
        if root is not None and not Path(root).is_absolute():
            raise ValueError(
                f'root must be an absolute path, got {str(root)!r}: a relative root would depend on '
                "the host process's working directory at some later moment. Make the intent explicit "
                "at the call site instead, e.g. `LocalSandbox(Path.cwd() / 'work')`."
            )
        self._owns_root = root is None
        self._given_root = None if root is None else Path(root)
        # Always the canonical spelling (symlinks resolved, no `..`), set on first use: the
        # kernel resolves a cwd like `link/..` through the symlink while lexical joins collapse
        # it as text, so a non-canonical root would point `run()` and `fs` at different
        # directories, breaking the protocol's one-environment contract.
        self._resolved_root: Path | None = None
        self._root_lock = anyio.Lock()
        self._id = f'local-{uuid.uuid4().hex}'
        self.fs = _LocalFilesystem()

    @property
    def ref(self) -> SandboxRef:
        return SandboxRef(sandbox_id=self._id)

    @property
    def root(self) -> Awaitable[Path]:
        """The directory commands run in, created on first use.

        Awaitable and never a plain value, which is the point: an operation cannot reach the root
        without going through the step that creates it. Backends for remote providers use the same
        shape for their provider handle, so create-or-attach happens on first use and no method
        can skip it.
        """
        return self._resolve_root()

    async def _resolve_root(self) -> Path:
        # The default temp root is created lazily, so a constructed-but-unused sandbox doesn't
        # leak a directory, and off the event loop, since `mkdtemp` and `resolve` are blocking
        # syscalls. The lock is what makes the two safe together: without it, two concurrent
        # first uses would each create a directory and one would leak.
        async with self._root_lock:
            if self._resolved_root is None:
                if self._given_root is None:
                    self._resolved_root = await run_in_executor(
                        lambda: Path(tempfile.mkdtemp(prefix='pydantic-ai-sandbox-')).resolve()
                    )
                else:
                    self._resolved_root = await run_in_executor(self._given_root.resolve)
            return self._resolved_root

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        # Under the root lock: an unlocked clear would race `_resolve_root()` — a first use
        # blocked on the lock could otherwise recreate a root mid-teardown that nothing
        # would ever remove.
        async with self._root_lock:
            if self._owns_root and self._resolved_root is not None:
                # Reset first so a reused sandbox lazily creates a fresh root instead of
                # resurrecting the deleted path.
                root, self._resolved_root = self._resolved_root, None
                try:
                    await run_in_executor(shutil.rmtree, root)
                except FileNotFoundError:
                    # A command or `fs.remove()` may have deleted the root already; exiting
                    # must not raise (it would mask the exception that ended the block).
                    pass

    async def working_dir(self) -> str:
        return str(await self.root)

    async def _spawn(
        self, command: SandboxCommand, cwd: str | None, env: Mapping[str, str]
    ) -> asyncio.subprocess.Process | Exception:
        """Start the command, returning the spawn failure instead of raising it.

        Returned rather than raised because `run` awaits this through `asyncio.shield`: if the
        run is cancelled or times out mid-spawn, nobody is left to receive the exception, and on
        Python 3.14 an abandoned shielded future reports it to the event loop's exception
        handler. Cancellation is deliberately *not* caught — it must keep propagating.
        """
        try:
            process_cwd = cwd or await self.root
            # `asyncio.create_subprocess_*`, not `anyio.open_process`: this backend closes the
            # private transport to release pipe descriptors that descendants keep open, and reaps
            # the direct child itself. Neither is reachable through anyio's process wrapper.
            # Each command leads its own process group, so the timeout kill takes out the whole
            # tree — killing only `sh` would leave its children running.
            if isinstance(command, str):
                return await asyncio.create_subprocess_shell(
                    command,
                    cwd=process_cwd,
                    env=env,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
            return await asyncio.create_subprocess_exec(
                *command,
                cwd=process_cwd,
                env=env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except Exception as error:
            return error

    async def run(
        self,
        command: SandboxCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        # `CommandResult` is the concrete carrier the built-in backends return; the protocol
        # `SandboxResult` stays structural so third-party backends can return their SDK's own
        # result object without wrapping it.
        deadline = None if timeout is None else time.monotonic() + timeout
        if cwd is not None and not Path(cwd).is_absolute():
            raise ValueError(
                f'cwd must be an absolute path, got {cwd!r}: a relative cwd would resolve against '
                "the host process's working directory, not the sandbox root"
            )
        # Keep the child environment small so the framework's credentials do not reach commands;
        # the caller can explicitly provide any additional variables it needs.
        merged_env = {key: os.environ[key] for key in ('PATH', 'HOME', 'LANG', 'TMPDIR') if key in os.environ}
        if env is not None:
            merged_env.update(env)
        if isinstance(command, str):
            if not shell:
                raise TypeError('a string command requires shell=True; pass an argv sequence otherwise')
        elif shell:
            raise TypeError('an argv sequence cannot be combined with shell=True; pass a single command string')

        spawn = asyncio.create_task(self._spawn(command, cwd, merged_env))
        try:
            # `anyio.fail_after` is the deadline idiom used across the codebase; `asyncio.shield`
            # is what keeps the spawn task itself alive through that deadline, which an anyio
            # shielded scope cannot do (it would also block `fail_after` from firing).
            with anyio.fail_after(None if deadline is None else max(0.0, deadline - time.monotonic())):
                outcome = await asyncio.shield(spawn)
        except anyio.get_cancelled_exc_class():
            spawn.add_done_callback(self._kill_abandoned_spawn)
            raise
        except TimeoutError as error:
            spawn.add_done_callback(self._kill_abandoned_spawn)
            raise SandboxTimeoutError(
                f'command timed out after {timeout} seconds and was killed',
                stdout='',
                stderr='',
                timeout=timeout,
            ) from error
        if isinstance(outcome, Exception):
            raise outcome
        process = outcome
        stdout_pipe, stderr_pipe = process.stdout, process.stderr
        if stdout_pipe is None or stderr_pipe is None:  # pragma: no cover
            # Unreachable: both are spawned with `PIPE`. Stated rather than asserted so an
            # optimized interpreter still fails loudly instead of raising `AttributeError` later.
            raise SandboxError('local sandbox could not capture the command output pipes')
        stdout_buffer = bytearray()
        stderr_buffer = bytearray()
        # Plain tasks rather than an anyio task group: a reader that trips the output ceiling
        # raises `SandboxError`, and a task group would deliver it wrapped in a
        # `BaseExceptionGroup`, changing the exception callers and tests see.
        reader_tasks = [
            asyncio.create_task(self._read_stream(stdout_pipe, stdout_buffer, stderr_buffer)),
            asyncio.create_task(self._read_stream(stderr_pipe, stderr_buffer, stdout_buffer)),
        ]

        try:
            exit_code = await self._wait_for_direct_child(process, reader_tasks, deadline)
            await self._drain_output(reader_tasks, deadline)
            self._close_transport(process)
        except TimeoutError as error:
            # The contract: a timeout kills the command first, then raises SandboxTimeoutError —
            # even when a hardened host denies the group kill, in which case the
            # denial rides along as the cause instead of replacing the promised type.
            denial = await self._kill_and_reap_and_close(process, reader_tasks)
            stdout = stdout_buffer.decode('utf-8', errors='replace')
            stderr = stderr_buffer.decode('utf-8', errors='replace')
            if denial is not None:
                raise SandboxTimeoutError(
                    f'command timed out after {timeout} seconds; killing its process group was '
                    'denied, so only the direct child was killed and grandchildren may survive',
                    stdout=stdout,
                    stderr=stderr,
                    timeout=timeout,
                ) from denial
            raise SandboxTimeoutError(
                f'command timed out after {timeout} seconds and was killed',
                stdout=stdout,
                stderr=stderr,
                timeout=timeout,
            ) from error
        except BaseException:
            # Cancellation or any other failure while the pipes were open: kill the group,
            # but let the in-flight exception keep propagating — replacing a cancellation
            # with a kill-denial error would break the caller's cancel scope. `returncode`
            # alone can't tell us the group is gone: a shell can exit while a background
            # child keeps the pipes open.
            await self._kill_and_reap_and_close(process, reader_tasks)
            raise
        return CommandResult(
            exit_code=exit_code,
            stdout=stdout_buffer.decode('utf-8', errors='replace'),
            stderr=stderr_buffer.decode('utf-8', errors='replace'),
        )

    @staticmethod
    async def _read_stream(stream: asyncio.StreamReader, buffer: bytearray, other_buffer: bytearray) -> None:
        while chunk := await stream.read(_READ_CHUNK_BYTES):
            buffer.extend(chunk)
            if len(buffer) + len(other_buffer) > _MAX_CAPTURE_BYTES:
                raise SandboxError(
                    f'local sandbox output exceeded {_MAX_CAPTURE_MIB} MiB safety limit; '
                    "redirect the command's "
                    'output to a file and read a window of it with `read_file` instead'
                )

    @staticmethod
    async def _wait_for_direct_child(
        process: asyncio.subprocess.Process,
        reader_tasks: list[asyncio.Task[None]],
        deadline: float | None,
    ) -> int:
        # Polled, not `await process.wait()`: that only returns once every pipe reaches EOF, which a
        # descendant holding the command's stdout can postpone indefinitely.
        returncode = process.returncode
        while returncode is None:
            # A finished reader means the output ceiling tripped: re-raise its `SandboxError`
            # here rather than waiting for a command that will never be read to completion.
            for task in reader_tasks:
                if task.done():
                    task.result()
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError
            await asyncio.sleep(_CHILD_POLL_INTERVAL if remaining is None else min(_CHILD_POLL_INTERVAL, remaining))
            returncode = process.returncode
        for task in reader_tasks:
            if task.done():
                task.result()
        return returncode

    @staticmethod
    async def _drain_output(reader_tasks: list[asyncio.Task[None]], deadline: float | None) -> None:
        pending_readers: set[asyncio.Task[None]] = set()
        for task in reader_tasks:
            if task.done():
                task.result()  # re-raises a tripped output ceiling
            else:
                pending_readers.add(task)
        if not pending_readers:
            return
        remaining = _OUTPUT_DRAIN_GRACE
        if deadline is not None:
            remaining = min(remaining, max(0.0, deadline - time.monotonic()))
        done, pending_readers = await asyncio.wait(pending_readers, timeout=remaining)
        for task in done:
            task.result()  # re-raises a tripped output ceiling
        if pending_readers:
            await cancel_and_drain(*pending_readers)

    async def _kill_and_reap_and_close(
        self,
        process: asyncio.subprocess.Process,
        reader_tasks: list[asyncio.Task[None]],
    ) -> PermissionError | None:
        cleanup = asyncio.create_task(self._kill_and_reap(process))
        try:
            # `asyncio.shield`, not an anyio shielded scope: what must survive an outer cancel is
            # the kill *task*, so that a cancelled run never abandons a live process group.
            denial = await asyncio.shield(cleanup)
        finally:
            await cancel_and_drain(*reader_tasks)
            self._close_transport(process)
        return denial

    @staticmethod
    def _close_transport(process: asyncio.subprocess.Process) -> None:
        # The private transport closes pipe descriptors retained by descendants after child exit.
        # Called from three places, which is why it is a method and not inlined.
        process._transport.close()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

    async def _kill_and_reap(self, process: asyncio.subprocess.Process) -> PermissionError | None:
        """Kill the group and reap the direct child, reporting a denied group kill.

        Reported instead of raised, so the caller decides which exception its contract owes.
        """
        try:
            self._kill(process)
        except PermissionError as error:
            return error
        finally:
            self._close_transport(process)
            # `wait()` on a killed child returns; it can still raise if the caller is cancelled,
            # which is why every call site runs inside `_kill_and_reap_and_close`'s shield.
            await process.wait()
        return None

    @staticmethod
    def _kill_abandoned_spawn(spawn: asyncio.Task[asyncio.subprocess.Process | Exception]) -> None:
        # The run that awaited this spawn was cancelled: nobody is left to receive a spawn
        # failure, and the loop's child watcher still reaps the direct child after the kill.
        if spawn.cancelled():  # pragma: no cover
            return
        outcome = spawn.result()
        if isinstance(outcome, Exception):
            return
        try:
            os.killpg(outcome.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            # `_kill`'s fallback, minus the propagation: nobody is left to receive the
            # error, so best-effort kill the direct child.
            with suppress(ProcessLookupError):
                outcome.kill()

    @staticmethod
    def _kill(process: asyncio.subprocess.Process) -> None:
        # Internal to `LocalSandbox`: `SandboxBackend` has no `kill` member, because not every
        # platform lets a client stop a running command.
        #
        # The child leads its own process group (`start_new_session=True`), so "already
        # exited" is the only benign failure. If a hardened host denies `killpg`, kill the
        # direct child as a fallback but still raise: grandchildren may survive, and the
        # caller must not believe the whole group was killed.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            try:
                process.kill()
            finally:
                raise


if TYPE_CHECKING:
    # Pins full structural conformance — signatures included — which `isinstance` cannot check.
    # Type-check time only: it never runs, and it is how a backend proves it satisfies the
    # protocols without inheriting from them.
    _conforms: SandboxBackend = LocalSandbox()
    _filesystem_backend_conforms: SupportsFilesystem = LocalSandbox()
