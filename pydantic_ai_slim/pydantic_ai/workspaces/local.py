"""A local implementation of the [workspace backend protocol][pydantic_ai.workspaces.WorkspaceBackend].

[`LocalWorkspaceBackend`][pydantic_ai.workspaces.LocalWorkspaceBackend] runs commands as plain host
subprocesses — it **isolates nothing**.
"""

from __future__ import annotations as _annotations

import asyncio
import os
import shutil
import signal
import stat as stat_module
from collections.abc import AsyncGenerator, Awaitable, Mapping, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from subprocess import DEVNULL, PIPE
from typing import cast

import anyio
import anyio.abc

from pydantic_ai._utils import BaseExceptionGroup, run_in_executor

from .protocol import (
    CommandResult,
    FileEntry,
    SupportsCommands,
    SupportsFilesystem,
    SupportsRealpath,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
    validate_timeout,
)

__all__ = ('LocalWorkspaceBackend',)

# Not secrets, and without them commands miss the host's tools and the user's configuration.
_INHERITED_ENV = ('PATH', 'HOME', 'LANG', 'LC_ALL', 'LC_CTYPE')
_MAX_CAPTURE_BYTES = 10 * 1024 * 1024
"""Ceiling on the combined stdout and stderr a single command may produce."""

_OUTPUT_DRAIN_GRACE = 2.0
"""How long to keep reading a command's pipes after the direct child has exited."""

# Before anyio 4.15, on asyncio, `Process.wait()` and `aclose()` also wait for the output pipes to close
# (https://github.com/agronholm/anyio/issues/1174), so a command that leaves a background child holding
# stdout open (`sleep 30 & echo done`) would hang `run()` until that child exits. On those versions
# `_wait_for_exit` polls `returncode` and `_close` closes our pipe ends first. Delete this workaround
# once `anyio>=4.15` is the minimum.
_ANYIO_WAITS_FOR_PIPES = tuple(int(part) for part in version('anyio').split('.')[:2]) < (4, 15)
_EXIT_POLL_INTERVAL = 0.005
_SPAWN_GRACE = 5.0
_REAP_GRACE = 2.0


def _waits_for_pipes() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:  # pragma: no cover - only reached on Trio, which CI does not run
        return False
    return _ANYIO_WAITS_FOR_PIPES


async def _shielded(awaitable: Awaitable[None], deadline: float) -> None:
    """Finish or cancel bounded work even when the caller is cancelled."""
    timed_out = False

    async def child() -> None:
        nonlocal timed_out
        # A child owns the operation until completion: native Task.cancel() cannot interrupt its shield.
        with anyio.move_on_after(max(0, deadline - anyio.current_time()), shield=True) as scope:
            await awaitable
        timed_out = scope.cancel_called

    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(child)
    except BaseExceptionGroup as group:
        # The child's own error, as a single exception rather than a group.
        error = group.exceptions[0]
        error.__suppress_context__ = True
        raise error
    if timed_out:
        raise TimeoutError('local workspace subprocess operation exceeded its grace period')


@dataclass
class _FileLock:
    lock: anyio.Lock = field(default_factory=anyio.Lock)
    users: int = 0


class LocalWorkspaceBackend(WorkspaceBackend, SupportsCommands, SupportsFilesystem, SupportsRealpath):
    """Run commands as subprocesses on this machine and use its filesystem (POSIX only).

    This isolates nothing: commands and absolute paths reach anywhere this process can. Commands
    inherit only `PATH`, `HOME` and locale (`LANG`, `LC_ALL`, `LC_CTYPE`), so they find the host's
    tools and use its text encoding without inheriting arbitrary secrets.
    Background jobs outlive `run()`; redirect their output to avoid waiting up to two seconds
    for inherited output pipes. The caller manages those jobs when the host exits.
    The directory is the environment: its [`ref`][pydantic_ai.workspaces.LocalWorkspaceBackend.ref]
    exists from construction, and the first operation raises
    [`WorkspaceUnavailableError`][pydantic_ai.workspaces.WorkspaceUnavailableError] if it is missing.

    Args:
        working_dir: Where commands start and relative paths resolve; `~` is expanded and a relative
            path is taken from the current directory. The caller creates and removes it.
        env: Environment variables for every command, on top of the inherited `PATH`, `HOME`, `LANG`,
            `LC_ALL` and `LC_CTYPE`; the per-call `env` goes on top.
    """

    def __init__(self, working_dir: str | Path, *, env: Mapping[str, str] | None = None):
        if os.name != 'posix':
            raise NotImplementedError(
                '`LocalWorkspaceBackend` only supports POSIX platforms at the moment: its timeout contract '
                'kills the whole process group. On other platforms, attach a container- or VM-based '
                'workspace instead.'
            )
        expanded = Path(working_dir).expanduser()
        # Absolute from here on, so a later change of the process's directory cannot move the workspace.
        # Symlinks are left for `working_dir()` to resolve on first use.
        absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
        self._working_dir = absolute
        # Resolved on first use, not here, because capabilities build backends inside the event loop.
        self._resolved_working_dir: Path | None = None
        # The ref has a stable lexical spelling, but the live directory retains the original
        # path: a symlink followed by `..` has different kernel and lexical meanings.
        self._ref = WorkspaceRef(provider='local', id=os.path.normpath(absolute))
        self._env = {name: os.environ[name] for name in _INHERITED_ENV if name in os.environ} | dict(env or {})
        self._file_locks: dict[str, _FileLock] = {}

    @property
    def ref(self) -> WorkspaceRef:
        """`WorkspaceRef(provider='local', id=<absolute working_dir>)`, available from construction."""
        return self._ref

    def _ensure_alive(self) -> Path:
        root = self._resolved_working_dir or self._working_dir.resolve()
        if not root.is_dir():
            raise WorkspaceUnavailableError(f'local workspace directory {self._working_dir!s} does not exist')
        self._resolved_working_dir = root
        return root

    async def _get_working_dir(self) -> Path:
        return await run_in_executor(self._ensure_alive)

    async def working_dir(self) -> str:
        return str(await self._get_working_dir())

    @staticmethod
    def _path(path: str) -> Path:
        target = Path(path)
        if not target.is_absolute():
            raise ValueError(f'path must be absolute, got {path!r}')
        return target

    @asynccontextmanager
    async def _file_lock(self, path: str) -> AsyncGenerator[None]:
        """Serialize this backend's reads and writes of one file.

        A write truncates in place from a worker thread, so parallel tool calls on the same file
        would otherwise interleave into mixed content or read a partial file. Keyed by the real
        path so aliases share a lock; an entry lives only while someone holds or waits on it.
        """
        key = await run_in_executor(os.path.realpath, path)
        entry = self._file_locks.setdefault(key, _FileLock())
        entry.users += 1
        try:
            async with entry.lock:
                yield
        finally:
            entry.users -= 1
            if not entry.users:
                del self._file_locks[key]

    # File operations run in a thread: filesystem calls block, and must not stall the event loop.

    async def read_bytes(self, path: str) -> bytes:
        def read() -> bytes:
            self._ensure_alive()
            # O_NONBLOCK lets us inspect FIFOs and devices without opening a blocking stream.
            fd = os.open(self._path(path), os.O_RDONLY | os.O_NONBLOCK)
            # Check before wrapping: `os.fdopen` would reject a directory without closing the fd.
            try:
                mode = os.fstat(fd).st_mode
                if stat_module.S_ISDIR(mode):
                    raise IsADirectoryError(path)
                if not stat_module.S_ISREG(mode):
                    raise OSError(f'not a regular file: {path!r}')
                with open(fd, 'rb', closefd=False) as file:
                    return file.read()
            finally:
                os.close(fd)

        async with self._file_lock(path):
            return await run_in_executor(read)

    async def write_bytes(self, path: str, data: bytes) -> None:
        """Write bytes to a file, creating missing parents and writing through an existing symlink.

        The file is rewritten in place. Reads and writes of one file through this backend are
        serialized; other processes and commands can still interleave with them.
        """

        def write() -> None:
            target = self._path(path)
            self._ensure_alive()
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
            except FileExistsError as error:
                # Only the parent itself existing as a file gets here; deeper ones raise `NotADirectoryError`.
                raise NotADirectoryError(path) from error
            with suppress(FileNotFoundError):
                # Opening a FIFO or device for writing would block or write somewhere unexpected.
                mode = target.stat().st_mode
                if not stat_module.S_ISREG(mode) and not stat_module.S_ISDIR(mode):
                    raise OSError(f'not a regular file: {path!r}')
            target.write_bytes(data)

        async with self._file_lock(path):
            await run_in_executor(write)

    async def stat(self, path: str) -> FileEntry:
        def stat() -> FileEntry:
            self._ensure_alive()
            target = self._path(path)
            info = target.stat()
            size = info.st_size
            is_dir = stat_module.S_ISDIR(info.st_mode)
            return FileEntry(name=target.name, path=path, is_dir=is_dir, size=None if is_dir else size)

        return await run_in_executor(stat)

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        def list_entries() -> list[FileEntry]:
            self._ensure_alive()
            entries: list[FileEntry] = []
            # `os.scandir`, not `Path.iterdir`: each `DirEntry` carries the type and stat data the
            # directory read already returned, so an ordinary entry costs one syscall instead of
            # the three that `iterdir` plus `is_dir` plus `stat` make.
            with os.scandir(self._path(path)) as scan:
                children = sorted(scan, key=lambda child: child.path)
            for child in children:
                try:
                    is_dir = child.is_dir()
                except OSError:
                    # A symlink loop has no resolvable target; keep the entry, not the failure.
                    is_dir = False
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
        def make() -> None:
            target = self._path(path)
            self._ensure_alive()
            target.mkdir(parents=True, exist_ok=True)

        await run_in_executor(make)

    async def remove(self, path: str) -> None:
        def remove() -> None:
            target = self._path(path)
            root = self._ensure_alive()
            # Never allow a recursive delete to take the workspace itself or its parents.
            if not target.is_symlink() and target.resolve() in (root, *root.parents):
                raise ValueError('cannot remove the workspace root or its ancestor')
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()  # files and symlinks (even to directories) unlink

        await run_in_executor(remove)

    async def exists(self, path: str) -> bool:
        def check() -> bool:
            self._ensure_alive()
            return self._path(path).exists()

        return await run_in_executor(check)

    async def realpath(self, path: str) -> str:
        def resolve() -> str:
            self._ensure_alive()
            return os.path.realpath(self._path(path))

        return await run_in_executor(resolve)

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        validate_timeout(timeout)
        merged_env = {**self._env, **(env or {})}
        if isinstance(command, str):
            if not shell:
                raise TypeError('a string command requires shell=True; pass an argv sequence otherwise')
        elif shell:
            raise TypeError('an argv sequence cannot be combined with shell=True; pass a single command string')

        working_dir = await self._get_working_dir()
        # Directory preparation is not command time; process startup still counts, so a
        # process spawned after the deadline is terminated by the existing cleanup path.
        absolute_deadline = None if timeout is None else anyio.current_time() + timeout
        process: anyio.abc.Process | None = None

        async def spawn() -> None:
            # Assigned here, not returned, so the cleanup below reaches a process that finished
            # starting after the caller was cancelled.
            nonlocal process
            process = await anyio.open_process(
                command,
                cwd=working_dir,
                env=merged_env,
                stdin=DEVNULL,
                stdout=PIPE,
                stderr=PIPE,
                start_new_session=True,
            )

        try:
            # Startup has its own finite bound, capped by the command deadline when supplied.
            spawn_deadline = min(anyio.current_time() + _SPAWN_GRACE, absolute_deadline or float('inf'))
            await _shielded(spawn(), spawn_deadline)
        except (FileNotFoundError, PermissionError) as error:
            if isinstance(error, FileNotFoundError):
                # A deleted workspace directory raises `WorkspaceUnavailableError` here.
                await self._get_working_dir()
            # Like `sh`, a program that is missing (127) or not executable (126) is a normal result.
            # Only the program itself: an unusable working directory raises the same error types and must still raise.
            if isinstance(command, str) or error.filename != command[0]:
                raise
            missing = isinstance(error, FileNotFoundError)
            return CommandResult(
                exit_code=127 if missing else 126,
                stdout='',
                stderr=f'{command[0]}: {"command not found" if missing else "Permission denied"}\n',
            )
        except BaseException as error:
            if process is not None:
                await self._terminate(process)
            if (
                isinstance(error, TimeoutError)
                and absolute_deadline is not None
                and anyio.current_time() >= absolute_deadline
            ):
                raise WorkspaceTimeoutError(f'command timed out after {timeout:g}s during startup') from error
            raise
        running_process = process
        assert running_process is not None

        stdout_buffer = bytearray()
        stderr_buffer = bytearray()

        try:
            exit_code = await self._wait_and_collect_output(
                running_process, stdout_buffer, stderr_buffer, absolute_deadline
            )
            await _shielded(self._close(running_process), anyio.current_time() + _REAP_GRACE)
        except BaseException as error:
            denial = await self._terminate(running_process)
            # Without a command timeout, a `TimeoutError` is only a stalled reap's grace period.
            if isinstance(error, TimeoutError) and timeout is not None:
                stdout = stdout_buffer.decode('utf-8', errors='replace')
                stderr = stderr_buffer.decode('utf-8', errors='replace')
                if denial is not None:
                    raise WorkspaceTimeoutError(
                        f'command timed out after {timeout:g}s; killing its process group was '
                        'denied, so only the direct child was killed and grandchildren may survive',
                        stdout=stdout,
                        stderr=stderr,
                    ) from denial
                raise WorkspaceTimeoutError(
                    f'command timed out after {timeout:g}s and was killed',
                    stdout=stdout,
                    stderr=stderr,
                ) from error
            raise
        # A process can still finish after its working directory is deleted; don't report
        # that as an ordinary exit from a live workspace.
        await self._get_working_dir()
        return CommandResult(
            exit_code=exit_code,
            stdout=stdout_buffer.decode('utf-8', errors='replace'),
            stderr=stderr_buffer.decode('utf-8', errors='replace'),
        )

    async def _wait_and_collect_output(
        self,
        process: anyio.abc.Process,
        stdout_buffer: bytearray,
        stderr_buffer: bytearray,
        absolute_deadline: float | None,
    ) -> int:
        """Wait for the command to exit while collecting both pipes; raise `TimeoutError` at the deadline."""
        stdout_pipe, stderr_pipe = process.stdout, process.stderr
        assert stdout_pipe is not None and stderr_pipe is not None
        remaining = None if absolute_deadline is None else absolute_deadline - anyio.current_time()
        exit_code: int | None = None
        overflowed = False

        async def collect(stream: anyio.abc.ByteReceiveStream, buffer: bytearray, other_buffer: bytearray) -> None:
            # Output is collected and returned whole in `CommandResult`; it is not streamed to the caller.
            nonlocal overflowed
            async for chunk in stream:
                buffer.extend(chunk)
                if len(buffer) + len(other_buffer) > _MAX_CAPTURE_BYTES:
                    # Stop the group rather than raise inside it, so the error below leaves this
                    # method as a plain `WorkspaceError` instead of an `ExceptionGroup`.
                    overflowed = True
                    tg.cancel_scope.cancel()
                    return

        with anyio.move_on_after(remaining):
            # Both pipes must be drained at once or a full unread pipe can block the command.
            async with anyio.create_task_group() as tg:
                tg.start_soon(collect, stdout_pipe, stdout_buffer, stderr_buffer)
                tg.start_soon(collect, stderr_pipe, stderr_buffer, stdout_buffer)
                exit_code = await self._wait_for_exit(process)
                # The command has exited; keep reading for a short grace period in case a background
                # child still holds a pipe open. The deadline above still bounds the grace period.
                tg.cancel_scope.deadline = anyio.current_time() + _OUTPUT_DRAIN_GRACE
        if overflowed:
            # Keep a small preview from each stream without retaining the entire oversized capture.
            raise WorkspaceOutputLimitError(
                "local workspace output exceeded 10 MiB safety limit; redirect the command's "
                'output to a file and read part of it instead',
                limit=_MAX_CAPTURE_BYTES,
                stdout=stdout_buffer[: 64 * 1024].decode('utf-8', errors='replace'),
                stderr=stderr_buffer[: 64 * 1024].decode('utf-8', errors='replace'),
            )
        if exit_code is None:
            raise TimeoutError
        return exit_code

    async def _wait_for_exit(self, process: anyio.abc.Process) -> int:
        """Return the exit code as soon as the command itself exits, whatever its children do with the pipes."""
        if not _waits_for_pipes():
            exit_code = await process.wait()
        else:
            while (exit_code := process.returncode) is None:
                await anyio.sleep(_EXIT_POLL_INTERVAL)
        # Subprocess APIs report -N, whereas shells report signal deaths as 128+N.
        return 128 - exit_code if exit_code < 0 else exit_code

    async def _close(self, process: anyio.abc.Process) -> None:
        """Release the process's pipes and reap it, without waiting for the pipes to close."""
        if _waits_for_pipes():
            # What anyio 4.15's `aclose()` does; the transport is only reachable through asyncio's
            # private `Process._transport`.
            transport = cast(
                asyncio.SubprocessTransport,
                process._process._transport,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
            )
            for fd in (1, 2):
                pipe = transport.get_pipe_transport(fd)
                assert pipe is not None  # `run` always spawns with `stdout=PIPE, stderr=PIPE`
                pipe.close()
        await process.aclose()

    async def _terminate(self, process: anyio.abc.Process) -> PermissionError | None:
        """Kill the process group and reap it; return a group-kill denial."""
        denial = None
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError as error:
            denial = error
            with suppress(ProcessLookupError):
                process.kill()
        # Teardown must not hold a cancelled caller indefinitely if a pipe or reap stalls.
        with suppress(TimeoutError):
            await _shielded(self._close(process), anyio.current_time() + _REAP_GRACE)
        return denial
