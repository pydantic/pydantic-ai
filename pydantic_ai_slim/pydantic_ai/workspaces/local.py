"""A local implementation of the [workspace backend protocol][pydantic_ai.workspaces.WorkspaceBackend].

[`LocalWorkspaceBackend`][pydantic_ai.workspaces.LocalWorkspaceBackend] runs commands as plain host
subprocesses — it **isolates nothing**.
"""

from __future__ import annotations as _annotations

import asyncio
import os
import shutil
import signal
from collections.abc import Awaitable, Mapping, Sequence
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
    WorkspaceError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)

__all__ = ('LocalWorkspaceBackend',)

# Not secrets, and without them commands miss the host's tools and the user's configuration.
_INHERITED_ENV = ('PATH', 'HOME')
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


def _waits_for_pipes() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:  # pragma: no cover - only reached on Trio, which CI does not run
        return False
    return _ANYIO_WAITS_FOR_PIPES


async def _shielded(awaitable: Awaitable[None]) -> None:
    """Await to completion even if the caller is cancelled meanwhile; the cancellation is raised after.

    Runs in a task-group child, because a shielded scope alone does not stop asyncio's `Task.cancel()`.
    """

    async def child() -> None:
        with anyio.CancelScope(shield=True):
            await awaitable

    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(child)
    except BaseExceptionGroup as group:
        # The child's own error, as a single exception rather than a group.
        error = group.exceptions[0]
        error.__suppress_context__ = True
        raise error


class LocalWorkspaceBackend(WorkspaceBackend, SupportsCommands, SupportsFilesystem, SupportsRealpath):
    """Run commands as subprocesses on this machine and use its filesystem (POSIX only).

    This isolates nothing: commands and absolute paths reach anywhere this process can. Commands
    inherit only `PATH` and `HOME`, so they find the host's tools and config but none of its secrets.
    The directory is the environment: its [`ref`][pydantic_ai.workspaces.LocalWorkspaceBackend.ref]
    exists from construction, and the first operation raises
    [`WorkspaceUnavailableError`][pydantic_ai.workspaces.WorkspaceUnavailableError] if it is missing.

    Args:
        working_dir: Where commands start and relative paths resolve; `~` is expanded and a relative
            path is taken from the current directory. The caller creates and removes it.
        env: Environment variables for every command, on top of `PATH` and `HOME`; the per-call `env` goes on top.
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
        self._ref = WorkspaceRef(provider='local', id=absolute.as_posix())
        self._env = {name: os.environ[name] for name in _INHERITED_ENV if name in os.environ} | dict(env or {})

    @property
    def ref(self) -> WorkspaceRef:
        """`WorkspaceRef(provider='local', id=<absolute working_dir>)`, available from construction."""
        return self._ref

    async def _get_working_dir(self) -> Path:
        # Symlinks resolved (macOS `/var`, `link/..`), so paths match what the kernel reports to commands.
        if self._resolved_working_dir is None:

            def resolve() -> Path:
                resolved = self._working_dir.resolve()
                if not resolved.is_dir():
                    raise WorkspaceUnavailableError(
                        f'local workspace directory {self._working_dir.as_posix()!r} does not exist; the '
                        'caller creates the directory before the run, and nothing recreates a removed one'
                    )
                return resolved

            self._resolved_working_dir = await run_in_executor(resolve)
        return self._resolved_working_dir

    async def working_dir(self) -> str:
        return str(await self._get_working_dir())

    @staticmethod
    def _path(path: str) -> Path:
        target = Path(path)
        if not target.is_absolute():
            raise ValueError(f'path must be absolute, got {path!r}')
        return target

    # File operations run in a thread: filesystem calls block, and must not stall the event loop.

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

    async def realpath(self, path: str) -> str:
        return await run_in_executor(os.path.realpath, self._path(path))

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        absolute_deadline = None if timeout is None else anyio.current_time() + timeout
        if cwd is not None and not Path(cwd).is_absolute():
            raise ValueError(
                f'cwd must be an absolute path, got {cwd!r}: a relative cwd would resolve against '
                "the host process's working directory, not the workspace's"
            )
        merged_env = {**self._env, **(env or {})}
        if isinstance(command, str):
            if not shell:
                raise TypeError('a string command requires shell=True; pass an argv sequence otherwise')
        elif shell:
            raise TypeError('an argv sequence cannot be combined with shell=True; pass a single command string')

        process: anyio.abc.Process | None = None

        async def spawn() -> None:
            # Assigned here, not returned, so the cleanup below reaches a process that finished
            # starting after the caller was cancelled.
            nonlocal process
            process = await anyio.open_process(
                command,
                cwd=cwd if cwd is not None else await self._get_working_dir(),
                env=merged_env,
                stdin=DEVNULL,
                stdout=PIPE,
                stderr=PIPE,
                start_new_session=True,
            )

        try:
            await _shielded(spawn())
        except BaseException:
            if process is not None:
                await self._terminate(process)
            raise
        running_process = process
        assert running_process is not None

        stdout_buffer = bytearray()
        stderr_buffer = bytearray()

        try:
            exit_code = await self._wait_and_collect_output(
                running_process, stdout_buffer, stderr_buffer, absolute_deadline
            )
            await self._close(running_process)
        except BaseException as error:
            denial = await self._terminate(running_process)
            if isinstance(error, TimeoutError):
                stdout = stdout_buffer.decode('utf-8', errors='replace')
                stderr = stderr_buffer.decode('utf-8', errors='replace')
                if denial is not None:
                    raise WorkspaceTimeoutError(
                        f'command timed out after {timeout} seconds; killing its process group was '
                        'denied, so only the direct child was killed and grandchildren may survive',
                        stdout=stdout,
                        stderr=stderr,
                    ) from denial
                raise WorkspaceTimeoutError(
                    f'command timed out after {timeout} seconds and was killed',
                    stdout=stdout,
                    stderr=stderr,
                ) from error
            raise
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
        if remaining is not None and remaining <= 0:
            raise TimeoutError
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
            raise WorkspaceError(
                "local workspace output exceeded 10 MiB safety limit; redirect the command's "
                'output to a file and read part of it instead'
            )
        if exit_code is None:
            raise TimeoutError
        return exit_code

    async def _wait_for_exit(self, process: anyio.abc.Process) -> int:
        """Return the exit code as soon as the command itself exits, whatever its children do with the pipes."""
        if not _waits_for_pipes():
            return await process.wait()
        while (exit_code := process.returncode) is None:
            await anyio.sleep(_EXIT_POLL_INTERVAL)
        return exit_code

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
        """Kill the process group and reap it; return the error if killing the group was denied."""
        try:
            self._kill(process)
        except PermissionError as denial:
            return denial
        finally:
            await _shielded(self._close(process))
        return None

    @staticmethod
    def _kill(process: anyio.abc.Process) -> None:
        # If the group kill is denied, kill the direct child but still raise because its children may survive.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            try:
                process.kill()
            finally:
                raise
