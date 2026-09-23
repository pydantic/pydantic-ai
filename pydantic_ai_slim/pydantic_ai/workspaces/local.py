"""A local implementation of the [workspace backend protocol][pydantic_ai.workspaces.WorkspaceBackend].

[`LocalWorkspaceBackend`][pydantic_ai.workspaces.LocalWorkspaceBackend] runs commands as plain host
subprocesses — it **isolates nothing**.
"""

from __future__ import annotations as _annotations

import asyncio
import os
import re
import shutil
import signal
from collections.abc import Awaitable, Mapping, Sequence
from importlib.metadata import version
from pathlib import Path
from subprocess import DEVNULL, PIPE
from typing import cast

import anyio
import sniffio
from typing_extensions import TypeVar

from pydantic_ai._utils import gather, run_in_executor

from .protocol import (
    CommandResult,
    FileEntry,
    SupportsCommands,
    SupportsFilesystem,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)

__all__ = ('LocalWorkspaceBackend',)

_MAX_CAPTURE_BYTES = 10 * 1024 * 1024
"""Ceiling on the combined stdout and stderr a single command may produce."""

_OUTPUT_DRAIN_GRACE = 2.0
"""How long to keep reading a command's pipes after the direct child has exited."""

T = TypeVar('T')

_ANYIO_VERSION_RE = re.compile(r'(\d+)\.(\d+)')
_match = _ANYIO_VERSION_RE.match(version('anyio'))
_PROCESS_WAIT_WAITS_FOR_PIPES = _match is not None and tuple(map(int, _match.groups())) < (4, 15)
"""Whether `anyio.abc.Process.wait()` on the asyncio backend also waits for the output pipes to close.

Before anyio 4.15.0 it delegated to asyncio's `Process.wait()`, which only returns once every
redirected pipe has disconnected (https://github.com/agronholm/anyio/issues/1174), so a command
that left a background child holding stdout open (`sleep 30 & echo done`) would hang `run()`
until that child exited, and `Process.aclose()` would hang the same way. On those versions
`_wait_for_exit` polls `returncode`, which asyncio sets the moment the child exits, and `_close`
releases the inherited pipe descriptors itself before `aclose()`.
"""
_EXIT_POLL_INTERVAL = 0.005


async def _shielded(awaitable: Awaitable[T]) -> T:
    """Wait for work that must finish even if the caller is cancelled.

    A plain shielded scope is not enough because `asyncio.Task.cancel()` and `asyncio.timeout()`
    still cancel the task doing the await. A task-group child is cancelled only by the group,
    which honors the child's shield.
    """

    async def run() -> T:
        result: list[T] = []
        with anyio.CancelScope(shield=True):
            result.append(await awaitable)
        return result[0]

    return (await gather(run()))[0]


class LocalWorkspaceBackend(WorkspaceBackend, SupportsCommands, SupportsFilesystem):
    """Run commands as subprocesses on this machine and use its filesystem.

    This isolates nothing and is not a jail. `working_dir` is only where commands start and what
    relative workspace paths resolve against: absolute paths, `..`, and commands reach anywhere on
    the host that this process can. Use it for trusted local work, tests, and development; run
    untrusted code in a container or VM through a provider workspace. Commands inherit only `PATH`,
    `HOME`, `LANG`, and `TMPDIR` when present, plus variables supplied through `env`.

    It supports POSIX platforms only. A command that calls `setsid` can move its own processes
    outside the process group that this workspace kills on cancellation or timeout.

    The directory is the environment: [`ref`][pydantic_ai.workspaces.LocalWorkspaceBackend.ref]
    names it from construction, and the first operation that needs it raises
    [`WorkspaceUnavailableError`][pydantic_ai.workspaces.WorkspaceUnavailableError] when it does
    not exist, the way a provider backend does for a sandbox that is gone. Nothing creates it.

    Args:
        working_dir: The default working directory for commands and the base for relative
            workspace paths. It must be absolute; a leading `~` is expanded to the user's home
            directory. It is not a confinement boundary. The caller creates and removes it. It is
            canonicalized on first use so
            [`working_dir()`][pydantic_ai.workspaces.WorkspaceBackend.working_dir] reports the
            directory commands actually run in.
    """

    def __init__(self, working_dir: str | Path):
        if os.name != 'posix':
            raise NotImplementedError(
                '`LocalWorkspaceBackend` only supports POSIX platforms at the moment: its timeout contract '
                'kills the whole process group. On other platforms, attach a container- or VM-based '
                'workspace instead.'
            )
        expanded = Path(working_dir).expanduser()
        if not expanded.is_absolute():
            raise ValueError(
                f'`working_dir` must be an absolute path or start with `~`, got {str(working_dir)!r}: a relative '
                "path would depend on the host process's working directory at some later moment. Make the "
                "intent explicit at the call site instead, e.g. `LocalWorkspaceBackend(Path.cwd() / 'work')`."
            )
        self._working_dir = expanded
        self._canonical_working_dir: Path | None = None
        self._ref = WorkspaceRef(provider='local', id=expanded.as_posix())

    @property
    def ref(self) -> WorkspaceRef:
        """`WorkspaceRef(provider='local', id=...)` naming `working_dir` as given, with `~` expanded.

        The directory is the environment, so this is the one backend whose ref precedes its first
        operation: it is available from construction and involves no I/O, so symlinks are not
        resolved. It says which directory on this host the workspace was configured with, and means
        nothing on another machine. Whether the directory exists is checked by the first operation,
        like [`working_dir()`][pydantic_ai.workspaces.WorkspaceBackend.working_dir], which raises
        [`WorkspaceUnavailableError`][pydantic_ai.workspaces.WorkspaceUnavailableError] when it
        is missing.
        """
        return self._ref

    async def _get_working_dir(self) -> Path:
        if self._canonical_working_dir is None:

            def resolve() -> Path:
                # Canonicalization keeps macOS `/var` symlinks and spellings such as `link/..` aligned
                # with the directory the kernel uses for the command's working directory.
                resolved = self._working_dir.resolve()
                if not resolved.is_dir():
                    raise WorkspaceUnavailableError(
                        f'local workspace directory {self._working_dir.as_posix()!r} does not exist; the '
                        'caller creates the directory before the run, and nothing recreates a removed one'
                    )
                return resolved

            # `resolve()` is idempotent, so concurrent first calls may safely compute it twice.
            self._canonical_working_dir = await run_in_executor(resolve)
        return self._canonical_working_dir

    async def working_dir(self) -> str:
        return str(await self._get_working_dir())

    @staticmethod
    def _path(path: str) -> Path:
        target = Path(path)
        if not target.is_absolute():
            raise ValueError(f'path must be absolute, got {path!r}')
        return target

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
        merged_env = {key: os.environ[key] for key in ('PATH', 'HOME', 'LANG', 'TMPDIR') if key in os.environ}
        if env is not None:
            merged_env.update(env)
        if isinstance(command, str):
            if not shell:
                raise TypeError('a string command requires shell=True; pass an argv sequence otherwise')
        elif shell:
            raise TypeError('an argv sequence cannot be combined with shell=True; pass a single command string')

        process: anyio.abc.Process | None = None

        async def spawn() -> anyio.abc.Process:
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
            return process

        try:
            # Store the result inside the shielded child so cleanup can reach a process whose caller
            # was cancelled while subprocess creation finished.
            running_process = await _shielded(spawn())
        except BaseException:
            if process is not None:
                await self._terminate(process)
            raise

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
                        timeout=timeout,
                    ) from denial
                raise WorkspaceTimeoutError(
                    f'command timed out after {timeout} seconds and was killed',
                    stdout=stdout,
                    stderr=stderr,
                    timeout=timeout,
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
                'output to a file and read a window of it with `read_file` instead'
            )
        if exit_code is None:
            raise TimeoutError
        return exit_code

    @staticmethod
    def _uses_pipe_bound_wait() -> bool:
        # Trio's `wait()` and `aclose()` never waited on the pipes; only asyncio (and uvloop) did.
        return _PROCESS_WAIT_WAITS_FOR_PIPES and sniffio.current_async_library() == 'asyncio'

    async def _wait_for_exit(self, process: anyio.abc.Process) -> int:
        """Return the exit code as soon as the command itself exits, whatever its children do with the pipes."""
        if not self._uses_pipe_bound_wait():
            return await process.wait()
        while (exit_code := process.returncode) is None:
            await anyio.sleep(_EXIT_POLL_INTERVAL)
        return exit_code

    async def _close(self, process: anyio.abc.Process) -> None:
        """Release the process's pipes and reap it, without waiting for the pipes to close."""
        if self._uses_pipe_bound_wait():
            # asyncio's `Process.wait()` (which old `aclose()` ends with) returns only once every
            # pipe has disconnected, so close our ends of the output pipes first, the way anyio
            # 4.15's `aclose()` does. The transport is reachable only through asyncio's private
            # `Process._transport`; this branch is dead on anyio >= 4.15.
            transport = cast(
                asyncio.SubprocessTransport,
                process._process._transport,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
            )
            for fd in (1, 2):
                # Both pipes exist: `run` always spawns with `stdout=PIPE, stderr=PIPE`.
                pipe = transport.get_pipe_transport(fd)
                assert pipe is not None
                pipe.close()
        await process.aclose()

    async def _terminate(self, process: anyio.abc.Process) -> PermissionError | None:
        async def terminate() -> PermissionError | None:
            denial: PermissionError | None = None
            try:
                self._kill(process)
            except PermissionError as error:
                denial = error
            finally:
                await self._close(process)
            return denial

        return await _shielded(terminate())

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
