"""A local implementation of the [workspace backend protocol][pydantic_ai.workspaces.WorkspaceBackend].

[`LocalWorkspace`][pydantic_ai.workspaces.LocalWorkspace] runs commands as plain host subprocesses —
it **isolates nothing**.
"""

from __future__ import annotations as _annotations

import asyncio
import os
import shutil
import signal
import time
from collections.abc import Awaitable, Mapping, Sequence
from pathlib import Path

import anyio
from typing_extensions import TypeVar

from pydantic_ai._utils import cancel_and_drain, gather, run_in_executor

from .protocol import (
    CommandResult,
    FileEntry,
    SupportsFilesystem,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceError,
    WorkspaceTimeoutError,
)

__all__ = ('LocalWorkspace',)

_MAX_CAPTURE_BYTES = 10 * 1024 * 1024
"""Ceiling on the combined stdout and stderr a single command may produce."""

_READ_CHUNK_BYTES = 64 * 1024
"""Bytes requested per pipe read."""

_CHILD_POLL_INTERVAL = 0.01
"""How often to check whether the command process has exited."""

_OUTPUT_DRAIN_GRACE = 2.0
"""How long to keep reading a command's pipes after the direct child has exited."""

T = TypeVar('T')


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


class LocalWorkspace(WorkspaceBackend, SupportsFilesystem):
    """Run commands as subprocesses on this machine and use its filesystem.

    This isolates nothing. Use it for trusted local work, tests, and development; run untrusted
    code in a container or VM through a provider workspace. Commands inherit only `PATH`, `HOME`,
    `LANG`, and `TMPDIR` when present, plus variables supplied through `env`.

    It supports POSIX platforms only. A command that calls `setsid` can move its own processes
    outside the process group that this workspace kills on cancellation or timeout.

    Args:
        root: The absolute working directory for commands and relative workspace paths. The caller
            creates and removes it. It is canonicalized on first use so
            [`working_dir()`][pydantic_ai.workspaces.WorkspaceBackend.working_dir] reports the
            directory commands actually run in.
    """

    def __init__(self, root: str | Path):
        if os.name != 'posix':
            raise NotImplementedError(
                '`LocalWorkspace` only supports POSIX platforms at the moment: its timeout contract '
                'kills the whole process group. On other platforms, attach a container- or VM-based '
                'workspace instead.'
            )
        root = Path(root)
        if not root.is_absolute():
            raise ValueError(
                f'root must be an absolute path, got {str(root)!r}: a relative root would depend on '
                "the host process's working directory at some later moment. Make the intent explicit "
                "at the call site instead, e.g. `LocalWorkspace(Path.cwd() / 'work')`."
            )
        self._root = root
        self._resolved_root: Path | None = None

    @property
    def ref(self) -> None:
        return None

    @property
    def root(self) -> Awaitable[Path]:
        """The canonical directory commands run in, resolved on first use."""
        return self._get_root()

    async def _get_root(self) -> Path:
        if self._resolved_root is None:
            # Canonicalization keeps macOS `/var` symlinks and roots such as `link/..` aligned with
            # the directory the kernel uses for the command's working directory.
            # `resolve()` is idempotent, so concurrent first calls may safely compute it twice.
            self._resolved_root = await run_in_executor(self._root.resolve)
        return self._resolved_root

    async def working_dir(self) -> str:
        return str(await self.root)

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

    async def _spawn(
        self, command: WorkspaceCommand, cwd: str | None, env: Mapping[str, str]
    ) -> asyncio.subprocess.Process:
        """Start the command in its own process group."""
        process_cwd = cwd if cwd is not None else await self.root
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

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        # The deadline includes subprocess creation; if it passes there, the first exit poll detects it.
        deadline = None if timeout is None else time.monotonic() + timeout
        if cwd is not None and not Path(cwd).is_absolute():
            raise ValueError(
                f'cwd must be an absolute path, got {cwd!r}: a relative cwd would resolve against '
                "the host process's working directory, not the workspace root"
            )
        merged_env = {key: os.environ[key] for key in ('PATH', 'HOME', 'LANG', 'TMPDIR') if key in os.environ}
        if env is not None:
            merged_env.update(env)
        if isinstance(command, str):
            if not shell:
                raise TypeError('a string command requires shell=True; pass an argv sequence otherwise')
        elif shell:
            raise TypeError('an argv sequence cannot be combined with shell=True; pass a single command string')

        process: asyncio.subprocess.Process | None = None

        async def spawn() -> None:
            nonlocal process
            process = await self._spawn(command, cwd, merged_env)

        try:
            # Store the result inside the shielded child so cleanup can reach a process whose caller
            # was cancelled while subprocess creation finished.
            await _shielded(spawn())
        except BaseException:
            if process is not None:
                await self._kill_and_reap_and_close(process, [])
            raise
        assert process is not None

        stdout_buffer = bytearray()
        stderr_buffer = bytearray()
        reader_tasks: list[asyncio.Task[None]] = []

        try:
            stdout_pipe, stderr_pipe = process.stdout, process.stderr
            if stdout_pipe is None or stderr_pipe is None:  # pragma: no cover
                raise WorkspaceError('local workspace could not capture the command output pipes')
            # Both pipes must be drained at once or a full unread pipe can block the command.
            reader_tasks = [
                asyncio.create_task(self._collect_output(stdout_pipe, stdout_buffer, stderr_buffer)),
                asyncio.create_task(self._collect_output(stderr_pipe, stderr_buffer, stdout_buffer)),
            ]
            exit_code = await self._wait_for_direct_child(process, reader_tasks, deadline)
            await self._drain_output(reader_tasks, deadline)
            self._close_transport(process)
        except TimeoutError as error:
            denial = await self._kill_and_reap_and_close(process, reader_tasks)
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
        except BaseException:
            await self._kill_and_reap_and_close(process, reader_tasks)
            raise
        return CommandResult(
            exit_code=exit_code,
            stdout=stdout_buffer.decode('utf-8', errors='replace'),
            stderr=stderr_buffer.decode('utf-8', errors='replace'),
        )

    @staticmethod
    async def _collect_output(stream: asyncio.StreamReader, buffer: bytearray, other_buffer: bytearray) -> None:
        """Read one pipe into a buffer until its writer closes it.

        Output is collected and returned whole in `CommandResult`; it is not streamed to the caller.
        """
        while chunk := await stream.read(_READ_CHUNK_BYTES):
            buffer.extend(chunk)
            if len(buffer) + len(other_buffer) > _MAX_CAPTURE_BYTES:
                raise WorkspaceError(
                    "local workspace output exceeded 10 MiB safety limit; redirect the command's "
                    'output to a file and read a window of it with `read_file` instead'
                )

    @staticmethod
    async def _wait_for_direct_child(
        process: asyncio.subprocess.Process,
        reader_tasks: list[asyncio.Task[None]],
        deadline: float | None,
    ) -> int:
        """Wait for the direct child to exit and return its code.

        The direct child is the process this workspace started (the shell, or the program itself);
        it may start processes of its own, which are not waited for. Exit is polled because on
        CPython 3.11+ `Process.wait()` waits for every pipe to reach end-of-file, so a background
        process that inherited stdout can keep it blocked after the direct child exits
        (https://github.com/python/cpython/issues/119710). This was fixed on main in July 2026 and
        backported to Python 3.13 and 3.14 patch releases, and AnyIO 4.15.0 fixed the same behavior
        (https://github.com/agronholm/anyio/issues/1174), but this project supports Python 3.10+
        with AnyIO 4.7.0+.
        """
        while True:
            for task in reader_tasks:
                if task.done():
                    task.result()
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError
            if (returncode := process.returncode) is not None:
                return returncode
            await asyncio.sleep(_CHILD_POLL_INTERVAL if remaining is None else min(_CHILD_POLL_INTERVAL, remaining))

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
        try:
            return await _shielded(self._kill_and_reap(process))
        finally:
            await cancel_and_drain(*reader_tasks)
            self._close_transport(process)

    @staticmethod
    def _close_transport(process: asyncio.subprocess.Process) -> None:
        """Release pipe descriptors that descendants may still hold open."""
        process._transport.close()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

    async def _kill_and_reap(self, process: asyncio.subprocess.Process) -> PermissionError | None:
        """Kill the group and reap the direct child, reporting a denied group kill.

        The caller decides whether a denied group kill should replace the current exception.
        """
        try:
            self._kill(process)
        except PermissionError as error:
            return error
        finally:
            self._close_transport(process)
            await process.wait()
        return None

    @staticmethod
    def _kill(process: asyncio.subprocess.Process) -> None:
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
