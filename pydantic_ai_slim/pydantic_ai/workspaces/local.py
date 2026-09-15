"""A local implementation of the [workspace backend protocol][pydantic_ai.workspaces.WorkspaceBackend].

[`LocalWorkspace`][pydantic_ai.workspaces.LocalWorkspace] runs commands as plain host subprocesses —
it **isolates nothing**.
"""

# anyio 4.15.0 is the floor because `Process.wait()` returns when the command exits even if a
# background child still holds a pipe open (https://github.com/agronholm/anyio/issues/1174), and
# `Process.aclose()` releases the pipe descriptors that child inherited.

from __future__ import annotations as _annotations

import os
import shutil
import signal
from collections.abc import Awaitable, Mapping, Sequence
from pathlib import Path
from subprocess import DEVNULL, PIPE

import anyio
from typing_extensions import TypeVar

from pydantic_ai._utils import gather, run_in_executor

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

        process: anyio.abc.Process | None = None

        async def spawn() -> anyio.abc.Process:
            nonlocal process
            process = await anyio.open_process(
                command,
                cwd=cwd if cwd is not None else await self.root,
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
            await running_process.aclose()
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
                exit_code = await process.wait()
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

    async def _terminate(self, process: anyio.abc.Process) -> PermissionError | None:
        async def terminate() -> PermissionError | None:
            denial: PermissionError | None = None
            try:
                self._kill(process)
            except PermissionError as error:
                denial = error
            finally:
                await process.aclose()
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
