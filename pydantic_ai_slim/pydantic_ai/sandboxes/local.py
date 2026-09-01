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
import uuid
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

import anyio
from typing_extensions import Self

from pydantic_ai._utils import run_in_executor

from .protocol import CommandResult, FileEntry, SandboxCommand, SandboxTimeoutError

if TYPE_CHECKING:
    from .protocol import SandboxBackend, SupportsFilesystem

__all__ = ('LocalSandbox',)


class _LocalFilesystem:
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
            for child in sorted(self._path(path).iterdir()):
                is_dir = child.is_dir()
                try:
                    # stat, not lstat: a symlinked file reports its target's size, matching `stat()`.
                    size = None if is_dir else child.stat().st_size
                except OSError:
                    # A broken symlink in the directory must not fail the whole listing.
                    size = None
                entries.append(FileEntry(name=child.name, path=str(child), is_dir=is_dir, size=size))
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
    `start()` is not implemented (use `run(timeout=...)` to bound commands).

    Deliberately no base class: it conforms to the protocol structurally, like any
    third-party backend would.

    Args:
        root: The working directory commands run in and relative paths resolve against; must
            be an absolute path (a relative one would silently depend on the host process's
            working directory). Defaults to a fresh temporary directory, created on first use
            and removed again when the sandbox is used as an async context manager. A
            caller-supplied `root` is never removed, and is canonicalized (symlinks resolved)
            on first use, so
            [`working_dir()`][pydantic_ai.sandboxes.SandboxBackend.working_dir] reports the
            directory commands actually run in.
    """

    provider = 'local'

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
        self._root: Path | None = None
        self._root_lock = anyio.Lock()
        self._id = f'local-{uuid.uuid4().hex}'
        self.fs = _LocalFilesystem()

    @property
    def sandbox_id(self) -> str:
        return self._id

    async def _root_path(self) -> Path:
        # The default temp root is created lazily, so a constructed-but-unused sandbox doesn't
        # leak a directory, and off the event loop, since `mkdtemp` and `resolve` are blocking
        # syscalls. The lock is what makes the two safe together: without it, two concurrent
        # first uses would each create a directory and one would leak.
        async with self._root_lock:
            if self._root is None:
                if self._given_root is None:
                    self._root = await run_in_executor(
                        lambda: Path(tempfile.mkdtemp(prefix='pydantic-ai-sandbox-')).resolve()
                    )
                else:
                    self._root = await run_in_executor(self._given_root.resolve)
            return self._root

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        # Under the root lock: an unlocked clear would race `_root_path()` — a first use
        # blocked on the lock could otherwise recreate a root mid-teardown that nothing
        # would ever remove.
        async with self._root_lock:
            if self._owns_root and self._root is not None:
                # Reset first so a reused sandbox lazily creates a fresh root instead of
                # resurrecting the deleted path.
                root, self._root = self._root, None
                try:
                    await run_in_executor(shutil.rmtree, root)
                except FileNotFoundError:
                    # A command or `fs.remove()` may have deleted the root already; exiting
                    # must not raise (it would mask the exception that ended the block).
                    pass

    async def working_dir(self) -> str:
        return str(await self._root_path())

    async def run(
        self,
        command: SandboxCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        if cwd is not None and not os.path.isabs(cwd):
            raise ValueError(
                f'cwd must be an absolute path, got {cwd!r}: a relative cwd would resolve against '
                "the host process's working directory, not the sandbox root"
            )
        # `env` overlays the host environment rather than replacing it, so passing one
        # variable doesn't strip PATH from the child.
        merged_env = {**os.environ, **env} if env is not None else None
        if shell:
            if not isinstance(command, str):
                raise TypeError('an argv sequence cannot be combined with shell=True; pass a single command string')
            spawn_coroutine = asyncio.create_subprocess_shell(
                command,
                cwd=cwd or await self._root_path(),
                env=merged_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # Each command leads its own process group, so the timeout kill takes out
                # the whole tree — killing only `sh` would leave its children running.
                start_new_session=True,
            )
        else:
            if isinstance(command, str):
                raise TypeError('a string command requires shell=True; pass an argv sequence otherwise')
            spawn_coroutine = asyncio.create_subprocess_exec(
                *command,
                cwd=cwd or await self._root_path(),
                env=merged_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )

        # If we're cancelled mid-spawn, the child may already be forked with nobody holding a
        # handle to kill its process group. Shield the spawn so we always get the handle back,
        # and return spawn failures instead of raising them: on Python 3.14, `shield` reports
        # an abandoned future's exception to the event loop's exception handler.
        async def guarded_spawn() -> asyncio.subprocess.Process | Exception:
            try:
                return await spawn_coroutine
            except Exception as error:
                return error

        spawn = asyncio.ensure_future(guarded_spawn())
        try:
            outcome = await asyncio.shield(spawn)
        except asyncio.CancelledError:
            spawn.add_done_callback(self._kill_abandoned_spawn)
            raise
        if isinstance(outcome, Exception):
            raise outcome
        process = outcome
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
        except (TimeoutError, asyncio.TimeoutError) as error:  # asyncio's is distinct on 3.10
            # The contract: a timeout kills the command first, then raises SandboxTimeoutError —
            # even when a hardened host denies the group kill, in which case the
            # denial rides along as the cause instead of replacing the promised type.
            denial = await self._kill_and_reap(process)
            if denial is not None:
                raise SandboxTimeoutError(
                    f'command timed out after {timeout} seconds; killing its process group was '
                    'denied, so only the direct child was killed and grandchildren may survive',
                    timeout=timeout,
                ) from denial
            raise SandboxTimeoutError(
                f'command timed out after {timeout} seconds and was killed', timeout=timeout
            ) from error
        except BaseException:
            # Cancellation or any other failure while the pipes were open: kill the group,
            # but let the in-flight exception keep propagating — replacing a cancellation
            # with a kill-denial error would break the caller's cancel scope. `returncode`
            # alone can't tell us the group is gone: a shell can exit while a background
            # child keeps the pipes open.
            await self._kill_and_reap(process)
            raise
        assert process.returncode is not None
        return CommandResult(
            exit_code=process.returncode,
            stdout=stdout.decode('utf-8', errors='replace'),
            stderr=stderr.decode('utf-8', errors='replace'),
        )

    async def _kill_and_reap(self, process: asyncio.subprocess.Process) -> PermissionError | None:
        """Kill the group and reap the direct child, reporting a denied group kill.

        Reported instead of raised, so the caller decides which exception its contract owes.
        """
        try:
            self._kill(process)
        except PermissionError as error:
            return error
        finally:
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
    _conforms: SandboxBackend = LocalSandbox()
    _filesystem_backend_conforms: SupportsFilesystem = LocalSandbox()
