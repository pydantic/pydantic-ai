"""A Windows implementation of the [workspace backend protocol][pydantic_ai.workspaces.WorkspaceBackend].

[`WindowsWorkspaceBackend`][pydantic_ai.workspaces.WindowsWorkspaceBackend] runs commands as plain host
subprocesses in PowerShell -- it **isolates nothing**.
"""

from __future__ import annotations as _annotations

import base64
import errno
import functools
import ntpath
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from subprocess import DEVNULL, PIPE

import anyio
import anyio.abc

from pydantic_ai._utils import run_in_executor

from .local import _HostWorkspaceBackend  # pyright: ignore[reportPrivateUsage]
from .protocol import CommandResult, WorkspaceCommand, WorkspacePlatform

__all__ = ('WindowsWorkspaceBackend',)

_INHERITED_ENV = (
    'PATH',
    'PATHEXT',
    'SYSTEMROOT',
    'SYSTEMDRIVE',
    'WINDIR',
    'COMSPEC',
    'TEMP',
    'TMP',
    'USERPROFILE',
    'HOMEDRIVE',
    'HOMEPATH',
    'APPDATA',
    'LOCALAPPDATA',
    'PROGRAMDATA',
    'PROGRAMFILES',
    'PROGRAMFILES(X86)',
    'PROGRAMW6432',
    'COMMONPROGRAMFILES',
    'PSMODULEPATH',
    'NUMBER_OF_PROCESSORS',
    'PROCESSOR_ARCHITECTURE',
)
"""Not secrets, and without them PowerShell, `git` and most tools fail to start or miss the user's configuration."""

_MAX_COMMAND_LINE = 32_767
"""Characters `CreateProcess` accepts in one command line, the program's own name included."""

_PREAMBLE = (
    "$ProgressPreference = 'SilentlyContinue'\n"
    'try { [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false) } catch {}\n'
    '$OutputEncoding = [Text.UTF8Encoding]::new($false)\n'
)
"""Write output as UTF-8, which `run` decodes, and keep progress bars out of stderr."""

_EPILOGUE = '\nif ($?) { exit 0 } elseif ($LASTEXITCODE) { exit $LASTEXITCODE } else { exit 1 }\n'
"""Exit with the last native command's code, as `sh` does; PowerShell would otherwise turn any failure into 1."""

_POWERSHELL_FLAGS = ('-NoLogo', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-OutputFormat', 'Text')

_PROCESS_SET_QUOTA = 0x0100
_PROCESS_TERMINATE = 0x0001

# Windows-only constants, read where they exist so this module imports, and type-checks, everywhere.
_O_BINARY: int = os.O_BINARY if sys.platform == 'win32' else 0
_CREATE_NEW_PROCESS_GROUP: int = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0


def _on_windows() -> bool:
    return sys.platform == 'win32'


def _powershell_command(executable: str, script: str) -> list[str]:
    """The argv that runs `script` in PowerShell, exiting with its last native command's code.

    The script is passed as `-EncodedCommand`, so no quoting rule of `CreateProcess` or PowerShell
    applies to it. Raises `OSError(E2BIG)` when the encoded script is too long for one command line.
    """
    encoded = base64.b64encode(f'{_PREAMBLE}{script}{_EPILOGUE}'.encode('utf-16-le')).decode('ascii')
    argv = [executable, *_POWERSHELL_FLAGS, '-EncodedCommand', encoded]
    if len(subprocess.list2cmdline(argv)) >= _MAX_COMMAND_LINE:
        raise OSError(errno.E2BIG, 'the PowerShell command is too long for one Windows command line')
    return argv


class _JobObjects:
    """The Windows Job Objects that let a timed-out command's whole process tree be ended at once."""

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        # `WinDLL` for `kernel32`'s `stdcall` convention, which 32-bit Windows needs. Read with `getattr`
        # because it exists only on Windows; it is a `CDLL` subclass.
        win_dll: type[ctypes.CDLL] = getattr(ctypes, 'WinDLL')
        kernel32 = win_dll('kernel32', use_last_error=True)
        self._create = kernel32.CreateJobObjectW
        self._create.restype = wintypes.HANDLE
        self._create.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
        self._open_process = kernel32.OpenProcess
        self._open_process.restype = wintypes.HANDLE
        self._open_process.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        self._assign = kernel32.AssignProcessToJobObject
        self._assign.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        self._terminate = kernel32.TerminateJobObject
        self._terminate.argtypes = (wintypes.HANDLE, wintypes.UINT)
        self._close = kernel32.CloseHandle
        self._close.argtypes = (wintypes.HANDLE,)

    def assign(self, pid: int) -> int | None:
        """A new Job Object holding process `pid` and every process it starts, or `None` if Windows refused."""
        job = self._create(None, None)
        if not job:  # pragma: no cover - Windows refuses only when out of resources
            return None
        process = self._open_process(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid)
        try:
            if process and self._assign(job, process):
                return job
        finally:
            if process:
                self._close(process)
        self._close(job)  # pragma: no cover - the command exited before it could be assigned
        return None  # pragma: no cover

    def terminate(self, job: int) -> None:
        self._terminate(job, 1)

    def close(self, job: int) -> None:
        self._close(job)


def _which(name: str, env: Mapping[str, str]) -> str | None:
    """`name` in an absolute directory on `env`'s `PATH`, trying each `PATHEXT` extension when it has none.

    Unlike `shutil.which`, never in the current directory: from there, a checkout's `rg.exe` would run in
    place of the real one.
    """
    extensions = [''] if ntpath.splitext(name)[1] else env.get('PATHEXT', '.COM;.EXE;.BAT;.CMD').split(';')
    for directory in env.get('PATH', '').split(';'):
        if ntpath.splitdrive(directory)[0] and ntpath.isabs(directory):
            for extension in extensions:
                if os.path.isfile(candidate := ntpath.join(directory, name + extension)):
                    return candidate
    return None


def _resolve_program(argv: Sequence[str], env: Mapping[str, str]) -> list[str]:
    """`argv` with a bare program name replaced by its path on `PATH`.

    `CreateProcess` looks for a bare name in this process's current directory before `PATH`, which
    `execvp` on POSIX does not. Raises `FileNotFoundError` when it is not on `PATH`.
    """
    program, *args = argv
    if ntpath.basename(program) != program or ntpath.splitdrive(program)[0]:
        return [program, *args]
    if (path := _which(program, env)) is None:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), program)
    return [path, *args]


@functools.cache
def _job_objects() -> _JobObjects:
    """Loaded on the first command, so building a backend does no I/O."""
    return _JobObjects()


class WindowsWorkspaceBackend(_HostWorkspaceBackend):
    r"""Run commands as subprocesses on this Windows machine, in PowerShell, and use its filesystem.

    The Windows counterpart of [`LocalWorkspaceBackend`][pydantic_ai.workspaces.LocalWorkspaceBackend], and
    like it, this isolates nothing: commands and absolute paths reach anywhere this process can. Paths are
    Windows paths (`C:\Users\me\project`), and a `shell=True` command runs in PowerShell: `pwsh` when it
    is on `PATH`, else Windows PowerShell. Its exit code is the last native command's, as in `sh`.

    Each command runs in a Windows Job Object, so a timeout or cancellation ends the processes it started
    along with it. A background job started with `Start-Process` stays in that Job Object: it outlives a
    command that returns, but not one that times out. Commands inherit only the variables Windows tools need
    to start (`PATH`, `PATHEXT`, `SYSTEMROOT`, `TEMP`, `USERPROFILE`, `APPDATA` and the like), not arbitrary secrets.

    Args:
        working_dir: Where commands start and relative paths resolve; `~` is expanded and a relative
            path is taken from the current directory. The caller creates and removes it.
        env: Environment variables for every command, on top of the inherited ones; the per-call `env` goes on top.
        powershell: The PowerShell executable; by default `pwsh` when it is on `PATH`, else `powershell.exe`.
    """

    _provider = 'windows'
    _inherited_env = _INHERITED_ENV

    def __init__(self, working_dir: str | Path, *, env: Mapping[str, str] | None = None, powershell: str | None = None):
        if not _on_windows():
            raise NotImplementedError(
                '`WindowsWorkspaceBackend` only runs on Windows; use `LocalWorkspaceBackend` on POSIX systems.'
            )
        super().__init__(working_dir, env=env)
        self._powershell = powershell
        # Keyed by PID: each command's Job Object, from its start until it is reaped.
        self._jobs: dict[int, int] = {}

    @property
    def platform(self) -> WorkspacePlatform:
        """Always `'windows'`: commands run in PowerShell, and paths are Windows paths."""
        return 'windows'

    def _executable(self, env: Mapping[str, str]) -> str:
        if self._powershell is None:
            self._powershell = _which('pwsh', env) or 'powershell.exe'
        return self._powershell

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        cwd: str | None = None,
    ) -> CommandResult:
        """Run `command`, a PowerShell script with `shell=True` or an argv, starting in `cwd` or the working directory."""
        return await self._run(command, shell=shell, env=env, timeout=timeout, cwd=cwd)

    async def _spawn(self, command: WorkspaceCommand, working_dir: Path, env: Mapping[str, str]) -> anyio.abc.Process:
        if isinstance(command, str):
            # Both search `PATH` on disk, so not on the event loop.
            command = _powershell_command(await run_in_executor(self._executable, env), command)
        command = await run_in_executor(_resolve_program, command, env)
        process = await anyio.open_process(
            command,
            cwd=working_dir,
            env=env,
            stdin=DEVNULL,
            stdout=PIPE,
            stderr=PIPE,
            creationflags=_CREATE_NEW_PROCESS_GROUP,
        )
        if (job := _job_objects().assign(process.pid)) is not None:
            self._jobs[process.pid] = job
        return process

    async def _program_error_result(
        self, command: WorkspaceCommand, error: FileNotFoundError | PermissionError
    ) -> CommandResult | None:
        # `CreateProcess` names no file when the program is missing: the working directory was checked
        # already, so the program is what is missing.
        if error.filename is None and not isinstance(command, str):
            error.filename = command[0]
        return await super()._program_error_result(command, error)

    def _kill_tree(self, process: anyio.abc.Process) -> PermissionError | None:
        job = self._jobs.get(process.pid)
        if job is not None:
            _job_objects().terminate(job)
        else:  # pragma: no cover - only when `assign` failed
            try:
                process.kill()
            except ProcessLookupError:
                pass
        return None

    async def _close(self, process: anyio.abc.Process) -> None:
        try:
            await super()._close(process)
        finally:
            # Closing the handle leaves the processes running: a background job outlives a command that returned.
            if (job := self._jobs.pop(process.pid, None)) is not None:
                _job_objects().close(job)

    def _open_for_read(self, target: Path) -> int:
        if target.is_dir():
            raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR), str(target))
        return os.open(target, os.O_RDONLY | _O_BINARY)

    async def write_bytes(self, path: str, data: bytes) -> None:
        if await run_in_executor(os.path.isdir, path):
            raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR), path)
        await super().write_bytes(path, data)

    async def make_dir(self, path: str) -> None:
        try:
            await super().make_dir(path)
        except FileExistsError:
            # Windows reports a file in place of a parent as the parent existing.
            if await run_in_executor(os.path.lexists, path):
                raise
            raise NotADirectoryError(errno.ENOTDIR, os.strerror(errno.ENOTDIR), path) from None
