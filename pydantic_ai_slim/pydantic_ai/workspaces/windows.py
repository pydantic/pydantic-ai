"""A Windows implementation of the [workspace backend protocol][pydantic_ai.workspaces.WorkspaceBackend].

[`WindowsWorkspaceBackend`][pydantic_ai.workspaces.WindowsWorkspaceBackend] runs commands as plain host
subprocesses in PowerShell -- it **isolates nothing**.
"""

from __future__ import annotations as _annotations

import errno
import functools
import ntpath
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from subprocess import DEVNULL, PIPE

import anyio
import anyio.abc

from pydantic_ai._utils import run_in_executor

from ._host import HostWorkspaceBackend
from .protocol import CommandResult, SupportsPlatform, SupportsRunInDirectory, WorkspaceCommand, WorkspacePlatform

__all__ = ('WindowsWorkspaceBackend', 'powershell_script')

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


def powershell_script(script: str) -> str:
    """`script` as [`WindowsWorkspaceBackend`][pydantic_ai.workspaces.WindowsWorkspaceBackend] runs a `shell=True` command.

    It writes its output as UTF-8 without progress bars, and exits with its last native command's code, as `sh`
    does, rather than PowerShell's 1 for any failure. For a tool that runs PowerShell some other way, such as a
    detached job, so that it behaves like a command run directly.
    """
    return f'{_PREAMBLE}{script}{_EPILOGUE}'


_SCRIPT_NAME = 'script.ps1'

_PRIVATE_DACL = 'D:P(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)(A;OICI;FA;;;OW)'
"""Full control for SYSTEM, Administrators and the owner only, inherited by what the directory holds."""

_DACL_SECURITY_INFORMATION = 0x00000004
_PROTECTED_DACL_SECURITY_INFORMATION = 0x80000000


def _make_private(directory: str) -> None:
    """Replace `directory`'s ACL, which it inherits from the temporary directory, with `_PRIVATE_DACL`.

    `mkdtemp` does the same itself since Python 3.11.10, 3.12.4 and 3.13, but not on earlier patch releases.
    """
    import ctypes
    from ctypes import wintypes

    # `WinDLL` exists only on Windows; it is a `CDLL` subclass.
    win_dll: type[ctypes.CDLL] = getattr(ctypes, 'WinDLL')
    advapi32 = win_dll('advapi32')
    convert = advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW
    convert.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p)
    convert.restype = wintypes.BOOL
    set_security = advapi32.SetFileSecurityW
    set_security.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p)
    set_security.restype = wintypes.BOOL
    local_free = win_dll('kernel32').LocalFree
    local_free.argtypes = (ctypes.c_void_p,)

    descriptor = ctypes.c_void_p()
    if not convert(_PRIVATE_DACL, 1, ctypes.byref(descriptor), None):  # pragma: no cover - the descriptor is fixed
        raise OSError(errno.EINVAL, 'invalid security descriptor', _PRIVATE_DACL)
    try:
        information = _DACL_SECURITY_INFORMATION | _PROTECTED_DACL_SECURITY_INFORMATION
        if not set_security(directory, information, descriptor):  # pragma: no cover - the owner can set its ACL
            raise OSError(errno.EACCES, 'could not make the directory private', directory)
    finally:
        local_free(descriptor)


def _write_script(script: str) -> str:
    """A new temporary directory holding `powershell_script(script)` as `_SCRIPT_NAME`.

    The script may hold secrets, so the directory is for this user alone: `mkdtemp` gives it mode `0o700`, and on
    Windows it gets `_PRIVATE_DACL` before the script is written. UTF-8 with a BOM, like the harness's `command.ps1`,
    so the file reads correctly with `-File` too; `ReadAllText` doesn't need it.
    """
    directory = tempfile.mkdtemp(prefix='pydantic-ai-')
    try:
        if sys.platform == 'win32':
            _make_private(directory)
        with open(os.path.join(directory, _SCRIPT_NAME), 'x', encoding='utf-8-sig', newline='') as file:
            file.write(powershell_script(script))
    except BaseException:
        _remove_script(directory)
        raise
    return directory


def _remove_script(directory: str) -> None:
    # An antivirus scan can hold the file open for a moment; a leftover in the temporary directory is harmless.
    shutil.rmtree(directory, ignore_errors=True)


def _powershell_command(executable: str, script_path: str) -> list[str]:
    """The argv that runs the script at `script_path` in PowerShell.

    The script stays off the command line: antivirus software flags an `-EncodedCommand` launch, and one
    command line holds only about 12,000 characters of an encoded script. PowerShell reads the file as text
    rather than running it with `-File`, so no execution policy, Group Policy's included, applies to it.
    """
    # PowerShell also ends a single-quoted string at a typographic single quote, so those are doubled too.
    literal = re.sub("(['\u2018\u2019\u201a\u201b])", r'\1\1', script_path)
    return [
        executable,
        *_POWERSHELL_FLAGS,
        '-Command',
        f"& ([ScriptBlock]::Create([IO.File]::ReadAllText('{literal}')))",
    ]


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
            if process and self._assign(job, process):  # pragma: no branch - refused only out of resources
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
    extensions = [''] if ntpath.splitext(name)[1] else env.get('PATHEXT', '.COM;.EXE').split(';')
    for directory in env.get('PATH', '').split(';'):
        if ntpath.splitdrive(directory)[0] and ntpath.isabs(directory):
            for extension in extensions:
                candidate = ntpath.join(directory, name + extension)
                if not _is_batch(candidate) and os.path.isfile(candidate):
                    return candidate
    return None


def _is_batch(program: str) -> bool:
    return ntpath.splitext(program)[1].lower() in ('.bat', '.cmd')


def _resolve_program(argv: Sequence[str], env: Mapping[str, str]) -> list[str]:
    """`argv` with a bare program name replaced by its path on `PATH`.

    `CreateProcess` looks for a bare name in this process's current directory before `PATH`, which
    `execvp` on POSIX does not. Raises `FileNotFoundError` when it is not on `PATH`, and
    `PermissionError` for a batch file: Windows runs one through `cmd.exe`, which parses its arguments
    as a command line, so a model-written argument such as a search pattern could run commands.
    """
    program, *args = argv
    if _is_batch(program):
        raise PermissionError(
            errno.EACCES, 'batch files run through `cmd.exe` and are refused as argv programs', program
        )
    if ntpath.basename(program) != program or ntpath.splitdrive(program)[0]:
        return [program, *args]
    if (path := _which(program, env)) is None:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), program)
    return [path, *args]


@functools.cache
def _job_objects() -> _JobObjects:
    """Loaded on the first command, so building a backend does no I/O."""
    return _JobObjects()


class WindowsWorkspaceBackend(HostWorkspaceBackend, SupportsPlatform, SupportsRunInDirectory):
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
        # Keyed by PID: each `shell=True` command's script directory, from its start until it is reaped.
        self._scripts: dict[int, str] = {}

    @property
    def workspace_platform(self) -> WorkspacePlatform:
        """Always `'windows'`: commands run in PowerShell, and paths are Windows paths."""
        return 'windows'

    def _executable(self) -> str:
        # From the backend's own environment, so no call's `env` decides it for later ones.
        if self._powershell is None:
            self._powershell = _which('pwsh', self._env) or 'powershell.exe'
        return self._powershell

    async def run_in_directory(
        self,
        command: WorkspaceCommand,
        *,
        cwd: str,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        """Run `command`, a PowerShell script with `shell=True` or an argv, starting in `cwd`."""
        return await self._run(command, shell=shell, env=env, timeout=timeout, cwd=cwd)

    async def _spawn(self, command: WorkspaceCommand, working_dir: Path, env: Mapping[str, str]) -> anyio.abc.Process:
        script_dir: str | None = None
        try:
            if isinstance(command, str):
                # Disk I/O and a `PATH` search, so not on the event loop.
                script_dir = await run_in_executor(_write_script, command)
                script_path = os.path.join(script_dir, _SCRIPT_NAME)
                command = _powershell_command(await run_in_executor(self._executable), script_path)
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
        except BaseException:
            if script_dir is not None:
                await run_in_executor(_remove_script, script_dir)
            raise
        if script_dir is not None:
            self._scripts[process.pid] = script_dir
        if (job := _job_objects().assign(process.pid)) is not None:  # pragma: no branch
            self._jobs[process.pid] = job
        return process

    def _kill_tree(self, process: anyio.abc.Process) -> tuple[PermissionError | None, PermissionError | None]:
        job = self._jobs.get(process.pid)
        if job is not None:
            _job_objects().terminate(job)
        else:  # pragma: no cover - only when `assign` failed
            try:
                process.kill()
            except ProcessLookupError:
                pass
        return None, None

    async def _close(self, process: anyio.abc.Process) -> None:
        try:
            await super()._close(process)
        finally:
            # Closing the handle leaves the processes running: a background job outlives a command that returned.
            if (job := self._jobs.pop(process.pid, None)) is not None:  # pragma: no branch
                _job_objects().close(job)
            if (script_dir := self._scripts.pop(process.pid, None)) is not None:
                await run_in_executor(_remove_script, script_dir)

    @staticmethod
    def _path(path: str) -> Path:
        # `file::$DATA` and `file:stream` name a file's data streams, so `.env::$DATA` would write `.env`
        # past a policy that matched the spelling.
        rest = ntpath.splitdrive(path)[1]
        if ':' in rest:
            raise ValueError(f'NTFS stream names are not supported, got {path!r}')
        # Windows drops a name's trailing dots and spaces, so `server.key.` would write `server.key`.
        if any(part.endswith(('.', ' ')) for part in re.split(r'[\\/]', rest) if part not in ('', '.', '..')):
            raise ValueError(f'names ending in a dot or a space are not supported, got {path!r}')
        return HostWorkspaceBackend._path(path)

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
