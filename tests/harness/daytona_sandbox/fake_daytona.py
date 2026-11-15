"""Controllable fake for the Daytona SDK boundary.

The exception classes and `SandboxState` are the real ones from the installed SDK, so the
backend's `isinstance` and state checks run against what production raises and reports.

Deletion follows the SDK: `AsyncSandbox.delete()` returns once Daytona accepts the request, and
the sandbox stays visible to `get` and `refresh_data` in the `destroying` state until the control
plane removes it (`FakeDaytona.purge`), after which lookups raise `DaytonaNotFoundError`. Toolbox
calls to a deleted sandbox raise `DaytonaNotFoundError` too, the same type as a missing path, which
is the ambiguity the backend has to resolve.
"""

from __future__ import annotations

import asyncio
import errno
import os
import posixpath
import re
import shlex
import shutil
import signal
import subprocess
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol

import anyio
import anyio.lowlevel
from daytona import (
    DaytonaBadRequestError,
    DaytonaConnectionError,
    DaytonaError,
    DaytonaNotFoundError,
    DaytonaTimeoutError,
    SandboxState,
)


class CreateParams(Protocol):
    name: str | None
    snapshot: str | None
    auto_stop_interval: int | None
    env_vars: dict[str, str] | None
    network_block_all: bool | None
    labels: dict[str, str] | None


class FakeProcess:
    def __init__(self, owner: FakeSandbox) -> None:
        self.owner = owner

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> SimpleNamespace:
        self.owner.check_alive()
        self.owner.exec_commands.append(command)
        if command == 'pwd -P':
            self.owner.workdir_calls += 1
            self.owner.workdir_started.set()
            if self.owner.workdir_gate is not None:
                await self.owner.workdir_gate.wait()
            if self.owner.workdir_error is not None:
                raise self.owner.workdir_error
            return SimpleNamespace(result=(cwd or self.owner.workdir) + '\n', exit_code=0)
        if command.startswith('realpath -m -- '):
            target = shlex.split(command)[-1]
            resolved = posixpath.normpath(posixpath.join(cwd or self.owner.workdir, target))
            return SimpleNamespace(result=resolved + '\n', exit_code=0)
        if command.startswith('[ -p '):
            return SimpleNamespace(result='', exit_code=1)
        if command.startswith('find '):
            return SimpleNamespace(result='', exit_code=0)
        if command.startswith('for f in '):
            # The output files' sizes, as `wc -c` prints them.
            paths = command.split(';')[0].split()[3:]
            sizes = [str(len(self.owner.files.get(path, b''))) for path in paths]
            return SimpleNamespace(result=self.owner.output_sizes_reply or '\n'.join(sizes) + '\n', exit_code=0)
        if command.startswith(('head -c ', 'tail -c ')):
            # Bounded excerpts of the output files, cut in the sandbox.
            for cut in command.split('; '):
                end, _, limit, _, path, _, part = cut.split()[:7]
                data = self.owner.files.get(path, b'')
                self.owner.files[part] = data[: int(limit)] if end == 'head' else data[-int(limit) :]
            return SimpleNamespace(result='', exit_code=0)
        if command.startswith('rm -f -- '):
            for path in shlex.split(command)[3:]:
                self.owner.files.pop(path, None)
            return SimpleNamespace(result='', exit_code=0)
        assert command.startswith('mkdir -p -- ')
        return SimpleNamespace(result='', exit_code=self.owner.mkdir_exit_code)

    async def create_session(self, session_id: str, request_timeout: float | None = None) -> None:
        self.owner.check_alive()
        if self.owner.state in (SandboxState.STOPPED, SandboxState.PAUSED):
            raise DaytonaError('sandbox stopped')
        if self.owner.process_create_gate is not None:
            await self.owner.process_create_gate.wait()
        self.owner.process_sessions.add(session_id)
        if self.owner.process_create_ack_gate is not None:
            await self.owner.process_create_ack_gate.wait()

    async def execute_session_command(
        self,
        session_id: str,
        request: object,
        timeout: int | None = None,
    ) -> SimpleNamespace:
        self.owner.check_alive()
        if self.owner.exec_error is not None:
            raise self.owner.exec_error
        self.owner.process_command = getattr(request, 'command')
        if self.owner.process_stdout is self.owner.process_reply:
            # The previous command's reply, not output a test set: each command gets its own.
            self.owner.process_stdout = []
        if not self.owner.process_stdout and not self.owner.process_stderr and not self.owner.process_hangs:
            output, self.owner.process_exit_code = self.owner.responder(self.owner.process_command, timeout)
            self.owner.process_stdout = self.owner.process_reply = [output]
        marker = re.search(r'pydantic-ai-end-[0-9a-f]{32}', self.owner.process_command)
        assert marker is not None
        # The wrapper sends the command's stdout and stderr to these files.
        self.owner.files[f'/tmp/{marker.group()}.out'] = ''.join(self.owner.process_stdout).encode()
        self.owner.files[f'/tmp/{marker.group()}.err'] = ''.join(self.owner.process_stderr).encode()
        return SimpleNamespace(cmd_id=f'{session_id}-cmd')

    async def get_session_command(
        self,
        session_id: str,
        command_id: str,
        request_timeout: float | None = None,
    ) -> SimpleNamespace:
        if self.owner.process_status_gate is not None:
            await self.owner.process_status_gate.wait()
        self.owner.process_status_polled.set()
        if self.owner.process_status_error is not None:
            raise self.owner.process_status_error
        return SimpleNamespace(exit_code=None if self.owner.process_hangs else self.owner.process_exit_code)

    async def delete_session(self, session_id: str, request_timeout: float | None = None) -> None:
        # A real HTTP request yields to the loop, where a cancelled caller's cleanup would be cut short.
        await anyio.lowlevel.checkpoint()
        self.owner.process_delete_calls += 1
        self.owner.check_alive()
        if (error := next(self.owner.process_delete_errors, None)) is not None:
            raise error
        self.owner.process_sessions.discard(session_id)


class FakeFileSystem:
    def __init__(self, owner: FakeSandbox) -> None:
        self.owner = owner

    async def get_file_info(self, path: str, request_timeout: float | None = None) -> SimpleNamespace:
        self._raise_if_needed()
        if path in self.owner.directories:
            return SimpleNamespace(size=0, is_dir=True)
        data = self.owner.files.get(path)
        if data is None:
            raise DaytonaNotFoundError(f'no file: {path}')
        return SimpleNamespace(size=len(data), is_dir=False)

    async def download_file(self, path: str, timeout: int | None = None) -> bytes:
        self._raise_if_needed()
        if path in self.owner.directories:
            raise DaytonaBadRequestError(f'path is a directory: {path}', status_code=400)
        if self.owner.download_error is not None:
            raise self.owner.download_error
        data = self.owner.files.get(path)
        if data is None:
            raise DaytonaNotFoundError(f'no file: {path}')
        return data

    async def upload_file(self, data: bytes, path: str, timeout: int = 1800) -> None:
        self._raise_if_needed()
        self.owner.files[path] = data

    async def list_files(
        self, path: str, depth: int | None = None, request_timeout: float | None = None
    ) -> list[SimpleNamespace]:
        self._raise_if_needed()
        prefix = '' if path in ('', '.') else path.rstrip('/') + '/'
        entries: dict[str, bool] = {}
        # File keys never end with '/', so the first segment under the prefix is never empty.
        for candidate in self.owner.files:
            if candidate.startswith(prefix):
                name, separator, _ = candidate[len(prefix) :].partition('/')
                entries[name] = bool(separator) or entries.get(name, False)
        for directory in self.owner.directories:
            if directory.startswith(prefix):
                name = directory[len(prefix) :].partition('/')[0]
                entries[name] = True
        return [
            SimpleNamespace(
                name=name,
                is_dir=is_dir,
                size=0 if is_dir else len(self.owner.files[prefix + name]),
            )
            for name, is_dir in entries.items()
        ]

    async def create_folder(self, path: str, mode: str, request_timeout: float | None = None) -> None:
        self._raise_if_needed()
        assert mode == '755'
        self.owner.directories.add(path)

    async def delete_file(self, path: str, recursive: bool = False, request_timeout: float | None = None) -> None:
        self._raise_if_needed()
        self.owner.files.pop(path, None)
        self.owner.directories.discard(path)
        if recursive:  # pragma: no branch - the sandbox protocol always requests recursive removal
            prefix = path.rstrip('/') + '/'
            self.owner.files = {key: value for key, value in self.owner.files.items() if not key.startswith(prefix)}
            self.owner.directories = {key for key in self.owner.directories if not key.startswith(prefix)}

    def _raise_if_needed(self) -> None:
        self.owner.check_alive()
        if self.owner.fs_error is not None:
            raise self.owner.fs_error


@contextmanager
def _host_errors(path: str, operation: str) -> Generator[None]:
    """Raise the SDK errors the toolbox's status codes turn into for these host errors.

    Apart from a missing path, the toolbox answers a Go `PathError` with a 400 whose text is the
    only thing telling the failures apart (observed live 2026-09-25). A `mkdir` over a file is
    `not a directory` there, since Go's `MkdirAll` reports that rather than `file exists`.
    """
    try:
        yield
    except FileNotFoundError as error:
        raise DaytonaNotFoundError(f'path not found: {path}', status_code=404) from error
    except IsADirectoryError as error:
        raise DaytonaError(f'Failed to {operation}: 400: open {path}: is a directory', status_code=400) from error
    except (NotADirectoryError, FileExistsError) as error:
        raise DaytonaBadRequestError(
            f'Failed to {operation}: bad request: {path}: not a directory', status_code=400
        ) from error


class _HostProcess(FakeProcess):
    """Mirrors `sandbox.process` by running commands on the host under `host_root`.

    Each session command is a host process in its own process group; deleting the session kills
    the group, as Daytona kills a deleted session's command.
    """

    def __init__(self, owner: FakeSandbox, host_root: Path) -> None:
        super().__init__(owner)
        self.host_root = host_root
        self.commands: dict[str, subprocess.Popen[bytes]] = {}

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> SimpleNamespace:
        self.owner.check_alive()
        if command.startswith('realpath -m -- '):
            target = shlex.split(command)[-1]
            # `os.path.realpath` leaves a symlink loop unresolved; `Path.resolve` raises on it before 3.13.
            resolved = os.path.realpath(os.path.join(cwd or self.host_root, target))
            return SimpleNamespace(result=f'{resolved}\n', exit_code=0)
        # `exec` reports stdout and stderr combined as `result`.
        done = await anyio.run_process(
            ['sh', '-c', command], cwd=cwd or self.host_root, check=False, stderr=subprocess.STDOUT
        )
        return SimpleNamespace(result=done.stdout.decode(), exit_code=done.returncode)

    async def execute_session_command(
        self,
        session_id: str,
        request: object,
        timeout: int | None = None,
    ) -> SimpleNamespace:
        self.owner.check_alive()
        # The command line sends its output to files itself.
        process = subprocess.Popen(
            ['sh', '-c', getattr(request, 'command')],
            cwd=self.host_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        command_id = f'{session_id}-cmd'
        self.commands[command_id] = process
        return SimpleNamespace(cmd_id=command_id)

    async def get_session_command(
        self,
        session_id: str,
        command_id: str,
        request_timeout: float | None = None,
    ) -> SimpleNamespace:
        self.owner.check_alive()
        return SimpleNamespace(exit_code=self.commands[command_id].poll())

    async def delete_session(self, session_id: str, request_timeout: float | None = None) -> None:
        process = self.commands.pop(f'{session_id}-cmd')
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def _loops(path: Path) -> bool:
    try:
        path.stat()
    except OSError as error:
        return error.errno == errno.ELOOP
    return False


class _HostFileSystem(FakeFileSystem):
    """Mirrors `sandbox.fs` on the host filesystem, so commands and file calls share one tree.

    Like the SDK's `FileInfo`, entries carry no symlink flag; `is_dir` follows a symlink, as
    `Path.is_dir` does. The live tier checks what Daytona itself reports.
    """

    async def get_file_info(self, path: str, request_timeout: float | None = None) -> SimpleNamespace:
        self._raise_if_needed()
        with _host_errors(path, 'get file info'):
            info = Path(path).stat()
        return SimpleNamespace(size=info.st_size, is_dir=Path(path).is_dir())

    async def download_file(self, path: str, timeout: int | None = None) -> bytes:
        self._raise_if_needed()
        with _host_errors(path, 'download file'):
            return Path(path).read_bytes()

    async def upload_file(self, data: bytes, path: str, timeout: int = 1800) -> None:
        self._raise_if_needed()
        # Like the toolbox (observed live): missing parents are created, and a symlink is written through.
        with _host_errors(path, 'upload files'):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(data)

    async def list_files(
        self, path: str, depth: int | None = None, request_timeout: float | None = None
    ) -> list[SimpleNamespace]:
        self._raise_if_needed()
        with _host_errors(path, 'list files'):
            children = sorted(Path(path).iterdir())
        # Like the toolbox (observed live), leave out a symlink that loops; `find` in the backend restores it.
        return [
            SimpleNamespace(
                name=child.name,
                is_dir=child.is_dir(),
                size=child.stat().st_size if child.exists() else child.lstat().st_size,
            )
            for child in children
            if not _loops(child)
        ]

    async def create_folder(self, path: str, mode: str, request_timeout: float | None = None) -> None:
        self._raise_if_needed()
        with _host_errors(path, 'create folder'):
            Path(path).mkdir(mode=int(mode, 8), parents=True, exist_ok=True)

    async def delete_file(self, path: str, recursive: bool = False, request_timeout: float | None = None) -> None:
        self._raise_if_needed()
        with _host_errors(path, 'delete file'):
            if Path(path).is_dir():
                shutil.rmtree(path)
            else:
                Path(path).unlink()


class FakeSandbox:
    def __init__(self, sandbox_id: str, host_root: Path | None = None) -> None:
        self.client: FakeClient | None = None
        self.id = sandbox_id
        self.name: str | None = None
        self.state = SandboxState.STARTED
        self.purged = False
        self.refresh_error: Exception | None = None
        # What a toolbox call to a deleted sandbox raises, when not the default 404.
        self.gone_error: Exception | None = None
        self.download_error: Exception | None = None
        self.start_calls: list[float | None] = []
        self.files: dict[str, bytes] = {}
        self.directories: set[str] = set()
        self.exec_error: Exception | None = None
        # Every `process.exec` command line, in order.
        self.exec_commands: list[str] = []
        self.fs_error: Exception | None = None
        self.workdir = '/srv/repo'
        self.workdir_error: Exception | None = None
        self.workdir_calls = 0
        self.workdir_gate: asyncio.Event | None = None
        self.workdir_started = asyncio.Event()
        self.mkdir_exit_code = 0
        self.responder: Callable[[str, int | None], tuple[str, int]] = lambda command, timeout: ('', 0)
        self.process_sessions: set[str] = set()
        self.process_command = ''
        # What the next session command writes to its stdout and stderr files.
        self.process_stdout: list[str] = []
        self.process_stderr: list[str] = []
        self.process_reply: list[str] | None = None
        self.process_hangs = False
        # Replaces the output files' sizes the backend asks for, to test a garbled reply.
        self.output_sizes_reply = ''
        self.process_exit_code: int | None = 0
        # Raised by successive `delete_session` calls; deletion succeeds once it is exhausted.
        self.process_delete_errors: Iterator[Exception] = iter(())
        self.process_delete_calls = 0
        self.process_status_error: Exception | None = None
        self.process_status_gate: asyncio.Event | None = None
        self.process_create_gate: asyncio.Event | None = None
        self.process_create_ack_gate: asyncio.Event | None = None
        self.process_status_polled = asyncio.Event()
        self.process = FakeProcess(self)
        self.fs = FakeFileSystem(self)
        if host_root is not None:
            self.workdir = str(host_root)
            self.process = _HostProcess(self, host_root)
            self.fs = _HostFileSystem(self)

    async def start(self, timeout: float | None = 60) -> None:
        self.start_calls.append(timeout)
        self.state = SandboxState.STARTED

    async def wait_for_sandbox_start(self, timeout: float | None = 60) -> None:
        self.state = SandboxState.STARTED

    async def refresh_data(self, request_timeout: float | None = None) -> None:
        if self.refresh_error is not None:
            raise self.refresh_error
        if self.purged:
            raise DaytonaNotFoundError(f'Sandbox with ID or name {self.id} not found', status_code=404)

    async def delete(self, timeout: float | None = 60, wait: bool = False) -> None:
        self.state = SandboxState.DESTROYING
        if isinstance(self.process, _HostProcess):
            # The real control plane stops command sessions when their sandbox is deleted;
            # merely changing the fake state leaves the host subprocess running.
            for process in self.process.commands.values():
                if process.poll() is None:  # pragma: no branch - a finished command's session is deleted before this
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()

    def check_alive(self) -> None:
        if self.state in (SandboxState.DESTROYING, SandboxState.DESTROYED):
            raise self.gone_error or DaytonaNotFoundError(
                f'Sandbox with ID or name {self.id} not found', status_code=404
            )


class FakeClient:
    def __init__(self, owner: FakeDaytona) -> None:
        self.owner = owner
        self.closed = False

    async def create(self, params: CreateParams, *, timeout: float = 60) -> FakeSandbox:
        if self.owner.create_error is not None:
            raise self.owner.create_error
        # Like the SDK, the sandbox exists once the create request is accepted, before `create`
        # returns: it then waits for the sandbox to start.
        sandbox = FakeSandbox(f'sb-{len(self.owner.create_params) + 1}', host_root=self.owner.host_root)
        sandbox.client = self
        sandbox.name = params.name
        self.owner.sandboxes.append(sandbox)
        self.owner.create_params.append(params)
        self.owner.create_started.set()
        if self.owner.create_gate is not None:
            await self.owner.create_gate.wait()
        if self.owner.lose_create_reply:
            raise DaytonaConnectionError('create response lost')
        if self.owner.slow_start:
            # The SDK's `create` gives up waiting for the start; the sandbox keeps starting.
            sandbox.state = SandboxState.STARTING
            raise DaytonaTimeoutError('Sandbox failed to become ready within the timeout period')
        return sandbox

    async def get(self, sandbox_id: str, request_timeout: float | None = None) -> FakeSandbox:
        if self.owner.get_error is not None:
            raise self.owner.get_error
        if self.owner.get_gate is not None:
            await self.owner.get_gate.wait()
        for sandbox in self.owner.sandboxes:
            if sandbox.id == sandbox_id or sandbox.name == sandbox_id:
                sandbox.client = self
                return sandbox
        raise DaytonaNotFoundError(f'no sandbox: {sandbox_id}')

    async def close(self) -> None:
        if self.owner.close_gate is not None:
            await self.owner.close_gate.wait()
        if (error := next(self.owner.close_errors, None)) is not None:
            raise error
        self.closed = True
        self.owner.closed_clients += 1


class FakeDaytona:
    def __init__(self) -> None:
        self.sandboxes: list[FakeSandbox] = []
        self.create_params: list[CreateParams] = []
        self.opened_clients = 0
        self.closed_clients = 0
        self.create_error: Exception | None = None
        self.lose_create_reply = False
        self.slow_start = False
        self.create_gate: asyncio.Event | None = None
        self.create_started = asyncio.Event()
        # Raised by successive `close` calls; closing succeeds once it is exhausted.
        self.close_errors: Iterator[Exception] = iter(())
        self.close_gate: asyncio.Event | None = None
        self.get_gate: asyncio.Event | None = None
        self.get_error: Exception | None = None
        # What constructing `AsyncDaytona()` raises, e.g. for a missing API key.
        self.client_error: Exception | None = None
        # When set, new sandboxes run commands and file operations on the host under this directory.
        self.host_root: Path | None = None

    def client(self, _config: object = None) -> FakeClient:
        if self.client_error is not None:
            raise self.client_error
        self.opened_clients += 1
        return FakeClient(self)

    def purge(self, sandbox: FakeSandbox) -> None:
        """Finish a deletion: the control plane forgets the sandbox."""
        sandbox.state = SandboxState.DESTROYED
        sandbox.purged = True
        self.sandboxes.remove(sandbox)

    def sandbox(self, sandbox_id: str = 'sb-existing') -> FakeSandbox:
        sandbox = FakeSandbox(sandbox_id)
        self.sandboxes.append(sandbox)
        return sandbox
