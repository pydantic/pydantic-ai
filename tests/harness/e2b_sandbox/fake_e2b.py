"""A controllable fake `e2b` SDK for E2BSandbox tests.

Tests never reach real E2B: the `fake_e2b` fixture replaces the backend's SDK module with a fake.
The fake records calls and lets each test decide what a command returns.

Fidelity to the real SDK is the point. The exception classes, `FileType`, `CommandResult`,
`CommandExitException`, and `WriteInfo` are the real ones, imported from the installed
package, so the backend's `isinstance` checks and its unwrapping of a non-zero exit are
exercised against the types production raises. Signatures are closed, every await suspends,
a command handle accumulates output before `wait()` returns (as the SDK's event pump does),
a non-zero exit raises rather than returns, and a missing path raises E2B's own filesystem
exception. Flattering the code under test here would hide production failures.
"""

from __future__ import annotations

import os
import posixpath
import shlex
import shutil
import stat
import subprocess
import tempfile
import types
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING, Literal, Protocol

import anyio
import anyio.lowlevel
from e2b import CommandExitException, CommandResult, FileType, WriteInfo
from e2b.exceptions import (
    AuthenticationException,
    FileNotFoundException,
    InvalidArgumentException,
    RateLimitException,
    SandboxException,
    SandboxNotFoundException,
    TimeoutException,
    format_sandbox_timeout_exception,
)

__all__ = (
    'FakeCommandCall',
    'FakeCreateCall',
    'FakeE2B',
    'FakeEntryInfo',
    'FakeSandbox',
)

# A responder maps (command line, deadline) to (stdout, stderr, exit_code).
Responder = Callable[[str, 'float | None'], 'tuple[str, str, int]']


def _echo_responder(command: str, timeout: float | None) -> tuple[str, str, int]:
    return f'{command}\n', '', 0


def _user_command(command: str) -> str:
    # The fake checks the outer launcher but responds to the user command, not its
    # registration shell. Quoting must survive both the E2B login shell and setsid.
    args = shlex.split(command)
    if args[:3] == ['setsid', 'sh', '-c']:
        return args[3].split('; exec ', 1)[1]
    if args[:2] == ['sh', '-c'] and '; exec ' in args[2]:
        return args[2].split('; exec ', 1)[1]
    return command


@dataclass(frozen=True)
class FakeCreateCall:
    template: str | None
    timeout: int | None
    envs: dict[str, str] | None
    secure: bool
    allow_internet_access: bool
    lifecycle: dict[str, str] | None


@dataclass(frozen=True)
class FakeCommandCall:
    command: str
    background: bool
    cwd: str | None
    envs: dict[str, str] | None
    timeout: float | None


@dataclass(frozen=True)
class FakeEntryInfo:
    """The `e2b.EntryInfo` members the backend reads.

    A subset rather than the real dataclass, which also carries mode, permissions, owner,
    group, and timestamps that no code here looks at. The `if TYPE_CHECKING` block below pins
    this subset against the real type so a drift in E2B's entry shape fails the type check.
    """

    name: str
    path: str
    type: FileType | None
    size: int
    symlink_target: str | None = None


class FakeCommandHandle:
    """Mirrors `e2b.AsyncCommandHandle` for the members the backend uses.

    Output is accumulated at construction rather than at `wait()`, the way the SDK's event
    pump fills its chunk lists as data arrives: that is what makes `stdout` readable after a
    deadline kill cancels the wait.
    """

    def __init__(
        self,
        control: FakeE2B,
        sandbox: FakeSandbox,
        *,
        pid: int,
        stdout: str,
        stderr: str,
        exit_code: int,
    ) -> None:
        self._control = control
        self._sandbox = sandbox
        self._pid = pid
        self._stdout = stdout
        self._stderr = stderr
        self._exit_code = exit_code

    def close(self) -> None:
        """Release a host-backed handle; in-memory handles have nothing to release."""

    @property
    def pid(self) -> int:
        return self._pid

    @property
    def stdout(self) -> str:
        return self._stdout

    @property
    def stderr(self) -> str:
        return self._stderr

    async def wait(self) -> CommandResult:
        # A real wait suspends; yield so a test can cancel it and so concurrent tool calls
        # actually interleave.
        await anyio.lowlevel.checkpoint()
        self._sandbox.check_alive()
        if self._control.wait_error is not None:
            raise self._control.wait_error
        if (
            self._control.command_hangs
            and 'kill -TERM ' not in self._sandbox.commands.calls[self._pid - self._control.next_pid].command
        ):
            await anyio.sleep_forever()
        if self._exit_code != 0:
            # The real SDK raises on a non-zero exit instead of returning a result.
            raise CommandExitException(
                stderr=self._stderr,
                stdout=self._stdout,
                exit_code=self._exit_code,
                error=f'exit status {self._exit_code}',
            )
        return CommandResult(stderr=self._stderr, stdout=self._stdout, exit_code=self._exit_code, error=None)


class FakeCommands:
    """Mirrors `sandbox.commands`: command execution and per-command kill."""

    def __init__(self, sandbox: FakeSandbox, control: FakeE2B) -> None:
        self._sandbox = sandbox
        self._control = control
        self.calls: list[FakeCommandCall] = []
        self.handles: list[FakeCommandHandle] = []
        self.killed_pids: list[int] = []
        self.group_stops: list[str] = []

    async def run(
        self,
        cmd: str,
        background: bool | None = None,
        envs: dict[str, str] | None = None,
        user: str | None = None,
        cwd: str | None = None,
        timeout: float | None = 60,
    ) -> FakeCommandHandle:
        # Closed signature on purpose: the real `run` rejects unknown kwargs, so the fake must
        # too, or a bad kwarg in the backend would only fail in production.
        del user
        await anyio.lowlevel.checkpoint()
        self._sandbox.check_alive()
        self.calls.append(FakeCommandCall(cmd, background is True, cwd, envs, timeout))
        assert background is True, 'the backend always starts commands in the background'
        if self._control.run_error is not None:
            raise self._control.run_error
        if cmd.startswith('sh -c ') and 'kill -TERM ' in cmd:
            self.group_stops.append(cmd)
            if self._control.kill_command_error is not None:
                raise self._control.kill_command_error
        user_command = _user_command(cmd)
        # The in-memory fake has no FIFO objects; `test -p` must report false rather
        # than inheriting a generic test responder's success status.
        stdout, stderr, exit_code = (
            ('', '', 1) if 'test -p ' in user_command else self._control.responder(user_command, timeout)
        )
        handle = FakeCommandHandle(
            self._control,
            self._sandbox,
            pid=self._control.next_pid + len(self.handles),
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
        )
        self.handles.append(handle)
        if self._control.start_response_held is not None:
            # E2B has started the process before the start event with its pid arrives.
            await self._control.start_response_held.wait()
        return handle

    async def kill(self, pid: int, request_timeout: float | None = None) -> bool:
        del request_timeout
        await anyio.lowlevel.checkpoint()
        self._sandbox.check_alive()
        self.killed_pids.append(pid)
        if self._control.kill_command_error is not None:
            raise self._control.kill_command_error
        return True


class FakeFilesystem:
    """Mirrors `sandbox.files`: an in-memory tree the tests can drive and inspect."""

    def __init__(self, sandbox: FakeSandbox, control: FakeE2B) -> None:
        self._sandbox = sandbox
        self._control = control
        self.files: dict[str, bytes] = {}
        self.directories: set[str] = set()
        self.removed: list[str] = []
        # Paths the sandbox user may not touch, the way a root-owned `/etc` refuses envd's user.
        self.denied: set[str] = set()
        # Symlinks by path, each reported with the target a test gives it.
        self.symlinks: dict[str, str] = {}

    async def read(
        self,
        path: str,
        format: Literal['bytes'],
        user: str | None = None,
        request_timeout: float | None = None,
        gzip: bool = False,
    ) -> bytearray:
        del user, request_timeout, gzip
        await self._check(path)
        # The backend only asks for bytes; anything else would be a silent behavior change.
        assert format == 'bytes', f'unexpected read format {format!r}'
        if self._control.read_error is not None:
            raise self._control.read_error
        if path in self.directories:
            # envd answers a read of a directory with a 400, which the SDK raises like this.
            raise InvalidArgumentException(f"path '{path}' is a directory")
        if path not in self.files:
            raise FileNotFoundException(path)
        # Real E2B hands back a `bytearray` for the `bytes` format.
        return bytearray(self.files[path])

    async def write(
        self,
        path: str,
        data: str | bytes,
        user: str | None = None,
        request_timeout: float | None = None,
    ) -> WriteInfo:
        del user, request_timeout
        await self._check(path)
        if path in self.directories:
            # envd's upload answers a directory with a 400 in its own words.
            raise InvalidArgumentException(f'path is a directory: {path}')
        self.files[path] = data.encode() if isinstance(data, str) else data
        self._add_parents(path)
        return WriteInfo(name=posixpath.basename(path), type=FileType.FILE, path=path)

    async def get_info(self, path: str, user: str | None = None, request_timeout: float | None = None) -> FakeEntryInfo:
        del user, request_timeout
        await self._check(path)
        return self._entry(path)

    async def list(
        self,
        path: str,
        depth: int | None = 1,
        user: str | None = None,
        request_timeout: float | None = None,
    ) -> list[FakeEntryInfo]:
        del user, request_timeout
        await self._check(path)
        assert depth == 1, f'unexpected list depth {depth!r}'
        if not await self._exists(path):
            raise FileNotFoundException(path)
        if path in self.files:
            raise InvalidArgumentException(f'path is not a directory: {path}')
        children = {
            posixpath.join(path, name)
            for entry in (*self.files, *self.directories)
            if entry != path and entry.startswith(f'{path.rstrip("/")}/')
            for name in (entry[len(path.rstrip('/')) + 1 :].split('/')[0],)
        }
        return [self._entry(child) for child in sorted(children)]

    async def exists(self, path: str, user: str | None = None, request_timeout: float | None = None) -> bool:
        del user, request_timeout
        await self._check(path)
        return await self._exists(path)

    async def make_dir(self, path: str, user: str | None = None, request_timeout: float | None = None) -> bool:
        del user, request_timeout
        await self._check(path)
        if path in self.files:
            raise InvalidArgumentException(f'path already exists but it is not a directory: {path}')
        created = path not in self.directories
        self.directories.add(path)
        self._add_parents(path)
        return created

    async def remove(self, path: str, user: str | None = None, request_timeout: float | None = None) -> None:
        del user, request_timeout
        await self._check(path)
        # envd removes with `os.RemoveAll`, so a missing path is not an error.
        self.removed.append(path)
        prefix = f'{path.rstrip("/")}/'
        for target in [target for target in self.files if target == path or target.startswith(prefix)]:
            del self.files[target]
        for directory in [d for d in self.directories if d == path or d.startswith(prefix)]:
            self.directories.discard(directory)

    async def _exists(self, path: str) -> bool:
        return path in self.files or path in self.directories

    def _entry(self, path: str) -> FakeEntryInfo:
        if (target := self.symlinks.get(path)) is not None:
            return FakeEntryInfo(
                name=posixpath.basename(path), path=path, type=None, size=len(target), symlink_target=target
            )
        if path in self.directories:
            return FakeEntryInfo(name=posixpath.basename(path), path=path, type=FileType.DIR, size=0)
        if path not in self.files:
            raise FileNotFoundException(path)
        return FakeEntryInfo(name=posixpath.basename(path), path=path, type=FileType.FILE, size=len(self.files[path]))

    def _add_parents(self, path: str) -> None:
        parent = posixpath.dirname(path)
        while parent and parent != '/':
            self.directories.add(parent)
            parent = posixpath.dirname(parent)

    async def _check(self, path: str) -> None:
        # Real E2B's filesystem API only accepts absolute paths; assert it here so a
        # regression that let a relative path through unresolved fails in the fake the way it
        # would in prod, instead of silently keying the in-memory store on a relative path.
        assert posixpath.isabs(path), f'E2B filesystem requires an absolute path, got {path!r}'
        await anyio.lowlevel.checkpoint()
        self._sandbox.check_alive()
        if self._control.fs_error is not None:
            raise self._control.fs_error
        # envd reports what the kernel refused as a 500 carrying Go's errno text, which the SDK
        # raises as an untyped `SandboxException`.
        if any(path == denied or path.startswith(f'{denied}/') for denied in self.denied):
            raise SandboxException(f'500: open {path}: permission denied')
        parent = posixpath.dirname(path)
        while parent != '/':
            if parent in self.files:
                raise SandboxException(f'500: stat {path}: not a directory')
            parent = posixpath.dirname(parent)


def _captured(stream: IO[bytes]) -> str:
    # `pread` reads from the start without moving the offset the child is still writing at.
    return os.pread(stream.fileno(), os.fstat(stream.fileno()).st_size, 0).decode(errors='replace')


class _HostCommandHandle(FakeCommandHandle):
    """A command handle over a real host process, for the conformance suite.

    Output goes to anonymous temporary files, so what the process printed before a deadline
    kill is readable afterwards, as the SDK's accumulated handle output is.
    """

    def __init__(
        self,
        control: FakeE2B,
        sandbox: FakeSandbox,
        process: subprocess.Popen[bytes],
        out: IO[bytes],
        err: IO[bytes],
    ) -> None:
        super().__init__(control, sandbox, pid=process.pid, stdout='', stderr='', exit_code=0)
        self.process = process
        self._out = out
        self._err = err

    @property
    def stdout(self) -> str:
        return _captured(self._out)

    @property
    def stderr(self) -> str:
        return _captured(self._err)

    def close(self) -> None:
        """Stop the process if it still runs and release its output files."""
        try:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait()
        finally:
            self._out.close()
            self._err.close()

    async def wait(self) -> CommandResult:
        while (exit_code := self.process.poll()) is None:
            self._sandbox.check_alive()
            await anyio.sleep(0.01)
        stdout, stderr = self.stdout, self.stderr
        self.close()
        if exit_code != 0:
            raise CommandExitException(
                stderr=stderr, stdout=stdout, exit_code=exit_code, error=f'exit status {exit_code}'
            )
        return CommandResult(stderr=stderr, stdout=stdout, exit_code=0, error=None)


class _HostCommands(FakeCommands):
    """Mirrors `sandbox.commands` by running each command on the host under `host_root`."""

    async def run(
        self,
        cmd: str,
        background: bool | None = None,
        envs: dict[str, str] | None = None,
        user: str | None = None,
        cwd: str | None = None,
        timeout: float | None = 60,
    ) -> FakeCommandHandle:
        del user
        await anyio.lowlevel.checkpoint()
        self._sandbox.check_alive()
        self.calls.append(FakeCommandCall(cmd, background is True, cwd, envs, timeout))
        assert background is True, 'the backend always starts commands in the background'
        assert self._control.host_root is not None
        # E2B runs `/bin/bash -l -c`; the host drops `-l` so the developer's login files stay out.
        out, err = tempfile.TemporaryFile(), tempfile.TemporaryFile()
        # macOS lacks the `setsid` executable: emulate its group isolation via
        # Popen's POSIX session flag while retaining the same registration shell.
        args = shlex.split(cmd)
        isolated = args[:3] == ['setsid', 'sh', '-c']
        if isolated:
            cmd = f'sh -c {shlex.quote(args[3])}'
        try:
            process = subprocess.Popen(
                ['/bin/bash', '-c', cmd],
                start_new_session=isolated,
                cwd=cwd or self._control.host_root,
                env={**os.environ, **(envs or {})},
                stdout=out,
                stderr=err,
            )
        except BaseException:
            # No handle exists yet to own these files if spawning fails.
            out.close()
            err.close()
            raise
        handle = _HostCommandHandle(self._control, self._sandbox, process, out, err)
        self.handles.append(handle)
        return handle

    async def kill(self, pid: int, request_timeout: float | None = None) -> bool:
        del request_timeout
        await anyio.lowlevel.checkpoint()
        self._sandbox.check_alive()
        self.killed_pids.append(pid)
        handle = next(handle for handle in self.handles if handle.pid == pid)
        assert isinstance(handle, _HostCommandHandle)
        handle.close()
        return True


@contextmanager
def _host_errors(path: str) -> Generator[None]:
    """Raise the SDK exceptions envd's status codes turn into for these host errors."""
    try:
        yield
    except FileNotFoundError as e:
        raise FileNotFoundException(f"path '{path}' does not exist") from e
    except IsADirectoryError as e:
        raise InvalidArgumentException(f"path '{path}' is a directory") from e
    except (NotADirectoryError, FileExistsError) as e:
        raise InvalidArgumentException(f"path '{path}' is not a directory") from e


class _HostFilesystem(FakeFilesystem):
    """Mirrors `sandbox.files` on the host filesystem, so commands and file calls share one tree."""

    async def read(
        self,
        path: str,
        format: Literal['bytes'],
        user: str | None = None,
        request_timeout: float | None = None,
        gzip: bool = False,
    ) -> bytearray:
        del user, request_timeout, gzip
        await self._check(path)
        assert format == 'bytes', f'unexpected read format {format!r}'
        with _host_errors(path):
            return bytearray(Path(path).read_bytes())

    async def write(
        self,
        path: str,
        data: str | bytes,
        user: str | None = None,
        request_timeout: float | None = None,
    ) -> WriteInfo:
        del user, request_timeout
        await self._check(path)
        with _host_errors(path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(data.encode() if isinstance(data, str) else data)
        return WriteInfo(name=posixpath.basename(path), type=FileType.FILE, path=path)

    async def get_info(self, path: str, user: str | None = None, request_timeout: float | None = None) -> FakeEntryInfo:
        del user, request_timeout
        await self._check(path)
        with _host_errors(path):
            return self._host_entry(path)

    async def list(
        self,
        path: str,
        depth: int | None = 1,
        user: str | None = None,
        request_timeout: float | None = None,
    ) -> list[FakeEntryInfo]:
        del user, request_timeout
        await self._check(path)
        assert depth == 1, f'unexpected list depth {depth!r}'
        with _host_errors(path):
            children = sorted(Path(path).iterdir())
        return [self._host_entry(posixpath.join(path, child.name)) for child in children]

    async def exists(self, path: str, user: str | None = None, request_timeout: float | None = None) -> bool:
        del user, request_timeout
        await self._check(path)
        return Path(path).exists()

    async def make_dir(self, path: str, user: str | None = None, request_timeout: float | None = None) -> bool:
        del user, request_timeout
        await self._check(path)
        created = not Path(path).is_dir()
        Path(path).mkdir(parents=True, exist_ok=True)
        return created

    async def remove(self, path: str, user: str | None = None, request_timeout: float | None = None) -> None:
        del user, request_timeout
        await self._check(path)
        # envd's `os.RemoveAll`: recursive, and a missing path is not an error.
        if Path(path).is_dir():
            shutil.rmtree(path)
        else:
            Path(path).unlink(missing_ok=True)

    def _host_entry(self, path: str) -> FakeEntryInfo:
        # Like envd: the link's own size, and for a symlink the fully resolved target (the link
        # itself when that fails) and the target's type (none when the target is missing).
        info = Path(path).lstat()
        kind: FileType | None = FileType.DIR if stat.S_ISDIR(info.st_mode) else FileType.FILE
        target = None
        if stat.S_ISLNK(info.st_mode):
            try:
                target = str(Path(path).resolve(strict=True))
            except (OSError, RuntimeError):
                target = path
            kind = (FileType.DIR if Path(target).is_dir() else FileType.FILE) if Path(target).exists() else None
        return FakeEntryInfo(
            name=posixpath.basename(path), path=path, type=kind, size=info.st_size, symlink_target=target
        )


class FakeSandbox:
    """Mirrors `e2b.AsyncSandbox` for the members the backend uses."""

    def __init__(self, control: FakeE2B, sandbox_id: str) -> None:
        self._control = control
        self.sandbox_id = sandbox_id
        self.files = FakeFilesystem(self, control)
        self.commands = FakeCommands(self, control)
        if control.host_root is not None:
            self.files = _HostFilesystem(self, control)
            self.commands = _HostCommands(self, control)
        self.killed = False

    async def kill(self) -> bool:
        # Real `kill` is a DELETE on the control plane that returns once E2B has removed the
        # sandbox, and `False` when it was already gone (a 404). From then on the SDK's own
        # docs show `is_running()` as `False`.
        await anyio.lowlevel.checkpoint()
        if self.killed:
            return False
        self.killed = True
        for handle in self.commands.handles:
            if isinstance(handle, _HostCommandHandle):
                handle.close()
        return True

    def check_alive(self) -> None:
        """Fail an envd call the way the SDK does once the sandbox is gone.

        E2B's proxy answers a request for a killed sandbox with a 502, which the SDK raises as
        a `TimeoutException` blaming the sandbox timeout -- the same type it uses for a slow
        request, so only the health probe tells the two apart.
        """
        if self.killed:
            raise format_sandbox_timeout_exception('The sandbox was not found')

    async def is_running(self, request_timeout: float | None = None) -> bool:
        del request_timeout
        await anyio.lowlevel.checkpoint()
        if self._control.is_running_error is not None:
            raise self._control.is_running_error
        return self._control.sandbox_is_running and not self.killed


class FakeAsyncSandboxFactory:
    """Mirrors the `AsyncSandbox.create` / `AsyncSandbox.connect` class methods."""

    def __init__(self, control: FakeE2B) -> None:
        self._control = control

    async def create(
        self,
        template: str | None = None,
        timeout: int | None = None,
        envs: dict[str, str] | None = None,
        secure: bool = True,
        allow_internet_access: bool = True,
        lifecycle: dict[str, str] | None = None,
    ) -> FakeSandbox:
        self._control.create_calls.append(
            FakeCreateCall(template, timeout, envs, secure, allow_internet_access, lifecycle)
        )
        if self._control.create_hangs:
            await anyio.sleep_forever()
        await anyio.lowlevel.checkpoint()
        if self._control.create_error is not None:
            raise self._control.create_error
        # E2B has created the sandbox before its response reaches the caller.
        sandbox = self._control.new_sandbox(f'sbx-{len(self._control.sandboxes) + 1}')
        if self._control.create_response_held is not None:
            await self._control.create_response_held.wait()
        return sandbox

    async def kill(self, id: str) -> bool:
        existing = next((sandbox for sandbox in self._control.sandboxes if sandbox.sandbox_id == id), None)
        return await existing.kill() if existing is not None else False

    async def connect(self, id: str, timeout: int | None = None) -> FakeSandbox:
        self._control.connect_calls.append((id, timeout))
        await anyio.lowlevel.checkpoint()
        if self._control.connect_error is not None:
            raise self._control.connect_error
        existing = next((sandbox for sandbox in self._control.sandboxes if sandbox.sandbox_id == id), None)
        if existing is not None and existing.killed:
            # The connect endpoint 404s for a killed sandbox, which the SDK raises like this.
            raise SandboxNotFoundException(f'Paused sandbox {id} not found')
        if existing is not None:
            return existing
        return self._control.new_sandbox(id)


@dataclass
class FakeE2B:
    """Control surface for the injected fake `e2b` module."""

    responder: Responder = _echo_responder
    sandboxes: list[FakeSandbox] = field(default_factory=list[FakeSandbox])
    create_calls: list[FakeCreateCall] = field(default_factory=list[FakeCreateCall])
    connect_calls: list[tuple[str, int | None]] = field(default_factory=list[tuple[str, 'int | None']])
    create_error: Exception | None = None
    create_hangs: bool = False
    # When set, `create` makes the sandbox and then holds its response until the event is set.
    create_response_held: anyio.Event | None = None
    connect_error: Exception | None = None
    run_error: Exception | None = None
    wait_error: Exception | None = None
    command_hangs: bool = False
    # When set, `commands.run` starts the command and then holds its response until the event is set.
    start_response_held: anyio.Event | None = None
    kill_command_error: Exception | None = None
    fs_error: Exception | None = None
    read_error: Exception | None = None
    is_running_error: Exception | None = None
    sandbox_is_running: bool = True
    next_pid: int = 4242
    # When set, sandboxes run commands and file operations on the host under this directory.
    host_root: Path | None = None

    def __post_init__(self) -> None:
        self.module = self._build_module()

    def close(self) -> None:
        """Reap host commands even when a test leaves the SDK handle unwaited."""
        for sandbox in self.sandboxes:
            for handle in sandbox.commands.handles:
                handle.close()

    def new_sandbox(self, sandbox_id: str) -> FakeSandbox:
        sandbox = FakeSandbox(self, sandbox_id)
        self.sandboxes.append(sandbox)
        return sandbox

    def _build_module(self) -> types.ModuleType:
        module = types.ModuleType('e2b')
        module.AsyncSandbox = FakeAsyncSandboxFactory(self)  # type: ignore[attr-defined]
        # The real classes, so the backend's `isinstance` checks and its unwrapping of a
        # non-zero exit run against exactly what production raises.
        module.AuthenticationException = AuthenticationException  # type: ignore[attr-defined]
        module.CommandExitException = CommandExitException  # type: ignore[attr-defined]
        module.CommandResult = CommandResult  # type: ignore[attr-defined]
        module.FileNotFoundException = FileNotFoundException  # type: ignore[attr-defined]
        module.FileType = FileType  # type: ignore[attr-defined]
        module.InvalidArgumentException = InvalidArgumentException  # type: ignore[attr-defined]
        module.RateLimitException = RateLimitException  # type: ignore[attr-defined]
        module.SandboxException = SandboxException  # type: ignore[attr-defined]
        module.SandboxNotFoundException = SandboxNotFoundException  # type: ignore[attr-defined]
        module.TimeoutException = TimeoutException  # type: ignore[attr-defined]
        return module


if TYPE_CHECKING:
    import e2b

    class _EntryInfoSurface(Protocol):
        """The `e2b.EntryInfo` members the backend reads.

        Pinned against both the fake and the real SDK type below, so a fake that drifts from
        E2B's own entry shape fails the type check instead of at the next live run.
        """

        @property
        def name(self) -> str: ...

        @property
        def path(self) -> str: ...

        @property
        def type(self) -> FileType | None: ...

        @property
        def size(self) -> int: ...

        @property
        def symlink_target(self) -> str | None: ...

    class _CommandHandleSurface(Protocol):
        """The `e2b.AsyncCommandHandle` members the backend reads.

        `stdout` / `stderr` are the accumulated output the deadline path reports, so a fake
        that stopped exposing them would hide the timeout behavior entirely.
        """

        @property
        def pid(self) -> int: ...

        @property
        def stdout(self) -> str: ...

        @property
        def stderr(self) -> str: ...

        async def wait(self) -> CommandResult: ...

    _fake_entry_conforms: _EntryInfoSurface = FakeEntryInfo(name='n', path='/n', type=FileType.FILE, size=0)
    _real_entry_conforms: _EntryInfoSurface = e2b.EntryInfo.__new__(e2b.EntryInfo)
    _fake_handle_conforms: _CommandHandleSurface = FakeCommandHandle(
        FakeE2B(), FakeSandbox(FakeE2B(), 'sbx'), pid=1, stdout='', stderr='', exit_code=0
    )
    _real_handle_conforms: _CommandHandleSurface = e2b.AsyncCommandHandle.__new__(e2b.AsyncCommandHandle)
