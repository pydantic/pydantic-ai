"""A fake `smolvm` CLI whose machines run each command on the host, so `SmolvmCliBackend` runs without a hypervisor.

It answers as smolvm 1.36 does (see `pydantic_ai_harness._smolvm_cli`'s module docstring): `machine exec` runs the
script it reads on its stdin in `sh`, and smolvm's own failures end with a `--error-format json` report on stderr.
A command runs in its own process group, which killing the CLI kills, as smolvm ends a command whose CLI goes.
Every other answer is canned, in this process: only a command needs a process of its own.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

import anyio
import anyio.abc
import pytest
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

_EXEC = (
    # `sh -s`, keeping what it read. A machine deleted meanwhile is reported as smolvm reports it.
    'cat > "$1"; sh "$1"; code=$?; if [ -e "$2" ]; then printf "%s\\n" "$3" >&2; exit 1; fi; exit "$code"'
)


@dataclass
class FakeProcess:
    """The CLI's process; `kill` ends its whole group, the command in the machine included."""

    process: anyio.abc.Process

    @property
    def stdin(self) -> anyio.abc.ByteSendStream | None:
        return self.process.stdin

    @property
    def stdout(self) -> anyio.abc.ByteReceiveStream | None:
        return self.process.stdout

    @property
    def stderr(self) -> anyio.abc.ByteReceiveStream | None:
        return self.process.stderr

    def kill(self) -> None:
        with suppress(ProcessLookupError):
            os.killpg(self.process.pid, signal.SIGKILL)

    async def wait(self) -> int:
        return await self.process.wait()

    async def aclose(self) -> None:
        # The command holds the CLI's stdout and stderr, as a command in a smolvm VM does not, so after `kill` they
        # reach EOF only once the whole group has exited, which can be after the CLI's own exit is reported. Before
        # anyio 4.14.1 (agronholm/anyio#1166), `Process.aclose()` returns at that exit without closing them, and
        # asyncio closes the process's transport only once they have closed, so a test could end with it open.
        # Read them to EOF first.
        for stream in (self.stdout, self.stderr):
            assert stream is not None
            with suppress(anyio.EndOfStream):
                while True:
                    await stream.receive()
        await self.process.aclose()


class CannedProcess:
    """An answer of smolvm's that runs nothing, such as `machine ls`: its output, already written, and its exit code."""

    def __init__(self, code: int, stdout: bytes, stderr: bytes, *, stdin: bool) -> None:
        self._code = code
        self.stdout = _written(stdout)
        self.stderr = _written(stderr)
        # Takes whatever `machine exec` is sent and drops it, as smolvm does when it fails before running anything.
        self.stdin: MemoryObjectSendStream[bytes] | None = None
        self._unread: MemoryObjectReceiveStream[bytes] | None = None
        if stdin:
            self.stdin, self._unread = anyio.create_memory_object_stream[bytes](math.inf)

    # Nothing kills an answer that is already written.
    def kill(self) -> None:  # pragma: no cover
        pass

    async def wait(self) -> int:
        return self._code

    async def aclose(self) -> None:
        for stream in (self.stdout, self.stderr, self.stdin, self._unread):
            if stream is not None:
                await stream.aclose()


def _written(data: bytes) -> MemoryObjectReceiveStream[bytes]:
    send, receive = anyio.create_memory_object_stream[bytes](1)
    if data:
        send.send_nowait(data)
    send.close()
    return receive


@dataclass
class FakeSmolvm:
    """The control surface: machines and their states, the calls made, and failures to inject."""

    scratch: Path
    """Where each `machine exec` leaves the script it read, and a deleted machine its marker."""
    machines: dict[str, str] = field(default_factory=dict[str, str])
    """Each machine's state, by name."""
    created: list[list[str]] = field(default_factory=list[list[str]])
    """The arguments of each `machine create`."""
    commands: list[list[str]] = field(default_factory=list[list[str]])
    """Every command line run on this machine."""
    failures: dict[str, tuple[str, str]] = field(default_factory=dict[str, tuple[str, str]])
    """The code and message each action named here fails with, such as `start` on a host that cannot run VMs."""
    wrapped_listing: bool = False
    """Whether `machine ls --json` puts its list under `machines`, as some versions do; 1.36 prints it bare."""
    workdirs: dict[str, set[str]] = field(default_factory=dict[str, set[str]])
    """The working directories each machine's commands ran in: its filesystem, which deleting it removes."""

    @property
    def scripts(self) -> list[str]:
        """What each `machine exec` read on its stdin."""
        return [path.read_text() for path in sorted(self.scratch.glob('*.sh'))]

    async def open_process(self, command: list[str], *, stdin: int) -> FakeProcess | CannedProcess:
        self.commands.append(command)
        executable, group, action, *args = command
        # Found in an absolute `PATH` directory, never in the current one.
        assert Path(executable).is_absolute() and Path(executable).stem == 'smolvm', executable
        assert (group, args[:2]) == ('machine', ['--error-format', 'json'])
        args = args[2:]
        if action == 'ls':
            machines = [{'name': name, 'state': state} for name, state in self.machines.items()]
            return await self._reply(stdin, 0, json.dumps({'machines': machines} if self.wrapped_listing else machines))
        name = args[args.index('--name') + 1]
        if action in self.failures:
            return await self._error(stdin, *self.failures[action])
        if action == 'create':
            self.created.append(args)
            self.machines[name] = 'stopped'
        elif name not in self.machines:
            return await self._error(stdin, 'NOT_FOUND', f'machine not found: {name}')
        elif action in ('start', 'resume', 'stop'):
            self.machines[name] = 'stopped' if action == 'stop' else 'running'
        elif action == 'delete':
            self.destroy(name)
        elif self.machines[name] != 'running':
            return await self._error(stdin, 'SMOLVM_ERROR', f'machine {name} is not running')
        else:
            return await self._exec(name, args, stdin)
        return await self._reply(stdin, 0)

    def destroy(self, name: str) -> None:
        """Delete a machine, and with it its filesystem, which ends a command watching its working directory."""
        del self.machines[name]
        (self.scratch / f'{name}.gone').touch()
        for workdir in self.workdirs.pop(name, set()):
            shutil.rmtree(workdir, ignore_errors=True)

    async def _exec(self, name: str, args: list[str], stdin: int) -> FakeProcess:
        assert stdin == subprocess.PIPE and args[args.index('--') + 1 :] == ['/bin/sh', '-s']
        workdir = args[args.index('--workdir') + 1] if '--workdir' in args else None
        if workdir is not None:
            self.workdirs.setdefault(name, set()).add(workdir)
        script = self.scratch / f'{len(self.commands):04}.sh'
        gone = json.dumps({'error': {'code': 'SMOLVM_ERROR', 'message': 'io operation failed: connection closed'}})
        run = ['sh', '-c', _EXEC, 'sh', str(script), str(self.scratch / f'{name}.gone'), gone]
        # A machine's commands get only what smolvm gives them, not this machine's environment.
        env = {'PATH': os.environ['PATH'], 'HOME': os.environ['HOME']}
        return FakeProcess(await anyio.open_process(run, stdin=stdin, cwd=workdir, env=env, start_new_session=True))

    async def _reply(self, stdin: int, code: int, stdout: str = '', stderr: str = '') -> CannedProcess:
        return CannedProcess(code, stdout.encode(), stderr.encode(), stdin=stdin == subprocess.PIPE)

    async def _error(self, stdin: int, code: str, message: str) -> CannedProcess:
        report = json.dumps({'error': {'code': code, 'message': message}})
        return await self._reply(stdin, 1, stderr=f'Error: failed\n{report}')


def install_fake_smolvm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeSmolvm:
    """Make `SmolvmCliBackend` run the fake, found in an absolute `PATH` directory, and return its control surface."""
    scratch = tmp_path / 'smolvm-scripts'
    scratch.mkdir()
    control = FakeSmolvm(scratch)
    monkeypatch.setattr('pydantic_ai_harness._smolvm_cli.open_process', control.open_process)
    monkeypatch.setenv('PATH', f'{put_smolvm_in(tmp_path / "smolvm-bin")}{os.pathsep}{os.environ["PATH"]}')
    return control


def put_smolvm_in(directory: Path) -> Path:
    """Make `directory` hold a `smolvm`, and a `smolvm.exe` for a Windows host, for `find_cli` to find."""
    directory.mkdir()
    for name in ('smolvm', 'smolvm.exe'):
        (directory / name).touch(mode=0o755)
    return directory
