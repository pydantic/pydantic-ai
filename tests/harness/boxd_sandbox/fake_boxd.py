"""Controllable fake for the boxd SDK boundary.

The exception classes, `Machine` records and `OutputChunk`s are the real ones from the installed SDK,
so the backend's `isinstance` and field checks run against what production raises and returns.

Commands run on the host under `host_root`, through `/bin/sh -c` in a session of their own, as boxd
runs them in the machine: the exit status arrives as soon as the command exits, even while a background
child holds its output open; closing the stream sends the command's process group SIGTERM and returns
without waiting for it to end; and a command killed by a signal exits 128 plus its number. Deleting a
machine kills its commands, and a command that ends while its machine is deleted or stopped raises
`APIConnectionError`, as production's does. A machine created from a snapshot keeps the snapshot's networks, isolated or not.
A fork copies the machine's record, not its files: every machine runs its commands in `host_root`.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from collections.abc import AsyncIterator, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
from boxd import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    BoxdError,
    ConflictError,
    ExecResult,
    Machine,
    MachineNetworking,
    NotFoundError,
    OutputChunk,
)
from boxd.models import OrgRef

# Output a command writes after it exits is read for this long, then dropped.
_DRAIN_GRACE = 0.05
_EXIT_POLL = 0.01
_SIZES = (1, 2, 4, 8)


def _not_found(machine_id: str) -> NotFoundError:
    return NotFoundError(f'VM {machine_id!r} not found', code='not_found', grpc_code=5)


async def wait_until_gone(pid_file: Path) -> None:
    """Wait for the host process whose pid is in `pid_file` to end; boxd stops a command without waiting for it."""
    pid = int(pid_file.read_text(encoding='utf-8'))
    with anyio.fail_after(10):
        while True:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            await anyio.sleep(0.01)  # pragma: lax no cover - depends on how fast the process dies


async def _exited(process: asyncio.subprocess.Process) -> None:
    """Wait for `process` to exit; `Process.wait()` also waits for a background child to close the pipes."""
    while process.returncode is None:
        await asyncio.sleep(_EXIT_POLL)


@dataclass
class FakeMachine:
    id: str
    name: str
    status: str = 'running'
    isolated: bool = True
    networks: list[str] = field(default_factory=list[str])
    create_kwargs: dict[str, Any] = field(default_factory=dict[str, Any])
    # Lookups that report the machine as `starting` before it runs, after a create or a start.
    boot_polls: int = 0
    # Lookups that still report a deleted machine as running, as production does for a moment.
    delete_lag: int = 0
    processes: list[asyncio.subprocess.Process] = field(default_factory=list[asyncio.subprocess.Process])
    starts: int = 0
    org: str | None = None
    # A copy's source machine: its id, and its status when it was forked.
    forked_from: tuple[str, str] | None = None

    def record(self, status: str | None = None) -> Machine:
        return Machine.model_construct(
            id=self.id,
            name=self.name,
            status=status or self.status,
            networking=MachineNetworking(networks=self.networks, isolated=self.isolated),
            org=None if self.org is None else OrgRef(id=f'org-{self.org}', name=self.org),
        )


class FakeExecStream:
    """`boxd.AsyncExecStream` over a host process."""

    def __init__(self, owner: FakeBoxd, machine: FakeMachine, line: str) -> None:
        self._owner = owner
        self._machine = machine
        self._line = line
        self._process: asyncio.subprocess.Process | None = None
        self._exit_code: int | None = None

    async def __aenter__(self) -> FakeExecStream:
        owner = self._owner
        owner.check_call()
        owner.check_runnable(self._machine)
        owner.lines.append(self._line)
        self._process = await asyncio.create_subprocess_exec(
            '/bin/sh',
            '-c',
            self._line,
            cwd=owner.host_root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        self._machine.processes.append(self._process)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def iter_chunks(self) -> AsyncIterator[OutputChunk]:
        process = self._process
        assert process is not None and process.stdout is not None and process.stderr is not None
        chunks: asyncio.Queue[OutputChunk] = asyncio.Queue()

        async def read(pipe: asyncio.StreamReader, is_stderr: bool) -> None:
            while data := await pipe.read(65536):
                await chunks.put(OutputChunk(data, is_stderr))

        readers = [asyncio.create_task(read(process.stdout, False)), asyncio.create_task(read(process.stderr, True))]
        self._owner.readers.extend(readers)
        exited = asyncio.create_task(_exited(process))
        try:
            while not exited.done():
                getter = asyncio.create_task(chunks.get())
                await asyncio.wait([getter, exited], return_when=asyncio.FIRST_COMPLETED)
                if getter.done():
                    yield getter.result()
                else:
                    getter.cancel()
            # A background child can hold the pipes open; read on briefly, as production returns at exit.
            await asyncio.wait(readers, timeout=_DRAIN_GRACE)
            while not chunks.empty():
                yield chunks.get_nowait()
        finally:
            for task in (*readers, exited):
                task.cancel()
        if self._owner.stream_error is not None:
            raise self._owner.stream_error
        if self._machine.status in ('destroyed', 'stopped'):
            raise APIConnectionError(
                f'lost the connection to the machine during exec: the machine was {self._machine.status}'
            )
        returncode = process.returncode
        assert returncode is not None
        if not self._owner.drop_exit_status:
            self._exit_code = returncode if returncode >= 0 else 128 - returncode

    @property
    def exit_code(self) -> int | None:
        return self._exit_code

    async def close(self) -> None:
        self._owner.closed_streams += 1
        process = self._process
        if process is not None and process.returncode is None:
            os.killpg(process.pid, signal.SIGTERM)


class FakeMachines:
    def __init__(self, owner: FakeBoxd) -> None:
        self._owner = owner

    async def create(
        self,
        name: str | None = None,
        *,
        org: str | None = None,
        from_snapshot: str | None = None,
        isolated: bool = False,
        vcpu: int | None = None,
        auto_suspend_timeout: int | None = None,
    ) -> Machine:
        owner = self._owner
        owner.check_call()
        owner.create_names.append(name)
        if owner.create_gate is not None and not owner.create_then_hang:
            await owner.create_gate.wait()
        if owner.create_errors:
            raise owner.create_errors.pop(0)
        if any(machine.name == name for machine in owner.machines.values()):
            raise ConflictError(f'name {name!r} is already taken', code='failed_precondition', grpc_code=9)
        if from_snapshot is not None and from_snapshot not in owner.snapshots:
            raise NotFoundError('snapshot not found', code='not_found', grpc_code=5)
        networks = [] if from_snapshot is None else owner.snapshots[from_snapshot]
        if vcpu is not None and vcpu not in _SIZES:
            raise APIStatusError(f'no machine size has {vcpu} vCPU', code='invalid_argument', grpc_code=3)
        kwargs = {
            'org': org,
            'from_snapshot': from_snapshot,
            'isolated': isolated,
            'vcpu': vcpu,
            'auto_suspend_timeout': auto_suspend_timeout,
        }
        machine = owner.add(
            name=name, isolated=isolated, networks=networks, create_kwargs=kwargs, boot_polls=owner.boot_polls
        )
        if owner.create_gate is not None and owner.create_then_hang:
            await owner.create_gate.wait()
        if owner.lose_create_reply:
            owner.lose_create_reply = False
            raise APIConnectionError('the connection dropped before the reply arrived')
        return machine.record()

    async def fork(self, source: str, name: str | None = None, *, isolated: bool = False) -> Machine:
        owner = self._owner
        owner.check_call()
        if owner.fork_errors:
            raise owner.fork_errors.pop(0)
        origin = owner.find(source)
        machine = owner.add(
            name=name, isolated=origin.isolated or isolated, networks=list(origin.networks), org=origin.org
        )
        machine.forked_from = (origin.id, origin.status)
        if owner.lose_fork_reply:
            owner.lose_fork_reply = False
            raise APIConnectionError('the connection dropped before the reply arrived')
        return machine.record()

    async def list(self, *, org: str | None = None) -> list[Machine]:
        owner = self._owner
        owner.check_call()
        owner.listed_orgs.append(org)
        return [machine.record() for machine in owner.machines.values() if machine.status != 'destroyed']

    async def get(self, machine_id: str) -> Machine:
        owner = self._owner
        owner.check_call()
        if owner.get_errors:
            raise owner.get_errors.pop(0)
        machine = owner.find(machine_id)
        if machine.status == 'destroyed':
            if machine.delete_lag:
                machine.delete_lag -= 1
                return machine.record('running')
            if not owner.report_destroyed:
                raise _not_found(machine_id)
        elif machine.boot_polls:
            machine.boot_polls -= 1
            return machine.record('starting')
        return machine.record()

    async def start(self, machine_id: str) -> None:
        owner = self._owner
        owner.check_call()
        machine = owner.find(machine_id)
        machine.starts += 1
        if owner.start_errors:
            raise owner.start_errors.pop(0)
        machine.status = 'running'
        machine.boot_polls = owner.boot_polls

    async def delete(self, machine_id: str) -> None:
        owner = self._owner
        owner.check_call()
        machine = owner.find(machine_id)
        machine.status = 'destroyed'
        machine.delete_lag = owner.delete_lag
        for process in machine.processes:
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGKILL)

    async def exec(self, machine_id: str, command: str | Sequence[str], *, timeout: float | None = None) -> ExecResult:
        owner = self._owner
        owner.check_call()
        machine = owner.find(machine_id)
        owner.check_runnable(machine)
        assert isinstance(command, str)
        owner.exec_commands.append(command)
        if command == 'true' and owner.probe_errors:
            raise owner.probe_errors.pop(0)
        process = await asyncio.create_subprocess_exec(
            '/bin/sh',
            '-c',
            command,
            cwd=owner.host_root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        assert process.returncode is not None
        return ExecResult(stdout=stdout.decode(), stderr=stderr.decode(), exit_code=process.returncode)

    def stream_exec(
        self, machine_id: str, *, command: str | Sequence[str], close_stdin: bool = False
    ) -> FakeExecStream:
        assert isinstance(command, str) and close_stdin
        return FakeExecStream(self._owner, self._owner.find(machine_id), command)


class FakeClient:
    """`boxd.AsyncBoxd`, opened from the environment or passed in."""

    def __init__(self, owner: FakeBoxd) -> None:
        self._owner = owner
        self.machines = FakeMachines(owner)
        self.closes = 0

    @property
    def closed(self) -> bool:
        return self.closes > 0

    async def close(self) -> None:
        self.closes += 1
        if self._owner.close_gate is not None:
            await self._owner.close_gate.wait()
        if self._owner.close_error is not None:
            raise self._owner.close_error


class FakeBoxd:
    """The fake SDK's control surface: its machines, the clients it opened, and failures to inject."""

    def __init__(self) -> None:
        self.machines: dict[str, FakeMachine] = {}
        self.clients: list[FakeClient] = []
        # Snapshot names, each with the networks a machine created from it keeps.
        self.snapshots: dict[str, list[str]] = {}
        self.host_root: Path | None = None
        self.create_names: list[str | None] = []
        self.listed_orgs: list[str | None] = []
        self.lines: list[str] = []
        self.exec_commands: list[str] = []
        # Every command stream's pipe readers, which `reap` finishes before it reads the pipes itself.
        self.readers: list[asyncio.Task[None]] = []
        self.closed_streams = 0
        self.boot_polls = 0
        self.delete_lag = 0
        self.reject_credentials = False
        self.report_destroyed = False
        self.lose_create_reply = False
        self.lose_fork_reply = False
        # With `create_gate`: create the machine, then wait, instead of waiting before creating it.
        self.create_then_hang = False
        self.drop_exit_status = False
        self.create_errors: list[BoxdError] = []
        self.fork_errors: list[BoxdError] = []
        self.get_errors: list[BaseException] = []
        self.probe_errors: list[BoxdError] = []
        self.start_errors: list[BoxdError] = []
        self.stream_error: BoxdError | None = None
        self.close_error: Exception | None = None
        self.create_gate: asyncio.Event | None = None
        self.close_gate: asyncio.Event | None = None

    def client(self) -> FakeClient:
        client = FakeClient(self)
        self.clients.append(client)
        return client

    def add(
        self,
        *,
        name: str | None = None,
        status: str = 'running',
        isolated: bool = True,
        networks: list[str] | None = None,
        create_kwargs: dict[str, Any] | None = None,
        boot_polls: int = 0,
        org: str | None = None,
    ) -> FakeMachine:
        machine_id = f'machine-{len(self.machines) + 1}'
        machine = FakeMachine(
            machine_id,
            name or machine_id,
            status=status,
            isolated=isolated,
            networks=networks or [],
            create_kwargs=create_kwargs or {},
            boot_polls=boot_polls,
            org=org,
        )
        self.machines[machine_id] = machine
        return machine

    async def reap(self) -> None:
        """Kill what the test left running and close its pipes, so nothing outlives the test's event loop.

        `close()` only signals a command, as production does, so one can still be ending when the test
        returns. One that has ended can still have output in its pipes, and asyncio keeps a pipe open
        until it is read to the end, so each one is read here.
        """
        processes = [process for machine in self.machines.values() for process in machine.processes]
        if not processes:
            return  # Nothing to close, and a test under Trio has no asyncio loop to close it on.
        for task in self.readers:
            task.cancel()
        await asyncio.gather(*self.readers, return_exceptions=True)
        for process in processes:
            assert process.stdout is not None and process.stderr is not None
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            with anyio.fail_after(10):
                await process.stdout.read()
                await process.stderr.read()
                await _exited(process)
        # asyncio closes a finished process's transport in callbacks on the loop's next turns.
        await asyncio.sleep(_EXIT_POLL)

    def find(self, machine_id: str) -> FakeMachine:
        for machine in self.machines.values():
            if machine_id in (machine.id, machine.name):
                return machine
        raise _not_found(machine_id)

    def check_call(self) -> None:
        if self.reject_credentials:
            raise AuthenticationError('invalid or expired API key')

    def check_runnable(self, machine: FakeMachine) -> None:
        if machine.status == 'destroyed':
            raise NotFoundError('VM is destroyed', code='not_found', grpc_code=5)
        if machine.status == 'stopped':
            raise ConflictError(
                f"VM {machine.name} is stopped, so it won't be resumed automatically.",
                code='failed_precondition',
                grpc_code=9,
            )
        # A command wakes a suspended or hibernated machine.
        machine.status = 'running'

    @property
    def only(self) -> FakeMachine:
        """The one machine the test made, created or added."""
        (machine,) = self.machines.values()
        return machine
