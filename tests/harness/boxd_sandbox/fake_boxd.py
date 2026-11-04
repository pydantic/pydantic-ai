"""Controllable fake for the boxd SDK boundary.

The exception classes, `Machine` records and the command stream (`AsyncExecStream`) are the real ones from the installed SDK,
so the backend's `isinstance` and field checks run against what production raises and returns.

Commands run on the host under `host_root`, through `/bin/sh -c` in a session of their own, as boxd
runs them in the machine: the exit status arrives as soon as the command exits, even while a background
child holds its output open; closing the stream sends the command's process group SIGTERM and returns
without waiting for it to end; and a command killed by a signal exits 128 plus its number. Deleting a
machine kills its commands, and a command that ends while its machine is deleted or stopped raises
`APIConnectionError`, as production's does. A machine created from a snapshot keeps the snapshot's networks, isolated or not.
A fork copies the machine's record, not its files: every machine runs its commands in `host_root`.

File operations run the SDK's own `AsyncFs` over the host's files, so machine paths are host paths. Each
call fails as boxd's does: a gRPC status with the reason in its trailer, which the SDK's `from_grpc_error`
turns into the exception production raises.
"""

from __future__ import annotations

import asyncio
import errno
import os
import shutil
import signal
import stat
import subprocess
from collections.abc import AsyncIterator, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import anyio
import grpc
from boxd import (
    APIConnectionError,
    APIStatusError,
    AsyncExecStream,
    AuthenticationError,
    BoxdError,
    ConflictError,
    ExecResult,
    Machine,
    MachineNetworking,
    NotFoundError,
)
from boxd._generated import api_pb2
from boxd.errors import REASON_TRAILER, from_grpc_error
from boxd.models import OrgRef
from boxd.resources.fs import AsyncFs
from boxd.resources.machines import AsyncMachines

# Output a command writes after it exits is read for this long, then dropped.
_DRAIN_GRACE = 0.05
# boxd's generated protobuf module has no type stubs.
_pb: Any = cast(Any, api_pb2)
_ExecChunk: Any = _pb.ExecChunk
_EXIT_POLL = 0.01
_SIZES = (1, 2, 4, 8)
# A file failure's errno, as the reason boxd names it and the status it sends.
_FS_FAILURES = {
    errno.ENOENT: ('not_found', grpc.StatusCode.NOT_FOUND),
    errno.EACCES: ('permission_denied', grpc.StatusCode.PERMISSION_DENIED),
    errno.EEXIST: ('file_exists', grpc.StatusCode.ALREADY_EXISTS),
    errno.EISDIR: ('is_a_directory', grpc.StatusCode.FAILED_PRECONDITION),
    errno.ENOTDIR: ('not_a_directory', grpc.StatusCode.FAILED_PRECONDITION),
}
_FILE, _DIRECTORY, _SYMLINK, _OTHER = 1, 2, 3, 4


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


class _FakeExecCall:
    """boxd's `Exec` gRPC call over a host process, under the SDK's own `AsyncExecStream`.

    It sends what boxd's server sends: an `ExecChunk` per piece of output, whose `exit_code` is the
    proto default 0, then a last chunk carrying the exit status.
    """

    def __init__(self, owner: FakeBoxd, requests: AsyncIterator[Any]) -> None:
        self._owner = owner
        self._requests = requests
        self._process: asyncio.subprocess.Process | None = None

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._chunks()

    async def _chunks(self) -> AsyncIterator[Any]:
        owner = self._owner
        init = await anext(aiter(self._requests))
        machine = owner.find(init.vm_id)
        owner.check_runnable(machine)
        owner.lines.append(init.command)
        process = self._process = await asyncio.create_subprocess_exec(
            '/bin/sh',
            '-c',
            init.command,
            cwd=owner.host_root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        machine.processes.append(process)
        assert process.stdout is not None and process.stderr is not None
        # `None` ends the stream, so no chunk depends on whether it was read before or after exit.
        chunks: asyncio.Queue[Any] = asyncio.Queue()

        async def read(pipe: asyncio.StreamReader, is_stderr: bool) -> None:
            while data := await pipe.read(65536):
                await chunks.put(_ExecChunk(data=data, is_stderr=is_stderr))

        async def end_after_exit() -> None:
            await _exited(process)
            # A background child can hold the pipes open; read on briefly, as production returns at exit.
            await asyncio.wait(readers, timeout=_DRAIN_GRACE)
            await chunks.put(None)

        readers = [asyncio.create_task(read(process.stdout, False)), asyncio.create_task(read(process.stderr, True))]
        owner.readers.extend(readers)
        ender = asyncio.create_task(end_after_exit())
        try:
            while (chunk := await chunks.get()) is not None:
                yield chunk
        finally:
            for task in (*readers, ender):
                task.cancel()
        if owner.stream_error is not None:
            raise owner.stream_error
        if machine.status in ('destroyed', 'stopped', 'failed'):
            raise APIConnectionError(
                f'lost the connection to the machine during exec: the machine was {machine.status}'
            )
        returncode = process.returncode
        assert returncode is not None
        if not owner.drop_exit_status:
            yield _ExecChunk(exit_code=returncode if returncode >= 0 else 128 - returncode)

    def cancel(self) -> None:
        process = self._process
        if process is not None and process.returncode is None:
            os.killpg(process.pid, signal.SIGTERM)


class _FakeTransport:
    def __init__(self, owner: FakeBoxd) -> None:
        self._owner = owner

    async def bidi(self, method: str, requests: AsyncIterator[Any]) -> _FakeExecCall:
        assert method == 'Exec'
        self._owner.check_call()
        return _FakeExecCall(self._owner, requests)


class _FakeRpcError(grpc.RpcError):
    """A failed boxd call as gRPC reports it: a status, its details, and the reason trailer."""

    def __init__(self, code: grpc.StatusCode, details: str, reason: str) -> None:
        self._code = code
        self._details = details
        self._reason = reason

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return self._details

    # grpc returns `(key, value)` pairs, which the SDK reads; typeshed's stub types them as objects.
    def trailing_metadata(self) -> Any:
        return ((REASON_TRAILER, self._reason),)


def _fs_failure(error: OSError, path: str) -> BoxdError:
    assert error.errno is not None
    reason, code = _FS_FAILURES[error.errno]
    return from_grpc_error(_FakeRpcError(code, f'{path}: {os.strerror(error.errno)}', reason))


def _type(mode: int) -> int:
    if stat.S_ISREG(mode):
        return _FILE
    if stat.S_ISDIR(mode):
        return _DIRECTORY
    return _SYMLINK if stat.S_ISLNK(mode) else _OTHER


def _entry(path: str) -> Any:
    """`path` as boxd describes it: the entry itself, and the size, mode and owner of what it resolves to."""
    link = os.lstat(path)
    try:
        target = os.stat(path)
    except OSError:
        target = None  # A dangling or looping symlink.
    info = target or link
    return _pb.FsEntry(
        name=os.fsencode(os.path.basename(path)),
        path=os.fsencode(path),
        type=_type(link.st_mode),
        target_type=0 if target is None else _type(target.st_mode),
        size=info.st_size,
        mode=stat.S_IMODE(info.st_mode),
        mtime_ns=info.st_mtime_ns,
        uid=info.st_uid,
        gid=info.st_gid,
        symlink_target=os.fsencode(os.readlink(path)) if stat.S_ISLNK(link.st_mode) else b'',
    )


async def _messages(*messages: Any) -> AsyncIterator[Any]:
    for message in messages:
        yield message


class _FakeFsTransport:
    """boxd's `Fs*` gRPC calls over the host's files, under the SDK's own `AsyncFs`."""

    def __init__(self, owner: FakeBoxd) -> None:
        self._owner = owner

    def _check(self, method: str, machine_id: str) -> None:
        owner = self._owner
        owner.check_call()
        owner.check_runnable(owner.find(machine_id))
        owner.fs_calls.append(method)
        if owner.agent_outdated:
            raise from_grpc_error(
                _FakeRpcError(grpc.StatusCode.FAILED_PRECONDITION, 'the boxd agent is too old', 'agent_outdated')
            )

    async def call(self, method: str, request: Any) -> Any:
        if method == 'FsWrite':
            return await self._write(request)
        self._check(method, request.vm_id)
        path = os.fsdecode(request.path)
        try:
            if method == 'FsStat':
                return _pb.FsStatResponse(entry=_entry(path))
            if method == 'FsMkdir':
                os.makedirs(path, exist_ok=True)
                return _pb.FsMkdirResponse()
            if method == 'FsRemove':
                # The backend removes recursively, and boxd removes a symlink itself.
                assert request.recursive
                if os.path.isdir(path) and not os.path.islink(path):
                    shutil.rmtree(path)
                else:
                    os.unlink(path)
                return _pb.FsRemoveResponse()
            assert method == 'FsRealpath'
            # Unlike boxd, a symlink loop is kept as written rather than refused; the protocol allows both.
            return _pb.FsRealpathResponse(path=os.fsencode(os.path.realpath(path)))
        except OSError as error:
            raise _fs_failure(error, path) from None

    async def server_stream(self, method: str, request: Any) -> AsyncIterator[Any]:
        self._check(method, request.vm_id)
        path = os.fsdecode(request.path)
        try:
            if method == 'FsList':
                entries = [_entry(os.path.join(path, name)) for name in os.listdir(path)]
                return _messages(_pb.FsListChunk(entries=entries))
            assert method == 'FsRead'
            entry = _entry(path)
            if entry.target_type == _DIRECTORY:
                raise IsADirectoryError(errno.EISDIR, os.strerror(errno.EISDIR))
            if entry.target_type != _FILE:
                # Refused before opening it: a FIFO with no writer would block.
                raise from_grpc_error(
                    _FakeRpcError(
                        grpc.StatusCode.FAILED_PRECONDITION, f'{path}: not a regular file', 'not_a_regular_file'
                    )
                )
            with open(path, 'rb') as file:
                data = file.read()
        except BoxdError:
            raise
        except OSError as error:
            raise _fs_failure(error, path) from None
        return _messages(_pb.FsReadChunk(entry=entry, data=data), _pb.FsReadChunk(eof=True, bytes_read=len(data)))

    async def _write(self, chunks: AsyncIterator[Any]) -> Any:
        first = await anext(chunks)
        data = bytearray(first.data)
        async for chunk in chunks:
            data.extend(chunk.data)
        self._check('FsWrite', first.vm_id)
        path = os.fsdecode(first.path)
        try:
            try:
                file = open(path, 'wb')
            except FileNotFoundError:
                # boxd creates missing parents, of a symlink's target too.
                os.makedirs(os.path.dirname(os.path.realpath(path)), exist_ok=True)
                file = open(path, 'wb')
            with file:
                file.write(data)
            return _pb.FsWriteResponse(bytes_written=len(data), entry=_entry(path))
        except OSError as error:
            raise _fs_failure(error, path) from None


class FakeMachines:
    def __init__(self, owner: FakeBoxd) -> None:
        self._owner = owner
        self._transport = _FakeTransport(owner)
        self.fs = AsyncFs(_FakeFsTransport(owner))

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
            name=name, isolated=isolated, networks=networks, create_kwargs=kwargs, boot_polls=owner.boot_polls, org=org
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
        return [
            machine.record()
            for machine in owner.machines.values()
            if machine.status != 'destroyed' and machine.org == org
        ]

    async def get(self, machine_id: str) -> Machine:
        owner = self._owner
        owner.check_call()
        if owner.get_errors:
            raise owner.get_errors.pop(0)
        machine = owner.find(machine_id)
        if machine.status == 'destroyed':
            raise _not_found(machine_id)
        if machine.boot_polls:
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
    ) -> AsyncExecStream:
        assert isinstance(command, str) and close_stdin
        # The SDK's own method and stream, so iteration, exit status and close run production code.
        return AsyncMachines.stream_exec(cast(Any, self), machine_id, command=command, close_stdin=close_stdin)


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
        self.lines: list[str] = []
        self.exec_commands: list[str] = []
        # The `Fs*` calls made, in order; with `agent_outdated`, each is refused as a machine's old agent refuses it.
        self.fs_calls: list[str] = []
        self.agent_outdated = False
        # Every command stream's pipe readers, which `reap` finishes before it reads the pipes itself.
        self.readers: list[asyncio.Task[None]] = []
        self.boot_polls = 0
        self.reject_credentials = False
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
