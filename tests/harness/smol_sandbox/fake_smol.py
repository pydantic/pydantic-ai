"""A fake `smol` module whose VMs run commands on the host, so the backend runs everywhere without a hypervisor.

Commands run as host processes, so the workspace conformance suite exercises real process, pipe and
filesystem behavior. The event shapes and errors mirror what the real SDK produced in the live probes
recorded in `pydantic_ai_harness.smol_sandbox._backend`'s module docstring.
"""

from __future__ import annotations

import asyncio
import codecs
import itertools
import os
import signal
import types
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import smol


@dataclass
class FakeVM:
    """One VM: `state` is what `AsyncMachine.state()` reports, `processes` its running commands."""

    id: str
    config: smol.MachineConfig
    state: str = 'running'
    processes: set[asyncio.subprocess.Process] = field(default_factory=set[asyncio.subprocess.Process])


class FakeSmol:
    """The control surface behind the fake module: inspect VMs and calls, or inject failures."""

    def __init__(self) -> None:
        self.vms: dict[str, FakeVM] = {}
        self.connections: list[tuple[str | None, smol.ConnectOptions | None]] = []
        """`(machine_id, conn)` for every create (`None` id) and connect."""
        self.commands: list[tuple[list[str], smol.ExecOptions | None]] = []
        self.create_error: smol.SmolError | None = None
        self.connect_error: smol.SmolError | None = None
        self.state_error: smol.SmolError | None = None
        self.exec_error: str | None = None
        """When set, every command ends with this error event, like an engine failure."""
        self.create_started = asyncio.Event()
        self.create_gate: asyncio.Event | None = None
        """When set, `create` waits for it, to cancel a caller mid-create."""
        self.ids = itertools.count(1)
        self.module = types.ModuleType('smol')
        self.module.__dict__.update(
            AsyncMachine=type('AsyncMachine', (FakeAsyncMachine,), {'control': self}),
            ConnectOptions=smol.ConnectOptions,
            ExecOptions=smol.ExecOptions,
            MachineConfig=smol.MachineConfig,
            ResourceSpec=smol.ResourceSpec,
            SmolError=smol.SmolError,
            wrap_native_error=smol.wrap_native_error,
        )

    async def destroy(self, machine_id: str) -> None:
        """Delete a VM, killing its running commands like the real engine."""
        vm = self.vms.pop(machine_id)
        vm.state = 'stopped'
        for process in vm.processes:
            _kill(process)


class FakeAsyncMachine:
    """`smol.AsyncMachine`, subclassed per `FakeSmol` to bind `control`."""

    control: ClassVar[FakeSmol]

    def __init__(self, vm: FakeVM) -> None:
        self._vm = vm

    @property
    def id(self) -> str:
        return self._vm.id

    @classmethod
    async def create(
        cls, config: smol.MachineConfig | None = None, conn: smol.ConnectOptions | None = None
    ) -> FakeAsyncMachine:
        cls.control.connections.append((None, conn))
        cls.control.create_started.set()
        if cls.control.create_gate is not None:
            await cls.control.create_gate.wait()
        if cls.control.create_error is not None:
            raise cls.control.create_error
        vm = FakeVM(id=f'smol-{next(cls.control.ids)}', config=config or smol.MachineConfig())
        cls.control.vms[vm.id] = vm
        return cls(vm)

    @classmethod
    async def connect(cls, machine_id: str, conn: smol.ConnectOptions | None = None) -> FakeAsyncMachine:
        cls.control.connections.append((machine_id, conn))
        if cls.control.connect_error is not None:
            raise cls.control.connect_error
        vm = cls.control.vms.get(machine_id)
        if vm is None:
            raise smol.SmolError('NOT_FOUND', f'VM not found: {machine_id}')
        vm.state = 'running'
        return cls(vm)

    async def state(self) -> str:
        if self.control.state_error is not None:
            raise self.control.state_error
        return self._vm.state

    async def delete(self, include_usage: bool = False) -> None:
        await self.control.destroy(self._vm.id)

    async def exec_stream(
        self, command: list[str], opts: smol.ExecOptions | None = None
    ) -> AsyncIterator[dict[str, object]]:
        self.control.commands.append((command, opts))
        workdir = opts.workdir if opts is not None else None
        if self.control.exec_error is not None:
            failure = self.control.exec_error
        elif self.control.vms.get(self._vm.id) is not self._vm:
            failure = f'[NOT_FOUND] VM not found: {self._vm.id}'
        elif workdir is not None and not Path(workdir).is_dir():
            failure = '[SMOLVM_ERROR] Agent error (streaming exec): No such file or directory (os error 2)'
        else:
            failure = None
        # The consumer stops at an `error` event, so nothing may follow its `yield`.
        if failure is not None:
            yield {'kind': 'error', 'message': failure}
        else:
            env = {'PATH': os.environ['PATH'], **((opts.env if opts is not None else None) or {})}
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=workdir,
                env=env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            self._vm.processes.add(process)
            events: asyncio.Queue[dict[str, object]] = asyncio.Queue()
            waiter = asyncio.create_task(self._report(process, events))
            try:
                while (event := await events.get())['kind'] not in ('exit', 'error'):
                    yield event
                yield event
            finally:
                # Synchronous, like the real SDK's `stream.close()`, so it completes under cancellation.
                _kill(process)
                self._vm.processes.discard(process)
                waiter.cancel()

    async def _report(self, process: asyncio.subprocess.Process, events: asyncio.Queue[dict[str, object]]) -> None:
        async def pump(stream: asyncio.StreamReader | None, kind: str) -> None:
            assert stream is not None
            decoder = codecs.getincrementaldecoder('utf-8')('replace')
            while chunk := await stream.read(65536):
                await events.put({'kind': kind, 'data': decoder.decode(chunk)})
            await events.put({'kind': kind, 'data': decoder.decode(b'', final=True)})

        readers = asyncio.gather(pump(process.stdout, 'stdout'), pump(process.stderr, 'stderr'))
        code = await process.wait()
        # Like the real engine, report the exit without waiting for a background child that
        # inherited the output pipes to close them.
        try:
            await asyncio.wait_for(asyncio.shield(readers), 0.5)
        except TimeoutError:
            readers.cancel()
        if self._vm.state != 'running':
            await events.put({'kind': 'error', 'message': 'io operation failed: connection closed'})
        else:
            await events.put({'kind': 'exit', 'exit_code': code})


def _kill(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:  # pragma: no cover - the process exited between the check and the kill
            pass
