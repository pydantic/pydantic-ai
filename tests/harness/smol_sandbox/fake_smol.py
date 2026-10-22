"""A fake `smol` module whose VMs run commands on the host, so the backend runs everywhere without a hypervisor.

Commands run as host processes, so the workspace conformance suite exercises real process, pipe and
filesystem behavior. They are confined to `root`, a test's temporary directory: a VM's own directory
is `root`, and a command elsewhere fails the test rather than touching the host. The event shapes and errors mirror what the real SDK produced in the live probes
recorded in `pydantic_ai_harness.smol_sandbox._backend`'s module docstring.
"""

from __future__ import annotations

import codecs
import os
import queue
import signal
import subprocess
import threading
import types
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, ClassVar

import smol


@dataclass
class FakeVM:
    """One VM: `state` is what `Machine.state()` reports, `processes` its running commands."""

    id: str
    config: smol.MachineConfig
    state: str = 'running'
    processes: set[subprocess.Popen[bytes]] = field(default_factory=set[subprocess.Popen[bytes]])


class FakeSmol:
    """The control surface behind the fake module: inspect VMs and calls, or inject failures.

    Like the real SDK's, the fake `Machine` blocks, so the backend calls it on worker threads, and
    the gates are `threading.Event`s.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.vms: dict[str, FakeVM] = {}
        self.connections: list[tuple[str | None, smol.ConnectOptions | None]] = []
        """`(machine_id, conn)` for every create (`None` id) and connect."""
        self.commands: list[tuple[list[str], smol.ExecOptions | None]] = []
        self.spawned: list[subprocess.Popen[bytes]] = []
        """Every command's process, in order."""
        self.create_error: smol.SmolError | None = None
        self.connect_error: smol.SmolError | None = None
        self.state_error: smol.SmolError | None = None
        self.open_error: smol.SmolError | None = None
        """When set, `exec_stream` raises it, like the SDK failing to start a command."""
        self.exec_error: str | None = None
        """When set, every command ends with this error event, like an engine failure."""
        self.exec_events: list[dict[str, object]] | None = None
        """When set, every command yields exactly these events, like a misbehaving engine or guest."""
        self.create_started = threading.Event()
        self.create_gate: threading.Event | None = None
        """When set, `create` waits for it, to cancel a caller mid-create."""
        self.connect_started = threading.Event()
        self.connect_gate: threading.Event | None = None
        """When set, `connect` boots the VM, then waits for it, to cancel a caller mid-connect."""
        self.open_started = threading.Event()
        self.open_gate: threading.Event | None = None
        """When set, `exec_stream` waits for it before starting the command, to cancel a caller mid-open."""
        self.module = types.ModuleType('smol')
        self.module.__dict__.update(
            AsyncMachine=smol.AsyncMachine,
            Machine=type('Machine', (FakeMachine,), {'control': self}),
            ConnectOptions=smol.ConnectOptions,
            ExecOptions=smol.ExecOptions,
            MachineConfig=smol.MachineConfig,
            ResourceSpec=smol.ResourceSpec,
            SmolError=smol.SmolError,
            wrap_native_error=smol.wrap_native_error,
        )

    def destroy(self, machine_id: str) -> None:
        """Delete a VM, killing its running commands like the real engine."""
        vm = self.vms.pop(machine_id)
        vm.state = 'stopped'
        for process in list(vm.processes):
            _kill(process)


class FakeMachine:
    """`smol.Machine`, subclassed per `FakeSmol` to bind `control`; the real `smol.AsyncMachine` wraps it."""

    control: ClassVar[FakeSmol]

    def __init__(self, vm: FakeVM) -> None:
        self._vm = vm

    @property
    def id(self) -> str:
        return self._vm.id

    @classmethod
    def create(cls, config: smol.MachineConfig | None = None, conn: smol.ConnectOptions | None = None) -> FakeMachine:
        cls.control.connections.append((None, conn))
        cls.control.create_started.set()
        if cls.control.create_gate is not None:
            cls.control.create_gate.wait()
        if cls.control.create_error is not None:
            raise cls.control.create_error
        # The backend always names its VMs; the real SDK would make up a `smol-` name otherwise.
        assert config is not None and config.name is not None
        vm = FakeVM(id=config.name, config=config)
        cls.control.vms[vm.id] = vm
        return cls(vm)

    @classmethod
    def connect(cls, machine_id: str, conn: smol.ConnectOptions | None = None) -> FakeMachine:
        cls.control.connections.append((machine_id, conn))
        if cls.control.connect_error is not None:
            raise cls.control.connect_error
        vm = cls.control.vms.get(machine_id)
        if vm is None:
            raise smol.SmolError('NOT_FOUND', f'VM not found: {machine_id}')
        vm.state = 'running'
        cls.control.connect_started.set()
        if cls.control.connect_gate is not None:
            cls.control.connect_gate.wait()
        return cls(vm)

    def state(self) -> str:
        if self.control.state_error is not None:
            raise self.control.state_error
        return self._vm.state

    def delete(self, include_usage: bool = False) -> None:
        self.control.destroy(self._vm.id)

    def exec_stream(self, command: list[str], opts: smol.ExecOptions | None = None) -> smol.ExecStream:
        self.control.commands.append((command, opts))
        self.control.open_started.set()
        if self.control.open_gate is not None:
            self.control.open_gate.wait()
        if self.control.open_error is not None:
            raise self.control.open_error
        if self.control.exec_events is not None:
            return smol.ExecStream(self.control.exec_events, lambda: None)
        workdir = Path(opts.workdir if opts is not None and opts.workdir is not None else self.control.root)
        assert workdir.is_relative_to(self.control.root), f'a fake VM command ran outside the test directory: {workdir}'
        if self.control.exec_error is not None:
            failure = self.control.exec_error
        elif self.control.vms.get(self._vm.id) is not self._vm:
            failure = f'[NOT_FOUND] VM not found: {self._vm.id}'
        elif not workdir.is_dir():
            failure = '[SMOLVM_ERROR] Agent error (streaming exec): No such file or directory (os error 2)'
        else:
            failure = None
        if failure is not None:
            return smol.ExecStream([{'kind': 'error', 'message': failure}], lambda: None)
        env = {'PATH': os.environ['PATH'], **((opts.env if opts is not None else None) or {})}
        process = subprocess.Popen(
            command,
            cwd=workdir,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        self._vm.processes.add(process)
        self.control.spawned.append(process)
        events: queue.Queue[dict[str, object]] = queue.Queue()
        reporter = threading.Thread(target=self._report, args=(process, events), daemon=True)
        reporter.start()

        def kill() -> None:
            # Like the engine, killing the command also reaps it before the stream ends.
            _kill(process)
            reporter.join()

        return smol.ExecStream(_drain(events), kill)

    def _report(self, process: subprocess.Popen[bytes], events: queue.Queue[dict[str, object]]) -> None:
        def pump(stream: IO[bytes] | None, kind: str) -> None:
            assert stream is not None
            decoder = codecs.getincrementaldecoder('utf-8')('replace')
            with stream:
                while chunk := os.read(stream.fileno(), 65536):
                    events.put({'kind': kind, 'data': decoder.decode(chunk)})
            events.put({'kind': kind, 'data': decoder.decode(b'', final=True)})

        # Unlike the real engine, this waits for the output pipes to close as well as for the exit.
        pumps = [
            threading.Thread(target=pump, args=(process.stdout, 'stdout'), daemon=True),
            threading.Thread(target=pump, args=(process.stderr, 'stderr'), daemon=True),
        ]
        for thread in pumps:
            thread.start()
        for thread in pumps:
            thread.join()
        code = process.wait()
        self._vm.processes.discard(process)
        if self._vm.state != 'running':
            events.put({'kind': 'error', 'message': 'io operation failed: connection closed'})
        else:
            events.put({'kind': 'exit', 'exit_code': code})


def _drain(events: queue.Queue[dict[str, object]]) -> Iterator[dict[str, object]]:
    while (event := events.get())['kind'] not in ('exit', 'error'):
        yield event
    yield event


def _kill(process: subprocess.Popen[bytes]) -> None:
    if process.returncode is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:  # pragma: no cover - the process was reaped between the check and the kill
            pass
