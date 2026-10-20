"""A fake `boxlite` module whose boxes run commands on the host, so the backend runs everywhere without a hypervisor.

Commands run as host processes, each leading its own session as an exec'd process does in a box, so
the workspace conformance suite exercises real process, pipe and filesystem behavior. The shapes and
errors mirror what the real SDK produced in the live probes recorded in
`pydantic_ai_harness.boxlite_sandbox._backend`'s module docstring.
"""

from __future__ import annotations

import asyncio
import codecs
import os
import signal
import subprocess
import types
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, ClassVar

import boxlite

# The SDK ships no type information.
sdk: Any = boxlite


class PanicException(BaseException):
    """Stands in for `pyo3_runtime.PanicException`, which the runtime raises when it can't open."""


@dataclass
class FakeBox:
    """One box: `status` is what `BoxStateInfo.status` reports, `processes` its running commands."""

    name: str
    options: Any
    status: str = 'configured'
    removing: bool = False
    stops: int = 0
    """How many times the box was stopped; a handle from before a stop is invalid, as in the real SDK."""
    processes: set[subprocess.Popen[bytes]] = field(default_factory=set[subprocess.Popen[bytes]])


class FakeBoxLite:
    """The control surface behind the fake module: inspect boxes and calls, or inject failures."""

    def __init__(self, root: Path) -> None:
        self.root = root
        """Where a command with no `cwd` runs, standing in for the box's own working directory."""
        self.boxes: dict[str, FakeBox] = {}
        self.commands: list[tuple[str, list[str], dict[str, str], str | None]] = []
        """`(program, args, env, cwd)` for every exec, the backend's own included."""
        self.spawned: list[subprocess.Popen[bytes]] = []
        self.runtime_error: BaseException | None = None
        """When set, `Boxlite.default()` raises it, like a runtime that can't open."""
        self.create_error: boxlite.BoxliteError | None = None
        self.start_error: boxlite.BoxliteError | None = None
        self.remove_error: boxlite.BoxliteError | None = None
        self.get_info_error: boxlite.BoxliteError | None = None
        self.exec_error: boxlite.BoxliteError | None = None
        """When set, every exec raises it, like the SDK failing to start a command."""
        self.read_chunks: list[str] | None = None
        """When set, every exec streams exactly these stdout chunks, like a misbehaving guest."""
        self.start_started = asyncio.Event()
        self.start_gate: asyncio.Event | None = None
        self.exec_started = asyncio.Event()
        self.exec_gate: asyncio.Event | None = None
        """When set, `exec` waits for it before starting the command, to cancel a caller mid-start."""
        """When set, `start()` waits for it, to cancel a caller mid-creation."""
        self.remove_started = asyncio.Event()
        self.remove_gate: asyncio.Event | None = None
        """When set, `remove` ends the box's commands, then waits for it before the box is gone."""
        self.module = types.ModuleType('boxlite')
        self.module.__dict__.update(
            Boxlite=type('Boxlite', (FakeRuntime,), {'control': self}),
            BoxOptions=sdk.BoxOptions,
            NetworkSpec=sdk.NetworkSpec,
            OutboundNetworkSpec=sdk.OutboundNetworkSpec,
            BoxliteError=boxlite.BoxliteError,
            NotFoundError=boxlite.NotFoundError,
            StoppedError=boxlite.StoppedError,
        )

    def stop(self, name: str) -> None:
        """Stop a box from outside the backend, ending its commands like the real runtime."""
        box = self.boxes[name]
        box.status = 'stopped'
        box.stops += 1
        _end(box)

    def close(self) -> None:
        for process in self.spawned:
            if process.returncode is None:
                _kill(process)
                process.wait()
            for stream in (process.stdin, process.stdout, process.stderr):
                assert stream is not None
                stream.close()


class FakeRuntime:
    """`boxlite.Boxlite`, subclassed per `FakeBoxLite` to bind `control`."""

    control: ClassVar[FakeBoxLite]

    @classmethod
    def default(cls) -> FakeRuntime:
        if cls.control.runtime_error is not None:
            raise cls.control.runtime_error
        return cls()

    async def create(self, options: Any, name: str | None = None) -> FakeHandle:
        if self.control.create_error is not None:
            raise self.control.create_error
        # The backend always names its boxes.
        assert name is not None and name not in self.control.boxes
        box = self.control.boxes[name] = FakeBox(name=name, options=options)
        return FakeHandle(self.control, box)

    async def get(self, id_or_name: str) -> FakeHandle | None:
        box = self.control.boxes.get(id_or_name)
        return None if box is None else FakeHandle(self.control, box)

    async def get_info(self, id_or_name: str) -> Any:
        if self.control.get_info_error is not None:
            raise self.control.get_info_error
        box = self.control.boxes.get(id_or_name)
        return None if box is None else types.SimpleNamespace(state=types.SimpleNamespace(status=box.status))

    async def remove(self, id_or_name: str, force: bool = False) -> None:
        assert force
        if self.control.remove_error is not None:
            raise self.control.remove_error
        box = self.control.boxes.get(id_or_name)
        if box is None:
            raise boxlite.NotFoundError(f'box not found: {id_or_name}')
        # Like the real box, it reports itself running while its commands end, and refuses new ones.
        box.removing = True
        _end(box)
        self.control.remove_started.set()
        if self.control.remove_gate is not None:
            await self.control.remove_gate.wait()
        # A removal running alongside this one may have deleted it first.
        self.control.boxes.pop(id_or_name, None)


class FakeHandle:
    """`boxlite.Box`: a handle on one box, which outlives the box it names."""

    def __init__(self, control: FakeBoxLite, box: FakeBox) -> None:
        self._control = control
        self._box = box
        self._stops = box.stops

    @property
    def name(self) -> str:
        return self._box.name

    async def start(self) -> None:
        self._control.start_started.set()
        if self._control.start_gate is not None:
            await self._control.start_gate.wait()
        if self._control.start_error is not None:
            raise self._control.start_error
        self._box.status = 'running'

    async def exec(
        self,
        command: str,
        args: list[str] | None = None,
        env: list[tuple[str, str]] | None = None,
        cwd: str | None = None,
    ) -> FakeExecution:
        args = args or []
        env_vars = dict(env or [])
        self._control.commands.append((command, args, env_vars, cwd))
        self._control.exec_started.set()
        if self._control.exec_gate is not None:
            await self._control.exec_gate.wait()
        if self._control.exec_error is not None:
            raise self._control.exec_error
        if self._control.boxes.get(self._box.name) is not self._box:
            raise boxlite.UpstreamUnavailableError('gRPC/tonic error: status: Unavailable')
        if self._box.stops != self._stops:
            raise boxlite.StoppedError(
                'stopped: Handle invalidated after stop(). Use runtime.get() to get a new handle.'
            )
        if self._box.removing:  # pragma: no cover - the backend waits out a deletion instead of reaching this
            raise boxlite.InternalError('internal error: execution_exists: Execution already exists')
        # A stopped box boots again on exec, as the real one does.
        self._box.status = 'running'
        if cwd is not None and not Path(cwd).is_dir():
            raise boxlite.InternalError('internal error: spawn_failed: exec process failed: failed to unix syscall')
        process = subprocess.Popen(
            [command, *args],
            cwd=self._control.root if cwd is None else cwd,
            env={'PATH': os.environ['PATH'], **env_vars},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        self._box.processes.add(process)
        self._control.spawned.append(process)
        return FakeExecution(process, self._box, self._control.read_chunks)


class FakeExecution:
    """`boxlite.Execution` over a host process; like the real one, `signal` reaches only that process."""

    def __init__(self, process: subprocess.Popen[bytes], box: FakeBox, chunks: list[str] | None) -> None:
        self._process = process
        self._box = box
        self._chunks = chunks

    async def wait(self) -> Any:
        code = await asyncio.to_thread(self._process.wait)
        self._box.processes.discard(self._process)
        return types.SimpleNamespace(exit_code=code)

    async def signal(self, signum: int) -> None:
        # Like the real SDK's, refused once the process has exited; its pid may belong to another one by now.
        if self._process.returncode is not None:  # pragma: lax no cover - an exit racing a stop
            raise boxlite.InternalError('internal error: Failed to send signal')
        os.kill(self._process.pid, signum)

    def stdout(self) -> AsyncIterator[str]:
        if self._chunks is not None:
            return _replay(self._chunks)
        return _read(self._process.stdout)

    def stderr(self) -> AsyncIterator[str]:
        return _replay([]) if self._chunks is not None else _read(self._process.stderr)


async def _read(stream: IO[bytes] | None) -> AsyncIterator[str]:
    assert stream is not None
    decoder = codecs.getincrementaldecoder('utf-8')('replace')
    while chunk := await asyncio.to_thread(os.read, stream.fileno(), 65536):
        yield decoder.decode(chunk)
    yield decoder.decode(b'', final=True)


async def _replay(chunks: list[str]) -> AsyncIterator[str]:
    for chunk in chunks:
        yield chunk


def _end(box: FakeBox) -> None:
    # The runtime sends TERM to every process in a box it stops or removes.
    for process in list(box.processes):
        # Never signal a reaped process: its pid may belong to another one by now.
        if process.returncode is None:  # pragma: no branch - reaped only when an ending races its wait
            os.kill(process.pid, signal.SIGTERM)


def _kill(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    # macOS refuses a group whose processes have exited but are not reaped yet with EPERM, not ESRCH.
    except (ProcessLookupError, PermissionError):  # pragma: no cover - depends on timing and platform
        pass
