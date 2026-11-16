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
from contextlib import suppress
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

    id: str
    name: str | None
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
        """Every box by its name, or by its id if it has none."""
        self.commands: list[tuple[str, list[str], dict[str, str], str | None]] = []
        """`(program, args, env, cwd)` for every exec, the backend's own included."""
        self.spawned: list[subprocess.Popen[bytes]] = []
        self.runtime_error: BaseException | None = None
        """When set, `Boxlite.default()` raises it, like a runtime that can't open."""
        self.create_error: BaseException | None = None
        """When set, `create` raises it; a `CancelledError` stands in for the SDK cancelling its own call."""
        self.start_error: boxlite.BoxliteError | None = None
        self.remove_error: boxlite.BoxliteError | None = None
        self.get_info_error: boxlite.BoxliteError | None = None
        self.exec_error: boxlite.BoxliteError | None = None
        """When set, every exec raises it, like the SDK failing to start a command."""
        self.read_chunks: list[str] | None = None
        """When set, every exec streams exactly these stdout chunks, like a misbehaving guest."""
        self.stream_error: boxlite.BoxliteError | None = None
        """When set, every exec's stdout raises it, like the SDK losing a command's output stream."""
        self.wait_error: boxlite.BoxliteError | None = None
        """When set, the next `wait()` raises it, like the SDK losing a command's exit status."""
        self.removed: list[str] = []
        """The names, or ids, of the boxes `remove` deleted, in order."""
        self.create_started = asyncio.Event()
        self.create_gate: asyncio.Event | None = None
        """When set, `create` waits for it before recording the box, to cancel a caller mid-create."""
        self.start_started = asyncio.Event()
        self.start_gate: asyncio.Event | None = None
        """When set, `start()` waits for it, to cancel a caller mid-boot."""
        self.exec_started = asyncio.Event()
        self.exec_gate: asyncio.Event | None = None
        """When set, `exec` waits for it before starting the command, to cancel a caller mid-start."""
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
            ConfigError=boxlite.ConfigError,
        )

    def find(self, id_or_name: str) -> FakeBox | None:
        """The box the real runtime finds by `id_or_name`, which is either its name or its id."""
        return next((box for box in self.boxes.values() if id_or_name in (box.name, box.id)), None)

    def stop(self, id_or_name: str) -> None:
        """Stop a box from outside the backend, ending its commands like the real runtime."""
        box = self.find(id_or_name)
        assert box is not None
        box.status = 'stopped'
        box.stops += 1
        _end(box)

    def close(self) -> None:
        for process in self.spawned:
            if process.returncode is None:  # pragma: no cover - only a failing test leaves a command running
                # macOS refuses a group whose processes have exited but are not reaped yet with EPERM, not ESRCH.
                with suppress(ProcessLookupError, PermissionError):
                    os.killpg(process.pid, signal.SIGKILL)
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
        self.control.create_started.set()
        if self.control.create_gate is not None:
            await self.control.create_gate.wait()
        # The backend always names its boxes; a test creates an unnamed one for the backend's `sandbox`.
        assert name is None or self.control.find(name) is None
        box = FakeBox(id=f'box{len(self.control.boxes)}', name=name, options=options)
        self.control.boxes[name or box.id] = box
        return FakeHandle(self.control, box)

    async def get(self, id_or_name: str) -> FakeHandle | None:
        box = self.control.find(id_or_name)
        return None if box is None else FakeHandle(self.control, box)

    async def get_info(self, id_or_name: str) -> Any:
        if self.control.get_info_error is not None:
            raise self.control.get_info_error
        box = self.control.find(id_or_name)
        return None if box is None else types.SimpleNamespace(state=types.SimpleNamespace(status=box.status))

    async def remove(self, id_or_name: str, force: bool = False) -> None:
        assert force
        if self.control.remove_error is not None:
            raise self.control.remove_error
        box = self.control.find(id_or_name)
        if box is None:
            raise boxlite.NotFoundError(f'box not found: {id_or_name}')
        # Like the real box, it reports itself running while its commands end, and refuses new ones.
        box.removing = True
        _end(box)
        self.control.remove_started.set()
        if self.control.remove_gate is not None:
            await self.control.remove_gate.wait()
        # A removal running alongside this one may have deleted it first.
        if self.control.boxes.pop(box.name or box.id, None) is not None:
            self.control.removed.append(box.name or box.id)


class FakeHandle:
    """`boxlite.Box`: a handle on one box, which outlives the box it names."""

    def __init__(self, control: FakeBoxLite, box: FakeBox) -> None:
        self._control = control
        self._box = box
        self._stops = box.stops

    @property
    def id(self) -> str:
        return self._box.id

    @property
    def name(self) -> str | None:
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
        if self._control.find(self._box.id) is not self._box:
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
        return FakeExecution(process, self._box, self._control)


class FakeExecution:
    """`boxlite.Execution` over a host process; like the real one, `signal` reaches only that process."""

    def __init__(self, process: subprocess.Popen[bytes], box: FakeBox, control: FakeBoxLite) -> None:
        self._process = process
        self._box = box
        self._control = control
        self._chunks = control.read_chunks
        self._error = control.stream_error

    async def wait(self) -> Any:
        if (error := self._control.wait_error) is not None:
            self._control.wait_error = None
            raise error
        code = await asyncio.to_thread(self._process.wait)
        self._box.processes.discard(self._process)
        return types.SimpleNamespace(exit_code=code)

    async def signal(self, signum: int) -> None:
        # Like the real SDK's, refused once the process has exited; its pid may belong to another one by now.
        try:
            if self._process.returncode is not None:  # pragma: lax no cover - an exit racing a stop
                raise ProcessLookupError
            os.kill(self._process.pid, signum)
        except ProcessLookupError:  # pragma: lax no cover - an exit racing a stop
            raise boxlite.InternalError('internal error: Failed to send signal') from None

    def stdout(self) -> AsyncIterator[str]:
        if self._error is not None:
            return _fail(self._error)
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


async def _fail(error: Exception) -> AsyncIterator[str]:
    raise error
    yield  # pragma: no cover - makes this an async generator


def _end(box: FakeBox) -> None:
    # The runtime sends TERM to every process in a box it stops or removes, including the commands each
    # script runs in a session of its own.
    for process in list(box.processes):
        # Never signal a reaped process: its pid may belong to another one by now.
        if process.returncode is None:  # pragma: no branch - reaped only when an ending races its wait
            # Listed first: once the script dies, its children are re-parented and `pgrep -P` misses them.
            children = subprocess.run(['pgrep', '-P', str(process.pid)], capture_output=True, text=True, check=False)
            # The script before its command, so it dies of the signal as in a real box. Signalled second, it
            # could see its command die first and exit 143 itself, which reads as the command's own exit.
            with suppress(ProcessLookupError):
                os.kill(process.pid, signal.SIGTERM)
            # Each may exit, and be reaped by the thread waiting for it, before it is signalled.
            for child in children.stdout.split():
                with suppress(ProcessLookupError, PermissionError):
                    os.killpg(int(child), signal.SIGTERM)
