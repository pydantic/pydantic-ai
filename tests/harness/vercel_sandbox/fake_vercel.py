"""Controllable fake for the Vercel Sandbox SDK boundary: `SandboxClient` and the `Sandbox` handles it returns.

The exception classes and `SandboxStatus` are the real ones from the installed SDK, so the backend's
`isinstance`, status-code and status checks run against what production raises and reports.

Commands run on the host under `host_root`, in a session of their own, as `Sandbox.run_process` runs them:
each output chunk is written to the `stdout` and `stderr` objects as it arrives, `on_start` gets a handle
whose `kill()` sends the process group SIGKILL, and the call returns once the command exits, even while a
background child holds its output open. Cancelling the call does not stop the command. Destroying or
stopping a sandbox kills its commands, and a command running then fails with `SandboxStreamError`.

Each lookup returns a `FakeHandle` bound to the client that made it, and a handle whose client is closed
fails, as the SDK's handles do.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

import anyio
import httpx2
from vercel.sandbox import SandboxApiError, SandboxStatus, SandboxStreamError

# Output a command writes after it exits is read for this long, then dropped.
_DRAIN_GRACE = 0.05
_EXIT_POLL = 0.01


def api_error(status_code: int, code: str = 'error') -> SandboxApiError:
    """A `SandboxApiError` as the SDK builds it from an API response."""
    response = httpx2.Response(status_code, request=httpx2.Request('GET', 'https://vercel.com/api/v2/sandboxes'))
    return SandboxApiError(response, f'Sandbox API error {status_code}', data={'error': {'code': code}})


def _raise_sink_error(readers: list[asyncio.Task[None]]) -> None:
    """Raise an output sink's error at once: the SDK writes each chunk inline, so its error ends the call."""
    for task in readers:
        if task.done():
            task.result()


@dataclass
class FakeProcess:
    """What `on_start` gets: the started command, killable by its process group."""

    process: asyncio.subprocess.Process

    async def kill(self) -> None:
        with suppress(ProcessLookupError, PermissionError):  # macOS reports EPERM for an exited group
            os.killpg(self.process.pid, signal.SIGKILL)


@dataclass
class FakeCompleted:
    returncode: int
    stdout: str | None = None
    stderr: str | None = None


@dataclass
class FakeSandbox:
    owner: FakeVercel
    name: str
    persistent: bool = True
    status: SandboxStatus = SandboxStatus.RUNNING
    create_kwargs: dict[str, Any] = field(default_factory=dict[str, Any])
    # Lookups that report the sandbox `pending` before it runs.
    pending_polls: int = 0
    processes: list[asyncio.subprocess.Process] = field(default_factory=list[asyncio.subprocess.Process])

    async def run_process(
        self,
        command: str,
        args: Sequence[str] = (),
        *,
        env: dict[str, str] | None = None,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
        capture_output: bool = False,
        on_start: Any = None,
        kill_after: float | None = None,
    ) -> FakeCompleted:
        owner = self.owner
        owner.check_call()
        owner.kill_afters.append(kill_after)
        if owner.run_error is not None:
            raise owner.run_error
        process = await asyncio.create_subprocess_exec(
            command,
            *args,
            cwd=owner.host_root,
            env={'PATH': os.environ['PATH'], **(env or {})},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        self.processes.append(process)
        if on_start is not None:
            on_start(FakeProcess(process))
        assert process.stdout is not None and process.stderr is not None
        captured: dict[str, list[str]] = {'stdout': [], 'stderr': []}

        async def read(pipe: asyncio.StreamReader, name: str, sink: TextIO | None) -> None:
            while data := await pipe.read(65536):
                text = data.decode(errors='replace')
                captured[name].append(text)
                if sink is not None:
                    sink.write(text)

        readers = [
            asyncio.create_task(read(process.stdout, 'stdout', stdout)),
            asyncio.create_task(read(process.stderr, 'stderr', stderr)),
        ]
        try:
            # `Process.wait()` would also wait for a background child to close the pipes.
            while process.returncode is None:
                _raise_sink_error(readers)
                await asyncio.sleep(_EXIT_POLL)
            await asyncio.wait(readers, timeout=_DRAIN_GRACE)
            _raise_sink_error(readers)
        finally:
            for task in readers:
                task.cancel()
        if self.status is not SandboxStatus.RUNNING:
            raise SandboxStreamError('The sandbox session stopped', code='sandbox_stopped')
        assert process.returncode is not None
        if not capture_output:
            return FakeCompleted(process.returncode)
        return FakeCompleted(process.returncode, ''.join(captured['stdout']), ''.join(captured['stderr']))

    async def destroy(self) -> None:
        self.owner.check_call()
        self.end(SandboxStatus.STOPPED)
        del self.owner.sandboxes[self.name]
        self.owner.destroyed.append(self.name)

    def end(self, status: SandboxStatus) -> None:
        """End the session as Vercel does when it stops or destroys the sandbox: its commands are killed."""
        self.status = status
        for process in self.processes:
            if process.returncode is None:
                with suppress(ProcessLookupError, PermissionError):  # macOS reports EPERM for an exited group
                    os.killpg(process.pid, signal.SIGKILL)


@dataclass
class FakeHandle:
    """A `Sandbox` handle: the sandbox's state, reached through the client that fetched it."""

    sandbox: FakeSandbox
    client: FakeClient

    def __getattr__(self, name: str) -> Any:
        if self.client.closed:  # pragma: no cover - only a backend that keeps a stale handle gets here
            raise RuntimeError('SDK session is closed')
        return getattr(self.sandbox, name)


class FakeClient:
    def __init__(self, owner: FakeVercel) -> None:
        self._owner = owner
        self.closed = False
        self.closes = 0

    async def get_or_create_sandbox(self, *, name: str, **kwargs: Any) -> tuple[FakeHandle, bool]:
        owner = self._owner
        owner.create_names.append(name)
        if owner.create_gate is not None:
            await owner.create_gate.wait()
        owner.check_call()
        if owner.create_errors:
            raise owner.create_errors.pop(0)
        if name in owner.sandboxes:
            return FakeHandle(owner.sandboxes[name], self), False
        sandbox = owner.add(name, persistent=kwargs['persistent'])
        sandbox.create_kwargs = kwargs
        if owner.lost_replies:
            owner.lost_replies -= 1
            raise httpx2.ReadError('the connection closed before the reply arrived')
        return FakeHandle(sandbox, self), True

    async def get_sandbox(self, *, name: str) -> FakeHandle:
        owner = self._owner
        owner.check_call()
        if owner.get_errors:
            raise owner.get_errors.pop(0)
        if (sandbox := owner.sandboxes.get(name)) is None:
            raise api_error(404, 'not_found')
        if sandbox.pending_polls:
            sandbox.pending_polls -= 1
            sandbox.status = SandboxStatus.PENDING if sandbox.pending_polls else SandboxStatus.RUNNING
        return FakeHandle(sandbox, self)

    async def aclose(self) -> None:
        self.closed = True
        self.closes += 1
        if self._owner.close_gate is not None:
            await self._owner.close_gate.wait()
        if self._owner.close_error is not None:
            raise self._owner.close_error


class FakeVercel:
    """The fake's control surface: its sandboxes, the clients opened, and errors to inject."""

    def __init__(self) -> None:
        self.host_root = Path('/')
        self.sandboxes: dict[str, FakeSandbox] = {}
        self.clients: list[FakeClient] = []
        self.create_names: list[str] = []
        self.destroyed: list[str] = []
        # While set, a create waits for it before answering.
        self.create_gate: asyncio.Event | None = None
        # Creates that make the sandbox, then lose the reply.
        self.lost_replies = 0
        # While set, closing a client waits for it.
        self.close_gate: asyncio.Event | None = None
        # Raised by the next creates, in order.
        self.create_errors: list[Exception] = []
        # Raised by every call while set.
        self.call_error: Exception | None = None
        # Raised by the next `get_sandbox` lookups, in order.
        self.get_errors: list[Exception] = []
        # The `kill_after` each `run_process` was given.
        self.kill_afters: list[float | None] = []
        # Raised by every `run_process` while set.
        self.run_error: Exception | None = None
        self.close_error: Exception | None = None

    def create(self) -> FakeClient:
        """Stands in for `SandboxClient.create()`."""
        client = FakeClient(self)
        self.clients.append(client)
        return client

    def add(
        self, name: str = 'existing', *, persistent: bool = True, status: SandboxStatus = SandboxStatus.RUNNING
    ) -> FakeSandbox:
        sandbox = self.sandboxes[name] = FakeSandbox(self, name, persistent=persistent, status=status)
        return sandbox

    @property
    def only(self) -> FakeSandbox:
        (sandbox,) = self.sandboxes.values()
        return sandbox

    def check_call(self) -> None:
        if self.call_error is not None:
            raise self.call_error

    async def reap(self) -> None:
        """Kill and wait for every command a test left running."""
        for sandbox in [*self.sandboxes.values()]:
            sandbox.end(sandbox.status)
            for process in sandbox.processes:
                with anyio.move_on_after(5):
                    await process.wait()
