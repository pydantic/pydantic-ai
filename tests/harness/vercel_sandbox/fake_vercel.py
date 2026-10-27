"""Controllable fake for the Vercel Sandbox SDK boundary.

Every sandbox runs its processes on the host, starting in `FakeVercel.host_root`, which stands in
for the sandbox's filesystem; so the backend's `sh` launcher, output collection, and kill run for
real. The exception classes and `SandboxStatus` are the real ones from the installed SDK, so the
backend's `isinstance` and status checks run against what production raises and reports.

Like the SDK, a sandbox exists once Vercel accepts the create request, before `create_sandbox`
returns. Deleting one ends its processes with SIGTERM, whose exit a `wait()` already in flight
reports as 143, as a live sandbox does; after that the process API answers 404. Unlike Vercel,
whose `wait()` returns only once every process holding the command's output has closed it, the
fake reports the exit as soon as the command itself ends.
"""

from __future__ import annotations

import asyncio
import codecs
import os
import signal
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import suppress
from datetime import timedelta
from pathlib import Path
from typing import Any

import anyio
import httpx2
from vercel.sandbox import SandboxApiError, SandboxStatus


def api_error(status: int, message: str = 'Vercel API error', *, code: str | None = None) -> SandboxApiError:
    """A `SandboxApiError` as the SDK raises it for a non-2xx answer."""
    data = None if code is None else {'error': {'code': code, 'message': message}}
    return SandboxApiError(httpx2.Response(status), message, data=data)


def _kill_group(process: asyncio.subprocess.Process, signum: int = signal.SIGKILL) -> None:
    # Each process leads its own group, so this also stops the children a test command started.
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signum)


class FakeReader:
    """One output stream of a host process, read like the SDK's `TextReader`."""

    def __init__(self, stream: asyncio.StreamReader, owner: FakeVercel) -> None:
        self._stream = stream
        self._owner = owner
        self._decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        self.closed = False

    async def read(self, size: int = -1) -> str:
        if self._owner.read_error is not None:
            raise self._owner.read_error
        data = await self._stream.read(size)
        return self._decoder.decode(data, final=not data)

    async def aclose(self) -> None:
        self.closed = True


class FakeProcess:
    def __init__(self, sandbox: FakeSandbox, process: asyncio.subprocess.Process) -> None:
        assert process.stdout is not None and process.stderr is not None
        self.sandbox = sandbox
        self.process = process
        self.stdout = FakeReader(process.stdout, sandbox.owner)
        self.stderr = FakeReader(process.stderr, sandbox.owner)
        self.kill_calls = 0

    async def wait(self) -> int:
        owner = self.sandbox.owner
        if owner.wait_error is not None:
            raise owner.wait_error
        # Not `process.wait()`, which also waits for the pipes to close.
        while (exit_code := self.process.returncode) is None:
            await asyncio.sleep(0.01)
        # Vercel reports a signal exit as a shell does: 128 plus the signal number.
        return 128 - exit_code if exit_code < 0 else exit_code

    async def kill(self) -> None:
        self.kill_calls += 1
        if self.sandbox.owner.kill_error is not None:
            raise self.sandbox.owner.kill_error
        _kill_group(self.process)


class FakeSandbox:
    def __init__(self, owner: FakeVercel, name: str, env: Mapping[str, str] | None = None) -> None:
        self.owner = owner
        owner._all_sandboxes.append(self)  # pyright: ignore[reportPrivateUsage]
        self.name = name
        self.env = dict(env or {})
        self.status = SandboxStatus.RUNNING
        self.deleted = False
        self.processes: list[FakeProcess] = []
        # Every process's argv and per-process env, in order.
        self.commands: list[list[str]] = []
        self.command_envs: list[Mapping[str, str] | None] = []
        self.kill_after: list[float | timedelta | None] = []
        self.destroy_calls: list[bool] = []

    async def create_process(
        self,
        command: str,
        args: Sequence[str] | None = None,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        sudo: bool = False,
        kill_after: float | timedelta | None = None,
        stdout: int = -1,
        stderr: int = -1,
    ) -> FakeProcess:
        owner = self.owner
        argv = [command, *(args or ())]
        self.commands.append(argv)
        self.command_envs.append(env)
        self.kill_after.append(kill_after)
        if self.deleted:
            raise api_error(404, 'Sandbox not found', code='not_found')
        if owner.process_error is not None:
            raise owner.process_error
        if owner.process_gate is not None:
            await owner.process_gate.wait()
        if owner.rewrite is not None:
            argv = owner.rewrite(argv)
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=owner.host_root,
            env={'PATH': os.environ.get('PATH', '/usr/bin:/bin'), **self.env, **(env or {})},
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        handle = FakeProcess(self, process)
        self.processes.append(handle)
        owner.process_started.set()
        return handle

    async def destroy(self, *, delete_orphan_snapshots: bool = False) -> FakeSandbox:
        if self.owner.destroy_error is not None:
            raise self.owner.destroy_error
        self.destroy_calls.append(delete_orphan_snapshots)
        self.deleted = True
        self.owner.sandboxes.pop(self.name, None)
        for handle in self.processes:
            _kill_group(handle.process, signal.SIGTERM)
        return self


class FakeClient:
    def __init__(self, owner: FakeVercel) -> None:
        self.owner = owner
        self.closed = False

    async def create_sandbox(self, *, name: str | None = None, **params: Any) -> FakeSandbox:
        owner = self.owner
        assert name is not None
        owner.create_calls.append({'name': name, **params})
        if owner.create_error is not None:
            raise owner.create_error
        sandbox = FakeSandbox(owner, name, params.get('env'))
        owner.sandboxes[name] = sandbox
        owner.create_started.set()
        if owner.create_gate is not None:
            await owner.create_gate.wait()
        if owner.lose_create_reply:
            raise httpx2.ConnectError('create response lost')
        return sandbox

    async def get_sandbox(self, *, name: str) -> FakeSandbox:
        owner = self.owner
        owner.get_calls.append(name)
        if (error := next(owner.get_errors, None)) is not None:
            raise error
        if owner.get_gate is not None:
            await owner.get_gate.wait()
        if (sandbox := owner.sandboxes.get(name)) is None:
            raise api_error(404, f'Sandbox {name} not found', code='not_found')
        return sandbox

    async def aclose(self) -> None:
        if self.owner.close_gate is not None:
            await self.owner.close_gate.wait()
        if self.owner.close_error is not None:
            raise self.owner.close_error
        self.closed = True
        self.owner.closed_clients += 1


class FakeVercel:
    """The fake's control surface; `client` stands in for `SandboxClient.create`."""

    def __init__(self, host_root: Path) -> None:
        self.host_root = host_root
        self.sandboxes: dict[str, FakeSandbox] = {}
        # Every sandbox, including deleted ones, whose processes teardown stops.
        self._all_sandboxes: list[FakeSandbox] = []
        self.clients: list[FakeClient] = []
        self.closed_clients = 0
        self.create_calls: list[dict[str, Any]] = []
        self.create_error: Exception | None = None
        self.create_gate: asyncio.Event | None = None
        self.create_started = asyncio.Event()
        self.lose_create_reply = False
        self.get_calls: list[str] = []
        # Raised by successive `get_sandbox` calls; lookups succeed once it is exhausted.
        self.get_errors: Iterator[Exception] = iter(())
        self.get_gate: asyncio.Event | None = None
        self.close_error: Exception | None = None
        self.close_gate: asyncio.Event | None = None
        self.destroy_error: Exception | None = None
        self.process_error: Exception | None = None
        self.process_gate: asyncio.Event | None = None
        self.process_started = asyncio.Event()
        self.read_error: Exception | None = None
        self.wait_error: Exception | None = None
        self.kill_error: Exception | None = None
        # Replaces the argv a process runs, after it is recorded.
        self.rewrite: Callable[[list[str]], list[str]] | None = None

    def client(self, *, options: object = None, httpx_client_factory: object = None) -> FakeClient:
        client = FakeClient(self)
        self.clients.append(client)
        return client

    @property
    def opened_clients(self) -> int:
        return len(self.clients)

    def sandbox(self, name: str = 'sb-existing') -> FakeSandbox:
        """A sandbox that exists before the test, as one created elsewhere."""
        sandbox = FakeSandbox(self, name)
        self.sandboxes[name] = sandbox
        return sandbox

    async def aclose(self) -> None:
        """Kill every process a test left running and drain its pipes, so each transport closes before the test ends.

        A process stopped mid-output, or one whose background child held its output open past the
        drain grace, still has unread data that would otherwise keep its pipe open until garbage collection.
        """
        for handle in self.processes:
            _kill_group(handle.process)
            with anyio.move_on_after(5):
                await handle.process.communicate()

    @property
    def processes(self) -> list[FakeProcess]:
        return [handle for sandbox in self._all_sandboxes for handle in sandbox.processes]
