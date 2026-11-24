"""Controllable fake for the Runta API boundary.

The backend's own client runs unchanged: `FakeRunta.handle` stands in for the HTTP transport, and
`FakeRunta.connect` for the `websockets` `connect` the client opens the exec socket with. Commands
run for real, in a local subprocess started in `FakeRunta.root`, so they share one host directory and
deadlines are real.

It follows what the live API was measured to do (2026-11-23): a runtime path takes a UUID only, a
create repeated with the same `Idempotency-Key` returns the same runtime, the exec handshake is
refused with an HTTP status unless the runtime is running, closing the socket kills a command that is
still running, and a process ended by a signal reports exit code -1.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import signal
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosedError, InvalidStatus
from websockets.http11 import Response

_DRAIN_GRACE = 0.2


@dataclass
class FakeRuntime:
    id: str
    name: str
    status: str
    revision: int = 1
    image: str | None = None
    resources: dict[str, object] = field(default_factory=dict[str, object])


class FakeExecSocket:
    """One exec WebSocket, from the start frame to the exit frame."""

    def __init__(self, fake: FakeRunta, runtime: FakeRuntime) -> None:
        self._fake = fake
        self._runtime = runtime
        self._frames: asyncio.Queue[str | None] = asyncio.Queue()
        self._process: asyncio.subprocess.Process | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self.closed = False

    async def send(self, message: str) -> None:
        frame = json.loads(message)
        if frame['type'] == 'start':
            self._fake.commands.append([frame['command'], *frame['args']])
            # The live API sends heartbeats between output frames.
            for injected in ('{"type": "heartbeat"}', *self._fake.frames_next):
                self._frames.put_nowait(injected)
            self._fake.frames_next = []
            self._process = await asyncio.create_subprocess_exec(
                frame['command'],
                *frame['args'],
                cwd=self._fake.root,
                env={**os.environ, **frame['env']},
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # Its own process group, so closing the socket can end the command and its children.
                start_new_session=True,
            )
            self._tasks.append(asyncio.create_task(self._serve(self._process)))

    async def _pump(self, kind: str, stream: asyncio.StreamReader) -> None:
        while chunk := await stream.read(65536):
            self._frames.put_nowait(json.dumps({'type': kind, 'data_base64': base64.b64encode(chunk).decode()}))

    async def _serve(self, process: asyncio.subprocess.Process) -> None:
        assert process.stdout is not None and process.stderr is not None
        pumps = [
            asyncio.create_task(self._pump('stdout', process.stdout)),
            asyncio.create_task(self._pump('stderr', process.stderr)),
        ]
        code = await process.wait()
        # A background child can keep the pipes open: report the exit after a short drain.
        _, pending = await asyncio.wait(pumps, timeout=_DRAIN_GRACE)
        for pump in pending:
            pump.cancel()
        if self._fake.drop_exit:
            if self._fake.lost_runtime_status == 'deleted':
                self._fake.runtimes.pop(self._runtime.id, None)
            elif self._fake.lost_runtime_status:
                self._runtime.status = self._fake.lost_runtime_status
            # The live API ends a command whose runtime was deleted with an error frame.
            lost = '{"type": "error", "message": "upstream internal error"}' if self._fake.lost_as_error else None
            self._frames.put_nowait(lost)
        elif self._fake.end_without_exit:
            self._frames.put_nowait('')
        else:
            self._frames.put_nowait(json.dumps({'type': 'exit', 'code': code if code >= 0 else -1}))

    def __aiter__(self) -> AsyncIterator[str]:
        return self._messages()

    async def _messages(self) -> AsyncIterator[str]:
        while True:
            frame = await self._frames.get()
            if frame is None:
                raise ConnectionClosedError(None, None)
            if frame == '':
                return
            yield frame

    async def close(self) -> None:
        self.closed = True
        process = self._process
        if process is not None and process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):  # pragma: no cover - the command exited as the socket closed
                pass
        for task in self._tasks:
            await asyncio.wait([task])


class FakeRunta:
    """An in-process Runta: runtimes in a dict, commands in local subprocesses under `root`."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.runtimes: dict[str, FakeRuntime] = {}
        self.requests: list[tuple[str, str]] = []
        self.commands: list[list[str]] = []
        self.sockets: list[FakeExecSocket] = []
        self.token = 'test-token'
        # Status a new runtime starts in; it runs once it has been read.
        self.initial_status = 'creating'
        # Set to break exec streams before their exit frame, as a dropped connection does.
        self.drop_exit = False
        # What becomes of a runtime whose exec stream breaks: `'deleted'`, a status such as `'crashed'`, or `''`.
        self.lost_runtime_status = ''
        self.lost_as_error = False
        # Set to close exec streams normally, but without their exit frame.
        self.end_without_exit = False
        # Frames the next exec stream sends before the command's own output.
        self.frames_next: list[str] = []
        # Statuses the next exec handshakes are refused with, oldest first.
        self.refuse_next: list[int] = []
        # Set to answer the next runtime read with a body that is not a runtime.
        self.malformed_next = False
        # Set to delete a runtime as soon as it is told to start or resume.
        self.delete_on_wake = False
        # Statuses queued for the next REST answers, oldest first.
        self.fail_next: list[int] = []
        self._keys: dict[str, str] = {}

    def create(self, name: str = 'existing', *, status: str = 'running') -> FakeRuntime:
        runtime = FakeRuntime(id=str(uuid.uuid4()), name=name, status=status)
        self.runtimes[runtime.id] = runtime
        return runtime

    @staticmethod
    def _json(status: int, body: object) -> httpx.Response:
        return httpx.Response(status, json=body)

    def _view(self, runtime: FakeRuntime) -> httpx.Response:
        data = {'id': runtime.id, 'display_name': runtime.name, 'status': runtime.status, 'revision': runtime.revision}
        return self._json(200, {'data': data})

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.method, request.url.path))
        if self.fail_next:
            return self._json(self.fail_next.pop(0), {'error': {'code': 'unavailable', 'message': 'injected'}})
        if request.headers.get('Authorization') != f'Bearer {self.token}':
            return self._json(401, {'error': {'code': 'unauthenticated', 'message': 'invalid API key'}})
        parts = request.url.path.strip('/').split('/')[2:]
        if not parts:
            return self._create(request)
        try:
            runtime_id = str(uuid.UUID(parts[0]))
        except ValueError:
            return self._json(422, {'error': {'code': 'invalid_argument', 'message': 'runtime_id must be a UUID'}})
        runtime = self.runtimes.get(runtime_id)
        if runtime is None:
            return self._json(404, {'error': {'code': 'not_found', 'message': f'runtime {runtime_id} not found'}})
        if request.method == 'GET':
            if self.malformed_next:
                self.malformed_next = False
                return self._json(200, {'data': {'id': runtime.id}})
            response = self._view(runtime)
            if runtime.status in ('creating', 'resuming'):
                runtime.status = 'running'
            return response
        if request.url.params.get('expected_revision') != str(runtime.revision):
            return self._json(409, {'error': {'code': 'failed_precondition', 'message': 'stale revision'}})
        runtime.revision += 1
        if request.method == 'DELETE':
            del self.runtimes[runtime_id]
        else:
            # The API returns before the change lands.
            runtime.status = 'resuming'
            if self.delete_on_wake:
                del self.runtimes[runtime_id]
        return self._view(runtime)

    def _create(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        key = request.headers['Idempotency-Key']
        if key not in self._keys:
            runtime = self.create(body['name'], status=self.initial_status)
            runtime.image = body.get('image', {}).get('id')
            runtime.resources = body.get('resources', {})
            self._keys[key] = runtime.id
        return self._view(self.runtimes[self._keys[key]])

    async def connect(
        self, url: str, *, additional_headers: dict[str, str], open_timeout: float, max_size: int | None
    ) -> FakeExecSocket:
        del open_timeout, max_size
        runtime = self.runtimes.get(url.split('/')[-3])
        if self.refuse_next:
            raise InvalidStatus(Response(self.refuse_next.pop(0), 'Refused', Headers()))
        if additional_headers.get('Authorization') != f'Bearer {self.token}':
            raise InvalidStatus(Response(401, 'Unauthorized', Headers()))
        if runtime is None:
            raise InvalidStatus(Response(404, 'Not Found', Headers()))
        if runtime.status != 'running':
            raise InvalidStatus(Response(409, 'Conflict', Headers()))
        socket = FakeExecSocket(self, runtime)
        self.sockets.append(socket)
        return socket
