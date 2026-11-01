"""A fake OpenSandbox server, behind the httpx transport the real SDK sends its requests through.

The SDK runs unchanged: it builds the requests, parses execd's event stream and converts failures into its own
exceptions. Only the server is fake. Its answers copy traffic recorded from `opensandbox-server` 1.2.0 with execd
1.2.0 on 2026-10-29:

- A create answers 202 with the sandbox; its endpoints point at execd, which answers `/ping` and streams
  `/command` as JSON events separated by blank lines: `init`, `ping`, `stdout` and `stderr` events carrying the
  output as it is written, newlines kept, then `execution_complete` for exit 0 or an `error` event carrying the exit code. A signal death
  reports `-1`. A missing `cwd` is refused with 400 before the command starts.
- Errors are JSON bodies with a runtime-prefixed `code` and the HTTP status the server uses.

Commands run on the host, through `bash -c` like execd's, each leading a process group of its own; the backend's
`/tmp/pydantic-ai` is redirected to `pid_dir`, so tests leave nothing in the host's `/tmp`. As in
production, a command keeps running when its client closes the stream, and killing a sandbox kills its commands
and breaks their streams. The knobs below reproduce what was seen on Kubernetes and on old execd versions.
"""

from __future__ import annotations

import asyncio
import codecs
import json
import os
import signal
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import parse_qsl

import anyio
import httpx
from opensandbox.config import ConnectionConfig

API_KEY = 'fake-key'
DOMAIN = 'opensandbox.invalid'
"""A name that cannot resolve, so a request that skips the fake's transport fails instead of leaving the host."""
EXECD_PORT = 44772
_NOT_FOUND = 'DOCKER::SANDBOX_NOT_FOUND'
_DRAIN_GRACE = 0.05


def _now() -> str:
    return datetime.now(UTC).isoformat().replace('+00:00', 'Z')


def _json(status: int, body: Any = None) -> httpx.Response:
    if body is None:
        return httpx.Response(status)
    return httpx.Response(status, json=body)


def _error(status: int, code: str, message: str) -> httpx.Response:
    return _json(status, {'code': code, 'message': message})


@dataclass
class FakeSandbox:
    id: str
    image: str
    metadata: dict[str, str]
    env: dict[str, str]
    resource: dict[str, str]
    expires_at: str | None
    state: str = 'Running'
    processes: dict[str, asyncio.subprocess.Process] = field(default_factory=lambda: {})
    streams: list[asyncio.Queue[tuple[str, Any]]] = field(default_factory=lambda: [])

    def info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            'id': self.id,
            'image': {'uri': self.image},
            'status': {'state': self.state, 'reason': self.state.upper(), 'lastTransitionAt': _now()},
            'metadata': self.metadata,
            'entrypoint': ['tail', '-f', '/dev/null'],
            'createdAt': _now(),
        }
        # A sandbox without a timeout has no `expiresAt` at all.
        if self.expires_at is not None:
            info['expiresAt'] = self.expires_at
        return info


class FakeOpenSandbox(httpx.AsyncBaseTransport):
    """The fake server. Pass `connection` to the backend; the attributes below steer it."""

    def __init__(self, pid_dir: str) -> None:
        self.pid_dir = pid_dir
        self.processes: list[asyncio.subprocess.Process] = []
        self.opened_transports = 0
        self.closed_transports = 0
        self.sandboxes: dict[str, FakeSandbox] = {}
        self.renewals: list[tuple[str, str]] = []
        self.requests: list[str] = []
        self.lose_create_response = False
        """Create the sandbox, then fail the request as a timeout, as an SDK request timeout on Kubernetes does."""
        self.during_create: Callable[[], object] | None = None
        """Called while a create is in flight, after the server made the sandbox, to cancel or fail around it."""
        self.refuse: tuple[int, str] | None = None
        """Answer every request with this status and error code."""
        self.endpoint_error: tuple[int, str] | None = None
        """Answer endpoint lookups with this status and code, which fails a create after the server made it."""
        self.unreachable = False
        """Fail every request to connect, as an unreachable server does."""
        self.stopping = False
        """Leave a killed sandbox reporting `Stopping`, as Kubernetes does for a few seconds, instead of gone."""
        self.api_key: str | None = API_KEY
        """The key the server requires; `None` runs it without authentication, as a local server may."""
        self.strip_newlines = False
        """Send output lines without their newline, as execd before 1.1.1 does."""
        self.image_working_dir = '/'
        """Where a command sent with no `cwd` starts, as execd starts it in the image's working directory."""
        self.connection = ConnectionConfig(domain=DOMAIN, api_key=API_KEY, transport=self)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.requests.append(f'{request.method} {path}')
        if self.unreachable:
            raise httpx.ConnectError('connection refused')
        if self.refuse is not None:
            return _error(*self.refuse, f'refused: {self.refuse[1]}')
        if request.url.host != DOMAIN:
            return await self._execd(request)
        if self.api_key is not None and request.headers.get('open-sandbox-api-key') != self.api_key:
            return _error(401, 'INVALID_API_KEY', 'Authentication credentials are invalid.')
        parts = path.strip('/').split('/')[1:]  # drop the `v1`
        match request.method, parts:
            case 'POST', ['sandboxes']:
                return await self._create(json.loads(request.content))
            case 'GET', ['sandboxes']:
                return self._list(dict(parse_qsl(request.url.query.decode())))
            case 'GET', ['sandboxes', sandbox_id]:
                return self._get(sandbox_id)
            case 'DELETE', ['sandboxes', sandbox_id]:
                return await self._kill(sandbox_id)
            case 'POST', ['sandboxes', sandbox_id, 'renew-expiration']:
                return self._renew(sandbox_id, json.loads(request.content))
            case 'GET', ['sandboxes', sandbox_id, 'endpoints', port]:
                if sandbox_id not in self.sandboxes:
                    return _error(404, _NOT_FOUND, f'Sandbox {sandbox_id} not found.')
                if self.endpoint_error is not None:
                    return _error(*self.endpoint_error, 'endpoint lookup failed')
                return _json(200, {'endpoint': f'execd.{DOMAIN}/{sandbox_id}/proxy/{port}'})
            case _:  # pragma: no cover - a request the backend never makes
                raise AssertionError(f'unexpected request {request.method} {path}')

    async def _create(self, body: dict[str, Any]) -> httpx.Response:
        timeout = body.get('timeout')
        sandbox = FakeSandbox(
            id=str(uuid.uuid4()),
            image=body['image']['uri'],
            metadata=body.get('metadata') or {},
            env=body.get('env') or {},
            resource=body.get('resourceLimits') or {},
            expires_at=None if timeout is None else (datetime.now(UTC) + timedelta(seconds=timeout)).isoformat(),
        )
        self.sandboxes[sandbox.id] = sandbox
        if self.during_create is not None:
            self.during_create()
        # A checkpoint, as a real request has, so a cancellation from `during_create` lands mid-create.
        await asyncio.sleep(0)
        if self.lose_create_response:
            sandbox.state = 'Pending'
            raise httpx.ReadTimeout('timed out')
        return _json(202, sandbox.info())

    def _list(self, query: dict[str, str]) -> httpx.Response:
        wanted = dict(item.split('=', 1) for item in query.get('metadata', '').split('&') if item)
        items = [s.info() for s in self.sandboxes.values() if wanted.items() <= s.metadata.items()]
        return _json(
            200,
            {
                'items': items,
                'pagination': {
                    'page': 1,
                    'pageSize': 20,
                    'totalItems': len(items),
                    'totalPages': 1,
                    'hasNextPage': False,
                },
            },
        )

    def _get(self, sandbox_id: str) -> httpx.Response:
        sandbox = self.sandboxes.get(sandbox_id)
        if sandbox is None:
            return _error(404, _NOT_FOUND, f'Sandbox {sandbox_id} not found.')
        return _json(200, sandbox.info())

    async def _kill(self, sandbox_id: str) -> httpx.Response:
        sandbox = self.sandboxes.get(sandbox_id)
        if sandbox is None or sandbox.state == 'Stopping':
            return _error(404, _NOT_FOUND, f'Sandbox {sandbox_id} not found.')
        await self.kill(sandbox_id)
        return _json(204)

    def _renew(self, sandbox_id: str, body: dict[str, str]) -> httpx.Response:
        if sandbox_id not in self.sandboxes:
            return _error(404, _NOT_FOUND, f'Sandbox {sandbox_id} not found.')
        self.sandboxes[sandbox_id].expires_at = body['expiresAt']
        self.renewals.append((sandbox_id, body['expiresAt']))
        return _json(200, {'expiresAt': body['expiresAt']})

    async def kill(self, sandbox_id: str) -> None:
        """Kill a sandbox as the runtime does: its commands die and their streams break."""
        sandbox = self.sandboxes[sandbox_id]
        sandbox.state = 'Stopping'
        if not self.stopping:
            del self.sandboxes[sandbox_id]
        for process in sandbox.processes.values():
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        self.break_streams(sandbox_id, sandbox)

    def break_streams(self, sandbox_id: str, sandbox: FakeSandbox | None = None) -> None:
        """Break the sandbox's command streams without stopping its commands, as a network failure does."""
        for queue in (sandbox or self.sandboxes[sandbox_id]).streams:
            queue.put_nowait(('broken', None))

    async def _execd(self, request: httpx.Request) -> httpx.Response:
        sandbox_id, _, _port, *rest = request.url.path.strip('/').split('/')
        sandbox = self.sandboxes.get(sandbox_id)
        if sandbox is None or sandbox.state == 'Pending':
            raise httpx.ConnectError('connection refused')
        match rest:
            case ['ping']:
                # A sandbox being deleted still answers for a few seconds, as on Kubernetes.
                return _json(200)
            case ['command']:
                if sandbox.state != 'Running':
                    raise httpx.ConnectError('connection refused')
                return await self._command(sandbox, json.loads(request.content))
            case _:  # pragma: no cover - a request the backend never makes
                raise AssertionError(f'unexpected execd request {request.url.path}')

    async def _command(self, sandbox: FakeSandbox, body: dict[str, Any]) -> httpx.Response:
        cwd = body.get('cwd') or self.image_working_dir
        if not os.path.isdir(cwd):
            return _error(400, 'INVALID_REQUEST_BODY', f'working directory does not exist: {cwd}')
        process = await asyncio.create_subprocess_exec(
            '/bin/bash',
            '-c',
            body['command'].replace('/tmp/pydantic-ai', self.pid_dir),
            cwd=cwd,
            env={'PATH': os.environ['PATH'], **sandbox.env, **(body.get('envs') or {})},
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        execution_id = uuid.uuid4().hex
        sandbox.processes[execution_id] = process
        self.processes.append(process)
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        sandbox.streams.append(queue)
        return httpx.Response(
            200,
            headers={'content-type': 'text/event-stream'},
            stream=_EventStream(self, sandbox, execution_id, process, queue),
        )


class _EventStream(httpx.AsyncByteStream):
    def __init__(
        self,
        server: FakeOpenSandbox,
        sandbox: FakeSandbox,
        execution_id: str,
        process: asyncio.subprocess.Process,
        queue: asyncio.Queue[tuple[str, Any]],
    ) -> None:
        self._server, self._sandbox, self._id, self._process, self._queue = (
            server,
            sandbox,
            execution_id,
            process,
            queue,
        )
        self._tasks: list[asyncio.Task[None]] = []

    async def __aiter__(self) -> AsyncIterator[bytes]:
        def event(payload: dict[str, Any]) -> bytes:
            # The SDK drops an event without an integer `timestamp`, as it would a malformed one.
            return json.dumps({**payload, 'timestamp': time.time_ns() // 1_000_000}).encode() + b'\n\n'

        yield event({'type': 'init', 'text': self._id})
        yield event({'type': 'ping', 'text': 'pong'})
        assert self._process.stdout is not None and self._process.stderr is not None
        self._tasks = [
            asyncio.create_task(self._read('stdout', self._process.stdout)),
            asyncio.create_task(self._read('stderr', self._process.stderr)),
            asyncio.create_task(self._wait()),
        ]
        # execd reports the exit as soon as the command exits, even while a background child holds its output
        # open: once the command has exited, output still arriving is read for a moment, the rest dropped.
        code: int | None = None
        open_pipes = 2
        while code is None or open_pipes:
            try:
                kind, value = (
                    await asyncio.wait_for(self._queue.get(), _DRAIN_GRACE)
                    if code is not None
                    else await self._queue.get()
                )
            except TimeoutError:
                break
            if kind == 'broken':
                raise httpx.RemoteProtocolError('peer closed connection without sending complete message body')
            if kind == 'exit':
                code = value
            elif kind == 'eof':
                open_pipes -= 1
            else:
                yield event({'type': kind, 'text': value})
        self._sandbox.processes.pop(self._id, None)
        if code == 0:
            yield event({'type': 'execution_complete', 'exit_code': 0, 'execution_time': 1})
        else:
            assert code is not None
            exit_code = code if code > 0 else -1
            yield event(
                {
                    'type': 'error',
                    'error': {
                        'ename': 'CommandExecError',
                        'evalue': str(exit_code),
                        'traceback': [],
                        'exit_code': exit_code,
                    },
                }
            )

    async def _wait(self) -> None:
        # Not `Process.wait()`, which before Python 3.14 also waits for a background child to close the pipes:
        # `returncode` is set as soon as the command itself exits.
        while self._process.returncode is None:
            await asyncio.sleep(0.005)
        self._queue.put_nowait(('exit', self._process.returncode))

    async def _read(self, kind: str, pipe: asyncio.StreamReader) -> None:
        # Output as it arrives, a partial line included, as execd sends it.
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        while chunk := await pipe.read(65536):
            text = decoder.decode(chunk)
            for piece in text.splitlines() if self._server.strip_newlines else [text]:
                self._queue.put_nowait((kind, piece))
        self._queue.put_nowait(('eof', None))

    async def aclose(self) -> None:
        # Closing the stream leaves the command running, as execd does.
        for task in self._tasks:
            task.cancel()
        with suppress(ValueError):
            self._sandbox.streams.remove(self._queue)


class OwnedTransport(httpx.AsyncBaseTransport):
    """What the SDK builds for itself when its config has no transport: the server's, counting how often it closes.

    Unlike httpx's own, which opens new connections when used after closing, it refuses, so a closed handle that
    is used again fails the test.
    """

    def __init__(self, server: FakeOpenSandbox) -> None:
        self._server = server
        self._closed = False
        server.opened_transports += 1

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        assert not self._closed, f'request on a closed transport: {request.method} {request.url}'
        return await self._server.handle_async_request(request)

    async def aclose(self) -> None:
        self._closed = True
        self._server.closed_transports += 1


async def reap(server: FakeOpenSandbox) -> None:
    """Kill every command still running and wait for each, so no test leaves host processes or pipes behind."""
    for process in server.processes:
        # The whole group, even once its leader has exited: a background child may still hold the pipes open.
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        # Reads what is left in its pipes, so a pipe paused on a full buffer reaches its end and closes.
        await process.communicate()


async def wait_until_gone(*pids: int) -> None:
    """Wait for host processes to end; a kill is sent, but the kernel reaps them on its own time."""

    def alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True  # pragma: lax no cover - depends on how fast the process dies

    with anyio.fail_after(5):
        while any(alive(pid) for pid in pids):
            await anyio.sleep(0.01)  # pragma: lax no cover - depends on how fast the process dies
