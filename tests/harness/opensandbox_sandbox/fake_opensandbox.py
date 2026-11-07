"""A fake OpenSandbox server, behind the httpx transport the real SDK sends its requests through.

The SDK runs unchanged: it builds the requests, parses execd's event stream and converts failures into its own
exceptions. Only the server is fake. Its core answers copy traffic recorded from `opensandbox-server` 1.2.0 with execd
1.2.0 on 2026-10-29:

- A create answers 202 with the sandbox; its endpoints point at execd, which answers `/ping` and streams
  `/command` as JSON events separated by blank lines: `init`, `ping`, `stdout` and `stderr` events carrying the
  output as it is written, newlines kept, then `execution_complete` for exit 0 or an `error` event carrying the
  exit code. A signal death reports `-1`. A missing `cwd` is refused with 400 before the command starts.
- Errors are JSON bodies with a runtime-prefixed `code` and the HTTP status the server uses.
- Secure endpoint headers model the published 1.2.0 gateway contract: a matching per-sandbox token is required.

Commands run on the host, through `bash -c` like execd's, each leading a process group of its own; the backend's
`/tmp/pydantic-ai` is redirected to `pid_dir`, so tests leave nothing in the host's `/tmp`. As in
production, a command keeps running when its client closes the stream, and killing a sandbox kills its commands
and breaks their streams. The knobs below reproduce what was seen on Kubernetes and on old execd versions.
"""

from __future__ import annotations

import asyncio
import codecs
import email.parser
import json
import os
import shutil
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
_IDLE_TURNS_AFTER_EXIT = 1000
"""Event loop turns with nothing to deliver after a command exits, which the readers need to queue what it wrote."""
_MAX_CHARS_AFTER_EXIT = 1 << 20
"""Characters of output after a command exits: more than it can have left unread in a full pipe, far below the backend's
limit."""


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
    network_policy: dict[str, Any] | None = None
    secure_access: bool = False
    state: str = 'Running'
    processes: dict[str, asyncio.subprocess.Process] = field(default_factory=lambda: {})
    streams: list[asyncio.Queue[tuple[str, Any]]] = field(default_factory=lambda: [])
    stream_breaks: int = 0
    """How often its streams were broken, so a command starting meanwhile can tell."""

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
        self.stale_running_queries = 0
        """Report a killed sandbox as `Running` for this many state queries before reporting `Stopping`."""
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
                headers = (
                    {'OpenSandbox-Secure-Access': f'fake-token-{sandbox_id}'}
                    if self.sandboxes[sandbox_id].secure_access
                    else {}
                )
                return _json(200, {'endpoint': f'execd.{DOMAIN}/{sandbox_id}/proxy/{port}', 'headers': headers})
            # A request the backend never makes.
            case _:  # pragma: no cover
                raise AssertionError(f'unexpected request {request.method} {path}')

    async def _create(self, body: dict[str, Any]) -> httpx.Response:
        timeout = body.get('timeout')
        sandbox = FakeSandbox(
            id=str(uuid.uuid4()),
            image=body['image']['uri'],
            metadata=body.get('metadata') or {},
            env=body.get('env') or {},
            resource=body.get('resourceLimits') or {},
            network_policy=body.get('networkPolicy'),
            secure_access=body.get('secureAccess', False),
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
        info = sandbox.info()
        if sandbox.state == 'Stopping' and self.stale_running_queries:
            self.stale_running_queries -= 1
            info['status']['state'] = 'Running'
        return _json(200, info)

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
        sandbox = sandbox or self.sandboxes[sandbox_id]
        sandbox.stream_breaks += 1
        for queue in sandbox.streams:
            queue.put_nowait(('broken', None))

    async def _execd(self, request: httpx.Request) -> httpx.Response:
        sandbox_id, _, _port, *rest = request.url.path.strip('/').split('/')
        sandbox = self.sandboxes.get(sandbox_id)
        if sandbox is None or sandbox.state == 'Pending':
            raise httpx.ConnectError('connection refused')
        if sandbox.secure_access and request.headers.get('OpenSandbox-Secure-Access') != f'fake-token-{sandbox_id}':
            return _error(401, 'UNAUTHORIZED', 'Missing or invalid sandbox endpoint token.')
        match rest:
            case ['ping']:
                # A sandbox being deleted still answers for a few seconds, as on Kubernetes.
                return _json(200)
            case ['command']:
                if sandbox.state != 'Running':
                    raise httpx.ConnectError('connection refused')
                return await self._command(sandbox, json.loads(request.content))
            case ['files', 'info']:
                return self._files_info(request)
            case ['files', 'download']:
                return self._files_download(request)
            case ['files', 'upload']:
                return await self._files_upload(request)
            case ['files', 'mv']:
                return self._files_mv(request)
            case ['files', 'permissions']:
                return self._files_permissions(request)
            case ['files']:
                return self._files_delete(request)
            case ['directories', 'list']:
                return self._files_list(request)
            case ['directories']:
                return self._directories(request)
            # A request the backend never makes.
            case _:  # pragma: no cover
                raise AssertionError(f'unexpected execd request {request.url.path}')

    async def _command(self, sandbox: FakeSandbox, body: dict[str, Any]) -> httpx.Response:
        cwd = body.get('cwd') or self.image_working_dir
        if not os.path.isdir(cwd):
            return _error(400, 'INVALID_REQUEST_BODY', f'working directory does not exist: {cwd}')
        stream_breaks = sandbox.stream_breaks
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
        # The command can run before this registers it, so a kill or a stream break that came meanwhile, and saw
        # neither the process nor its stream, takes effect here.
        if sandbox.state != 'Running':  # pragma: lax no cover
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        if sandbox.stream_breaks != stream_breaks:  # pragma: lax no cover
            queue.put_nowait(('broken', None))
        return httpx.Response(
            200,
            headers={'content-type': 'text/event-stream'},
            stream=_EventStream(self, sandbox, execution_id, process, queue),
        )

    def _entry_info(self, path: str) -> dict[str, Any] | None:
        """execd `get_files_info`: symlinks are not followed, missing paths just omit the key."""
        try:
            st = os.lstat(path)
        except OSError:
            return None
        kind = st.st_mode & 0o170000
        # Unix file-type masks: symlink, directory, FIFO, regular file.
        if kind == 0o120000:
            entry_type = 'symlink'
        elif kind == 0o040000:
            entry_type = 'directory'
        elif kind == 0o010000:
            entry_type = 'other'
        else:
            entry_type = 'file'
        return {
            'path': path,
            'type': entry_type,
            'mode': int(f'{st.st_mode & 0o777:o}'),
            'size': st.st_size,
            'modified_at': datetime.fromtimestamp(st.st_mtime, tz=UTC).isoformat(),
            'created_at': datetime.fromtimestamp(st.st_ctime, tz=UTC).isoformat(),
            'owner': str(st.st_uid),
            'group': str(st.st_gid),
        }

    def _files_info(self, request: httpx.Request) -> httpx.Response:
        infos = {}
        for path in request.url.params.get_list('path'):
            info = self._entry_info(path)
            if info is not None:
                infos[path] = info
        return httpx.Response(200, json=infos)

    def _files_download(self, request: httpx.Request) -> httpx.Response:
        path = request.url.params['path']
        info = self._entry_info(path)
        if info is None:
            return _error(404, 'NOT_FOUND', f'no such file: {path}')
        if info['type'] == 'directory':
            # execd answers a download of a directory with a 500.
            return _error(500, 'INTERNAL_ERROR', f'cannot read a directory: {path}')
        offset = int(range_start[6:].split('-')[0]) if (range_start := request.headers.get('range', 'bytes=0-')) else 0
        try:
            with open(path, 'rb') as file:
                file.seek(offset)
                data = file.read()
        except OSError as error:
            # execd answers a file the shell cannot open with a 500 carrying the errno text.
            return _error(500, 'INTERNAL_ERROR', f'open {path}: {(error.strerror or str(error)).lower()}')
        return httpx.Response(200, content=data, headers={'accept-ranges': 'bytes'})

    async def _files_upload(self, request: httpx.Request) -> httpx.Response:
        # The SDK's direct-to-execd upload streams an async-generator body, so the request can only be read
        # asynchronously. Each entry arrives as two parts: a `metadata` JSON part, then the `file` part.
        ctype = request.headers.get('content-type', '')
        if not ctype.startswith('multipart/form-data'):  # pragma: no cover
            raise AssertionError(f'unexpected /files/upload content-type {ctype!r}')
        body = await request.aread()
        message = email.parser.BytesParser().parsebytes(b'Content-Type: ' + ctype.encode() + b'\r\n\r\n' + body)
        parts: list[tuple[str, bytes]] = []
        for part in message.walk():
            if part.is_multipart():
                continue
            name = str(part.get_param('name', header='content-disposition'))
            raw = part.get_payload(decode=True)
            parts.append((name, raw if isinstance(raw, bytes) else b''))
        # execd applies each entry's mode to the target, replacing any previous one; the backend preserves an
        # existing file's mode by passing it back.
        for metadata_part, file_part in zip(parts[::2], parts[1::2], strict=True):
            metadata = json.loads(metadata_part[1])
            path, mode = str(metadata['path']), int(str(metadata.get('mode', 755)), 8)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            try:
                with open(path, 'wb') as file:
                    file.write(file_part[1])
            except OSError as error:
                # execd answers a file the shell cannot open with a 500 carrying the errno text.
                return _error(500, 'INTERNAL_ERROR', f'open {path}: {(error.strerror or str(error)).lower()}')
            os.chmod(path, mode)
        return _json(200)

    def _files_list(self, request: httpx.Request) -> httpx.Response:
        path = request.url.params['path']
        info = self._entry_info(path)
        if info is None:
            return _error(404, 'NOT_FOUND', f'no such directory: {path}')
        if info['type'] != 'directory':
            # execd refuses to list a directory reached through a symlink.
            return _error(500, 'INTERNAL_ERROR', f'not a directory: {path}')
        entries: list[dict[str, Any]] = []
        for child in sorted(os.listdir(path)):
            child_info = self._entry_info(os.path.join(path, child))
            if child_info is not None:
                entries.append(child_info)
        return httpx.Response(200, json=entries)

    def _files_mv(self, request: httpx.Request) -> httpx.Response:
        for entry in json.loads(request.content):
            src = entry.get('src') or entry.get('source')
            dest = entry.get('dest') or entry.get('destination')
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            os.replace(src, dest)
        return _json(200)

    def _files_permissions(self, request: httpx.Request) -> httpx.Response:
        for path, spec in json.loads(request.content).items():
            os.chmod(path, int(str(spec['mode']), 8))
        return _json(200)

    def _files_delete(self, request: httpx.Request) -> httpx.Response:
        for path in request.url.params.get_list('path'):
            # A missing path deletes without error; a directory must go through delete_directories.
            if os.path.lexists(path) and not (os.path.isdir(path) and not os.path.islink(path)):
                os.remove(path)
        return _json(200)

    def _directories(self, request: httpx.Request) -> httpx.Response:
        if request.method == 'POST':
            for path in json.loads(request.content):
                os.makedirs(path, exist_ok=True)
        else:
            for path in request.url.params.get_list('path'):
                # A missing directory deletes without error.
                shutil.rmtree(path, ignore_errors=True)
        return _json(200)


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
        # open. What the command wrote is in its pipes by then, and is delivered whatever the machine's speed: the
        # drain only counts turns with nothing queued, and stops for a background child's output only after more
        # than any pipe holds. The rest is dropped.
        code: int | None = None
        idle_turns_left = _IDLE_TURNS_AFTER_EXIT
        chars_after_exit = 0
        open_pipes = 2
        while code is None or open_pipes:
            if code is not None:
                if not idle_turns_left or chars_after_exit > _MAX_CHARS_AFTER_EXIT:
                    break
                if self._queue.empty():
                    idle_turns_left -= 1
                    await asyncio.sleep(0)
                    continue
            kind, value = await self._queue.get()
            if kind == 'broken':
                raise httpx.RemoteProtocolError('peer closed connection without sending complete message body')
            if kind == 'exit':
                code = value
            elif kind == 'eof':
                open_pipes -= 1
            else:
                if code is not None:
                    chars_after_exit += len(value)
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
        with anyio.CancelScope(shield=True):
            await asyncio.gather(*self._tasks, return_exceptions=True)
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
        # Depends on how fast the process dies.
        return True  # pragma: lax no cover

    with anyio.fail_after(5):
        while any(alive(pid) for pid in pids):
            # Depends on how fast the process dies.
            await anyio.sleep(0.01)  # pragma: lax no cover
