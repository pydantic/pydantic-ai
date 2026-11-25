"""A controllable fake of the Platinum API, served to the real `platinum` SDK through `httpx.MockTransport`.

Only the HTTP exchange is faked: the SDK's client, retries, error classes, sandbox and process handles and its
process stream decoder are the installed ones, so the backend runs against what production raises and returns.

The fake's sandboxes are this machine. An exec or managed process runs its command as a host process in a session of
its own, as Platinum runs each one in its own process group: a timed-out exec has its group killed and answers with
the output so far, and the sandbox stops waiting for output shortly after the command exits, even while a
background child holds the pipes. Deleting a sandbox kills its commands, and an exec it interrupts answers `502`.
File operations act on host paths and answer with the status, `code` and `strerror` text Platinum's API uses.
"""

from __future__ import annotations

import asyncio
import base64
import errno
import json
import os
import shutil
import signal
import stat
import subprocess
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeGuard
from urllib.parse import parse_qs, urlsplit

import anyio
import httpx
from platinum.aio import AsyncPlatinum

from tests._timeouts import READINESS_WAIT_TIMEOUT

API_URL = 'http://platinum.test'
# Platinum's documented limits: an exec's output, one write, one read.
MAX_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_WRITE_BYTES = 128 * 1024 * 1024 - 4096
MAX_READ_BYTES = 256 * 1024 * 1024
# Output a background child keeps writing after its command exits is read for this long, then left.
_DRAIN_GRACE = 0.2

Json = dict[str, object]


def is_str_keyed_dict(value: object) -> TypeGuard[dict[str, object]]:
    """Whether `value` is a JSON object, narrowed so its values are `object` rather than unknown."""
    return isinstance(value, dict) and all(isinstance(key, str) for key in value)  # pyright: ignore[reportUnknownVariableType]


def is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


@dataclass
class Outcome:
    """How a command ended, as the sandbox reports it."""

    exit_code: int
    stdout: bytes = b''
    stderr: bytes = b''
    timed_out: bool = False
    signal: int | None = None
    dropped: int = 0
    """Bytes a managed process's stdout lost before it was read."""


class _Command(asyncio.SubprocessProtocol):
    """A host process's output, and when it exited: `Process.wait()` would also wait for a background child to
    close the pipes, which the sandbox doesn't."""

    def __init__(self) -> None:
        self.stdout = bytearray()
        self.stderr = bytearray()
        self.exited = asyncio.Event()
        self.closed = asyncio.Event()

    def pipe_data_received(self, fd: int, data: bytes) -> None:
        (self.stdout if fd == 1 else self.stderr).extend(data)

    def process_exited(self) -> None:
        self.exited.set()

    def connection_lost(self, exc: Exception | None) -> None:
        self.closed.set()


@dataclass
class FakeSandbox:
    id: str
    name: str | None
    state: str = 'running'
    env: dict[str, str] = field(default_factory=dict[str, str])
    body: Json = field(default_factory=Json)
    """The create request's body."""
    jobs: set[asyncio.Task[Outcome]] = field(default_factory=set[asyncio.Task[Outcome]])
    """The commands running in it, killed when it is deleted."""
    next_states: list[str] = field(default_factory=list[str])
    """States the next lookups report, one each, before it settles in `state`."""


@dataclass
class FakeProcess:
    sandbox: FakeSandbox
    body: Json
    task: asyncio.Task[Outcome]
    killed: bool = False


@dataclass
class Request:
    method: str
    path: str
    query: dict[str, list[str]]
    body: object


ExecHook = Callable[[list[str], int | None], 'Outcome | httpx.Response | Exception | None']
"""Given a command and its timeout in ms, an outcome or response to answer instead of running it, or an
exception for the transport to raise."""


@dataclass
class FakePlatinum:
    """The control plane, plus what tests set and inspect."""

    sandboxes: dict[str, FakeSandbox] = field(default_factory=dict[str, FakeSandbox])
    processes: dict[str, FakeProcess] = field(default_factory=dict[str, FakeProcess])
    requests: list[Request] = field(default_factory=list[Request])
    clients: list[AsyncPlatinum] = field(default_factory=list[AsyncPlatinum])
    closed: list[AsyncPlatinum] = field(default_factory=list[AsyncPlatinum])
    """The clients something closed, in order."""
    exec_hook: ExecHook | None = None
    route_hook: Callable[[Request], Awaitable[httpx.Response | None]] | None = None
    """Given any request, a response to answer instead of serving it; it can also wait, or raise for the transport."""
    max_output_bytes: int = MAX_OUTPUT_BYTES
    max_write_bytes: int = MAX_WRITE_BYTES
    max_read_bytes: int = MAX_READ_BYTES
    max_list_entries: int = 100_000
    _children: list[asyncio.SubprocessTransport] = field(default_factory=list[asyncio.SubprocessTransport])
    _count: int = 0

    def client(self, **_: object) -> AsyncPlatinum:
        """A real `AsyncPlatinum` whose requests reach this fake; it stands in for the SDK's constructor."""
        client = AsyncPlatinum(token='pt_test', api_url=API_URL, transport=httpx.MockTransport(self.handle), retries=0)
        close = client.aclose

        async def aclose() -> None:
            self.closed.append(client)
            await close()

        setattr(client, 'aclose', aclose)
        self.clients.append(client)
        return client

    def add(self, state: str = 'running', name: str | None = None) -> str:
        """A sandbox that exists already, as one created earlier would."""
        self._count += 1
        sandbox = FakeSandbox(id=f'sbx_{self._count:04d}', name=name, state=state)
        self.sandboxes[sandbox.id] = sandbox
        return sandbox.id

    def calls(self, method: str, suffix: str = '') -> list[Request]:
        return [request for request in self.requests if request.method == method and request.path.endswith(suffix)]

    def commands(self) -> list[list[str]]:
        """The argv of every exec and process start."""
        commands: list[list[str]] = []
        for request in self.requests:
            if request.method == 'POST' and request.path.endswith(('/exec', '/processes')):
                assert is_str_keyed_dict(request.body)
                cmd = request.body.get('cmd')
                assert is_object_list(cmd)
                commands.append([str(arg) for arg in cmd])
        return commands

    async def reap(self) -> None:
        """Kill every command still running and close its pipes, so none outlives the test."""
        for sandbox in self.sandboxes.values():
            if sandbox.jobs:  # pragma: lax no cover - only a test that ends with a command still running
                for job in sandbox.jobs:
                    job.cancel()
                await asyncio.wait(sandbox.jobs)
        for child in self._children:
            with suppress(ProcessLookupError):
                os.killpg(child.get_pid(), signal.SIGKILL)
            child.close()
        for client in self.clients:
            if client not in self.closed:
                await client.aclose()

    @staticmethod
    def error(status: int, code: str, message: str = '', **extra: object) -> httpx.Response:
        return httpx.Response(status, json={'code': code, 'error': message or code, **extra})

    async def handle(self, request: httpx.Request) -> httpx.Response:
        url = urlsplit(str(request.url))
        query = parse_qs(url.query)
        body: object = None
        if request.headers.get('content-type', '').startswith('application/json'):
            body = json.loads(request.content)
        recorded = Request(request.method, url.path, query, body)
        self.requests.append(recorded)
        if self.route_hook is not None and (hooked := await self.route_hook(recorded)) is not None:
            return hooked
        parts = url.path.strip('/').split('/')
        assert parts[:2] == ['v1', 'sandboxes'], url.path
        if len(parts) == 2:
            return await self._sandboxes(request.method, body)
        sandbox = self.sandboxes.get(parts[2])
        if sandbox is None or sandbox.state == 'deleted':
            return self.error(404, 'sandbox_not_found', 'sandbox not found')
        operation = parts[3] if len(parts) > 3 else ''
        if operation in ('', 'start'):
            return self._lifecycle(request.method, sandbox, operation)
        if sandbox.state != 'running':
            return self.error(409, 'sandbox_not_running', 'sandbox not running', state=sandbox.state)
        if operation == 'exec':
            assert is_str_keyed_dict(body)
            return await self._exec(sandbox, body)
        if operation == 'processes':
            return await self._processes(request.method, sandbox, parts[4:], body)
        assert operation == 'files', url.path
        return await self._files(request, parts[4:], query)

    async def _sandboxes(self, method: str, body: object) -> httpx.Response:
        if method == 'GET':
            rows = [{'id': s.id, 'name': s.name, 'state': s.state} for s in self.sandboxes.values()]
            return httpx.Response(200, json={'rows': rows[::-1], 'total': len(rows), 'has_more': False})
        assert is_str_keyed_dict(body)
        name = body.get('name')
        assert name is None or isinstance(name, str)
        if any(s.name == name and s.state != 'deleted' for s in self.sandboxes.values()):
            return self.error(409, 'name_taken', f'a sandbox named {name} already exists')
        sandbox = self.sandboxes[self.add(name=name)]
        env = body.get('env')
        sandbox.env = {str(k): str(v) for k, v in env.items()} if is_str_keyed_dict(env) else {}
        sandbox.body = body
        return httpx.Response(201, json={'id': sandbox.id, 'state': 'running', 'name': name})

    def _lifecycle(self, method: str, sandbox: FakeSandbox, operation: str) -> httpx.Response:
        if operation == 'start':
            sandbox.state = 'running'
        elif method == 'DELETE':
            sandbox.state = 'deleted'
            for job in sandbox.jobs:
                job.cancel()
            return httpx.Response(200, json={'ok': True})
        elif sandbox.next_states:
            return httpx.Response(
                200, json={'id': sandbox.id, 'state': sandbox.next_states.pop(0), 'name': sandbox.name}
            )
        return httpx.Response(200, json={'id': sandbox.id, 'state': sandbox.state, 'name': sandbox.name})

    def _start(
        self, sandbox: FakeSandbox, argv: list[str], timeout_ms: int | None, env: object
    ) -> asyncio.Task[Outcome]:
        extra = {str(k): str(v) for k, v in env.items()} if is_str_keyed_dict(env) else {}
        environment = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'LC_ALL': 'C.UTF-8', **sandbox.env, **extra}
        task = asyncio.ensure_future(self._run(argv, environment, timeout_ms))
        sandbox.jobs.add(task)
        task.add_done_callback(sandbox.jobs.discard)
        return task

    async def _run(self, argv: list[str], env: dict[str, str], timeout_ms: int | None) -> Outcome:
        transport, child = await asyncio.get_running_loop().subprocess_exec(
            _Command,
            *argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env=env,
        )
        self._children.append(transport)
        group = transport.get_pid()
        timed_out = False
        try:
            try:
                await asyncio.wait_for(child.exited.wait(), None if timeout_ms is None else timeout_ms / 1000)
            except TimeoutError:
                timed_out = True
                os.killpg(group, signal.SIGKILL)  # the whole group, as the sandbox does
                await child.exited.wait()
            with suppress(TimeoutError):
                # real-wait: the sandbox stops reading a finished command's pipes after a short grace.
                await asyncio.wait_for(child.closed.wait(), _DRAIN_GRACE)
        except asyncio.CancelledError:
            with suppress(ProcessLookupError):
                os.killpg(group, signal.SIGKILL)
            raise
        finally:
            transport.close()
        code = transport.get_returncode()
        code = -1 if code is None else code
        stdout, stderr = bytes(child.stdout), bytes(child.stderr)
        if code < 0:
            return Outcome(exit_code=128 - code, stdout=stdout, stderr=stderr, timed_out=timed_out, signal=-code)
        return Outcome(exit_code=code, stdout=stdout, stderr=stderr)

    async def _exec(self, sandbox: FakeSandbox, body: Json) -> httpx.Response:
        cmd, timeout_ms = body.get('cmd'), body.get('timeout_ms')
        assert is_object_list(cmd) and isinstance(timeout_ms, int)
        argv = [str(arg) for arg in cmd]
        hooked = self.exec_hook(argv, timeout_ms) if self.exec_hook else None
        if isinstance(hooked, Exception):
            raise hooked
        if isinstance(hooked, httpx.Response):
            return hooked
        if hooked is None:
            try:
                hooked = await asyncio.shield(self._start(sandbox, argv, timeout_ms, None))
            except asyncio.CancelledError:
                if sandbox.state != 'deleted':
                    raise
                return self.error(502, 'guest_unreachable', 'the sandbox stopped answering')
        if len(hooked.stdout) + len(hooked.stderr) > self.max_output_bytes:
            return self.error(413, 'output_too_large', 'output too large', max_bytes=self.max_output_bytes)
        # The sandbox omits empty fields.
        result: Json = {'ok': not hooked.timed_out, 'exit_code': -1 if hooked.timed_out else hooked.exit_code}
        result.update(
            {k: v.decode(errors='replace') for k, v in (('stdout', hooked.stdout), ('stderr', hooked.stderr)) if v}
        )
        if hooked.timed_out:
            result.update(error='timeout', timed_out=True)
            return httpx.Response(200, json={'result': result, 'error': 'timeout'})
        return httpx.Response(200, json={'result': result})

    async def _processes(self, method: str, sandbox: FakeSandbox, rest: list[str], body: object) -> httpx.Response:
        if not rest:
            assert is_str_keyed_dict(body)
            cmd, timeout_ms = body.get('cmd'), body.get('timeout_ms')
            assert is_object_list(cmd) and (timeout_ms is None or isinstance(timeout_ms, int))
            argv = [str(arg) for arg in cmd]
            hooked = self.exec_hook(argv, timeout_ms) if self.exec_hook else None
            assert not isinstance(hooked, httpx.Response | Exception), 'a managed process start does not fail this way'
            if hooked is None:
                task = self._start(sandbox, argv, timeout_ms, body.get('env'))
            else:
                task = asyncio.ensure_future(asyncio.sleep(0, result=hooked))
            self._count += 1
            handle = f'proc_{self._count:04d}'
            self.processes[handle] = FakeProcess(sandbox, body, task)
            return httpx.Response(200, json={'handle': handle, 'pid': 100, 'pty': False, 'running': True})
        process = self.processes[rest[0]]
        if method == 'DELETE':
            process.killed = True
            process.task.cancel()
            return httpx.Response(200, json={'ok': True})
        assert rest[1:] == ['stream'], rest
        return self._stream(await asyncio.shield(process.task))

    @staticmethod
    def _stream(outcome: Outcome) -> httpx.Response:
        """The process's output and exit as server-sent events, as Platinum streams them."""
        frames: list[str] = []
        offsets = {'stdout': 0, 'stderr': 0}
        for name, data in (('stdout', outcome.stdout), ('stderr', outcome.stderr)):
            if data:
                offsets[name] = len(data)
                payload: Json = {'data': base64.b64encode(data).decode(), 'encoding': 'base64', 'offset': len(data)}
                if name == 'stdout' and outcome.dropped:
                    payload['dropped'] = outcome.dropped
                frames.append(f'event: {name}\ndata: {json.dumps(payload)}\n\n')
        exit_payload: Json = {
            'exit_code': outcome.exit_code,
            'timed_out': outcome.timed_out,
            'stdout_offset': offsets['stdout'],
            'stderr_offset': offsets['stderr'],
        }
        if outcome.signal:
            exit_payload['signal'] = outcome.signal
        frames.append(f'event: exit\ndata: {json.dumps(exit_payload)}\n\n')
        return httpx.Response(200, text=''.join(frames), headers={'content-type': 'text/event-stream'})

    async def _files(self, request: httpx.Request, rest: list[str], query: dict[str, list[str]]) -> httpx.Response:
        path = query['path'][0]
        assert '..' not in path.split('/'), 'Platinum refuses a `..` segment, so the backend normalizes paths'
        operation = rest[0] if rest else ''
        if operation == 'stat':
            return self._stat(path)
        if operation == 'list':
            return self._list(path)
        if operation == 'mkdir':
            return self._mkdir(path)
        if request.method == 'DELETE':
            return self._delete(path, recurse=query.get('recurse') == ['true'])
        if request.method == 'PUT':
            return self._write(path, request.content)
        return await self._read(path)

    def _stat(self, path: str) -> httpx.Response:
        try:
            info = os.stat(path)
        except OSError as error:
            return self.error(404, 'not_found', _strerror('stat', path, error))
        return httpx.Response(
            200,
            json={
                'ok': True,
                'size': info.st_size,
                'mode': stat.S_IMODE(info.st_mode),
                'is_dir': stat.S_ISDIR(info.st_mode),
            },
        )

    def _list(self, path: str) -> httpx.Response:
        try:
            with os.scandir(path) as scan:
                found = list(scan)
        except OSError as error:
            return self.error(404, 'not_found', _strerror('open', path, error))
        if len(found) > self.max_list_entries:
            return self.error(413, 'list_too_large', 'listing too large')
        entries: list[Json] = []
        for entry in found:
            info = entry.stat(follow_symlinks=False)
            mode = stat.S_IMODE(info.st_mode)
            entries.append(
                {'name': entry.name, 'size': info.st_size, 'mode': mode, 'is_dir': stat.S_ISDIR(info.st_mode)}
            )
        return httpx.Response(200, json={'ok': True, 'entries': entries})

    def _mkdir(self, path: str) -> httpx.Response:
        try:
            os.makedirs(path, exist_ok=True)
        except (FileExistsError, NotADirectoryError):
            return self.error(
                409,
                'not_a_directory',
                f'cannot mkdir {path}: a component of the path is a regular file, not a directory',
            )
        return httpx.Response(200, json={'ok': True})

    def _delete(self, path: str, *, recurse: bool) -> httpx.Response:
        if not os.path.lexists(path):
            if recurse:
                return httpx.Response(200, json={'ok': True})
            return self.error(404, 'path_not_found', f'no such file or directory: {path}')
        if os.path.isdir(path) and not os.path.islink(path):
            if recurse:
                shutil.rmtree(path)
            elif os.listdir(path):
                return self.error(
                    409,
                    'directory_not_empty',
                    f'directory {path} is not empty; pass recurse=true to delete it and everything under it',
                )
            else:
                os.rmdir(path)
        else:
            os.remove(path)
        return httpx.Response(200, json={'ok': True})

    def _write(self, path: str, content: bytes) -> httpx.Response:
        if len(content) > self.max_write_bytes:
            return self.error(
                413, 'payload_too_large', 'file too large for a single write', max_bytes=self.max_write_bytes
            )
        target = Path(path)
        if target.is_dir():
            return self.error(409, 'not_a_file', f'cannot write {path}: it is a directory')
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except (FileExistsError, NotADirectoryError):
            return self.error(
                409,
                'not_a_directory',
                f'cannot write {path}: a component of the path is a regular file, not a directory',
            )
        try:
            with open(path, 'wb') as file:  # through a symlink, as the sandbox's open does
                file.write(content)
        except PermissionError as error:
            # Platinum answers a write the sandbox refused with a server error carrying the `strerror` text.
            return self.error(500, 'internal', _strerror('open', path, error))
        return httpx.Response(200, json={'ok': True})

    async def _read(self, path: str) -> httpx.Response:
        target = Path(path)
        if target.is_dir():
            return self.error(404, 'not_found', 'is a directory')
        if target.is_fifo():
            await asyncio.Event().wait()  # the sandbox's open waits for a writer
        try:
            data = target.read_bytes()
        except OSError as error:
            code = 'path_not_found' if error.errno == errno.ENOENT else 'not_found'
            return self.error(404, code, _strerror('open', path, error))
        if len(data) > self.max_read_bytes:
            return self.error(413, 'file_too_large', 'file too large', size=len(data), max_bytes=self.max_read_bytes)
        return httpx.Response(200, content=data)


def _strerror(operation: str, path: str, error: OSError) -> str:
    """The sandbox's error text for a failed system call, as the API passes it on."""
    return f'{operation} {path}: {os.strerror(error.errno or 0).lower()}'


async def wait_until_gone(pid_file: Path) -> None:
    """Wait for the host process whose pid is in `pid_file` to end, as a stopped command's does."""
    pid = int(pid_file.read_text(encoding='utf-8'))
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        # real-wait: polls a host process that a stop request killed; there is nothing to wait on.
        while True:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            await anyio.sleep(0.01)  # pragma: lax no cover - depends on how fast the process dies
