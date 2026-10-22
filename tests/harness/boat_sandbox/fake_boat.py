"""A stateful fake of Boat's v1 REST API, served through `httpx.MockTransport`.

Answers carry Boat's real envelopes, status codes, error codes and messages (observed live on
2026-10-21), so the backend's translation runs against what production returns. Each sandbox is a
pair of host directories, its home and its `/tmp`: commands are real `bash -c` processes in their
own session, so stopping one kills its process group, and file calls act on the host filesystem,
limited to those two directories like Boat's file API.

Knobs, all keyed by route name (`create`, `get`, `update`, `resume`, `delete`, `run`, `start`,
`status`, `kill`, `read`, `write`, `upload_url`, `download_url`, `transfer`):

- `errors[route]`: answers or exceptions served, in order, instead of handling the request; a `None`
  entry lets that request through.
- `lost_replies[route]`: how many requests are handled and then answered with `httpx.ReadTimeout`.
- `gates[route]`: an event a request waits for before it is handled; `reached[route]` is set first.

`calls` records every request that reached a route, with its parsed JSON body.

Boat's refusals of a malformed request are kept as tripwires, marked uncovered: a correct backend
never sends one, and the refusal fails the test that made it.
"""

from __future__ import annotations

import asyncio
import base64
import errno
import os
import re
import signal
import stat
import subprocess
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterator
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

import anyio.to_thread
import httpx
from pydantic import TypeAdapter

BOAT_API_URL = 'https://boat.dev/api/v1'
MACHINE_HOST = 'box-node-fake.on.boat.dev'
MACHINE_TYPES = ('small', 'default', 'large', 'xlarge')
_ID_ALPHABET = '23456789abcdefghjkmnpqrstuvwxyz'
_STREAM_LIMIT = 8 * 1024 * 1024
_OFFSET_PAGE = 1024 * 1024
_FILE_LIMIT = 5 * 1024 * 1024
_COMMAND_LIMIT = 131071
_ROOTS_MESSAGE = 'Path must resolve under /home/user or /tmp (work directory: /home/user).'
_BODY = TypeAdapter(dict[str, object])
_NODE_ERRORS = {
    errno.ENOENT: 'ENOENT: no such file or directory',
    errno.EACCES: 'EACCES: permission denied',
    errno.ENOTDIR: 'ENOTDIR: not a directory',
    errno.EISDIR: 'EISDIR: illegal operation on a directory',
    errno.ELOOP: 'ELOOP: too many symbolic links encountered',
}


def boat_error(status: int, code: str, message: str, **details: object) -> httpx.Response:
    """An error answer in Boat's envelope."""
    return httpx.Response(
        status,
        json={
            'ok': False,
            'type': 'sandbox.error',
            'status': status,
            'code': code,
            'message': message,
            'error': {
                'code': code,
                'message': message,
                'status': status,
                'details': {'error': code, 'message': message, **details},
            },
            'requestId': 'req_00000000000000000000000000000000',
        },
    )


def boat_ok(kind: str, http_status: int = 200, /, **body: object) -> httpx.Response:
    """A successful answer in Boat's envelope."""
    return httpx.Response(http_status, json={'ok': True, 'type': kind, **body})


def _direct_failed(message: str) -> httpx.Response:
    return boat_error(400, 'sandbox_direct_failed', message)


def _tail(data: bytes, limit: int) -> tuple[str, bool]:
    return data[-limit:].decode(errors='replace'), len(data) > limit


@dataclass
class Call:
    """A request that reached one of the fake's routes."""

    route: str
    request: httpx.Request
    body: dict[str, object]


class FakeCommand:
    """A detached command: a host process writing its output to log files, as Boat's agent does."""

    def __init__(self, process: subprocess.Popen[bytes], out: Path, err: Path) -> None:
        self.process = process
        self.out = out
        self.err = err
        self.stopped = False

    def kill(self) -> None:
        """Kill the command's process group, as Boat kills a stopped command's whole tree."""
        # The group outlives its leader while a background child is left in it.
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(self.process.pid, signal.SIGKILL)
        self.process.wait()


class FakeSandbox:
    def __init__(self, sandbox_id: str, root: Path, *, machine_type: str, boot_polls: int) -> None:
        self.id = sandbox_id
        self.state = 'provisioning'
        self.machine_type = machine_type
        self.home = root / sandbox_id / 'home'
        self.tmp = root / sandbox_id / 'tmp'
        self.logs = root / sandbox_id / 'logs'
        for directory in (self.home, self.tmp, self.logs):
            directory.mkdir(parents=True)
        # How many lookups report the sandbox as still starting.
        self.boot_polls = boot_polls
        # Boat's auto-stop in seconds, `None` when it is off.
        self.ttl: int | None = 3600
        self.resume_calls = 0
        self.commands: dict[int, FakeCommand] = {}

    def info(self) -> dict[str, object]:
        return {
            'id': self.id,
            'name': f'Box {self.id}',
            'state': self.state,
            'type': self.machine_type,
            'ttlSeconds': self.ttl,
        }

    def environment(self) -> dict[str, str]:
        return {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'HOME': str(self.home), 'TMPDIR': str(self.tmp)}

    def kill_all(self) -> None:
        for command in self.commands.values():
            command.kill()


class FakeClient(httpx.AsyncClient):
    """An API client whose requests the fake answers, counting its own closing."""

    def __init__(self, fake: FakeBoat, api_key: str) -> None:
        super().__init__(
            transport=httpx.MockTransport(fake.handle),
            base_url=BOAT_API_URL,
            headers={'Authorization': f'Bearer {api_key}'},
        )
        self.fake = fake

    async def aclose(self) -> None:
        if self.fake.close_gate is not None:
            await self.fake.close_gate.wait()
        if (error := next(self.fake.close_errors, None)) is not None:
            raise error
        await super().aclose()
        self.fake.closed_clients += 1


Handler = Callable[[httpx.Request, re.Match[str], dict[str, object]], Awaitable[httpx.Response]]


class FakeBoat:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.api_key = 'test-key'
        self.sandboxes: dict[str, FakeSandbox] = {}
        self.deleted: set[str] = set()
        # Named snapshots a create may start from.
        self.snapshots: set[str] = set()
        self.boot_polls = 0
        self.create_bodies: list[dict[str, object]] = []
        self.requests: list[httpx.Request] = []
        self.calls: list[Call] = []
        self.errors: dict[str, list[httpx.Response | Exception | None]] = defaultdict(list)
        self.lost_replies: dict[str, int] = defaultdict(int)
        self.gates: dict[str, asyncio.Event] = {}
        self.reached: dict[str, asyncio.Event] = defaultdict(asyncio.Event)
        self.clients: list[FakeClient] = []
        self.opened_clients = 0
        self.closed_clients = 0
        # Raised by successive client `aclose` calls; closing succeeds once it is exhausted.
        self.close_errors: Iterator[Exception] = iter(())
        self.close_gate: asyncio.Event | None = None
        self._created = 0
        self._idempotency: dict[str, tuple[dict[str, object], str]] = {}
        sandbox, command = r'/sandboxes/(?P<id>[^/]+)', r'/commands/(?P<pid>[^/]+)'
        self._routes: list[tuple[str, re.Pattern[str], str, Handler]] = [
            ('POST', re.compile(r'/sandboxes'), 'create', self._create),
            ('GET', re.compile(sandbox), 'get', self._get),
            ('PATCH', re.compile(sandbox), 'update', self._update),
            ('DELETE', re.compile(sandbox), 'delete', self._delete),
            ('POST', re.compile(f'{sandbox}/resume'), 'resume', self._resume),
            ('POST', re.compile(f'{sandbox}/commands'), 'command', self._command),
            ('GET', re.compile(f'{sandbox}{command}'), 'status', self._status),
            ('DELETE', re.compile(f'{sandbox}{command}'), 'kill', self._kill),
            ('GET', re.compile(f'{sandbox}/files'), 'read', self._read),
            ('PUT', re.compile(f'{sandbox}/files'), 'write', self._write),
            ('POST', re.compile(f'{sandbox}/files/(?P<op>upload|download)-url'), 'transfer_url', self._transfer_url),
        ]

    # --- Control surface ---------------------------------------------------------------------------------------------

    def client(self, api_key: str) -> FakeClient:
        """Stands in for `open_client`: an API client served by this fake."""
        self.opened_clients += 1
        client = FakeClient(self, api_key)
        self.clients.append(client)
        return client

    def supplied_client(self) -> FakeClient:
        """A caller-owned API client, as an application passes with `client=`; not one the backend opened."""
        client = FakeClient(self, self.api_key)
        self.clients.append(client)
        return client

    def bodies(self, route: str) -> list[dict[str, object]]:
        """The JSON bodies of the requests that reached `route`, in order."""
        return [call.body for call in self.calls if call.route == route]

    def sandbox(self, state: str = 'idle') -> FakeSandbox:
        """A sandbox that exists before the test, as one an earlier run created."""
        sandbox = self._new_sandbox('default')
        sandbox.state = state
        return sandbox

    def stop(self, sandbox_id: str) -> None:
        """Stop a sandbox as Boat's auto-stop does; its processes keep running, as on Boat's first minutes."""
        self.sandboxes[sandbox_id].state = 'archived'

    async def close(self) -> None:
        for sandbox in self.sandboxes.values():
            sandbox.kill_all()
        for client in self.clients:
            if not client.is_closed:
                await httpx.AsyncClient.aclose(client)

    # --- Dispatch ----------------------------------------------------------------------------------------------------

    async def handle(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        self.requests.append(request)
        url = request.url
        if url.host == MACHINE_HOST:
            route, body, call = 'transfer', {}, partial(self._transfer, request)
        else:
            assert url.host == 'boat.dev' and url.path.startswith('/api/v1/'), f'not a Boat API URL: {url}'
            if request.headers.get('authorization') != f'Bearer {self.api_key}':
                return boat_error(401, 'unauthorized', 'unauthorized')
            route, body, call = self._route(request)
        self.calls.append(Call(route, request, body))
        self.reached[route].set()
        if (gate := self.gates.get(route)) is not None:
            await gate.wait()
        if self.errors[route] and (injected := self.errors[route].pop(0)) is not None:
            if isinstance(injected, Exception):
                raise injected
            return injected
        response = await call()
        if self.lost_replies[route]:
            self.lost_replies[route] -= 1
            raise httpx.ReadTimeout('reply lost', request=request)
        return response

    def _route(self, request: httpx.Request) -> tuple[str, dict[str, object], Callable[[], Awaitable[httpx.Response]]]:
        path = request.url.path.removeprefix('/api/v1')
        for method, pattern, route, handler in self._routes:
            if request.method == method and (match := pattern.fullmatch(path)) is not None:
                body = _BODY.validate_json(request.content) if request.content else {}
                if route == 'command':
                    route = 'start' if body.get('detached') is True else 'run'
                elif route == 'transfer_url':
                    route = f'{match["op"]}_url'
                return route, body, partial(handler, request, match, body)
        raise AssertionError(f'Boat has no {request.method} {path}')  # pragma: no cover - tripwire

    def _sandbox(self, match: re.Match[str]) -> FakeSandbox | httpx.Response:
        sandbox = self.sandboxes.get(match['id'])
        return boat_error(404, 'not_found', 'not_found') if sandbox is None else sandbox

    def _running(self, match: re.Match[str], *, files: bool = False) -> FakeSandbox | httpx.Response:
        """The sandbox an agent call reaches, or Boat's refusal while it is stopped.

        While a stop is still archiving the sandbox, its machine answers file calls but refuses commands.
        """
        sandbox = self._sandbox(match)
        if isinstance(sandbox, FakeSandbox) and sandbox.state == 'archived':
            return boat_error(409, 'machine_not_running', 'Sandbox machine is not running.')
        if isinstance(sandbox, FakeSandbox) and sandbox.state == 'archiving' and not files:
            message = 'Sandbox is not ready (state: archiving). Resume or recover it before running commands.'
            return boat_error(409, 'sandbox_not_ready', message, retryable=False, state='archiving')
        return sandbox

    def _new_sandbox(self, machine_type: str) -> FakeSandbox:
        number, self._created = self._created, self._created + 1
        suffix = ''
        for _ in range(8):
            number, digit = divmod(number, len(_ID_ALPHABET))
            suffix = _ID_ALPHABET[digit] + suffix
        sandbox = FakeSandbox(f'bx_{suffix}', self.root, machine_type=machine_type, boot_polls=self.boot_polls)
        self.sandboxes[sandbox.id] = sandbox
        return sandbox

    # --- Sandboxes ---------------------------------------------------------------------------------------------------

    async def _create(self, request: httpx.Request, match: re.Match[str], body: dict[str, object]) -> httpx.Response:
        key = request.headers.get('idempotency-key')
        assert key is not None, 'every creation is sent with an Idempotency-Key, so a retry finds its sandbox'
        if key in self._idempotency:
            earlier, sandbox_id = self._idempotency[key]
            if earlier != body:  # pragma: no cover - tripwire: a retry must repeat the body it was first sent with
                message = 'This Idempotency-Key was already used with a different request body.'
                return boat_error(409, 'idempotency_key_reused', message)
            sandbox = self.sandboxes.get(sandbox_id)
            info = sandbox.info() if sandbox is not None else {'id': sandbox_id, 'state': 'archived'}
            return boat_ok('sandbox.created', 202, status='ready', sandbox=info)
        machine_type = body.get('type', 'default')
        if machine_type not in MACHINE_TYPES:
            return boat_error(400, 'invalid_type', 'type must be one of small, default, large, xlarge')
        snapshot = body.get('from')
        if snapshot is not None and snapshot not in self.snapshots:
            message = f'No named snapshot "{snapshot}". Run boat snapshots to list yours.'
            return boat_error(404, 'snapshot_not_found', message)
        ttl = body.get('ttlSeconds', 3600)
        if ttl is not None and (type(ttl) is not int or not 0 <= ttl <= 2592000):  # pragma: no cover - tripwire
            return boat_error(400, 'invalid_ttl', 'ttlSeconds must be an integer between 1 and 2592000, or null.')
        assert isinstance(machine_type, str)
        sandbox = self._new_sandbox(machine_type)
        # Boat reads `0` as its default.
        sandbox.ttl = 3600 if ttl == 0 else ttl
        self.create_bodies.append(body)
        self._idempotency[key] = (body, sandbox.id)
        return boat_ok('sandbox.created', 202, status='provisioning', ttlSeconds=sandbox.ttl, sandbox=sandbox.info())

    async def _get(self, request: httpx.Request, match: re.Match[str], body: dict[str, object]) -> httpx.Response:
        sandbox = self._sandbox(match)
        if isinstance(sandbox, httpx.Response):
            return sandbox
        if sandbox.state in ('provisioning', 'provisioned', 'cloning'):
            if sandbox.boot_polls:
                sandbox.boot_polls -= 1
                sandbox.state = 'cloning'
            else:
                sandbox.state = 'idle'
        return boat_ok('sandbox.info', sandbox=sandbox.info())

    async def _update(self, request: httpx.Request, match: re.Match[str], body: dict[str, object]) -> httpx.Response:
        sandbox = self._sandbox(match)
        if isinstance(sandbox, httpx.Response):
            return sandbox
        ttl = body.get('ttlSeconds')
        # The backend only sets the auto-stop again, with the sandbox's own.
        assert body == {'ttlSeconds': sandbox.ttl} and type(ttl) is int and 1 <= ttl <= 2592000
        return boat_ok('sandbox.updated', sandbox=sandbox.info())

    async def _resume(self, request: httpx.Request, match: re.Match[str], body: dict[str, object]) -> httpx.Response:
        sandbox = self._sandbox(match)
        # The backend resumes only a sandbox it just found stopped.
        assert isinstance(sandbox, FakeSandbox) and sandbox.state in ('archived', 'archiving')
        sandbox.resume_calls += 1
        sandbox.state = 'ready'
        return boat_ok('sandbox.resuming', id=sandbox.id, status='running', sandbox=sandbox.info())

    async def _delete(self, request: httpx.Request, match: re.Match[str], body: dict[str, object]) -> httpx.Response:
        sandbox_id = match['id']
        operation = {'id': f'bdop_{sandbox_id}', 'kind': 'sandbox', 'targetId': sandbox_id, 'status': 'pending'}
        if sandbox_id in self.deleted:
            return boat_ok('sandbox.deleting', 202, operation=operation)
        sandbox = self._sandbox(match)
        if isinstance(sandbox, httpx.Response):
            return sandbox
        if request.headers.get('x-ascii-confirm-delete') != sandbox_id:  # pragma: no cover - tripwire
            message = f'Set X-Ascii-Confirm-Delete to the exact target id: {sandbox_id}'
            return boat_error(409, 'delete_confirmation_required', message, targetId=sandbox_id)
        # Deleting a sandbox stops its machine at once, and every process on it.
        sandbox.kill_all()
        del self.sandboxes[sandbox_id]
        self.deleted.add(sandbox_id)
        return boat_ok('sandbox.deleting', 202, operation=operation)

    # --- Commands ----------------------------------------------------------------------------------------------------

    async def _command(self, request: httpx.Request, match: re.Match[str], body: dict[str, object]) -> httpx.Response:
        sandbox = self._running(match)
        if isinstance(sandbox, httpx.Response):
            return sandbox
        command = body.get('command')
        assert isinstance(command, str) and command
        if len(command.encode()) > _COMMAND_LIMIT:
            return _direct_failed("E2BIG: argument list too long, posix_spawn 'systemd-run'")
        if body.get('detached') is True:
            return self._start(sandbox, command)
        timeout = body.get('timeoutSeconds', 30)
        assert type(timeout) is int and 1 <= timeout <= 600, 'Boat refuses a timeout outside 1-600 seconds'
        return await anyio.to_thread.run_sync(_run_sync, sandbox, command, timeout)

    def _start(self, sandbox: FakeSandbox, command: str) -> httpx.Response:
        out, err = sandbox.logs / f'{len(sandbox.commands)}.log', sandbox.logs / f'{len(sandbox.commands)}.err.log'
        with out.open('wb') as stdout, err.open('wb') as stderr:
            process = subprocess.Popen(
                ['bash', '-c', command],
                cwd=sandbox.home,
                env=sandbox.environment(),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
        sandbox.commands[process.pid] = FakeCommand(process, out, err)
        return boat_ok(
            'command.started', success=True, processId=process.pid, pid=process.pid, command=command, cwd='.'
        )

    async def _status(self, request: httpx.Request, match: re.Match[str], body: dict[str, object]) -> httpx.Response:
        sandbox = self._running(match)
        if isinstance(sandbox, httpx.Response):
            return sandbox
        assert match['pid'].isdigit() and int(match['pid']) >= 1, 'Boat refuses a process ID that is not positive'
        command = sandbox.commands.get(int(match['pid']))
        if command is None:
            return boat_ok(
                'command.status', status='lost', known=False, running=False, exitCode=None, stdout='', stderr=''
            )
        code = command.process.poll()
        status: dict[str, object] = {
            'processId': command.process.pid,
            'status': 'running' if code is None else 'exited',
            'running': code is None,
            'exitCode': None if code is None else code if code >= 0 else 128 - code,
        }
        out, err = command.out.read_bytes(), command.err.read_bytes()
        query = request.url.params
        if 'stdoutOffset' in query or 'stderrOffset' in query:
            start_out, start_err = int(query.get('stdoutOffset', len(out))), int(query.get('stderrOffset', len(err)))
            page_out, page_err = out[start_out : start_out + _OFFSET_PAGE], err[start_err : start_err + _OFFSET_PAGE]
            status |= {
                'stdout': page_out.decode(errors='replace'),
                'stderr': page_err.decode(errors='replace'),
                'stdoutOffset': start_out,
                'stderrOffset': start_err,
                'stdoutNextOffset': start_out + len(page_out),
                'stderrNextOffset': start_err + len(page_err),
                'more': start_out + len(page_out) < len(out) or start_err + len(page_err) < len(err),
            }
        else:
            # Boat ignores a `tailBytes` that is not a positive integer.
            raw = query.get('tailBytes', '')
            limit = int(raw) if raw.isdigit() and int(raw) > 0 else _STREAM_LIMIT
            (stdout, stdout_cut), (stderr, stderr_cut) = _tail(out, limit), _tail(err, limit)
            status |= {
                'known': True,
                'stdout': stdout,
                'stderr': stderr,
                'stdoutTruncated': stdout_cut,
                'stderrTruncated': stderr_cut,
            }
        return boat_ok('command.status', success=True, **status)

    async def _kill(self, request: httpx.Request, match: re.Match[str], body: dict[str, object]) -> httpx.Response:
        sandbox = self._running(match)
        if isinstance(sandbox, httpx.Response):
            return sandbox
        command = sandbox.commands.get(int(match['pid']))
        if command is None:
            return _direct_failed(f'Process {match["pid"]} was not started by a detached command.')
        signaled = command.process.poll() is None
        command.kill()
        command.stopped = True
        return boat_ok('command.stopped', processId=command.process.pid, running=False, signaled=signaled)

    # --- Files -------------------------------------------------------------------------------------------------------

    def _resolve(self, sandbox: FakeSandbox, raw: object) -> Path | httpx.Response:
        """The host path of a file-API path, or Boat's refusal of a path outside its roots."""
        assert isinstance(raw, str)
        path = Path(os.path.normpath(sandbox.home / raw))
        if not _under(path, sandbox.home, sandbox.tmp):
            return boat_error(400, 'invalid_path', _ROOTS_MESSAGE)
        # A symlink is followed, and its target must resolve under the roots as well.
        if not _under(Path(os.path.realpath(path)), sandbox.home, sandbox.tmp):
            return _direct_failed(_ROOTS_MESSAGE)
        return path

    async def _read(self, request: httpx.Request, match: re.Match[str], body: dict[str, object]) -> httpx.Response:
        sandbox = self._running(match, files=True)
        if isinstance(sandbox, httpx.Response):
            return sandbox
        path = self._resolve(sandbox, request.url.params.get('path'))
        if isinstance(path, httpx.Response):
            return path
        try:
            info = path.stat()
        except OSError as error:
            return _direct_failed(_node_error(error, 'stat', path))
        if not stat.S_ISREG(info.st_mode):
            # A FIFO too, without opening it.
            return _direct_failed('Path is not a file.')
        if info.st_size > _FILE_LIMIT:
            return _direct_failed(
                f'File is too large for read_file ({info.st_size} bytes > {_FILE_LIMIT}). '
                f'Use a download URL instead: POST /sandboxes/{sandbox.id}/files/download-url.'
            )
        # Boat's file API reads as root, so a file it found is a file it can read.
        data = path.read_bytes()
        encoding = request.url.params.get('encoding')
        content = base64.b64encode(data).decode() if encoding == 'base64' else data.decode(errors='replace')
        return boat_ok(
            'file.read', success=True, path=str(path), encoding=encoding or 'utf8', size=len(data), content=content
        )

    async def _write(self, request: httpx.Request, match: re.Match[str], body: dict[str, object]) -> httpx.Response:
        sandbox = self._running(match, files=True)
        if isinstance(sandbox, httpx.Response):
            return sandbox
        path = self._resolve(sandbox, body.get('path'))
        if isinstance(path, httpx.Response):
            return path
        content = body.get('content')
        text = content if isinstance(content, str) else ''
        data = base64.b64decode(text) if body.get('encoding') == 'base64' else text.encode()
        if len(data) > _FILE_LIMIT:  # pragma: no cover - tripwire: a larger file goes through an upload URL
            return _direct_failed(
                f'File is too large for write_file ({len(data)} bytes > {_FILE_LIMIT}). '
                f'Use an upload URL instead: POST /sandboxes/{sandbox.id}/files/upload-url.'
            )
        if path.is_symlink() and not path.exists():
            return _direct_failed(_ROOTS_MESSAGE)
        if path.exists() and not path.is_file() and not path.is_dir():
            # Boat's agent answers a write to a FIFO only once something opens it for reading, so the
            # client's request times out first; opening it here would block the event loop.
            raise httpx.ReadTimeout('timed out', request=request)
        if (error := _write_file(path, data)) is not None:
            return _direct_failed(error)
        return boat_ok('file.written', success=True, path=str(path), size=len(data))

    async def _transfer_url(
        self, request: httpx.Request, match: re.Match[str], body: dict[str, object]
    ) -> httpx.Response:
        sandbox = self._running(match)
        if isinstance(sandbox, httpx.Response):
            return sandbox
        raw = body.get('path')
        assert isinstance(raw, str)
        path = Path(os.path.normpath(sandbox.home / raw))
        # Boat signs the URL without looking at the file.
        if not _under(path, sandbox.home, sandbox.tmp):
            return boat_error(400, 'invalid_path', _ROOTS_MESSAGE)
        op = 'write' if match['op'] == 'upload' else 'read'
        query = urlencode({'op': op, 'sandbox': sandbox.id, 'path': str(path), 'sig': 'fake-signature'})
        method = 'PUT' if op == 'write' else 'GET'
        return boat_ok(f'file.{match["op"]}_url', url=f'https://{MACHINE_HOST}/file-transfer?{query}', method=method)

    async def _transfer(self, request: httpx.Request) -> httpx.Response:
        # The URL is signed: Boat's API key must not travel to the sandbox's machine.
        assert 'authorization' not in request.headers, 'the Boat API key was sent to a file-transfer URL'
        query = parse_qs(urlsplit(str(request.url)).query)
        sandbox = self.sandboxes.get(query['sandbox'][0])
        if sandbox is None or sandbox.state not in ('ready', 'idle', 'running'):
            raise httpx.ConnectError('the sandbox machine is not running', request=request)
        path = Path(query['path'][0])
        if query['op'][0] == 'read':
            if not path.exists():
                return httpx.Response(404, json={'error': 'not_found', 'message': 'File not found.'})
            # A download follows the file API refusing the file as too large, so it is one.
            return httpx.Response(200, content=path.read_bytes(), headers={'content-type': 'application/octet-stream'})
        if path.exists() and not path.is_file():
            return httpx.Response(
                400, json={'error': 'not_a_file', 'message': 'Path exists and is not a regular file.'}
            )
        if (error := _write_file(path, request.content)) is not None:
            return httpx.Response(400, json={'error': 'write_failed', 'message': error})
        return httpx.Response(200, json={'success': True, 'path': str(path), 'size': len(request.content)})


def _under(path: Path, *roots: Path) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _node_error(error: OSError, syscall: str, path: Path) -> str:
    """The Node.js wording Boat's agent reports a failed file call in."""
    code = _NODE_ERRORS.get(error.errno or 0, f'EIO: i/o error ({error.strerror})')
    return f"{code}, {syscall} '{path}'"


def _write_file(path: Path, data: bytes) -> str | None:
    """Write like Boat's agent: missing parents are created, in place, through a symlink; or its error message."""
    for parent in reversed(path.parents):
        if parent.exists() and not parent.is_dir():
            return f"EEXIST: file already exists, mkdir '{parent}'"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as file:
            file.write(data)
    except OSError as error:
        return _node_error(error, 'open', path)
    return None


def _run_sync(sandbox: FakeSandbox, command: str, timeout: int) -> httpx.Response:
    """A synchronous command: Boat answers once its output closes or its timeout stops it."""
    process = subprocess.Popen(
        ['bash', '-c', command],
        cwd=sandbox.home,
        env=sandbox.environment(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    timed_out = False
    try:
        out, err = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:  # pragma: no cover - only an internal command that hangs reaches it
        timed_out = True
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
        out, err = process.communicate()
    code = process.returncode
    (stdout, stdout_cut), (stderr, stderr_cut) = _tail(out, _STREAM_LIMIT), _tail(err, _STREAM_LIMIT)
    return boat_ok(
        'command.finished',
        success=code == 0,
        exitCode=code if code >= 0 else 128 - code,
        stdout=stdout,
        stderr=stderr,
        stdoutTruncated=stdout_cut,
        stderrTruncated=stderr_cut,
        timedOut=timed_out,
    )
