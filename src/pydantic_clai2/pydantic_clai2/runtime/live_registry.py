"""Private per-process discovery of live CLAI sessions over a local control socket."""

import asyncio
import ctypes
import json
import logging
import os
import secrets
import socket
import stat
import struct
import sys
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import TypeAdapter, ValidationError

from pydantic_clai2.config.private_files import write_private
from pydantic_clai2.config.settings_store import state_dir
from pydantic_clai2.plugins import LiveSession

_PROTOCOL = 1
_MAX_FRAME = 65_536
_TIMEOUT = 0.5
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class LiveEntry:
    """One session in a process's live listing."""

    id: str
    handle: str
    conversation_id: str
    title: str
    workspace: str
    model: str
    status: Literal['idle', 'running', 'waiting']
    branch: str = ''
    waiting_for: str = ''
    status_since: datetime | None = None
    attached: int = 0
    focused: bool = False
    accepts_messages: bool = True
    inbox: int = 0


@dataclass(frozen=True, kw_only=True)
class LiveRecord:
    """A private on-disk snapshot, verified against its owner's control socket."""

    schema: int
    pid: int
    instance: str
    protocol: int
    features: list[str]
    socket: str
    started_at: datetime
    updated_at: datetime
    sessions: list[LiveEntry] = field(default_factory=lambda: list[LiveEntry]())


_RECORD = TypeAdapter(LiveRecord)
_ENTRIES = TypeAdapter(list[LiveEntry])
_FRAME = TypeAdapter(dict[str, object])


def _private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    mode = path.lstat()
    if not stat.S_ISDIR(mode.st_mode) or mode.st_uid != os.getuid() or mode.st_mode & 0o077:
        raise PermissionError(f'Unsafe CLAI directory {path}')
    return path


def _socket_directory() -> Path:
    base = os.environ.get('XDG_RUNTIME_DIR') or os.environ.get('TMPDIR')
    if base:
        return _private_directory(Path(base) / 'pydantic-clai2')
    return _private_directory(Path('/tmp') / f'pydantic-clai2-{os.getuid()}')


def _owned_socket(path: Path) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISSOCK(info.st_mode) and info.st_uid == os.getuid() and not info.st_mode & 0o077


def _unlink_if_unchanged(path: Path, inode: int) -> None:
    try:
        if path.lstat().st_ino == inode:
            path.unlink()
    except FileNotFoundError:
        pass


def _peer_uid(connection: socket.socket) -> int | None:
    if sys.platform == 'darwin':
        uid = ctypes.c_uint()
        gid = ctypes.c_uint()
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.getpeereid(connection.fileno(), ctypes.byref(uid), ctypes.byref(gid)) != 0:
            return None
        return uid.value
    if hasattr(socket, 'SO_PEERCRED'):
        _, uid, _ = struct.unpack('3i', connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        return uid
    return None


async def request(path: Path, frame: dict[str, object]) -> dict[str, object]:
    """Exchange one bounded control frame with a process socket."""
    payload = json.dumps(frame).encode() + b'\n'
    if len(payload) > _MAX_FRAME:
        raise ValueError('Control frame exceeds 64 KiB')
    reader, writer = await asyncio.wait_for(asyncio.open_unix_connection(str(path)), _TIMEOUT)
    try:
        writer.write(payload)
        await asyncio.wait_for(writer.drain(), _TIMEOUT)
        line = await asyncio.wait_for(reader.readline(), _TIMEOUT)
        if not line or len(line) > _MAX_FRAME or not line.endswith(b'\n'):
            raise ValueError('Invalid control frame')
        return _FRAME.validate_json(line)
    finally:
        writer.close()
        await writer.wait_closed()


async def listing(*, directory: Path | None = None) -> list[LiveRecord]:
    """Find reachable processes; remove only records whose exact socket and inode are safely stale."""
    directory = directory or state_dir() / 'live'
    if os.name != 'posix' or not directory.exists():
        return []
    _private_directory(directory)
    found: list[LiveRecord] = []
    for path in directory.glob('*.json'):
        try:
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                continue
            record = _RECORD.validate_json(path.read_bytes())
            if (
                record.schema != 1
                or record.protocol != _PROTOCOL
                or path.name != f'{record.pid}.json'
                or record.pid <= 0
            ):
                continue
            socket_path = Path(record.socket)
            if socket_path != _socket_directory() / f'{record.pid}.sock' or not _owned_socket(socket_path):
                # An absent socket is stale, but never follow an arbitrary path supplied by a record.
                if not socket_path.exists() and socket_path == _socket_directory() / f'{record.pid}.sock':
                    _unlink_if_unchanged(path, info.st_ino)
                continue
            socket_inode = socket_path.lstat().st_ino
            try:
                os.kill(record.pid, 0)
            except ProcessLookupError:
                # Only a refused connection proves the socket is abandoned. Never unlink a new
                # listener after PID reuse, even if its old record has not been refreshed yet.
                try:
                    await request(socket_path, {'op': 'hello', 'protocol': _PROTOCOL})
                except (ConnectionRefusedError, FileNotFoundError):
                    _unlink_if_unchanged(socket_path, socket_inode)
                    _unlink_if_unchanged(path, info.st_ino)
                continue
            except PermissionError:
                continue
            try:
                hello = await request(socket_path, {'op': 'hello', 'protocol': _PROTOCOL})
            except ConnectionRefusedError:
                _unlink_if_unchanged(path, info.st_ino)
                continue
            except (OSError, ValueError, TimeoutError, asyncio.LimitOverrunError):
                # Timeout is inconclusive; another process may be starting or unresponsive.
                continue
            if hello.get('instance') == record.instance and hello.get('protocol') == _PROTOCOL:
                found.append(record)
            else:
                _unlink_if_unchanged(path, info.st_ino)
        except (OSError, ValidationError, ValueError):
            continue
    return found


class LiveRegistry:
    """One process's socket and atomically replaced listing, scoped to `serving()`."""

    def __init__(self, sessions: Callable[[], list[LiveSession]], *, directory: Path | None = None) -> None:
        self.sessions = sessions
        self.directory = directory or state_dir() / 'live'
        self.pid = os.getpid()
        self.instance = secrets.token_hex(16)
        self.started_at = datetime.now(UTC)
        self.socket_path: Path | None = None
        self._server: asyncio.AbstractServer | None = None
        self._timer: asyncio.TimerHandle | None = None
        self._unsubscribe: Callable[[], None] | None = None
        self._handles: dict[str, str] = {}
        self._statuses: dict[str, tuple[str, datetime]] = {}
        self._waiting: dict[str, set[str]] = {}

    def question(self, session_id: str, request_id: str, *, answered: bool = False) -> None:
        """Keep a live question visible until it is answered or its turn ends."""
        waiting = self._waiting.setdefault(session_id, set())
        if answered:
            waiting.discard(request_id)
        else:
            waiting.add(request_id)
        self.changed()

    def snapshot(self) -> LiveRecord:
        now = datetime.now(UTC)
        sessions = self.sessions()
        live_ids = {session.id for session in sessions}
        self._handles = {key: value for key, value in self._handles.items() if key in live_ids}
        self._statuses = {key: value for key, value in self._statuses.items() if key in live_ids}
        self._waiting = {key: value for key, value in self._waiting.items() if key in live_ids}
        entries: list[LiveEntry] = []
        for session in sessions:
            if session.id not in self._handles:
                name = session.workspace.name or 'session'
                handle = name
                suffix = 2
                while handle in self._handles.values():
                    handle = f'{name}-{suffix}'
                    suffix += 1
                self._handles[session.id] = handle
            if not session.running:
                self._waiting.pop(session.id, None)
            status: Literal['idle', 'running', 'waiting'] = (
                'waiting' if self._waiting.get(session.id) else 'running' if session.running else 'idle'
            )
            previous = self._statuses.get(session.id)
            if previous is None or previous[0] != status:
                self._statuses[session.id] = (status, now)
            entries.append(
                LiveEntry(
                    id=session.id,
                    handle=self._handles[session.id],
                    conversation_id=session.conversation.conversation_id,
                    title=session.conversation.title or '',
                    workspace=str(session.workspace),
                    model=session.status.model,
                    status=status,
                    status_since=self._statuses[session.id][1],
                    waiting_for='an ask_user question' if status == 'waiting' else '',
                )
            )
        return LiveRecord(
            schema=1,
            pid=self.pid,
            instance=self.instance,
            protocol=_PROTOCOL,
            features=['list'],
            socket=str(self.socket_path),
            started_at=self.started_at,
            updated_at=now,
            sessions=entries,
        )

    def changed(self) -> None:
        if self._timer is None:
            self._timer = asyncio.get_running_loop().call_later(0.25, self.flush)

    def flush(self) -> None:
        """Persist the current session snapshot immediately, or remove it if no sessions are open."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        try:
            path = self.directory / f'{self.pid}.json'
            if not self.sessions():
                self._forget_record()
                return
            write_private(path, _RECORD.dump_json(self.snapshot()).decode())
        except OSError:
            _LOGGER.exception('Could not update CLAI live registry')

    def _forget_record(self) -> None:
        path = self.directory / f'{self.pid}.json'
        try:
            info = path.lstat()
            record = _RECORD.validate_json(path.read_bytes())
            if record.instance == self.instance:
                _unlink_if_unchanged(path, info.st_ino)
        except (FileNotFoundError, ValidationError):
            pass

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            connection = writer.get_extra_info('socket')
            if connection is None or _peer_uid(connection) != os.getuid():
                return
            line = await asyncio.wait_for(reader.readline(), _TIMEOUT)
            if not line or len(line) > _MAX_FRAME or not line.endswith(b'\n'):
                return
            request = _FRAME.validate_json(line)
            if request.get('op') == 'hello':
                result: dict[str, object] = {'instance': self.instance, 'protocol': _PROTOCOL, 'features': ['list']}
            elif request.get('op') == 'list':
                result = {'sessions': _ENTRIES.dump_python(self.snapshot().sessions, mode='json')}
            else:
                result = {'error': 'unsupported'}
            payload = json.dumps(result).encode() + b'\n'
            if len(payload) <= _MAX_FRAME:
                writer.write(payload)
                await writer.drain()
        except (OSError, ValueError, TimeoutError, asyncio.LimitOverrunError):
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    @asynccontextmanager
    async def serving(self, subscribe: Callable[[Callable[[], None]], Callable[[], None]]):
        if os.name != 'posix':
            raise OSError('Live session discovery requires Unix sockets')
        _private_directory(self.directory)
        path = _socket_directory() / f'{self.pid}.sock'
        if len(os.fsencode(path)) >= 104:
            raise OSError('CLAI control socket path is too long')
        if path.exists() or path.is_symlink():
            if not _owned_socket(path):
                raise PermissionError(f'Unsafe CLAI socket {path}')
            old_inode = path.lstat().st_ino
            try:
                await request(path, {'op': 'hello', 'protocol': _PROTOCOL})
            except (ConnectionRefusedError, FileNotFoundError):
                _unlink_if_unchanged(path, old_inode)
                if path.exists():
                    raise OSError(f'CLAI socket changed during cleanup at {path}')
            else:
                raise OSError(f'CLAI socket already active at {path}')
        self.socket_path = path
        listener = socket.socket(socket.AF_UNIX)
        bound_inode: int | None = None
        try:
            previous_umask = os.umask(0o077)
            try:
                listener.bind(str(path))
                bound_inode = path.lstat().st_ino
            finally:
                os.umask(previous_umask)
            path.chmod(0o600)
            listener.listen()
            listener.setblocking(False)
            self._server = await asyncio.start_unix_server(self._serve, sock=listener, limit=_MAX_FRAME + 1)
        except BaseException:
            listener.close()
            if bound_inode is not None:
                _unlink_if_unchanged(path, bound_inode)
            raise
        socket_inode = path.lstat().st_ino
        self._unsubscribe = subscribe(self.changed)
        self.flush()
        try:
            yield
        finally:
            self._unsubscribe()
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._server.close()
            await self._server.wait_closed()
            _unlink_if_unchanged(path, socket_inode)
            self._forget_record()
