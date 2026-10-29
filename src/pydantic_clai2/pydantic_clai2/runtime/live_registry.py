"""Private process discovery and typed, same-UID Unix control frames.

The discovery file contains only process identity and socket path. A `hello` carries fresh sessions;
requests are bounded at 64 KiB and replies at 1 MiB, with explicit errors for oversized session data.
"""

import asyncio
import ctypes
import os
import secrets
import socket
import stat
import struct
import sys
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from anyio import CancelScope
from pydantic import Field, TypeAdapter, ValidationError

from pydantic_clai2.config.private_files import write_private
from pydantic_clai2.config.settings_store import state_dir
from pydantic_clai2.plugins import LiveSession

ProtocolVersion = Literal[1]
_PROTOCOL: ProtocolVersion = 1
_MAX_REQUEST = 65_536
_MAX_REPLY = 1_048_576
_TIMEOUT = 0.5
LiveStatus = Literal['idle', 'running', 'waiting']


@dataclass(frozen=True, kw_only=True)
class LiveEntry:
    """One session in a process's live listing."""

    id: str
    name: str
    conversation_id: str
    title: str
    workspace: str
    model: str
    status: LiveStatus
    waiting_for: str | None = None
    status_since: datetime | None = None


@dataclass(frozen=True, kw_only=True)
class LiveRecord:
    """Static private discovery record."""

    pid: int
    instance: str
    socket: str


@dataclass(frozen=True, kw_only=True)
class LiveProcess(LiveRecord):
    """A verified process and its current sessions."""

    sessions: list[LiveEntry]


@dataclass(frozen=True, kw_only=True)
class HelloRequest:
    """Request the current process identity and sessions."""

    op: Literal['hello'] = 'hello'
    protocol: ProtocolVersion = _PROTOCOL


@dataclass(frozen=True, kw_only=True)
class Hello:
    """Current process identity and live sessions."""

    op: Literal['hello_reply']
    instance: str
    protocol: ProtocolVersion = _PROTOCOL
    sessions: list[LiveEntry]


@dataclass(frozen=True, kw_only=True)
class ControlError:
    """A typed control-protocol failure."""

    op: Literal['error']
    error: Literal['unsupported', 'response_too_large']
    message: str


Frame = Annotated[HelloRequest | Hello | ControlError, Field(discriminator='op')]
_FRAME: TypeAdapter[Frame] = TypeAdapter(Frame)
_RECORD = TypeAdapter(LiveRecord)


class ControlProtocolError(RuntimeError):
    """The peer returned a typed control-protocol error."""


class ResponseTooLargeError(ControlProtocolError):
    """The process has more session data than its reply limit permits."""


def _private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return _check_private_directory(path)


def _check_private_directory(path: Path) -> Path:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise PermissionError(f'Unsafe CLAI directory {path}')
    return path


def _socket_directory() -> Path:
    base = os.environ.get('XDG_RUNTIME_DIR') or os.environ.get('TMPDIR')
    return _private_directory((Path(base).absolute() if base else Path('/tmp')) / f'pydantic-clai2-{os.getuid()}')


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


def peer_uid(connection: socket.socket) -> int | None:
    """The connecting process's UID, or `None` when the platform cannot verify it."""
    if sys.platform == 'darwin':
        uid, gid = ctypes.c_uint(), ctypes.c_uint()
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.getpeereid(connection.fileno(), ctypes.byref(uid), ctypes.byref(gid)) != 0:
            return None
        return uid.value
    peercred = getattr(socket, 'SO_PEERCRED', None)
    if peercred is None:
        return None
    _, uid, _ = struct.unpack('3i', connection.getsockopt(socket.SOL_SOCKET, peercred, 12))
    return uid


async def request(path: Path, frame: HelloRequest) -> Hello:
    """Exchange a typed, bounded control frame."""
    payload = _FRAME.dump_json(frame) + b'\n'
    if len(payload) > _MAX_REQUEST:
        raise ValueError('Control request exceeds 64 KiB')
    reader, writer = await asyncio.wait_for(asyncio.open_unix_connection(str(path), limit=_MAX_REPLY + 1), _TIMEOUT)
    try:
        writer.write(payload)
        await asyncio.wait_for(writer.drain(), _TIMEOUT)
        line = await asyncio.wait_for(reader.readline(), _TIMEOUT)
        if not line or len(line) > _MAX_REPLY or not line.endswith(b'\n'):
            raise ValueError('Invalid control frame')
        try:
            reply = _FRAME.validate_json(line)
        except ValidationError as exc:
            raise ControlProtocolError('Invalid control frame') from exc
        if isinstance(reply, ControlError):
            if reply.error == 'response_too_large':
                raise ResponseTooLargeError(reply.message)
            raise ControlProtocolError(reply.message)
        if isinstance(reply, HelloRequest):
            raise ControlProtocolError('Unexpected hello request reply')
        return reply
    finally:
        writer.close()
        with suppress(OSError):
            await writer.wait_closed()


def _record_socket(record: LiveRecord) -> Path:
    path = Path(record.socket)
    if (
        not path.is_absolute()
        or path.name != f'{record.pid}.sock'
        or path.parent.name != f'pydantic-clai2-{os.getuid()}'
    ):
        raise ValueError('Invalid CLAI control socket path')
    _check_private_directory(path.parent)
    return path


async def listing(*, directory: Path | None = None) -> list[LiveProcess]:
    """Fetch current sessions from each verified process."""
    directory = directory or state_dir() / 'live'
    if os.name != 'posix' or not directory.exists():
        return []
    _private_directory(directory)
    found: list[LiveProcess] = []
    for path in directory.glob('*.json'):
        try:
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                continue
            record = _RECORD.validate_json(path.read_bytes())
            if path.name != f'{record.pid}.json' or record.pid <= 0:
                continue
            sock = _record_socket(record)
            if not _owned_socket(sock):
                if not sock.exists():
                    _unlink_if_unchanged(path, info.st_ino)
                continue
            inode = sock.lstat().st_ino
            try:
                os.kill(record.pid, 0)
            except ProcessLookupError:
                try:
                    await request(sock, HelloRequest())
                except (ConnectionRefusedError, FileNotFoundError):
                    _unlink_if_unchanged(sock, inode)
                    _unlink_if_unchanged(path, info.st_ino)
                continue
            except PermissionError:
                continue
            try:
                hello = await request(sock, HelloRequest())
            except ConnectionRefusedError:
                _unlink_if_unchanged(path, info.st_ino)
                continue
            except ResponseTooLargeError:
                raise
            except (OSError, ValueError, TimeoutError, ControlProtocolError, asyncio.LimitOverrunError):
                continue
            if hello.instance == record.instance:
                found.append(
                    LiveProcess(pid=record.pid, instance=record.instance, socket=record.socket, sessions=hello.sessions)
                )
            else:
                _unlink_if_unchanged(path, info.st_ino)
        except (OSError, ValidationError, ValueError):
            continue
    return found


class LiveRegistry:
    """Serve one process's live sessions and maintain its static discovery record."""

    def __init__(self, sessions: Callable[[], list[LiveSession]], *, directory: Path | None = None) -> None:
        self.sessions = sessions
        self.directory = directory or state_dir() / 'live'
        self.pid = os.getpid()
        self.instance = secrets.token_hex(16)
        self.socket_path: Path | None = None
        self._server: asyncio.AbstractServer | None = None
        self._unsubscribe: Callable[[], None] | None = None
        self._statuses: dict[str, tuple[LiveStatus, datetime]] = {}

    def _track(self) -> list[LiveSession]:
        now = datetime.now(UTC)
        sessions = self.sessions()
        ids = {session.id for session in sessions}
        self._statuses = {key: value for key, value in self._statuses.items() if key in ids}
        for session in sessions:
            status: LiveStatus = (
                'waiting' if session.waiting_for is not None else 'running' if session.running else 'idle'
            )
            previous = self._statuses.get(session.id)
            if previous is None or previous[0] != status:
                self._statuses[session.id] = (status, now)
        return sessions

    def snapshot(self) -> Hello:
        """Current session metadata, with status-transition times retained between requests."""
        entries: list[LiveEntry] = []
        for session in self._track():
            status, since = self._statuses[session.id]
            entries.append(
                LiveEntry(
                    id=session.id,
                    name=session.workspace.name or 'session',
                    conversation_id=session.conversation.conversation_id,
                    title=session.conversation.title or '',
                    workspace=str(session.workspace),
                    model=session.status.model,
                    status=status,
                    status_since=since,
                    waiting_for=session.waiting_for,
                )
            )
        return Hello(op='hello_reply', instance=self.instance, sessions=entries)

    def changed(self) -> None:
        """Track session transitions in memory; discovery metadata is never rewritten."""
        self._track()

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            connection = writer.get_extra_info('socket')
            if connection is None or peer_uid(connection) != os.getuid():
                return
            line = await asyncio.wait_for(reader.readline(), _TIMEOUT)
            if not line or len(line) > _MAX_REQUEST or not line.endswith(b'\n'):
                return
            try:
                frame = _FRAME.validate_json(line)
            except ValidationError:
                frame = None
            if not isinstance(frame, HelloRequest):
                reply: Frame = ControlError(op='error', error='unsupported', message='Unsupported control request')
            else:
                reply = self.snapshot()
            payload = _FRAME.dump_json(reply) + b'\n'
            if len(payload) > _MAX_REPLY:
                payload = (
                    _FRAME.dump_json(
                        ControlError(
                            op='error', error='response_too_large', message='Live session response exceeds 1 MiB'
                        )
                    )
                    + b'\n'
                )
            writer.write(payload)
            await writer.drain()
        except (OSError, ValueError, TimeoutError, asyncio.LimitOverrunError):
            pass
        finally:
            writer.close()
            with suppress(OSError):
                await writer.wait_closed()

    @asynccontextmanager
    async def serving(self, subscribe: Callable[[Callable[[], None]], Callable[[], None]]) -> AsyncGenerator[None]:
        """Serve current sessions and track transitions until the context ends; publish discovery metadata once."""
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
                await request(path, HelloRequest())
            except (ConnectionRefusedError, FileNotFoundError):
                _unlink_if_unchanged(path, old_inode)
                if path.exists():
                    raise OSError(f'CLAI socket changed during cleanup at {path}')
            else:
                raise OSError(f'CLAI socket already active at {path}')
        self.socket_path = path
        listener = socket.socket(socket.AF_UNIX)
        inode: int | None = None
        try:
            previous_umask = os.umask(0o077)
            try:
                listener.bind(str(path))
                inode = path.lstat().st_ino
            finally:
                os.umask(previous_umask)
            path.chmod(0o600)
            listener.listen()
            listener.setblocking(False)
            self._server = await asyncio.start_unix_server(self._serve, sock=listener, limit=_MAX_REQUEST + 1)
        except BaseException:
            listener.close()
            if inode is not None:
                _unlink_if_unchanged(path, inode)
            raise
        socket_inode = path.lstat().st_ino
        record_path = self.directory / f'{self.pid}.json'
        try:
            write_private(
                record_path,
                _RECORD.dump_json(LiveRecord(pid=self.pid, instance=self.instance, socket=str(path))).decode(),
            )
            self.changed()
            self._unsubscribe = subscribe(self.changed)
            yield
        finally:
            if self._unsubscribe is not None:
                self._unsubscribe()
                self._unsubscribe = None
            with CancelScope(shield=True):
                self._server.close()
                await self._server.wait_closed()
                _unlink_if_unchanged(path, socket_inode)
                try:
                    record_info = record_path.lstat()
                    record = _RECORD.validate_json(record_path.read_bytes())
                    if record.instance == self.instance:
                        _unlink_if_unchanged(record_path, record_info.st_ino)
                except (FileNotFoundError, ValidationError):
                    pass
