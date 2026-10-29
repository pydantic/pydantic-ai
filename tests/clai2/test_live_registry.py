"""Socket discovery uses real in-process Unix sockets and private files."""

import asyncio
import ctypes
import os
import socket
import struct
import sys
from collections.abc import Callable
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from pydantic_clai2.cli import cli
from pydantic_clai2.config.private_files import write_private
from pydantic_clai2.runtime import live_registry
from pydantic_clai2.runtime.live_registry import (
    ControlError,
    ControlProtocolError,
    Hello,
    HelloRequest,
    LiveEntry,
    LiveRecord,
    LiveRegistry,
    ResponseTooLargeError,
    listing,
    peer_uid,
    request,
)

_RECORD = TypeAdapter(LiveRecord)


def subscribe(callback: Callable[[], None]) -> Callable[[], None]:
    return lambda: None


@pytest.fixture
def private_runtime(runtime_directory: Path) -> Path:
    directory = runtime_directory / 'state' / 'live'
    directory.mkdir(parents=True, mode=0o700)
    return directory


def save_record(directory: Path, record: LiveRecord) -> Path:
    path = directory / f'{record.pid}.json'
    write_private(path, _RECORD.dump_json(record).decode())
    return path


async def test_control_lifecycle_and_bounded_frames(private_runtime: Path) -> None:
    registry = LiveRegistry(lambda: [], directory=private_runtime)
    async with registry.serving(subscribe):
        path = registry.socket_path
        assert path is not None
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.parent.stat().st_mode & 0o777 == 0o700
        record = private_runtime / f'{registry.pid}.json'
        assert record.stat().st_mode & 0o777 == 0o600
        assert set(TypeAdapter(dict[str, object]).validate_json(record.read_bytes())) == {'pid', 'instance', 'socket'}
        reply = await request(path, HelloRequest())
        assert reply.instance == registry.instance and reply.sessions == []
        assert reply.protocol == 1
        reader, writer = await asyncio.open_unix_connection(str(path))
        writer.write(b'x' * 70_000 + b'\n')
        await writer.drain()
        with suppress(ConnectionResetError):
            assert await asyncio.wait_for(reader.read(), 10) == b''
        writer.close()
        with suppress(ConnectionResetError):
            await writer.wait_closed()
    assert not path.exists() and not record.exists()


@pytest.mark.parametrize(
    'frame', [b'{\n', b'[]\n', b'{"op":"list"}\n', b'{"op":"unknown"}\n', b'{"op":"hello","protocol":2}\n']
)
async def test_invalid_or_unknown_frame_returns_typed_error(private_runtime: Path, frame: bytes) -> None:
    registry = LiveRegistry(lambda: [], directory=private_runtime)
    async with registry.serving(subscribe):
        assert registry.socket_path is not None
        reader, writer = await asyncio.open_unix_connection(str(registry.socket_path))
        writer.write(frame)
        await writer.drain()
        reply = TypeAdapter(ControlError).validate_json(await reader.readline())
        assert reply.error == 'unsupported'
        writer.close()
        await writer.wait_closed()


@pytest.mark.parametrize('cross_environment', [False, True])
async def test_listing_verifies_nonce_across_environments(
    private_runtime: Path, monkeypatch: pytest.MonkeyPatch, cross_environment: bool
) -> None:
    registry = LiveRegistry(lambda: [], directory=private_runtime)
    async with registry.serving(subscribe):
        path = registry.socket_path
        assert path is not None
        record_path = private_runtime / f'{registry.pid}.json'
        record = _RECORD.validate_json(record_path.read_bytes())
        if cross_environment:
            monkeypatch.setenv('XDG_RUNTIME_DIR', str(private_runtime / 'another-terminal'))
        result = await listing(directory=private_runtime)
        assert [(process.pid, process.instance, process.sessions) for process in result] == [
            (registry.pid, registry.instance, [])
        ]
        save_record(private_runtime, replace(record, instance='reused-pid'))
        assert await listing(directory=private_runtime) == []
        assert not record_path.exists() and path.exists()


@pytest.mark.parametrize('size', [100_000, 1_100_000])
async def test_large_hello_is_bounded_and_oversize_is_visible(private_runtime: Path, size: int) -> None:
    registry = LiveRegistry(lambda: [], directory=private_runtime)
    async with registry.serving(subscribe):
        assert registry.socket_path is not None
        registry.snapshot = lambda: Hello(
            op='hello_reply',
            instance=registry.instance,
            protocol=1,
            sessions=[
                LiveEntry(
                    id='x',
                    name='folder',
                    conversation_id='x',
                    title='x' * size,
                    workspace='x',
                    model='test',
                    status='idle',
                )
            ],
        )
        if size > 1_048_576:
            with pytest.raises(ResponseTooLargeError, match='exceeds 1 MiB'):
                await request(registry.socket_path, HelloRequest())
            with pytest.raises(ResponseTooLargeError, match='exceeds 1 MiB'):
                await listing(directory=private_runtime)
        else:
            reply = await request(registry.socket_path, HelloRequest())
            assert len(reply.sessions[0].title) == size


@pytest.mark.parametrize(
    ('reply', 'error', 'message'),
    [
        (b'', ValueError, 'Invalid control frame'),
        (b'{}', ValueError, 'Invalid control frame'),
        (b'{}\n', ControlProtocolError, 'Invalid control frame'),
        (b'{"op":"hello","protocol":1}\n', ControlProtocolError, 'Unexpected hello request reply'),
        (b'{"op":"error","error":"unsupported","message":"no"}\n', ControlProtocolError, 'no'),
    ],
)
async def test_client_rejects_invalid_replies(
    private_runtime: Path, reply: bytes, error: type[Exception], message: str
) -> None:
    async def respond(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.readline()
        writer.write(reply)
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    path = private_runtime / 'reply.sock'
    async with await asyncio.start_unix_server(respond, path=str(path)):
        with pytest.raises(error, match=message):
            await request(path, HelloRequest())


@pytest.mark.parametrize('result', [0, -1])
def test_darwin_peer_uid(monkeypatch: pytest.MonkeyPatch, result: int) -> None:
    class LibC:
        @staticmethod
        def getpeereid(fd: int, uid: ctypes.c_void_p, gid: ctypes.c_void_p) -> int:
            ctypes.cast(uid, ctypes.POINTER(ctypes.c_uint))[0] = os.getuid()
            return result

    def library(name: str | None, *, use_errno: bool) -> LibC:
        return LibC()

    monkeypatch.setattr(live_registry.sys, 'platform', 'darwin')
    monkeypatch.setattr(live_registry.ctypes, 'CDLL', library)
    with socket.socket() as connection:
        assert peer_uid(connection) == (os.getuid() if result == 0 else None)


@pytest.mark.parametrize('uid', [501, 4_294_967_294])
def test_linux_peer_uid(monkeypatch: pytest.MonkeyPatch, uid: int) -> None:
    def credentials(connection: socket.socket, level: int, option: int, size: int) -> bytes:
        assert (level, option, size) == (socket.SOL_SOCKET, 12345, 12)
        return struct.pack('iII', os.getpid(), uid, os.getgid())

    monkeypatch.setattr(live_registry.sys, 'platform', 'linux')
    monkeypatch.setattr(socket, 'SO_PEERCRED', 12345, raising=False)
    monkeypatch.setattr(socket.socket, 'getsockopt', credentials)
    with socket.socket() as connection:
        assert peer_uid(connection) == uid


def test_missing_peer_credentials_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(live_registry.sys, 'platform', 'linux')
    monkeypatch.delattr(socket, 'SO_PEERCRED', raising=False)
    with socket.socket() as connection:
        assert peer_uid(connection) is None


async def test_socket_rejects_wrong_uid(private_runtime: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def other_uid(connection: socket.socket) -> int:
        return os.getuid() + 1

    registry = LiveRegistry(lambda: [], directory=private_runtime)
    async with registry.serving(subscribe):
        assert registry.socket_path is not None
        monkeypatch.setattr(live_registry, 'peer_uid', other_uid)
        reader, writer = await asyncio.open_unix_connection(str(registry.socket_path))
        writer.write(b'{"op":"hello","protocol":1}\n')
        await writer.drain()
        with suppress(ConnectionResetError):
            assert await reader.read() == b''
        writer.close()
        with suppress(ConnectionResetError):
            await writer.wait_closed()


@pytest.mark.parametrize('outcome', ['timeout', 'permission', 'missing', 'invalid', 'malformed'])
async def test_unverified_records_preserve_live_sockets(
    private_runtime: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    registry = LiveRegistry(lambda: [], directory=private_runtime)
    async with registry.serving(subscribe):
        path = registry.socket_path
        assert path is not None
        record_path = private_runtime / f'{registry.pid}.json'
        record = _RECORD.validate_json(record_path.read_bytes())
        if outcome == 'timeout':

            async def timeout(path: Path, frame: HelloRequest) -> Hello:
                raise TimeoutError

            monkeypatch.setattr(live_registry, 'request', timeout)
        elif outcome == 'permission':

            def denied(pid: int, signal: int) -> None:
                raise PermissionError

            monkeypatch.setattr(live_registry.os, 'kill', denied)
        elif outcome == 'missing':
            path.unlink()
        elif outcome == 'invalid':
            save_record(private_runtime, replace(record, socket=str(private_runtime / 'unrelated.sock')))
        else:
            write_private(record_path, '{')
        assert await listing(directory=private_runtime) == []
        assert record_path.exists() == (outcome != 'missing')
        assert path.exists() == (outcome != 'missing')


async def test_cleanup_preserves_socket_replaced_during_probe(
    private_runtime: Path, runtime_directory: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid = os.getpid() + 100_000_000
    folder = runtime_directory / f'pydantic-clai2-{os.getuid()}'
    folder.mkdir(mode=0o700)
    path = folder / f'{pid}.sock'
    with socket.socket(socket.AF_UNIX) as abandoned, socket.socket(socket.AF_UNIX) as replacement:
        abandoned.bind(str(path))
        path.chmod(0o600)
        record_path = save_record(private_runtime, LiveRecord(pid=pid, instance='old', socket=str(path)))

        async def replaced_during_probe(path: Path, frame: HelloRequest) -> Hello:
            path.rename(path.with_suffix('.old'))
            replacement.bind(str(path))
            path.chmod(0o600)
            replacement.listen()
            raise ConnectionRefusedError

        monkeypatch.setattr(live_registry, 'request', replaced_during_probe)
        assert await listing(directory=private_runtime) == []
        assert path.exists() and not record_path.exists()


async def test_abandoned_socket_and_dead_pid_are_cleaned(private_runtime: Path, runtime_directory: Path) -> None:
    pid = os.getpid() + 100_000_000
    folder = runtime_directory / f'pydantic-clai2-{os.getuid()}'
    folder.mkdir(mode=0o700)
    path = folder / f'{pid}.sock'
    with socket.socket(socket.AF_UNIX) as abandoned:
        abandoned.bind(str(path))
    path.chmod(0o600)
    record_path = save_record(private_runtime, LiveRecord(pid=pid, instance='dead', socket=str(path)))
    assert await listing(directory=private_runtime) == []
    assert not path.exists() and not record_path.exists()


async def test_second_owner_cannot_replace_live_socket(private_runtime: Path) -> None:
    first = LiveRegistry(lambda: [], directory=private_runtime)
    second = LiveRegistry(lambda: [], directory=private_runtime)
    async with first.serving(subscribe):
        with pytest.raises(OSError, match='already active'):
            async with second.serving(subscribe):
                pass
        assert first.socket_path is not None
        assert (await request(first.socket_path, HelloRequest())).instance == first.instance


@pytest.mark.parametrize('variable', ['TMPDIR', 'XDG_RUNTIME_DIR'])
async def test_relative_runtime_environment_is_discoverable(
    private_runtime: Path, monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    monkeypatch.chdir(private_runtime.parent.parent)
    monkeypatch.delenv('XDG_RUNTIME_DIR', raising=False)
    monkeypatch.setenv(variable, 'runtime')
    registry = LiveRegistry(lambda: [], directory=private_runtime)
    async with registry.serving(subscribe):
        assert [process.instance for process in await listing(directory=private_runtime)] == [registry.instance]
        assert registry.socket_path is not None and registry.socket_path.is_absolute()


async def test_socket_directory_symlink_is_not_followed(private_runtime: Path, runtime_directory: Path) -> None:
    target = runtime_directory / 'private-target'
    target.mkdir(mode=0o700)
    (runtime_directory / f'pydantic-clai2-{os.getuid()}').symlink_to(target, target_is_directory=True)
    registry = LiveRegistry(lambda: [], directory=private_runtime)
    with pytest.raises(PermissionError, match='Unsafe CLAI directory'):
        async with registry.serving(subscribe):
            pass
    assert list(target.iterdir()) == []


async def test_failed_record_write_cleans_up_listener(private_runtime: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(path: Path, value: str, *, replace: bool = True) -> None:
        raise OSError('disk full')

    registry = LiveRegistry(lambda: [], directory=private_runtime)
    monkeypatch.setattr(live_registry, 'write_private', fail)
    with pytest.raises(OSError, match='disk full'):
        async with registry.serving(subscribe):
            pass
    assert registry.socket_path is not None and not registry.socket_path.exists()
    assert list(private_runtime.iterdir()) == []


def test_sessions_command_rejects_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, 'platform', 'win32')
    monkeypatch.setattr(sys, 'argv', ['clai2', 'sessions'])
    with pytest.raises(SystemExit, match='requires Unix sockets'):
        cli.run()
