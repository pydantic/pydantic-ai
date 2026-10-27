"""Socket discovery is tested in-process, with real Unix sockets and private files."""

import asyncio
import os
import socket
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from pydantic import TypeAdapter

from pydantic_clai2.builtin_plugins.workspaces.registry import write_private
from pydantic_clai2.runtime import live_registry
from pydantic_clai2.runtime.live_registry import LiveRegistry, listing, request


def subscribe(callback: Callable[[], None]) -> Callable[[], None]:
    """A listener whose lifetime is controlled by the registry context."""
    return lambda: None


@pytest.fixture
def private_runtime(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    with TemporaryDirectory(prefix='clai2-', dir='/tmp') as root:
        monkeypatch.setenv('XDG_RUNTIME_DIR', root)
        directory = Path(root) / 'state' / 'live'
        directory.mkdir(parents=True, mode=0o700)
        yield directory


async def test_control_lifecycle_and_bounded_frames(private_runtime: Path) -> None:
    registry = LiveRegistry(lambda: [], directory=private_runtime)
    async with registry.serving(subscribe):
        assert registry.socket_path is not None
        assert registry.socket_path.stat().st_mode & 0o777 == 0o600
        assert (registry.socket_path.parent.stat().st_mode & 0o777) == 0o700
        assert (await request(registry.socket_path, {'op': 'hello', 'protocol': 1})) == {
            'instance': registry.instance,
            'protocol': 1,
            'features': ['list'],
        }
        assert await request(registry.socket_path, {'op': 'list'}) == {'sessions': []}
        assert await request(registry.socket_path, {'op': 'unknown'}) == {'error': 'unsupported'}
        reader, writer = await asyncio.open_unix_connection(str(registry.socket_path))
        writer.write(b'x' * 70_000 + b'\n')
        await writer.drain()
        assert await asyncio.wait_for(reader.read(), 2) == b''
        writer.close()
        await writer.wait_closed()
    assert not registry.socket_path.exists()


@pytest.mark.parametrize('frame', [b'{\n', b'[]\n'])
async def test_malformed_frame_is_closed_without_server_errors(private_runtime: Path, frame: bytes) -> None:
    registry = LiveRegistry(lambda: [], directory=private_runtime)
    loop = asyncio.get_running_loop()
    errors: list[dict[str, object]] = []
    previous = loop.get_exception_handler()

    def record_error(loop: asyncio.AbstractEventLoop, context: dict[str, object]) -> None:
        errors.append(context)

    loop.set_exception_handler(record_error)
    try:
        async with registry.serving(subscribe):
            reader, writer = await asyncio.open_unix_connection(str(registry.socket_path))
            writer.write(frame)
            await writer.drain()
            assert await reader.read() == b''
            writer.close()
            await writer.wait_closed()
        assert not errors
    finally:
        loop.set_exception_handler(previous)


async def test_listing_validates_instance_and_does_not_unlink_replacement_socket(private_runtime: Path) -> None:
    registry = LiveRegistry(lambda: [], directory=private_runtime)
    async with registry.serving(subscribe):
        record = registry.snapshot()
        path = private_runtime / f'{registry.pid}.json'
        adapter = TypeAdapter(type(record))
        write_private(path, adapter.dump_json(record).decode())
        assert [found.instance for found in await listing(directory=private_runtime)] == [record.instance]
        write_private(path, adapter.dump_json(replace(record, instance='reused-pid')).decode())
        assert await listing(directory=private_runtime) == []
        assert not path.exists()
        assert registry.socket_path is not None and registry.socket_path.exists()
        write_private(path, adapter.dump_json(record).decode())
        path.chmod(0o644)
        assert await listing(directory=private_runtime) == []
        assert path.exists()


async def test_stale_dead_pid_cannot_delete_live_socket(private_runtime: Path) -> None:
    registry = LiveRegistry(lambda: [], directory=private_runtime)
    async with registry.serving(subscribe):
        record = registry.snapshot()
        stale_pid = registry.pid + 100_000_000
        path = private_runtime / f'{stale_pid}.json'
        assert registry.socket_path is not None
        stale_socket = registry.socket_path.with_name(f'{stale_pid}.sock')
        with socket.socket(socket.AF_UNIX) as dead:
            dead.bind(str(stale_socket))
        stale_socket.chmod(0o600)
        write_private(
            path,
            TypeAdapter(type(record)).dump_json(replace(record, pid=stale_pid, socket=str(stale_socket))).decode(),
        )
        assert await listing(directory=private_runtime) == []
        assert not path.exists()
        assert not stale_socket.exists()
        assert registry.socket_path.exists()


async def test_cleanup_preserves_socket_replaced_during_probe(
    private_runtime: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = LiveRegistry(lambda: [], directory=private_runtime)
    dead_pid = registry.pid + 100_000_000
    socket_directory = private_runtime.parent.parent / 'pydantic-clai2'
    socket_directory.mkdir(mode=0o700)
    socket_path = socket_directory / f'{dead_pid}.sock'
    with socket.socket(socket.AF_UNIX) as abandoned, socket.socket(socket.AF_UNIX) as replacement:
        abandoned.bind(str(socket_path))
        socket_path.chmod(0o600)
        record = replace(registry.snapshot(), pid=dead_pid, socket=str(socket_path))
        write_private(private_runtime / f'{dead_pid}.json', TypeAdapter(type(record)).dump_json(record).decode())

        async def replaced_during_probe(path: Path, frame: dict[str, object]) -> dict[str, object]:
            path.rename(path.with_suffix('.old'))
            replacement.bind(str(path))
            path.chmod(0o600)
            replacement.listen()
            raise ConnectionRefusedError

        monkeypatch.setattr(live_registry, 'request', replaced_during_probe)
        assert await listing(directory=private_runtime) == []
        assert socket_path.exists(), 'cleanup must not unlink the replacement listener'


async def test_second_owner_cannot_replace_live_socket(private_runtime: Path) -> None:
    first = LiveRegistry(lambda: [], directory=private_runtime)
    second = LiveRegistry(lambda: [], directory=private_runtime)
    async with first.serving(subscribe):
        with pytest.raises(OSError, match='already active'):
            async with second.serving(subscribe):
                pass
        assert first.socket_path is not None
        assert (await request(first.socket_path, {'op': 'hello'}))['instance'] == first.instance


async def test_reused_pid_with_refused_socket_is_cleaned(private_runtime: Path) -> None:
    registry = LiveRegistry(lambda: [], directory=private_runtime)
    socket_directory = private_runtime.parent.parent / 'pydantic-clai2'
    socket_directory.mkdir(mode=0o700)
    socket_path = socket_directory / f'{registry.pid}.sock'
    with socket.socket(socket.AF_UNIX) as dead:
        dead.bind(str(socket_path))
    socket_path.chmod(0o600)
    registry.socket_path = socket_path
    path = private_runtime / f'{registry.pid}.json'
    write_private(path, TypeAdapter(type(registry.snapshot())).dump_json(registry.snapshot()).decode())
    assert await listing(directory=private_runtime) == []
    assert not path.exists()
    async with registry.serving(subscribe):
        assert await request(socket_path, {'op': 'hello'})
    assert not socket_path.exists()


async def test_rejects_unsafe_directory_and_wrong_uid(private_runtime: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = LiveRegistry(lambda: [], directory=private_runtime)
    private_runtime.chmod(0o755)
    with pytest.raises(PermissionError):
        async with registry.serving(subscribe):
            pass
    private_runtime.chmod(0o700)
    async with registry.serving(subscribe):
        assert registry.socket_path is not None

        def other_uid(connection: socket.socket) -> int:
            return os.getuid() + 1

        monkeypatch.setattr(live_registry, '_peer_uid', other_uid)
        reader, writer = await asyncio.open_unix_connection(str(registry.socket_path))
        writer.write(b'{"op":"hello","protocol":1}\n')
        await writer.drain()
        assert await reader.read() == b''
        writer.close()
        await writer.wait_closed()
