"""Controllable fake for the Sprites SDK boundary.

The Sprite handles, clients, exception classes and the exec WebSocket protocol handler
(`sprites.websocket.WSCommand`) are the real ones from the installed SDK; only the network calls
are replaced. `SpriteTransport.connect` stands in for the `websockets` `connect` that `WSCommand`
calls, so the SDK builds the real exec URL, sends stdin EOF, and reads the real frames. Commands run
for real: an exec socket runs the URL's `cmd` argv in a local subprocess, in its `dir`
(`SpriteTransport.root` by default) and this process's environment, so commands share one host
directory and deadlines are real. Output streams back as STDOUT frames (both pipes), then an EXIT
frame. Closing the socket kills a command that is still running, as a positive
`max_run_after_disconnect` makes the Sprite do (the fake does it at once instead of after that time).

Like the live Sprite, the command starts as soon as the socket opens, and its output is streamed only
once the client has attached, which the fake takes to be the client's stdin EOF frame. The live
Sprite replays the last 16 or 64 KiB printed before that; the fake takes the worst case and replays
nothing. An exec URL longer than `SpriteTransport.url_limit` is refused with HTTP 414, as the live
Sprite refuses one of about 40 KB. File writes (`AsyncSpritePath.stat` and `write_bytes`) go to the
same host directory.

Deletion follows the SDK: `destroy_sprite` (and `AsyncSprite.delete()`) returns once the API accepts
the request, after which `get_sprite` raises `NotFoundError` and an exec handshake with the deleted
Sprite fails with HTTP 404 (`websockets.exceptions.InvalidStatus`, which the SDK parses into an
`APIError`).
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import IO
from urllib.parse import parse_qs, unquote, urlsplit

import anyio
from sprites import AsyncSprite, AsyncSpritesClient
from sprites.async_filesystem import AsyncSpritePath
from sprites.exceptions import FileNotFoundError_, IsADirectoryError_, NotADirectoryError_, NotFoundError
from sprites.types import FileStat
from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus
from websockets.http11 import Response

_STDOUT, _EXIT, _STDIN_EOF = 1, 3, 4


class FakeSocketTransport:
    def __init__(self, sprite_transport: SpriteTransport) -> None:
        self.sprite_transport = sprite_transport

    def abort(self) -> None:
        self.sprite_transport.aborted += 1


class FakeExecSocket:
    """One exec WebSocket, from the handshake to the EXIT frame."""

    def __init__(self, sprite_transport: SpriteTransport, url: str) -> None:
        self.sprite_transport = sprite_transport
        self.query_url = url
        self.query = parse_qs(urlsplit(url).query)
        self.transport = FakeSocketTransport(sprite_transport)
        # Read by the SDK when the stream ends without an EXIT frame.
        self.close_code: int | None = None
        self.close_reason: str | None = None
        # Set on the client's stdin EOF; output printed before then is not streamed.
        self.attached = False
        self._frames: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._loop = asyncio.get_running_loop()
        self.process = subprocess.Popen(
            self.query['cmd'],
            cwd=self.query.get('dir', [str(sprite_transport.root)])[0],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Its own process group, so closing the socket ends the command's children too.
            start_new_session=True,
        )
        self.thread = threading.Thread(target=self._execute)
        self.thread.start()

    def _execute(self) -> None:
        process = self.process
        assert process.stdout is not None and process.stderr is not None
        readers = [
            threading.Thread(target=self._pump, args=(process.stdout,)),
            threading.Thread(target=self._pump, args=(process.stderr,)),
        ]
        for reader in readers:
            reader.start()
        code = process.wait()
        for reader in readers:
            reader.join()
        if self.sprite_transport.connection_dropped:
            self.close_code = 1006
            self._send(None)
        else:
            exit_code = code if self.sprite_transport.exit_override is None else self.sprite_transport.exit_override
            if self.sprite_transport.delete_on_exit:
                self.sprite_transport.names.discard(unquote(urlsplit(self.query_url).path.split('/')[3]))
            self._send(bytes([_EXIT, exit_code % 256]))

    def _pump(self, source: IO[bytes]) -> None:
        # The live Sprite delivered stderr on the stdout stream in some runs and not others; this
        # takes the worst case every time, so the backend must not rely on the stderr stream.
        while chunk := os.read(source.fileno(), 4096):
            if self.attached:  # pragma: no branch - only an ungated command prints before it attaches
                self._send(bytes([_STDOUT]) + chunk)
        source.close()

    def _send(self, frame: bytes | None) -> None:
        self._loop.call_soon_threadsafe(self._frames.put_nowait, frame)

    def __aiter__(self) -> FakeExecSocket:
        return self

    async def __anext__(self) -> bytes:
        frame = await self._frames.get()
        if frame is None:
            raise StopAsyncIteration
        return frame

    async def send(self, message: bytes) -> None:
        # The backend sends no stdin data, only the EOF the SDK sends once the socket is open.
        assert message == bytes([_STDIN_EOF]) and self.process.stdin is not None
        if self.sprite_transport.release_stdin_eof is not None:
            await self.sprite_transport.release_stdin_eof.wait()
        self.attached = True
        self.process.stdin.close()

    async def close(self) -> None:
        sprite_transport = self.sprite_transport
        sprite_transport.exec_close_started.set()
        if sprite_transport.release_exec_close is not None:
            await sprite_transport.release_exec_close.wait()
        if sprite_transport.exec_close_hang:
            await anyio.sleep(1)
        if sprite_transport.exec_close_error is not None:
            raise sprite_transport.exec_close_error
        sprite_transport.exec_closes += 1
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGKILL)
        self._frames.put_nowait(None)


class SpriteTransport:
    """SDK acquisition and exec WebSocket fake; exec commands run in local subprocesses."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.names: set[str] = set()
        self.created: list[str] = []
        self.create_started = asyncio.Event()
        self.release_create: asyncio.Event | None = None
        self.release_get: asyncio.Event | None = None
        self.clients: list[AsyncSpritesClient] = []
        self.execs: list[FakeExecSocket] = []
        self.exec_started = asyncio.Event()
        self.get_error: Exception | None = None
        self.get_error_once: Exception | None = None
        self.create_error: Exception | None = None
        self.create_error_after_commit: Exception | None = None
        self.close_error: Exception | None = None
        self.close_calls = 0
        self.close_started = asyncio.Event()
        self.release_close: asyncio.Event | None = None
        self.connect_error: Exception | None = None
        self.connect_error_once: Exception | None = None
        # The socket closes before the command reports an exit status.
        self.connection_dropped = False
        self.exec_close_error: Exception | None = None
        self.exec_close_hang = False
        self.exec_close_started = asyncio.Event()
        self.release_exec_close: asyncio.Event | None = None
        self.exec_closes = 0
        self.aborted = 0
        self.exit_override: int | None = None
        self.delete_on_exit = False
        # Holds the client's stdin EOF, and with it the stream's attachment, until set.
        self.release_stdin_eof: asyncio.Event | None = None
        self.url_limit = 40_000
        self.fs_writes: list[str] = []

    def client(self, token: str) -> AsyncSpritesClient:
        client = AsyncSpritesClient(token=token)
        self.clients.append(client)
        return client

    async def connect(self, url: str, **kwargs: object) -> FakeExecSocket:
        if self.connect_error_once is not None:
            error = self.connect_error_once
            self.connect_error_once = None
            raise error
        if self.connect_error is not None:
            raise self.connect_error
        # /v1/sprites/{name}/exec
        name = unquote(urlsplit(url).path.split('/')[3])
        if name not in self.names:
            raise InvalidStatus(Response(404, 'Not Found', Headers()))
        if len(url) > self.url_limit:
            raise InvalidStatus(Response(414, 'URI Too Long', Headers()))
        socket = FakeExecSocket(self, url)
        self.execs.append(socket)
        self.exec_started.set()
        return socket

    async def get(self, client: AsyncSpritesClient, name: str) -> AsyncSprite:
        if self.get_error_once is not None:
            error = self.get_error_once
            self.get_error_once = None
            raise error
        if self.get_error is not None:
            raise self.get_error
        if self.release_get is not None:
            await self.release_get.wait()
        if name not in self.names:
            raise NotFoundError(name)
        return AsyncSprite(name, client)

    async def create(self, client: AsyncSpritesClient, name: str, *, runtime: str | None) -> AsyncSprite:
        if self.create_error is not None:
            raise self.create_error
        self.names.add(name)
        self.created.append(name)
        self.create_started.set()
        if self.create_error_after_commit is not None:
            raise self.create_error_after_commit
        if self.release_create is not None:
            await self.release_create.wait()
        return AsyncSprite(name, client)

    async def destroy(self, client: AsyncSpritesClient, name: str) -> None:
        self.names.discard(name)

    async def close(self, client: AsyncSpritesClient) -> None:
        self.close_calls += 1
        self.close_started.set()
        if self.release_close is not None:
            await self.release_close.wait()
        if self.close_error is not None:
            error = self.close_error
            self.close_error = None
            raise error

    def _fs_path(self, path: AsyncSpritePath) -> Path:
        # The filesystem API answers 404 for a deleted Sprite as for a missing path.
        if path._fs._sprite.name not in self.names:  # pyright: ignore[reportPrivateUsage]
            raise FileNotFoundError_('fs', str(path))
        return Path(str(path))

    async def fs_stat(self, path: AsyncSpritePath) -> FileStat:
        target = self._fs_path(path)
        # The API lists a directory's entries and the SDK reports the first; the fake keeps that.
        entries = sorted(target.iterdir()) if target.is_dir() else [target] if target.exists() else []
        if not entries:
            raise FileNotFoundError_('stat', str(path))
        entry = entries[0]
        entry_stat = entry.stat()
        return FileStat(
            name=entry.name,
            path=str(entry),
            size=entry_stat.st_size,
            mode=f'{entry_stat.st_mode & 0o777:o}',
            mod_time=datetime.now(),
            is_dir=entry.is_dir(),
        )

    async def fs_write(self, path: AsyncSpritePath, data: bytes, mode: int) -> None:
        target = self._fs_path(path)
        if target.is_dir():
            raise IsADirectoryError_('write', str(path))
        if any(parent.exists() and not parent.is_dir() for parent in target.parents):
            raise NotADirectoryError_('write', str(path))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        target.chmod(mode)
        self.fs_writes.append(str(path))
