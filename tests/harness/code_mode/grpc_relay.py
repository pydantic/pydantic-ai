"""An in-process gRPC server that bridges each remote Monty session to a fresh `monty subprocess` child.

`pydantic_monty.AsyncMontyRemote` opens one `/monty.v1.MontySession/Session` stream per worker. The
stdio worker reads and writes the same protobuf messages with a four-byte little-endian length prefix,
so the relay only adds and removes that prefix. It never parses a message.
"""

from __future__ import annotations

import asyncio
import socket
import struct
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from typing import Any

from grpclib.const import Cardinality, Handler, Status
from grpclib.encoding.base import CodecBase
from grpclib.exceptions import GRPCError
from grpclib.server import Server, Stream
from pydantic_monty._binary import find_monty_binary  # the lookup `AsyncMonty()` uses for local workers

SESSION_METHOD = '/monty.v1.MontySession/Session'
_LENGTH_PREFIX = struct.Struct('<I')


class _RawCodec(CodecBase):
    """Pass each gRPC message through as the bytes it arrived as."""

    @property
    def __content_subtype__(self) -> str:
        return 'proto'

    def encode(self, message: Any, message_type: Any) -> bytes:
        return message

    def decode(self, data: bytes, message_type: Any) -> Any:
        return data


@dataclass
class MontyRelay:
    """Serves remote Monty sessions and records the metadata each stream opened with."""

    refuse: bool = False
    """Refuse every stream with `UNAUTHENTICATED`, as a server does for a bad key."""
    url: str = ''
    metadata: list[Mapping[str, str]] = field(default_factory=list[Mapping[str, str]])
    _streams: list[Stream[bytes, bytes]] = field(default_factory=list[Stream[bytes, bytes]])

    def __mapping__(self) -> dict[str, Handler]:
        return {SESSION_METHOD: Handler(self._session, Cardinality.STREAM_STREAM, bytes, bytes)}

    async def drop_streams(self) -> None:
        """Reset every open stream, as a lost worker connection does."""
        await asyncio.gather(*(stream.cancel() for stream in self._streams))

    async def _session(self, stream: Stream[bytes, bytes]) -> None:
        assert stream.metadata is not None
        self.metadata.append({key: value for key, value in stream.metadata.items() if isinstance(value, str)})
        if self.refuse:
            raise GRPCError(Status.UNAUTHENTICATED, 'unauthorized')
        self._streams.append(stream)
        child = await asyncio.create_subprocess_exec(
            find_monty_binary(), 'subprocess', stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE
        )
        stdin, stdout = child.stdin, child.stdout
        assert stdin is not None and stdout is not None

        async def to_child() -> None:
            async for message in stream:
                stdin.write(_LENGTH_PREFIX.pack(len(message)) + message)
                await stdin.drain()
            stdin.close()

        # The client sends nothing until the server admits the stream with its response headers.
        await stream.send_initial_metadata()
        to_child_task = asyncio.create_task(to_child())
        try:
            with suppress(asyncio.IncompleteReadError):
                while True:
                    (length,) = _LENGTH_PREFIX.unpack(await stdout.readexactly(_LENGTH_PREFIX.size))
                    await stream.send_message(await stdout.readexactly(length))
        finally:
            to_child_task.cancel()
            await asyncio.gather(to_child_task, return_exceptions=True)
            with suppress(ProcessLookupError):
                child.kill()
            await child.wait()


@asynccontextmanager
async def serve_monty_relay(port: int = 0, *, refuse: bool = False) -> AsyncGenerator[MontyRelay]:
    """Serve a `MontyRelay` on a loopback port (ephemeral by default), with its `url` set."""
    relay = MontyRelay(refuse=refuse)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('127.0.0.1', port))
    relay.url = f'http://127.0.0.1:{sock.getsockname()[1]}'
    server = Server([relay], codec=_RawCodec())
    await server.start(sock=sock)
    try:
        yield relay
    finally:
        server.close()
        await server.wait_closed()
