"""A loopback gRPC server for Monty transport tests, with refusal and subprocess bridge modes.

`pydantic_monty.AsyncMontyRemote` opens one `/monty.v1.MontySession/Session` stream per worker. The
stdio worker reads and writes the same protobuf messages with a four-byte little-endian length prefix,
so the relay only adds and removes that prefix. It never parses a message. Tests can use
`serve_monty_relay(refusal_causes=...)` to exercise native refusal decoding without starting workers.
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

    def encode(self, message: Any, message_type: Any) -> bytes:  # pragma: no cover -- only bridged sessions (#8824)
        return message

    def decode(self, data: bytes, message_type: Any) -> Any:  # pragma: no cover -- only bridged sessions (#8824)
        return data


@dataclass
class MontyRelay:
    """Serves remote Monty sessions and records the metadata each stream opened with."""

    refuse: bool = False
    """Refuse every stream with `UNAUTHENTICATED`, as a server does for a bad key."""
    refusal_causes: tuple[str, ...] | None = None
    """Refuse streams with these causes in order, repeating the last cause after exhaustion."""
    url: str = ''
    metadata: list[Mapping[str, str]] = field(default_factory=list[Mapping[str, str]])
    _streams: list[Stream[bytes, bytes]] = field(default_factory=list[Stream[bytes, bytes]])

    def __mapping__(self) -> dict[str, Handler]:
        return {SESSION_METHOD: Handler(self._session, Cardinality.STREAM_STREAM, bytes, bytes)}

    async def drop_streams(self) -> None:  # pragma: no cover -- only the skipped #8824 tests bridge a session
        """Reset every open stream, as a lost worker connection does."""
        await asyncio.gather(*(stream.cancel() for stream in self._streams))

    async def _session(self, stream: Stream[bytes, bytes]) -> None:
        assert stream.metadata is not None
        self.metadata.append({key: value for key, value in stream.metadata.items() if isinstance(value, str)})
        if self.refusal_causes is not None:
            cause = self.refusal_causes[min(len(self.metadata), len(self.refusal_causes)) - 1]
            status = {
                'caller_cap': Status.RESOURCE_EXHAUSTED,
                'capacity': Status.RESOURCE_EXHAUSTED,
                'worker_unavailable': Status.UNAVAILABLE,
                'unauthenticated': Status.UNAUTHENTICATED,
                'permission_denied': Status.PERMISSION_DENIED,
            }.get(cause, Status.RESOURCE_EXHAUSTED)
            await stream.send_trailing_metadata(
                status=status, status_message='refused', metadata={'monty-cause': cause}
            )
            return
        if self.refuse:
            raise GRPCError(Status.UNAUTHENTICATED, 'unauthorized')
        return await self._bridge(stream)  # pragma: no cover -- only the skipped #8824 tests bridge a session

    async def _bridge(self, stream: Stream[bytes, bytes]) -> None:  # pragma: no cover -- only the skipped #8824 tests
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
async def serve_monty_relay(
    port: int = 0, *, refuse: bool = False, refusal_causes: tuple[str, ...] | None = None
) -> AsyncGenerator[MontyRelay]:
    """Serve a relay with its URL set; explicit causes override `refuse` and must not be empty."""
    if refusal_causes == ():
        raise ValueError('refusal_causes must contain at least one cause')
    relay = MontyRelay(refuse=refuse, refusal_causes=refusal_causes)
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
