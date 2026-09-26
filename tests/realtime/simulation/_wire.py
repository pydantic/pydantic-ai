"""In-memory WebSocket transport for the simulated OpenAI-protocol and GPT-Live servers.

The real connection classes dial `websockets.connect`, which `Network.patch` replaces with a factory
handing out `FakeWebSocket`s wired to a simulated server. Frames the server emits are queued *in
flight* until the simulation delivers them, so the trace controls exactly which provider frames the
client has read at any point: that is how a response terminal gets to race a client send, or a
straggler delta gets to arrive after a cancel. (Handshake replies skip the queue: the trace only
controls the live session.)

Outbound frames reach the server synchronously inside `send()`, after a number of event-loop yields
(send latency) drawn from the simulation's seeded random source, so a send can be in flight while the
receive pump handles other frames. Faults are injected here too: a send can fail before or after the
server received it, the whole connection can drop, and a re-dial can be refused.
"""

from __future__ import annotations as _annotations

import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator, Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
from unittest import mock

import websockets
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from websockets.frames import Close

SendFault = Literal['lost', 'ambiguous']
"""How an injected send failure behaves.

`'lost'` fails before the server sees the frame; `'ambiguous'` fails after it did, which is what a
socket dying while the frame is on the wire looks like to the client: it can't tell whether the frame
arrived. Either way the connection is dropped, as a real socket error would.
"""

_HANDSHAKE_FRAME_TYPES = frozenset({'session.update', 'session.start'})


class WireServer(Protocol):
    """What a simulated server exposes to its transport."""

    def on_connect(self, socket: FakeWebSocket, url: str) -> None:
        """A new socket was dialed at `url`; emit whatever the provider sends first."""
        ...

    def on_client_frame(self, socket: FakeWebSocket, frame: dict[str, Any]) -> None:
        """React to one frame the client sent on `socket`."""
        ...

    def on_client_read(self, socket: FakeWebSocket, frame: dict[str, Any]) -> None:
        """Note that the client read `frame`: what it has seen is part of the ground truth."""
        ...

    def on_disconnect(self, socket: FakeWebSocket) -> None:
        """`socket` dropped: whatever the server had in flight on it is lost."""
        ...


class DialRefused(OSError):
    """A simulated dial the network refused."""


@dataclass
class _Closed:
    ok: bool
    code: int
    reason: str


@dataclass(eq=False)
class FakeWebSocket:
    """One simulated socket: a server-to-client queue plus a record of both directions."""

    network: Network
    index: int
    inbox: deque[str | _Closed] = field(default_factory=deque[str | _Closed])
    in_flight: deque[str] = field(default_factory=deque[str])
    sent: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    """Frames the server received on this socket, in order."""
    received: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    """Frames the client actually read, in order."""
    broken: bool = False
    closed_by_client: bool = False
    close_code: int | None = None
    close_reason: str = ''
    _readable: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def alive(self) -> bool:
        return not self.broken and not self.closed_by_client

    # --- client side (what the connection class calls) --------------------------------------------

    async def send(self, message: str | bytes) -> None:
        text = message.decode() if isinstance(message, bytes) else message
        frame = json.loads(text)
        handshake = frame.get('type') in _HANDSHAKE_FRAME_TYPES
        if not handshake:
            for _ in range(self.network.latency()):
                await asyncio.sleep(0)
        if not self.alive:
            raise ConnectionClosedError(None, Close(1006, 'simulated drop'))
        fault = None if handshake else self.network.take_send_fault()
        if fault is not None:
            last_read = self.received[-1].get('type') if self.received else None
            self.network.failed_sends.append((frame.get('type'), last_read, fault))
        if fault == 'lost':
            self.break_connection()
            raise ConnectionClosedError(None, Close(1006, 'simulated send failure'))
        self.sent.append(frame)
        self.network.server.on_client_frame(self, frame)
        if fault == 'ambiguous':
            self.break_connection()
            raise ConnectionClosedError(None, Close(1006, 'simulated send failure'))

    async def recv(self, decode: bool | None = None) -> str | bytes:
        while not self.inbox:
            self._readable.clear()
            await self._readable.wait()
        item = self.inbox[0]
        if isinstance(item, _Closed):
            # Left at the head, so every later read sees the same close.
            self.close_code, self.close_reason = item.code, item.reason
            close = Close(item.code, item.reason)
            if item.ok:  # pragma: lax no cover (the connection usually stops reading before its own close)
                raise ConnectionClosedOK(close, None)
            raise ConnectionClosedError(close, None)
        self.inbox.popleft()
        frame = json.loads(item)
        self.received.append(frame)
        self.network.server.on_client_read(self, frame)
        return item.encode() if decode is False else item

    async def __aiter__(self) -> AsyncIterator[str]:
        while True:
            try:
                item = await self.recv()
            except ConnectionClosedOK:  # pragma: lax no cover (as above)
                return
            assert isinstance(item, str)
            yield item

    async def close(self, code: int = 1000, reason: str = '') -> None:
        if self.alive:
            self.closed_by_client = True
            self.in_flight.clear()
            self._push(_Closed(ok=True, code=code, reason=reason))

    # --- server side --------------------------------------------------------------------------------

    def emit(self, frame: dict[str, Any], *, immediately: bool = False) -> None:
        """Queue a server frame; it reaches the client once the simulation delivers it."""
        if not self.alive:  # pragma: lax no cover (a race: the server answers a frame on a socket that just dropped)
            return
        text = json.dumps(frame)
        if immediately:
            self._push(text)
        else:
            self.in_flight.append(text)

    def deliver(self, count: int | None = None) -> int:
        """Move up to `count` in-flight frames (all by default) to the client's read queue."""
        moved = 0
        while self.in_flight and (count is None or moved < count):
            self._push(self.in_flight.popleft())
            moved += 1
        return moved

    def break_connection(self) -> None:
        """Drop the socket: frames still in flight are lost, reads fail once the delivered ones are read."""
        if not self.alive:  # pragma: lax no cover (a drop racing the client's own close)
            return
        self.broken = True
        self.in_flight.clear()
        self._push(_Closed(ok=False, code=1006, reason='simulated drop'))
        self.network.server.on_disconnect(self)

    def _push(self, item: str | _Closed) -> None:
        self.inbox.append(item)
        self._readable.set()


@dataclass(eq=False)
class Network:
    """The simulated network between the client and one simulated server."""

    server: WireServer
    latency: Callable[[], int] = lambda: 0
    """How many loop iterations the next send yields before it reaches the server."""
    sockets: list[FakeWebSocket] = field(default_factory=list[FakeWebSocket])
    send_faults: deque[SendFault] = field(default_factory=deque[SendFault])
    dial_failures: int = 0
    """How many upcoming dials fail before one succeeds."""
    failed_sends: list[tuple[str | None, str | None, SendFault]] = field(
        default_factory=list[tuple[str | None, str | None, SendFault]]
    )
    """`(frame type, type of the last frame the client had read, fault)` for every send a fault failed."""

    @property
    def socket(self) -> FakeWebSocket | None:
        """The most recently dialed socket."""
        return self.sockets[-1] if self.sockets else None

    def take_send_fault(self) -> SendFault | None:
        return self.send_faults.popleft() if self.send_faults else None

    def drop(self) -> None:
        socket = self.socket
        assert socket is not None
        socket.break_connection()

    def deliver(self, count: int | None = None) -> int:
        socket = self.socket
        return socket.deliver(count) if socket is not None else 0

    def in_flight(self) -> int:
        socket = self.socket
        return len(socket.in_flight) if socket is not None and socket.alive else 0

    def dial(self, url: str = '') -> FakeWebSocket:
        if self.dial_failures:
            self.dial_failures -= 1
            raise DialRefused('simulated dial failure')
        socket = FakeWebSocket(self, len(self.sockets))
        self.sockets.append(socket)
        self.server.on_connect(socket, url)
        return socket

    @contextmanager
    def patch(self) -> Generator[None]:
        """Route every `websockets.connect` to this network."""
        network = self

        class _Connect:
            def __init__(self, url: str = '', *args: Any, **kwargs: Any) -> None:
                self._url = url
                self._socket: FakeWebSocket | None = None

            async def __aenter__(self) -> FakeWebSocket:
                self._socket = network.dial(self._url)
                return self._socket

            async def __aexit__(self, *exc: object) -> None:
                assert self._socket is not None
                await self._socket.close()

        with mock.patch.object(websockets, 'connect', _Connect):
            yield
