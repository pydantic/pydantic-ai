"""The Responses WebSockets a model and its request copies share, one per event loop."""

from __future__ import annotations as _annotations

import asyncio
import threading
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import anyio
from openai import APIConnectionError

from ...exceptions import ModelConnectionError

if TYPE_CHECKING:
    from ._responses_websocket import ResponsesWebSocket, ResponsesWebSocketStream


@dataclass(eq=False)
class _ResponsesWebSocketState:
    """A model's socket and request capacity on one event loop."""

    loop: asyncio.AbstractEventLoop
    leases: int = 0
    contexts: int = 0
    reusable: bool = False
    """An agent run or an outer model scope has made this state reusable across requests."""
    in_flight: int = 0
    socket: ResponsesWebSocket | None = None
    open_lock: anyio.Lock | None = None
    connection_failed: bool = False

    async def reserve(
        self,
        connect: Callable[[], Awaitable[ResponsesWebSocket]],
        response_id: str | None,
        headers: Mapping[str, str],
    ) -> ResponsesWebSocketStream | None:
        """Reserve a lane, opening a socket first if there's none, or the current one failed, closed or aged out.

        Returns `None` when opening a connection failed, the socket has no free lane, opened with other headers, or when
        there's no socket left to use once the request's lease has ended. Only requests on this state's loop call this.
        """
        socket = self.socket
        if socket is None or not socket.accepts_requests:
            if self.open_lock is None:
                self.open_lock = anyio.Lock()
            async with self.open_lock:
                if self.socket is socket:
                    # The lease that let this request choose the socket may have ended while it waited for the lock,
                    # and a socket opened now would only serve this one request.
                    if not self.leases or self.connection_failed:
                        return None
                    try:
                        self.socket = await connect()
                    except (APIConnectionError, ModelConnectionError):
                        # Do not repeat an unavailable transport's handshake for every tool-loop request.
                        self.connection_failed = True
                        raise
                    if socket is not None:
                        # A socket replaced near OpenAI's time limit closes once its responses finish.
                        await socket.close_if_idle()
            socket = self.socket
            assert socket is not None
        if socket.headers != headers:
            return None
        return socket.reserve(response_id)


@dataclass(eq=False)
class ResponsesWebSocketHolder:
    """Loop-local sockets shared by a model and its request copies.

    Each loop opens its own socket; teardown on another loop schedules closure on the socket's loop.
    Entries are removed when their last lease and request finish, including after failed handshakes.
    """

    states: dict[asyncio.AbstractEventLoop, _ResponsesWebSocketState] = field(
        default_factory=dict[asyncio.AbstractEventLoop, _ResponsesWebSocketState]
    )
    # A context can exit on another thread. Never hold this lock during network I/O.
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def acquire(self, *, context: bool = False, reusable: bool = True) -> _ResponsesWebSocketState | None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None
        with self.lock:
            if (state := self.states.get(loop)) is None:
                state = self.states[loop] = _ResponsesWebSocketState(loop)
            state.leases += 1
            state.contexts += context
            state.reusable |= reusable
            return state

    async def exit(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        with self.lock:
            state = self.states.get(loop) if loop is not None else None
            if state is None or not state.contexts:
                # A context closed on another loop has no scope identity. Release an entered loop's lease;
                # in-flight requests keep its socket alive, and response provenance stays with that socket.
                state = next((state for state in self.states.values() if state.contexts), None)
            if state is not None:
                state.contexts -= 1
                state.leases -= 1
        if state is not None:
            await self.close_if_unused(state)

    async def release(self, state: _ResponsesWebSocketState) -> None:
        with self.lock:
            state.leases -= 1
        await self.close_if_unused(state)

    @asynccontextmanager
    async def lease(self) -> AsyncGenerator[None]:
        state = self.acquire()
        try:
            yield
        finally:
            if state is not None:
                await self.release(state)

    async def close_if_unused(self, state: _ResponsesWebSocketState) -> None:
        with self.lock:
            if state.leases or state.in_flight or self.states.get(state.loop) is not state:
                return
            self.states.pop(state.loop)
            socket, state.socket = state.socket, None
            state.open_lock = None
        if socket is not None:
            if not state.loop.is_running():
                # Its asyncio transport belongs to a stopped loop and cannot be awaited from another one.
                return
            if state.loop is not asyncio.get_running_loop():
                future = asyncio.run_coroutine_threadsafe(socket.close(), state.loop)
                await anyio.to_thread.run_sync(future.result)
            else:
                await socket.close()
