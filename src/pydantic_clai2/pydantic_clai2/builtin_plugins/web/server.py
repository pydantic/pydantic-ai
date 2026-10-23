"""Serve the web UI from CLAI's own event loop, so handlers and the shell share one loop and never a thread."""

import asyncio
import logging
import socket
import sys
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Self

import uvicorn

from pydantic_clai2.builtin_plugins.web.app import ASGIApp

LOOPBACK = frozenset({'127.0.0.1', 'localhost', '::1'})
"""Addresses that only this machine reaches."""


class _Server(uvicorn.Server):
    """A uvicorn server that leaves CLAI's Ctrl-C handling alone."""

    @contextmanager
    def capture_signals(self) -> Generator[None]:
        yield


class WebServer:
    """A uvicorn server on its own bound socket, served from a task on the running loop."""

    def __init__(self, *, sock: socket.socket, server: _Server, task: asyncio.Task[None]) -> None:
        self._socket = sock
        self._server = server
        self._task = task
        self.port: int = sock.getsockname()[1]

    @classmethod
    def serve(cls, app: Callable[[int], ASGIApp], *, host: str, port: int) -> Self:
        """Bind `host:port` (0 picks a free port) and serve `app(port)`; `OSError` when the address is taken.

        Logs stay out of the terminal, which the editor owns.
        """
        sock = socket.socket(socket.AF_INET6 if ':' in host else socket.AF_INET, socket.SOCK_STREAM)
        try:
            if sys.platform != 'win32':
                # Rebind a fixed port at once after `/web stop`; on Windows this option would let others share it.
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            config = uvicorn.Config(
                app(sock.getsockname()[1]), log_config=None, access_log=False, lifespan='off', ws='websockets-sansio'
            )
            # Here rather than in the task, where a broken configuration would fail after `/web` reported success.
            config.load()
        except BaseException:
            sock.close()
            raise
        for name in ('uvicorn', 'uvicorn.error', 'uvicorn.access'):
            logging.getLogger(name).disabled = True
        server = _Server(config)
        task = asyncio.create_task(server.serve(sockets=[sock]), name='clai2-web')
        return cls(sock=sock, server=server, task=task)

    async def stop(self) -> None:
        """Close every connection and wait for the server to finish."""
        self._server.should_exit = True
        self._server.force_exit = True
        await self._task
        self._socket.close()
