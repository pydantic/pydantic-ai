"""In-process ACP *wire* harness: drive a `PydanticAIACPAgent` through the real SDK router and codec.

The direct-call tests invoke adapter methods straight -- below the JSON-RPC router and JSON
serialization. That boundary cannot see two whole classes of bug: a method the SDK router rejects
before it reaches the adapter (e.g. an unstable method without `use_unstable_protocol`), and a
frame whose *serialized bytes* differ from the Python object the adapter handed over (e.g. an
`ensure_ascii` expansion that overruns the client's read buffer).

`wire_agent` closes that gap without a subprocess: it connects a real `ClientSideConnection` to
`acp.run_agent` over an in-memory `socket.socketpair()`, in one event loop. Every request crosses
the SDK's method routing and serialization, and every `session/update` arrives as bytes the client
re-parses -- exactly as an editor would drive it. The client-side `StreamReader` keeps asyncio's
default 64 KiB line limit, so an oversized frame fails the read just as it would over stdio.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
from types import TracebackType

import acp
from acp.client.connection import ClientSideConnection

from tests.experimental.acp._acp_clients import RecordingClientBase  # pyright: ignore[reportMissingTypeStubs]


class WireClient(RecordingClientBase):
    """A real ACP client reached over the wire, recording the `session/update`s it receives.

    The protocol-level conformance tests in this batch do not drive permission, filesystem, or
    terminal methods; those stay stubbed by `RecordingClientBase`.
    """

    def texts(self) -> str:
        """The concatenated agent message text received over the wire (other update kinds skipped)."""
        out: list[str] = []
        for update in self.updates:
            if getattr(update, 'session_update', '') == 'agent_message_chunk':
                out.append(getattr(getattr(update, 'content', None), 'text', ''))
        return ''.join(out)


class wire_agent:
    """Serve `adapter` over an in-memory socket pair; `async with` yields a connected client connection.

    The agent runs as a background task for the lifetime of the context; on exit the task is
    cancelled and both stream ends are closed. Set `unstable=False` to drive the agent without
    `use_unstable_protocol`, the configuration in which the SDK router rejects unstable methods.

    A class rather than an `@asynccontextmanager` generator only because the bundled type stubs
    flag that decorator as deprecated under strict checking.
    """

    def __init__(
        self,
        adapter: acp.Agent,
        client: WireClient | None = None,
        *,
        unstable: bool = True,
    ) -> None:
        self._adapter = adapter
        self._client = client or WireClient()
        self._unstable = unstable
        self._server: asyncio.Task[None] | None = None
        self._writers: tuple[asyncio.StreamWriter, asyncio.StreamWriter] | None = None

    async def __aenter__(self) -> tuple[ClientSideConnection, WireClient]:
        agent_sock, client_sock = socket.socketpair()
        # Each side's `input_stream` is the writer it sends to the peer with; `output_stream` is
        # the reader it receives on -- the order both SDK connection constructors require.
        agent_reader, agent_writer = await asyncio.open_connection(sock=agent_sock)
        client_reader, client_writer = await asyncio.open_connection(sock=client_sock)
        self._writers = (client_writer, agent_writer)
        self._server = asyncio.ensure_future(
            acp.run_agent(
                self._adapter,
                input_stream=agent_writer,
                output_stream=agent_reader,
                use_unstable_protocol=self._unstable,
            )
        )
        conn = acp.connect_to_agent(
            self._client, input_stream=client_writer, output_stream=client_reader, use_unstable_protocol=self._unstable
        )
        return conn, self._client

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        assert self._server is not None and self._writers is not None
        self._server.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._server
        for writer in self._writers:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
