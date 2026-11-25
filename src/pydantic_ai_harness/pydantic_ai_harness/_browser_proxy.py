"""A loopback SOCKS5 proxy the browser capabilities send Chromium's connections through.

A request guard resolves a name to classify it, but Chromium resolves the name again before it connects, so a
record that changes between the two lookups would reach an address the guard never saw. Through a SOCKS5 proxy,
Chromium sends the name instead of resolving it. The proxy resolves it through the cache the guard's lookup filled,
and connects only to the addresses the policy checked.

A listener applying `PRIVATE_ADDRESS_BLOCK` carries the traffic the request guards also see. A connection names
only its host and port, and Chromium shares one across requests of different kinds to the same host, so the domain
rules, which depend on the kind, stay with the guards. `CDPBrowser` also sends its WebSockets, which CDP `Fetch`
never pauses, through a listener applying the run's whole policy.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from dataclasses import replace
from ipaddress import IPv4Address, IPv6Address

import anyio

from pydantic_ai_harness._browser_egress import EgressPolicy, EgressRequest, resolve_host, url_host

SETUP_TIMEOUT_SECONDS = 10.0
"""How long a connection gets to name its destination and reach it."""
CLOSE_GRACE_SECONDS = 1.0
"""How long a closed connection's unsent data gets to reach its peer before it is dropped."""
READ_SIZE = 65_536
REPLY_CONNECTED = b'\x05\x00\x00\x01\x00\x00\x00\x00\x00\x00'
REPLY_NOT_ALLOWED = b'\x05\x02\x00\x01\x00\x00\x00\x00\x00\x00'

PRIVATE_ADDRESS_BLOCK = EgressPolicy()
"""The policy a listener for traffic the request guards see applies: the private-address block alone, on every name."""


@asynccontextmanager
async def socks_proxy(policy: EgressPolicy) -> AsyncGenerator[int]:
    """Serve the proxy on a loopback port for the lifetime of the block, and yield the port."""
    connections: set[asyncio.Task[None]] = set()
    stopping = False

    def accepted(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if stopping:
            writer.close()
            return
        task = asyncio.create_task(relay(policy, reader, writer), name='browser-proxy-relay')
        connections.add(task)
        task.add_done_callback(connections.discard)
        task.add_done_callback(lambda _: writer.is_closing() or writer.transport.abort())

    server = await asyncio.start_server(accepted, '127.0.0.1', 0)
    try:
        yield server.sockets[0].getsockname()[1]
    finally:
        stopping = True
        server.close()
        for task in connections:
            task.cancel()
        with anyio.CancelScope(shield=True):
            await asyncio.gather(*connections, return_exceptions=True)


async def relay(policy: EgressPolicy, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Relay one connection if the policy allows its destination; anything malformed closes it unrelayed."""
    writers = [writer]
    try:
        with suppress(OSError, EOFError, ValueError):
            with anyio.fail_after(SETUP_TIMEOUT_SECONDS):
                host, port = await read_destination(reader, writer)
                destination = await open_destination(policy, host, port)
            if destination is None:
                writer.write(REPLY_NOT_ALLOWED)
                await writer.drain()
                return
            upstream_reader, upstream_writer = destination
            writers.append(upstream_writer)
            writer.write(REPLY_CONNECTED)
            await writer.drain()
            async with anyio.create_task_group() as tg:

                async def pipe(source: asyncio.StreamReader, sink: asyncio.StreamWriter) -> None:
                    with suppress(OSError):
                        while data := await source.read(READ_SIZE):
                            sink.write(data)
                            await sink.drain()
                    tg.cancel_scope.cancel()

                tg.start_soon(pipe, reader, upstream_writer)
                tg.start_soon(pipe, upstream_reader, writer)
    finally:
        # A transport with unsent data stays open until its peer reads it, which a stalled peer never does.
        for open_writer in writers:
            open_writer.close()
        unclosed = list(writers)
        try:
            with anyio.move_on_after(CLOSE_GRACE_SECONDS, shield=True):
                for open_writer in writers:
                    with suppress(OSError):
                        await open_writer.wait_closed()
                    unclosed.remove(open_writer)
        finally:
            # On Python 3.13 and older, aborting a transport whose close has finished raises `AttributeError`.
            for open_writer in unclosed:
                open_writer.transport.abort()


async def read_destination(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> tuple[str, int]:
    """Answer the SOCKS5 greeting and return the host and port asked for, raising `ValueError` on anything else."""
    version, method_count = await reader.readexactly(2)
    methods = await reader.readexactly(method_count)
    if version != 5 or 0 not in methods:
        raise ValueError('not a SOCKS5 client offering no authentication')
    writer.write(b'\x05\x00')
    await writer.drain()
    _, command, _, address_type = await reader.readexactly(4)
    if command != 1:
        raise ValueError('only CONNECT is supported')
    match address_type:
        case 1:
            host = str(IPv4Address(await reader.readexactly(4)))
        case 3:
            size = (await reader.readexactly(1))[0]
            host = (await reader.readexactly(size)).decode('ascii')
        case 4:
            host = str(IPv6Address(await reader.readexactly(16)))
        case _:
            raise ValueError('unknown address type')
    return host, int.from_bytes(await reader.readexactly(2))


async def open_destination(
    policy: EgressPolicy, host: str, port: int
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter] | None:
    """Connect to `host` if the policy allows it, or return `None`.

    A connection is asked about as a WebSocket, `data` to `ws://host:port/`, the one kind of connection whose whole
    policy a listener applies; `PRIVATE_ADDRESS_BLOCK` reads neither the kind nor the scheme. The lookup is made here
    rather than in `refusal_reason` so the connection goes to the addresses the policy checked. A host that holds a
    NUL or does not parse back to itself cannot be classified, so it is refused.
    """
    if '\0' in host:
        return None
    authority = f'[{host}]' if ':' in host else host
    request = EgressRequest(url=f'ws://{authority}:{port}/', kind='data', resource_type='websocket', top_level=False)
    parsed = url_host(request.url)
    if parsed is None or parsed != host.lower():
        return None
    needs_lookup = policy.needs_resolution(request)
    addresses = await resolve_host(parsed) if needs_lookup else None
    checked = replace(request, resolved_addresses=addresses or (), resolution_failed=needs_lookup and addresses is None)
    if policy.refuse(checked) is not None or (needs_lookup and not addresses):
        return None
    targets = addresses or (parsed,)
    for address in targets[:-1]:
        with suppress(OSError):
            return await asyncio.open_connection(address, port)
    return await asyncio.open_connection(targets[-1], port)
