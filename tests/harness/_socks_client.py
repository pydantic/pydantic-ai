"""Speak SOCKS5 to the browser capabilities' loopback proxy the way Chromium does, and give it a peer to reach."""

from __future__ import annotations

import asyncio
import ipaddress
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass

import pytest

PUBLIC_ADDRESS = '93.184.216.34'
"""What the suites' stub resolver answers with: an address the private-address block lets through."""


def ipv4(host: str, port: int) -> bytes:
    return b'\x01' + ipaddress.IPv4Address(host).packed + port.to_bytes(2)


def ipv6(host: str, port: int) -> bytes:
    return b'\x04' + ipaddress.IPv6Address(host).packed + port.to_bytes(2)


def name(host: str, port: int) -> bytes:
    return b'\x03' + bytes([len(host)]) + host.encode() + port.to_bytes(2)


async def socks_connect(
    port: int, destination: bytes, *, greeting: bytes = b'\x05\x01\x00', command: int = 1
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter, bytes]:
    """Speak SOCKS5 to the proxy as Chrome does; the reply is empty when the proxy closes without answering."""
    reader, writer = await asyncio.open_connection('127.0.0.1', port)
    return reader, writer, await socks_handshake(reader, writer, destination, greeting=greeting, command=command)


async def socks_handshake(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    destination: bytes,
    *,
    greeting: bytes = b'\x05\x01\x00',
    command: int = 1,
) -> bytes:
    writer.write(greeting)
    await writer.drain()
    if await reader.read(2) != b'\x05\x00':
        return b''
    writer.write(bytes([5, command, 0]) + destination)
    await writer.drain()
    return await reader.read(10)


@dataclass
class EchoServer:
    port: int
    connections: list[asyncio.StreamWriter]


@asynccontextmanager
async def serve_echo() -> AsyncGenerator[EchoServer]:
    """A loopback server that sends back what it reads, and records each connection it accepts."""
    connections: list[asyncio.StreamWriter] = []

    async def echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        connections.append(writer)
        with suppress(ConnectionError):
            while data := await reader.read(1024):
                writer.write(data)
                await writer.drain()
        writer.close()

    server = await asyncio.start_server(echo, '127.0.0.1', 0)
    yield EchoServer(server.sockets[0].getsockname()[1], connections)
    server.close()
    await server.wait_closed()


def dial_public_on_loopback(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Send connections to `PUBLIC_ADDRESS` to loopback instead, and return each address dialed that is not loopback.

    A test cannot serve the public address the private-address block admits, so loopback stands in for it. Loopback
    dials are left out of the record because the test's own connections to the proxy are among them.
    """
    dialed: list[str] = []
    real_open_connection = asyncio.open_connection

    async def dial(host: str, port: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        if host != '127.0.0.1':
            dialed.append(host)
        return await real_open_connection('127.0.0.1' if host == PUBLIC_ADDRESS else host, port)

    monkeypatch.setattr(asyncio, 'open_connection', dial)
    return dialed
