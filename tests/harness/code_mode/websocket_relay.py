"""Bridge each WebSocket connection to a fresh `monty subprocess` child.

The WebSocket transport sends one binary protocol frame per message. The stdio
worker uses the same frame with a four-byte little-endian length prefix, so this
relay only adds and removes that prefix.
"""

from __future__ import annotations

import argparse
import asyncio
import struct

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

_LENGTH_PREFIX = struct.Struct('<I')


async def _bridge_connection(websocket: ServerConnection, monty_bin: str) -> None:  # pragma: no cover
    """Shuttle frames until either peer closes, then reap the worker."""
    child = await asyncio.create_subprocess_exec(
        monty_bin,
        'subprocess',
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
    )
    stdin = child.stdin
    stdout = child.stdout
    assert stdin is not None
    assert stdout is not None

    async def websocket_to_child() -> None:
        async for message in websocket:
            body = message.encode() if isinstance(message, str) else message
            stdin.write(_LENGTH_PREFIX.pack(len(body)) + body)
            await stdin.drain()
        stdin.close()

    async def child_to_websocket() -> None:
        while True:
            (length,) = _LENGTH_PREFIX.unpack(await stdout.readexactly(_LENGTH_PREFIX.size))
            await websocket.send(await stdout.readexactly(length))

    websocket_task = asyncio.create_task(websocket_to_child())
    child_task = asyncio.create_task(child_to_websocket())
    try:
        done, _ = await asyncio.wait(
            {websocket_task, child_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in done:
            try:
                task.result()
            except (asyncio.IncompleteReadError, ConnectionError, ConnectionClosed):
                pass
    finally:
        for task in (websocket_task, child_task):
            task.cancel()
        await asyncio.gather(websocket_task, child_task, return_exceptions=True)
        if child.returncode is None:
            child.kill()
        await child.wait()


async def _serve_relay(host: str, port: int, monty_bin: str) -> None:  # pragma: no cover
    """Print the bound URL after startup and serve until terminated."""

    async def handler(websocket: ServerConnection) -> None:
        await _bridge_connection(websocket, monty_bin)

    async with serve(handler, host, port, max_size=None) as server:
        bound_host, bound_port = next(iter(server.sockets)).getsockname()[:2]
        authority = f'[{bound_host}]' if ':' in bound_host else bound_host
        print(f'ws://{authority}:{bound_port}', flush=True)
        await asyncio.get_running_loop().create_future()


def main() -> None:  # pragma: no cover
    """Parse the subprocess fixture arguments and run the relay."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--monty-bin', required=True)
    args = parser.parse_args()
    try:
        asyncio.run(_serve_relay(args.host, args.port, args.monty_bin))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':  # pragma: no cover
    main()
