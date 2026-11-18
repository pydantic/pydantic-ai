"""A local MongoDB wire-protocol server that records the driver metadata of each connection handshake.

It speaks only OP_MSG: the workspace constrains `pymongo>=4.18.2`, which sends the `hello` handshake as OP_MSG.
"""

from __future__ import annotations

import asyncio
import re
import struct
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from importlib.metadata import version
from typing import TypeGuard

import pymongo
from bson import BSON, encode

from tests.harness.conftest import IsStr

_OP_MSG = 2013

HARNESS_DRIVER: dict[str, object] = {
    'name': IsStr(regex=r'PyMongo\|.*\|pydantic-ai-harness'),
    # pymongo >= 4.19 adds an empty version segment for each built-in `|c` / `|async` name segment.
    'version': IsStr(regex=rf'{re.escape(pymongo.version)}\|+{re.escape(version("pydantic-ai-harness"))}'),
}


def is_object_dict(value: object) -> TypeGuard[dict[object, object]]:
    return isinstance(value, dict)


@asynccontextmanager
async def handshake_server() -> AsyncGenerator[tuple[str, list[dict[object, object]]]]:
    """Yield a connection string and the list that collects each handshake's `client.driver` document.

    The server answers a handshake as a writable primary, `find` with an empty batch, and any other command with `ok`.
    """
    drivers: list[dict[object, object]] = []

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                length, request_id, _, opcode = struct.unpack('<iiii', await reader.readexactly(16))
                assert opcode == _OP_MSG
                # The command document follows 4 bytes of flag bits and a 1-byte section kind.
                command = BSON((await reader.readexactly(length - 16))[5:]).decode()
                client = command.get('client')
                if is_object_dict(client):
                    driver = client['driver']
                    assert is_object_dict(driver)
                    drivers.append(driver)
                    reply: dict[str, object] = {
                        'ok': 1,
                        'isWritablePrimary': True,
                        'minWireVersion': 0,
                        'maxWireVersion': 17,
                        'logicalSessionTimeoutMinutes': 30,
                    }
                elif 'find' in command:
                    namespace = f'{command["$db"]}.{command["find"]}'
                    reply = {'ok': 1, 'cursor': {'id': 0, 'ns': namespace, 'firstBatch': []}}
                else:
                    reply = {'ok': 1}
                body = struct.pack('<iB', 0, 0) + encode(reply)
                writer.write(struct.pack('<iiii', 16 + len(body), 1, request_id, _OP_MSG) + body)
                await writer.drain()
        except asyncio.IncompleteReadError:
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle, '127.0.0.1', 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f'mongodb://127.0.0.1:{port}/?directConnection=true&serverSelectionTimeoutMS=1000', drivers
    finally:
        server.close()
        await server.wait_closed()
