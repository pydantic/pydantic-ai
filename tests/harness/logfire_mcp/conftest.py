from __future__ import annotations

import subprocess
import sys
from collections.abc import AsyncIterator

import anyio
import pytest

# The `mcp` SDK's server, not FastMCP's: the slim install has only the FastMCP client.
_WHOAMI_SERVER = """
import asyncio, socket, uvicorn
from mcp.server.fastmcp import Context, FastMCP

mcp = FastMCP('whoami', stateless_http=True)

@mcp.tool()
async def whoami(ctx: Context) -> str:
    await asyncio.sleep(0.05)  # keep concurrent runs overlapping
    return ctx.request_context.request.headers['authorization']

server_socket = socket.create_server(('127.0.0.1', 0))
print(server_socket.getsockname()[1], flush=True)
uvicorn.run(mcp.streamable_http_app(), fd=server_socket.fileno(), log_level='warning')
"""


@pytest.fixture
async def whoami_url() -> AsyncIterator[str]:
    """A streamable HTTP MCP server whose tool reports the caller's `Authorization` header."""
    async with await anyio.open_process([sys.executable, '-c', _WHOAMI_SERVER], stdout=subprocess.PIPE) as process:
        assert process.stdout is not None
        port = int((await process.stdout.receive()).decode().strip())
        yield f'http://127.0.0.1:{port}/mcp'
        process.terminate()
