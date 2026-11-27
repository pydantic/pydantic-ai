"""The MCP server the stdio handler tests spawn as `python -m tests.mcp_log_server`.

Its tool sends a log message, which reaches the client's `log_handler` from the connection rather than
from the tool call. It lives apart from `tests/mcp_server.py`, whose tool list recorded requests pin.
"""

from fastmcp.server import Context, FastMCP

mcp: FastMCP[None] = FastMCP('Pydantic AI MCP Log Server')


@mcp.tool()
async def log_text(text: str, ctx: Context) -> str:
    """Send `text` back to the client as an info log message."""
    await ctx.info(text)
    return text


if __name__ == '__main__':
    mcp.run()
