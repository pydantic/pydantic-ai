"""The MCP server the stdio handler tests spawn as `python -m tests.mcp_log_server`.

Its tools send a log message and a sampling request, which reach the client's `log_handler` and sampling
handler from the connection rather than from the tool call. It lives apart from `tests/mcp_server.py`,
whose tool list recorded requests pin.
"""

from fastmcp.server import Context, FastMCP
from mcp.types import SamplingMessage, TextContent

mcp: FastMCP[None] = FastMCP('Pydantic AI MCP Log Server')


@mcp.tool()
async def log_text(text: str, ctx: Context) -> str:
    """Send `text` back to the client as an info log message."""
    await ctx.info(text)
    return text


@mcp.tool()
async def sample_text(ctx: Context) -> str:
    """Ask the client's sampling model for a reply."""
    message = SamplingMessage(role='user', content=TextContent(type='text', text='Say hi'))
    # A legacy session's server-initiated sampling request, which the 2026-07-28 protocol deprecates.
    await ctx.session.create_message([message], max_tokens=5, related_request_id=ctx.request_id)  # pyright: ignore[reportDeprecated]
    return 'sampled'


if __name__ == '__main__':
    mcp.run()
