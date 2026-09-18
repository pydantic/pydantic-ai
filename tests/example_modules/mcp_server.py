"""The MCP server the documentation examples spawn.

Written against the `fastmcp` server API rather than `mcp.server.fastmcp`, which is the MCP SDK v1
server that SDK v2 removed; see the note in `tests/mcp_server.py`.
"""

from typing import Any

from fastmcp.server import Context, FastMCP

mcp: FastMCP[None] = FastMCP('Pydantic AI MCP Server')


@mcp.tool()
async def get_weather_forecast(location: str) -> str:
    """Get the weather forecast for a location."""
    return f'The weather in {location} is sunny and 26 degrees Celsius.'


@mcp.tool()
async def echo_deps(ctx: Context) -> dict[str, Any]:
    """Echo the run context.

    Args:
        ctx: Context object containing request and session information.

    Returns:
        Dictionary with an echo message and the deps.
    """

    deps: Any = getattr(getattr(ctx.request_context, 'meta', None), 'deps')
    return {'echo': 'This is an echo message', 'deps': deps}


if __name__ == '__main__':
    mcp.run()
