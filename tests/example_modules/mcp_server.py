from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSessionT
from mcp.shared.context import LifespanContextT, RequestT

mcp = FastMCP('Pydantic AI MCP Server')


@mcp.tool()
async def echo_deps(ctx: Context[ServerSessionT, LifespanContextT, RequestT]) -> dict[str, Any]:
    """Echo the run context.

    Args:
        ctx: Context object containing request and session information.

    Returns:
        Dictionary with an echo message and the deps.
    """

    deps: Any = getattr(ctx.request_context.meta, 'deps')
    return {'echo': 'This is an echo message', 'deps': deps}


if __name__ == '__main__':
    mcp.run()
