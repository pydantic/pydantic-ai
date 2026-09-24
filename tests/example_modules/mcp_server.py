from typing import Any

from fastmcp import Context, FastMCP

mcp = FastMCP('Pydantic AI MCP Server')


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
    assert ctx.request_context is not None
    deps: Any = getattr(ctx.request_context.meta, 'deps')
    return {'echo': 'This is an echo message', 'deps': deps}


if __name__ == '__main__':
    mcp.run()
