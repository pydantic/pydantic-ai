from mcp.server.fastmcp import Context, FastMCP

mcp = FastMCP('PydanticAI MCP Server')
log_level = 'unset'


@mcp.tool()
async def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit.

    Args:
        celsius: Temperature in Celsius

    Returns:
        Temperature in Fahrenheit
    """
    return (celsius * 9 / 5) + 32


@mcp.tool()
async def get_log_level(ctx: Context) -> str:  # type: ignore
    """Get the current log level.

    Returns:
        The current log level.
    """
    await ctx.info('this is a log message')
    return log_level


@mcp._mcp_server.set_logging_level()  # pyright: ignore[reportPrivateUsage]
async def set_logging_level(level: str) -> None:
    global log_level
    log_level = level


if __name__ == '__main__':
    mcp.run()
