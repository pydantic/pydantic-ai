from mcp.server.fastmcp import FastMCP

mcp = FastMCP('PydanticAI MCP Server')


@mcp.tool()
async def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit.

    Args:
        celsius: Temperature in Celsius

    Returns:
        Temperature in Fahrenheit
    """
    return (celsius * 9 / 5) + 32


mcp.run()
