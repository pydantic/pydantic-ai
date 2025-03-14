"""Simple MCP Server that can be used to test the MCP protocol.

Run with:

    uv run -m pydantic_ai_examples.mcp_server
"""

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


if __name__ == '__main__':
    mcp.run()
