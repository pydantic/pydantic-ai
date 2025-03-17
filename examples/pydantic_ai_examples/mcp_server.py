"""Simple MCP Server that can be used to test the MCP protocol.

Run with:

    uv run -m pydantic_ai_examples.mcp_server --transport <TRANSPORT>

TRANSPORT can be either `sse` or `stdio`.
"""

import argparse

from mcp.server.fastmcp import FastMCP

mcp = FastMCP('PydanticAI MCP Server', port=8005)


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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--transport', type=str, default='stdio', choices=('sse', 'stdio')
    )
    args = parser.parse_args()

    mcp.run(transport=args.transport)
