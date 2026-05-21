"""v2 form: MCPToolset + StdioTransport."""
from pydantic_ai.mcp import MCPToolset
from fastmcp.client.transports import StdioTransport


def trigger():
    return MCPToolset(StdioTransport(command='python', args=['-c', 'pass']))


if __name__ == '__main__':
    trigger()
