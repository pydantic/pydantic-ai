"""v2 form: toolsets=[MCPToolset(...)]."""
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from fastmcp.client.transports import StdioTransport


def trigger():
    s = MCPToolset(StdioTransport(command='python', args=['-c', 'pass']))
    return Agent('openai-chat:gpt-4o', toolsets=[s])


if __name__ == '__main__':
    trigger()
