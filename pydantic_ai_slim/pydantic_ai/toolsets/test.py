from pydantic_ai import Agent
from pydantic_ai.toolsets.fastmcp import FastMCPToolset
from fastmcp.client.transports import StdioTransport

server = StdioTransport(  # (1)!
    command='uv', args=['run', 'mcp-run-python', 'stdio']
)
agent = Agent('openai:gpt-4o', toolsets=[FastMCPToolset(mcp=server)])

async def main():
    async with agent:  # (2)!