"""v1: mcp_servers= ctor kwarg."""
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio


def trigger():
    # DEPRECATION: A7_mcp_servers
    s = MCPServerStdio('python', args=['-c', 'pass'])
    return Agent('openai-chat:gpt-4o', mcp_servers=[s])


EXPECT = '`mcp_servers` is deprecated, use `toolsets` instead'

if __name__ == '__main__':
    trigger()
