"""v1: MCPServerStdio."""
from pydantic_ai.mcp import MCPServerStdio


def trigger():
    # DEPRECATION: D1_mcp_stdio
    return MCPServerStdio('python', args=['-c', 'pass'])


EXPECT = '`MCPServerStdio` is deprecated'

if __name__ == '__main__':
    trigger()
