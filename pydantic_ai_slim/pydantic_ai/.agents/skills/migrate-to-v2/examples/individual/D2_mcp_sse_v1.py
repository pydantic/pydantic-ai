"""v1: MCPServerSSE."""
from pydantic_ai.mcp import MCPServerSSE


def trigger():
    # DEPRECATION: D2_mcp_sse
    return MCPServerSSE('http://localhost:8000/sse')


EXPECT = '`MCPServerSSE` is deprecated'

if __name__ == '__main__':
    trigger()
