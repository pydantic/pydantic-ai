"""v2 form: MCPToolset(url) — SSE transport inferred from /sse suffix."""
from pydantic_ai.mcp import MCPToolset


def trigger():
    return MCPToolset('http://localhost:8000/sse')


if __name__ == '__main__':
    trigger()
