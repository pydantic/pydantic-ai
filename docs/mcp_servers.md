# MCP Servers

**PydanticAI** supports integration with
[MCP (Model Control Protocol) Servers](https://modelcontextprotocol.io/introduction),
allowing you to extend agent capabilities through external services. This integration enables
dynamic tool discovery and remote execution.

## Install

To use MCP servers, you need to either install [`pydantic-ai`](install.md), or install
[`pydantic-ai-slim`](install.md#slim-install) with the `mcp` optional group:

```bash
pip/uv-add 'pydantic-ai-slim[mcp]'
```

!!! note
    MCP integration requires Python 3.10 or higher.

## Usage

The [MCPServer][pydantic_ai.mcp.MCPServer] must be used within an async context manager to ensure
proper initialization and cleanup of resources. You can use either an HTTP/SSE server or a
stdio-based server.

### HTTP/SSE Server

```python {title="basic_mcp_setup.py"}
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServer

async with MCPServer.sse(url="http://localhost:8000/sse") as mcp_server:
    agent = Agent("your-model", mcp_servers=[mcp_server])
    # Use the agent here
```

### Stdio Server

For stdio-based servers,

```python {title="stdio_mcp_setup.py"}
import asyncio

from pydantic_ai.agent import Agent
from pydantic_ai.mcp import MCPServer


async def main():
    async with MCPServer.stdio('python', ['-m', 'pydantic_ai.mcp']) as server:
        agent = Agent('openai:gpt-4o', mcp_servers=[server])
        result = await agent.run('Can you convert 30 degrees celsius to fahrenheit?')
    print(result.data)


asyncio.run(main())
```

### Multiple Servers

You can connect to multiple MCP servers simultaneously:

```python {title="multiple_mcp_servers.py"}
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServer

async with MCPServer.sse(url="http://localhost:8000") as local_server, \
         MCPServer.sse(url="https://remote-mcp.example.com") as remote_server:
    agent = Agent("your-model", mcp_servers=[local_server, remote_server])
    # Use the agent here
```
