# MCP Servers

PydanticAI supports integration with MCP (Model Control Protocol) Servers, allowing you to extend agent capabilities
through external services. This integration enables dynamic tool discovery and remote execution.

## Install

To use MCP servers, you need to either install [`pydantic-ai`](install.md), or install
[`pydantic-ai-slim`](install.md#slim-install) with the `mcp` optional group:

```bash
pip/uv-add 'pydantic-ai-slim[mcp]'
```

!!! note
    MCP integration requires Python 3.10 or higher.

## Usage

The [MCPServer][pydantic_ai.mcp.MCPServer] must be used within a context manager to ensure proper initialization and cleanup of
resources. You can use either an HTTP/SSE server or a stdio-based server.

### HTTP/SSE Server

```python {title="basic_mcp_setup.py"}
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServer

async with MCPServer(config={'url': "http://localhost:8000"}) as mcp_server:
    agent = Agent("your-model", mcp_servers={"local": mcp_server})
    # Use the agent here
```

### Stdio Server

For stdio-based servers, you can use the [`StdioMCPServerConfig`][pydantic_ai.mcp.StdioMCPServerConfig] class to configure the server.

```python {title="stdio_mcp_setup.py"}
import os
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServer, StdioMCPServerConfig

# Configure the stdio server
config = StdioMCPServerConfig(
    command='/path/to/your/python',  # Path to Python interpreter
    args=['-m', 'your_mcp_module'],  # Module to run
    env={
        'API_TOKEN': os.getenv('API_TOKEN'),
    },
)

async def main():
    async with MCPServer(config=config) as server:
        agent = Agent(
            'your-model',
            mcp_servers={'your_service': server},
            system_prompt="Your system prompt here"
        )
        result = await agent.run('Your prompt here')
        print(result.data)

if __name__ == '__main__':
    asyncio.run(main())
```

### Multiple Servers

You can connect to multiple MCP servers simultaneously:

```python {title="multiple_mcp_servers.py"}
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServer

async with MCPServer(url="http://localhost:8000") as local_server, \
         MCPServer(config={'url': "https://remote-mcp.example.com"}) as remote_server:
    agent = Agent("your-model", mcp_servers={"local": local_server, "remote": remote_server})
    # Use the agent here
```
