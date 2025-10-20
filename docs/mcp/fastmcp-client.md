# FastMCP

Pydantic AI can also use a [FastMCP Client](https://gofastmcp.com/clients/) to connect to local and remote MCP servers. FastMCP is a higher-level MCP Client framework that makes building and using MCP servers easier for humans, it supports additional capabilities on top of the MCP specification like [Tool Transformation](https://gofastmcp.com/patterns/tool-transformation), [oauth](https://gofastmcp.com/clients/auth/oauth) and more!

### FastMCP Tools {#fastmcp-tools}

The [FastMCP](https://fastmcp.dev) Client can also be used with Pydantic AI with the provided [`FastMCPToolset`][pydantic_ai.toolsets.fastmcp.FastMCPToolset] [toolset](toolsets.md).

To use the `FastMCPToolset`, you will need to install `pydantic-ai-slim[fastmcp]`.

A FastMCP Toolset can be created from:
- A FastMCP Client: `FastMCPToolset(client=Client(...))`
- A FastMCP Transport: `FastMCPToolset(StdioTransport(command='uv', args=['run', 'mcp-run-python', 'stdio']))`
- A FastMCP Server: `FastMCPToolset(FastMCP('my_server'))`
- An HTTP URL: `FastMCPToolset('http://localhost:8000/mcp')`
- An SSE URL: `FastMCPToolset('http://localhost:8000/sse')`
- A Python Script: `FastMCPToolset('my_server.py')`
- A Node.js Script: `FastMCPToolset('my_server.js')`
- A JSON MCP Configuration: `FastMCPToolset({'mcpServers': {'my_server': {'command': 'python', 'args': ['-c', 'print("test")']}}})`

Connecting your agent to an HTTP MCP Server is as simple as:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.toolsets.fastmcp import FastMCPToolset

toolset = FastMCPToolset('http://localhost:8000/mcp')

agent = Agent('openai:gpt-5', toolsets=[toolset])
```

You can also create a toolset from a JSON MCP Configuration. FastMCP supports additional capabilities on top of the MCP specification, like Tool Transformation in the MCP configuration that you can take advantage of with the `FastMCPToolset`.

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.toolsets.fastmcp import FastMCPToolset

mcp_config = {
    'mcpServers': {
        'time_mcp_server': {
            'command': 'uvx',
            'args': ['mcp-server-time']
        }
    }
}

toolset = FastMCPToolset(mcp_config)

agent = Agent('openai:gpt-5', toolsets=[toolset])
```

Toolsets can also be created from a FastMCP Server:

```python {test="skip"}
from fastmcp import FastMCP

from pydantic_ai import Agent
from pydantic_ai.toolsets.fastmcp import FastMCPToolset

fastmcp_server = FastMCP('my_server')
@fastmcp_server.tool()
async def my_tool(a: int, b: int) -> int:
    return a + b

toolset = FastMCPToolset(fastmcp_server)

agent = Agent('openai:gpt-5', toolsets=[toolset])
```
