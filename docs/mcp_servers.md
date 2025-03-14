# MCP Servers

**PydanticAI** supports integration with
[MCP (Model Control Protocol) Servers](https://modelcontextprotocol.io/introduction),
allowing you to extend agent capabilities through external services. This integration enables
dynamic tool discovery.

## Install

To use MCP servers, you need to either install [`pydantic-ai`](install.md), or install
[`pydantic-ai-slim`](install.md#slim-install) with the `mcp` optional group:

```bash
pip/uv-add 'pydantic-ai-slim[mcp]'
```

!!! note
    MCP integration requires Python 3.10 or higher.

## Usage

PydanticAI comes with two ways to connect to MCP servers:

- [`MCPRemoteServer`][pydantic_ai.mcp.MCPRemoteServer] which connects to an MCP server using the [HTTP SSE](TODO) transport
- [`MCPSubprocessServer`][pydantic_ai.mcp.MCPSubprocessServer] which runs the server as a subprocess and connects to it using the [stdio](TODO) transport

Examples of both are shown below.

### MCP Remote Server

You can have a MCP server running on a remote server. In this case, you'd use the
[`MCPRemoteServer`][pydantic_ai.mcp.MCPRemoteServer] class:

```python {title="basic_mcp_setup.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPRemoteServer

server = MCPRemoteServer(url='http://localhost:8000/sse')
agent = Agent('openai:gpt-4o', mcp_servers=[server])


async def main():
    async with agent.run_mcp_servers():
        result = await agent.run('Can you convert 30 degrees celsius to fahrenheit?')
    print(result.data)
    #> '30 degrees Celsius is equal to 86 degrees Fahrenheit.'
```

This will connect to the MCP server at the given URL and use the SSE transport.

### MCP Subprocess Server

We also have a subprocess-based server that can be used to run the MCP server in a separate process.
In this case, you'd use the [`MCPSubprocessServer`][pydantic_ai.mcp.MCPSubprocessServer] class,
when using `MCPSubprocessServer` you need to run the server with the [`run_mcp_servers`][pydantic_ai.Agent.run_mcp_servers]
context manager before running the server.

```python {title="stdio_mcp_setup.py"}
from pydantic_ai.agent import Agent
from pydantic_ai.mcp import MCPSubprocessServer

server = MCPSubprocessServer('python', ['-m', 'pydantic_ai_examples.mcp_server'])
agent = Agent('openai:gpt-4o', mcp_servers=[server])


async def main():
    async with agent.run_mcp_servers():
        result = await agent.run('Can you convert 30 degrees celsius to fahrenheit?')
    print(result.data)
    #> 30 degrees Celsius is equal to 86 degrees Fahrenheit.
```

This will start the MCP server in a separate process and connect to it using the stdio transport.
