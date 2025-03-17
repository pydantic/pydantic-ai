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

## Running MCP Servers

Before diving into how to use MCP servers with PydanticAI, let's look at how to run MCP servers
with different transports.

To run MCP servers, you'll need to install the MCP CLI package:

```bash
pip/uv-add 'mcp[cli]'
```

Here's a simple MCP server that provides a temperature conversion tool. We will later assume this is the server we connect to from our agent:

```python {title="temperature_mcp_server.py" py="3.10"}
from mcp.server.fastmcp import FastMCP

mcp = FastMCP('Temperature Conversion Server')


@mcp.tool()
async def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit.

    Args:
        celsius: Temperature in Celsius
    """
    return (celsius * 9 / 5) + 32


if __name__ == '__main__':
    mcp.run('stdio')  # (1)!
```

1. Run with stdio transport (for subprocess communication).

The same server can be run with [SSE transport](https://modelcontextprotocol.io/docs/concepts/transports#server-sent-events-sse)
for HTTP communication:

```python {title="temperature_mcp_server_sse.py" py="3.10"}
from temperature_mcp_server import mcp

if __name__ == '__main__':
    mcp.run('sse', port=8000)  # (1)!
```

1. Run with SSE transport on port 8000.

## Usage

PydanticAI comes with two ways to connect to MCP servers:

- [`MCPServerSSE`][pydantic_ai.mcp.MCPServerSSE] which connects to an MCP server using the [HTTP SSE](https://modelcontextprotocol.io/docs/concepts/transports#server-sent-events-sse) transport
- [`MCPServerStdio`][pydantic_ai.mcp.MCPServerStdio] which runs the server as a subprocess and connects to it using the [stdio](https://modelcontextprotocol.io/docs/concepts/transports#standard-input%2Foutput-stdio) transport

Examples of both are shown below.

### MCP Remote Server

You can have a MCP server running on a remote server. In this case, you'd use the
[`MCPServerSSE`][pydantic_ai.mcp.MCPServerSSE] class:

```python {title="mcp_remote_server.py" py="3.10"}
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerSSE

server = MCPServerSSE(url='http://localhost:8005/sse')
agent = Agent('openai:gpt-4o', mcp_servers=[server])


async def main():
    async with agent.run_mcp_servers():
        result = await agent.run('Can you convert 30 degrees celsius to fahrenheit?')
    print(result.data)
    #> 30 degrees Celsius is equal to 86 degrees Fahrenheit.
```

This will connect to the MCP server at the given URL and use the
[SSE transport](https://modelcontextprotocol.io/docs/concepts/transports#server-sent-events-sse).

### MCP Subprocess Server

We also have a subprocess-based server that can be used to run the MCP server in a separate process.
In this case, you'd use the [`MCPServerStdio`][pydantic_ai.mcp.MCPServerStdio] class,
when using `MCPServerStdio` you need to run the server with the [`run_mcp_servers`][pydantic_ai.Agent.run_mcp_servers]
context manager before running the server.

```python {title="mcp_subprocess_server.py" py="3.10"}
from pydantic_ai.agent import Agent
from pydantic_ai.mcp import MCPServerStdio

server = MCPServerStdio('python', ['-m', 'pydantic_ai_examples.mcp_server'])
agent = Agent('openai:gpt-4o', mcp_servers=[server])


async def main():
    async with agent.run_mcp_servers():
        result = await agent.run('Can you convert 30 degrees celsius to fahrenheit?')
    print(result.data)
    #> 30 degrees Celsius is equal to 86 degrees Fahrenheit.
```

This will start the MCP server in a separate process and connect to it using the
[stdio transport](https://modelcontextprotocol.io/docs/concepts/transports#standard-input%2Foutput-stdio).
