# MCP (Model Context Protocol) Reference

Source: `pydantic_ai_slim/pydantic_ai/mcp.py`

## Overview

PydanticAI supports MCP servers as toolsets, allowing agents to use tools provided by external MCP servers.

## MCP Server Types

### MCPServerStreamableHTTP (recommended)

Connect to an HTTP-based MCP server using the Streamable HTTP transport:

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP

server = MCPServerStreamableHTTP('http://localhost:8080/mcp')
agent = Agent('openai:gpt-5', toolsets=[server])

async with agent:
    result = await agent.run('Use the tools available.')
```

### MCPServerSSE

Connect using Server-Sent Events transport:

```python
from pydantic_ai.mcp import MCPServerSSE

server = MCPServerSSE('http://localhost:8080/sse')
agent = Agent('openai:gpt-5', toolsets=[server])
```

### MCPServerStdio

Connect to a subprocess-based MCP server:

```python
from pydantic_ai.mcp import MCPServerStdio

server = MCPServerStdio('npx', args=['-y', '@modelcontextprotocol/server-filesystem', '/tmp'])
agent = Agent('openai:gpt-5', toolsets=[server])
```

## Important: Context Manager

MCP servers must be used within an async context manager (`async with agent:`):

```python
agent = Agent('openai:gpt-5', toolsets=[server])

# REQUIRED: enter the agent context to start MCP servers
async with agent:
    result = await agent.run('prompt')
```

## Client Identification

Identify your application to MCP servers during connection. Useful for server logs, custom behavior, and debugging:

```python
from mcp import types as mcp_types
from pydantic_ai.mcp import MCPServerSSE

server = MCPServerSSE(
    'http://localhost:3001/sse',
    client_info=mcp_types.Implementation(
        name='MyApplication',
        version='2.1.0',
    ),
)
```

All MCP server types (`MCPServerStdio`, `MCPServerStreamableHTTP`, `MCPServerSSE`) support `client_info`.

## Elicitation — Interactive Input from Servers

Elicitation allows MCP servers to request structured input from the client during a session. Instead of requiring all information upfront, servers can ask for it as needed.

### How Elicitation Works

1. User makes a request (e.g., "Book a table at that Italian place")
2. Server needs more information and sends an `ElicitRequest`
3. Client presents the request to the user
4. User provides response, declines, or cancels
5. Client sends `ElicitResult` back to the server
6. Server continues processing with the structured data

### Setting Up Elicitation

Provide an `elicitation_callback` when creating your MCP server:

```python
from mcp.client.session import ClientSession
from mcp.shared.context import RequestContext
from mcp.types import ElicitRequestParams, ElicitResult

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio


async def handle_elicitation(
    context: RequestContext[ClientSession, Any, Any],
    params: ElicitRequestParams,
) -> ElicitResult:
    """Handle elicitation requests from MCP server."""
    print(f'\n{params.message}')

    if not params.requestedSchema:
        response = input('Response: ')
        return ElicitResult(action='accept', content={'response': response})

    # Collect data for each field in the schema
    properties = params.requestedSchema['properties']
    data = {}

    for field, info in properties.items():
        description = info.get('description', field)
        value = input(f'{description}: ')

        # Convert to proper type
        if info.get('type') == 'integer':
            data[field] = int(value)
        else:
            data[field] = value

    confirm = input('\nConfirm? (y/n/c): ').lower()

    if confirm == 'y':
        return ElicitResult(action='accept', content=data)
    elif confirm == 'n':
        return ElicitResult(action='decline')
    else:
        return ElicitResult(action='cancel')


server = MCPServerStdio(
    'python', args=['restaurant_server.py'],
    elicitation_callback=handle_elicitation
)

agent = Agent('openai:gpt-5', toolsets=[server])
```

### Elicitation Actions

| Action | Description |
|--------|-------------|
| `accept` | User provided the requested data |
| `decline` | User declined to provide the data |
| `cancel` | User cancelled the entire operation |

### Supported Schema Types

MCP elicitation supports: `string`, `number`, `boolean`, and `enum` types with flat object structures only.

## MCP Sampling

MCP servers can request LLM completions from the client (proxy LLM calls through the client):

```python
from pydantic_ai.mcp import MCPServerStdio

server = MCPServerStdio(
    'python',
    args=['my_server.py'],
    allow_sampling=True,  # Enable sampling (default: True)
)
```

Set `allow_sampling=False` to disable sampling requests from servers.

## load_mcp_servers()

Load MCP server configurations from a JSON file:

```python
from pydantic_ai.mcp import load_mcp_servers

servers = load_mcp_servers('mcp_config.json')
agent = Agent('openai:gpt-5', toolsets=servers)
```

## FastMCPToolset — Simplified MCP Client

A higher-level MCP client based on FastMCP. Supports multiple connection types:

```python
from pydantic_ai import Agent
from pydantic_ai.toolsets.fastmcp import FastMCPToolset

# From a FastMCP Server (in-process, no network)
from fastmcp import FastMCP
fastmcp_server = FastMCP('my_server')
toolset = FastMCPToolset(fastmcp_server)

# From a Streamable HTTP URL
toolset = FastMCPToolset('http://localhost:8000/mcp')

# From an HTTP SSE URL
toolset = FastMCPToolset('http://localhost:8000/sse')

# From a Python script
toolset = FastMCPToolset('my_server.py')

# From a Node.js script
toolset = FastMCPToolset('my_server.js')

# From JSON MCP configuration
toolset = FastMCPToolset({
    'mcpServers': {
        'time_server': {'command': 'uvx', 'args': ['mcp-run-python', 'stdio']},
        'weather_server': {'command': 'python', 'args': ['weather_server.py']},
    }
})

agent = Agent('openai:gpt-5', toolsets=[toolset])
```

Install extra: `pydantic-ai-slim[fastmcp]`

Note: FastMCPToolset does not yet support elicitation or sampling.

## MCP Resources

Access resources provided by MCP servers:

```python
from pydantic_ai.mcp import Resource, ResourceTemplate

# Resources are exposed as tool capabilities by the MCP server
```

## MCPError

Raised when an MCP server returns an error:

```python
from pydantic_ai.mcp import MCPError

# MCPError has: message, code, data
```

## Key Types

| Type | Import | Description |
|------|--------|-------------|
| `MCPServerStreamableHTTP` | `pydantic_ai.mcp.MCPServerStreamableHTTP` | HTTP MCP server |
| `MCPServerSSE` | `pydantic_ai.mcp.MCPServerSSE` | SSE MCP server |
| `MCPServerStdio` | `pydantic_ai.mcp.MCPServerStdio` | Stdio MCP server |
| `FastMCPToolset` | `pydantic_ai.toolsets.fastmcp.FastMCPToolset` | FastMCP-based client |
| `load_mcp_servers` | `pydantic_ai.mcp.load_mcp_servers` | Load from config file |
| `MCPError` | `pydantic_ai.mcp.MCPError` | MCP error type |
| `Resource` | `pydantic_ai.mcp.Resource` | MCP resource |
| `ResourceTemplate` | `pydantic_ai.mcp.ResourceTemplate` | MCP resource template |
