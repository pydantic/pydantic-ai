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
agent = Agent('openai:gpt-4o', toolsets=[server])

async with agent:
    result = await agent.run('Use the tools available.')
```

### MCPServerSSE

Connect using Server-Sent Events transport:

```python
from pydantic_ai.mcp import MCPServerSSE

server = MCPServerSSE('http://localhost:8080/sse')
agent = Agent('openai:gpt-4o', toolsets=[server])
```

### MCPServerStdio

Connect to a subprocess-based MCP server:

```python
from pydantic_ai.mcp import MCPServerStdio

server = MCPServerStdio('npx', args=['-y', '@modelcontextprotocol/server-filesystem', '/tmp'])
agent = Agent('openai:gpt-4o', toolsets=[server])
```

## Important: Context Manager

MCP servers must be used within an async context manager (`async with agent:`):

```python
agent = Agent('openai:gpt-4o', toolsets=[server])

# REQUIRED: enter the agent context to start MCP servers
async with agent:
    result = await agent.run('prompt')
```

## load_mcp_servers()

Load MCP server configurations from a JSON file:

```python
from pydantic_ai.mcp import load_mcp_servers

servers = load_mcp_servers('mcp_config.json')
agent = Agent('openai:gpt-4o', toolsets=servers)
```

## FastMCP — Create MCP Servers

Use the FastMCP toolset to create and use MCP servers inline:

```python
from pydantic_ai import Agent
from pydantic_ai.toolsets.fastmcp import FastMCPToolset

# Create a FastMCP server toolset
toolset = FastMCPToolset('my-server')

@toolset.tool
def greet(name: str) -> str:
    """Greet someone."""
    return f'Hello, {name}!'

agent = Agent('openai:gpt-4o', toolsets=[toolset])
```

Install extra: `pydantic-ai-slim[fastmcp]`

## MCP Resources

Access resources provided by MCP servers:

```python
from pydantic_ai.mcp import Resource, ResourceTemplate

# Resources are exposed as tool capabilities by the MCP server
```

## MCP Sampling

MCP servers can request LLM completions from the client:

```python
# Sampling is handled automatically when the MCP server requests it
# The agent's model is used for sampling requests
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
| `load_mcp_servers` | `pydantic_ai.mcp.load_mcp_servers` | Load from config file |
| `MCPError` | `pydantic_ai.mcp.MCPError` | MCP error type |
| `Resource` | `pydantic_ai.mcp.Resource` | MCP resource |
| `ResourceTemplate` | `pydantic_ai.mcp.ResourceTemplate` | MCP resource template |
