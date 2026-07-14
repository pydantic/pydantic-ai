# Model Context Protocol (MCP)

Pydantic AI supports [Model Context Protocol (MCP)](https://modelcontextprotocol.io) in multiple ways:

1. [Agents](../agent.md) can connect to MCP servers and use their tools — see [Use MCP servers](#use-mcp-servers) below.
2. Agents can be used within MCP servers. [Learn more](server.md)

## What is MCP?

The Model Context Protocol is a standardized protocol that allows AI applications (including programmatic agents like Pydantic AI, coding agents like [cursor](https://www.cursor.com/), and desktop applications like [Claude Desktop](https://claude.ai/download)) to connect to external tools and services using a common interface.

As with other protocols, the dream of MCP is that a wide range of applications can speak to each other without the need for specific integrations.

There is a great list of MCP servers at [github.com/modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers).

Some examples of what this means:

- Pydantic AI could use a web search service implemented as an MCP server to implement a deep research agent
- Cursor could connect to the [Pydantic Logfire](https://github.com/pydantic/logfire-mcp) MCP server to search logs, traces and metrics to gain context while fixing a bug
- Pydantic AI, or any other MCP client could connect to our [Run Python](https://github.com/pydantic/mcp-run-python) MCP server to run arbitrary Python code in a sandboxed environment

## Use MCP servers

The recommended way to give an agent access to an MCP server is the [`MCP` capability](../capabilities.md#mcp). It runs the MCP server locally by default — keeping credentials, hooks, and tracing under your control — and lets you opt into the model provider's [native MCP support](../native-tools.md#mcp-server-tool) with a single `native=True` flag, so the same agent works across providers without code changes:

```python {title="mcp_capability.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP

agent = Agent(
    'openai:gpt-5.2',
    capabilities=[
        # Runs the MCP server locally by default
        MCP(url='https://mcp.example.com/api'),

        # Opt into native MCP — falls back to local if the model doesn't support it
        MCP(url='https://mcp.example.com/other', native=True),
    ],
)
```

Pass a URL as the first argument to enable both the local fallback and (with `native=True`) provider-native MCP. On the local side, `local=` accepts any [`MCPToolset`][pydantic_ai.mcp.MCPToolset] input — a URL, FastMCP transport, pre-built `fastmcp.Client`, in-process `FastMCP` server, or local script path. See the [capability documentation](../capabilities.md#mcp) for the full set of inputs and configuration options.

For lower-level access — managing the toolset lifecycle directly, sharing one MCP server across multiple agents, or passing advanced transport / client configuration that doesn't fit the capability shape — use `MCPToolset` directly via `toolsets=[...]`. See the [MCP client documentation](client.md) for details.

If you only need the model provider's native MCP support without a local fallback, you can use [`MCPServerTool`][pydantic_ai.native_tools.MCPServerTool] as a [native tool](../native-tools.md#mcp-server-tool) directly.

For building MCP servers with Pydantic AI agents, see the [MCP server documentation](server.md).
