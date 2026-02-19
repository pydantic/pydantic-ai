# Model Context Protocol (MCP)

Pydantic AI supports [Model Context Protocol (MCP)](https://modelcontextprotocol.io) in multiple ways:

1. [Agents](../agent.md) can connect to MCP servers and use their tools using three different methods:
    1. Pydantic AI can act as an MCP client and connect directly to local and remote MCP servers. [Learn more](client.md) about [`MCPServer`][pydantic_ai.mcp.MCPServer].
    2. Pydantic AI can use the [FastMCP Client](https://gofastmcp.com/clients/client/) to connect to local and remote MCP servers, whether or not they're built using [FastMCP Server](https://gofastmcp.com/servers). [Learn more](fastmcp-client.md) about [`FastMCPToolset`][pydantic_ai.toolsets.fastmcp.FastMCPToolset].
    3. Some model providers can themselves connect to remote MCP servers using a "built-in tool". [Learn more](../builtin-tools.md#mcp-server-tool) about [`MCPServerTool`][pydantic_ai.builtin_tools.MCPServerTool].
2. Agents can be used within MCP servers. [Learn more](server.md)

## What is MCP?

The Model Context Protocol is a standardized protocol that allow AI applications (including programmatic agents like Pydantic AI, coding agents like [cursor](https://www.cursor.com/), and desktop applications like [Claude Desktop](https://claude.ai/download)) to connect to external tools and services using a common interface.

As with other protocols, the dream of MCP is that a wide range of applications can speak to each other without the need for specific integrations.

There is a great list of MCP servers at [github.com/modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers).

Some examples of what this means:

- Pydantic AI could use a web search service implemented as an MCP server to implement a deep research agent
- Cursor could connect to the [Pydantic Logfire](https://github.com/pydantic/logfire-mcp) MCP server to search logs, traces and metrics to gain context while fixing a bug
- Pydantic AI, or any other MCP client could connect to our [Run Python](https://github.com/pydantic/mcp-run-python) MCP server to run arbitrary Python code in a sandboxed environment
