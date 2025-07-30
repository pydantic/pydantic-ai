# Model Context Protocol (MCP)

Pydantic AI supports [Model Context Protocol (MCP)](https://modelcontextprotocol.io) in three ways:

1. [Agents](../agents.md) act as an MCP Client, connecting to MCP servers to use their tools, [learn more …](client.md)
2. Agents can be used within MCP servers, [learn more …](server.md)
3. As part of Pydantic AI, we're building a number of MCP servers, [see below](#mcp-servers)

## What is MCP?

The Model Context Protocol is a standardized protocol that allow AI applications (including programmatic agents like Pydantic AI, coding agents like [cursor](https://www.cursor.com/), and desktop applications like [Claude Desktop](https://claude.ai/download)) to connect to external tools and services using a common interface.

As with other protocols, the dream of MCP is that a wide range of applications can speak to each other without the need for specific integrations.

There is a great list of MCP servers at [github.com/modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers).

Some examples of what this means:

- Pydantic AI could use a web search service implemented as an MCP server to implement a deep research agent
- Cursor could connect to the [Pydantic Logfire](https://github.com/pydantic/logfire-mcp) MCP server to search logs, traces and metrics to gain context while fixing a bug
- Pydantic AI, or any other MCP client could connect to our [Run Python](run-python.md) MCP server to run arbitrary Python code in a sandboxed environment

## MCP Servers

To add functionality to Pydantic AI while making it as widely usable as possible, we're implementing some functionality as MCP servers.

So far, we've only implemented one MCP server as part of Pydantic AI:

- [Run Python](run-python.md): A sandboxed Python interpreter that can run arbitrary code, with a focus on security and safety.
