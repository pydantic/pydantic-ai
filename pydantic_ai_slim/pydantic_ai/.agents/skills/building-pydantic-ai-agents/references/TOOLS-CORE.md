# Tools Core

Read this file when the user wants to add function tools, organize toolsets, connect MCP servers, or use explicit common search tools.

## Add Tools to an Agent

Use `@agent.tool_plain` for pure functions and `@agent.tool` for tools that need `RunContext`.

```python
import random

from pydantic_ai import Agent, RunContext

agent = Agent('google:gemini-3-flash-preview', deps_type=str)


@agent.tool_plain
def roll_dice() -> str:
    return str(random.randint(1, 6))


@agent.tool
def get_player_name(ctx: RunContext[str]) -> str:
    return ctx.deps
```

Use `Tool(fn)` when tools are defined outside the agent file or shared between agents.

## Choosing a Tool Registration Method

Default choices:

- `@agent.tool` when the tool needs deps, usage, retry count, or message history
- `@agent.tool_plain` when the tool is a plain function
- `Tool(...)` in `tools=[...]` when the tool should be reusable across agents
- `FunctionToolset` when multiple related tools should be managed as a group

## Organize or Restrict Which Tools an Agent Can Use

Use toolsets when the user has multiple related tools or wants cross-cutting behavior applied to a group.

Examples:

- `FunctionToolset` for a bundle of Python tools
- MCP servers, which are themselves toolsets
- wrapper toolsets for approval, deferred loading, or other cross-cutting behavior

## Access Usage Stats, Message History, or Retry Count in Tools

Route this to `@agent.tool` with `RunContext`.

Useful `RunContext` fields include:

- `ctx.deps`
- `ctx.usage`
- `ctx.messages`
- `ctx.retry`

## Use MCP Servers

Attach an MCP server via the `MCP` capability — it runs the MCP server locally by default and lets you opt into the model provider's native MCP support with `native=True`. Accepts a URL string, script path, `fastmcp.client.transports.*Transport`, or pre-built `fastmcp.Client`. See the [MCP capability docs](https://ai.pydantic.dev/capabilities/#mcp).

```python
from fastmcp.client.transports import StdioTransport

from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP

agent = Agent(
    'openai:gpt-5.2',
    capabilities=[MCP(StdioTransport(command='python', args=['mcp_server.py']))],
)


async def main():
    async with agent:
        result = await agent.run('What is the weather in Paris?')
        print(result.output)
```

For HTTP servers, pass the URL directly: `MCP(url='https://example.com/mcp')` (Streamable HTTP) or `MCP(url='https://example.com/sse')` (SSE).

When you need to manage the toolset lifecycle yourself, share an MCP server across multiple agents, or use FastMCP-specific configuration that doesn't fit the capability shape, use [`MCPToolset`](https://ai.pydantic.dev/mcp/client/) directly with the same inputs and pass it via `toolsets=[...]`.

## Search with DuckDuckGo, Tavily, or Exa

Use common tools when the user wants explicit search tools rather than provider-adaptive capabilities.

```python
from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

agent = Agent(
    'openai:gpt-5.2',
    tools=[duckduckgo_search_tool()],
    instructions='Search DuckDuckGo for the given query and return the results.',
)
```

Good default split:

- use `WebSearch()` capability when the user wants model-agnostic search with native fallback
- use `duckduckgo_search_tool()` / Tavily / Exa when the user explicitly wants those engines as tools
