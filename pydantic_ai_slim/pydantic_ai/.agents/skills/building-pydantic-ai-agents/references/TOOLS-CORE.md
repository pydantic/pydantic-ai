# Tools Core

Read this file when the user wants to add function tools, organize toolsets, connect MCP servers, or use explicit common search tools.

## Add Tools to an Agent

Use `@agent.tool_plain` for pure functions and `@agent.tool` for tools that need `RunContext`.

```python
import random

from pydantic_ai import Agent, RunContext

agent = Agent('google:gemini-3-flash-preview', name='dice_tools_agent', deps_type=str)


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

## Describe Enum Options to the Model

An `Enum` parameter or field reaches the model as a bare list of values. Mix in `UseEnumMemberDocstrings` to
send the docstring written under each member as that option's description; the enum then renders as `anyOf` of
described `const`s. This is the enum counterpart to `model_config = ConfigDict(use_attribute_docstrings=True)`
on a Pydantic model — an `Enum` has no config to carry a flag, and any assigned class attribute becomes a
member, so a base class is the marker.

```python
from enum import Enum

from pydantic_ai import Agent, UseEnumMemberDocstrings

agent = Agent('openai:gpt-5.2', name='ticket_agent')


class Urgency(UseEnumMemberDocstrings, str, Enum):
    """How urgent the ticket is."""

    low = 'low'
    """Can wait a week."""
    high = 'high'
    """Needs attention today."""


@agent.tool_plain
def set_urgency(urgency: Urgency) -> str:
    """Set the urgency of the ticket."""
    return f'Urgency set to {urgency.value}.'
```

Without the mix-in the docstrings are ignored and the schema is unchanged. `Literal`s have nowhere to write a
docstring; use a described `Enum` when the model needs to know what each option means.

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
- `ctx.realtime` — whether the run is a realtime session
- `ctx.realtime_session` — the live `RealtimeSession` in tools and `on_event` hooks (`None` in
  classic runs, setup hooks, and throughout `wrap_run`, whose context copy predates the connection)

Inside a realtime tool, `await ctx.realtime_session.close()` hangs up cleanly: the calling tool does
not resume, and its call is recorded as interrupted. Use `ctx.cancel()` instead when the session
context should raise `RunCancelled`; that route also records the call as interrupted.

## Use MCP Servers

For URL-based MCP servers, use the `MCP` capability — it runs the MCP server locally by default and lets you opt into the model provider's native MCP support with `native=True`. See the [MCP capability docs](https://pydantic.dev/docs/ai/capabilities/mcp/).

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP

agent = Agent(
    'openai:gpt-5.2',
    name='mcp_agent',
    capabilities=[
        MCP('https://mcp.example.com/api'),               # Streamable HTTP, local-only by default
        MCP('https://mcp.example.com/other', native=True),  # opt into native with local fallback
    ],
)
```

The capability's first positional arg is the URL. To pass any other `MCPToolset` input — a `fastmcp.client.transports.*Transport`, pre-built `fastmcp.Client`, in-process `FastMCP` server, or local script path — use the `local=` keyword:

```python
from fastmcp.client.transports import StdioTransport

from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP

agent = Agent(
    'openai:gpt-5.2',
    name='mcp_stdio_agent',
    capabilities=[MCP(local=StdioTransport(command='python', args=['mcp_server.py']))],
)
```

When you need to manage the toolset lifecycle yourself, share an MCP server across multiple agents, or use FastMCP-specific configuration that doesn't fit the capability shape, use [`MCPToolset`](https://pydantic.dev/docs/ai/mcp/client/) directly and pass it via `toolsets=[...]`. Its `tool_error_behavior` controls how a tool error from the server surfaces: `'retry'` (default) raises `ModelRetry`, `'failed'` raises `ToolFailed` (recorded as `outcome='failed'`), and `'error'` raises the raw `fastmcp` `ToolError`. For SEP-1686 tools with optional task support, set `prefer_tasks=False` to use normal calls; required tasks still use task-augmented execution.

An agent inside an MCP server can use `MCPSamplingModel(ctx.session)` to request inference from the connected client. It accepts prior function tool calls, results, and retries via `message_history` using the MCP 2025-11-25 sampling format. This preserves completed tool exchanges; it does not enable new tool calls or multimodal tool results. See [MCP sampling](https://pydantic.dev/docs/ai/mcp/server/#mcp-sampling).

## Search with DuckDuckGo, Tavily, or Exa

Use common tools when the user wants explicit search tools rather than provider-adaptive capabilities.

```python
from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

agent = Agent(
    'openai:gpt-5.2',
    name='duckduckgo_search_agent',
    tools=[duckduckgo_search_tool()],
    instructions='Search DuckDuckGo for the given query and return the results.',
)
```

Good default split:

- use `WebSearch()` capability when the user wants model-agnostic search with native fallback
- use `duckduckgo_search_tool()` / Tavily / Exa when the user explicitly wants those engines as tools
