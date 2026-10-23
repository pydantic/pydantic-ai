---
description: "Connect a Pydantic AI agent to local or remote MCP servers with the MCP capability, calling tools from your own process or via provider-native MCP."
---

# MCP Capability

[`MCP`][pydantic_ai.capabilities.MCP] is a [provider-adaptive capability](overview.md#provider-adaptive-tools) and the primary entry point for [MCP](../mcp/overview.md) in Pydantic AI. It runs the MCP server locally by default — keeping credentials, hooks, and tracing under your control — and supports both URL-based servers and direct client / toolset / transport inputs.

Backed by [`MCPServerTool`][pydantic_ai.native_tools.MCPServerTool] on the native side (see [MCP Server Tool](../native-tools.md#mcp-server-tool) for provider support and configuration) — pass `native=MCPServerTool(...)` directly when you need full control (e.g. a different `id`, `authorization_token`, or `description` than the capability would derive). On the local side, `local=` accepts any [`MCPToolset`][pydantic_ai.mcp.MCPToolset] input (URL, `fastmcp.Client`, transport, in-process `FastMCP` server, script path, …) — non-toolset inputs are wrapped in `MCPToolset` automatically.

Without an explicit `id=`, the capability uses the server URL's hostname as its [`id`][pydantic_ai.capabilities.AbstractCapability.id], which also names its local toolset and [durable](../durable_execution/overview.md) operations. Only the hostname is used, because a URL's path or query can carry a credential. Two `MCP` capabilities for servers on the same host would get the same `id`, so the agent rejects them and asks for an explicit `id=` on each.

```python {title="mcp.py" test="skip" lint="skip"}
from pydantic_ai.capabilities import MCP
from pydantic_ai.native_tools import MCPServerTool

# URL-based MCP server, running locally (requires `pydantic-ai-slim[mcp]`)
MCP('https://mcp.example.com/api')

# Local client without a URL — pass any `MCPToolset` input
# (URL, `fastmcp.Client`, transport, in-process `FastMCP` server, script path, etc.)
MCP(local=my_fastmcp_client)

# Native preferred; URL-based local fallback
MCP('https://mcp.example.com/api', native=True)

# Strict native-only (no local — does not require the `mcp` extra)
MCP('https://mcp.example.com/api', native=True, local=False)

# Explicit native + explicit local — independent configuration on each side
# (e.g. provider-relay URL for native, direct connection for local)
MCP(
    native=MCPServerTool(
        id='public-mcp',
        url='https://relay.example.com/mcp',
        authorization_token='relay-token',
    ),
    local=my_fastmcp_client,
)
```

## Per-user credentials

In an app where each user has their own credentials for the MCP server, one agent serves all of them, so `authorization_token` and `headers` can't be fixed when the capability is created. Pass a function instead: it's called with the [run context][pydantic_ai.tools.RunContext] at the start of each run, so it can read the current user's token from the run's [dependencies](../dependencies.md):

```python {title="mcp_capability_per_user.py"}
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import MCP


@dataclass
class Deps:
    mcp_token: str


def mcp_token(ctx: RunContext[Deps]) -> str:
    return f'Bearer {ctx.deps.mcp_token}'


agent = Agent(
    '{{ openai:medium }}',
    deps_type=Deps,
    capabilities=[MCP('https://mcp.example.com/api', authorization_token=mcp_token)],
)


async def main():
    result = await agent.run('What is 7 plus 5?', deps=Deps(mcp_token='<token>'))
    print(result.output)
    #> The answer is 12.
```

_(To run this example, ensure `asyncio` is imported and add `asyncio.run(main())`; no other changes are needed.)_

The value the function returns is used for that run on both sides: the native [`MCPServerTool`][pydantic_ai.native_tools.MCPServerTool] is built with it, and the local toolset connects in an MCP session of its own, so concurrent runs never share a session between users. Returning `None` sends no `Authorization` header. `headers` accepts a function the same way. See [per-user authentication](../mcp/client.md#per-user-authentication) for why each user needs a session of their own.

For lower-level access — managing the [`MCPToolset`][pydantic_ai.mcp.MCPToolset] lifecycle directly, advanced transport / client configuration, or using MCP servers without going through a capability — see the [MCP documentation](../mcp/overview.md).
