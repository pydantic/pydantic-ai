---
description: "Give a Pydantic AI agent Linear tools through Linear's hosted MCP server: read and update issues, projects, and comments, with per-user tokens or OAuth."
---

# Linear

Let an agent read and change Linear issues, projects, teams, and comments. `Linear` gives the agent every tool Linear's hosted MCP server offers, including tools that make changes. The credential you connect with decides what those tools can reach.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Install and connect

```bash
pip/uv-add "pydantic-ai-harness[linear]" "pydantic-ai-slim[openai]"
```

Set `LINEAR_ACCESS_TOKEN` to a Linear API key or OAuth access token, or pass `auth=` a token. On your own machine, `auth='oauth'` signs you in through the browser instead. See the [provider setup](https://linear.app/docs/mcp).

```python
from pydantic_ai import Agent
from pydantic_ai_harness.linear import Linear

agent = Agent('{{ openai:medium }}', capabilities=[Linear()])
result = agent.run_sync('Summarize the resources I can access')
print(result.output)
```

## Per-user credentials

`auth` decides which Linear account each run uses:

| `auth` | Account used |
| --- | --- |
| Not set, `None`, or `''` | `LINEAR_ACCESS_TOKEN`. If that is not set either, creating the agent raises an error. |
| An API key or OAuth token | That token, for every run. |
| `'oauth'` | The account you sign in to through the browser. This only works on your own machine. |
| A function | Called at the start of each run. The token it returns is used for that run. If it returns `None` or `''`, that run has no Linear tools. A function never uses `LINEAR_ACCESS_TOKEN`, and must not return `'oauth'`. |

A fixed token or `LINEAR_ACCESS_TOKEN` suits a script or an agent on your own machine, where every run is the same account.

In an app where each user connects their own Linear account, one agent serves all of them, so the token cannot be fixed when the agent is created. Pass a function that reads the current user's token from the run's deps:

```python
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.linear import Linear


@dataclass
class Deps:
    linear_token: str | None


def linear_token(ctx: RunContext[Deps]) -> str | None:
    return ctx.deps.linear_token


agent = Agent('{{ openai:medium }}', deps_type=Deps, capabilities=[Linear(auth=linear_token)])
```

Each run connects as its own user, so concurrent runs never share an account. `read_only=True` still applies to every run.

Your app gets each user's token, stores it, and refreshes it. For example, a settings page where each user pastes their own Linear API key, or a "Connect Linear" button that signs them in with Linear OAuth and saves the access token to their account. Before each run, load it (this can be async) and put it in the deps; the function only reads it.

With durable execution such as Temporal, read the credential from the run's deps rather than from a global, since the function may run in another process. The capability's `id` defaults to `linear`, so `defer_loading=True` works without one. To add more than one `Linear` to an agent, give each a distinct `id` and `tool_prefix`; two that share an `id` but differ raise an error.

## Provider settings

`read_only=True` connects to Linear's read-only endpoint instead of the full one. OAuth token scopes can limit access further.

## Tool names

The agent sees each of the server's tools under the `linear_` prefix, as in `linear_list_issues`, so they don't collide with other tools on the agent. Set `tool_prefix` to use a different prefix, or `tool_prefix=None` to keep the names the server gives them.

`read_only` selects tools by what the server marks as read-only, not by name, so the prefix does not change what it keeps. A [toolset wrapper](../toolsets.md) on `capability.get_toolset()`, such as `filtered()`, sees the names the agent sees, prefix included.

## Upgrading

`Linear` used to give the agent the server's tool names unprefixed. A tool such as `list_issues` now reaches the agent as `linear_list_issues`. Code that refers to them by name, such as a `filtered()` wrapper, an approval rule, or a `ToolGuardrail(tools=[...])`, needs the new names. A rule that no longer matches stops applying, so an approval rule or guardrail that named an old tool lets its calls through without approval or checks until you update it. Pass `tool_prefix=None` to keep the old names. If you wrapped `Linear` in [PrefixTools](../capabilities/prefix-tools.md) to avoid collisions, you can drop the wrapper and set `tool_prefix` instead; keeping it adds both prefixes.

- **Durable runs in flight.** A Temporal or DBOS run replays the model responses it recorded, and their tool calls use the old names, so a run started before the change that calls a `Linear` tool after it no longer replays. Activity and step names come from the capability's `id`, which is unchanged. Let those runs finish, keep `tool_prefix=None` until they have, or deploy the change with [Temporal worker versioning](https://docs.temporal.io/production-deployment/worker-deployments/worker-versioning) or a new DBOS application version.

## Tool selection and approval

To filter tools or require approval in your application, wrap the toolset with the existing [toolset wrappers](../toolsets.md). For example, this asks for approval before every tool call:

```python
from pydantic_ai import Agent
from pydantic_ai.messages import DeferredToolRequests
from pydantic_ai_harness.linear import Linear

capability = Linear()
agent = Agent(
    '{{ openai:medium }}',
    toolsets=[capability.get_toolset().approval_required()],
    output_type=[str, DeferredToolRequests],
)
```

Handle the approval requests with the [deferred tools workflow](../deferred-tools.md). To cap the size of tool output, add [Tool Output Limits](tool-output-limits.md).

## Connection customization

Use `auth` in almost every case. Pass `client` only when you need control of the connection itself: your own FastMCP client or transport, for example one with a different authentication scheme, a proxy, or MCP handlers. The client then owns the URL and authentication, so passing `client` together with `auth` raises an error. With a `client`, `read_only=True` keeps only the tools the server marks as read-only. If none are marked, the agent gets no Linear tools and emits an `MCPReadOnlyNoToolsWarning`. `include_instructions=False` stops the server's own instructions from reaching the agent.

A `client` is one connection shared by every run; see [Per-user credentials](#per-user-credentials) to connect each user separately. To use two connections, give each a distinct `id` and `tool_prefix`.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/linear/)
