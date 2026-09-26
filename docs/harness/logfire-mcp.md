---
description: "Give a Pydantic AI agent Logfire tools through the hosted Logfire MCP server: query traces and telemetry and manage projects, with per-user API keys."
---

# Logfire MCP

Let an agent query Logfire telemetry and manage Logfire projects. `LogfireMCP` gives the agent every tool Logfire's hosted MCP server offers, including tools that make changes. The credential you connect with decides what those tools can reach.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Install and connect

```bash
pip/uv-add "pydantic-ai-harness[logfire-mcp]" "pydantic-ai-slim[openai]"
```

Set `LOGFIRE_API_KEY`, or pass `auth=` an API key. On your own machine, `auth='oauth'` signs you in through the browser instead. See the [provider setup](https://pydantic.dev/docs/logfire/guides/mcp-server/).

```python
from pydantic_ai import Agent
from pydantic_ai_harness.logfire_mcp import LogfireMCP

agent = Agent('openai:gpt-5.6-sol', capabilities=[LogfireMCP()])
result = agent.run_sync('Summarize the resources I can access')
print(result.output)
```

## Per-user credentials

`auth` decides which Logfire account each run uses:

| `auth` | Account used |
| --- | --- |
| Not set, `None`, or `''` | `LOGFIRE_API_KEY`. If that is not set either, creating the agent raises an error. |
| An API key | That key, for every run. |
| `'oauth'` | The account you sign in to through the browser. This only works on your own machine. |
| A function | Called at the start of each run. The key it returns is used for that run. If it returns `None` or `''`, that run has no Logfire tools. A function never uses `LOGFIRE_API_KEY`, and must not return `'oauth'`. |

A fixed key or `LOGFIRE_API_KEY` suits a script or an agent on your own machine, where every run is the same account.

In an app where each user connects their own Logfire account, one agent serves all of them, so the credential cannot be fixed when the agent is created. Pass a function that reads the current user's credential from the run's deps:

```python
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.logfire_mcp import LogfireMCP


@dataclass
class Deps:
    logfire_token: str | None


def logfire_token(ctx: RunContext[Deps]) -> str | None:
    return ctx.deps.logfire_token


agent = Agent('openai:gpt-5.6-sol', deps_type=Deps, capabilities=[LogfireMCP(auth=logfire_token)])
```

Each run connects as its own user, so concurrent runs never share an account.

Your app gets each user's credential, stores it, and refreshes it. For example, a settings page where each user pastes their own API key, or a "Connect Logfire" button that signs them in with OAuth and saves the token to their account. Before each run, load it (this can be async) and put it in the deps; the function only reads it.

When users differ in more than their credential, such as users whose data is in the EU region, build the whole capability for each run with a [dynamic capability](/ai/capabilities/custom/#dynamically-building-a-capability):

```python
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import DynamicCapability
from pydantic_ai_harness.logfire_mcp import LOGFIRE_EU_MCP_URL, LOGFIRE_US_MCP_URL, LogfireMCP


@dataclass
class Deps:
    logfire_token: str | None
    logfire_region: str = 'us'


def logfire(ctx: RunContext[Deps]) -> LogfireMCP[Deps] | None:
    if ctx.deps.logfire_token is None:
        return None
    url = LOGFIRE_EU_MCP_URL if ctx.deps.logfire_region == 'eu' else LOGFIRE_US_MCP_URL
    return LogfireMCP(auth=ctx.deps.logfire_token, url=url)


agent = Agent('openai:gpt-5.6-sol', deps_type=Deps, capabilities=[DynamicCapability(logfire, id='logfire')])
```

With durable execution such as Temporal, read the credential from the run's deps rather than from a global, since the function may run in another process. The capability's `id` defaults to `logfire-mcp`, so `defer_loading=True` works without one. To add more than one `LogfireMCP` to an agent, give each a distinct `id` and wrap them in [PrefixTools](/ai/capabilities/prefix-tools/), since their tool names are the same; two that share an `id` but differ raise an error.

## Provider settings

The default endpoint is Logfire's US region. Set `url=LOGFIRE_EU_MCP_URL` for EU data, or pass the MCP URL of a self-hosted Logfire. The API key's scopes decide which projects and actions are allowed.

The capability adds short guidance to the agent's instructions: the current UTC time, a reminder that timestamps in examples are not the current time, that queries cover a short time window unless widened, and that Logfire links should be created only when asked for. `include_instructions=False` turns this off, along with the server's own instructions.

## Tool selection and approval

`read_only=True` keeps only the tools the server marks as read-only. If the server does not mark its read tools, this can leave none. The credential is still what controls access.

To filter tools or require approval in your application, wrap the toolset with the existing [toolset wrappers](/ai/tools-toolsets/toolsets/). For example, this asks for approval before every tool call:

```python
from pydantic_ai import Agent
from pydantic_ai.messages import DeferredToolRequests
from pydantic_ai_harness.logfire_mcp import LogfireMCP

capability = LogfireMCP()
agent = Agent(
    'openai:gpt-5.6-sol',
    toolsets=[capability.get_toolset().approval_required()],
    instructions=capability.get_instructions(),
    output_type=[str, DeferredToolRequests],
)
```

Handle the approval requests with the [deferred tools workflow](/ai/tools-toolsets/deferred-tools/). To cap the size of tool output, add [Tool Output Limits](tool-output-limits.md).

## Connection customization

Use `auth` in almost every case. Pass `client` only when you need control of the connection itself: your own FastMCP client or transport, for example one with a different authentication scheme, a proxy, or MCP handlers. The client then owns the URL and authentication, so passing `client` together with `auth` or `url` raises an error. `read_only` and `include_instructions` still apply.

A `client` is one connection shared by every run; see [Per-user credentials](#per-user-credentials) to connect each user separately. To use two connections whose tool names overlap, give them distinct `id`s and add [PrefixTools](/ai/capabilities/prefix-tools/).

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/logfire_mcp/)
