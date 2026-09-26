---
title: Grain
description: Connect a Pydantic AI agent to Grain's hosted MCP server to search and read meetings, transcripts, and notes.
---

# Grain

`Grain` connects an agent to [Grain](https://grain.com)'s hosted MCP server so it can search and read the meetings, transcripts, notes, and deals the signed-in user can see.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/grain/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Before you start

Grain MCP works on every Grain plan; the deal and coaching tools need the Business or Enterprise plan. Grain MCP takes an OAuth access token.

Grain issues these tokens only through OAuth. Your application gets one by running the standard MCP authorization flow: the server names its authorization server and supports dynamic client registration, so any MCP OAuth client library can sign the user in. Store the token it returns and pass it as `auth` or `GRAIN_ACCESS_TOKEN`.

## Installation

```bash
pip/uv-add "pydantic-ai-harness[grain]" "pydantic-ai-slim[openai]"
```

The second package installs the OpenAI provider the example uses. For another model, install that provider's extra instead.

## Connect

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Grain

agent = Agent('openai:gpt-5.6-sol', capabilities=[Grain()])
result = agent.run_sync('Summarize my most recent Grain meeting')
print(result.output)
```

Set `GRAIN_ACCESS_TOKEN` to a Grain access token, or pass `auth=` a token. On your own machine, `auth='oauth'` signs you in through the browser instead. To serve several users from one agent, pass a function instead (see [Per-user credentials](#per-user-credentials)).

## Per-user credentials

`auth` decides which Grain account each run uses:

| `auth` | Account used |
| --- | --- |
| Not set, `None`, or `''` | `GRAIN_ACCESS_TOKEN`. If that is not set either, creating the agent raises an error. |
| An access token | That token, for every run. |
| `'oauth'` | The account you sign in to through the browser. This only works on your own machine. |
| A function | Called at the start of each run. The token it returns is used for that run. If it returns `None` or `''`, that run has no Grain tools. A function never uses `GRAIN_ACCESS_TOKEN`, and must not return `'oauth'`. |

A fixed token or `GRAIN_ACCESS_TOKEN` suits a script or an agent on your own machine, where every run is the same account.

In an app where each user connects their own Grain account, one agent serves all of them, so the token cannot be fixed when the agent is created. Pass a function that reads the current user's token from the run's deps:

```python
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai_harness import Grain


@dataclass
class Deps:
    grain_token: str | None


def grain_token(ctx: RunContext[Deps]) -> str | None:
    return ctx.deps.grain_token


agent = Agent('openai:gpt-5.6-sol', deps_type=Deps, capabilities=[Grain(auth=grain_token)])
```

Each run connects as its own user, so concurrent runs never share an account.

Your app gets each user's token, stores it, and refreshes it. For example, a "Connect Grain" button that runs the OAuth flow from [Before you start](#before-you-start) and saves the token to their account. Before each run, load it (this can be async) and put it in the deps; the function only reads it.

With durable execution such as Temporal, read the token from the run's deps rather than from a global, since the function may run in another process. The capability's `id` defaults to `grain`, so `defer_loading=True` works without one. To add more than one `Grain` to an agent, give each a distinct `id` and wrap them in [PrefixTools](/ai/capabilities/prefix-tools/), since their tool names are the same; two that share an `id` but differ raise an error.

## Tool selection and approval

`read_only=True` gives the agent only the tools that Grain's server labels as read-only, and leaves out all the others. If Grain has not labeled its read tools, the agent gets no Grain tools at all. The token's permissions still decide what the agent can reach.

To filter tools or require approval in your application, wrap the toolset with the existing [toolset wrappers](/ai/tools-toolsets/toolsets/). For example, this asks for approval before every tool call, so you decide which meeting data reaches the model:

```python
from pydantic_ai import Agent
from pydantic_ai.messages import DeferredToolRequests
from pydantic_ai_harness import Grain

capability = Grain()
agent = Agent(
    'openai:gpt-5.6-sol',
    toolsets=[capability.get_toolset().approval_required()],
    output_type=[str, DeferredToolRequests],
)
```

Handle the approval requests with the [deferred tools workflow](/ai/tools-toolsets/deferred-tools/). To cap the size of tool output, add [Tool Output Limits](tool-output-limits.md).

## Connection customization

Use `auth` in almost every case. Pass `client` only when you need control of the connection itself: your own FastMCP client or transport, for example one with a proxy or MCP handlers. The client then owns the URL and authentication, so passing `client` together with `auth` raises an error. `read_only` and `include_instructions` still apply. `include_instructions=False` stops the server's own instructions from reaching the agent.

A `client` is one connection shared by every run; see [Per-user credentials](#per-user-credentials) to connect each user separately.

## Telemetry

`Grain` emits no spans of its own. Core's [instrumentation](/ai/capabilities/instrumentation/) already records each Grain tool call as a tool span, and connecting makes no decision worth a span of its own.

## Define the agent in YAML or JSON

Loading a YAML file also needs the `spec` extra:

```bash
pip/uv-add "pydantic-ai-slim[spec]"
```

```yaml
# agent.yaml
model: openai:gpt-5.6-sol
capabilities:
  - Grain: {}
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Grain

agent = Agent.from_file('agent.yaml', custom_capability_types=[Grain])
```

Pass `custom_capability_types` so the loader can create `Grain` from the file.

## API reference

::: pydantic_ai_harness.grain.Grain
