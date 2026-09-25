---
description: "Give a Pydantic AI agent Gmail, Google Calendar, Drive, Docs, and Sheets tools through Google's hosted MCP servers, with per-user OAuth tokens."
---

# Google Workspace

Let an agent use Gmail, Calendar, Drive, and other Google Workspace products. `GoogleWorkspace` gives the agent every tool Google's hosted MCP servers offer for the products you select, including tools that send, change, and delete. The token you connect with decides what those tools can reach.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Install and connect

```bash
pip/uv-add "pydantic-ai-harness[google-workspace]" "pydantic-ai-slim[openai]"
```

Set `GOOGLE_ACCESS_TOKEN` to a Google OAuth access token, or pass `auth=` a token. See the [provider setup](https://developers.google.com/workspace/guides/configure-mcp-servers).

```python
from pydantic_ai import Agent
from pydantic_ai_harness.google_workspace import GoogleWorkspace

agent = Agent('openai:gpt-5.6-sol', capabilities=[GoogleWorkspace(services=['gmail', 'calendar'])])
result = agent.run_sync('Summarize the resources I can access')
print(result.output)
```

## Per-user credentials

`auth` decides which Google account each run uses:

| `auth` | Account used |
| --- | --- |
| Not set, `None`, or `''` | `GOOGLE_ACCESS_TOKEN`. If that is not set either, creating the agent raises an error. |
| An access token | That token, for every run. |
| A function | Called at the start of each run. The token it returns is used for that run. If it returns `None` or `''`, that run has no Google Workspace tools. A function never uses `GOOGLE_ACCESS_TOKEN`, and must not return `'oauth'`. |

A fixed token or `GOOGLE_ACCESS_TOKEN` suits a script or an agent on your own machine, where every run is the same account.

In an app where each user connects their own Google account, one agent serves all of them, so the token cannot be fixed when the agent is created. Pass a function that reads the current user's token from the run's deps:

```python
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.google_workspace import GoogleWorkspace


@dataclass
class Deps:
    google_token: str | None


def google_token(ctx: RunContext[Deps]) -> str | None:
    return ctx.deps.google_token


agent = Agent(
    'openai:gpt-5.6-sol',
    deps_type=Deps,
    capabilities=[GoogleWorkspace(services=['gmail', 'calendar'], auth=google_token)],
)
```

Each run connects as its own user, so concurrent runs never share an account.

Your app gets each user's token, stores it, and refreshes it. For example, a "Connect Google" button that signs them in with Google OAuth, saves the refresh token to their account, and exchanges it for a fresh access token when the old one expires. Before each run, load it (this can be async) and put it in the deps; the function only reads it.

With durable execution such as Temporal, read the credential from the run's deps rather than from a global, since the function may run in another process. The capability's `id` defaults to `google-workspace-<products>` with the products sorted, such as `google-workspace-calendar-gmail`, so `defer_loading=True` works without one. Two for different products can share an agent; to add two for the same ones, give each a distinct `id` and wrap them in [PrefixTools](/ai/capabilities/prefix-tools/), since their tool names are the same.

## Provider settings

`services` selects one product or a list: `gmail`, `drive`, `docs`, `sheets`, `slides`, `calendar`, `chat`, or `people`. Tool names start with the product, such as `gmail_search_threads`.

Register a Google OAuth client yourself and request the scopes the selected products need; Google does not support automatic client registration. Access tokens expire after about an hour, so refresh them in your app.

## Tool selection and approval

`read_only=True` keeps only the tools the server marks as read-only. If the server does not mark its read tools, this can leave none. The token is still what controls access.

To filter tools or require approval in your application, wrap the toolset with the existing [toolset wrappers](/ai/tools-toolsets/toolsets/). For example, this asks for approval before every tool call:

```python
from pydantic_ai import Agent
from pydantic_ai.messages import DeferredToolRequests
from pydantic_ai_harness.google_workspace import GoogleWorkspace

capability = GoogleWorkspace(services=['gmail', 'calendar'])
agent = Agent(
    'openai:gpt-5.6-sol',
    toolsets=[capability.get_toolset().approval_required()],
    output_type=[str, DeferredToolRequests],
)
```

Handle the approval requests with the [deferred tools workflow](/ai/tools-toolsets/deferred-tools/). To cap the size of tool output, add [Tool Output Limits](tool-output-limits.md).

## Connection customization

`include_instructions=False` stops the servers' own instructions from reaching the agent.

A fixed `auth` is one set of connections shared by every run; see [Per-user credentials](#per-user-credentials) to connect each user separately. To use two connections whose tool names overlap, give them distinct `id`s and add [PrefixTools](/ai/capabilities/prefix-tools/).

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/google_workspace/)
