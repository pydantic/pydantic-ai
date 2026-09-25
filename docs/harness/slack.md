---
description: "Give a Pydantic AI agent Slack tools through Slack's hosted MCP server: read and send messages, browse channels, and edit canvases as a Slack user."
---

# Slack

Let an agent read and send Slack messages, browse channels, and work with canvases. `Slack` gives the agent every tool Slack's hosted MCP server offers, including tools that make changes. The token you connect with decides what those tools can reach.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Install and connect

```bash
pip/uv-add "pydantic-ai-harness[slack]" "pydantic-ai-slim[openai]"
```

Set `SLACK_USER_TOKEN` to a Slack user token, or pass `auth=` a token. See the [provider setup](https://docs.slack.dev/ai/slack-mcp-server/).

```python
from pydantic_ai import Agent
from pydantic_ai_harness.slack import Slack

agent = Agent('openai:gpt-5.6-sol', capabilities=[Slack()])
result = agent.run_sync('Summarize the resources I can access')
print(result.output)
```

## Per-user credentials

`auth` decides which Slack account each run uses:

| `auth` | Account used |
| --- | --- |
| Not set, `None`, or `''` | `SLACK_USER_TOKEN`. If that is not set either, creating the agent raises an error. |
| A token | That token, for every run. |
| A function | Called at the start of each run. The token it returns is used for that run. If it returns `None` or `''`, that run has no Slack tools. A function never uses `SLACK_USER_TOKEN`, and must not return `'oauth'`. |

A fixed token or `SLACK_USER_TOKEN` suits a script or an agent on your own machine, where every run is the same Slack user.

In an app where each user connects their own Slack account, one agent serves all of them, so the token cannot be fixed when the agent is created. Pass a function that reads the current user's token from the run's deps:

```python
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.slack import Slack


@dataclass
class Deps:
    slack_user_token: str | None


def slack_token(ctx: RunContext[Deps]) -> str | None:
    return ctx.deps.slack_user_token


agent = Agent('openai:gpt-5.6-sol', deps_type=Deps, capabilities=[Slack(auth=slack_token)])
```

Each run connects as its own user, so concurrent runs never share an account.

Your app gets each user's token, stores it, and refreshes it. For example, a "Connect Slack" button that signs them in to your Slack app with OAuth and saves their user token (`xoxp-`) to their account. Before each run, load it (this can be async) and put it in the deps; the function only reads it.

When users differ in more than their credential, such as giving some users read-only access, build the whole capability for each run with a [dynamic capability](/ai/capabilities/custom/#dynamically-building-a-capability):

```python
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import DynamicCapability
from pydantic_ai_harness.slack import Slack


@dataclass
class Deps:
    slack_user_token: str | None
    read_only: bool = False


def slack(ctx: RunContext[Deps]) -> Slack[Deps] | None:
    if not ctx.deps.slack_user_token:
        return None
    return Slack(auth=ctx.deps.slack_user_token, read_only=ctx.deps.read_only)


agent = Agent('openai:gpt-5.6-sol', deps_type=Deps, capabilities=[DynamicCapability(slack, id='slack')])
```

With durable execution such as Temporal, read the token from the run's deps rather than from a global, since the function may run in another process. The capability's `id` defaults to `slack`, so `defer_loading=True` works without one. To add more than one `Slack` to an agent, give each a distinct `id` and wrap them in [PrefixTools](/ai/capabilities/prefix-tools/), since their tool names are the same; two that share an `id` but differ raise an error.

## Provider settings

Slack's hosted MCP server takes user tokens (`xoxp-`), not bot tokens (`xoxb-`). Register a Slack app and grant its user-token scopes as Slack's MCP documentation describes.

The tools act as the token's user, so messages the agent posts and canvases it edits appear as that user. This capability gives an agent Slack tools; it does not receive Slack messages or start runs from them.

## Tool selection and approval

`read_only=True` keeps only the tools the server marks as read-only. If the server does not mark its read tools, this can leave none. The token is still what controls access.

To filter tools or require approval in your application, wrap the toolset with the existing [toolset wrappers](/ai/tools-toolsets/toolsets/). For example, this asks for approval before every tool call:

```python
from pydantic_ai import Agent
from pydantic_ai.messages import DeferredToolRequests
from pydantic_ai_harness.slack import Slack

capability = Slack()
agent = Agent(
    'openai:gpt-5.6-sol',
    toolsets=[capability.get_toolset().approval_required()],
    output_type=[str, DeferredToolRequests],
)
```

Handle the approval requests with the [deferred tools workflow](/ai/tools-toolsets/deferred-tools/). To cap the size of tool output, add [Tool Output Limits](tool-output-limits.md).

## Connection customization

Use `auth` in almost every case. Pass `client` only when you need control of the connection itself: your own FastMCP client or transport, for example one with a different authentication scheme, a proxy, or MCP handlers. The client then owns the URL and authentication, so passing `client` together with `auth` raises an error. `read_only` and `include_instructions` still apply. `include_instructions=False` stops the server's own instructions from reaching the agent.

A `client` is one connection shared by every run; see [Per-user credentials](#per-user-credentials) to connect each user separately. To use two connections whose tool names overlap, give them distinct `id`s and add [PrefixTools](/ai/capabilities/prefix-tools/).

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/slack/)
