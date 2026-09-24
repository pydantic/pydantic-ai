# Notion

Let an agent search and change content in a Notion workspace. `Notion` gives the agent every tool Notion's hosted MCP server offers, including tools that make changes. The agent acts with the permissions of the Notion user it connects as.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Install and connect

```bash
pip/uv-add "pydantic-ai-harness[notion]" "pydantic-ai-slim[openai]"
```

Set `NOTION_ACCESS_TOKEN` to a Notion OAuth access token, or pass `auth=` a token. On your own machine, `auth='oauth'` signs you in through the browser instead. See the [provider setup](https://developers.notion.com/guides/mcp/build-mcp-client).

```python
from pydantic_ai import Agent
from pydantic_ai_harness.notion import Notion

agent = Agent('openai:gpt-5.6-sol', capabilities=[Notion()])
result = agent.run_sync('Summarize the resources I can access')
print(result.output)
```

## Per-user credentials

`auth` decides which Notion account each run uses:

| `auth` | Account used |
| --- | --- |
| Not set, `None`, or `''` | `NOTION_ACCESS_TOKEN`. If that is not set either, creating the agent raises an error. |
| An access token | That token, for every run. |
| `'oauth'` | The account you sign in to through the browser. This only works on your own machine. |
| A function | Called at the start of each run. The token it returns is used for that run. If it returns `None` or `''`, that run has no Notion tools. A function never uses `NOTION_ACCESS_TOKEN`, and must not return `'oauth'`. |

A fixed token or `NOTION_ACCESS_TOKEN` suits a script or an agent on your own machine, where every run is the same account.

In an app where each user connects their own Notion account, one agent serves all of them, so the token cannot be fixed when the agent is created. Pass a function that reads the current user's token from the run's deps:

```python
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.notion import Notion


@dataclass
class Deps:
    notion_token: str | None


def notion_token(ctx: RunContext[Deps]) -> str | None:
    return ctx.deps.notion_token


agent = Agent('openai:gpt-5.6-sol', deps_type=Deps, capabilities=[Notion(auth=notion_token)])
```

Each run connects as its own user, so concurrent runs never share an account. `read_only` applies to every user.

Your app gets each user's token, stores it, and refreshes it. For example, a "Connect Notion" button that signs them in with Notion OAuth and saves the access token to their account. Before each run, load it (this can be async) and put it in the deps; the function only reads it.

With durable execution such as Temporal, read the credential from the run's deps rather than from a global, since the function may run in another process. The capability's `id` defaults to `notion`, so `defer_loading=True` works without one. To add more than one `Notion` to an agent, give each a distinct `id` and wrap them in [PrefixTools](/ai/capabilities/prefix-tools/), since their tool names are the same; two that share an `id` but differ raise an error.

## Provider settings

Connect with a Notion OAuth access token. Notion integration tokens are a different kind of credential and do not work with the hosted MCP server. Notion decides which pages and tools the user can reach. Some search and connected-source tools need a matching Notion plan and permissions.

## Tool selection and approval

`read_only=True` keeps only the tools the server marks as read-only. If the server does not mark its read tools, this can leave none. The credential is still what controls access.

To filter tools or require approval in your application, wrap the toolset with the existing [toolset wrappers](/ai/tools-toolsets/toolsets/). For example, this asks for approval before every tool call:

```python
from pydantic_ai import Agent
from pydantic_ai.messages import DeferredToolRequests
from pydantic_ai_harness.notion import Notion

capability = Notion()
agent = Agent(
    'openai:gpt-5.6-sol',
    toolsets=[capability.get_toolset().approval_required()],
    output_type=[str, DeferredToolRequests],
)
```

Handle the approval requests with the [deferred tools workflow](/ai/tools-toolsets/deferred-tools/). To cap the size of tool output, add [Tool Output Limits](tool-output-limits.md).

## Connection customization

Use `auth` in almost every case. Pass `client` only when you need control of the connection itself: your own FastMCP client or transport, for example one with a different authentication scheme, a proxy, or MCP handlers. The client then owns the URL and authentication, so passing `client` together with `auth` raises an error. `read_only` still applies. `include_instructions=False` stops the server's instructions from reaching the model.

A `client` is one connection shared by every run; see [Per-user credentials](#per-user-credentials) to connect each user separately. To use two connections whose tool names overlap, give them distinct `id`s and add [PrefixTools](/ai/capabilities/prefix-tools/).

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/notion/)
