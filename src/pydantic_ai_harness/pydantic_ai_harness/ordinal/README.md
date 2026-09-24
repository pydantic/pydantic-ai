# Ordinal

`Ordinal` connects an agent to [Ordinal](https://www.tryordinal.com)'s hosted MCP
server so it can draft, schedule, and analyze social posts in the signed-in
user's workspaces.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/ordinal/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

## Before you start

Ordinal MCP needs the Pro plan or higher, the same as the REST API. You need an
Ordinal account with access to at least one workspace. Ordinal MCP takes an
OAuth access token; there is no API key to copy.

Ordinal issues these tokens only through OAuth. Your application gets one by
running the standard MCP authorization flow: the server names its authorization
server and supports dynamic client registration, so any MCP OAuth client
library can sign the user in. Store the token it returns and pass it as `auth`
or `ORDINAL_ACCESS_TOKEN`.

If you set up the old server at `https://app.tryordinal.com/api/mcp` with a
workspace API key, remove it. Do not use the old and new servers together,
because their duplicate tool names confuse the agent.

## Installation

uv:

```bash
uv add "pydantic-ai-harness[ordinal]" "pydantic-ai-slim[openai]"
```

pip:

```bash
pip install "pydantic-ai-harness[ordinal]" "pydantic-ai-slim[openai]"
```

The second package installs the OpenAI provider the example uses. For another
model, install that provider's extra instead.

## Connect

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Ordinal

agent = Agent('openai:gpt-5.6-sol', capabilities=[Ordinal()])
result = agent.run_sync('List my Ordinal workspaces')
print(result.output)
```

Set `ORDINAL_ACCESS_TOKEN` to an Ordinal access token, or pass `auth=` a token.
On your own machine, `auth='oauth'` signs you in through the browser instead. To
serve several users from one agent, pass a function instead (see [Per-user
credentials](#per-user-credentials)). Then start by listing workspaces. Every
other Ordinal tool needs a `workspaceSlug` from `ordinal_get_workspace_context`.

## Per-user credentials

`auth` decides which Ordinal account each run uses:

| `auth` | Account used |
| --- | --- |
| Not set, `None`, or `''` | `ORDINAL_ACCESS_TOKEN`. If that is not set either, creating the agent raises an error. |
| An access token | That token, for every run. |
| `'oauth'` | The account you sign in to through the browser. This only works on your own machine. |
| A function | Called at the start of each run. The token it returns is used for that run. If it returns `None` or `''`, that run has no Ordinal tools. A function never uses `ORDINAL_ACCESS_TOKEN`, and must not return `'oauth'`. |

A fixed token or `ORDINAL_ACCESS_TOKEN` suits a script or an agent on your own
machine, where every run is the same account.

In an app where each user connects their own Ordinal account, one agent serves
all of them, so the token cannot be fixed when the agent is created. Pass a
function that reads the current user's token from the run's deps:

```python
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai_harness import Ordinal


@dataclass
class Deps:
    ordinal_token: str | None


def ordinal_token(ctx: RunContext[Deps]) -> str | None:
    return ctx.deps.ordinal_token


agent = Agent('openai:gpt-5.6-sol', deps_type=Deps, capabilities=[Ordinal(auth=ordinal_token)])
```

Each run connects as its own user, so concurrent runs never share an account.

Your app gets each user's token, stores it, and refreshes it. For example, a
"Connect Ordinal" button that runs the OAuth flow from [Before you
start](#before-you-start) and saves the token to their account. Before each run,
load it (this can be async) and put it in the deps; the function only reads it.

With durable execution such as Temporal, read the token from the run's deps
rather than from a global, since the function may run in another process. The
capability's `id` defaults to `ordinal`, so `defer_loading=True` works without
one. To add more than one `Ordinal` to an agent, give each a distinct `id` and
wrap them in [PrefixTools](https://pydantic.dev/docs/ai/capabilities/prefix-tools/),
since their tool names are the same; two that share an `id` but differ raise an
error.

## Tool selection and approval

To filter tools or require approval in your application, wrap the toolset with
the existing [toolset
wrappers](https://pydantic.dev/docs/ai/tools-toolsets/toolsets/). For example,
this asks for approval before every tool call, which suits tools that publish or
schedule posts:

```python
from pydantic_ai import Agent
from pydantic_ai.messages import DeferredToolRequests
from pydantic_ai_harness import Ordinal

capability = Ordinal()
agent = Agent(
    'openai:gpt-5.6-sol',
    toolsets=[capability.get_toolset().approval_required()],
    output_type=[str, DeferredToolRequests],
)
```

Handle the approval requests with the [deferred tools
workflow](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/). To cap
the size of tool output, add [Tool Output
Limits](https://pydantic.dev/docs/ai/harness/tool-output-limits/).

## Connection customization

Use `auth` in almost every case. Pass `client` only when you need control of the
connection itself: your own FastMCP client or transport, for example one with a
proxy or MCP handlers. The client then owns the URL and authentication, so
passing `client` together with `auth` raises an error.
`include_instructions=False` stops the server's own instructions from reaching
the agent.

A `client` is one connection shared by every run; see [Per-user
credentials](#per-user-credentials) to connect each user separately.

## Define the agent in YAML or JSON

Loading a YAML file also needs the `spec` extra.

uv:

```bash
uv add "pydantic-ai-slim[spec]"
```

pip:

```bash
pip install "pydantic-ai-slim[spec]"
```

```yaml
# agent.yaml
model: openai:gpt-5.6-sol
capabilities:
  - Ordinal: {}
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Ordinal

agent = Agent.from_file('agent.yaml', custom_capability_types=[Ordinal])
```

Pass `custom_capability_types` so the loader can create `Ordinal` from the file.
