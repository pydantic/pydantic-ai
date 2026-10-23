# Google Workspace

Let an agent use Gmail, Calendar, Drive, Docs, Sheets, Slides, Chat, and People through Google's hosted MCP servers. `GoogleWorkspace` gives the agent every tool those servers offer for the products you select, including tools that send, change, and delete. The token's OAuth scopes limit what those tools can reach.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Install and connect

uv:

```bash
uv add "pydantic-ai-harness[google-workspace]" "pydantic-ai-slim[openai]"
```

pip:

```bash
pip install "pydantic-ai-harness[google-workspace]" "pydantic-ai-slim[openai]"
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
| A function | Called at the start of each run, and can be async. The token it returns is used for that run. If it returns `None` or `''`, that run has no Google Workspace tools. A function never uses `GOOGLE_ACCESS_TOKEN`, and returning `'oauth'` raises an error. |

Use a fixed token for a script that always acts as one account. In an app where each user connects their own Google account, pass a function that reads the current user's token from the run's deps:

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

Each run opens its own connections with the token the function returns.

Your app owns the OAuth flow: store each user's refresh token and exchange it for an access token. Before each run, put that in the deps, or do the exchange in an async `auth` function. Access tokens expire after about an hour, and the capability does not refresh them: it reads a fixed token once when the agent is created, and a function's token once at the start of each run.

With durable execution such as Temporal, read the credential from the run's deps rather than from a global, since the function may run in another process.

The capability's `id` defaults to `google-workspace-<products>` with the products sorted, such as `google-workspace-calendar-gmail`, so `defer_loading=True` works without one. Two instances whose products do not overlap can share an agent. Two whose products overlap, such as the same product for two accounts, register the same tool names: wrap one in [PrefixTools](https://pydantic.dev/docs/ai/capabilities/prefix-tools/), and when their products are the same, also give it a distinct `id`.

## Provider settings

`services` selects one product or a list: `gmail`, `drive`, `docs`, `sheets`, `slides`, `calendar`, `chat`, or `people`. Tool names start with the product, such as `gmail_search_threads`. Unless you set `description`, the one the agent sees when it loads the capability on demand names the selected products.

Register a Google OAuth client yourself and request the scopes the selected products need. Google does not support automatic client registration, so `auth='oauth'` raises an error.

## Tool selection and approval

`read_only=True` keeps only the tools the server marks as read-only. If the server does not mark its read tools, the agent gets none and emits an `MCPReadOnlyNoToolsWarning`. The token is still what controls access.

Tool results include text other people wrote, such as email bodies and shared documents, and that text can carry instructions aimed at the model. Google's setup page describes this [indirect prompt injection](https://developers.google.com/workspace/guides/configure-mcp-servers) risk. For an agent that only reads or summarizes, set `read_only=True`. Otherwise, require approval for tool calls or add [Prompt Injection Defender](https://pydantic.dev/docs/ai/harness/prompt-injection-defender/).

To filter tools or require approval in your application, wrap the toolset with the existing [toolset wrappers](https://pydantic.dev/docs/ai/tools-toolsets/toolsets/). For example, this asks for approval before every tool call:

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

Handle the approval requests with the [deferred tools workflow](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/). To cap the size of tool output, add [Tool Output Limits](https://pydantic.dev/docs/ai/harness/tool-output-limits/).

## Connection customization

`include_instructions=False` stops the servers' own instructions from reaching the agent.

A fixed `auth` is one set of connections shared by every run; see [Per-user credentials](#per-user-credentials) to connect each user separately.

## Telemetry

`GoogleWorkspace` emits no spans of its own. Core's [instrumentation](https://pydantic.dev/docs/ai/capabilities/instrumentation/) already records each Google Workspace tool call as a tool span, and connecting makes no decision worth a span of its own.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/google_workspace/)
