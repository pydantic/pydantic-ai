# PostHog

`PostHog` connects an agent to [PostHog](https://posthog.com)'s hosted MCP
server so it can query product analytics, run SQL, and manage feature flags,
experiments, dashboards, surveys, and error tracking, including tools that make
changes. The key you connect with decides what those tools can reach.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/posthog/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

## Before you start

Create a personal API key with the MCP Server preset
([US](https://us.posthog.com/settings/user-api-keys?preset=mcp_server) or
[EU](https://eu.posthog.com/settings/user-api-keys?preset=mcp_server) Cloud). US
and EU Cloud accounts use the same server. See the [provider
setup](https://posthog.com/docs/model-context-protocol).

Some PostHog tools use an LLM on PostHog's side. Those need AI data processing
enabled for your organization and may be billed as PostHog AI usage.

## Installation

uv:

```bash
uv add "pydantic-ai-harness[posthog]" "pydantic-ai-slim[openai]"
```

pip:

```bash
pip install "pydantic-ai-harness[posthog]" "pydantic-ai-slim[openai]"
```

The second package installs the OpenAI provider the example uses. For another
model, install that provider's extra instead.

## Connect

```python
from pydantic_ai import Agent
from pydantic_ai_harness import PostHog

agent = Agent('openai:gpt-5.6-sol', capabilities=[PostHog()])
result = agent.run_sync('Which feature flags are active in my project?')
print(result.output)
```

Set `POSTHOG_PERSONAL_API_KEY` to a PostHog personal API key, or pass `auth=` a
key. On your own machine, `auth='oauth'` signs you in through the browser
instead. To serve several users from one agent, pass a function instead (see
[Per-user credentials](#per-user-credentials)).

## Per-user credentials

`auth` decides which PostHog account each run uses:

| `auth` | Account used |
| --- | --- |
| Not set, `None`, or `''` | `POSTHOG_PERSONAL_API_KEY`. If that is not set either, creating the agent raises an error. |
| A personal API key or OAuth access token | That key, for every run. |
| `'oauth'` | The account you sign in to through the browser. This only works on your own machine. |
| A function | Called at the start of each run. The key it returns is used for that run. If it returns `None` or `''`, that run has no PostHog tools. A function never uses `POSTHOG_PERSONAL_API_KEY`, and must not return `'oauth'`. |

A fixed key or `POSTHOG_PERSONAL_API_KEY` suits a script or an agent on your own
machine, where every run is the same account.

In an app where each user connects their own PostHog account, one agent serves
all of them, so the key cannot be fixed when the agent is created. Pass a
function that reads the current user's key from the run's deps:

```python
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai_harness import PostHog


@dataclass
class Deps:
    posthog_key: str | None


def posthog_key(ctx: RunContext[Deps]) -> str | None:
    return ctx.deps.posthog_key


agent = Agent('openai:gpt-5.6-sol', deps_type=Deps, capabilities=[PostHog(auth=posthog_key)])
```

Each run connects as its own user, so concurrent runs never share an account.
`read_only` and `features` still apply to every run.

Your app gets each user's key, stores it, and refreshes it if it is an OAuth
token. For example, a settings page where each user pastes their own personal
API key, or a "Connect PostHog" button that signs them in with PostHog OAuth and
saves the access token to their account. Before each run, load it (this can be
async) and put it in the deps; the function only reads it.

With durable execution such as Temporal, read the key from the run's deps rather
than from a global, since the function may run in another process. The
capability's `id` defaults to `posthog`, so `defer_loading=True` works without
one. To add more than one `PostHog` to an agent, give each a distinct `id` and
wrap them in
[PrefixTools](https://pydantic.dev/docs/ai/capabilities/prefix-tools/), since
their tool names are the same; two that share an `id` but differ raise an error.

## Provider settings

By default the server offers everything through a single `posthog` tool that the
agent drives with commands, so narrow what it can reach on the server:

- `features=['flags', 'insights', 'sql']` limits the server to those [feature
  groups](https://github.com/PostHog/posthog/blob/master/services/mcp/README.md#feature-filtering).
  Leave it unset for every group.
- `read_only=True` asks the server for its read-only mode, which drops every
  tool that makes changes.

The key's scopes, and the organizations and projects it can reach, still decide
what the agent can access.

## Tool selection and approval

To filter tools or require approval in your application, wrap the toolset with
the existing [toolset
wrappers](https://pydantic.dev/docs/ai/tools-toolsets/toolsets/). For example,
this asks for approval before every tool call:

```python
from pydantic_ai import Agent
from pydantic_ai.messages import DeferredToolRequests
from pydantic_ai_harness import PostHog

capability = PostHog()
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
connection itself: your own FastMCP client or transport, for example one
pointing at a PostHog MCP server you run for a self-hosted instance, a proxy, or
MCP handlers. The client then owns the URL and authentication, so passing
`client` together with `auth` or `features` raises an error. With a client,
`read_only=True` keeps only the tools the server marks as read-only, instead of
asking the server for read-only mode. PostHog's default mode serves one
`posthog` tool that is not marked read-only, so this filter leaves no PostHog
tools; have your client send the `x-posthog-read-only: true` header instead.
`include_instructions=False` stops the server's own instructions from reaching
the agent.

A `client` is one connection shared by every run; see [Per-user
credentials](#per-user-credentials) to connect each user separately.

## Telemetry

`PostHog` emits no spans of its own. Core's
[instrumentation](https://pydantic.dev/docs/ai/capabilities/instrumentation/)
already records each PostHog tool call as a tool span, and connecting makes no
decision worth a span of its own.

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
  - PostHog:
      features: [flags, insights]
      read_only: true
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import PostHog

agent = Agent.from_file('agent.yaml', custom_capability_types=[PostHog])
```

Pass `custom_capability_types` so the loader can create `PostHog` from the file.
