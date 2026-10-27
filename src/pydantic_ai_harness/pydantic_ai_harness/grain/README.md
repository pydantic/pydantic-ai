# Grain

`Grain` connects an agent to [Grain](https://grain.com)'s hosted MCP server.
The agent can search and read the meetings, transcripts, notes, and deals the
signed-in user can see, and create clips and tag meetings unless you set
`read_only=True`.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/grain/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Before you start

Grain MCP works on every Grain plan. The deal and coaching tools need the
Business or Enterprise plan.

Grain MCP takes an OAuth access token, which Grain issues only through the
standard MCP authorization flow. The server supports dynamic client
registration, so any MCP OAuth client library can sign the user in. Pass the
token it returns as `auth` or `GRAIN_ACCESS_TOKEN`.

## Installation

uv:

```bash
uv add "pydantic-ai-harness[grain]" "pydantic-ai-slim[openai]"
```

pip:

```bash
pip install "pydantic-ai-harness[grain]" "pydantic-ai-slim[openai]"
```

The second package installs the OpenAI provider the example uses. For another
model, install that provider's extra instead.

## Connect

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Grain

agent = Agent('openai:gpt-5.6-sol', capabilities=[Grain()])
result = agent.run_sync('Summarize my most recent Grain meeting')
print(result.output)
```

Set `GRAIN_ACCESS_TOKEN` to a Grain access token, or pass `auth=` a token.
On your own machine, `auth='oauth'` signs you in through the browser instead. To
serve several users from one agent, pass a function instead (see [Per-user
credentials](#per-user-credentials)).

## Per-user credentials

`auth` decides which Grain account each run uses:

| `auth` | Account used |
| --- | --- |
| Not set, `None`, or `''` | `GRAIN_ACCESS_TOKEN`. If that is not set either, creating the agent raises an error. |
| An access token | That token, for every run. |
| `'oauth'` | The account you sign in to through the browser. This only works on your own machine. |
| A function | Called at the start of each run, and can be async. The token it returns is used for that run. If it returns `None` or `''`, that run has no Grain tools. A function never uses `GRAIN_ACCESS_TOKEN`, and must not return `'oauth'`. |

A fixed token or `GRAIN_ACCESS_TOKEN` suits a script or an agent on your own
machine, where every run is the same account.

In an app where each user connects their own Grain account, one agent serves
all of them, so the token cannot be fixed when the agent is created. Pass a
function that reads the current user's token from the run's deps:

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

The function runs once per run, so each run connects with the token from its
own deps.

Your app gets each user's token, stores it, and refreshes it. For example, a
"Connect Grain" button that runs the OAuth flow from [Before you
start](#before-you-start) and saves the token to their account. Before each run,
put it in the deps, or load it in an async `auth` function.

With durable execution such as Temporal, read the token from the run's deps,
not a global: the function may run in another process.

Grain connects when the run starts, even with `defer_loading=True`. If the
token is invalid or expired, the run raises an `httpx.HTTPStatusError` with
status 401, so return `None` from the function when you know the token is no
longer valid.

The capability's `id` defaults to `grain`, so `defer_loading=True` works
without one. To use two Grain accounts in one agent, give each `Grain` its own
`id` and `tool_prefix`. Two that share an `id` but differ raise an
error.

## Tool names

The agent sees each of the server's tools under the `grain_` prefix, as in
`grain_<tool>`, so they don't collide with other tools on the agent. Set
`tool_prefix` to use a different prefix, or `tool_prefix=None` to keep the names
the server gives them.

`read_only` selects tools by what the server marks as read-only, not by name, so
the prefix does not change what it keeps. A [toolset
wrapper](https://pydantic.dev/docs/ai/tools-toolsets/toolsets/) on
`capability.get_toolset()`, such as `filtered()`, sees the names the agent sees,
prefix included.

## Upgrading

`Grain` used to give the agent the server's tool names unprefixed. They now
reach the agent as `grain_<tool>`. Code that refers to them by name, such as a
`filtered()` wrapper, an approval rule, or a `ToolGuardrail(tools=[...])`, needs
the new names. A rule that no longer matches stops applying, so an approval rule
or guardrail that named an old tool lets its calls through without approval or
checks until you update it. Pass `tool_prefix=None` to keep the old names. If you
wrapped `Grain` in
[PrefixTools](https://pydantic.dev/docs/ai/capabilities/prefix-tools/) to avoid
collisions, you can drop the wrapper and set `tool_prefix` instead; keeping it
adds both prefixes.

- **Durable runs in flight.** A Temporal or DBOS run replays the model responses
  it recorded, and their tool calls use the old names, so a run started before
  the change that calls a `Grain` tool after it no longer replays. Activity and
  step names come from the capability's `id`, which is unchanged. Let those runs
  finish, keep `tool_prefix=None` until they have, or deploy the change with
  [Temporal worker
  versioning](https://docs.temporal.io/production-deployment/worker-deployments/worker-versioning)
  or a new DBOS application version.

## Tool selection and approval

`read_only=True` gives the agent only the tools that Grain's server labels as
read-only, and leaves out all the others. If Grain has not labeled its read
tools, the agent gets no Grain tools and emits an `MCPReadOnlyNoToolsWarning`. The token's
permissions still decide what the agent can reach.

Transcripts and meeting titles contain text from everyone in the meeting,
including external guests, and that text reaches the model as tool output. Use
`read_only=True` or require approval if the agent should not change Grain data.
`include_instructions=False` stops the server's own instructions from reaching
the agent.

To filter tools or require approval in your application, wrap the toolset with
the existing [toolset
wrappers](https://pydantic.dev/docs/ai/tools-toolsets/toolsets/). For example,
this asks for approval before every tool call, so you decide which meeting data
reaches the model:

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

Handle the approval requests with the [deferred tools
workflow](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/). To cap
the size of tool output, add [Tool Output
Limits](https://pydantic.dev/docs/ai/harness/tool-output-limits/).

## Connection customization

Pass `client` (a FastMCP client or transport) to control the connection
yourself, for example to add a proxy or MCP handlers. The client owns the URL
and authentication, so it cannot be combined with `auth`. `read_only` and
`include_instructions` still apply.

A `client` is one connection shared by every run; see [Per-user
credentials](#per-user-credentials) to connect each user separately.

## Telemetry

`Grain` emits no spans of its own. Core
[instrumentation](https://pydantic.dev/docs/ai/capabilities/instrumentation/)
records each Grain tool call as a tool span.

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
  - Grain: {}
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Grain

agent = Agent.from_file('agent.yaml', custom_capability_types=[Grain])
```

Pass `custom_capability_types` so the loader can create `Grain` from the file.
