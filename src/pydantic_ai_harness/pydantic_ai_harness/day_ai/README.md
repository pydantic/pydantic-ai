# Day AI

`DayAI` connects an agent to [Day AI](https://day.ai)'s hosted MCP server so it
can search and update CRM records, read meeting context, and draft emails in the
signed-in user's workspace.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/day_ai/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Before you start

Day AI MCP needs a paid Day AI Agent tier; the tools and limits of your tier and
workspace role apply. Day AI MCP takes an OAuth access token; there is no API key
to copy.

Day AI issues these tokens only through OAuth. Your application gets one by
running the standard MCP authorization flow: the server names its authorization
server and supports dynamic client registration, so any MCP OAuth client
library can sign the user in. Store the token it returns and pass it as `auth`
or `DAY_AI_ACCESS_TOKEN`.

## Installation

uv:

```bash
uv add "pydantic-ai-harness[day-ai]" "pydantic-ai-slim[openai]"
```

pip:

```bash
pip install "pydantic-ai-harness[day-ai]" "pydantic-ai-slim[openai]"
```

The second package installs the OpenAI provider the example uses. For another
model, install that provider's extra instead.

## Connect

```python
from pydantic_ai import Agent
from pydantic_ai_harness import DayAI

agent = Agent('openai:gpt-5.6-sol', capabilities=[DayAI()])
result = agent.run_sync('Which Day AI opportunities are in the proposal stage?')
print(result.output)
```

Set `DAY_AI_ACCESS_TOKEN` to a Day AI access token, or pass `auth=` a token. On
your own machine, `auth='oauth'` signs you in through the browser instead. To
serve several users from one agent, pass a function instead (see [Per-user
credentials](#per-user-credentials)).

## Per-user credentials

`auth` decides which Day AI account each run uses:

| `auth` | Account used |
| --- | --- |
| Not set, `None`, or `''` | `DAY_AI_ACCESS_TOKEN`. If that is not set either, creating the agent raises an error. |
| An access token | That token, for every run. |
| `'oauth'` | The account you sign in to through the browser. This only works on your own machine. |
| A function | Called at the start of each run, and can be async. The token it returns is used for that run. If it returns `None` or `''`, that run has no Day AI tools. A function never uses `DAY_AI_ACCESS_TOKEN`, and must not return `'oauth'`. |

A fixed token or `DAY_AI_ACCESS_TOKEN` suits a script or an agent on your own
machine, where every run is the same account.

In an app where each user connects their own Day AI account, one agent serves
all of them, so the token cannot be fixed when the agent is created. Pass a
function that reads the current user's token from the run's deps:

```python
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai_harness import DayAI


@dataclass
class Deps:
    day_ai_token: str | None


def day_ai_token(ctx: RunContext[Deps]) -> str | None:
    return ctx.deps.day_ai_token


agent = Agent('openai:gpt-5.6-sol', deps_type=Deps, capabilities=[DayAI(auth=day_ai_token)])
```

Each run connects as its own user, so concurrent runs never share an account.

Your app gets each user's token, stores it, and refreshes it. For example, a
"Connect Day AI" button that runs the OAuth flow from [Before you
start](#before-you-start) and saves the token to their account. Before each run,
put it in the deps, or load it in an async `auth` function.

With durable execution such as Temporal, read the token from the run's deps
rather than from a global, since the function may run in another process.
Temporal stores deps in workflow history, so encrypt history with a [payload
codec](https://docs.temporal.io/develop/python/converters-and-encryption#custom-payload-codec)
before you put tokens there.

The capability's `id` defaults to `day_ai`, so `defer_loading=True` works
without one. To add more than one `DayAI` to an agent, give each a distinct `id`
and `tool_prefix`; two that share an `id` but differ raise an error.

## Tool names

The agent sees each of the server's tools under the `day_ai_` prefix, as in
`day_ai_<tool>`, so they don't collide with other tools on the agent. Set
`tool_prefix` to use a different prefix, or `tool_prefix=None` to keep the names
the server gives them.

A [toolset wrapper](https://pydantic.dev/docs/ai/tools-toolsets/toolsets/) on
`capability.get_toolset()`, such as `filtered()`, sees the names the agent sees,
prefix included.

## Upgrading

`DayAI` used to give the agent the server's tool names unprefixed. They now
reach the agent as `day_ai_<tool>`. Code that refers to them by name, such as a
`filtered()` wrapper, an approval rule, or a `ToolGuardrail(tools=[...])`, needs
the new names. A rule that no longer matches stops applying, so an approval rule
or guardrail that named an old tool lets its calls through without approval or
checks until you update it. Pass `tool_prefix=None` to keep the old names. If you
wrapped `DayAI` in
[PrefixTools](https://pydantic.dev/docs/ai/capabilities/prefix-tools/) to avoid
collisions, you can drop the wrapper and set `tool_prefix` instead; keeping it
adds both prefixes.

- **Durable runs in flight.** A Temporal or DBOS run replays the model responses
  it recorded, and their tool calls use the old names, so a run started before
  the change that calls a `DayAI` tool after it no longer replays. Activity and
  step names come from the capability's `id`, which is unchanged. Let those runs
  finish, keep `tool_prefix=None` until they have, or deploy the change with
  [Temporal worker
  versioning](https://docs.temporal.io/production-deployment/worker-deployments/worker-versioning)
  or a new DBOS application version.

## Tool selection and approval

Day AI's server does not mark any tool as read-only, so there is no `read_only`
option: the agent gets every tool your tier and role allow, including ones that
change CRM records and send notifications.

Tool results can include text written outside your workspace, such as inbound
email and meeting notes, and that text can steer the model into calling those
tools. To filter tools or require approval, wrap the toolset with the existing
[toolset wrappers](https://pydantic.dev/docs/ai/tools-toolsets/toolsets/). For
example, this asks for approval before every tool call:

```python
from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai_harness import DayAI

capability = DayAI()
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

## Telemetry

`DayAI` emits no spans of its own. Core's
[instrumentation](https://pydantic.dev/docs/ai/capabilities/instrumentation/)
already records each Day AI tool call as a tool span, and connecting makes no
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
  - DayAI: {}
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import DayAI

agent = Agent.from_file('agent.yaml', custom_capability_types=[DayAI])
```

Pass `custom_capability_types` so the loader can create `DayAI` from the file.
