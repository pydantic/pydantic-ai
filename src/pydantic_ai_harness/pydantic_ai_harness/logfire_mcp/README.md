# Logfire MCP

Let an agent query Logfire telemetry and manage Logfire projects. `LogfireMCP` gives the agent every tool Logfire's hosted MCP server offers, including tools that make changes. The credential you connect with decides what those tools can reach.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Install and connect

uv:

```bash
uv add "pydantic-ai-harness[logfire-mcp]" "pydantic-ai-slim[openai]"
```

pip:

```bash
pip install "pydantic-ai-harness[logfire-mcp]" "pydantic-ai-slim[openai]"
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
| A function | Called at the start of each run, and can be async. The key it returns is used for that run. If it returns `None` or `''`, that run has no Logfire tools, though the capability's guidance stays in the instructions. A function never uses `LOGFIRE_API_KEY`, and must not return `'oauth'`. |

In an app where each user has their own Logfire account, pass a function that reads the run's credential from deps:

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

Your app stores and refreshes each user's API key or OAuth token. Before each run, put it in the deps, or load it in an async `auth` function.

When users differ in more than their credential, such as users whose data is in the EU region, or when a user without a credential should get no Logfire instructions either, build the whole capability for each run with a [dynamic capability](https://pydantic.dev/docs/ai/capabilities/custom/#dynamically-building-a-capability):

```python
from dataclasses import dataclass
from typing import Literal

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import DynamicCapability
from pydantic_ai_harness.logfire_mcp import LOGFIRE_EU_MCP_URL, LOGFIRE_US_MCP_URL, LogfireMCP


@dataclass
class Deps:
    logfire_token: str | None
    logfire_region: Literal['us', 'eu'] = 'us'


def logfire(ctx: RunContext[Deps]) -> LogfireMCP[Deps] | None:
    if ctx.deps.logfire_token is None:
        return None
    url = LOGFIRE_EU_MCP_URL if ctx.deps.logfire_region == 'eu' else LOGFIRE_US_MCP_URL
    return LogfireMCP(auth=ctx.deps.logfire_token, url=url)


agent = Agent('openai:gpt-5.6-sol', deps_type=Deps, capabilities=[DynamicCapability(logfire, id='logfire')])
```

With durable execution such as Temporal, read the credential from the run's deps rather than from a global, since the function may run in another process. The capability's `id` defaults to `logfire-mcp`, so `defer_loading=True` works without one. To add more than one `LogfireMCP` to an agent, give each a distinct `id` and `tool_prefix`; two that share an `id` but differ raise an error.

## Provider settings

The default endpoint is Logfire's US region. Set `url=LOGFIRE_EU_MCP_URL` for EU data, or pass the MCP URL of a self-hosted Logfire. The API key is sent to `url` as a bearer token, so use an `https` URL. The API key's scopes decide which projects and actions are allowed.

The capability adds short guidance to the agent's instructions: the current UTC hour (when the hour changes during a conversation, the new hour is appended to the message history rather than rewriting the instructions, so the prompt cache keeps the conversation so far), a reminder that timestamps in examples are not the current time, that queries cover a short time window unless widened, that Logfire links should be created only when asked for, and that tool names carry the `tool_prefix` the server's own text leaves out. `include_instructions=False` turns this off, along with the server's own instructions.

## Tool names

The agent sees each of the server's tools under the `logfire_` prefix, as in `logfire_query_run`, so they don't collide with other tools on the agent. Set `tool_prefix` to use a different prefix, or `tool_prefix=None` to keep the names the server gives them.

`read_only` selects tools by what the server marks as read-only, not by name, so the prefix does not change what it keeps. A [toolset wrapper](https://pydantic.dev/docs/ai/tools-toolsets/toolsets/) on `capability.get_toolset()`, such as `filtered()`, sees the names the agent sees, prefix included.

## Tool selection and approval

`read_only=True` keeps only the tools the server marks as read-only. If the server does not mark its read tools, the agent gets none and emits an `MCPReadOnlyNoToolsWarning`. The credential is still what controls access.

Query results carry your telemetry, which can include text your app's users wrote, such as request paths, error messages, and logged prompts. Treat it as untrusted input: an instruction hidden in it can steer the agent toward a tool that makes changes. When the agent only needs to read, use `read_only=True` and an API key scoped to reading.

To filter tools or require approval in your application, wrap the toolset with the existing [toolset wrappers](https://pydantic.dev/docs/ai/tools-toolsets/toolsets/). For example, this asks for approval before every tool call:

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

Handle the approval requests with the [deferred tools workflow](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/). Query results can be large; to cap the size of tool output, add [Tool Output Limits](https://pydantic.dev/docs/ai/harness/tool-output-limits/).

## Connection customization

Use `auth` in almost every case. Pass `client` only when you need control of the connection itself: your own FastMCP client or transport, for example one with a different authentication scheme, a proxy, or MCP handlers. The client then owns the URL and authentication, so passing `client` together with a non-empty `auth` or a `url` other than the default raises an error. `read_only` and `include_instructions` still apply.

A `client` is one connection shared by every run; see [Per-user credentials](#per-user-credentials) to connect each user separately. To use two connections, give each a distinct `id` and `tool_prefix`.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/logfire_mcp/)
