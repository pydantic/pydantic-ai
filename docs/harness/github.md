# GitHub

Let an agent read and change GitHub repositories, issues, pull requests, and other resources. `GitHub` gives the agent the tools in GitHub's default tool groups, including tools that make changes; `toolsets` picks other groups. The credential you connect with decides what those tools can reach.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Install and connect

```bash
pip/uv-add "pydantic-ai-harness[github]" "pydantic-ai-slim[openai]"
```

Set `GITHUB_TOKEN` to a GitHub personal access token, or pass `auth=` a token. See the [provider setup](https://github.com/github/github-mcp-server/blob/main/docs/remote-server.md).

```python
from pydantic_ai import Agent
from pydantic_ai_harness.github import GitHub

agent = Agent('openai:gpt-5.6-sol', capabilities=[GitHub()])
result = agent.run_sync('Summarize the resources I can access')
print(result.output)
```

## Per-user credentials

`auth` decides which GitHub account each run uses:

| `auth` | Account used |
| --- | --- |
| Not set, `None`, or `''` | `GITHUB_TOKEN`. If that is not set either, creating the agent raises an error. |
| A token | That token, for every run. |
| A function | Called at the start of each run. The token it returns is used for that run. If it returns `None` or `''`, that run has no GitHub tools. A function never uses `GITHUB_TOKEN`, and must not return `'oauth'`. |

A fixed token or `GITHUB_TOKEN` suits a script or an agent on your own machine, where every run is the same account.

In an app where each user connects their own GitHub account, one agent serves all of them, so the token cannot be fixed when the agent is created. Pass a function that reads the current user's token from the run's deps:

```python
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.github import GitHub


@dataclass
class Deps:
    github_token: str | None


def github_token(ctx: RunContext[Deps]) -> str | None:
    return ctx.deps.github_token


agent = Agent('openai:gpt-5.6-sol', deps_type=Deps, capabilities=[GitHub(auth=github_token)])
```

Each run connects as its own user, so concurrent runs never share an account. `read_only`, `toolsets`, and `url` still apply to every run.

Your app gets each user's token, stores it, and refreshes it. For example, a "Connect GitHub" button that signs them in through your GitHub App with OAuth and saves the user access token to their account, or a settings page where each user pastes their own personal access token. Before each run, load it (this can be async) and put it in the deps; the function only reads it.

When users differ in more than their credential, such as a user whose organization is on a GitHub Enterprise Cloud data-residency endpoint, build the whole capability for each run with a [dynamic capability](/ai/capabilities/custom/#dynamically-building-a-capability):

```python
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import DynamicCapability
from pydantic_ai_harness.github import GITHUB_MCP_URL, GitHub


@dataclass
class Deps:
    github_token: str | None
    github_mcp_url: str | None = None


def github(ctx: RunContext[Deps]) -> GitHub[Deps] | None:
    if not ctx.deps.github_token:
        return None
    return GitHub(auth=ctx.deps.github_token, url=ctx.deps.github_mcp_url or GITHUB_MCP_URL)


agent = Agent('openai:gpt-5.6-sol', deps_type=Deps, capabilities=[DynamicCapability(github, id='github')])
```

With durable execution such as Temporal, read the credential from the run's deps rather than from a global, since the function may run in another process. The capability's `id` defaults to `github`, so `defer_loading=True` works without one. To add more than one `GitHub` to an agent, give each a distinct `id` and wrap them in [PrefixTools](/ai/capabilities/prefix-tools/), since their tool names are the same; two that share an `id` but differ raise an error.

## Provider settings

`toolsets=['repos', 'issues', 'actions']` picks which of GitHub's tool groups the server offers. Leave it unset to get the server's default groups. `read_only=True` asks the server for its read-only mode. Set `url` to use a GitHub Enterprise Cloud endpoint.

The capability does not limit which repositories the agent can reach. Set that with the token's or GitHub App's permissions. To send other headers, or to use custom authentication, pass a configured `client`.

## Tool selection and approval

To filter tools or require approval in your application, wrap the toolset with the existing [toolset wrappers](/ai/tools-toolsets/toolsets/). For example, this asks for approval before every tool call:

```python
from pydantic_ai import Agent
from pydantic_ai.messages import DeferredToolRequests
from pydantic_ai_harness.github import GitHub

capability = GitHub()
agent = Agent(
    'openai:gpt-5.6-sol',
    toolsets=[capability.get_toolset().approval_required()],
    output_type=[str, DeferredToolRequests],
)
```

Handle the approval requests with the [deferred tools workflow](/ai/tools-toolsets/deferred-tools/). To cap the size of tool output, add [Tool Output Limits](tool-output-limits.md).

## Connection customization

Use `auth` in almost every case. Pass `client` only when you need control of the connection itself: your own FastMCP client or transport, for example one with a different authentication scheme, a proxy, or MCP handlers. The client then owns the URL and authentication, so passing `client` together with `auth`, `url`, or `toolsets` raises an error. With a client, `read_only=True` keeps only the tools the server marks as read-only, instead of asking the server for read-only mode. `include_instructions=False` stops the server's instructions from reaching the model.

A `client` is one connection shared by every run; see [Per-user credentials](#per-user-credentials) to connect each user separately. To use two connections whose tool names overlap, give them distinct `id`s and add [PrefixTools](/ai/capabilities/prefix-tools/).

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/github/)
