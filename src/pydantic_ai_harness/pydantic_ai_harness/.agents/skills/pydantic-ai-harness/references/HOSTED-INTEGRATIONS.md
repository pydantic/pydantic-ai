# Hosted Integrations

Capabilities that connect an agent to a SaaS product's hosted MCP server: GitHub, Linear, Notion,
Slack, Google Workspace, Logfire, Ordinal, Grain, Day AI, PostHog, Pylon, and StackOne. Each one is an
`AbstractCapability` that returns an `MCPToolset` pointed at the vendor's endpoint, with the URL,
auth header, env var, read-only mode where the service supports one, and a stable `id` already set. The MCP connection runs in your
process, not through the model provider's native MCP support. For any server not
listed here, use core `MCP` (see the end of this page).

## Choose a capability

| Class | Import from | Extra | Credential (`auth` or env var) | Notable options |
|---|---|---|---|---|
| `GitHub` | `pydantic_ai_harness.github` | `github` | `GITHUB_TOKEN` (PAT or GitHub App user token) | `toolsets`, `url` (GHE Cloud), server-side `read_only` |
| `Linear` | `pydantic_ai_harness.linear` | `linear` | `LINEAR_ACCESS_TOKEN` (API key or OAuth token), `'oauth'` | `read_only` uses Linear's read-only endpoint |
| `Notion` | `pydantic_ai_harness.notion` | `notion` | `NOTION_ACCESS_TOKEN` (OAuth token only), `'oauth'` | `read_only` |
| `Slack` | `pydantic_ai_harness.slack` | `slack` | `SLACK_USER_TOKEN` (`xoxp-` user token, not `xoxb-`) | `read_only` |
| `GoogleWorkspace` | `pydantic_ai_harness.google_workspace` | `google-workspace` | `GOOGLE_ACCESS_TOKEN` (OAuth access token) | `services` (required), `read_only`, no `client` |
| `LogfireMCP` | `pydantic_ai_harness.logfire_mcp` | `logfire-mcp` | `LOGFIRE_API_KEY`, `'oauth'` | `url` (US default, EU, self-hosted), `read_only` |
| `Ordinal` | `pydantic_ai_harness.ordinal` | `ordinal` | `ORDINAL_ACCESS_TOKEN`, `'oauth'` | no `read_only` |
| `Grain` | `pydantic_ai_harness.grain` | `grain` | `GRAIN_ACCESS_TOKEN`, `'oauth'` | `read_only` |
| `DayAI` | `pydantic_ai_harness.day_ai` | `day-ai` | `DAY_AI_ACCESS_TOKEN`, `'oauth'` | no `read_only` (server has no annotations) |
| `PostHog` | `pydantic_ai_harness.posthog` | `posthog` | `POSTHOG_PERSONAL_API_KEY`, `'oauth'` | `features`, server-side `read_only` |
| `Pylon` | `pydantic_ai_harness.pylon` | `pylon` | `PYLON_ACCESS_TOKEN` (OAuth only), `'oauth'` | `read_only` |
| `StackOne` | `pydantic_ai_harness.stackone` | `stackone` | `api_key=` or `STACKONE_API_KEY` | `account_id` (required), `actions`, `tool_mode` |

Every class imports from its submodule. `Ordinal`, `Grain`, `DayAI`, `PostHog`, `Pylon`, and
`StackOne` are also exported from top-level `pydantic_ai_harness`; the others are submodule-only. Every extra installs `pydantic-ai-slim[mcp]`.

```bash
uv add "pydantic-ai-harness[github]"   # or linear, notion, slack, google-workspace, logfire-mcp, ...
```

## Shared pattern

All classes except `StackOne` take the same keyword-only fields:

- `auth`: a token string, `'oauth'`, a function `(RunContext) -> str | None`, or `None`.
- `read_only: bool = False` (absent on `Ordinal` and `DayAI`).
- `include_instructions: bool = True`: forward the server's own MCP instructions to the agent.
- `client`: your own FastMCP client or transport, for a custom auth scheme, proxy, or handlers
  (absent on `GoogleWorkspace`). It owns the connection, so combining it with `auth` (and with
  `url`/`toolsets`/`features` where those exist) raises `UserError`.
- `id`, `description`, `defer_loading` from `AbstractCapability`. The default `id` is fixed
  (`'github'`, `'linear'`, `'notion'`, `'slack'`, `'logfire-mcp'`, `'ordinal'`, `'grain'`, `'day_ai'`,
  `'posthog'`, `'pylon'`), so `defer_loading=True` and durable execution need no explicit `id`.

```python
from pydantic_ai import Agent

from pydantic_ai_harness.github import GitHub

agent = Agent(
    'test',
    capabilities=[GitHub(auth='ghp_example', read_only=True, toolsets=['repos', 'issues'])],
)
```

### How `auth` resolves

| `auth` | Behaviour |
|---|---|
| unset, `None`, or `''` | Reads the env var. If that is unset too, `Agent(...)` raises `UserError` at construction. |
| a string | That token for every run. |
| `'oauth'` | FastMCP browser sign-in. Local machine only; it would hang a server. Supported where the vendor has OAuth client registration: Linear, Notion, LogfireMCP, Ordinal, Grain, DayAI, PostHog, Pylon. |
| a function | Called at the start of each run. Its return value is that run's token. `None` or `''` means that run has no tools from this integration. A function never falls back to the env var, and returning `'oauth'` raises `UserError`. |

Use the function form when one agent serves many users. Read the token from deps, not a global:
under Temporal/DBOS/Prefect the function can run in another process.

```python
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext

from pydantic_ai_harness.linear import Linear


@dataclass
class Deps:
    linear_token: str | None


def linear_token(ctx: RunContext[Deps]) -> str | None:
    return ctx.deps.linear_token


agent = Agent('test', deps_type=Deps, capabilities=[Linear(auth=linear_token)])
```

Your app obtains, stores, and refreshes each user's token (for example, an OAuth "Connect" flow).
When users differ in more than the token (say, a different GitHub Enterprise `url`), build the whole
capability per run with core `DynamicCapability`.

### `read_only`

Two mechanisms:

- **Server-side mode**: GitHub (`X-MCP-Readonly` header), Linear (`/mcp/readonly` endpoint), PostHog
  (`x-posthog-read-only` header). The server decides which tools exist.
- **Annotation filter**: every other class, and GitHub/Linear/PostHog when you pass `client`. Keeps only
  tools whose MCP annotations set `readOnlyHint: true`. Unmarked tools are dropped, so a server that
  does not annotate can leave the agent with no tools.

`read_only` is not an access boundary. The token's scopes are. PostHog's default mode serves one
`posthog` tool that is not marked read-only, so `PostHog(client=..., read_only=True)` leaves no tools.

GitHub's server-side `read_only` also removes every write tool, comments included. For "read-only plus named
writes", leave `read_only` off and filter the toolset on the MCP annotations (in
`ToolDefinition.metadata['annotations']`, possibly `None`; `ToolDefinition` is in `pydantic_ai.tools`),
passing it as a toolset:

```python {test="skip" lint="skip"}
def allowed(ctx: RunContext[None], tool: ToolDefinition) -> bool:
    hints = (tool.metadata or {}).get('annotations') or {}
    return hints.get('readOnlyHint') is True or tool.name in {'add_issue_comment'}


agent = Agent('test', toolsets=[GitHub(auth='ghp_example').get_toolset().filtered(allowed)])
```

### Two of the same integration

One instance is one connection to one account. Two with the same `id` and identical settings merge
into one. Two with the same `id` and different settings raise `UserError` at agent construction. To
keep two accounts, give each its own `id`. Their tool names are identical, so also wrap each in core
`PrefixTools`:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import PrefixTools

from pydantic_ai_harness.github import GitHub

agent = Agent(
    'test',
    capabilities=[
        PrefixTools(GitHub(id='github-work', auth='ghp_work'), prefix='work'),
        PrefixTools(GitHub(id='github-oss', auth='ghp_oss'), prefix='oss'),
    ],
)
```

### Approval and output size

The capabilities do not add approval. Wrap the toolset and pass it as a toolset instead of a
capability, then handle `DeferredToolRequests` with the core deferred-tools workflow:

```python
from pydantic_ai import Agent
from pydantic_ai.messages import DeferredToolRequests

from pydantic_ai_harness.slack import Slack

slack = Slack(auth='xoxp-example')
agent = Agent(
    'test',
    toolsets=[slack.get_toolset().approval_required()],
    output_type=[str, DeferredToolRequests],
)
```

Add `ToolOutputLimits` to cap large tool results.

## Where one differs

**GitHub.** `toolsets=['repos', 'issues', 'actions']` picks GitHub's tool groups (`None` keeps the
server's defaults, which include write tools). `toolsets=[]` raises, and so does an entry containing a
comma. `url` targets GitHub Enterprise Cloud (`GITHUB_MCP_URL` is the default). The capability does not
limit which repositories are reachable; the token's permissions do.

**GoogleWorkspace.** `services` is the first positional argument: one of, or a list of, `'gmail'`,
`'drive'`, `'docs'`, `'sheets'`, `'slides'`, `'calendar'`, `'chat'`, `'people'`. Each service is its own MCP
server. Tool names are prefixed with the service (`gmail_search_threads`). The default `id` is
`google-workspace-<sorted services>` (for example `google-workspace-calendar-gmail`), so two instances
for different services coexist. There is no `client` and no merge: two for the same services collide.
Google does not support automatic client registration, so register your own OAuth client with the
needed scopes. Access tokens expire after about an hour.

```python
from pydantic_ai import Agent

from pydantic_ai_harness.google_workspace import GoogleWorkspace

workspace = GoogleWorkspace(['gmail', 'calendar'], auth='ya29.example', read_only=True)
agent = Agent('test', capabilities=[workspace])
print(workspace.id)
#> google-workspace-calendar-gmail
```

**LogfireMCP.** `url=LOGFIRE_EU_MCP_URL` for EU data, or a self-hosted MCP URL (import the constants
from `pydantic_ai_harness.logfire_mcp`). With `include_instructions=True` it also adds query guidance and
the current UTC time (taken from the request timestamp, so Temporal-safe) to the instructions.

**PostHog.** By default the server exposes a single command-driven `posthog` tool. `features='flags'`
or `['flags', 'insights']` limits the server to those feature groups. `features=[]` raises (the server
reads empty as "everything"). One URL serves US and EU; the token decides the region.

**Notion / Pylon.** OAuth access tokens only. Notion integration tokens and Pylon REST API tokens are
rejected by the servers.

**DayAI.** Needs a paid Day AI Agent tier. There is no `read_only`, because the server does not
annotate its tools. Note the underscore in the default `id`, `'day_ai'`.

**StackOne** is the odd one out: HTTP Basic auth plus an account header, not bearer auth.

- `StackOne(account_id, *, api_key=None, base_url='https://api.stackone.com', actions=(), tool_mode=None,
  include_instructions=True, metadata=None, client=None)`. There is no `auth` and no `read_only`.
- `tool_mode='search_execute'` (the default when `actions` is empty) exposes two meta-tools: a search tool
  that returns runtime `action_id`s and an execute tool that takes them. `'individual'` registers one
  tool per action. Passing `actions` (case-insensitive `fnmatch` globs over `{connector}_{action}_{entity}`
  names, such as `'*_list_*'`) switches the default to `individual`. `actions` with an explicit
  `'search_execute'` raises `UserError`.
- The default `id` is `stackone-<account_id>`. Two accounts on the same provider still clash on tool
  names, so wrap each in `PrefixTools`.
- `metadata` is merged onto every tool, so `CodeMode(tools={...})` or `prepare_tools` can select them.
- `StackOneToolset` (same arguments, keyword-only) is the raw toolset for `approval_required()` and other
  toolset wrappers. URL values for `base_url` and `client` must be HTTPS.

```python
from pydantic_ai import Agent

from pydantic_ai_harness import StackOne
from pydantic_ai_harness.stackone import StackOneToolset

hr = StackOne('hr-account', api_key='sk-example', actions=['*_list_*'])
writes = StackOneToolset(
    account_id='hr-account', api_key='sk-example', actions=['workday_create_worker']
).approval_required()
agent = Agent('test', capabilities=[hr], toolsets=[writes])
print(hr.id)
#> stackone-hr-account
```

## Agent specs

All twelve classes can be loaded from YAML or JSON. Pass the class in `custom_capability_types`, and
keep secrets out of the file by leaving `auth`/`api_key` unset so the env var is used. Install each
integration's extra, plus `pydantic-ai-slim[spec]` for YAML files:

```yaml
capabilities:
  - Grain:
      read_only: true
  - StackOne:
      account_id: hr-account
      actions: ['*_list_*']
```

```python {test="skip"}
from pydantic_ai import Agent

from pydantic_ai_harness import Grain, StackOne

agent = Agent.from_file('agent.yaml', custom_capability_types=[Grain, StackOne])
```

A function `auth` and a custom `client` cannot come from a spec.

## Servers not listed: core `MCP`

For any other MCP server, use core `pydantic_ai.capabilities.MCP`, not a harness class:
`MCP(url, authorization_token=..., headers=..., allowed_tools=[...])` runs a local client. Pass
`native=True` to also let a provider with native MCP support call the server itself, or `local=` for a
stdio or in-process server. Core `MCP` has no env-var lookup, per-run token function, or read-only
filter. Add those with a `DynamicToolset`/`DynamicCapability` and `.filtered(...)` if you need them. See
the `building-pydantic-ai-agents` skill.

## See also

- https://pydantic.dev/docs/ai/harness/github/
- https://pydantic.dev/docs/ai/harness/linear/
- https://pydantic.dev/docs/ai/harness/notion/
- https://pydantic.dev/docs/ai/harness/slack/
- https://pydantic.dev/docs/ai/harness/google-workspace/
- https://pydantic.dev/docs/ai/harness/logfire-mcp/
- https://pydantic.dev/docs/ai/harness/ordinal/
- https://pydantic.dev/docs/ai/harness/grain/
- https://pydantic.dev/docs/ai/harness/day-ai/
- https://pydantic.dev/docs/ai/harness/posthog/
- https://pydantic.dev/docs/ai/harness/pylon/
- https://pydantic.dev/docs/ai/harness/stackone/
- https://pydantic.dev/docs/ai/mcp/client/
