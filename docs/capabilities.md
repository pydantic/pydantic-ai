# Capabilities

A capability is a reusable, composable unit of agent behavior. Instead of threading multiple arguments through your `Agent` constructor — [instructions](agent.md#instructions) here, [model settings](agent.md#model-run-settings) there, a [toolset](toolsets.md) somewhere else, a [history processor](message-history.md#processing-message-history) on yet another parameter — you can bundle related behavior into a single capability and pass it via the [`capabilities`][pydantic_ai.agent.Agent.__init__] parameter.

Capabilities can provide any combination of:

* **Tools** — via [toolsets](toolsets.md) or [native tools](native-tools.md)
* **Lifecycle hooks** — intercept and modify model requests, tool calls, and the overall run
* **Instructions** — static or dynamic [instruction](agent.md#instructions) additions
* **Model settings** — static or per-step [model settings](agent.md#model-run-settings)

This makes them the primary extension point for Pydantic AI. Whether you're building a memory system, a guardrail, a cost tracker, or an approval workflow, a capability is the right abstraction.

## On-demand capabilities {#on-demand-capabilities}

A multi-workflow agent normally sends every workflow's instructions and tool schemas on every turn, and applies every workflow's settings and hooks for the whole run — even though most requests need just one workflow. That cost grows with each workflow you add: more input tokens, and worse tool selection once the visible tool set passes the ~30–50-tool mark where models start picking the wrong one (the same pressure behind [tool search](tools-advanced.md#tool-search)).

Mark a capability with `defer_loading=True` and give it a stable `id`, and it collapses to a one-line catalog entry — its `id` plus an optional `description` — that the model pulls in on demand. Here's the minimal shape:

```python {title="on_demand_capability.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability

refunds = Capability(
    id='refunds',
    description='Use for refund eligibility, refund status, or processing a refund.',
    instructions='Always confirm the order ID before issuing a refund.',
    defer_loading=True,
)


@refunds.tool_plain
def refund_status(order_id: str) -> str:
    """Look up the refund status for an order."""
    return f'Order {order_id}: refund issued on 2026-05-01.'


agent = Agent(
    'openai-responses:gpt-5.4',
    instructions='You are a customer support assistant.',
    capabilities=[refunds],
)
```

On the first turn, the refund workflow is collapsed to a catalog entry. The model sees its base instructions, the framework-managed `load_capability` tool, and the catalog appended to the instructions:

```
The following capabilities are deferred and can be loaded using the `load_capability` tool:
- refunds: Use for refund eligibility, refund status, or processing a refund.
```

The model does not receive the refund instructions or the `refund_status` tool definition yet, so it has no reason to call the tool. Depending on the active model, Pydantic AI may also send provider/tool-search plumbing to preserve the hidden state; that plumbing does not expose the refund tool definition until the capability is loaded. The exchange unfolds across model requests within a single `agent.run_sync` call:

1. **Request 1.** The model sees the catalog above and the user's prompt. It calls the `load_capability` tool with `id='refunds'`.
2. **Load.** Pydantic AI returns the capability's instructions — *"Always confirm the order ID before issuing a refund."* — as the tool result and exposes the `refund_status` definition on the next request.
3. **Request 2.** The model now sees those instructions in history and `refund_status` in its tool list. It calls `refund_status(order_id='ABC-123')` and answers the user from the result.

Already-loaded capabilities stay loaded for the rest of the run — the model never needs to re-open one.

Loading activates the whole bundle, not just instructions: the capability's function tools, model settings, and lifecycle hooks come live together (see [What you can defer](#what-you-can-defer)). It's a one-line change to a capability you already register, it works on [every provider](#cross-provider-behavior), and it [survives history replay](#resumable-across-runs).

!!! note
    The `load_capability` tool name is reserved whenever any on-demand capability is present. Capability `id` values must be stable and explicit — see [Resumable across runs](#resumable-across-runs).

!!! note "Deferred instructions reach client-facing message history"
    A deferred capability's instructions come back as the `load_capability` tool *result*, so they land in the run's message history — including the copy a [UI adapter](ui/overview.md) serializes to the client. Instructions on an always-on capability stay in the server-side system prompt instead. If a capability's instructions shouldn't be exposed to the client, keep it always-on rather than deferred.

### What you can defer

Every part of a capability bundle activates together as a single unit:

| Part | Before load | After load |
|---|---|---|
| Instructions (static or dynamic) | Not sent | Returned as the `load_capability` tool result; included in subsequent requests |
| Function tools | Not exposed | Exposed on the next request |
| Model settings (static or per-step) | Not applied | Merged into the run's settings for subsequent requests |
| Lifecycle [hooks](#hooking-into-the-lifecycle) | Do not fire | Fire after the capability is loaded |
| [Native tools](native-tools.md) | Not exposed | Exposed on the next request — see [Cache implications](#cache-implications) |

### When to use it

**Reach for on-demand capabilities when:**

- the agent serves multiple distinct workflows (refunds, returns, fraud review, account security…) where most turns need one
- a workflow needs *more than instructions* — its own tools, raised reasoning effort, an approval hook — and those should travel together as a unit
- you want skills-style progressive disclosure but also want the loaded bundle to bring tools and settings, not just a runbook

**Skip it when:**

- the capability is used on most turns — the discovery round-trip costs more than the tokens it saves
- you have a flat catalog of individually-discoverable tools with no shared instructions — use [tool search](tools-advanced.md#tool-search) instead, which discovers individual tools by name rather than loading bundles

If you've used [Anthropic's Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills), this is the same idea generalised: a skill is a markdown file the model can pull in on demand. An on-demand capability does that *plus* typed function tools, per-step model settings, and lifecycle hooks.

### Retrofitting an existing capability

`defer_loading=True` is not specific to the [`Capability`][pydantic_ai.capabilities.Capability] convenience class. The shared fields live on [`AbstractCapability`][pydantic_ai.capabilities.AbstractCapability], and built-in capabilities expose `id`, `description`, and `defer_loading` on construction. For custom capabilities, set those attributes on the instance.

```python {title="defer_existing_capability.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP

agent = Agent(
    'openai-responses:gpt-5.4',
    capabilities=[
        MCP(
            url='https://mcp.example.com/analytics',
            native=True,
            id='analytics-mcp',
            description='Use for analytics queries, dashboards, and metric lookups.',
            defer_loading=True,
        ),
    ],
)
```

Until the model loads `analytics-mcp`, none of the MCP server's tool definitions enter the prompt. The same flag works on [`WebSearch`][pydantic_ai.capabilities.WebSearch], [`WebFetch`][pydantic_ai.capabilities.WebFetch], [`Hooks`][pydantic_ai.capabilities.Hooks], and any custom [`AbstractCapability`][pydantic_ai.capabilities.AbstractCapability] subclass — see [Building custom capabilities](#building-custom-capabilities) for adding `defer_loading` to your own subclass.

!!! note "Deferred `MCP`: set a stable `id`"
    [`MCP`][pydantic_ai.capabilities.MCP] derives its `id` from the server URL when you omit one, so `defer_loading=True` works without an explicit `id`. Pass one anyway if you persist and [resume](#resumable-across-runs) conversations: a URL-derived id changes if the URL does (different environment, path version, …), which silently breaks the resumed capability's loaded state.

### Resumable across runs {#resumable-across-runs}

Loaded-capability state lives in message history, not in the agent. When a conversation is persisted to a database and resumed later — possibly on a different process, machine, or model — Pydantic AI reconstructs the loaded set from the `load_capability` tool call/return pairs in history. Capabilities the model loaded earlier stay loaded; capabilities it never loaded stay collapsed in the catalog. No re-discovery round-trip on resume.

This is why deferred capabilities require a stable explicit `id`: history replay matches calls to capabilities by id, so a class-derived id would silently break the moment a class is renamed. The same property makes cross-provider replay work — a run that loaded `refunds` on Anthropic and continued on OpenAI Responses keeps `refunds` loaded after the switch.

History carries *which* capability ids were loaded, not the capabilities themselves: the resuming agent must be constructed with the same capabilities (matching `id`s), just as it must be constructed with the same tools. State lives in history; definitions live in code.

### Runtime state in `RunContext`

Several [`RunContext`][pydantic_ai.tools.RunContext] fields expose progressive-disclosure state to tools, hooks, and capability-owned callbacks:

- `ctx.loaded_capability_ids` — deferred capability IDs explicitly loaded through the `load_capability` tool, reconstructed from message history and updated when a capability loads during the current step.
- `ctx.available_capability_ids` — the currently-live capability IDs: always-available capabilities plus `ctx.loaded_capability_ids`.
- `ctx.capability_loaded` — only meaningful while Pydantic AI is running a capability-owned hook or callback. It is scoped to that capability; deferred hooks and callbacks are skipped until this value would be true.
- `ctx.discovered_tool_names` — deferred function tools revealed by tool search. This is tool-level discovery, separate from capability-level loading.
- `ctx.available_tool_names` — function tool names currently known as available: always-visible tools from the current step's assembled tool manager plus tool-search discoveries reconstructed from history. Early hooks such as `before_run` may see only the history-derived discovered names, or an empty set if none exist yet, before tool definitions have been prepared. See [Hook ordering](hooks.md#hook-ordering) for how hook timing affects what is populated.
- `ctx.usage_limits` — the [`UsageLimits`][pydantic_ai.usage.UsageLimits] the run is enforcing (defaulting to `UsageLimits()` when none were passed, so it's only `None` outside of a run), alongside `ctx.usage` for the usage so far. A capability can read the run's limits to disclose or adapt to the remaining budget (e.g. budget disclosure) without being configured with a duplicate copy. Treat it as read-only: it's the live object the run enforces against, so mutating a field would change what the run enforces on subsequent requests.

Loading a capability updates the capability state immediately, but the loaded bundle's function tools, native tools, and model settings take effect on the next model request.

### Cross-provider behavior

On-demand capabilities work on every model. Where the provider exposes a native progressive-disclosure surface — Anthropic tool search on Sonnet 4.5+/Opus 4.5+/Haiku 4.5+, OpenAI Responses `tool_search` on GPT-5.4+ — Pydantic AI uses that surface so deferred function tools stay out of the prompt prefix. Standalone deferred tools can use the provider's hosted search; tools owned by on-demand capabilities use client-executed local search through the native surface so tools from unloaded capabilities cannot leak. On other providers, a local `search_tools` function tool handles discovery: the initial context shrinks the same way, but cache stability across loads is not guaranteed.

#### Cache implications {#cache-implications}

Calling the `load_capability` tool reveals capability behavior between requests. Whether that breaks the provider's prompt-cache prefix depends on what's revealed:

| What loads | Cache prefix |
|---|---|
| Instructions only | **Stable** — instructions land in the message history, not the request prefix. |
| Function tools on a model with native [tool search](tools-advanced.md#tool-search) (OpenAI Responses, Anthropic) | **Stable** — the function tools visible to the provider don't change across loads. |
| Function tools on other models (local `search_tools` fallback) | **May break between turns** — function-tool visibility changes as capabilities load. |
| Native tools | **Always breaks the prefix on load** — native tool definitions are part of the request prefix on every provider. |

When preserving the cache prefix matters, prefer instruction-only or function-tool-only on-demand capabilities on a model with native tool search. The provider-specific mechanics that keep the prefix stable live in [tools-advanced.md](tools-advanced.md#tool-search).

### The `Capability` convenience class

[`Capability`][pydantic_ai.capabilities.Capability] bundles instructions, function tools, and toolsets without subclassing. Register tools with the decorator that mirrors [`@agent.tool`](tools.md#registering-function-tools-via-decorator):

```python {title="capability_decorator.py"}
from pydantic_ai import RunContext
from pydantic_ai.capabilities import Capability

refunds = Capability(
    id='refunds',
    description='Use for refund eligibility and refund status.',
    instructions='Always confirm the order ID before issuing a refund.',
    defer_loading=True,
)


@refunds.tool
def refund_status(ctx: RunContext[None], order_id: str) -> str:
    """Look up the refund status for an order."""
    return f'Order {order_id}: refund issued on 2026-05-01.'
```

In addition to `@capability.tool` and `@capability.tool_plain`, you can pass existing functions or [`Tool`][pydantic_ai.tools.Tool] instances via `tools=`, or hand in one or more [toolsets](toolsets.md) via `toolsets=`. For dynamic instructions, use the [`@capability.instructions`][pydantic_ai.capabilities.Capability.instructions] decorator. For a dynamic catalog entry, pass a callable as `description=`.

`@capability.tool` and `@capability.tool_plain` mirror [`@agent.tool`](tools.md#registering-function-tools-via-decorator) exactly, including the `defer_loading` argument. On a deferred capability that per-tool flag is a no-op — the capability gates all its tools as a unit — so it only has an effect on a non-deferred `Capability`, where it opts an individual tool into [tool search](tools-advanced.md#tool-search) discovery.

For anything beyond instructions, function tools, toolsets, and descriptions — model settings, hooks, native tools, wrapper toolsets, or custom per-run logic — subclass [`AbstractCapability`][pydantic_ai.capabilities.AbstractCapability] directly. When subclassing, override [`get_description`][pydantic_ai.capabilities.AbstractCapability.get_description] if the catalog entry needs to vary by run.

!!! note "Setting `id` for durable execution"
    A toolset contributed by a capability — via `Capability(tools=[...])` or an [`MCP`](#mcp) server running locally — inherits its `id` from the capability's [`id`][pydantic_ai.capabilities.AbstractCapability.id]. [Durable execution](durable_execution/overview.md) identifies each leaf toolset by its `id`, so pass `Capability(id='...', tools=[...])` or `MCP(id='...', url='...')` when combining a capability with Temporal, DBOS, or Prefect. Temporal requires an `id` for every leaf toolset and DBOS for every MCP server — both raise at construction without one. (`MCP` also derives one from the server URL when no `id` is given.) A URL-derived `id` can collide when two different servers share a host and final path segment (`https://a.com/api` and `https://a.com/v2/api` both derive `a.com-api`); DBOS raises at construction and Temporal when the worker starts, so pass an explicit `id` to disambiguate them.

### Beyond instructions: tools, settings, hooks, native tools {#beyond-instructions}

The [`Capability`][pydantic_ai.capabilities.Capability] example above deferred instructions and a function tool, but the same flag gates the whole bundle — what the model knows, what it can do, and how it does it (see [What you can defer](#what-you-can-defer)). The snippets below show the remaining pieces in turn: model settings, hooks, and native tools.

#### Deferred model settings

[`get_model_settings`][pydantic_ai.capabilities.AbstractCapability.get_model_settings] is collected during capability assembly, but its settings are only applied after the deferred capability is loaded. That means per-step settings like raised reasoning effort only apply for workflows the model opts into:

```python {title="deferred_model_settings.py"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, ModelSettings
from pydantic_ai.capabilities import AbstractCapability


@dataclass
class DeepReasoning(AbstractCapability[Any]):
    def get_model_settings(self) -> ModelSettings:
        return ModelSettings(extra_body={'reasoning_effort': 'high'})


agent = Agent(
    'openai-responses:gpt-5.4',
    capabilities=[
        DeepReasoning(
            id='deep-reasoning',
            description='Use for multi-step planning or hard analytical problems.',
            defer_loading=True,
        ),
    ],
)
```

#### Lifecycle hooks with deferred workflows

Hooks can live on deferred capabilities too. They do not run until the model loads the capability that owns them:

```python {title="deferred_hooks.py"}
from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability


@dataclass
class AccountSecurityWorkflow(AbstractCapability[None]):
    id: str = 'account-security'
    description: str = 'Use when the next action may be destructive.'
    defer_loading: bool = True

    def get_instructions(self) -> str:
        return 'Confirm the customer identity before taking destructive action.'

    async def before_tool_execute(self, ctx, *, call, tool_def, args):
        # Inspect the call, prompt the operator, raise to block.
        return args


agent = Agent('openai-responses:gpt-5.4', capabilities=[AccountSecurityWorkflow()])
```

!!! note "Checking other capabilities"
    `ctx.capability_loaded` is scoped to the capability whose hook is currently running. For an always-on hook capability, it is always true. To check whether another deferred capability has been loaded, look for its ID in `ctx.loaded_capability_ids`, for example `if 'account-security' in ctx.loaded_capability_ids:`. If a hook must enforce a rule before a workflow is loaded, keep that hook in an always-available capability and inspect `ctx.loaded_capability_ids`.

#### Deferred native tools

Any [native capability](#native-capabilities) (`WebSearch`, `WebFetch`, `MCP`, …) can be deferred the same way. The native tool definition only enters the request after the `load_capability` tool loads the capability — see [Cache implications](#cache-implications) for the trade-off:

```python {title="deferred_native_tool.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[
        WebSearch(
            local='duckduckgo',
            id='web-research',
            description='Use when the question requires up-to-date information.',
            defer_loading=True,
        ),
    ],
)
```

### Putting it together: a multi-workflow support agent

A realistic on-demand capability rarely consists of just one piece. The example below defines a customer-support agent with two deferred workflows that exercise different parts of the bundle:

- `orders` — instructions plus a function tool, defined inline with [`Capability`][pydantic_ai.capabilities.Capability].
- `account-security` — instructions, a function tool, raised reasoning effort, *and* an approval hook, all bundled as one [`AbstractCapability`][pydantic_ai.capabilities.AbstractCapability] subclass.

For those workflows, turn 1 exposes only the two-line catalog. Base instructions, always-on tools, the framework-managed `load_capability` tool, and any provider/tool-search plumbing still appear as usual. Loading `account-security` activates the runbook, the destructive tool, the higher reasoning effort, *and* the approval gate together — that's what we mean by bundle-level disclosure.

```python {title="support_agent.py"}
from dataclasses import dataclass

from pydantic_ai import Agent, ModelSettings, RunContext
from pydantic_ai.capabilities import AbstractCapability, Capability
from pydantic_ai.toolsets import AgentToolset, FunctionToolset


@dataclass
class Store:
    orders: dict[str, str]


# Workflow 1: instructions + function tool, defined inline.
orders = Capability[Store](
    id='orders',
    description='Use for order tracking, delivery status, or questions involving an order ID.',
    instructions='Quote the order ID and item name when discussing an order.',
    defer_loading=True,
)


@orders.tool
def order_status(ctx: RunContext[Store], order_id: str) -> str:
    """Look up shipping or delivery status for an order."""
    return ctx.deps.orders.get(order_id, f'No order found with id {order_id}.')


# Workflow 2: instructions + tool + per-step model settings + approval hook,
# all hidden until the model loads `account-security`.
security_tools = FunctionToolset[Store]()


@security_tools.tool
def revoke_sessions(ctx: RunContext[Store], account_id: str) -> str:
    """Revoke all active sessions for an account."""
    return f'Revoked sessions for {account_id}.'


@dataclass
class AccountSecurity(AbstractCapability[Store]):
    id: str = 'account-security'
    description: str = 'Use for suspicious logins, account takeover, or session revocation.'
    defer_loading: bool = True

    def get_instructions(self) -> str:
        return 'Confirm the customer identity before revoking sessions.'

    def get_toolset(self) -> AgentToolset[Store]:
        return security_tools

    def get_model_settings(self) -> ModelSettings:
        # Raise reasoning effort just for sensitive workflows.
        return ModelSettings(extra_body={'reasoning_effort': 'high'})

    async def before_tool_execute(self, ctx, *, call, tool_def, args):
        # Approval gate for destructive actions, active once the model has loaded `account-security`.
        return args


support_agent = Agent(
    'openai-responses:gpt-5.4',
    deps_type=Store,
    instructions='You are a customer-support agent for an e-commerce store.',
    capabilities=[orders, AccountSecurity()],
)
```

A "where is my order?" request loads only `orders`. A "someone is logging into my account" request loads only `account-security` — and from that point on, every tool call in the run passes through the approval hook *and* benefits from the raised reasoning effort, without either being visible to the model on requests that never touched the workflow.

### Enforcing read-before-act

Want the model to actually *read the runbook* before taking a destructive action? Make the runbook a deferred capability, then check `ctx.loaded_capability_ids` in a one-method hook:

```python {title="runbook_required.py"}
from dataclasses import dataclass, field

from pydantic_ai import Agent, ModelRetry
from pydantic_ai.capabilities import AbstractCapability, Capability


@dataclass
class RunbookRequired(AbstractCapability[None]):
    """Bounces a tool call back until the matching runbook has been loaded."""

    requirements: dict[str, str] = field(default_factory=dict)

    async def before_tool_execute(self, ctx, *, call, tool_def, args):
        required = self.requirements.get(tool_def.name)
        if required and required not in ctx.loaded_capability_ids:
            raise ModelRetry(
                f'Call the `load_capability` tool with `id={required!r}` and follow its '
                f'guidance before calling `{tool_def.name}`.'
            )
        return args


refund_policy = Capability(
    id='refund-policy',
    description='Read before issuing refunds. Eligibility rules and approval limits.',
    instructions=(
        'Refunds over $500 require manager approval. '
        'Refunds outside the 30-day window require a documented exception.'
    ),
    defer_loading=True,
)


agent = Agent(
    'openai-responses:gpt-5.4',
    capabilities=[
        refund_policy,
        RunbookRequired(requirements={'issue_refund': 'refund-policy'}),
    ],
)


@agent.tool_plain
def issue_refund(order_id: str, amount: float) -> str:
    """Issue a refund for an order."""
    return f'Refund of ${amount} issued for {order_id}.'
```

The model sees `issue_refund` from turn 1. If it tries to call it before opening `refund-policy`, the hook bounces the call back with a message pointing at the exact `load_capability` tool call to make. The model loads the policy, the policy text lands in its recent context, and the refund runs *within* the rules — and only then. Same shape for any tool-and-runbook pair.

Because the loaded set is just runtime data on [`RunContext`][pydantic_ai.tools.RunContext], the pattern generalises: dynamic instructions can warn when a risky pair of workflows is open, audit hooks can tag traces with the loaded set, escalation hooks can require an extra confirmation when both `payments` and `account-security` are active.

### Loading skills from Markdown files

If you already keep your skills as Markdown files with YAML frontmatter — the format used by [Anthropic Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — you can wrap each one in a [`Capability`][pydantic_ai.capabilities.Capability] with a few lines of glue.

Given a skill file `skills/refunds.md`:

```markdown {title="skills/refunds.md"}
---
id: refunds
description: Use for refund eligibility, refund status, or processing a refund.
---
Always confirm the order ID before issuing a refund.
Never issue refunds over $500 without manager approval.
```

Load it into an agent as an on-demand capability:

```python {title="skill_from_markdown.py" test="skip"}
from pathlib import Path

import yaml

from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability


def load_skill(path: Path) -> Capability:
    _, frontmatter, body = path.read_text().split('---', 2)
    meta = yaml.safe_load(frontmatter)
    return Capability(
        id=meta['id'],
        description=meta['description'],
        instructions=body.strip(),
        defer_loading=True,
    )


agent = Agent(
    'openai-responses:gpt-5.4',
    instructions='You are a customer support assistant.',
    capabilities=[load_skill(p) for p in Path('skills').glob('*.md')],
)
```

Each file shows up in the model's catalog as its `id` plus `description`; the body is only sent once the model calls the `load_capability` tool. To go beyond instructions — add function tools, model settings, or hooks for a particular skill — subclass [`AbstractCapability`][pydantic_ai.capabilities.AbstractCapability] as in the examples above.

!!! note "Composes with"
    On-demand capabilities are orthogonal to the rest of the framework — they layer onto features you may already be using:

    - **[Tool search](tools-advanced.md#tool-search)** — capability-level `defer_loading=True` gates the whole bundle as a unit; for per-*tool* discovery, set tool-level `defer_loading=True` on a non-deferred capability or on `@agent.tool`.
    - **[MCP servers](mcp/client.md)** — the [`MCP`][pydantic_ai.capabilities.MCP] capability accepts `defer_loading=True`, hiding the server's full tool list until the model opts in.
    - **[Native tools](native-tools.md)** — [`WebSearch`][pydantic_ai.capabilities.WebSearch], [`WebFetch`][pydantic_ai.capabilities.WebFetch], [`ImageGeneration`][pydantic_ai.capabilities.ImageGeneration], and [`MCP`][pydantic_ai.capabilities.MCP] all defer the same way as function tools (see [Cache implications](#cache-implications)).
    - **[Hooks](hooks.md)** — lifecycle hooks declared on a deferred capability (or via a deferred [`Hooks`][pydantic_ai.capabilities.Hooks] capability) stay dormant until the model opts in.
    - **[Message history](message-history.md)** — loaded state round-trips through history, so persisted conversations resume in the same state (see [Resumable across runs](#resumable-across-runs)).

## Native capabilities

Pydantic AI ships with several capabilities that cover common needs:

| Capability | What it provides | Spec |
|---|---|:---:|
| [`Thinking`][pydantic_ai.capabilities.Thinking] | Enables model [thinking/reasoning](thinking.md) at configurable effort | Yes |
| [`Hooks`][pydantic_ai.capabilities.Hooks] | Decorator-based [lifecycle hook](hooks.md) registration | — |
| [`Instrumentation`][pydantic_ai.capabilities.Instrumentation] | OpenTelemetry/Logfire tracing — see [Debugging and Monitoring](logfire.md) | Yes |
| [`WebSearch`][pydantic_ai.capabilities.WebSearch] | Web search — native by default, optional [local fallback](common-tools.md#duckduckgo-search-tool) via `local='duckduckgo'` | Yes |
| [`WebFetch`][pydantic_ai.capabilities.WebFetch] | URL fetching — native by default, optional [local fallback](common-tools.md#web-fetch-tool) via `local=True` | Yes |
| [`ImageGeneration`][pydantic_ai.capabilities.ImageGeneration] | Image generation — native by default, optional subagent fallback via `fallback_model` | Yes |
| [`XSearch`][pydantic_ai.capabilities.XSearch] | X search — native on xAI, explicit subagent fallback via `fallback_model` | Yes |
| [`MCP`][pydantic_ai.capabilities.MCP] | MCP server — runs locally by default; `native=True` opts into the model provider's native MCP support | Yes |
| [`ToolSearch`][pydantic_ai.capabilities.ToolSearch] | Discovery of [deferred tools](tools-advanced.md#tool-search) — native when supported, local `search_tools` function tool otherwise | Yes |
| [`PrepareTools`][pydantic_ai.capabilities.PrepareTools] | Filters or modifies function [tool definitions](tools.md) per step | — |
| [`PrepareOutputTools`][pydantic_ai.capabilities.PrepareOutputTools] | Filters or modifies [output tool][pydantic_ai.output.ToolOutput] definitions per step | — |
| [`PrefixTools`][pydantic_ai.capabilities.PrefixTools] | Wraps a capability and prefixes its tool names | Yes |
| [`NativeTool`][pydantic_ai.capabilities.NativeTool] | Registers a [native tool](native-tools.md) with the agent | Yes |
| [`Capability`][pydantic_ai.capabilities.Capability] | Bundles instructions, function tools, and toolsets without subclassing | — |
| [`Toolset`][pydantic_ai.capabilities.Toolset] | Wraps an [`AbstractToolset`][pydantic_ai.toolsets.AbstractToolset] | — |
| [`IncludeToolReturnSchemas`][pydantic_ai.capabilities.IncludeToolReturnSchemas] | Includes return type schemas in tool definitions sent to the model | Yes |
| [`SetToolMetadata`][pydantic_ai.capabilities.SetToolMetadata] | Merges metadata key-value pairs onto selected tools | Yes |
| [`RaiseContentFilterError`][pydantic_ai.capabilities.RaiseContentFilterError] | Raises [`ContentFilterError`][pydantic_ai.exceptions.ContentFilterError] whenever a model response has `finish_reason='content_filter'` | Yes |
| [`HandleDeferredToolCalls`][pydantic_ai.capabilities.HandleDeferredToolCalls] | Resolves [deferred tool calls](deferred-tools.md#resolving-deferred-calls-with-a-handler) inline with a handler function | — |
| [`ProcessHistory`][pydantic_ai.capabilities.ProcessHistory] | Wraps a [history processor](message-history.md#processing-message-history) | — |
| [`ProcessEventStream`][pydantic_ai.capabilities.ProcessEventStream] | Forwards agent stream events to a handler function | — |
| [`ThreadExecutor`][pydantic_ai.capabilities.ThreadExecutor] | Uses a custom thread executor for [sync functions](tools-advanced.md#thread-executor-for-long-running-servers) | — |

The **Spec** column indicates whether the capability can be used in [agent specs](agent-spec.md) (YAML/JSON). Capabilities marked **—** take non-serializable arguments (callables, toolset objects) and can only be used in Python code.

```python {title="native_capabilities.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking, WebSearch

agent = Agent(
    'anthropic:claude-opus-4-6',
    instructions='You are a research assistant. Be thorough and cite sources.',
    capabilities=[
        Thinking(effort='high'),
        WebSearch(local='duckduckgo'),
    ],
)
```

[Instructions](agent.md#instructions) and [model settings](agent.md#model-run-settings) are configured directly via the `instructions` and `model_settings` parameters on `Agent` (or [`AgentSpec`][pydantic_ai.agent.spec.AgentSpec]). Capabilities are for behavior that goes beyond simple configuration — tools, lifecycle hooks, and custom extensions. They compose well, especially when you want to reuse the same configuration across multiple agents or load it from a [spec file](agent-spec.md).

### Thinking

The [`Thinking`][pydantic_ai.capabilities.Thinking] capability enables model [thinking/reasoning](thinking.md) at a configurable effort level. It's the simplest way to enable thinking across providers:

```python {title="thinking_capability.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[Thinking(effort='high')])
result = agent.run_sync('What is the capital of France?')
print(result.output)
#> The capital of France is Paris.
```

See [Thinking](thinking.md) for provider-specific details and the [unified thinking settings](thinking.md#unified-thinking-settings).

### Compaction

Provider-specific compaction capabilities manage conversation context size by compacting older messages into summaries:

| Provider | Capability | Details |
|----------|-----------|---------|
| OpenAI Responses API | [`OpenAICompaction`][pydantic_ai.models.openai.OpenAICompaction] | [OpenAI compaction](models/openai.md#message-compaction) |
| Anthropic | [`AnthropicCompaction`][pydantic_ai.models.anthropic.AnthropicCompaction] | [Anthropic compaction](models/anthropic.md#message-compaction) |

### ThreadExecutor

The [`ThreadExecutor`][pydantic_ai.capabilities.ThreadExecutor] capability provides a custom [`Executor`][concurrent.futures.Executor] for running sync tool functions and other sync callbacks in threads. This is useful in long-running servers (e.g. FastAPI) where the default ephemeral threads from [`anyio.to_thread.run_sync`][anyio.to_thread.run_sync] can accumulate under sustained load:

```python {test="skip"}
from concurrent.futures import ThreadPoolExecutor

from pydantic_ai import Agent
from pydantic_ai.capabilities import ThreadExecutor

executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix='agent-worker')
agent = Agent('openai:gpt-5.2', capabilities=[ThreadExecutor(executor)])
```

See [Thread executor for long-running servers](tools-advanced.md#thread-executor-for-long-running-servers) for more details.

### Hooks

The [`Hooks`][pydantic_ai.capabilities.Hooks] capability provides decorator-based [lifecycle hook](hooks.md) registration — the easiest way to intercept model requests, tool calls, and other events without subclassing [`AbstractCapability`][pydantic_ai.capabilities.AbstractCapability]:

```python {test="skip" lint="skip"}
from pydantic_ai import Agent, ModelRequestContext, RunContext
from pydantic_ai.capabilities import Hooks

hooks = Hooks()

@hooks.on.before_model_request
async def log_request(ctx: RunContext, request_context: ModelRequestContext) -> ModelRequestContext:
    agent_name = ctx.agent.name if ctx.agent else 'unknown'
    print(f'[{agent_name}] Sending {len(request_context.messages)} messages')
    return request_context

agent = Agent('openai:gpt-5.2', name='my_agent', capabilities=[hooks])
```

All hooks receive [`RunContext`][pydantic_ai.tools.RunContext], which provides access to the running agent via [`ctx.agent`][pydantic_ai.tools.RunContext.agent] — useful for logging, metrics, and other cross-cutting concerns that need to identify which agent is running.

Hooks can also push follow-up messages into the conversation via
[`RunContext.enqueue`][pydantic_ai.tools.RunContext.enqueue] — useful for capability
authors that need to surface an event to the model mid-run without rebuilding the
cached system prompt. See [Injecting messages mid-run](message-history.md#injecting-messages-mid-run).

See the dedicated [Hooks](hooks.md) page for the full API: decorator and constructor registration, timeouts, tool filtering, wrap hooks, per-event hooks, and more.

### Provider-adaptive tools

[`WebSearch`][pydantic_ai.capabilities.WebSearch], [`WebFetch`][pydantic_ai.capabilities.WebFetch], [`ImageGeneration`][pydantic_ai.capabilities.ImageGeneration], [`XSearch`][pydantic_ai.capabilities.XSearch], and [`MCP`][pydantic_ai.capabilities.MCP] each cover a single capability (web search, URL fetch, image generation, X search, MCP) across two implementations:

- **Native** — invoked by the model provider when the model supports it. The work happens on the provider's side (e.g. Anthropic's web search runs server-side, returning results inline).
- **Local** — runs in your Python process. Used when the model doesn't support the native tool; your code does the work (e.g. calling DuckDuckGo directly).

Because these capabilities contribute model-facing tools, their `id`, `description`, and `defer_loading` fields are meaningful: set them when that tool should stay hidden until the model loads the matching workflow with the `load_capability` tool. This includes [`ImageGeneration`][pydantic_ai.capabilities.ImageGeneration] when image generation should only be available for an image-specific workflow, whether it resolves to a native image tool or a fallback subagent tool.

Configure each side via the `native=` and `local=` kwargs. `native=` accepts `True` (use the capability's default [native tool](native-tools.md) instance), `False` (disable native), or an explicit instance like `WebSearchTool(...)` for fine-grained config. `local=` accepts `True` (the bundled local fallback), `False` (disable local), a named strategy string where supported, or any callable, [`Tool`][pydantic_ai.tools.Tool], or [`AbstractToolset`][pydantic_ai.toolsets.AbstractToolset]. Optional installs needed for the local fallback are opt-in — the capability raises a [`UserError`][pydantic_ai.exceptions.UserError] at construction (with an install hint) when you ask for a local strategy whose extra isn't installed.

```python {title="provider_adaptive_tools.py" test="skip" lint="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP, ImageGeneration, WebFetch, WebSearch, XSearch

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[
        # Native when supported; DuckDuckGo fallback on unsupported models
        WebSearch(local='duckduckgo'),
        # Native when supported; markdownify-based fallback on unsupported models
        WebFetch(local=True),
        # Native when supported; subagent fallback via `fallback_model`
        ImageGeneration(fallback_model='openai-responses:gpt-5.4'),
        # Native on xAI; on other models, explicitly delegate to an xAI model
        XSearch(fallback_model='xai:grok-4.3'),
        # Runs the MCP server locally by default; pass `native=True` to also advertise native MCP
        MCP('https://mcp.example.com/api'),
    ],
)
```

`MCP` defaults the other way from the others: because MCP carries credentials, it runs locally by default and you opt into native MCP with `native=True`. The others default to native and you opt into local with `local=`.

[`XSearch`][pydantic_ai.capabilities.XSearch] is slightly different from [`WebSearch`][pydantic_ai.capabilities.WebSearch] and [`WebFetch`][pydantic_ai.capabilities.WebFetch]: there is no default non-xAI fallback. If your agent is not running on an xAI model, set `fallback_model` explicitly to an xAI model that supports [`XSearchTool`][pydantic_ai.native_tools.XSearchTool].

Some constraint fields require the native tool (the bundled local fallback can't enforce them) — passing them locks the capability to the native path. If the model doesn't support the native tool, the capability raises a [`UserError`][pydantic_ai.exceptions.UserError].

```python {title="constraints.py" test="skip" lint="skip"}
# Limit to 5 searches per run — requires native (the local fallback can't track call count)
WebSearch(max_uses=5)

# Only fetch example.com — enforced locally when native is unavailable
WebFetch(allowed_domains=['example.com'], local=True)
```

#### WebSearch

[`WebSearch`][pydantic_ai.capabilities.WebSearch] defaults to native-only. Backed by [`WebSearchTool`][pydantic_ai.native_tools.WebSearchTool] on the native side (see [Web Search Tool](native-tools.md#web-search-tool) for provider support and configuration) — pass `native=WebSearchTool(...)` directly when you need full control over the native instance.

For the local side, pass `local='duckduckgo'` (or `local=True`) for a [DuckDuckGo](common-tools.md#duckduckgo-search-tool) fallback (requires the `duckduckgo` optional group); for other search providers, use a [Tavily][pydantic_ai.common_tools.tavily.tavily_search_tool] or [Exa][pydantic_ai.common_tools.exa.ExaSearchTool] wrapper from [`common_tools`](common-tools.md), or any callable, [`Tool`][pydantic_ai.tools.Tool], or [`AbstractToolset`][pydantic_ai.toolsets.AbstractToolset].

Native constraint fields: `search_context_size`, `user_location`, `blocked_domains`, `allowed_domains`, `max_uses`. The domain and `max_uses` constraints require native support (the shipped DuckDuckGo fallback doesn't enforce them).

```python {title="web_search.py" test="skip" lint="skip"}
from pydantic_ai.capabilities import WebSearch

# Native-only — raises on models without native web search
WebSearch()

# Native preferred; DuckDuckGo fallback (needs `pydantic-ai-slim[duckduckgo]`)
WebSearch(local='duckduckgo')

# Native preferred; custom callable as fallback
def my_search(query: str) -> str: ...
WebSearch(local=my_search)
```

#### WebFetch

[`WebFetch`][pydantic_ai.capabilities.WebFetch] defaults to native-only. Backed by [`WebFetchTool`][pydantic_ai.native_tools.WebFetchTool] on the native side (see [Web Fetch Tool](native-tools.md#web-fetch-tool) for provider support and configuration) — pass `native=WebFetchTool(...)` directly for full control.

For the local side, pass `local=True` for the bundled [markdownify-based fetch tool](common-tools.md#web-fetch-tool) (requires the `web-fetch` optional group), or any callable, [`Tool`][pydantic_ai.tools.Tool], or [`AbstractToolset`][pydantic_ai.toolsets.AbstractToolset].

Native constraint fields: `allowed_domains`, `blocked_domains`, `max_uses`, `enable_citations`, `max_content_tokens`. Only `max_uses` requires native; domain filters are enforced locally when native isn't available.

```python {title="web_fetch.py" test="skip" lint="skip"}
from pydantic_ai.capabilities import WebFetch

# Native-only — raises on models without native web fetch
WebFetch()

# Native preferred; markdownify-based fallback (needs `pydantic-ai-slim[web-fetch]`)
WebFetch(local=True)

# Domain filters enforced locally when native isn't available
WebFetch(allowed_domains=['example.com'], local=True)
```

#### ImageGeneration

[`ImageGeneration`][pydantic_ai.capabilities.ImageGeneration] defaults to native-only. Backed by [`ImageGenerationTool`][pydantic_ai.native_tools.ImageGenerationTool] on the native side (see [Image Generation Tool](native-tools.md#image-generation-tool) for provider support and configuration) — pass `native=ImageGenerationTool(...)` directly for full control.

For the local side, pass `fallback_model='…'` to delegate unsupported requests to a subagent running an image-generation-capable model (e.g. `openai-responses:gpt-5.4`), or `local=` with any callable, [`Tool`][pydantic_ai.tools.Tool], or [`AbstractToolset`][pydantic_ai.toolsets.AbstractToolset] for a custom generator.

```python {title="image_generation.py" test="skip" lint="skip"}
from pydantic_ai.capabilities import ImageGeneration

# Native-only — raises on models without native image generation
ImageGeneration()

# Native preferred; subagent fallback for unsupported models
ImageGeneration(fallback_model='openai-responses:gpt-5.4')

# Native preferred; custom callable as fallback
def my_generator(prompt: str) -> bytes: ...
ImageGeneration(local=my_generator)
```

#### MCP

[`MCP`][pydantic_ai.capabilities.MCP] is the primary entry point for [MCP](mcp/overview.md) in Pydantic AI. It runs the MCP server locally by default — keeping credentials, hooks, and tracing under your control — and supports both URL-based servers and direct client / toolset / transport inputs.

Backed by [`MCPServerTool`][pydantic_ai.native_tools.MCPServerTool] on the native side (see [MCP Server Tool](native-tools.md#mcp-server-tool) for provider support and configuration) — pass `native=MCPServerTool(...)` directly when you need full control (e.g. a different `id`, `authorization_token`, or `description` than the capability would derive). On the local side, `local=` accepts any [`MCPToolset`][pydantic_ai.mcp.MCPToolset] input (URL, `fastmcp.Client`, transport, in-process `FastMCP` server, script path, …) — non-toolset inputs are wrapped in `MCPToolset` automatically.

```python {title="mcp.py" test="skip" lint="skip"}
from pydantic_ai.capabilities import MCP
from pydantic_ai.native_tools import MCPServerTool

# URL-based MCP server, running locally (requires `pydantic-ai-slim[mcp]`)
MCP('https://mcp.example.com/api')

# Local client without a URL — pass any `MCPToolset` input
# (URL, `fastmcp.Client`, transport, in-process `FastMCP` server, script path, etc.)
MCP(local=my_fastmcp_client)

# Native preferred; URL-based local fallback
MCP('https://mcp.example.com/api', native=True)

# Strict native-only (no local — does not require the `mcp` extra)
MCP('https://mcp.example.com/api', native=True, local=False)

# Explicit native + explicit local — independent configuration on each side
# (e.g. provider-relay URL for native, direct connection for local)
MCP(
    native=MCPServerTool(
        id='public-mcp',
        url='https://relay.example.com/mcp',
        authorization_token='relay-token',
    ),
    local=my_fastmcp_client,
)
```

For lower-level access — managing the [`MCPToolset`][pydantic_ai.mcp.MCPToolset] lifecycle directly, advanced transport / client configuration, or using MCP servers without going through a capability — see the [MCP documentation](mcp/overview.md).

#### Building your own

All four capabilities are subclasses of [`NativeOrLocalTool`][pydantic_ai.capabilities.NativeOrLocalTool], which you can use directly or subclass to build your own provider-adaptive tools. For example, to pair [`CodeExecutionTool`][pydantic_ai.native_tools.CodeExecutionTool] with a local fallback:

```python {title="custom_native_or_local.py" test="skip" lint="skip"}
from pydantic_ai.native_tools import CodeExecutionTool
from pydantic_ai.capabilities import NativeOrLocalTool

cap = NativeOrLocalTool(native=CodeExecutionTool(), local=my_local_executor)
```

### ToolSearch

The [`ToolSearch`][pydantic_ai.capabilities.ToolSearch] capability handles discovery of tools marked with `defer_loading=True`, so agents with large toolsets only pay tokens for the tools the model needs. Like the [provider-adaptive tools](#provider-adaptive-tools) above, it picks the best path for the active model — native server-executed search on Anthropic and OpenAI Responses, a local `search_tools` function tool elsewhere — and is auto-injected into every agent with zero overhead when no deferred tools exist.

Pass an explicit [`ToolSearch`][pydantic_ai.capabilities.ToolSearch] to pick a specific [`strategy`][pydantic_ai.capabilities.ToolSearch.strategy] (`'keywords'`, `'bm25'`, `'regex'`, or a custom callable) or tune the local fallback:

```python {title="tool_search_capability.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import ToolSearch

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[ToolSearch(strategy='keywords')])
```

See [Tool Search](tools-advanced.md#tool-search) for when to reach for it, the full strategy table, and provider support details.

### PrepareTools and PrepareOutputTools

[`PrepareTools`][pydantic_ai.capabilities.PrepareTools] and [`PrepareOutputTools`][pydantic_ai.capabilities.PrepareOutputTools] wrap a [`ToolsPrepareFunc`][pydantic_ai.tools.ToolsPrepareFunc] as a capability, for filtering or modifying [tool definitions](tools.md) per step. `PrepareTools` handles function tools; `PrepareOutputTools` handles [output tools][pydantic_ai.output.ToolOutput].

```python {title="prepare_tools_native.py"}
from pydantic_ai import Agent, RunContext, ToolDefinition
from pydantic_ai.capabilities import PrepareTools


async def hide_dangerous(ctx: RunContext, tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
    return [td for td in tool_defs if not td.name.startswith('delete_')]


agent = Agent('openai:gpt-5.2', capabilities=[PrepareTools(hide_dangerous)])


@agent.tool_plain
def delete_file(path: str) -> str:
    """Delete a file."""
    return f'deleted {path}'


@agent.tool_plain
def read_file(path: str) -> str:
    """Read a file."""
    return f'contents of {path}'


result = agent.run_sync('hello')
# The model only sees `read_file`, not `delete_file`
```

For more complex tool preparation logic, see [Tool preparation](#tool-preparation) under lifecycle hooks.

### PrefixTools

[`PrefixTools`][pydantic_ai.capabilities.PrefixTools] wraps another capability and prefixes all of its tool names, useful for namespacing when composing multiple capabilities that might have conflicting tool names:

```python {title="prefix_tools_example.py" test="skip" lint="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP, PrefixTools

agent = Agent(
    'openai:gpt-5.2',
    capabilities=[
        PrefixTools(MCP(url='https://api1.example.com', native=True), prefix='api1'),
        PrefixTools(MCP(url='https://api2.example.com', native=True), prefix='api2'),
    ],
)
```

Every [`AbstractCapability`][pydantic_ai.capabilities.AbstractCapability] has a convenience method [`prefix_tools`][pydantic_ai.capabilities.AbstractCapability.prefix_tools] that returns a [`PrefixTools`][pydantic_ai.capabilities.PrefixTools] wrapper:

```python {title="prefix_convenience.py" test="skip" lint="skip"}
MCP(url='https://mcp.example.com/api', native=True).prefix_tools('mcp')
```

### IncludeToolReturnSchemas

[`IncludeToolReturnSchemas`][pydantic_ai.capabilities.IncludeToolReturnSchemas] includes return type schemas in tool definitions sent to the model. For models that natively support return schemas (e.g. Google Gemini), the schema is passed as a structured field in the API request. For other models, it is injected into the tool description as JSON text.

```python {title="include_return_schemas.py" lint="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import IncludeToolReturnSchemas
from pydantic_ai.models.test import TestModel


test_model = TestModel()
agent = Agent(test_model, capabilities=[IncludeToolReturnSchemas()])


@agent.tool_plain
def get_temperature(city: str) -> float:
    """Get the temperature for a city."""
    return 21.0


result = agent.run_sync('What is the temperature in Paris?')
params = test_model.last_model_request_parameters
assert params is not None
td = params.function_tools[0]
assert td.include_return_schema is True
```

_(This example is complete, it can be run "as is")_

Use the `tools` parameter to select which tools should include return schemas. It accepts a list of tool names, a metadata dict for matching, or a callable predicate:

```python {title="include_return_schemas_selective.py" lint="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import IncludeToolReturnSchemas
from pydantic_ai.models.test import TestModel


test_model = TestModel()
agent = Agent(
    test_model,
    capabilities=[IncludeToolReturnSchemas(tools=['get_temperature'])],
)


@agent.tool_plain
def get_temperature(city: str) -> float:
    """Get the temperature for a city."""
    return 21.0


@agent.tool_plain
def get_greeting(name: str) -> str:
    """Get a greeting."""
    return f'Hello, {name}!'


result = agent.run_sync('Hello')
params = test_model.last_model_request_parameters
assert params is not None
temp_tool = next(t for t in params.function_tools if t.name == 'get_temperature')
greet_tool = next(t for t in params.function_tools if t.name == 'get_greeting')
assert temp_tool.include_return_schema is True
assert greet_tool.include_return_schema is None
```

_(This example is complete, it can be run "as is")_

The same effect can be achieved at the toolset level using [`.include_return_schemas()`][pydantic_ai.toolsets.AbstractToolset.include_return_schemas] — see [toolset composition](toolsets.md#including-return-schemas).

### SetToolMetadata

[`SetToolMetadata`][pydantic_ai.capabilities.SetToolMetadata] merges metadata key-value pairs onto selected tools. This is useful for tagging tools with configuration that other capabilities or custom logic can inspect:

```python {title="set_tool_metadata.py" lint="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import SetToolMetadata
from pydantic_ai.models.test import TestModel


test_model = TestModel()
agent = Agent(
    test_model,
    capabilities=[SetToolMetadata(tools=['search'], sensitive=True)],
)


@agent.tool_plain
def search(query: str) -> str:
    """Search for information."""
    return f'Results for: {query}'


@agent.tool_plain
def greet(name: str) -> str:
    """Greet someone."""
    return f'Hello, {name}!'


result = agent.run_sync('Search for pydantic')
params = test_model.last_model_request_parameters
assert params is not None
search_tool = next(t for t in params.function_tools if t.name == 'search')
greet_tool = next(t for t in params.function_tools if t.name == 'greet')
assert search_tool.metadata is not None and search_tool.metadata.get('sensitive') is True
assert greet_tool.metadata is None or greet_tool.metadata.get('sensitive') is None
```

_(This example is complete, it can be run "as is")_

The same effect can be achieved at the toolset level using [`.with_metadata()`][pydantic_ai.toolsets.AbstractToolset.with_metadata] — see [toolset composition](toolsets.md#setting-tool-metadata).

### RaiseContentFilterError

[`RaiseContentFilterError`][pydantic_ai.capabilities.RaiseContentFilterError] opts into treating any model response with `finish_reason='content_filter'` as a [`ContentFilterError`][pydantic_ai.exceptions.ContentFilterError], even when the provider returns partial text or refusal text:

```python {title="raise_content_filter_error.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import RaiseContentFilterError
from pydantic_ai.exceptions import ContentFilterError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel


def filtered_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(
        parts=[TextPart(content='I cannot help with that.')],
        finish_reason='content_filter',
        provider_details={'finish_reason': 'content_filter'},
    )


agent = Agent(FunctionModel(filtered_response), capabilities=[RaiseContentFilterError()])

try:
    agent.run_sync('Tell me how to make a weapon.')
except ContentFilterError as exc:
    print(exc.message)
    #> Content filter triggered. Finish reason: 'content_filter'
```

_(This example is complete, it can be run "as is")_

By default, Pydantic AI only raises [`ContentFilterError`][pydantic_ai.exceptions.ContentFilterError] when a `content_filter` response is *empty*: if the provider returns partial text or refusal text alongside `finish_reason='content_filter'`, that text becomes ordinary agent output and no error is raised (see [finish reason handling](models/overview.md#finish-reason-example)). This capability extends the check to *every* `content_filter` response, so partial and refusal text raise too. When it raises, the full [`ModelResponse`][pydantic_ai.messages.ModelResponse] is serialized into [`ContentFilterError.body`][pydantic_ai.exceptions.UnexpectedModelBehavior.body] so the partial text remains inspectable.

### ReinjectSystemPrompt

[`ReinjectSystemPrompt`][pydantic_ai.capabilities.ReinjectSystemPrompt] ensures the agent's configured [`system_prompt`](agent.md#system-prompts) is at the head of the first [`ModelRequest`][pydantic_ai.messages.ModelRequest] on every model request. By default, if any [`SystemPromptPart`][pydantic_ai.messages.SystemPromptPart] is already present in the history, the capability is a no-op (so multi-agent handoff and user-managed system prompts remain authoritative). Set `replace_existing=True` to instead strip any existing `SystemPromptPart`s before prepending the agent's configured prompt — useful when the history comes from an untrusted source and the server's prompt must win.

Useful when `message_history` comes from a source that doesn't round-trip system prompts — UI frontends, database persistence layers, conversation compaction pipelines. Without this capability, an agent configured with a `system_prompt` will silently run without it if the history doesn't already include one.

```python {title="reinject_system_prompt.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import ReinjectSystemPrompt
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

agent = Agent('test', system_prompt='You are a helpful assistant.', capabilities=[ReinjectSystemPrompt()])

# History that's missing the system prompt (e.g. reconstructed from a UI frontend).
history = [
    ModelRequest(parts=[UserPromptPart(content='Hi')]),
    ModelResponse(parts=[TextPart(content='Hello!')]),
]

# Without the capability, the agent would run without its configured system prompt.
# With the capability, the system prompt is reinjected at the head of the first request.
result = agent.run_sync('Follow up', message_history=history)
first_request = result.all_messages()[0]
assert isinstance(first_request, ModelRequest)
assert first_request.parts[0].content == 'You are a helpful assistant.'
```

_(This example is complete, it can be run "as is")_

The [UI adapters](ui/ag-ui.md) (AG-UI, Vercel AI) automatically add this capability with `replace_existing=True` in their `manage_system_prompt='server'` mode.

## Building custom capabilities

To build your own capability, subclass [`AbstractCapability`][pydantic_ai.capabilities.AbstractCapability] and override the methods you need. There are two categories: **configuration methods** that are called at agent construction (except [`get_wrapper_toolset`][pydantic_ai.capabilities.AbstractCapability.get_wrapper_toolset] which is called per-run), and **lifecycle hooks** that fire during each run.

Custom capability classes can be plain classes or dataclasses. The shared metadata attributes — [`id`][pydantic_ai.capabilities.AbstractCapability.id], [`description`][pydantic_ai.capabilities.AbstractCapability.description], and [`defer_loading`][pydantic_ai.capabilities.AbstractCapability.defer_loading] — are optional declarations on the capability object for always-available capabilities. If `id` is omitted there, Pydantic AI derives a run-local id from the class name and disambiguates duplicates within the run. Deferred capabilities require an explicit stable `id`.

```python {title="custom_capability_plain.py"}
from typing import Any

from pydantic_ai.capabilities import AbstractCapability


class MyCapability(AbstractCapability[Any]):
    """A custom capability."""
```

Use a dataclass when you want generated constructor parameters for your own configuration fields, or for the shared metadata fields:

```python {title="custom_capability_dataclass.py"}
from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability


@dataclass
class MyCapability(AbstractCapability[None]):
    label: str
```

If you define a custom `__init__`, set only the metadata you want to expose. There is no `super().__init__()` or `__post_init__()` requirement:

```python {title="custom_capability_init.py"}
from pydantic_ai.capabilities import AbstractCapability


class MyCapability(AbstractCapability[None]):
    def __init__(
        self,
        label: str,
        *,
        id: str | None = None,
        description: str | None = None,
        defer_loading: bool = False,
    ) -> None:
        self.id = id
        self.description = description
        self.defer_loading = defer_loading
        self.label = label
```

When [`defer_loading=True`](#on-demand-capabilities), provide a stable explicit `id`; history replay depends on it, and Pydantic AI rejects deferred capabilities without one. For always-available capabilities, omitting `id` still derives a run-local id from the class name.

### Providing tools

A capability that provides tools returns a [toolset](toolsets.md) from [`get_toolset`][pydantic_ai.capabilities.AbstractCapability.get_toolset]. This can be a pre-built [`AbstractToolset`][pydantic_ai.toolsets.AbstractToolset] instance, or a callable that receives [`RunContext`][pydantic_ai.tools.RunContext] and returns one dynamically:

```python {title="custom_capability_tools.py"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AgentToolset, FunctionToolset

math_toolset = FunctionToolset()


@math_toolset.tool_plain
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@math_toolset.tool_plain
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@dataclass
class MathTools(AbstractCapability[Any]):
    """Provides basic math operations."""

    def get_toolset(self) -> AgentToolset[Any] | None:
        return math_toolset


agent = Agent('openai:gpt-5.2', capabilities=[MathTools()])
result = agent.run_sync('What is 2 + 3?')
print(result.output)
#> The answer is 5.0
```

For [native tools](native-tools.md), override [`get_native_tools`][pydantic_ai.capabilities.AbstractCapability.get_native_tools] to return a sequence of [`AgentNativeTool`][pydantic_ai.tools.AgentNativeTool] instances (which includes both [`AbstractNativeTool`][pydantic_ai.native_tools.AbstractNativeTool] objects and callables that receive [`RunContext`][pydantic_ai.tools.RunContext]).

#### Toolset wrapping

[`get_wrapper_toolset`][pydantic_ai.capabilities.AbstractCapability.get_wrapper_toolset] lets a capability wrap the agent's entire assembled toolset with a [`WrapperToolset`](toolsets.md#changing-tool-execution). This is more powerful than providing tools — it can intercept tool execution, add logging, or apply cross-cutting behavior.

The wrapper receives the combined non-output toolset (after the [`prepare_tools`](#tool-preparation) hook has wrapped it). Output tools are added separately and are not affected.

```python {title="wrapper_toolset_example.py"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets.wrapper import WrapperToolset


@dataclass
class LoggingToolset(WrapperToolset[Any]):
    """Logs all tool calls."""

    async def call_tool(
        self, tool_name: str, tool_args: dict[str, Any], *args: Any, **kwargs: Any
    ) -> Any:
        print(f'  Calling tool: {tool_name}')
        return await super().call_tool(tool_name, tool_args, *args, **kwargs)


@dataclass
class LogToolCalls(AbstractCapability[Any]):
    """Wraps the agent's toolset to log all tool calls."""

    def get_wrapper_toolset(self, toolset: AbstractToolset[Any]) -> AbstractToolset[Any]:
        return LoggingToolset(wrapped=toolset)


agent = Agent('openai:gpt-5.2', capabilities=[LogToolCalls()])


@agent.tool_plain
def greet(name: str) -> str:
    """Greet someone."""
    return f'Hello, {name}!'


result = agent.run_sync('hello')
# Tool calls are logged as they happen
```

!!! note
    `get_wrapper_toolset` wraps the non-output *toolset* once per run (during toolset assembly). The [`prepare_tools`](#tool-preparation) and [`prepare_output_tools`](#tool-preparation) hooks also flow through `PreparedToolset` wrappers, so all three integrate at the toolset level — `get_wrapper_toolset` runs around `prepare_tools` (it sees the prepared defs), and `prepare_output_tools` wraps the output toolset independently.

### Providing instructions

[`get_instructions`][pydantic_ai.capabilities.AbstractCapability.get_instructions] adds [instructions](agent.md#instructions) to the agent. Since it's called once at agent construction, return a callable if you need dynamic values:

```python {title="custom_capability_config.py"}
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability


@dataclass
class KnowsCurrentTime(AbstractCapability[Any]):
    """Tells the agent what time it is."""

    def get_instructions(self):
        def _get_time(ctx: RunContext[Any]) -> str:
            return f'The current date and time is {datetime.now().isoformat()}.'

        return _get_time


agent = Agent('openai:gpt-5.2', capabilities=[KnowsCurrentTime()])
result = agent.run_sync('What time is it?')
print(result.output)
#> The current time is 3:45 PM.
```

Instructions can also use [template strings](agent-spec.md#template-strings) (`TemplateStr('Hello {{name}}')`) for Handlebars-style templates rendered against the agent's [dependencies](dependencies.md). In Python code, a callable with [`RunContext`][pydantic_ai.tools.RunContext] is generally preferred for IDE autocomplete.

### Providing model settings

[`get_model_settings`][pydantic_ai.capabilities.AbstractCapability.get_model_settings] returns [model settings](agent.md#model-run-settings) as a dict or a callable for per-step settings.

When model settings need to vary per step — for example, enabling thinking only on retry, or forcing a specific [`tool_choice`](tools-advanced.md#dynamic-tool-choice-via-capabilities) until a tool has been called — return a callable:

```python {title="dynamic_settings.py"}
from dataclasses import dataclass

from pydantic_ai import Agent, ModelSettings, RunContext
from pydantic_ai.capabilities import AbstractCapability


@dataclass
class ThinkingOnRetry(AbstractCapability):
    """Enables thinking mode when the agent is retrying."""

    def get_model_settings(self):
        def resolve(ctx: RunContext) -> ModelSettings:
            if ctx.run_step > 1:
                return ModelSettings(thinking='high')
            return ModelSettings()

        return resolve


agent = Agent('openai:gpt-5.2', capabilities=[ThinkingOnRetry()])
result = agent.run_sync('hello')
print(result.output)
#> Hello! How can I help you today?
```

The callable receives a [`RunContext`][pydantic_ai.tools.RunContext] where `ctx.model_settings` contains the merged result of all layers resolved before this capability (model defaults and agent-level settings).

### Configuration methods reference

| Method | Return type | Purpose |
|---|---|---|
| [`get_toolset()`][pydantic_ai.capabilities.AbstractCapability.get_toolset] | [`AgentToolset`][pydantic_ai.toolsets.AgentToolset] ` \| None` | A [toolset](toolsets.md) to register (or a callable for [dynamic toolsets](toolsets.md#dynamically-building-a-toolset)) |
| [`get_native_tools()`][pydantic_ai.capabilities.AbstractCapability.get_native_tools] | `Sequence[`[`AgentNativeTool`][pydantic_ai.tools.AgentNativeTool]`]` | [Native tools](native-tools.md) to register (including callables) |
| [`get_wrapper_toolset()`][pydantic_ai.capabilities.AbstractCapability.get_wrapper_toolset] | [`AbstractToolset`][pydantic_ai.toolsets.AbstractToolset] ` \| None` | [Wrap the agent's assembled toolset](#toolset-wrapping) |
| [`get_instructions()`][pydantic_ai.capabilities.AbstractCapability.get_instructions] | [`AgentInstructions`][pydantic_ai._instructions.AgentInstructions] ` \| None` | [Instructions](agent.md#instructions) (static strings, [template strings](agent-spec.md#template-strings), or callables) |
| [`get_model_settings()`][pydantic_ai.capabilities.AbstractCapability.get_model_settings] | [`AgentModelSettings`][pydantic_ai.agent.abstract.AgentModelSettings] ` \| None` | [Model settings](agent.md#model-run-settings) dict, or a callable for per-step settings |

### Hooking into the lifecycle

Capabilities can hook into five lifecycle points, each with up to four variants:

* **`before_*`** — fires before the action, can modify inputs
* **`after_*`** — fires after the action succeeds (in reverse capability order), can modify outputs
* **`wrap_*`** — full middleware control: receives a `handler` callable and decides whether/how to call it
* **`on_*_error`** — fires when the action fails (after `wrap_*` has had its chance to recover), can observe, transform, or recover from errors

!!! tip
    For quick, application-level hooks without subclassing, use the [`Hooks`](hooks.md) capability instead.

#### Run hooks

| Hook | Signature | Purpose |
|---|---|---|
| [`before_run`][pydantic_ai.capabilities.AbstractCapability.before_run] | `(ctx: RunContext) -> None` | Observe-only notification that a run is starting |
| [`after_run`][pydantic_ai.capabilities.AbstractCapability.after_run] | `(ctx: RunContext, *, result: AgentRunResult) -> AgentRunResult` | Modify the final result |
| [`wrap_run`][pydantic_ai.capabilities.AbstractCapability.wrap_run] | `(ctx: RunContext, *, handler: WrapRunHandler) -> AgentRunResult` | Wrap the entire run |
| [`on_run_error`][pydantic_ai.capabilities.AbstractCapability.on_run_error] | `(ctx: RunContext, *, error: BaseException) -> AgentRunResult` | Handle run errors (see [error hooks](#error-hooks)) |

`wrap_run` supports error recovery: if `handler()` raises and `wrap_run` catches the exception and returns a result instead, the error is suppressed and the recovery result is used. This works with both [`agent.run()`][pydantic_ai.agent.AbstractAgent.run] and [`agent.iter()`][pydantic_ai.agent.Agent.iter].

#### Node hooks

| Hook | Signature | Purpose |
|---|---|---|
| [`before_node_run`][pydantic_ai.capabilities.AbstractCapability.before_node_run] | `(ctx: RunContext, *, node: AgentNode) -> AgentNode` | Observe or replace the node before execution |
| [`after_node_run`][pydantic_ai.capabilities.AbstractCapability.after_node_run] | `(ctx: RunContext, *, node: AgentNode, result: NodeResult) -> NodeResult` | Modify the result (next node or `End`) |
| [`wrap_node_run`][pydantic_ai.capabilities.AbstractCapability.wrap_node_run] | `(ctx: RunContext, *, node: AgentNode, handler: WrapNodeRunHandler) -> NodeResult` | Wrap each graph node execution |
| [`on_node_run_error`][pydantic_ai.capabilities.AbstractCapability.on_node_run_error] | `(ctx: RunContext, *, node: AgentNode, error: Exception) -> NodeResult` | Handle node errors (see [error hooks](#error-hooks)) |

[`wrap_node_run`][pydantic_ai.capabilities.AbstractCapability.wrap_node_run] fires for every node in the [agent graph](agent.md#iterating-over-an-agents-graph) ([`UserPromptNode`][pydantic_ai.UserPromptNode], [`ModelRequestNode`][pydantic_ai.ModelRequestNode], [`CallToolsNode`][pydantic_ai.CallToolsNode]). Override this to observe node transitions, add per-step logging, or modify graph progression:

!!! note
    `wrap_node_run` hooks are called automatically by [`agent.run()`][pydantic_ai.agent.AbstractAgent.run], [`agent.run_stream()`][pydantic_ai.agent.AbstractAgent.run_stream], and [`agent_run.next()`][pydantic_ai.run.AgentRun.next]. However, they are **not** called when iterating with bare `async for node in agent_run:` over [`agent.iter()`][pydantic_ai.agent.Agent.iter], since that uses the graph run's internal iteration. Always use `agent_run.next(node)` to advance the run if you need `wrap_node_run` hooks to fire.

```python {title="node_logging_example.py"}
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import (
    AbstractCapability,
    AgentNode,
    NodeResult,
    WrapNodeRunHandler,
)


@dataclass
class NodeLogger(AbstractCapability[Any]):
    """Logs each node that executes during a run."""

    nodes: list[str] = field(default_factory=list)

    async def wrap_node_run(
        self, ctx: RunContext[Any], *, node: AgentNode[Any], handler: WrapNodeRunHandler[Any]
    ) -> NodeResult[Any]:
        self.nodes.append(type(node).__name__)
        return await handler(node)


logger = NodeLogger()
agent = Agent('openai:gpt-5.2', capabilities=[logger])
agent.run_sync('hello')
print(logger.nodes)
#> ['UserPromptNode', 'ModelRequestNode', 'CallToolsNode']
```

You can also use `wrap_node_run` to modify graph progression — for example, limiting the number of model requests per run:

```python {title="node_modification_example.py" test="skip" lint="skip"}
from dataclasses import dataclass
from typing import Any

from pydantic_graph import End

from pydantic_ai import ModelRequestNode, RunContext
from pydantic_ai.capabilities import AbstractCapability, AgentNode, NodeResult, WrapNodeRunHandler
from pydantic_ai.result import FinalResult


@dataclass
class MaxModelRequests(AbstractCapability[Any]):
    """Limits the number of model requests per run by ending early."""

    max_requests: int = 5
    count: int = 0

    async def for_run(self, ctx: RunContext[Any]) -> 'MaxModelRequests':
        return MaxModelRequests(max_requests=self.max_requests)  # fresh per run

    async def wrap_node_run(
        self, ctx: RunContext[Any], *, node: AgentNode[Any], handler: WrapNodeRunHandler[Any]
    ) -> NodeResult[Any]:
        if isinstance(node, ModelRequestNode):
            self.count += 1
            if self.count > self.max_requests:
                return End(FinalResult(output='Max model requests reached'))
        return await handler(node)
```

See [Iterating Over an Agent's Graph](agent.md#iterating-over-an-agents-graph) for more about the agent graph and its node types.

#### Model request hooks

| Hook | Signature | Purpose |
|---|---|---|
| [`before_model_request`][pydantic_ai.capabilities.AbstractCapability.before_model_request] | `(ctx: RunContext, request_context: ModelRequestContext) -> ModelRequestContext` | Modify messages, settings, parameters, or model before the model call |
| [`after_model_request`][pydantic_ai.capabilities.AbstractCapability.after_model_request] | `(ctx: RunContext, *, request_context: ModelRequestContext, response: ModelResponse) -> ModelResponse` | Modify the model's response |
| [`wrap_model_request`][pydantic_ai.capabilities.AbstractCapability.wrap_model_request] | `(ctx: RunContext, *, request_context: ModelRequestContext, handler: WrapModelRequestHandler) -> ModelResponse` | Wrap the model call |
| [`on_model_request_error`][pydantic_ai.capabilities.AbstractCapability.on_model_request_error] | `(ctx: RunContext, *, request_context: ModelRequestContext, error: Exception) -> ModelResponse` | Handle model request errors (see [error hooks](#error-hooks)) |

[`ModelRequestContext`][pydantic_ai.models.ModelRequestContext] bundles `model`, `messages`, `model_settings`, and `model_request_parameters` into a single object, making the signature future-proof. To swap the model for a given request, set `request_context.model` to a different [`Model`][pydantic_ai.models.Model] instance.

To skip the model call entirely and provide a replacement response, raise [`SkipModelRequest(response)`][pydantic_ai.exceptions.SkipModelRequest] from `before_model_request` or `wrap_model_request`.

#### Tool hooks

Tool processing has two phases: **validation** (parsing and validating the model's JSON arguments against the tool's schema) and **execution** (running the tool function). Each phase has its own hooks.

All tool hooks receive a `tool_def` parameter with the [`ToolDefinition`][pydantic_ai.tools.ToolDefinition].

**Validation hooks** — `args` is the raw `str | dict[str, Any]` from the model before validation, or the validated `dict[str, Any]` after:

| Hook | Signature | Purpose |
|---|---|---|
| [`before_tool_validate`][pydantic_ai.capabilities.AbstractCapability.before_tool_validate] | `(ctx: RunContext, *, call: ToolCallPart, tool_def: ToolDefinition, args: RawToolArgs) -> RawToolArgs` | Modify raw args before validation (e.g. JSON repair) |
| [`after_tool_validate`][pydantic_ai.capabilities.AbstractCapability.after_tool_validate] | `(ctx: RunContext, *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs) -> ValidatedToolArgs` | Modify validated args |
| [`wrap_tool_validate`][pydantic_ai.capabilities.AbstractCapability.wrap_tool_validate] | `(ctx: RunContext, *, call: ToolCallPart, tool_def: ToolDefinition, args: RawToolArgs, handler: WrapToolValidateHandler) -> ValidatedToolArgs` | Wrap the validation step |
| [`on_tool_validate_error`][pydantic_ai.capabilities.AbstractCapability.on_tool_validate_error] | `(ctx: RunContext, *, call: ToolCallPart, tool_def: ToolDefinition, args: RawToolArgs, error: Exception) -> ValidatedToolArgs` | Handle validation errors (see [error hooks](#error-hooks)) |

To skip validation and provide pre-validated args, raise [`SkipToolValidation(args)`][pydantic_ai.exceptions.SkipToolValidation] from `before_tool_validate` or `wrap_tool_validate`.

**Execution hooks** — `args` is always the validated `dict[str, Any]`:

| Hook | Signature | Purpose |
|---|---|---|
| [`before_tool_execute`][pydantic_ai.capabilities.AbstractCapability.before_tool_execute] | `(ctx: RunContext, *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs) -> ValidatedToolArgs` | Modify args before execution |
| [`after_tool_execute`][pydantic_ai.capabilities.AbstractCapability.after_tool_execute] | `(ctx: RunContext, *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs, result: Any) -> Any` | Modify execution result |
| [`wrap_tool_execute`][pydantic_ai.capabilities.AbstractCapability.wrap_tool_execute] | `(ctx: RunContext, *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs, handler: WrapToolExecuteHandler) -> Any` | Wrap execution |
| [`on_tool_execute_error`][pydantic_ai.capabilities.AbstractCapability.on_tool_execute_error] | `(ctx: RunContext, *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs, error: Exception) -> Any` | Handle execution errors (see [error hooks](#error-hooks)) |

To skip execution and provide a replacement result, raise [`SkipToolExecution(result)`][pydantic_ai.exceptions.SkipToolExecution] from `before_tool_execute` or `wrap_tool_execute`.

#### Output hooks

Like tool processing, [output](output.md) processing has two phases: **validation** (parsing the model's raw output against the output schema) and **processing** (extracting the value and calling any [output function](output.md#output-functions)). Each phase has its own hooks.

All output hooks receive an `output_context` parameter with [`OutputContext`][pydantic_ai.capabilities.OutputContext] (mode, output type, schema info, and tool call details for [tool output](output.md#tool-output)).

**Validate hooks** fire only for structured output that requires parsing (prompted, native, tool, union output). They do not fire for plain text or image output. **Process hooks** fire for **all output types** including text, structured, and image output. For [tool output](output.md#tool-output), only output hooks fire — tool hooks are skipped entirely.

**Validation hooks** — fire for structured output only; `output` is `str` (raw text) or `dict` (tool args):

| Hook | Signature | Purpose |
|---|---|---|
| [`before_output_validate`][pydantic_ai.capabilities.AbstractCapability.before_output_validate] | `(ctx, *, output_context, output: RawOutput) -> RawOutput` | Modify raw output before validation (e.g. JSON repair) |
| [`after_output_validate`][pydantic_ai.capabilities.AbstractCapability.after_output_validate] | `(ctx, *, output_context, output: Any) -> Any` | Modify validated output |
| [`wrap_output_validate`][pydantic_ai.capabilities.AbstractCapability.wrap_output_validate] | `(ctx, *, output_context, output: RawOutput, handler) -> Any` | Wrap the validation step |
| [`on_output_validate_error`][pydantic_ai.capabilities.AbstractCapability.on_output_validate_error] | `(ctx, *, output_context, output: RawOutput, error: ValidationError \| ModelRetry) -> Any` | Handle validation errors (see [error hooks](#error-hooks)) |

**Processing hooks** — fire for all output types; `output` is the validated/raw output. Output validators ([`@agent.output_validator`][pydantic_ai.Agent.output_validator]) run inside the processing pipeline (within `wrap_output_process`), so `after_output_process` sees the fully validated result:

| Hook | Signature | Purpose |
|---|---|---|
| [`before_output_process`][pydantic_ai.capabilities.AbstractCapability.before_output_process] | `(ctx, *, output_context, output: Any) -> Any` | Modify output before processing |
| [`after_output_process`][pydantic_ai.capabilities.AbstractCapability.after_output_process] | `(ctx, *, output_context, output: Any) -> Any` | Modify processed result |
| [`wrap_output_process`][pydantic_ai.capabilities.AbstractCapability.wrap_output_process] | `(ctx, *, output_context, output: Any, handler) -> Any` | Wrap processing |
| [`on_output_process_error`][pydantic_ai.capabilities.AbstractCapability.on_output_process_error] | `(ctx, *, output_context, output: Any, error: Exception) -> Any` | Handle processing errors (see [error hooks](#error-hooks)) |

Output validate and process hooks can raise [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] to ask the model to try again with a custom message — the same pattern used in [output functions](output.md#output-functions) and [output validators](output.md#output-validator-functions). See [Triggering retries with `ModelRetry`](hooks.md#triggering-retries-with-modelretry) for the full pattern.

#### Tool preparation

Capabilities can filter or modify which tool definitions the model sees on each step via two hooks:

- [`prepare_tools`][pydantic_ai.capabilities.AbstractCapability.prepare_tools] — receives **function** tools only. Use this for filtering or modifications to tools the model can call directly.
- [`prepare_output_tools`][pydantic_ai.capabilities.AbstractCapability.prepare_output_tools] — receives [output tools][pydantic_ai.output.ToolOutput] only, with `ctx.retry`/`ctx.max_retries` reflecting the **output** side of the agent retry budget, matching the [output hook](#output-hooks) lifecycle.

Both hooks operate at the toolset level — the result flows into both the model's request parameters and `ToolManager.tools`, so filtering also blocks tool execution.

```python {title="prepare_tools_example.py"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, RunContext, ToolDefinition
from pydantic_ai.capabilities import AbstractCapability


@dataclass
class HideDangerousTools(AbstractCapability[Any]):
    """Hides tools matching certain name prefixes from the model."""

    hidden_prefixes: tuple[str, ...] = ('delete_', 'drop_')

    async def prepare_tools(
        self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        return [td for td in tool_defs if not any(td.name.startswith(p) for p in self.hidden_prefixes)]


agent = Agent('openai:gpt-5.2', capabilities=[HideDangerousTools()])


@agent.tool_plain
def delete_file(path: str) -> str:
    """Delete a file."""
    return f'deleted {path}'


@agent.tool_plain
def read_file(path: str) -> str:
    """Read a file."""
    return f'contents of {path}'


result = agent.run_sync('hello')
# The model only sees `read_file`, not `delete_file`
```

For simple cases, the built-in [`PrepareTools`][pydantic_ai.capabilities.PrepareTools] / [`PrepareOutputTools`][pydantic_ai.capabilities.PrepareOutputTools] capabilities wrap a callable without a custom subclass.

#### Event stream hook

For runs with event streaming ([`run_stream_events`][pydantic_ai.agent.AbstractAgent.run_stream_events], [`event_stream_handler`][pydantic_ai.agent.Agent.__init__], [UI event streams](ui/overview.md)), capabilities can observe or transform the event stream:

| Hook | Signature | Purpose |
|---|---|---|
| [`wrap_run_event_stream`][pydantic_ai.capabilities.AbstractCapability.wrap_run_event_stream] | `(ctx: RunContext, *, stream: AsyncIterable[AgentStreamEvent]) -> AsyncIterable[AgentStreamEvent]` | Observe, filter, or transform streamed events |

```python {title="event_stream_example.py"}
from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import Any

from pydantic_ai import AgentStreamEvent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    PartStartEvent,
    TextPart,
    ToolCallEvent,
    ToolResultEvent,
)


@dataclass
class StreamAuditor(AbstractCapability[Any]):
    """Logs tool calls and text output during streamed runs."""

    async def wrap_run_event_stream(
        self,
        ctx: RunContext[Any],
        *,
        stream: AsyncIterable[AgentStreamEvent],
    ) -> AsyncIterable[AgentStreamEvent]:
        async for event in stream:
            if isinstance(event, ToolCallEvent):
                print(f'Tool called: {event.part.tool_name}')
            elif isinstance(event, ToolResultEvent):
                print(f'Tool result: {event.part.content!r}')
            elif isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                print(f'Text: {event.part.content!r}')
            yield event
```

Matching against [`ToolCallEvent`][pydantic_ai.messages.ToolCallEvent] and [`ToolResultEvent`][pydantic_ai.messages.ToolResultEvent] handles both function tool calls ([`FunctionToolCallEvent`][pydantic_ai.messages.FunctionToolCallEvent] / [`FunctionToolResultEvent`][pydantic_ai.messages.FunctionToolResultEvent]) and output tool calls ([`OutputToolCallEvent`][pydantic_ai.messages.OutputToolCallEvent] / [`OutputToolResultEvent`][pydantic_ai.messages.OutputToolResultEvent]). Match against the specific subclass when you need to treat them differently. [Deferred tool calls](deferred-tools.md#observing-deferred-tool-calls-in-a-stream) additionally emit batch-level [`DeferredToolRequestsEvent`][pydantic_ai.messages.DeferredToolRequestsEvent] / [`DeferredToolResultsEvent`][pydantic_ai.messages.DeferredToolResultsEvent].

For building web UIs that transform streamed events into protocol-specific formats (like SSE), see the [UI event streams](ui/overview.md) documentation and the [`UIEventStream`][pydantic_ai.ui.UIEventStream] base class.

#### Error hooks

Each lifecycle point has an `on_*_error` hook — the error counterpart to `after_*`. While `after_*` hooks fire on success, `on_*_error` hooks fire on failure (after `wrap_*` has had its chance to recover):

```
before_X → wrap_X(handler)
  ├─ success ─────────→ after_X (modify result)
  └─ failure → on_X_error
        ├─ re-raise ──→ (error propagates, after_X not called)
        └─ recover ───→ after_X (modify recovered result)
```

Error hooks use **raise-to-propagate, return-to-recover** semantics:

- **Raise the original error** — propagates the error unchanged *(default)*
- **Raise a different exception** — transforms the error
- **Return a result** — suppresses the error and uses the returned value

| Hook | Fires when | Recovery type |
|---|---|---|
| [`on_run_error`][pydantic_ai.capabilities.AbstractCapability.on_run_error] | Agent run fails | Return [`AgentRunResult`][pydantic_ai.run.AgentRunResult] |
| [`on_node_run_error`][pydantic_ai.capabilities.AbstractCapability.on_node_run_error] | Graph node fails | Return next node or [`End`][pydantic_graph.basenode.End] |
| [`on_model_request_error`][pydantic_ai.capabilities.AbstractCapability.on_model_request_error] | Model request fails | Return [`ModelResponse`][pydantic_ai.messages.ModelResponse] |
| [`on_tool_validate_error`][pydantic_ai.capabilities.AbstractCapability.on_tool_validate_error] | Tool validation fails | Return validated args `dict` |
| [`on_tool_execute_error`][pydantic_ai.capabilities.AbstractCapability.on_tool_execute_error] | Tool execution fails | Return any tool result |
| [`on_output_validate_error`][pydantic_ai.capabilities.AbstractCapability.on_output_validate_error] | Output validation fails | Return validated output |
| [`on_output_process_error`][pydantic_ai.capabilities.AbstractCapability.on_output_process_error] | Output execution fails | Return any output result |

```python {title="error_hooks_example.py" test="skip" lint="skip"}
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import ModelRequestContext, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse, TextPart


@dataclass
class ErrorLogger(AbstractCapability[Any]):
    """Logs all errors that occur during agent runs."""

    errors: list[str] = field(default_factory=list)

    async def on_model_request_error(
        self, ctx: RunContext[Any], *, request_context: ModelRequestContext, error: Exception
    ) -> ModelResponse:
        self.errors.append(f'Model error: {error}')
        # Return a fallback response to recover
        return ModelResponse(parts=[TextPart(content='Service temporarily unavailable.')])

    async def on_tool_execute_error(
        self, ctx: RunContext[Any], *, call: Any, tool_def: Any, args: dict[str, Any], error: Exception
    ) -> Any:
        self.errors.append(f'Tool {call.tool_name} failed: {error}')
        raise error  # Re-raise to let the normal retry flow handle it
```

#### Deferred tool calls

Capabilities can resolve [deferred tool calls](deferred-tools.md) — calls that require approval, or that are executed externally — directly from the agent run, without ending the run and waiting for a follow-up:

| Hook | Signature | Purpose |
|---|---|---|
| [`handle_deferred_tool_calls`][pydantic_ai.capabilities.AbstractCapability.handle_deferred_tool_calls] | `(ctx: RunContext, *, requests: DeferredToolRequests) -> DeferredToolResults \| None` | Resolve some or all pending approval/external calls inline |

Multiple capabilities can each handle a subset: dispatch accumulates results across the chain, passing only the still-unresolved requests to the next capability. Returning `None` (or a [`DeferredToolResults`][pydantic_ai.tools.DeferredToolResults] with no entries) declines handling. Anything still unresolved bubbles up as a [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests] output for the caller to handle.

For application code that just needs to plug in a handler, use the dedicated [`HandleDeferredToolCalls`][pydantic_ai.capabilities.HandleDeferredToolCalls] capability — see [Resolving deferred calls with a handler](deferred-tools.md#resolving-deferred-calls-with-a-handler).

### Wrapping capabilities

[`WrapperCapability`][pydantic_ai.capabilities.WrapperCapability] wraps another capability and delegates all methods to it — similar to [`WrapperToolset`][pydantic_ai.toolsets.WrapperToolset] for toolsets. Subclass it to override specific methods while delegating the rest:

```python {title="wrapper_capability_example.py" test="skip" lint="skip"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai import ModelRequestContext, RunContext
from pydantic_ai.capabilities import WrapperCapability


@dataclass
class AuditedCapability(WrapperCapability[Any]):
    """Wraps any capability and logs its model requests."""

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        print(f'Request from {type(self.wrapped).__name__}')
        return await super().before_model_request(ctx, request_context)
```

The built-in [`PrefixTools`][pydantic_ai.capabilities.PrefixTools] is an example of a `WrapperCapability` — it wraps another capability and prefixes its tool names.

### Per-run state isolation

By default, a capability instance is shared across all runs of an agent. If your capability accumulates mutable state that should not leak between runs, override [`for_run`][pydantic_ai.capabilities.AbstractCapability.for_run] to return a fresh instance:

```python {title="per_run_state.py"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, ModelRequestContext, RunContext
from pydantic_ai.capabilities import AbstractCapability


@dataclass
class RequestCounter(AbstractCapability[Any]):
    """Counts model requests per run."""

    count: int = 0

    async def for_run(self, ctx: RunContext[Any]) -> 'RequestCounter':
        return RequestCounter()  # fresh instance for each run

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        self.count += 1
        return request_context


counter = RequestCounter()
agent = Agent('openai:gpt-5.2', capabilities=[counter])

# The shared counter stays at 0 because for_run returns a fresh instance
agent.run_sync('first run')
agent.run_sync('second run')
print(counter.count)
#> 0
```

### Dynamically building a capability

Capabilities can be built dynamically ahead of each agent run using a function that takes the agent [`RunContext`][pydantic_ai.tools.RunContext] and returns a capability or `None`. This is useful when the capability — its instructions, model settings, hooks, or contributed toolset — depends on information specific to a run, like its [dependencies](./dependencies.md).

To register a dynamic capability, pass a function that takes [`RunContext`][pydantic_ai.tools.RunContext] to the `capabilities` argument of the [`Agent`][pydantic_ai.Agent] constructor or `agent.run()`. Sync and async functions are both supported. The function is called once per run and the returned capability replaces it for the rest of the run, so its instructions, model settings, toolsets, native tools, and hooks all flow through normally.

```python {title="dynamic_capability.py"}
from dataclasses import dataclass
from typing import Literal

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel


@dataclass
class Skill(AbstractCapability[str]):
    """Per-user skill loaded from a database at run time."""

    name: str
    role: Literal['admin', 'guest']

    def get_instructions(self) -> str:
        return f'You can use the {self.name} skill (role: {self.role}).'


# Pretend this comes from a database keyed by user.
SKILLS = {
    'alice': Skill(name='refunds', role='admin'),
    'bob': Skill(name='lookup', role='guest'),
}


def user_skill(ctx: RunContext[str]) -> AbstractCapability[str] | None:
    return SKILLS.get(ctx.deps)


agent = Agent(TestModel(), deps_type=str, capabilities=[user_skill])

result = agent.run_sync('hi', deps='alice')
print(result.all_messages()[0].instructions)
#> You can use the refunds skill (role: admin).
```

_(This example is complete, it can be run "as is")_

To return more than one capability from a single factory, wrap them in a [`CombinedCapability`][pydantic_ai.capabilities.CombinedCapability].

!!! note "Durable execution (Temporal, DBOS, Prefect)"

    A dynamic capability whose resolved capability contributes only instructions, model settings, native tools, hooks, or `prepare_tools`/`get_wrapper_toolset` (i.e. no `get_toolset()` of its own) works seamlessly with durable execution — the factory runs in the workflow alongside the rest of the agent loop. This covers the common "load this user's skill from the database and add its instructions" pattern.

    However, dynamic capabilities that contribute their own toolset via `get_toolset()` are not yet supported with durable execution. The toolset is only known at run time, so it bypasses the durable wrapper's construction-time toolset registration and would attempt I/O directly inside the workflow. As a workaround, register the toolsets statically via `Agent(toolsets=[...])` (where they get wrapped properly) and have the dynamic capability reference them indirectly — e.g. via [`prepare_tools`][pydantic_ai.capabilities.AbstractCapability.prepare_tools] to scope which tools are visible per-run, rather than constructing the toolset inside the factory. Full support is tracked in [#5253](https://github.com/pydantic/pydantic-ai/issues/5253).

### Composition and middleware semantics

When multiple capabilities are passed to an agent, they are composed into a single [`CombinedCapability`][pydantic_ai.capabilities.CombinedCapability] that follows **middleware semantics** — the same pattern used by web frameworks like Django and Starlette:

* **Configuration** is merged: instructions concatenate, model settings merge additively (later capabilities override earlier ones), toolsets combine, native tools collect.
* **`before_*`** hooks fire in capability order (outermost to innermost): `cap1 → cap2 → cap3`.
* **`after_*`** hooks fire in reverse order (innermost to outermost): `cap3 → cap2 → cap1`.
* **`wrap_*`** hooks nest as middleware: `cap1` wraps `cap2` wraps `cap3` wraps the actual operation. The first capability is the **outermost** layer.
* **`get_wrapper_toolset`** follows the same nesting: the first capability's wrapper is outermost.

This means the first capability in the list has the first and last say on the operation — it sees the original input before any other capability, and it sees the final output after all inner capabilities have processed it.

### Ordering

By default, capabilities are composed in the order you list them. When a capability needs to be at a specific position regardless of where the user lists it, override [`get_ordering`][pydantic_ai.capabilities.AbstractCapability.get_ordering] to return a [`CapabilityOrdering`][pydantic_ai.capabilities.CapabilityOrdering]:

```python {title="capability_ordering_example.py"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai.capabilities import (
    AbstractCapability,
    CapabilityOrdering,
    CombinedCapability,
)


@dataclass
class InstrumentationCapability(AbstractCapability[Any]):
    """Must wrap all other capabilities to trace everything."""

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='outermost')


@dataclass
class PlainCapability(AbstractCapability[Any]):
    pass


# InstrumentationCapability ends up first regardless of list order
combined = CombinedCapability([PlainCapability(), InstrumentationCapability()])
assert type(combined.capabilities[0]) is InstrumentationCapability
```

The available constraints are:

* **`position`** — `'outermost'` or `'innermost'`. Places the capability in a tier before (or after) all capabilities without that position. Multiple capabilities can share a tier; original list order breaks ties within it.
* **`wraps`** — list of capabilities this one wraps around (is outside of). Each entry can be a capability **type** (matches all instances via `issubclass`) or a specific **instance** (matches by identity). Use when your capability needs to see the output of another: `CapabilityOrdering(wraps=[OtherCapability])`.
* **`wrapped_by`** — list of capabilities that wrap around this one (are outside of it). Accepts types or instances, like `wraps`. The inverse of `wraps`.
* **`requires`** — list of capability types that must be present. Raises [`UserError`][pydantic_ai.exceptions.UserError] if any are missing. Does not imply ordering.

When constraints are declared, [`CombinedCapability`][pydantic_ai.capabilities.CombinedCapability] topologically sorts its children at construction time, preserving user-provided order as a tiebreaker.

[`Hooks`][pydantic_ai.capabilities.Hooks] supports ordering via the `ordering` parameter, so you can declare ordering constraints without subclassing:

```python {title="hooks_ordering_example.py"}
from pydantic_ai.capabilities import CapabilityOrdering, CombinedCapability, Hooks

logging_hooks = Hooks(ordering=CapabilityOrdering(position='outermost'))
rate_limit_hooks = Hooks(ordering=CapabilityOrdering(wrapped_by=[logging_hooks]))

# logging_hooks ends up outermost; rate_limit_hooks is wrapped by it
combined = CombinedCapability([rate_limit_hooks, logging_hooks])
assert combined.capabilities[0] is logging_hooks
assert combined.capabilities[1] is rate_limit_hooks
```

## Examples

### Guardrail (PII redaction)

A guardrail is a capability that intercepts model requests or responses to enforce safety rules. Here's one that scans model responses for potential PII and redacts it:

```python {title="guardrail_example.py"}
import re
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, ModelRequestContext, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse, TextPart


@dataclass
class PIIRedactionGuardrail(AbstractCapability[Any]):
    """Redacts email addresses and phone numbers from model responses."""

    async def after_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        for part in response.parts:
            if isinstance(part, TextPart):
                # Redact email addresses
                part.content = re.sub(
                    r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
                    '[EMAIL REDACTED]',
                    part.content,
                )
                # Redact phone numbers (simple US pattern)
                part.content = re.sub(
                    r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                    '[PHONE REDACTED]',
                    part.content,
                )
        return response


agent = Agent('openai:gpt-5.2', capabilities=[PIIRedactionGuardrail()])
result = agent.run_sync("What's Jane's contact info?")
print(result.output)
#> You can reach Jane at [EMAIL REDACTED] or [PHONE REDACTED].
```

### Logging middleware

The `wrap_*` pattern is useful when you need to observe or time both the input and output of an operation. Here's a capability that logs every model request and tool call:

```python {title="logging_middleware_example.py"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, ModelRequestContext, RunContext, ToolDefinition
from pydantic_ai.capabilities import (
    AbstractCapability,
    WrapModelRequestHandler,
    WrapToolExecuteHandler,
)
from pydantic_ai.messages import ModelResponse, ToolCallPart


@dataclass
class VerboseLogging(AbstractCapability[Any]):
    """Logs model requests and tool executions."""

    async def wrap_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        print(f'  Model request (step {ctx.run_step}, {len(request_context.messages)} messages)')
        #>   Model request (step 1, 1 messages)
        response = await handler(request_context)
        print(f'  Model response: {len(response.parts)} parts')
        #>   Model response: 1 parts
        return response

    async def wrap_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> Any:
        print(f'  Tool call: {call.tool_name}({args})')
        result = await handler(args)
        print(f'  Tool result: {result!r}')
        return result


agent = Agent('openai:gpt-5.2', capabilities=[VerboseLogging()])
result = agent.run_sync('hello')
print(f'Output: {result.output}')
#> Output: Hello! How can I help you today?
```

## Pydantic AI Harness

[**Pydantic AI Harness**](https://pydantic.dev/docs/ai/harness/) is the official capability library for Pydantic AI -- standalone capabilities like memory, guardrails, context management, and [code mode](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/code_mode) live there rather than in core. See [What goes where?](https://pydantic.dev/docs/ai/harness/#what-goes-where) for the full breakdown, or jump to the [capability matrix](https://github.com/pydantic/pydantic-ai-harness#capability-matrix).

## Third-party capabilities

Capabilities are the recommended way for third-party packages to extend Pydantic AI, since they can bundle tools with hooks, instructions, and model settings. See [Extensibility](extensibility.md) for the full ecosystem, including [third-party toolsets](toolsets.md#third-party-toolsets) that can also be wrapped as capabilities.

### Task Management

Capabilities for task planning and progress tracking help agents organize complex work:

* [`pydantic-ai-todo`](https://github.com/vstorm-co/pydantic-ai-todo) - `TodoCapability` with `add_todo`, `read_todos`, `write_todos`, `update_todo_status`, and `remove_todo` tools. Supports subtasks, dependencies, and PostgreSQL persistence. Also available as a lower-level `TodoToolset`.

### Context Management

Capabilities for managing long conversations help agents stay within context limits:

* [`summarization-pydantic-ai`](https://github.com/vstorm-co/summarization-pydantic-ai) - Four capabilities for managing long conversations: `ContextManagerCapability` (real-time token tracking, auto-compression at a configurable threshold, and large tool-output truncation); `SummarizationCapability` (LLM-powered history compression); `SlidingWindowCapability` (zero-cost message trimming); `LimitWarnerCapability` (injects a finish-soon hint before hard context limits). Also available as standalone `history_processors`: `SummarizationProcessor`, `SlidingWindowProcessor`, and `LimitWarnerProcessor`.

### Multi-Agent Orchestration

Capabilities for spawning and delegating to specialized subagents help agents tackle complex, parallelizable work:

* [`subagents-pydantic-ai`](https://github.com/vstorm-co/subagents-pydantic-ai) - `SubAgentCapability` adds tools for multi-agent delegation: `task` (spawn a subagent), `check_task`, `wait_tasks`, `list_active_tasks`, `soft_cancel_task`, `hard_cancel_task`, and `answer_subagent`. Supports sync, async, and auto execution modes, nested subagents, and runtime agent creation. Also available as a lower-level toolset via `create_subagent_toolset`.

### Guardrails & Safety

Capabilities for cost control, input/output filtering, and tool permissions help keep agents safe and within budget:

* [`pydantic-ai-shields`](https://github.com/vstorm-co/pydantic-ai-shields) - Ready-to-use guardrail capabilities: `CostTracking` (tracks token usage and USD cost per run, raises `BudgetExceededError` on budget overrun); `ToolGuard` (block or require approval for specific tools); `InputGuard` and `OutputGuard` (custom sync or async validation functions); `PromptInjection`, `PiiDetector`, `SecretRedaction`, `BlockedKeywords`, and `NoRefusals` content shields.

### File Operations & Sandboxing

Capabilities for filesystem access and sandboxed code execution help agents work with files and run code safely:

* [`pydantic-ai-backend`](https://github.com/vstorm-co/pydantic-ai-backend) - `ConsoleCapability` registers `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, and `execute` tools with a fine-grained permission system. Backends include `StateBackend` (in-memory, for testing), `LocalBackend` (real filesystem), `DockerSandbox` (isolated container execution), and `CompositeBackend` (routing across backends). Also available as a lower-level `ConsoleToolset`.

### Agent Skills

Capabilities that implement [Agent Skills](https://agentskills.io) support help agents efficiently discover and perform specific tasks:

* [`pydantic-ai-skills`](https://github.com/DougTrajano/pydantic-ai-skills) - `SkillsCapability` implements Agent Skills support with progressive disclosure (load skills on-demand to reduce tokens). Supports filesystem and programmatic skills; compatible with [agentskills.io](https://agentskills.io).

### Data & Analytics

Capabilities for querying and analyzing structured data help agents answer questions over files and databases:

* [`pydantic-ai-chdb`](https://github.com/chdb-io/pydantic-ai-chdb) - `ChDBCapability` gives agents analytical SQL over local files (Parquet/CSV/JSON), object storage, and remote databases with [chDB](https://clickhouse.com/docs/en/chdb), the in-process ClickHouse engine — the engine itself needs no server or connection string to run (remote sources are reached via ClickHouse table functions, which take their own credentials). Registers `run_select_query` (read-only ClickHouse SQL with parameter binding), `list_databases`, `list_tables`, `describe_table`, `get_sample_data`, `list_functions`, and `attach_file` (opt-in writable sessions) tools plus schema-first instructions. Sessions default to the engine-level `readonly=2` setting with capped results, and typed engine errors are mapped to [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] so the model can correct its queries. Works with [agent specs](agent-spec.md) out of the box, so it can be loaded via [`from_spec`][pydantic_ai.capabilities.AbstractCapability.from_spec] / [`Agent.from_spec`][pydantic_ai.Agent.from_spec]. Also available as a lower-level [toolset](toolsets.md) via [`ChDBCapability(...).get_toolset()`][pydantic_ai.capabilities.AbstractCapability.get_toolset].

To add your package to this page, open a pull request.

## Publishing capabilities

To make a custom capability usable in [agent specs](agent-spec.md), it needs a [`get_serialization_name`][pydantic_ai.capabilities.AbstractCapability.get_serialization_name] (defaults to the class name) and a constructor that accepts serializable arguments. The default [`from_spec`][pydantic_ai.capabilities.AbstractCapability.from_spec] implementation calls `cls(*args, **kwargs)`, so for simple dataclasses no override is needed:

```python {title="custom_spec_capability.py"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, AgentSpec
from pydantic_ai.capabilities import AbstractCapability


@dataclass
class RateLimit(AbstractCapability[Any]):
    """Limits requests per minute."""

    rpm: int = 60


# In YAML: `- RateLimit: {rpm: 30}`
# In Python:
agent = Agent.from_spec(
    AgentSpec(model='test', capabilities=[{'RateLimit': {'rpm': 30}}]),
    custom_capability_types=[RateLimit],
)
```

Users register custom capability types via the `custom_capability_types` parameter on [`Agent.from_spec`][pydantic_ai.Agent.from_spec] or [`Agent.from_file`][pydantic_ai.Agent.from_file].

Override [`from_spec`][pydantic_ai.capabilities.AbstractCapability.from_spec] when the constructor takes types that can't be represented in YAML/JSON. The spec fields should mirror the dataclass fields, but with serializable types:

```python {title="from_spec_override_example.py" test="skip" lint="skip"}
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import RunContext, ToolDefinition
from pydantic_ai.capabilities import AbstractCapability


@dataclass
class ConditionalTools(AbstractCapability[Any]):
    """Hides tools unless a condition is met."""

    condition: Callable[[RunContext[Any]], bool]  # not serializable
    hidden_tools: list[str] = field(default_factory=list)

    @classmethod
    def from_spec(cls, hidden_tools: list[str]) -> 'ConditionalTools[Any]':
        # In the spec, there's no condition callable — always hide
        return cls(condition=lambda ctx: True, hidden_tools=hidden_tools)

    async def prepare_tools(
        self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        if self.condition(ctx):
            return [td for td in tool_defs if td.name not in self.hidden_tools]
        return tool_defs
```

In YAML this would be `- ConditionalTools: {hidden_tools: [dangerous_tool]}`. In Python code, the full constructor is available: `ConditionalTools(condition=my_check, hidden_tools=['dangerous_tool'])`.

See [Extensibility](extensibility.md) for packaging conventions and the broader extension ecosystem.
