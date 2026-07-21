# On-Demand Capabilities

A multi-workflow agent normally sends every workflow's instructions and tool schemas on every turn, and applies every workflow's settings and hooks for the whole run — even though most requests need just one workflow. That cost grows with each workflow you add: more input tokens, and worse tool selection once the visible tool set passes the ~30–50-tool mark where models start picking the wrong one (the same pressure behind [tool search](../tools-advanced.md#tool-search)).

Mark a [capability](overview.md) with `defer_loading=True` and give it a stable `id`, and it collapses to a one-line catalog entry — its `id` plus an optional `description` — that the model pulls in on demand. Here's the minimal shape:

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

```text
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
    The `load_capability` tool name is reserved whenever any on-demand capability is present. Capability `id` values must be stable — set one explicitly unless the capability derives a stable `id` itself, as [`MCP`][pydantic_ai.capabilities.MCP] does from its server URL. See [Resumable across runs](#resumable-across-runs).

!!! note "Deferred instructions reach client-facing message history"
    A deferred capability's instructions come back as the `load_capability` tool *result*, so they land in the run's message history — including the copy a [UI adapter](../ui/overview.md) serializes to the client. Instructions on an always-on capability stay in the server-side system prompt instead. If a capability's instructions shouldn't be exposed to the client, keep it always-on rather than deferred.

## What you can defer

Every part of a capability bundle activates together as a single unit:

| Part | Before load | After load |
|---|---|---|
| Instructions (static or dynamic) | Not sent | Returned as the `load_capability` tool result; included in subsequent requests |
| Function tools | Not exposed | Exposed on the next request |
| Model settings (static or per-step) | Not applied | Merged into the run's settings for subsequent requests |
| Lifecycle [hooks](custom.md#hooking-into-the-lifecycle) | Do not fire | Fire after the capability is loaded |
| [Native tools](../native-tools.md) | Not exposed | Exposed on the next request — see [Cache implications](#cache-implications) |

## When to use it

**Reach for on-demand capabilities when:**

- the agent serves multiple distinct workflows (refunds, returns, fraud review, account security…) where most turns need one
- a workflow needs *more than instructions* — its own tools, raised reasoning effort, an approval hook — and those should travel together as a unit
- you want skills-style progressive disclosure but also want the loaded bundle to bring tools and settings, not just a runbook

**Skip it when:**

- the capability is used on most turns — the discovery round-trip costs more than the tokens it saves
- you have a flat catalog of individually-discoverable tools with no shared instructions — use [tool search](../tools-advanced.md#tool-search) instead, which discovers individual tools by name rather than loading bundles

If you've used [Anthropic's Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills), this is the same idea generalised: a skill is a markdown file the model can pull in on demand. An on-demand capability does that *plus* typed function tools, per-step model settings, and lifecycle hooks.

## Retrofitting an existing capability

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

Until the model loads `analytics-mcp`, none of the MCP server's tool definitions enter the prompt. The same flag works on [`WebSearch`][pydantic_ai.capabilities.WebSearch], [`WebFetch`][pydantic_ai.capabilities.WebFetch], [`Hooks`][pydantic_ai.capabilities.Hooks], and any custom [`AbstractCapability`][pydantic_ai.capabilities.AbstractCapability] subclass — see [Building custom capabilities](custom.md) for adding `defer_loading` to your own subclass.

!!! note "Deferred `MCP`: set a stable `id`"
    [`MCP`][pydantic_ai.capabilities.MCP] derives its `id` from the server URL when you omit one, so `defer_loading=True` works without an explicit `id`. Pass one anyway if you persist and [resume](#resumable-across-runs) conversations: a URL-derived id changes if the URL does (different environment, path version, …), which silently breaks the resumed capability's loaded state.

## Resumable across runs {#resumable-across-runs}

Loaded-capability state lives in message history, not in the agent. When a conversation is persisted to a database and resumed later — possibly on a different process, machine, or model — Pydantic AI reconstructs the loaded set from the `load_capability` tool call/return pairs in history. Capabilities the model loaded earlier stay loaded; capabilities it never loaded stay collapsed in the catalog. No re-discovery round-trip on resume.

This is why deferred capabilities require a stable explicit `id`: history replay matches calls to capabilities by id, so a class-derived id would silently break the moment a class is renamed. The same property makes cross-provider replay work — a run that loaded `refunds` on Anthropic and continued on OpenAI Responses keeps `refunds` loaded after the switch.

History carries *which* capability ids were loaded, not the capabilities themselves: the resuming agent must be constructed with the same capabilities (matching `id`s), just as it must be constructed with the same tools. State lives in history; definitions live in code.

## Runtime state in `RunContext`

Several [`RunContext`][pydantic_ai.tools.RunContext] fields expose progressive-disclosure state to tools, hooks, and capability-owned callbacks:

- `ctx.loaded_capability_ids` — deferred capability IDs explicitly loaded through the `load_capability` tool, reconstructed from message history and updated when a capability loads during the current step.
- `ctx.available_capability_ids` — the currently-live capability IDs: always-available capabilities plus `ctx.loaded_capability_ids`.
- `ctx.capability_loaded` — only meaningful while Pydantic AI is running a capability-owned hook or callback. It is scoped to that capability; deferred hooks and callbacks are skipped until this value would be true.
- `ctx.discovered_tool_names` — deferred function tools revealed by tool search. This is tool-level discovery, separate from capability-level loading.
- `ctx.available_tool_names` — function tool names currently known as available: always-visible tools from the current step's assembled tool manager plus tool-search discoveries reconstructed from history. Early hooks such as `before_run` may see only the history-derived discovered names, or an empty set if none exist yet, before tool definitions have been prepared. See [Hook ordering](../hooks.md#hook-ordering) for how hook timing affects what is populated.
- `ctx.usage_limits` — the [`UsageLimits`][pydantic_ai.usage.UsageLimits] the run is enforcing (defaulting to `UsageLimits()` when none were passed, so it's only `None` outside of a run), alongside `ctx.usage` for the usage so far. A capability can read the run's limits to disclose or adapt to the remaining budget (e.g. budget disclosure) without being configured with a duplicate copy. Treat it as read-only: it's the live object the run enforces against, so mutating a field would change what the run enforces on subsequent requests.

Loading a capability updates the capability state immediately, but the loaded bundle's function tools, native tools, and model settings take effect on the next model request.

## Cross-provider behavior

On-demand capabilities work on every model. Where the provider exposes a native progressive-disclosure surface — Anthropic tool search on Sonnet 4.5+/Opus 4.5+/Haiku 4.5+, OpenAI Responses `tool_search` on GPT-5.4+ — Pydantic AI uses that surface so deferred function tools stay out of the prompt prefix. Standalone deferred tools can use the provider's hosted search; tools owned by on-demand capabilities use client-executed local search through the native surface so tools from unloaded capabilities cannot leak. On other providers, a local `search_tools` function tool handles discovery: the initial context shrinks the same way, but cache stability across loads is not guaranteed.

### Cache implications {#cache-implications}

Calling the `load_capability` tool reveals capability behavior between requests. Whether that breaks the provider's prompt-cache prefix depends on what's revealed:

| What loads | Cache prefix |
|---|---|
| Instructions only | **Stable** — instructions land in the message history, not the request prefix. |
| Function tools on a model with native [tool search](../tools-advanced.md#tool-search) (OpenAI Responses, Anthropic) | **Stable** — the function tools visible to the provider don't change across loads. |
| Function tools on other models (local `search_tools` fallback) | **May break between turns** — function-tool visibility changes as capabilities load. |
| Native tools | **Always breaks the prefix on load** — native tool definitions are part of the request prefix on every provider. |

When preserving the cache prefix matters, prefer instruction-only or function-tool-only on-demand capabilities on a model with native tool search. The provider-specific mechanics that keep the prefix stable live in [tools-advanced.md](../tools-advanced.md#tool-search).

## The `Capability` convenience class

[`Capability`][pydantic_ai.capabilities.Capability] bundles instructions, function tools, and toolsets without subclassing. Register tools with the decorator that mirrors [`@agent.tool`](../tools.md#registering-function-tools-via-decorator):

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

In addition to `@capability.tool` and `@capability.tool_plain`, you can pass existing functions or [`Tool`][pydantic_ai.tools.Tool] instances via `tools=`, or hand in one or more [toolsets](../toolsets.md) via `toolsets=`. For dynamic instructions, use the [`@capability.instructions`][pydantic_ai.capabilities.Capability.instructions] decorator. For a dynamic catalog entry, pass a callable as `description=`.

`@capability.tool` and `@capability.tool_plain` mirror [`@agent.tool`](../tools.md#registering-function-tools-via-decorator) exactly, including the `defer_loading` argument. On a deferred capability that per-tool flag is a no-op — the capability gates all its tools as a unit — so it only has an effect on a non-deferred `Capability`, where it opts an individual tool into [tool search](../tools-advanced.md#tool-search) discovery.

For anything beyond instructions, function tools, toolsets, and descriptions — model settings, hooks, native tools, wrapper toolsets, or custom per-run logic — subclass [`AbstractCapability`][pydantic_ai.capabilities.AbstractCapability] directly. When subclassing, override [`get_description`][pydantic_ai.capabilities.AbstractCapability.get_description] if the catalog entry needs to vary by run.

!!! note "Setting `id` for durable execution"
    A toolset contributed by a capability — via `Capability(tools=[...])` or an [`MCP`](mcp.md) server running locally — inherits its `id` from the capability's [`id`][pydantic_ai.capabilities.AbstractCapability.id]. [Durable execution](../durable_execution/overview.md) identifies each leaf toolset by its `id`, so pass `Capability(id='...', tools=[...])` or `MCP(id='...', url='...')` when combining a capability with Temporal, DBOS, or Prefect. Temporal requires an `id` for every leaf toolset and DBOS for every MCP server — both raise at construction without one. (`MCP` also derives one from the server URL when no `id` is given.) A URL-derived `id` can collide when two different servers share a host and final path segment (`https://a.com/api` and `https://a.com/v2/api` both derive `a.com-api`); DBOS raises at construction and Temporal when the worker starts, so pass an explicit `id` to disambiguate them.

## Beyond instructions: tools, settings, hooks, native tools {#beyond-instructions}

The [`Capability`][pydantic_ai.capabilities.Capability] example above deferred instructions and a function tool, but the same flag gates the whole bundle — what the model knows, what it can do, and how it does it (see [What you can defer](#what-you-can-defer)). The snippets below show the remaining pieces in turn: model settings, hooks, and native tools.

### Deferred model settings

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

### Lifecycle hooks with deferred workflows

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

### Deferred native tools

Any [native capability](overview.md#built-in-capabilities) (`WebSearch`, `WebFetch`, `MCP`, …) can be deferred the same way. The native tool definition only enters the request after the `load_capability` tool loads the capability — see [Cache implications](#cache-implications) for the trade-off:

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

## Putting it together: a multi-workflow support agent

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
        # Approval gate: inspect the call and raise to block, active once the model has loaded `account-security`.
        return args


support_agent = Agent(
    'openai-responses:gpt-5.4',
    deps_type=Store,
    instructions='You are a customer-support agent for an e-commerce store.',
    capabilities=[orders, AccountSecurity()],
)
```

A "where is my order?" request loads only `orders`. A "someone is logging into my account" request loads only `account-security` — and from that point on, every tool call in the run passes through the approval hook *and* benefits from the raised reasoning effort, without either being visible to the model on requests that never touched the workflow.

## Enforcing read-before-act

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

## Loading skills from Markdown files

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

    - **[Tool search](../tools-advanced.md#tool-search)** — capability-level `defer_loading=True` gates the whole bundle as a unit; for per-*tool* discovery, set tool-level `defer_loading=True` on a non-deferred capability or on `@agent.tool`.
    - **[MCP servers](../mcp/client.md)** — the [`MCP`][pydantic_ai.capabilities.MCP] capability accepts `defer_loading=True`, hiding the server's full tool list until the model opts in.
    - **[Native tools](../native-tools.md)** — [`WebSearch`][pydantic_ai.capabilities.WebSearch], [`WebFetch`][pydantic_ai.capabilities.WebFetch], [`ImageGeneration`][pydantic_ai.capabilities.ImageGeneration], and [`MCP`][pydantic_ai.capabilities.MCP] all defer the same way as function tools (see [Cache implications](#cache-implications)).
    - **[Hooks](../hooks.md)** — lifecycle hooks declared on a deferred capability (or via a deferred [`Hooks`][pydantic_ai.capabilities.Hooks] capability) stay dormant until the model opts in.
    - **[Message history](../message-history.md)** — loaded state round-trips through history, so persisted conversations resume in the same state (see [Resumable across runs](#resumable-across-runs)).
