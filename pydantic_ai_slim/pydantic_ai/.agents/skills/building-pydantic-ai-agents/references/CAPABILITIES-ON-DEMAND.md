# Capabilities on Demand

Read this file when designing progressive disclosure of any kind, when an agent has information it does not need on most turns, or when the user asks about deferred capabilities, capabilities on demand, `defer_loading=True` on capabilities, or the `load_capability` tool.

## Mental Model

Capabilities on demand are bundle-level progressive disclosure for Pydantic AI. The model initially sees a compact catalog of deferred capability `id` values, plus `description` values when provided, and the framework-managed `load_capability` tool. When the model calls `load_capability(id)`, Pydantic AI returns that capability's instructions and makes its function tools available.

Be opinionated: every capability should be reviewed for whether `defer_loading=True` would benefit the system before accepting eager loading. If the model does not need a piece of information, a specialist instruction set, or a tool schema on most turns, do not put it in the eager prompt by default. Suggest capabilities on demand for named bundles, tool search for long-tail tools, or a narrower always-on instruction if the behavior really is universal.

Use this for specialist behavior where instructions and tools should travel together:

- support workflows such as refunds, returns, account management, or fraud review
- domain-specific tool bundles where most requests need only one bundle
- agents that would otherwise load many capability instructions and tool schemas on every turn

Use tool search instead when the agent has a large flat tool catalog and the model should discover individual tools. Tool search uses `search_tools`; capabilities on demand use `load_capability`.

Do not wait for the user to say "progressive disclosure." Raise it during design and review whenever an agent is accumulating optional context, many domain runbooks, large policy text, rarely used tools, or multiple specialist workflows.

## Opinionated Design Rules

- Treat `defer_loading=True` as a design question for every capability, not a niche option users must ask for.
- Keep the base agent prompt small: identity, task boundaries, global safety, and the routing instruction needed to decide what to load.
- Put specialist runbooks behind capabilities on demand when they are useful only for a subset of requests.
- Put broad tool catalogs behind tool search when the tools are individually discoverable and do not need shared instructions.
- Keep hot-path tools and universal instructions eager when they are used most turns.
- Prefer a few coherent capability bundles over dozens of tiny capabilities that force the model to plan its own dependency graph.
- Do not hide information the model needs to decide which capability to load; that belongs in the capability description or always-on routing instructions.

## Minimal Pattern

Every deferred capability needs a stable explicit `id` and `defer_loading=True`. A concise `description` is optional; add one when the `id` alone is not enough for routing.

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability

refunds = Capability(
    id='refunds',
    description='Refund policy tools and instructions.',
    instructions='Use the refund policy before answering refund questions.',
    defer_loading=True,
)


@refunds.tool_plain
def lookup_refund_policy(order_id: str) -> str:
    """Look up whether an order is eligible for a refund."""
    return f'{order_id} is eligible for a refund for 30 days after purchase.'


agent = Agent(
    'anthropic:claude-sonnet-4-6',
    instructions='Answer as a support assistant.',
    capabilities=[refunds],
)
```

Prefer `Capability` for static instructions and function tools. Subclass `AbstractCapability` only when the description, instructions, model settings, hooks, or toolset needs custom logic.

## Runtime Semantics

Initial request:

- deferred capability instructions are not included
- deferred capability function tools are present in the framework toolset but marked with `defer_loading=True`, so they are hidden from the model-facing visible set
- non-deferred capabilities are treated as already loaded
- the framework adds `load_capability` if any deferred capability exists

When `load_capability` succeeds:

- the call is typed as a capability-load message part
- the return may include resolved capability instructions and owned toolset instructions
- the capability id is added to `ctx.available_capability_ids`
- tools owned by the loaded capability become visible on later steps
- `load_capability` remains visible so the tool set stays stable

Message history matters. Loaded capability state is reconstructed from matching `LoadCapabilityCallPart` and `LoadCapabilityReturnPart` pairs in message history. If a history processor removes those parts, the model may need to load the capability again.

## Dynamic Descriptions and Instructions

Use `get_description()` when the catalog text depends on run context. Use dynamic instructions when load-time instructions need deps or current run state.

```python
from dataclasses import dataclass

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability


@dataclass
class SupportDeps:
    plan: str
    account_id: str


@dataclass
class AccountCapability(AbstractCapability[SupportDeps]):
    def get_description(self, ctx: RunContext[SupportDeps] | None) -> str:
        plan = ctx.deps.plan if ctx else 'the current'
        return f'Account-management tools for {plan} plan customers.'

    def get_instructions(self):
        def load_instructions(ctx: RunContext[SupportDeps]) -> str:
            return f'Use account ID {ctx.deps.account_id} for account-management tools.'

        return load_instructions


account_capability = AccountCapability(id='account-management', defer_loading=True)
```

## Composition Rules

- Capability `id` values must be unique in a run.
- Deferred capability ids must be explicit and stable; auto-generated ids are rejected because history replay cannot rely on them.
- `load_capability` is reserved when any deferred capability exists.
- Deferred capability instructions and model settings activate only after the capability is loaded.
- Function tools are supported. Native tools are not currently lazy-loaded; keep native tools always on or set `defer_loading=False`.
- Capability-level `defer_loading=True` gates the bundle as a unit. Individual tool `defer_loading` values inside that capability do not expose tools before loading or keep them behind `search_tools` after loading.

## Choosing Between Deferral Mechanisms

Use capabilities on demand when the model needs a named package of instructions plus tools.

Keep a capability eager when its instructions, hooks, model settings, or tools materially improve most turns, or when hiding it would make routing unreliable.

Use tool search when the model needs individual tool discovery across many tools or MCP server endpoints.

Use deferred tool calls when the issue is execution timing, approval, or external execution. Deferred tool calls decide whether a visible tool call can run now; they do not control whether the model can see a capability.

When in doubt, ask this question: "Would a high-quality answer to most user prompts get worse if this information were absent until requested?" If no, recommend progressive disclosure.

## Testing Checklist

- Assert the first model request only exposes `load_capability` plus always-on tools.
- Assert the model can call `load_capability` with the expected id.
- Assert the next request exposes the loaded capability's function tools.
- Assert returned instructions include capability and owned toolset instructions when applicable.
- Add a history replay case if the feature depends on loaded capability state across runs.
