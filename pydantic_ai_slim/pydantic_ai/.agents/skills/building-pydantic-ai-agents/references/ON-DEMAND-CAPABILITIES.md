# Capabilities on Demand

Read this file when designing progressive disclosure of any kind, when an agent has information it does not need on most turns, or when the user asks about deferred capabilities, capabilities on demand, `defer_loading=True` on capabilities, or the `load_capability` tool.

## Mental Model

Capabilities on demand are bundle-level progressive disclosure for Pydantic AI. The model initially sees a compact catalog of deferred capability `id` values, plus `description` values when provided, and the framework-managed `load_capability` tool. When the model calls `load_capability(id)`, Pydantic AI returns that capability's instructions; its function tools, native tools, and model settings are reflected on the next model request, and its hooks can fire for later hook points in the run.

Be opinionated: review every capability for whether `defer_loading=True` would benefit the system before accepting eager loading. If the model does not need a piece of information, a specialist instruction set, or a tool schema on most turns, do not put it in the eager prompt by default.

Use this for specialist behavior where instructions and tools should travel together:

- support workflows such as refunds, returns, account management, or fraud review
- domain-specific tool bundles where most requests need only one bundle
- agents that would otherwise load many capability instructions and tool schemas on every turn

Use tool search instead when the agent has a large flat tool catalog and the model should discover individual tools. Tool search uses `search_tools`; capabilities on demand use `load_capability`.

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

`Capability` is a convenience helper for simple bundles of instructions, descriptions, function tools, and toolsets. It accepts callable descriptions, dynamic instruction functions, and dynamic toolset functions. Use a custom `AbstractCapability` for model settings, hooks, native tools, wrapper toolsets, reusable public behavior, or custom per-run logic. Wrapper toolsets are applied during per-run toolset assembly; if wrapper behavior should wait for a deferred capability to load, gate that behavior inside the wrapper.

## Runtime Semantics

Initial request:

- deferred capability instructions are not included
- deferred capability function tools are present in the framework toolset but marked with `defer_loading=True`; they go through client-executed local search, so the provider's hosted search never sees them, and they are not callable until the capability loads
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

Use `get_description()` when the catalog text depends on run context. Return a callable (with or without `RunContext`) that produces the description string. Use dynamic instructions when load-time instructions need deps or current run state.

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
    def get_description(self):
        def describe(ctx: RunContext[SupportDeps]) -> str:
            return f'Account-management tools for {ctx.deps.plan} plan customers.'

        return describe

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
- Both function and native tools defer with the capability. Deferring a native tool delays its definition entering the request, which breaks the prompt-cache prefix on load — only worth it for tools that materially bloat the prompt.
- Capability-level `defer_loading=True` gates the bundle as a unit. Once the model loads the capability, all tools owned by that deferred capability become visible together. Use tool-level `defer_loading=True` outside a deferred capability when individual tools should stay behind `search_tools`.

## Choosing Between Deferral Mechanisms

Capabilities on demand (`load_capability`) and tool search (`search_tools`) are covered above. The third mechanism is **deferred tool calls**: use these when the issue is execution timing, approval, or external execution. Deferred tool calls decide whether a *visible* tool call can run now; they do not control whether the model can see a capability.

When in doubt: "Would a high-quality answer to most user prompts get worse if this information were absent until requested?" If no, recommend progressive disclosure.
