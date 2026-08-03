# Capabilities on Demand

Read this file when designing progressive disclosure of any kind, when an agent has information it does not need on most turns, or when the user asks about deferred capabilities, capabilities on demand, `defer_loading=True` on capabilities, or the `load_capability` tool.

## Mental Model

Capabilities on demand are bundle-level progressive disclosure for Pydantic AI. The model initially sees a compact catalog of deferred capability `id` values, plus `description` values when provided, and the framework-managed `load_capability` tool. When the model calls `load_capability(id)`, Pydantic AI returns that capability's instructions; its function tools, native tools, and model settings are reflected on the next model request, and its hooks can fire for later hook points in the run.

Loaded function tools are recorded in durable message history with `ToolAvailabilityDeltaPart`. Treat it as framework control state: it names tools that became available, while their current definitions remain in the model request parameters.

Provider adapters project that control state without changing the history. OpenAI Responses uses an `additional_tools` input item. In a mixed corpus, the deferred tool and `tool_search` deliberately remain in `tools` alongside that item; keeping them there preserves a byte-identical `tools` prefix and avoids leaving `tool_search` with an empty deferred corpus. OpenAI-compatible endpoints that don't implement `additional_tools` announce the change when the tool schema is already visible, or use a synthesized `search_tools` exchange when its result must reveal a withheld schema. Do not copy tool definitions into `ToolAvailabilityDeltaPart`.

### Tool-availability history portability

Stored history can describe availability through model-driven discovery or application-driven
control. Preserve that distinction when switching models:

| Stored representation | Anthropic with `tool_addition` | Anthropic with native search only | OpenAI Responses with native search | First-party OpenAI Responses without native search | OpenAI-compatible Responses without `additional_tools` | Gemini | OpenAI Chat Completions |
|---|---|---|---|---|---|---|---|
| Local `search_tools` call and result | Native search | Native search | Native search | Local search | Local search | Local search | Local search |
| Anthropic native search | Native search | Native search | Native search | Local search | Local search | Local search | Local search |
| OpenAI native search | Native search | Native search | Native search | Local search | Local search | Local search | Local search |
| `ToolAvailabilityDeltaPart` | `tool_addition` | Native search | `additional_tools` | `additional_tools` | Announcement or local search | Announcement | Announcement |
| `search_tools` result with `metadata['discovered_tools']` | Native search | Native search | Native search | Local search | Local search | Local search | Local search |

**Native search** is a paired provider-native search call and result with the native search tool; the
revealed tool remains in the deferred corpus. **Local search** is a paired `search_tools` function
call and result with the local search tool; the revealed tool is eager in the function-tool list.
For a capability-only corpus, a provider-native availability change includes neither a search
exchange nor a search tool. In a mixed corpus, the search tool stays on the wire for the tools that
remain searchable.

A genuine search records a query chosen by the model and the matches it received. Never rewrite it
as `tool_addition` or `additional_tools`, which would recast discovery as application-driven
control. Use those provider-native control items only for `ToolAvailabilityDeltaPart`. Where the
target has no availability-change primitive and the schema is already visible, announce
`The following tool(s) are now available: {names}`. Synthesize a complete local search exchange only
when its result must reveal a schema that is actually withheld; the tool must not remain locked
behind `defer_loading`.

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
    name='support_agent',
    instructions='Answer as a support assistant.',
    capabilities=[refunds],
)
```

`Capability` is a convenience helper for simple bundles of instructions, descriptions, function tools, and toolsets. It accepts callable descriptions, dynamic instruction functions, and dynamic toolset functions. Use a custom `AbstractCapability` for model settings, hooks, native tools, wrapper toolsets, reusable public behavior, or custom per-run logic. Wrapper toolsets are applied during per-run toolset assembly; if wrapper behavior should wait for a deferred capability to load, gate that behavior inside the wrapper.

## Runtime Semantics

Initial request:

- deferred capability instructions are not included
- deferred capability function tools are present in the framework toolset but marked with `defer_loading=True`, and they are not callable until the capability loads
- capability-owned tools are hidden but never searchable, so when every deferred tool is capability-owned no tool search is advertised at all — not the provider's and not the local `search_tools` function. Anthropic declares the tools with the wire `defer_loading` flag and reveals them in place; OpenAI Responses rejects `defer_loading` without a `tool_search` tool, so it leaves them out of `tools` and reveals them with an `additional_tools` item. Either way `tools` is byte-identical across the load. Add a standalone `defer_loading=True` tool and search returns for that one, running client-side so a query can't surface a tool whose capability hasn't loaded
- non-deferred capabilities are treated as already loaded
- the framework adds `load_capability` if any deferred capability exists

When `load_capability` succeeds:

- the call is typed as a capability-load message part
- the return may include resolved capability instructions and owned toolset instructions
- the capability id is added to `ctx.available_capability_ids`
- tools owned by the loaded capability become visible on later steps
- `load_capability` remains visible so the tool set stays stable

Use `ctx.is_tool_available(tool_def)` when a wrapping toolset needs to decide whether a definition it holds is currently visible. The definition form remains reliable inside `get_tools`; the name form looks in the current resolved `ctx.tools` snapshot and is intended for model-request hooks and tool execution.

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
