# Enable Coder delegation in CLAI

Status: design proposal only. No runtime changes, dependency upgrade, push, or PR.

## Confirmed cause

- `Coder` defaults `sub_agents=True`, adding `SubAgents(include_self=True)`.
- CLAI's `DEFAULT_PLUGINS` explicitly sets `sub_agents=False`. `PluginSettings` also supplies `False` for saved Coder declarations that omit it.
- `Session.prompt()` passes active plugins through `agent.run(capabilities=...)`.
- Self-delegation starts another run of `ctx.agent`, which does not inherit those run-level plugins. Harness rejects this configuration before the model request.

Reproduced with installed Harness `0.52.0`: disabling delegation exposes the six coding tools; enabling it at run level raises the documented `UserError`. A dependency bump alone does not resolve this contract mismatch.

## Recommended direction

Add an explicit, opt-in run-capability inheritance mode to Harness self-delegation. Keep the existing agent-bound mode and its validation unchanged by default. CLAI opts into the new mode; other Agent and Coder users keep their current behavior.

The child must use the same underlying agent, a fresh history, and the capability configuration granted to the parent turn. This preserves agent-registered tools and hooks while carrying CLAI's active plugin tools, instructions, and guardrails. Keep Harness's existing workspace, dependency, usage-budget, cancellation, and nesting-limit behavior.

Do not mutate the shared agent, rebuild an approximate copy of a custom agent, or remove the binding check without supplying the missing capabilities.

## Resolve before implementation

1. Establish a supported way to snapshot the original run-level capability declarations, preserving their ordering, factories, and precedence over agent-bound capabilities. Rebind them for each child run rather than copying parent run state.
2. Core exposes `RunContext.root_capability` and `RunContext.capabilities`, but these describe the effective runtime tree. Do not assume forwarding either safely recreates declarations: it could duplicate agent-bound contributions, lose combined wrappers, or reuse already-resolved dynamic capabilities.
3. Prefer a Harness-only implementation if existing public APIs can satisfy that contract. If they cannot, propose a narrowly scoped core facility for accessing the run's capability declarations. No change to ordinary run behavior or automatic inheritance.
4. Audit stateful CLAI plugins, particularly persistence, compaction, AskUser, telemetry, and speculative CodeMode. Child runs must not overwrite the foreground conversation or share unsafe per-run state. Any exclusion must be explicit and must never discard approval or guardrail hooks silently.

## Integration and compatibility

- Expose the opt-in mode through Coder only after the inheritance contract is proven.
- Enable delegation in the stock CLAI declaration using that mode.
- Preserve explicit saved `sub_agents=False`. Preserve existing saved declarations until migration/default behavior is agreed; test omitted and explicit values separately.
- Plugin changes between turns affect subsequent delegations. Concurrent forks keep independent snapshots.
- Update CLAI's README, plugin contract, customization guide, and relevant skills once implementation is approved.

## Acceptance tests

Use existing CLAI and Harness test directories with `FunctionModel`/`TestModel`:

- Stock Coder exposes `delegate_task`; a child successfully uses a plugin tool and the parent's workspace.
- A plugin approval/guardrail hook runs in the child and can deny its tool call.
- Dynamic capabilities bind independently per child, without duplicate tools or instructions.
- Disabled plugins are absent on the next turn; concurrent forks do not exchange grants.
- Explicit delegation opt-out still works; ordinary agent-bound delegation and the existing unsupported run-level configuration retain their behavior.
- Nested delegates respect depth and shared usage limits; cancellation drains child work.
- Child persistence and speculation do not corrupt foreground state.

Run focused pytest, Ruff, and single-file Pyright checks, followed by a fresh terminal check. Keep the work local until asked to publish.
