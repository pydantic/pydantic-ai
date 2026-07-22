# Capabilities and Hooks

Read this file when the user wants reusable agent behavior, provider-adaptive tools, or lifecycle interception.

## Add Capabilities to an Agent

Capabilities bundle reusable behavior and compose automatically.

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking, WebSearch

agent = Agent(
    'anthropic:claude-opus-4-6',
    name='capabilities_agent',
    capabilities=[
        Thinking(effort='high'),
        WebSearch(),
    ],
)
```

Provider-adaptive capabilities to reach for first:

- `Thinking`
- `WebSearch`
- `WebFetch`
- `ImageGeneration`
- `MCP`

For stricter safety handling, use `RaiseContentFilterError` to raise `ContentFilterError` whenever a model response has `finish_reason='content_filter'`, including responses with partial text.

Use capabilities when the user wants behavior that should survive model/provider changes.

## Enable Thinking Across Providers

Use the unified `Thinking` capability or the `thinking` model setting.

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking

capability_agent = Agent('anthropic:claude-opus-4-6', name='capability_agent', capabilities=[Thinking(effort='high')])
settings_agent = Agent('anthropic:claude-opus-4-6', name='settings_agent', model_settings={'thinking': 'high'})
```

Supported effort values:

- `True`
- `False`
- `'minimal'`
- `'low'`
- `'medium'`
- `'high'`
- `'xhigh'`

## Intercept Agent Lifecycle with Hooks

Use `Hooks` for decorator-based lifecycle interception.

```python
from pydantic_ai import Agent, RunContext, ToolDefinition
from pydantic_ai.capabilities import ValidatedToolArgs
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models import ModelRequestContext

hooks = Hooks()


@hooks.on.before_model_request
async def log_request(ctx: RunContext, request_context: ModelRequestContext) -> ModelRequestContext:
    print(f'Sending {len(request_context.messages)} messages')
    return request_context


@hooks.on.before_tool_execute(tools=['send_email'])
async def audit_tool(
    ctx: RunContext[None],
    *,
    call: ToolCallPart,
    tool_def: ToolDefinition,
    args: ValidatedToolArgs,
) -> ValidatedToolArgs:
    print(f'Executing {call.tool_name}')
    return args


agent = Agent('openai:gpt-5.2', name='hooks_agent', capabilities=[hooks])
```

Important hook families:

- run-level hooks
- node-level hooks
- model-request hooks
- tool-validation hooks
- tool-execution hooks
- event-stream hooks

Use hooks when the user wants observability, auditing, or light interception without adding a new abstraction.

## Build a Custom Capability

Subclass `AbstractCapability` when the user wants reusable behavior that combines tools, hooks, instructions, or model settings into one package.

Reach for a custom capability when:

- the same bundle should be reused across multiple agents
- `Hooks` alone is not enough
- the behavior should be installable or declarative

Keep custom capabilities focused. If the user only needs one tool or one hook, do not introduce a capability.

For every capability, consider whether `defer_loading=True` would improve the system by keeping instructions and tool schemas out of the eager context. Keep it eager only when the model benefits from that capability on most turns, when its hooks/settings must always apply, or when deferral would make capability selection unreliable.

Use `for_agent(agent)` when a capability needs the agent's model, name, or toolsets. Return a bound copy instead of mutating the original so one capability instance can safely be attached to multiple agents. The returned copy supplies all subsequent `get_*` contributions and hooks. Binding sees the constructor model exactly as supplied, including an unresolved string; default model inference happens afterwards only if the bound tree has no `resolve_model_id()` hook and model checking was not deferred. `CombinedCapability` and `WrapperCapability` bind their children automatically. Static per-run capabilities bind once per run. A `CapabilityFunc` result also binds before its own `for_run()` runs, while a specialized run-bound value returned by another capability's `for_run()` is not rebound.

## Select a Model Dynamically

Implement `get_model()` when reusable policy should choose the model, or use `SelectModel(selector)` for the common callable-only case. Return a model or model ID for a static choice, or return a sync/async callable accepting `ModelSelectionContext` to choose before every request step. The context exposes the agent, run dependencies, lower-precedence configured model on step one (then the previous step's model), step number, messages, and accumulated usage. Keep `get_model()` cheap; put I/O in an async selector. Static choices are resolved once per run, while a selector runs once per new logical request step and not again for same-step continuation. A model-less agent can be bootstrapped by a selector because the callable is first evaluated during run setup, after dependencies and history are available.

Explicit `run(model=...)`, run-spec, and `agent.override(model=...)` choices win and skip capability selection. Later capabilities override earlier model contributions. Same-step continuation remains pinned to its selected model; pass an explicit model when resuming a suspended provider-side request in another run.

Keep selection separate from construction. Use the `resolve_model_id()` hook, or the `ResolveModelId` convenience capability, when tenant, region, credentials, or another dependency controls how a selected string becomes a `Model` instance. Resolution uses the first non-`None` result in capability order; model selection uses the last non-`None` contribution.

Bootstrap strings use the post-`for_agent`, pre-`for_run` resolver chain. If `for_run()` replaces the capability, strings selected for step one and later use the replacement's resolver chain. Return a `FallbackModel` as the selected model when request failures, rather than routing policy, should trigger fallback.

Both hooks are eager: deferred capabilities do not select or resolve models. Run-spec capabilities can bootstrap a model-less agent, but `CapabilityFunc` and `for_run()` need an existing model to construct their `RunContext`; they can replace it starting on step one. Do not use adaptive selection with durable execution yet, and pass an explicit model when resuming a selector-backed suspended request in another run.

## Defer Capability Loading

For capabilities on demand, load [Capabilities on Demand](./ON-DEMAND-CAPABILITIES.md). Use it when the user mentions deferred capabilities, capability progressive disclosure, `defer_loading=True` on a capability, or `load_capability`; also use it proactively when an agent design includes optional instructions, specialist workflows, long-tail tools, or context the model does not need on most turns.
