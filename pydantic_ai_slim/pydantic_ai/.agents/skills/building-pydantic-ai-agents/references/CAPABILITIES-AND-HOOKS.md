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

From tool-validation and tool-execution hooks you can raise `ModelRetry` (the model should retry the call) or `ToolFailed` (the call is done and failed — the model sees the result and adapts, without consuming the retry budget) to redirect a tool call in one place instead of per tool.

At wrap boundaries, `ModelRetry` is control flow and bypasses `on_model_request_error`, `on_tool_execute_error`, and `on_output_process_error`. `ToolFailed` bypasses only `on_tool_execute_error`; from model-request or output-process hooks it is an ordinary exception passed to the corresponding error hook.

For deferrals (`ApprovalRequired`, `CallDeferred`), the rule is that a tool call can only be deferred once its arguments have been validated, since whoever resolves it is shown those arguments. So they are allowed from `after_tool_validate`, from `wrap_tool_validate` after `handler()` returns, and from every tool-execution hook — prefer `before_tool_execute`, since deferring after the tool body ran means its side effects happened and its result is discarded. Raising one from `before_tool_validate`, from `wrap_tool_validate` before `handler()`, or from `on_tool_validate_error` is a `UserError`. For a per-tool decision, use the tool's `args_validator` instead of a hook.

Use hooks when the user wants observability, auditing, or light interception without adding a new abstraction.

## Publish and React to Capability Events

A capability announces things to other capabilities and to the host application with `CapabilityEvent`, not with the application-owned `CustomEvent` covered in AGENTS-CORE.md. The split is enforced: emitting a `CustomEvent` from a capability raises `UserError`, and so does emitting a `CapabilityEvent` from application code.

Define each event as a dataclass subclass carrying a namespace, and await `ctx.emit(event)` from an async capability hook or a tool the capability contributes. The namespace plus the event name form the serialized `kind` (`workspace.file_read`), which is the wire identifier, so a published capability should pin `name=` on each event. The payload can't reuse the envelope's field names: `data`, `capability_id`, `tool_call_id`, `tool_name`, and `event_kind` are rejected at class definition.

React with `@on_event(SomeEvent, OtherEvent)` on an async capability method; a bare `@on_event` sees every `AgentStreamEvent`. Listeners run in capability order, then definition order, and the emitting capability receives its own events. This is the way to coordinate two capabilities without a shared object between them.

Always name the classes when you can. They are what lets dispatch skip a capability without descending into it — a bare `@on_event`, or an overridden `on_event()`, opts that capability into every event in the run. Override `listens_to(event)` alongside a custom `on_event()` if you can report something narrower.

Capability events reach the agent run event stream but UI adapters do not forward them, since they are internal signals. To surface one to a frontend, react to it from application code and emit your own `CustomEvent` with the public payload.

For application-level listening, prefer `@agent.on_event(SomeEvent)` — it takes the same event classes, filtering and `timeout=` as `Hooks.on.event` without a separate capability, runs after the capabilities' own listeners, and survives an overridden root capability. Reach for a `Hooks` capability when you also want hook families other than events, or when you need to choose where it sits among the other capabilities.

An event class can declare `dispatch='immediate'` when listeners must act before the emitter continues — a decision event, where a listener mutates a field the emitter then reads. Default dispatch delivers the event at its stream position instead, keeping listener work off the emitter's latency path.

```python
from dataclasses import dataclass
from typing import Any

from pydantic_ai import CapabilityEvent, RunContext
from pydantic_ai.capabilities import AbstractCapability, on_event


@dataclass(kw_only=True)
class FileReadEvent(CapabilityEvent, namespace='workspace', name='file_read'):
    path: str


class Watcher(AbstractCapability[Any]):
    @on_event(FileReadEvent)
    async def _on_read(self, ctx: RunContext[Any], event: FileReadEvent) -> None:
        if event.path.endswith('AGENTS.md'):
            ctx.enqueue('Follow the instructions in the discovered AGENTS.md file.')
```

## Build a Custom Capability

Subclass `AbstractCapability` when the user wants reusable behavior that combines tools, hooks, instructions, or model settings into one package.

Reach for a custom capability when:

- the same bundle should be reused across multiple agents
- `Hooks` alone is not enough
- the behavior should be installable or declarative

Keep custom capabilities focused. If the user only needs one tool or one hook, do not introduce a capability.

Give a capability an `id` when its instructions should be addressable. You declare `InstructionPart.name`; the framework issues `InstructionPart.id` as an `InstructionId`, whose source identifies the agent, toolset, or capability and whose optional name identifies one named part. It renders and serializes as `agent`, `toolset:<id>`, `capability:<id>`, or one of those followed by `:<name>`. Compare the structured source and name in Python; use `str(part.id)` only when a persisted string key is required. Naming a part whose source has no `id` of its own leaves `id` as `None`: the name still travels with the part, but nothing addresses it.

### Isolate Mutable State Per Run

Assume a capability instance may outlive and be reused across runs. By default, `for_run()` returns `self`, so every run uses that same instance. Mutating its fields from hooks, tools, dynamic callables, or other code executed during a run will therefore share state across sequential and concurrent runs.

Before adding an instance field, classify it as immutable configuration, a deliberately shared concurrency-safe resource, or per-run state. If code executed during a run mutates per-run state, override `for_run()` to return a fresh instance. Do not mutate and return `self`, and do not rely on clearing shared state in `after_run()`; overlapping runs can still interfere, and `after_run()` is skipped when a run produces no result. Use `wrap_run()` with `try`/`finally` when external resources need cleanup on errors or cancellation.

For dataclass capabilities, keep configuration in normal fields and declare per-run state with `init=False` so `dataclasses.replace()` preserves configuration while reinitializing the run state:

```python
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic_ai import AgentRunResult, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import ModelRequestContext

logger = logging.getLogger(__name__)


@dataclass
class RequestCounter(AbstractCapability[Any]):
    label: str
    _requests: int = field(default=0, init=False, repr=False)

    async def for_run(self, ctx: RunContext[Any]) -> RequestCounter:
        return replace(self)

    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        self._requests += 1
        return request_context

    async def after_run(
        self,
        ctx: RunContext[Any],
        *,
        result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        logger.info('%s made %d model requests', self.label, self._requests)
        return result
```

`replace()` is shallow: do not mutate nested mutable configuration from any code executed during a run. Make that data run-local too, or copy it explicitly in `for_run()`. Capabilities with no mutable per-run state can keep the default `for_run()` implementation.

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
