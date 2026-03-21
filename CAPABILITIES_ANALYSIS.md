# Capabilities API — Architecture Sketch

## Core Abstraction: `AbstractCapability[AgentDepsT]`

A dataclass + ABC with two categories of methods:

**Static (called once at Agent construction):**
- `get_instructions()` -> `Instructions[AgentDepsT] | None`
- `get_model_settings()` -> `ModelSettings | Callable[[RunContext], ModelSettings] | None`
- `get_toolset()` -> `AbstractToolset | ToolsetFunc | None`
- `get_builtin_tools()` -> `Sequence[AbstractBuiltinTool | BuiltinToolFunc]`

**Dynamic (called per model request):**
- `before_model_request(ctx, request_context)` -> `BeforeModelRequestContext`
- `after_model_request(ctx, *, response)` -> `ModelResponse`

**Spec support (for YAML/JSON):**
- `get_serialization_name()` -> class name or `None` to opt out
- `from_spec(*args, **kwargs)` -> instance

### Composition: `CombinedCapability`

All capabilities on an agent are merged into a single `CombinedCapability` (`self._root_capability`), which:
- Flattens instructions, builtin tools
- Merges model settings (static merged eagerly, dynamic callables chained)
- Combines toolsets via `CombinedToolset`
- Chains `before_model_request` forward, `after_model_request` reversed

### Built-in Capabilities

| Class | What it provides |
|---|---|
| `Instructions` | Static/template instructions |
| `ModelSettings` | Static or dynamic model settings |
| `Thinking` | Provider-specific thinking settings (placeholder) |
| `WebSearch` | The web search builtin tool |
| `Toolset` | Wraps an `AbstractToolset` (no spec support) |
| `HistoryProcessorCapability` | Wraps a `HistoryProcessor` into `before_model_request` (no spec support) |

### Agent Integration

```
Agent(capabilities=[...])
  |
  +-- Construction time:
  |   _root_capability = CombinedCapability(capabilities + [HistoryProcessorCapability(hp) for hp in history_processors])
  |   _instructions += root_capability.get_instructions()
  |   _builtin_tools += root_capability.get_builtin_tools()
  |   toolsets += root_capability.get_toolset()
  |
  +-- Per run step (get_model_settings closure):
  |   model defaults -> agent settings -> capability settings -> run settings
  |
  +-- Per model request (_agent_graph.py):
      request_context = root_capability.before_model_request(ctx, ...)
      # ...model call...
      response = root_capability.after_model_request(ctx, response=response)
```

### YAML/JSON Spec (`AgentSpec`)

```yaml
model: anthropic:claude-opus-4-6
capabilities:
  - Instructions: "You are helpful"     # single-arg shorthand
  - Thinking                             # no-arg shorthand
  - ModelSettings:                       # kwargs shorthand
      max_tokens: 4096
  - WebSearch
  - MyCustomCapability:                  # custom types via custom_capability_types
      key: value
```

`Agent.from_spec(spec, custom_capability_types=[MyCustomCapability])` resolves names via a registry of `DEFAULT_CAPABILITY_TYPES` + custom types, calling `cls.from_spec(...)` for each.

---

## Hooks Branch (builds on capabilities)

Expands `AbstractCapability` with 4 lifecycle hook groups, each with before/after/wrap:

| Lifecycle | before | after | wrap |
|---|---|---|---|
| **Run** | `before_run(ctx)` | `after_run(ctx, *, result)` | `wrap_run(ctx, *, handler)` |
| **Model Request** | `before_model_request(ctx, request_context)` (already existed) | `after_model_request(ctx, *, response)` (already existed) | `wrap_model_request(ctx, *, request_context, handler)` |
| **Tool Validate** | `before_tool_validate(ctx, *, call, args)` | `after_tool_validate(ctx, *, call, args)` | `wrap_tool_validate(ctx, *, call, args, handler)` |
| **Tool Execute** | `before_tool_execute(ctx, *, call, args)` | `after_tool_execute(ctx, *, call, args, result)` | `wrap_tool_execute(ctx, *, call, args, handler)` |

Plus `wrap_run_event_stream(ctx, *, stream)` for streaming.

**Skip exceptions:** `SkipModelRequest(response)`, `SkipToolValidation(validated_args)`, `SkipToolExecution(result)` -- can be raised in before/wrap hooks to short-circuit.

**CombinedCapability** chains: before forward, after reversed, wrap nested (outer wraps inner).

## Toolset-State Branch (builds on hooks)

Adds per-run and per-step lifecycle to `AbstractToolset`:
- `for_run(ctx)` -> `AbstractToolset` -- return a fresh instance for run isolation (default: `self`)
- `for_run_step(ctx)` -> `AbstractToolset` -- return a per-step instance (default: `self`)

Also adds `for_run(ctx)` to `AbstractCapability` so capabilities can create per-run state too.

Refactors `DynamicToolset` to use these hooks instead of the `copy()` + `visit_and_replace` mechanism, and removes ref-counting from `CombinedToolset`.

---

# Feature-to-Capability Mapping (Third-Party Libraries)

## 1. Middleware / Hooks

| Library Feature | Maps to | Notes |
|---|---|---|
| **deepagents** `before_run` / `after_run` | `before_run` / `after_run` hooks | Direct match |
| **deepagents** `before_model_request` | `before_model_request` | Direct match |
| **deepagents** `before_tool_call` (with ALLOW/DENY/ASK) | `before_tool_execute` + `SkipToolExecution` | Raise `SkipToolExecution("Denied: reason")` to block. For ASK, raise `ApprovalRequired` |
| **deepagents** `after_tool_call` | `after_tool_execute` | Direct match |
| **deepagents** `on_tool_error` | `wrap_tool_execute` (catch exceptions in the wrapper) | Clean fit -- `wrap_tool_execute` pattern handles this |
| **deepagents** `on_error` | `wrap_run` (catch exceptions) | Same pattern |
| **deepagents** `CostTrackingMiddleware` | Capability with `wrap_run` + `after_model_request` | Track usage in `after_model_request`, enforce budget in `before_model_request` |
| **deepagents** `ContextManagerMiddleware` | Capability with `before_model_request` (for auto-compression) + `after_model_request` (for token tracking) | Clean fit |
| **deepagents** `HooksMiddleware` (shell command hooks) | Capability with `before_tool_execute` / `after_tool_execute` that shells out | Clean fit |
| **deepagents** `CheckpointMiddleware` | Capability with `after_model_request` to snapshot | Clean fit |
| **ya-agent-sdk** per-tool pre/post hooks with `CallMetadata` | `wrap_tool_execute` on a capability | `wrap_tool_execute` naturally scopes state -- local variables in the wrapper are visible before and after the inner `handler()` call. No need for explicit `CallMetadata`. |
| **ya-agent-sdk** `GlobalHooks` (pre/post on all tools) | Same hooks, just without tool-name filtering | Our hooks already fire for all tools. Check `call.tool_name` inside the hook if needed. |
| **ya-agent-sdk** stream-level hooks (`on_runtime_ready`, `pre_node_hook`, `pre_event_hook`, etc.) | `wrap_run` + `wrap_run_event_stream` | See dedicated section below on `pre_event_hook`. |
| **code_puppy** callback system (30+ phases) | Various capability hooks | Most phases map to our hooks. `register_tools` is just what capabilities *are* (see dedicated section below). |
| **code_puppy** hook engine (config-driven, shell command hooks) | Capability with spec support | A `ShellHooks` capability that reads config and dispatches shell commands in `before_tool_execute`/`after_tool_execute` |
| **code_puppy** plugin system | Capabilities ARE the plugin system | Each plugin becomes a capability |

**Verdict: Clean fit.** The before/after/wrap triple at run, model-request, tool-validate, and tool-execute levels covers every hook point these libraries need. `wrap_tool_execute` naturally scopes per-call state (local variables), so no explicit `CallMetadata` dict is needed. Tool-name filtering is just an `if` check inside the hook.

## 2. Message History Processing

| Library Feature | Maps to | Notes |
|---|---|---|
| **deepagents** `PatchToolCallsProcessor` (fix orphaned tool calls/returns) | Capability with `before_model_request` | Modify `request_context.messages` to patch orphans. Or just a `HistoryProcessor` wrapped in `HistoryProcessorCapability` |
| **deepagents** `EvictionProcessor` (truncate large tool outputs) | Capability with `after_tool_execute` (truncate at source) or `before_model_request` (truncate in history) | Both approaches work |
| **deepagents** `SummarizationProcessor` | Capability with `before_model_request` | This is exactly what `HistoryProcessorCapability` already wraps. A `Compaction` capability could use the `before_model_request` hook to summarize when approaching context limits |
| **deepagents** `SlidingWindowProcessor` | Same pattern | Simpler version -- just drop old messages |
| **deepagents** history archive search tool | Capability providing both `before_model_request` (to persist messages) and `get_toolset()` (search tool) | Perfect use case for a capability that spans hooks + tools |
| **ya-agent-sdk** `create_compact_filter()` (LLM-based compaction) | `Compaction` capability with `before_model_request` | Same pattern as deepagents' summarization |
| **ya-agent-sdk** 12+ history filters (system_prompt, runtime_instructions, bus_message, image, tool_args, etc.) | Each could be a capability or grouped into a few | Most are `before_model_request` hooks. Some (like `image` filter, `tool_args` fixer) could be `before_model_request` or `before_tool_validate` |
| **code_puppy** `message_history_accumulator` (dedup + compaction trigger) | Capability with `before_model_request` | Clean fit |
| **code_puppy** summarization/truncation strategies | `Compaction` capability | Same pattern |
| **code_puppy** `prune_interrupted_tool_calls` | Capability with `before_model_request` | Same as deepagents' PatchToolCalls |
| **code_puppy** `filter_huge_messages` | Capability with `before_model_request` | Straightforward |

**Verdict: Clean fit.** `before_model_request` is the natural hook point for all history processing. The existing `HistoryProcessorCapability` wrapper validates this, and richer capabilities (like `Compaction`) can use `before_model_request` directly with access to `RunContext` (which `HistoryProcessor` callables already receive).

## 3. Memory

| Library Feature | Maps to | Notes |
|---|---|---|
| **deepagents** `AgentMemoryToolset` (read/write/update MEMORY.md + system prompt injection) | `Memory` capability with `get_instructions()` + `get_toolset()` | Perfect capability use case -- it provides both instructions (injected memory content) and tools (read/write/update) |
| **ya-agent-sdk** `MemoryManager` (key-value + XML injection in runtime instructions) | Same pattern | `get_instructions()` for injection, `get_toolset()` for manipulation tools |
| **code_puppy** pickle-based session persistence | `Sessions` capability or just agent-level config | Session persistence is more of an infrastructure concern than a per-request capability, but a `Sessions` capability could provide the save/load tools |

**Verdict: Clean fit.** Memory is a textbook capability -- it needs both instructions injection and tools.

## 4. Tool Management

| Library Feature | Maps to | Notes |
|---|---|---|
| **deepagents** tool approval via `DeferredToolRequests` | Capability with `before_tool_execute` that raises `ApprovalRequired` | Or a standalone `Approval` capability |
| **deepagents** `SkillsToolset` (discover/load/run skills) | `Skills` capability with `get_toolset()` + `get_instructions()` | Skills list in instructions, load/run as tools |
| **deepagents** `ContextToolset` (project file injection) | `ContextFiles` capability with `get_instructions()` | Just instructions, no tools needed |
| **ya-agent-sdk** tool capability tags + superseding | `get_toolset()` returning a custom toolset that filters in `get_tools(ctx)` | The toolset itself handles availability logic. `get_tools(ctx)` already receives `RunContext` so dynamic filtering is possible today. |
| **ya-agent-sdk** `ToolSearchToolSet` (dynamic tool discovery) | Capability with `get_toolset()` | The meta-toolset is the capability's toolset |
| **ya-agent-sdk** `SkillToolset` (markdown skills) | Same as deepagents skills | `Skills` capability |
| **code_puppy** tool registry + per-agent tool selection | `get_toolset()` returning a filtered toolset | Each capability provides its own tools, agent picks capabilities |
| **code_puppy** file permission handler | Capability with `before_tool_execute` | Check permissions, show diff, prompt user |
| **code_puppy** shell safety agent | Capability with `before_tool_execute` that runs a sub-agent for risk assessment | Clean fit -- the sub-agent call happens inside the hook |
| **code_puppy** universal constructor (dynamic tool creation at runtime) | Capability with `get_toolset()` + `for_run` for state | Would need toolset-state's `for_run_step` to handle dynamically added tools |

**Verdict: Clean fit.** Tool management maps to either `get_toolset()` (for providing tools), `get_instructions()` (for injecting tool-related context), or `before_tool_execute`/`wrap_tool_execute` (for gating/approval).

## 5. Guardrails

| Library / Pattern | Maps to | Notes |
|---|---|---|
| **Input guardrails** (PII detection, prompt injection, content moderation) | Capability with `before_model_request` | Validate `request_context.messages`, raise on violation |
| **Concurrent input guardrails** (OpenAI Agents SDK pattern: run alongside model request, cancel if fails) | Capability with `wrap_model_request` | Fire guardrail as background task, start model call, cancel response if guardrail fails. Zero added latency on pass. |
| **Output guardrails** (hallucination detection, toxicity, secret redaction) | Capability with `after_model_request` | Validate/transform `response`. Or use existing output validators. |
| **Tool guardrails** (approval, argument validation, permission decisions) | Capability with `before_tool_validate` / `before_tool_execute` + `SkipToolExecution` or `ApprovalRequired` | ALLOW/DENY/ASK pattern maps directly |
| **Auto-retry with feedback** (Guardrails AI `reask`, jagreehal `max_retries`) | Capability with `wrap_model_request` or `wrap_run` | Retry loop inside the wrapper, with feedback injected into messages |
| **Schema-driven validation** (Guardrails AI) | Already covered by `output_type` + Pydantic validation | Pydantic AI's core feature |
| **jagreehal/pydantic-ai-guardrails** `GuardedAgent` (6 input + 12 output guardrails) | Individual `Guardrail` capabilities or grouped | Each guardrail (PII, toxicity, secret_redaction, etc.) becomes a capability |
| **vstorm-co/pydantic-ai-middleware** `AsyncGuardrailMiddleware` (BLOCKING / CONCURRENT / ASYNC_POST timing modes) | `before_model_request` (BLOCKING), `wrap_model_request` (CONCURRENT), `after_run` (ASYNC_POST) | All three timing modes have natural hook mappings |
| **vstorm-co/pydantic-ai-middleware** `ParallelMiddleware` (ALL_MUST_PASS, ANY_PASSES, etc.) | A `ParallelGuardrails` capability that runs multiple checks via `asyncio.gather` in `wrap_model_request` | Aggregation strategies are internal to the capability |
| **PR #3938** `InputGuardrail` / `OutputGuardrail` with `tripwire_triggered` | Capability hooks + custom exceptions | The PR's design becomes unnecessary -- capabilities subsume it |
| **Issue #4598** runtime governance (tool-level, inter-agent trust, audit) | Capability with `before_tool_execute` + `wrap_run` for audit logging | Blocking hooks are exactly what governance needs |
| **NeMo Guardrails** Colang dialog flow modeling | Out of scope | A framework-on-top, not a single capability |

**Verdict: Clean fit.** Every guardrail pattern maps to some combination of `before_model_request`, `after_model_request`, `wrap_model_request`, and `before_tool_execute`. The concurrent-with-model-request pattern (OpenAI's key innovation) maps beautifully to `wrap_model_request`. PR #3938 and issue #1197 are fully subsumed.

## 6. Sub-agents and Multi-Agent

| Library Feature | Maps to | Notes |
|---|---|---|
| **deepagents** `SubAgentToolset` (task/check_task/answer) | `SubAgents` capability with `get_toolset()` | Provides delegation tools. Needs access to deps for agent creation |
| **deepagents** `DynamicAgentRegistry` | Same capability, more tools | Dynamic agent creation tools |
| **deepagents** `AgentTeam` (shared todo, message bus) | `Teams` capability with `get_toolset()` | Provides team coordination tools |
| **ya-agent-sdk** subagent configs from markdown | `SubAgents` capability constructed from config files | `from_spec()` could load markdown configs |
| **ya-agent-sdk** `MessageBus` (inter-agent communication) | `MessageBus` capability instance shared between agents | See dedicated section below |
| **code_puppy** `invoke_agent` / `list_agents` | `SubAgents` capability | Same pattern |

**How MessageBus would work with capabilities:**

A `MessageBus` capability instance is shared across multiple agents by passing the same object to each:

```python
bus = MessageBus()
coordinator = Agent('model', capabilities=[bus, SubAgents([worker_config])])
worker = Agent('model', capabilities=[bus, ...])
```

The `MessageBus` capability:
- `before_model_request`: injects pending messages into the conversation
- `get_toolset()`: provides `send_message`, `broadcast`, `receive` tools
- `for_run()`: creates a per-run subscription cursor

The `SubAgents` capability:
- `get_toolset()`: provides `delegate`, `check_task`, `cancel_task` tools
- `get_instructions()`: lists available sub-agents
- The delegate tool internally creates and runs a child agent, potentially with a shared `MessageBus` instance
- `for_run()`: creates per-run task tracking state

Sub-agent creation happens inside the capability (factory pattern) since it needs `RunContext` for deps, model, etc. The capability holds configs and creates agents on-demand during tool execution.

## 7. Streaming / Events

| Library Feature | Maps to | Notes |
|---|---|---|
| **deepagents** cost tracking callbacks | Capability with `after_model_request` | Track usage from `response.usage` |
| **ya-agent-sdk** lifecycle events (ModelRequestStart, ToolCallsComplete, etc.) | `wrap_run_event_stream` + `wrap_model_request` + `wrap_tool_execute` | Event emission in wrap hooks |
| **ya-agent-sdk** merged multi-agent event stream | Infrastructure-level concern, not per-capability | Agent-level, not capability-level |

## 8. Configuration / Context

| Library Feature | Maps to | Notes |
|---|---|---|
| **deepagents** `OutputStyle` (response style injection) | `Instructions` capability | Just instructions |
| **ya-agent-sdk** model presets | `ModelSettings` capability | Just settings |
| **ya-agent-sdk** Jinja2 system prompt templates | `Instructions` capability (which already supports `TemplateStr`) | Clean fit |
| **code_puppy** AGENT.md/AGENTS.md rules loading | `ContextFiles` capability with `get_instructions()` | Reads files, injects as instructions |
| **code_puppy** model-specific prompt handling | `before_model_request` to transform based on model | Or model-level concern |

## 9. Features That Need Monkey-Patching Today

These are telling -- they show what Pydantic AI doesn't expose cleanly:

| code_puppy Monkey-Patch | Our Solution |
|---|---|
| `ToolManager._call_tool` for pre/post callbacks | `before_tool_execute` / `after_tool_execute` hooks -- **eliminates the need** |
| `ToolManager._call_tool` for JSON repair | `before_tool_validate` hook -- **eliminates the need** |
| `_clean_message_history` disable | `before_model_request` gives full control -- **eliminates the need** |
| `_process_message_history` validation skip | `before_model_request` -- **eliminates the need** |

**This is a strong validation signal.** The hooks system directly addresses the pain points that drove code_puppy to monkey-patch.

---

# Detailed Analysis of Specific Questions

## ya-agent-sdk `pre_event_hook` -- should that be a simpler option vs full `wrap_run_event_stream`?

**No.** Research shows ya-agent-sdk's `pre_event_hook` is observe-only -- it can't filter or transform events, just observe and inject extras via a queue. That's the 80% case (logging, metrics), but the 20% case (filtering, transforming, buffering) needs a full wrapper.

`wrap_run_event_stream` handles both: if you just want to observe, your wrapper yields events unchanged with a side effect. If you want to filter, you skip yields. The full wrapper is already simple enough that a convenience callback wouldn't save meaningful boilerplate. And adding both creates two extension points for the same lifecycle, which is what we're trying to avoid.

## code_puppy `register_tools` -- what's the point?

It's a plugin system for **injecting new tools** at agent construction time. Plugins return `{"name": ..., "register_func": ...}` dicts that get merged into the global tool registry. Without it, plugins can't add tools.

In our model, this is just what capabilities *are*. A plugin provides a capability, the capability's `get_toolset()` returns its tools. No special hook needed -- `Agent(capabilities=[...])` is the extension point.

## Wrapper toolset injection for tool filtering

There's a gap: if a capability wants to control which tools the model *sees* (not just execution gating), it has to mess with `model_request_parameters.function_tools` in `before_model_request`, which is awkward.

The cleanest solution would be a `prepare_tools`-style hook on capabilities: `prepare_tools(ctx, tools: list[ToolsetTool]) -> list[ToolsetTool]` that fires per-step and can filter/modify the tool list before it's sent to the model. This already exists as `prepare_tools` on `Agent` but not on capabilities. Making it available on capabilities would be a natural follow-up.

Alternatively, a new method like `get_toolset_wrapper() -> Callable[[AbstractToolset], AbstractToolset]` that wraps the combined toolset after all capabilities have contributed theirs. But `prepare_tools` is lighter and more consistent with the hooks pattern.

## Graph Progression / Node Interception via Capabilities

Today, capabilities only hook into the model request/response cycle. They cannot intercept graph node transitions (prevent End, force extra model request, substitute a different node).

Manual driving via `agent_run.next(node)` gives users *external* control between nodes, but that's not available to capabilities.

**Suggested addition (follow-up):** A `wrap_run_step` hook:

```python
async def wrap_run_step(
    self,
    ctx: RunContext[AgentDepsT],
    *,
    node: AgentNode,
    handler: Callable[[AgentNode], Awaitable[AgentNode]],
) -> AgentNode:
    """Wraps each step of the agent run.

    Receives the node about to execute. Call handler(node) to execute it
    and get the next node. Can:
    - Modify the node before execution
    - Inspect/modify the returned next node
    - Call handler multiple times (retry)
    - Return a different node entirely (redirect)
    - Raise to abort
    """
    return await handler(node)
```

This gives capabilities full control over graph progression without exposing graph internals. Composes via CombinedCapability the same way `wrap_model_request` does (nested middleware).

This is a power-user feature that could be a follow-up. The current hooks cover 95% of use cases.

## Dynamic Tool Availability

`AbstractToolset.get_tools(ctx)` already receives `RunContext` and is called per-step. A toolset can already return different tools per step based on `ctx`. The `for_run_step` hook from toolset-state adds *lifecycle management* (enter/exit for fresh instances), but the filtering itself was already possible via `get_tools(ctx)`.

---

# Conclusion

**Every significant feature across all three libraries + the guardrails ecosystem maps to some combination of:**

1. `get_instructions()` / `get_model_settings()` / `get_toolset()` / `get_builtin_tools()` -- static contribution at construction time
2. `before/after/wrap_*` hooks -- dynamic behavior per request/tool call
3. `for_run` / `for_run_step` -- per-run state isolation

**The strongest validation signals:**
- code_puppy's monkey-patches become unnecessary
- PR #3938 (guardrails) is fully subsumed
- vstorm's middleware library maps 1:1 to our hooks
- The concurrent-guardrails pattern (OpenAI's key innovation) maps beautifully to `wrap_model_request`

**Potential follow-ups (not blockers):**
- `prepare_tools` hook on capabilities for tool visibility filtering
- `wrap_run_step` hook for graph progression control
- Concrete `SubAgents` / `MessageBus` capability designs (deserve their own issues)
