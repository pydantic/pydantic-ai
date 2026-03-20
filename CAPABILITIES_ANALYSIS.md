# Capabilities API — Architecture Sketch

## Core Abstraction: `AbstractCapability[AgentDepsT]`

A dataclass + ABC with two categories of methods:

**Static (called once at Agent construction):**
- `get_instructions()` → `Instructions[AgentDepsT] | None`
- `get_model_settings()` → `ModelSettings | Callable[[RunContext], ModelSettings] | None`
- `get_toolset()` → `AbstractToolset | ToolsetFunc | None`
- `get_builtin_tools()` → `Sequence[AbstractBuiltinTool | BuiltinToolFunc]`

**Dynamic (called per model request):**
- `before_model_request(ctx, request_context)` → `BeforeModelRequestContext`
- `after_model_request(ctx, *, response)` → `ModelResponse`

**Spec support (for YAML/JSON):**
- `get_serialization_name()` → class name or `None` to opt out
- `from_spec(*args, **kwargs)` → instance

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
  │
  ├─ Construction time:
  │   _root_capability = CombinedCapability(capabilities + [HistoryProcessorCapability(hp) for hp in history_processors])
  │   _instructions += root_capability.get_instructions()
  │   _builtin_tools += root_capability.get_builtin_tools()
  │   toolsets += root_capability.get_toolset()
  │
  ├─ Per run step (get_model_settings closure):
  │   model defaults → agent settings → capability settings → run settings
  │
  └─ Per model request (_agent_graph.py):
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

**Skip exceptions:** `SkipModelRequest(response)`, `SkipToolValidation(validated_args)`, `SkipToolExecution(result)` — can be raised in before/wrap hooks to short-circuit.

**CombinedCapability** chains: before forward, after reversed, wrap nested (outer wraps inner).

## Toolset-State Branch (builds on hooks)

Adds per-run and per-step lifecycle to `AbstractToolset`:
- `for_run(ctx)` → `AbstractToolset` — return a fresh instance for run isolation (default: `self`)
- `for_run_step(ctx)` → `AbstractToolset` — return a per-step instance (default: `self`)

Also adds `for_run(ctx)` to `AbstractCapability` so capabilities can create per-run state too.

Refactors `DynamicToolset` to use these hooks instead of the `copy()` + `visit_and_replace` mechanism, and removes ref-counting from `CombinedToolset`.

---

# Feature-to-Capability Mapping

## 1. Middleware / Hooks

| Library Feature | Maps to | Notes |
|---|---|---|
| **deepagents** `before_run` / `after_run` | `before_run` / `after_run` hooks | Direct match |
| **deepagents** `before_model_request` | `before_model_request` | Direct match |
| **deepagents** `before_tool_call` (with ALLOW/DENY/ASK) | `before_tool_execute` + `SkipToolExecution` | Raise `SkipToolExecution("Denied: reason")` to block. For ASK, raise `ApprovalRequired` |
| **deepagents** `after_tool_call` | `after_tool_execute` | Direct match |
| **deepagents** `on_tool_error` | `wrap_tool_execute` (catch exceptions in the wrapper) | Clean fit — `wrap_model_request` pattern handles this |
| **deepagents** `on_error` | `wrap_run` (catch exceptions) | Same pattern |
| **deepagents** `CostTrackingMiddleware` | Capability with `wrap_run` + `after_model_request` | Track usage in `after_model_request`, enforce budget in `before_model_request` |
| **deepagents** `ContextManagerMiddleware` | Capability with `before_model_request` (for auto-compression) + `after_model_request` (for token tracking) | Clean fit |
| **deepagents** `HooksMiddleware` (shell command hooks) | Capability with `before_tool_execute` / `after_tool_execute` that shells out | Clean fit |
| **deepagents** `CheckpointMiddleware` | Capability with `after_model_request` to snapshot | Clean fit |
| **ya-agent-sdk** per-tool pre/post hooks with `CallMetadata` | `before_tool_execute` / `after_tool_execute` on a capability, with state shared via the capability instance | The `CallMetadata` dict pattern maps to per-call state on the capability. Would need a way to pass state between before and after — our hooks don't have this explicitly but a capability is stateful, so it can use an instance dict |
| **ya-agent-sdk** `GlobalHooks` (pre/post on all tools) | Same hooks, just without tool-name filtering | Our hooks already fire for all tools. Filtering by name would be done inside the hook impl |
| **ya-agent-sdk** stream-level hooks (`on_runtime_ready`, `pre_node_hook`, `pre_event_hook`, etc.) | `wrap_run` + `wrap_run_event_stream` | `wrap_run` covers `on_runtime_ready`/`on_agent_start`/`on_agent_complete`. `wrap_run_event_stream` covers event-level hooks. **Gap:** No explicit `pre_node_hook`/`post_node_hook` (before/after each model request node vs tool call node). Our `before_model_request`/`after_model_request` covers model nodes. Tool nodes are covered by `before_tool_execute`/`after_tool_execute`. So the node-level abstraction is implicitly covered |
| **code_puppy** callback system (30+ phases) | Various capability hooks | Most phases map to our hooks. Some are very specific (e.g., `register_tools`, `register_agents`) and are better handled by `Agent` construction |
| **code_puppy** hook engine (config-driven, shell command hooks) | Capability with spec support | A `ShellHooks` capability that reads config and dispatches shell commands in `before_tool_execute`/`after_tool_execute` |
| **code_puppy** plugin system | Capabilities ARE the plugin system | Each plugin becomes a capability |

**Verdict: Clean fit.** The before/after/wrap triple at run, model-request, tool-validate, and tool-execute levels covers every hook point these libraries need. The one nuance is per-tool-call state sharing between before and after hooks — a capability instance can handle this with a `dict` keyed by call ID, or we could consider adding a `CallMetadata` to the hook context.

## 2. Message History Processing

| Library Feature | Maps to | Notes |
|---|---|---|
| **deepagents** `PatchToolCallsProcessor` (fix orphaned tool calls/returns) | Capability with `before_model_request` | Modify `request_context.messages` to patch orphans. Or just a `HistoryProcessor` wrapped in `HistoryProcessorCapability` |
| **deepagents** `EvictionProcessor` (truncate large tool outputs) | Capability with `after_tool_execute` (truncate at source) or `before_model_request` (truncate in history) | Both approaches work |
| **deepagents** `SummarizationProcessor` | Capability with `before_model_request` | This is exactly what `HistoryProcessorCapability` already wraps. A `Compaction` capability could use the `before_model_request` hook to summarize when approaching context limits |
| **deepagents** `SlidingWindowProcessor` | Same pattern | Simpler version — just drop old messages |
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
| **deepagents** `AgentMemoryToolset` (read/write/update MEMORY.md + system prompt injection) | `Memory` capability with `get_instructions()` + `get_toolset()` | Perfect capability use case — it provides both instructions (injected memory content) and tools (read/write/update) |
| **ya-agent-sdk** `MemoryManager` (key-value + XML injection in runtime instructions) | Same pattern | `get_instructions()` for injection, `get_toolset()` for manipulation tools |
| **code_puppy** pickle-based session persistence | `Sessions` capability or just agent-level config | Session persistence is more of an infrastructure concern than a per-request capability, but a `Sessions` capability could provide the save/load tools |

**Verdict: Clean fit.** Memory is a textbook capability — it needs both instructions injection and tools.

## 4. Tool Management

| Library Feature | Maps to | Notes |
|---|---|---|
| **deepagents** tool approval via `DeferredToolRequests` | Capability with `before_tool_execute` that raises `ApprovalRequired` | Or a standalone `Approval` capability |
| **deepagents** `SkillsToolset` (discover/load/run skills) | `Skills` capability with `get_toolset()` + `get_instructions()` | Skills list in instructions, load/run as tools |
| **deepagents** `ContextToolset` (project file injection) | `ContextFiles` capability with `get_instructions()` | Just instructions, no tools needed |
| **ya-agent-sdk** tool capability tags + superseding | `get_toolset()` returning a custom toolset that filters | The toolset itself handles availability logic. A capability could wrap this |
| **ya-agent-sdk** `ToolSearchToolSet` (dynamic tool discovery) | Capability with `get_toolset()` | The meta-toolset is the capability's toolset |
| **ya-agent-sdk** `SkillToolset` (markdown skills) | Same as deepagents skills | `Skills` capability |
| **code_puppy** tool registry + per-agent tool selection | `get_toolset()` returning a filtered toolset | Each capability provides its own tools, agent picks capabilities |
| **code_puppy** file permission handler | Capability with `before_tool_execute` | Check permissions, show diff, prompt user |
| **code_puppy** shell safety agent | Capability with `before_tool_execute` that runs a sub-agent for risk assessment | Clean fit — the sub-agent call happens inside the hook |
| **code_puppy** universal constructor (dynamic tool creation at runtime) | Capability with `get_toolset()` + `for_run` for state | Would need toolset-state's `for_run_step` to handle dynamically added tools |

**Verdict: Clean fit.** Tool management maps to either `get_toolset()` (for providing tools), `get_instructions()` (for injecting tool-related context), or `before_tool_execute`/`wrap_tool_execute` (for gating/approval).

## 5. Sub-agents

| Library Feature | Maps to | Notes |
|---|---|---|
| **deepagents** `SubAgentToolset` (task/check_task/answer) | `SubAgents` capability with `get_toolset()` | Provides delegation tools. Needs access to deps for agent creation |
| **deepagents** `DynamicAgentRegistry` | Same capability, more tools | Dynamic agent creation tools |
| **deepagents** `AgentTeam` (shared todo, message bus) | `Teams` capability with `get_toolset()` | Provides team coordination tools |
| **ya-agent-sdk** subagent configs from markdown | `SubAgents` capability constructed from config files | `from_spec()` could load markdown configs |
| **ya-agent-sdk** `MessageBus` (inter-agent communication) | Part of a `SubAgents` or `Teams` capability | Could also inject messages via `before_model_request` |
| **code_puppy** `invoke_agent` / `list_agents` | `SubAgents` capability | Same pattern |

**Verdict: Clean fit.** Sub-agent orchestration is a capability that provides tools (delegate, check, cancel) and possibly instructions (available agents list) and hooks (inject bus messages in `before_model_request`).

## 6. Streaming / Events

| Library Feature | Maps to | Notes |
|---|---|---|
| **deepagents** cost tracking callbacks | Capability with `after_model_request` | Track usage from `response.usage` |
| **ya-agent-sdk** lifecycle events (ModelRequestStart, ToolCallsComplete, etc.) | `wrap_run_event_stream` + `wrap_model_request` + `wrap_tool_execute` | Event emission in wrap hooks |
| **ya-agent-sdk** merged multi-agent event stream | Infrastructure-level concern, not per-capability | Agent-level, not capability-level |

**Verdict: Mostly clean.** `wrap_run_event_stream` handles most observability needs.

## 7. Configuration / Context

| Library Feature | Maps to | Notes |
|---|---|---|
| **deepagents** `OutputStyle` (response style injection) | `Instructions` capability | Just instructions |
| **ya-agent-sdk** model presets | `ModelSettings` capability | Just settings |
| **ya-agent-sdk** Jinja2 system prompt templates | `Instructions` capability (which already supports `TemplateStr`) | Clean fit |
| **code_puppy** AGENT.md/AGENTS.md rules loading | `ContextFiles` capability with `get_instructions()` | Reads files, injects as instructions |
| **code_puppy** model-specific prompt handling | `before_model_request` to transform based on model | Or model-level concern |

**Verdict: Clean fit.**

## 8. Features That Need Monkey-Patching Today

These are telling — they show what Pydantic AI doesn't expose cleanly:

| code_puppy Monkey-Patch | Our Solution |
|---|---|
| `ToolManager._call_tool` for pre/post callbacks | `before_tool_execute` / `after_tool_execute` hooks — **eliminates the need** |
| `ToolManager._call_tool` for JSON repair | `before_tool_validate` hook — **eliminates the need** |
| `_clean_message_history` disable | `before_model_request` gives full control — **eliminates the need** |
| `_process_message_history` validation skip | `before_model_request` — **eliminates the need** |

**This is a strong validation signal.** The hooks system directly addresses the pain points that drove code_puppy to monkey-patch.

---

# Gaps / Things to Consider

1. **Per-tool-call state sharing between before/after hooks**: deepagents' `CallMetadata` and ya-agent-sdk's `CallMetadata` both pass a mutable dict from pre-hook to post-hook for the same tool invocation. Our hooks don't have this. A capability can use its own instance state keyed by `call.tool_call_id`, but that's not as ergonomic. Worth considering whether `wrap_tool_execute` is sufficient (it naturally scopes state) or if we want explicit shared context.

2. **Tool-name filtering in hooks**: All three libraries support filtering hooks to specific tools. Our hooks fire for ALL tools, requiring the capability to filter internally. This is fine for code-level capabilities but less ergonomic for config-driven hooks. A `tool_names: set[str] | None` parameter on the hook methods (or a declarative filter on the capability) could help.

3. **`for_run` on capabilities**: The toolset-state branch adds `for_run()` to `AbstractCapability`, which is essential for capabilities that accumulate per-run state (cost tracking, memory writes, etc.). **This confirms you should merge toolset-state.**

4. **No `before_run_step` / `after_run_step`**: ya-agent-sdk has `pre_node_hook`/`post_node_hook` which fire before/after each agent graph node (model request or tool calls). Our hooks cover the model request side but not the "run step" granularity explicitly. This is implicitly covered by `before_model_request` + tool hooks, but there's no hook that fires "once per step" regardless of what kind of step it is. Probably fine — the implicit coverage is sufficient.

5. **Dynamic tool availability**: ya-agent-sdk's `is_available(ctx)` per tool and tag-based superseding are toolset-level concerns, not capability-level. `for_run_step` on toolsets enables this — tools can change availability per step. **Another point for merging toolset-state.**

---

# Recommendation

**Merge hooks into capabilities first, then toolset-state into that.** The review issues are real but minor:
- Test file placement is a style fix
- The `Instructions` re-export and `root_capability` visibility change need quick decisions
- The `'running tools'` span re-addition needs a revert
- The streaming `wrap_model_request` complexity is inherent to the feature

The three libraries validate that the abstraction covers the full feature space. Every feature — middleware, history processing, memory, tool approval, skills, sub-agents, context injection, cost tracking, compaction — maps to some combination of:
- `get_instructions()` / `get_model_settings()` / `get_toolset()` / `get_builtin_tools()` (static contribution)
- `before/after/wrap_*` hooks (dynamic behavior)
- `for_run` / `for_run_step` (per-run state isolation)

The strongest validation signal is that code_puppy's monkey-patches become unnecessary — our hooks provide the extension points that forced them to patch internals.
