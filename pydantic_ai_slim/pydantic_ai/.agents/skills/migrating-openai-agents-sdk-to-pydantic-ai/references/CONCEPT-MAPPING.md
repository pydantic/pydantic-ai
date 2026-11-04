# Concept Mapping

Use this reference only for features present in the active source path. Confirm the source and target behavior against the installed versions before editing code.

## Core agent and tools

| OpenAI Agents SDK | First Pydantic AI seam | What to verify |
|---|---|---|
| `Agent`, `Runner.run()` / `run_sync()` | reusable `Agent`, `run()` / `run_sync()` | input, output, errors, model requests, and tool loop |
| static or callable `instructions` | `instructions` or `@agent.instructions` | dynamic values and what becomes model-visible |
| `RunContextWrapper.context` | typed `deps` and `RunContext.deps` | trusted identity and clients never become tool arguments |
| `@function_tool` / `FunctionTool` | `@agent.tool`, `@agent.tool_plain`, `Tool`, or a toolset | name, description, schema, validation, retries, error shape, and side effects |
| `output_type` | `output_type`, optionally `NativeOutput` or `ToolOutput` | provider transport, validation, retry, and public wire shape |
| `tool_use_behavior` terminal result | output function or `ToolOutput` | which call wins, whether sibling tools execute, and terminal value |
| `ModelSettings` and `RunConfig` | agent/run `model`, `model_settings`, `UsageLimits`, and hooks | precedence, provider support, retry/counting, and failure behavior |
| `ModelBehaviorError` / tool input correction | Pydantic validation plus `ModelRetry` where the model can correct the call | retry budget and caller-visible terminal error |
| `Agent.clone(...)` variants | construct an explicit configured variant; use `Agent.override(...)` only for a scoped temporary override | shared tool containers, concurrent isolation, and variant lifetime |

OpenAI hosted tools do not all share one replacement:

- Map web search, file search, code execution, image generation, and hosted MCP to Pydantic AI [native tools](https://pydantic.dev/docs/ai/tools-toolsets/native-tools/) only when the model profile supports them and their outputs/events are sufficient.
- Map local function tools and local MCP servers to Pydantic AI tools or [`MCPToolset`](https://pydantic.dev/docs/ai/mcp/client/).
- Treat OpenAI `ComputerTool`, `ShellTool`, `ApplyPatchTool`, and `SandboxAgent` as execution-boundary decisions. Harness [`FileSystem`](https://pydantic.dev/docs/ai/harness/filesystem/), [`Shell`](https://pydantic.dev/docs/ai/harness/shell/), `Coder`, and [`ModalSandbox`](https://pydantic.dev/docs/ai/harness/modal-sandbox/) are candidates, not drop-in equivalents. Preserve workspace, process, network, secret, and cleanup boundaries with real integration tests.
- OpenAI hosted execution and skills may be provider-owned. Harness skills expose `SKILL.md` guidance to the model; they do not imply the same hosted container, file materialization, or provider tool behavior.

## Orchestration

OpenAI exposes two distinct LLM-directed patterns:

| Source behavior | Candidate target | Semantic difference |
|---|---|---|
| `Agent.as_tool()` | a Pydantic AI tool that runs a specialist, or Harness `SubAgents` | manager remains in control in both designs, but history, usage, limits, approval, and event propagation require explicit tests |
| handoff | application router, a typed routing output followed by another agent run, or an explicit orchestration layer | OpenAI changes the active agent inside the same run; a Pydantic AI tool call normally returns to the caller |

Do not replace a handoff with a subagent merely because both invoke a specialist. Inventory these contracts first:

- whether the specialist or manager writes the final answer;
- whether the receiving agent sees full, filtered, or nested history;
- whether handoff metadata is model-generated and validated;
- which input/output guardrails run;
- lifecycle event names and ordering;
- whether `last_agent` selects the next conversational turn;
- parent/child usage, limits, retries, and cancellation.

Use plain Python for deterministic routing. Use [`pydantic-graph`](https://pydantic.dev/docs/ai/graph/graph/) only when explicit typed nodes, branching, or persisted workflow state remain useful.

## Guardrails and hooks

Harness provides [`InputGuardrail`, `OutputGuardrail`, and `ToolGuardrail`](https://pydantic.dev/docs/ai/harness/guardrails/), but matching names are not parity evidence.

| OpenAI behavior | Required decision or proof |
|---|---|
| input guardrails apply to the first agent only | identify the actual public input boundary and test handoff/subagent paths |
| input guardrails run in parallel by default | choose sequential blocking when no model/tool work may start, or accept and test speculative work |
| output guardrails apply to the final agent only | prove the same terminal boundary and sanitization/error contract |
| tool guardrails wrap eligible local tools, not all hosted tools or handoffs | enumerate tools actually covered and enforce security below the model layer |
| tripwires raise SDK-specific exceptions | preserve the API error shape with an adapter or accept a documented change |
| `RunHooks` span a run and `AgentHooks` scope to one agent | map observations to Pydantic AI hooks/capabilities and test order across delegation |
| `call_model_input_filter` replaces model input immediately before a request | use a focused model-request hook or message-history processor and prove every request shape |

Guardrails are policy, not authentication or isolation. Authorization belongs inside application services and protected tools, using authenticated dependencies.

## State, sessions, and approval

Choose one state strategy per observed contract:

| OpenAI source | Meaning | Target owner |
|---|---|---|
| `result.to_input_list()` | caller-managed replay-ready conversation input | core `message_history`, serialized with `ModelMessagesTypeAdapter`; application owns storage |
| `Session` | SDK loads, merges, and persists client-managed history | application history repository, or Harness `StepPersistence` only when its settled snapshots/event/effect semantics are wanted |
| `previous_response_id` | OpenAI Responses server-side chain | `OpenAIResponsesModelSettings.openai_previous_response_id`; verify storage/ZDR and reasoning continuity |
| OpenAI `conversation_id` | OpenAI Conversations API state | `OpenAIResponsesModelSettings.openai_conversation_id`; do not confuse it with Pydantic AI's correlation `conversation_id` |
| `RunState` plus interruptions | resumable paused execution with approval decisions | core deferred tools with inline `HandleDeferredToolCalls` or stored `DeferredToolRequests`/`DeferredToolResults`, Harness step persistence, or a durable integration according to crash/replay requirements |

Pydantic AI's `conversation_id` groups runs and traces; it is not itself a message store. Message history preserves conversation context, not arbitrary workflow/checkpoint state.

For human approval:

1. Use deferred tools or raise `ApprovalRequired` based on the call and trusted dependencies.
2. Choose the flow per pending call: resolve with `HandleDeferredToolCalls` when the decision is available during the same call; otherwise include `DeferredToolRequests` in `output_type`, store the paused messages and complete request—or an equivalent pending-action record with category, validated arguments, and metadata—at an authenticated server-side boundary, and resume in a later run with `DeferredToolResults`, a new run ID, and the same conversation ID. A handler may resolve some calls and let the rest bubble up.
3. Re-check authorization and idempotency inside the tool before the side effect.
4. Test approve, deny, foreign/unknown ID, stale schema, and duplicate decisions. For the later-run flow, also test duplicate resume and process restart to the extent the source promised them.

Use a Pydantic AI durable integration when the source path promises replay or crash recovery across model/tool steps. Use Harness `StepPersistence` when its snapshot, event-log, continuation/fork, and effect-ledger contract fits. Neither follows merely from the word "session."

## Streaming and observability

OpenAI `run_streamed().stream_events()` can expose raw Responses events, run-item events, and agent lifecycle events. Pydantic AI exposes model deltas, tool/lifecycle events, and final results through several APIs; raw event types and completion timing differ.

Choose the smallest Pydantic AI streaming surface that includes the required lifecycle, then adapt it to the existing public schema. `run_stream()` commits the first matching output and may skip tool calls emitted alongside or after it; use a loop-completing event or graph surface unless that terminal behavior is part of the source contract. Test incremental delivery, order, IDs, tool and final events, approval interruption, cancellation, early consumer exit, and terminal errors with the real client boundary.

OpenAI tracing is enabled by default and has OpenAI-specific span/export behavior. Pydantic AI uses OpenTelemetry and integrates directly with Logfire. Retaining an existing exporter, dual-running temporarily, and switching to Logfire have different dashboard, alert, privacy, retention, and cost consequences. Trace similarity can corroborate a test, but cannot prove state, authorization, exactly-once side effects, or public streaming delivery.

Primary references: [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/), [agents](https://openai.github.io/openai-agents-python/agents/), [running agents](https://openai.github.io/openai-agents-python/running_agents/), [orchestration](https://openai.github.io/openai-agents-python/multi_agent/), [sessions](https://openai.github.io/openai-agents-python/sessions/), [guardrails](https://openai.github.io/openai-agents-python/guardrails/), [human-in-the-loop](https://openai.github.io/openai-agents-python/human_in_the_loop/), [streaming](https://openai.github.io/openai-agents-python/streaming/), and [Pydantic AI comparisons](https://pydantic.dev/docs/ai/comparisons/vs-openai-agents-sdk/).
