# Concept Mapping

Map behavior only after tracing the active ADK caller path. Keep existing application infrastructure unless the migration explicitly includes it.

| Google ADK source | Normal Pydantic AI target | Focused proof |
|---|---|---|
| `LlmAgent(model, instruction, tools)` | Core `Agent(model, instructions, tools)` | Prompt, tool calls, final output, and errors |
| Dynamic `instruction` / deprecated `global_instruction` / `GlobalInstructionPlugin` | Instructions functions using typed `RunContext` dependencies; attach tree-wide instructions to every migrated agent or one shared capability | Trusted values reach prompts but not tool schemas and apply to the same agents |
| `input_schema` / `output_schema` | Existing input adapter plus `output_type`; typed graph/node data where applicable | Accepted/rejected payloads and wire shape |
| `FunctionTool` / Python callable | `@agent.tool`, `@agent.tool_plain`, `Tool`, or `FunctionToolset` | Name, schema, validation, error, and side effect |
| `ToolContext` / `CallbackContext` | `RunContext` plus typed dependencies; application services for persistence | Identity and clients are not model arguments |
| `Runner.run_async()` / `Event.is_final_response()` | `Agent.run()`, `run_stream_events()`, or an application adapter | Sync/async form, IDs, event order, final response |
| `Session.events` | Normalized `ModelMessage` history when it is model context | Multi-turn replay and persisted reload |
| `Session.state`, including `app:`, `user:`, and `temp:` keys | Application/workflow state with explicit scope and lifetime | Cross-session isolation, invocation lifetime, concurrency |
| `MemoryService` and `load_memory` | Existing retrieval service, or Harness [`Memory`](https://pydantic.dev/docs/ai/harness/memory/) if its store/search policy matches | Ingestion, search ranking, tenant scope, deletion |
| `ArtifactService` and artifact deltas | Existing versioned blob/file service | Version lookup, user/session scope, cleanup, errors |
| `output_key` | Explicit Python/graph output flow or an application state write | Downstream value and persistence timing |
| `Workflow` graph / `node` / `Event.route` | Plain async Python or `pydantic_graph` | Routes, typed inputs/outputs, fan-out/join, failures |
| Dynamic `ctx.run_node()` | Plain async composition, `pydantic_graph`, and a durable backend only when required | Scheduling order, execution IDs, retries, restart |
| `SequentialAgent` / `ParallelAgent` / `LoopAgent` | Plain Python, `pydantic_graph`, or explicit agent composition | Order, context isolation, stop condition, race/failure behavior |
| `sub_agents` with `mode='chat'` | Core delegation/programmatic hand-off/model routing | Which agent sees which history and answers the user |
| `sub_agents` with `mode='task'` or `mode='single_turn'` | A typed agent tool or Harness `SubAgents` only when isolation, return, interaction, and concurrency semantics match | Input visibility, return control, user interaction, parallelism, output |
| `AgentTool` | A typed tool that runs another `Agent`; Harness `SubAgents` when the task is self-contained | Input prompt, usage propagation, history isolation, returned output |
| `before_*` / `after_*` callbacks | Core [`Hooks`](https://pydantic.dev/docs/ai/core-concepts/hooks/) | Order, mutation, short-circuit, errors, sync/async behavior |
| Runner-wide `BasePlugin` | One reusable capability or application instrumentation/policy | Global coverage, lifecycle, early exit, cleanup |
| `require_confirmation`, `ToolConfirmation` | `requires_approval` or `ApprovalRequiredToolset` to gate the tool; resolve inline with `HandleDeferredToolCalls`, or return `DeferredToolRequests` and resume a later run with `DeferredToolResults`; use `ToolApproved.override_args` and deferred metadata when the confirmation payload changes execution | Correlation, approve/deny, overridden arguments, metadata, no side effect before approval |
| Workflow `RequestInput` | Application or graph pause/resume boundary | Prompt, response correlation, restart; do not imply authorization |
| `LongRunningFunctionTool` | Deferred external execution, application jobs, or durable steps | Pending response, correlation, completion/error delivery |
| `McpToolset` | Core [MCP](https://pydantic.dev/docs/ai/capabilities/mcp/) toolsets | Transport lifecycle, tool filtering/names, auth, errors |
| `SkillToolset` | Harness [`Skills`](https://pydantic.dev/docs/ai/harness/skills/) for portable `SKILL.md` instructions | Discovery, selection, instruction loading |
| ADK code executor / environment | Harness [`FileSystem`](https://pydantic.dev/docs/ai/harness/filesystem/) and [`Shell`](https://pydantic.dev/docs/ai/harness/shell/), or [`ModalSandbox`](https://pydantic.dev/docs/ai/harness/modal-sandbox/) | Containment, credentials, timeout, persistence, output |
| Event and text streaming | `run_stream_events()` for lifecycle events; `run_stream()` for output streaming | Chunk boundaries, event types/order, completion, cancellation |
| Context compaction | Core/Harness compaction selected by the observed policy | Preserved pinned content and behavior near token limits |
| Resume/replay | Core durable integrations or Harness step persistence, chosen by required guarantees | Kill/restart and idempotent external effects |
| Realtime/live | Core realtime when modality/provider contracts match | Audio/text turns, interruption, tool calls, session closure |
| A2A, CLI/web/API server, deploy, evals | Keep the application/protocol boundary; migrate separately if requested | Existing consumer and operational tests |

## Routing rules

- Prefer core for the agent loop, normalized messages, typed dependencies, tools, output, hooks, streaming, MCP, approvals, and durable-runtime primitives.
- Use Harness for optional reusable policy such as memory, skills, subagents, planning, workspace tools, context management, guardrails, or an isolated execution environment.
- Keep auth, stores, queues, endpoints, deployment, and product state in the application.
- Record a gap when the target cannot preserve a contract through a supported public API. Build a bounded adapter only after the impact is known.

Google ADK's current concepts are documented under [agents](https://adk.dev/agents/), [workflows](https://adk.dev/workflows/), [runtime](https://adk.dev/runtime/), [sessions](https://adk.dev/sessions/), [tools](https://adk.dev/tools-custom/), [callbacks](https://adk.dev/callbacks/), [plugins](https://adk.dev/plugins/), and [skills](https://adk.dev/skills/). Check the installed source because experimental and workflow APIs move quickly.
