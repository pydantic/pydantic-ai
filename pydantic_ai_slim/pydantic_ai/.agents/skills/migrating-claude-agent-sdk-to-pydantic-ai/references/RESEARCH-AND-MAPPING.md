# Research and concept mapping

Use this reference conditionally after tracing the source path. It is a decision guide, not a requirement to reproduce every Claude Code feature.

## Primary documentation

Inspect the installed versions before migrating because the Claude Agent SDK, Pydantic AI, and Harness continue to evolve.

- Claude Agent SDK: [Python reference](https://code.claude.com/docs/en/agent-sdk/python), [agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop), and [sessions](https://code.claude.com/docs/en/agent-sdk/sessions).
- Pydantic AI: [agents](https://pydantic.dev/docs/ai/core-concepts/agent/), [message history](https://pydantic.dev/docs/ai/core-concepts/message-history/), [hooks](https://pydantic.dev/docs/ai/core-concepts/hooks/), and [deferred tools](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/).
- Pydantic AI Harness: [overview](https://pydantic.dev/docs/ai/harness/), [shell execution](https://pydantic.dev/docs/ai/harness/shell/), [subagents](https://pydantic.dev/docs/ai/harness/subagents/), and [step persistence](https://pydantic.dev/docs/ai/harness/step-persistence/).

The source SDK launches a bundled Claude Code process, passes configuration through CLI flags and a control protocol, and parses stream-JSON frames into Python message classes. That subprocess architecture is not a caller contract by itself. Preserve configured behavior, ordered outputs/events, errors, state, and external effects.

## Ownership map

| Observed Claude Agent SDK behavior | Target owner and likely seam | Focused proof |
|---|---|---|
| `query()` model/tool loop and `ResultMessage.result` | **Core:** `Agent.run()` / `run_sync()`, `AgentRunResult.output` | Assert caller output, errors, required usage data, and side effects without preserving the subprocess envelope. |
| `ClaudeAgentOptions.system_prompt`, model, thinking, limits | **Core:** instructions, model/provider settings, `UsageLimits(request_limit=..., cost_limit=...)`; **Harness:** `SpendLimits` only for observed cross-run spend policy | Assert that limit exhaustion raises `UsageLimitExceeded` and translate it if callers expect a Claude result subtype. Core cost limits are best-effort, depend on available pricing, and may be checked after a billed response; retain provider/application spend controls when the source promises a hard cap. |
| Custom `@tool` and in-process SDK MCP server | **Core:** `@agent.tool`, `@agent.tool_plain`, `Tool`, toolsets; MCP only for protocol interoperability | Assert schema validation, retry/error behavior, concurrency, output content, and side effects. |
| Built-in `Read`/`Write`/`Edit`/`Glob`/`Grep`/`Bash` | **Harness:** `FileSystem`, `Shell`, `RepoContext`, or `Coder`; **Application/infrastructure:** `ModalSandbox` or existing isolation | Assert file/process outcomes, path escape resistance, command containment, and output limits. |
| Built-in web tools | **Core:** `WebSearch` and `WebFetch` capabilities, backed by `WebSearchTool` and `WebFetchTool` where the provider supports them; otherwise retain the application integration | Assert result shape, citations/events consumed by callers, credentials, and egress policy. |
| External MCP servers | **Core:** `MCPToolset`, `load_mcp_toolsets()` | Assert discovery, collisions/prefixes, transport, credentials, errors, lifecycle, and any required live control. |
| `allowed_tools`, `disallowed_tools`, permission modes, `can_use_tool` | **Core:** tool preparation; `requires_approval` or `ApprovalRequiredToolset` to gate the tool; resolve inline with `HandleDeferredToolCalls`, or return `DeferredToolRequests` and resume a later run with `DeferredToolResults`; optional **Harness** policy; **Application:** authorization/UI/audit | Assert availability and each pre-effect allow, deny, or ask outcome independently, including source callback precedence. |
| Structured `output_format` and `structured_output` | **Core:** typed `output_type`, Pydantic models, output validators and modes | Assert validated values and invalid-output retry exhaustion; do not silently parse fallback text. |
| Completed assistant/tool/result messages | **Core:** normalized messages and agent stream events; **Application:** retained event adapter | Compare a stable-field golden trace and include trailing source events. |
| Raw partial `StreamEvent`s | **Core:** `run_stream()`, `run_stream_events()`, `event_stream_handler`, or `agent.iter()` according to consumer intent | Assert reconstruction, order, terminal detection, no duplicate final text, and cancellation. |
| Live multi-turn client | **Core:** repeated `Agent.run(..., message_history=...)`; **Application:** connection/UI loop | Assert next-turn context plus observed mid-run input or interrupt behavior. |
| Disk resume, `SessionStore`, transcript list/read/rename/tag | **Core:** serialized normalized messages; **Application:** storage, indexing, tenancy, metadata | Round-trip through the real store and continue in a fresh process. |
| Session fork/truncating resume | **Application + Core:** copy validated history and start a correlated conversation; optional **Harness:** `StepPersistence` for settled snapshots | Assert branch point, source immutability, new lineage, tool-call pairing, and side-effect safety. |
| File checkpoint rewind | **Application/infrastructure:** VCS, overlay, snapshot, or workspace owner | Assert filesystem state independently of message history; otherwise record a tested gap. |
| Hooks and matchers | **Core:** `Hooks` or a custom capability; **Application:** audit/integration effects | Golden-test firing, stable inputs, decisions, mutation, errors, ordering, retry, and streaming behavior. |
| Model-directed `AgentDefinition` subagents | **Harness:** `SubAgents`; **Core:** agent-as-tool for a fixed handoff | Assert task-only input, isolated history, dependencies, budgets, handback, events, and recursion policy. |
| Agent Skills | **Harness:** `Skills` | Assert `SKILL.md` discovery/loading and test any resources, scripts, or setting-source behavior separately. |
| Local plugins | Decompose used skills, agents, hooks, commands, and MCP servers across **Core**, **Harness**, and **Application** | Assert each used feature; record packaging or namespace parity as a gap only when caller-visible. |
| Planning/task tracking | **Harness:** `Planning` only for an observed model-owned plan | Assert persistence and tenant/session keys only when required. |
| Cross-run model memory | **Harness:** `Memory` for its notebook semantics; otherwise retain application storage/retrieval | Assert restart, namespace isolation, concurrency, and bounded injection. |
| Compaction/context management | **Core:** history processors/provider compaction; optional **Harness:** compaction, output limits, conversation search | Assert retained facts, tool pairing, thresholds, cache behavior, and retrieval of omitted history. |
| Interrupts and cancellation | **Core:** cancellation token/task cancellation; **Application:** transport semantics | Assert stopped work, terminal shape, cleanup, and next-turn continuation. |
| Durable restart/replay | **Core:** durable runtime integration; optional Harness durability capabilities | Kill and restart at each promised boundary; assert external effects are idempotent. |
| Cost, usage, rate-limit and telemetry events | **Core:** run usage and OTel; optional **Harness:** spend policy; **Application:** billing and retained adapters | Assert required metrics, event/trace mapping, and privacy settings; do not treat estimated cost as billing. |
| HTTP, WebSocket, queues, deployment and auth | **Application/infrastructure** | Exercise the retained production boundary, including tenancy, scheduling, secrets, scaling, and isolation. |
