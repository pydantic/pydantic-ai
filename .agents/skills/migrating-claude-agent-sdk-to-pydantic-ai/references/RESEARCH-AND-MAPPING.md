# Research and concept mapping

Use this reference conditionally after tracing the source path. It is a decision guide, not a requirement to reproduce every Claude Code feature.

## Research baseline

Research was refreshed on 2026-09-15 against these primary sources:

- Claude Agent SDK for Python `0.2.152`, release commit `a8b1e285f97f8dbcb7b10226d74ba0d551b493f4`, bundled Claude Code CLI `2.1.259`: [source and types](https://github.com/anthropics/claude-agent-sdk-python/tree/a8b1e285f97f8dbcb7b10226d74ba0d551b493f4), [Python reference](https://code.claude.com/docs/en/agent-sdk/python), [agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop), and [sessions](https://code.claude.com/docs/en/agent-sdk/sessions).
- Pydantic AI commit `fbedb5911418a83812cd7de7e45fa23f84f05587`, five commits after `v2.43.0`: [agent API](https://github.com/pydantic/pydantic-ai/blob/fbedb5911418a83812cd7de7e45fa23f84f05587/docs/agent.md), [message history](https://github.com/pydantic/pydantic-ai/blob/fbedb5911418a83812cd7de7e45fa23f84f05587/docs/message-history.md), [hooks](https://github.com/pydantic/pydantic-ai/blob/fbedb5911418a83812cd7de7e45fa23f84f05587/docs/hooks.md), and [deferred tools](https://github.com/pydantic/pydantic-ai/blob/fbedb5911418a83812cd7de7e45fa23f84f05587/docs/deferred-tools.md).
- Pydantic AI Harness `0.31.0`, commit `d8787b74fa043908bca6dd0f5cc7f1f2e1c3a50a`: [capability matrix](https://github.com/pydantic/pydantic-ai-harness/blob/v0.31.0/README.md) and [core boundary](https://github.com/pydantic/pydantic-ai-harness/blob/v0.31.0/agent_docs/core-boundary.md). Harness is a `0.x` package; inspect the installed version before using examples here.

The source SDK launches a bundled Claude Code process, passes configuration through CLI flags and a control protocol, and parses stream-JSON frames into Python message classes. That subprocess architecture is not a caller contract by itself. Preserve configured behavior, ordered outputs/events, errors, state, and external effects.

## Ownership map

| Observed Claude Agent SDK behavior | Target owner and likely seam | Semantic difference to prove |
|---|---|---|
| `query()` model/tool loop and `ResultMessage.result` | **Core:** `Agent.run()` / `run_sync()`, `AgentRunResult.output` | Pydantic AI returns one run result rather than exposing Claude Code's subprocess result envelope. Preserve caller output, errors, usage requirements, and side effects, not the envelope. |
| `ClaudeAgentOptions.system_prompt`, model, thinking, limits | **Core:** agent/run instructions, model/provider, model settings, `UsageLimits`; **Harness:** `SpendLimits` only for an observed cross-run budget policy | Claude Code presets and `max_turns`/estimated USD result subtypes have no name-for-name equivalent. Test the actual stopping and reporting contract. |
| Custom `@tool` and in-process SDK MCP server | **Core:** `@agent.tool`, `@agent.tool_plain`, `Tool`, toolsets; use MCP only when protocol interoperability matters | Trusted services belong in `deps`, not model arguments. Test schema validation, retry/error semantics, concurrency, output content, and side effects. |
| Built-in `Read`/`Write`/`Edit`/`Glob`/`Grep`/`Bash` | **Harness:** `FileSystem`, `Shell`, `RepoContext`, or `Coder`; **Application/infrastructure:** workspace and isolation | Tool names, patches, output truncation, and permission defaults differ. Test the file/process outcome and escape resistance. Shell policy is not isolation. |
| Built-in web tools | **Core:** web-search/web-fetch capabilities where supported; keep an existing application integration when it owns the contract | Provider-native and local fallbacks have different events, citations, network policy, and credentials. Test the caller-visible result and egress boundary. |
| External MCP servers | **Core:** `MCPToolset`, `load_mcp_toolsets()` | Familiar config shapes do not guarantee every Claude-specific field. Test discovery, collisions/prefixes, transport, credentials, error behavior, and lifecycle. Live toggle/status/reconnect parity may be a gap. |
| `allowed_tools`, `disallowed_tools`, permission modes, `can_use_tool` | **Core:** tool exposure/preparation, approval-required tools, deferred results; optional **Harness** guardrail policy; **Application:** authenticated policy/UI/audit | `allowed_tools` auto-approves but does not restrict availability. Source callbacks can be shadowed by prior allow rules. Prove availability and pre-effect allow/deny/ask outcomes separately. |
| Structured `output_format` and `structured_output` | **Core:** typed `output_type`, Pydantic models, output validators and modes | Target output is a validated Python value. Test invalid-output retry exhaustion as well as success; do not fall back to parsing text silently. |
| Completed assistant/tool/result messages | **Core:** normalized model messages plus agent stream events; **Application:** adapter for a retained public event schema | Event classes and terminal markers differ. Capture a golden trace and drain the source stream when trailing events matter. |
| Raw partial `StreamEvent`s | **Core:** `run_stream()`, `run_stream_events()`, `event_stream_handler`, or `agent.iter()` according to consumer intent | Output deltas, lifecycle events, and graph nodes are separate target surfaces. Test reconstruction, ordering, completion, duplicate-final-text avoidance, and cancellation. |
| Live multi-turn client | **Core:** repeated `Agent.run(..., message_history=...)`; **Application:** connection/UI loop | A persistent target client object is unnecessary. Preserve the next-turn context and any mid-run input/interrupt behavior the caller observes. |
| Disk resume, `SessionStore`, transcript list/read/rename/tag | **Core:** serialized normalized messages; **Application:** storage, indexing, tenancy and metadata | Pydantic AI does not own Claude transcript JSONL or session helpers. Store `all_messages_json()` or application records and prove restart behavior. |
| Session fork/truncating resume | **Application + Core:** select/copy validated history and begin a new correlated conversation; optional **Harness:** `StepPersistence` for settled snapshots | `conversation_id` is correlation, not a resume token. Prove branch point, source immutability, new lineage, tool-call pairing, and side-effect safety. |
| File checkpoint rewind | **Application/infrastructure:** VCS, overlay, snapshot, or workspace owner | Message-history rewind does not rewind files. Treat exact checkpoint/rewind behavior as a gap unless the application supplies and tests it. |
| Hooks and matchers | **Core:** `Hooks` or a custom capability; **Application:** audit/integration effects | Match by lifecycle and allowed mutation rather than names. Claude same-event matchers may run concurrently; target capabilities run in order. Golden-test firing, inputs, decisions, errors, retry and streaming behavior. |
| Model-directed `AgentDefinition` subagents | **Harness:** `SubAgents`; **Core:** agent-as-tool for a fixed handoff | Children have independent history and receive an explicit task. Test isolation, dependencies, budgets, output handback, events and recursion policy. Keep deterministic workflows in application code. |
| Agent Skills | **Harness:** `Skills` | Harness exposes `SKILL.md` instructions on demand. Verify discovery and loading; bundled resources/scripts and Claude setting-source behavior are not automatic parity. Visibility is not secret protection. |
| Local plugins | Decompose into **Core**, **Harness**, and **Application** owners | Do not recreate Claude plugin packaging. Map each used skill, agent, hook, command, and MCP server. Packaging/namespace parity is a gap unless caller-visible. |
| Planning/task tracking | **Harness:** `Planning` when a model-owned plan is observable | A plan is not message history, workflow state, or memory. Configure a store only when persistence is required and test tenant/session keys. |
| Cross-run model memory | **Harness:** `Memory` for the notebook semantics it supports; otherwise **Application:** retain the existing store/retrieval | Do not replace product/domain memory by name. Test restart, namespace isolation, concurrency and bounded injection. |
| Compaction/context management | **Core:** history processors/provider compaction; optional **Harness:** model-agnostic compaction, output limits, conversation search | Compaction is not persistence. Test retained facts, tool pairing, thresholds, cache behavior, and whether omitted history remains retrievable. |
| Interrupts and cancellation | **Core:** cancellation token/task cancellation; **Application:** transport semantics | Prove which in-flight model/tool work stops, terminal error/result shape, cleanup, and whether the next turn can continue. |
| Durable restart/replay | **Core:** Temporal/DBOS/Prefect/other durable integrations when required; optional Harness durability capabilities | This is separate from chat-history resume. Test process death at model/tool/approval boundaries and idempotency of effects. |
| Cost, usage, rate-limit and telemetry events | **Core:** run usage, OTel instrumentation; optional **Harness:** spend policy; **Application:** billing and retained event adapters | Estimated client cost is not authoritative billing. Source event identity and trace IDs do not carry over automatically. Test required metrics and privacy configuration. |
| HTTP, WebSocket, queues, deployment and auth | **Application/infrastructure** | Keep the existing public boundary and workload shape. Pydantic AI CLI/web helpers do not replace production authentication, tenancy, scheduling, secrets, scaling, or isolation. |

## State and security checks

Ask five independent questions before selecting persistence:

1. Which model messages must the next turn see?
2. Which application/workflow state must survive?
3. Which in-flight work must resume after failure?
4. Which model-owned plan or long-term memory must persist?
5. Which workspace changes must survive, branch, or rewind?

Likewise, keep tool exposure, automatic approval, authenticated authorization, and process isolation separate. A model-facing deny rule can reduce accidents, but only the application and infrastructure can enforce tenant identity, protect secrets, and contain arbitrary code.
