---
name: migrating-agno-to-pydantic-ai
description: Migrate Python Agno applications to Pydantic AI and, only when needed, Pydantic AI Harness. Use when source code imports `agno` or relies on Agno agents, teams, workflows, sessions, memory, knowledge, tools, hooks, guardrails, approvals, skills, streaming, or AgentOS.
---

# Migrate Agno to Pydantic AI

Preserve observable behavior, not Agno's object model. Migrate the smallest complete caller path and keep application infrastructure in place.

## Trace the source before choosing a target

1. Read repository instructions, dependencies, tests, and the runtime entrypoint. Record the installed Agno, Pydantic AI, and Harness versions.
2. Trace one real request from `Agent.run()` / `arun()`, `Team.run()` / `arun()`, a `Workflow`, or an AgentOS endpoint through instructions, model and tool calls, hooks, session loading, memory or knowledge retrieval, events, results, state, and side effects. Establish a focused baseline or characterization test.
3. Separate these source contracts when present:
   - trusted dependencies, model-chosen tool input, session state, and workflow state;
   - conversation history, session summaries, user memories, knowledge retrieval, workflow checkpoints, and model-owned plans;
   - final content, structured output, token deltas, run events, pauses, and terminal status;
   - confirmation, user input, external tool execution, authenticated authorization, and process isolation;
   - team delegation, deterministic workflow control, AgentOS transport, queues, and deployment.
4. Record each observed contract, its owner, semantic difference, and executable proof. An unused Agno feature is not migration scope.

Read [Research and concept mapping](references/RESEARCH-AND-MAPPING.md) for the detected features. Read [Verification and cutover](references/VERIFICATION-AND-CUTOVER.md) before implementation.

## Choose the smallest target

- **Core:** use `pydantic_ai.Agent` for the agent loop, typed dependencies, tools, outputs, normalized messages, generic hooks, streaming, MCP clients, approvals, usage limits, instrumentation, realtime, and durable-runtime integrations.
- **Graph:** use `pydantic_graph` only when explicit typed nodes, branches, and joins remain useful; use plain async Python for simple fixed control flow.
- **Harness:** add `pydantic-ai-harness` only for observed reusable policy such as memory notebooks, planning, model-directed subagents, Agent Skills, coding tools, guardrails, model-agnostic compaction, or step persistence. Harness capabilities compose through the core agent loop; Harness is not a second runtime.
- **Evals:** add the separate `pydantic-evals` package for observed datasets and evaluators.
- **Application:** retain authentication, databases, vector search, queues, AgentOS/API routes, storage schemas, deployment, product state, and UI transports unless the requested slice includes them.
- **Gap:** name behavior that no supported public seam preserves, explain its impact, and test a bounded adapter. Do not rebuild Agno's registry, database abstraction, event taxonomy, or AgentOS merely to hide a difference.

The normal migration is one reusable `Agent`, application services supplied through typed dependencies, typed tools and output, and application-owned storage of serialized `result.all_messages()` for later turns. Preserve an existing HTTP, job, or UI boundary with a small adapter and its current field names.

## Apply high-risk gates

- Agno `dependencies`, `user_id`, `session_id`, and authenticated identity are trusted runtime context, not model arguments. Put them in typed dependencies and authorize storage or effects in application code.
- Agno sessions combine distinct owners. Map chat history to serialized Pydantic AI messages; keep mutable product/session state in an application store; treat summaries, user memories, knowledge retrieval, workflow checkpoints, and Harness `Memory` as separate contracts. Choose and test record conversion, a read-through adapter, or starting fresh as an accepted change.
- Inspect a `Team`'s actual delegation mode and caller-visible member events. Use Harness `SubAgents` only for model-directed, isolated task delegation. Keep routing, broadcast, deterministic fan-out, shared-context collaboration, aggregation, and retries in application or graph code unless parity is proved.
- Keep Agno `Workflow` steps, conditions, routers, loops, and parallel joins deterministic. Do not move them into an agent prompt. Graph state alone is not a persisted checkpoint.
- Agno pauses for confirmation, user input, and external execution have different payloads and owners. Core deferred tools can preserve pending model tool calls; the application still owns identity, authorization, UI, audit, persistence, correlation, and idempotency. Prove restart when the source pause survives one.
- Map pre/post hooks, tool hooks, and guardrails by firing point, mutation, short-circuit behavior, ordering, retries, streaming visibility, and persistence. Similar hook names do not prove lifecycle parity.
- Map consumer intent for streaming. Pydantic AI output streaming, run-event streaming, capability events, UI adapters, and graph iteration are different surfaces; none promises Agno's run-event taxonomy or resume cursor.
- Agno Skills can expose instructions, references, and scripts. Harness `Skills` loads `SKILL.md` instructions on demand but does not load bundled resources or execute scripts; add explicit `FileSystem`, `Shell`, or application tools only for observed, trusted behavior.
- AgentOS routes, auth, session APIs, telemetry, control plane, database selection, and deployment are product infrastructure. Keep or replace each deliberately rather than treating an agent port as an AgentOS port.
- Tool allowlists, path checks, and guardrails are policy, not OS isolation. Use a container, VM, or cloud sandbox when untrusted execution is in scope.

## Implement and prove one vertical slice

1. Preserve the supported caller boundary and replace only agent-owned internals.
2. Start core-only. Add `pydantic_graph`, Harness, Evals, or a durable runtime only after a source contract requires it.
3. Test inputs, typed outputs, errors, event order, tool arguments/results, hook decisions, state across turns, and side effects at that boundary. Use deterministic models and fake application services offline; add a focused recorded or live test only when provider behavior is the contract.
4. For persistence, pauses, concurrent workflows, or external effects, test interruption and restart, lineage, authorization, failure aggregation, and idempotency at the exact boundary Agno promised.
5. Remove `agno`, AgentOS setup, and Agno storage or event adapters only after no retained path needs them.

Explain any consequential semantic change before implementing it: state the source behavior, target behavior, caller impact, recommended choice, and remaining risk.

## Completion

Apply the completion criterion in [Verification and cutover](references/VERIFICATION-AND-CUTOVER.md). Label evidence from fakes, recordings, live providers, and operational tests accurately.
