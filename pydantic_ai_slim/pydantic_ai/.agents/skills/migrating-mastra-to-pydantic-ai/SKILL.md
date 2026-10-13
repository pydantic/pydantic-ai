---
name: migrating-mastra-to-pydantic-ai
description: Migrate TypeScript Mastra applications to Python with Pydantic AI and, only when needed, Pydantic AI Harness. Use when source code imports `@mastra/*` or relies on Mastra agents, tools, workflows, memory, processors, streaming, approvals, skills, or subagents.
---

# Migrate Mastra to Pydantic AI

Preserve observable behavior, not TypeScript or Mastra's object model. Migrate the smallest complete caller path and keep application infrastructure in place.

## Trace the source before choosing a target

1. Read repository instructions, dependencies, tests, and the runtime entrypoint. Record the installed Mastra, Pydantic AI, and Harness versions.
2. Trace one real request from its public entrypoint, whether an `Agent.generate()` / `Agent.stream()` call or a workflow run, through the tools, processors, memory, events, results, state, and side effects that callers use. Establish a focused baseline or characterization test.
3. Separate these source contracts when present:
   - request context, model-chosen tool input, workflow input/output, and shared workflow state;
   - conversation messages, semantic recall, working or observational memory, workflow snapshots, and model-owned plans;
   - final text or objects, token deltas, lifecycle events, tripwires, and terminal workflow status;
   - tool approval, authenticated authorization, and process or sandbox isolation;
   - deterministic workflows, agent-selected delegation, queues, schedules, and durable execution.
4. Record each observed contract, its owner, semantic difference, and executable proof. An unused Mastra feature is not migration scope.

Read [Research and concept mapping](references/RESEARCH-AND-MAPPING.md) for the detected source features. Read [Verification and cutover](references/VERIFICATION-AND-CUTOVER.md) before implementation.

## Choose the smallest target

- **Core:** use `pydantic_ai.Agent` for the agent loop, typed dependencies, tools, outputs, normalized messages, generic hooks, streaming, MCP clients, approvals, usage limits, instrumentation, realtime, and durable-runtime integrations.
- **Graph:** use `pydantic_graph` only when explicit typed nodes, branches, and joins remain useful; use plain async Python for simple fixed control flow.
- **Harness:** add `pydantic-ai-harness` only for observed reusable policy such as memory notebooks, planning, model-directed subagents, Agent Skills, coding tools, guardrails, model-agnostic compaction strategies, or step persistence. Harness capabilities compose through the core agent loop; Harness is not a workflow engine or second runtime.
- **Evals:** add the separate `pydantic-evals` package for observed datasets, cases, and evaluators.
- **Application:** retain authentication, databases, vector search, queues, schedulers, deployment, product state, API/UI transports, session lookup, and tenant policy unless the requested slice includes them.
- **Gap:** name behavior that no supported public seam preserves, explain its impact, and test a bounded adapter. Do not reproduce the Mastra registry, server, storage schema, or event taxonomy merely to hide a difference.

The normal migration is one reusable `Agent`, application services supplied through typed dependencies, ordinary Pydantic AI tools, a typed output when callers expect structure, and application-owned storage of `result.all_messages()` for later turns. Preserve an existing HTTP, job, or UI boundary with a small adapter and its current field names. If the Mastra call is only in-process, migrate its nearest caller in the same slice or explicitly agree on a new application-owned service boundary; do not add a transport by default.

## Apply high-risk gates

- Mastra `RequestContext` is trusted run context, not model input. Map authenticated identity and services to typed dependencies; expose only model-chosen values as tool parameters.
- Mastra memory combines distinct behaviors. Map thread messages to serialized Pydantic AI message history, keep semantic retrieval behind an application service or tool, and use Harness `Memory` only when its model-owned notebook semantics preserve the observed working-memory contract. For existing Mastra records, explicitly choose and test one-time conversion, a read-through adapter, or starting fresh as an accepted change. Do not treat message history, a notebook, or Harness `StepPersistence` as a workflow snapshot.
- A Mastra workflow is deterministic application control flow. Use plain Python or `pydantic_graph`; do not move `.branch()`, `.parallel()`, `.foreach()`, loops, or suspend conditions into an agent prompt. Preserve concurrency, aggregation, failure, and cancellation semantics explicitly.
- Mastra workflow suspension persists a resumable snapshot. Keep branch and workflow-state persistence in the application or underlying durable engine and prove restart at the promised step. Pydantic AI durable integrations make agent model, tool, and MCP operations durable inside that workflow; they do not own its surrounding control flow. Deferred tools preserve agent tool calls, not arbitrary workflow state.
- Map processors by firing point, input/output mutation, tripwire behavior, ordering, streaming visibility, and persistence. Use the narrowest of core history processors, output validators, hooks, Harness input/output/tool guardrails, a custom capability, or an application adapter, and golden-test any caller-visible event or error.
- Approval pauses a validated action; it does not authenticate the approver. Keep identity, authorization, audit, correlation, and idempotency in the application, and prove the effect does not run before approval.
- Map consumer intent for streaming. Pydantic AI output streaming, run-event streaming, UI adapters, and graph iteration are different surfaces; none promises Mastra's `fullStream` chunk taxonomy.
- Use Harness `SubAgents` only for model-directed delegation. Keep deterministic fan-out, aggregation, retries, and scorer-driven verification loops in application or graph code.
- Mastra Agent Skills may include references and dynamic resolution. Harness `Skills` loads on-demand `SKILL.md` instructions only and does not automatically scan `.agents` or `.claude`; use Harness `FileSystem` or an application tool for required references, and test per-request selection separately.
- Mastra Studio, server adapters, storage providers, deployment targets, logs, trace storage, and live-eval scheduling are product infrastructure. Retain or replace them deliberately rather than treating the agent port as equivalent.
- Mastra durable agents, workflow time travel, exact snapshot compatibility, observational-memory compaction, and exact raw event identity are gaps unless a supported target seam is proven for the observed caller.

## Implement and prove one vertical slice

1. Preserve the supported caller boundary and replace only the agent-owned internals.
2. Start core-only. Add `pydantic_graph`, Harness, Pydantic Evals, or a durable runtime only after a source contract requires it.
3. Test inputs, typed outputs, errors, event order, tool arguments/results, processor decisions, state across turns, and side effects at that boundary. Use deterministic models and fake application services offline; add a focused recorded or live test only when provider behavior is the contract.
4. For persistence, approvals, concurrent workflows, or external effects, test interruption and restart, lineage, authorization, failure aggregation, and idempotency at the exact boundary promised by Mastra.
5. Remove `@mastra/*`, Mastra server setup, and Mastra storage or event adapters only after no retained path needs them.

Explain any consequential semantic change before implementing it: state the source behavior, target behavior, caller impact, recommended choice, and remaining risk.

## Completion

Apply the completion criterion in [Verification and cutover](references/VERIFICATION-AND-CUTOVER.md). Label evidence from fakes, recordings, and live providers accurately.
