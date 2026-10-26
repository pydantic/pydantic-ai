---
name: migrating-vercel-ai-sdk-and-eve-to-pydantic-ai
description: Migrate TypeScript Vercel AI SDK or Eve applications to Python with Pydantic AI and, only when needed, Pydantic AI Harness. Use when source code imports `ai`, `@ai-sdk/*`, or `eve`, or relies on `generateText`, `streamText`, `ToolLoopAgent`, `useChat`, Eve agents, tools, skills, subagents, approvals, sessions, channels, schedules, or sandboxes.
---

# Migrate Vercel AI SDK and Eve to Pydantic AI

Preserve the application's observable behavior, not the source APIs. Vercel AI SDK Core, AI SDK UI, and Eve have different responsibilities: identify which one owns each source behavior before choosing a target.

## Trace the source before choosing a target

1. Read repository instructions, dependencies, lockfiles, tests, and the runtime entrypoint. Resolve `workspace:*` dependencies to concrete package versions and record the installed AI SDK, Eve, Pydantic AI, and Harness versions.
2. Trace one real request from its public entrypoint through messages, model calls, tools, approvals, streams, persistence, side effects, and the result the caller consumes. Establish a focused baseline or characterization test.
3. Classify every observed contract:
   - **AI SDK Core:** model calls, step loops, tool execution, structured output, and provider routing;
   - **AI SDK UI:** `UIMessage`, `useChat`, transports, client tools, stream chunks, and reconnect behavior;
   - **Eve:** filesystem-defined agents, durable sessions, skills, subagents, state, connections, channels, schedules, and sandboxes;
   - **Application/platform:** authentication, authorization, storage, queues, deployment, credentials, product UI, and hosted Vercel services.
4. Record each contract, its owner, semantic difference, and executable proof. An unused feature is not migration scope.

Read [Research and concept mapping](references/RESEARCH-AND-MAPPING.md) for the detected source features. Read [Verification and cutover](references/VERIFICATION-AND-CUTOVER.md) before implementation.

## Choose the smallest target

- **Core:** use `pydantic_ai.Agent` for the agent loop, typed dependencies, tools, outputs, messages, streaming, MCP clients, approvals, usage limits, instrumentation, and durable-runtime integrations.
- **Graph:** use `pydantic_graph` only when explicit typed nodes, branches, and joins remain useful; use plain async Python for simple fixed control flow.
- **Harness:** add `pydantic-ai-harness` only for observed reusable policy such as Agent Skills, model-directed subagents, planning, memory notebooks, coding tools, guardrails, or step persistence. Harness capabilities compose through the core agent loop; Harness is not an Eve-compatible runtime.
- **Evals:** add the separate `pydantic-evals` package for observed datasets, cases, and evaluators.
- **Application:** retain UI, HTTP transports, auth, persistence, reconnect state, schedulers, deployment, tenant policy, and hosted services unless the requested slice includes replacing them.
- **Gap:** name behavior that no supported public seam preserves, explain its impact, and test a bounded adapter. Do not recreate the AI SDK or Eve object model merely to hide a difference.

The normal AI SDK migration is one reusable `Agent` behind the existing caller. Treat `@ai-sdk/workflow` as a durable-runtime migration, not an ordinary agent loop. When a TypeScript or React client uses AI SDK UI, keep it and serve its exact installed-version protocol with `VercelAIAdapter`; do not migrate the UI unnecessarily. When the source is only an in-process TypeScript call, migrate its nearest caller in the same slice or explicitly agree on a new application-owned service boundary.

The normal Eve migration is a core agent plus only the Harness capabilities that match observed behavior. Keep Eve's durable session driver, product endpoints, schedules, channels, authentication, and hosted services in the application until each has a proved replacement.

## Apply high-risk gates

- `generateText()` and `streamText()` are single-step by default, `ToolLoopAgent` defaults to a longer loop, and `WorkflowAgent` has no maximum step count unless configured. Preserve the observed stop conditions, usage limits, and `prepareStep` changes to models, tools, instructions, messages, or runtime context; do not copy defaults blindly.
- Distinguish local, provider-executed, browser/client, and MCP tools. Preserve where execution and credentials live, input/result schemas, error recovery, cancellation propagation, and side-effect idempotency.
- An AI SDK async-generator tool can expose preliminary results before its final output. An ordinary Pydantic AI tool returns once; reproduce consumed intermediate UI states with explicit custom events or record an intentional change.
- Complete structured output is validated; partially streamed output may be incomplete, and combining tools with structured output may require another step. Test the caller-visible timing and failure shape.
- `UIMessage` is client-controlled application state, not trusted provider history or dependency context. Validate incoming messages, keep identity and services server-side, and convert only the permitted content to model messages.
- Match the installed AI SDK wire contract. The current docs call the SSE-based UI-message wire format the Data Stream Protocol, while older implementations can use line-prefixed frames under a similar name. Request bodies, headers, frames, chunk ordering, tool states, error/abort behavior, and reconnect endpoints are not interchangeable. Configure `VercelAIAdapter` for the installed SDK version and golden-test the boundary.
- AI SDK approval is commonly a two-request conversation; Eve can durably park a session; Pydantic AI can resolve a deferred call inline or in a later run. Persist the complete pending request, bind a decision to validated arguments and call identity, authenticate the approver, and prove the protected effect runs exactly once.
- Pydantic AI message history cannot continue past unresolved tool calls. If Eve accepts an unrelated turn while approval remains pending, keep the pending branch separate in application state; do not append an unrelated prompt to that incomplete history.
- Eve sessions, turns, steps, replay units, event cursors, background subagents, and NDJSON session streams are one durable protocol. Core message history and Harness `StepPersistence` do not automatically reproduce it. Retain an application-owned session adapter or mark the difference until restart and replay tests prove the promised behavior.
- Eve `defineState` data is separate from conversation history, and durable memory has its own lifetime and sharing rules. Map each store independently and test session, user, and tenant isolation.
- Eve skills can materialize supporting files. Harness `Skills` loads `SKILL.md` instructions only; provide required references, assets, or scripts through `FileSystem` or application tools and test them explicitly.
- Eve subagents may be durable child sessions with background result batching and separate streams. Harness `SubAgents` runs an isolated agent and returns its result through a tool. Use it only when that narrower lifecycle preserves the observed contract; otherwise retain orchestration in the application.
- A filesystem or shell policy is not sandbox isolation. Preserve Eve's sandbox, credential-brokering, and network boundaries with a real container, VM, or supported cloud sandbox when untrusted execution is in scope.
- Eve channels, schedules, connections/OAuth, its default HTTP API, Workflow SDK persistence backends, Vercel Agent Runs, and deployment integrations are application or platform concerns. Replace and test them deliberately rather than attributing them to Core or Harness.
- Preserve provider routing, gateway allow-lists/fallbacks, provider options, retry semantics, telemetry correlation, and evaluation datasets only when the traced path uses them.

## Implement and prove one vertical slice

1. Preserve the supported caller boundary and replace only the agent-owned internals.
2. Start core-only. Add `pydantic_graph`, Harness, Pydantic Evals, or a durable runtime only after an observed contract requires it.
3. Test inputs, outputs, errors, message history, event order, tool arguments/results, step counts, state, and side effects at that boundary. Use deterministic models and fake application services offline; add a focused recording or live test only when provider behavior is the contract.
4. For approvals, reconnects, durable sessions, background work, or external effects, test interruption and fresh-process restart, stable correlation, authorization, replay, cancellation, and idempotency.
5. Remove `ai`, `@ai-sdk/*`, or `eve` only after no retained UI, transport, agent, or platform path needs them.

Explain any consequential semantic change before implementing it: state the source behavior, target behavior, caller impact, recommended choice, and remaining risk.

## Completion

Apply the completion criterion in [Verification and cutover](references/VERIFICATION-AND-CUTOVER.md). Label evidence from fakes, recordings, live providers, and real sandboxes accurately.
