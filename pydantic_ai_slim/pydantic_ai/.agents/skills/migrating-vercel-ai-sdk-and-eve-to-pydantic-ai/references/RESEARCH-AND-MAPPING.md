# Research and concept mapping

Use this reference conditionally after tracing the source path. It is a decision guide, not a requirement to reproduce the entire Vercel AI SDK or Eve platform.

## Primary documentation

Inspect installed versions before migrating because AI SDK, Eve, Pydantic AI, and Harness continue to evolve.

- Vercel AI SDK: [agents](https://ai-sdk.dev/docs/agents/overview), [tools](https://ai-sdk.dev/docs/ai-sdk-core/tools-and-tool-calling), [structured output](https://ai-sdk.dev/docs/ai-sdk-core/generating-structured-data), [chatbot](https://ai-sdk.dev/docs/ai-sdk-ui/chatbot), [stream protocol](https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol), and [resume streams](https://ai-sdk.dev/docs/ai-sdk-ui/chatbot-resume-streams).
- Eve: [documentation](https://eve.dev/docs) and [repository](https://github.com/vercel/eve). Inspect the installed package and source because its durable protocol is evolving.
- Pydantic AI: [agents](https://pydantic.dev/docs/ai/core-concepts/agent/), [message history](https://pydantic.dev/docs/ai/core-concepts/message-history/), [deferred tools](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/), [MCP](https://pydantic.dev/docs/ai/mcp/client/), [durable execution](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/), and [Vercel AI integration](https://pydantic.dev/docs/ai/integrations/ui/vercel-ai/).
- Pydantic AI Harness: [overview](https://pydantic.dev/docs/ai/harness/), [skills](https://pydantic.dev/docs/ai/harness/skills/), [subagents](https://pydantic.dev/docs/ai/harness/subagents/), [memory](https://pydantic.dev/docs/ai/harness/memory/), [sandbox](https://pydantic.dev/docs/ai/harness/modal-sandbox/), and [step persistence](https://pydantic.dev/docs/ai/harness/step-persistence/).

AI SDK and Eve are TypeScript while Pydantic AI is Python. Capture retained transport and storage boundaries with language-neutral fixtures and keep their field names, using Pydantic aliases where needed.

## Gate on the installed AI SDK version

Read the matching [v5](https://ai-sdk.dev/docs/migration-guides/migration-guide-5-0), [v6](https://ai-sdk.dev/docs/migration-guides/migration-guide-6-0), or [v7](https://ai-sdk.dev/docs/migration-guides/migration-guide-7-0) migration guide before choosing fixtures. Important boundaries differ:

| Source generation | Check before porting |
|---|---|
| Before v5 | Legacy message shapes and line-prefixed stream frames may require an application adapter rather than the current UI path. |
| v5 | `UIMessage.parts`, `ModelMessage`, `stopWhen`, transport-based `useChat`, and start/delta/end stream parts changed the client contract. |
| v6 | `ToolLoopAgent`, async `convertToModelMessages`, current output APIs, and explicit approval states replaced earlier names or lifecycles. |
| v7 | `stepCountIs` became `isStepCount`, `needsApproval` moved to per-call `toolApproval`, `onFinish` became `onEnd`, and usage/tool-call results aggregate across steps; Node and ESM requirements also changed. |

Pydantic AI's `VercelAIAdapter` supports explicit SDK versions 5, 6, and 7. Its default is a compatibility choice, not evidence of the source version; configure it deliberately and test the emitted wire contract.

## AI SDK Core and UI ownership map

| Observed source behavior | Target owner and likely seam | Focused proof |
|---|---|---|
| `generateText()` / `streamText()` and text, steps, usage, or finish reason | **Core:** `Agent.run()` / `run_stream()` / `run_stream_events()`; **Application:** result adapter | Assert caller-visible result, exact required trajectory fields, step count, usage, errors, and effects. |
| `ToolLoopAgent`, `stopWhen`, `prepareStep` | **Core:** reusable `Agent`, `UsageLimits`, dynamic instructions and tool preparation; **Application/Graph:** explicit dynamic model or workflow policy | Exercise every observed stop condition and per-step mutation without assuming identical defaults. |
| `@ai-sdk/workflow` `WorkflowAgent` | **Application/durable runtime:** workflow state, persistence, approvals, and resume lifecycle; **Core durable integration:** agent model/tool operations inside a supported workflow | Interrupt and restart at model, tool, approval, and effect boundaries; assert replay, state, correlation, and idempotency. |
| Local `tool()` with Zod or JSON schema | **Core:** typed functions, `Tool`, `Tool.from_schema`, function toolsets | Assert advertised schema, call/result IDs, error recovery, timeout, cancellation, approval, and effects. |
| Async-generator tool with preliminary results | **Core/Application:** custom events plus final tool result when the client consumes intermediate states; otherwise an intentional change | Assert every consumed preliminary state, final output, order, cancellation, and behavior when the generator fails. |
| Provider-executed tool | **Core/provider:** matching provider tool when available; otherwise **Gap/Application** | Verify who executes it, provider payloads, credentials, result shape, and fallback behavior. |
| Client-side tool with no server executor | **Application/UI:** retain browser execution; **Core:** deferred/external tool call | Prove the server does not execute it and the validated result resumes the correct call. |
| Dynamic MCP tools | **Core:** `MCPToolset`; **Application:** auth and lifecycle | Assert discovery, schemas, name collisions, transport, credentials, errors, and version skew. |
| `Output.object`, `array`, `choice`, or `json` | **Core:** Pydantic output type and explicit output mode when transport matters | Assert complete validation, invalid-output behavior, partial-stream expectations, and any extra step after tools. |
| Tool approval request/response | **Core:** `requires_approval` or `ApprovalRequiredToolset`; resolve inline with `HandleDeferredToolCalls`, or later with `DeferredToolRequests` and `DeferredToolResults`; **Application:** policy, auth, persistence, audit, UI | Exercise approve, deny, invalid/replayed decisions, call correlation, and exactly-once effects. |
| `UIMessage`, `useChat`, and `ChatTransport` | **Application:** retained TS/React client; **Core UI adapter:** `VercelAIAdapter` | Golden-test request fields, validation, stream headers/chunks, tool states, errors, and final reconstruction for the installed SDK version. |
| Persisted UI messages and `ModelMessage` conversion | **Application:** UI-message store and trust policy; **Core:** normalized server-side history | Continue in a fresh process and prove client input cannot set trusted instructions, provider metadata, credentials, or ownership. |
| Resumable chat stream | **Application:** active-stream/message persistence and reconnect endpoint | Disconnect and reconnect; assert continuation, HTTP no-stream behavior, stable correlation, and no duplicate assistant message or effect. |
| `onStepFinish`, response `onFinish`, or v7 `onEnd` lifecycle callbacks | **Core/Application:** hooks or event stream handling; the `on_complete` parameter of `VercelAIAdapter.dispatch_request()` or `run_stream()` for completed-run persistence | Assert firing point, per-step versus terminal behavior, errors, persisted messages, and exactly-once writes. |
| Provider string, provider instance, Gateway routing or fallback | **Core/provider:** model/provider configuration; **Application:** allow-list and policy | Exercise each configured route, fallback order, provider options, and failure shape. |
| AI SDK subagent implemented as a tool | **Core:** agent-as-tool; **Harness:** `SubAgents` only for matching reusable delegation policy | Assert fresh child context, input projection, result projection, usage/cancellation/error propagation, and approval limitation. |
| OpenTelemetry and provider metadata | **Core:** instrumentation; optional Logfire; **Application:** trace and result adapter | Assert trace correlation, privacy policy, required spans, usage, and retained dashboards. |

## Eve ownership map

| Observed Eve behavior | Target owner and likely seam | Focused proof |
|---|---|---|
| `instructions.md`, `agent.ts`, and filesystem compilation | **Core:** explicit `Agent` construction; **Application:** configuration loader only if runtime discovery is required | Construct every configured agent and assert instruction, model, tool, and environment selection. |
| `tools/` with `defineTool` | **Core:** typed tools/toolsets; **Application:** dependencies and protected services | Assert schema, executor result, error/retry, approval, cancellation, and effects. |
| `defineWorkflowTool` and durable waits | **Application/durable runtime:** workflow and wait lifecycle; **Core durable integration:** model/tool operations within a supported workflow | Kill and restart around the wait; assert correlation, validated resume data, replay, and idempotency. |
| `skills/` with descriptions, `load_skill`, and materialized files | **Harness:** `Skills` for `SKILL.md`; **Harness/Application:** `FileSystem` or explicit tools for other files | Assert discovery, request-scoped loading, instruction injection, and access to every required reference, asset, or script. |
| Built-in or declared `subagents/` | **Harness:** `SubAgents` for synchronous isolated delegation; **Application/Gap:** durable children, background results, and child streams | Assert child history/state isolation, dependency/tool sharing, handback, cancellation, errors, events, and restart. |
| `defineState` durable session state | **Application:** separately typed and persisted state | Assert mutation, schema/version handling, fresh-process continuation, concurrency, and non-sharing with children. |
| Long-lived memory | **Harness:** `Memory` only for matching model-owned notebook semantics; otherwise **Application:** retained memory provider | Assert lifetime, namespace, locking, attribution, bounded injection, and sharing rules independently from messages and state. |
| Session → turn → step durability and step replay | **Application/durable runtime:** session driver, replay unit, event log; optional **Harness:** `StepPersistence` only for its narrower documented contract | Interrupt model, tool, and effect boundaries; assert which step re-runs, event identity, ordering, and exactly-once external effects. |
| Eve HTTP session API, NDJSON stream, event cursor | **Application/Gap:** compatibility endpoint and durable event store | Golden-test status codes, payloads, framing, cursor semantics, reconnect, retry, cancel, and terminal state. `VercelAIAdapter` does not implement Eve's protocol. |
| `connections/`, MCP/OpenAPI discovery, app/user OAuth | **Core:** MCP/toolsets where compatible; **Application:** connection registry, OAuth, token persistence, tenant policy | Exercise discovery, qualification, auth parking/resume, refresh, revocation, isolation, and errors. |
| `channels/` and `schedules/` | **Application/platform:** channel adapters, scheduler, queues, and deployment | Exercise trigger identity, authorization, deduplication, delivery, retry, cancellation, timezone, and failure behavior. |
| Sandbox adapters and `/workspace` boundary | **Harness:** a supported real sandbox when it matches; **Application:** existing sandbox and credential broker | Run containment, network, secret, filesystem, resource, timeout, and cleanup tests in the actual isolation boundary. |
| Evals and agent-run observability | **Evals:** `pydantic_evals`; **Core:** instrumentation; **Application:** HTTP eval target, sampling, storage, and dashboards | Re-run the same dataset and essential metrics; assert trace/run correlation and retained privacy policy. |
| Workflow SDK backends, Vercel Sandbox, Cron, Connect, Blob, Gateway, Agent Runs, and deployment | **Application/platform** | Verify each retained or replacement service at its public boundary; do not claim Core or Harness equivalence. |

## Do not collapse distinct state

Keep these stores separate unless the source contract proves they are interchangeable:

- UI messages restored by the frontend;
- provider-facing normalized message history;
- Eve per-session structured state;
- long-lived memory across sessions;
- durable step/session records and event cursors;
- model-owned plans or Harness memory notebooks;
- application identity, authorization, and tenant configuration.

The same fact appearing in two prompts does not prove equivalent ownership, lifetime, consistency, or restart behavior.
