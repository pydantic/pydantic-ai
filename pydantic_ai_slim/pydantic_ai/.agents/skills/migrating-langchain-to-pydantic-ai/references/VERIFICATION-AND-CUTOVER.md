# Verification and Cutover

Use this reference before editing a production agent or declaring a migration complete.

## Contents

- [Characterize the old system](#characterize-the-old-system)
- [Classify every claim](#classify-every-claim)
- [Build the test pyramid](#build-the-test-pyramid)
- [Verify operational semantics](#verify-operational-semantics)
- [Cut over safely](#cut-over-safely)
- [Completion checklist](#completion-checklist)

## Characterize the old system

Capture the applicable behavior at stable boundaries; do not invent requirements for features the source path does not use:

- accepted request and context schema;
- final output and error schema;
- tool names, descriptions, argument schemas, visibility, and side effects;
- dynamic prompt/tool/model selection;
- middleware/hook order;
- state fields and reducer behavior;
- message, checkpoint, store, thread, interrupt, and replay semantics;
- stream modes and client-visible events;
- usage, iteration, concurrency, timeout, and retry limits;
- auth, tenant, filesystem, shell, network, and secret boundaries;
- traces, metrics, and eval dimensions;
- deployment, queue, scheduler, and webhook contracts.

Record at least one success trace and representative failure traces for the risks the selected path actually has.

## Classify every claim

Apply a status to each observed contract, not to the migration as a whole:

| Status | Required evidence |
|---|---|
| `verified-equivalent` | Source and target preserve the same externally observable contract in executable checks against the target project's versions and boundaries. |
| `verified-adapter` | Internal semantics differ, but an adapter preserves the public contract in executable checks. |
| `intentional-change` | The difference and impact were explained and explicitly accepted by the user or owner. |
| `external-owner` | A named application or infrastructure component preserves the contract, with evidence at that boundary. |
| `not-applicable` | The observed source path does not provide or consume this behavior. |
| `unverified` | The behavior was not exercised, the evidence is incomplete, or only a candidate design exists. Do not call it equivalent. |
| `blocked` | A required contract has no acceptable proved construction. Do not cut over that slice. |

Trace similarity, successful imports, matching class names, a happy-path demo, and an author-side spike are not sufficient for `verified-equivalent`. Link each verified status to its test or boundary assertion and report every `unverified`, `intentional-change`, and `blocked` item to the user.

## Build the test pyramid

### Deterministic unit tests

- Test tools as ordinary functions/services, including authorization and idempotency.
- Use `TestModel` for schema/tool registration checks.
- Use `FunctionModel` when exact requests, tool calls, retry, or failure behavior matters.
- Assert `ModelRequestParameters` or captured run messages when tool schema and instructions are part of the contract.
- Run the repository's static checker across the migrated slice. When output inference loses a real union, parameterize `Agent[DepsT, OutputT]` explicitly rather than weakening the owned boundary to `Any`.
- Round-trip persisted model messages and typed workflow records through their production serializer, and prove malformed records fail validation.
- For dynamic instructions, test at least two dependency values and prove the emitted instructions change while secrets and authenticated identity remain absent from model-visible content.
- Test both approval decisions: denial must not execute the protected tool; an approved resume must execute it exactly once with the original authenticated dependencies and correlation ID.
- Test graph/application transitions without a live model.

### Integration tests

- Exercise each real provider family used in production with a minimal recorded or low-cost case.
- Test MCP/server lifecycle and transport failure.
- Test database, vector store, sandbox, filesystem, shell, queue, and webhook adapters at their real boundary.
- Test streaming consumers against the application-owned event schema.
- Test persistence across a real process restart when restart survival is promised.

### Evals

Port representative LangSmith or custom datasets to `pydantic_evals` without changing prompts or graders at the same time. Compare:

- task success and output validity;
- tool choice and trajectory constraints;
- citation/evidence quality;
- latency and time to first event;
- model requests, tokens, and cost;
- retry, failure, and escalation rates;
- unsafe or unauthorized attempts.

Do not require identical prose unless wording is a public contract. Require identical structured fields, safety boundaries, and side effects where they are contracts.

## Verify operational semantics

### Persistence and recovery

When the source promises them, prove separately:

- conversation continuation;
- workflow-state restoration;
- approval correlation and resume;
- replay/fork behavior;
- pending-write or idempotency behavior;
- crash recovery during a model call, tool call, and external write.

Pydantic AI message history proves only conversation continuity. Use a durable execution integration or application workflow persistence when the old system promised graph state, pending writes, replay/fork, effect deduplication, or process-restart recovery.

### Concurrency and limits

When the slice fans out, accepts concurrent requests for one thread, or shares limits, test caps, cancellation, timeout, rate-limit backoff, partial child failure, deterministic result aggregation, and duplicate work before persistence conflicts. Verify that parent and child agents do not silently receive separate unlimited budgets.

### Security

Test the security boundaries the slice exposes. Examples include cross-tenant access, path traversal for filesystem tools, SSRF for URL-fetching tools, secret exposure, and approval bypass. For deferred approval, forge a foreign, unknown, or already-consumed tool-call ID and reject it at an authenticated server-side correlation boundary. Enforce failures below the model layer.

### Streaming

Check the event behavior promised by the public stream: event order, relevant correlation IDs, partial text, final-result emission, and any documented reconnect, backpressure, or cancellation behavior. Prove incremental delivery with a real client; model-level time to first chunk does not detect application buffering. If synchronous request construction, retrieval, or tools can block the event loop, preserve any source worker/thread boundary or use a native async path. When responsiveness before the first chunk is an observed contract, test that unrelated event-loop work still advances; do not move thread-affine clients across threads blindly. `run_stream` may treat the first valid final output as terminal; use `run(event_stream_handler=...)`, `run_stream_events`, or `iter` when all tool events must complete. Test early consumer exit, cleanup, and late producer errors.

### Observability

If the source uses another observability system, explain the retain, temporary dual-export, and switch-to-Logfire choices before editing it. Recommend Logfire for its first-party Pydantic AI integration, but preserve the existing system unless the user accepts the wider operational migration. A switch includes dashboards, alerts, evaluations, trace ingestion, privacy/retention controls, credentials, and documentation—not only the runtime SDK. When the user chooses Logfire, configure it at application startup and make content capture an explicit privacy decision. An installed SDK or outer request/graph span does not prove that a nested Pydantic AI model call, tokens, errors, usage, or tool calls remain observable. Probe the active success and failure lifecycle at the model boundary, including parent correlation and existing application metrics. Compare normalized source and target facts rather than raw telemetry schemas. Sampling, missing instrumentation, and disabled content limit what a trace can substantiate. Follow [Logfire-Assisted Migration Verification](LOGFIRE-VERIFICATION.md); traces complement rather than replace contract tests.

## Cut over safely

1. Add a framework-neutral adapter and route whole runs by a stable flag.
2. Shadow read-only traffic first. Redact sensitive data in comparison logs.
3. Compare outputs, trajectories, limits, and traces automatically.
4. Canary low-risk write traffic with idempotency keys and rollback controls.
5. Increase traffic only after predefined quality, latency, cost, and safety thresholds hold.
6. Stop new LangChain feature work in the migrated slice.
7. Remove `tool_from_langchain`, `LangChainToolset`, message converters, and dual observability after the rollback window.
8. Remove LangChain/LangGraph dependencies only after the strict inventory has no errors, a repository-wide text search is clean or explained, dependency and entrypoint graphs show no runtime use, notebooks/config/plugin registries have been checked, and the original runtime tests still pass.

Avoid dual-running side-effectful agents unless tools are in dry-run mode or every external write is deduplicated.

## Completion checklist

- [ ] Every observed contract has an evidence-backed status; a low-risk slice may use a concise residual-risk note instead of a ledger.
- [ ] Public request, response, error, and event contracts pass.
- [ ] Tool schemas, authorization, side effects, retries, and approval pass.
- [ ] State, history, persistence, resume, and recovery promises pass.
- [ ] Streaming and cancellation pass with real clients.
- [ ] Unit, integration, and eval thresholds pass.
- [ ] When Logfire is enabled, its privacy settings are deliberate and traces correlate the applicable app, agent, model, tool, and subagent boundaries without being treated as sole parity proof.
- [ ] No hidden LangChain callbacks, globals, messages, or `RunnableConfig` assumptions remain.
- [ ] Transitional bridges have been removed or have owners and removal dates.
- [ ] Dependency files and operational documentation match the new runtime.

Primary references: [Pydantic AI testing](https://pydantic.dev/docs/ai/guides/testing/), [Pydantic Evals](https://pydantic.dev/docs/ai/evals/evals/), [Logfire integration](https://pydantic.dev/docs/ai/integrations/logfire/), and [durable execution](https://pydantic.dev/docs/ai/integrations/durable_execution/overview/).
