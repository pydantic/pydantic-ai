# Verification and Cutover

Use this reference before editing a production agent or declaring a migration complete.

## Contents

- [Characterize the old system](#characterize-the-old-system)
- [Build the test pyramid](#build-the-test-pyramid)
- [Verify operational semantics](#verify-operational-semantics)
- [Cut over safely](#cut-over-safely)
- [Completion checklist](#completion-checklist)

## Characterize the old system

Capture behavior at stable boundaries:

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

Record at least one success trace and representative traces for invalid tool arguments, tool failure, provider failure, approval, resume, cancellation, and context pressure.

## Build the test pyramid

### Deterministic unit tests

- Test tools as ordinary functions/services, including authorization and idempotency.
- Use `TestModel` for schema/tool registration checks.
- Use `FunctionModel` when exact requests, tool calls, retry, or failure behavior matters.
- Assert `ModelRequestParameters` or captured run messages when tool schema and instructions are part of the contract.
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

Prove separately:

- conversation continuation;
- workflow-state restoration;
- approval correlation and resume;
- replay/fork behavior;
- pending-write or idempotency behavior;
- crash recovery during a model call, tool call, and external write.

Pydantic AI message history proves only conversation continuity. Use a durable execution integration or application workflow persistence when the old system promised graph state, pending writes, replay/fork, effect deduplication, or process-restart recovery.

### Concurrency and limits

Test fan-out caps, shared usage limits, cancellation, timeout, rate-limit backoff, partial child failure, and deterministic result aggregation. Verify that parent and child agents do not silently receive separate unlimited budgets.

### Security

Attempt cross-tenant access, path traversal, symlink escape, command bypass, SSRF, prompt injection into tool arguments, secret exposure, and approval bypass. Forge a deferred approval with a foreign, unknown, or already-consumed tool-call ID and reject it at an authenticated server-side correlation boundary. Enforce failures below the model layer.

### Streaming

Check event order, tool-call/result correlation IDs, partial text semantics, final-result emission, reconnect behavior, backpressure, and client cancellation. `run_stream` may treat the first valid final output as terminal; use `run_stream_events` or `iter` when all tool events must complete.

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

- [ ] Every migration-ledger row is native, intentionally external, or documented as remaining work.
- [ ] Public request, response, error, and event contracts pass.
- [ ] Tool schemas, authorization, side effects, retries, and approval pass.
- [ ] State, history, persistence, resume, and recovery promises pass.
- [ ] Streaming and cancellation pass with real clients.
- [ ] Unit, integration, and eval thresholds pass.
- [ ] Logfire traces correlate app request, agent run, model calls, tools, and subagents.
- [ ] No hidden LangChain callbacks, globals, messages, or `RunnableConfig` assumptions remain.
- [ ] Transitional bridges have been removed or have owners and removal dates.
- [ ] Dependency files and operational documentation match the new runtime.

Primary references: [Pydantic AI testing](https://pydantic.dev/docs/ai/guides/testing/), [Pydantic Evals](https://pydantic.dev/docs/ai/evals/evals/), [Logfire integration](https://pydantic.dev/docs/ai/integrations/logfire/), and [durable execution](https://pydantic.dev/docs/ai/integrations/durable_execution/overview/).
