# Verification and Cutover

Use this reference before changing a high-risk boundary or declaring a production migration complete.

## Characterize the source

Capture the applicable behavior at stable boundaries:

- accepted request/context and final output/error schema;
- tool definitions, visibility, validation, side effects, retries, concurrency, and terminal-tool behavior;
- handoff destinations, input filtering, active-agent changes, next-turn ownership, and guardrail scope;
- manual history, session merge/storage, provider-side state, interruption, approval, and resume behavior;
- raw, run-item, lifecycle, and public streaming events;
- model settings, turn limits, usage, cancellation, and timeout behavior;
- authentication, tenant, filesystem, process, network, secret, and sandbox boundaries;
- traces, metrics, eval dimensions, queues, deployment, and service integrations.

Record a success trace and representative failure traces for the risks the selected path actually has. Do not invent requirements for unused SDK features.

## Classify every observed contract

| Status | Evidence |
|---|---|
| `verified-equivalent` | source and target preserve the same externally observable contract in executable checks |
| `verified-adapter` | internal semantics differ, but a focused adapter preserves the public contract |
| `intentional-change` | the difference and impact were explained and explicitly accepted |
| `external-owner` | a named application or infrastructure component preserves it at a tested boundary |
| `not-applicable` | the active source path does not provide or consume it |
| `unverified` | evidence is incomplete; do not call it equivalent |
| `blocked` | a required contract has no acceptable proved construction; do not cut over |

Matching class names, successful imports, a happy-path demo, and similar traces are insufficient.

## Proportionate tests

For an ordinary stateless agent, focused boundary tests and a residual-risk note are enough. Use `TestModel` for tool/schema registration and `FunctionModel` when exact messages, tool calls, retries, or event order matter. Test tools as normal functions/services, including authorization and idempotency.

Add targeted integration tests when the slice uses:

- a real provider family or OpenAI native tool;
- local or hosted MCP;
- databases, sessions, vector stores, sandboxes, filesystem/shell tools, queues, or webhooks;
- public SSE, WebSocket, AG-UI, or another streaming adapter;
- persistence or approval that must survive process restart.

For multi-agent paths, prove final-answer ownership, history passed to each agent, typed routing metadata, parent/child usage and limits, partial failure, cancellation, and event correlation. Do not silently give nested agents independent unlimited budgets.

For guardrails, exercise allow, block, replacement/retry where applicable, exceptions, and every agent/tool boundary that the source actually covered. A security policy must fail below the model layer even if a guardrail, tool filter, prompt, or approval record is bypassed.

For session and HITL paths, separately prove:

- next-turn conversation continuity;
- atomic append/merge behavior under concurrent writers when promised;
- pending approval correlation and authenticated resume;
- denial executes no protected effect;
- approval executes the effect once with original trusted dependencies;
- stale, foreign, unknown, and already-consumed call IDs fail closed;
- crash recovery, replay, or fork semantics only when the source promised them.

For streaming, test the real consumer boundary for incremental delivery, event schema and order, correlation IDs, final completion, cancellation/disconnect, backpressure, early exit cleanup, and terminal errors. Model-level streaming does not prove the application is not buffering.

Port representative datasets to Pydantic Evals only when the source has evals or task quality is a release criterion. Compare task success, structured validity, tool trajectory constraints, latency/time to first event, requests/tokens/cost, retries, escalation, and unsafe attempts. Do not require identical prose unless wording is a public contract.

## Cut over

1. Place the migrated implementation behind the existing framework-neutral boundary and route complete runs with a stable flag.
2. Shadow only read-only or sandboxed traffic. Redact sensitive comparison data.
3. Compare normalized outputs, trajectories, limits, side effects, and traces against predefined thresholds.
4. Canary writes only with server-side authorization, idempotency keys, and rollback controls.
5. Increase traffic after quality, latency, cost, safety, persistence, and operational thresholds pass.
6. Remove adapters and the `openai-agents` dependency only after repository search, dependency/entrypoint inspection, original tests, and production rollback criteria show no retained path needs them.

Avoid dual-running side-effectful agents unless every effect is dry-run, sandboxed, or deduplicated.

## Completion checklist

- [ ] Every observed contract has an evidence-backed status.
- [ ] Public request, output, error, and event contracts pass.
- [ ] Tool schemas, permissions, side effects, retries, and terminal behavior pass.
- [ ] Handoff or delegation ownership and history behavior pass.
- [ ] Session, continuation, approval, and recovery promises pass.
- [ ] Security boundaries and real execution environments pass.
- [ ] Focused unit, integration, type, and eval checks appropriate to the slice pass.
- [ ] Observability and privacy choices are deliberate and traces are not treated as sole proof.
- [ ] Transitional adapters have owners and removal criteria.
- [ ] Dependency files and operational documentation match the new runtime.

Primary references: [Pydantic AI testing](https://pydantic.dev/docs/ai/guides/testing/), [message history](https://pydantic.dev/docs/ai/core-concepts/message-history/), [deferred tools](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/), [durable execution](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/), [Pydantic Evals](https://pydantic.dev/docs/ai/evals/evals/), and [Logfire](https://pydantic.dev/docs/ai/integrations/logfire/).
