# Logfire-Assisted Migration Verification

Use Logfire to expose what the migrated Pydantic AI application actually did. Treat it as diagnostic and corroborating evidence, not as proof that two frameworks have identical semantics.

## Contents

- [Choose with the user](#choose-with-the-user)
- [Instrument deliberately](#instrument-deliberately)
- [Compare source and target runs](#compare-source-and-target-runs)
- [Observe streaming at both boundaries](#observe-streaming-at-both-boundaries)
- [Protect sensitive content](#protect-sensitive-content)
- [Test without exporting](#test-without-exporting)
- [Know what traces cannot prove](#know-what-traces-cannot-prove)

## Choose with the user

When the source already has observability, present the decision before changing it. Recommend Logfire as the first-party Pydantic AI experience, but do not treat that recommendation as authorization to replace an established system.

| Choice | When it fits | Cost or residual |
|---|---|---|
| retain the existing system | continuity and a narrow agent-runtime migration matter most | prove that nested Pydantic AI model/tool lifecycles remain visible; an outer framework span is insufficient |
| add Logfire temporarily alongside it | sanitized shadow comparison or migration debugging justifies dual telemetry | duplicate export, privacy, sampling, cost, and trace-correlation concerns; set a removal condition |
| switch to Logfire | the user wants native Pydantic AI instrumentation and accepts an observability migration | migrate or retire dashboards, alerts, evaluations, trace ingestion, retention/privacy controls, credentials, and operational documentation |

Explain which option you recommend, why it fits the application, what will change for operators, and what remains unverified. If observability was not placed in scope, retain it and offer the recommendation rather than expanding the migration.

## Instrument deliberately

Configure Logfire during application startup, before constructing or running agents, and install Pydantic AI instrumentation once:

```python
import logfire

logfire.configure()
logfire.instrument_pydantic_ai(
    include_content=False,
    include_binary_content=False,
)
```

Name reusable agents so their runs are distinguishable. Instrument a specific agent instead of all agents when the application has a narrower trace or privacy boundary. Inspect the installed `logfire` and `pydantic_ai` versions before passing instrumentation settings: telemetry data-format versions evolve independently of the Pydantic AI package, and their event roles and attributes can differ.

The default instrumentation can expose agent runs, model requests, tool calls, retries, errors, token usage, and timing. Add small application spans around boundaries that Pydantic AI does not own, such as retrieval, persistence, queues, approval records, public streaming, and external writes. Reuse the application's request, thread, tenant, and idempotency correlation identifiers as safe attributes; do not put secrets in them.

Inventory the source model/tool callback lifecycle and application metrics before replacing it. Keeping a LangSmith, Langfuse, or other callback on an outer LangGraph node does not automatically instrument a nested Pydantic AI call. Preserve the existing model/tool observations through a tested adapter or make an explicit observability change; verify success, token, completion, and error paths plus parent correlation.

Do not add one span per token. Prefer a bounded set of spans and metrics: first event, terminal event, event counts, queue delay, cancellation, and failure.

## Compare source and target runs

During a shadow or replay comparison:

1. Run the same sanitized input, dependencies, model settings, and deterministic fixtures through the source and target behind the same application boundary.
2. Send both traces to the same OpenTelemetry backend when practical. LangChain and LangGraph can export LangSmith OpenTelemetry traces to Logfire; set `LANGSMITH_OTEL_ENABLED=true` and `LANGSMITH_TRACING=true` before importing those frameworks. Set `LANGSMITH_OTEL_ONLY=true` only when intentionally sending exclusively through OpenTelemetry rather than also retaining LangSmith export. Keep source and target trace namespaces distinct.
3. Normalize comparable facts instead of diffing raw spans. Compare model-request count, tool calls and their causal relationships, safe arguments and results, retries, errors, usage, latency, model-visible messages when allowed, and application boundary events. Use application sequence IDs or executable assertions—not raw span arrival order—when tool order is a contract.
4. Investigate every unexplained difference. A trace that merely looks similar is not equivalence evidence.
5. Link a validated claim to the executable test that establishes it and use the trace to explain the trajectory.

Do not dual-run side-effectful agents unless tools are dry-run, sandboxed, or protected by durable idempotency keys. Observing a single write does not prove exactly-once behavior.

## Observe streaming at both boundaries

Keep these measurements separate:

- **Instrumented model-call boundary:** Pydantic AI can record the time from issuing the streaming request until the wrapped response surfaces its first chunk. This includes transport and client-SDK behavior; it is not provider-internal timing.
- **Application boundary:** measure when the API, SSE, or WebSocket consumer receives its first public event and its terminal event.

Model time to first chunk does not prove client time to first event. A service can buffer an entire agent run after the model begins streaming. Test the public client contract for:

- exact event schema and order;
- incremental delivery rather than post-run buffering;
- tool, retry, partial-output, final-result, and terminal-error events that the source exposes;
- correlation identifiers and usage placement;
- disconnect and cancellation propagation;
- backpressure and bounded buffering;
- reconnect or resume behavior, when promised;
- cleanup and late producer errors after an early consumer exit.

Choose the Pydantic AI streaming API from the required lifecycle. `run_stream()` may stop at the first matching final output. Use `run(event_stream_handler=...)`, `run_stream_events()`, or `iter()` when the contract requires complete tool execution or lower-level event control. When handling raw events, assemble both initial part events and later deltas. A Logfire trace can reveal the resulting trajectory, but only a real client test proves delivery behavior.

## Protect sensitive content

Pydantic AI instrumentation includes prompts, completions, tool arguments, and tool results by default. That data may contain personal, proprietary, credential, or tenant information.

- Prefer `InstrumentationSettings(include_content=False)` when raw content is unnecessary or unsafe. Structural telemetry remains useful.
- Do not assume generic scrubbing makes LLM content safe. Logfire deliberately does not apply generic regex scrubbing to free-form LLM message attributes because it would be both noisy and incomplete. Define organization-specific controls and retention rules before enabling content capture.
- Avoid full HTTPX header/body capture as a default. It is a temporary diagnostic escalation that can expose provider authorization and raw payloads; Bedrock and other non-HTTPX transports need separate treatment.
- Sanitize shadow inputs and comparison attributes. Keep authenticated identity and secrets out of model-visible content and trace attributes.

State the privacy choice and its consequence: with content disabled, traces cannot substantiate prompt, argument, or output equivalence.

## Test without exporting

Use Logfire's in-memory test exporter or Pytest `capfire` fixture to assert a small stable set of spans and attributes without sending telemetry to a remote project. Test that important application spans correlate with the agent run and that failures/cancellation are observable. Avoid full raw-trace snapshots as the only assertion; telemetry schemas change and raw traces include incidental data.

For short-lived commands, workers, or tests, flush telemetry at shutdown rather than on every request. Sampling can omit spans. SDK tail sampling is process-local and can fragment a distributed trace; use collector-side tail sampling when the entire distributed trace must share a decision. Never use sampled production traces as exhaustive parity evidence.

Instrument every service participating in the tested path and verify trace-context propagation across HTTP, queues, workers, and detached tasks. An application-created background task can outlive its request span or lose its parent; retain a safe business correlation ID and do not treat a broken trace tree as proof that work did not happen. Likewise, cancellation can leave incomplete telemetry, so resource and side-effect assertions remain authoritative.

## Know what traces cannot prove

Logfire can help demonstrate that an exercised run used the expected model, tools, retry path, observed token usage, and timing. It cannot by itself prove:

- deterministic output equivalence from a nondeterministic model;
- public streaming delivery, backpressure, reconnect, or cancellation;
- checkpoint, fork, replay, or process-restart semantics;
- authorization or tenant isolation;
- exactly-once external side effects;
- behavior that was not instrumented or was removed by sampling.

Use deterministic characterization tests, provider integration tests, real clients, persistence/restart probes, database and side-effect assertions, and security tests for those claims. Report missing telemetry as missing evidence, not a passing result.

Primary references: [Pydantic AI with Logfire](https://pydantic.dev/docs/logfire/integrations/llms/pydanticai/), [Pydantic AI instrumentation settings](https://pydantic.dev/docs/ai/api/models/instrumented/), [LangChain and LangGraph with Logfire](https://pydantic.dev/docs/logfire/integrations/llms/langchain/), [Pydantic AI streaming](https://pydantic.dev/docs/ai/core-concepts/agent/#streaming-events-and-final-output), [Logfire scrubbing](https://pydantic.dev/docs/logfire/instrument/scrubbing/), [Logfire sampling](https://pydantic.dev/docs/logfire/instrument/sampling/), and [Logfire testing](https://pydantic.dev/docs/logfire/reference/testing/).
