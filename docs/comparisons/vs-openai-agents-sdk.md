# Pydantic AI vs OpenAI Agents SDK

The OpenAI Agents SDK is OpenAI's agent library: sessions, guardrails, handoffs, and new OpenAI
features first. Pydantic AI runs on any model, including OpenAI.

## Side by side

| | OpenAI Agents SDK | Pydantic AI |
|---|---|---|
| Language | Python | Python |
| Model providers | OpenAI first; others via LiteLLM | [Multiple](../models/overview.md) |
| Durable execution | Yes | [5+ integrations](../durable_execution/overview.md) |
| Observability | Their dashboard | [OpenTelemetry](../capabilities/instrumentation.md), including [Pydantic Logfire](../logfire.md) |
| Extensibility | Tools, guardrails, handoffs | [Capabilities and toolsets](../extensibility.md) |
| Interfaces | None built in | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) |
| Build your own harness | Yes | [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) |
| Coding harness | Yes | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| Graph library | No | [`pydantic-graph`](../graph.md) |
| Multi-agent | Handoffs | [Sub-agents](../multi-agent-applications.md), [hand-offs](../multi-agent-applications.md#programmatic-agent-hand-off), or [graph](../graph.md) |
| Realtime voice | Yes | [Realtime](../realtime/overview.md) |
| Image generation | Yes | [Image Generation](../image-generation.md) |
| License | MIT | MIT |

| | OpenAI Agents SDK | Pydantic AI |
|---|---|---|
| Structured output | Yes | [Structured output](../output.md#structured-output) |
| Guardrails | Yes | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Yes | [Execution environments](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser | Yes | [Web & research](https://pydantic.dev/docs/ai/harness/#web--research) |
| Memory | Yes | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes | [Compaction](../capabilities/compaction.md) |
| Evals | Yes | [Pydantic Evals](../evals.md) |
| Test without API keys | Yes | [`TestModel`, `FunctionModel`](../testing.md) |
