# Pydantic AI vs OpenAI Agents SDK

| | OpenAI Agents SDK | Pydantic AI |
|---|---|---|
| Language | Python | Python |
| Model providers | OpenAI first; others via LiteLLM | [Multiple](../models/overview.md) |
| Durable execution | Yes | [5+ integrations](../durable_execution/overview.md) |
| Observability | OpenAI tracing, OTel via adapters | [OpenTelemetry](../capabilities/instrumentation.md), including [Pydantic Logfire](../logfire.md) |
| Interfaces | None built in | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental) |
| Extensibility | Tools, guardrails, handoffs | [Capabilities and toolsets](../extensibility.md) |
| Realtime voice | Yes | [Realtime](../realtime/overview.md) |
| Image generation | Via provider tools | [Image Generation](../image-generation.md) |
| Multi-agent | Handoffs | [Delegation (tools or `SubAgents`), hand-off in your code, or graph](../multi-agent-applications.md) |
| Graph library | No | [`pydantic-graph`](../graph.md) |
| Build your own harness | Yes | [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) |
| Coding harness | Yes | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| License | MIT | MIT |

| | OpenAI Agents SDK | Pydantic AI |
|---|---|---|
| Structured output | Yes | [Structured output](../output.md#structured-output) |
| Memory | Yes | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes | [Compaction](../capabilities/compaction.md) |
| Guardrails | Yes | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Yes | [Execution environments](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser | Bring your own `Computer` | [Web & research](https://pydantic.dev/docs/ai/harness/#web--research) |
| Evals | Yes | [Pydantic Evals](../evals.md) |
