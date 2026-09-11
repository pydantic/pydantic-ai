# Pydantic AI vs Vercel AI SDK

The Vercel AI SDK is the wire to a React UI: streaming, tool cards, approval in the browser. Pydantic
AI is a Python agent. [`VercelAIAdapter`][pydantic_ai.ui.vercel_ai.VercelAIAdapter] speaks their
protocol, so the browser can stay theirs.

## Side by side

| | Vercel AI SDK | Pydantic AI |
|---|---|---|
| Language | TypeScript | Python |
| Model providers | Multiple | [Multiple](../models/overview.md) |
| Durable execution | Yes | [5+ integrations](../durable_execution/overview.md) |
| Observability | OpenTelemetry | [OpenTelemetry](../capabilities/instrumentation.md), including [Pydantic Logfire](../logfire.md) |
| Extensibility | Middleware, tools | [Capabilities and toolsets](../extensibility.md) |
| Interfaces | React chat UI, stream protocol | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental), A2A |
| Build your own harness | Yes | [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) |
| Coding harness | Yes | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| Graph library | Yes | [`pydantic-graph`](../graph.md) |
| Multi-agent | You compose it | [Sub-agents](../multi-agent-applications.md), [hand-offs](../multi-agent-applications.md#programmatic-agent-hand-off), or [graph](../graph.md) |
| Realtime voice | Yes | [Realtime](../realtime/overview.md) |
| Image generation | Yes | [Image Generation](../image-generation.md) |
| License | Apache-2.0 | MIT |

| | Vercel AI SDK | Pydantic AI |
|---|---|---|
| Structured output | Yes | [Structured output](../output.md#structured-output) |
| Guardrails | Yes | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Yes | [Execution environments](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser | No | [Web & research](https://pydantic.dev/docs/ai/harness/#web--research) |
| Memory | No | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes | [Compaction](../capabilities/compaction.md) |
| Evals | No | [Pydantic Evals](../evals.md) |
| Test without API keys | Yes | [`TestModel`, `FunctionModel`](../testing.md) |
