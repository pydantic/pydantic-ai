# Pydantic AI vs Mastra

Mastra is a TypeScript agent framework that comes with batteries: step workflows, memory, processors, evals, a local playground and a hosted Studio. For a team that wants one opinionated stack end to end, it covers a lot of ground. Pydantic AI vs Mastra is a language choice first: we bring the same range to Python, composed from a plain [`Agent`][pydantic_ai.Agent].

| | Mastra | Pydantic AI |
|---|---|---|
| Language | TypeScript | Python |
| Model providers | Multiple | [Multiple](../models/overview.md) |
| Durable execution | Yes | [5+ integrations](../durable_execution/overview.md) |
| Observability | OpenTelemetry | [OpenTelemetry](../capabilities/instrumentation.md), including [Pydantic Logfire](../logfire.md) |
| Interfaces | Playground, Studio | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental) |
| Extensibility | Tools, processors, scorers, workflows | [Capabilities and toolsets](../extensibility.md) |
| Realtime voice | Yes | [Realtime](../realtime/overview.md) |
| Image generation | No | [Image Generation](../image-generation.md) |
| Multi-agent | Workflows, sub-agents | [Delegation (tools or `SubAgents`), hand-off in your code, or graph](../multi-agent-applications.md) |
| Graph library | No (step workflows) | [`pydantic-graph`](../graph.md) |
| Build your own harness | Yes | [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) |
| Coding harness | Yes | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| License | Apache-2.0 (core); EE for some features | MIT |

| | Mastra | Pydantic AI |
|---|---|---|
| Structured output | Yes | [Structured output](../output.md#structured-output) |
| Memory | Yes | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Token limit: truncate or abort | [Compaction](../capabilities/compaction.md) |
| Guardrails | Yes | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Yes | [Execution environments](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser | Yes | [Web & research](https://pydantic.dev/docs/ai/harness/#web--research) |
| Evals | Yes | [Pydantic Evals](../evals.md) |
