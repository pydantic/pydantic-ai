# Pydantic AI vs Vercel AI SDK

The Vercel AI SDK is the default way to add model calls, tool loops and streaming chat to a TypeScript app, with `useChat` hooks and a broad provider registry. If your product is the frontend, start there. Pydantic AI vs Vercel AI SDK is mostly a language question: we speak the [Vercel AI stream protocol](../ui/vercel-ai.md), so their UI renders our Python agents.

| | Vercel AI SDK | Pydantic AI |
|---|---|---|
| Language | TypeScript | Python |
| Model providers | Multiple | [Multiple](../models/overview.md) |
| Durable execution | Yes | [5+ integrations](../durable_execution/overview.md) |
| Observability | OpenTelemetry | [OpenTelemetry](../capabilities/instrumentation.md), including [Pydantic Logfire](../logfire.md) |
| Interfaces | React chat UI, stream protocol | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental) |
| Extensibility | Middleware, tools | [Capabilities and toolsets](../extensibility.md) |
| Realtime voice | Yes (experimental) | [Realtime](../realtime/overview.md) |
| Image generation | Yes | [Image Generation](../image-generation.md) |
| Multi-agent | You compose it | [Delegation (tools or `SubAgents`), hand-off in your code, or graph](../multi-agent-applications.md) |
| Graph library | No (separate Workflow DevKit) | [`pydantic-graph`](../graph.md) |
| Build your own harness | Yes | [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) |
| Coding harness | Yes | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| License | Apache-2.0 | MIT |

| | Vercel AI SDK | Pydantic AI |
|---|---|---|
| Structured output | Yes | [Structured output](../output.md#structured-output) |
| Memory | No | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Prunes reasoning and tool calls | [Compaction](../capabilities/compaction.md) |
| Guardrails | Yes | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Yes (experimental) | [Execution environments](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser | No | [Web & research](https://pydantic.dev/docs/ai/harness/#web--research) |
| Evals | No | [Pydantic Evals](../evals.md) |
