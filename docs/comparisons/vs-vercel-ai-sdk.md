# Pydantic AI vs Vercel AI SDK

The Vercel AI SDK adds model calls, tool loops and streaming chat to a TypeScript app, with `useChat` hooks and a broad provider registry. Pydantic AI vs Vercel AI SDK is mostly a language question: we speak the [Vercel AI stream protocol](../ui/vercel-ai.md), so their UI renders our Python agents.

## Framework

| | Vercel AI SDK | Pydantic AI and [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Language | TypeScript | Python |
| License | Apache-2.0 | MIT |
| Model providers | Many | [Many](../models/overview.md) |
| Extensibility | Middleware, tools | [Capabilities and toolsets](../extensibility.md) |
| Build a custom harness | Yes | [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) |
| Coding harness | Yes | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| Interfaces | React chat UI, stream protocol | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental) |
| Realtime voice | Yes (experimental) | [Realtime](../realtime/overview.md) |
| Image generation | Yes | [Image Generation](../image-generation.md) |
| Durable execution | Yes | [5+ integrations](../durable_execution/overview.md) |
| Observability | OpenTelemetry | [OpenTelemetry](../capabilities/instrumentation.md), including [Pydantic Logfire](../logfire.md) |
| Evals | No | [Pydantic Evals](../evals.md) |

## Features

| | Vercel AI SDK | Pydantic AI and [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Structured output | Yes | [Structured output](../output.md#structured-output) |
| Multi-agent | You compose it | [Delegation (tools or `SubAgents`), hand-off in your code, or graph](../multi-agent-applications.md) |
| Graph library | No (separate Workflow DevKit) | [`pydantic-graph`](../graph.md) |
| Memory | No | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Prunes reasoning and tool calls | [Compaction](../capabilities/compaction.md) |
| Guardrails | Yes | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Yes (experimental) | [Execution environments](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser use | No | [Web & research](https://pydantic.dev/docs/ai/harness/#web--research) |
