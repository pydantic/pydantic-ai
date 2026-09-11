# Pydantic AI vs Vercel AI SDK

The Vercel AI SDK adds model calls, tool loops and streaming chat to a TypeScript app, with `useChat` hooks and a broad provider registry; [Eve](https://eve.dev), a separate Apache-2.0 package, adds an agent harness on top. Pydantic AI is the Python back end for that UI: a typed [`Agent`][pydantic_ai.Agent], the [Harness SDK](https://pydantic.dev/docs/ai/harness/), and a [Vercel AI stream adapter](../ui/vercel-ai.md) so `useChat` renders our agents.

## Framework

| | Vercel AI SDK | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Language | TypeScript | Python |
| License | Apache-2.0 | MIT |
| Model providers | Many | [Many](../models/overview.md) |
| Extensibility | Middleware, tools | [Capabilities and toolsets](../extensibility.md); [50+ with the Harness SDK](https://pydantic.dev/docs/ai/harness/) |
| Harnesses | [Eve](https://eve.dev), a separate package; adapters drive external harnesses | Built-in [`Coder`](https://pydantic.dev/docs/ai/harness/coder/) and [`Researcher`](https://pydantic.dev/docs/ai/harness/researcher/), or compose your own |
| Observability | OpenTelemetry | [OpenTelemetry](../logfire.md#using-opentelemetry), including [Pydantic Logfire](https://pydantic.dev/logfire) |
| Interfaces | React chat UI, stream protocol | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental) |
| Realtime voice | Yes (experimental) | [Realtime](../realtime/overview.md) |
| Evals | No | [Pydantic Evals](../evals.md) |
| Durable execution | Yes | [5+ integrations](../durable_execution/overview.md) |
| Image generation | Yes | [Image Generation](../image-generation.md) |

## Features

| | Vercel AI SDK | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Sub-agents | Yes (Eve) | [Subagents](https://pydantic.dev/docs/ai/harness/subagents/), [delegation](../multi-agent-applications.md), or [`pydantic-graph`](../graph.md) |
| Planning | Yes (Eve) | [Planning](https://pydantic.dev/docs/ai/harness/planning/) |
| Skills | Yes (Eve); provider-hosted in the SDK | [Skills](https://pydantic.dev/docs/ai/harness/skills/) |
| Memory | No | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes (Eve); `pruneMessages` in the SDK | [Compaction](../capabilities/compaction.md) |
| Guardrails | Yes | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Yes (experimental) | [Execution environments](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser use | No | [Web & research](https://pydantic.dev/docs/ai/harness/#web--research) |
