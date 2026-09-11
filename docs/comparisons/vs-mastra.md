# Pydantic AI vs Mastra

Mastra is a TypeScript agent framework with step workflows, memory, processors, evals, a local playground and a hosted Studio. Pydantic AI brings the same range to Python: a typed [`Agent`][pydantic_ai.Agent], [memory](https://pydantic.dev/docs/ai/harness/memory/), [guardrails](https://pydantic.dev/docs/ai/harness/guardrails/), [Pydantic Evals](../evals.md) and a [web chat UI](../web.md), composed from [capabilities](../capabilities/overview.md) rather than fixed constructs.

## Framework

| | Mastra | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Language | TypeScript | Python |
| License | Apache-2.0 (core); EE for some features | MIT |
| Model providers | Many | [Many](../models/overview.md) |
| Extensibility | Tools, processors, scorers, workflows | [Capabilities and toolsets](../extensibility.md); [30+ in the Harness SDK](https://pydantic.dev/docs/ai/harness/) |
| Harnesses | Mastra Code, or your own | Built-in [`Coder`](https://pydantic.dev/docs/ai/harness/coder/) and [`Researcher`](https://pydantic.dev/docs/ai/harness/researcher/), or compose your own |
| Observability | OpenTelemetry | [OpenTelemetry](../logfire.md#using-opentelemetry), including [Pydantic Logfire](https://pydantic.dev/logfire) |
| Interfaces | Playground, Studio | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental) |
| Realtime voice | Yes | [Realtime](../realtime/overview.md) |
| Evals | Yes | [Pydantic Evals](../evals.md) |
| Durable execution | Yes | [5+ integrations](../durable_execution/overview.md) |
| Image generation | No | [Image Generation](../image-generation.md) |

## Features

| | Mastra | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Sub-agents | TBD-FACTCHECK | [Subagents](https://pydantic.dev/docs/ai/harness/subagents/), [delegation](../multi-agent-applications.md), or [`pydantic-graph`](../graph.md) |
| Planning | TBD-FACTCHECK | [Planning](https://pydantic.dev/docs/ai/harness/planning/) |
| Skills | TBD-FACTCHECK | [Skills](https://pydantic.dev/docs/ai/harness/skills/) |
| Memory | Yes | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Token limit: truncate or abort | [Compaction](../capabilities/compaction.md) |
| Guardrails | Yes | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Yes | [Execution environments](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser use | Yes | [Web & research](https://pydantic.dev/docs/ai/harness/#web--research) |
