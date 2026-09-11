# Pydantic AI vs CrewAI

CrewAI models an agent system as a crew: roles, tasks and a process that runs them, with `Flow` for deterministic control flow around the crew and a large first-party tool package. Pydantic AI starts from a plain typed [`Agent`][pydantic_ai.Agent] and lets you pick the orchestration: [sub-agents](https://pydantic.dev/docs/ai/harness/subagents/) and [delegation](../multi-agent-applications.md) for a crew, [`pydantic-graph`](../graph.md) for a flow, and a [Harness SDK](https://pydantic.dev/docs/ai/harness/) of tools and capabilities.

## Framework

| | CrewAI | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Language | Python | Python |
| License | MIT | MIT |
| Model providers | Many | [Many](../models/overview.md) |
| Extensibility | Tools on agents and crews | [Capabilities and toolsets](../extensibility.md); [30+ in the Harness SDK](https://pydantic.dev/docs/ai/harness/) |
| Harnesses | Build your own | Built-in [`Coder`](https://pydantic.dev/docs/ai/harness/coder/) and [`Researcher`](https://pydantic.dev/docs/ai/harness/researcher/), or compose your own |
| Observability | OpenTelemetry | [OpenTelemetry](../logfire.md#using-opentelemetry), including [Pydantic Logfire](https://pydantic.dev/logfire) |
| Interfaces | Enterprise UI, A2A | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental) |
| Realtime voice | No | [Realtime](../realtime/overview.md) |
| Evals | Yes | [Pydantic Evals](../evals.md) |
| Durable execution | Yes | [5+ integrations](../durable_execution/overview.md) |
| Image generation | Yes | [Image Generation](../image-generation.md) |

## Features

| | CrewAI | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Sub-agents | Yes | [Subagents](https://pydantic.dev/docs/ai/harness/subagents/), [delegation](../multi-agent-applications.md), or [`pydantic-graph`](../graph.md) |
| Planning | Planner loop, no task list | [Planning](https://pydantic.dev/docs/ai/harness/planning/) |
| Skills | Yes | [Skills](https://pydantic.dev/docs/ai/harness/skills/) |
| Memory | Yes | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes | [Compaction](../capabilities/compaction.md) |
| Guardrails | Yes | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Yes | [Execution environments](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser use | Yes | [Web & research](https://pydantic.dev/docs/ai/harness/#web--research) |
