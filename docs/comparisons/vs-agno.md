# Pydantic AI vs Agno

Agno is a Python agent framework that optimizes for breadth: a very large tool catalogue, teams, step workflows, and AgentOS to run and watch them. Pydantic AI matches that breadth with a typed core and strict Pydantic validation: [capabilities](../capabilities/overview.md) and the [Harness SDK](https://pydantic.dev/docs/ai/harness/) for the batteries, [sub-agents](https://pydantic.dev/docs/ai/harness/subagents/) for teams, [`pydantic-graph`](../graph.md) for workflows, and [Pydantic Logfire](https://pydantic.dev/logfire) to run and watch them.

## Framework

| | Agno | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Language | Python | Python |
| License | Apache-2.0 | MIT |
| Model providers | Many | [Many](../models/overview.md) |
| Extensibility | Tools, toolkits | [Capabilities and toolsets](../extensibility.md); [30+ in the Harness SDK](https://pydantic.dev/docs/ai/harness/) |
| Harnesses | Build your own | Built-in [`Coder`](https://pydantic.dev/docs/ai/harness/coder/) and [`Researcher`](https://pydantic.dev/docs/ai/harness/researcher/), or compose your own |
| Observability | OpenTelemetry | [OpenTelemetry](../logfire.md#using-opentelemetry), including [Pydantic Logfire](https://pydantic.dev/logfire) |
| Interfaces | AG-UI, A2A, chat platforms | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental) |
| Realtime voice | No | [Realtime](../realtime/overview.md) |
| Evals | Yes | [Pydantic Evals](../evals.md) |
| Durable execution | Yes | [5+ integrations](../durable_execution/overview.md) |
| Image generation | Yes | [Image Generation](../image-generation.md) |

## Features

| | Agno | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Sub-agents | Yes | [Subagents](https://pydantic.dev/docs/ai/harness/subagents/), [delegation](../multi-agent-applications.md), or [`pydantic-graph`](../graph.md) |
| Planning | Teams only | [Planning](https://pydantic.dev/docs/ai/harness/planning/) |
| Skills | Yes | [Skills](https://pydantic.dev/docs/ai/harness/skills/) |
| Memory | Yes | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | History window, session summaries | [Compaction](../capabilities/compaction.md) |
| Guardrails | Yes | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Yes | [Execution environments](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser use | Yes | [Web & research](https://pydantic.dev/docs/ai/harness/#web--research) |
