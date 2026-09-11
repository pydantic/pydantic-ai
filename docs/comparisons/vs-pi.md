# Pydantic AI vs Pi

Pi is a TypeScript coding agent from Earendil Works: a terminal agent you extend with hooks, skills and packages, or embed through `createAgentSession`. In Pydantic AI, a [coding agent](https://pydantic.dev/docs/ai/harness/coder/) is one [configuration](../capabilities/overview.md) of a general [`Agent`][pydantic_ai.Agent], and the [skills](https://pydantic.dev/docs/ai/harness/skills/), [sandbox](https://pydantic.dev/docs/ai/harness/#execution-environments) and [sub-agents](https://pydantic.dev/docs/ai/harness/subagents/) are each a capability you can swap.

Pydantic AI is one part of a stack: the [Harness SDK](https://pydantic.dev/docs/ai/harness/) for capabilities and complete agents, [Pydantic Evals](../evals.md), [Pydantic Graph](../graph.md), [Pydantic Logfire](https://pydantic.dev/logfire) for observability, and [Pydantic](https://pydantic.dev/docs/validation/latest/get-started/) itself for validation. The tables below cover the whole of it.

## Framework

| | Pi | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Language | TypeScript | Python |
| License | MIT | MIT |
| Model providers | Many | [Many](../models/overview.md) |
| Extensibility | Extensions, skills, `pi install` | [Capabilities and toolsets](../extensibility.md); [50+ with the Harness SDK](https://pydantic.dev/docs/ai/harness/) |
| Harnesses | Pi itself; extend it or embed it | Built-in [`Coder`](https://pydantic.dev/docs/ai/harness/coder/) and [`Researcher`](https://pydantic.dev/docs/ai/harness/researcher/), or compose your own |
| Observability | Telemetry contract, no exporter | [OpenTelemetry](../logfire.md#using-opentelemetry), including [Pydantic Logfire](https://pydantic.dev/logfire) |
| Durable execution | No | [5+ integrations](../durable_execution/overview.md) |
| Interfaces | CLI | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental) |
| Realtime voice | No | [Realtime](../realtime/overview.md) |
| Evals | No | [Pydantic Evals](../evals.md) |
| Image generation | No | [Image Generation](../image-generation.md) |

## Features

| | Pi | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Multi-agent | No | [Subagents](https://pydantic.dev/docs/ai/harness/subagents/), [delegation](../multi-agent-applications.md), or [`pydantic-graph`](../graph.md) |
| Planning | No | [Planning](https://pydantic.dev/docs/ai/harness/planning/) |
| Skills | Yes | [Skills](https://pydantic.dev/docs/ai/harness/skills/) |
| Memory | No | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes | [Compaction](../capabilities/compaction.md) |
| Guardrails | Yes (`tool_call` hook) | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | No | [Execution environments](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser use | No | [Web & research](https://pydantic.dev/docs/ai/harness/#web--research) |

## FAQ

**Can I build a coding agent like this in Python?** Yes. Give your agent the
[`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) capability, or start from the harness
repository's [complete coding agent](https://github.com/pydantic/pydantic-ai-harness/blob/main/examples/coding_agent.py),
built from the pieces `Coder` puts together.
