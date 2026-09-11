# Pydantic AI vs CrewAI

CrewAI models an agent system as a crew: roles, tasks and a process that runs them, with `Flow` for deterministic control flow around the crew and a large first-party tool package. It suits teams that think in job descriptions. Pydantic AI vs CrewAI comes down to abstraction level: we hand you a plain typed [`Agent`][pydantic_ai.Agent] and leave the orchestration pattern to you.

| | CrewAI | Pydantic AI |
|---|---|---|
| Language | Python | Python |
| Model providers | Multiple | [Multiple](../models/overview.md) |
| Durable execution | Yes | [5+ integrations](../durable_execution/overview.md) |
| Observability | OpenTelemetry | [OpenTelemetry](../capabilities/instrumentation.md), including [Pydantic Logfire](../logfire.md) |
| Interfaces | Enterprise UI, A2A | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental) |
| Extensibility | Tools on agents and crews | [Capabilities and toolsets](../extensibility.md) |
| Realtime voice | No | [Realtime](../realtime/overview.md) |
| Image generation | Yes (`crewai-tools`) | [Image Generation](../image-generation.md) |
| Multi-agent | Roles, tasks, `Process` | [Delegation (tools or `SubAgents`), hand-off in your code, or graph](../multi-agent-applications.md) |
| Graph library | Yes (`Flow`) | [`pydantic-graph`](../graph.md) |
| Build your own harness | Yes | [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) |
| Coding harness | No | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| License | MIT | MIT |

| | CrewAI | Pydantic AI |
|---|---|---|
| Structured output | Yes | [Structured output](../output.md#structured-output) |
| Memory | Yes | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes | [Compaction](../capabilities/compaction.md) |
| Guardrails | Yes | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Yes (`crewai-tools`) | [Execution environments](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser | Yes (`crewai-tools`) | [Web & research](https://pydantic.dev/docs/ai/harness/#web--research) |
| Evals | Yes | [Pydantic Evals](../evals.md) |
