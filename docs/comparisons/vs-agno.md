# Pydantic AI vs Agno

Agno is a Python agent framework that optimizes for breadth and speed: a very large tool catalogue, teams, step workflows, and AgentOS to run and watch them. If you want batteries and a control plane from one install, it delivers. Pydantic AI vs Agno trades that breadth for a smaller typed core, strict Pydantic validation, and [capabilities](../extensibility.md) you add when you need them.

| | Agno | Pydantic AI |
|---|---|---|
| Language | Python | Python |
| Model providers | Multiple | [Multiple](../models/overview.md) |
| Durable execution | Yes | [5+ integrations](../durable_execution/overview.md) |
| Observability | OpenTelemetry | [OpenTelemetry](../capabilities/instrumentation.md), including [Pydantic Logfire](../logfire.md) |
| Interfaces | AG-UI, A2A, chat platforms | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental) |
| Extensibility | Tools, toolkits | [Capabilities and toolsets](../extensibility.md) |
| Realtime voice | No | [Realtime](../realtime/overview.md) |
| Image generation | Yes | [Image Generation](../image-generation.md) |
| Multi-agent | Teams (delegation), workflows | [Delegation (tools or `SubAgents`), hand-off in your code, or graph](../multi-agent-applications.md) |
| Graph library | No (step workflows) | [`pydantic-graph`](../graph.md) |
| Build your own harness | Yes | [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) |
| Coding harness | No | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| License | Apache-2.0 | MIT |

| | Agno | Pydantic AI |
|---|---|---|
| Structured output | Yes | [Structured output](../output.md#structured-output) |
| Memory | Yes | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | History window, session summaries | [Compaction](../capabilities/compaction.md) |
| Guardrails | Yes | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Yes | [Execution environments](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser | Yes | [Web & research](https://pydantic.dev/docs/ai/harness/#web--research) |
| Evals | Yes | [Pydantic Evals](../evals.md) |
