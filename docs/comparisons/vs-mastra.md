# Pydantic AI vs Mastra

Mastra is TypeScript all-in-one: agents, workflows, memory, evals, a playground, Studio, Cloud.
Pydantic AI is Python. One extension point (a capability) instead of tools, processors, guardrails,
and scorers as separate concepts.

## Side by side

| | Mastra | Pydantic AI |
|---|---|---|
| Language | TypeScript | Python |
| License | Apache-2.0 (core); EE for some features | MIT |
| Model providers | Many | [Any](../models/overview.md), plus [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] |
| Durable execution | `createDurableAgent()`, Inngest | Yes — [5+ integrations](../durable_execution/overview.md) |
| Observability | OTel, their endpoint | OpenTelemetry, any backend including [Pydantic Logfire](../logfire.md) |
| Extensibility | Tools, processors, scorers, workflows | [Capabilities](../extensibility.md) and [toolsets](../toolsets.md) |
| Interfaces | Playground, Studio | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental), [A2A](https://github.com/datalayer/fasta2a) (community) |
| Build your own harness | Yes | Yes |
| Coding harness | `createCodingAgent()` | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| Graph library | Workflows in core | [`pydantic-graph`](../graph.md) |
| Multi-agent | Workflows, sub-agents | [Sub-agents](../multi-agent-applications.md), graph, or `async` |
| Realtime voice | Voice extras | [Realtime](../realtime/overview.md) |
| Image generation | No | [Image generation](../image-generation.md) |

| | Mastra | Pydantic AI |
|---|---|---|
| Structured output | Yes | [Type on the agent](../output.md) |
| Guardrails | Processors | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Yes | [Yes](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser | Yes | [Yes](https://pydantic.dev/docs/ai/harness/#web-research) |
| Memory | Yes (working, observational, semantic) | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | No | [Compaction](../capabilities/compaction.md) |
| Evals | Yes | [Pydantic Evals](../evals.md) |
| Test without API keys | Yes | [`TestModel`](../testing.md), [`FunctionModel`](../testing.md) |

Cells linked to pydantic.dev/docs/ai/harness ship in [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/),
a separate package.

[Install Pydantic AI](../install.md).
