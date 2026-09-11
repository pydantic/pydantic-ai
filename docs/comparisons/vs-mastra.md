# Pydantic AI vs Mastra

Mastra is TypeScript all-in-one: agents, workflows, memory, evals, a playground, Studio, Cloud.
Pydantic AI is Python. One extension point (a capability) instead of tools, processors, guardrails,
and scorers as separate concepts.

## Side by side

| | Mastra | Pydantic AI |
|---|---|---|
| Language | TypeScript | Python |
| Model providers | Multiple | [Multiple](../models/overview.md) |
| Durable execution | Yes (`createDurableAgent()`, Inngest) | Yes — [5+ integrations](../durable_execution/overview.md) |
| Observability | OpenTelemetry | [OpenTelemetry](../capabilities/instrumentation.md) |
| Extensibility | Tools, processors, scorers, workflows | [Capabilities and toolsets](../extensibility.md) |
| Interfaces | Playground, Studio | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental), A2A |
| Build your own harness | Yes | [Yes](https://pydantic.dev/docs/ai/harness/#build-your-own) |
| Coding harness | Yes (`createCodingAgent()`) | Yes ([`Coder()`](https://pydantic.dev/docs/ai/harness/coder/)) |
| Graph library | Yes (workflows in core) | Yes ([`pydantic-graph`](../graph.md)) |
| Multi-agent | Workflows, sub-agents | [Sub-agents](../multi-agent-applications.md), [hand-offs](../multi-agent-applications.md#programmatic-agent-hand-off), or [graph](../graph.md) |
| Realtime voice | Yes (voice packages) | [Yes](../realtime/overview.md) |
| Image generation | No | [Yes](../image-generation.md) |
| License | Apache-2.0 (core); EE for some features | MIT |

| | Mastra | Pydantic AI |
|---|---|---|
| Structured output | Yes | [Yes](../output.md) |
| Guardrails | Yes (processors) | [Yes](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Yes | [Yes](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser | Yes | [Yes](https://pydantic.dev/docs/ai/harness/#web-research) |
| Memory | Yes (working, observational, semantic) | [Yes](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes (`TokenLimiterProcessor`, `summarizeThread`) | [Yes](../capabilities/compaction.md) |
| Evals | Yes | [Yes](../evals.md) |
| Test without API keys | Yes | [Yes](../testing.md) |

Cells linked to pydantic.dev/docs/ai/harness ship in [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/),
a separate package.

[Install Pydantic AI](../install.md).
