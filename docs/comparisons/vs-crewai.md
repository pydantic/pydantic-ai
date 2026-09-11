# Pydantic AI vs CrewAI

CrewAI is a team: roles, tasks, a process mode. Pydantic AI has no crew. Multi-agent work is Python:
call, branch, `asyncio.gather`, or [`pydantic-graph`](../graph.md).

## Side by side

| | CrewAI | Pydantic AI |
|---|---|---|
| Language | Python | Python |
| Model providers | Many | [Any](../models/overview.md), plus [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] |
| Durable execution | `Crew.from_checkpoint` | Yes — [5+ integrations](../durable_execution/overview.md) |
| Observability | OTel, their endpoint | [OpenTelemetry](../capabilities/instrumentation.md), any backend including [Pydantic Logfire](../logfire.md) |
| Extensibility | Tools on agents and crews | [Capabilities](../extensibility.md) and [toolsets](../toolsets.md) |
| Interfaces | Enterprise UI, A2A | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental), [A2A](https://github.com/datalayer/fasta2a) (community) |
| Build your own harness | Yes | [Yes](https://pydantic.dev/docs/ai/harness/#build-your-own) |
| Coding harness | No (E2B toolkits) | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| Graph library | No (`Process` modes) | [`pydantic-graph`](../graph.md) |
| Multi-agent | Roles, tasks, `Process` | [Sub-agents](../multi-agent-applications.md), [graph](../graph.md), or `async` |
| Realtime voice | No | [Realtime](../realtime/overview.md) |
| Image generation | Yes (`DallETool`) | [Image generation](../image-generation.md) |
| License | MIT | MIT |

| | CrewAI | Pydantic AI |
|---|---|---|
| Structured output | Yes | [Type on the agent](../output.md) |
| Guardrails | Yes (task guardrails) | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Yes (E2B and others) | [Yes](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser | Yes | [Yes](https://pydantic.dev/docs/ai/harness/#web-research) |
| Memory | On `Agent` and `Crew` | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | `respect_context_window` | [Compaction](../capabilities/compaction.md) |
| Evals | Yes (experimental) | [Pydantic Evals](../evals.md) |
| Test without API keys | No (`crewai test` hits a live model) | [`TestModel`](../testing.md), [`FunctionModel`](../testing.md) |

Cells linked to pydantic.dev/docs/ai/harness ship in [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/),
a separate package.

[Install Pydantic AI](../install.md).
