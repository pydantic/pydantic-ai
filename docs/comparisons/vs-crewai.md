# Pydantic AI vs CrewAI

CrewAI is a team: roles, tasks, a process mode. Pydantic AI has no crew. Multi-agent work is Python:
call, branch, `asyncio.gather`, or [`pydantic-graph`](../graph.md).

## Side by side

| | CrewAI | Pydantic AI |
|---|---|---|
| Language | Python | Python |
| Model providers | Many | [Multiple](../models/overview.md) |
| Durable execution | `Crew.from_checkpoint` | Yes — [5+ integrations](../durable_execution/overview.md) |
| Observability | OpenTelemetry | [OpenTelemetry](../capabilities/instrumentation.md) |
| Extensibility | Tools on agents and crews | [Capabilities and toolsets](../extensibility.md) |
| Interfaces | Enterprise UI, A2A | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental), A2A |
| Build your own harness | Yes | [Yes](https://pydantic.dev/docs/ai/harness/#build-your-own) |
| Coding harness | No (E2B toolkits) | Yes ([`Coder()`](https://pydantic.dev/docs/ai/harness/coder/)) |
| Graph library | No (`Process` modes) | Yes ([`pydantic-graph`](../graph.md)) |
| Multi-agent | Roles, tasks, `Process` | [Sub-agents](../multi-agent-applications.md), [hand-offs](../multi-agent-applications.md#programmatic-agent-hand-off), or [graph](../graph.md) |
| Realtime voice | No | [Yes](../realtime/overview.md) |
| Image generation | Yes (`DallETool`) | [Yes](../image-generation.md) |
| License | MIT | MIT |

| | CrewAI | Pydantic AI |
|---|---|---|
| Structured output | Yes | Yes ([type on the agent](../output.md)) |
| Guardrails | Yes (task guardrails) | [Yes](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Yes (E2B and others) | [Yes](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser | Yes | [Yes](https://pydantic.dev/docs/ai/harness/#web-research) |
| Memory | On `Agent` and `Crew` | [Yes](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | `respect_context_window` | [Yes](../capabilities/compaction.md) |
| Evals | Yes (experimental) | Yes ([Pydantic Evals](../evals.md)) |
| Test without API keys | No (`crewai test` hits a live model) | Yes ([`TestModel`, `FunctionModel`](../testing.md)) |

Cells linked to pydantic.dev/docs/ai/harness ship in [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/),
a separate package.

[Install Pydantic AI](../install.md).
