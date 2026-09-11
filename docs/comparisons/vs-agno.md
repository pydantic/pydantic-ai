# Pydantic AI vs Agno

Agno is a library plus **AgentOS**, a FastAPI runtime with auth, a UI, and storage. Pydantic AI is
only the library: an agent you put in the application you already run.

## Side by side

| | Agno | Pydantic AI |
|---|---|---|
| Language | Python | Python |
| Model providers | Many | [Multiple](../models/overview.md) |
| Durable execution | Agent `db` / `checkpoint` | Yes — [5+ integrations](../durable_execution/overview.md) |
| Observability | OpenTelemetry | [OpenTelemetry](../capabilities/instrumentation.md) |
| Extensibility | Tools, toolkits | [Capabilities and toolsets](../extensibility.md) |
| Interfaces | AG-UI, A2A | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental), A2A |
| Build your own harness | Yes | [Yes](https://pydantic.dev/docs/ai/harness/#build-your-own) |
| Coding harness | No (shell and Python toolkits) | Yes ([`Coder()`](https://pydantic.dev/docs/ai/harness/coder/)) |
| Graph library | Workflows | Yes ([`pydantic-graph`](../graph.md)) |
| Multi-agent | Teams, workflows | [Sub-agents](../multi-agent-applications.md), [hand-offs](../multi-agent-applications.md#programmatic-agent-hand-off), or [graph](../graph.md) |
| Realtime voice | No | [Yes](../realtime/overview.md) |
| Image generation | Yes | [Yes](../image-generation.md) |
| License | Apache-2.0 | MIT |

| | Agno | Pydantic AI |
|---|---|---|
| Structured output | Yes | Yes ([type on the agent](../output.md)) |
| Guardrails | Yes | [Yes](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Yes (integrations) | [Yes](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser | Yes | [Yes](https://pydantic.dev/docs/ai/harness/#web-research) |
| Memory | Yes (`MemoryManager`) | [Yes](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Tool-result compression | [Yes](../capabilities/compaction.md) |
| Evals | Yes | Yes ([Pydantic Evals](../evals.md)) |
| Test without API keys | You write a model | Yes ([`TestModel`, `FunctionModel`](../testing.md)) |

Cells linked to pydantic.dev/docs/ai/harness ship in [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/),
a separate package.

[Install Pydantic AI](../install.md).
