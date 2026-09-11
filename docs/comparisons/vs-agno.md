# Pydantic AI vs Agno

Agno is a library plus **AgentOS**, a FastAPI runtime with auth, a UI, and storage. Pydantic AI is
only the library: an agent you put in the application you already run.

## Side by side

| | Agno | Pydantic AI |
|---|---|---|
| Language | Python | Python |
| Model providers | Many | [Any](../models/overview.md), plus [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] |
| Durable execution | Agent `db` / `checkpoint` | Yes — [5+ integrations](../durable_execution/overview.md) |
| Observability | OTel, their endpoint | [OpenTelemetry](../capabilities/instrumentation.md), any backend including [Pydantic Logfire](../logfire.md) |
| Extensibility | Tools, toolkits | [Capabilities](../extensibility.md) and [toolsets](../toolsets.md) |
| Interfaces | AG-UI, A2A | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental), [A2A](https://github.com/datalayer/fasta2a) (community) |
| Build your own harness | Yes | [Yes](https://pydantic.dev/docs/ai/harness/#build-your-own) |
| Coding harness | No (shell and Python toolkits) | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| Graph library | Workflows | [`pydantic-graph`](../graph.md) |
| Multi-agent | Teams, workflows | [Sub-agents](../multi-agent-applications.md), [graph](../graph.md), or `async` |
| Realtime voice | No | [Realtime](../realtime/overview.md) |
| Image generation | Yes | [Image generation](../image-generation.md) |
| License | Apache-2.0 | MIT |

| | Agno | Pydantic AI |
|---|---|---|
| Structured output | Yes | [Type on the agent](../output.md) |
| Guardrails | Yes | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Yes (integrations) | [Yes](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser | Yes | [Yes](https://pydantic.dev/docs/ai/harness/#web-research) |
| Memory | Yes (`MemoryManager`) | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Tool-result compression | [Compaction](../capabilities/compaction.md) |
| Evals | Yes | [Pydantic Evals](../evals.md) |
| Test without API keys | You write a model | [`TestModel`](../testing.md), [`FunctionModel`](../testing.md) |

Cells linked to pydantic.dev/docs/ai/harness ship in [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/),
a separate package.

[Install Pydantic AI](../install.md).
