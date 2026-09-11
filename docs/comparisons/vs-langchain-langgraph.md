# Pydantic AI vs LangChain & LangGraph

A LangChain **agent** is a graph: `create_agent()` returns a `CompiledStateGraph`. Pydantic AI is a
typed [`Agent`][pydantic_ai.Agent], with [`pydantic-graph`](../graph.md) when you actually need a graph.

## Side by side

| | LangChain & LangGraph | Pydantic AI |
|---|---|---|
| Language | Python | Python |
| License | MIT | MIT |
| Model providers | Many | [Any](../models/overview.md), plus [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] |
| Durable execution | Checkpointers | Yes — [5+ integrations](../durable_execution/overview.md) |
| Observability | LangSmith | [OpenTelemetry](../capabilities/instrumentation.md), any backend including [Pydantic Logfire](../logfire.md) |
| Extensibility | Middleware, callbacks | [Capabilities](../extensibility.md) and [toolsets](../toolsets.md) |
| Interfaces | LangServe, Studio | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental), [A2A](https://github.com/datalayer/fasta2a) (community) |
| Build your own harness | Yes | [Yes](https://pydantic.dev/docs/ai/harness/#build-your-own) |
| Coding harness | Deep Agents | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| Graph library | LangGraph | [`pydantic-graph`](../graph.md) |
| Multi-agent | LangGraph (handoffs, supervisors, teams) | [Sub-agents](../multi-agent-applications.md), [graph](../graph.md), or `async` |
| Realtime voice | No | [Realtime](../realtime/overview.md) |
| Image generation | Third party | [Image generation](../image-generation.md) |

| | LangChain & LangGraph | Pydantic AI |
|---|---|---|
| Structured output | Yes (`with_structured_output`) | [Type on the agent](../output.md) |
| Guardrails | Middleware | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Yes (integrations) | [Yes](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser | Yes (integrations) | [Yes](https://pydantic.dev/docs/ai/harness/#web-research) |
| Memory | Checkpointers, store | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes (`SummarizationMiddleware`) | [Compaction](../capabilities/compaction.md) |
| Evals | Yes (LangSmith) | [Pydantic Evals](../evals.md) |
| Test without API keys | Yes (fake chat models) | [`TestModel`](../testing.md), [`FunctionModel`](../testing.md) |

Cells linked to pydantic.dev/docs/ai/harness ship in [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/),
a separate package.

## FAQ

**Do you have a graph library?** Yes. [`pydantic-graph`](../graph.md). Most multi-agent work is still
ordinary [async Python](../multi-agent-applications.md).

[Install Pydantic AI](../install.md).
