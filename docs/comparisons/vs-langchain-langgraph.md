# Pydantic AI vs LangChain & LangGraph

A LangChain **agent** is a graph: `create_agent()` returns a `CompiledStateGraph`. Pydantic AI is a
typed [`Agent`][pydantic_ai.Agent], with [`pydantic-graph`](../graph.md) when you actually need a graph.

## Side by side

| | LangChain & LangGraph | Pydantic AI |
|---|---|---|
| Language | Python | Python |
| Model providers | Multiple | [Multiple](../models/overview.md) |
| Durable execution | Yes | [5+ integrations](../durable_execution/overview.md) |
| Observability | LangSmith | [OpenTelemetry](../capabilities/instrumentation.md), including [Pydantic Logfire](../logfire.md) |
| Extensibility | Middleware, callbacks | [Capabilities and toolsets](../extensibility.md) |
| Interfaces | LangSmith Agent Server, Fleet | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) |
| Build your own harness | Yes | [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) |
| Coding harness | Yes | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| Graph library | Yes | [`pydantic-graph`](../graph.md) |
| Multi-agent | LangGraph (handoffs, supervisors, teams) | [Sub-agents, hand-offs, or graph](../multi-agent-applications.md) |
| Realtime voice | No | [Realtime](../realtime/overview.md) |
| Image generation | Yes | [Image Generation](../image-generation.md) |
| License | MIT | MIT |

| | LangChain & LangGraph | Pydantic AI |
|---|---|---|
| Structured output | Yes | [Structured output](../output.md#structured-output) |
| Guardrails | Yes | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Yes | [Execution environments](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser | Yes | [Web & research](https://pydantic.dev/docs/ai/harness/#web--research) |
| Memory | Yes | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes | [Compaction](../capabilities/compaction.md) |
| Evals | Yes | [Pydantic Evals](../evals.md) |
| Test without API keys | Yes | [Testing](../testing.md) |

## FAQ

**Do you have a graph library?** Yes. [`pydantic-graph`](../graph.md). Most multi-agent work is still
ordinary [async Python](../multi-agent-applications.md).
