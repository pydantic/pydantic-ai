# Pydantic AI vs LangChain & LangGraph

| | LangChain & LangGraph | Pydantic AI |
|---|---|---|
| Language | Python | Python |
| Model providers | Multiple | [Multiple](../models/overview.md) |
| Durable execution | Yes | [5+ integrations](../durable_execution/overview.md) |
| Observability | OpenTelemetry via LangSmith | [OpenTelemetry](../capabilities/instrumentation.md), including [Pydantic Logfire](../logfire.md) |
| Interfaces | LangSmith Agent Server, Fleet | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental) |
| Extensibility | Middleware, callbacks | [Capabilities and toolsets](../extensibility.md) |
| Realtime voice | No | [Realtime](../realtime/overview.md) |
| Image generation | Via provider tools | [Image Generation](../image-generation.md) |
| Multi-agent | LangGraph (handoffs, supervisors, teams) | [Delegation (tools or `SubAgents`), hand-off in your code, or graph](../multi-agent-applications.md) |
| Graph library | Yes (LangGraph `StateGraph`) | [`pydantic-graph`](../graph.md) |
| Build your own harness | Yes | [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) |
| Coding harness | Yes (`deepagents`) | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| License | MIT | MIT |

| | LangChain & LangGraph | Pydantic AI |
|---|---|---|
| Structured output | Yes | [Structured output](../output.md#structured-output) |
| Memory | Yes | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes | [Compaction](../capabilities/compaction.md) |
| Guardrails | Yes | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Yes | [Execution environments](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser | Via provider tools | [Web & research](https://pydantic.dev/docs/ai/harness/#web--research) |
| Evals | Yes | [Pydantic Evals](../evals.md) |

## FAQ

**Do you have a graph library?** Yes. [`pydantic-graph`](../graph.md).
