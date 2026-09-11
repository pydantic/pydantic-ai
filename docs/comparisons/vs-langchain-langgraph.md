# Pydantic AI vs LangChain & LangGraph

A LangChain **agent** is a graph: `create_agent()` returns a `CompiledStateGraph`. Pydantic AI is a
typed [`Agent`][pydantic_ai.Agent], with [`pydantic-graph`](../graph.md) when you actually need a graph.

## Side by side

| | LangChain & LangGraph | Pydantic AI |
|---|---|---|
| Language | Python | Python |
| Model providers | Many | [Multiple](../models/overview.md) |
| Durable execution | Checkpointers | Yes — [5+ integrations](../durable_execution/overview.md) |
| Observability | LangSmith | [OpenTelemetry](../capabilities/instrumentation.md) |
| Extensibility | Middleware, callbacks | [Capabilities and toolsets](../extensibility.md) |
| Interfaces | LangServe, Studio | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental), A2A |
| Build your own harness | Yes | [Yes](https://pydantic.dev/docs/ai/harness/#build-your-own) |
| Coding harness | Deep Agents | Yes ([`Coder()`](https://pydantic.dev/docs/ai/harness/coder/)) |
| Graph library | LangGraph | Yes ([`pydantic-graph`](../graph.md)) |
| Multi-agent | LangGraph (handoffs, supervisors, teams) | [Sub-agents](../multi-agent-applications.md), [hand-offs](../multi-agent-applications.md#programmatic-agent-hand-off), or [graph](../graph.md) |
| Realtime voice | No | [Yes](../realtime/overview.md) |
| Image generation | Third party | [Yes](../image-generation.md) |
| License | MIT | MIT |

| | LangChain & LangGraph | Pydantic AI |
|---|---|---|
| Structured output | Yes (`with_structured_output`) | [Yes](../output.md) |
| Guardrails | Middleware | [Yes](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Yes (integrations) | [Yes](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser | Yes (integrations) | [Yes](https://pydantic.dev/docs/ai/harness/#web-research) |
| Memory | Checkpointers, store | [Yes](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes (`SummarizationMiddleware`) | [Yes](../capabilities/compaction.md) |
| Evals | Yes (LangSmith) | [Yes](../evals.md) |
| Test without API keys | Yes (fake chat models) | [Yes](../testing.md) |

Cells linked to pydantic.dev/docs/ai/harness ship in [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/),
a separate package.

## FAQ

**Do you have a graph library?** Yes. [`pydantic-graph`](../graph.md). Most multi-agent work is still
ordinary [async Python](../multi-agent-applications.md).

[Install Pydantic AI](../install.md).
