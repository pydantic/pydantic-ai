# Pydantic AI vs LangChain & LangGraph

LangChain is a large Python ecosystem: LangGraph underneath it for graph-based control flow, `deepagents` for its coding harness, and a large catalogue of integrations. Pydantic AI vs LangChain comes down to surface area: one typed [`Agent`][pydantic_ai.Agent], plain Python control flow, and validation from the library you already use.

## Framework

| | LangChain & LangGraph | Pydantic AI and [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Language | Python | Python |
| License | MIT | MIT |
| Model providers | Many | [Many](../models/overview.md) |
| Extensibility | Middleware, callbacks | [Capabilities and toolsets](../extensibility.md) |
| Build a custom harness | Yes | [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) |
| Coding harness | Yes (`deepagents`) | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| Interfaces | LangSmith Agent Server, Fleet | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental) |
| Realtime voice | No | [Realtime](../realtime/overview.md) |
| Image generation | Via provider tools | [Image Generation](../image-generation.md) |
| Durable execution | Yes | [5+ integrations](../durable_execution/overview.md) |
| Observability | OpenTelemetry via LangSmith | [OpenTelemetry](../capabilities/instrumentation.md), including [Pydantic Logfire](../logfire.md) |
| Evals | Yes | [Pydantic Evals](../evals.md) |

## Features

| | LangChain & LangGraph | Pydantic AI and [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Structured output | Yes | [Structured output](../output.md#structured-output) |
| Multi-agent | LangGraph (handoffs, supervisors, teams) | [Delegation (tools or `SubAgents`), hand-off in your code, or graph](../multi-agent-applications.md) |
| Graph library | Yes (LangGraph `StateGraph`) | [`pydantic-graph`](../graph.md) |
| Memory | Yes | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes | [Compaction](../capabilities/compaction.md) |
| Guardrails | Yes | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Yes | [Execution environments](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser use | Via provider tools | [Web & research](https://pydantic.dev/docs/ai/harness/#web--research) |

## FAQ

**Do you have a graph library?** Yes. [`pydantic-graph`](../graph.md).
