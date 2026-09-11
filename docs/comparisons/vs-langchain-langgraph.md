# Pydantic AI vs LangChain & LangGraph

LangChain is a large Python ecosystem: LangGraph underneath it for graph-based control flow, `deepagents` for its coding harness, and a large catalogue of integrations. Pydantic AI covers the same ground from one typed [`Agent`][pydantic_ai.Agent] with plain Python control flow, [`pydantic-graph`](../graph.md) when you want an explicit graph, a [Harness SDK](https://pydantic.dev/docs/ai/harness/) of ready-made capabilities and complete agents, and validation from the library you already use.

## Framework

| | LangChain & LangGraph | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Language | Python | Python |
| License | MIT | MIT |
| Model providers | Many | [Many](../models/overview.md) |
| Extensibility | Middleware, callbacks | [Capabilities and toolsets](../extensibility.md); [30+ in the Harness SDK](https://pydantic.dev/docs/ai/harness/) |
| Harnesses | `deepagents`, or your own on LangGraph | Built-in [`Coder`](https://pydantic.dev/docs/ai/harness/coder/) and [`Researcher`](https://pydantic.dev/docs/ai/harness/researcher/), or compose your own |
| Observability | OpenTelemetry via LangSmith | [OpenTelemetry](../logfire.md#using-opentelemetry), including [Pydantic Logfire](https://pydantic.dev/logfire) |
| Interfaces | LangSmith Agent Server, Fleet | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental) |
| Realtime voice | No | [Realtime](../realtime/overview.md) |
| Evals | Yes | [Pydantic Evals](../evals.md) |
| Durable execution | Yes | [5+ integrations](../durable_execution/overview.md) |
| Image generation | Provider-hosted tools only | [Image Generation](../image-generation.md) |

## Features

| | LangChain & LangGraph | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Sub-agents | Yes | [Subagents](https://pydantic.dev/docs/ai/harness/subagents/), [delegation](../multi-agent-applications.md), or [`pydantic-graph`](../graph.md) |
| Planning | Yes | [Planning](https://pydantic.dev/docs/ai/harness/planning/) |
| Skills | Yes | [Skills](https://pydantic.dev/docs/ai/harness/skills/) |
| Memory | Yes | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes | [Compaction](../capabilities/compaction.md) |
| Guardrails | Yes | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Yes | [Execution environments](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser use | Provider-hosted tools only | [Web & research](https://pydantic.dev/docs/ai/harness/#web--research) |

## FAQ

**Do you have a graph library?** Yes. [`pydantic-graph`](../graph.md).
