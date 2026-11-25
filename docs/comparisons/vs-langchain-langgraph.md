# Pydantic AI vs LangChain & LangGraph

LangChain is a large Python ecosystem: LangGraph underneath it for graph-based control flow, `deepagents` for its coding harness, and a large catalogue of integrations. Pydantic AI does it from one typed [`Agent`][pydantic_ai.Agent] with plain Python control flow: [`pydantic-graph`](../graph.md) when you want an explicit graph, a [Harness SDK](https://pydantic.dev/docs/ai/harness/) of ready-made capabilities and complete agents, and validation from the library you already use.

Pydantic AI is one part of a stack: the [Harness SDK](https://pydantic.dev/docs/ai/harness/) for capabilities and complete agents, [Pax](../harness/pax.md) to run any of your agents in a coding terminal, [Pydantic Evals](../evals.md) for evaluation, [Pydantic Graph](../graph.md) for graph-based workflows, [Pydantic Logfire](https://pydantic.dev/logfire) for observability, and [Pydantic](https://pydantic.dev/docs/validation/latest/get-started/) itself for validation. The tables below cover the whole of it.

**Already built on LangChain?** The [`migrating-langchain-to-pydantic-ai`](../framework-migration.md) skill ports an existing application to Pydantic AI one working path at a time, preserving behavior rather than translating API names.

## Framework

| | LangChain & LangGraph | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Language | Python | Python |
| License | MIT | MIT |
| Model providers | Many | [Many](../models/overview.md) |
| Extensibility | Middleware, callbacks | [Capabilities and toolsets](../extensibility.md); [100+ with the Harness SDK](https://pydantic.dev/docs/ai/harness/) |
| Harnesses | `deepagents`, or your own on LangGraph | Built-in [`Coder`](https://pydantic.dev/docs/ai/harness/coder/) and [`Researcher`](https://pydantic.dev/docs/ai/harness/researcher/), or compose your own, and run any of them in [Pax](../harness/pax.md) |
| Observability | OpenTelemetry via LangSmith | [OpenTelemetry](../logfire.md#using-opentelemetry), including [Pydantic Logfire](https://pydantic.dev/logfire) |
| Enterprise support | With [LangSmith Enterprise](https://www.langchain.com/enterprise) | [For the framework itself](../enterprise-support.md), no platform required |
| Durable execution | Yes | [Nine integrations](../durable_execution/overview.md) |
| Interfaces | LangSmith Agent Server, Fleet | [Pax](../harness/pax.md) coding terminal (`pax --agent` runs yours), [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental), [GitHub Agentic Workflows](https://pydantic.dev/docs/ai/harness/gh-aw/) |
| Realtime voice | No | [Realtime](../realtime/overview.md) |
| Evals | Yes | [Pydantic Evals](../evals.md) |
| Image generation | Provider-hosted tools only | [Image Generation](../image-generation.md) |

## Features

| | LangChain & LangGraph | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Multi-agent | Yes | [Subagents](https://pydantic.dev/docs/ai/harness/subagents/), [delegation](../multi-agent-applications.md), or [`pydantic-graph`](../graph.md) |
| Planning | Yes | [Planning](https://pydantic.dev/docs/ai/harness/planning/) |
| Skills | Yes | [Skills](https://pydantic.dev/docs/ai/harness/skills/) |
| Memory | Yes | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes | [Compaction](../capabilities/compaction.md) |
| Guardrails | Yes | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Yes | [Your machine, SSH and 11 sandboxes](../workspace.md#supported-workspaces), with per-path [trust](../workspace-custom.md#workspace-authority) |
| Browser use | Provider-hosted tools only | [Web & research](https://pydantic.dev/docs/ai/harness/#web--research), and [computer use](../harness/computer-use.md) on a desktop in 11 sandboxes |

## FAQ

**Do you have a graph library?** Yes. [`pydantic-graph`](../graph.md): typed nodes, edges from return
types, and persistence for pausing and resuming. Reach for it when the control flow is a real state
machine; plain Python and [sub-agents](https://pydantic.dev/docs/ai/harness/subagents/) cover the rest.

**Can I control which content my agent treats as instructions?** Yes. Everything a capability adds to the conversation is a typed part that records who wrote it (you, your user, or nobody you trust), and Pydantic AI speaks it in the voice that authority allows; what a capability enqueues can be lowered, never raised, as it moves through a run. Tool results, fetched pages and files stay data on every provider, and text that imitates the framework's own frames is flagged to the model. You can mark a workspace's files as untrusted, path by path. See [Capability parts](../message-history.md#capability-parts) and [Marking who wrote the files](../workspace-custom.md#workspace-authority).

*Last checked: 2026-09-24*

## Sources

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [LangGraph durable execution](https://docs.langchain.com/oss/python/langgraph/durable-execution)
