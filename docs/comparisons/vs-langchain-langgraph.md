# Pydantic AI vs LangChain & LangGraph

A LangChain **agent** is a graph: `create_agent()` returns a `CompiledStateGraph`. Pydantic AI is a
typed [`Agent`][pydantic_ai.Agent], with [`pydantic-graph`](../graph.md) when you actually need a graph.

## Side by side

| | LangChain & LangGraph | Pydantic AI |
|---|---|---|
| Native Python SDK | Yes | Yes |
| License | MIT | MIT |
| Model providers | Many | Any, plus [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] |
| Extensibility | Middleware, callbacks | [Capabilities](../extensibility.md) |
| Skills | Yes | Yes ([Skills](https://pydantic.dev/docs/ai/harness/skills/)) |
| Interfaces | LangServe, Studio | [`to_cli_sync()`](../cli.md), [`to_web()`](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/), [A2A extra](https://github.com/datalayer/fasta2a) |
| Realtime voice | No | Yes ([realtime](../realtime/overview.md)) |
| Agent graph | LangGraph | [`pydantic-graph`](../graph.md) |
| Coding agent | Deep Agents | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| Research agent | Open Deep Research | [`Researcher()`](https://pydantic.dev/docs/ai/harness/researcher/) |
| Web search | Yes | Yes ([WebSearch](../capabilities/web-search.md)) |
| Code sandboxes | Community / E2B | [Modal](https://pydantic.dev/docs/ai/harness/modal-sandbox/), [Monty](https://github.com/pydantic/monty), [Code Mode](https://pydantic.dev/docs/ai/harness/code-mode/) |
| Image generation | Integrations | Yes (OpenAI, Google, xAI) |
| Browser | Integrations | Yes ([Browser Use](https://pydantic.dev/docs/ai/harness/browser-use/) and [Playwright](https://pydantic.dev/docs/ai/harness/playwright/)) |
| Structured output | Yes (`with_structured_output`) | Yes (type on the agent) |
| Human in the loop | Yes (`interrupt`) | Yes (tool approval) |
| Guardrails | Middleware | Yes ([harness](https://pydantic.dev/docs/ai/harness/guardrails/)) |
| Spend limits | No | Yes ([cost_limit](../agent.md#usage-limits), [spend](https://pydantic.dev/docs/ai/harness/spend/)) |
| MCP | Client (adapters) | Client and [server](../mcp/server.md) |
| Memory | Checkpointers, store | [Harness Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes (Deep Agents) | Yes ([compaction](https://pydantic.dev/docs/ai/harness/compaction/)) |
| Multi-agent | LangGraph | [Delegation, graph, or `async`](../multi-agent-applications.md) |
| Durable execution | Checkpointers | [Temporal, DBOS, Prefect, Restate, Lambda, Kitaru, Airflow](../durable_execution/overview.md) |
| Tracing | LangSmith | OpenTelemetry |
| Evals | Yes (LangSmith) | Yes ([Pydantic Evals](../evals.md)) |
| Test without API keys | Yes (fake chat models) | Yes |
| Embeddings | Yes (large catalogue) | Yes |
| Integrations | Large catalogue | Wrap theirs in a tool |

## FAQ

**Do you have a graph library?** Yes. [`pydantic-graph`](../graph.md). Most multi-agent work is still
ordinary [async Python](../multi-agent-applications.md).

**Can I get a coding agent without Deep Agents?** Yes.
[`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) on the same typed [`Agent`][pydantic_ai.Agent].
