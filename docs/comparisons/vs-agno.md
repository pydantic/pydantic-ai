# Pydantic AI vs Agno

Agno is a library plus **AgentOS**, a FastAPI runtime with auth, a UI, and storage. Pydantic AI is
only the library: an agent you put in the application you already run.

## Side by side

| | Agno | Pydantic AI |
|---|---|---|
| Native Python SDK | Yes | Yes |
| License | Apache-2.0 | MIT |
| Model providers | Many | Any, plus [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] |
| Extensibility | Toolkits | [Capabilities](../extensibility.md) |
| Skills | Yes | Yes ([Skills](https://pydantic.dev/docs/ai/harness/skills/)) |
| Interfaces | AG-UI, A2A | [`to_cli_sync()`](../cli.md), [`to_web()`](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/), [A2A extra](https://github.com/datalayer/fasta2a) |
| Realtime voice | No | Yes ([realtime](../realtime/overview.md)) |
| Agent graph | Workflows | [`pydantic-graph`](../graph.md) |
| Coding agent | Shell / Python toolkits | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| Research agent | No | [`Researcher()`](https://pydantic.dev/docs/ai/harness/researcher/) |
| Web search | Third party (DuckDuckGo, Tavily, Exa) | Native and third party ([WebSearch](../capabilities/web-search.md)) |
| Code sandboxes | E2B, Daytona, Superserve | [Modal](https://pydantic.dev/docs/ai/harness/modal-sandbox/) |
| Image generation | Yes | Yes (OpenAI, Google, xAI) |
| Browser | Yes | Yes ([Browser Use](https://pydantic.dev/docs/ai/harness/browser-use/) and [Playwright](https://pydantic.dev/docs/ai/harness/playwright/)) |
| Structured output | Yes | Yes (type on the agent) |
| Human in the loop | Yes (confirmation on tools) | Yes (tool approval) |
| Guardrails | Yes | Yes ([harness](https://pydantic.dev/docs/ai/harness/guardrails/)) |
| Spend limits | Claude passthrough | Yes ([cost_limit](../agent.md#usage-limits), [spend](https://pydantic.dev/docs/ai/harness/spend/)) |
| MCP | Client | Client and [server](../mcp/server.md) |
| Memory | Yes (`MemoryManager`) | Yes ([harness Memory](https://pydantic.dev/docs/ai/harness/memory/); thinner) |
| Compaction | Tool-result compression | Yes ([compaction](https://pydantic.dev/docs/ai/harness/compaction/)) |
| Multi-agent | Teams | [Delegation, graph, or `async`](../multi-agent-applications.md) |
| Durable execution | Agent `db` / `checkpoint` | [Temporal, DBOS, Prefect, Restate, Lambda, Kitaru, Airflow](../durable_execution/overview.md) |
| Tracing | OpenInference | OpenTelemetry |
| Evals | Yes | Yes ([Pydantic Evals](../evals.md)) |
| Test without API keys | Yes | Yes |
| Embeddings | Knowledge | Yes |

## FAQ

**Does the agent need its own service?** No. It goes in the app you already run.

**Can I get a chat UI without AgentOS?** Yes. [`to_web()`][pydantic_ai.agent.Agent.to_web] and
[`to_cli_sync()`](../cli.md).
