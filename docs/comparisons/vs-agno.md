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
| Interfaces | AG-UI, A2A | [`to_cli_sync()`](../cli.md), [`to_web()`](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental), [A2A](https://github.com/datalayer/fasta2a) (community) |
| Realtime voice | No | Yes ([realtime](../realtime/overview.md)) |
| Agent graph | Workflows | [`pydantic-graph`](../graph.md) |
| Coding agent | Shell / Python toolkits | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| Research agent | No | [`Researcher()`](https://pydantic.dev/docs/ai/harness/researcher/) |
| Web search | Third party | Native ([WebSearch](../capabilities/web-search.md)) and third party ([DuckDuckGo](../common-tools.md#duckduckgo-search-tool), [Tavily](../common-tools.md#tavily-search-tool), [You.com](https://pydantic.dev/docs/ai/harness/youdotcom/), [Exa](https://pydantic.dev/docs/ai/harness/exa-search/)) |
| Code sandboxes | Third party | [Modal](https://pydantic.dev/docs/ai/harness/modal-sandbox/) |
| Image generation | Yes | Yes (OpenAI, Google, xAI) |
| Browser | Yes | Yes ([Browser Use](https://pydantic.dev/docs/ai/harness/browser-use/) and [Playwright](https://pydantic.dev/docs/ai/harness/playwright/)) |
| Structured output | Yes | Yes (type on the agent) |
| Human in the loop | Yes (confirmation on tools) | Yes (tool approval) |
| Guardrails | Yes | Yes ([harness](https://pydantic.dev/docs/ai/harness/guardrails/)) |
| Spend limits | `tool_call_limit` | Yes ([cost_limit](../agent.md#usage-limits), [spend](https://pydantic.dev/docs/ai/harness/spend/)) |
| MCP | Client and server | Client and [server](../mcp/server.md) |
| Memory | Yes (`MemoryManager`) | Yes ([harness Memory](https://pydantic.dev/docs/ai/harness/memory/); thinner) |
| Compaction | Tool-result compression | Yes ([compaction](../capabilities/compaction.md), [harness](https://pydantic.dev/docs/ai/harness/compaction/)) |
| Multi-agent | Teams | [Delegation, graph, or `async`](../multi-agent-applications.md) |
| Durable execution | Agent `db` / `checkpoint` | [Temporal, DBOS, Prefect, Restate](../durable_execution/overview.md), [Lambda](https://pydantic.dev/docs/ai/harness/aws-lambda/); Kitaru and Airflow (external) |
| Tracing | OTel, their endpoint | OpenTelemetry, any backend |
| Evals | Yes | Yes ([Pydantic Evals](../evals.md)) |
| Test without API keys | You write a model | Yes ([TestModel](../testing.md), [FunctionModel](../testing.md)) |
| Embeddings | Knowledge | Yes |

Harness-linked cells ship in [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/), a separate package.

[Install Pydantic AI](../install.md).
