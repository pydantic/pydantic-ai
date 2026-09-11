# Pydantic AI vs CrewAI

CrewAI is a team: roles, tasks, a process mode. Pydantic AI has no crew. Multi-agent work is Python:
call, branch, `asyncio.gather`, or [`pydantic-graph`](../graph.md).

## Side by side

| | CrewAI | Pydantic AI |
|---|---|---|
| Native Python SDK | Yes | Yes |
| License | MIT | MIT |
| Model providers | Many | Any, plus [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] |
| Extensibility | Tools on agents and crews | [Capabilities](../extensibility.md) |
| Skills | Yes | Yes ([Skills](https://pydantic.dev/docs/ai/harness/skills/)) |
| Interfaces | Enterprise UI, A2A | [`to_cli_sync()`](../cli.md), [`to_web()`](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental), [A2A](https://github.com/datalayer/fasta2a) (community) |
| Realtime voice | No | Yes ([realtime](../realtime/overview.md)) |
| Agent graph | `Process` (sequential / hierarchical) | [`pydantic-graph`](../graph.md) |
| Coding agent | E2B tools | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| Research agent | No | [`Researcher()`](https://pydantic.dev/docs/ai/harness/researcher/) |
| Web search | Third party | Native ([WebSearch](../capabilities/web-search.md)) and third party ([DuckDuckGo](../common-tools.md#duckduckgo-search-tool), [Tavily](../common-tools.md#tavily-search-tool), [You.com](https://pydantic.dev/docs/ai/harness/youdotcom/), [Exa](https://pydantic.dev/docs/ai/harness/exa-search/)) |
| Code sandboxes | Third party | [Modal](https://pydantic.dev/docs/ai/harness/modal-sandbox/) |
| Image generation | Yes (`DallETool`) | Yes (OpenAI, Google, xAI) |
| Browser | Yes | Yes ([Browser Use](https://pydantic.dev/docs/ai/harness/browser-use/) and [Playwright](https://pydantic.dev/docs/ai/harness/playwright/)) |
| Structured output | Yes | Yes (type on the agent) |
| Human in the loop | Yes (human tools) | Yes (tool approval) |
| Guardrails | Yes (task guardrails) | Yes ([harness](https://pydantic.dev/docs/ai/harness/guardrails/)) |
| Spend limits | `max_iter` / `max_rpm` | Yes ([cost_limit](../agent.md#usage-limits), [spend](https://pydantic.dev/docs/ai/harness/spend/)) |
| MCP | Client | Client and [server](../mcp/server.md) |
| Memory | On `Agent` and `Crew` | [Harness Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | `respect_context_window` | Yes ([compaction](../capabilities/compaction.md), [harness](https://pydantic.dev/docs/ai/harness/compaction/)) |
| Multi-agent | Roles, tasks, `Process` | [Delegation, graph, or `async`](../multi-agent-applications.md) |
| Durable execution | `Crew.from_checkpoint` | [Temporal, DBOS, Prefect, Restate](../durable_execution/overview.md), [Lambda](https://pydantic.dev/docs/ai/harness/aws-lambda/); Kitaru and Airflow (external) |
| Tracing | OTel, their endpoint | OpenTelemetry, any backend |
| Evals | Yes (experimental) | Yes ([Pydantic Evals](../evals.md)) |
| Test without API keys | No (`crewai test` hits a live model) | Yes ([TestModel](../testing.md), [FunctionModel](../testing.md)) |
| Embeddings | Knowledge | Yes |

Harness-linked cells ship in [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/), a separate package.

[Install Pydantic AI](../install.md).
