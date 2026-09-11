# Pydantic AI vs Mastra

Mastra is TypeScript all-in-one: agents, workflows, memory, evals, a playground, Studio, Cloud.
Pydantic AI is Python. One extension point (a capability) instead of tools, processors, guardrails,
and scorers as separate concepts.

## Side by side

| | Mastra | Pydantic AI |
|---|---|---|
| Native Python SDK | No, TypeScript | Yes |
| License | Apache-2.0 (core); EE for some features | MIT |
| Model providers | Many | Any, plus [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] |
| Extensibility | Tools, processors, scorers, workflows | [Capabilities](../extensibility.md) |
| Skills | Yes | Yes ([Skills](https://pydantic.dev/docs/ai/harness/skills/)) |
| Interfaces | Playground, Studio | [`to_cli_sync()`](../cli.md), [`to_web()`](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental), [A2A](https://github.com/datalayer/fasta2a) (community) |
| Realtime voice | Voice extras | Yes ([realtime](../realtime/overview.md)) |
| Agent graph | Workflows in core | [`pydantic-graph`](../graph.md) |
| Coding agent | `createCodingAgent()` | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| Research agent | No | [`Researcher()`](https://pydantic.dev/docs/ai/harness/researcher/) |
| Web search | Native | Native ([WebSearch](../capabilities/web-search.md)) and third party ([DuckDuckGo](../common-tools.md#duckduckgo-search-tool), [Tavily](../common-tools.md#tavily-search-tool), [You.com](https://pydantic.dev/docs/ai/harness/youdotcom/), [Exa](https://pydantic.dev/docs/ai/harness/exa-search/)) |
| Code sandboxes | Yes | [Modal](https://pydantic.dev/docs/ai/harness/modal-sandbox/) |
| Image generation | No | Yes (OpenAI, Google, xAI) |
| Browser | Yes | Yes ([Browser Use](https://pydantic.dev/docs/ai/harness/browser-use/) and [Playwright](https://pydantic.dev/docs/ai/harness/playwright/)) |
| Structured output | Yes | Yes (type on the agent) |
| Human in the loop | Yes (processors / approval) | Yes (tool approval) |
| Guardrails | Processors | Yes ([harness](https://pydantic.dev/docs/ai/harness/guardrails/)) |
| Spend limits | `TokenCostControl` | Yes ([cost_limit](../agent.md#usage-limits), [spend](https://pydantic.dev/docs/ai/harness/spend/)) |
| MCP | Client and server | Client and [server](../mcp/server.md) |
| Memory | Yes (working, observational, semantic) | Yes ([harness Memory](https://pydantic.dev/docs/ai/harness/memory/); thinner) |
| Compaction | No | Yes ([compaction](../capabilities/compaction.md), [harness](https://pydantic.dev/docs/ai/harness/compaction/)) |
| Multi-agent | Workflows, sub-agents | [Delegation, graph, or `async`](../multi-agent-applications.md) |
| Durable execution | `createDurableAgent()`, Inngest | [Temporal, DBOS, Prefect, Restate](../durable_execution/overview.md), [Lambda](https://pydantic.dev/docs/ai/harness/aws-lambda/); Kitaru and Airflow (external) |
| Tracing | OTel, their endpoint | OpenTelemetry, any backend |
| Evals | Yes | Yes ([Pydantic Evals](../evals.md)) |
| Test without API keys | Yes | Yes ([TestModel](../testing.md), [FunctionModel](../testing.md)) |
| Embeddings | Yes (with memory) | Yes |

Harness-linked cells ship in [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/), a separate package.

[Install Pydantic AI](../install.md).
