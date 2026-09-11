# Pydantic AI vs Vercel AI SDK

The Vercel AI SDK is the wire to a React UI: streaming, tool cards, approval in the browser. Pydantic
AI is a Python agent. [`VercelAIAdapter`][pydantic_ai.ui.vercel_ai.VercelAIAdapter] speaks their
protocol, so the browser can stay theirs.

## Side by side

| | Vercel AI SDK | Pydantic AI |
|---|---|---|
| Native Python SDK | No, TypeScript | Yes |
| License | Apache-2.0 | MIT |
| Model providers | Many | Any, plus [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] |
| Extensibility | Middleware, tools | [Capabilities](../extensibility.md) |
| Skills | Yes (`uploadSkill`) | Yes ([Skills](https://pydantic.dev/docs/ai/harness/skills/)) |
| Interfaces | React `useChat` / Vercel AI stream | [`to_cli_sync()`](../cli.md), [`to_web()`](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental), [A2A](https://github.com/datalayer/fasta2a) (community) |
| Realtime voice | Yes (experimental) | Yes ([realtime](../realtime/overview.md)) |
| Agent graph | `@ai-sdk/workflow` (sibling) | [`pydantic-graph`](../graph.md) |
| Coding agent | `HarnessAgent` | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| Research agent | No | [`Researcher()`](https://pydantic.dev/docs/ai/harness/researcher/) |
| Web search | Native and third party | Native ([WebSearch](../capabilities/web-search.md)) and third party ([DuckDuckGo](../common-tools.md#duckduckgo-search-tool), [Tavily](../common-tools.md#tavily-search-tool), [You.com](https://pydantic.dev/docs/ai/harness/youdotcom/), [Exa](https://pydantic.dev/docs/ai/harness/exa-search/)) |
| Code sandboxes | `experimental_sandbox` (you host) | [Modal](https://pydantic.dev/docs/ai/harness/modal-sandbox/) |
| Image generation | Yes (`generateImage`) | Yes (OpenAI, Google, xAI) |
| Browser | No | Yes ([Browser Use](https://pydantic.dev/docs/ai/harness/browser-use/) and [Playwright](https://pydantic.dev/docs/ai/harness/playwright/)) |
| Structured output | Yes (`generateObject`) | Yes (type on the agent) |
| Human in the loop | Yes (approval in the browser) | Yes (tool approval) |
| Guardrails | Middleware | Yes ([harness](https://pydantic.dev/docs/ai/harness/guardrails/)) |
| Spend limits | No | Yes ([cost_limit](../agent.md#usage-limits), [spend](https://pydantic.dev/docs/ai/harness/spend/)) |
| MCP | Client | Client and [server](../mcp/server.md) |
| Memory | No | [Harness Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | `pruneMessages` | Yes ([compaction](../capabilities/compaction.md), [harness](https://pydantic.dev/docs/ai/harness/compaction/)) |
| Multi-agent | You compose it | [Delegation, graph, or `async`](../multi-agent-applications.md) |
| Durable execution | `WorkflowAgent` (`@ai-sdk/workflow`) | [Temporal, DBOS, Prefect, Restate](../durable_execution/overview.md), [Lambda](https://pydantic.dev/docs/ai/harness/aws-lambda/); Kitaru and Airflow (external) |
| Tracing | OTel, their endpoint | OpenTelemetry, any backend |
| Evals | No | Yes ([Pydantic Evals](../evals.md)) |
| Test without API keys | Yes (`MockLanguageModelV4`) | Yes ([TestModel](../testing.md), [FunctionModel](../testing.md)) |
| Embeddings | Yes (`embed`) | Yes |

Harness-linked cells ship in [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/), a separate package.

[Install Pydantic AI](../install.md).
