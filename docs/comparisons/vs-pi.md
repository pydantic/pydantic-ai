# Pydantic AI vs Pi

Pi is a TypeScript coding agent you can embed without forking: extensions, skills, and `pi install`
packages. Pydantic AI is a typed Python [`Agent`][pydantic_ai.Agent].
[`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) is a capability on that same object.

## Side by side

| | Pi | Pydantic AI |
|---|---|---|
| Native Python SDK | No, TypeScript | Yes |
| License | MIT | MIT |
| Model providers | Many | Any, plus [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] |
| Extensibility | Extensions, skills, `pi install` | [Capabilities](../extensibility.md) |
| Skills | Yes (`pi install`) | Yes ([Skills](https://pydantic.dev/docs/ai/harness/skills/)) |
| Interfaces | CLI | [`to_cli_sync()`](../cli.md), [`to_web()`](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental), [A2A](https://github.com/datalayer/fasta2a) (community) |
| Realtime voice | No | Yes ([realtime](../realtime/overview.md)) |
| Agent graph | No | [`pydantic-graph`](../graph.md) |
| Coding agent | The product | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| Research agent | No | [`Researcher()`](https://pydantic.dev/docs/ai/harness/researcher/) |
| Web search | Third party | Native ([WebSearch](../capabilities/web-search.md)) and third party ([DuckDuckGo](../common-tools.md#duckduckgo-search-tool), [Tavily](../common-tools.md#tavily-search-tool), [You.com](https://pydantic.dev/docs/ai/harness/youdotcom/), [Exa](https://pydantic.dev/docs/ai/harness/exa-search/)) |
| Code sandboxes | Packages, not core | [Modal](https://pydantic.dev/docs/ai/harness/modal-sandbox/) |
| Image generation | No | Yes (OpenAI, Google, xAI) |
| Browser | No | Yes ([Browser Use](https://pydantic.dev/docs/ai/harness/browser-use/) and [Playwright](https://pydantic.dev/docs/ai/harness/playwright/)) |
| Structured output | Terminating tool you write | Yes (type on the agent) |
| Human in the loop | Packages, not core | Yes (tool approval) |
| Guardrails | No | Yes ([harness](https://pydantic.dev/docs/ai/harness/guardrails/)) |
| Spend limits | No | Yes ([cost_limit](../agent.md#usage-limits), [spend](https://pydantic.dev/docs/ai/harness/spend/)) |
| MCP | Packages, not core | Client and [server](../mcp/server.md) |
| Memory | Sessions | [Harness Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes | Yes ([compaction](../capabilities/compaction.md), [harness](https://pydantic.dev/docs/ai/harness/compaction/)) |
| Multi-agent | Packages | [Delegation, graph, or `async`](../multi-agent-applications.md) |
| Durable execution | Session tree on disk | [Temporal, DBOS, Prefect, Restate](../durable_execution/overview.md), [Lambda](https://pydantic.dev/docs/ai/harness/aws-lambda/); Kitaru and Airflow (external) |
| Tracing | Logs | OpenTelemetry, any backend |
| Evals | No | Yes ([Pydantic Evals](../evals.md)) |
| Test without API keys | Yes (`registerProvider()`) | Yes ([TestModel](../testing.md), [FunctionModel](../testing.md)) |
| Embeddings | No | Yes |

Harness-linked cells ship in [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/), a separate package.

## FAQ

**Can I build a coding agent like this in Python?** Yes. [Code Puppy](https://github.com/mpfaffenberger/code_puppy)
and [Pydantic Deep Agents](https://github.com/vstorm-co/pydantic-deepagents) (Vstorm) did.
[`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) if you want ours.

[Install Pydantic AI](../install.md).
