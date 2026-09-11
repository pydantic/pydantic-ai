# Pydantic AI vs Google ADK

Google ADK is the Gemini-native kit: `LlmAgent`, a `Runner`, Vertex, Search, A2A, a web UI. Pydantic
AI isn't tied to a cloud.

## Side by side

| | Google ADK | Pydantic AI |
|---|---|---|
| Native Python SDK | Yes | Yes |
| License | Apache-2.0 | MIT |
| Model providers | Gemini first (`LiteLlm`, `AnthropicLlm` exist) | Any, plus [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] |
| Extensibility | Tools, plugins | [Capabilities](../extensibility.md) |
| Skills | Yes | Yes ([Skills](https://pydantic.dev/docs/ai/harness/skills/)) |
| Interfaces | CLI, web, A2A | [`to_cli_sync()`](../cli.md), [`to_web()`](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental), [A2A](https://github.com/datalayer/fasta2a) (community) |
| Realtime voice | Yes (Gemini Live) | Yes ([realtime](../realtime/overview.md)) |
| Agent graph | `SequentialAgent`, `LoopAgent`, `ParallelAgent` | [`pydantic-graph`](../graph.md) |
| Coding agent | `AntigravityAgent` (experimental) | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| Research agent | No | [`Researcher()`](https://pydantic.dev/docs/ai/harness/researcher/) |
| Web search | Native | Native ([WebSearch](../capabilities/web-search.md)) and third party ([DuckDuckGo](../common-tools.md#duckduckgo-search-tool), [Tavily](../common-tools.md#tavily-search-tool), [You.com](https://pydantic.dev/docs/ai/harness/youdotcom/), [Exa](https://pydantic.dev/docs/ai/harness/exa-search/)) |
| Code sandboxes | Gemini code execution | [Modal](https://pydantic.dev/docs/ai/harness/modal-sandbox/) |
| Image generation | No | Yes (OpenAI, Google, xAI) |
| Browser | Yes (Computer Use) | Yes ([Browser Use](https://pydantic.dev/docs/ai/harness/browser-use/) and [Playwright](https://pydantic.dev/docs/ai/harness/playwright/)) |
| Structured output | Yes | Yes (type on the agent) |
| Human in the loop | Yes (confirmation / long-running tools) | Yes (tool approval) |
| Guardrails | Callbacks | Yes ([harness](https://pydantic.dev/docs/ai/harness/guardrails/)) |
| Spend limits | No | Yes ([cost_limit](../agent.md#usage-limits), [spend](https://pydantic.dev/docs/ai/harness/spend/)) |
| MCP | Client and server | Client and [server](../mcp/server.md) |
| Memory | App / user / invocation state | [Harness Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | `EventsCompactionConfig` | Yes ([compaction](../capabilities/compaction.md), [harness](https://pydantic.dev/docs/ai/harness/compaction/)) |
| Multi-agent | `LoopAgent`, `ParallelAgent` | [Delegation, graph, or `async`](../multi-agent-applications.md) |
| Durable execution | Vertex | [Temporal, DBOS, Prefect, Restate](../durable_execution/overview.md), [Lambda](https://pydantic.dev/docs/ai/harness/aws-lambda/); Kitaru and Airflow (external) |
| Tracing | OTel, their endpoint | OpenTelemetry, any backend |
| Evals | Yes (Vertex) | Yes ([Pydantic Evals](../evals.md)) |
| Test without API keys | Yes (subclass `BaseLlm`) | Yes ([TestModel](../testing.md), [FunctionModel](../testing.md)) |
| Embeddings | Yes (Vertex) | Yes |

Harness-linked cells ship in [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/), a separate package.

[Install Pydantic AI](../install.md).
