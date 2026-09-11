# Pydantic AI vs LiveKit Agents

LiveKit Agents is the realtime runtime: WebRTC rooms, telephony, turn detection, STT/LLM/TTS
pipelines. Pydantic AI is a typed [`Agent`][pydantic_ai.Agent] that also holds a
[spoken conversation](../realtime/overview.md). Same tools, dependencies, and observability as text.

Use LiveKit when the product is a room. Use Pydantic AI when the product is an agent that can also
speak.

## Side by side

| | LiveKit Agents | Pydantic AI |
|---|---|---|
| Native Python SDK | Yes (also Node) | Yes |
| License | Apache-2.0 | MIT |
| Model providers | Many (plugins) | Any, plus [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] |
| Extensibility | Pipeline nodes (`stt_node`, `llm_node`, …) | [Capabilities](../extensibility.md) |
| Skills | No | Yes ([Skills](https://pydantic.dev/docs/ai/harness/skills/)) |
| Interfaces | WebRTC rooms, telephony | [`to_cli_sync()`](../cli.md), [`to_web()`](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental), [A2A](https://github.com/datalayer/fasta2a) (community) |
| Realtime voice | The product (WebRTC, telephony) | Yes ([realtime](../realtime/overview.md)) |
| Agent graph | Media pipeline, not a workflow graph | [`pydantic-graph`](../graph.md) |
| Coding agent | No | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| Research agent | No | [`Researcher()`](https://pydantic.dev/docs/ai/harness/researcher/) |
| Web search | Native | Native ([WebSearch](../capabilities/web-search.md)) and third party ([DuckDuckGo](../common-tools.md#duckduckgo-search-tool), [Tavily](../common-tools.md#tavily-search-tool), [You.com](https://pydantic.dev/docs/ai/harness/youdotcom/), [Exa](https://pydantic.dev/docs/ai/harness/exa-search/)) |
| Code sandboxes | `CodeInterpreter` (provider) | [Modal](https://pydantic.dev/docs/ai/harness/modal-sandbox/) |
| Image generation | No | Yes (OpenAI, Google, xAI) |
| Browser | `ComputerUse` / `plugins.browser` | Yes ([Browser Use](https://pydantic.dev/docs/ai/harness/browser-use/) and [Playwright](https://pydantic.dev/docs/ai/harness/playwright/)) |
| Structured output | You wire `response_format` | Yes (type on the agent) |
| Human in the loop | Frontend tools | Yes (tool approval) |
| Guardrails | No | Yes ([harness](https://pydantic.dev/docs/ai/harness/guardrails/)) |
| Spend limits | No | Yes ([cost_limit](../agent.md#usage-limits), [spend](https://pydantic.dev/docs/ai/harness/spend/)) |
| MCP | Client | Client and [server](../mcp/server.md) |
| Memory | No | [Harness Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes (`truncate`) | Yes ([compaction](../capabilities/compaction.md), [harness](https://pydantic.dev/docs/ai/harness/compaction/)) |
| Multi-agent | Handoffs in a room | [Delegation, graph, or `async`](../multi-agent-applications.md) |
| Durable execution | Agent server orchestration | [Temporal, DBOS, Prefect, Restate](../durable_execution/overview.md), [Lambda](https://pydantic.dev/docs/ai/harness/aws-lambda/); Kitaru and Airflow (external) |
| Tracing | OTel, their endpoint | OpenTelemetry, any backend |
| Evals | Yes (`livekit.agents.evals`) | Yes ([Pydantic Evals](../evals.md)) |
| Test without API keys | Yes | Yes ([TestModel](../testing.md), [FunctionModel](../testing.md)) |
| Embeddings | No | Yes |

Harness-linked cells ship in [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/), a separate package.

[Install Pydantic AI](../install.md).
