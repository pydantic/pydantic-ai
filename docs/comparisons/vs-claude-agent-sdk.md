# Pydantic AI vs Claude Agent SDK

The Claude Agent SDK is Claude Code as a library: it wraps the `claude` CLI (TypeScript) and you
configure that program in data. Pydantic AI is a native Python [`Agent`][pydantic_ai.Agent].
[`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) is the coding stack in your process, on any
model.

## Side by side

| | Claude Agent SDK | Pydantic AI |
|---|---|---|
| Native Python SDK | No, wraps the `claude` CLI | Yes |
| License | MIT | MIT |
| Model providers | Claude (Anthropic, Bedrock, Vertex, Foundry) | Any, plus [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] |
| Extensibility | Hooks, `allowed_tools` | [Capabilities](../extensibility.md) |
| Skills | Yes | Yes ([Skills](https://pydantic.dev/docs/ai/harness/skills/)) |
| On-demand capabilities | No | Yes ([on-demand](../capabilities/on-demand.md)) |
| Interfaces | CLI | [`to_cli_sync()`](../cli.md), [`to_web()`](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) |
| Realtime voice | No | Yes ([realtime](../realtime/overview.md)) |
| Agent graph | No | [`pydantic-graph`](../graph.md) |
| Coding agent | The CLI | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| Research agent | No | [`Researcher()`](https://pydantic.dev/docs/ai/harness/researcher/) |
| Web search | Yes | Yes ([WebSearch](../capabilities/web-search.md)) |
| Code sandboxes | CLI permissions | [Modal](https://pydantic.dev/docs/ai/harness/modal-sandbox/), [Monty](https://github.com/pydantic/monty), [Code Mode](https://pydantic.dev/docs/ai/harness/code-mode/) |
| Image generation | No | Yes (OpenAI, Google, xAI) |
| Browser | Yes (Chrome) | Yes ([Browser Use](https://pydantic.dev/docs/ai/harness/browser-use/) and [Playwright](https://pydantic.dev/docs/ai/harness/playwright/)) |
| Structured output | You configure the CLI | Yes (type on the agent) |
| Human in the loop | CLI permissions | Yes (tool approval) |
| Guardrails | CLI permissions | Yes ([harness](https://pydantic.dev/docs/ai/harness/guardrails/)) |
| Spend limits | `max_budget_usd` | Yes ([cost_limit](../agent.md#usage-limits), [spend](https://pydantic.dev/docs/ai/harness/spend/)) |
| MCP | Client | Client and [server](../mcp/server.md) |
| Memory | Sessions | [Harness Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes | Yes ([compaction](https://pydantic.dev/docs/ai/harness/compaction/)) |
| Multi-agent | CLI sub-agents | [Delegation, graph, or `async`](../multi-agent-applications.md) |
| Durable execution | Sessions | [Temporal, DBOS, Prefect, Restate, Lambda, Kitaru, Airflow](../durable_execution/overview.md) |
| Tracing | CLI telemetry | OpenTelemetry |
| Evals | No | Yes ([Pydantic Evals](../evals.md)) |
| Test without API keys | No (launch the CLI) | Yes |
| Embeddings | No | Yes |
| Deployment | Wherever `claude` runs | Anywhere |

## FAQ

**Is Pydantic AI a native Python SDK?** Yes. The Claude Agent SDK is a wrapper around a TypeScript
CLI.

**Can I get a coding agent without spawning `claude`?** Yes.
[`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) in your process, on any model, including
Claude.
