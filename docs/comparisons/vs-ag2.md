# Pydantic AI vs AG2

AG2 1.0 is a rewrite. The AutoGen module is gone (`import autogen` fails). What replaced it is close
to us: typed `Agent`, `AgentSpec`, `Inject`/`Depends`. Durability is the fork: their `Task` checkpoint
store, or the same agent inside [Temporal, DBOS, Prefect, Restate, Lambda, Kitaru, or Airflow](../durable_execution/overview.md).

## Side by side

| | AG2 | Pydantic AI |
|---|---|---|
| Native Python SDK | Yes | Yes |
| License | Apache-2.0 | MIT |
| Model providers | Many | Any, plus [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] |
| Extensibility | Tools, `Inject` / `Depends` | [Capabilities](../extensibility.md) |
| Skills | Yes | Yes ([Skills](https://pydantic.dev/docs/ai/harness/skills/)) |
| Interfaces | A2A, ACP extra | [`to_cli_sync()`](../cli.md), [`to_web()`](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) |
| Realtime voice | No | Yes ([realtime](../realtime/overview.md)) |
| Agent graph | Group chat / swarm | [`pydantic-graph`](../graph.md) |
| Coding agent | No | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| Research agent | No | [`Researcher()`](https://pydantic.dev/docs/ai/harness/researcher/) |
| Web search | Yes | Yes ([WebSearch](../capabilities/web-search.md)) |
| Code sandboxes | Daytona, Docker | [Modal](https://pydantic.dev/docs/ai/harness/modal-sandbox/), [Monty](https://github.com/pydantic/monty), [Code Mode](https://pydantic.dev/docs/ai/harness/code-mode/) |
| Image generation | Yes (OpenAI, Gemini) | Yes (OpenAI, Google, xAI) |
| Browser | No | Yes ([Browser Use](https://pydantic.dev/docs/ai/harness/browser-use/) and [Playwright](https://pydantic.dev/docs/ai/harness/playwright/)) |
| Structured output | Yes | Yes (type on the agent) |
| Human in the loop | Yes (human input) | Yes (tool approval) |
| Guardrails | No | Yes ([harness](https://pydantic.dev/docs/ai/harness/guardrails/)) |
| Spend limits | No | Yes ([cost_limit](../agent.md#usage-limits), [spend](https://pydantic.dev/docs/ai/harness/spend/)) |
| MCP | Client (extra) | Client and [server](../mcp/server.md) |
| Memory | No | [Harness Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | No | Yes ([compaction](https://pydantic.dev/docs/ai/harness/compaction/)) |
| Multi-agent | Group chat, swarm | [Delegation, graph, or `async`](../multi-agent-applications.md) |
| Durable execution | `Task` checkpoint store | [Temporal, DBOS, Prefect, Restate, Lambda, Kitaru, Airflow](../durable_execution/overview.md) |
| Tracing | Logs | OpenTelemetry |
| Evals | No | Yes ([Pydantic Evals](../evals.md)) |
| Test without API keys | Yes | Yes |
| Embeddings | No | Yes |
| Agent as data | `AgentSpec` | YAML, templates checked when you construct |

## FAQ

**Can I ship an agent as YAML?** Yes. Templates are checked when you construct, so a typo fails before
a customer hits it.

**If the process dies?** The same [`Agent`][pydantic_ai.Agent], inside [Temporal, DBOS, Prefect, Restate, Lambda, Kitaru, or Airflow](../durable_execution/overview.md).
