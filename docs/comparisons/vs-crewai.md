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
| Interfaces | Enterprise UI | [`to_cli_sync()`](../cli.md), [`to_web()`](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) |
| Realtime voice | No | Yes ([realtime](../realtime/overview.md)) |
| Agent graph | `Process` (sequential / hierarchical) | [`pydantic-graph`](../graph.md) |
| Coding agent | E2B tools | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| Research agent | No | [`Researcher()`](https://pydantic.dev/docs/ai/harness/researcher/) |
| Web search | Yes | Yes ([WebSearch](../capabilities/web-search.md)) |
| Code sandboxes | E2B | [Modal](https://pydantic.dev/docs/ai/harness/modal-sandbox/), [Monty](https://github.com/pydantic/monty), [Code Mode](https://pydantic.dev/docs/ai/harness/code-mode/) |
| Image generation | Yes (`DallETool`) | Yes (OpenAI, Google, xAI) |
| Browser | Yes (Browserbase) | Yes ([Browser Use](https://pydantic.dev/docs/ai/harness/browser-use/) and [Playwright](https://pydantic.dev/docs/ai/harness/playwright/)) |
| Structured output | Yes | Yes (type on the agent) |
| Human in the loop | Yes (human tools) | Yes (tool approval) |
| Guardrails | Yes (task guardrails) | Yes ([harness](https://pydantic.dev/docs/ai/harness/guardrails/)) |
| Spend limits | No | Yes ([cost_limit](../agent.md#usage-limits), [spend](https://pydantic.dev/docs/ai/harness/spend/)) |
| MCP | Client | Client and [server](../mcp/server.md) |
| Memory | On `Agent` and `Crew` | [Harness Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | No | Yes ([compaction](https://pydantic.dev/docs/ai/harness/compaction/)) |
| Multi-agent | Roles, tasks, `Process` | [Delegation, graph, or `async`](../multi-agent-applications.md) |
| Durable execution | `Crew.from_checkpoint` | [Temporal, DBOS, Prefect, Restate, Lambda, Kitaru, Airflow](../durable_execution/overview.md) |
| Tracing | Their platform | OpenTelemetry |
| Evals | Yes (experimental) | Yes ([Pydantic Evals](../evals.md)) |
| Test without API keys | No (`crewai test` hits a live model) | Yes |
| Embeddings | Knowledge | Yes |
| Deployment | CrewAI enterprise, or your app | Anywhere |

## FAQ

**How do I do multi-agent without a crew?** An agent as a tool, a router, `asyncio.gather`, or
[`pydantic-graph`](../graph.md).

**Can one of those agents be a coding agent?** Yes.
[`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) on that [`Agent`][pydantic_ai.Agent].
