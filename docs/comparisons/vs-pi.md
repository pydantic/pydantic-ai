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
| Interfaces | CLI | [`to_cli_sync()`](../cli.md), [`to_web()`](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/), [A2A extra](https://github.com/datalayer/fasta2a) |
| Realtime voice | No | Yes ([realtime](../realtime/overview.md)) |
| Agent graph | No | [`pydantic-graph`](../graph.md) |
| Coding agent | The product | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| Research agent | No | [`Researcher()`](https://pydantic.dev/docs/ai/harness/researcher/) |
| Web search | No | Yes ([WebSearch](../capabilities/web-search.md)) |
| Code sandboxes | No | [Modal](https://pydantic.dev/docs/ai/harness/modal-sandbox/), [Monty](https://github.com/pydantic/monty), [Code Mode](https://pydantic.dev/docs/ai/harness/code-mode/) |
| Image generation | No | Yes (OpenAI, Google, xAI) |
| Browser | No | Yes ([Browser Use](https://pydantic.dev/docs/ai/harness/browser-use/) and [Playwright](https://pydantic.dev/docs/ai/harness/playwright/)) |
| Structured output | Terminating tool you write | Yes (type on the agent) |
| Human in the loop | No | Yes (tool approval) |
| Guardrails | No | Yes ([harness](https://pydantic.dev/docs/ai/harness/guardrails/)) |
| Spend limits | No | Yes ([cost_limit](../agent.md#usage-limits), [spend](https://pydantic.dev/docs/ai/harness/spend/)) |
| MCP | Packages, not core | Client and [server](../mcp/server.md) |
| Memory | Sessions | [Harness Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes | Yes ([compaction](https://pydantic.dev/docs/ai/harness/compaction/)) |
| Multi-agent | Packages | [Delegation, graph, or `async`](../multi-agent-applications.md) |
| Durable execution | Session tree on disk | [Temporal, DBOS, Prefect, Restate, Lambda, Kitaru, Airflow](../durable_execution/overview.md) |
| Tracing | Logs | OpenTelemetry |
| Evals | No | Yes ([Pydantic Evals](../evals.md)) |
| Test without API keys | Yes (stub `streamFn`) | Yes |
| Embeddings | No | Yes |

## FAQ

**Can I code like Pi?** Yes. [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) on a typed
[`Agent`][pydantic_ai.Agent]. Files, shell, planning, and sub-agents.

**Can that coding agent live in the Python app I already have?** Yes. Types, tests, and Temporal stay
attached.
