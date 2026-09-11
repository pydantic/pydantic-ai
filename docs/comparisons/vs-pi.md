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
| On-demand capabilities | No | Yes ([on-demand](../capabilities/on-demand.md)) |
| Interfaces | CLI | [`to_cli_sync()`](../cli.md), [`to_web()`](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) |
| Realtime voice | No | Yes ([realtime](../realtime/overview.md)) |
| Agent graph | No | [`pydantic-graph`](../graph.md) |
| Coding agent | The product | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| Research agent | No | [`Researcher()`](https://pydantic.dev/docs/ai/harness/researcher/) |
| Code sandboxes | No | [Modal](https://pydantic.dev/docs/ai/harness/modal-sandbox/) and [Monty](https://github.com/pydantic/monty) |
| Image generation | No | Yes |
| Browser | No | Yes ([Browser Use](https://pydantic.dev/docs/ai/harness/browser-use/)) |
| Structured output | Terminating tool you write | Yes (type on the agent) |
| Human in the loop | No | Yes (tool approval) |
| Guardrails | No | Yes ([harness](https://pydantic.dev/docs/ai/harness/guardrails/)) |
| MCP | Packages, not core | Client and [server](../mcp/server.md) |
| Memory | Sessions | [Harness Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Multi-agent | Packages | [Delegation, graph, or `async`](../multi-agent-applications.md) |
| Durable execution | Session tree on disk | [Temporal, DBOS, Prefect, Restate, Lambda, Kitaru, Airflow](../durable_execution/overview.md) |
| Tracing | Logs | OpenTelemetry |
| Evals | No | Yes ([Pydantic Evals](../evals.md)) |
| Test without API keys | Yes (stub `streamFn`) | Yes |
| Embeddings | No | Yes |
| Deployment | Wherever Node runs | Anywhere |

## FAQ

**Can I code like Pi?** Yes. [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) on a typed
[`Agent`][pydantic_ai.Agent]. Files, shell, planning, and sub-agents.

**Can that coding agent live in the Python app I already have?** Yes. Types, tests, and Temporal stay
attached.
