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
| Interfaces | Playground, Studio | [`to_cli_sync()`](../cli.md), [`to_web()`](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) |
| Realtime voice | Voice extras | Yes ([realtime](../realtime/overview.md)) |
| Agent graph | Workflows in core | [`pydantic-graph`](../graph.md) |
| Coding agent | No | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| Research agent | No | [`Researcher()`](https://pydantic.dev/docs/ai/harness/researcher/) |
| Web search | Yes | Yes ([WebSearch](../capabilities/web-search.md)) |
| Code sandboxes | No | [Modal](https://pydantic.dev/docs/ai/harness/modal-sandbox/), [Monty](https://github.com/pydantic/monty), [Code Mode](https://pydantic.dev/docs/ai/harness/code-mode/) |
| Image generation | No | Yes (OpenAI, Google, xAI) |
| Browser | Yes (AgentBrowser, Stagehand) | Yes ([Browser Use](https://pydantic.dev/docs/ai/harness/browser-use/) and [Playwright](https://pydantic.dev/docs/ai/harness/playwright/)) |
| Structured output | Yes | Yes (type on the agent) |
| Human in the loop | Yes (processors / approval) | Yes (tool approval) |
| Guardrails | Processors | Yes ([harness](https://pydantic.dev/docs/ai/harness/guardrails/)) |
| Spend limits | Metrics, not a ceiling | Yes ([cost_limit](../agent.md#usage-limits), [spend](https://pydantic.dev/docs/ai/harness/spend/)) |
| MCP | Client | Client and [server](../mcp/server.md) |
| Memory | Yes (working, observational, semantic) | Yes ([harness Memory](https://pydantic.dev/docs/ai/harness/memory/); thinner) |
| Compaction | No | Yes ([compaction](https://pydantic.dev/docs/ai/harness/compaction/)) |
| Multi-agent | Workflows | [Delegation, graph, or `async`](../multi-agent-applications.md) |
| Durable execution | `createDurableAgent()`, Inngest | [Temporal, DBOS, Prefect, Restate, Lambda, Kitaru, Airflow](../durable_execution/overview.md) |
| Tracing | Studio, Cloud | OpenTelemetry |
| Evals | Yes | Yes ([Pydantic Evals](../evals.md)) |
| Test without API keys | Yes | Yes |
| Embeddings | Yes (with memory) | Yes |

## FAQ

**Can the UI stay in TypeScript?** Yes. A Python agent behind HTTP. UI adapters exist.

**Can I get a chat UI without Studio?** Yes. [`to_web()`][pydantic_ai.agent.Agent.to_web].
