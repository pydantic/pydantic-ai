# Pydantic AI vs OpenAI Agents SDK

The OpenAI Agents SDK is OpenAI's agent library: sessions, guardrails, handoffs, and new OpenAI
features first. Pydantic AI runs on any model, including OpenAI.

## Side by side

| | OpenAI Agents SDK | Pydantic AI |
|---|---|---|
| Native Python SDK | Yes | Yes |
| License | MIT | MIT |
| Model providers | OpenAI first; others via LiteLLM | Any, plus [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] |
| Extensibility | Tools, guardrails, handoffs | [Capabilities](../extensibility.md) |
| Skills | Yes (sandbox) | Yes ([Skills](https://pydantic.dev/docs/ai/harness/skills/)) |
| Interfaces | None of these | [`to_cli_sync()`](../cli.md), [`to_web()`](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/), [A2A extra](https://github.com/datalayer/fasta2a) |
| Realtime voice | Yes (OpenAI Realtime) | Yes ([realtime](../realtime/overview.md)) |
| Agent graph | No (handoffs only) | [`pydantic-graph`](../graph.md) |
| Coding agent | `SandboxAgent` | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| Research agent | No | [`Researcher()`](https://pydantic.dev/docs/ai/harness/researcher/) |
| Web search | Yes (hosted) | Yes ([WebSearch](../capabilities/web-search.md)) |
| Code sandboxes | Hosted tools plus sandbox clients | [Modal](https://pydantic.dev/docs/ai/harness/modal-sandbox/), [Monty](https://github.com/pydantic/monty), [Code Mode](https://pydantic.dev/docs/ai/harness/code-mode/) |
| Image generation | Yes (OpenAI) | Yes (OpenAI, Google, xAI) |
| Browser | `ComputerTool` (you host) | Yes ([Browser Use](https://pydantic.dev/docs/ai/harness/browser-use/) and [Playwright](https://pydantic.dev/docs/ai/harness/playwright/)) |
| Structured output | Yes | Yes (type on the agent) |
| Human in the loop | Yes (guardrails, approvals) | Yes (tool approval) |
| Guardrails | Input, output, tool tripwires | Yes ([harness](https://pydantic.dev/docs/ai/harness/guardrails/)) |
| Spend limits | No | Yes ([cost_limit](../agent.md#usage-limits), [spend](https://pydantic.dev/docs/ai/harness/spend/)) |
| MCP | Client | Client and [server](../mcp/server.md) |
| Memory | Sessions | [Harness Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes | Yes ([compaction](https://pydantic.dev/docs/ai/harness/compaction/)) |
| Multi-agent | Handoffs | [Delegation, graph, or `async`](../multi-agent-applications.md) |
| Durable execution | Temporal integration | [Temporal, DBOS, Prefect, Restate, Lambda, Kitaru, Airflow](../durable_execution/overview.md) |
| Tracing | Their dashboard | OpenTelemetry |
| Evals | Yes (platform) | Yes ([Pydantic Evals](../evals.md)) |
| Test without API keys | Yes (`ScriptedModel`) | Yes |
| Embeddings | Yes (OpenAI) | Yes |

## FAQ

**Can I keep using OpenAI models?** Yes, including the Responses API, and any other provider with
[`FallbackModel`][pydantic_ai.models.fallback.FallbackModel].

**Can I attach a coding harness?** Yes. [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) on
the same [`Agent`][pydantic_ai.Agent].
