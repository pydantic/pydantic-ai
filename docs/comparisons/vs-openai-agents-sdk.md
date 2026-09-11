# Pydantic AI vs OpenAI Agents SDK

The OpenAI Agents SDK is OpenAI's agent library: sessions, guardrails, handoffs, and new OpenAI
features first. Pydantic AI runs on any model, including OpenAI.

## Side by side

| | OpenAI Agents SDK | Pydantic AI |
|---|---|---|
| Language | Python | Python |
| License | MIT | MIT |
| Model providers | OpenAI first; others via LiteLLM | [Any](../models/overview.md), plus [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] |
| Durable execution | Temporal integration | Yes — [5+ integrations](../durable_execution/overview.md) |
| Observability | Their dashboard | OpenTelemetry, any backend including [Pydantic Logfire](../logfire.md) |
| Extensibility | Tools, guardrails, handoffs | [Capabilities](../extensibility.md) and [toolsets](../toolsets.md) |
| Interfaces | None built in | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental), [A2A](https://github.com/datalayer/fasta2a) (community) |
| Build your own harness | Yes | Yes |
| Coding harness | `SandboxAgent` | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| Graph library | No (handoffs only) | [`pydantic-graph`](../graph.md) |
| Multi-agent | Handoffs | [Sub-agents](../multi-agent-applications.md), graph, or `async` |
| Realtime voice | Yes (OpenAI Realtime) | [Realtime](../realtime/overview.md) |
| Image generation | Yes (OpenAI) | [Image generation](../image-generation.md) |

| | OpenAI Agents SDK | Pydantic AI |
|---|---|---|
| Structured output | Yes | [Type on the agent](../output.md) |
| Guardrails | Input, output, tool tripwires | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Hosted tools plus sandbox clients | [Yes](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser | `ComputerTool` (you host) | [Yes](https://pydantic.dev/docs/ai/harness/#web-research) |
| Memory | Sessions | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes | [Compaction](../capabilities/compaction.md) |
| Evals | Yes (platform) | [Pydantic Evals](../evals.md) |
| Test without API keys | Yes (`ScriptedModel`) | [`TestModel`](../testing.md), [`FunctionModel`](../testing.md) |

Cells linked to pydantic.dev/docs/ai/harness ship in [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/),
a separate package.

[Install Pydantic AI](../install.md).
