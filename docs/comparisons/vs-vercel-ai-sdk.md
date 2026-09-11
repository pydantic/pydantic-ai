# Pydantic AI vs Vercel AI SDK

The Vercel AI SDK is the wire to a React UI: streaming, tool cards, approval in the browser. Pydantic
AI is a Python agent. [`VercelAIAdapter`][pydantic_ai.ui.vercel_ai.VercelAIAdapter] speaks their
protocol, so the browser can stay theirs.

## Side by side

| | Vercel AI SDK | Pydantic AI |
|---|---|---|
| Language | TypeScript | Python |
| License | Apache-2.0 | MIT |
| Model providers | Many | [Any](../models/overview.md), plus [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] |
| Durable execution | `WorkflowAgent` (`@ai-sdk/workflow`) | Yes — [5+ integrations](../durable_execution/overview.md) |
| Observability | OTel, their endpoint | [OpenTelemetry](../capabilities/instrumentation.md), any backend including [Pydantic Logfire](../logfire.md) |
| Extensibility | Middleware, tools | [Capabilities](../extensibility.md) and [toolsets](../toolsets.md) |
| Interfaces | React chat UI (`useChat`), AI stream | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental), [A2A](https://github.com/datalayer/fasta2a) (community) |
| Build your own harness | Yes | [Yes](https://pydantic.dev/docs/ai/harness/#build-your-own) |
| Coding harness | `HarnessAgent` | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| Graph library | `@ai-sdk/workflow` (sibling) | [`pydantic-graph`](../graph.md) |
| Multi-agent | You compose it | [Sub-agents](../multi-agent-applications.md), [graph](../graph.md), or `async` |
| Realtime voice | Yes (experimental) | [Realtime](../realtime/overview.md) |
| Image generation | Yes (`generateImage`) | [Image generation](../image-generation.md) |

| | Vercel AI SDK | Pydantic AI |
|---|---|---|
| Structured output | Yes (`generateObject`) | [Type on the agent](../output.md) |
| Guardrails | Middleware | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | `experimental_sandbox` (you host) | [Yes](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser | No | [Yes](https://pydantic.dev/docs/ai/harness/#web-research) |
| Memory | No | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | `pruneMessages` | [Compaction](../capabilities/compaction.md) |
| Evals | No | [Pydantic Evals](../evals.md) |
| Test without API keys | Yes (`MockLanguageModelV4`) | [`TestModel`](../testing.md), [`FunctionModel`](../testing.md) |

Cells linked to pydantic.dev/docs/ai/harness ship in [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/),
a separate package.

[Install Pydantic AI](../install.md).
