# Pydantic AI vs Google ADK

Google ADK is the Gemini-native kit: `LlmAgent`, a `Runner`, Vertex, Search, A2A, a web UI. Pydantic
AI isn't tied to a cloud.

## Side by side

| | Google ADK | Pydantic AI |
|---|---|---|
| Language | Python | Python |
| Model providers | Gemini first (`LiteLlm`, `AnthropicLlm` exist) | [Any](../models/overview.md), plus [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] |
| Durable execution | Vertex | Yes — [5+ integrations](../durable_execution/overview.md) |
| Observability | OTel, their endpoint | [OpenTelemetry](../capabilities/instrumentation.md), any backend including [Pydantic Logfire](../logfire.md) |
| Extensibility | Tools, plugins | [Capabilities](../extensibility.md) and [toolsets](../toolsets.md) |
| Interfaces | CLI, web, A2A | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental), [A2A](https://github.com/datalayer/fasta2a) (community) |
| Build your own harness | Yes | [Yes](https://pydantic.dev/docs/ai/harness/#build-your-own) |
| Coding harness | `AntigravityAgent` (experimental) | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| Graph library | `SequentialAgent`, `LoopAgent`, `ParallelAgent` | [`pydantic-graph`](../graph.md) |
| Multi-agent | `LoopAgent`, `ParallelAgent` | [Sub-agents](../multi-agent-applications.md), [graph](../graph.md), or `async` |
| Realtime voice | Yes (Gemini Live) | [Realtime](../realtime/overview.md) |
| Image generation | No | [Image generation](../image-generation.md) |
| License | Apache-2.0 | MIT |

| | Google ADK | Pydantic AI |
|---|---|---|
| Structured output | Yes | [Type on the agent](../output.md) |
| Guardrails | Callbacks | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Gemini code execution | [Yes](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser | Yes (Computer Use) | [Yes](https://pydantic.dev/docs/ai/harness/#web-research) |
| Memory | App / user / invocation state | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | `EventsCompactionConfig` | [Compaction](../capabilities/compaction.md) |
| Evals | Yes (Vertex) | [Pydantic Evals](../evals.md) |
| Test without API keys | Yes (subclass `BaseLlm`) | [`TestModel`](../testing.md), [`FunctionModel`](../testing.md) |

Cells linked to pydantic.dev/docs/ai/harness ship in [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/),
a separate package.

[Install Pydantic AI](../install.md).
