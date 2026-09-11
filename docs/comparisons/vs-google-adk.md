# Pydantic AI vs Google ADK

Google ADK is the Gemini-native kit: `LlmAgent`, a `Runner`, Vertex, Search, A2A, a web UI. Pydantic
AI isn't tied to a cloud.

## Side by side

| | Google ADK | Pydantic AI |
|---|---|---|
| Language | Python | Python |
| Model providers | Gemini first (`LiteLlm`, `AnthropicLlm` exist) | [Multiple](../models/overview.md) |
| Durable execution | Yes (Vertex) | Yes — [5+ integrations](../durable_execution/overview.md) |
| Observability | OpenTelemetry | [OpenTelemetry](../capabilities/instrumentation.md) |
| Extensibility | Tools, plugins | [Capabilities and toolsets](../extensibility.md) |
| Interfaces | CLI, web, A2A | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental), A2A |
| Build your own harness | Yes | [Yes](https://pydantic.dev/docs/ai/harness/#build-your-own) |
| Coding harness | Yes (`AntigravityAgent`, experimental) | Yes ([`Coder()`](https://pydantic.dev/docs/ai/harness/coder/)) |
| Graph library | Yes (`SequentialAgent`, `LoopAgent`, `ParallelAgent`) | Yes ([`pydantic-graph`](../graph.md)) |
| Multi-agent | Sub-agents, transfer | [Sub-agents](../multi-agent-applications.md), [hand-offs](../multi-agent-applications.md#programmatic-agent-hand-off), or [graph](../graph.md) |
| Realtime voice | Yes (Gemini Live) | [Yes](../realtime/overview.md) |
| Image generation | No | [Yes](../image-generation.md) |
| License | Apache-2.0 | MIT |

| | Google ADK | Pydantic AI |
|---|---|---|
| Structured output | Yes | [Yes](../output.md) |
| Guardrails | Yes (callbacks) | [Yes](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Yes (Gemini code execution) | [Yes](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser | Yes (Computer Use) | [Yes](https://pydantic.dev/docs/ai/harness/#web-research) |
| Memory | Yes (state, `MemoryService`) | [Yes](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes (`EventsCompactionConfig`) | [Yes](../capabilities/compaction.md) |
| Evals | Yes (`adk eval`, Vertex) | [Yes](../evals.md) |
| Test without API keys | Yes (subclass `BaseLlm`) | [Yes](../testing.md) |

Cells linked to pydantic.dev/docs/ai/harness ship in [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/),
a separate package.

[Install Pydantic AI](../install.md).
