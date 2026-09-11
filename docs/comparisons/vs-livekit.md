# Pydantic AI vs LiveKit Agents

LiveKit Agents is the realtime runtime: WebRTC rooms, telephony, turn detection, STT/LLM/TTS
pipelines. Pydantic AI is a typed [`Agent`][pydantic_ai.Agent] that also holds a
[spoken conversation](../realtime/overview.md). Same tools, dependencies, and observability as text.

Use LiveKit when the product is a room. Use Pydantic AI when the product is an agent that can also
speak.

## Side by side

| | LiveKit Agents | Pydantic AI |
|---|---|---|
| Language | Python (also Node) | Python |
| Model providers | Many (plugins) | [Multiple](../models/overview.md) |
| Durable execution | Agent server orchestration | Yes — [5+ integrations](../durable_execution/overview.md) |
| Observability | OpenTelemetry | [OpenTelemetry](../capabilities/instrumentation.md) |
| Extensibility | Pipeline nodes (`stt_node`, `llm_node`, …) | [Capabilities and toolsets](../extensibility.md) |
| Interfaces | WebRTC rooms, telephony | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental), A2A |
| Build your own harness | Yes | [Yes](https://pydantic.dev/docs/ai/harness/#build-your-own) |
| Coding harness | No | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| Graph library | No (media pipelines) | [`pydantic-graph`](../graph.md) |
| Multi-agent | Handoffs in a room | [Sub-agents](../multi-agent-applications.md), [graph](../graph.md), or `async` |
| Realtime voice | The product (WebRTC, telephony) | [Realtime](../realtime/overview.md) |
| Image generation | No | [Image generation](../image-generation.md) |
| License | Apache-2.0 | MIT |

| | LiveKit Agents | Pydantic AI |
|---|---|---|
| Structured output | You wire `response_format` | [Type on the agent](../output.md) |
| Guardrails | No | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | `CodeInterpreter` (provider) | [Yes](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser | `ComputerUse` / `plugins.browser` | [Yes](https://pydantic.dev/docs/ai/harness/#web-research) |
| Memory | No | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes (`truncate`) | [Compaction](../capabilities/compaction.md) |
| Evals | Yes (`livekit.agents.evals`) | [Pydantic Evals](../evals.md) |
| Test without API keys | Yes | [`TestModel`](../testing.md), [`FunctionModel`](../testing.md) |

Cells linked to pydantic.dev/docs/ai/harness ship in [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/),
a separate package.

[Install Pydantic AI](../install.md).
