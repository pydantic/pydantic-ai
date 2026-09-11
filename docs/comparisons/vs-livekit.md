# Pydantic AI vs LiveKit Agents

LiveKit Agents is the reference stack for realtime voice: WebRTC rooms, telephony, turn detection, and plugins for every STT, LLM and TTS vendor. If you are shipping a phone or voice product, that transport layer is the point. Pydantic AI vs LiveKit Agents is a question of scope: [voice](../realtime/overview.md) is one frontend on the same typed agent that also runs headless or behind your API.

| | LiveKit Agents | Pydantic AI |
|---|---|---|
| Language | Python (also Node) | Python |
| Model providers | Multiple (plugins) | [Multiple](../models/overview.md) |
| Durable execution | No | [5+ integrations](../durable_execution/overview.md) |
| Observability | OpenTelemetry | [OpenTelemetry](../capabilities/instrumentation.md), including [Pydantic Logfire](../logfire.md) |
| Interfaces | WebRTC rooms, telephony | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental) |
| Extensibility | Pipeline nodes (`stt_node`, `llm_node`, …) | [Capabilities and toolsets](../extensibility.md) |
| Realtime voice | Yes | [Realtime](../realtime/overview.md) |
| Image generation | No | [Image Generation](../image-generation.md) |
| Multi-agent | Handoffs in a room | [Delegation (tools or `SubAgents`), hand-off in your code, or graph](../multi-agent-applications.md) |
| Graph library | No | [`pydantic-graph`](../graph.md) |
| Build your own harness | Yes | [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) |
| Coding harness | No | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| License | Apache-2.0 | MIT |

| | LiveKit Agents | Pydantic AI |
|---|---|---|
| Structured output | Yes | [Structured output](../output.md#structured-output) |
| Memory | No | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes | [Compaction](../capabilities/compaction.md) |
| Guardrails | No | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Via provider tools | [Execution environments](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser | Yes | [Web & research](https://pydantic.dev/docs/ai/harness/#web--research) |
| Evals | Yes | [Pydantic Evals](../evals.md) |
