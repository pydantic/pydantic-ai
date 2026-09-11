# Pydantic AI vs LiveKit Agents

LiveKit Agents is built around realtime voice transport: WebRTC rooms, telephony, turn detection, and plugins for STT, LLM and TTS vendors. Pydantic AI vs LiveKit Agents is a question of scope: [voice](../realtime/overview.md) is one frontend on the same typed agent that also runs headless or behind your API.

## Framework

| | LiveKit Agents | Pydantic AI and [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Language | Python (also Node) | Python |
| License | Apache-2.0 | MIT |
| Model providers | Many (plugins) | [Many](../models/overview.md) |
| Extensibility | Pipeline nodes (`stt_node`, `llm_node`, …) | [Capabilities and toolsets](../extensibility.md) |
| Build a custom harness | Yes | [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) |
| Coding harness | No | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| Interfaces | WebRTC rooms, telephony | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental) |
| Realtime voice | Yes | [Realtime](../realtime/overview.md) |
| Image generation | No | [Image Generation](../image-generation.md) |
| Durable execution | No | [5+ integrations](../durable_execution/overview.md) |
| Observability | OpenTelemetry | [OpenTelemetry](../capabilities/instrumentation.md), including [Pydantic Logfire](../logfire.md) |
| Evals | Yes | [Pydantic Evals](../evals.md) |

## Features

| | LiveKit Agents | Pydantic AI and [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Structured output | Yes | [Structured output](../output.md#structured-output) |
| Multi-agent | Handoffs in a room | [Delegation (tools or `SubAgents`), hand-off in your code, or graph](../multi-agent-applications.md) |
| Graph library | No | [`pydantic-graph`](../graph.md) |
| Memory | No | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes | [Compaction](../capabilities/compaction.md) |
| Guardrails | No | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Via provider tools | [Execution environments](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser use | Yes | [Web & research](https://pydantic.dev/docs/ai/harness/#web--research) |
