# Pydantic AI vs LiveKit Agents

LiveKit Agents is built around realtime voice transport: WebRTC rooms, telephony, turn detection, and plugins for STT, LLM and TTS vendors. In Pydantic AI, [realtime voice](../realtime/overview.md) is one interface on the same typed [`Agent`][pydantic_ai.Agent] that also runs headless, behind your API or in a [web chat](../web.md), with the [Harness SDK](https://pydantic.dev/docs/ai/harness/) available in every mode.

## Framework

| | LiveKit Agents | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Language | Python (also Node) | Python |
| License | Apache-2.0 | MIT |
| Model providers | Many (plugins) | [Many](../models/overview.md) |
| Extensibility | Pipeline nodes (`stt_node`, `llm_node`, …) | [Capabilities and toolsets](../extensibility.md); [30+ in the Harness SDK](https://pydantic.dev/docs/ai/harness/) |
| Harnesses | Build your own | Built-in [`Coder`](https://pydantic.dev/docs/ai/harness/coder/) and [`Researcher`](https://pydantic.dev/docs/ai/harness/researcher/), or compose your own |
| Observability | OpenTelemetry | [OpenTelemetry](../logfire.md#using-opentelemetry), including [Pydantic Logfire](https://pydantic.dev/logfire) |
| Interfaces | WebRTC rooms, telephony | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental) |
| Realtime voice | Yes | [Realtime](../realtime/overview.md) |
| Evals | Yes | [Pydantic Evals](../evals.md) |
| Durable execution | No | [5+ integrations](../durable_execution/overview.md) |
| Image generation | No | [Image Generation](../image-generation.md) |

## Features

| | LiveKit Agents | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Sub-agents | TBD-FACTCHECK | [Subagents](https://pydantic.dev/docs/ai/harness/subagents/), [delegation](../multi-agent-applications.md), or [`pydantic-graph`](../graph.md) |
| Planning | TBD-FACTCHECK | [Planning](https://pydantic.dev/docs/ai/harness/planning/) |
| Skills | TBD-FACTCHECK | [Skills](https://pydantic.dev/docs/ai/harness/skills/) |
| Memory | No | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes | [Compaction](../capabilities/compaction.md) |
| Guardrails | No | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Provider-hosted tools only | [Execution environments](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser use | Yes | [Web & research](https://pydantic.dev/docs/ai/harness/#web--research) |
