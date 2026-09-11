# Pydantic AI vs LiveKit Agents

LiveKit Agents is a voice-agent framework built on LiveKit's WebRTC transport: rooms, SIP telephony, turn detection, noise cancellation, mid-call handoffs, and plugins for STT, LLM and TTS vendors, with a cascaded pipeline as the default. Pydantic AI's [realtime support](../realtime/overview.md) is a speech-to-speech agent loop on four providers behind one API, and it is the same typed [`Agent`][pydantic_ai.Agent] that runs as text, in a [web chat](../web.md) or behind your API: the call uses the same tools, dependencies and [capabilities](../realtime/capabilities.md), becomes ordinary message history you can [hand to a text agent](../realtime/history.md#handing-off-to-a-text-agent) for structured output, and is traced end to end in [Logfire](https://pydantic.dev/logfire). You bring the transport; with LiveKit, the transport is the product.

## Framework

| | LiveKit Agents | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Language | Python (also Node) | Python |
| License | Apache-2.0 | MIT |
| Model providers | Many (plugins) | [Many](../models/overview.md) |
| Extensibility | Pipeline nodes (`stt_node`, `llm_node`, …) | [Capabilities and toolsets](../extensibility.md); [50+ with the Harness SDK](https://pydantic.dev/docs/ai/harness/) |
| Harnesses | Build your own | Built-in [`Coder`](https://pydantic.dev/docs/ai/harness/coder/) and [`Researcher`](https://pydantic.dev/docs/ai/harness/researcher/), or compose your own |
| Observability | OpenTelemetry | [OpenTelemetry](../logfire.md#using-opentelemetry), including [Pydantic Logfire](https://pydantic.dev/logfire) |
| Interfaces | WebRTC rooms, telephony, text sessions | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental) |
| Realtime voice | Speech-to-speech and cascaded STT + LLM + TTS | [Speech-to-speech](../realtime/overview.md), four providers |
| Evals | Yes | [Pydantic Evals](../evals.md) |
| Durable execution | No | [5+ integrations](../durable_execution/overview.md) |
| Image generation | No | [Image Generation](../image-generation.md) |

## Features

| | LiveKit Agents | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Sub-agents | Yes | [Subagents](https://pydantic.dev/docs/ai/harness/subagents/), [delegation](../multi-agent-applications.md), or [`pydantic-graph`](../graph.md) |
| Planning | No | [Planning](https://pydantic.dev/docs/ai/harness/planning/) |
| Skills | No | [Skills](https://pydantic.dev/docs/ai/harness/skills/) |
| Memory | No | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Truncation only | [Compaction](../capabilities/compaction.md) |
| Guardrails | No | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | No | [Execution environments](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser use | No | [Web & research](https://pydantic.dev/docs/ai/harness/#web--research) |

## Realtime, side by side

Our realtime support means speech-to-speech models: one persistent connection, audio in and audio out, on the four providers below. LiveKit also runs the cascaded pipeline, which we do not. If you are choosing a voice stack, these are the rows that decide it:

| | LiveKit Agents | Pydantic AI |
|---|---|---|
| Speech-to-speech providers | Nine plugins | [Four](../realtime/overview.md#provider-support) behind one API: OpenAI, Azure OpenAI, Gemini Live, xAI; ElevenLabs in [#7964](https://github.com/pydantic/pydantic-ai/pull/7964) |
| Cascaded STT + LLM + TTS | Yes, the default; dozens of STT and TTS plugins | Not built in; [compose it yourself](../realtime/overview.md#other-ways-to-build-voice) around a text agent |
| Audio transport | WebRTC rooms via LiveKit server or Cloud | Yours: [browser WebRTC sideband or WebSocket relay](../realtime/deployment.md) |
| Telephony | SIP in and out, DTMF, transfers; numbers on Cloud | [Bridge a provider](../realtime/deployment.md#siptelephony-bridge) such as Twilio |
| Turn detection | Silero VAD, own turn-detector model, adaptive interruption | [Provider turn detection, barge-in, push-to-talk](../realtime/turns.md) |
| Noise cancellation | Krisp and ai-coustics plugins; enhanced models on Cloud | Provider-side only |
| Hand off to another agent mid-call | Yes, context carried over | No; [delegate from a tool](../realtime/tools.md#delegating-work-during-a-call) instead |
| Tools mid-call | `@function_tool`, MCP | [The same tools, toolsets and dependencies](../realtime/tools.md) as a text agent |
| Capabilities mid-call | No equivalent | [Capabilities and hooks](../realtime/capabilities.md) run in the session |
| After the call | `session.history`, `SessionReport` JSON | [`Agent.run()` on the call's history](../realtime/history.md#handing-off-to-a-text-agent) for structured output or follow-up |
| Observability | OpenTelemetry; Insights on Cloud | [OpenTelemetry](../realtime/observability.md): session, turn and tool spans, usage attributed per response |
| Evals | pytest framework with an LLM judge; simulations on Cloud | [Pydantic Evals](../evals.md) on the text hand-off; nothing realtime-specific yet |
| Deployment | Agent server, dispatch, jobs; Cloud or self-host | [Your process, your backend](../realtime/deployment.md) |
| The same agent without voice | Text-only sessions, still a room and a server | [`run()`, CLI, web chat, AG-UI, Vercel AI](../interfaces.md) |
