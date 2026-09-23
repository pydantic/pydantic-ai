# Pydantic AI vs Pipecat

Pipecat is an open-source Python framework for realtime voice and multimodal agents, maintained by Daily: frame processors compose into a pipeline, transports carry the audio, and speech-to-text, LLM, text-to-speech and speech-to-speech services slot in as stages. Pydantic AI's [realtime support](../realtime/overview.md) is a speech-to-speech agent loop on four providers behind one API, and it is the same typed [`Agent`][pydantic_ai.Agent] that runs as text, in a [web chat](../web.md) or behind your API: the call uses the same tools, dependencies and [capabilities](../realtime/capabilities.md), becomes ordinary message history you can [hand to a text agent](../realtime/history.md#handing-off-to-a-text-agent) for structured output, and is traced end to end in [Logfire](https://pydantic.dev/logfire). With Pipecat the pipeline is the product and the model is one stage in it; here the agent is the product and voice is one of its interfaces.

Pydantic AI is one part of a stack: the [Harness SDK](https://pydantic.dev/docs/ai/harness/) for capabilities and complete agents, [Pydantic Evals](../evals.md), [Pydantic Graph](../graph.md), [Pydantic Logfire](https://pydantic.dev/logfire) for observability, and [Pydantic](https://pydantic.dev/docs/validation/latest/get-started/) itself for validation. The tables below cover the whole of it.

## Framework

| | Pipecat | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Language | Python | Python |
| License | BSD-2-Clause | MIT |
| Model providers | Many (services) | [Many](../models/overview.md) |
| Extensibility | Frame processors and services | [Capabilities and toolsets](../extensibility.md); [50+ with the Harness SDK](https://pydantic.dev/docs/ai/harness/) |
| Harnesses | Build your own | Built-in [`Coder`](https://pydantic.dev/docs/ai/harness/coder/) and [`Researcher`](https://pydantic.dev/docs/ai/harness/researcher/), or compose your own |
| Observability | OpenTelemetry | [OpenTelemetry](../logfire.md#using-opentelemetry), including [Pydantic Logfire](https://pydantic.dev/logfire) |
| Durable execution | No | [Seven integrations](../durable_execution/overview.md) |
| Interfaces | WebRTC and WebSocket transports, telephony, client SDKs | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental) |
| Realtime voice | Cascaded STT + LLM + TTS, and speech-to-speech services | [Speech-to-speech](../realtime/overview.md), four providers |
| Evals | Yes | [Pydantic Evals](../evals.md) |
| Image generation | Image-generation services | [Image Generation](../image-generation.md) |

## Realtime, side by side

Our realtime support means speech-to-speech models: one persistent connection, audio in and audio out, on the four providers below. Pipecat also runs the cascaded pipeline, its primary path, which we do not. If you are choosing a voice stack, these are the rows that decide it:

| | Pipecat | Pydantic AI |
|---|---|---|
| Speech-to-speech providers | Services for several providers | [Four](../realtime/overview.md#provider-support) behind one API: OpenAI, Azure OpenAI, Gemini Live, xAI; ElevenLabs in [#7964](https://github.com/pydantic/pydantic-ai/pull/7964) |
| Cascaded STT + LLM + TTS | Yes, the primary path; dozens of STT, LLM and TTS services | Not built in; [compose it yourself](../realtime/overview.md#other-ways-to-build-voice) around a text agent |
| Audio transport | First-party WebRTC and WebSocket transports, Daily and LiveKit among them | Yours: [browser WebRTC sideband or WebSocket relay](../realtime/deployment.md) |
| Telephony | PSTN and SIP in and out, DTMF, transfers | [Bridge a provider](../realtime/deployment.md#siptelephony-bridge) such as Twilio |
| Turn detection | Silero VAD, own Smart Turn model, interruptions | [Provider turn detection, barge-in, push-to-talk](../realtime/turns.md) |
| Noise cancellation | Krisp, ai-coustics and other filters | Provider-side only |
| Hand off to another agent mid-call | Yes, between workers on a shared bus | No; [delegate from a tool](../realtime/tools.md#delegating-work-during-a-call) instead |
| Structured conversation flows | Pipecat Flows: a node graph in YAML, JSON or Python, with a visual editor; cascaded pipelines only, not speech-to-speech | No flow graph; instructions and tools steer the call |
| Tools mid-call | Direct functions, MCP | [The same tools, toolsets and dependencies](../realtime/tools.md) as a text agent |
| Capabilities mid-call | No equivalent | [Capabilities and hooks](../realtime/capabilities.md), with documented limits |
| After the call | The pipeline's `LLMContext` messages | [`Agent.run()` on the call's history](../realtime/history.md#handing-off-to-a-text-agent) for structured output or follow-up |
| Observability | OpenTelemetry, opt-in; turn and per-service spans | [OpenTelemetry](../realtime/observability.md): session, turn and tool spans, usage attributed per response |
| Evals | Pipecat Evals: scripted and simulated scenarios, LLM judge | [Pydantic Evals](../evals.md) on the text hand-off; nothing realtime-specific yet |
| Deployment | A Python process; Pipecat Cloud or self-host | [Your process, your backend](../realtime/deployment.md) |
| The same agent without voice | Text-only bots over a WebSocket transport, still a pipeline and a worker | [`run()`, CLI, web chat, AG-UI, Vercel AI](../interfaces.md) |
