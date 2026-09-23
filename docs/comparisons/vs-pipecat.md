# Pydantic AI vs Pipecat

Pipecat is an open-source Python framework for realtime voice and multimodal agents, maintained by Daily. Frames move through frame processors, you compose those into a pipeline, and a worker runs it: transports carry the audio, and services for speech-to-text, LLMs, text-to-speech and speech-to-speech models slot in as stages. Around that it ships Smart Turn detection, noise-cancellation filters, telephony serializers, Flows for structured conversations, and an eval harness. Pydantic AI's [realtime support](../realtime/overview.md) is a speech-to-speech agent loop on four providers behind one API, and it is the same typed [`Agent`][pydantic_ai.Agent] that runs as text, in a [web chat](../web.md) or behind your API: the call uses the same tools, dependencies and [capabilities](../realtime/capabilities.md), becomes ordinary message history you can [hand to a text agent](../realtime/history.md#handing-off-to-a-text-agent) for structured output, and is traced end to end in [Logfire](https://pydantic.dev/logfire). With Pipecat the pipeline is the product and the model is one stage in it; here the agent is the product and voice is one of its interfaces.

Pydantic AI is one part of a stack: the [Harness SDK](https://pydantic.dev/docs/ai/harness/) for capabilities and complete agents, [Pydantic Evals](../evals.md), [Pydantic Graph](../graph.md), [Pydantic Logfire](https://pydantic.dev/logfire) for observability, and [Pydantic](https://pydantic.dev/docs/validation/latest/get-started/) itself for validation. The tables below cover the whole of it.

## Framework

| | Pipecat | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Language | Python, with client SDKs in JavaScript, Swift, Kotlin and C++ | Python |
| License | BSD-2-Clause | MIT |
| Model providers | Many (services) | [Many](../models/overview.md) |
| Extensibility | Frame processors and services | [Capabilities and toolsets](../extensibility.md); [50+ with the Harness SDK](https://pydantic.dev/docs/ai/harness/) |
| Harnesses | Build your own; processors wrap LangChain and AWS Strands agents | Built-in [`Coder`](https://pydantic.dev/docs/ai/harness/coder/) and [`Researcher`](https://pydantic.dev/docs/ai/harness/researcher/), or compose your own |
| Observability | OpenTelemetry | [OpenTelemetry](../logfire.md#using-opentelemetry), including [Pydantic Logfire](https://pydantic.dev/logfire) |
| Durable execution | No | [Seven integrations](../durable_execution/overview.md) |
| Interfaces | WebRTC and WebSocket transports, telephony, client SDKs | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental) |
| Realtime voice | Cascaded STT + LLM + TTS, and speech-to-speech services | [Speech-to-speech](../realtime/overview.md), four providers |
| Evals | Yes | [Pydantic Evals](../evals.md) |
| Image generation | Image-generation services | [Image Generation](../image-generation.md) |

## Realtime, side by side

Our realtime support means speech-to-speech models: one persistent connection, audio in and audio out, on the four providers below. Pipecat leads with the cascaded pipeline, which we do not run, and offers speech-to-speech services alongside it. If you are choosing a voice stack, these are the rows that decide it:

| | Pipecat | Pydantic AI |
|---|---|---|
| Speech-to-speech providers | Services for OpenAI Realtime, Gemini Live, AWS Nova Sonic, Grok Voice, Inworld and Ultravox | [Four](../realtime/overview.md#provider-support) behind one API: OpenAI, Azure OpenAI, Gemini Live, xAI; ElevenLabs in [#7964](https://github.com/pydantic/pydantic-ai/pull/7964) |
| Cascaded STT + LLM + TTS | Yes, the primary path; dozens of STT, LLM and TTS services | Not built in; [compose it yourself](../realtime/overview.md#other-ways-to-build-voice) around a text agent |
| Audio transport | First-party transports for Daily, SmallWebRTC, WebSocket and FastAPI WebSocket, plus LiveKit, Tavus and HeyGen | Yours: [browser WebRTC sideband or WebSocket relay](../realtime/deployment.md) |
| Telephony | Dial-in and dial-out over PSTN and SIP with Daily, Twilio, Telnyx, Plivo and Exotel; DTMF both ways, cold and warm transfer | [Bridge a provider](../realtime/deployment.md#siptelephony-bridge) such as Twilio |
| Turn detection | Silero VAD, the bundled Smart Turn model on by default, interruptions on by default, pluggable turn-start and turn-stop strategies | [Provider turn detection, barge-in, push-to-talk](../realtime/turns.md) |
| Noise cancellation | Krisp VIVA, Koala, ai-coustics and RNNoise audio filters; Krisp ships only on Pipecat Cloud | Provider-side only |
| Hand off to another agent mid-call | Yes: workers on a shared bus, activated and deactivated from a tool call | No; [delegate from a tool](../realtime/tools.md#delegating-work-during-a-call) instead |
| Structured conversation flows | Pipecat Flows, in the core package: a graph of nodes, declarative YAML or JSON or programmatic Python, with a visual editor; cascaded text LLMs only, not speech-to-speech | No flow graph; instructions and tools steer the call, and [`pydantic-graph`](../graph.md) is a general async state machine you can drive a session from rather than a flow the realtime layer knows about |
| Tools mid-call | Functions whose schema is derived from the signature and docstring, plus an MCP client | [The same tools, toolsets and dependencies](../realtime/tools.md) as a text agent |
| Capabilities mid-call | No equivalent; frame processors and observers sit in the pipeline instead | [Capabilities and hooks](../realtime/capabilities.md), with documented limits |
| After the call | The `LLMContext` the aggregators built, plus background context summarization | [`Agent.run()` on the call's history](../realtime/history.md#handing-off-to-a-text-agent) for structured output or follow-up |
| Observability | OpenTelemetry, opt-in per worker: conversation, turn, STT, LLM and TTS spans, with TTFB, token and character metrics | [OpenTelemetry](../realtime/observability.md): session, turn and tool spans, usage attributed per response |
| Evals | Pipecat Evals: scripted and simulated scenarios run against the real pipeline and graded by a judge LLM, in text or audio mode | [Pydantic Evals](../evals.md) on the text hand-off; nothing realtime-specific yet |
| Deployment | A Python process; Pipecat Cloud and Pipecat Enterprise, or self-host | [Your process, your backend](../realtime/deployment.md) |
| The same agent without voice | Not a separate surface: the agent is its pipeline, assembled around realtime media | [`run()`, CLI, web chat, AG-UI, Vercel AI](../interfaces.md) |
