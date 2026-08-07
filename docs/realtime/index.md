# Realtime (speech-to-speech)

Pydantic AI's realtime support lets an agent hold a live, spoken conversation. It streams the
user's audio to a speech-to-speech model and streams the model's spoken reply back over one
persistent connection, so latency is low and interruptions feel natural.

A realtime session uses the same agent tools, dependencies, instructions, message history, usage
limits, and observability as the rest of Pydantic AI. Your application owns audio capture and
playback; Pydantic AI runs the provider-agnostic agent loop on your backend.

## Choose your path

| Path | Best for | Where Pydantic AI fits |
| --- | --- | --- |
| **Native speech-to-speech with Pydantic AI** | Low-latency voice agents with server-side tools and shared history | Runs the complete realtime agent described here |
| **Browser WebRTC + server sideband** | Browser voice agents that want browser-direct media *and* server-side tools and history | Negotiates the call and runs the agent over its control plane while the browser owns the audio; see [Browser / WebRTC](lifecycle.md#browser-webrtc) |
| **Browser talks directly to the provider** | Provider-native, UI-only experiences using an ephemeral token | Use the provider SDK for the media session; Pydantic AI can power separate backend workflows |
| **Batch STT → text agent → TTS** | Text-model choice, structured output, or independent speech components | Compose a standard [agent](../agent.md) with chosen STT and TTS services |

## Quickstart

Install Pydantic AI with the OpenAI and realtime dependencies:

```bash
pip install "pydantic-ai-slim[realtime,openai]"
```

Set `OPENAI_API_KEY`, then send a text prompt and save the spoken reply as
`realtime-response.wav`:

```python
import asyncio
import wave

from pydantic_ai import Agent
from pydantic_ai.realtime import (
    PartDeltaEvent,
    RealtimeTurnCompleteEvent,
    SpeechPartDelta,
)
from pydantic_ai.realtime.openai import OpenAIRealtimeModel

agent = Agent(instructions='Keep your replies short and conversational.')
OUTPUT_PATH = 'realtime-response.wav'


async def main() -> None:
    audio = bytearray()
    model = OpenAIRealtimeModel('gpt-realtime')
    async with agent.realtime(model).session() as session:
        await session.send('Tell me a fun fact about octopuses.')
        async for event in session:
            match event:
                case PartDeltaEvent(delta=SpeechPartDelta(audio_chunk=chunk)) if chunk:
                    audio.extend(chunk)
                case RealtimeTurnCompleteEvent():
                    break

        with wave.open(OUTPUT_PATH, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(session.profile['audio_output_sample_rate'])
            wav_file.writeframes(audio)


asyncio.run(main())
```

Play `realtime-response.wav` with any audio player. The
[text-to-audio example](../examples/realtime-text-to-audio.md) adds a streamed transcript,
command-line prompt, and empty-audio check. For a microphone and speaker implementation, see the
[complete voice assistant example](../examples/realtime-voice.md).

## How sessions work

Your backend opens the provider connection and runs a
[`RealtimeSession`][pydantic_ai.realtime.RealtimeSession]. Stream content in with
[`send()`][pydantic_ai.realtime.RealtimeSession.send] or
[`send_audio()`][pydantic_ai.realtime.RealtimeSession.send_audio], and iterate the session for
content, tool, turn, error, and reconnect events.

```text
device ↔ media bridge ↔ RealtimeSession ↔ provider
                         ├── typed tools
                         └── message history
                         (your backend)
```

The media reaches the session one of two ways: it flows **through** the backend (your app streams
microphone bytes in and plays audio events back — the diagram above), or it flows **around** it,
browser ↔ provider directly over WebRTC, while the backend attaches a control-plane
[sideband](lifecycle.md#browser-webrtc) to the same call. Either way the tools, history, and secrets
stay server-side.

Keep provider keys on the server. Browser, WebRTC, and telephony stacks are transports that connect
the user to this backend session; see [Connection lifecycle](lifecycle.md#connecting-a-frontend).

## Learn by task

- [Audio, transcripts, images, and events](audio.md) covers the PCM wire contract, playback,
  captions, image input, and the event vocabulary.
- [Turns and interruptions](turns.md) covers automatic turn detection, barge-in, output
  truncation, and push-to-talk.
- [Tools and capabilities](tools.md) covers function tools, provider-native tools, concurrency,
  capability hooks, and delegation during a call.
- [History and handoff](history.md) covers retained transcripts, audio and images, session seeding,
  and continuing with a standard text agent.
- [Connection lifecycle](lifecycle.md) covers frontend transports, reconnection, session limits,
  errors, and troubleshooting.
- [Usage and observability](observability.md) covers usage limits, cost accounting, Logfire, and
  gateway trace propagation.
- The [API reference](../api/realtime.md) lists session and codec types and explains how to
  implement another provider.

## Provider support

All providers implement the same [`RealtimeModel`][pydantic_ai.realtime.RealtimeModel] interface.
Provider pages are the canonical source for installation, model names, settings, feature support,
and quirks:

| Provider | Audio output | Text output | Image input | State-restoring reconnect |
| --- | :---: | :---: | :---: | :---: |
| [OpenAI](openai.md) | ✓ | ✓ | ✓ | Replays local history |
| [Azure OpenAI](azure.md) | ✓ | ✓ | ✓ | Replays local history |
| [Google Gemini](gemini.md) | ✓ | ✓ | ✓ | ✓, when enabled |
| [xAI](xai.md) | ✓ | ✗ | ✗ | ✓ |

For portable branching, inspect [`RealtimeModel.profile`][pydantic_ai.realtime.RealtimeModel.profile]
or [`RealtimeSession.profile`][pydantic_ai.realtime.RealtimeSession.profile]. The
[`RealtimeModelProfile`][pydantic_ai.realtime.RealtimeModelProfile] reports audio sample rates and
support for image input, manual turn control, interruption, output truncation, text output, session
seeding, audio/image seeding, asynchronous tool calls, native tools, and thinking.

Each flag is resolved from the model's defaults and the provider's, then the model's `profile=`
argument — the same three layers a standard [`Model`][pydantic_ai.models.Model] resolves. Pass
`profile=` when the model name doesn't identify the model and the inferred facts are wrong, most
often with an Azure deployment named something other than its model:

```python {test="skip"}
from pydantic_ai.realtime.azure import AzureRealtimeModel

# The deployment serves a reasoning model, but nothing in its name says so.
model = AzureRealtimeModel('voice-prod', profile={'supports_thinking': True})
```

A partial dict is merged over the resolved profile; pass a callable
`(resolved) -> RealtimeModelProfile` instead to replace it wholesale.

## Shared settings

[`RealtimeModelSettings`][pydantic_ai.realtime.RealtimeModelSettings] defines common settings for
`tool_choice`, `parallel_tool_calls`, `max_tokens`, `input_transcription_model`, `output_modality`,
`handshake_timeout`, [`turn_detection`][pydantic_ai.realtime.TurnDetection], and
[`thinking`][pydantic_ai.realtime.RealtimeModelSettings.thinking]. Set defaults on the realtime
model or pass settings for one session; per-session values override model defaults.

The agent's regular `model_settings` and capability `get_model_settings()` contributions do not
configure realtime sessions. Unsupported shared settings are ignored, matching request-response
models — with one deliberate exception: `output_modality='text'` on a model whose profile reports
`supports_text_output=False` (Gemini Live and xAI) raises a `UserError` before connecting, because
silently answering with speech would be worse than not starting. Voices and detailed controls are
provider-specific: use `openai_voice`, `google_voice`, or
`xai_voice` on the corresponding provider settings. See the provider pages for defaults and
limitations.

## Relationship to standard agent runs

[`Agent.realtime()`][pydantic_ai.agent.Agent.realtime] is the long-lived, bidirectional sibling of
[`run()`][pydantic_ai.agent.AbstractAgent.run] and
[`iter()`][pydantic_ai.agent.AbstractAgent.iter]. It accepts realtime settings, dependencies,
instructions, toolsets, capabilities, usage limits, metadata, and `message_history`. Input arrives
through the live session instead of a single `user_prompt`.

Realtime sessions have no `output_type`, output-validation retries, graph pause for out-of-band
deferred tools, or graph node/model-request hooks. Use [tools and capabilities](tools.md) for
in-session work, and [hand off to a text agent](history.md#handing-off-to-a-text-agent) for structured
output or deeper reasoning.

## Limitations

- Sessions run server-side; SIP is not built in. Browser-direct WebRTC is supported on OpenAI and
  Azure OpenAI as a [server sideband](lifecycle.md#browser-webrtc), not on Gemini Live or xAI.
- Provider resumption handles cannot be persisted and resumed in another process.
- Dynamic instructions are resolved once when the session connects.
- History processors do not transform `message_history` before realtime seeding; preprocess it
  before opening the session when filtering or redaction is required.
- Realtime-specific exchange hooks are not yet available; use supported tool hooks and session
  events.
- Interactive human-in-the-loop tool approval is not yet supported. A
  [`HandleDeferredToolCalls`][pydantic_ai.capabilities.HandleDeferredToolCalls] handler resolves
  approvals from policy, immediately; asking a person mid-call and resuming on their answer needs a
  design of its own. See [approval-gated tools](tools.md#deferred-and-approval-required-tools).
