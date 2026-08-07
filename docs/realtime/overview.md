# Realtime (speech-to-speech)

Pydantic AI's realtime support lets an agent hold a live, spoken conversation. It streams the
user's audio to a speech-to-speech model and streams the model's spoken reply back over one
persistent connection, so latency is low and interruptions feel natural.

A realtime session uses the same agent tools, dependencies, instructions, message history, usage
limits, and observability as the rest of Pydantic AI, and that's the point: the conversation runs on
your backend, so mid-call the agent can look up an order, check availability, or act on the
logged-in user's data with the same tools a text agent would use. That's what this path buys you
over wiring a browser directly to a provider — a phone support line, an in-app voice assistant, or
a drive-through kiosk are all this same shape. Your application owns audio capture and playback;
Pydantic AI runs the provider-agnostic agent loop.

## Quickstart

Install Pydantic AI with the OpenAI and realtime dependencies, and set `OPENAI_API_KEY`:

```bash
pip install "pydantic-ai-slim[realtime,openai]"
```

A complete voice agent is one agent, one session, and three small loops — microphone in, speaker
out, and a transcript log. The model hears the user, calls your tool on your backend, and answers
out loud:

```python {title="reservations.py" dunder_name="not_main"}
import asyncio
from collections.abc import AsyncIterator

from pydantic_ai import Agent
from pydantic_ai.realtime import RealtimeSession
from pydantic_ai.realtime.openai import OpenAIRealtimeModel

agent = Agent(instructions='You take reservations for The Terrace. Keep replies short.')


@agent.tool_plain
def check_availability(day: str, party_size: int) -> str:
    """Check whether a table is free."""
    return f'One table for {party_size} is free at 7 pm {day}.'


async def stream_microphone(session: RealtimeSession) -> None:
    ...  # capture signed 16-bit mono PCM chunks and `await session.send_audio(chunk)`


async def play_audio(chunks: AsyncIterator[bytes]) -> None:
    async for chunk in chunks:
        ...  # write the PCM chunk to your speaker


async def main() -> None:
    model = OpenAIRealtimeModel('gpt-realtime')
    async with agent.realtime(model).session() as session:
        microphone = asyncio.create_task(stream_microphone(session))
        speaker = asyncio.create_task(play_audio(session.stream_audio()))

        async for part in session.stream_transcripts():
            print(f'{part.speaker}: {part.transcript}')
            #> user: Hi! Do you have a table for two tomorrow night?
            #> assistant: We do: 7 pm, table for two. Want me to book it?
            if part.speaker == 'assistant':
                break  # keep listening in a real call; we stop after one exchange

        await session.close()
        await asyncio.gather(microphone, speaker)


if __name__ == '__main__':
    asyncio.run(main())
```

The audio placeholders are the only part Pydantic AI doesn't provide, because they depend on your
audio stack: capture and play at the sample rates `session.profile` reports (see
[Provider support](#provider-support) below). The
[voice assistant example](../examples/realtime-voice.md) fills them in with `sounddevice` for a
runnable microphone-and-speaker loop; the
[text-to-audio example](../examples/realtime-text-to-audio.md) skips audio input entirely by
sending a text prompt and saving the spoken reply to a WAV file.

## How sessions work

Your backend opens the provider connection and runs a
[`RealtimeSession`][pydantic_ai.realtime.RealtimeSession]. Stream content in with
[`send()`][pydantic_ai.realtime.RealtimeSession.send] or
[`send_audio()`][pydantic_ai.realtime.RealtimeSession.send_audio], and iterate the session for
content, tool, turn, error, and reconnect events, or consume the dedicated
[`stream_audio()`][pydantic_ai.realtime.RealtimeSession.stream_audio] and
[`stream_transcripts()`][pydantic_ai.realtime.RealtimeSession.stream_transcripts] views as the
quickstart does.

```text
device ↔ media bridge ↔ RealtimeSession ↔ provider
                         ├── typed tools
                         └── message history
                         (your backend)
```

The *media bridge* is whatever moves audio between the user's device and your backend — and it's
how you deploy this beyond a local microphone. Keep provider keys on the server; the frontend only
ever talks to you:

- **Browser → your WebSocket:** a page captures microphone audio and relays PCM chunks over a
  WebSocket to your backend, which pumps them into the session and streams the reply back. The
  [camera example](../examples/realtime-camera.md) is this shape end to end.
- **WebRTC media room:** a platform such as LiveKit handles echo cancellation, jitter, devices, and
  telephony, while a server-side participant runs the realtime session.
- **SIP/telephony bridge:** a telephony provider terminates the phone call and bridges its audio
  stream to the backend session.

See [Connection lifecycle](lifecycle.md#connecting-a-frontend) for more on each shape.

## Learn by task

- [Audio and transcripts](audio.md) covers the PCM wire contract, playback,
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
| [Google Gemini](gemini.md) | ✓ | ✗ | ✓ | ✓, when enabled |
| [xAI](xai.md) | ✓ | ✗ | ✗ | ✓ |

For portable branching, inspect [`RealtimeModel.profile`][pydantic_ai.realtime.RealtimeModel.profile]
or [`RealtimeSession.profile`][pydantic_ai.realtime.RealtimeSession.profile]: the
[`RealtimeModelProfile`][pydantic_ai.realtime.RealtimeModelProfile] reports the audio sample rates
to capture and play at, plus one flag per capability in the table above and beyond. Profiles resolve
the same way as for a standard [`Model`][pydantic_ai.models.Model] — defaults, then the provider's
knowledge of the model name, then your `profile=` argument on top. Pass `profile=` when the model
name doesn't identify the model and the inferred facts are wrong, most often with an Azure
deployment named something other than its model:

```python {test="skip"}
from pydantic_ai.realtime.azure import AzureRealtimeModel

# The deployment serves a reasoning model, but nothing in its name says so.
model = AzureRealtimeModel('voice-prod', profile={'supports_thinking': True})
```

A partial dict is merged over the resolved profile; pass a callable
`(resolved) -> RealtimeModelProfile` instead to replace it wholesale.

## Shared settings

[`RealtimeModelSettings`][pydantic_ai.realtime.RealtimeModelSettings] defines the settings shared
across realtime providers, from `tool_choice` to
[`turn_detection`][pydantic_ai.realtime.TurnDetection]. Set defaults on the realtime model or pass
settings for one session; per-session values override model defaults. Voices and detailed controls
are provider-specific — `openai_voice`, `google_voice`, `xai_voice` and friends live on the
corresponding provider settings classes, with defaults and limitations on the provider pages.

The agent's regular `model_settings` and capability `get_model_settings()` contributions do not
configure realtime sessions. Unsupported shared settings are ignored, matching request-response
models, with one deliberate exception:

!!! note "Asking for text on a speech-only model fails fast"
    `output_modality='text'` on a model whose profile reports `supports_text_output=False`
    (Gemini Live and xAI) raises a `UserError` before connecting: silently answering with speech
    would be worse than not starting.

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

## Other ways to build voice

Not every voice product needs the realtime agent loop:

| Path | Best for | Where Pydantic AI fits |
| --- | --- | --- |
| **Native speech-to-speech with Pydantic AI** | Low-latency voice agents with server-side tools and shared history | Runs the complete realtime agent described here |
| **Browser talks directly to the provider** | Provider-native, UI-only experiences using an ephemeral token | Use the provider SDK for the media session; Pydantic AI can power separate backend workflows |
| **Batch STT → text agent → TTS** | Text-model choice, structured output, or independent speech components | Compose a standard [agent](../agent.md) with chosen speech-to-text and text-to-speech services |

Browser-direct provider sessions move the agent loop into the client and give up Pydantic AI's
server-side tools and history; use that path only when the provider-native client experience is the
actual goal.

## Limitations

- Sessions run server-side over WebSocket; browser-direct WebRTC and SIP are not built in.
- Provider resumption handles cannot be persisted and resumed in another process.
- Dynamic instructions are resolved once when the session connects.
- History processors do not transform `message_history` before realtime seeding; preprocess it
  before opening the session when filtering or redaction is required.
- Realtime-specific exchange hooks are not yet available; use supported tool hooks and session
  events.
- Interactive human-in-the-loop tool approval is not yet supported: a
  [`HandleDeferredToolCalls`][pydantic_ai.capabilities.HandleDeferredToolCalls] handler resolves
  approvals from policy, immediately. See
  [approval-gated tools](tools.md#deferred-and-approval-required-tools).
