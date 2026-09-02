# Realtime (speech-to-speech)

Pydantic AI's realtime support lets an agent hold a live, spoken conversation. It streams the
user's audio to a speech-to-speech model and streams the model's spoken reply back over one
persistent connection, so latency is low and interruptions feel natural.

A realtime session uses the same agent [tools](tools.md), [dependencies](../dependencies.md),
[instructions](../agent.md#instructions), [message history](../message-history.md),
[capabilities](capabilities.md), [usage limits](../agent.md#usage-limits), and
[observability](observability.md) as the rest of Pydantic AI, and that's the point: mid-call the
agent can look up an order, check availability, or act on the logged-in user's data with the same
tools and dependencies a text agent would use. The call itself becomes ordinary message history
that you can [hand to `Agent.run()`](history.md#handing-off-to-a-text-agent) for summarization or
structured follow-up, the same code runs against [four providers](#provider-support), and usage
limits and [Logfire](../logfire.md) tracing are built in. Your application owns the audio transport —
bridged through your backend, or [browser-direct over WebRTC](deployment.md#browser-webrtc-server-sideband)
on OpenAI and Azure — while Pydantic AI runs the provider-agnostic agent loop.

## Quickstart

Install Pydantic AI with the OpenAI realtime dependencies, and set `OPENAI_API_KEY`:

```bash
pip/uv-add "pydantic-ai-slim[openai-realtime]"
```

A complete voice agent is one agent, one session, and three small loops — microphone in, speaker
out, and a transcript log. The model hears the user, calls your tool on your backend, and answers
out loud:

```python {title="reservations.py" dunder_name="not_main"}
import asyncio
import contextlib
from collections.abc import AsyncIterator

from pydantic_ai import Agent

agent = Agent(instructions='You take reservations for The Terrace. Keep replies short.')


@agent.tool_plain
async def check_availability(day: str, party_size: int) -> str:
    """Check whether a table is free."""
    return f'One table for {party_size} is free at 7 pm {day}.'


async def microphone_chunks() -> AsyncIterator[bytes]:
    yield b'...'  # capture signed 16-bit mono PCM chunks from your microphone


async def play_audio(chunks: AsyncIterator[bytes]) -> None:
    async for chunk in chunks:
        ...  # write the PCM chunk to your speaker


async def main():
    async with agent.realtime('openai:gpt-realtime').session() as session:
        microphone = asyncio.create_task(session.send_audio(microphone_chunks()))
        speaker = asyncio.create_task(play_audio(session.stream_audio()))

        async for part in session.stream_transcripts():
            print(f'{part.speaker}: {part.transcript}')
            #> user: Hi! Do you have a table for two tomorrow night?
            #> assistant: We do: 7 pm, table for two. Want me to book it?
            if part.speaker == 'assistant':
                break  # keep listening in a real call; we stop after one exchange

    # Leaving the `async with` block closes the session, which ends the speaker's audio stream —
    # but the microphone reads an external source, so stop it explicitly.
    microphone.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await microphone
    await speaker


if __name__ == '__main__':
    asyncio.run(main())
```

_(This example is complete, it can be run "as is" — after filling in the two audio placeholders,
which depend on your audio stack)_

Capture and play at the sample rates the model expects — they're reported by the model's profile
and can differ between input and output (see [Provider support](#provider-support) below). The
[voice assistant example](../examples/realtime-voice.md) fills the placeholders in with
[`listentome`](https://github.com/Kludex/listentome) for a runnable microphone-and-speaker loop; the
[text-to-audio example](../examples/realtime-text-to-audio.md) skips audio input entirely by
sending a text prompt and saving the spoken reply to a WAV file.

## How sessions work

Your backend opens the provider connection and runs a
[`RealtimeSession`][pydantic_ai.realtime.RealtimeSession]. Stream content in with
[`send()`][pydantic_ai.realtime.RealtimeSession.send] or
[`send_audio()`][pydantic_ai.realtime.RealtimeSession.send_audio], and iterate the session for its
[event stream](events.md) — content, tool, turn, error, and reconnect events — or consume the
dedicated [`stream_audio()`][pydantic_ai.realtime.RealtimeSession.stream_audio] and
[`stream_transcripts()`][pydantic_ai.realtime.RealtimeSession.stream_transcripts] views as the
quickstart does.

```text
device ↔ media bridge ↔ RealtimeSession ↔ provider
                         ├── typed tools
                         └── message history
                         (your backend)
```

The *media bridge* is whatever moves audio between the user's device and your backend — a browser
WebSocket or a telephony bridge. It's how you deploy this beyond a local microphone; see
[Connecting a frontend](deployment.md) for each setup. On OpenAI and Azure the browser can instead
exchange media with the provider directly over [WebRTC](deployment.md#browser-webrtc-server-sideband),
with your backend running this same loop over a control-plane sideband rather than a media bridge.

## Learn by task

- [Audio, images, and transcripts](audio.md) covers the PCM wire contract, playback, captions,
  input transcription, and image input.
- [Events](events.md) covers the session event vocabulary, which events are shared with standard
  runs, and the turn boundary.
- [Turns and interruptions](turns.md) covers automatic turn detection, barge-in, output
  truncation, and push-to-talk.
- [Tools](tools.md) covers function tools, provider-native tools, concurrency, approval, and
  delegation during a call.
- [Capabilities and hooks](capabilities.md) covers how capabilities and their hooks map onto a
  session.
- [History and handoff](history.md) covers retained transcripts, audio and images, session seeding,
  and continuing with a standard text agent.
- [Connecting a frontend](deployment.md) covers the transport options between user devices and your
  backend.
- [Connection lifecycle](lifecycle.md) covers the session lifecycle, reconnection, session limits,
  and errors.
- [Usage and observability](observability.md) covers usage limits, cost accounting, Logfire, and
  gateway trace propagation.
- [Troubleshooting](troubleshooting.md) indexes common problems by symptom.
- The [API reference](../api/realtime.md) lists session and codec types and explains how to
  implement another provider.

## Provider support

All providers implement the same [`RealtimeModel`][pydantic_ai.realtime.RealtimeModel] interface.
Provider pages are the canonical source for installation, model names, settings, feature support,
and quirks:

| Provider | Audio output | Image input | Text output | [Browser WebRTC](deployment.md#browser-webrtc-server-sideband) | Async tool calls | [Thinking](../capabilities/thinking.md) | State-restoring reconnect |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [OpenAI](openai.md) | ✓ | ✓ | ✓ | ✓ | ✓ | `gpt-realtime-2*` models | Replays local history |
| [Azure OpenAI](azure.md) | ✓ | ✓ | ✓ | ✓ | ✓ | `gpt-realtime-2*` models | Replays local history |
| [Google Gemini](gemini.md) | ✓ | ✓ | ✗ | ✗ | Opt-in, native-audio models | ✓ | ✓, when enabled |
| [xAI](xai.md) | ✓ | ✗ | ✗ | ✗ | ✗ | `grok-voice-latest` and `-think-` models | ✓ |

For portable branching, inspect [`RealtimeModel.profile`][pydantic_ai.realtime.RealtimeModel.profile]
or [`RealtimeSession.profile`][pydantic_ai.realtime.RealtimeSession.profile]: the
[`RealtimeModelProfile`][pydantic_ai.realtime.RealtimeModelProfile] reports the audio sample rates
to capture and play at, plus one flag per capability in the table above and beyond. Profiles resolve
the same way as for a standard [`Model`][pydantic_ai.models.Model]
(see [Inspecting a model's profile](../models/overview.md#inspecting-a-models-profile)) — defaults,
then the provider's knowledge of the model name, then your `profile=` argument on top. Pass
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

Realtime sessions have their own settings type, playing the role that
[model run settings](../agent.md#model-run-settings) play for standard runs:
[`RealtimeModelSettings`][pydantic_ai.realtime.RealtimeModelSettings] defines the settings shared
across realtime providers, from `tool_choice` to
[`turn_detection`][pydantic_ai.realtime.TurnDetection]. Set defaults with `settings=` on the
realtime model constructor, or pass `realtime(model_settings=...)` for one session; per-session
values override model defaults:

```python
from pydantic_ai import Agent
from pydantic_ai.realtime import RealtimeModelSettings

agent = Agent(instructions='You are a helpful voice assistant.')
realtime = agent.realtime(
    'openai:gpt-realtime', model_settings=RealtimeModelSettings(output_modality='audio')
)
```

Voices and detailed controls are provider-specific — `openai_voice`, `google_voice`, `xai_voice`
and friends live on the corresponding provider settings classes, with defaults and limitations on
the provider pages.

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
[`iter()`][pydantic_ai.agent.AbstractAgent.iter], and its parameters mirror theirs:

```python {test="skip" lint="skip"}
agent.realtime(
    model,                # 'openai:gpt-realtime', or a RealtimeModel instance
    deps=...,             # dependencies, as in run()/iter()
    model_settings=...,   # RealtimeModelSettings
    instructions=...,     # combined with the agent's instructions
    toolsets=...,         # additional toolsets for the session
    capabilities=...,     # additional capabilities for the session
    usage=..., usage_limits=...,
    message_history=...,  # prior conversation to seed the session with
)
```

It accepts the same [dependencies](../dependencies.md), [instructions](../agent.md#instructions),
[toolsets](../toolsets.md), [capabilities](../capabilities/overview.md),
[usage limits](../agent.md#usage-limits), and [`message_history`](../message-history.md) as a
standard run. Input arrives through the live session instead of a single `user_prompt`:

| Standard-run feature | In a realtime session |
| --- | --- |
| Function tools and [tool hooks](capabilities.md#capability-stages-in-a-session) | ✓ — validation, retries, and execution hooks run as in a standard run |
| [Run hooks](capabilities.md#run-hooks) (`before_run`, `after_run`, `wrap_run`, `on_run_error`) | ✓ — once around the session |
| [Capabilities](capabilities.md), including third-party | ✓ — resolved once at connect |
| [Event stream](events.md) | ✓ — iterate the session, or attach [`ProcessEventStream`][pydantic_ai.capabilities.ProcessEventStream] |
| `output_type` and output validators | ✗ — [delegate to a text agent](tools.md#delegating-work-during-a-call) |
| Graph node and model-request hooks (e.g. `before_model_request`) | ✗ — no agent graph |
| History processors at seeding | ✗ — [preprocess before opening](capabilities.md#seeded-history-is-not-processed) |
| `event_stream_handler` parameter | ✗ — use [`ProcessEventStream`][pydantic_ai.capabilities.ProcessEventStream] |

See [Capabilities and hooks](capabilities.md) for the full mapping, and
[hand off to a text agent](history.md#handing-off-to-a-text-agent) for structured output or deeper
reasoning.

## Other ways to build voice

The same realtime loop deploys to a browser or phone over [WebRTC or a WebSocket
relay](deployment.md) without changing the agent code. If the realtime agent loop isn't the right
fit for a product, two alternatives sit outside it:

- **Batch STT → text agent → TTS.** Compose a standard [agent](../agent.md) with your own
  speech-to-text and text-to-speech services when you want a specific text model, structured output,
  or independently chosen speech components.
- **Browser directly to the provider.** A provider-native, UI-only experience using an ephemeral
  token: the provider's own SDK owns the session, so there is no server-side agent loop, tools, or
  shared history — unlike the [WebRTC sideband](deployment.md#browser-webrtc-server-sideband), where
  the browser owns the media but your backend still runs the agent. Pydantic AI can still power
  separate backend workflows.

## Limitations

| Limitation | Tracking |
| --- | --- |
| SIP is not built in; bridge telephony through a provider such as Twilio. | [Connecting a frontend](deployment.md#siptelephony-bridge) |
| New tools cannot be advertised mid-session, so `defer_loading=True` tools and tool-contributing capabilities are [rejected](capabilities.md#deferred-capability-loading). | [#7288](https://github.com/pydantic/pydantic-ai/issues/7288) |
| Realtime-specific exchange hooks are not yet available; use supported [tool hooks](capabilities.md) and [session events](events.md). | [#7190](https://github.com/pydantic/pydantic-ai/issues/7190), [#7191](https://github.com/pydantic/pydantic-ai/issues/7191) |
| Provider resumption handles cannot be persisted and resumed in another process. | [#7302](https://github.com/pydantic/pydantic-ai/issues/7302) |
| Dynamic instructions are resolved once when the session connects. | [#7303](https://github.com/pydantic/pydantic-ai/issues/7303) |
| History processors do not transform `message_history` before realtime seeding; [preprocess it](capabilities.md#seeded-history-is-not-processed) before opening the session when filtering or redaction is required. | [#7299](https://github.com/pydantic/pydantic-ai/issues/7299) |
| Interactive human-in-the-loop tool approval is not supported: a [`HandleDeferredToolCalls`][pydantic_ai.capabilities.HandleDeferredToolCalls] handler resolves approvals [from policy, immediately](tools.md#deferred-and-approval-required-tools). | [#7301](https://github.com/pydantic/pydantic-ai/issues/7301) |
| [`RunContext.enqueue()`][pydantic_ai.tools.RunContext.enqueue] accepts [one plain-text prompt per call](tools.md#enqueuing-prompts-from-tools), unlike its [standard-run form](../message-history.md#injecting-messages-mid-run). | [#7300](https://github.com/pydantic/pydantic-ai/issues/7300) |
| Gemini Live tool results are JSON-only: binary content attached to a [tool return](tools.md#function-tools) raises. | [#7362](https://github.com/pydantic/pydantic-ai/issues/7362) |
