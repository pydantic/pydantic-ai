# Realtime (speech-to-speech)

Some providers offer **realtime** models that exchange audio over a persistent bidirectional
connection (WebSocket or HTTP/2) instead of the request-response pattern of the standard
[`Model`][pydantic_ai.models.Model] interface. These models listen and speak at the same time,
detect when the user starts talking (so the model can be interrupted), and can call tools mid
conversation.

Pydantic AI exposes this through a small provider-agnostic layer in
[`pydantic_ai.realtime`][pydantic_ai.realtime], with [`Agent.realtime_session`][pydantic_ai.Agent.realtime_session]
as the high-level entry point: it reuses the agent's tools and instructions and runs the tool-call
loop for you.

!!! note "Realtime is separate from the text `Model`"
    A realtime session is not a `Model` request. It opens a connection you stream audio into and
    iterate events out of. Use a realtime model for the conversational surface, and a normal text
    [`Agent`][pydantic_ai.Agent] when you need structured output or heavier reasoning (see
    [Delegating to a text agent](#delegating-to-a-text-agent)).

## Installation

The OpenAI provider uses WebSockets, available via the `realtime` optional group:

```bash
pip install "pydantic-ai-slim[realtime]"
```

## Quickstart

```python {test="skip" lint="skip"}
from pydantic_ai import Agent
from pydantic_ai.realtime import AudioDelta, Transcript
from pydantic_ai.realtime.openai import OpenAIRealtimeModel

agent = Agent(instructions='You are a helpful voice assistant.')


@agent.tool_plain
def get_weather(city: str) -> str:
    return f'Sunny in {city}'


async def main():
    model = OpenAIRealtimeModel('gpt-realtime')
    async with agent.realtime_session(model=model) as session:
        await session.send_audio(microphone_chunk)  # PCM16 audio bytes
        async for event in session:
            if isinstance(event, AudioDelta):
                speaker.play(event.data)
            elif isinstance(event, Transcript) and event.is_final:
                print('assistant:', event.text)
```

You stream content in with the session's `send_*` helpers and consume events by iterating the
session:

| Method | Sends |
| --- | --- |
| [`send_audio`][pydantic_ai.realtime.RealtimeSession.send_audio] | A chunk of microphone audio (PCM16). |
| [`send_text`][pydantic_ai.realtime.RealtimeSession.send_text] | A complete text turn. |
| [`send_image`][pydantic_ai.realtime.RealtimeSession.send_image] | An image as conversation context (e.g. a video frame). |

## Events

Iterating a [`RealtimeSession`][pydantic_ai.realtime.RealtimeSession] yields:

| Event | Meaning |
| --- | --- |
| [`AudioDelta`][pydantic_ai.realtime.AudioDelta] | A chunk of the model's audio output. |
| [`Transcript`][pydantic_ai.realtime.Transcript] | Transcription of the model's speech / its text output (`is_final` marks the end of a turn). |
| [`InputTranscript`][pydantic_ai.realtime.InputTranscript] | Transcription of the user's speech (streamed; `is_final` marks the completed turn). |
| [`SpeechStarted`][pydantic_ai.realtime.SpeechStarted] | The provider detected the user started speaking (barge-in). |
| [`SpeechStopped`][pydantic_ai.realtime.SpeechStopped] | The user stopped speaking; the model is about to respond. |
| [`ToolCallStarted`][pydantic_ai.realtime.ToolCallStarted] | The session began executing a tool the model requested. |
| [`ToolCallCompleted`][pydantic_ai.realtime.ToolCallCompleted] | The tool finished and its result was sent back to the model. |
| [`TurnComplete`][pydantic_ai.realtime.TurnComplete] | The model finished a turn (`interrupted=True` if the user barged in). |
| [`Usage`][pydantic_ai.realtime.Usage] | Token usage for a completed model response (see [Usage and cost](#usage-and-cost)). |
| [`RateLimits`][pydantic_ai.realtime.RateLimits] | An updated rate-limit snapshot from the provider. |
| [`Reconnected`][pydantic_ai.realtime.Reconnected] | The connection dropped and was automatically re-established (see [Reconnecting](#reconnecting)). |
| [`SessionError`][pydantic_ai.realtime.SessionError] | The provider reported an error (`recoverable=False` means the connection dropped). |

## Configuring the session

Session behaviour is configured on the provider model. For OpenAI, on
[`OpenAIRealtimeModel`][pydantic_ai.realtime.openai.OpenAIRealtimeModel]:

```python {test="skip" lint="skip"}
from pydantic_ai.realtime.openai import OpenAIRealtimeModel, SemanticVAD

model = OpenAIRealtimeModel(
    'gpt-realtime',
    voice='alloy',
    turn_detection=SemanticVAD(eagerness='high'),  # how eagerly the model takes its turn
    input_noise_reduction='near_field',            # tuned for a headset mic
    output_speed=1.1,                              # speak slightly faster
)
```

`tool_choice` and `parallel_tool_calls` are read from `model_settings` passed to
`realtime_session`. (GA realtime sessions have no `temperature`, so it is not forwarded.)

## Turn-taking and barge-in

By default the provider uses server-side voice activity detection
([`ServerVAD`][pydantic_ai.realtime.openai.ServerVAD]): it decides when the user has started and
stopped speaking, commits the audio, and triggers a response — and interrupts the model when the
user barges in. [`SemanticVAD`][pydantic_ai.realtime.openai.SemanticVAD] uses a model to decide turn
boundaries instead.

When the user barges in you get a [`SpeechStarted`][pydantic_ai.realtime.SpeechStarted] event; stop
playing any buffered model audio immediately, and call
[`interrupt`][pydantic_ai.realtime.RealtimeSession.interrupt] to cancel the model's in-progress
response. Pass `audio_end_ms` (how many milliseconds of the response the user actually heard) so the
provider truncates its stored transcript to match — otherwise the model "remembers" saying words the
user never heard:

```python {test="skip" lint="skip"}
async for event in session:
    if isinstance(event, SpeechStarted):
        speaker.flush()  # drop buffered audio locally
        await session.interrupt(audio_end_ms=speaker.played_ms())
```

`interrupt()` is server-side only — it does not flush your local playback buffer; that is the
caller's responsibility.

### Push-to-talk (manual turn-taking)

Disable automatic detection with `turn_detection=None` and drive the turn yourself: stream audio,
[`commit_audio`][pydantic_ai.realtime.RealtimeSession.commit_audio] to end the user's turn, then
[`create_response`][pydantic_ai.realtime.RealtimeSession.create_response] to ask the model to reply.
[`clear_audio`][pydantic_ai.realtime.RealtimeSession.clear_audio] discards uncommitted audio.

```python {test="skip" lint="skip"}
model = OpenAIRealtimeModel('gpt-realtime', turn_detection=None)
async with agent.realtime_session(model=model) as session:
    await session.send_audio(chunk)
    await session.commit_audio()
    await session.create_response()
```

## Images

Send an image as conversation context (for example a video frame) with
[`send_image`][pydantic_ai.realtime.RealtimeSession.send_image]. An image does not itself trigger a
response — the model picks it up on the next turn (via VAD or `create_response`).

```python {test="skip" lint="skip"}
await session.send_image(jpeg_bytes, mime_type='image/jpeg')
```

## Tool calling

Tools registered on the agent are offered to the realtime model. When the model calls one, the
session emits `ToolCallStarted`, runs the tool, sends the result back, and emits `ToolCallCompleted`.
A result is always returned to the model, even when the arguments fail to parse or the tool raises,
so the conversation never stalls.

### Background tools

By default a tool runs synchronously: the session waits for it before reading more of the model's
output. For slow tools you can run them in the **background** so the model keeps speaking while the
work happens, with the result delivered once it is ready. This mirrors firing off a subagent and
carrying on.

```python {test="skip" lint="skip"}
async with agent.realtime_session(model=model, background_tools={'deep_research'}) as session:
    ...
```

## Usage and cost

The session accumulates token usage as the model responds. Read it from
[`RealtimeSession.usage`][pydantic_ai.realtime.RealtimeSession.usage] — a
[`RunUsage`][pydantic_ai.usage.RunUsage] with input/output tokens (including audio and cached
breakdowns) and tool-call counts. Each completed response is also surfaced as a
[`Usage`][pydantic_ai.realtime.Usage] event, and providers may emit
[`RateLimits`][pydantic_ai.realtime.RateLimits] snapshots.

```python {test="skip" lint="skip"}
async with agent.realtime_session(model=model) as session:
    async for event in session:
        ...
    print(session.usage)  # cumulative tokens + tool calls for the session
```

## Observability with Logfire

Realtime sessions emit OpenTelemetry spans when the agent is instrumented — call
`logfire.instrument_pydantic_ai()` (or set `instrument=True` on the agent). You get a session span
carrying cumulative usage and, when `include_content` is enabled, the conversation transcript, with
an `execute_tool` span per tool call (including any delegated text-agent run) nested underneath. See
[Debugging and monitoring](logfire.md).

```python {test="skip" lint="skip"}
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()
# realtime_session spans now appear in Logfire
```

## Reconnecting

A long-lived connection can drop. Pass a
[`ReconnectPolicy`][pydantic_ai.realtime.openai.ReconnectPolicy] to transparently re-dial with
exponential backoff, re-apply the session configuration, and emit a
[`Reconnected`][pydantic_ai.realtime.Reconnected] event:

```python {test="skip" lint="skip"}
from pydantic_ai.realtime.openai import OpenAIRealtimeModel, ReconnectPolicy

model = OpenAIRealtimeModel('gpt-realtime', reconnect=ReconnectPolicy(max_attempts=5))
```

Reconnecting restores the session configuration but **not** server-side conversation state (the
audio buffer and prior turns) — treat a `Reconnected` event as the start of a fresh turn. Without a
policy (the default), a dropped connection surfaces as a non-recoverable
[`SessionError`][pydantic_ai.realtime.SessionError] (`recoverable=False`) and ends the stream, so the
app can restart the session itself.

## Delegating to a text agent

Realtime models do not support structured output, and are typically weaker at multi-step reasoning
than a frontier text model. The robust pattern is to keep the realtime model as the conversational
surface and expose a single tool that delegates the hard work to a normal [`Agent`][pydantic_ai.Agent]
with an `output_type`:

```python {test="skip" lint="skip"}
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.realtime.openai import OpenAIRealtimeModel


class Answer(BaseModel):
    summary: str
    confidence: float


supervisor = Agent('openai:gpt-5', output_type=Answer)
voice = Agent(instructions='Answer using the `consult` tool, then read the summary aloud.')


@voice.tool_plain
async def consult(question: str) -> str:
    result = await supervisor.run(question)
    return result.output.summary


async def main():
    async with voice.realtime_session(model=OpenAIRealtimeModel('gpt-realtime')) as session:
        ...
```

## Implementing a provider

A provider implements two ABCs: [`RealtimeModel`][pydantic_ai.realtime.RealtimeModel]
(opens a connection) and [`RealtimeConnection`][pydantic_ai.realtime.RealtimeConnection]
(sends [`RealtimeInput`][pydantic_ai.realtime.RealtimeInput] and yields
[`RealtimeEvent`][pydantic_ai.realtime.RealtimeEvent]). The OpenAI provider in
[`pydantic_ai.realtime.openai`][pydantic_ai.realtime.openai] is a reference implementation; the same
shape applies to Gemini Live, Amazon Nova Sonic, and others. Inputs a provider doesn't support
(e.g. `ImageInput`, or the manual turn-taking verbs) should raise `NotImplementedError`.
```
