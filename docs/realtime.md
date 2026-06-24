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

You stream audio in with [`send_audio`][pydantic_ai.realtime.RealtimeSession.send_audio] (or text
with [`send_text`][pydantic_ai.realtime.RealtimeSession.send_text]) and consume events by iterating
the session.

## Events

Iterating a [`RealtimeSession`][pydantic_ai.realtime.RealtimeSession] yields:

| Event | Meaning |
| --- | --- |
| [`AudioDelta`][pydantic_ai.realtime.AudioDelta] | A chunk of the model's audio output. |
| [`Transcript`][pydantic_ai.realtime.Transcript] | Transcription of the model's speech (`is_final` marks the end of a turn). |
| [`InputTranscript`][pydantic_ai.realtime.InputTranscript] | Transcription of the user's speech. |
| [`ToolCallStarted`][pydantic_ai.realtime.ToolCallStarted] | The session began executing a tool the model requested. |
| [`ToolCallCompleted`][pydantic_ai.realtime.ToolCallCompleted] | The tool finished and its result was sent back to the model. |
| [`TurnComplete`][pydantic_ai.realtime.TurnComplete] | The model finished a turn (`interrupted=True` if the user barged in). |
| [`SessionError`][pydantic_ai.realtime.SessionError] | The provider reported an error. |

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

## Interruptions

Realtime providers detect when the user starts speaking and stop the current response. The turn that
was cut off arrives as `TurnComplete(interrupted=True)`; stop playback of any buffered audio when you
see it.

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
shape applies to Gemini Live, Amazon Nova Sonic, and others.
```
