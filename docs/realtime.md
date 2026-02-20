# Realtime Sessions

Realtime sessions provide bidirectional streaming connections to models that support live audio (and optionally video/image) input and output. Unlike the standard [`Agent.run()`][pydantic_ai.Agent.run] interface which uses request-response, realtime sessions maintain a persistent connection where you continuously feed content in and receive events out.

This is useful for building voice assistants, live transcription, and interactive multimodal applications.

## Installation

For OpenAI Realtime (WebSocket):

```bash
pip install "pydantic-ai-slim[realtime]"
```

For Gemini Live (uses the `google-genai` SDK):

```bash
pip install "pydantic-ai-slim[google]"
```

## Quick Start

```python
from pydantic_ai import Agent
from pydantic_ai.realtime import AudioDelta, Transcript, ToolCallCompleted, TurnComplete
from pydantic_ai.realtime.openai import OpenAIRealtimeModel

agent = Agent()


@agent.tool_plain
def get_weather(city: str) -> str:
    return f'Sunny in {city}'


async def main():
    model = OpenAIRealtimeModel('gpt-4o-realtime-preview')
    async with agent.realtime_session(model=model) as session:
        # Send audio data (e.g. from a microphone)
        await session.send_audio(b'...')

        async for event in session:
            match event:
                case AudioDelta(data=data):
                    # Play audio response
                    ...
                case Transcript(text=text):
                    # Display transcript
                    ...
                case ToolCallCompleted(tool_name=name, result=result):
                    # Tool was auto-executed
                    ...
                case TurnComplete():
                    break
```

## Input Types

Content is sent to the model via the [`send()`][pydantic_ai.realtime.RealtimeSession.send] method using typed input dataclasses. Each provider accepts a specific subset of input types:

| Input type | Description | OpenAI | Gemini Live | Nova Sonic |
|---|---|---|---|---|
| [`AudioInput`][pydantic_ai.realtime.AudioInput] | Raw audio chunk | Yes | Yes | Yes |
| [`ImageInput`][pydantic_ai.realtime.ImageInput] | Image frame | No | Yes | No |
| [`ToolResult`][pydantic_ai.realtime.ToolResult] | Tool call response | Yes | Yes | Yes |

Sending an unsupported input type raises `NotImplementedError` at runtime. When using a specific provider class directly, type checkers will also flag the mismatch at check time.

A convenience [`send_audio()`][pydantic_ai.realtime.RealtimeSession.send_audio] method is available for the most common case:

```python
# These are equivalent:
await session.send_audio(audio_bytes)
await session.send(AudioInput(data=audio_bytes))
```

## Event Types

Events are yielded when iterating over a session. The session intercepts tool calls and auto-executes them, emitting `ToolCallStarted` and `ToolCallCompleted` in their place.

| Event | Description |
|---|---|
| [`AudioDelta`][pydantic_ai.realtime.AudioDelta] | A chunk of audio from the model |
| [`Transcript`][pydantic_ai.realtime.Transcript] | Model output transcription (partial or final) |
| [`InputTranscript`][pydantic_ai.realtime.InputTranscript] | Transcription of user's speech input |
| [`ToolCallStarted`][pydantic_ai.realtime.ToolCallStarted] | Agent began executing a tool call |
| [`ToolCallCompleted`][pydantic_ai.realtime.ToolCallCompleted] | Agent finished executing a tool call |
| [`TurnComplete`][pydantic_ai.realtime.TurnComplete] | The model finished its turn |
| [`SessionError`][pydantic_ai.realtime.SessionError] | An error occurred in the session |

## Tool Integration

When using [`Agent.realtime_session()`][pydantic_ai.Agent.realtime_session], the agent's registered tools are automatically wired into the session. When the model requests a tool call, the session:

1. Yields a `ToolCallStarted` event
2. Executes the tool using the agent's toolset
3. Sends the result back to the model
4. Yields a `ToolCallCompleted` event

This happens transparently - you don't need to handle `ToolCall` events yourself.

!!! note "Dynamic instructions"
    Only static string instructions are supported in realtime sessions. Dynamic instructions registered via `@agent.instructions` are not used - pass explicit instructions via the `instructions` parameter instead.

## Providers

### OpenAI Realtime

Uses WebSocket to connect to the [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime).

- **Audio format**: 24kHz PCM16 mono, little-endian (input and output)
- **Input**: audio and tool results only
- **Models**: `gpt-4o-realtime-preview` and variants

```python
from pydantic_ai.realtime.openai import OpenAIRealtimeModel

model = OpenAIRealtimeModel(
    model='gpt-4o-realtime-preview',
    voice='alloy',  # optional: alloy, echo, shimmer, etc.
)
```

### Gemini Live

Uses the [Gemini Live API](https://ai.google.dev/gemini-api/docs/live) via the `google-genai` SDK.

- **Audio format**: 16kHz PCM16 mono input, 24kHz PCM16 mono output
- **Input**: audio, images, and tool results
- **Models**: `gemini-2.5-flash-native-audio-preview` and variants

#### Google AI (API key)

```python
from pydantic_ai.realtime.gemini import GeminiRealtimeModel

model = GeminiRealtimeModel(
    model='gemini-2.5-flash-native-audio-preview',
    voice='Kore',  # optional: Kore, Puck, Charon, etc.
)
```

Set the `GOOGLE_API_KEY` or `GEMINI_API_KEY` environment variable, or pass `api_key` directly.

#### Vertex AI

```python
from pydantic_ai.realtime.gemini import GeminiRealtimeModel

model = GeminiRealtimeModel(
    model='gemini-2.5-flash-native-audio-preview',
    project='my-gcp-project',
    location='us-central1',
)
```

### Bedrock Nova Sonic

_Coming soon._ Uses [Amazon Bedrock bidirectional streaming](https://docs.aws.amazon.com/nova/latest/userguide/speech-bidirection.html) via HTTP/2.

- **Audio format**: 16kHz PCM16 mono input, 24kHz output
- **Input**: audio and tool results only
- **Models**: `amazon.nova-sonic-v1:0`, `amazon.nova-sonic-v2:0`

## Using RealtimeConnection Directly

For advanced use cases where you want manual control over tool execution, you can use a `RealtimeModel` directly without the agent wrapper:

```python
from pydantic_ai.realtime import AudioInput, ToolCall, ToolResult
from pydantic_ai.realtime.openai import OpenAIRealtimeModel

model = OpenAIRealtimeModel()

async with model.connect(instructions='Be helpful') as conn:
    await conn.send(AudioInput(data=audio_bytes))

    async for event in conn:
        if isinstance(event, ToolCall):
            # Handle tool calls manually
            result = my_tool_handler(event.tool_name, event.args)
            await conn.send(ToolResult(tool_call_id=event.tool_call_id, output=result))
```

At this level you receive raw `ToolCall` events and are responsible for executing tools and sending results back.
