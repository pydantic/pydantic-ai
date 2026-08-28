# Audio, images, and transcripts

A realtime session accepts live audio, text, and supported images while exposing separate views for
playback and captions. Use the high-level session views for media and transcripts; consume the main
[event stream](events.md) for tools, turn boundaries, reconnects, and errors.

## Audio wire contract

You send and receive raw audio samples; there is no container or codec in the live path.
[`send_audio()`][pydantic_ai.realtime.RealtimeSession.send_audio] accepts raw, signed 16-bit
little-endian mono PCM — a single chunk, or an async iterable of chunks (a microphone stream, a
WebSocket receive loop) that it forwards until the iterable ends, so a whole capture loop can be
one task. [`stream_audio()`][pydantic_ai.realtime.RealtimeSession.stream_audio]
returns the same format. Capture at
[`session.audio_input_sample_rate`][pydantic_ai.realtime.RealtimeSession.audio_input_sample_rate]
and play at
[`session.audio_output_sample_rate`][pydantic_ai.realtime.RealtimeSession.audio_output_sample_rate];
input and output rates can differ.

Start with 100 ms input chunks to balance interactive cadence with per-chunk overhead, then tune for
your transport. The provider pages list their model-specific rates and constraints:
[OpenAI](openai.md#feature-support-and-limitations),
[Azure OpenAI](azure.md#feature-support-and-limitations),
[Google Gemini](gemini.md#feature-support-and-limitations), and
[xAI](xai.md#feature-support-and-limitations).

For a complete microphone and speaker loop with bounded buffers, playback accounting, and clean
shutdown, use the [realtime voice assistant example](../examples/realtime-voice.md).

## Consuming audio and transcripts

Run media views alongside the main iterator:

```python
import asyncio
from collections.abc import AsyncIterator

from pydantic_ai import Agent
from pydantic_ai.messages import SpeechPart
from pydantic_ai.realtime import RealtimeTurnCompleteEvent

agent = Agent(instructions='You are a helpful voice assistant.')


async def play_audio(chunks: AsyncIterator[bytes]) -> None:
    async for chunk in chunks:
        ...  # Write the PCM16 chunk to your speaker or audio output stream.


async def show_transcripts(parts: AsyncIterator[SpeechPart]) -> None:
    async for part in parts:
        print(part.speaker, part.transcript)
        #> assistant Hello from the realtime assistant.


async def main():
    async with agent.realtime('openai:gpt-realtime').session() as session:
        audio_task = asyncio.create_task(play_audio(session.stream_audio()))
        transcript_task = asyncio.create_task(show_transcripts(session.stream_transcripts()))
        async for event in session:
            if isinstance(event, RealtimeTurnCompleteEvent):
                break

    # Leaving the `async with` block closes the session, which ends every live view.
    await asyncio.gather(audio_task, transcript_task)
```

Each view is independently bounded; a slow consumer drops its oldest item rather than stalling
tools, turn tracking, or other consumers.
Subscriptions begin when iteration starts, so unused views do not buffer.
[`close()`][pydantic_ai.realtime.RealtimeSession.close] discards pending items and ends every live
iterator; [`closed`][pydantic_ai.realtime.RealtimeSession.closed] reports the state.

### Live captions

For live captions, pass `delta=True` to
[`stream_transcripts()`][pydantic_ai.realtime.RealtimeSession.stream_transcripts]. Each
[`TranscriptUpdate`][pydantic_ai.realtime.TranscriptUpdate] includes the speaker, new delta, full
transcript so far, and an index identifying the turn. Replace a caption by index rather than blindly
appending, because speech recognition can revise earlier words:

```python
from pydantic_ai.realtime import RealtimeSession

bubbles: dict[int, tuple[str, str]] = {}


async def show_captions(session: RealtimeSession) -> None:
    async for update in session.stream_transcripts(delta=True):
        bubbles[update.index] = (update.speaker, update.transcript)
```

## Input transcription

The shared `input_transcription_model` setting controls whether user speech reaches history as text:

| Value | Behavior |
| --- | --- |
| `'auto'` (default) | Uses the provider's recommended transcription path. |
| A model ID | Pins a dedicated transcription model on providers that support one. |
| `None` | Disables input transcription. |

OpenAI, Azure OpenAI, and xAI use dedicated transcription models. Gemini uses native transcription,
configured with `google_input_transcription`: a pinned model ID in the shared setting is ignored
(native transcription stays on), and only `None` turns it off. Provider-specific defaults and
deployment constraints live on the provider pages.

Disabling transcription changes what a spoken turn contributes to history, replay, and text-agent
handoff; see [History and handoff](history.md#retaining-audio) before relying on it. A
[WebRTC sideband](deployment.md#browser-webrtc-server-sideband) receives no audio bytes to retain, so without input
transcription its user turns contain no spoken text.

## Images

Beyond audio and text, a session accepts the same image content as
[multimodal input](../input.md#image-input) to a standard run. Send an image as context with
[`send()`][pydantic_ai.realtime.RealtimeSession.send]. An image does not trigger a response by
itself; the model uses it on the next voice, text, or manually-created turn.

```python
from pydantic_ai import BinaryContent


async def send_image(session):
    jpeg_bytes = b'...'
    await session.send(BinaryContent(data=jpeg_bytes, media_type='image/jpeg'))
```

Streaming images continuously approximates live video: the
[camera example](../examples/realtime-camera.md) sends one camera frame per second alongside
microphone audio. For continuous streams like that, use the session's image-retention controls to
bound local history; see [Retaining images](history.md#retaining-images). Gemini-specific live-video
settings belong on the [Gemini provider page](gemini.md#settings).

## Edge cases

- Audio and transcript iterators deliberately drop old buffered items when consumers fall behind.
  [Logfire attributes](observability.md#logfire-instrumentation) report those drops.
- Provider speech/interruption signals differ. Use the profile flags and the
  [turns guide](turns.md#barge-in) rather than branching on provider names.
