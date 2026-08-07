# Audio and transcripts

A realtime session accepts live audio, text, and supported images while exposing separate views for
playback, captions, and control events. Use the high-level session views for media and captions;
consume the main event iterator for tools, turn boundaries, reconnects, and errors.

## Audio wire contract

You send and receive raw audio samples; there is no container or codec in the live path.
[`send_audio()`][pydantic_ai.realtime.RealtimeSession.send_audio] accepts raw, signed 16-bit
little-endian mono PCM. [`stream_audio()`][pydantic_ai.realtime.RealtimeSession.stream_audio]
returns the same format. Capture at `session.profile['audio_input_sample_rate']` and play at
`session.profile['audio_output_sample_rate']`; input and output rates can differ.

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
from pydantic_ai.realtime.openai import OpenAIRealtimeModel

agent = Agent(instructions='You are a helpful voice assistant.')


async def play_audio(chunks: AsyncIterator[bytes]) -> None:
    async for chunk in chunks:
        ...  # Write the PCM16 chunk to your speaker or audio output stream.


async def show_transcripts(parts: AsyncIterator[SpeechPart]) -> None:
    async for part in parts:
        print(part.speaker, part.transcript)
        #> assistant Hello from the realtime assistant.


async def main() -> None:
    async with agent.realtime(OpenAIRealtimeModel('gpt-realtime')).session() as session:
        audio_task = asyncio.create_task(play_audio(session.stream_audio()))
        transcript_task = asyncio.create_task(show_transcripts(session.stream_transcripts()))
        async for event in session:
            if isinstance(event, RealtimeTurnCompleteEvent):
                break
        await session.close()
        await asyncio.gather(audio_task, transcript_task)
```

Each view is independently bounded; a slow consumer drops its oldest item rather than stalling
tools, turn tracking, or other consumers.
Subscriptions begin when iteration starts, so unused views do not buffer.
[`close()`][pydantic_ai.realtime.RealtimeSession.close] discards pending items and ends every live
iterator; [`closed`][pydantic_ai.realtime.RealtimeSession.closed] reports the state.

For live captions, pass `delta=True` to
[`stream_transcripts()`][pydantic_ai.realtime.RealtimeSession.stream_transcripts]. Each
[`TranscriptUpdate`][pydantic_ai.realtime.TranscriptUpdate] includes the speaker, new delta, full
transcript so far, and an index identifying the turn. Replace a caption by index rather than blindly
appending, because speech recognition can revise earlier words:

```python
from collections.abc import AsyncIterator

from pydantic_ai.realtime import TranscriptUpdate

bubbles: dict[int, tuple[str, str]] = {}


async def show_captions(updates: AsyncIterator[TranscriptUpdate]) -> None:
    async for update in updates:
        bubbles[update.index] = (update.speaker, update.transcript)
```

## Event reference

Iterating a [`RealtimeSession`][pydantic_ai.realtime.RealtimeSession] yields shared message events
from [`pydantic_ai.messages`][pydantic_ai.messages] plus realtime control events.

| Event | Meaning |
| --- | --- |
| [`PartStartEvent`][pydantic_ai.messages.PartStartEvent] | A speech, text, or tool part started. |
| [`PartDeltaEvent`][pydantic_ai.messages.PartDeltaEvent] | Incremental speech audio/transcript or text content. |
| [`PartEndEvent`][pydantic_ai.messages.PartEndEvent] | A finalized part; retained speech audio appears here, not at part start. |
| [`FunctionToolCallEvent`][pydantic_ai.messages.FunctionToolCallEvent] | A local function tool began executing. |
| [`FunctionToolResultEvent`][pydantic_ai.messages.FunctionToolResultEvent] | A local function tool completed or returned a retry prompt. |
| [`DeferredToolRequestsEvent`][pydantic_ai.messages.DeferredToolRequestsEvent] | An inline capability handler resolved deferred requests. |
| [`DeferredToolResultsEvent`][pydantic_ai.messages.DeferredToolResultsEvent] | Inline deferred results are ready for normal tool processing. |
| [`RealtimeInputSpeechStartEvent`][pydantic_ai.realtime.RealtimeInputSpeechStartEvent] | The provider detected that the user started speaking, when supported. |
| [`RealtimeInputSpeechEndEvent`][pydantic_ai.realtime.RealtimeInputSpeechEndEvent] | The provider detected the end of user speech, when supported. |
| [`RealtimeResponseInterruptedEvent`][pydantic_ai.realtime.RealtimeResponseInterruptedEvent] | The provider reported an interrupted model response. |
| [`RealtimeInputTranscriptionErrorEvent`][pydantic_ai.realtime.RealtimeInputTranscriptionErrorEvent] | One user turn could not be transcribed; the session remains usable. |
| [`RealtimeTurnCompleteEvent`][pydantic_ai.realtime.RealtimeTurnCompleteEvent] | The model finished replying and no tool remains active. |
| [`RealtimeSessionReconnectEvent`][pydantic_ai.realtime.RealtimeSessionReconnectEvent] | The connection was automatically re-established. |
| [`RealtimeSessionErrorEvent`][pydantic_ai.realtime.RealtimeSessionErrorEvent] | A recoverable provider error occurred; the session remains usable. |

Use [`RealtimeTurnCompleteEvent`][pydantic_ai.realtime.RealtimeTurnCompleteEvent] as the exchange boundary. A model
can speak, call a tool, and speak again, so receiving speech does not imply that the turn is done.

### Reading raw audio events

As an advanced alternative to `stream_audio()`, play
[`SpeechPartDelta.audio_chunk`][pydantic_ai.messages.SpeechPartDelta.audio_chunk] from raw
[`PartDeltaEvent`][pydantic_ai.messages.PartDeltaEvent]s. Model audio arrives in full whether or not
history retention is enabled. When output audio is retained, the final
[`SpeechPart`][pydantic_ai.messages.SpeechPart] contains the whole turn again as a WAV snapshot for
history; do not play both or the turn will play twice.

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
handoff; see [History and handoff](history.md#audio-retention) before relying on it.

## Images

Send an image as context with [`send()`][pydantic_ai.realtime.RealtimeSession.send]. An image does
not trigger a response by itself; the model uses it on the next voice, text, or manually-created
turn.

```python
from pydantic_ai import BinaryContent


async def send_image(session):
    jpeg_bytes = b'...'
    await session.send(BinaryContent(data=jpeg_bytes, media_type='image/jpeg'))
```

For continuous camera input, use the session's image-retention controls to bound local history; see
[Retaining images](history.md#retaining-images). Gemini-specific live-video settings belong on the
[Gemini provider page](gemini.md#settings).

## Edge cases

- Audio and transcript iterators deliberately drop old buffered items when consumers fall behind.
  [Logfire attributes](observability.md#logfire-instrumentation) report those drops.
- Provider speech/interruption signals differ. Use the profile flags and the
  [turns guide](turns.md#barge-in) rather than branching on provider names.
