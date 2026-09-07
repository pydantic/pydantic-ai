# Turns and interruptions

Realtime providers normally use voice activity detection (VAD) to decide when the user starts and
stops speaking and when the model should respond. Pydantic AI exposes a shared configuration for
portable behavior, explicit interruption for providers that support it, and manual turn control for
push-to-talk applications.

## Automatic turn detection

Automatic detection is enabled by default. Configure common behavior with
[`TurnDetection`][pydantic_ai.realtime.TurnDetection]: `sensitivity` maps to the closest provider
control, while `prefix_padding_ms` and `silence_duration_ms` pass through where supported.

```python
from pydantic_ai.realtime.openai import OpenAIRealtimeModel, OpenAIRealtimeModelSettings

settings = OpenAIRealtimeModelSettings(
    turn_detection={'sensitivity': 'high', 'silence_duration_ms': 400}
)
model = OpenAIRealtimeModel('gpt-realtime', settings=settings)
```

Use provider-specific settings only when the shared controls are insufficient:
`openai_turn_detection`, `xai_turn_detection`, and `google_vad` fully override `turn_detection`.
Their accepted values, defaults, and limitations are documented on the
[OpenAI](openai.md#settings), [Azure OpenAI](azure.md#settings),
[Google Gemini](gemini.md#settings), and [xAI](xai.md#settings) pages.

## Text turns

Sending a string creates a complete user turn and asks the model to reply:

```python
from pydantic_ai import BinaryImage
from pydantic_ai.realtime import RealtimeSession


async def send_turns(session: RealtimeSession, image: BinaryImage) -> None:
    await session.send('Greet the visitor.')

    # Add context for a later voice or text turn without asking for a reply.
    await session.send('The visitor is called Ada.', respond=False)

    # Show an image and ask for a reply in one operation.
    await session.send(image, respond=True)
```

Images are context-only by default. Asking for a response to an image requires a model that supports
manual turn control.

Do not call `create_response()` after `send('...')`: the text turn already asks for a response, so
the pair asks twice and can make the model say the same thing twice.

## Barge-in

With server-side turn detection, providers interrupt the model when they detect new user speech.
What remains is the local half of the problem: audio already queued for playback that the user
will never hear, and a provider-side transcript that would otherwise record unheard words.

When playback drains the session's single
[`stream_audio()`][pydantic_ai.realtime.RealtimeSession.stream_audio] iterator — writing each
chunk to the device before pulling the next — the session can handle that half itself. Pass
`handle_barge_in=True` when opening the session:

```python
from pydantic_ai import Agent

agent = Agent(instructions='You are a helpful voice assistant.')


async def main():
    realtime = agent.realtime('openai:gpt-realtime')
    async with realtime.session(handle_barge_in=True) as session:
        async for chunk in session.stream_audio():
            ...  # write the chunk to your speaker, waiting until the device consumed it
```

When the user speaks over the model, the session discards the buffered audio the user will never
hear, truncates the provider's transcript to what was actually played, and cancels the response —
doing nothing when the previous reply was heard in full, since the speech-start signal also fires
on ordinary user turns. A reply that has not reached its first audio chunk is still stopped, so
speaking over the model's thinking time works like speaking over its voice. Provider differences
are absorbed: on a model without output truncation
(xAI) the response is cancelled without a truncation point, and when the provider interrupts
itself without reporting speech onset (Gemini) only the local flush is performed. The events still
reach your iterator, already handled — react to them for UI state or to flush your audio layer's
own in-flight block, the one buffer the session cannot reach. The truncation point is the last
chunk boundary the device reached, so it attributes at most one chunk less than was really heard,
never more. Without that single iterator — no `stream_audio()` consumer, or several — there is no
playback position to attribute, and the flag stands down in favour of the manual paths below.

As an alternative, handle barge-in yourself. The signals: providers whose profile declares
[`emits_input_speech_events`][pydantic_ai.realtime.RealtimeModelProfile.emits_input_speech_events]
(OpenAI, Azure OpenAI, and xAI) emit
[`RealtimeInputSpeechStartEvent`][pydantic_ai.realtime.RealtimeInputSpeechStartEvent] when user speech begins.
Gemini emits [`RealtimeResponseInterruptedEvent`][pydantic_ai.realtime.RealtimeResponseInterruptedEvent] when it
interrupts model output instead. Read the flag rather than waiting on an event a provider never
sends.

While playback keeps the single device-paced iterator, staying in control of the trigger costs one
line: the session still tracks the playback position for you, as
[`played_audio_bytes`][pydantic_ai.realtime.RealtimeSession.played_audio_bytes] (a chunk counts as
played once the consumer comes back for the next one), and passing it to
[`interrupt(played_bytes=...)`][pydantic_ai.realtime.RealtimeSession.interrupt] gets the same
flush-attribute-truncate-cancel treatment as `handle_barge_in=True`:

```python
import asyncio
from collections.abc import AsyncIterator

from pydantic_ai.realtime import RealtimeInputSpeechStartEvent, RealtimeSession


async def conversation(session: RealtimeSession) -> None:
    async def play_audio(chunks: AsyncIterator[bytes]) -> None:
        async for chunk in chunks:
            ...  # write the chunk to your speaker, waiting until the device consumed it

    playback = asyncio.create_task(play_audio(session.stream_audio()))
    async for event in session:
        if isinstance(event, RealtimeInputSpeechStartEvent):
            await session.interrupt(played_bytes=session.played_audio_bytes)
    playback.cancel()
```

A playback loop that instead buffers ahead of the device makes `played_audio_bytes` read too far —
count actual device consumption yourself and pass that. This handler covers the providers that
report speech onset; on Gemini, which interrupts itself and leaves only the local flush to do,
prefer `handle_barge_in=True`, which performs that flush for you.

Interrupting between the provider's speech onset and the start of its next response sends only the
truncation on the models whose own turn detection cancels the response being spoken over (OpenAI
and Azure OpenAI by default, and xAI): a second, client-side cancel racing the provider's can be
applied to the *next* response and silence the reply to the barge-in. This holds for every form of
`interrupt()`. An interruption you raise outside that window — a stop button, a tool cutting the
model off — still cancels, since nothing else is stopping it.

Finally, when playback doesn't drain a single session-long `stream_audio()` iterator — several
consumers, a playback layer that buffers ahead of the device, or a transport where the session
never touches the audio — keep your own accounting and pass `played_ms` (or nothing). `Speaker`
here stands in for your playback layer — anything that can report and flush buffered audio:

```python
from typing import Protocol

from pydantic_ai.realtime import RealtimeInputSpeechStartEvent, RealtimeSession


class Speaker(Protocol):
    def has_unplayed_audio(self) -> bool: ...
    def flush(self) -> None: ...
    def played_ms(self) -> int: ...


async def handle_events(session: RealtimeSession, speaker: Speaker):
    async for event in session:
        if isinstance(event, RealtimeInputSpeechStartEvent) and speaker.has_unplayed_audio():
            speaker.flush()
            if session.profile.get('supports_output_truncation', False):
                await session.interrupt(played_ms=speaker.played_ms())
            elif session.profile.get('supports_interruption', False):
                await session.interrupt()
```

With `played_ms`, all of the session-side conveniences above are yours to reimplement: track
unplayed audio before interrupting, and flush buffered playback yourself — `interrupt()` with
`played_ms` never flushes.

On a [WebRTC sideband](deployment.md#browser-webrtc-server-sideband) there is a third buffer between those two: the
provider generates audio well ahead of playback and keeps streaming what it already produced, so
stopping the model is not enough to stop the voice. `interrupt()` drops that outbound buffer too,
which is what actually ends the turn for the listener. The browser still owns its own playback buffer
and should flush it on barge-in, as above.

History records a known cutoff on
[`SpeechPart.interrupted_at_ms`][pydantic_ai.messages.SpeechPart.interrupted_at_ms] and marks the
response state as interrupted. When this history is sent to a text model, Pydantic AI adds a readable
interruption note to the prepared request without modifying stored history.

## Speaking first

Send a text turn to have the agent open the conversation, with playback already running. Wait for
the greeting's finalized [`SpeechPart`][pydantic_ai.messages.SpeechPart], which arrives once it has
been generated, then let your playback loop drain before opening the microphone. A fixed sleep tells
you neither.

```python
import asyncio
from collections.abc import AsyncIterator

from pydantic_ai import Agent
from pydantic_ai.messages import SpeechPart

agent = Agent(instructions='You are a welcoming museum guide.')


async def play_audio(chunks: AsyncIterator[bytes]) -> None:
    async for chunk in chunks:
        ...  # Write the PCM16 chunk to your speaker or audio output stream.


async def wait_for_assistant_speech(parts: AsyncIterator[SpeechPart]) -> None:
    async for part in parts:
        if part.speaker == 'assistant':
            return


async def main():
    async with agent.realtime('openai:gpt-realtime').session() as session:
        playback = asyncio.create_task(play_audio(session.stream_audio()))
        greeted = asyncio.create_task(wait_for_assistant_speech(session.stream_transcripts()))
        await session.send('Greet the visitor.')
        await greeted
        ...  # wait for the speaker to drain, then open the microphone and start sending audio
    await playback  # the audio view ends once the session has closed
```

With manual turn control, [`create_response()`][pydantic_ai.realtime.RealtimeSession.create_response]
can request the greeting without adding a text turn. If a response is already active, the request is
held until that response completes and is dropped if the user barges in, so returning from
`create_response()` does not mean speech has started.

Server VAD enables `interrupt_response` by default, so any detected speech cancels a greeting in flight. This
includes speaker echo and microphone transients while the audio path opens; keeping the microphone
closed until the greeting has played avoids that race.

## Push-to-talk

Disable automatic detection with `turn_detection=False` on models whose profile declares
`supports_manual_turn_control`. Stream audio, call
[`commit_audio()`][pydantic_ai.realtime.RealtimeSession.commit_audio] to end the user turn, then
[`create_response()`][pydantic_ai.realtime.RealtimeSession.create_response]. The explicit
`create_response()` call is needed because with turn detection off, committing the buffer only
finalizes the user's input; nothing triggers a reply until you ask for one. Use
[`clear_audio()`][pydantic_ai.realtime.RealtimeSession.clear_audio] to discard uncommitted input.

```python
from pydantic_ai import Agent
from pydantic_ai.realtime.openai import OpenAIRealtimeModel, OpenAIRealtimeModelSettings

agent = Agent()
model = OpenAIRealtimeModel(
    'gpt-realtime', settings=OpenAIRealtimeModelSettings(turn_detection=False)
)


async def main():
    async with agent.realtime(model).session() as session:
        await session.send_audio(b'...')
        await session.commit_audio()
        await session.create_response()
```

Gemini does not expose manual turn verbs through Pydantic AI; `turn_detection=False` raises
[`UserError`][pydantic_ai.exceptions.UserError] before connecting.

## Checking what the model supports

These are [*model profile*](../models/overview.md#inspecting-a-models-profile) flags describing
what a provider connection can do — not to be confused with
[capabilities](../capabilities/overview.md), which add behavior to an agent. Branch on
[`RealtimeModelProfile`][pydantic_ai.realtime.RealtimeModelProfile] rather than provider names:

| Profile flag | Gates |
| --- | --- |
| `supports_manual_turn_control` | `commit_audio()`, `clear_audio()`, and `create_response()` |
| `supports_interruption` | `interrupt()` |
| `supports_output_truncation` | `interrupt(played_ms=...)` |

Calling an unsupported method raises [`UserError`][pydantic_ai.exceptions.UserError] before a
control message is sent. Current provider support is summarized on each provider page.

## Edge cases

- Push-to-talk silence usually means `commit_audio()` or `create_response()` was omitted.
- If playback triggers speech detection, add echo cancellation in the device or WebRTC layer and
  flush playback promptly on real barge-in.
