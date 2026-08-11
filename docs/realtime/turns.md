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

## Barge-in

With server-side turn detection, providers interrupt the model when they detect new user speech.
Your application still owns audio already queued for playback and must flush that local buffer.

Providers whose profile declares
[`emits_input_speech_events`][pydantic_ai.realtime.RealtimeModelProfile.emits_input_speech_events]
(OpenAI, Azure OpenAI, and xAI) emit
[`RealtimeInputSpeechStartEvent`][pydantic_ai.realtime.RealtimeInputSpeechStartEvent] when user speech begins.
Gemini emits [`RealtimeResponseInterruptedEvent`][pydantic_ai.realtime.RealtimeResponseInterruptedEvent] when it
interrupts model output instead. These are the signals to flush playback; read the flag rather than
waiting on an event a provider never sends.

[`interrupt()`][pydantic_ai.realtime.RealtimeSession.interrupt] handles the server-side half of the
problem. When supported, pass how many milliseconds actually played so the provider does not record
unheard words as part of the conversation. `Speaker` here stands in for your playback layer —
anything that can report and flush buffered audio:

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
            if session.profile['supports_output_truncation']:
                await session.interrupt(played_ms=speaker.played_ms())
            elif session.profile['supports_interruption']:
                await session.interrupt()
```

The speech-start event also occurs on ordinary user turns when nothing is playing. Track unplayed
audio before interrupting. `interrupt()` never flushes the local speaker buffer.

On a [WebRTC sideband](deployment.md#browser-webrtc-server-sideband) there is a third buffer between those two: the
provider generates audio well ahead of playback and keeps streaming what it already produced, so
stopping the model is not enough to stop the voice. `interrupt()` drops that outbound buffer too,
which is what actually ends the turn for the listener. The browser still owns its own playback buffer
and should flush it on barge-in, as above.

History records a known cutoff on
[`SpeechPart.interrupted_at_ms`][pydantic_ai.messages.SpeechPart.interrupted_at_ms] and marks the
response state as interrupted. When this history is sent to a text model, Pydantic AI adds a readable
interruption note to the prepared request without modifying stored history.

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
