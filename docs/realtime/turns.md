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
from pydantic_ai.realtime import TurnDetection
from pydantic_ai.realtime.openai import OpenAIRealtimeModel, OpenAIRealtimeModelSettings

settings = OpenAIRealtimeModelSettings(
    turn_detection=TurnDetection(sensitivity='high', silence_duration_ms=400)
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

OpenAI, Azure OpenAI, and xAI emit
[`InputSpeechStartEvent`][pydantic_ai.realtime.InputSpeechStartEvent] when user speech begins.
Gemini emits [`ResponseInterruptedEvent`][pydantic_ai.realtime.ResponseInterruptedEvent] when it
interrupts model output instead. These are the signals to flush playback.

[`interrupt()`][pydantic_ai.realtime.RealtimeSession.interrupt] handles the server-side half of the
problem. When supported, pass how many milliseconds actually played so the provider does not record
unheard words as part of the conversation:

```python {test="skip"}
from typing import Any

from pydantic_ai.realtime import InputSpeechStartEvent


async def handle_events(session: Any, speaker: Any):
    async for event in session:
        if isinstance(event, InputSpeechStartEvent) and speaker.has_unplayed_audio():
            speaker.flush()
            if session.profile['supports_output_truncation']:
                await session.interrupt(played_ms=speaker.played_ms())
            elif session.profile['supports_interruption']:
                await session.interrupt()
```

The speech-start event also occurs on ordinary user turns when nothing is playing. Track unplayed
audio before interrupting. `interrupt()` never flushes the local speaker buffer.

History records a known cutoff on
[`SpeechPart.interrupted_at_ms`][pydantic_ai.messages.SpeechPart.interrupted_at_ms] and marks the
response state as interrupted. When this history is sent to a text model, Pydantic AI adds a readable
interruption note to the prepared request without modifying stored history.

## Push-to-talk

Disable automatic detection with `turn_detection=False` on models whose profile declares
`supports_manual_turn_control`. Stream audio, call
[`commit_audio()`][pydantic_ai.realtime.RealtimeSession.commit_audio] to end the user turn, then
[`create_response()`][pydantic_ai.realtime.RealtimeSession.create_response]. Use
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

## Capability checks

Branch on [`RealtimeModelProfile`][pydantic_ai.realtime.RealtimeModelProfile] rather than provider
names:

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
- A model may speak, call a tool, and speak again. Treat
  [`TurnCompleteEvent`][pydantic_ai.realtime.TurnCompleteEvent], not the end of one speech part, as
  the exchange boundary.
- xAI can cancel output but cannot report how much audio played, so call `interrupt()` without
  `played_ms`. Gemini handles interruption server-side and exposes no explicit `interrupt()`.
