# xAI Grok Voice

[`XaiRealtimeModel`][pydantic_ai.realtime.xai.XaiRealtimeModel] brings Grok Voice into the typed,
server-side realtime agent loop. Start with the [realtime quickstart](index.md#quickstart) or the
[text-to-audio example](../examples/realtime-text-to-audio.md).

## Setup

```bash
pip install "pydantic-ai-slim[realtime,xai,openai]"
```

Set `XAI_API_KEY`. The provider uses `xai-sdk` plus the `openai` package for event types from the
OpenAI Realtime protocol that Grok Voice follows. Use `provider='xai'` or pass an
[`XaiProvider`][pydantic_ai.providers.xai.XaiProvider] with `api_key=`. Custom `api_host` is
unsupported, and a provider constructed with only `xai_client=` cannot open the WebSocket because
the connection requires the API key.

## Model names

Use a Grok Voice ID such as `grok-voice-latest` or a pinned `grok-voice-think-*` model.
`grok-voice-latest` follows xAI's current flagship and can change underneath an application; pin a
version when behavior must remain stable. Use the
[official xAI voice documentation](https://docs.x.ai/docs/guides/voice-agent) for the canonical
model list.

## Settings

[`XaiRealtimeModelSettings`][pydantic_ai.realtime.xai.XaiRealtimeModelSettings] extends the
[shared settings](index.md#shared-settings):

```python
from pydantic_ai.realtime import TurnDetection
from pydantic_ai.realtime.xai import XaiRealtimeModel, XaiRealtimeModelSettings

settings = XaiRealtimeModelSettings(
    xai_voice='eve',
    turn_detection=TurnDetection(sensitivity='low'),
    input_transcription_model='auto',
)
model = XaiRealtimeModel('grok-voice-latest', settings=settings)
```

`xai_voice` selects the provider voice; `eve` is the default. For exact server-VAD threshold or
automatic-response behavior, set `xai_turn_detection=` with
[`ServerVAD`][pydantic_ai.realtime.openai.ServerVAD]; it fully overrides shared `turn_detection`.
Set `turn_detection=False` for push-to-talk.

Input transcription defaults to `'auto'`. The provider sends cumulative transcript snapshots that
can revise earlier words, so caption UIs should render the full
[`TranscriptUpdate.transcript`][pydantic_ai.realtime.TranscriptUpdate.transcript].

### Reasoning

`grok-voice-latest` and `grok-voice-think-*` models support shared `thinking`. The provider exposes
only `'high'` and `'none'`: every enabled effort maps to `'high'`, while `False` maps to `'none'`.
Other Grok Voice models ignore the setting.

## Feature support and limitations

| Feature | Support | Notes |
| --- | --- | --- |
| Audio format | Full feature support | Mono PCM16, 24 kHz input and output |
| Text output | Unsupported | Grok Voice always produces audio |
| Image input | Unsupported | Audio/text input only |
| Manual turns | Full feature support | `turn_detection=False` plus commit/create verbs |
| Interruption | Limited parameter support | `interrupt()` works; output truncation with `played_ms` does not |
| Input transcription | Full feature support | Dedicated provider path; `'auto'` by default |
| Native tools | Unsupported | Configure local fallbacks for web capabilities |
| Usage | Full feature support | Audio-token buckets and `billable_audio_seconds` in `RunUsage.details` |
| State-restoring reconnect | Full feature support | Native resumption is automatic with a reconnect policy |

## Gateway

xAI realtime gateway routing is not currently exposed. Connect through `provider='xai'` or an
`XaiProvider`.

## Session resumption

With a [`ReconnectPolicy`][pydantic_ai.realtime.ReconnectPolicy], xAI automatically enables native
resumption, restores prior turns, and suppresses the provider's replay burst from the local event
stream. The handle stays in memory and cannot resume in another process.

## Provider-specific quirks

- Grok Voice always speaks: its profile reports `supports_text_output=False`, so `output_modality='text'`
  raises a `UserError` before connecting. Read the answer from the transcript on the `SpeechPart`.
- xAI supports cancellation but not output truncation. Flush local playback and call `interrupt()`
  without `played_ms`.
- The protocol resembles OpenAI Realtime, but feature support comes from the xAI model profile;
  avoid assuming every OpenAI behavior is available.
