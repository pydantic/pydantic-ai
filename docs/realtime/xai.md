# xAI Grok Voice

[`XaiRealtimeModel`][pydantic_ai.realtime.xai.XaiRealtimeModel] brings Grok Voice into the typed,
server-side realtime agent loop. See the [realtime overview](index.md).

## Installation

```bash
pip install "pydantic-ai-slim[realtime,xai]"
```

The provider uses WebSockets plus `xai-sdk`.

## Configuration

xAI follows the OpenAI Realtime protocol. Use `provider='xai'` (default, reads `XAI_API_KEY`) or
an [`XaiProvider`][pydantic_ai.providers.xai.XaiProvider] with `api_key=`. Custom `api_host` is
unsupported, and a provider built only with `xai_client=` cannot connect because the WebSocket
needs the API key.

### Voice and turn settings

[`XaiRealtimeModel`][pydantic_ai.realtime.xai.XaiRealtimeModel] supports the shared
[`TurnDetection`][pydantic_ai.realtime.TurnDetection] configuration:

```python
from pydantic_ai.realtime import TurnDetection
from pydantic_ai.realtime.xai import XaiRealtimeModel, XaiRealtimeModelSettings

settings = XaiRealtimeModelSettings(
    voice='eve',                                    # eve (default), ara, rex, sal, leo, or a custom ID
    turn_detection=TurnDetection(sensitivity='low'),  # or False for push-to-talk
    input_transcription_model='auto',               # the default (see Transcribing user input)
)
model = XaiRealtimeModel('grok-voice-latest', settings=settings)
```

For an exact server-VAD threshold or automatic-response behavior, use the
`xai_turn_detection=` escape hatch with [`ServerVAD`][pydantic_ai.realtime.openai.ServerVAD]; it
fully overrides `turn_detection`. The shared
realtime settings can be configured on the model or passed to `realtime()`. Grok
Voice reports the input transcript as cumulative snapshots that can correct earlier text. Rendering
[`SpeechPartDelta.transcript`][pydantic_ai.messages.SpeechPartDelta.transcript] (or
[`TranscriptUpdate.transcript`][pydantic_ai.realtime.TranscriptUpdate.transcript]) handles that with no
xAI-specific code; see [Transcribing user input](index.md#transcribing-user-input).

Input transcription defaults to `'auto'`; see
[Transcribing user input](index.md#transcribing-user-input). Usage includes audio-token buckets and
`billable_audio_seconds` in [`RunUsage.details`][pydantic_ai.usage.RunUsage.details].

### Reasoning

`grok-voice-latest` and `grok-voice-think-fast-1.0` support the shared
[`thinking`][pydantic_ai.realtime.RealtimeModelSettings.thinking] setting. The API exposes only
`'high'` and `'none'`: every enabled Pydantic AI thinking level maps to `'high'`, while `False`
maps to `'none'`. Other xAI voice models silently ignore the setting.

xAI always produces audio, so it silently ignores `output_modality='text'`.

### Native session resumption

With a [`ReconnectPolicy`][pydantic_ai.realtime.ReconnectPolicy], xAI automatically enables native
resumption, restores prior turns, and suppresses the provider replay burst locally. Its handle stays
in memory and cannot resume in another process. See [Reconnecting](index.md#reconnecting).
