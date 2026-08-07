# Google Gemini Live

[`GoogleRealtimeModel`][pydantic_ai.realtime.google.GoogleRealtimeModel] connects an agent to Gemini
Live, including native audio, live images, and provider-native tools. Start with the
[realtime quickstart](index.md#quickstart) or [camera example](../examples/realtime-camera.md).

## Setup

```bash
pip install "pydantic-ai-slim[realtime,google]"
```

Authentication comes from `provider`, mirroring
[`GoogleModel`][pydantic_ai.models.google.GoogleModel]. Use `provider='google'` for the Gemini
Developer API or `provider='google-cloud'` for Vertex AI/ADC. Pass a
[`GoogleProvider`][pydantic_ai.providers.google.GoogleProvider] or
[`GoogleCloudProvider`][pydantic_ai.providers.google_cloud.GoogleCloudProvider] for custom
credentials, project, region, or client.

## Model names

Use a Gemini Live model ID, for example `gemini-2.5-flash-native-audio-latest` or
`gemini-3.1-flash-live-preview`. Native-audio and other Live models differ in thinking,
asynchronous tools, and output behavior. Use the
[official Gemini Live documentation](https://ai.google.dev/gemini-api/docs/live) as the canonical
model and availability source.

The profile describes the speech-to-speech Live models. Gemini also serves specialist streaming
models on the same endpoint that behave differently — `gemini-robotics-er-2-streaming-preview`, for
instance, is text-only and rejects audio output. Point a session at one of those and correct the
facts with [`profile=`](index.md#provider-support), e.g.
`GoogleRealtimeModel('gemini-robotics-er-2-streaming-preview', profile={'supports_text_output': True})`.

## Settings

[`GoogleRealtimeModelSettings`][pydantic_ai.realtime.google.GoogleRealtimeModelSettings] extends the
[shared settings](index.md#shared-settings) with Google generation and Live controls:

```python
from pydantic_ai.realtime.google import (
    AutomaticVAD,
    ContextCompression,
    GoogleRealtimeModel,
    GoogleRealtimeModelSettings,
)

settings = GoogleRealtimeModelSettings(
    temperature=0.7,
    top_p=0.9,
    google_voice='Puck',
    google_language_code='en-US',
    google_affective_dialog=True,
    google_proactive_audio=True,
    google_vad=AutomaticVAD(start_sensitivity='high', end_sensitivity='low'),
    google_turn_coverage='all_video',
    google_context_compression=ContextCompression(trigger_tokens=16000, target_tokens=8000),
)
model = GoogleRealtimeModel('gemini-2.5-flash-native-audio-latest', settings=settings)
```

| Setting | Purpose |
| --- | --- |
| `google_voice`, `google_language_code`, `google_multi_speaker` | Voice, output language, and per-speaker voices |
| `google_affective_dialog`, `google_proactive_audio` | Emotion-aware delivery and model-decided speech on native-audio models |
| `google_vad` | Exact automatic VAD; fully overrides shared `turn_detection` |
| `google_activity_handling`, `google_turn_coverage` | Interruption behavior and which input belongs to a turn |
| `google_input_transcription`, `google_output_transcription` | Native transcription switches, enabled by default |
| `google_context_compression` | Sliding-window compression for long sessions |
| `google_enable_session_resumption` | Native state restoration with a reconnect policy |
| `google_async_tool_calls` | Lets supported native-audio models continue speaking during tools |
| `google_config_overrides` | Raw `LiveConnectConfig` keys merged last as a forward-compatibility escape hatch |

`google_voice` is the provider voice setting. `google_thinking_config` takes precedence over shared
`thinking` when a token budget or other Gemini-specific control is needed.

!!! warning "Keep automatic VAD enabled"
    Pydantic AI does not expose Gemini activity markers or manual turn verbs. Do not set
    `AutomaticVAD(disabled=True)`; shared `turn_detection=False` is rejected for the same reason.

### Asynchronous tool calls

Gemini normally pauses generation while a function tool is outstanding. Set
`google_async_tool_calls=True` on supported native-audio models to let it continue speaking. This is
best for slow tools; a fast result can interrupt speech that barely started and leave an empty
interrupted turn in history. Other Live models ignore the setting.

### Native tools

Gemini Live maps [`WebSearch`][pydantic_ai.capabilities.WebSearch] to Google Search grounding, the
only native tool it supports — no Live model runs native code execution or URL context, so neither
[`CodeExecutionTool`][pydantic_ai.native_tools.CodeExecutionTool] nor
[`WebFetch`][pydantic_ai.capabilities.WebFetch] is advertised in `supported_native_tools`. Give
those a `local=` fallback and the session runs the local tool instead: `CodeExecutionTool(local=...)`,
or `WebFetch(native=False, local=True)`, which requires the `web-fetch` optional group
(`pip install "pydantic-ai-slim[realtime,google,web-fetch]"`).

Gemini 2.5 also cannot combine native Google Search grounding with function tools; choose native
grounding or local function-tool fallbacks unless using a model that supports the combination.

## Feature support and limitations

| Feature | Support | Notes |
| --- | --- | --- |
| Audio format | Full feature support | Mono PCM16, 16 kHz input and 24 kHz output |
| Text output | Unsupported | Every speech-to-speech Live model rejects a `TEXT` response modality, so `output_modality='text'` raises. Read the answer from the transcript on the `SpeechPart` |
| Image/live video input | Full feature support | `google_turn_coverage='all_video'` keeps streamed frames in context |
| Manual turns | Unsupported | Automatic VAD is required |
| Explicit interruption/truncation | Unsupported | Gemini interrupts server-side and emits `RealtimeResponseInterruptedEvent` |
| Input transcription | Full feature support | Native transcription, enabled by default; no separate model ID |
| Native tools | Limited parameter support | Google Search grounding only; URL context and code execution fall back to a `local=` tool (see above) |
| Usage | Full feature support | Token and modality breakdowns; function-call usage may arrive on a later turn |
| State-restoring reconnect | Full feature support | Requires session resumption plus a reconnect policy |

## Gateway

Use `provider='gateway/google'` or infer a gateway-prefixed model:

```python
from pydantic_ai.realtime import infer_realtime_model

model = infer_realtime_model('gateway/google:gemini-live-2.5-flash')
```

The gateway proxies Gemini Live through the Vertex upstream, so configure a region that supports
the Live API. `gateway/google-cloud` is an alias. See
[Gateway trace propagation](observability.md#gateway-trace-propagation).

## Session resumption

Enable both `google_enable_session_resumption=True` and a
[`ReconnectPolicy`][pydantic_ai.realtime.ReconnectPolicy]. Reconnection uses the latest in-memory
server handle and emits `state_restored=True`.

!!! note "The connection cap can briefly interrupt a turn"
    Gemini sends `GoAway` shortly before its provider-defined connection cap. Pydantic AI reconnects
    after the drop rather than proactively on `GoAway`, so a long call can briefly drop mid-turn.
    This is tracked in [#6643](https://github.com/pydantic/pydantic-ai/issues/6643).

## Provider-specific quirks

- Gemini reports response interruption but not user speech-start/end events, so local playback is
  flushed on `RealtimeResponseInterruptedEvent`, and Gemini sessions record no `user speech` span (see
  [Logfire instrumentation](observability.md#logfire-instrumentation)).
- Seeded function calls/results are represented as readable text because Live cannot accept
  function parts in seeded turns.
- Native transcription can produce only a completed sentence on some models. Caption UIs should
  replace text from `TranscriptUpdate.transcript` rather than assume incremental deltas.
