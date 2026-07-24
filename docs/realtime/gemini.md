# Google Gemini Live

[`GoogleRealtimeModel`][pydantic_ai.realtime.google.GoogleRealtimeModel] connects an agent to Gemini
Live, including native audio and live image input. See the [realtime overview](index.md).

## Installation

```bash
pip install "pydantic-ai-slim[google]"
```

The provider uses `google-genai`.

## Configuration

### Live settings

[`GoogleRealtimeModel`][pydantic_ai.realtime.google.GoogleRealtimeModel] exposes Gemini Live's knobs
as optional fields, grouped by concern. Google-only generation parameters use
[`GoogleRealtimeModelSettings`][pydantic_ai.realtime.google.GoogleRealtimeModelSettings], which extends
the shared settings with `temperature`, `top_p`, `top_k`, `seed`, `google_thinking_config`, and
`google_video_resolution`.

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
    voice='Puck',
    google_language_code='en-US',
    google_affective_dialog=True,
    google_proactive_audio=True,
    google_vad=AutomaticVAD(start_sensitivity='high', end_sensitivity='low'),
    google_turn_coverage='all_video',
    google_context_compression=ContextCompression(trigger_tokens=16000, target_tokens=8000),
    google_config_overrides={'explicit_vad_signal': True},
)
model = GoogleRealtimeModel('gemini-2.5-flash-native-audio-latest', settings=settings)
```

| Field | What it does |
| --- | --- |
| `voice`, `google_language_code`, `google_multi_speaker` ([`MultiSpeaker`][pydantic_ai.realtime.google.MultiSpeaker]) | Prebuilt voice, output language, per-speaker voices |
| `google_affective_dialog`, `google_proactive_audio` | Emotion-aware delivery; let the model decide when to speak (native-audio models) |
| `google_vad` ([`AutomaticVAD`][pydantic_ai.realtime.google.AutomaticVAD]) | Finer Gemini-specific VAD control: `disabled`, separate start/end sensitivity, padding/silence; fully overrides `turn_detection` |
| `google_activity_handling`, `google_turn_coverage` | Whether activity interrupts; which input a turn covers (`activity_only`/`all_input`/`all_video`) |
| `google_input_transcription`, `google_output_transcription`, `google_transcription_language_codes` | Transcription on/off and language hints |
| `google_context_compression` ([`ContextCompression`][pydantic_ai.realtime.google.ContextCompression]) | Sliding-window compression for long sessions |
| `google_enable_session_resumption`, `reconnect` | Transparent resume on a dropped connection (see [Reconnecting](index.md#reconnecting)) |
| `google_config_overrides` | Raw keys merged last into the `LiveConnectConfig` — forward-compat escape hatch |

!!! warning "Keep automatic VAD enabled"
    Do not set `AutomaticVAD(disabled=True)` through `RealtimeSession`: Pydantic AI does not expose
    Gemini activity markers or manual turn controls. Use automatic VAD; `turn_detection=False` is
    rejected for the same reason.

Authentication comes from the `provider` argument, mirroring
[`GoogleModel`][pydantic_ai.models.google.GoogleModel]: pass `provider='google'` (the default,
Gemini Developer API) or `provider='google-cloud'` for Vertex AI / ADC, or a
[`GoogleProvider`][pydantic_ai.providers.google.GoogleProvider] instance for a custom key or client,
or a [`GoogleCloudProvider`][pydantic_ai.providers.google_cloud.GoogleCloudProvider] instance for
custom Google Cloud credentials, project, or region.

Gemini uses `google_input_transcription`, enabled by default; there is no separate transcription
model. See [Transcribing user input](index.md#transcribing-user-input).

### Routing through the gateway

Route a session through the [Pydantic AI Gateway](../gateway.md) by naming the upstream provider. The
gateway credentials come from [`gateway_provider`][pydantic_ai.providers.gateway.gateway_provider]:

```python
from pydantic_ai.realtime.google import GoogleRealtimeModel

model = GoogleRealtimeModel('gemini-live-2.5-flash', provider='gateway/google')
```

Or infer it from a gateway-prefixed identifier:

```python
from pydantic_ai.realtime import infer_realtime_model

model = infer_realtime_model('gateway/google:gemini-live-2.5-flash')
```

The gateway proxies Gemini Live over **Vertex** (the `google-vertex` route), so the provider row it
routes to must be in a region that supports the Live API. `gateway/google-cloud` is an alias for the
same route. See the shared [gateway notes](index.md#gateway) for trace propagation over the handshake.

### Session resumption

Set `google_enable_session_resumption=True` together with
[`ReconnectPolicy`][pydantic_ai.realtime.ReconnectPolicy]. The session re-dials after a drop from
the latest server handle and emits `state_restored=True`. See
[Reconnecting](index.md#reconnecting).

!!! note "Proactive resume before the connection cap is not yet supported"
    Gemini Live closes a connection at its session-duration cap and sends a `GoAway` warning shortly
    before doing so. Pydantic AI reconnects on the drop (restoring prior turns from the resumption
    handle), but does not yet resume *proactively* on `GoAway`, so a long call can briefly drop mid-turn
    at the cap. Tracked in [#6643](https://github.com/pydantic/pydantic-ai/issues/6643).
