# Realtime (speech-to-speech)

Some providers offer **realtime** models that exchange audio over a persistent bidirectional
WebSocket instead of the request-response pattern of the standard
[`Model`][pydantic_ai.models.Model] interface. These models listen and speak at the same time,
detect when the user starts talking (so the model can be interrupted), and can call tools mid
conversation.

Pydantic AI exposes this through a small provider-agnostic layer in
[`pydantic_ai.realtime`][pydantic_ai.realtime], with [`Agent.realtime_session`][pydantic_ai.agent.Agent.realtime_session]
as the high-level entry point: it reuses the agent's tools and instructions and runs the tool-call
loop for you.

!!! note "Realtime is separate from the text `Model`"
    A realtime session is not a `Model` request. It opens a connection you stream audio into and
    iterate events out of. Use a realtime model for the conversational surface, and a normal text
    [`Agent`][pydantic_ai.Agent] when you need structured output or heavier reasoning (see
    [Delegating to a text agent](#delegating-to-a-text-agent)).

!!! tip "Prefer to see it running first?"
    The [voice assistant example](examples/realtime-voice.md) is a complete, runnable script —
    microphone in, speakers out, one tool. Come back here for the *why*.

!!! note "Connecting a browser or other frontend"
    A session runs on *your* server, not in the browser: audio flows **browser ⟷ your app server
    ⟷ provider**. Your frontend streams microphone audio to your server (e.g. over a WebSocket),
    which feeds it into the session with [`session.send_audio()`][pydantic_ai.realtime.RealtimeSession.send_audio]
    and streams the session's audio events back out. The [camera example](examples/realtime-camera.md)
    is a full browser-to-server demo (mic, speaker, and live video frames over a WebSocket). Direct
    browser-to-provider transport (WebRTC) is listed under [Limitations](#limitations).

## Installation

The OpenAI provider ([`OpenAIRealtimeModel`][pydantic_ai.realtime.openai.OpenAIRealtimeModel]) uses
WebSockets plus the `openai` client, available via the `realtime` and `openai` optional groups:

```bash
pip install "pydantic-ai-slim[realtime,openai]"
```

Authentication and the base URL come from the `provider` argument, mirroring
[`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel]: pass `provider='openai'` (the
default, reads `OPENAI_API_KEY`) or an [`OpenAIProvider`][pydantic_ai.providers.openai.OpenAIProvider]
instance for a custom key or base URL. The realtime transport is opened separately with
`websockets`, so a custom `OpenAIProvider` `httpx` client is not used for the WebSocket connection.
OpenAI-compatible endpoints that expose a realtime API work too. Azure OpenAI's separate realtime
endpoint is not supported through `OpenAIRealtimeModel`; use Azure AI Voice Live below.

### Azure AI Voice Live

[Azure AI Voice Live](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/voice-live)
uses WebSockets and is available through the `realtime` optional group:

```bash
pip install "pydantic-ai-slim[realtime]"
```

Set `AZURE_VOICELIVE_ENDPOINT`, `AZURE_VOICELIVE_API_VERSION`, and
`AZURE_VOICELIVE_API_KEY`, then use the `azure-voicelive:` model prefix:

```python {test="skip"}
from pydantic_ai import Agent

agent = Agent(instructions='You are a helpful voice assistant.')


async def main():
    async with agent.realtime_session(model='azure-voicelive:gpt-realtime') as session:
        await session.send('Say hello.')
        async for event in session:
            ...
```

Or initialise the model and provider directly:

```python {test="skip"}
from pydantic_ai.providers.azure_voicelive import AzureVoiceLiveProvider
from pydantic_ai.realtime.azure import AzureRealtimeModel

model = AzureRealtimeModel(
    'gpt-realtime',
    provider=AzureVoiceLiveProvider(
        endpoint='https://your-resource.services.ai.azure.com',
        api_version='2026-04-10',
        api_key='your-api-key',
    ),
)
```

Voice Live offers managed models rather than requiring an Azure OpenAI deployment. See the official
[model and region matrix](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/regions?tabs=voice-live)
for currently available model identifiers. [`AzureVoiceLiveProvider`][pydantic_ai.providers.azure_voicelive.AzureVoiceLiveProvider]
uses the documented Azure `api-key` WebSocket header; Microsoft Entra ID authentication is not yet
exposed by this provider.

[`AzureRealtimeModel`][pydantic_ai.realtime.azure.AzureRealtimeModel] supports the shared realtime
settings and maps a string `voice` to an OpenAI voice on Voice Live. Azure-specific standard/custom
voices, noise suppression, echo cancellation, avatars, and animation are not yet exposed as model
settings.

The Gemini provider ([`GoogleRealtimeModel`][pydantic_ai.realtime.google.GoogleRealtimeModel]) uses
the `google-genai` SDK, available via the `google` optional group:

```bash
pip install "pydantic-ai-slim[google]"
```

The xAI Grok Voice provider ([`XaiRealtimeModel`][pydantic_ai.realtime.xai.XaiRealtimeModel]) uses
WebSockets plus the `xai-sdk` client (for [`XaiProvider`][pydantic_ai.providers.xai.XaiProvider]),
available via the `realtime` and `xai` optional groups:

```bash
pip install "pydantic-ai-slim[realtime,xai]"
```

xAI's realtime API is a clone of the OpenAI Realtime protocol. Use `provider='xai'` (the default,
which reads `XAI_API_KEY`) or an [`XaiProvider`][pydantic_ai.providers.xai.XaiProvider] constructed
with `api_key=`. Realtime does not support a custom `api_host`, and a provider constructed only with
`xai_client=` cannot be used because the WebSocket connection needs access to the API key.

All four implement the same [`RealtimeModel`][pydantic_ai.realtime.RealtimeModel] interface, so the
rest of this guide applies to any of them — swap `OpenAIRealtimeModel('gpt-realtime')` for
`AzureRealtimeModel('gpt-realtime')`, `GoogleRealtimeModel('gemini-2.5-flash-native-audio-latest')`,
or `XaiRealtimeModel('grok-voice-latest')`. A few provider differences are worth knowing: send mono
PCM16 input at **16 kHz** for Gemini and **24 kHz** for OpenAI, Azure Voice Live, and xAI; all four
produce mono PCM16 output at **24 kHz**. Gemini produces a single
response modality per session and natively accepts **live video frames**
sent as [`ImageInput`][pydantic_ai.realtime.ImageInput] (stream camera/screen frames with
[`send`][pydantic_ai.realtime.RealtimeSession.send] for "show me this" interactions). xAI
Grok Voice supports cancellation-based barge-in but not output truncation (see
[model profile](#model-profile)). xAI reports response usage, including audio-token buckets and
`billable_audio_seconds` in [`RunUsage.details`][pydantic_ai.usage.RunUsage.details].

## Quickstart

```python {test="skip" lint="skip"}
from pydantic_ai import Agent
from pydantic_ai.messages import SpeechPart, SpeechPartDelta
from pydantic_ai.realtime import PartDeltaEvent, PartEndEvent
from pydantic_ai.realtime.openai import OpenAIRealtimeModel

agent = Agent(instructions='You are a helpful voice assistant.')


@agent.tool_plain
def get_weather(city: str) -> str:
    return f'Sunny in {city}'


async def main():
    model = OpenAIRealtimeModel('gpt-realtime')
    async with agent.realtime_session(model=model) as session:
        await session.send_audio(microphone_chunk)  # PCM16 audio bytes
        async for event in session:
            match event:
                case PartDeltaEvent(delta=SpeechPartDelta(audio_chunk=chunk)) if chunk:
                    speaker.play(chunk)
                case PartEndEvent(part=SpeechPart(speaker='assistant', transcript=transcript)):
                    print('assistant:', transcript)
```

You stream content in with the session's `send_*` helpers and consume events by iterating the
session:

| Method | Sends |
| --- | --- |
| [`send_audio`][pydantic_ai.realtime.RealtimeSession.send_audio] | A chunk of raw mono PCM16 microphone audio at the model profile's input sample rate. |
| [`send`][pydantic_ai.realtime.RealtimeSession.send] | Plain text, image/audio [`BinaryContent`][pydantic_ai.messages.BinaryContent], a typed [`RealtimeSessionInput`][pydantic_ai.realtime.RealtimeSessionInput], or a sequence of these. |

## Events

A session speaks the same event vocabulary as a standard streamed agent run: iterating a
[`RealtimeSession`][pydantic_ai.realtime.RealtimeSession] yields the shared message/part events from
[`pydantic_ai.messages`][pydantic_ai.messages] for content, plus realtime control-plane events.

Spoken content (both the user's and the model's) streams as a
[`SpeechPart`][pydantic_ai.messages.SpeechPart] (distinguished by
`speaker`), assembled through the standard part events:

| Event | Meaning |
| --- | --- |
| [`PartStartEvent`][pydantic_ai.messages.PartStartEvent] | A new part started — a `SpeechPart` (assistant or user), a `ToolCallPart`, or a plain `TextPart` when [`output_modality='text'`](#configuring-the-session). |
| [`PartDeltaEvent`][pydantic_ai.messages.PartDeltaEvent] | A [`SpeechPartDelta`][pydantic_ai.messages.SpeechPartDelta]: `audio_chunk` for playback and/or `transcript_delta` for incremental text (a [`TextPartDelta`][pydantic_ai.messages.TextPartDelta] in text mode). |
| [`PartEndEvent`][pydantic_ai.messages.PartEndEvent] | The completed part: speech has `transcript` and optional retained `audio`, text has `content`, and tool parts carry their own fields. |
| [`FunctionToolCallEvent`][pydantic_ai.messages.FunctionToolCallEvent] | The session began executing a tool the model requested (carries the `ToolCallPart`). |
| [`FunctionToolResultEvent`][pydantic_ai.messages.FunctionToolResultEvent] | The tool finished and produced a `ToolReturnPart`. Normally the result is sent to the model; a provider-cancelled call records a synthetic cancellation result only in local history. |

The remaining realtime control-plane events:

| Event | Meaning |
| --- | --- |
| [`InputSpeechStartEvent`][pydantic_ai.realtime.InputSpeechStartEvent] | OpenAI/Azure/xAI detected speech onset, or Gemini reported activity that interrupted model output. |
| [`InputSpeechEndEvent`][pydantic_ai.realtime.InputSpeechEndEvent] | OpenAI/Azure/xAI detected the end of speech; Gemini does not emit this event. |
| [`TurnCompleteEvent`][pydantic_ai.realtime.TurnCompleteEvent] | The model finished a turn. `interrupted` reflects cancellation on OpenAI/Azure/xAI; Gemini reports `False` even after barge-in. |
| [`ReconnectedEvent`][pydantic_ai.realtime.ReconnectedEvent] | The connection dropped and was automatically re-established (see [Reconnecting](#reconnecting)). |
| [`SessionErrorEvent`][pydantic_ai.realtime.SessionErrorEvent] | The provider reported a **recoverable** error mid-session; the session keeps running. A non-recoverable error instead raises [`RealtimeError`][pydantic_ai.realtime.RealtimeError]. |

## Message history

A realtime session builds the same [`ModelMessage`][pydantic_ai.messages.ModelMessage]
[history](message-history.md) a standard agent run does, so a voice conversation composes with the
rest of the framework: seed a session from an earlier conversation, and hand a finished session off
to a text [`Agent.run`][pydantic_ai.agent.AbstractAgent.run] for summarization, structured
extraction, or follow-up. Spoken turns are recorded as
[`SpeechPart`][pydantic_ai.messages.SpeechPart]s; everything else is the
ordinary [`ModelRequest`][pydantic_ai.messages.ModelRequest] /
[`ModelResponse`][pydantic_ai.messages.ModelResponse] shape, including tool calls and results.
Images and video frames streamed with `send()` are provider context but are not stored in session
history; retain them separately if a later handoff must include them.

The session exposes two snapshots (each a copy, so it won't change as the session continues):

| Method | Returns |
| --- | --- |
| [`session.all_messages()`][pydantic_ai.realtime.RealtimeSession.all_messages] | The seeded history plus completed spoken/text turns and tool rounds recorded by this session. |
| [`session.new_messages()`][pydantic_ai.realtime.RealtimeSession.new_messages] | Only the recorded messages created during this session. |

Seed a session with `message_history=` and hand its history off to a text agent afterwards:

```python {test="skip" lint="skip"}
from pydantic_ai import Agent
from pydantic_ai.realtime.openai import OpenAIRealtimeModel

voice = Agent(instructions='You are a helpful voice assistant.')
notetaker = Agent('openai:gpt-5', instructions='Summarize the conversation as bullet points.')


async def main(prior_history):
    async with voice.realtime_session(
        model=OpenAIRealtimeModel('gpt-realtime'),
        message_history=prior_history,  # resume an earlier conversation
    ) as session:
        async for event in session:
            ...  # stream audio, run tools

    # Hand the spoken conversation to a text agent.
    summary = await notetaker.run(message_history=session.all_messages())
    print(summary.output)
```

Seeding projects every replayable part of the prior conversation into the provider's initial
conversation items: text, speech transcripts, tag-wrapped thinking, function-tool calls and results,
and images. OpenAI, Azure Voice Live, and xAI replay function tools as native call/result items; Gemini Live cannot put
function parts in seeded turns, so it uses readable `[Tool call: ...]` and
`[Tool "..." returned: ...]` / `[Tool "..." error: ...]` text instead. Thinking signatures and
provider details are not replayed because they belong to the provider session that produced them.
Provider-executed native-tool metadata is also omitted: the answer it produced is already in the
history, and the execution itself cannot be replayed in a new session.

Content that cannot be represented is rejected with a [`UserError`][pydantic_ai.exceptions.UserError]
instead of being dropped silently. In particular, video, documents, uploaded-file references, and
model-generated files cannot be seeded. Speech transcripts are preferred over retained audio. When a
user [`SpeechPart`][pydantic_ai.messages.SpeechPart] has no transcript, OpenAI and Azure Voice Live can
replay retained input audio; Gemini and xAI cannot. Enable `input_transcription_model` or `audio_retention` when
capturing the original session, or filter unseedable parts from `message_history` before connecting.
Assistant speech always requires a transcript. See the [model profile](#model-profile) for provider
support.

### Retaining audio

By default only transcripts are kept on the history parts. Pass
`audio_retention=` to [`realtime_session`][pydantic_ai.agent.Agent.realtime_session] to also retain
the spoken audio as WAV [`BinaryContent`][pydantic_ai.messages.BinaryContent] on the `SpeechPart`s,
at the cost of memory. Streaming input and `SpeechPartDelta.audio_chunk` output remain raw PCM16;
only finalized history audio is wrapped in a WAV container.

| [`audio_retention`][pydantic_ai.realtime.AudioRetention] | Retains |
| --- | --- |
| `'transcript_only'` (default) | Transcripts only; no audio bytes. |
| `'input'` | The user's spoken audio. |
| `'output'` | The model's spoken audio. |
| `'both'` | Both sides' audio. |

When you [hand off](#delegating-to-a-text-agent) to a standard model, retained user audio is forwarded
to models whose profile declares audio-input support; other models receive the transcript instead.
Assistant speech is always handed off as transcript text.

## Configuring the session

Session behaviour is configured through realtime model settings. The simplest path uses a
provider-prefixed model ID and per-session settings:

```python {test="skip" lint="skip"}
from pydantic_ai.realtime import TurnDetection
from pydantic_ai.realtime.openai import OpenAIRealtimeModelSettings

settings = OpenAIRealtimeModelSettings(
    max_tokens=2_000,
    parallel_tool_calls=True,
    voice='alloy',
    turn_detection=TurnDetection(sensitivity='high', silence_duration_ms=400),
    openai_input_noise_reduction='near_field',
    openai_output_speed=1.1,
)

async with agent.realtime_session(
    model='openai:gpt-realtime', model_settings=settings
) as session:
    ...
```

[`RealtimeModelSettings`][pydantic_ai.realtime.RealtimeModelSettings] defines the common settings
vocabulary for realtime providers: `tool_choice`, `parallel_tool_calls`, `max_tokens`, `voice`,
`input_transcription_model`, `output_modality`, `handshake_timeout`,
[`turn_detection`][pydantic_ai.realtime.TurnDetection], and
[`thinking`][pydantic_ai.realtime.RealtimeModelSettings.thinking] (see [Reasoning](#reasoning)). You
can instead set defaults on a model and override individual values for one session. Gemini ignores
`tool_choice`, `parallel_tool_calls`, `input_transcription_model`, and `handshake_timeout`; use its
Google-prefixed transcription fields instead. xAI ignores `output_modality` and `thinking` and
always produces audio output. Provider-specific setting docstrings document other exceptions.
Azure Voice Live ignores `parallel_tool_calls` and `thinking`.

```python {test="skip" lint="skip"}
async with agent.realtime_session(
    model=OpenAIRealtimeModel('gpt-realtime', settings=settings),
    model_settings=OpenAIRealtimeModelSettings(max_tokens=4_000),
) as session:
    ...
```

For finer OpenAI control, `openai_turn_detection` accepts
[`ServerVAD`][pydantic_ai.realtime.openai.ServerVAD] or
[`SemanticVAD`][pydantic_ai.realtime.openai.SemanticVAD] and fully overrides the shared
`turn_detection` setting:

```python {test="skip" lint="skip"}
from pydantic_ai.realtime.openai import SemanticVAD

settings = OpenAIRealtimeModelSettings(
    turn_detection=TurnDetection(sensitivity='high'),
    openai_turn_detection=SemanticVAD(eagerness='high'),
)
```

To keep the prompt-cached audio prefix stable as a long session grows (cached audio is far cheaper
than re-encoding it), OpenAI also accepts
[`openai_truncation`][pydantic_ai.realtime.openai.OpenAIRealtimeModelSettings.openai_truncation] —
`'auto'`, `'disabled'`, or a `{'type': 'retention_ratio', 'retention_ratio': 0.8}` dict.

The agent's regular `model_settings` and capability `get_model_settings()` contributions do not apply
to realtime sessions. OpenAI, Azure Voice Live, and xAI realtime sessions do not expose
`temperature` through Pydantic AI.

### Reasoning

The cross-provider [`thinking`][pydantic_ai.realtime.RealtimeModelSettings.thinking] setting mirrors
the unified [`thinking`][pydantic_ai.settings.ModelSettings.thinking] on the request-response models:
`True` enables reasoning at the provider default, and
`'minimal'`/`'low'`/`'medium'`/`'high'`/`'xhigh'` selects an effort level.

It applies only to realtime models that support reasoning — reported by the model's
[`supports_thinking`][pydantic_ai.realtime.RealtimeModelProfile.supports_thinking] profile flag. This
includes OpenAI's `gpt-realtime-2` family (e.g. `gpt-realtime-2.1` and `gpt-realtime-2.1-mini`) and
Gemini's native-audio Live models. The GA `gpt-realtime` is a standard speech-to-speech model without
reasoning, so a `thinking` setting is ignored with a warning rather than sent (the API would otherwise
reject it).

```python {test="skip" lint="skip"}
async with agent.realtime_session(
    model=OpenAIRealtimeModel('gpt-realtime-2.1', settings=OpenAIRealtimeModelSettings(thinking='low')),
) as session:
    ...
```

On OpenAI the effort maps to `reasoning.effort`. OpenAI realtime does not accept a disabled effort,
so `False` omits `reasoning` and leaves the model's default behavior unchanged. On Gemini the setting
maps to a thinking level, and `False` disables thinking. xAI ignores `thinking`. For finer Gemini
control — a token budget, or thought summaries — set
[`google_thinking_config`][pydantic_ai.realtime.google.GoogleRealtimeModelSettings.google_thinking_config],
which takes precedence over `thinking`.

!!! note "Reasoning traces are not surfaced"
    `thinking` tunes the model's reasoning effort (a quality-vs-latency trade-off that improves the
    answer); the session does not surface the reasoning itself as `ThinkingPart`s. OpenAI's realtime
    API never streams reasoning (`reasoning.effort` is input-only), and Gemini's native-audio
    *audio*-output sessions emit no thought summaries. Gemini's *text*-output sessions can emit thought
    summaries, but the realtime session does not surface them.

### Gemini configuration

[`GoogleRealtimeModel`][pydantic_ai.realtime.google.GoogleRealtimeModel] exposes Gemini Live's knobs
as optional fields, grouped by concern. Google-only generation parameters use
[`GoogleRealtimeModelSettings`][pydantic_ai.realtime.google.GoogleRealtimeModelSettings], which extends
the shared settings with `temperature`, `top_p`, `top_k`, `seed`, `google_thinking_config`, and
`google_video_resolution`.

```python {test="skip" lint="skip"}
from pydantic_ai.realtime.google import (
    AutomaticVAD,
    ContextCompression,
    GoogleRealtimeModel,
    GoogleRealtimeModelSettings,
    MultiSpeaker,
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
| `google_enable_session_resumption`, `reconnect` | Transparent resume on a dropped connection (see [Reconnecting](#reconnecting)) |
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

### xAI Grok Voice configuration

[`XaiRealtimeModel`][pydantic_ai.realtime.xai.XaiRealtimeModel] supports the shared
[`TurnDetection`][pydantic_ai.realtime.TurnDetection] configuration:

```python {test="skip" lint="skip"}
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
realtime settings can be configured on the model or passed to `realtime_session`. Grok
Voice reports the input transcript as cumulative snapshots that can
retroactively correct earlier text, so live partials are not streamed — the transcript is surfaced at
the end of each user turn.

### Transcribing user input

To hand a voice session off to a text agent, the user's turns need to reach history as text. OpenAI and
xAI transcribe the user's audio with a dedicated model, set via `input_transcription_model`:

| Value | Behaviour |
| --- | --- |
| `'auto'` (default) | The provider's recommended transcription model, so user turns are captured under the default `audio_retention='transcript_only'`. Pin a specific id when the transcription model must remain fixed. |
| An explicit id (e.g. `'gpt-4o-transcribe'`) | Used verbatim. Known ids autocomplete via [`KnownRealtimeTranscriptionModelName`][pydantic_ai.realtime.KnownRealtimeTranscriptionModelName], but any string works. |
| `None` | Transcription disabled — no transcription model is sent. No user transcripts arrive, so [`audio_retention`](#retaining-audio) **must** be `'input'`/`'both'` to keep the raw audio; each user turn is then finalized as an audio-only [`SpeechPart`][pydantic_ai.messages.SpeechPart] (no transcript, so not usable for a text handoff). Disabling transcription while `audio_retention` doesn't retain input audio raises a `UserError`, since the user's turns would otherwise be silently dropped from history. |

Gemini transcribes with `google_input_transcription` (on by default) rather than a model id: the
Live model transcribes natively, so there is no separate transcription model to choose. If
`google_input_transcription=False`, set `audio_retention='input'` or `'both'`; otherwise session
creation raises [`UserError`][pydantic_ai.exceptions.UserError] because user turns could not be
recorded. If `google_output_transcription=False`, retain output audio to keep assistant audio turns
in history at all. Transcript-less assistant audio cannot be handed off to a text agent or seeded
into another realtime session.

OpenAI and Gemini may stream partial user transcripts. xAI suppresses revisable partial snapshots
and emits the finalized user transcript at the end of the turn.

## Turn-taking and barge-in

By default each provider uses its automatic voice activity detection: it decides when the user has
started and stopped speaking and when to trigger a response. Configure the common behavior with
[`TurnDetection`][pydantic_ai.realtime.TurnDetection]: `sensitivity` maps to the closest provider
knob, while `prefix_padding_ms` and `silence_duration_ms` pass through on OpenAI, xAI, and Gemini.
For finer control, `openai_turn_detection`, `xai_turn_detection`, and `google_vad` fully override the
shared setting. OpenAI's escape hatch also supports
[`SemanticVAD`][pydantic_ai.realtime.openai.SemanticVAD].

[Push-to-talk](#push-to-talk-manual-turn-taking) drives turns manually instead of by detection.
Gemini's native-audio models can also decide on their own when to speak via `google_proactive_audio`.

When the user barges in you get a [`InputSpeechStartEvent`][pydantic_ai.realtime.InputSpeechStartEvent] event; stop
playing any buffered model audio immediately, and call
[`interrupt`][pydantic_ai.realtime.RealtimeSession.interrupt] to cancel the model's in-progress
response. Pass `audio_end_ms` (how many milliseconds of the response the user actually heard) so the
provider truncates its stored transcript to match — otherwise the model "remembers" saying words the
user never heard:

```python {test="skip" lint="skip"}
async for event in session:
    if isinstance(event, InputSpeechStartEvent):
        speaker.flush()  # drop buffered audio locally
        await session.interrupt(audio_end_ms=speaker.played_ms())
```

`interrupt()` is server-side only — it does not flush your local playback buffer; that is the
caller's responsibility. Explicit `interrupt()` and manual turn-taking require provider support (see
[model profile](#model-profile)); Gemini Live handles barge-in automatically and exposes neither. The
`audio_end_ms` truncation additionally needs [`supports_output_truncation`](#model-profile), which xAI Grok
Voice lacks — call `interrupt()` without `audio_end_ms` there.

### Push-to-talk (manual turn-taking)

Automatic detection is **on by default**; disable it with `turn_detection=False` for push-to-talk.
This is only supported on providers that expose manual turn control through Pydantic AI (OpenAI,
Azure Voice Live, and xAI — see [`supports_manual_turn_control`](#model-profile)); Gemini Live has no manual turn verbs, so
`turn_detection=False` raises a `UserError` there. Drive the turn yourself: stream audio,
[`commit_audio`][pydantic_ai.realtime.RealtimeSession.commit_audio] to end the user's turn, then
[`create_response`][pydantic_ai.realtime.RealtimeSession.create_response] to ask the model to reply.
[`clear_audio`][pydantic_ai.realtime.RealtimeSession.clear_audio] discards uncommitted audio.

```python {test="skip" lint="skip"}
model = OpenAIRealtimeModel(
    'gpt-realtime', settings=OpenAIRealtimeModelSettings(turn_detection=False)
)
async with agent.realtime_session(model=model) as session:
    await session.send_audio(chunk)
    await session.commit_audio()
    await session.create_response()
```

## Model profile

Realtime providers differ in what they support, so features like manual turn-taking and barge-in
aren't universal. Each model reports its support through
[`RealtimeModel.profile`][pydantic_ai.realtime.RealtimeModel.profile], a
[`RealtimeModelProfile`][pydantic_ai.realtime.RealtimeModelProfile] — the realtime counterpart to the
[`ModelProfile`][pydantic_ai.profiles.ModelProfile] of a standard [`Model`][pydantic_ai.models.Model]:

The profile also exposes `audio_input_sample_rate` and `audio_output_sample_rate`, in Hz. OpenAI,
Azure Voice Live, and xAI use 24 kHz in both directions; Gemini uses 16 kHz input and 24 kHz output.

| [`RealtimeModelProfile`][pydantic_ai.realtime.RealtimeModelProfile] flag | Gates | OpenAI | Azure Voice Live | Gemini | xAI |
| --- | --- | :---: | :---: | :---: | :---: |
| [`supports_image_input`][pydantic_ai.realtime.RealtimeModelProfile.supports_image_input] | [`send`](#images) | ✅ | ✅ | ✅ | ❌ |
| [`supports_manual_turn_control`][pydantic_ai.realtime.RealtimeModelProfile.supports_manual_turn_control] | [`commit_audio`/`clear_audio`/`create_response`](#push-to-talk-manual-turn-taking) | ✅ | ✅ | ❌ | ✅ |
| [`supports_interruption`][pydantic_ai.realtime.RealtimeModelProfile.supports_interruption] | [`interrupt`](#turn-taking-and-barge-in) | ✅ | ✅ | ❌ | ✅ |
| [`supports_output_truncation`][pydantic_ai.realtime.RealtimeModelProfile.supports_output_truncation] | [`interrupt(audio_end_ms=…)`](#turn-taking-and-barge-in) | ✅ | ✅ | ❌ | ❌ |
| [`supports_session_seeding`][pydantic_ai.realtime.RealtimeModelProfile.supports_session_seeding] | [`message_history=`](#message-history) | ✅ | ✅ | ✅ | ✅ |
| [`supports_seeding_images`][pydantic_ai.realtime.RealtimeModelProfile.supports_seeding_images] | Images in `message_history` | ✅ | ✅ | ✅ | ❌ |
| [`supports_seeding_audio`][pydantic_ai.realtime.RealtimeModelProfile.supports_seeding_audio] | Transcript-less retained user audio in `message_history` | ✅ | ✅ | ❌ | ❌ |
| [`supports_thinking`][pydantic_ai.realtime.RealtimeModelProfile.supports_thinking] | [`thinking`](#reasoning) | `gpt-realtime-2*` | ❌ | Native-audio models | ❌ |

Gemini Live drives turns with automatic VAD only and interrupts server-side on its own, so it
exposes neither the manual turn verbs nor an explicit `interrupt()`. xAI Grok Voice supports
cancelling a response (`interrupt()`) but not truncating its audio to the point the user actually
heard, so `supports_output_truncation` is off — `interrupt(audio_end_ms=…)`
raises while `interrupt()` works. Calling a method the model doesn't support raises a clear
[`UserError`][pydantic_ai.exceptions.UserError] *before* anything is sent, so you can branch on the
model's `profile` up front rather than handle a mid-session failure:

```python {test="skip" lint="skip"}
from pydantic_ai.realtime.openai import OpenAIRealtimeModel

model = OpenAIRealtimeModel('gpt-realtime')
if model.profile['supports_interruption']:
    await session.interrupt(audio_end_ms=speaker.played_ms())
```

## Images

Send an image as conversation context (for example a video frame) with
[`send`][pydantic_ai.realtime.RealtimeSession.send]. An image does not itself trigger a
response — the model picks it up on the next turn (via VAD or `create_response`).

```python {test="skip" lint="skip"}
from pydantic_ai import BinaryContent

await session.send(BinaryContent(data=jpeg_bytes, media_type='image/jpeg'))
```

**Live vision (Gemini).** For a "show the camera and ask about it" experience, stream frames
continuously and set `google_turn_coverage='all_video'` so every frame stays in context. Because a frame
alone never triggers a turn, drive proactive narration by periodically sending a short text turn
("say what changed, else stay silent"); combine with `google_proactive_audio=True` (native-audio) so the
model keeps quiet when nothing changed.

## Tool calling

Tools registered on the agent are offered to the realtime model. When the model calls one, the
session emits a [`FunctionToolCallEvent`][pydantic_ai.messages.FunctionToolCallEvent], runs the tool,
normally sends the result back, and emits a
[`FunctionToolResultEvent`][pydantic_ai.messages.FunctionToolResultEvent]. Parse failures and tool
exceptions are returned as string results so the conversation does not stall.
If the provider cancels an in-flight call, Pydantic AI cancels the task and emits a synthetic
cancellation result for valid local history, but does not send that abandoned result back to the
provider.

Realtime provider tool-output channels are string-only. If a tool returns
[`ToolReturn`][pydantic_ai.messages.ToolReturn], only `return_value` is sent and recorded; `content` and
`metadata` are not preserved.

### Concurrent tools

Every tool call runs concurrently with the session. The model can keep talking (or stay silent) while
the tool works, and receives the result as soon as it is ready, so slow tools do not freeze the
conversation. The [`FunctionToolResultEvent`][pydantic_ai.messages.FunctionToolResultEvent] streams
when the tool finishes; [`all_messages()`][pydantic_ai.realtime.RealtimeSession.all_messages] keeps
the result adjacent to its call so the history remains valid for a text-agent handoff.

!!! note "Deferred and approval-required tools"
    Deferred and approval-required calls can be resolved inline by
    [`HandleDeferredToolCalls`][pydantic_ai.capabilities.HandleDeferredToolCalls]. If no handler
    resolves the call, the model receives an explanation that the tool cannot complete during a
    realtime session; the session cannot pause for an out-of-band result.

### Built-in tools (web search)

Provider-native tools run server-side. Add them as you would for a normal run — via the high-level
[`WebSearch`][pydantic_ai.capabilities.WebSearch] / [`WebFetch`][pydantic_ai.capabilities.WebFetch]
capabilities (or the lower-level [`NativeTool`][pydantic_ai.capabilities.NativeTool]) — and they flow
into the session. **Gemini** maps [`WebSearch`][pydantic_ai.capabilities.WebSearch] to Grounding with
Google Search and [`WebFetch`][pydantic_ai.capabilities.WebFetch] to URL context, so the model can
search the web and read a page mid-conversation:

```python {test="skip" lint="skip"}
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.messages import NativeToolReturnPart, PartEndEvent
from pydantic_ai.realtime.google import GoogleRealtimeModel

agent = Agent(instructions='Answer questions, searching the web when useful.')

async with agent.realtime_session(
    model=GoogleRealtimeModel('gemini-2.5-flash-native-audio-latest'),
    capabilities=[WebSearch()],
) as session:
    async for event in session:
        if isinstance(event, PartEndEvent) and isinstance(event.part, NativeToolReturnPart):
            # Cite what the model grounded its answer on.
            print(event.part.content)
```

`WebFetch()` works the same way on models that support URL context — but see the caveats below before
combining it with `WebSearch` on Gemini 2.5.

When the model grounds an answer on web results, its citations are carried in the
[`NativeToolReturnPart`][pydantic_ai.messages.NativeToolReturnPart] content, matching a regular run.
The grounding (and any code the model ran via
[`CodeExecutionTool`][pydantic_ai.native_tools.CodeExecutionTool]) is also recorded in
[`all_messages`][pydantic_ai.realtime.RealtimeSession.all_messages] as the built-in-tool call/return
parts a classic run would produce, so it survives the handoff to
[`Agent.run`][pydantic_ai.agent.AbstractAgent.run].

Gemini Live supports `WebSearch` / `WebFetch` (web search and URL context) and code execution (add
[`CodeExecutionTool`][pydantic_ai.native_tools.CodeExecutionTool] via
[`NativeTool`][pydantic_ai.capabilities.NativeTool]). The OpenAI and xAI realtime providers support no
native tools. Each model declares the tools it runs server-side in its
[`supported_native_tools`][pydantic_ai.realtime.RealtimeModelProfile.supported_native_tools] profile;
passing an unsupported one raises a [`UserError`][pydantic_ai.exceptions.UserError] naming what the
model does support, before the session connects.

!!! warning "`WebFetch` (URL context) isn't supported natively on native-audio models"
    Gemini's native-audio Live models support `WebSearch` (Grounding with Google Search)
    but **not** native `WebFetch` (`url_context`): the connection opens, but the session drops with
    `Unexpected function call` the first time the model tries to fetch a URL.

    Use the **local fallback** instead — `WebFetch(native=False, local=True)`. It registers an ordinary
    function tool (which the native-audio model *does* support) that fetches the page in your own
    process, so it works on any Live model. The fetch runs from your network and needs the `web-fetch`
    optional group. The same applies to `WebSearch(native=False, local='duckduckgo')` if you need
    search on a model without grounding.

!!! warning "Don't combine native Google Search grounding with function tools on Gemini 2.5"
    Gemini 2.5 models (including native-audio) can't use Grounding with Google Search **and** function
    calling in the same session — only Gemini 3 supports that combination. So pairing native
    `WebSearch()` (grounding) with *any* function tool (including a local `WebFetch` fallback) leaves
    the function tool uncallable: the model will say it can't use it. Pick one — native grounding, or
    function tools (use local fallbacks for both search and fetch) — unless you're on Gemini 3.

## Usage and cost

The session accumulates token usage as the model responds. Read it from
[`RealtimeSession.usage`][pydantic_ai.realtime.RealtimeSession.usage] — a
[`RunUsage`][pydantic_ai.usage.RunUsage] with input/output tokens (including audio and cached
breakdowns) and tool-call counts. Usage updates are accumulated on `session.usage` and are not
yielded as session events.

```python {test="skip" lint="skip"}
async with agent.realtime_session(model=model) as session:
    async for event in session:
        ...
    print(session.usage)  # cumulative tokens + tool calls for the session
```

Pass `usage` to accumulate into a shared [`RunUsage`][pydantic_ai.usage.RunUsage] (e.g. to total a
voice session and follow-up text runs together), and `usage_limits` to cap a session. Token and
tool-call limits are enforced as usage accrues; a breach raises
[`UsageLimitExceeded`][pydantic_ai.exceptions.UsageLimitExceeded] from the session's event iterator,
matching how `run` / `iter` surface a usage limit.

```python {test="skip" lint="skip"}
from pydantic_ai.usage import RunUsage, UsageLimits

shared = RunUsage()
async with agent.realtime_session(
    model=model, usage=shared, usage_limits=UsageLimits(total_tokens_limit=100_000)
) as session:
    ...
```

## Observability with Logfire

Realtime sessions emit OpenTelemetry spans when the agent is instrumented — call
`logfire.instrument_pydantic_ai()` (or set `instrument=True` on the agent). You get a session span
(reported as an agent invocation, like a classic run) carrying cumulative usage and the conversation
transcript (content is redacted when `include_content` is disabled). The session span uses
`gen_ai.operation.name='invoke_agent'` and sets `gen_ai.output.type` to `'speech'` or `'text'`.
Nested under it, each assistant response gets a `chat {model}` span with
`gen_ai.output.type` set to the same value, and each tool call gets an
`execute_tool` span (including any delegated text-agent run). See
[Debugging and monitoring](logfire.md).

OpenAI, Azure Voice Live, and xAI `chat` spans carry the response's own usage, including function-call-only responses.
Gemini finalizes a function-call response before the provider reports usage; that response has zero
usage, while Gemini's later completed turn carries the reported turn usage. The cumulative
`session.usage` and session span remain authoritative totals.

```python {test="skip" lint="skip"}
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()
# realtime_session spans now appear in Logfire
```

### Routing through a gateway

Realtime models take a `provider=` just like standard models, so you can route a session through the
[Pydantic AI Gateway](gateway.md) (or any OpenAI-compatible endpoint that
exposes a realtime API) by naming the upstream provider:

```python {test="skip" lint="skip"}
from pydantic_ai.realtime.openai import OpenAIRealtimeModel

model = OpenAIRealtimeModel('gpt-realtime', provider='gateway/openai')
```

When a span is active as the session connects, the client propagates
[W3C trace context](https://www.w3.org/TR/trace-context/) over the realtime WebSocket handshake, so a
gateway that emits its own spans nests them under the same trace. The provider connection is
established before the realtime session span opens, so wrap the entire session context in your own
span to ensure gateway handshake spans join the trace:

```python {test="skip" lint="skip"}
with logfire.span('voice call'):
    async with agent.realtime_session(model=model) as session:
        ...
```

## Reconnecting

A long-lived connection can drop. Pass a
[`ReconnectPolicy`][pydantic_ai.realtime.ReconnectPolicy] to transparently re-dial with
exponential backoff, re-apply the session configuration, and emit a
[`ReconnectedEvent`][pydantic_ai.realtime.ReconnectedEvent] event:

```python {test="skip" lint="skip"}
from pydantic_ai.realtime import ReconnectPolicy
from pydantic_ai.realtime.openai import OpenAIRealtimeModel

model = OpenAIRealtimeModel('gpt-realtime', reconnect=ReconnectPolicy(max_attempts=5))
```

For OpenAI, Azure Voice Live, and xAI, reconnecting restores the session configuration but **not** server-side
conversation state (the audio buffer and prior turns), so treat a
[`ReconnectedEvent`][pydantic_ai.realtime.ReconnectedEvent] as the start of a fresh turn. Without a
policy (the default), a dropped connection raises [`RealtimeError`][pydantic_ai.realtime.RealtimeError]
from the session iterator, so the app can open a new session itself.

Gemini reconnects via **session resumption**, which *does* restore conversation state. Enable it with
both `google_enable_session_resumption=True` and a
[`ReconnectPolicy`][pydantic_ai.realtime.ReconnectPolicy] — the session re-dials from the
latest resumption handle the server issued:

```python {test="skip" lint="skip"}
from pydantic_ai.realtime import ReconnectPolicy
from pydantic_ai.realtime.google import GoogleRealtimeModel, GoogleRealtimeModelSettings

model = GoogleRealtimeModel(
    'gemini-2.5-flash-native-audio-latest',
    settings=GoogleRealtimeModelSettings(google_enable_session_resumption=True),
    reconnect=ReconnectPolicy(max_attempts=5),
)
```

## Relationship to `run` / `iter`

[`Agent.realtime_session`][pydantic_ai.agent.Agent.realtime_session] is the realtime sibling of
[`run`][pydantic_ai.agent.AbstractAgent.run] / [`iter`][pydantic_ai.agent.AbstractAgent.iter]. It
accepts the parameters that map to a long-lived, bidirectional session and intentionally omits the
ones that are specific to the request-response graph (faking them would be misleading).

| `run` / `iter` parameter | In `realtime_session`? |
| --- | --- |
| `deps`, `model_settings` | ✅ realtime-specific settings; regular agent/capability settings do not apply |
| `instructions` | ✅ additive (combined with the agent's); dynamic `@agent.instructions` evaluated once at connect |
| `toolsets` | ✅ extra toolsets for the session |
| `capabilities` | ⚠️ **setup + tool hooks only** — see [Capabilities](#capabilities) below |
| `usage`, `usage_limits` | ✅ accumulate / enforce (request, token, and tool-call limits; see [Usage and cost](#usage-and-cost)) |
| `metadata`, `conversation_id` | ✅ set on the `RunContext` and telemetry span; `conversation_id` is also stamped on session-built history |
| `message_history` | ✅ seeds replayable text, transcripts, thinking, tool rounds, images, and supported retained user audio; included in `all_messages()` |
| `output_type` | ❌ no structured output → [delegate](#delegating-to-a-text-agent) |
| Provider-side session handle/resumption | ❌ no persistent handle API; for Gemini's in-process recovery see [session resumption](#reconnecting) |
| `user_prompt` | ❌ stream input with `send_audio()` or `send()` instead |
| `retries` | ❌ no per-session argument or output-validation retries; agent-level tool retries still apply |
| `deferred_tool_results`, `event_stream_handler` | ❌ graph-only (the session *is* the event stream) |

Realtime-only: `audio_retention` keeps spoken audio bytes on the history parts. Tool calls always run
concurrently with the session.

[Deferred tools](deferred-tools.md) work in a session to the extent they can be resolved *live*: a
[`HandleDeferredToolCalls`][pydantic_ai.capabilities.HandleDeferredToolCalls] capability handler is
invoked inline when a tool [requires approval](deferred-tools.md) or raises
[`CallDeferred`][pydantic_ai.exceptions.CallDeferred], and can approve, deny, or answer the call on
the spot. What has no realtime analog is the graph's *pause*: a session can't end with a
`DeferredToolRequests` output and wait for out-of-band results, so when no handler resolves the
call, the model instead receives an explanation that the tool can't be completed during a realtime
session, and the conversation continues. Similarly,
[`ctx.enqueue()`][pydantic_ai.tools.RunContext.enqueue] raises when called from a realtime tool —
there is no between-steps delivery point; send follow-up context into the conversation with
[`session.send()`][pydantic_ai.realtime.RealtimeSession.send] instead.

```python {test="skip" lint="skip"}
from pydantic_ai.realtime.openai import OpenAIRealtimeModel
from pydantic_ai.usage import UsageLimits

async with agent.realtime_session(
    model=OpenAIRealtimeModel('gpt-realtime'),
    toolsets=[extra_toolset],            # extra tools for this session
    capabilities=[my_capability],        # tool-lifecycle hooks run
    usage_limits=UsageLimits(total_tokens_limit=100_000),
    metadata={'tenant': 'acme'},
) as session:
    ...
```

`realtime_session` lives on [`AbstractAgent`][pydantic_ai.agent.AbstractAgent], so it's available on
[`WrapperAgent`][pydantic_ai.agent.WrapperAgent] and wrapped agents (durable, instrumented, …) just
like `run`/`iter`. [`agent.override(...)`][pydantic_ai.agent.AbstractAgent.override] of `name`, `deps`,
`tools`, `toolsets`, `instructions`, `metadata`, and `native_tools` is honored. Regular model and model
setting overrides do not select or configure the realtime model. A capability replacement supplied
through `override(spec=...)` is not applied to realtime sessions.

### Capabilities

A [capability][pydantic_ai.capabilities.AbstractCapability] passed to the agent or to
`realtime_session(capabilities=...)` participates in a session through the parts of its lifecycle that
map onto a live, bidirectional connection. A session sets the connection up once and then executes
tools as the model calls them — it has no request → graph → response run — so the hooks tied to those
stages have nothing to fire on and are **silently skipped**.

What a capability contributes to a session:

- **Setup** — its toolsets, native (built-in) tools, and instructions are applied when
  the session connects, and its `for_run` runs. [`Instrumentation`][pydantic_ai.capabilities.Instrumentation]
  is wired in exactly as for `run`/`iter` (see [Observability](#observability-with-logfire)).
  Regular capability model settings intentionally do not apply to realtime sessions.
- **Tool hooks** — every tool call routes through the same tool manager as `run`/`iter`, so
  `prepare_tools`, the `tool_validate` hooks (`before`/`after`/`wrap`/`on_error`), and the
  `tool_execute` hooks (`before`/`after`/`wrap`/`on_error`) all run. This is where guards, argument
  rewriting, retries, and per-tool instrumentation live.

What does **not** run in a session (no corresponding stage):

- run-level hooks (`before`/`after`/`wrap`/`on_error` `run`),
- graph-node hooks (`*_node_run`),
- model-request hooks (`*_model_request`) — the realtime model is a persistent connection, not a
  request-response [`Model`][pydantic_ai.models.Model],
- event-stream hooks (`wrap_run_event_stream` and per-event `on_event`) — a session emits its own
  [event stream](#events), not an agent-run one,
- output hooks (`*_output_validate`, `*_output_process`, `prepare_output_tools`) — a session has no
  `output_type` ([delegate](#delegating-to-a-text-agent) structured output to a text agent),
- deferred capability loading.

So a capability that scopes, guards, or instruments **tools**, or contributes tools/toolsets/native
tools/instructions, works unchanged in a session; one that hooks the **model request, run, graph, or
output** is inert here. There are no session- or turn-level capability hooks; use the tool hooks and
the [session event stream](#events).

## Delegating to a text agent

Realtime models do not support structured output, and are typically weaker at multi-step reasoning
than a frontier text model. The robust pattern is to keep the realtime model as the conversational
surface and expose a single tool that delegates the hard work to a normal [`Agent`][pydantic_ai.Agent]
with an `output_type`:

```python {test="skip" lint="skip"}
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.realtime.openai import OpenAIRealtimeModel


class Answer(BaseModel):
    summary: str
    confidence: float


supervisor = Agent('openai:gpt-5', output_type=Answer)
voice = Agent(instructions='Answer using the `consult` tool, then read the summary aloud.')


@voice.tool_plain
async def consult(question: str) -> str:
    result = await supervisor.run(question)
    return result.output.summary


async def main():
    async with voice.realtime_session(model=OpenAIRealtimeModel('gpt-realtime')) as session:
        ...
```

The delegated run executes concurrently, so the model can keep talking while the analysis runs.

## Limitations

Some capabilities are intentionally out of scope:

- **Browser-direct transport (WebRTC).** Sessions run server-side over WebSocket; there is no direct browser-to-provider WebRTC path.
- **Telephony (SIP).** Connecting a session to a phone call over SIP is not built in.
- **Session resumption beyond automatic reconnect.** You can't persist a handle and resume a session in a later process; recovery is limited to in-process [reconnection](#reconnecting).
- **Bounded structured-output runs.** A session has no `output_type` or `session.run()` with an output schema — [delegate to a text agent](#delegating-to-a-text-agent) for structured results.
- **Realtime-specific capability hooks.** Capabilities run once for setup and apply their instructions, toolsets, and native tools; tool-lifecycle hooks also run, but there are no before/after-exchange hooks.
- **Dynamic instructions mid-session.** Instructions are resolved once at connect and not re-evaluated during the session.
- **Provider-limited audio replay when seeding history.** OpenAI and Azure Voice Live can replay retained user audio when a [`SpeechPart`][pydantic_ai.messages.SpeechPart] has no transcript. Gemini and xAI cannot seed retained audio, and no provider can seed assistant audio; use transcripts or filter those parts before connecting (see [Message history](#message-history)).
- **Proactive resume before Gemini's session cap.** Gemini Live signals an upcoming disconnect (`GoAway`) near its session-length limit, but the session only [reconnects](#reconnecting) after a drop.

## Implementing a provider

A provider implements two ABCs: [`RealtimeModel`][pydantic_ai.realtime.RealtimeModel]
(opens a connection) and [`RealtimeConnection`][pydantic_ai.realtime.RealtimeConnection]
(sends [`RealtimeInput`][pydantic_ai.realtime.RealtimeInput] and yields the low-level
[`RealtimeCodecEvent`][pydantic_ai.realtime.RealtimeCodecEvent] vocabulary, which the session
translates into user-facing [`RealtimeEvent`][pydantic_ai.realtime.RealtimeEvent]s). The OpenAI
provider in [`pydantic_ai.realtime.openai`][pydantic_ai.realtime.openai] is a reference
implementation; the same shape applies to Azure Voice Live, Gemini Live, xAI Grok Voice, and others. Inputs a provider
doesn't support (e.g. `ImageInput`, or the manual turn-taking verbs) should raise `NotImplementedError`.
