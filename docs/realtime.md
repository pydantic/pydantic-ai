# Realtime (speech-to-speech)

Some providers offer **realtime** models that exchange audio over a persistent bidirectional
connection (WebSocket or HTTP/2) instead of the request-response pattern of the standard
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

## Installation

The OpenAI provider ([`OpenAIRealtimeModel`][pydantic_ai.realtime.openai.OpenAIRealtimeModel]) uses
WebSockets plus the `openai` client, available via the `realtime` and `openai` optional groups:

```bash
pip install "pydantic-ai-slim[realtime,openai]"
```

Authentication, base URL, and HTTP client come from the `provider` argument, mirroring
[`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel]: pass `provider='openai'` (the
default, reads `OPENAI_API_KEY`) or an [`OpenAIProvider`][pydantic_ai.providers.openai.OpenAIProvider]
instance for a custom key, base URL, or client. OpenAI-compatible endpoints that expose a realtime
API work too; Azure OpenAI is not supported.

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

xAI's realtime API is a clone of the OpenAI Realtime protocol. Authentication and the base URL come
from the `provider` argument, mirroring [`XaiModel`][pydantic_ai.models.xai.XaiModel]: pass
`provider='xai'` (the default, reads `XAI_API_KEY`) or an
[`XaiProvider`][pydantic_ai.providers.xai.XaiProvider] instance for a custom key.

All three implement the same [`RealtimeModel`][pydantic_ai.realtime.RealtimeModel] interface, so the
rest of this guide applies to any of them — swap `OpenAIRealtimeModel('gpt-realtime')` for
`GoogleRealtimeModel('gemini-2.5-flash-native-audio-latest')` or `XaiRealtimeModel('grok-voice-latest')`. A few
provider differences are worth knowing: Gemini expects **16 kHz** PCM input audio (OpenAI and xAI use
24 kHz), produces a single response modality per session, and natively accepts **live video frames**
sent as [`ImageInput`][pydantic_ai.realtime.ImageInput] (stream camera/screen frames with
[`send_image`][pydantic_ai.realtime.RealtimeSession.send_image] for "show me this" interactions). xAI
Grok Voice supports cancellation-based barge-in but not output truncation (see
[model profile](#model-profile)), and reports no per-response token usage.

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
| [`send_audio`][pydantic_ai.realtime.RealtimeSession.send_audio] | A chunk of microphone audio (PCM16). |
| [`send_text`][pydantic_ai.realtime.RealtimeSession.send_text] | A complete text turn. |
| [`send_image`][pydantic_ai.realtime.RealtimeSession.send_image] | An image as conversation context (e.g. a video frame). |

## Events

A session speaks the same event vocabulary as a standard streamed agent run: iterating a
[`RealtimeSession`][pydantic_ai.realtime.RealtimeSession] yields the shared message/part events from
[`pydantic_ai.messages`][pydantic_ai.messages] for content, plus realtime control-plane events.

Spoken content (both the user's and the model's) streams as an
[`SpeechPart`][pydantic_ai.messages.SpeechPart] (distinguished by
`speaker`), assembled through the standard part events:

| Event | Meaning |
| --- | --- |
| [`PartStartEvent`][pydantic_ai.messages.PartStartEvent] | A new part started — a `SpeechPart` (assistant or user) or a `ToolCallPart`. |
| [`PartDeltaEvent`][pydantic_ai.messages.PartDeltaEvent] | An [`SpeechPartDelta`][pydantic_ai.messages.SpeechPartDelta]: `audio_chunk` for playback and/or `transcript_delta` for incremental text. |
| [`PartEndEvent`][pydantic_ai.messages.PartEndEvent] | A part completed; `part.transcript` holds the full transcript (and retained `audio`, per `audio_retention`). |
| [`FunctionToolCallEvent`][pydantic_ai.messages.FunctionToolCallEvent] | The session began executing a tool the model requested (carries the `ToolCallPart`). |
| [`FunctionToolResultEvent`][pydantic_ai.messages.FunctionToolResultEvent] | The tool finished and its `ToolReturnPart` result was sent back to the model. |

The remaining realtime control-plane events:

| Event | Meaning |
| --- | --- |
| [`SpeechStartedEvent`][pydantic_ai.realtime.SpeechStartedEvent] | The provider detected the user started speaking (barge-in). |
| [`SpeechStoppedEvent`][pydantic_ai.realtime.SpeechStoppedEvent] | The user stopped speaking; the model is about to respond. |
| [`TurnCompleteEvent`][pydantic_ai.realtime.TurnCompleteEvent] | The model finished a turn (`interrupted=True` if the user barged in). |
| [`SessionUsageEvent`][pydantic_ai.realtime.SessionUsageEvent] | Token usage for a completed model response (see [Usage and cost](#usage-and-cost)). |
| [`RateLimitsEvent`][pydantic_ai.realtime.RateLimitsEvent] | An updated rate-limit snapshot from the provider. |
| [`ReconnectedEvent`][pydantic_ai.realtime.ReconnectedEvent] | The connection dropped and was automatically re-established (see [Reconnecting](#reconnecting)). |
| [`SourcesEvent`][pydantic_ai.realtime.SourcesEvent] | Web pages the model grounded its answer on, when using a built-in web tool (see [Built-in tools](#built-in-tools-web-search)). |
| [`SessionErrorEvent`][pydantic_ai.realtime.SessionErrorEvent] | The provider reported an error (`recoverable=False` means the connection dropped). |

## Message history

A realtime session builds the same [`ModelMessage`][pydantic_ai.messages.ModelMessage]
[history](message-history.md) a standard agent run does, so a voice conversation composes with the
rest of the framework: seed a session from an earlier conversation, and hand a finished session off
to a text [`Agent.run`][pydantic_ai.agent.AbstractAgent.run] for summarization, structured
extraction, or follow-up. Spoken turns are recorded as
[`SpeechPart`][pydantic_ai.messages.SpeechPart]s; everything else is the
ordinary [`ModelRequest`][pydantic_ai.messages.ModelRequest] /
[`ModelResponse`][pydantic_ai.messages.ModelResponse] shape, including tool calls and results.

The session exposes two snapshots (each a copy, so it won't change as the session continues):

| Method | Returns |
| --- | --- |
| [`session.all_messages()`][pydantic_ai.realtime.RealtimeSession.all_messages] | The seeded history plus everything from this session. |
| [`session.new_messages()`][pydantic_ai.realtime.RealtimeSession.new_messages] | Only the messages created during this session. |

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

Seeding is a **text/transcript projection**: the prior conversation's transcripts and text are
replayed to the provider as initial conversation items, but audio is not (see
[Not yet supported](#not-yet-supported)). Seeding requires a provider that accepts it — see
[model profile](#model-profile).

### Retaining audio

By default only transcripts are kept on the history parts. Pass
`audio_retention=` to [`realtime_session`][pydantic_ai.agent.Agent.realtime_session] to also retain
the raw PCM audio bytes (as [`BinaryContent`][pydantic_ai.messages.BinaryContent]) on the
`SpeechPart`s, at the cost of memory:

| [`audio_retention`][pydantic_ai.realtime.AudioRetention] | Retains |
| --- | --- |
| `'transcript_only'` (default) | Transcripts only; no audio bytes. |
| `'input'` | The user's spoken audio. |
| `'output'` | The model's spoken audio. |
| `'both'` | Both sides' audio. |

Retained audio is kept on the history parts, but when you [hand off](#delegating-to-a-text-agent) to a
standard model it is forwarded as the transcript text, not the audio itself: audio
forwarding requires the target model's profile to declare audio-input support, which none do.

## Configuring the session

Session behaviour is configured on the provider model. For OpenAI, on
[`OpenAIRealtimeModel`][pydantic_ai.realtime.openai.OpenAIRealtimeModel]:

```python {test="skip" lint="skip"}
from pydantic_ai.realtime import RealtimeModelSettings
from pydantic_ai.realtime.openai import OpenAIRealtimeModel, SemanticVAD

model = OpenAIRealtimeModel(
    'gpt-realtime',
    settings=RealtimeModelSettings(max_tokens=2_000, parallel_tool_calls=True),
    voice='alloy',
    turn_detection=SemanticVAD(eagerness='high'),  # how eagerly the model takes its turn
    input_noise_reduction='near_field',            # tuned for a headset mic
    output_speed=1.1,                              # speak slightly faster
)
```

[`RealtimeModelSettings`][pydantic_ai.realtime.RealtimeModelSettings] contains the settings shared by
all realtime providers: `tool_choice`, `parallel_tool_calls`, and `max_tokens`. Set defaults on the
model as above, then override individual values for one session:

```python {test="skip" lint="skip"}
async with agent.realtime_session(
    model=model,
    model_settings=RealtimeModelSettings(max_tokens=4_000),
) as session:
    ...
```

The agent's regular `model_settings` and capability `get_model_settings()` contributions do not apply
to realtime sessions. OpenAI and xAI realtime sessions do not accept `temperature`.

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

model = GoogleRealtimeModel(
    'gemini-2.5-flash-native-audio-latest',
    settings=GoogleRealtimeModelSettings(temperature=0.7, top_p=0.9),
    voice='Puck',
    language_code='en-US',                              # output language
    affective_dialog=True,                              # emotion-aware delivery (native-audio)
    proactive_audio=True,                               # model decides when to speak (native-audio)
    vad=AutomaticVAD(start_sensitivity='high', end_sensitivity='low'),
    turn_coverage='all_video',                          # keep every video frame in context
    context_compression=ContextCompression(trigger_tokens=16000, target_tokens=8000),
    config_overrides={'explicit_vad_signal': True},     # escape hatch for unmodelled SDK fields
)
```

| Field | What it does |
| --- | --- |
| `voice`, `language_code`, `multi_speaker` ([`MultiSpeaker`][pydantic_ai.realtime.google.MultiSpeaker]) | Prebuilt voice, output language, per-speaker voices |
| `affective_dialog`, `proactive_audio` | Emotion-aware delivery; let the model decide when to speak (native-audio models) |
| `vad` ([`AutomaticVAD`][pydantic_ai.realtime.google.AutomaticVAD]) | VAD `disabled`, start/end sensitivity, padding/silence |
| `activity_handling`, `turn_coverage` | Whether activity interrupts; which input a turn covers (`activity_only`/`all_input`/`all_video`) |
| `input_transcription`, `output_transcription`, `transcription_language_codes` | Transcription on/off and language hints |
| `context_compression` ([`ContextCompression`][pydantic_ai.realtime.google.ContextCompression]) | Sliding-window compression for long sessions |
| `enable_session_resumption`, `reconnect` | Transparent resume on a dropped connection (see [Reconnecting](#reconnecting)) |
| `config_overrides` | Raw keys merged last into the `LiveConnectConfig` — forward-compat escape hatch |

Authentication comes from the `provider` argument, mirroring
[`GoogleModel`][pydantic_ai.models.google.GoogleModel]: pass `provider='google'` (the default,
Gemini Developer API) or `provider='google-cloud'` for Vertex AI / ADC, or a
[`GoogleProvider`][pydantic_ai.providers.google.GoogleProvider] instance for a custom key or client.

### xAI Grok Voice configuration

[`XaiRealtimeModel`][pydantic_ai.realtime.xai.XaiRealtimeModel] keeps a minimal surface, reusing the
OpenAI provider's [`ServerVAD`][pydantic_ai.realtime.openai.ServerVAD] turn-detection knobs:

```python {test="skip" lint="skip"}
from pydantic_ai.realtime.openai import ServerVAD
from pydantic_ai.realtime.xai import XaiRealtimeModel

model = XaiRealtimeModel(
    'grok-voice-latest',
    voice='eve',                                       # eve (default), ara, rex, sal, leo, or a custom ID
    turn_detection=ServerVAD(threshold=0.85),          # or None for push-to-talk
    input_transcription_model='auto',                  # the default (see Transcribing user input)
)
```

The shared realtime settings can be configured on the model or passed to `realtime_session`. Grok
Voice reports the input transcript as cumulative snapshots that can
retroactively correct earlier text, so live partials are not streamed — the transcript is surfaced at
the end of each user turn.

### Transcribing user input

To hand a voice session off to a text agent, the user's turns need to reach history as text. OpenAI and
xAI transcribe the user's audio with a dedicated model, set via `input_transcription_model`:

| Value | Behaviour |
| --- | --- |
| `'auto'` (default) | The provider's recommended transcription model, so user turns are captured under the default `audio_retention='transcript_only'`. The concrete model behind `'auto'` can change across releases; pin a specific id to opt out of that. |
| An explicit id (e.g. `'gpt-4o-transcribe'`) | Used verbatim. Known ids autocomplete via [`KnownRealtimeTranscriptionModelName`][pydantic_ai.realtime.KnownRealtimeTranscriptionModelName], but any string works. |
| `None` | Transcription disabled — no transcription model is sent. No user transcripts arrive, so [`audio_retention`](#message-history) **must** be `'input'`/`'both'` to keep the raw audio; each user turn is then finalized as an audio-only [`SpeechPart`][pydantic_ai.messages.SpeechPart] (no transcript, so not usable for a text handoff). Disabling transcription while `audio_retention` doesn't retain input audio raises a `UserError`, since the user's turns would otherwise be silently dropped from history. |

Gemini transcribes with a boolean `input_transcription` (on by default) rather than a model id: the
Live model transcribes natively, so there is no separate transcription model to choose.

## Turn-taking and barge-in

By default the provider uses server-side voice activity detection
([`ServerVAD`][pydantic_ai.realtime.openai.ServerVAD]): it decides when the user has started and
stopped speaking, commits the audio, and triggers a response — and interrupts the model when the
user barges in. [`SemanticVAD`][pydantic_ai.realtime.openai.SemanticVAD] uses a model to decide turn
boundaries instead. [Push-to-talk](#push-to-talk-manual-turn-taking) drives turns manually instead of
by detection. Gemini's native-audio models also decide on their own when to speak, via
`proactive_audio`.

When the user barges in you get a [`SpeechStartedEvent`][pydantic_ai.realtime.SpeechStartedEvent] event; stop
playing any buffered model audio immediately, and call
[`interrupt`][pydantic_ai.realtime.RealtimeSession.interrupt] to cancel the model's in-progress
response. Pass `audio_end_ms` (how many milliseconds of the response the user actually heard) so the
provider truncates its stored transcript to match — otherwise the model "remembers" saying words the
user never heard:

```python {test="skip" lint="skip"}
async for event in session:
    if isinstance(event, SpeechStartedEvent):
        speaker.flush()  # drop buffered audio locally
        await session.interrupt(audio_end_ms=speaker.played_ms())
```

`interrupt()` is server-side only — it does not flush your local playback buffer; that is the
caller's responsibility. Explicit `interrupt()` and manual turn-taking require provider support (see
[model profile](#model-profile)); Gemini Live handles barge-in automatically and exposes neither. The
`audio_end_ms` truncation additionally needs [`supports_output_truncation`](#model-profile), which xAI Grok
Voice lacks — call `interrupt()` without `audio_end_ms` there.

### Push-to-talk (manual turn-taking)

Disable automatic detection with `turn_detection=None` and drive the turn yourself: stream audio,
[`commit_audio`][pydantic_ai.realtime.RealtimeSession.commit_audio] to end the user's turn, then
[`create_response`][pydantic_ai.realtime.RealtimeSession.create_response] to ask the model to reply.
[`clear_audio`][pydantic_ai.realtime.RealtimeSession.clear_audio] discards uncommitted audio.

```python {test="skip" lint="skip"}
model = OpenAIRealtimeModel('gpt-realtime', turn_detection=None)
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

| [`RealtimeModelProfile`][pydantic_ai.realtime.RealtimeModelProfile] flag | Gates | OpenAI | Gemini | xAI |
| --- | --- | :---: | :---: | :---: |
| [`supports_image_input`][pydantic_ai.realtime.RealtimeModelProfile.supports_image_input] | [`send_image`](#images) | ✅ | ✅ | ❌ |
| [`supports_manual_turn_control`][pydantic_ai.realtime.RealtimeModelProfile.supports_manual_turn_control] | [`commit_audio`/`clear_audio`/`create_response`](#push-to-talk-manual-turn-taking) | ✅ | ❌ | ✅ |
| [`supports_interruption`][pydantic_ai.realtime.RealtimeModelProfile.supports_interruption] | [`interrupt`](#turn-taking-and-barge-in) | ✅ | ❌ | ✅ |
| [`supports_output_truncation`][pydantic_ai.realtime.RealtimeModelProfile.supports_output_truncation] | [`truncate_output` / `interrupt(audio_end_ms=…)`](#turn-taking-and-barge-in) | ✅ | ❌ | ❌ |
| [`supports_session_seeding`][pydantic_ai.realtime.RealtimeModelProfile.supports_session_seeding] | [`message_history=`](#message-history) | ✅ | ✅ | ✅ |

Gemini Live drives turns with automatic VAD only and interrupts server-side on its own, so it
exposes neither the manual turn verbs nor an explicit `interrupt()`. xAI Grok Voice supports
cancelling a response (`interrupt()`) but not truncating its audio to the point the user actually
heard, so `supports_output_truncation` is off — `truncate_output()` and `interrupt(audio_end_ms=…)`
raise while `interrupt()` works. Calling a method the model doesn't support raises a clear
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
[`send_image`][pydantic_ai.realtime.RealtimeSession.send_image]. An image does not itself trigger a
response — the model picks it up on the next turn (via VAD or `create_response`).

```python {test="skip" lint="skip"}
await session.send_image(jpeg_bytes, mime_type='image/jpeg')
```

**Live vision (Gemini).** For a "show the camera and ask about it" experience, stream frames
continuously and set `turn_coverage='all_video'` so every frame stays in context. Because a frame
alone never triggers a turn, drive proactive narration by periodically sending a short text turn
("say what changed, else stay silent"); combine with `proactive_audio=True` (native-audio) so the
model keeps quiet when nothing changed.

## Tool calling

Tools registered on the agent are offered to the realtime model. When the model calls one, the
session emits a [`FunctionToolCallEvent`][pydantic_ai.messages.FunctionToolCallEvent], runs the tool,
sends the result back, and emits a [`FunctionToolResultEvent`][pydantic_ai.messages.FunctionToolResultEvent].
A result is always returned to the model, even when the arguments fail to parse or the tool raises,
so the conversation never stalls.

### Background tools

By default a tool runs synchronously: the session waits for it before reading more of the model's
output. For slow tools you can run them in the **background** so the model keeps speaking while the
work happens, with the result delivered once it is ready. This mirrors firing off a subagent and
carrying on.

```python {test="skip" lint="skip"}
async with agent.realtime_session(model=model, background_tools={'deep_research'}) as session:
    ...
```

### Built-in tools (web search)

Provider-native tools run server-side. Add them as you would for a normal run — via the high-level
[`WebSearch`][pydantic_ai.capabilities.WebSearch] / [`WebFetch`][pydantic_ai.capabilities.WebFetch]
capabilities (or the lower-level [`NativeTool`][pydantic_ai.capabilities.NativeTool]) — and they flow
into the session. **Gemini** maps [`WebSearch`][pydantic_ai.capabilities.WebSearch] to Grounding with
Google Search and [`WebFetch`][pydantic_ai.capabilities.WebFetch] to URL context, so the model can
search the web and read a page mid-conversation:

```python {test="skip" lint="skip"}
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.realtime import SourcesEvent
from pydantic_ai.realtime.google import GoogleRealtimeModel

agent = Agent(instructions='Answer questions, searching the web when useful.')

async with agent.realtime_session(
    model=GoogleRealtimeModel('gemini-2.5-flash-native-audio-latest'),
    capabilities=[WebSearch()],
) as session:
    async for event in session:
        if isinstance(event, SourcesEvent):
            # Cite what the model grounded its answer on.
            for source in event.sources:
                print(source.title, source.url)
```

`WebFetch()` works the same way on models that support URL context — but see the caveats below before
combining it with `WebSearch` on Gemini 2.5.

When the model grounds an answer on web results, the session emits a
[`SourcesEvent`][pydantic_ai.realtime.SourcesEvent] event carrying the search queries and the
[`WebSource`][pydantic_ai.realtime.WebSource] pages it used — surface these as citations in your UI.
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
breakdowns) and tool-call counts. Each completed response is also surfaced as a
[`SessionUsageEvent`][pydantic_ai.realtime.SessionUsageEvent] event, and providers may emit
[`RateLimitsEvent`][pydantic_ai.realtime.RateLimitsEvent] snapshots.

```python {test="skip" lint="skip"}
async with agent.realtime_session(model=model) as session:
    async for event in session:
        ...
    print(session.usage)  # cumulative tokens + tool calls for the session
```

Pass `usage` to accumulate into a shared [`RunUsage`][pydantic_ai.usage.RunUsage] (e.g. to total a
voice session and follow-up text runs together), and `usage_limits` to cap a session. Token and
tool-call limits are enforced as usage accrues; on breach the session emits a non-recoverable
[`SessionErrorEvent`][pydantic_ai.realtime.SessionErrorEvent] and ends.

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
carrying cumulative usage and, when `include_content` is enabled, the conversation transcript.
Nested under it, each assistant response gets a `chat {model}` span (mirroring a classic model
request: that response's input/output messages and per-turn usage), and each tool call an
`execute_tool` span (including any delegated text-agent run). See
[Debugging and monitoring](logfire.md).

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
gateway that emits its own spans nests them under the same trace. The session and `chat` spans only
open once you start iterating, so to guarantee the gateway's spans join the trace, wrap the session in
your own span:

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

Reconnecting restores the session configuration but **not** server-side conversation state (the
audio buffer and prior turns) — treat a `ReconnectedEvent` event as the start of a fresh turn. Without a
policy (the default), a dropped connection surfaces as a non-recoverable
[`SessionErrorEvent`][pydantic_ai.realtime.SessionErrorEvent] (`recoverable=False`) and ends the stream, so the
app can restart the session itself.

Gemini reconnects via **session resumption**, which *does* restore conversation state. Enable it with
both `enable_session_resumption=True` and a
[`ReconnectPolicy`][pydantic_ai.realtime.ReconnectPolicy] — the session re-dials from the
latest resumption handle the server issued:

```python {test="skip" lint="skip"}
from pydantic_ai.realtime import ReconnectPolicy
from pydantic_ai.realtime.google import GoogleRealtimeModel

model = GoogleRealtimeModel(
    'gemini-2.5-flash-native-audio-latest',
    enable_session_resumption=True,
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
| `usage`, `usage_limits` | ✅ accumulate / enforce (token + tool-call limits; see [Usage and cost](#usage-and-cost)) |
| `metadata`, `conversation_id` | ✅ set on the `RunContext` (and telemetry span) for tools/correlation |
| `message_history` | ✅ seeds the session (text/transcript projection; audio not replayed) and is included in `all_messages()` |
| `output_type` | ❌ no structured output → [delegate](#delegating-to-a-text-agent) |
| `conversation_id` (as history) | ❌ live conversation state lives on the provider; for Gemini see [session resumption](#reconnecting) |
| `user_prompt` | ❌ stream input with `send_audio` / `send_text` / `send_image` instead |
| `retries`, `deferred_tool_results`, `event_stream_handler` | ❌ graph-only (the session *is* the event stream) |

Realtime-only: `background_tools` (run a tool concurrently while the model keeps talking) and
`audio_retention` (keep spoken audio bytes on the history parts).

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
like `run`/`iter`. [`agent.override(...)`][pydantic_ai.agent.AbstractAgent.override] of `deps`,
`toolsets`, `instructions`, `metadata`, and `native_tools` is honored (spec-based capability override
is the one exception, since capabilities only run their tool hooks here).

### Capabilities

A [capability][pydantic_ai.capabilities.AbstractCapability] passed to the agent or to
`realtime_session(capabilities=...)` participates in a session through the parts of its lifecycle that
map onto a live, bidirectional connection. A session sets the connection up once and then executes
tools as the model calls them — it has no request → graph → response run — so the hooks tied to those
stages have nothing to fire on and are **silently skipped**.

What a capability contributes to a session:

- **Setup** — its toolsets, native (built-in) tools, and instructions are applied when
  the session connects, and its `for_run` runs, resolved through the same
  `Agent._resolve_run_capabilities` as a graph run. [`Instrumentation`][pydantic_ai.capabilities.Instrumentation]
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
output** is inert here. Session- and turn-level capability hooks (to observe or wrap a whole exchange)
are a planned addition; today, use the tool hooks and the [session event stream](#events).

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

For a slow delegated run, mark the tool as a [background tool](#background-tools) so the model keeps
talking while the analysis runs.

## Not yet supported

Some capabilities are intentionally out of scope:

- **Browser-direct transport (WebRTC).** Sessions run server-side over WebSocket; there is no direct browser-to-provider WebRTC path.
- **Telephony (SIP).** Connecting a session to a phone call over SIP is not built in.
- **Session resumption beyond automatic reconnect.** You can't persist a handle and resume a session in a later process; recovery is limited to in-process [reconnection](#reconnecting).
- **Bounded structured-output runs.** A session has no `output_type` or `session.run()` with an output schema — [delegate to a text agent](#delegating-to-a-text-agent) for structured results.
- **Realtime-specific capability hooks.** Capabilities run their [tool-lifecycle hooks](#relationship-to-run-iter) only; there are no before/after-exchange hooks.
- **Dynamic instructions mid-session.** Instructions are resolved once at connect and not re-evaluated during the session.
- **Audio replay when seeding history.** Seeding a session with `message_history=` projects text and transcripts only; prior audio is not replayed (see [Message history](#message-history)).

## Implementing a provider

A provider implements two ABCs: [`RealtimeModel`][pydantic_ai.realtime.RealtimeModel]
(opens a connection) and [`RealtimeConnection`][pydantic_ai.realtime.RealtimeConnection]
(sends [`RealtimeInput`][pydantic_ai.realtime.RealtimeInput] and yields
[`RealtimeEvent`][pydantic_ai.realtime.RealtimeEvent]). The OpenAI provider in
[`pydantic_ai.realtime.openai`][pydantic_ai.realtime.openai] is a reference implementation; the same
shape applies to Gemini Live, Amazon Nova Sonic, and others. Inputs a provider doesn't support
(e.g. `ImageInput`, or the manual turn-taking verbs) should raise `NotImplementedError`.
