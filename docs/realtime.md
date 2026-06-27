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
    Two complete, runnable apps are the fastest way in: a [voice finance assistant](examples/realtime-finance.md)
    that delegates to a text agent, and a [talk-and-show camera assistant](examples/realtime-camera.md)
    you can open on your phone. Come back here for the *why*.

## Installation

The OpenAI provider ([`OpenAIRealtimeModel`][pydantic_ai.realtime.openai.OpenAIRealtimeModel]) uses
WebSockets, available via the `realtime` optional group:

```bash
pip install "pydantic-ai-slim[realtime]"
```

The Gemini provider ([`GoogleRealtimeModel`][pydantic_ai.realtime.google.GoogleRealtimeModel]) uses
the `google-genai` SDK, available via the `google` optional group:

```bash
pip install "pydantic-ai-slim[google]"
```

Both implement the same [`RealtimeModel`][pydantic_ai.realtime.RealtimeModel] interface, so the rest
of this guide applies to either — swap `OpenAIRealtimeModel('gpt-realtime')` for
`GoogleRealtimeModel('gemini-live-2.5-flash')`. A few provider differences are worth knowing: Gemini
expects **16 kHz** PCM input audio (OpenAI uses 24 kHz), produces a single response modality per
session, and natively accepts **live video frames** sent as
[`ImageInput`][pydantic_ai.realtime.ImageInput] (stream camera/screen frames with
[`send_image`][pydantic_ai.realtime.RealtimeSession.send_image] for "show me this" interactions).

## Quickstart

```python {test="skip" lint="skip"}
from pydantic_ai import Agent
from pydantic_ai.realtime import AudioDelta, Transcript
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
            if isinstance(event, AudioDelta):
                speaker.play(event.data)
            elif isinstance(event, Transcript) and event.is_final:
                print('assistant:', event.text)
```

You stream content in with the session's `send_*` helpers and consume events by iterating the
session:

| Method | Sends |
| --- | --- |
| [`send_audio`][pydantic_ai.realtime.RealtimeSession.send_audio] | A chunk of microphone audio (PCM16). |
| [`send_text`][pydantic_ai.realtime.RealtimeSession.send_text] | A complete text turn. |
| [`send_image`][pydantic_ai.realtime.RealtimeSession.send_image] | An image as conversation context (e.g. a video frame). |

## Events

Iterating a [`RealtimeSession`][pydantic_ai.realtime.RealtimeSession] yields:

| Event | Meaning |
| --- | --- |
| [`AudioDelta`][pydantic_ai.realtime.AudioDelta] | A chunk of the model's audio output. |
| [`Transcript`][pydantic_ai.realtime.Transcript] | Transcription of the model's speech / its text output (`is_final` marks the end of a turn). |
| [`InputTranscript`][pydantic_ai.realtime.InputTranscript] | Transcription of the user's speech (streamed; `is_final` marks the completed turn). |
| [`SpeechStarted`][pydantic_ai.realtime.SpeechStarted] | The provider detected the user started speaking (barge-in). |
| [`SpeechStopped`][pydantic_ai.realtime.SpeechStopped] | The user stopped speaking; the model is about to respond. |
| [`ToolCallStarted`][pydantic_ai.realtime.ToolCallStarted] | The session began executing a tool the model requested. |
| [`ToolCallCompleted`][pydantic_ai.realtime.ToolCallCompleted] | The tool finished and its result was sent back to the model. |
| [`TurnComplete`][pydantic_ai.realtime.TurnComplete] | The model finished a turn (`interrupted=True` if the user barged in). |
| [`Usage`][pydantic_ai.realtime.Usage] | Token usage for a completed model response (see [Usage and cost](#usage-and-cost)). |
| [`RateLimits`][pydantic_ai.realtime.RateLimits] | An updated rate-limit snapshot from the provider. |
| [`Reconnected`][pydantic_ai.realtime.Reconnected] | The connection dropped and was automatically re-established (see [Reconnecting](#reconnecting)). |
| [`Sources`][pydantic_ai.realtime.Sources] | Web pages the model grounded its answer on, when using a built-in web tool (see [Built-in tools](#built-in-tools-web-search)). |
| [`SessionError`][pydantic_ai.realtime.SessionError] | The provider reported an error (`recoverable=False` means the connection dropped). |

## Configuring the session

Session behaviour is configured on the provider model. For OpenAI, on
[`OpenAIRealtimeModel`][pydantic_ai.realtime.openai.OpenAIRealtimeModel]:

```python {test="skip" lint="skip"}
from pydantic_ai.realtime.openai import OpenAIRealtimeModel, SemanticVAD

model = OpenAIRealtimeModel(
    'gpt-realtime',
    voice='alloy',
    turn_detection=SemanticVAD(eagerness='high'),  # how eagerly the model takes its turn
    input_noise_reduction='near_field',            # tuned for a headset mic
    output_speed=1.1,                              # speak slightly faster
)
```

`tool_choice` and `parallel_tool_calls` are read from `model_settings` passed to
`realtime_session`. (GA realtime sessions have no `temperature`, so it is not forwarded.)

### Gemini configuration

[`GoogleRealtimeModel`][pydantic_ai.realtime.google.GoogleRealtimeModel] exposes Gemini Live's knobs
as optional fields, grouped by concern. Generation parameters come from `model_settings` (consistent
with the rest of pydantic-ai), so `temperature`, `top_p`, `top_k`, `max_tokens`, `seed`,
`google_thinking_config`, and `google_video_resolution` all flow through.

```python {test="skip" lint="skip"}
from pydantic_ai.realtime.google import AutomaticVAD, ContextCompression, GoogleRealtimeModel, MultiSpeaker

model = GoogleRealtimeModel(
    'gemini-live-2.5-flash-native-audio',
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
| `vertexai`, `project`, `location` | Use Vertex AI / ADC instead of an API key |

## Turn-taking and barge-in

By default the provider uses server-side voice activity detection
([`ServerVAD`][pydantic_ai.realtime.openai.ServerVAD]): it decides when the user has started and
stopped speaking, commits the audio, and triggers a response — and interrupts the model when the
user barges in. [`SemanticVAD`][pydantic_ai.realtime.openai.SemanticVAD] uses a model to decide turn
boundaries instead.

When the user barges in you get a [`SpeechStarted`][pydantic_ai.realtime.SpeechStarted] event; stop
playing any buffered model audio immediately, and call
[`interrupt`][pydantic_ai.realtime.RealtimeSession.interrupt] to cancel the model's in-progress
response. Pass `audio_end_ms` (how many milliseconds of the response the user actually heard) so the
provider truncates its stored transcript to match — otherwise the model "remembers" saying words the
user never heard:

```python {test="skip" lint="skip"}
async for event in session:
    if isinstance(event, SpeechStarted):
        speaker.flush()  # drop buffered audio locally
        await session.interrupt(audio_end_ms=speaker.played_ms())
```

`interrupt()` is server-side only — it does not flush your local playback buffer; that is the
caller's responsibility.

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
model keeps quiet when nothing changed. The [realtime camera example](examples/realtime-camera.md)
implements exactly this with a *Watch* toggle.

## Tool calling

Tools registered on the agent are offered to the realtime model. When the model calls one, the
session emits `ToolCallStarted`, runs the tool, sends the result back, and emits `ToolCallCompleted`.
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
from pydantic_ai.realtime import Sources
from pydantic_ai.realtime.google import GoogleRealtimeModel

agent = Agent(instructions='Answer questions, searching the web when useful.')

async with agent.realtime_session(
    model=GoogleRealtimeModel('gemini-live-2.5-flash-native-audio'),
    capabilities=[WebSearch()],
) as session:
    async for event in session:
        if isinstance(event, Sources):
            # Cite what the model grounded its answer on.
            for source in event.sources:
                print(source.title, source.url)
```

`WebFetch()` works the same way on models that support URL context — but see the caveats below before
combining it with `WebSearch` on Gemini 2.5.

When the model grounds an answer on web results, the session emits a
[`Sources`][pydantic_ai.realtime.Sources] event carrying the search queries and the
[`WebSource`][pydantic_ai.realtime.WebSource] pages it used — surface these as citations in your UI.

Only `WebSearch` / `WebFetch` (web search and URL context) are supported on Gemini Live today; other
native tools raise a `UserError`. The OpenAI realtime provider does not support native tools yet (it
raises `UserError`).

!!! warning "`WebFetch` (URL context) isn't supported natively on native-audio models"
    The `gemini-live-2.5-flash-native-audio` model supports `WebSearch` (Grounding with Google Search)
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
[`Usage`][pydantic_ai.realtime.Usage] event, and providers may emit
[`RateLimits`][pydantic_ai.realtime.RateLimits] snapshots.

```python {test="skip" lint="skip"}
async with agent.realtime_session(model=model) as session:
    async for event in session:
        ...
    print(session.usage)  # cumulative tokens + tool calls for the session
```

Pass `usage` to accumulate into a shared [`RunUsage`][pydantic_ai.usage.RunUsage] (e.g. to total a
voice session and follow-up text runs together), and `usage_limits` to cap a session. Token and
tool-call limits are enforced as usage accrues; on breach the session emits a non-recoverable
[`SessionError`][pydantic_ai.realtime.SessionError] and ends.

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
carrying cumulative usage and, when `include_content` is enabled, the conversation transcript, with
an `execute_tool` span per tool call (including any delegated text-agent run) nested underneath. See
[Debugging and monitoring](logfire.md).

```python {test="skip" lint="skip"}
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()
# realtime_session spans now appear in Logfire
```

## Reconnecting

A long-lived connection can drop. Pass a
[`ReconnectPolicy`][pydantic_ai.realtime.openai.ReconnectPolicy] to transparently re-dial with
exponential backoff, re-apply the session configuration, and emit a
[`Reconnected`][pydantic_ai.realtime.Reconnected] event:

```python {test="skip" lint="skip"}
from pydantic_ai.realtime.openai import OpenAIRealtimeModel, ReconnectPolicy

model = OpenAIRealtimeModel('gpt-realtime', reconnect=ReconnectPolicy(max_attempts=5))
```

Reconnecting restores the session configuration but **not** server-side conversation state (the
audio buffer and prior turns) — treat a `Reconnected` event as the start of a fresh turn. Without a
policy (the default), a dropped connection surfaces as a non-recoverable
[`SessionError`][pydantic_ai.realtime.SessionError] (`recoverable=False`) and ends the stream, so the
app can restart the session itself.

Gemini reconnects via **session resumption**, which *does* restore conversation state. Enable it with
both `enable_session_resumption=True` and a
[`ReconnectPolicy`][pydantic_ai.realtime.google.ReconnectPolicy] — the session re-dials from the
latest resumption handle the server issued:

```python {test="skip" lint="skip"}
from pydantic_ai.realtime.google import GoogleRealtimeModel, ReconnectPolicy

model = GoogleRealtimeModel(
    'gemini-live-2.5-flash',
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
| `deps`, `model_settings` | ✅ same |
| `instructions` | ✅ additive (combined with the agent's); dynamic `@agent.instructions` evaluated once at connect |
| `toolsets` | ✅ extra toolsets for the session |
| `capabilities` | ⚠️ **tool-lifecycle hooks only** — `prepare_tools` and `before`/`after`/`wrap`/`on_error` `tool_execute`. The session executes tools but has no model-request/graph/output stages, so those hooks (and deferred loading / capability toolsets) don't run |
| `usage`, `usage_limits` | ✅ accumulate / enforce (token + tool-call limits; see [Usage and cost](#usage-and-cost)) |
| `metadata`, `conversation_id` | ✅ set on the `RunContext` (and telemetry span) for tools/correlation |
| `output_type` | ❌ no structured output → [delegate](#delegating-to-a-text-agent) |
| `message_history` / `conversation_id` (as history) | ❌ conversation state lives on the provider; for Gemini see [session resumption](#reconnecting) |
| `user_prompt` | ❌ stream input with `send_audio` / `send_text` / `send_image` instead |
| `retries`, `deferred_tool_results`, `event_stream_handler` | ❌ graph-only (the session *is* the event stream) |

Realtime-only: `background_tools` (run a tool concurrently while the model keeps talking).

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

The [realtime finance example](examples/realtime-finance.md) builds this out into a full voice app —
including [background tools](#background-tools) so the model keeps talking while a slow analysis runs.

## Implementing a provider

A provider implements two ABCs: [`RealtimeModel`][pydantic_ai.realtime.RealtimeModel]
(opens a connection) and [`RealtimeConnection`][pydantic_ai.realtime.RealtimeConnection]
(sends [`RealtimeInput`][pydantic_ai.realtime.RealtimeInput] and yields
[`RealtimeEvent`][pydantic_ai.realtime.RealtimeEvent]). The OpenAI provider in
[`pydantic_ai.realtime.openai`][pydantic_ai.realtime.openai] is a reference implementation; the same
shape applies to Gemini Live, Amazon Nova Sonic, and others. Inputs a provider doesn't support
(e.g. `ImageInput`, or the manual turn-taking verbs) should raise `NotImplementedError`.
```
