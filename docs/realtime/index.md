# Realtime (speech-to-speech)

Pydantic AI's realtime support lets an agent hold a live, spoken conversation: it streams the user's
audio to a speech-to-speech model and streams the model's spoken reply back, turn by turn. The model
hears and speaks directly — no separate transcribe-then-generate-then-synthesize pipeline — so latency
is low and tone, timing, and interruptions feel natural.

What Pydantic AI brings is everything *around* the model. A realtime session is a real Pydantic AI
agent: it runs your typed tools with their dependencies, retries, and validation **server-side**; it
builds the same message history your text agents produce; it tracks usage and enforces limits; and it's
instrumented with [Logfire](../logfire.md) out of the box — all provider-agnostic across OpenAI, Azure,
xAI, and Google. **You bring the audio transport; Pydantic AI is the intelligence behind the voice.**

## One agent, many modalities

A realtime session isn't a separate world from the rest of your agent — it's the *same* agent wearing a
voice: the same tools, dependencies, instructions, and, crucially, the same conversation history.
Because a session records canonical `ModelRequest`/`ModelResponse` messages, you can:

- **hand a call off to a text agent** — pass `session.all_messages()` straight into `Agent.run()` to
  finish a task in the background, summarize it, or continue over chat;
- **seed a call from a prior conversation** — start a voice session with the history of an earlier text
  or voice interaction;
- **share one agent across channels** — the voice bot on the phone and the assistant in your app can be
  the same `Agent`, with the same behavior and the same audit trail.

Your investment in tools, evals, and observability carries straight over to voice. See
[Message history](#message-history) and
[One agent, many modalities: technical details](#one-agent-many-modalities-technical-details).

## How it works

Your Python backend opens the provider WebSocket and runs the
[`RealtimeSession`][pydantic_ai.realtime.RealtimeSession]. Your application owns audio capture and
playback: it streams microphone bytes into the session and plays audio events as they arrive. Unlike
batch STT → text generation → TTS, the model hears and speaks directly in one live session.
Browser/WebRTC and telephony stacks remain transport concerns that connect users to the backend.

## Choose your path

| Path | Best for | Agent loop runs | Where Pydantic AI fits |
| --- | --- | --- | --- |
| **Native speech-to-speech with Pydantic AI** | Low-latency voice agents with server-side tools and shared history | On your backend | Runs the complete provider-agnostic realtime agent described here |
| **Browser talks directly to the provider** | Provider-native, UI-only experiences using an ephemeral token | In the browser/provider SDK | Can power separate backend workflows; use the provider SDK for the direct media session |
| **Batch STT → text agent → TTS** | Text-model choice, structured output, or independent speech components | In a normal Pydantic AI run | Compose a standard [agent](../agent.md) with chosen STT and TTS services |

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

The audio capture and playback seams are yours to implement; the
[voice assistant example](../examples/realtime-voice.md) shows a complete, runnable `sounddevice`
microphone/speaker version of this quickstart, and [Connecting a frontend](#connecting-a-frontend)
covers browser/telephony transports.

## The event loop

Handle the events your application needs: play audio deltas, render transcripts, show tool lifecycle
updates, mark turns complete, react to reconnection, and surface recoverable errors.

```python {test="skip" lint="skip"}
async for event in session:
    match event:
        case PartDeltaEvent(delta=SpeechPartDelta(audio_chunk=chunk)) if chunk:
            play_audio(chunk)
        case PartEndEvent(part=SpeechPart(transcript=transcript)):
            show_transcript(transcript)
        case FunctionToolCallEvent():
            show_tool_status('running')
        case FunctionToolResultEvent():
            show_tool_status('complete')
        case TurnCompleteEvent(interrupted=interrupted):
            finish_turn(interrupted)
        case ReconnectedEvent(state_restored=state_restored):
            show_reconnected(state_restored)
        case SessionErrorEvent(message=message):
            show_error(message)
```

### Event reference

A session speaks the same event vocabulary as a standard streamed agent run: iterating a
[`RealtimeSession`][pydantic_ai.realtime.RealtimeSession] yields the shared message/part events from
[`pydantic_ai.messages`][pydantic_ai.messages] for content, plus realtime control-plane events.

Spoken content (both the user's and the model's) streams as a
[`SpeechPart`][pydantic_ai.messages.SpeechPart] (distinguished by
`speaker`), assembled through the standard part events:

| Event | Meaning |
| --- | --- |
| [`PartStartEvent`][pydantic_ai.messages.PartStartEvent] | A new part started — a `SpeechPart` (assistant or user), a `ToolCallPart`, or a plain `TextPart` when [`output_modality='text'`](#configuring-shared-settings). |
| [`PartDeltaEvent`][pydantic_ai.messages.PartDeltaEvent] | A [`SpeechPartDelta`][pydantic_ai.messages.SpeechPartDelta]: `audio_chunk` for playback and/or `transcript_delta` for incremental text (a [`TextPartDelta`][pydantic_ai.messages.TextPartDelta] in text mode). |
| [`PartEndEvent`][pydantic_ai.messages.PartEndEvent] | The completed part: speech has `transcript` and optional retained `audio`, text has `content`, and tool parts carry their own fields. |
| [`FunctionToolCallEvent`][pydantic_ai.messages.FunctionToolCallEvent] | The session began executing a tool the model requested (carries the `ToolCallPart`). |
| [`FunctionToolResultEvent`][pydantic_ai.messages.FunctionToolResultEvent] | The tool finished and produced a [`ToolReturnPart`][pydantic_ai.messages.ToolReturnPart] or [`RetryPromptPart`][pydantic_ai.messages.RetryPromptPart]. Normally the result is sent to the model; a provider-cancelled call records a synthetic cancellation result only in local history. |
| [`DeferredToolRequestsEvent`][pydantic_ai.messages.DeferredToolRequestsEvent] | The original deferred or approval-required requests resolved by an inline capability handler. |
| [`DeferredToolResultsEvent`][pydantic_ai.messages.DeferredToolResultsEvent] | The inline handler supplied results and normal tool processing continues. |

The remaining realtime control-plane events:

| Event | Meaning |
| --- | --- |
| [`InputSpeechStartEvent`][pydantic_ai.realtime.InputSpeechStartEvent] | OpenAI/Azure/xAI detected speech onset, or Gemini reported activity that interrupted model output. |
| [`InputSpeechEndEvent`][pydantic_ai.realtime.InputSpeechEndEvent] | OpenAI/Azure/xAI detected the end of speech; Gemini does not emit this event. |
| [`InputTranscriptionFailedEvent`][pydantic_ai.realtime.InputTranscriptionFailedEvent] | The provider could not transcribe a user audio turn. The session continues, and `item_id` and `content_index` identify the affected turn when available. |
| [`TurnCompleteEvent`][pydantic_ai.realtime.TurnCompleteEvent] | The model finished a turn. `interrupted` reflects cancellation or barge-in across all providers. |
| [`ReconnectedEvent`][pydantic_ai.realtime.ReconnectedEvent] | The connection dropped and was automatically re-established. Conversation state is restored for Gemini and xAI; see [Reconnecting](#reconnecting). |
| [`SessionErrorEvent`][pydantic_ai.realtime.SessionErrorEvent] | The provider reported a **recoverable** error mid-session; the session keeps running. A non-recoverable error instead raises [`RealtimeError`][pydantic_ai.realtime.RealtimeError]. |

## Core tasks

### Configuring shared settings

[`RealtimeModelSettings`][pydantic_ai.realtime.RealtimeModelSettings] defines the common settings
vocabulary: `tool_choice`, `parallel_tool_calls`, `max_tokens`, `voice`,
`input_transcription_model`, `output_modality`, `handshake_timeout`,
[`turn_detection`][pydantic_ai.realtime.TurnDetection], and
[`thinking`][pydantic_ai.realtime.RealtimeModelSettings.thinking]. Pass settings for one session,
set defaults on the realtime model, or combine both — per-session values override model defaults:

```python {test="skip" lint="skip"}
from pydantic_ai.realtime.openai import OpenAIRealtimeModel, OpenAIRealtimeModelSettings

defaults = OpenAIRealtimeModelSettings(voice='alloy', max_tokens=2_000)
model = OpenAIRealtimeModel('gpt-realtime', settings=defaults)

async with agent.realtime_session(
    model=model,
    model_settings=OpenAIRealtimeModelSettings(max_tokens=4_000),
) as session:
    ...
```

The agent's regular `model_settings` and capability `get_model_settings()` contributions do not
apply to realtime sessions. Gemini ignores `tool_choice`, `parallel_tool_calls`,
`input_transcription_model`, and `handshake_timeout`; xAI ignores `output_modality` and
`thinking` and always produces audio. OpenAI, Azure OpenAI, and xAI do not expose `temperature`
through Pydantic AI. See the provider pages for provider-specific settings and exceptions.

### Turn-taking and barge-in

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
        # `interrupt()` (and `audio_end_ms`) require provider support — OpenAI and Azure OpenAI here.
        # Gemini and xAI handle barge-in themselves; see the model profile reference below.
        await session.interrupt(audio_end_ms=speaker.played_ms())
```

`interrupt()` is server-side only — it does not flush your local playback buffer; that is the
caller's responsibility. Explicit `interrupt()` and manual turn-taking require provider support (see
[model profile](#model-profile-reference)); Gemini Live handles barge-in automatically and exposes neither. The
`audio_end_ms` truncation additionally needs [`supports_output_truncation`](#model-profile-reference), which xAI Grok
Voice lacks — call `interrupt()` without `audio_end_ms` there.

#### Push-to-talk (manual turn-taking)

Automatic detection is **on by default**; disable it with `turn_detection=False` for push-to-talk.
This is only supported on providers that expose manual turn control through Pydantic AI (OpenAI,
Azure OpenAI, and xAI — see [`supports_manual_turn_control`](#model-profile-reference)); Gemini Live has no manual turn verbs, so
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

### Tool calling

Tools registered on the agent are offered to the realtime model. When the model calls one, the
session emits a [`FunctionToolCallEvent`][pydantic_ai.messages.FunctionToolCallEvent], runs the tool,
normally sends the result back, and emits a
[`FunctionToolResultEvent`][pydantic_ai.messages.FunctionToolResultEvent]. Parse failures and
[`ModelRetry`][pydantic_ai.exceptions.ModelRetry] produce a
[`RetryPromptPart`][pydantic_ai.messages.RetryPromptPart], matching a standard agent run. Other tool
exceptions end the session and propagate to the caller.
If the provider cancels an in-flight call, Pydantic AI cancels the task and emits a synthetic
cancellation result for valid local history, but does not send that abandoned result back to the
provider.

Realtime provider tool-output channels are string-only, but session history preserves the structured
`return_value`, `content`, and `metadata` of a [`ToolReturn`][pydantic_ai.messages.ToolReturn]. The
return value is rendered as text only when it is sent over the provider's tool-output channel.
OpenAI and xAI receive `content` as a follow-up user conversation item; Gemini receives its textual
representation.

#### Concurrent tools

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

#### Built-in tools (web search)

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

### Message history

A realtime session builds the same [`ModelMessage`][pydantic_ai.messages.ModelMessage]
[history](../message-history.md) a standard agent run does, so a voice conversation composes with the
rest of the framework: seed a session from an earlier conversation, and hand a finished session off
to a text [`Agent.run`][pydantic_ai.agent.AbstractAgent.run] for summarization, structured
extraction, or follow-up. Spoken turns are recorded as
[`SpeechPart`][pydantic_ai.messages.SpeechPart]s; everything else is the
ordinary [`ModelRequest`][pydantic_ai.messages.ModelRequest] /
[`ModelResponse`][pydantic_ai.messages.ModelResponse] shape, including tool calls and results.
Images and video frames streamed with `send()` are recorded as user image turns by default, so a
later handoff includes the visual context the realtime model received. For high-rate camera or screen
streams, use `retain_images_every_n` to sample the recorded frames and limit history memory use.

The session exposes two snapshots (each a copy, so it won't change as the session continues):

| Method | Returns |
| --- | --- |
| [`session.all_messages()`][pydantic_ai.realtime.RealtimeSession.all_messages] | The seeded history plus completed spoken/text turns, retained images, and tool rounds recorded by this session. |
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
and images. OpenAI, Azure OpenAI, and xAI replay function tools as native call/result items; Gemini Live cannot put
function parts in seeded turns, so it uses readable `[Tool call: ...]` and
`[Tool "..." returned: ...]` / `[Tool "..." error: ...]` text instead. Thinking signatures and
provider details are not replayed because they belong to the provider session that produced them.
Provider-executed native-tool metadata is also omitted: the answer it produced is already in the
history, and the execution itself cannot be replayed in a new session.

Content that cannot be represented is rejected with a [`UserError`][pydantic_ai.exceptions.UserError]
instead of being dropped silently. In particular, video, documents, uploaded-file references, and
model-generated files cannot be seeded. Speech transcripts are preferred over retained audio. When a
user [`SpeechPart`][pydantic_ai.messages.SpeechPart] has no transcript, OpenAI and Azure OpenAI can
replay retained input audio; Gemini and xAI cannot. Enable `input_transcription_model` or `audio_retention` when
capturing the original session, or filter unseedable parts from `message_history` before connecting.
Assistant speech always requires a transcript. See the [model profile](#model-profile-reference) for provider
support.

#### Retaining audio

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

#### Retaining images

`retain_images_every_n=1` (the default) records every image sent with
[`session.send()`][pydantic_ai.realtime.RealtimeSession.send]. Set it to `N > 1` to keep the first
image and then one of every `N` sent images. The provider still receives every frame; sampling only
affects local [`all_messages()`][pydantic_ai.realtime.RealtimeSession.all_messages] and
[`new_messages()`][pydantic_ai.realtime.RealtimeSession.new_messages] history. Keeping every image is
the least surprising choice for one-off vision prompts, while sampling avoids unbounded memory growth
for continuous video.

### Transcribing user input

To hand a voice session off to a text agent, the user's turns need to reach history as text. OpenAI and
xAI transcribe the user's audio with a dedicated model, set via `input_transcription_model`:

| Value | Behaviour |
| --- | --- |
| `'auto'` (default) | The provider's recommended transcription model, so user turns are captured under the default `audio_retention='transcript_only'`. Pin a specific id when the transcription model must remain fixed. |
| An explicit id (e.g. `'gpt-4o-transcribe'`) | Used verbatim. Known ids autocomplete via [`KnownRealtimeTranscriptionModelName`][pydantic_ai.realtime.KnownRealtimeTranscriptionModelName], but any string works. |
| `None` | Transcription disabled — no transcription model is sent. No user transcripts arrive, so [`audio_retention`](#retaining-audio) **must** be `'input_audio'`/`'all'` to keep the raw audio; each user turn is then finalized as an audio-only [`SpeechPart`][pydantic_ai.messages.SpeechPart] (no transcript, so not usable for a text handoff). Disabling transcription while `audio_retention` doesn't retain input audio raises a `UserError`, since the user's turns would otherwise be silently dropped from history. |

Gemini transcribes with `google_input_transcription` (on by default) rather than a model id: the
Live model transcribes natively, so there is no separate transcription model to choose. If
`google_input_transcription=False`, set `audio_retention='input_audio'` or `'both'`; otherwise session
creation raises [`UserError`][pydantic_ai.exceptions.UserError] because user turns could not be
recorded. If `google_output_transcription=False`, retain output audio to keep assistant audio turns
in history at all. Transcript-less assistant audio cannot be handed off to a text agent or seeded
into another realtime session.

OpenAI and Gemini may stream partial user transcripts. xAI suppresses revisable partial snapshots
and emits the finalized user transcript at the end of the turn.

### Images

Send an image as conversation context (for example a video frame) with
[`send`][pydantic_ai.realtime.RealtimeSession.send]. An image does not itself trigger a
response — the model picks it up on the next turn (via VAD, a text turn, or `create_response` where
manual turn-taking is supported).

```python {test="skip" lint="skip"}
from pydantic_ai import BinaryContent

await session.send(BinaryContent(data=jpeg_bytes, media_type='image/jpeg'))
```

**Live vision (Gemini).** For a "show the camera and ask about it" experience, stream frames
continuously and set `google_turn_coverage='all_video'` so every frame stays in context. Because a frame
alone never triggers a turn, drive proactive narration by periodically sending a short text turn
("say what changed, else stay silent"); combine with `google_proactive_audio=True` (native-audio) so the
model keeps quiet when nothing changed.

### Usage and cost

The session accumulates token usage as the model responds. Read it from
[`RealtimeSession.usage`][pydantic_ai.realtime.RealtimeSession.usage] — a
[`RunUsage`][pydantic_ai.usage.RunUsage] with input/output tokens (including audio and cached
breakdowns) and tool-call counts. Usage updates are accumulated on `session.usage` and are not
yielded as session events.

Input-transcription (ASR) usage is reported separately in
[`RunUsage.details`][pydantic_ai.usage.RunUsage.details] under `input_transcription_*` buckets. It is
not included in the response token totals or attributed to any [`ModelResponse`][pydantic_ai.messages.ModelResponse],
because transcription of the user's input audio uses a separate model and billing meter.

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

### Observability with Logfire

Realtime sessions emit OpenTelemetry spans when the agent is instrumented — call
`logfire.instrument_pydantic_ai()` (or set `instrument=True` on the agent). You get a session span
(reported as an agent invocation, like a classic run) carrying cumulative usage and the conversation
transcript (content is redacted when `include_content` is disabled). The session span uses
`gen_ai.operation.name='invoke_agent'` and sets `gen_ai.output.type` to `'speech'` or `'text'`.
Nested under it, each assistant response gets a `chat {model}` span with
`gen_ai.output.type` set to the same value, and each tool call gets an
`execute_tool` span (including any delegated text-agent run). See
[Debugging and monitoring](../logfire.md).

OpenAI, Azure OpenAI, and xAI `chat` spans carry the response's own usage, including function-call-only responses.
Gemini finalizes a function-call response before the provider reports usage; that response has zero
usage, while Gemini's later completed turn carries the reported turn usage. The cumulative
`session.usage` and session span remain authoritative totals.

```python {test="skip" lint="skip"}
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()
# realtime_session spans now appear in Logfire
```

## Connecting a frontend

Pydantic AI is the *agent* — it runs server-side so your keys, tools, and business logic stay on your
backend. You connect a media transport to it. There are three common shapes:

- **Browser → your backend → provider (a WebSocket relay).** Your server runs the `RealtimeSession` and
  relays audio to and from the browser. Keys and tools stay server-side and you keep the full agent
  loop. This is what the [realtime camera example](../examples/realtime-camera.md) does.
- **A media room (WebRTC) with a server-side agent participant.** A platform like LiveKit carries
  browser/mobile/telephony media into a room; your backend joins as a participant running the
  `RealtimeSession`. You get browser-grade media (echo cancellation, jitter handling, device SDKs) *and*
  the server-side agent loop.
- **The phone (SIP/telephony).** A provider like Twilio or LiveKit terminates the call and bridges its
  audio to your backend session.

What you don't do is wire the browser *straight* to the model provider: that moves the agent loop into
the browser and gives up server-side tools, history, and secrets. Pydantic AI is the brain; bring the
transport that reaches your users.

```text
device ↔ media bridge ↔ RealtimeSession ↔ provider
                         ├── typed tools
                         └── message history
                         (your backend)
```

Keep provider keys on the server. Give browsers and rooms only short-lived tokens scoped to their
connection; never ship backend credentials to a client.

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

For OpenAI and Azure OpenAI, reconnecting restores the session configuration but **not** server-side
conversation state (the audio buffer and prior turns), so treat a
[`ReconnectedEvent`][pydantic_ai.realtime.ReconnectedEvent] with `state_restored=False` as the start of a fresh
turn. Without a
policy (the default), a dropped connection raises [`RealtimeError`][pydantic_ai.realtime.RealtimeError]
from the session iterator, so the app can open a new session itself.

Gemini and xAI reconnect via native **session resumption**, which restores prior turns. xAI suppresses
the provider's resumption replay burst from the local event stream and enables resumption automatically
whenever a [`ReconnectPolicy`][pydantic_ai.realtime.ReconnectPolicy] is set. Its conversation handle is
managed in memory and cannot be persisted to resume a session in another process. Their
[`ReconnectedEvent`][pydantic_ai.realtime.ReconnectedEvent] has `state_restored=True`.

For Gemini, enable resumption with
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

## One agent, many modalities: technical details

### Relationship to `run` / `iter`

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
| Provider-side session handle/resumption | ⚠️ no persistent handle API; Gemini and xAI support in-process [session resumption](#reconnecting) |
| `user_prompt` | ❌ stream input with `send_audio()` or `send()` instead |
| `retries` | ❌ no per-session argument or output-validation retries; agent-level tool retries still apply |
| `deferred_tool_results`, `event_stream_handler` | ❌ graph-only (the session *is* the event stream) |

Realtime-only: `audio_retention` keeps spoken audio bytes on the history parts, while
`retain_images_every_n` controls image-history sampling. Tool calls always run concurrently with the
session.

[Deferred tools](../deferred-tools.md) work in a session to the extent they can be resolved *live*: a
[`HandleDeferredToolCalls`][pydantic_ai.capabilities.HandleDeferredToolCalls] capability handler is
invoked inline when a tool [requires approval](../deferred-tools.md) or raises
[`CallDeferred`][pydantic_ai.exceptions.CallDeferred], and can approve, deny, or answer the call on
the spot. What has no realtime analog is the graph's *pause*: a session can't end with a
`DeferredToolRequests` output and wait for out-of-band results, so when no handler resolves the
call, the model instead receives an explanation that the tool can't be completed during a realtime
session, and the conversation continues.

[`ctx.enqueue()`][pydantic_ai.tools.RunContext.enqueue] accepts one plain-text prompt per call from a
realtime tool. The default `priority='asap'` sends it into the live conversation promptly;
`priority='when_idle'` sends it after the next [`TurnCompleteEvent`](#event-reference). Delivered text is
recorded as a normal user turn. Multimodal content and prebuilt message/part sequences are rejected
because the realtime live-input channel cannot preserve their full classic-run semantics.

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

#### Capabilities

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
  [event stream](#event-reference), not an agent-run one,
- output hooks (`*_output_validate`, `*_output_process`, `prepare_output_tools`) — a session has no
  `output_type` ([delegate](#delegating-to-a-text-agent) structured output to a text agent),
- history-processing hooks — `ProcessHistory` and other classic-run history processors are not
  applied to `message_history` before realtime session seeding; the supplied history is projected
  directly into provider seed items,
- deferred capability loading.

So a capability that scopes, guards, or instruments **tools**, or contributes tools/toolsets/native
tools/instructions, works unchanged in a session; one that hooks the **model request, run, graph, or
output** is inert here. There are no session- or turn-level capability hooks; use the tool hooks and
the [session event stream](#event-reference).

### Delegating to a text agent

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

## Troubleshooting

### No audio

**Symptom:** no useful speech is heard. **Cause:** the input/output PCM rate or channel count does not
match the profile. **Fix:** send mono PCM16 at `audio_input_sample_rate` and play it at
`audio_output_sample_rate`.

### The model never responds

**Symptom:** push-to-talk audio produces silence. **Cause:** manual mode does not infer turn end or
request a response. **Fix:** call `commit_audio()`, then `create_response()`.

### The model interrupts itself

**Symptom:** playback triggers input speech. **Cause:** the microphone hears speaker echo. **Fix:**
enable echo cancellation in the device/WebRTC layer and stop playback promptly on real barge-in.

### Truncation cuts the wrong point

**Symptom:** interrupted context is cut incorrectly. **Cause:** `audio_end_ms` measures audio
received, not played. **Fix:** pass the duration the user actually *heard*.

### Tools seem to stall

**Symptom:** a turn appears to wait on a slow tool. **Cause:** the tool itself is long-running.
**Fix:** surface lifecycle events, let the model keep talking where appropriate, and see
[Concurrent tools](#concurrent-tools).

### Reconnect lost the conversation

**Symptom:** earlier server context is gone. **Cause:** providers emitting `state_restored=False`
start fresh. **Fix:** treat the event as a fresh turn; see [Reconnecting](#reconnecting).

### Gemini reaches its session limit

**Symptom:** Gemini sends `GoAway` and later drops. **Cause:** a provider-defined session length
limit. **Fix:** combine [`ReconnectPolicy`][pydantic_ai.realtime.ReconnectPolicy] with
`google_enable_session_resumption=True`; resumption uses the latest server handle after the drop.

## Provider support

All providers implement the same [`RealtimeModel`][pydantic_ai.realtime.RealtimeModel] interface.
OpenAI, Azure, and xAI use mono PCM16 at 24 kHz both ways; Gemini uses 16 kHz input and 24 kHz output.

| Provider | Audio/text output | Image input | Manual turns | Output truncation | Semantic VAD | Input transcription | Native tools | Usage | State-restoring reconnect |
| --- | --- | :---: | :---: | :---: | :---: | --- | --- | --- | :---: |
| [OpenAI](openai.md) | Audio or text | ✓ | ✓ | ✓ | ✓ | Dedicated model; `'auto'` default | ✗ | Tokens, audio, cached breakdowns | ✗ |
| [Azure OpenAI](azure.md) | Audio or text | ✓ | ✓ | ✓ | ✓ | Dedicated model; `'auto'` default | ✗ | Tokens, audio, cached breakdowns | ✗ |
| [xAI](xai.md) | Audio | ✗ | ✓ | ✗ | ✗ | Dedicated model; `'auto'` default | ✗ | Tokens, audio buckets, billable seconds | ✓ |
| [Google Gemini](gemini.md) | One modality/session: audio or text | ✓ | ✗ | ✗ | ✗ | Native; on by default | Search, URL context, code execution; model-dependent | Tokens and modality breakdowns | ✓, when enabled |

Gemini handles barge-in automatically instead of through explicit `interrupt()`. Provider/model
variants can differ, especially for native tools and reasoning; inspect the profile before branching.

### Model profile reference

Realtime providers differ in what they support, so features like manual turn-taking and barge-in
aren't universal. Each model reports its support through
[`RealtimeModel.profile`][pydantic_ai.realtime.RealtimeModel.profile], a
[`RealtimeModelProfile`][pydantic_ai.realtime.RealtimeModelProfile] — the realtime counterpart to the
[`ModelProfile`][pydantic_ai.profiles.ModelProfile] of a standard [`Model`][pydantic_ai.models.Model]:

The profile also exposes `audio_input_sample_rate` and `audio_output_sample_rate`, in Hz. OpenAI,
Azure OpenAI, and xAI use 24 kHz in both directions; Gemini uses 16 kHz input and 24 kHz output.

| [`RealtimeModelProfile`][pydantic_ai.realtime.RealtimeModelProfile] flag | Gates | OpenAI | Azure OpenAI | Gemini | xAI |
| --- | --- | :---: | :---: | :---: | :---: |
| [`supports_image_input`][pydantic_ai.realtime.RealtimeModelProfile.supports_image_input] | [`send`](#images) | ✅ | ✅ | ✅ | ❌ |
| [`supports_manual_turn_control`][pydantic_ai.realtime.RealtimeModelProfile.supports_manual_turn_control] | [`commit_audio`/`clear_audio`/`create_response`](#push-to-talk-manual-turn-taking) | ✅ | ✅ | ❌ | ✅ |
| [`supports_interruption`][pydantic_ai.realtime.RealtimeModelProfile.supports_interruption] | [`interrupt`](#turn-taking-and-barge-in) | ✅ | ✅ | ❌ | ✅ |
| [`supports_output_truncation`][pydantic_ai.realtime.RealtimeModelProfile.supports_output_truncation] | [`interrupt(audio_end_ms=…)`](#turn-taking-and-barge-in) | ✅ | ✅ | ❌ | ❌ |
| [`supports_session_seeding`][pydantic_ai.realtime.RealtimeModelProfile.supports_session_seeding] | [`message_history=`](#message-history) | ✅ | ✅ | ✅ | ✅ |
| [`supports_seeding_images`][pydantic_ai.realtime.RealtimeModelProfile.supports_seeding_images] | Images in `message_history` | ✅ | ✅ | ✅ | ❌ |
| [`supports_seeding_audio`][pydantic_ai.realtime.RealtimeModelProfile.supports_seeding_audio] | Transcript-less retained user audio in `message_history` | ✅ | ✅ | ❌ | ❌ |
| [`supports_thinking`][pydantic_ai.realtime.RealtimeModelProfile.supports_thinking] | [`thinking`](openai.md#reasoning) | `gpt-realtime-2*` | `gpt-realtime-2*` | Native-audio models | ❌ |

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

### Gateway

#### Routing through a gateway

Gateway routing covers OpenAI realtime (`gateway/openai`) and Gemini Live (`gateway/google`, which
proxies the Vertex upstream — see [Gemini gateway](gemini.md#routing-through-the-gateway)).

Realtime models take a `provider=` just like standard models, so you can route a session through the
[Pydantic AI Gateway](../gateway.md) (or, for OpenAI, any OpenAI-compatible endpoint that exposes a
realtime API) by naming the upstream provider:

```python {test="skip" lint="skip"}
from pydantic_ai.realtime.openai import OpenAIRealtimeModel

model = OpenAIRealtimeModel('gpt-realtime', provider='gateway/openai')
```

You can also infer the model from a gateway-prefixed identifier. This reads gateway credentials from
[`gateway_provider`][pydantic_ai.providers.gateway.gateway_provider]:

```python {test="skip" lint="skip"}
from pydantic_ai.realtime import infer_realtime_model

model = infer_realtime_model('gateway/openai:gpt-realtime')
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

## Limitations

Some capabilities are intentionally out of scope:

- **Browser-direct transport (WebRTC).** Sessions run server-side over WebSocket; there is no direct browser-to-provider WebRTC path.
- **Telephony (SIP).** Connecting a session to a phone call over SIP is not built in.
- **Session resumption beyond automatic reconnect.** You can't persist a handle and resume a session in a later process; recovery is limited to in-process [reconnection](#reconnecting).
- **Bounded structured-output runs.** A session has no `output_type` or `session.run()` with an output schema — [delegate to a text agent](#delegating-to-a-text-agent) for structured results.
- **Realtime-specific capability hooks.** Capabilities run once for setup and apply their instructions, toolsets, and native tools; tool-lifecycle hooks also run, but there are no before/after-exchange hooks.
- **History processing during realtime seeding.** History processors run on classic agent runs, not when a realtime session seeds `message_history`; preprocess the history before passing it when redaction, summarization, or filtering is required.
- **Dynamic instructions mid-session.** Instructions are resolved once at connect and not re-evaluated during the session.
- **Provider-limited audio replay when seeding history.** OpenAI and Azure OpenAI can replay retained user audio when a [`SpeechPart`][pydantic_ai.messages.SpeechPart] has no transcript. Gemini and xAI cannot seed retained audio, and no provider can seed assistant audio; use transcripts or filter those parts before connecting (see [Message history](#message-history)).
- **Proactive resume before Gemini's session cap.** Gemini Live signals an upcoming disconnect (`GoAway`) near its session-length limit, but the session only [reconnects](#reconnecting) after a drop.

## Implementing a provider

A provider implements two ABCs: [`RealtimeModel`][pydantic_ai.realtime.RealtimeModel]
(opens a connection) and [`RealtimeConnection`][pydantic_ai.realtime.codec.RealtimeConnection]
(sends [`RealtimeInput`][pydantic_ai.realtime.codec.RealtimeInput] and yields the low-level
[`RealtimeCodecEvent`][pydantic_ai.realtime.codec.RealtimeCodecEvent] vocabulary, which the session
translates into user-facing [`RealtimeEvent`][pydantic_ai.realtime.RealtimeEvent]s). The OpenAI
provider in [`pydantic_ai.realtime.openai`][pydantic_ai.realtime.openai] is a reference
implementation; the same shape applies to Azure OpenAI, Gemini Live, xAI Grok Voice, and others. Inputs a provider
doesn't support (e.g. `ImageInput`, or the manual turn-taking verbs) should raise `NotImplementedError`.
