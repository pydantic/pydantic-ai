# OpenAI GPT-Live

[`OpenAILiveModel`][pydantic_ai.realtime.openai_live.OpenAILiveModel] connects an agent to OpenAI's GPT-Live
API. Live is a separate protocol from the [OpenAI Realtime API](openai.md), not a model served by it:
the Live model runs the spoken conversation and hands the thinking to a *backend* model that Pydantic
AI configures with the agent's instructions and tools. Start with the
[realtime quickstart](overview.md#quickstart) for the shape of a session.

A Live session is more constrained than a Realtime one. It owns turn-taking entirely, takes no
images, and bills by the second rather than by the token, so read
[Feature support and limitations](#feature-support-and-limitations) before porting code between the
two.

## Setup

To use GPT-Live, install `pydantic-ai-slim` with the `openai-realtime` optional group, which bundles
the `openai` package together with the realtime WebSocket transport. Live's event types arrived in
`openai` 3.12, and the group floors it there:

```bash
pip/uv-add "pydantic-ai-slim[openai-realtime]"
```

Set `OPENAI_API_KEY` as described in the
[OpenAI model documentation](../models/openai.md#configuration). Authentication and base URL come
from `provider`, mirroring [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel]; pass an
[`OpenAIProvider`][pydantic_ai.providers.openai.OpenAIProvider] for a custom key or base URL. The
WebSocket opens separately, so a custom provider `httpx` client is not used for it. Azure OpenAI does
not serve GPT-Live: an `azure` provider raises
[`UserError`][pydantic_ai.exceptions.UserError] and points at
[`AzureRealtimeModel`][pydantic_ai.realtime.azure.AzureRealtimeModel] and the
[Azure page](azure.md) instead.

## Model names

Use `gpt-live-1`. Any model name starting with `gpt-live` routes to
[`OpenAILiveModel`][pydantic_ai.realtime.openai_live.OpenAILiveModel]; every other `openai:` realtime name
routes to [`OpenAIRealtimeModel`][pydantic_ai.realtime.openai.OpenAIRealtimeModel], because the
provider prefix alone cannot tell the two protocols apart. Use the
[official OpenAI model documentation](https://platform.openai.com/docs/models) as the canonical model
list.

The *backend* model is a second model name nested inside the session: it defaults to `gpt-5.6-sol`
and is set with `openai_live_delegation`, described below.

## How delegation works

Live splits one agent across two models:

| Piece | Where it goes |
| --- | --- |
| The agent's [instructions](../agent.md#instructions) | The backend model, which does the work |
| The agent's [tools](tools.md) | The backend model, advertised as function tools |
| `openai_live_instructions` | The Live model, which does the talking |

When the Live model decides it cannot answer from the conversation alone, it opens a *delegation* and
the backend model takes over, calling the agent's tools as it goes. Those calls arrive as ordinary
[`ToolCall`][pydantic_ai.realtime.codec.ToolCall]s, so the session runs them through the same
[tool loop](tools.md#function-tools) as every other provider, including validation, retries,
[dependencies](../dependencies.md), and [capability hooks](capabilities.md), and sends the
results back. The Live model keeps speaking while that happens, so tool latency does not create dead
air.

That split is why the agent's instructions describe the *work* and `openai_live_instructions`
describes the *speech*:

```python
from pydantic_ai import Agent
from pydantic_ai.realtime.openai_live import OpenAILiveModelSettings

agent = Agent(
    instructions='You handle order questions for The Terrace. Quote prices exactly as returned.'
)


@agent.tool_plain
async def lookup_order(order_id: str) -> str:
    """Look up an order by its ID."""
    return f'Order {order_id} ships tomorrow.'


realtime = agent.realtime(
    'openai:gpt-live-1',
    model_settings=OpenAILiveModelSettings(
        openai_live_instructions=(
            'Speak warmly and briefly. Say you are checking before you delegate a lookup.'
        ),
        openai_live_delegation={'model': 'gpt-5.6-sol', 'reasoning_effort': 'low'},
    ),
)
```

Live can also delegate to the *client* rather than to a backend model, leaving your application to
answer in prose. Pydantic AI configures backend delegation, so a client delegation is reported as a
[`RealtimeSessionErrorEvent`][pydantic_ai.realtime.RealtimeSessionErrorEvent] with code
`live_client_delegation` rather than stalling the call silently.

To run the harder reasoning under your own control instead, expose a tool that
[delegates to a standard agent](tools.md#delegating-work-during-a-call); that works here exactly as
it does on the other providers.

## Settings

[`OpenAILiveModelSettings`][pydantic_ai.realtime.openai_live.OpenAILiveModelSettings] is the realtime
counterpart of [model run settings](../agent.md#model-run-settings), and extends the
[shared settings](overview.md#shared-settings):

```python
from pydantic_ai.realtime.openai_live import OpenAILiveModel, OpenAILiveModelSettings

settings = OpenAILiveModelSettings(
    openai_voice='marin',
    openai_live_turn_silence_ms=1_500,
    openai_live_store=True,
    openai_live_delegation={'model': 'gpt-5.6-sol', 'verbosity': 'low'},
)
model = OpenAILiveModel('gpt-live-1', settings=settings)
```

| Setting | Purpose |
| --- | --- |
| `openai_voice` | The Live voice, e.g. `marin` (the provider default). Immutable once the session has started |
| `openai_live_instructions` | How the Live model speaks: pacing, style, and when to delegate |
| `openai_live_delegation` | The backend the session delegates to. [`OpenAILiveResponsesDelegation`][pydantic_ai.realtime.openai_live.OpenAILiveResponsesDelegation] carries `model`, extra `instructions`, `reasoning_effort`, `verbosity`, `max_output_tokens`, `parallel_tool_calls`, and `service_tier` |
| `openai_live_turn_silence_ms` | How long the model must stay quiet before the [turn boundary](#the-turn-boundary-is-inferred) is reported. Defaults to 2000 |
| `openai_live_store` | Whether OpenAI stores the session for later retrieval. Defaults to `False` |

Voice, audio format, and instructions are fixed for the life of the session; only the delegation
backend can be reconfigured, which is why these are session-start settings rather than things to
change mid-call. Live exposes no turn-detection, truncation, or token-limit controls, and the shared
settings that name them [raise rather than being ignored](#what-raises). It has no temperature or
other sampling control at all, on either the spoken model or the delegated backend.

## The turn boundary is inferred

Live sends no end-of-response frame and no transcript-done event, so there is nothing on the wire
that marks the end of a reply. The connection infers one: once the model has produced no speech for
`openai_live_turn_silence_ms` and no delegated work is outstanding, it reports the turn complete and
the session emits [`RealtimeTurnCompleteEvent`][pydantic_ai.realtime.RealtimeTurnCompleteEvent]. The
end of the *user's* turn is inferred the same way.

The profile reports `synthesizes_turn_boundary=True` so an application can tell an inference from a
protocol fact (see [Provider support](overview.md#provider-support) for how to read a profile). Treat
the [turn boundary](events.md#the-turn-boundary) as a good guess here: a long dramatic pause can end
a turn early, and code that must not act on a partial reply should confirm against the transcript.
Lower the threshold for snappier turn-taking, raise it when replies contain long silences.

Delegated work suspends the clock. The model goes quiet while the backend thinks, and ending the turn
there would finalize a reply that is still coming.

## Text is context, not a user turn

Live has no client event that puts text into the conversation as the user's own words. Text is
delivered as context to the Live model instead, and it relays or answers it:

```python
from pydantic_ai.realtime import RealtimeSession


async def send_context(session: RealtimeSession) -> None:
    # Speakable: the model says this, or something close to it.
    await session.send('Tell the caller their table is ready.')

    # Silent: the model takes it into account without speaking it.
    await session.send('The caller is a returning guest named Ada.', respond=False)
```

Both forms are capped at 500 tokens by the provider.
[`enqueue()`](tools.md#enqueuing-prompts) delivers text the same way once the model is idle.

!!! warning "Text only lands while audio is flowing"
    A Live session's timeline advances with its audio, so text sent to a session whose microphone is
    not streaming is deferred rather than delivered. Keep
    [`send_audio()`][pydantic_ai.realtime.RealtimeSession.send_audio] running for the life of the
    call, streaming silence if the user is not speaking. That is why a text-driven session with no
    audio input, the shape the [text-to-audio example](../examples/realtime-text-to-audio.md) uses,
    does not work on Live.

Seeding is text-only in the same spirit: [`message_history=`](history.md#seeding-a-session) replays
text, transcripts, and thinking text, with tool rounds rendered as readable text because the protocol
has nowhere to put function parts. Audio and images in seeded history raise
[`UserError`][pydantic_ai.exceptions.UserError] rather than being dropped.

## Usage is measured in seconds

Live bills audio duration, not tokens. The session reports a running total of billable seconds, and
Pydantic AI records the increment in `details` on the session's
[`RunUsage`][pydantic_ai.usage.RunUsage] under `billable_audio_seconds`. The value belongs to the
session rather than to any one [`ModelResponse`][pydantic_ai.messages.ModelResponse], because Live
meters the call as a whole.

!!! warning "Audio seconds are reported on a timer"
    Live reports its running total periodically rather than per turn, so a call that ends within the
    first reporting interval records no `billable_audio_seconds` at all, and any call loses the
    seconds elapsed since the last report. Treat the figure as a floor on what the call cost, not an
    exact total, and reconcile against your OpenAI usage dashboard when the exact number matters.

The Live model itself reports no token counts, but the Responses backend it delegates to is billed
per token like any other model, and that usage is accumulated with its cache and reasoning
breakdowns intact. In a call that delegates, most of the token cost is there.

Those tokens are priced against the *backend's* model, not against `gpt-live-1`, because that is
what spent them — so a delegated turn's cost is right even though the
[`ModelResponse`][pydantic_ai.messages.ModelResponse] it lands on carries Live's name. The backend's
request is also what a `per_request_input_tokens_limit` is measured against, since it is the only
thing in a Live session that spends input tokens.

Session duration has a sharp consequence for [usage limits](observability.md#usage-and-limits): no
[`UsageLimits`][pydantic_ai.usage.UsageLimits] field caps it, so token and cost limits bound the
delegated backend but never the spoken call itself. Tool-call and request limits still apply. Cap
the call with your own timer or by closing the session.

## Feature support and limitations

| Feature | Support | Notes |
| --- | --- | --- |
| Audio format | Full feature support | Mono PCM16, 24 kHz input and output |
| Text input | Limited parameter support | [Context, not a user turn](#text-is-context-not-a-user-turn); capped at 500 tokens and delivered only while audio flows |
| Text output | Unsupported | Live always speaks, so `output_modality='text'` raises. Read the answer from the transcript on the [`SpeechPart`][pydantic_ai.messages.SpeechPart] |
| Image input | Unsupported | Audio and text only |
| Manual turns | Unsupported | Live owns turn-taking; `turn_detection` and the [commit/create verbs](turns.md#push-to-talk) raise |
| Interruption/truncation | Unsupported | [`interrupt()`](turns.md#barge-in) raises; Live handles barge-in itself |
| Turn boundary | Limited parameter support | [Inferred from silence](#the-turn-boundary-is-inferred), not reported by the provider |
| Input transcription | Full feature support | Always on in both directions; no [model to choose](audio.md#input-transcription) and no way to disable it |
| Input speech events | Unsupported | No speech start/end frames, so a "listening" indicator should read the profile rather than wait for events |
| Native tools | Unsupported | Configure [local fallbacks](tools.md#native-tools) for web capabilities |
| Async tool calls | Full feature support | The Live model keeps talking while the backend works |
| Thinking | Unsupported | Set backend reasoning with `openai_live_delegation={'reasoning_effort': ...}` instead |
| Usage | Limited parameter support | [Seconds, not tokens](#usage-is-measured-in-seconds); no duration-based `UsageLimits` field |
| Browser WebRTC | Unsupported | Bridge media through your backend; see [Connecting a frontend](deployment.md) |
| Reconnection | Unsupported | Automatic [reconnection](lifecycle.md#reconnecting) is not implemented for Live, so the [`reconnect`][pydantic_ai.realtime.RealtimeModelSettings.reconnect] policy is ignored and a dropped connection ends the session. Open a new one, seeding it with the previous session's history |

See [Audio, images, and transcripts](audio.md), [Turns and interruptions](turns.md),
[Tools](tools.md), and [Connection lifecycle](lifecycle.md) for the provider-agnostic workflows.

### What raises

Live refuses a stated requirement it cannot meet rather than accepting and ignoring it. These raise
[`UserError`][pydantic_ai.exceptions.UserError]:

- `turn_detection`, `max_tokens`, and `input_transcription_model`, before the session connects.
- `output_modality='text'`, because the profile reports `supports_text_output=False`
  (see [Shared settings](overview.md#shared-settings)).
- [`commit_audio()`][pydantic_ai.realtime.RealtimeSession.commit_audio],
  [`clear_audio()`][pydantic_ai.realtime.RealtimeSession.clear_audio],
  [`create_response()`][pydantic_ai.realtime.RealtimeSession.create_response], and
  [`interrupt()`][pydantic_ai.realtime.RealtimeSession.interrupt].
- Sending an image, and seeding history that contains audio or images.
- A [`ToolReturn`][pydantic_ai.messages.ToolReturn] whose `content` carries media. Live takes no
  media at all, so the result is refused before anything is sent rather than reaching the backend
  without the material that explains it. Put what the model needs in the tool's return value.

## Gateway

Whether the [Pydantic AI Gateway](../gateway.md) can route GPT-Live has not been verified. Connect
through `provider='openai'` or an
[`OpenAIProvider`][pydantic_ai.providers.openai.OpenAIProvider].

## Provider-specific quirks

- Live streams output audio as a continuous track for the whole session, ten frames a second and
  digitally silent between replies, so an arriving frame says nothing about whether the model is
  speaking. Pydantic AI drops the idle silence, and
  [`stream_audio()`][pydantic_ai.realtime.RealtimeSession.stream_audio] behaves as it does on every
  other provider: audio arrives when the model talks. Silences *inside* a reply are forwarded, so a
  mid-sentence pause does not become a gap in playback.
- Reasoning happens on the delegated backend and is not surfaced as
  [`ThinkingPart`][pydantic_ai.messages.ThinkingPart]s; the profile reports
  `supports_thinking=False` and the shared [`thinking`](../capabilities/thinking.md) setting does not
  apply. Use `openai_live_delegation` to configure the backend's effort.
- [Seeded](history.md#seeding-a-session) function calls and results are represented as readable text,
  as they are on [Gemini Live](gemini.md), for the same protocol reason.
- An [allow-list `tool_choice`](../agent.md#model-run-settings) is applied by trimming the tools
  advertised to the backend, because Live accepts only the declarative modes on the wire.
