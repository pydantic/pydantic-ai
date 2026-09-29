---
description: "Use OpenAI's GPT-Live and gpt-realtime voice models with Pydantic AI through OpenAILiveModel and OpenAIRealtimeModel: setup, model routing, delegation, voices, VAD settings, reasoning and browser WebRTC."
---

# OpenAI

Pydantic AI connects an agent to two families of OpenAI voice models, each spoken to over its own
protocol by its own model class:

- [`OpenAILiveModel`][pydantic_ai.realtime.openai_live.OpenAILiveModel] connects an agent to the
  GPT-Live family, over the GPT-Live API. The Live model runs the spoken conversation and hands the
  thinking to a *backend* model that Pydantic AI configures with the agent's instructions and tools
  (see [How GPT-Live works](#gpt-live-models)).
- [`OpenAIRealtimeModel`][pydantic_ai.realtime.openai.OpenAIRealtimeModel] connects an agent to
  OpenAI's gpt-realtime family of native speech-to-speech models, over the Realtime API. Live is a
  separate protocol, not a model served by the Realtime API.

You rarely need to choose between the classes yourself: an `openai:` model name picks the right one
(see [Model names](#model-names)). Start with the [realtime quickstart](overview.md#quickstart) for the
shape of a session, or the [text-to-audio example](../examples/realtime-text-to-audio.md) for
gpt-realtime.

## Setup

To use OpenAI realtime models, install `pydantic-ai-slim` with the `openai-realtime` optional
group, which bundles the `openai` package together with the realtime WebSocket transport. GPT-Live's
event types arrived in `openai` 3.12, and the group floors it there:

```bash
pip/uv-add "pydantic-ai-slim[openai-realtime]"
```

Set `OPENAI_API_KEY` as described in the [OpenAI model documentation](../models/openai.md#configuration).
Authentication and base URL come from `provider`, mirroring
[`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel]. The default `provider='openai'`
reads the environment; pass an [`OpenAIProvider`][pydantic_ai.providers.openai.OpenAIProvider] for a
custom key or base URL. The realtime WebSocket opens separately, so a custom provider `httpx` client
is not used for it. Sessions run over a server-side WebSocket by default; for browser voice, the
browser can exchange media directly over [WebRTC](#browser-webrtc) while your backend runs the agent
(see [Connecting a frontend](deployment.md#browser-webrtc-server-sideband)).

## Model names

| Family | Model class | Example model names | Protocol |
| --- | --- | --- | --- |
| [GPT-Live](#gpt-live-models) | [`OpenAILiveModel`][pydantic_ai.realtime.openai_live.OpenAILiveModel] | `gpt-live-1` | GPT-Live API (`/live/sessions`) |
| gpt-realtime | [`OpenAIRealtimeModel`][pydantic_ai.realtime.openai.OpenAIRealtimeModel] | `gpt-realtime`, `gpt-realtime-2.1`, `gpt-realtime-2.1-mini` | Realtime API (`/realtime`) |

Model availability and aliases can change; use the
[official OpenAI model documentation](https://platform.openai.com/docs/models) as the canonical model
list.

The provider prefix alone cannot tell the two protocols apart, so the model name picks the class: an
`openai:` (or `gateway/openai:`) model name starting with `gpt-live` routes to
[`OpenAILiveModel`][pydantic_ai.realtime.openai_live.OpenAILiveModel], and every other name routes to
[`OpenAIRealtimeModel`][pydantic_ai.realtime.openai.OpenAIRealtimeModel]:

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-5.6-sol', instructions='You are a helpful voice assistant.')

# GPT-Live, through `OpenAILiveModel`:
live = agent.realtime('openai:gpt-live-1')

# gpt-realtime, through `OpenAIRealtimeModel`:
realtime = agent.realtime('openai:gpt-realtime')
```

Two cases don't route by name:

- The `azure:` prefix always builds an
  [`AzureRealtimeModel`][pydantic_ai.realtime.azure.AzureRealtimeModel], whatever the model name.
  Azure OpenAI does not serve GPT-Live, and passing an `azure` provider to
  [`OpenAILiveModel`][pydantic_ai.realtime.openai_live.OpenAILiveModel] raises
  [`UserError`][pydantic_ai.exceptions.UserError]. See the [Azure page](azure.md).
- Constructing a model class directly skips the name routing: the class you construct is the protocol
  you get. When you instantiate one for model-level configuration, as the settings examples below do,
  pick the class for the model's family.

## How GPT-Live works {#gpt-live-models}

A Live session is more constrained than a gpt-realtime one. It owns turn-taking entirely, takes an
image only for its backend to look at, and bills by the second rather than by the token, so read
[its feature support](#gpt-live-feature-support-and-limitations) before porting code between the two.

### The backend model {#backend-model}

The *backend* model, the one that does the delegated work, is a second model name. Pydantic AI picks
it from the first of these that names one:

1. `openai_live_delegation={'model': ...}` in the [settings](#gpt-live-settings).
2. A model named after a `+` in the realtime model name: `'openai:gpt-live-1+gpt-6-luna'` delegates to
   `gpt-6-luna`.
3. The agent's own model, since the backend runs the agent's instructions and tools. It counts when it
   is an OpenAI model reached the same way as the Live model, directly or through the same gateway
   route, so an agent built on `'openai:gpt-5.6-sol'` delegates to `gpt-5.6-sol`. That holds with
   `defer_model_check=True` too: the name is resolved when the session connects, as the agent's own
   run would resolve it.
4. `'auto'`: the model Pydantic AI currently recommends
   ([`AUTO_BACKEND_MODEL`][pydantic_ai.realtime.openai_live.AUTO_BACKEND_MODEL]), which moves as OpenAI
   releases new models.

There is always a backend, so a session never fails for want of one; pin it (1 or 2) when its behavior
needs to stay put.

Live accepts any backend name when the session starts and only tries it when it first delegates. A
backend model that doesn't exist, or that your account can't use, would then fail every delegation, so
the first failure ends the session with a
[`RealtimeError`][pydantic_ai.realtime.RealtimeError] (code `live_backend_model_unavailable`) naming
the model, rather than leaving a call that can talk but never look anything up.

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-5.6-sol', instructions='You look things up.')

# Delegates to the agent's model, gpt-5.6-sol:
realtime = agent.realtime('openai:gpt-live-1')

# Delegates to gpt-6-luna instead:
realtime = agent.realtime('openai:gpt-live-1+gpt-6-luna')
```

### How delegation works

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
results back. Speech and delegated work run independently, so the Live model can keep talking while
that happens rather than leaving dead air.

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

Live has no command to cancel a delegation, so once the backend has the work it runs to completion,
tools included, even if the user takes the question back ("never mind") in the meantime, and the Live
model may still speak its result. Put tools that change state behind
[approval](tools.md#deferred-and-approval-required-tools) if a retracted request must not go through.

### The turn boundary is inferred

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

### Text is context, not a user turn

Live has no client event that puts text into the conversation as the user's own words. Text is
delivered as context to the Live model instead, and it relays or answers it:

```python
from pydantic_ai.realtime import RealtimeSession


async def send_context(session: RealtimeSession) -> None:
    # Speakable: the model says this, or something close to it.
    await session.send('Tell the caller their table is ready.')

    # Not a request to speak: the model takes it into account, and decides for itself whether to mention it.
    await session.send('The caller is a returning guest named Ada.', respond=False)
```

`respond=False` does not make the text silent. Live only promises that it doesn't *request* speech,
and in our testing the model usually acknowledged it out loud anyway ("Welcome back, Ada"). Word it as
background ("Internal note: ...") and say in `openai_live_instructions` what the model should keep to
itself.

Both forms are capped at 500 tokens by the provider. Longer text raises
[`UserError`][pydantic_ai.exceptions.UserError] before anything is sent, so it is neither recorded in
history nor waited for by [`wait_for_reply()`][pydantic_ai.realtime.RealtimeSession.wait_for_reply];
split it, or give long material to the backend as a tool result instead.
[`enqueue()`](tools.md#enqueuing-prompts) delivers text the same way once the model is idle. The
[`EnqueuedMessagesEvent`][pydantic_ai.messages.EnqueuedMessagesEvent] it produces marks when the text
was sent and recorded in history, not when the model took it in: as the warning below explains, that
only happens once audio is flowing.

!!! warning "Text only lands while audio is flowing"
    A Live session's timeline advances with its audio, so text sent to a session whose microphone is
    not streaming is deferred rather than delivered. Keep
    [`send_audio()`][pydantic_ai.realtime.RealtimeSession.send_audio] running for the life of the
    call, streaming silence if the user is not speaking. That is why a text-driven session with no
    audio input, the shape the [text-to-audio example](../examples/realtime-text-to-audio.md) uses,
    does not work on Live. It is also why
    [`wait_for_reply()`][pydantic_ai.realtime.RealtimeSession.wait_for_reply] after a `send()` waits
    until audio is flowing, and, since Live can take text as context without answering it, bound
    that wait with a timeout rather than relying on a reply always following.

Seeding is text-only in the same spirit: [`message_history=`](history.md#seeding-a-session) replays
text, transcripts, and thinking text, with tool rounds rendered as readable text because the protocol
has nowhere to put function parts. Audio and images in seeded history
raise [`UserError`][pydantic_ai.exceptions.UserError] rather than being dropped. Live accepts up to 128
seeded messages and 8,192 tokens in total, so seed a long conversation with its recent end. Within a
session, Live manages its own context: once it nears the limit, it continues from a summary of the
older conversation, so a long call does not keep every early detail verbatim. The fraction of its
context in use, as Live reports it, is
[`context_window_used`](history.md#context-window), which drops once Live summarizes.

### Images go to the backend

Live's voice model sees no images; its delegated backend does. Send one with `respond=True`, and the
backend runs on it straight away: Live opens a delegation, and speaks what the backend makes of the
image. The session tracks that like any other reply, so
[`wait_for_reply()`][pydantic_ai.realtime.RealtimeSession.wait_for_reply] waits for it.

```python
from pydantic_ai import BinaryImage
from pydantic_ai.realtime import RealtimeSession


async def show(session: RealtimeSession, photo: bytes) -> None:
    await session.send(BinaryImage(data=photo, media_type='image/jpeg'), respond=True)
```

An image sent without `respond=True` raises rather than being added as context. Queued for the backend
alone, it goes unseen until the backend next runs, and the voice model, not knowing it exists, answers
a question about it without looking. The profile reports this as `image_input_requires_response=True`.

### Usage is measured in seconds

Live bills audio duration, not tokens. The session reports a running total of billable seconds, and
Pydantic AI records the increase as `audio_seconds` on the session's
[`RunUsage`][pydantic_ai.usage.RunUsage], priced against the Live model, so `session.usage.cost`
includes the call itself once [genai-prices](https://github.com/pydantic/genai-prices) knows its rate
(see [keeping model prices up to date](../agent.md#keeping-model-prices-up-to-date) if your data
predates `gpt-live-1`). The value belongs to the session rather than to any one [`ModelResponse`][pydantic_ai.messages.ModelResponse], because Live
meters the call as a whole.

!!! warning "The last seconds of a call you close may not be recorded"
    Live reports its running total periodically rather than per turn, and sends the final total when
    the session ends. When Live ends the session, that final total is recorded. When you close it,
    Pydantic AI does not wait for it yet, so the seconds since Live's last periodic report are
    missing, and a short call can record none. Reconcile against your OpenAI usage dashboard when the
    exact number matters.

The Live model itself reports no token counts, but the Responses backend it delegates to is billed
per token like any other model, and that usage is accumulated with its cache and reasoning
breakdowns intact. In a call that delegates, most of the token cost is there.

Those tokens are priced against the *backend's* model, not against `gpt-live-1`, because that is
what spent them — so a delegated turn's cost is right even though the
[`ModelResponse`][pydantic_ai.messages.ModelResponse] it lands on carries Live's name. That response
records the backend model under `delegated_model` in its `provider_details`, and the backend response's
ID under `delegated_response_id`, so the cost can be recalculated from its `usage` later, or attributed
to the model and response that spent it. (`provider_response_id` stays unset: Live assigns no ID to the
reply it speaks.) The backend's
request is also what a `per_request_input_tokens_limit` is measured against, since it is the only
thing in a Live session that spends input tokens.

For [usage limits](observability.md#usage-and-limits), that means a `cost_limit` on
[`UsageLimits`][pydantic_ai.usage.UsageLimits] bounds the whole call, spoken seconds and backend tokens
together, as long as both are priced; token limits bound only the backend. No field caps duration
directly, so cap an unpriced call with your own timer or by closing the session.

`usage.requests`, and the `request_limit` that bounds it, count the backend's responses rather than
every [`ModelResponse`][pydantic_ai.messages.ModelResponse] the session records: on Live most of those
are spoken replies, turns whose boundaries are [inferred](#the-turn-boundary-is-inferred), while the
backend's responses are the requests that spend tokens. The profile says so with
`responses_are_requests=False`. A backend response is counted when its usage arrives, after it has
run, so the limit ends the session at the first response past it rather than before that response
starts.

## Settings

Each family has its own settings class, the realtime counterpart of
[model run settings](../agent.md#model-run-settings), extending the
[shared settings](overview.md#shared-settings). Both select the provider voice with `openai_voice`.

### GPT-Live {#gpt-live-settings}

[`OpenAILiveModelSettings`][pydantic_ai.realtime.openai_live.OpenAILiveModelSettings]:

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
| `openai_live_data_channel` | Which events a [WebRTC](#browser-webrtc) browser may send and receive over its data channel. Defaults to none, since the sideband runs the session |

Voice, audio format, and the starting instructions are fixed for the life of the session, which is
why these are session-start settings rather than things to change mid-call. (The Live API can append
to the instructions and reconfigure the delegation backend mid-session; Pydantic AI does not expose
either yet.) Live exposes no turn-detection, truncation, or token-limit controls, and the shared
settings that name them [raise rather than being ignored](#what-raises). `tool_choice` reaches the
backend when it can't loop: `'auto'`, `'none'`, and [`ToolOrOutput`][pydantic_ai.settings.ToolOrOutput]
to limit the tools, applied by trimming the tools the backend is given. `'required'` and lists of
tool names raise, because the backend applies the choice to every response of a delegation,
including the one meant to answer after the tools have run, so it would call tools until a limit
ended the session. It has no temperature or other sampling control at all, on either the spoken model or
the delegated backend.

### gpt-realtime {#gpt-realtime-settings}

[`OpenAIRealtimeModelSettings`][pydantic_ai.realtime.openai.OpenAIRealtimeModelSettings] adds noise
reduction, output speed, exact [turn detection](turns.md), and truncation:

```python
from pydantic_ai.realtime.openai import (
    OpenAIRealtimeModel,
    OpenAIRealtimeModelSettings,
)

settings = OpenAIRealtimeModelSettings(
    max_tokens=2_000,
    openai_voice='alloy',
    turn_detection={'sensitivity': 'high', 'silence_duration_ms': 400},
    openai_input_noise_reduction='near_field',
    openai_output_speed=1.1,
    openai_turn_detection={'type': 'semantic_vad', 'eagerness': 'high'},
    openai_truncation={'type': 'retention_ratio', 'retention_ratio': 0.8},
)
model = OpenAIRealtimeModel('gpt-realtime', settings=settings)
```

`openai_turn_detection` accepts [`ServerVAD`][pydantic_ai.realtime.openai.ServerVAD] or
[`SemanticVAD`][pydantic_ai.realtime.openai.SemanticVAD] and overrides shared
[`turn_detection`](turns.md#automatic-turn-detection).
`openai_truncation` also accepts `'auto'` or `'disabled'`; retention ratio preserves a stable,
cacheable prefix as the session grows. OpenAI realtime does not expose `temperature` through
Pydantic AI.

Input transcription defaults to `'auto'`; set a supported transcription model ID to pin it or
`None` to disable it. See [Input transcription](audio.md#input-transcription).

## Reasoning {#reasoning}

GPT-Live reasons on its delegated backend: set the effort with
`openai_live_delegation={'reasoning_effort': ...}`. The profile reports `supports_thinking=False`, so
the shared [`thinking`](../capabilities/thinking.md) setting does not apply.

On gpt-realtime, the shared [`thinking`][pydantic_ai.realtime.RealtimeModelSettings.thinking] setting
(see [Thinking](../capabilities/thinking.md)) applies to models whose profile reports
`supports_thinking`, including the `gpt-realtime-2` family. `True` uses the provider default and an
effort string selects a level. `False` sends `reasoning.effort: 'none'`, which turns reasoning off. The
GA `gpt-realtime` ignores the setting.

Neither family surfaces reasoning traces as [`ThinkingPart`][pydantic_ai.messages.ThinkingPart]s:
the Realtime API exposes effort as input only, and Live's reasoning happens on the backend.

## Browser WebRTC

For browser voice agents, OpenAI recommends WebRTC: the audio flows browser ↔ OpenAI directly, while
your backend attaches a control-plane **sideband** to run the agent.
[`AgentRealtime`][pydantic_ai.agent.AgentRealtime] exposes two signaling helpers, both resolving and
binding the agent's session configuration (instructions, tools, voice, VAD) server-side:

- [`answer_webrtc_offer`][pydantic_ai.agent.AgentRealtime.answer_webrtc_offer] — the **secure** path:
  relay the browser's SDP offer to OpenAI from your server, returning the SDP answer and a
  [`WebRTCSession`][pydantic_ai.realtime.WebRTCSession] to attach a sideband to with
  [`agent.realtime(model).session(provider_session=…)`][pydantic_ai.agent.AgentRealtime.session]. The browser
  never sees a token.
- [`create_client_secret`][pydantic_ai.agent.AgentRealtime.create_client_secret] — mint a short-lived
  [`RealtimeClientSecret`][pydantic_ai.realtime.RealtimeClientSecret] (ephemeral token) for a browser
  that negotiates the WebRTC call itself, when you don't relay the SDP through your backend.
  gpt-realtime only: GPT-Live has no client secrets.

A GPT-Live session is configured once, by the offer, so a sideband attaching to it can't seed
`message_history`; the browser's data channel stays closed unless `openai_live_data_channel` opens it.

See [Connecting a frontend](deployment.md#browser-webrtc-server-sideband) for the topology, the
secure offer-relay flow, and the sideband trust model, and the
[realtime WebRTC example](../examples/realtime-webrtc.md) for a runnable FastAPI and browser app.

## Feature support and limitations

Neither family supports [native tools](tools.md#native-tools); configure local fallbacks for web
capabilities. Both run [tool calls asynchronously](tools.md#concurrent-tool-execution), so the model
keeps talking while a tool runs. See [Audio, images, and transcripts](audio.md),
[Turns and interruptions](turns.md), [Tools](tools.md), and [Connection lifecycle](lifecycle.md) for
the provider-agnostic workflows.

### GPT-Live {#gpt-live-feature-support-and-limitations}

| Feature | Support | Notes |
| --- | --- | --- |
| Audio format | Limited parameter support | Mono PCM16 at 24 kHz by default, input and output; 16 kHz by setting both `audio_input_sample_rate` and `audio_output_sample_rate` to `16000` through [`profile=`](overview.md#provider-support). The API also offers 8 kHz G.711, which Pydantic AI does not expose |
| Text input | Limited parameter support | [Context, not a user turn](#text-is-context-not-a-user-turn): capped at 500 tokens, delivered only while audio flows, and possibly spoken even with `respond=False` |
| Text output | Unsupported | Live always speaks; read the answer from the transcript on the [`SpeechPart`][pydantic_ai.messages.SpeechPart] |
| Image input | Limited parameter support | [For the backend, with `respond=True`](#images-go-to-the-backend) |
| Manual turns and interruption | Unsupported | Live owns turn-taking and handles barge-in itself, but reports nothing when it does, so a reply the user cut off is recorded as complete, not interrupted. The [turn boundary is inferred](#the-turn-boundary-is-inferred) from silence |
| Input transcription | Full feature support | Always on in both directions; no [model to choose](audio.md#input-transcription) and no way to disable it |
| Input speech events | Unsupported | No speech start/end frames, so a "listening" indicator should read the profile rather than wait for events |
| Thinking | Unsupported | Set the backend's effort instead; see [Reasoning](#reasoning) |
| Usage | Limited parameter support | [Seconds, not tokens](#usage-is-measured-in-seconds); no duration-based `UsageLimits` field |
| Reconnection | Unsupported | Automatic [reconnection](lifecycle.md#reconnecting) is not implemented for Live, so the [`reconnect`][pydantic_ai.realtime.RealtimeModelSettings.reconnect] policy is ignored and a dropped connection ends the session. Open a new one, seeding it with the previous session's history |

#### What raises

Live refuses a stated requirement it cannot meet rather than accepting and ignoring it. These raise
[`UserError`][pydantic_ai.exceptions.UserError]:

- Before the session connects: `turn_detection`, `max_tokens`, `input_transcription_model`,
  `output_modality='text'` (the profile reports `supports_text_output=False`; see
  [Shared settings](overview.md#shared-settings)), and a `tool_choice` of `'required'` or a list of
  tool names.
- [`commit_audio()`][pydantic_ai.realtime.RealtimeSession.commit_audio],
  [`clear_audio()`][pydantic_ai.realtime.RealtimeSession.clear_audio],
  [`create_response()`][pydantic_ai.realtime.RealtimeSession.create_response], and
  [`interrupt()`][pydantic_ai.realtime.RealtimeSession.interrupt].
- An image sent without `respond=True`, text over the 500-token cap, and seeded history that contains
  audio or images.
- A [`ToolReturn`][pydantic_ai.messages.ToolReturn] whose `content` carries media, which Pydantic AI
  does not route to the delegated backend yet. It is refused before anything is sent rather than
  reaching the backend without the material that explains it. Text `content` is sent to the backend as
  a message after the tool's result.

### gpt-realtime {#gpt-realtime-feature-support-and-limitations}

| Feature | Support | Notes |
| --- | --- | --- |
| Audio format | Limited parameter support | Mono PCM16, 24 kHz input and output; the API also offers 8 kHz G.711, which Pydantic AI does not expose |
| Text output | Full feature support | Select with `output_modality='text'` |
| Image input | Full feature support | [Images](audio.md#images) provide context for the next turn |
| Manual turns and interruption | Full feature support | `turn_detection=False` plus [commit/create verbs](turns.md#push-to-talk); [`interrupt(played_ms=...)`](turns.md#barge-in) records the heard cutoff |
| Input transcription | Full feature support | [Dedicated model](audio.md#input-transcription); `'auto'` by default |
| Usage | Full feature support | Token, audio, and cache breakdowns |
| Reconnection | Full feature support | The connection has no resumable server handle, so Pydantic AI [replays completed local history](lifecycle.md#state-restoration) into a new session; in-flight media is lost |

## Provider-specific quirks

### GPT-Live {#gpt-live-provider-specific-quirks}

- Live streams output audio as a continuous track for the whole session (in our recordings, ten
  frames a second, silent between replies), so an arriving frame says nothing about whether the model
  is speaking. Pydantic AI drops the idle silence, so
  [`stream_audio()`][pydantic_ai.realtime.RealtimeSession.stream_audio] yields audio only when the
  model talks, as on every other provider. Silences of up to half a second *inside* a reply are
  forwarded, so a mid-sentence pause doesn't become a gap in playback; a longer one arrives as a gap.
- Live's transcript fragments carry their own spacing, except the first fragment of a new segment,
  which Pydantic AI separates from a sentence that ended the previous one.
- When Live ends the session itself, the reply in progress is recorded as interrupted and the session
  raises [`RealtimeError`][pydantic_ai.realtime.RealtimeError] with a code naming the reason:
  `live_session_expired` (the duration limit), `live_session_content` (the safety filter), or
  `live_session_connection_lost`.
- A delegated backend that fails, stops short, or reports an error is surfaced as a recoverable
  [`RealtimeSessionErrorEvent`][pydantic_ai.messages.RealtimeSessionErrorEvent] instead: the call
  goes on. When Live reports such a failure only as a session-level error that names no delegation
  (for instance `Responses handoff incomplete.` when the backend runs out of `max_output_tokens`),
  Pydantic AI treats the delegated work in flight as the work that failed, with code
  `live_delegation_failed`, so the turn still ends.

### gpt-realtime {#gpt-realtime-provider-specific-quirks}

- A response the server can't generate, for example because its safety system rejected an image in
  the conversation, arrives as a recoverable
  [`RealtimeSessionErrorEvent`][pydantic_ai.realtime.RealtimeSessionErrorEvent] carrying the
  provider's `code` (such as `input_image_safety_violation`), followed by an empty response with
  `finish_reason='error'` and the error in `provider_details['error']`. The rejected content stays in
  the server's conversation, so every later response fails the same way:
  [seed a new session](history.md#seeding-a-session) with history that leaves it out.

## Gateway

To route through the [Pydantic AI Gateway](../gateway.md), use a `gateway/`-prefixed model string.
The gateway does not route GPT-Live yet, so `'gateway/openai:gpt-live-1'` fails to connect. Connect
through `provider='openai'` or an [`OpenAIProvider`][pydantic_ai.providers.openai.OpenAIProvider] for
now. Once it does, the backend follows the same rule as a direct connection: an agent built on
`'gateway/openai:gpt-6-luna'` delegates to `gpt-6-luna`, because both go through the same gateway
route.

gpt-realtime routes through the gateway today:

```python
from pydantic_ai import Agent

agent = Agent(instructions='You are a helpful voice assistant.')
realtime = agent.realtime('gateway/openai:gpt-realtime')
```

Credentials come from
[`gateway_provider`][pydantic_ai.providers.gateway.gateway_provider]. OpenAI-compatible endpoints
that expose the realtime protocol can also be supplied through an `OpenAIProvider`. See
[Gateway trace propagation](observability.md#gateway-trace-propagation).
