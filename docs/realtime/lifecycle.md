# Connection lifecycle

A realtime model uses one persistent provider connection. Your backend owns that session and the
media bridge to the user; a reconnect policy can recover dropped connections and provider session
limits without changing the application event loop.

## Connecting a frontend

Keep provider keys, tools, and business logic on the server. Common transport shapes are:

- **Browser ↔ provider over WebRTC, with a server sideband (recommended for browsers):** the browser
  exchanges audio with the provider directly, while your backend attaches a control-plane connection
  to the *same* call and runs the full agent loop. See [Browser / WebRTC](#browser-webrtc) below and
  the [realtime WebRTC example](../examples/realtime-webrtc.md).
- **Browser → backend → provider:** relay audio over your own WebSocket. The
  [realtime camera example](../examples/realtime-camera.md) demonstrates this shape.
- **WebRTC media room:** let a platform such as LiveKit handle echo cancellation, jitter, devices,
  and telephony while a server-side participant runs the realtime session.
- **SIP/telephony bridge:** terminate the call with a telephony provider and bridge its audio to the
  backend session.

```text
device ↔ media bridge ↔ RealtimeSession ↔ provider
                         ├── typed tools
                         └── message history
                         (your backend)
```

Browser-direct provider sessions *without* a sideband move the agent loop into the client and give up
Pydantic AI's server-side tools and history. Use that path only when the provider-native client
experience is the actual goal. Give browsers only short-lived tokens scoped to their connection, or —
better — negotiate on their behalf so a token never reaches the client at all; never ship backend
credentials to a client.

### Browser / WebRTC

For browser voice agents, OpenAI and Azure OpenAI both recommend WebRTC: the audio flows **browser ↔
provider directly**, so latency is low and the browser's media stack (echo cancellation, jitter
buffering, device handling) does the hard part. Pydantic AI stays the control plane by attaching a
**sideband** — a normal realtime control connection to the *same* call, identified by a `call_id`, over
which it runs instructions, tools, and history while the browser owns the audio.

```text
   browser ──mic/speaker audio (WebRTC media)──▶  provider
          ◀─────────────────────────────────────
      │  SDP offer                                  ▲ control connection (call_id)
      ▼                                             │
   your backend ──answer_webrtc_offer()──▶ provider ──session(provider_session=…)──┘
                (relays the SDP, gets a call_id)     (runs tools, builds history)
```

The secure flow keeps the API key server-side — the browser never holds a token:

1. The browser creates an `RTCPeerConnection`, captures the microphone, and sends its SDP **offer** to
   your backend.
2. Your backend relays it with
   [`AgentRealtime.answer_webrtc_offer`][pydantic_ai.agent.AgentRealtime.answer_webrtc_offer], which returns the
   provider's SDP **answer** and a [`WebRTCSession`][pydantic_ai.realtime.WebRTCSession] carrying the `call_id`.
3. Your backend returns the answer to the browser (media now flows browser ↔ provider) and attaches the
   sideband with
   [`agent.realtime(model).session(provider_session=call)`][pydantic_ai.agent.AgentRealtime.session].

```python
import asyncio

from pydantic_ai import Agent
from pydantic_ai.realtime.openai import OpenAIRealtimeModel

agent = Agent(instructions='You are a helpful voice assistant.')


@agent.tool_plain
def get_weather(city: str) -> str:
    return f'Sunny in {city}'


realtime = agent.realtime(OpenAIRealtimeModel('gpt-realtime'))


# In your `POST /offer` handler, `sdp_offer` is the browser's offer (the request body):
async def handle_offer(sdp_offer: str) -> str:
    answer = await realtime.answer_webrtc_offer(sdp_offer)

    async def run_sideband() -> None:
        async with realtime.session(provider_session=answer.session) as session:
            async for event in session:  # the agent runs tools and builds history here
                print(event)

    asyncio.create_task(run_sideband())  # attach the sideband, then return the answer to the browser
    return answer.sdp
```

Because a sideband session doesn't own the audio transport, [`send_audio()`][pydantic_ai.realtime.RealtimeSession.send_audio],
[`commit_audio()`][pydantic_ai.realtime.RealtimeSession.commit_audio],
[`clear_audio()`][pydantic_ai.realtime.RealtimeSession.clear_audio], and
[`stream_audio()`][pydantic_ai.realtime.RealtimeSession.stream_audio] raise. Send microphone audio and
consume the remote audio track in the browser instead. `audio_retention` must stay `'transcript_only'`
(the browser has the audio; the session still records transcripts). The event loop,
[`stream_transcripts()`][pydantic_ai.realtime.RealtimeSession.stream_transcripts], tools, and message
history remain available, so the backend can render captions, run tools, and hand the call off to a
text agent.

!!! note "Capturing a sideband's user transcripts needs input transcription"
    Because the sideband never receives the user's audio, the only way to capture the *words* a user
    speaks is a [transcription model](audio.md#input-transcription) — the
    `audio_retention='input_audio'` fallback can't apply (there's no audio to retain, so `audio_retention`
    stays `'transcript_only'`). Without transcription the user's turns are still represented in history,
    but as content-less [`SpeechPart`][pydantic_ai.messages.SpeechPart]s. To capture what users say,
    keep transcription enabled — on Azure OpenAI the transcription model must be **deployed** on your
    resource, or it fails with `DeploymentNotFound` (see the [Azure page](azure.md#browser-webrtc-and-microsoft-entra-id));
    the session records content-less turns and keeps running rather than failing.

WebRTC is available for **OpenAI and Azure OpenAI** (see the [OpenAI](openai.md#browser-webrtc) and
[Azure](azure.md#browser-webrtc-and-microsoft-entra-id) provider pages, including Azure's Microsoft
Entra ID support and the alternative ephemeral-token flow via
[`AgentRealtime.create_client_secret`][pydantic_ai.agent.AgentRealtime.create_client_secret]). The agent-level
methods resolve and bake in the agent's instructions, tools, capabilities, and model settings; the
corresponding methods on [`RealtimeModel`][pydantic_ai.realtime.RealtimeModel] are the lower-level
signaling mechanism they build on. Gemini Live and xAI Grok Voice are WebSocket-only and don't offer a
WebRTC sideband; use a relay or media room for those.

The runnable [realtime WebRTC example](../examples/realtime-webrtc.md) shows the whole flow end to end.

#### Trust model: what the browser can see and do

On a sideband the browser is a **peer on the same provider session**, not a client of your backend. It
holds its own data channel to the provider, and that channel carries client events in both directions
by design: the browser can send
[`session.update`, `conversation.item.create` and `response.create`](https://platform.openai.com/docs/guides/realtime-conversations)
just as your backend can, so it can replace the instructions and `tool_choice` your agent resolved, or
ask for a response of its own. OpenAI's own realtime console installs a function tool this way.
Session configuration attached to an ephemeral secret is documented as overridable by the client
connection too, so [`create_client_secret`][pydantic_ai.agent.AgentRealtime.create_client_secret]
behaves the same way.

The session's resolved instructions and tool definitions are visible to the browser as well, in the
provider's `session.created` and `session.updated` events. Azure OpenAI relays offers with
`webrtcfilter=on`, which withholds those from the browser's data channel (see the
[Azure page](azure.md#browser-webrtc-and-microsoft-entra-id)); it filters only what is sent *to* the
browser, and OpenAI has no equivalent.

What follows from that is specific to a sideband: **tool calls execute on your backend, with your
[dependencies](../dependencies.md)** — so any tool registered on a sideband session is reachable by
that call's end user. Two habits follow:

- **Authorize inside the tool, against `deps`.** The identity you attached when opening the session is
  the thing to check, because it is the only part of the exchange the browser cannot influence.
  Instructions are visible to the browser *and* replaceable by it, so they shape behavior, not
  permissions.
- **Keep sensitive workflows off the call.** Hand off to a standard agent run
  ([History and handoff](history.md#handing-off-to-a-text-agent)) for work that shouldn't be reachable
  from the session at all.

[Approval-gated tools](tools.md#deferred-and-approval-required-tools) are not a substitute for either.
Approval guards against the *model* acting without sign-off; a policy handler resolving calls inline is
not an authorization boundary against the caller, the same distinction the
[UI adapter trust model](../ui/overview.md#trust-model-for-client-submitted-messages) draws.

!!! note "This is the boundary WebRTC already draws, not a vulnerability"
    A browser holding an ephemeral secret can create and configure provider sessions on its own, so a
    sideband does not widen what it can reach — it is the same trust boundary as the browser talking to
    the provider directly, and your provider API key stays on the server either way (with
    [`answer_webrtc_offer`][pydantic_ai.agent.AgentRealtime.answer_webrtc_offer] the browser never
    receives a token at all). What the sideband adds is your tools, which is why they are the part to
    authorize. This mirrors the
    [trust boundary for client-supplied history](../message-history.md#trust-boundary-for-client-supplied-history):
    a report that a browser can rewrite its own session's instructions or ask for a response describes
    this boundary working as designed.

#### Knowing when the model is speaking

An ordinary session sees the model's audio go by, so it knows when the model is talking. A sideband
doesn't — the media never reaches it — and *generation* is not a usable stand-in: the provider
produces audio far faster than it plays, so a response can be complete while the listener still has
many seconds of speech to hear. Measured on a twenty-second answer, the model stayed audible for
another **24 to 36 seconds after**
[`RealtimeTurnCompleteEvent`][pydantic_ai.realtime.RealtimeTurnCompleteEvent], on OpenAI and Azure alike.

So the provider reports the boundary directly, as
[`RealtimeOutputSpeechStartEvent`][pydantic_ai.realtime.RealtimeOutputSpeechStartEvent] and
[`RealtimeOutputSpeechEndEvent`][pydantic_ai.realtime.RealtimeOutputSpeechEndEvent]. Drive a
"speaking" indicator from these rather than from
[`RealtimeTurnCompleteEvent`][pydantic_ai.realtime.RealtimeTurnCompleteEvent], which fires when the
model stops *generating*. They also bracket a `speak {model}` span in
[Logfire](observability.md#logfire-instrumentation), so a trace shows how long the model was actually audible.

Both events arrive on Azure too: its `webrtcfilter=on` restricts what the *browser's* data channel
receives (see [Azure](azure.md#browser-webrtc-and-microsoft-entra-id)), and leaves the session's own
control connection — where these come from — untouched.

[`interrupt()`][pydantic_ai.realtime.RealtimeSession.interrupt] handles the other half of the same
problem: it drops the audio the provider has already produced but not yet played, so a barge-in stops
the voice instead of only stopping generation. Measured across OpenAI and Azure, the model went silent
**within 60 ms** of the call, and
[`RealtimeOutputSpeechEndEvent`][pydantic_ai.realtime.RealtimeOutputSpeechEndEvent] followed
immediately — so an indicator driven off these events clears itself on barge-in.

## Reconnecting

Pass [`ReconnectPolicy`][pydantic_ai.realtime.ReconnectPolicy] to redial with exponential backoff,
reapply configuration, and emit
[`RealtimeSessionReconnectEvent`][pydantic_ai.realtime.RealtimeSessionReconnectEvent]:

```python
from pydantic_ai.realtime import ReconnectPolicy
from pydantic_ai.realtime.openai import OpenAIRealtimeModel

model = OpenAIRealtimeModel('gpt-realtime', reconnect=ReconnectPolicy(max_attempts=5))
```

`max_attempts` bounds retries for one drop. `max_reconnects` bounds recoveries across the entire
session, preventing an endpoint that repeatedly accepts and closes connections from redialing
forever.

Without a policy, an unexpected provider close raises
[`RealtimeError`][pydantic_ai.realtime.RealtimeError] from the session iterator.

### State restoration

OpenAI and Azure OpenAI have no cross-connection server state, so Pydantic AI replays local message
history into the new session. Prior transcript turns survive; in-flight audio does not. A reconnect
therefore always begins a fresh turn.

Gemini and xAI use native in-process session resumption. xAI enables it automatically when a policy
is present. Gemini additionally requires `google_enable_session_resumption=True`; see the
[Gemini resumption settings](gemini.md#session-resumption). Their handles live only in memory and
cannot be persisted for another process.

[`RealtimeSessionReconnectEvent.state_restored`][pydantic_ai.realtime.RealtimeSessionReconnectEvent.state_restored]
reports whether conversation state was recovered. Treat `False` as a fresh context.

Resumption restores the conversation, not a generation in flight: a reply the drop cut off is never
continued on the new connection. The session closes it as an interrupted response — keeping any
partial transcript in history — and ends its turn before the
[`RealtimeSessionReconnectEvent`][pydantic_ai.realtime.RealtimeSessionReconnectEvent], so queued messages waiting for
the turn boundary still flush; the model then stays quiet until the next input.

## Provider session limits

Providers cap individual connection duration. A reconnect policy is also how an application survives
those limits. Exact limits and provider behavior can change, so provider pages are canonical:

- [OpenAI session behavior](openai.md#feature-support-and-limitations)
- [Azure OpenAI session behavior](azure.md#feature-support-and-limitations)
- [Gemini session resumption](gemini.md#session-resumption)
- [xAI native session resumption](xai.md#session-resumption)

Gemini sends `GoAway` shortly before its cap but Pydantic AI currently reconnects only after the
connection drops, so a long call can briefly drop mid-turn.

## Errors

Realtime sessions use the standard Pydantic AI exception hierarchy:

| Exception | Raised when |
| --- | --- |
| [`UserError`][pydantic_ai.exceptions.UserError] | The application requests an operation the model or transport doesn't support (`answer_webrtc_offer()` on a WebSocket-only model, `stream_audio()` on a [WebRTC sideband](#browser-webrtc)), passes incompatible settings, lacks credentials, or misuses the session. |
| [`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError] | The provider rejects the WebSocket upgrade with an HTTP status. |
| [`RealtimeError`][pydantic_ai.realtime.RealtimeError] | The connection fails, times out, closes unexpectedly, returns an invalid frame, or exhausts reconnect attempts. |
| [`UsageLimitExceeded`][pydantic_ai.exceptions.UsageLimitExceeded] | A configured [usage limit](observability.md#usage-and-limits) is exceeded. |

[`RealtimeError`][pydantic_ai.realtime.RealtimeError] subclasses
[`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError], so `except ModelAPIError` covers HTTP and
non-HTTP provider failures together.

Recoverable failures arrive as events: [`RealtimeSessionErrorEvent`][pydantic_ai.realtime.RealtimeSessionErrorEvent]
for provider operations and
[`RealtimeInputTranscriptionErrorEvent`][pydantic_ai.realtime.RealtimeInputTranscriptionErrorEvent] for one failed
user transcription. The session remains usable after either event.

Failures surface from the responsible call where possible; a failed `send_audio()` raises there.
Receive-loop and tool failures propagate from session iteration.

One close is *not* an error: on a [WebRTC sideband](#browser-webrtc) the browser owns the call, so its
hanging up cleanly is the normal end of one. The event stream simply ends, with no
[`RealtimeSessionErrorEvent`][pydantic_ai.realtime.RealtimeSessionErrorEvent] and no reconnect
attempt — a dropped socket (an abnormal close) is still treated as a drop.

## Troubleshooting

### No audio

Send mono PCM16 at `session.profile['audio_input_sample_rate']` and play it at
`session.profile['audio_output_sample_rate']`. Do not assume the rates match.

### The model never responds

In push-to-talk mode, call `commit_audio()` and then `create_response()` after sending audio.

### The model interrupts itself

The microphone is probably hearing speaker output. Add echo cancellation in the device/WebRTC
layer and stop local playback on real [barge-in](turns.md#barge-in).

### Tools seem to stall

The local tool runs concurrently, but the provider may pause speech while awaiting the result. Show
tool lifecycle events and review [concurrent tool execution](tools.md#concurrent-tool-execution).

### A reconnect lost context

Inspect `RealtimeSessionReconnectEvent.state_restored`. If false, begin a fresh conversation; if true but a
current utterance vanished, that in-flight media was outside the restored completed-turn history.

### Gemini reaches its session limit

Combine [`ReconnectPolicy`][pydantic_ai.realtime.ReconnectPolicy] with
`google_enable_session_resumption=True`. Recovery uses the latest in-memory server handle after the
drop.
