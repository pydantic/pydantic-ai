# Connection lifecycle

A realtime model uses one persistent provider connection. Your backend owns that session and the
media bridge to the user; a reconnect policy can recover dropped connections and provider session
limits without changing the application event loop.

## Connecting a frontend

Keep provider keys, tools, and business logic on the server. Common transport shapes are:

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

Browser-direct provider sessions move the agent loop into the client and give up Pydantic AI's
server-side tools and history. Use that path only when the provider-native client experience is the
actual goal.

## Reconnecting

Pass [`ReconnectPolicy`][pydantic_ai.realtime.ReconnectPolicy] to redial with exponential backoff,
reapply configuration, and emit
[`SessionReconnectEvent`][pydantic_ai.realtime.SessionReconnectEvent]:

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

[`SessionReconnectEvent.state_restored`][pydantic_ai.realtime.SessionReconnectEvent.state_restored]
reports whether conversation state was recovered. Treat `False` as a fresh context.

Resumption restores the conversation, not a generation in flight: a reply the drop cut off is never
continued on the new connection. The session closes it as an interrupted response — keeping any
partial transcript in history — and ends its turn before the
[`SessionReconnectEvent`][pydantic_ai.realtime.SessionReconnectEvent], so queued messages waiting for
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
| [`UserError`][pydantic_ai.exceptions.UserError] | The application requests an unsupported operation, passes incompatible settings, lacks credentials, or misuses the session. |
| [`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError] | The provider rejects the WebSocket upgrade with an HTTP status. |
| [`RealtimeError`][pydantic_ai.realtime.RealtimeError] | The connection fails, times out, closes unexpectedly, returns an invalid frame, or exhausts reconnect attempts. |
| [`UsageLimitExceeded`][pydantic_ai.exceptions.UsageLimitExceeded] | A configured [usage limit](observability.md#usage-and-limits) is exceeded. |

[`RealtimeError`][pydantic_ai.realtime.RealtimeError] subclasses
[`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError], so `except ModelAPIError` covers HTTP and
non-HTTP provider failures together.

Recoverable failures arrive as events: [`SessionErrorEvent`][pydantic_ai.realtime.SessionErrorEvent]
for provider operations and
[`InputTranscriptionErrorEvent`][pydantic_ai.realtime.InputTranscriptionErrorEvent] for one failed
user transcription. The session remains usable after either event.

Failures surface from the responsible call where possible; a failed `send_audio()` raises there.
Receive-loop and tool failures propagate from session iteration.

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

Inspect `SessionReconnectEvent.state_restored`. If false, begin a fresh conversation; if true but a
current utterance vanished, that in-flight media was outside the restored completed-turn history.

### Gemini reaches its session limit

Combine [`ReconnectPolicy`][pydantic_ai.realtime.ReconnectPolicy] with
`google_enable_session_resumption=True`. Recovery uses the latest in-memory server handle after the
drop.
