# Connection lifecycle

A realtime model uses one persistent provider connection. Your backend owns that session and the
media bridge to the user; a reconnect policy can recover dropped connections and provider session
limits without changing the application event loop.

## Connecting a frontend

Keep provider keys, tools, and business logic on the server; connect user devices to your backend,
not to the provider.

**Browser → backend → provider.** Build a WebSocket endpoint on your backend that accepts the
browser's microphone audio and pumps it into
[`send_audio()`][pydantic_ai.realtime.RealtimeSession.send_audio], while relaying
[`stream_audio()`][pydantic_ai.realtime.RealtimeSession.stream_audio] output back for playback. The
[realtime camera example](../examples/realtime-camera.md) demonstrates this shape end to end.

**WebRTC media room.** Let a platform such as LiveKit handle echo cancellation, jitter, device
handling, and telephony; you build an agent worker that joins the room as a server-side participant
and bridges the room's audio track to the realtime session.

**SIP/telephony bridge.** Terminate the phone call with a telephony provider such as Twilio, then
build the service that connects its media stream (e.g. Twilio Media Streams over WebSocket) to the
backend session, transcoding between the line's codec and PCM16.

A minimal FastAPI relay for the first shape — the browser sends raw PCM16 binary frames and plays
the frames it receives:

```python
import asyncio

from fastapi import FastAPI, WebSocket

from pydantic_ai import Agent

agent = Agent(instructions='You are a helpful voice assistant.')
app = FastAPI()


@app.websocket('/voice')
async def voice_socket(websocket: WebSocket):
    await websocket.accept()
    async with agent.realtime('openai:gpt-realtime').session() as session:

        async def pump_input():
            while True:
                await session.send_audio(await websocket.receive_bytes())

        input_task = asyncio.create_task(pump_input())
        try:
            async for chunk in session.stream_audio():
                await websocket.send_bytes(chunk)
        finally:
            input_task.cancel()
```

Browser-direct provider sessions move the agent loop into the client and give up Pydantic AI's
server-side tools and history. Use that path only when the provider-native client experience is the
actual goal.

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
history into the new session. Prior transcript turns survive; in-flight audio does not.

Gemini and xAI use native in-process session resumption. xAI enables it automatically when a policy
is present. Gemini additionally requires `google_enable_session_resumption=True`; see the
[Gemini resumption settings](gemini.md#session-resumption). Their handles live only in memory and
cannot be persisted for another process.

[`RealtimeSessionReconnectEvent.state_restored`][pydantic_ai.realtime.RealtimeSessionReconnectEvent.state_restored]
reports whether conversation state was recovered, by either mechanism.

A reply the drop cut off follows the same flag. With state restored, the recorded response simply
stays open: output on the new connection continues it, and the turn completes with the response
terminal as usual — except on Gemini, which closes the cut reply as an interrupted response (keeping
any partial transcript in history) before the
[`RealtimeSessionReconnectEvent`][pydantic_ai.realtime.RealtimeSessionReconnectEvent] and stays
quiet until the next input. With state lost, treat the session as a fresh context: before emitting
the event, the session settles everything the provider lost — the partial reply is recorded as an
interrupted response, running tool calls get cancelled returns, and the turn ends so queued messages
waiting for the boundary still flush.

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

Recoverable failures arrive as events: [`RealtimeSessionErrorEvent`][pydantic_ai.realtime.RealtimeSessionErrorEvent]
for provider operations and
[`RealtimeInputTranscriptionErrorEvent`][pydantic_ai.realtime.RealtimeInputTranscriptionErrorEvent] for one failed
user transcription. The session remains usable after either event.

Failures surface from the responsible call where possible; a failed `send_audio()` raises there.
Receive-loop and tool failures propagate from session iteration.

## Troubleshooting

### No audio, or no useful speech

Send mono PCM16 at `session.profile['audio_input_sample_rate']` and play it at
`session.profile['audio_output_sample_rate']`. Do not assume the rates match. See the
[audio wire contract](audio.md#audio-wire-contract).

### The model never responds

In push-to-talk mode, call `commit_audio()` and then `create_response()` after sending audio. See
[push-to-talk](turns.md#push-to-talk).

### The model interrupts itself

The microphone is probably hearing speaker output. Add echo cancellation in the device/WebRTC
layer and stop local playback on real [barge-in](turns.md#barge-in).

### Tools seem to stall

The local tool runs concurrently, but the provider may pause speech while awaiting the result. Show
tool lifecycle events and review [concurrent tool execution](tools.md#concurrent-tool-execution).

### A reconnect lost context

Inspect `RealtimeSessionReconnectEvent.state_restored`. If false, begin a fresh conversation; if true but a
current utterance vanished, that in-flight media was outside the restored completed-turn history.
See [state restoration](#state-restoration).

### Gemini reaches its session limit

Combine [`ReconnectPolicy`][pydantic_ai.realtime.ReconnectPolicy] with
`google_enable_session_resumption=True`. Recovery uses the latest in-memory server handle after the
drop. See [Gemini session resumption](gemini.md#session-resumption).
