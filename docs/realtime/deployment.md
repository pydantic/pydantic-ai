---
description: "Deploy a Pydantic AI realtime voice agent: connect browsers and phone calls via WebRTC, a WebSocket relay or a Twilio bridge, and host the long-lived sessions on your server."
---

# Deployment

Keep provider keys, tools, and business logic on the server; connect user devices to your backend,
not straight to the provider. How audio travels between the device and the model depends on the
client and the provider:

- **[Browser WebRTC + server sideband](#browser-webrtc-server-sideband)** — the recommended browser
  path on OpenAI and Azure OpenAI: the browser exchanges media directly with the provider (lowest
  latency) while your backend runs the agent over a control-plane sideband.
- **[Browser → backend WebSocket relay](#browser-backend-websocket-relay)** — works with every
  provider, including Gemini Live, xAI, and ElevenLabs Agents: the browser streams audio to your backend, which owns the
  media bridge.
- **[Twilio Media Streams](#twilio-media-streams)** — for phone calls: Twilio streams the call's audio to
  your backend, which bridges it to the session.

[Hosting the backend](#hosting) covers what the server running these sessions needs: long-lived
connections, routing, timeouts, and draining on deploy.

In every setup the session — with its [tools](tools.md), [history](history.md), and
[usage limits](observability.md#usage-and-limits) — runs on your backend. Wiring a browser straight
to the provider with the provider's own SDK instead moves the agent loop into the client and gives
up all of that; prefer one of the setups above.

## Browser WebRTC + server sideband {#browser-webrtc-server-sideband}

For browser voice agents on OpenAI and Azure OpenAI, the browser carries microphone and speaker
audio directly over WebRTC while the backend attaches a control-plane **sideband** to the same call.
Media never touches your backend, so latency stays low, while tools, history, dependencies, and
provider credentials stay on the server. Gemini Live, xAI, and ElevenLabs Agents do not offer this sideband transport —
use the [WebSocket relay](#browser-backend-websocket-relay) there.

```text
browser ── WebRTC media ── provider
   │                         ▲
   └─ SDP offer → backend ───┘ sideband identified by call_id
```

Relay the browser's offer with
[`AgentRealtime.answer_webrtc_offer`][pydantic_ai.agent.AgentRealtime.answer_webrtc_offer], return
the SDP answer to the browser, then attach the returned call handle:

```python
import asyncio

from pydantic_ai import Agent

agent = Agent(instructions='You are a concise voice assistant.')
realtime = agent.realtime('{{ openai:realtime }}')
sideband_tasks: set[asyncio.Task[None]] = set()


async def handle_offer(sdp_offer: str) -> str:
    answer = await realtime.answer_webrtc_offer(sdp_offer)

    async def run_sideband() -> None:
        async with realtime.session(provider_session=answer.session) as session:
            async for event in session:
                print(event)

    task = asyncio.create_task(run_sideband())
    sideband_tasks.add(task)  # (1)!
    task.add_done_callback(sideband_tasks.discard)
    return answer.sdp
```

1. asyncio keeps only a weak reference to a task, so hold one yourself until the call ends; the done
   callback also surfaces an error the sideband session raised instead of leaving it as a "never
   retrieved" warning.

The secure offer-relay flow never gives the browser a token. As an alternative,
[`AgentRealtime.create_client_secret`][pydantic_ai.agent.AgentRealtime.create_client_secret] mints a
short-lived credential for client-led negotiation: the browser posts its offer to the secret's
[`webrtc_url`][pydantic_ai.realtime.RealtimeClientSecret.webrtc_url] with the secret as its
bearer token. Either way the browser is a peer on the provider
session and can send provider-native control events, so authorize every server-side tool against
trusted [`deps`](../dependencies.md), not session instructions supplied to the model.

To continue an earlier conversation, create the `AgentRealtime` per call inside the offer handler,
`realtime = agent.realtime(model, message_history=...)`, and answer the offer and open the sideband from
that same object. GPT-Live takes the history when the offer starts the call, and a sideband opened with
other history refuses to attach; the other providers seed it when the sideband attaches.

!!! warning "The browser can read seeded history"
    Seeding a WebRTC call with [`message_history`](history.md) sends those prior turns into the
    **shared** provider conversation that the browser is a peer on, so a call participant can read
    them — including confidential tool results — over the data channel (Azure's `webrtcfilter=on`
    still forwards conversation-item events). Only seed a call with history that is safe for the
    browser to see; keep confidential context in [`deps`](../dependencies.md) and tool logic instead.

A sideband does not own the audio transport: its `send_audio()`, `commit_audio()`, `clear_audio()`,
and `stream_audio()` methods raise, and `audio_retention` must remain `'transcript_only'`. Enable
[input transcription](audio.md#input-transcription) when user speech must appear in history. On
gpt-realtime and Azure OpenAI, use
[`RealtimeOutputSpeechStartEvent`][pydantic_ai.realtime.RealtimeOutputSpeechStartEvent] /
[`RealtimeOutputSpeechEndEvent`][pydantic_ai.realtime.RealtimeOutputSpeechEndEvent] for speaking
indicators, and [`interrupt()`][pydantic_ai.realtime.RealtimeSession.interrupt] clears the provider's
outbound WebRTC audio buffer so [barge-in](turns.md#barge-in) stops playback. GPT-Live emits neither
event and doesn't support `interrupt()` (see
[its feature support](openai.md#gpt-live-feature-support-and-limitations)). A dropped sideband
follows the same [reconnect](lifecycle.md#reconnecting) rules, with one wrinkle covered there: a
clean close is treated as the browser hanging up.

Closing a sideband session, or leaving its `async with` block, only detaches your server: the
browser's call stays up, and billed, until the browser hangs up. To end the call from the server,
call [`hang_up()`][pydantic_ai.realtime.RealtimeSession.hang_up] instead (it works from a
[tool](tools.md#ending-the-session-from-a-tool) too), or
[`hang_up(answer.session)`][pydantic_ai.agent.AgentRealtime.hang_up] on `agent.realtime(model)` when
no sideband is attached. OpenAI supports this for both gpt-realtime and GPT-Live, and so does Azure OpenAI.

The [realtime WebRTC example](../examples/realtime-webrtc.md) demonstrates the full FastAPI and
browser flow. Provider-specific setup (Azure's Microsoft Entra ID and `webrtcfilter`) lives on the
[Azure](azure.md#browser-webrtc-and-microsoft-entra-id) page.

## Browser → backend WebSocket relay {#browser-backend-websocket-relay}

When the browser can't use WebRTC — or the provider is Gemini Live, xAI, or ElevenLabs Agents — build a WebSocket
endpoint on your backend that accepts the browser's microphone audio and pumps it into
[`send_audio()`][pydantic_ai.realtime.RealtimeSession.send_audio], while relaying
[`stream_audio()`][pydantic_ai.realtime.RealtimeSession.stream_audio] output back for playback. Here
the backend owns the media bridge. The [realtime camera example](../examples/realtime-camera.md)
demonstrates this setup end to end; a minimal FastAPI relay — the browser sends raw PCM16 binary
frames and plays the frames it receives — is:

```python
import asyncio

from fastapi import FastAPI, WebSocket

from pydantic_ai import Agent

agent = Agent(instructions='You are a helpful voice assistant.')
app = FastAPI()


@app.websocket('/voice')
async def voice_socket(websocket: WebSocket):
    await websocket.accept()

    async def microphone():
        while True:
            yield await websocket.receive_bytes()

    async with agent.realtime('{{ openai:realtime }}').session() as session:

        async def playback():
            async for chunk in session.stream_audio():
                await websocket.send_bytes(chunk)

        playback_task = asyncio.create_task(playback())
        try:
            await session.send_audio(microphone())
        finally:
            playback_task.cancel()
```

[`send_audio()`][pydantic_ai.realtime.RealtimeSession.send_audio] consumes the async iterator for the
whole call, so when the browser disconnects, `receive_bytes()` raises, the `finally` stops playback,
and leaving the `async with` block hangs up the provider session. Driving the input from a bare
`asyncio.create_task` instead would swallow that error and leave the billed session open with nobody
listening.

`handle_barge_in=True` is a no-op for this relay: `played_audio_bytes` counts a chunk as played when
the relay forwards it, before the browser has actually played it, so the session sees no unplayed
audio to flush. The relay must obtain the browser's real playback position, call
[`interrupt(played_bytes=...)`][pydantic_ai.realtime.RealtimeSession.interrupt] with that count, and
tell the browser to stop playback and clear its own buffer: the session can only drop what it has
not forwarded yet. Passing `played_ms=` records the provider-side cutoff but never flushes audio
queued by the session or buffered in the browser.

## Twilio Media Streams {#twilio-media-streams}

For phone calls, Twilio's [Media Streams](https://www.twilio.com/docs/voice/media-streams) carry a call's
audio over a WebSocket to your backend, which bridges it to the session, as the
[relay](#browser-backend-websocket-relay) does for a browser. Your server answers Twilio's incoming-call webhook
with [TwiML](https://www.twilio.com/docs/voice/twiml/stream) that connects the call to a `<Stream>`, and Twilio
then opens the stream's WebSocket and exchanges [JSON messages](https://www.twilio.com/docs/voice/media-streams/websocket-messages)
on it. The [Twilio example](../examples/realtime-twilio.md) runs the whole call, webhook included. The stream
end of it is below. Its mu-law helpers come from the example's `audio` module, which you can copy or import from
the `pydantic-ai-examples` package; they aren't part of Pydantic AI.

```python
import base64
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import anyio
from fastapi import FastAPI, WebSocket
from pydantic_ai_examples.realtime_twilio.audio import mulaw_to_pcm16, pcm16_to_mulaw

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import RealtimeInputSpeechStartEvent

agent = Agent(deps_type=str, instructions='You answer the phone for a restaurant. Keep replies short.')
app = FastAPI()
# The calls your signed `/twiml` webhook answered, as in the example: only their streams get a session.
answered_calls: set[str] = set()


@agent.instructions
def caller_number(ctx: RunContext[str]) -> str:
    return f'The caller is calling from {ctx.deps}.'


@dataclass
class PlaybackMarks:
    """How much of the session's audio Twilio has played, from the marks it echoes."""

    played_bytes: int = 0
    pending: dict[str, int] = field(default_factory=dict[str, int])
    sent: int = 0

    def name_next(self, pcm_bytes: int) -> str:
        self.sent += 1
        self.pending[str(self.sent)] = pcm_bytes
        return str(self.sent)


async def twilio_messages(websocket: WebSocket) -> AsyncIterator[dict[str, Any]]:
    async for text in websocket.iter_text():
        yield json.loads(text)


@app.websocket('/media-stream')
async def media_stream(websocket: WebSocket):
    await websocket.accept()
    messages = twilio_messages(websocket)
    await anext(messages)  # `connected`
    start = await anext(messages)
    if start['start']['callSid'] not in answered_calls:
        await websocket.close(code=1008)
        return
    answered_calls.discard(start['start']['callSid'])
    stream_sid = start['streamSid']
    marks = PlaybackMarks()

    async with agent.realtime(
        '{{ openai:realtime }}', deps=start['start']['customParameters']['caller']
    ).session() as session:

        async def caller_audio() -> AsyncIterator[bytes]:
            async for message in messages:
                if message['event'] == 'media':
                    mulaw = base64.b64decode(message['media']['payload'])
                    yield mulaw_to_pcm16(mulaw, rate=session.audio_input_sample_rate)
                elif message['event'] == 'mark':
                    marks.played_bytes += marks.pending.pop(message['mark']['name'], 0)
                elif message['event'] == 'dtmf':
                    await session.send(f'The caller pressed {message["dtmf"]["digit"]}.')
                elif message['event'] == 'stop':
                    return

        async def playback():
            async for chunk in session.stream_audio():
                mulaw = pcm16_to_mulaw(chunk, rate=session.audio_output_sample_rate)
                payload = base64.b64encode(mulaw).decode()
                mark = {'name': marks.name_next(len(chunk))}  # before a `clear` can drop the chunk
                await websocket.send_json({'event': 'media', 'streamSid': stream_sid, 'media': {'payload': payload}})
                await websocket.send_json({'event': 'mark', 'streamSid': stream_sid, 'mark': mark})

        async def barge_in():
            async for event in session:
                if isinstance(event, RealtimeInputSpeechStartEvent):
                    played = marks.played_bytes
                    # Unheard, but still behind the next interruption's position.
                    marks.played_bytes += sum(marks.pending.values())
                    marks.pending.clear()
                    await websocket.send_json({'event': 'clear', 'streamSid': stream_sid})
                    await session.interrupt(played_bytes=played)

        async with anyio.create_task_group() as tg:
            tg.start_soon(playback)
            tg.start_soon(barge_in)
            await session.send_audio(caller_audio())
            tg.cancel_scope.cancel()
```

What a bridge has to get right:

- **The line's audio.** Twilio sends and plays 8 kHz G.711 mu-law, base64-encoded. A session takes mono PCM16 at
  [`audio_input_sample_rate`][pydantic_ai.realtime.RealtimeSession.audio_input_sample_rate] and speaks at
  [`audio_output_sample_rate`][pydantic_ai.realtime.RealtimeSession.audio_output_sample_rate], and Pydantic AI
  doesn't take G.711 yet ([#9038](https://github.com/pydantic/pydantic-ai/issues/9038)), so the bridge converts
  and resamples both ways. The example's `audio` module does it with the standard library: both session rates are
  whole multiples of 8 kHz.
- **What the caller heard.** Twilio buffers the audio it's sent and plays it in order, so a bridge that forwards
  each [`stream_audio()`][pydantic_ai.realtime.RealtimeSession.stream_audio] chunk as it arrives has, as far as the
  session can tell, played all of it, and `handle_barge_in=True` does nothing, as for the
  [relay](#browser-backend-websocket-relay). Send a `mark` after each chunk instead: Twilio echoes it once the
  audio ahead of it has played. When the caller speaks, the bytes of the chunks whose marks came back are what to
  pass to [`interrupt(played_bytes=...)`][pydantic_ai.realtime.RealtimeSession.interrupt], which truncates the
  reply there so history records what the caller heard.
- **`clear` echoes every pending mark.** Interrupting the session doesn't stop audio Twilio already has: send
  `clear` too. Twilio then drops what it hasn't played and echoes every mark still pending, though nobody heard
  that audio. Read the played count and forget the pending marks before sending `clear`, so those echoes count
  for nothing, and name each chunk's mark before sending the chunk, so a `clear` sent in between drops the mark
  too. Then add the bytes of those chunks to the count: the session handed them over and can't learn that the
  caller skipped them, so every later `interrupt()` would otherwise land that many bytes early.
- **Models that interrupt themselves.** Gemini and ElevenLabs Agents cut their own reply short when the caller
  talks and report a [`RealtimeResponseInterruptedEvent`][pydantic_ai.realtime.RealtimeResponseInterruptedEvent]
  rather than a speech start: send `clear` on it too, with no `interrupt()` call, as the example does.
  [GPT-Live](openai.md#gpt-live-feature-support-and-limitations) reports neither and has no `interrupt()`: it
  stops talking over the caller by itself, and sends its audio at the pace it speaks it (ten 100 ms frames a
  second in our recordings), so little of it waits in Twilio's buffer to play on.
- **Per-call context.** The stream URL takes no query string. Put the call's context in `<Parameter>` elements of
  the `<Stream>`, read them from the `start` message's `customParameters` before opening the session, and pass
  them as the session's [`deps`](../dependencies.md).
- **Key presses.** Twilio reports the caller's keypad as `dtmf` messages. A session has no keypad input, so send
  the digit as text.
- **Ending the call.** Feed [`send_audio()`][pydantic_ai.realtime.RealtimeSession.send_audio] from the stream's
  messages inside the `async with` block: Twilio's `stop` ends the iterator, as does the socket closing, and either
  leaves the block, which hangs up the provider session. Twilio carries on with the TwiML after `<Connect>` once
  your server closes the socket, so with nothing after it, closing the socket ends the call.
- **Signatures.** Check `X-Twilio-Signature` on Twilio's requests with the `RequestValidator` of
  [Twilio's SDK](https://www.twilio.com/docs/usage/webhooks/webhooks-security), as Twilio asks, rather than with
  code of your own. Twilio signs the public URL it called, which a tunnel or proxy hides from your server, so
  validate against that. Then accept a stream only for a call that webhook answered, as the snippet does with
  `answered_calls`, which the example's webhook fills.

## Hosting the backend {#hosting}

A realtime session is not a request. It is a task that holds a WebSocket to the provider open for the
whole call (on a [WebRTC sideband](#browser-webrtc-server-sideband), the control-plane WebSocket),
next to the client's own connection in the [relay](#browser-backend-websocket-relay) and telephony
setups. Each [`session()`][pydantic_ai.agent.AgentRealtime.session] opens its own provider connection,
and one process can run many sessions on its event loop. Unlike a
[standard agent](../production.md#deploy), where any instance can continue any conversation, a call
lives in the process that holds it: its provider connection, running tools, and (on Gemini) resumption
handles, which [can't be moved to another process](https://github.com/pydantic/pydantic-ai/issues/7302).

A WebSocket stays on the instance that accepted it, so the relay needs no extra routing; if your app
splits one call across several requests, route them to the instance holding the session. A client that
reconnects elsewhere gets a new provider session; to carry the conversation over, store
[`all_messages()`][pydantic_ai.realtime.RealtimeSession.all_messages] and
[seed](history.md#seeding-a-session) the new session with it.

In the relay and telephony setups, where the client's connection is the request, set the platform's
request or connection timeout above your longest call. Providers cap a connection's
duration too, and a `reconnect` policy survives that cap on most of them (see
[provider session limits](lifecycle.md#provider-session-limits)), but nothing recovers the client's
connection when your platform cuts it: in the relay above, the browser socket closing ends the session.
Enforce your own maximum call length with a [watchdog](lifecycle.md#ending-a-call) instead, ending a
sideband call as described below.

Scale on concurrent calls rather than request rate; a relay also carries every call's audio through
the backend, so watch its bandwidth and CPU. A deploy has to drain: stop sending new calls to an
instance and let its sessions finish before it exits, which can take far longer than the few seconds of
shutdown grace many platforms give by default. To end calls sooner, end them deliberately; on a
sideband that means [`hang_up()`][pydantic_ai.realtime.RealtimeSession.hang_up], because a sideband
whose process exits without hanging up leaves the browser's call up, and billed, with no agent behind
it (see [the sideband section](#browser-webrtc-server-sideband) and
[Ending a call](lifecycle.md#ending-a-call)).

Long-running compute fits this directly: containers or VMs behind a load balancer that supports
WebSockets. Platforms that cap requests work within their limits: Cloud Run, for example, counts a
relay's WebSocket against its request timeout, which can be raised to an hour, so a longer call has to
continue on a new connection and session. Request-scoped functions don't fit. With AWS Lambda behind
an Amazon API Gateway WebSocket API, API Gateway holds the client's connection but invokes the function
once per message, so no invocation lives long enough to hold the provider connection between messages.
The session has to run on long-lived compute instead. API Gateway can still front it, but then you
build the routing yourself: a registry that sends each message to the instance holding that call's
session, audio base64-encoded in text frames (it rejects binary ones) and sent back through its
`@connections` API, and a connection that closes after ten idle minutes or two hours.

[Browser WebRTC](#browser-webrtc-server-sideband) takes media off your servers: audio flows directly
between the browser and the provider, and your backend carries only the sideband. But in the
[offer handler above](#browser-webrtc-server-sideband), the sideband runs as a background task after
the handler has returned the SDP answer, so no request or connection stays open for the call and the
platform can't see it. Give the sideband CPU outside requests (on Cloud Run, instance-based rather than
the default request-based billing) or run it on a worker or VM; scale on your own count of live
sideband tasks, since request metrics show an idle instance; and on shutdown, wait for the tasks in
`sideband_tasks` or hang up their calls before the process exits.
