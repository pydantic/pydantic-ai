# Connecting a frontend

Keep provider keys, tools, and business logic on the server; connect user devices to your backend,
not to the provider. Three transport shapes cover most deployments:

**Browser → backend → provider.** The shape available today: build a WebSocket endpoint on your
backend that accepts the browser's microphone audio and pumps it into
[`send_audio()`][pydantic_ai.realtime.RealtimeSession.send_audio], while relaying
[`stream_audio()`][pydantic_ai.realtime.RealtimeSession.stream_audio] output back for playback. The
[realtime camera example](../examples/realtime-camera.md) demonstrates this shape end to end.

**Browser-direct WebRTC.** On OpenAI and Azure OpenAI, the browser can exchange media directly with
the provider while your backend attaches a sideband to the same call and keeps running the agent
loop, tools, and history. See [Browser / WebRTC](lifecycle.md#browser-webrtc) and the runnable
[WebRTC example](../examples/realtime-webrtc.md). Gemini Live and xAI are WebSocket-only.

**SIP/telephony bridge.** Terminate the phone call with a telephony provider such as Twilio, then
build the service that connects its media stream (e.g. Twilio Media Streams over WebSocket) to the
backend session, transcoding between the line's codec and PCM16.

## A minimal WebSocket relay

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

In every shape the session — with its [tools](tools.md), [history](history.md), and
[usage limits](observability.md#usage-and-limits) — runs on your backend. Wiring a browser to the
provider with the provider's own SDK instead moves the agent loop into the client and gives up all
of that; prefer the shapes above.
