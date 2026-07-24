# Realtime WebRTC voice agent (browser + server sideband)

A browser voice agent where the **browser exchanges audio with OpenAI directly over WebRTC** (lowest
latency) while a **Pydantic AI sideband** on the server runs the agent's tools, builds history, and
keeps the API key off the client.

This is the recommended topology for browser voice agents (see the
[realtime guide](https://ai.pydantic.dev/realtime/#browser--webrtc)). The server never sits in the
audio path — it is the control plane:

```
browser ──mic/speaker audio (WebRTC media)──▶  OpenAI Realtime
       ◀─────────────────────────────────────
   │  SDP offer (POST /offer)                    ▲ control WebSocket (call_id)
   ▼                                             │
FastAPI backend  ──answer_webrtc_offer()──▶ OpenAI  ──realtime_session(provider_session=…)──┘
                 (relays SDP, gets call_id)        (runs tools, builds history)
```

## Run

Set `OPENAI_API_KEY` in a `.env` at the repo root, then:

```bash
uv run --all-packages uvicorn pydantic_ai_examples.realtime_webrtc.app:app
```

Open <http://localhost:8000> (localhost is a secure context, so the browser allows the microphone) and
click **Start call**. Ask "What time is it in Tokyo?" or "What's your refund policy?" to trigger a
server-side tool.

Overrides: `WEBRTC_REALTIME_MODEL` (default `gpt-realtime`), `WEBRTC_REALTIME_VOICE` (default `marin`).

The app is instrumented with [Logfire](https://ai.pydantic.dev/logfire/): set `LOGFIRE_TOKEN` (e.g. in
the same `.env`) to see the realtime session, model turns, and tool calls as traces.

## How it works

1. The browser creates an `RTCPeerConnection`, captures the microphone, and `POST`s its SDP **offer**
   to `/offer`.
2. The backend calls [`answer_webrtc_offer`][pydantic_ai.realtime.RealtimeModel.answer_webrtc_offer],
   which relays the offer to OpenAI (using the server's key) and returns the SDP **answer** plus a
   [`WebRTCSession`][pydantic_ai.realtime.WebRTCSession] (`call_id`).
3. The backend attaches a sideband session with
   [`agent.realtime(model).session(provider_session=call)`][pydantic_ai.agent.AgentRealtime.session] and returns the
   answer to the browser. Media now flows browser ↔ OpenAI; the sideband runs the tools and records the
   conversation in `session.all_messages()`.

Because the session doesn't own the audio transport, its `send_audio` / `commit_audio` / `clear_audio`
methods are unavailable — the browser owns the media.

## Use it from your phone (HTTPS)

The microphone only works in a secure context, so a phone needs HTTPS. Expose the local server with a
Cloudflare quick tunnel (no account needed):

```bash
cloudflared tunnel --url http://localhost:8000
```

Open the printed `https://<...>.trycloudflare.com` URL on the phone and allow the microphone.
