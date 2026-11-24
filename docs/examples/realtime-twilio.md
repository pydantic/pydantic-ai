---
description: "A Pydantic AI phone agent example: Twilio Media Streams bridged to a realtime voice session, with barge-in that knows what the caller heard."
---

This example answers phone calls with a realtime voice agent. Twilio streams each call's audio to the
server over [Media Streams](../realtime/deployment.md#twilio-media-streams), and the server bridges it to a
Pydantic AI realtime session: it converts the line's 8 kHz mu-law to the session's PCM16 and back, and keeps
track of what the caller has heard, so a reply they talk over is recorded as far as they heard it.

```text
caller ── phone ── Twilio ──POST /twiml (signed webhook)──▶ FastAPI backend
                     │                                         │
                     └──── Media Stream WebSocket ─────────────┤ /media-stream
                           (8 kHz mu-law, marks, clear)        ▼
                                                     realtime session ── provider
```

Demonstrates:

- the [Twilio Media Streams bridge](../realtime/deployment.md#twilio-media-streams): TwiML that connects the call
  to a stream, and per-call context passed through the stream's custom parameters as the session's
  [dependencies](../dependencies.md)
- barge-in on a buffered transport: `mark` messages count what Twilio has played, `clear` drops the rest, and
  [`interrupt(played_bytes=...)`][pydantic_ai.realtime.RealtimeSession.interrupt] truncates the reply there
- keypad presses sent to the model as text, and the session hung up when the call ends
- checking Twilio's request signatures with Twilio's SDK

## Running the Example

You'll need a Twilio phone number, an `OPENAI_API_KEY` with realtime access, and a public HTTPS address for
the server. Set these in a `.env` file at the repo root:

```dotenv
OPENAI_API_KEY=...
TWILIO_AUTH_TOKEN=...
PUBLIC_URL=https://<your public host>
```

`TWILIO_AUTH_TOKEN` is your Twilio account's auth token, which the server checks Twilio's signatures with.
To run against another realtime model, such as a [GPT-Live model](../realtime/openai.md#model-names), set
`TWILIO_REALTIME_MODEL` to its name.

With [dependencies installed and your key set](./setup.md#usage), start the server:

```bash
uv run --all-packages uvicorn pydantic_ai_examples.realtime_twilio.app:app
```

To give it a public address from your machine, a
[Cloudflare quick tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/)
works, and prints the address to put in `PUBLIC_URL`:

```bash
cloudflared tunnel --url http://localhost:8000
```

In the Twilio console, set the phone number's "A call comes in" webhook to `<PUBLIC_URL>/twiml`, using
`HTTP POST`, then call the number and ask when the restaurant is open.

## Example Code

The server: the webhook, the stream, and the bridge between Twilio and the session.

```snippet {path="/examples/pydantic_ai_examples/realtime_twilio/app.py"}```

The conversion between the line's audio and the session's, with the standard library alone:

```snippet {path="/examples/pydantic_ai_examples/realtime_twilio/audio.py"}```
