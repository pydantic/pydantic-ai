# Realtime phone agent (Twilio Media Streams)

Answers phone calls with a Pydantic AI realtime voice agent. Twilio streams each call's audio to this
server over Media Streams, and the server bridges it to a realtime session, converting the line's 8 kHz
mu-law to PCM16 and back, and tracking what the caller has heard for barge-in (see the
[deployment guide](https://pydantic.dev/docs/ai/realtime/deployment/#twilio-media-streams)).

## Run

Set `OPENAI_API_KEY`, `TWILIO_AUTH_TOKEN` and `PUBLIC_URL` (the server's public HTTPS address) in a `.env`
at the repo root, then:

```bash
uv run --all-packages uvicorn pydantic_ai_examples.realtime_twilio.app:app
```

Point your Twilio number's "A call comes in" webhook at `<PUBLIC_URL>/twiml` (`HTTP POST`) and call it.
