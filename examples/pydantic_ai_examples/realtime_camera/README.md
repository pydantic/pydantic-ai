# Realtime camera + voice assistant

A minimal "talk and show" assistant: speak to a **Gemini Live** model and point your camera at
things to ask about them. The browser streams microphone audio (PCM16, 16 kHz) and ~1 fps JPEG
camera frames into a realtime session and plays the model's audio back — no tools, no supervisor,
just conversation with live vision.

It showcases the Gemini provider's native **live video** support: each camera frame is an
[`ImageInput`][pydantic_ai.realtime.ImageInput] sent with
[`send_image`][pydantic_ai.realtime.RealtimeSession.send_image]; the audio is the usual
speech-to-speech path. (See the [realtime guide](https://ai.pydantic.dev/realtime/).)

## Run

Set `GOOGLE_API_KEY` (Gemini Live access) in a `.env` at the repo root, then:

```bash
uv run --all-packages uvicorn pydantic_ai_examples.realtime_camera.app:app
```

Open <http://localhost:8000> and tap **Start** — `localhost` is a secure context, so the browser
allows camera + microphone. Talk, and hold something up to the camera.

Overrides: `CAMERA_REALTIME_MODEL` (default `gemini-live-2.5-flash`; try
`gemini-live-2.5-flash-native-audio`), `CAMERA_REALTIME_VOICE` (default `Puck`).

### Watch mode — make the assistant react to the camera

By default the model only replies when you speak; a camera frame is passive context, so the model
won't say "I see two fingers" until you ask. The assistant always receives **every** frame
(`turn_coverage='all_input'`), and the **Watch** toggle makes it proactive: while on, the browser
nudges the model every couple of seconds to report what changed.

For the best experience use a **native-audio** model and set `CAMERA_PROACTIVE=true` so the model
stays silent when nothing changed instead of replying to every nudge:

```bash
export CAMERA_REALTIME_MODEL=gemini-live-2.5-flash-native-audio
export CAMERA_PROACTIVE=true     # model decides when to speak (native-audio only)
export CAMERA_AFFECTIVE=true     # optional: emotion-aware delivery (native-audio only)
```

Other overrides: `CAMERA_TURN_COVERAGE` (`activity_only` | `all_input` | `all_video`; default
`all_input`, which works everywhere — the newer `all_video` isn't accepted on Vertex yet),
`CAMERA_WATCH_PROMPT` (what the model is asked on each nudge). Watch mode costs extra tokens, since
it prompts the model on a timer.

### Web search

The assistant can **search the web** (Gemini's Grounding with Google Search) via
`Agent(capabilities=[WebSearch()])` — on by default. Ask it something current ("what's the weather
where I'm pointing?", "who won last night?") and it grounds the answer, then the UI shows the
**sources** it used as clickable chips. Set `CAMERA_WEB_SEARCH=false` to disable, or if your
model/region doesn't support grounding.

> URL reading (`WebFetch`) isn't enabled here: the Live native-audio model can't combine Google Search
> grounding with function calling in one session, so a fetch tool alongside grounding wouldn't be
> callable. See the [realtime guide](https://ai.pydantic.dev/realtime/#built-in-tools-web-search) for
> how to use `WebFetch` (including its local fallback) on its own.

### Using a work Google Cloud account (Vertex AI)

If your organization disallows Gemini API keys, use **Vertex AI** with Application Default
Credentials instead — no API key:

```bash
gcloud auth application-default login
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=your-project
export GOOGLE_CLOUD_LOCATION=us-central1   # a region where the Live API is available
```

The demo detects `GOOGLE_GENAI_USE_VERTEXAI` and connects through Vertex; everything else is the same.

## Use it from your phone (HTTPS)

Camera and microphone only work in a **secure context**, so a phone needs HTTPS — not the
machine's `http://<lan-ip>:8000`. The easiest way is a Cloudflare quick tunnel (no account needed):

```bash
# install once, e.g. on macOS: brew install cloudflared
cloudflared tunnel --url http://localhost:8000
```

It prints a `https://<random>.trycloudflare.com` URL — open that on your phone, tap **Start**, and
allow camera + mic. The tunnel forwards both HTTP and the WebSocket, so audio and frames flow
straight through. (`ngrok http 8000` works the same way.)

## How it works

```
browser ──mic PCM16@16k (binary WS)──┐
        ──camera JPEG frames (JSON WS)─┤→  FastAPI /ws  →  Agent.realtime_session(GoogleRealtimeModel)
        ◀──model audio (binary WS)─────┘                    └ send_audio / send_image / receive
        ◀──transcripts (JSON WS)───────┘
```

`app.py` bridges the WebSocket to a single Gemini Live session: inbound binary frames are forwarded
as audio, `{"type":"image"}` messages as video frames, and `{"type":"text"}` as typed turns; outbound
audio is streamed back as binary and transcripts as JSON. The session closes (and its telemetry span
flushes) when the socket drops.
