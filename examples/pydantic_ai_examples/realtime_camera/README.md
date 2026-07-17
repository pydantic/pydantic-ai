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

Overrides: `CAMERA_REALTIME_MODEL` (default `gemini-2.5-flash-native-audio-latest`; try
`gemini-3.1-flash-live-preview`), `CAMERA_REALTIME_VOICE` (default `Puck`).

The app is instrumented with [Logfire](https://ai.pydantic.dev/logfire/): if a `LOGFIRE_TOKEN` is
set (e.g. in the same `.env`), the realtime session, model turns, and tool calls show up as traces;
without a token nothing is sent and the app works as normal.

### Watch mode — make the assistant react to the camera

By default the model only replies when you speak; a camera frame is passive context, so the model
won't say "I see two fingers" until you ask. The assistant always receives **every** frame
(`turn_coverage='all_input'`), and the **Watch** toggle makes it proactive: while on, the browser
nudges the model every couple of seconds to report what changed.

The default model is a **native-audio** one, so for the best experience set `CAMERA_PROACTIVE=true` —
the model then stays silent when nothing changed instead of replying to every nudge:

```bash
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

### Redraw a sketch

Hold a **hand-drawn diagram** up to the camera — a system design, flow chart, wireframe — and ask
the assistant to *"clean this up"* or *"redraw it properly"*. It calls the `redraw_diagram` tool,
which is a regular [`@agent.tool`][pydantic_ai.Agent.tool] that the
[`realtime_session`][pydantic_ai.Agent.realtime_session] executes automatically. The tool hands the
current camera frame to a **separate vision agent** — a plain [`Agent`][pydantic_ai.Agent] running
Gemini, using the same `GOOGLE_API_KEY` as the live session — that returns a clean, self-contained
HTML version of the drawing. The browser renders it in an overlay and can export it to PNG
client-side.

Tools [run concurrently by design][pydantic_ai.Agent.realtime_session], so the voice conversation
keeps flowing while the diagram is drawn, then the model announces it when it appears.

The frames streamed to Gemini are small and low-detail to keep the live session cheap, so for the
redraw the server asks the browser for a one-off **high-resolution snapshot** (the camera is opened at
up to 1920×1080 and captured at ~1600 px wide) that goes only to the drawing agent — enough detail to
read hand-written labels without paying for high-res frames on every turn.

```bash
export CAMERA_DRAW_MODEL=google:gemini-3.5-flash  # optional: any `provider:model` vision model
export CAMERA_DRAW=false                            # optional: turn the feature off
```

> Because Gemini Live can't combine function calling with Google Search grounding in one session,
> enabling the drawing tool turns [web search](#web-search) off.

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
