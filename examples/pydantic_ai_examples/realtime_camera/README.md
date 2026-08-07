# Realtime camera + voice assistant

A browser camera agent that watches and narrates. It streams microphone PCM and one downscaled JPEG
camera frame per second into a provider-agnostic realtime session, then plays model audio and renders
transcripts. The server reads both PCM sample rates from the selected model profile, so Gemini's
16 kHz input and OpenAI/Azure's 24 kHz input are handled correctly.

The launch demo also keeps its complete feature set:

- **Watch** periodically prompts an idle model to narrate visual changes.
- **Web search** grounds current answers and displays HTTP(S) citation chips when the selected model
  supports native search.
- **Redraw** lets the realtime agent describe a sketch to a separate drawing agent. Generated HTML
  is displayed in a sandboxed, network-blocked iframe and can be exported as PNG.
- The **settings panel** includes the model picker, voice, modality, VAD, and Gemini-specific options.

xAI realtime is not offered because it does not support camera image input.

## Run locally

Set credentials for the provider you intend to select (`GOOGLE_API_KEY`, `OPENAI_API_KEY`, or
`AZURE_OPENAI_*`) in a `.env` at the repository root, then run:

```bash
uv run --all-packages uvicorn pydantic_ai_examples.realtime_camera.app:app
```

Open <http://localhost:8000>, select **Start**, and allow camera and microphone access. `localhost`
is a browser secure context.

The picker accepts only the models in `ALLOWED_MODELS` in `app.py`. To use a different deployment,
set `CAMERA_REALTIME_MODEL` before starting the server; that configured value is added to the
allowlist and becomes the default.

Useful environment settings:

```bash
export CAMERA_REALTIME_MODEL=openai:gpt-realtime-2.1
export CAMERA_REALTIME_VOICE=marin
export CAMERA_DRAW_MODEL=gateway/anthropic:claude-sonnet-5
export CAMERA_DRAW=true
export CAMERA_WEB_SEARCH=true
export CAMERA_PROACTIVE=false
export CAMERA_AFFECTIVE=false
export CAMERA_TURN_COVERAGE=all_input
```

Leaving `CAMERA_REALTIME_VOICE` empty uses the provider's default. Language, turn coverage,
proactive audio, and affective dialog are Gemini-only. OpenAI and Azure map either VAD sensitivity
control to their shared turn-detection sensitivity.

The app is instrumented with [Logfire](https://ai.pydantic.dev/logfire/). When `LOGFIRE_TOKEN` is
set, realtime sessions, model turns, and tool calls appear as traces; otherwise nothing is sent.

## Watch, search, and drawing

Camera frames are passive context. Watch mode sends `CAMERA_WATCH_PROMPT` every few seconds only
while the model is listening, which avoids interrupting an in-flight response. It consumes tokens
while enabled. Gemini native-audio models can use `CAMERA_PROACTIVE=true` to remain silent when
nothing notable changed.

`CAMERA_WEB_SEARCH=true` enables [`WebSearch`][pydantic_ai.capabilities.WebSearch] only when the
selected realtime model profile supports it. Grounding sources are rendered as citation chips.

`CAMERA_DRAW=true` registers the regular `redraw_diagram` function tool. The realtime model passes a
detailed text description to a separate [`Agent`][pydantic_ai.Agent], which returns a self-contained
HTML diagram. Drawing and search can coexist when the selected realtime model supports both.

For Vertex AI, use Application Default Credentials:

```bash
gcloud auth application-default login
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=your-project
export GOOGLE_CLOUD_LOCATION=us-central1
```

## Security boundary

This is a local development example, not an internet-facing service. It checks that WebSocket
`Origin` matches `Host`, allowlists model IDs, limits concurrent sessions to 8, and caps audio and
JSON messages. These controls reduce accidental exposure; they do not provide authentication,
per-user quotas, or production abuse protection.

Do not publish it with a Cloudflare quick tunnel, ngrok, or a public reverse proxy. To use it from
another device, place it behind authentication and TLS on a network you control, then add
appropriate user-level rate limits and deployment-specific access controls.

## Realtime bridge

The essential data path is concentrated in `_run_session`:

```text
browser ── PCM16 + JPEG/text ──▶ FastAPI /ws ──▶ RealtimeSession
browser ◀── PCM16 + JSON events ──────────────── RealtimeSession
```

The server first sends `session_config` with the selected model profile's input and output sample
rates. Only then does the browser create its audio contexts and begin microphone capture. Two
concurrent pumps subsequently forward browser input and model events until either side disconnects.
