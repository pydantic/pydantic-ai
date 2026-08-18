This camera agent streams microphone audio and one camera frame per second into a
[realtime session](../realtime/overview.md), then plays and captions the spoken response. Point it at
objects to ask about them, enable *Watch* for proactive narration, or show it a sketch to redraw.

The example demonstrates:

- provider-agnostic [realtime sessions](../realtime/overview.md) with profile-derived PCM sample rates
- [image input](../realtime/audio.md#images) using
  [`BinaryContent`][pydantic_ai.messages.BinaryContent]
- live vision with `turn_coverage='all_input'` and a *Watch* toggle
- a regular function tool that delegates diagram rendering to a second
  [`Agent`][pydantic_ai.Agent]
- [web search](../realtime/tools.md#native-tools) with
  [`WebSearch`][pydantic_ai.capabilities.WebSearch] and clickable citations
- a model picker and provider-aware voice, modality, VAD, and Gemini settings

## Running the Example

Add credentials for a picker model to the repository-root `.env`, for example:

```dotenv
GOOGLE_API_KEY=your-google-api-key
```

The sketch-redraw tool delegates to a separate drawing agent — `google:gemini-3.5-flash` by
default, which reuses the same `GOOGLE_API_KEY`. Set `CAMERA_DRAW_MODEL` to any other
`provider:model` your credentials cover, or `CAMERA_DRAW=false` to disable drawing; the rest of the
assistant works either way.

With [dependencies installed and environment variables set](./setup.md#usage), start the local
server:

```bash
python/uv-run -m pydantic_ai_examples.realtime_camera.app
```

Open <http://localhost:8000>, select **Start**, and allow camera and microphone access.

The model defaults to `google:gemini-3.1-flash-live-preview`; set `CAMERA_REALTIME_MODEL` to change
it, or use the picker to switch to any Google, OpenAI, or Azure OpenAI `provider:model` per session
(xAI realtime doesn't support camera image input). The selected model's realtime profile supplies
the browser's PCM input and output sample rates: Gemini input uses 16 kHz, while OpenAI and Azure
input uses 24 kHz.

!!! warning "Keep the example local"
    The WebSocket uses provider credentials from the server and has no user authentication. The
    example checks that the browser's origin matches the host it's served from, but that is a
    development safeguard rather than production access control.

    Do not expose the server through a Cloudflare quick tunnel, ngrok, or a public reverse proxy.
    For another device, deploy behind authentication and TLS on a network you control, with
    user-level quotas and rate limits appropriate to your environment.

## Watch mode

Camera frames add visual context but do not start a model turn. *Watch* periodically sends a short
text turn while the model is idle, prompting it to report a visual change without interrupting
speech already in progress. Set `CAMERA_WATCH_PROMPT` to customize that instruction.

Gemini native-audio models can decide that nothing needs saying:

```bash
export CAMERA_PROACTIVE=true
export CAMERA_AFFECTIVE=true
```

`CAMERA_TURN_COVERAGE` defaults to `all_input`, which works with both the Gemini Developer API and
Vertex AI. Watch mode consumes tokens while enabled.

## Search and citations

With `CAMERA_WEB_SEARCH=true` (the default), the example adds
[`WebSearch`][pydantic_ai.capabilities.WebSearch] when the selected model profile supports native
search. Native-tool return events are converted into citation chips; the browser accepts only
HTTP(S) source URLs.

## Redraw a diagram

With `CAMERA_DRAW=true` (the default), the realtime agent can call `redraw_diagram`. It gives a
detailed textual description of the visible sketch to a separate
[`Agent`][pydantic_ai.Agent], which produces self-contained HTML. The browser displays that HTML in
an opaque-origin iframe that blocks scripts and network access, and retains the PNG export action.

The default is a fast small model because the user is waiting on a live call: the redraw's latency
is dominated by HTML output tokens, so a larger model mostly adds thinking time, not quality.
Configure the drawing model independently:

```bash
export CAMERA_DRAW_MODEL=anthropic:claude-haiku-4-5
```

Drawing and web search remain enabled together when the selected realtime model supports both.
Tools [run concurrently][pydantic_ai.agent.AgentRealtime.session], so drawing does not replace the
voice conversation.

## Vertex AI

Use Application Default Credentials when your organization does not allow Gemini API keys:

```bash
gcloud auth application-default login
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=your-project
export GOOGLE_CLOUD_LOCATION=us-central1
```

## How the bridge works

The browser and provider are connected by two small concurrent pumps in `_run_session`:

```text
browser ── PCM16 + JPEG/text ──▶ FastAPI /ws ──▶ RealtimeSession
browser ◀── PCM16 + JSON events ──────────────── RealtimeSession
```

Before microphone capture begins, the server sends `session_config` over the JSON channel with the
profile-derived audio rates. The inbound pump then forwards PCM, image, text, and Watch
messages. The event pump returns audio, transcripts, barge-in notifications, grounding citations,
drawing updates, and turn completion. Either side ending cancels the other pump and closes the
session cleanly.

## Example Code

The server contains the realtime bridge and the subordinate Watch, grounding, and drawing helpers:

```snippet {path="/examples/pydantic_ai_examples/realtime_camera/app.py"}```

The build-free browser captures media, waits for session configuration, and renders every demo
feature:

```snippet {path="/examples/pydantic_ai_examples/realtime_camera/index.html"}```
