This example sends microphone audio and one camera frame per second from a browser to a
[Gemini Live realtime session](../realtime.md), then plays the model's spoken responses. You can ask
about objects in view or show the assistant a sketch to redraw.

By default, the assistant includes a `redraw_diagram` function tool that hands a captured frame to a
vision agent and renders the resulting diagram in the browser. Gemini 2.5 cannot combine function
calling with native Google Search grounding, so drawing and web search are mutually exclusive. The
default `CAMERA_DRAW=true` disables web search; set `CAMERA_DRAW=false` and leave
`CAMERA_WEB_SEARCH=true` to use grounding instead.

Demonstrates:

- [realtime sessions](../realtime.md) with the [Gemini provider][pydantic_ai.realtime.google.GoogleRealtimeModel]
- [sending images](../realtime.md#images) — each camera frame is an
  [`BinaryContent`][pydantic_ai.messages.BinaryContent] sent with [`send`][pydantic_ai.realtime.RealtimeSession.send]
- [live vision](../realtime.md#images) — `turn_coverage='all_input'` plus a *Watch* toggle so the
  assistant reacts to what changes, not just to your voice
- function tools — the default `redraw_diagram` tool converts a sketch into a clean diagram
- [web search](../realtime.md#built-in-tools-web-search) — the
  [`WebSearch`][pydantic_ai.capabilities.WebSearch] capability (Grounding with Google Search), so it
  can answer with current facts and render citations from native-tool return part events when drawing
  is disabled

## Running the Example

You'll need access to a **Gemini Live** model. The simplest path is a `GOOGLE_API_KEY` from
[Google AI Studio](https://aistudio.google.com/apikey), in a `.env` file at the repo root:

```dotenv
GOOGLE_API_KEY=...
```

With [dependencies installed and your key set](./setup.md#usage), start the server:

```bash
uvicorn pydantic_ai_examples.realtime_camera.app:app
```

Open <http://localhost:8000>, tap **Start**, allow camera and microphone access, and ask about what
the camera sees.

!!! note "Camera and mic need a secure context"
    Browsers grant camera and microphone access on `localhost` or over HTTPS. Use the HTTPS setup
    below when opening the example from another device.

### Triggering turns with Watch

Camera frames are passive context and do not trigger a model turn. Without speech or text input, the
model stays quiet even as the scene changes.

The example handles this in two ways:

1. The assistant always receives **every** frame (`turn_coverage='all_input'`), so the live scene is
   in context whenever it *does* answer.
2. The **Watch** toggle periodically sends a short text turn asking the model to describe changes or
   stay silent when nothing notable changed.

!!! note "`all_video` vs `all_input` on Vertex"
    `turn_coverage='all_input'` is the default because it works on both supported Google API
    surfaces. Vertex's `v1beta1` API does not accept `all_video` (all video, but audio only during
    speech); use `all_input` on Vertex.

With a native-audio model, proactive audio lets the model decide whether a Watch prompt needs a
spoken response:

```bash
export CAMERA_PROACTIVE=true     # the model decides when to talk (native-audio only)
export CAMERA_AFFECTIVE=true     # optional: emotion-aware delivery
```

!!! tip "Watch mode costs tokens"
    Watch prompts the model on a timer, so it uses tokens even while no one is speaking. Turn it off
    when timed scene updates are not needed.

### Use it from your phone (HTTPS)

To use the example from a phone, expose the local server over HTTPS with a
[Cloudflare quick tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/)
(no account needed):

```bash
cloudflared tunnel --url http://localhost:8000
```

Open the printed `https://<random>.trycloudflare.com` URL on the phone, tap **Start**, and allow
camera and microphone access. The tunnel also forwards the WebSocket. `ngrok http 8000` is an
alternative.

### Using a work Google Cloud account (Vertex AI)

If your organization doesn't allow Gemini API keys, use **Vertex AI** with Application Default
Credentials instead — no key at all:

```bash
gcloud auth application-default login
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=your-project
export GOOGLE_CLOUD_LOCATION=us-central1   # a region where the Live API is available
```

The demo detects `GOOGLE_GENAI_USE_VERTEXAI` and connects through Vertex; everything else is the same.

!!! note "Vertex model names differ"
    The Live models available on Vertex aren't always named like their AI Studio counterparts, and
    they vary by project and region. If a connection fails with "Publisher model … was not found",
    list what you actually have and set `CAMERA_REALTIME_MODEL` to one of them:

    ```python {test="skip" lint="skip"}
    from google import genai

    client = genai.Client(vertexai=True, project='your-project', location='us-central1')
    print([m.name for m in client.models.list() if 'live' in (m.name or '')])
    ```

## Example Code

The server — it bridges the browser to a single Gemini Live session, forwarding mic audio and camera
frames in, and audio and transcripts out:

```snippet {path="/examples/pydantic_ai_examples/realtime_camera/app.py"}```

The browser — it captures the camera and microphone, sends ~1 frame per second, plays the audio back,
and drives the *Watch* toggle. Plain HTML and JavaScript, no build step:

```snippet {path="/examples/pydantic_ai_examples/realtime_camera/index.html"}```
