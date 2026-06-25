Point your phone's camera at something and *ask about it* — out loud — like the camera mode in the
ChatGPT and Gemini apps. Hold up a plant: "what is this?" Show it a circuit board: "which pin is
ground?" The model sees what you see and answers in its own voice.

It's a wonderfully small amount of code, because [Gemini Live](../realtime.md) accepts video frames
natively. The browser streams your microphone *and* a frame a second from the camera into one
realtime session, and plays the audio back. No tools, no supervisor — just conversation, with eyes.

Demonstrates:

- [realtime sessions](../realtime.md) with the [Gemini provider][pydantic_ai.realtime.google.GoogleRealtimeModel]
- [sending images](../realtime.md#images) — each camera frame is an
  [`ImageInput`][pydantic_ai.realtime.ImageInput] sent with
  [`send_image`][pydantic_ai.realtime.RealtimeSession.send_image]
- [live vision](../realtime.md#images) — `turn_coverage='all_input'` plus a *Watch* toggle so the
  assistant reacts to what changes, not just to your voice
- [web search](../realtime.md#built-in-tools-web-search) — the
  [`WebSearch`][pydantic_ai.capabilities.WebSearch] capability (Grounding with Google Search), so it
  can answer with current facts and render the [`Sources`][pydantic_ai.realtime.Sources] it grounded
  on as citation chips in the UI (on by default)

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

Open <http://localhost:8000>, tap **Start**, allow camera + microphone, and start talking. Hold
things up to the camera and ask away.

!!! note "Camera and mic need a secure context"
    Browsers only grant camera/microphone on `localhost` or over HTTPS — which is exactly why the
    *next* section exists, so you can use it from your phone.

### A frame alone won't make it talk — that's where *Watch* comes in

Here's a subtlety worth understanding. In a realtime session, what makes the model *take its turn*
is your **voice** (the server detects when you stop speaking). A camera frame is passive context — it
rides along, but on its own it doesn't trigger a reply. So if you silently show two fingers, the
model stays quiet until you ask "how many fingers?".

The example handles this in two complementary ways:

1. The assistant always receives **every** frame (`turn_coverage='all_input'`), so the live scene is
   in context whenever it *does* answer.
2. The **Watch** toggle makes it proactive: while it's on, the browser nudges the model every couple
   of seconds — "say what changed, otherwise stay silent" — so it narrates the scene on its own.

!!! note "`all_video` vs `all_input` on Vertex"
    `turn_coverage='all_input'` is the default because it works everywhere. The newer `all_video`
    value (all video, but audio only during speech) isn't accepted on Vertex's `v1beta1` API yet — if
    you set `CAMERA_TURN_COVERAGE=all_video` there you'll get an "invalid value" error, so stick with
    `all_input` on Vertex.

For the best version of this, use a **native-audio** model and let it decide when to speak, so it
stays quiet when nothing's changed instead of replying to every nudge:

```bash
export CAMERA_REALTIME_MODEL=gemini-live-2.5-flash-native-audio
export CAMERA_PROACTIVE=true     # the model decides when to talk (native-audio only)
export CAMERA_AFFECTIVE=true     # optional: emotion-aware delivery
```

!!! tip "Watch mode costs tokens"
    Because *Watch* prompts the model on a timer, it uses tokens even while you're quiet. It's great
    for a demo or a hands-free "keep an eye on this" moment; turn it off when you just want to chat.

### Use it from your phone (HTTPS)

This example is *made* to be used from a phone — that's where a camera you can wave around is most
fun. Phones need HTTPS, so expose your local server with a [Cloudflare](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/)
quick tunnel (no account needed):

```bash
cloudflared tunnel --url http://localhost:8000
```

It prints a `https://<random>.trycloudflare.com` URL — open *that* on your phone, tap **Start**, and
allow camera + mic. The tunnel forwards the WebSocket too, so audio and frames flow straight through.
(`ngrok http 8000` works the same way.)

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
