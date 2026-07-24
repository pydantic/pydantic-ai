This example is a browser voice agent where the **browser exchanges audio with the provider (OpenAI
or Azure OpenAI) directly over WebRTC** (lowest latency), while a
[Pydantic AI sideband](../realtime/index.md#browser-webrtc) on the server runs the agent's tools,
builds message history, and keeps the API key off the client.

It's the recommended topology for browser voice agents: the server never sits in the audio path — it is
the control plane.

```text
   browser ──mic/speaker audio (WebRTC media)──▶  OpenAI / Azure OpenAI Realtime
          ◀─────────────────────────────────────
      │  SDP offer (POST /offer)                    ▲ control WebSocket (call_id)
      ▼                                             │
   FastAPI backend ──answer_webrtc_offer()──▶ provider ──session(provider_session=…)──┘
                   (relays the SDP, gets a call_id)     (runs tools, builds history)
```

Demonstrates:

- [browser WebRTC + server sideband](../realtime/index.md#browser-webrtc) with the
  [OpenAI][pydantic_ai.realtime.openai.OpenAIRealtimeModel] and
  [Azure OpenAI][pydantic_ai.realtime.azure.AzureRealtimeModel] providers
- [`AgentRealtime.answer_webrtc_offer`][pydantic_ai.agent.AgentRealtime.answer_webrtc_offer] — relaying the
  browser's SDP offer server-side, with the agent's instructions and tools baked in, so the API key
  never reaches the client
- [`agent.realtime(model).session(provider_session=…)`][pydantic_ai.agent.AgentRealtime.session] — running the
  agent's [tools](../realtime/index.md#tool-calling) over the call's control plane while the browser owns
  the audio

## Running the Example

You'll need an `OPENAI_API_KEY` with realtime access, in a `.env` file at the repo root:

```dotenv
OPENAI_API_KEY=...
```

To run against **Azure OpenAI** instead, point `WEBRTC_REALTIME_MODEL` at your realtime deployment:

```dotenv
WEBRTC_REALTIME_MODEL=azure:gpt-realtime
AZURE_OPENAI_ENDPOINT=https://my-resource.openai.azure.com
AZURE_OPENAI_API_KEY=...
```

!!! note "Azure needs an input-transcription deployment"
    Azure resolves models against your resource's **deployments**: the resource needs a realtime
    deployment (the `azure:<deployment-name>` segment above) *and* an input-transcription deployment,
    because a [sideband session](../realtime/index.md#browser-webrtc) records the user's turns as
    transcripts. The default is `gpt-realtime-whisper`; if your transcription deployment is named
    differently, set `WEBRTC_TRANSCRIPTION_MODEL` to its name.

With [dependencies installed and your key set](./setup.md#usage), start the server:

```bash
uvicorn pydantic_ai_examples.realtime_webrtc.app:app
```

Open <http://localhost:8000>, click **Start call**, allow microphone access, and ask "What time is it
in Tokyo?" or "What's your refund policy?" to trigger a server-side tool.

Overrides: `WEBRTC_REALTIME_MODEL` (default `openai:gpt-realtime`), `WEBRTC_REALTIME_VOICE` (default
`marin`), `WEBRTC_TRANSCRIPTION_MODEL` (default: the provider's `'auto'` choice).

!!! note "The microphone needs a secure context"
    Browsers grant microphone access on `localhost` or over HTTPS. To open the example from another
    device, expose the local server over HTTPS with a
    [Cloudflare quick tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/):

    ```bash
    cloudflared tunnel --url http://localhost:8000
    ```

## Example Code

The server — it relays the SDP offer to OpenAI, attaches the sideband session, and runs the tools:

```snippet {path="/examples/pydantic_ai_examples/realtime_webrtc/app.py"}```

The browser — it captures the microphone, negotiates WebRTC through the backend, and plays the audio
back. Plain HTML and JavaScript, no build step:

```snippet {path="/examples/pydantic_ai_examples/realtime_webrtc/index.html"}```
