Add browser WebRTC + server sideband support for realtime speech-to-speech (`answer_webrtc_offer`, `create_client_secret`, `realtime_session(provider_session=…)`, Azure Entra ID)

- Closes #<issue>

## Summary

Browser voice agents want the audio to flow **browser ↔ provider directly** over WebRTC (lowest
latency, browser-grade media handling), while the tools, history, and secrets stay server-side. This PR
adds that topology to the realtime module: the server relays the browser's SDP offer to the provider,
gets back an SDP answer and a `call_id`, and then attaches a **sideband** — a normal realtime control
connection to the *same* call — over which our existing `RealtimeSession` runs the agent loop while the
browser owns the audio.

Built in the shared OpenAI-protocol layer, so **OpenAI and Azure OpenAI** both get it. Gemini Live and
xAI are WebSocket-only and out of scope (see xAI findings below).

Highlights:

- **SDP relay (secure path):** `OpenAIRealtimeModel.answer_webrtc_offer(sdp_offer, …)` POSTs the offer +
  session config as multipart to `/realtime/calls` (via the provider's raw `httpx` client — the SDK's
  `realtime.calls.create` helper sends a boundary-less `Content-Type`), parses the `call_id` from the
  `Location` header, and returns a `WebRTCAnswer`. The browser never holds a token.
- **Ephemeral tokens:** `OpenAIRealtimeModel.create_client_secret(…)` mints a `RealtimeClientSecret` for
  the alternative flow where the browser negotiates the call itself.
- **Sideband session:** `agent.realtime_session(model, provider_session=call)` attaches the control
  connection (`wss://…/realtime?call_id=…`), reusing the existing `_openai_protocol` codec, session
  state machine, and tool loop unchanged. The control WS applies the session config with `session.update`
  and waits for `session.updated` (the call already exists, so there's no `session.created` handshake).
- **`owns_media` profile flag:** `True` by default; the sideband session runs with `owns_media=False`,
  so its audio methods (`send_audio`/`commit_audio`/`clear_audio`) raise a clear `UserError` and
  `audio_retention` must stay `'transcript_only'` (transcripts still build the full history, so handoff
  to a text agent keeps working).
- **Azure Microsoft Entra ID:** `AzureRealtimeModel(credential=DefaultAzureCredential())` authenticates
  the signaling calls with an Entra bearer token (scope `https://ai.azure.com/.default`, "Cognitive
  Services User" role) instead of the API key; Azure uses `/realtime/calls?webrtcfilter=on`. No new
  hard dependency — the credential is accepted via a structural `AzureTokenCredential` Protocol.
- Docs rethink (browser/WebRTC as a first-class path, reconciled with the server-side narrative),
  a runnable browser example, and tests.

The ordering (relay the SDP → get the `call_id` → attach the sideband, before the browser has the
answer) means the call always exists before the control connection attaches, so the "connect too early
→ 404" race can't occur, and the tools are live before the browser can speak.

## New public API surface (please review)

Artifacts (in `pydantic_ai.realtime`, defined in `_base.py`):

- `RealtimeClientSecret(value: str, expires_at: datetime, provider_details: dict | None)`
- `WebRTCCall(provider_name: str, call_id: str, provider_details: dict | None)`
- `WebRTCAnswer(sdp: str, call: WebRTCCall)`

Profile:

- `RealtimeModelProfile.owns_media: bool` (default `True` in `DEFAULT_REALTIME_PROFILE`).

Methods on `RealtimeModel` (base raises `UserError`; implemented by `OpenAIRealtimeModel`, inherited by
`AzureRealtimeModel`):

- `async create_client_secret(*, instructions=None, tools=None, model_settings=None, expires_after_seconds=None) -> RealtimeClientSecret`
- `async answer_webrtc_offer(sdp_offer, *, instructions=None, tools=None, model_settings=None) -> WebRTCAnswer`
- `connect_webrtc(call, *, messages, model_settings, model_request_parameters) -> AbstractAsyncContextManager[RealtimeConnection]`

Agent:

- `Agent.realtime_session(..., provider_session: WebRTCCall | None = None)` (also on `AbstractAgent` /
  `WrapperAgent`).

Azure:

- `AzureRealtimeModel(..., credential: AzureTokenCredential | None = None)` and the
  `AzureTokenCredential` structural Protocol (in `pydantic_ai.realtime.azure`).

Open questions for a maintainer:

1. **Naming**: `WebRTCCall` / `WebRTCAnswer` / `provider_session=` vs alternatives (e.g. `RealtimeCall`).
   The param is named `provider_session` to leave room for non-WebRTC sidebands (e.g. SIP) later.
2. **`create_client_secret` scope**: kept as a model-level primitive only (no agent-level convenience),
   since the primary, fully-documented flow is the secure SDP relay. Add an agent-level convenience later
   if there's demand?
3. **Entra scope** is fixed to `https://ai.azure.com/.default`. Worth exposing an override for sovereign
   clouds, or leave for a follow-up?

## xAI WebRTC findings

**xAI does not support the WebRTC sideband and is scoped out.** Verified against xAI's docs
(`docs.x.ai` voice-agent + ephemeral-tokens pages): xAI realtime is **WebSocket-only**
(`wss://api.x.ai/v1/realtime?model=…`). There is no `/v1/realtime/calls` SDP endpoint and no
`call_id`-based server-side control/observer connection. Its only conversation identifier is
`conversation_id`, which is a session-resumption/cache mechanism, not a second control connection. xAI
*does* expose ephemeral tokens (`/v1/realtime/client_secrets`) for a browser to open a direct WebSocket,
but that's a different topology (no server sideband) and isn't included here. `XaiRealtimeModel`
therefore inherits the base `RealtimeModel` methods, which raise a clear `UserError`.

## Tests

All network-free or cassette-backed; `uv run pytest tests/realtime` passes (473 passed, 44 skipped).

- `tests/realtime/test_webrtc.py` — `httpx.MockTransport`-driven tests for `create_client_secret` and
  `answer_webrtc_offer` (request shape, multipart body, auth headers, `call_id` parsing, error paths),
  Azure `webrtcfilter=on` + Entra bearer minting, and the base-model rejections.
- `tests/realtime/test_openai.py::test_connect_webrtc_sideband_handshake` — the sideband control-WS
  handshake (attach by `call_id`, `session.update`-first, served-model capture) with fake WebSockets.
- `tests/realtime/test_openai_ws.py::test_webrtc_sideband_text_turn` — **recorded, offline-replayable**
  end-to-end: HTTP SDP relay (VCR cassette) + sideband control WebSocket (WS cassette) → agent turn +
  history + `owns_media` gating. Recorded live against OpenAI. (The WS cassette is stored under a
  dedicated `test_openai_ws_sideband/` subdir to avoid colliding with the VCR cassette path.)
- `tests/realtime/test_session.py` — `owns_media=False` gating of the audio methods.

**Cassettes still needing live recording:** none for OpenAI — all recorded and replay offline. Azure
WebRTC signaling + Entra were validated with unit tests (mock transport / fake credential) but have **no
live-recorded cassette** (no Azure realtime resource / Entra-enabled resource available in this
environment); record equivalents against a real Azure resource if desired.

## Example

`examples/pydantic_ai_examples/realtime_webrtc/` — a runnable FastAPI + browser demo of the secure flow
(server relays the SDP, browser owns the media, server runs tools). Docs page:
`docs/examples/realtime-webrtc.md`.

## Docs

- `docs/realtime/index.md` — new "Browser / WebRTC" section (topology, secure flow, code) and reconciled
  "How it works" / "Choose your path" / "Connecting a frontend" framing; `owns_media` note in the profile
  reference.
- `docs/realtime/openai.md`, `docs/realtime/azure.md` — provider-specific WebRTC sections (Azure includes
  Entra ID).
- `docs/api/realtime.md` — WebRTC artifacts/methods in the overview.

## Credit

The WebRTC + server-sideband approach and a reference implementation by **Nathan Gage** informed this
work (the `httpx` multipart workaround for the SDK's broken `Content-Type`, and the `call_id` sideband
design). The example README and this PR credit him; please confirm whether to keep the credit in the
user-facing example README.
