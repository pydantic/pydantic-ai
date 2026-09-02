# `pydantic_ai.realtime.elevenlabs`

The ElevenLabs Agents realtime provider. Requires the `elevenlabs-realtime` optional group
(`pip install "pydantic-ai-slim[elevenlabs-realtime]"`), which pulls in only the `realtime`
WebSocket transport; the REST preflight uses Pydantic AI's own HTTP client, so no ElevenLabs SDK is
needed.

ElevenLabs has no direct realtime conversational model, so
[`ElevenLabsRealtimeModel`][pydantic_ai.realtime.elevenlabs.ElevenLabsRealtimeModel] wraps a hosted
agent addressed by its `agent_id` and speaks the agent WebSocket protocol with its own codec. A
connect-time REST preflight fetches the agent, checks every requested per-conversation override
against the agent's override-permission allowlist (raising a
[`UserError`][pydantic_ai.exceptions.UserError] naming the exact toggle otherwise), and reconciles
the session's tools with the agent's client tools (error on mismatch by default, opt-in sync).
Turn-taking is owned by ElevenLabs' server-side turn model: there is no manual turn control, no
client-initiated interruption, and no session resumption. Authentication comes from an
[`ElevenLabsProvider`][pydantic_ai.providers.elevenlabs.ElevenLabsProvider], whose `base_url`
selects the data-residency region.

::: pydantic_ai.realtime.elevenlabs
