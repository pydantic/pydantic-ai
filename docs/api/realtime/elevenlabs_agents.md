# `pydantic_ai.realtime.elevenlabs_agents`

The ElevenLabs Agents realtime provider. Requires the `elevenlabs-agents-realtime` optional group
(`pip install "pydantic-ai-slim[elevenlabs-agents-realtime]"`), which pulls in only the `realtime`
WebSocket transport; the REST preflight uses Pydantic AI's own HTTP client, so no ElevenLabs SDK is
needed.

[`ElevenLabsAgentsRealtimeModel`][pydantic_ai.realtime.elevenlabs_agents.ElevenLabsAgentsRealtimeModel] wraps a hosted
agent addressed by its `agent_id` and speaks the agent WebSocket protocol with its own codec. A
connect-time REST preflight fetches the agent, checks each enumerated per-conversation override
setting against the agent's override-permission allowlist (raising a
[`UserError`][pydantic_ai.exceptions.UserError] naming the exact toggle otherwise; the raw
`elevenlabs_agents_config_override` escape hatch is deliberately not preflight-validated and is judged by
the server after the handshake instead), and reconciles the session's tools with the agent's client
tools (error on mismatch by default, opt-in sync).
Turn-taking is owned by ElevenLabs' server-side turn model: there is no manual turn control, no
client-initiated interruption, and no session resumption. Authentication comes from an
[`ElevenLabsAgentsProvider`][pydantic_ai.providers.elevenlabs_agents.ElevenLabsAgentsProvider], whose `base_url`
selects the data-residency region.

::: pydantic_ai.realtime.elevenlabs_agents
