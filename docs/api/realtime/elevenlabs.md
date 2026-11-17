# `pydantic_ai.realtime.elevenlabs`

The ElevenLabs realtime provider, for the Eleven Realtime speech-to-speech models. Requires the `elevenlabs-realtime`
optional group (`pip install "pydantic-ai-slim[elevenlabs-realtime]"`), which includes `openai` because the
model reuses the OpenAI Realtime codec, whose event types come from the OpenAI SDK.

[`ElevenLabsRealtimeModel`][pydantic_ai.realtime.elevenlabs.ElevenLabsRealtimeModel] is an
[`OpenAIRealtimeModel`][pydantic_ai.realtime.openai.OpenAIRealtimeModel] pointed at ElevenLabs, authenticated
by the same [`ElevenLabsProvider`][pydantic_ai.providers.elevenlabs.ElevenLabsProvider] as ElevenLabs Agents.
For a hosted ElevenLabs agent, use
[`ElevenLabsAgentsRealtimeModel`][pydantic_ai.realtime.elevenlabs_agents.ElevenLabsAgentsRealtimeModel] instead.

::: pydantic_ai.realtime.elevenlabs
