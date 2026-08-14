# `pydantic_ai.realtime.google`

The Gemini Live API provider. Requires the `google` optional group
(`pip install "pydantic-ai-slim[google]"`).

[`GoogleRealtimeModel`][pydantic_ai.realtime.google.GoogleRealtimeModel] runs over the `google-genai`
SDK (which manages the WebSocket transport). Gemini expects **16 kHz** PCM input (output is 24 kHz),
produces one response modality per session, and natively accepts live video frames sent as
[`BinaryImage`][pydantic_ai.messages.BinaryImage]. It exposes Gemini Live's session and generation
configuration through [`GoogleRealtimeModelSettings`][pydantic_ai.realtime.google.GoogleRealtimeModelSettings] —
shared turn-taking via [`TurnDetection`][pydantic_ai.realtime.TurnDetection], with finer Gemini-specific
control via [`AutomaticVAD`][pydantic_ai.realtime.google.AutomaticVAD] in `google_vad` plus
`google_activity_handling`/`google_turn_coverage`, voice via `google_voice` or a
[`MultiSpeaker`][pydantic_ai.realtime.google.MultiSpeaker] in `google_multi_speaker`,
and long-session [`ContextCompression`][pydantic_ai.realtime.google.ContextCompression] — with
resilience via session resumption + a [`ReconnectPolicy`][pydantic_ai.realtime.ReconnectPolicy] in the
`reconnect` setting.

::: pydantic_ai.realtime.google
