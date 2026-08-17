# `pydantic_ai.realtime.xai`

The xAI Grok Voice realtime API provider. Requires the `realtime`, `xai`, and `openai` optional
groups (`pip install "pydantic-ai-slim[realtime,xai,openai]"`) — `openai` because the model reuses
the OpenAI Realtime codec, whose event types come from the OpenAI SDK.

xAI's realtime API is a clone of the OpenAI Realtime protocol, so
[`XaiRealtimeModel`][pydantic_ai.realtime.xai.XaiRealtimeModel] reuses the OpenAI codec (event
mapping, seeding, the WebSocket connection). Turn-taking uses the shared
[`TurnDetection`][pydantic_ai.realtime.TurnDetection] (or `False` for push-to-talk); for exact
server-VAD control, `xai_turn_detection` accepts
[`ServerVAD`][pydantic_ai.realtime.openai.ServerVAD] and fully overrides the shared setting. It
diverges only where xAI does: it supports
cancellation-based interruption but not output truncation, has no image input, and streams input
transcription as cumulative snapshots that may revise earlier text, rather than as incremental deltas.
Authentication comes from an
[`XaiProvider`][pydantic_ai.providers.xai.XaiProvider], mirroring [`XaiModel`][pydantic_ai.models.xai.XaiModel].

::: pydantic_ai.realtime.xai
