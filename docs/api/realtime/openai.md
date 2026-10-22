# `pydantic_ai.realtime.openai`

The OpenAI Realtime API provider. Requires the `realtime` and `openai` optional groups
(`pip install "pydantic-ai-slim[realtime,openai]"`).

[`OpenAIRealtimeModelSettings`][pydantic_ai.realtime.openai.OpenAIRealtimeModelSettings] configures
the session, including shared turn-taking via [`TurnDetection`][pydantic_ai.realtime.TurnDetection]
(or `False` for push-to-talk). For finer control, `openai_turn_detection` accepts
[`ServerVAD`][pydantic_ai.realtime.openai.ServerVAD] or
[`SemanticVAD`][pydantic_ai.realtime.openai.SemanticVAD] and fully overrides the shared setting.
Resilience comes from the `reconnect` setting: a
[`ReconnectPolicy`][pydantic_ai.realtime.ReconnectPolicy] in
[`RealtimeModelSettings`][pydantic_ai.realtime.RealtimeModelSettings].

::: pydantic_ai.realtime.openai
