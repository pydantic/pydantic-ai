# `pydantic_ai.realtime.openai_live`

The OpenAI GPT-Live API provider. Requires the `openai-realtime` optional group
(`pip install "pydantic-ai-slim[openai-realtime]"`), which floors `openai` at 3.12, the release that
added Live's event types.

GPT-Live is a different protocol from the [OpenAI Realtime API](openai.md), not a model served by it,
so [`OpenAILiveModel`][pydantic_ai.realtime.openai_live.OpenAILiveModel] shares no event mapping with
[`OpenAIRealtimeModel`][pydantic_ai.realtime.openai.OpenAIRealtimeModel]; the model name is what picks
between them. The Live model runs the spoken conversation and delegates the work to a backend model
configured through
[`OpenAILiveResponsesDelegation`][pydantic_ai.realtime.openai_live.OpenAILiveResponsesDelegation] in
[`OpenAILiveModelSettings`][pydantic_ai.realtime.openai_live.OpenAILiveModelSettings], which receives the
agent's instructions and tools and calls them as ordinary
[`ToolCall`][pydantic_ai.realtime.codec.ToolCall]s.

Live owns turn-taking entirely, so there is no manual turn control, interruption, or truncation, and
no turn-detection setting. It sends no end-of-response frame either: the connection synthesizes
[`ResponseDone`][pydantic_ai.realtime.codec.ResponseDone] after `openai_live_turn_silence_ms` of
silence, and the profile reports `synthesizes_turn_boundary=True`. Text is delivered as context
rather than as a user turn, seeding is text-only, and usage is reported as billable audio seconds
instead of tokens. Authentication comes from an
[`OpenAIProvider`][pydantic_ai.providers.openai.OpenAIProvider]; Azure OpenAI does not serve Live. See
the [GPT-Live documentation](../../realtime/openai-live.md) for the full story.

::: pydantic_ai.realtime.openai_live
