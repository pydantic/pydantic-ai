# `pydantic_ai.realtime.codec`

The lower-level *codec* vocabulary, for implementing a realtime provider or consuming a
[`RealtimeConnection`][pydantic_ai.realtime.codec.RealtimeConnection] directly: the raw events a
connection yields, the turn-control verbs and inputs it accepts, and the model-profile merge helpers.
Most users only need the session-level API in [`pydantic_ai.realtime`](../realtime.md).

::: pydantic_ai.realtime.codec
