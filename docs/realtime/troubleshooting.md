# Troubleshooting

Below are suggestions on how to fix some common problems with realtime sessions, each linking to
the page that covers the underlying behavior. For issues not listed here or addressed in the
documentation, see the general [troubleshooting page](../troubleshooting.md), ask in the
[Pydantic Slack](../help.md), or create an issue on
[GitHub](https://github.com/pydantic/pydantic-ai/issues).

## No audio, or no useful speech

Send mono PCM16 at `session.audio_input_sample_rate` and play it at
`session.audio_output_sample_rate`. Do not assume the rates match. See the
[audio wire contract](audio.md#audio-wire-contract).

## The model never responds

In push-to-talk mode, call `commit_audio()` and then `create_response()` after sending audio. See
[push-to-talk](turns.md#push-to-talk).

## The model interrupts itself

The microphone is probably hearing speaker output. Add echo cancellation in the device/WebRTC
layer and stop local playback on real [barge-in](turns.md#barge-in).

## Tools seem to stall

The local tool runs concurrently, but the provider may pause speech while awaiting the result. Show
tool lifecycle events and review [concurrent tool execution](tools.md#concurrent-tool-execution).

## A reconnect lost context

Inspect `RealtimeSessionReconnectEvent.state_restored`. If false, begin a fresh conversation; if
true but a current utterance vanished, that in-flight media was outside the restored completed-turn
history. See [state restoration](lifecycle.md#state-restoration).

## Gemini reaches its session limit

Set the `reconnect` setting to a [`ReconnectPolicy`][pydantic_ai.realtime.ReconnectPolicy]; Gemini
session resumption is enabled automatically alongside it. Recovery uses the latest in-memory server
handle after the drop. See [Gemini session resumption](gemini.md#session-resumption) and
[provider session limits](lifecycle.md#provider-session-limits).
