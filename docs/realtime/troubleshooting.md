---
description: "Fix common Pydantic AI realtime voice problems: no audio, a model that never replies or answers twice, echo cutoffs, stalled tools, lost reconnects."
---

# Realtime troubleshooting

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

## The model says the same thing twice

`send('...')` already asks the model to reply. Do not follow it with `create_response()`, which asks
for a second response. See [text turns](turns.md#text-turns).

## The model interrupts itself

The microphone is probably hearing speaker output. Add echo cancellation in the device/WebRTC
layer and stop local playback on real [barge-in](turns.md#barge-in).

## The greeting never plays, or the model replies twice when the visitor speaks

Speaker echo or microphone transients probably cancelled the greeting while the audio path opened.
Mute microphone capture until the greeting has played, while continuing to send digital silence so
server VAD can close any open speech segment; see [Muting the microphone](turns.md#muting-the-microphone)
and [Speaking first](turns.md#speaking-first).
To confirm the cause, iterate the event stream and look for a
[`RealtimeResponseInterruptedEvent`][pydantic_ai.realtime.RealtimeResponseInterruptedEvent] on the
first response and any
[`RealtimeSessionErrorEvent`][pydantic_ai.realtime.RealtimeSessionErrorEvent], or inspect the
[Logfire trace](observability.md#logfire-instrumentation).

## Tools seem to stall

The local tool runs concurrently, but the provider may pause speech while awaiting the result. Show
tool lifecycle events and review [concurrent tool execution](tools.md#concurrent-tool-execution).

## A reconnect lost context

Inspect `RealtimeSessionReconnectEvent.state_restored`. If false, the drop cut off a turn in flight:
the session closed any cut reply as interrupted and kept the earlier turns, so ask again or resend
the cut turn rather than starting over. On Gemini, a drop before the first resumption handle starts
a fresh session instead: the session's history keeps the earlier turns, but the model no longer has
them, so restate any context the next turn depends on. If true but a current utterance vanished,
that in-flight media was outside the restored completed-turn history. See
[state restoration](lifecycle.md#state-restoration).

## Gemini reaches its session limit

Set the `reconnect` setting to a [`ReconnectPolicy`][pydantic_ai.realtime.ReconnectPolicy]; Gemini
session resumption is enabled automatically alongside it. Recovery uses the latest in-memory server
handle after the drop. See [Gemini session resumption](gemini.md#session-resumption) and
[provider session limits](lifecycle.md#provider-session-limits).

## Azure Voice Live fails with "Only Azure voice is supported"

Cascade models such as `gpt-5` and `gpt-4.1` speak only through Azure text-to-speech voices, so
they reject `openai_voice`. Choose their voice with `azure_voice_live_voice` instead. See
[Voice Live voices](azure.md#voices).

## ElevenLabs Agents fails to connect

The connect-time preflight raises a [`UserError`][pydantic_ai.exceptions.UserError] naming what to change: an
override toggle the agent doesn't permit (see
[Configuration ownership](elevenlabs.md#configuration-ownership)), or a difference between the session's tools and
the agent's client tools (see [Tool synchronization](elevenlabs.md#tool-synchronization)). A bad API key or an
unknown agent id raises a [`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError] instead.
