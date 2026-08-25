# Events

Iterating a [`RealtimeSession`][pydantic_ai.realtime.RealtimeSession] yields the session's event
stream: content parts, tool activity, turn boundaries, reconnects, and recoverable errors. The
high-level [`stream_audio()`][pydantic_ai.realtime.RealtimeSession.stream_audio] and
[`stream_transcripts()`][pydantic_ai.realtime.RealtimeSession.stream_transcripts] views described in
[Audio, images, and transcripts](audio.md) are derived from this same stream, so most applications
iterate the session for control flow and leave media to the views.

## Event reference

| Event | Meaning |
| --- | --- |
| [`PartStartEvent`][pydantic_ai.messages.PartStartEvent] | A speech, text, or tool part started. |
| [`PartDeltaEvent`][pydantic_ai.messages.PartDeltaEvent] | Incremental speech audio/transcript or text content. |
| [`PartEndEvent`][pydantic_ai.messages.PartEndEvent] | A finalized part; retained speech audio appears here, not at part start. |
| [`FunctionToolCallEvent`][pydantic_ai.messages.FunctionToolCallEvent] | A local function tool began executing. |
| [`FunctionToolResultEvent`][pydantic_ai.messages.FunctionToolResultEvent] | A local function tool completed or returned a retry prompt. |
| [`DeferredToolRequestsEvent`][pydantic_ai.messages.DeferredToolRequestsEvent] | An inline capability handler resolved deferred requests. |
| [`DeferredToolResultsEvent`][pydantic_ai.messages.DeferredToolResultsEvent] | Inline deferred results are ready for normal tool processing. |
| [`RealtimeInputSpeechStartEvent`][pydantic_ai.realtime.RealtimeInputSpeechStartEvent] | The provider detected that the user started speaking, when the profile declares [`emits_input_speech_events`][pydantic_ai.realtime.RealtimeModelProfile.emits_input_speech_events]. |
| [`RealtimeInputSpeechEndEvent`][pydantic_ai.realtime.RealtimeInputSpeechEndEvent] | The provider detected the end of user speech, when the profile declares [`emits_input_speech_events`][pydantic_ai.realtime.RealtimeModelProfile.emits_input_speech_events]. |
| [`RealtimeResponseInterruptedEvent`][pydantic_ai.realtime.RealtimeResponseInterruptedEvent] | The provider reported an interrupted model response. |
| [`RealtimeInputTranscriptionErrorEvent`][pydantic_ai.realtime.RealtimeInputTranscriptionErrorEvent] | One user turn could not be transcribed; the session remains usable. |
| [`RealtimeOutputSpeechStartEvent`][pydantic_ai.realtime.RealtimeOutputSpeechStartEvent] / [`RealtimeOutputSpeechEndEvent`][pydantic_ai.realtime.RealtimeOutputSpeechEndEvent] | The model became, or stopped being, audible. These are emitted on a [WebRTC sideband](deployment.md#browser-webrtc-server-sideband), where the provider owns audio playback. |
| [`RealtimeTurnCompleteEvent`][pydantic_ai.realtime.RealtimeTurnCompleteEvent] | The model finished replying and no tool remains active. |
| [`RealtimeSessionReconnectEvent`][pydantic_ai.realtime.RealtimeSessionReconnectEvent] | The connection was automatically re-established. |
| [`RealtimeSessionErrorEvent`][pydantic_ai.realtime.RealtimeSessionErrorEvent] | A recoverable provider error occurred; the session remains usable. |

## Shared and realtime-only events

The first seven rows are [`AgentStreamEvent`][pydantic_ai.messages.AgentStreamEvent] members from
[`pydantic_ai.messages`][pydantic_ai.messages] — the same events a
[standard streamed run](../agent.md#streaming-all-events) yields, so event-handling code written for
a text agent (rendering parts, logging tool calls) works on a session unchanged. The `Realtime*`
rows are [`RealtimeEvent`][pydantic_ai.realtime.RealtimeEvent] members that only a session emits:
speech detection, interruption, turn completion, reconnection, and recoverable errors have no
equivalent in a request-response run.

A capability's [event stream hooks](../hooks.md#event-stream-hooks) see both kinds flow through the
same stream; see [Capabilities and hooks](capabilities.md#the-event-stream).

## The turn boundary

Use [`RealtimeTurnCompleteEvent`][pydantic_ai.realtime.RealtimeTurnCompleteEvent] as the exchange
boundary. A model can speak, call a tool, and speak again, so receiving speech — or a tool result —
does not imply that the turn is done.

## Reading raw audio events

The audio stream *is* these events under the hood:
[`stream_audio()`][pydantic_ai.realtime.RealtimeSession.stream_audio] is a bounded view over the
speech part deltas, and most applications should use it. As an advanced alternative, play
[`SpeechPartDelta.audio_chunk`][pydantic_ai.messages.SpeechPartDelta.audio_chunk] from raw
[`PartDeltaEvent`][pydantic_ai.messages.PartDeltaEvent]s. Model audio arrives in full whether or not
history retention is enabled. When output audio is retained, the final
[`SpeechPart`][pydantic_ai.messages.SpeechPart] contains the whole turn again as a WAV snapshot for
[history](history.md#retaining-audio); do not play both or the turn will play twice.
