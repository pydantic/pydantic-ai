# Realtime (speech-to-speech) sessions

For voice models that stream audio over a persistent connection (OpenAI Realtime, Azure OpenAI,
Gemini Live, xAI Grok Voice, ElevenLabs' Eleven Realtime, or ElevenLabs Agents), use
`agent.realtime().session()` instead of `run()`. It reuses the agent's tools and instructions and runs
the tool loop for you. Stream input with `send_audio`/`send`, and iterate the
session to consume the **same part/event vocabulary as a streamed run** — `PartStartEvent` /
`PartDeltaEvent` / `PartEndEvent` carrying `SpeechPart`s and `ToolCallPart`s, plus
`FunctionToolCallEvent` / `FunctionToolResultEvent`, plus realtime control events (`RealtimeInputSpeechStartEvent`,
`RealtimeInputSpeechEndEvent`, `RealtimeResponseInterruptedEvent`, ...). A part's `index` identifies it for the whole
session: when text of an output item's transcript arrives after a later item started, the session streams that part
again under the same `index`, so keep parts by `index`. Use `RealtimeTurnCompleteEvent` as the exchange boundary: it
comes when what `wait_for_reply()` waits for is over (every reply owed has come, tool rounds included, each from the
response that answers it: not one that was already under way when the input arrived). This is not
always the end of audible speech: on WebRTC sidebands, track playback with `RealtimeOutputSpeechStartEvent` and
`RealtimeOutputSpeechEndEvent`. On OpenAI Realtime and Azure OpenAI, a tool-heavy agent can set
`openai_prune_tool_items` to delete answered tool rounds from the server conversation, which otherwise bills every
later response for them (history keeps them). Before
passing raw microphone bytes to `send_audio`, convert them to mono PCM16 at `session.audio_input_sample_rate`; raw
chunks carry no sample-rate metadata.
After `RealtimeTurnCompleteEvent` (or a greeting's finalized `SpeechPart`), await
`session.wait_for_playback()` before closing the session or opening the microphone. It waits for the single
device-paced `stream_audio()` view to account for all audio emitted so far — played, discarded on a barge-in or a
full buffer, or emitted before the view subscribed; it requires exactly one audio view.
To align spoken words with playback (e.g. karaoke-style captions), set `output_speech_timestamps=True`: on a model
that reports them (Azure AI Voice Live with an Azure voice, or ElevenLabs Agents), `SpeechPartDelta.timestamps` streams `SpeechTimestamp`s
whose `start_ms` counts from the start of that part's audio, and `SpeechPart.timestamps` records them in history.
ElevenLabs Agents wraps an agent hosted on ElevenLabs, addressed as `elevenlabs-agents:<agent_id>`: the agent must
permit the conversation overrides the session sends (instructions, voice, ...) in its Security tab, and its client
tools must match the session's unless `elevenlabs_agents_tool_sync` is `'sync'` (which updates the agent) or `'off'`.
Eleven Realtime (`elevenlabs:eleven_realtime_v1_mini` / `elevenlabs:eleven_realtime_v1_max`) instead takes its
instructions, tools, and history from the agent, like OpenAI Realtime; set its voice with `elevenlabs_voice`.
OpenAI GPT-Live (`openai:gpt-live-1`) splits the agent across two models: the agent's instructions, system prompt
and tools go only to the Responses backend that does delegated work, never to the voice model the caller hears. Put
the persona, language and speaking style in `openai_live_instructions` (it replaces the default Live prompt, so say
when to delegate there too); left in the agent's instructions, the voice model never sees them (it runs on its
default prompt) and carries that style into delegated answers only in part.

```python {test="skip"}
import anyio

from pydantic_ai import Agent
from pydantic_ai.messages import (
    PartDeltaEvent,
    PartEndEvent,
    SpeechPart,
    SpeechPartDelta,
)
from pydantic_ai.realtime import RealtimeSessionErrorEvent, RealtimeTurnCompleteEvent
from pydantic_ai.realtime.openai import OpenAIRealtimeModelSettings

agent = Agent(instructions='You are a helpful voice assistant.')


async def main(microphone_chunk: bytes):
    settings = OpenAIRealtimeModelSettings(openai_voice='alloy', turn_detection=False)
    async with agent.realtime(
        'openai:gpt-realtime', model_settings=settings
    ).session() as session:
        # The chunk must already be mono PCM16 at `session.audio_input_sample_rate`.
        await session.send_audio(microphone_chunk)
        await session.commit_audio()
        await session.create_response()
        # Input transcription can finish after the model exchange. Give it a
        # bounded grace period so a missing transcript cannot hang the session.
        turn_complete = user_turn_complete = False
        with anyio.move_on_after(None) as transcript_wait:
            async for event in session:
                match event:
                    case PartDeltaEvent(delta=SpeechPartDelta(audio_chunk=chunk)) if chunk:
                        ...  # play audio out
                    case PartEndEvent(part=SpeechPart(speaker='user', transcript=t)):
                        if t is not None:
                            print('user said:', t)
                        user_turn_complete = True
                    case RealtimeTurnCompleteEvent():
                        turn_complete = True
                        transcript_wait.deadline = anyio.current_time() + 1
                    case RealtimeSessionErrorEvent(message=message, recoverable=True):
                        # The connection remains usable, but this turn may not complete.
                        raise RuntimeError(message)
                if turn_complete and user_turn_complete:
                    break

    # A session builds ordinary ModelMessage history: hand it off to a text agent.
    notes = Agent('openai:gpt-5.2', instructions='Summarize.')
    await notes.run(message_history=session.all_messages())
```

Key facts for building realtime agents:

- **A string sent with `session.send()` solicits a response**: use `respond=False` to add passive
  text context. Images are context-only by default; use `respond=True` to ask for a response to an
  image. Never pair `session.send('...')` with `session.create_response()`, because that asks twice.
  A string sent during a reply queues on OpenAI/Azure/xAI/Eleven Realtime and Gemini 2.5, but interrupts the active
  reply on Gemini 3.1. On OpenAI GPT-Live a string goes to the delegated backend as the user's message and
  runs it (tools included), and Live speaks the answer, often after a filler line; `respond=False` is context only
  (which Live may still mention), Live only speaks while audio is flowing (set `openai_live_idle_audio=True` for a
  session with no microphone), and `respond=False` text over 500 tokens raises `UserError` (a longer typed turn still runs the backend, but Live isn't told what was typed). Gemini speech models reject text output before connect, except the Vertex
  `gemini-live-2.5-flash` half-cascade, which answers in text.
- **Application notes mid-session are `session.send(SystemPromptPart(...), respond=False)`**: not a string,
  which the model takes as the user's words. It goes out on the provider's operator channel (a `system` message on
  OpenAI/Azure, a contextual update on ElevenLabs Agents, a note to GPT-Live's voice model, `<system>`-tagged
  user text elsewhere), is recorded as a `SystemPromptPart`, and with the default `respond=True` gets a reply.
- **History handoff is the marquee integration**: `session.all_messages()` / `session.new_messages()`
  return real `ModelMessage`s; seed with `realtime(model, message_history=...).session()`, or with
  `conversation=result.conversation` to carry the running usage and `conversation_id` too. Transcripts
  stay attached to the user turn they describe even when they arrive after its response, and a turn
  started while the model is still answering (barge-in) is recorded after that answer. A reported
  speech segment whose transcript never arrives remains represented by retained audio or a content-less
  `SpeechPart` when the session closes. Transcripts are what carry over; a model whose profile sets
  `supports_seeding_audio` can also replay retained transcript-less *user* audio recorded at its input
  rate, and assistant audio is never replayed. Streamed images all reach the provider, but
  history keeps a sampled (`retain_images_every_n`) and bounded (`retain_images_max`, default `100`,
  oldest evicted first) record. Audio kept by `audio_retention` is bounded too (`retain_audio_max_seconds`,
  default `1800`, oldest evicted first, transcripts kept).
- **Usage and cost**: each recorded `ModelResponse` carries its response usage, while `session.usage`
  is cumulative; priced models get a `genai-prices` cost and enforce `UsageLimits.cost_limit`.
- **Context window**: `session.context_window_used` (and `ctx.context_window_used` in a session's tools)
  is the fraction in use: reported by OpenAI GPT-Live, computed from the latest response's tokens on
  OpenAI/Azure/Gemini, taken from the agent's `context_usage` reports on ElevenLabs Agents (`None` until the
  first), and `None` on xAI and Eleven Realtime. It can drop after server-side compaction or truncation, which
  no provider announces; tune it with `openai_truncation` (OpenAI Realtime and Azure, not GPT-Live) or
  `google_context_compression` (Gemini).
- **No `output_type`**: realtime models don't do structured output. Delegate hard work to a text
  agent behind a tool, or hand off history afterwards.
- **Check the model profile before calling profile-gated methods**: `model.profile` (a
  `RealtimeModelProfile`, the realtime counterpart to `ModelProfile`) reports
  `supports_manual_turn_control`, `supports_interruption`, `supports_image_input`,
  `supports_output_truncation`, and `supports_session_seeding`. OpenAI Realtime and Azure OpenAI
  support all of these; OpenAI GPT-Live supports only `supports_session_seeding` (from text) and
  `supports_image_input` with `image_input_requires_response` (an image goes to its backend, sent with
  `respond=True`), since it owns turn-taking; Gemini Live lacks `supports_manual_turn_control`,
  `supports_interruption`, and `supports_output_truncation` (automatic VAD only); ElevenLabs Agents supports none of
  them, nor images or seeding, since the hosted agent owns turn-taking; Eleven Realtime supports all of them but
  images (and seeds text only). Calling an unsupported method raises `UserError` up front.
- **Turn detection**: use the shared `TurnDetection` setting for sensitivity, prefix padding, and
  silence duration across providers (Eleven Realtime maps it onto semantic VAD, the only kind it has). Use
  `openai_turn_detection`, `xai_turn_detection`, or
  `google_vad` only for finer provider-specific control; when present, they fully override the shared
  setting. Automatic detection is on by default (`True`); set `turn_detection=False` for push-to-talk
  (OpenAI/Azure/xAI/Eleven Realtime only — Gemini and ElevenLabs Agents have no manual turn controls and raise).
- **Barge-in** (the user speaking over the model): pass `handle_barge_in=True` to `.session()` and
  the session owns the local half — flushing the audio the user will never hear, truncating the
  provider's transcript to what was played, and adding a client cancel only on providers whose own
  turn detection isn't already cancelling. Off by default, and it needs playback to drain a single
  device-paced `stream_audio()` iterator (the position it tracks); with none or several it stands
  down. To keep the trigger yourself, `session.interrupt(played_bytes=session.played_audio_bytes)`
  gets the same treatment on your own signal. A playback layer that buffers ahead of the device
  makes `played_audio_bytes` read too far: count real device consumption and pass `played_ms`.
  A reply cut off this way is recorded as interrupted with `SpeechPart.interrupted_at_ms`, even when it
  had already finished generating (the usual case): a later snapshot holds an interrupted copy at the
  same position, so persist the latest `all_messages()` rather than appending only what's past what you
  stored. Its transcript stays what the model generated (except on ElevenLabs Agents, which reports what was heard:
  the transcript becomes that, or `None` for a message never heard, and the generated text moves to the part's
  `provider_details['original_transcript']`); text handoff and seeding add an interruption note.
- **Mute with server VAD**: keep sending zero-valued PCM16 frames at the normal cadence. Sending
  nothing can leave an open speech segment open; pure tones do not reliably trigger speech VAD.
  Under manual turn control, stop sending and call `clear_audio()` instead.
- **Tools**: every tool runs in the background, so a slow tool never blocks the session. Whether
  the model keeps speaking and answering meanwhile is the profile's `async_tool_call_mode`:
  `'always'` (OpenAI, Azure, GPT-Live, xAI, `gemini-3.8-live-extended-thinking`), `'optional'`
  (Gemini native-audio and `gemini-3.8-live`, on only with the shared `async_tool_calls=True`
  setting, which the other models ignore), or `'never'` (other Gemini Live models, Eleven Realtime, ElevenLabs Agents). Don't use the
  deprecated `google_async_tool_calls` setting or `supports_async_tool_calls` profile flag.
  `gemini-3.8-live-extended-thinking` has no blocking mode and reasons in the background —
  it speaks a filler, runs the tool, and speaks again inside one exchange, so read
  `RealtimeTurnCompleteEvent` or await `session.wait_for_reply()` rather than watching each response
  to know it's done. An unhandled tool
  exception is raised from session iteration while it is active; otherwise it ends `stream_audio()` and
  `stream_transcripts()` and is raised when the session context closes. The next outbound method
  raises an already-ended receive side's failure instead, and every failure is delivered only once.
  Its call is recorded with `outcome='failed'`, leaving history valid for a standard-agent handoff.
  The model gets the generic result `{"error":"The tool raised an unhandled error."}` (never the
  exception's message) and answers it, so `wait_for_reply()` waits for that answer; after a
  `UsageLimitExceeded` nothing is sent and the exchange ends. To give the model something better than
  the generic failure, raise `ToolFailed` from the tool, or use an `on_tool_execute_error` capability
  to return a replacement result or raise `ModelRetry`. To end the call from a tool, await `ctx.realtime_session.hang_up()` for a clean
  hang-up (the tool does not resume, its call is recorded as interrupted, and a concurrent
  `send_audio()` async iterable returns cleanly at its next chunk), or call `ctx.cancel()` to make
  the session context raise `RunCancelled`. A watchdog can also await `session.close()` safely:
  cancelling the watchdog does not interrupt teardown, and the session context waits for teardown
  before exiting. While iteration is running the loop ends cleanly and `session.result` is settled.
- **Approval**: approval-gated and deferred tools are resolved inline by a `HandleDeferredToolCalls`
  handler (and refused without one); as in a run, `DeferredToolRequestsEvent` is emitted before the
  handler runs, and `DeferredToolResultsEvent` once it has resolved the call.
- **Late event consumption is bounded**: while nothing is iterating the session, it retains only the
  most recent 512 `PartDeltaEvent`s and the most recent 512 structural events, so a long call that
  nobody iterates cannot grow without bound. Parts are dropped whole, so a late iterator never sees a
  delta without its `PartStartEvent`. A parked failure is always retained. An active
  `async for event in session` remains lossless.
- **Browser WebRTC (OpenAI and Azure OpenAI)**: for browser voice agents, relay the browser's SDP
  offer server-side with `agent.realtime(model).answer_webrtc_offer(sdp_offer)` — the agent's
  resolved instructions and tools are baked in and the API key stays on the server — then attach a
  control-plane **sideband** with `.session(provider_session=answer.session)`. The browser owns the
  audio; the sideband session runs tools and builds history (its audio methods raise, and
  `audio_retention` must stay `'transcript_only'`). Closing the sideband only detaches it: call
  `session.hang_up()` (or `agent.realtime(model).hang_up(answer.session)`) to end the browser's call (OpenAI and Azure OpenAI).
- **Browser WebSocket relays**: `handle_barge_in=True` cannot know browser playback position because
  forwarded chunks count as played. Have the browser report real playback and pass it to
  `interrupt(played_bytes=...)`; `played_ms=` does not flush session-queued audio.

See the [Realtime guide](https://pydantic.dev/docs/ai/realtime/overview/) for the full walkthrough.
