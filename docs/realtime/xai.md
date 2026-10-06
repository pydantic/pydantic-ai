---
description: "Connect a Pydantic AI agent to xAI Grok Voice with XaiRealtimeModel: setup, grok-voice model names, voices, turn detection, reasoning and session resumption."
---

# xAI Grok Voice

[`XaiRealtimeModel`][pydantic_ai.realtime.xai.XaiRealtimeModel] brings Grok Voice into the typed,
server-side realtime agent loop. Start with the [realtime quickstart](overview.md#quickstart) or the
[text-to-audio example](../examples/realtime-text-to-audio.md).

## Setup

To use Grok Voice, install `pydantic-ai-slim` with the `xai-realtime` optional group. Alongside
`xai-sdk`, the bundle includes the `openai` package, because Grok Voice's realtime API reuses the
OpenAI Realtime protocol's event types:

```bash
pip/uv-add "pydantic-ai-slim[xai-realtime]"
```

Set `XAI_API_KEY` as described in the [xAI model documentation](../models/xai.md#configuration).
Use `provider='xai'` or pass an
[`XaiProvider`][pydantic_ai.providers.xai.XaiProvider] with `api_key=`. Custom `api_host` is
unsupported, and a provider constructed with only `xai_client=` cannot open the WebSocket because
the connection requires the API key.

## Model names

Use a Grok Voice ID such as `grok-voice-latest` or a pinned `grok-voice-think-*` model.
`grok-voice-latest` follows xAI's current flagship and can change underneath an application; pin a
version when behavior must remain stable. Use the
[official xAI voice documentation](https://docs.x.ai/developers/model-capabilities/audio/voice-agent) for the canonical
model list.

## Settings

[`XaiRealtimeModelSettings`][pydantic_ai.realtime.xai.XaiRealtimeModelSettings] — the realtime
counterpart of [model run settings](../agent.md#model-run-settings) — extends the
[shared settings](overview.md#shared-settings):

```python
from pydantic_ai.realtime.xai import XaiRealtimeModel, XaiRealtimeModelSettings

settings = XaiRealtimeModelSettings(
    xai_voice='eve',
    turn_detection={'sensitivity': 'low'},
    input_transcription_model='auto',
)
model = XaiRealtimeModel('grok-voice-latest', settings=settings)
```

`xai_voice` selects the provider voice; when unset, xAI picks its own server-side default
(currently `eve`). For an exact server-VAD threshold, prefix padding, or silence duration, set
`xai_turn_detection=` with [`ServerVAD`][pydantic_ai.realtime.openai.ServerVAD]; it fully overrides
shared [`turn_detection`](turns.md#automatic-turn-detection). xAI echoes `create_response: False`
back but still responds when the user stops speaking, so use `turn_detection=False` for
[push-to-talk](turns.md#push-to-talk) instead.

xAI accepts the shared `max_tokens` and `parallel_tool_calls` settings but ignores them: replies
aren't capped, and the model can still call several tools at once.

[Input transcription](audio.md#input-transcription) defaults to `'auto'`. Unlike the incremental
deltas described in [live captions](audio.md#live-captions), xAI sends cumulative transcript
snapshots that can revise earlier words, so caption UIs should render the full
[`TranscriptUpdate.transcript`][pydantic_ai.realtime.TranscriptUpdate.transcript] rather than
append deltas.

### Reasoning

`grok-voice-latest` and `grok-voice-think-*` models support the shared
[`thinking`](../capabilities/thinking.md) setting. The provider exposes
only `'high'` and `'none'`: every enabled effort maps to `'high'`, while `False` maps to `'none'`.
Other Grok Voice models ignore the setting.

## Feature support and limitations

| Feature | Support | Notes |
| --- | --- | --- |
| Audio format | Limited parameter support | Mono PCM16 at 24 kHz by default, input and output; the other PCM rates xAI lists (8 to 48 kHz) by setting `audio_input_sample_rate` and `audio_output_sample_rate` through [`profile=`](overview.md#provider-support). The API also offers G.711 and Opus, which Pydantic AI does not expose |
| Text output | Unsupported | Grok Voice always produces audio |
| Image input | Unsupported | Grok Voice doesn't see images: xAI accepts an image item but ignores it, so [`send()`][pydantic_ai.realtime.RealtimeSession.send] rejects images with a `UserError` before sending |
| Manual turns | Full feature support | `turn_detection=False` plus [commit/create verbs](turns.md#push-to-talk) |
| Interruption | Limited parameter support | [`interrupt()`](turns.md#barge-in) works, and `played_bytes=` also discards the unheard audio; `played_ms` raises, because [xAI's truncation](#provider-specific-quirks) is unreliable at barge-in |
| Input transcription | Full feature support | [Dedicated provider path](audio.md#input-transcription); `'auto'` by default |
| Native tools | Unsupported | xAI offers web search, X search, file search, and MCP tools, which Pydantic AI does not expose yet; configure [local fallbacks](tools.md#native-tools) for web capabilities |
| Async tool calls | Full feature support | Grok Voice keeps talking, and answers the user, while a [tool runs](tools.md#concurrent-tool-execution) |
| Usage | Full feature support | Audio-token buckets and `billable_audio_seconds` in `RunUsage.details` |
| State-restoring reconnect | Full feature support | Native [resumption](#session-resumption) is automatic with a reconnect policy |

See [Audio, images, and transcripts](audio.md), [Turns and interruptions](turns.md),
[Tools](tools.md), and [Connection lifecycle](lifecycle.md) for the provider-agnostic workflows.

## Gateway

Grok Voice is not currently available through the [Pydantic AI Gateway](../gateway.md). Connect
through `provider='xai'` or an `XaiProvider`.

## Session resumption

With a [`ReconnectPolicy`][pydantic_ai.realtime.ReconnectPolicy], xAI automatically enables native
resumption for [state-restoring reconnects](lifecycle.md#state-restoration): it restores prior
turns and suppresses the provider's replay burst from the local event stream. The handle is xAI's
conversation ID, and xAI keeps the conversation for 30 minutes of inactivity, but Pydantic AI holds
the ID only in memory, so another process can't resume it yet
([#7302](https://github.com/pydantic/pydantic-ai/issues/7302)).

A conversation that runs past xAI's maximum duration ends with a `max_duration` error. Resuming it
would only run into the same limit, so the session raises a
[`RealtimeError`][pydantic_ai.realtime.RealtimeError] instead of reconnecting.

## Provider-specific quirks

- Grok Voice always speaks: its profile reports `supports_text_output=False`, so `output_modality='text'`
  raises a `UserError` before connecting. Read the answer from the transcript on the `SpeechPart`.
- xAI documents `conversation.item.truncate` and confirms it with `conversation.item.truncated`,
  but in our testing it only takes effect once the response has ended, and not reliably even then.
  Sent just before the cancel, as barge-in needs, it changes nothing, so the profile reports
  `supports_output_truncation=False`. `interrupt(played_bytes=...)` still discards the unheard audio
  and cancels the response, while a plain `interrupt()` only cancels. Either way xAI keeps everything
  generated up to the cancel, which can be more than the user heard.
- The protocol resembles OpenAI Realtime, but feature support comes from the xAI model profile;
  avoid assuming every OpenAI behavior is available.
