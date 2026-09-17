# ElevenLabs Agents

[`ElevenLabsRealtimeModel`][pydantic_ai.realtime.elevenlabs.ElevenLabsRealtimeModel] connects an
agent to the [ElevenLabs Agents platform](https://elevenlabs.io/docs/agents-platform/overview),
where the conversation runs on a hosted agent. Start with the
[realtime quickstart](overview.md#quickstart).

ElevenLabs offers no direct speech-to-speech model. Its Agents platform runs a cascaded pipeline
(realtime ASR, a configurable LLM, ElevenLabs TTS, and a server-side turn-taking model) behind a
pre-provisioned agent, and the agent WebSocket is the only conversational API. The model therefore
wraps a *hosted agent* addressed by its `agent_id` — Pydantic AI takes ownership of as much of
the conversation as ElevenLabs allows per conversation (see
[Configuration ownership](#configuration-ownership)).

## Setup

To use ElevenLabs agents, install `pydantic-ai-slim` with the `elevenlabs-realtime` optional group,
which bundles the realtime WebSocket transport — the REST calls around the WebSocket go through
Pydantic AI's own HTTP client, so no ElevenLabs SDK is required:

```bash
pip/uv-add "pydantic-ai-slim[elevenlabs-realtime]"
```

Set `ELEVENLABS_API_KEY`, or pass an
[`ElevenLabsProvider`][pydantic_ai.providers.elevenlabs.ElevenLabsProvider] with `api_key=`. The
key is required even for public agents, because the connect-time preflight authenticates with it.
The provider's `base_url` doubles as the region switch: pass a
[data-residency host](https://elevenlabs.io/docs/product-guides/administration/data-residency) such
as `https://api.eu.residency.elevenlabs.io` (the India and Singapore hosts follow the same pattern)
to keep both the REST calls and the WebSocket inside a region.

## Model names

Use the id of an agent created in the [ElevenLabs dashboard](https://elevenlabs.io/app/agents) or
through the ElevenLabs API, for example `agent_0101k2...`, or `elevenlabs:agent_0101k2...` as a
model string. The agent's own configuration decides the ASR setup, audio formats, turn-taking, and
which per-conversation overrides are permitted. Every
[`ModelResponse`][pydantic_ai.messages.ModelResponse] is stamped with the agent id as its
`model_name`; the LLM the agent runs on is part of the agent configuration and is not reported per
response.

## Configuration ownership

Pydantic AI is the source of truth wherever ElevenLabs allows a per-conversation override, and
fails loudly where it does not:

- **Instructions**: when the Pydantic AI agent defines instructions, the ElevenLabs agent must
  permit the *System prompt* override; otherwise the preflight raises a
  [`UserError`][pydantic_ai.exceptions.UserError] naming the exact toggle. Without local
  instructions, the ElevenLabs-side prompt is inherited silently.
- **Overrides**: first message, language, voice, TTS knobs, the pipeline LLM, and text-only mode
  are sent in the conversation-initiation override payload. Every override is gated by a per-field
  toggle under *Settings* > *Security* > *Overrides* in the agents dashboard
  (`platform_settings.overrides` via the API); the preflight checks each override setting against
  the agent's allowlist before dialing. The raw `elevenlabs_config_override` escape hatch is the one
  exception; see [Settings](#settings).
- **Tools**: client tools are workspace entities referenced by the agent, so they cannot be
  declared inline per conversation. By default the preflight errors when the session's tools and
  the agent's client tools differ in any way; see [Tool synchronization](#tool-synchronization).
- **Stays with the agent**: turn-taking and VAD, ASR configuration, audio formats, and platform
  settings (auth, privacy, guardrails, knowledge bases, workflows).

## Settings

[`ElevenLabsRealtimeModelSettings`][pydantic_ai.realtime.elevenlabs.ElevenLabsRealtimeModelSettings] —
the realtime counterpart of [model run settings](../agent.md#model-run-settings) — extends the
[shared settings](overview.md#shared-settings) with the agent's per-conversation overrides
(greeting, language, voice, TTS delivery, the pipeline LLM), dynamic variables, and the tool
synchronization mode:

```python
from pydantic_ai.realtime.elevenlabs import (
    ElevenLabsRealtimeModel,
    ElevenLabsRealtimeModelSettings,
)

settings = ElevenLabsRealtimeModelSettings(
    elevenlabs_voice_id='EXAVITQu4vr4xnSDxMaL',
    elevenlabs_language='de',
    elevenlabs_first_message='Hallo! Wie kann ich helfen?',
    elevenlabs_tts={'stability': 0.4, 'speed': 1.05},
)
model = ElevenLabsRealtimeModel('agent_0101k2example', settings=settings)
```

| Setting | Purpose |
| --- | --- |
| `elevenlabs_first_message`, `elevenlabs_language` | The greeting the agent opens with and the conversation language |
| `elevenlabs_voice_id`, `elevenlabs_tts` | The provider voice and the TTS delivery knobs (model, stability, speed, similarity boost) |
| `elevenlabs_llm` | The LLM the ElevenLabs pipeline runs, e.g. `gpt-5.2` |
| `elevenlabs_dynamic_variables` | Values substituted into `{{placeholders}}` in the agent's prompts and messages |
| `elevenlabs_custom_llm_extra_body` | Extra request body for an agent backed by a custom LLM; gated by its own security toggle |
| `elevenlabs_user_id` | Opaque end-user identifier for ElevenLabs-side analytics |
| `elevenlabs_tool_sync` | How the session's tools are reconciled with the agent's client tools; see [Tool synchronization](#tool-synchronization) |
| `elevenlabs_config_override` | Raw override keys merged last as a forward-compatibility escape hatch |

Each `elevenlabs_*` override setting requires its toggle on the agent, as described above.
`elevenlabs_config_override` is the one setting deliberately not checked against the permission
allowlist — being raw is its point — so a field the agent does not permit there surfaces as the
server closing the socket (code 1008, naming the field) after the handshake, not as a local
[`UserError`][pydantic_ai.exceptions.UserError].

Of the shared settings:

- `output_modality='text'` maps to the text-only override, toggle-gated like the rest.
- `tool_choice='none'` and allow-lists restrict the tool set the preflight checks or syncs. A call
  to any other tool is rejected on receipt with an error result, whatever the sync mode, so the
  agent's dashboard tools can never widen the run's tool set.
- `turn_detection` cannot be configured: ElevenLabs' server-side turn model is always on, so any
  value other than `True` raises. There is no [push-to-talk](turns.md#push-to-talk).
- `input_transcription_model=None` raises: ASR drives the agent pipeline and cannot be disabled.
- `reconnect` raises: conversations cannot be resumed, so a reconnect policy would silently start a
  conversation that remembers nothing.
- `max_tokens`, `parallel_tool_calls`, and `thinking` have no per-conversation surface and are
  ignored, matching the [shared settings contract](overview.md#shared-settings).

## Tool synchronization

`elevenlabs_tool_sync` controls how the session's
[`ToolDefinition`][pydantic_ai.tools.ToolDefinition]s are reconciled with the agent's client tools
at connect time:

- `'error'` (default): raise a [`UserError`][pydantic_ai.exceptions.UserError] describing every
  difference (missing, extra, or differing tools), so the two can never silently disagree.
- `'sync'`: make the workspace match Pydantic AI over REST before dialing: create missing client
  tools, update differing ones, and re-point the agent's `tool_ids`. This mutates workspace state
  shared by every conversation with the agent, which is why it is opt-in. Treat it as a
  deploy-time step — a warm-up connect after a tool change — rather than something a customer's
  first call should pay for: the writes run sequentially at roughly half a second each, and two
  processes syncing the same agent concurrently race on the final `tool_ids` re-point, last
  writer wins.
- `'off'`: trust the agent's configuration, for example with a read-scoped API key. Calls to tools
  the session does not advertise are still rejected on receipt.

Server-side webhook, MCP, and system tools on the agent are ElevenLabs-owned and never touched, and
tool executions the server runs itself are not surfaced as session tool calls. The agent's attached
tools are read from the resolved `prompt.tools` in its configuration; should ElevenLabs stop
returning them inline (the field is deprecated in favor of `tool_ids`), each attached id is fetched
from the workspace instead, so the comparison keeps working.

ElevenLabs stores tool parameters in its own restricted schema dialect rather than JSON Schema.
`type`, `description`, `enum`, `items`, `properties`, and `required` are supported; other keywords
(such as `additionalProperties` or numeric bounds) are not; and **every parameter needs a
description**. Nested models (`$defs`/`$ref`) are inlined, and optional (nullable) parameters are
collapsed onto their non-null type before conversion, with optionality carried by `required` alone.
The dialect has no way to say `null` — a required nullable parameter is advertised as its
non-null type, and the agent's LLM, which only ever sees the advertised schema, always supplies a
value for it. `'sync'` therefore fails loudly for a tool whose parameters lack descriptions (add
them in the function docstring) or use a shape the dialect cannot express (a union of types, or a
recursive reference). The `'error'` comparison checks what the dialect can express, comparing a
parameter's description only when the local schema declares one.

## Feature support and limitations

| Feature | Support | Notes |
| --- | --- | --- |
| Audio format | Full feature support | Mono PCM16; rates are fixed per agent (16 kHz default) and validated at connect, see [Audio formats](#audio-formats) |
| Text output | Limited parameter support | `output_modality='text'` maps to the text-only override, which needs its toggle on the agent |
| Image input | Unsupported | Audio/text input only |
| Manual turns | Unsupported | ElevenLabs' server-side turn model owns turn-taking; `turn_detection` must stay `True` |
| Interruption | Limited parameter support | Barge-in is detected server-side and reported as `RealtimeResponseInterruptedEvent`; there is no client `interrupt()` verb |
| Session seeding | Unsupported | The conversation WebSocket has no history-seeding channel |
| Input transcription | Full feature support | Always on: per-utterance final transcripts from the agent's ASR; `input_transcription_model=None` raises |
| Native tools | Unsupported | Server-side tools live on the agent, outside the session |
| Usage | Limited parameter support | Context tokens only, and only with `context_usage` in the agent's `client_events` (off by default); no output tokens or credits on the socket, cost appears post-hoc on the conversations API |
| Context window | Limited parameter support | `context_window` is `None` because the agent's LLM is configurable; pin it via `profile={'context_window': ...}`, or read the live value from `ElevenLabsRealtimeConnection.context_limit_tokens` once the agent reports `context_usage` |
| State-restoring reconnect | Unsupported | Conversations cannot be resumed; `reconnect` raises |

See [Audio, images, and transcripts](audio.md), [Turns and interruptions](turns.md),
[Tools](tools.md), and [Connection lifecycle](lifecycle.md) for the provider-agnostic workflows.

## Gateway

ElevenLabs is not currently available through the [Pydantic AI Gateway](../gateway.md). Connect
through `provider='elevenlabs'` or an `ElevenLabsProvider`.

## Audio formats

Audio formats are fixed per agent, not per conversation. The model profile defaults to ElevenLabs'
16 kHz PCM defaults; the connect handshake echoes the agent's actual formats, and the model raises a
[`RealtimeError`][pydantic_ai.realtime.RealtimeError] when they disagree with the profile. For an
agent configured differently (e.g. 24 kHz), pass the actual rates via `profile=`:

```python
from pydantic_ai.realtime.elevenlabs import ElevenLabsRealtimeModel

model = ElevenLabsRealtimeModel(
    'agent_0101k2example',
    profile={'audio_input_sample_rate': 24000, 'audio_output_sample_rate': 24000},
)
```

Telephony (`ulaw_8000`) agents are not supported.

## Provider-specific quirks

- In audio mode, the assistant's transcript arrives as one final
  [`OutputTranscript`][pydantic_ai.realtime.codec.OutputTranscript] per response, after the
  response's audio has streamed; in text-only mode the text additionally streams as incremental
  deltas ahead of it. Synthesis outruns playback, so a whole spoken response is typically delivered
  (audio included) while the user is still hearing its first words.
- After a barge-in, the server truncates the stored transcript itself and reports the corrected
  text, which the session keeps on the interrupted response's `provider_details` under
  `corrected_agent_response` when the response was still streaming. A barge-in during playback of
  an already-delivered response surfaces only as an interruption event: the session's history keeps
  the full generated text even though the user heard less of it.
- Usage reports LLM context consumption only (as `input_tokens`), and only when `context_usage` is
  in the agent's `conversation.client_events` list (it is not by default). Reports arrive after
  each turn completes and accumulate into the run total without attaching to a specific response.
  The same report names the LLM's context window; it is not summed into usage but kept on
  [`ElevenLabsRealtimeConnection.context_limit_tokens`][pydantic_ai.realtime.elevenlabs.ElevenLabsRealtimeConnection.context_limit_tokens].
  Look up conversation cost post-hoc via the ElevenLabs conversations API: every finalized
  [`ModelResponse`][pydantic_ai.messages.ModelResponse] carries the server-assigned id in
  `provider_details['conversation_id']`, so it survives into persisted history, and a consumer
  holding the live connection can read
  [`ElevenLabsRealtimeConnection.conversation_id`][pydantic_ai.realtime.elevenlabs.ElevenLabsRealtimeConnection.conversation_id]
  directly.
- Client tools whose ElevenLabs registration sets `expects_response: false` are fire-and-forget:
  the session still executes them, but their results are never sent back to the agent.
