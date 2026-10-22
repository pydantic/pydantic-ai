---
description: "Connect a Pydantic AI agent to ElevenLabs: Eleven Realtime speech-to-speech models with ElevenLabsRealtimeModel, or a hosted ElevenLabs agent with ElevenLabsAgentsRealtimeModel."
---

# ElevenLabs

Pydantic AI connects to [ElevenLabs](https://elevenlabs.io) in two ways:

- [ElevenLabs Agents](#elevenlabs-agents) runs the conversation on an agent hosted and configured on
  [ElevenAgents](https://elevenlabs.io/docs/eleven-agents/overview), ElevenLabs' agent platform (its voice, LLM,
  knowledge base, and tools), which Pydantic AI takes over per conversation.
- [Eleven Realtime](#eleven-realtime) runs ElevenLabs' speech-to-speech models directly, with the instructions,
  tools, and history coming from your Pydantic AI agent, as with every other
  [realtime provider](overview.md#provider-support).

The difference is where the agent is defined. Use Eleven Realtime unless you need the ElevenAgents platform: your
Pydantic AI agent defines the conversation, as it does with any other realtime provider, and a reconnect replays
the history.

Use ElevenLabs Agents when the agent lives on the ElevenAgents platform, for its knowledge bases, workflows, and
guardrails, or to run a cascaded pipeline with an LLM of your choice. Pydantic AI then overrides the agent's
instructions and settings per conversation, and by default checks its client tools against your agent's tools
(or [syncs them](#tool-synchronization) to match; see [Configuration ownership](#configuration-ownership)). An
ElevenAgents agent can also run on an Eleven Realtime model, in [Realtime voice mode](#configuration-ownership):
that's configured on the agent, and you connect to it through ElevenLabs Agents.

## ElevenLabs Agents

Use an `elevenlabs-agents:<agent_id>` model string, or
[`ElevenLabsAgentsRealtimeModel`][pydantic_ai.realtime.elevenlabs_agents.ElevenLabsAgentsRealtimeModel] for
model-level configuration, to connect an agent to the
[ElevenAgents](https://elevenlabs.io/docs/eleven-agents/overview) platform, where the conversation runs on a
hosted agent. Start with the [realtime quickstart](overview.md#quickstart).

ElevenLabs Agents runs a cascaded pipeline (realtime ASR, a configurable LLM, ElevenLabs TTS, and
a server-side turn-taking model) behind a pre-provisioned agent, which the conversation reaches
over the agent WebSocket. The model wraps that *hosted agent*, addressed by its `agent_id`, and
Pydantic AI takes ownership of as much of the conversation as ElevenLabs allows per conversation
(see [Configuration ownership](#configuration-ownership)).

### Setup

To use ElevenLabs Agents, install `pydantic-ai-slim` with the `elevenlabs-agents-realtime` optional group,
which bundles the realtime WebSocket transport:

```bash
pip/uv-add "pydantic-ai-slim[elevenlabs-agents-realtime]"
```

Set `ELEVENLABS_API_KEY`, or pass an
[`ElevenLabsProvider`][pydantic_ai.providers.elevenlabs.ElevenLabsProvider] with `api_key=`. The
key is required even for public agents, because the connect-time preflight authenticates with it.
The provider's `base_url` doubles as the region switch: pass a
[data-residency host](https://elevenlabs.io/docs/product-guides/administration/data-residency) such
as `https://api.eu.residency.elevenlabs.io` (or the India and Singapore hosts,
`api.in.residency.elevenlabs.io` and `api.sg.residency.elevenlabs.io`) to keep both the REST calls and the
WebSocket inside a region.

Connecting starts with a REST preflight that fetches the agent: a bad key or an unknown agent id raises a
[`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError] from it, and an API that can't be reached raises a
[`RealtimeError`][pydantic_ai.realtime.RealtimeError]. With `elevenlabs_agents_tool_sync` set to `'error'` (the
default) or `'off'`, the key only reads; `'sync'` also needs write access to agents and tools.

### Before you connect

1. Create the agent in the [ElevenLabs dashboard](https://elevenlabs.io/app/agents) or through the ElevenLabs API,
   and copy its id.
2. Under *Security* > *Overrides* in the agent's settings, turn on *System prompt* if the Pydantic AI agent has
   instructions, plus the toggle of each override setting you use (see [Settings](#settings)).
3. Under *Advanced* > *Client events*, add `context_usage`, which reports context usage and marks where a spoken
   response ends (see [Usage and cost](#usage-and-cost)).
4. Pick a [tool synchronization](#tool-synchronization) mode: with the default, `'error'`, a difference between the
   session's tools and the agent's client tools fails the connect.

This agent keeps the ElevenLabs agent's client tools in sync with its own:

```python {title="elevenlabs_agents_tools.py"}
from pydantic_ai import Agent
from pydantic_ai.realtime.elevenlabs_agents import ElevenLabsAgentsRealtimeModelSettings

agent = Agent(instructions='You are a friendly concierge. Keep answers short.')


@agent.tool_plain
async def find_restaurant(cuisine: str, party_size: int) -> str:
    """Find a restaurant with a free table.

    Args:
        cuisine: The kind of food the guests want, e.g. 'Italian'.
        party_size: How many guests need a seat.
    """
    return f'Trattoria Roma has a table for {party_size} ({cuisine}).'


realtime_agent = agent.realtime(
    'elevenlabs-agents:agent_0101k2example',
    model_settings=ElevenLabsAgentsRealtimeModelSettings(elevenlabs_agents_tool_sync='sync'),
)
```

ElevenLabs needs a description for every tool parameter, which the `Args:` section of the docstring provides.

### Model names

Use `elevenlabs-agents:<agent_id>` as a model string, for example `elevenlabs-agents:agent_0101k2...`, or pass the
agent id, `agent_0101k2...`, to `ElevenLabsAgentsRealtimeModel`. The agent's own configuration decides the ASR setup, audio formats, turn-taking, and
which per-conversation overrides are permitted. Every
[`ModelResponse`][pydantic_ai.messages.ModelResponse] is stamped with the agent id as its
`model_name`; the LLM the agent runs on is part of the agent configuration and is not reported per
response.

That LLM doesn't always answer. When it errors, returns nothing or an invalid tool call, or goes quiet for longer
than the agent's `prompt.cascade_timeout_seconds` (4 seconds by default; in [Realtime voice mode](https://elevenlabs.io/docs/eleven-agents/customization/realtime-models), ElevenLabs sets this wait itself), ElevenLabs answers the turn with a backup
LLM from its fallback cascade. The response's `model_name` is still the agent id: only a `context_usage` report
(see [Usage and cost](#usage-and-cost)) names the LLM, and it isn't tied to a response. The conversations API reports
the answering LLM afterwards, per message, as `producing_llm`. To turn the cascade off, set the agent's `prompt.backup_llm_config` to `{"preference": "disabled"}`.
To choose the backup LLMs, set it to `{"preference": "override", "order": [...]}`.

### Configuration ownership

Pydantic AI is the source of truth wherever ElevenLabs allows a per-conversation override, and
fails loudly where it does not:

- **Instructions**: when the Pydantic AI agent defines instructions, the ElevenLabs agent must
  permit the *System prompt* override; otherwise the preflight raises a
  [`UserError`][pydantic_ai.exceptions.UserError] naming the exact toggle. Without local
  instructions, the ElevenLabs-side prompt is inherited silently.
- **Overrides**: first message, language, voice, TTS knobs, the pipeline LLM, and text-only mode
  are sent in the conversation-initiation override payload. Every override is gated by a per-field
  toggle under *Security* > *Overrides* in the agent's settings
  (`platform_settings.overrides` via the API); the preflight checks each override setting against
  the agent's allowlist before dialing. The raw `elevenlabs_agents_config_override` escape hatch is the one
  exception; see [Settings](#settings).
- **Tools**: client tools are workspace entities referenced by the agent, so they cannot be
  declared inline per conversation. By default the preflight errors when the session's tools and
  the agent's client tools differ in any way; see [Tool synchronization](#tool-synchronization).
- **Stays with the agent**: turn-taking and VAD, ASR configuration, audio formats, the voice mode, and platform
  settings (auth, privacy, guardrails, knowledge bases, workflows).
- **Realtime voice mode**: an agent set to a
  [Realtime model](https://elevenlabs.io/docs/eleven-agents/customization/realtime-models) (its
  `conversation_config.realtime_model`, which can't be overridden per conversation) has ElevenLabs pick the LLM and
  TTS model. It would ignore `elevenlabs_agents_llm` and an `elevenlabs_agents_tts` `model_id`, so the preflight
  raises a [`UserError`][pydantic_ai.exceptions.UserError] for either. Every other override works as in Custom mode.

### Settings

[`ElevenLabsAgentsRealtimeModelSettings`][pydantic_ai.realtime.elevenlabs_agents.ElevenLabsAgentsRealtimeModelSettings] —
the realtime counterpart of [model run settings](../agent.md#model-run-settings) — extends the
[shared settings](overview.md#shared-settings) with the agent's per-conversation overrides
(greeting, language, voice, TTS delivery, the pipeline LLM), dynamic variables, and the tool
synchronization mode:

```python
from pydantic_ai.realtime.elevenlabs_agents import (
    ElevenLabsAgentsRealtimeModel,
    ElevenLabsAgentsRealtimeModelSettings,
)

settings = ElevenLabsAgentsRealtimeModelSettings(
    elevenlabs_agents_voice_id='EXAVITQu4vr4xnSDxMaL',
    elevenlabs_agents_language='de',
    elevenlabs_agents_first_message='Hallo! Wie kann ich helfen?',
    elevenlabs_agents_tts={'stability': 0.4, 'speed': 1.05},
)
model = ElevenLabsAgentsRealtimeModel('agent_0101k2example', settings=settings)
```

| Setting | Purpose |
| --- | --- |
| `elevenlabs_agents_first_message`, `elevenlabs_agents_language` | The greeting the agent opens with and the conversation language |
| `elevenlabs_agents_voice_id`, `elevenlabs_agents_tts` | The provider voice and the TTS delivery knobs (model, stability, speed, similarity boost) |
| `elevenlabs_agents_llm` | The LLM the ElevenLabs pipeline runs, by its ElevenLabs model id |
| `elevenlabs_agents_dynamic_variables` | Values substituted into `{{placeholders}}` in the agent's prompts and messages |
| `elevenlabs_agents_custom_llm_extra_body` | Extra request body for an agent backed by a custom LLM; gated by its own security toggle |
| `elevenlabs_agents_user_id` | Opaque end-user identifier for ElevenLabs-side analytics |
| `elevenlabs_agents_strip_audio_tags` | Whether expressive audio tags such as `[laughs]` are removed from the agent's transcripts (`True` by default); see [Provider-specific quirks](#provider-specific-quirks) |
| `elevenlabs_agents_tool_sync` | How the session's tools are reconciled with the agent's client tools; see [Tool synchronization](#tool-synchronization) |
| `elevenlabs_agents_config_override` | Raw override keys merged last as a forward-compatibility escape hatch |

Each `elevenlabs_agents_*` override setting requires its toggle on the agent, as described above;
`elevenlabs_agents_dynamic_variables`, `elevenlabs_agents_user_id`, `elevenlabs_agents_strip_audio_tags`, and
`elevenlabs_agents_tool_sync` need none.
`elevenlabs_agents_config_override` is the one setting deliberately not checked against the permission
allowlist — being raw is its point — so a field the agent does not permit there surfaces as the
server closing the socket (code 1008, naming the field) after the handshake, not as a local
[`UserError`][pydantic_ai.exceptions.UserError].

Of the shared settings:

- `output_modality='text'` maps to the text-only override, toggle-gated like the rest.
- `tool_choice='none'` and [`ToolOrOutput`][pydantic_ai.settings.ToolOrOutput] allow-lists restrict
  the conversation to those tools through the agent's *Tool IDs* override, toggle-gated like the
  rest: the agent keeps every tool of the run attached, and the conversation leaves the others out.
  A choice that forces a tool call (`'required'`, or a list of tool names) raises, as on the other
  providers.
- `turn_detection` cannot be configured: ElevenLabs' server-side turn model is always on, so any
  value other than `True` raises. There is no [push-to-talk](turns.md#push-to-talk).
- `input_transcription_model=None` raises: ASR drives the agent pipeline and cannot be disabled.
- `reconnect` raises: a dropped connection ends the session, and a reconnect policy would silently start a
  conversation that remembers nothing.
- `max_tokens`, `parallel_tool_calls`, `async_tool_calls`, and `thinking` have no per-conversation
  surface and are ignored, matching the [shared settings contract](overview.md#shared-settings).

### Audio formats

Audio formats are fixed per agent, not per conversation. The model profile defaults to ElevenLabs'
16 kHz PCM defaults; the connect handshake echoes the agent's actual formats, and the model raises a
[`RealtimeError`][pydantic_ai.realtime.RealtimeError] when they disagree with the profile. For an
agent configured differently (e.g. 24 kHz), pass the actual rates via `profile=`:

```python
from pydantic_ai.realtime.elevenlabs_agents import ElevenLabsAgentsRealtimeModel

model = ElevenLabsAgentsRealtimeModel(
    'agent_0101k2example',
    profile={'audio_input_sample_rate': 24000, 'audio_output_sample_rate': 24000},
)
```

Telephony (`ulaw_8000`) agents are not supported.

### Tool synchronization

`elevenlabs_agents_tool_sync` controls how the session's
[`ToolDefinition`][pydantic_ai.tools.ToolDefinition]s are reconciled with the agent's client tools
at connect time:

- `'error'` (default): raise a [`UserError`][pydantic_ai.exceptions.UserError] describing every
  difference (missing, extra, or differing tools), so the two can never silently disagree.
- `'sync'`: make the workspace match Pydantic AI over REST before dialing: create missing client
  tools, update differing ones, and re-point the agent's `tool_ids`. A differing tool that another
  agent also uses isn't changed under it: this agent gets a copy of its own. This mutates workspace
  state shared by every conversation with the agent, which is why it is opt-in. Treat it as a
  deploy-time step — a warm-up connect after a tool change — rather than something a customer's
  first call should pay for: the writes run one after another, and two processes syncing the same
  agent concurrently race on the final `tool_ids` re-point, last writer wins. Synced tools get
  ElevenLabs' default 20-second response timeout (`response_timeout_secs`); a tool that already has a
  timeout of its own keeps it when it is updated.
- `'off'`: trust the agent's own tool definitions, e.g. for client tools set up on the ElevenLabs side
  with settings Pydantic AI doesn't sync. A call to a tool the session does not define reaches the tool
  manager like any other call and is answered with the standard unknown-tool error, which lands in the
  session's history.

Server-side webhook, MCP, and system tools on the agent are ElevenLabs-owned and never touched, and
ElevenLabs runs them itself. Its calls to webhook and system tools (such as `end_call`) aren't recorded in the
session's history; its calls to MCP tools are recorded as a
[`NativeToolCallPart`][pydantic_ai.messages.NativeToolCallPart] and
[`NativeToolReturnPart`][pydantic_ai.messages.NativeToolReturnPart]. MCP tools that require approval on
the ElevenLabs side aren't supported. The session denies such a call (with a warning) rather than leave
the conversation waiting up to 300 seconds for an approval that can't come. The agent's LLM is told that the user
denied the call, and ElevenLabs reports nothing further about it. Give the agent those tools through a Pydantic AI
[MCP toolset](../mcp/client.md) instead, which is the recommended way, or turn approval off for the tool
in the agent's configuration.

ElevenLabs stores tool parameters in its own restricted schema dialect rather than JSON Schema.
`type`, `description`, `enum`, `items`, `properties`, and `required` are supported; other keywords
(such as `additionalProperties` or numeric bounds) are not; and **every parameter needs a
description**. Nested models (`$defs`/`$ref`) are inlined, and optional (nullable) parameters are
collapsed onto their non-null type before conversion, with optionality carried by `required` alone.
The dialect has no way to say `null` — a required nullable parameter is advertised as its
non-null type, and the agent's LLM, which only ever sees the advertised schema, always supplies a
value for it. `'sync'` therefore fails loudly for a tool whose parameters lack descriptions (add
them in the function docstring, and set
[`require_parameter_descriptions=True`](../tools.md#function-tools-and-schema) to catch a missing one when the
tool is defined) or use a shape the dialect cannot express (a union of types, or a
recursive reference). The `'error'` comparison checks what the dialect can express, comparing a
parameter's description only when the local schema declares one.

### Feature support and limitations

| Feature | Support | Notes |
| --- | --- | --- |
| Audio format | Full feature support | Mono PCM16; rates are fixed per agent (16 kHz default) and validated at connect, see [Audio formats](#audio-formats) |
| Text output | Limited parameter support | `output_modality='text'` maps to the text-only override, which needs its toggle on the agent |
| Image input | Unsupported | [`send()`][pydantic_ai.realtime.RealtimeSession.send] rejects images with a `UserError` |
| Manual turns | Unsupported | ElevenLabs' server-side turn model owns turn-taking; `turn_detection` must stay `True` |
| Explicit interruption/truncation | Unsupported | ElevenLabs [interrupts server-side](turns.md#barge-in) and emits `RealtimeResponseInterruptedEvent`; `interrupt()` raises |
| Session seeding | Unsupported | The conversation WebSocket has no history-seeding channel |
| Input transcription | Full feature support | Always on: per-utterance final transcripts from the agent's ASR; `input_transcription_model=None` raises |
| Native tools | Unsupported | None can be configured from the session; configure [local fallbacks](tools.md#native-tools) for web capabilities. The agent's own MCP tool calls are recorded as `NativeToolCallPart`s named `mcp_server:<server id>` |
| Async tool calls | Unsupported | The agent waits for each tool result (see [concurrent tool execution](tools.md#concurrent-tool-execution)) |
| Usage | Limited parameter support | Context tokens only, with `context_usage` in the agent's `client_events`; no cost (see [Usage and cost](#usage-and-cost)) |
| Context window | Limited parameter support | Known only once the agent reports `context_usage`: `session.context_window_used` then gives the share of the LLM's context in use (see [Usage and cost](#usage-and-cost)) |
| Session duration | Limited parameter support | Capped by the agent's [maximum conversation duration](https://elevenlabs.io/docs/eleven-agents/customization/conversation-flow) (600 s by default, up to 7,200 s, set on the agent, not per conversation); at the cap the session gets a non-recoverable `RealtimeSessionErrorEvent` and raises a [`RealtimeError`][pydantic_ai.realtime.RealtimeError] |
| State-restoring reconnect | Unsupported | A dropped connection ends the session, and `reconnect` raises, see [Provider session limits](lifecycle.md#provider-session-limits) |

See [Audio, images, and transcripts](audio.md), [Turns and interruptions](turns.md),
[Tools](tools.md), and [Connection lifecycle](lifecycle.md) for the provider-agnostic workflows.

### Gateway

ElevenLabs is not currently available through the [Pydantic AI Gateway](../gateway.md). Connect
with an `elevenlabs-agents:<agent_id>` model string, or with `provider='elevenlabs'` or an
`ElevenLabsProvider`.

### Usage and cost

ElevenLabs reports the LLM's context consumption only, as `input_tokens`, and only when
`context_usage` is in the agent's `conversation.client_events` list (it is not by default, and adding it
is recommended: it also tells the session where a spoken response ends, see
[Provider-specific quirks](#provider-specific-quirks); a session on an agent without it warns once). A report arrives when the agent has finished generating a
response, with its last audio chunk (well before that audio has played), and counts toward the session's usage without belonging to any one response. When ElevenLabs
answers a turn with a fallback LLM, the report names that model and its context window. The context window it names is kept on
[`ElevenLabsAgentsRealtimeConnection.context_limit_tokens`][pydantic_ai.realtime.elevenlabs_agents.ElevenLabsAgentsRealtimeConnection.context_limit_tokens],
and [`context_window_used`][pydantic_ai.realtime.RealtimeSession.context_window_used] is the ratio of
the two.

No price is known for the platform, so [`usage.cost`][pydantic_ai.usage.RunUsage.cost] is always
`None` and a [`cost_limit`][pydantic_ai.usage.UsageLimits.cost_limit] can't be enforced. Look the
cost up afterwards through the ElevenLabs conversations API: `GET /v1/convai/conversations/{conversation_id}`
reports the conversation's cost in credits as `metadata.cost`, its length as `metadata.call_duration_secs`, and a
breakdown, including the LLM tokens used per model, under `metadata.charging`. Every
[`ModelResponse`][pydantic_ai.messages.ModelResponse] the conversation ends, cut off and lost ones included,
carries the conversation's id in `provider_details['conversation_id']` (one still under way when the session
itself closes has none), and the live connection has it as
[`ElevenLabsAgentsRealtimeConnection.conversation_id`][pydantic_ai.realtime.elevenlabs_agents.ElevenLabsAgentsRealtimeConnection.conversation_id].

### Provider-specific quirks

- In audio mode, a response's transcript arrives whole once the agent's LLM has finished, which can be
  before or during its audio; only text-only mode streams it. Synthesis outruns playback, so a spoken
  response is usually delivered in full while the user is still hearing its first words.
- With a TTS model that voices expressive audio tags, such as `eleven_v3_conversational` or `eleven_v4_turbo`,
  the agent's LLM writes tags like `[laughs]` into its reply to direct the delivery, and ElevenLabs sends them as
  part of the response's transcript. Pydantic AI removes them, from the streamed transcript and the one in history
  alike, and keeps them, in order and without brackets, in the response's `provider_details['audio_tags']`
  (`['laughs']`). Only lowercase words in square brackets count as tags, so citations (`[1]`), Markdown links,
  and placeholders like `[Your Name]` stay in the text. Set `elevenlabs_agents_strip_audio_tags=False` to keep the
  transcript as ElevenLabs sends it (`[laughs] Hi there.`). To keep the LLM from writing tags in the first place,
  turn off expressive mode on the agent (`tts.expressive_mode: false`), which also gives up the expressive delivery.
  This works only in Custom voice mode: in [Realtime voice mode](https://elevenlabs.io/docs/eleven-agents/customization/realtime-models), ElevenLabs manages expressive mode itself.
- A barge-in cuts each agent message down to what the user heard, as ElevenLabs reports it. Synthesis
  outruns playback, so this is usually after the message's response has ended. History then holds an
  [interrupted copy](history.md#transcription-and-history-edge-cases) of the response, in the same position,
  whose speech part has the heard text as its transcript: the cut-off message ends in `...`, and a message
  the user heard none of (one after it in the same turn) has no transcript. What the agent said is kept in
  the part's `provider_details['original_transcript']`. A response cut off while it is still being
  generated also carries the heard text in `provider_details['corrected_agent_response']`.
- A text turn goes out under an `id` naming it, and each reply says which text turns it answers (its
  `in_response_to_ids`), so [`wait_for_reply()`][pydantic_ai.realtime.RealtimeSession.wait_for_reply] waits for
  the reply that answers the text: the one under way, if it names the text, and otherwise the next one. A reply
  that names a text also answers any text sent before it, as the agent folds text sent while it speaks into one
  reply, which can name only the latest. (Text sent while the agent speaks usually cuts it off, and the next reply
  answers it.)
- [`send(text, respond=False)`](turns.md#text-turns) is a `contextual_update`: the agent reads it from
  its next turn on, without stopping or replying.
- A response's parallel tool calls each run as soon as they arrive.
- With `elevenlabs_agents_tool_sync='off'`, client tools registered on the ElevenLabs side with
  `expects_response: false` are fire-and-forget: the session still runs them, but their results are never sent
  back to the agent. (The other modes require every client tool to expect a response.)
- A conversation waiting in the workspace's concurrency queue hears ElevenLabs' hold audio, which
  isn't part of the conversation; one that times out there ends with a
  [`RealtimeError`][pydantic_ai.realtime.RealtimeError] saying so.

## Eleven Realtime

Use an `elevenlabs:<model>` model string, or
[`ElevenLabsRealtimeModel`][pydantic_ai.realtime.elevenlabs.ElevenLabsRealtimeModel] for model-level
configuration, to talk to ElevenLabs' [Eleven Realtime](https://elevenlabs.io/docs/eleven-agents/customization/realtime-models)
speech-to-speech models. Unlike [ElevenLabs Agents](#elevenlabs-agents), there is no hosted agent to configure:
the session speaks [ElevenLabs' OpenAI Realtime-compatible API](https://elevenlabs.io/docs/eleven-agents/guides/openai-realtime-compatibility),
and Pydantic AI configures it from your agent. Start with the [realtime quickstart](overview.md#quickstart).

### Setup

To use Eleven Realtime, install `pydantic-ai-slim` with the `elevenlabs-realtime` optional group. It includes
the `openai` package, because the API reuses the OpenAI Realtime protocol's event types:

```bash
pip/uv-add "pydantic-ai-slim[elevenlabs-realtime]"
```

Set `ELEVENLABS_API_KEY`, or pass an
[`ElevenLabsProvider`][pydantic_ai.providers.elevenlabs.ElevenLabsProvider] with `api_key=`. The same key
and provider serve [ElevenLabs Agents](#elevenlabs-agents). To keep the connection in a
[data-residency region](https://elevenlabs.io/docs/overview/administration/data-residency), pass the region's
host as `base_url=`:

```python
from pydantic_ai import Agent
from pydantic_ai.providers.elevenlabs import ElevenLabsProvider
from pydantic_ai.realtime.elevenlabs import ElevenLabsRealtimeModel

model = ElevenLabsRealtimeModel(
    '{{ elevenlabs:realtime.name }}',
    provider=ElevenLabsProvider(base_url='https://api.eu.residency.elevenlabs.io'),
)
agent = Agent(instructions='You are a friendly voice assistant.')
```

### Model names

Use the `elevenlabs:` prefix with a model name, e.g. `agent.realtime('{{ elevenlabs:realtime }}')`:

| Model | Optimized for |
| --- | --- |
| `eleven_realtime_v1_mini` | Price and latency |
| `eleven_realtime_v1_max` | Quality: it reasons before it responds, for complex instructions and multi-step tool use |

`eleven_realtime_v1_max` needs a [`max_tokens`][pydantic_ai.realtime.RealtimeModelSettings.max_tokens] of at
least 1024 (or none), isn't available in the India and Singapore data-residency regions, and costs more per
token. See [ElevenLabs' model list](https://elevenlabs.io/docs/eleven-agents/customization/realtime-models) for
the details.

### Settings

[`ElevenLabsRealtimeModelSettings`][pydantic_ai.realtime.elevenlabs.ElevenLabsRealtimeModelSettings]
extends the [shared settings](overview.md#shared-settings):

```python
from pydantic_ai.realtime.elevenlabs import ElevenLabsRealtimeModel, ElevenLabsRealtimeModelSettings

settings = ElevenLabsRealtimeModelSettings(
    elevenlabs_voice='george',
    elevenlabs_language='en',
    elevenlabs_max_duration_seconds=1800,
    turn_detection={'sensitivity': 'high'},
)
model = ElevenLabsRealtimeModel('{{ elevenlabs:realtime.name }}', settings=settings)
```

- `elevenlabs_voice` takes an ElevenLabs voice ID (e.g. `'JBFqnCBsd6RMkjVDRZzb'`) or a default voice's name;
  the server's default is `'brian'`.
- `elevenlabs_language` sets the conversation's language (e.g. `'es'` or `'pt-br'`), which ElevenLabs uses for
  transcription, speech, and the model's replies. It defaults to English.
- `elevenlabs_max_duration_seconds` caps the session's length, from 60 to 7,200 seconds; see
  [Session length](#session-length).
- `elevenlabs_strip_audio_tags` (default `True`) removes expressive audio tags such as `[cheerful]` from the
  model's transcripts, as for ElevenLabs Agents; see
  [Provider-specific quirks](#eleven-realtime-quirks).

Turn detection is ElevenLabs' semantic VAD, which decides from what the user said, not only from silence,
whether they're done. The shared [`turn_detection`](turns.md#automatic-turn-detection) setting maps onto it:
`sensitivity` becomes its `eagerness` (`'low'`, `'medium'`, or `'high'`), and `prefix_padding_ms` and
`silence_duration_ms` are passed through. `turn_detection=False` turns it off for
[push-to-talk](turns.md#push-to-talk).

The session is configured through the OpenAI Realtime protocol, so `ElevenLabsRealtimeModelSettings` also takes
the [`OpenAIRealtimeModelSettings`][pydantic_ai.realtime.openai.OpenAIRealtimeModelSettings] fields.
`openai_turn_detection` takes a [`SemanticVAD`][pydantic_ai.realtime.openai.SemanticVAD], which sets
`create_response` and `interrupt_response` too; a [`ServerVAD`][pydantic_ai.realtime.openai.ServerVAD] raises
a `UserError`, as ElevenLabs only supports semantic VAD. ElevenLabs accepts and then ignores the other
OpenAI fields, so Pydantic AI doesn't send them: `openai_voice` (use `elevenlabs_voice`),
`openai_input_noise_reduction`, `openai_output_speed`, `openai_truncation`, and `openai_prune_tool_items`.
Of the shared settings, `parallel_tool_calls` and `thinking` are ignored the same way.

[Input transcription](audio.md#input-transcription) is always on: each spoken turn's transcript arrives whole
once the turn is committed, usually after the reply has started. It takes no model, so
`input_transcription_model` only accepts `'auto'` (the default); `None` still leaves transcripts on, and any
other value raises a `UserError`.

### Feature support and limitations

| Feature | Support | Notes |
| --- | --- | --- |
| Audio format | Limited parameter support | Mono PCM16 at 24 kHz, input and output. The input rate is configurable (set `audio_input_sample_rate` through [`profile=`](overview.md#provider-support)); output is always 24 kHz. The API also offers G.711, which Pydantic AI does not expose |
| Text output | Unsupported | ElevenLabs answers text input in text, but then ignores audio input altogether, so `output_modality='text'` raises a `UserError` |
| Image input | Unsupported | ElevenLabs drops images from the conversation, so [`send()`][pydantic_ai.realtime.RealtimeSession.send] rejects them with a `UserError` before sending |
| Manual turns | Full feature support | `turn_detection=False` plus [commit/create verbs](turns.md#push-to-talk) |
| Interruption | Full feature support | [`interrupt()`](turns.md#barge-in) cancels the response, and `played_ms` truncates what the model remembers saying to what the user heard |
| Session seeding | Limited parameter support | User and assistant text only; seeded audio and images are unsupported |
| Input transcription | Limited parameter support | Always on, with whole-turn transcripts; no transcription model to choose |
| Native tools | Unsupported | ElevenLabs offers MCP tools, which Pydantic AI does not expose yet |
| Async tool calls | Unsupported | The model answers the user while a [tool runs](tools.md#concurrent-tool-execution), but then abandons the call, see [Provider-specific quirks](#eleven-realtime-quirks) |
| Usage | Unsupported | ElevenLabs reports no usage, so token counts are zero and `usage.cost` is `None` |
| Reconnection | Full feature support | Pydantic AI [replays completed local history](lifecycle.md#state-restoration) into a new session; in-flight media is lost |

See [Audio, images, and transcripts](audio.md), [Turns and interruptions](turns.md),
[Tools](tools.md), and [Connection lifecycle](lifecycle.md) for the provider-agnostic workflows.

### Browser connections

ElevenLabs serves OpenAI's client-secret endpoint, but not its WebRTC one, so
[browser WebRTC](deployment.md#browser-webrtc-server-sideband) isn't available.
[`create_client_secret()`][pydantic_ai.realtime.openai.OpenAIRealtimeModel.create_client_secret] still mints the
short-lived credential ElevenLabs' own browser guide uses: a browser running OpenAI's Realtime SDK passes it as
the API key and opens a WebSocket to `wss://api.elevenlabs.io/v1/convai/realtime` itself. The secret is bound
to the session your agent configures, but the conversation then runs in the browser, outside Pydantic AI.

### Gateway

Eleven Realtime is not currently available through the [Pydantic AI Gateway](../gateway.md). Connect with an
`elevenlabs:<model>` model string, or with `provider='elevenlabs'` or an `ElevenLabsProvider`.

### Session length

ElevenLabs closes a session once it reaches its maximum duration (600 seconds unless
`elevenlabs_max_duration_seconds` says otherwise). With a [reconnect policy](lifecycle.md#reconnecting), the
session reconnects, and the new session gets the [replayed local history](lifecycle.md#state-restoration), so the
call carries on. Without a reconnect policy, or once its reconnects are used up, the session raises a
[`RealtimeError`][pydantic_ai.realtime.RealtimeError] with ElevenLabs' close reason.

ElevenLabs also closes a session that hasn't received anything from the client for 60 seconds, which happens
whenever no microphone audio is streaming, as in a text-only or push-to-talk session. To keep such a session open,
the connection sends an empty audio append (which adds no audio) after 45 seconds without sending anything.

### Provider-specific quirks {#eleven-realtime-quirks}

- ElevenLabs' semantic VAD can report that the user stopped speaking, then hear them carry on in the same turn,
  so [`RealtimeInputSpeechStartEvent`][pydantic_ai.realtime.RealtimeInputSpeechStartEvent] and
  [`RealtimeInputSpeechEndEvent`][pydantic_ai.realtime.RealtimeInputSpeechEndEvent] can come more than once per
  turn. The turn joins the history once ElevenLabs commits it.
- Eleven Realtime writes expressive audio tags such as `[cheerful]` into its replies to direct the delivery, and
  sends them as part of the transcript. Pydantic AI removes them, from the streamed transcript and the one in history
  alike, and keeps them, in order and without brackets, in the response's `provider_details['audio_tags']`
  (`['cheerful']`). Only lowercase words in square brackets count as tags, so citations (`[1]`), Markdown links, and
  placeholders like `[Your Name]` stay in the text. Set `elevenlabs_strip_audio_tags=False` to keep the transcript
  as ElevenLabs sends it.
- When ElevenLabs replies to something the user said while a tool call is still running, it abandons the call: its
  result never reaches the model, and a response asked for after it never comes. The session then
  gets a recoverable [`RealtimeSessionErrorEvent`][pydantic_ai.realtime.RealtimeSessionErrorEvent] of type
  `'tool_calls_abandoned'` naming the calls. The tool keeps running and its result is still recorded in history, but
  no reply to it is waited for: [`RealtimeTurnCompleteEvent`][pydantic_ai.realtime.RealtimeTurnCompleteEvent] and
  [`wait_for_reply()`][pydantic_ai.realtime.RealtimeSession.wait_for_reply] come when the response that cut in ends.
- ElevenLabs makes parallel tool calls as back-to-back responses of one call each, and answers all their results
  with one reply, which the session waits for once:
  [`RealtimeTurnCompleteEvent`][pydantic_ai.realtime.RealtimeTurnCompleteEvent] and
  [`wait_for_reply()`][pydantic_ai.realtime.RealtimeSession.wait_for_reply] come when it ends.
- `eleven_realtime_v1_max` sometimes answers a question that needs a tool with two responses: a spoken preamble
  ("Let me check the weather…") that completes on its own, then the tool call in a response nobody asked for.
  Nothing links the two, so [`RealtimeTurnCompleteEvent`][pydantic_ai.realtime.RealtimeTurnCompleteEvent] and
  [`wait_for_reply()`][pydantic_ai.realtime.RealtimeSession.wait_for_reply] can come after the preamble, and the
  tool call and the answer follow as an exchange of their own. History is unaffected. Don't assume the first turn
  completion after such a question carries the answer: keep consuming the session's events.
- The protocol follows OpenAI Realtime, but feature support comes from the ElevenLabs model profile; avoid
  assuming every OpenAI behavior is available.
