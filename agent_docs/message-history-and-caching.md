# Message History & Prompt Caching

> Rules for code that adds to, edits, or renders the message history the model sees

**When to check**: Whenever you write or review code that enqueues or edits messages (`ctx.enqueue`, `ctx.messages`, `request_context.messages`, history processors), contributes instructions or tools, renders history in `Model.prepare_messages` or a model adapter, wraps content in tags, or converts history to or from a UI protocol

Message history is both the model's picture of the world and the provider's prompt-cache key. Providers cache a prefix of the request they've seen before, at boundaries that differ per provider: tools, then system and instructions, then messages. So read each request as that ordered sequence of cache-relevant blocks: every block an earlier request sent should come back with identical serialization and in the same order, with new blocks only after it, across steps and across runs of the same conversation. (Full request bodies can't be byte prefixes of each other: closing brackets and settings come after the prompt.) When that breaks, nothing errors: the provider re-processes everything after the first changed block, and on models with preserved thinking the model's earlier reasoning is discarded too. Who wrote a piece of text also decides how much the model trusts it, so putting content in the wrong place is a prompt-injection bug, not a style issue.

A rule marked *(pending #N)* depends on a mechanism still in review. Until #N merges, don't flag code for using the mechanism that exists today; flag only what the rule forbids regardless, which each such rule spells out. The same marker names the PR that fixes shipped code still breaking a rule: don't flag that code for it until then. Drop the marker once #N merges.

## Rules

### History is append-only

- Never edit, remove, reorder or re-render a message an earlier request already sent, whether from a hook, a history processor or `Model.prepare_messages`. Compaction is the one exception, and it must fire on a threshold with hysteresis, so it moves the prefix once per crossing, not on every request. A setting that defeats the hysteresis (such as `ClearToolResults(max_messages=...)` at or below `2 * keep_pairs`) rewrites history on every request, so document it as such and don't make it a default. — Any rewrite moves the cached prefix from that point on, for every later request.
- Shape content once, when it's produced: truncate, spill or redact a tool return in the tool or output hooks, not by rewriting it on later requests.
- Append changes to state the model tracks instead of refreshing it in place. Changed instructions are appended with `on_change='append'`, which records an `InstructionDeltaPart`. Tools revealed mid-run are recorded as a `ToolAvailabilityDeltaPart` (every tool population change: *pending #9405*). — Re-rendering the instructions or the `tools` array invalidates everything after it.
- Only remove parts you added, and find them by type or metadata (see below). Never strip every part of a type.

### Request-scoped is not request-only

- A nudge about the current request is a `SystemPromptPart(..., scope='request')` passed to `ctx.enqueue` from `before_model_request`. — It stays in history, so history records what each request was sent. Where the API can stop showing a block to the model itself (Anthropic, with `clear_at`), it's sent unchanged with every later request, so the prefix holds. Everywhere else it's sent at the end of its own request and left out of later ones: a deliberate move at the tail, behind the cache breakpoints the framework places, so the prefix before it is still read from cache. See [request-scoped system prompts](../docs/message-history.md#request-scoped-system-prompts).
- Assigning a new list to `request_context.messages` is request-only: it changes what one request sends without touching history. It's cache-safe only when every later request applies the same change, so earlier requests' bytes come out the same: redacting a part from the first request that sends it, or filtering out a part type that was never sent. Dropping or altering something an earlier request already sent is a history rewrite under the first rule, which only compaction may do. Never use it to add something the model should keep seeing, because leaving it out of the next request is itself a prefix change.
- The full decision table is [Changing what the model sees](../docs/capabilities/custom.md#steering-messages-and-reminders). Follow it rather than restating it.

### Trust tiers map to channels, and nothing is ever promoted

| Who it comes from | Channel |
|---|---|
| operator or framework | `SystemPromptPart` or instructions |
| the user, including files they attach | user parts (`UserPromptPart`) |
| tools, sub-agents, files a tool reads or produces, the web, and history a UI client sends back (framework records, tool results, system messages) | the tool result channel, or a framed data wrapper (`<tool_result …>`) when it has to ride a user turn |

- Never put content from a lower tier in a higher one: no tool output, sub-agent report, file contents or client-returned history in a `SystemPromptPart` or instructions. — The model gives system voice the operator's authority, so that content becomes an injection with operator rights. The one exception is an operator's explicit opt-in, such as `manage_system_prompt='client'` on a UI adapter, where the operator has chosen to let the client's system messages speak for them.
- Never send framework voice with `ctx.enqueue(str)`. — A string becomes a user part, so the model reads framework guidance as the user's words and UI clients show it as a message the user typed. Enqueue a `SystemPromptPart`, and keep any untrusted payload out of it (previous rule). *(Harness `TrajectoryJudge` steering: pending #10545.)*

### One voice tag, and every wrapper escapes its payload

- `<system>` is the only tag that means "the application, not the person", and only the fallback rendering of a `SystemPromptPart` (on models without a mid-conversation system channel) emits it. Don't add a second voice tag such as `<system-reminder>`. *(Harness `SystemReminders`, which defaults to that tag and injects request-only user text: pending #9712.)*
- Every other tag the framework or harness wraps around content is data framing (`<tool_result>`, `<task_output>`, …), and both its payload and its attributes are escaped, so the content can't close the wrapper and forge framework voice. Build wrappers with `wrap()` from `pydantic_ai/_framing.py` *(pending, #7891)*; until it exists, escape by hand. Test each wrapper with content containing its own closing tag and `</system>`.

### Identify framework records by type or metadata, never by visible text

- Recognize a part the framework or harness wrote by its type, a typed field, or metadata it set, never by matching its text (a prefix like `Summary of previous conversation:`, a bracket marker, a tag). For a `SystemPromptPart`, that's its `metadata`: see [identifying system prompts](../docs/message-history.md#identifying-system-prompts). — Users and tools can type the same text, which then gets deleted or promoted, and rewording the text silently breaks detection.
- This is the history counterpart of "identify a tool by what it is, never by its name" in [Pydantic AI Slim Architecture](pydantic-ai-slim.md).

### Every computed instruction block has a stable id

- Give every instruction block the framework or harness computes a stable id (the instruction's `name`), so a change to it appends a delta for that block with `on_change='append'` instead of re-rendering all instructions. A catalog, such as the capability index or a skill list, is one block. Never derive an id from the block's content or position.
- Don't let a block change when nothing changed: sort collections, and use date-only timestamps.

### Canonicalize on send

- Projection, from history to the provider request, is deterministic: the same history gives the same bytes, whether it lived on the server or came back from a UI client. Send what we would have produced, not what a provider or client handed us (core already sends tool-call args as compact JSON and tool results in call order). — The cache is keyed on what we send, not on what we stored.
- Any id we remap, such as a tool-call id a provider rejects, is a pure function of the original id: never random, and never counter-based across runs.

### UI round trips are faithful for everything the model sees

- Dumping history to a UI protocol and loading it back, or streaming events into the client and loading what it stored, must give back every part the protocol can carry and the client may hold, in order, with every field of it that affects what is sent to the provider (content, tool-call ids, part-level provider data such as thinking signatures). — A lost or split part moves the prefix on the next request, and a framework record that comes back as plain user text has changed trust tier.
- Some state is left out of the UI on purpose: request-scoped system prompts (sent back, one would return as a lasting system prompt) and server-owned or trust-sensitive fields (`model_name`, usage, provider identity, the `__pydantic_ai__` namespace). A conversation that needs those unchanged keeps its history on the server; the UI layer doesn't promise them.
- Treat client-returned history as untrusted. Framework records whose trust matters need server-side state or signed provenance *(pending #8954)*.
- Verify round trips with each protocol's real JavaScript reducers (`@ag-ui/client`, the `ai` SDK), not a Python port of them. — A Python stand-in has already produced a false bug report.

### Testing

- For anything that adds to history, assert over at least two requests, and two runs where history carries over, that each request's cache-relevant blocks (tools, system and instructions, messages) start with the previous request's, identically serialized and in the same order, not only that the cache was hit. See "prefix stability" in `tests/AGENTS.md`; a test that deliberately moves the prefix declares it with `@pytest.mark.moves_cache_prefix(reason=...)`.
