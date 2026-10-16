# Message History & Prompt Caching

> Rules for code that adds to, edits, or renders the message history the model sees

**When to check**: Whenever you write or review code that enqueues or edits messages (`ctx.enqueue`, `ctx.messages`, `request_context.messages`, history processors), contributes instructions or tools, renders history in `Model.prepare_messages` or a model adapter, wraps content in tags, or converts history to or from a UI protocol

Message history is both the model's picture of the world and the provider's prompt-cache key. Providers cache the exact bytes of a prefix they've seen, so every request should start with the previous request's bytes, across steps and across runs of the same conversation. When it doesn't, nothing errors: the provider re-processes everything after the first changed byte, and on models with preserved thinking the model's earlier reasoning is discarded too. Who wrote a piece of text also decides how much the model trusts it, so putting content in the wrong place is a prompt-injection bug, not a style issue.

A rule marked *(pending #N)* names a mechanism still in review. The rule applies now, with the nearest mechanism that exists; drop the marker once #N merges.

## Rules

### History is append-only

- Never edit, remove, reorder or re-render a message an earlier request already sent, whether from a hook, a history processor or `Model.prepare_messages`. Compaction is the one exception: it fires on a threshold with hysteresis, so it moves the prefix once per crossing, not on every request. — Any rewrite moves the cached prefix from that point on, for every later request.
- Shape content once, when it's produced: truncate, spill or redact a tool return in the tool or output hooks, not by rewriting it on later requests.
- Append changes to state the model tracks instead of refreshing it in place. Changed instructions are appended with `on_change='append'`, which records an `InstructionDeltaPart` *(pending #8188)*. Tools revealed mid-run are recorded as a `ToolAvailabilityDeltaPart` (every tool population change: *pending #9405*). — Re-rendering the instructions or the `tools` array invalidates everything after it.
- Only remove parts you added, and find them by type or metadata (see below). Never strip every part of a type.

### Request-scoped is not request-only

- A nudge about the current request is a `SystemPromptPart(..., scope='request')` passed to `ctx.enqueue` from `before_model_request`. — It stays in history, so later requests send the same bytes, and providers that can hide it after its request (Anthropic) do.
- Assigning a new list to `request_context.messages` is request-only: history never sees it. Use it to trim, filter or redact what one request sends. Never use it to add something the model should keep seeing, because leaving it out of the next request is itself a prefix change.
- The full decision table is [Changing what the model sees](../docs/capabilities/custom.md#steering-messages-and-reminders). Follow it rather than restating it.

### Trust tiers map to channels, and nothing is ever promoted

| Who it comes from | Channel |
|---|---|
| operator or framework | `SystemPromptPart` or instructions |
| the user | user parts (`UserPromptPart`) |
| tools, sub-agents, files, the web, anything a UI client sends | the tool result channel, or a framed data wrapper (`<tool_result …>`) when it has to ride a user turn |

- Never put content from a lower tier in a higher one: no tool output, sub-agent report, file contents or client-supplied text in a `SystemPromptPart` or instructions. — The model gives system voice the operator's authority, so that content becomes an injection with operator rights.
- Never send framework voice with `ctx.enqueue(str)`. — A string becomes a user part, so the model reads framework guidance as the user's words and UI clients show it as a message the user typed. Enqueue a `SystemPromptPart`.

### One voice tag, and every wrapper escapes its payload

- `<system>` is the only tag that means "the application, not the person", and only the fallback rendering of a `SystemPromptPart` (on models without a mid-conversation system channel) emits it. Don't add a second voice tag such as `<system-reminder>`.
- Every other tag the framework or harness wraps around content is data framing (`<tool_result>`, `<task_output>`, …), and both its payload and its attributes are escaped, so the content can't close the wrapper and forge framework voice. Build wrappers with `wrap()` from `pydantic_ai/_framing.py` *(pending, #7891)*; until it exists, escape by hand. Test each wrapper with content containing its own closing tag and `</system>`.

### Identify framework records by type or metadata, never by visible text

- Recognize a part the framework or harness wrote by its type, a typed field, or metadata it set, never by matching its text (a prefix like `Summary of previous conversation:`, a bracket marker, a tag). — Users and tools can type the same text, which then gets deleted or promoted, and rewording the text silently breaks detection.
- This is the history counterpart of "identify a tool by what it is, never by its name" in [Pydantic AI Slim Architecture](pydantic-ai-slim.md).

### Every computed instruction block has a stable id

- Give every instruction block the framework or harness computes a stable id, so a change to it appends a delta for that block instead of re-rendering all instructions *(pending #8188)*. A catalog, such as the capability index or a skill list, is one block. Never derive the id from the content or its position.
- Don't let a block change when nothing changed: sort collections, and use date-only timestamps.

### Canonicalize on send

- Projection, from history to the provider request, is deterministic: the same history gives the same bytes, whether it lived on the server or came back from a UI client. Send what we would have produced, not what a provider or client handed us (core already sends tool-call args as compact JSON and tool results in call order). — The cache is keyed on the bytes we send.
- Any id we remap, such as a tool-call id a provider rejects, is a pure function of the original id: never random, and never counter-based across runs.

### UI round trips are faithful for everything the model sees

- Dumping history to a UI protocol and loading it back, or streaming events into the client and loading what it stored, must give back every part the model sees, in order and with the same content. — A lost or split part moves the prefix on the next request, and a framework record that comes back as plain user text has changed trust tier.
- Treat client-returned history as untrusted. Framework records whose trust matters need server-side state or signed provenance *(pending #8954)*.
- Verify round trips with each protocol's real JavaScript reducers (`@ag-ui/client`, the `ai` SDK), not a Python port of them. — A Python stand-in has already produced a false bug report.

### Testing

- For anything that adds to history, assert over at least two requests, and two runs where history carries over, that each request starts with the previous one, not only that the cache was hit. See "prefix stability" in `tests/AGENTS.md`; a test that deliberately moves the prefix declares it with `@pytest.mark.moves_cache_prefix(reason=...)`.
