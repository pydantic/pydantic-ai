<!-- braindump: rules extracted from PR review patterns -->

# pydantic_ai_slim/pydantic_ai/models/ Guidelines

## API Design

<!-- rule:912 -->
- Silently ignore unsupported generic tuning settings (`temperature`, sampling params, penalties, …) at runtime and document them in docstrings — a model that simply no-ops an unsupported knob keeps client code portable across models; failing noisily would break that portability (provider-namespaced settings like `google_*`/`openai_*` are governed by a separate rule below, not this one)
<!-- rule:81 -->
- Apply identical response processing to both `request()` and `request_stream()` — if `request()` calls `_process_response()`, `request_stream()` must apply it to each chunk — Ensures streaming and non-streaming code paths support the same message types (`ToolCallPart`, `NativeToolCallPart`, `TextPart`, etc.) with consistent behavior, preventing bugs where features work in one mode but fail in the other
<!-- rule:598 -->
- Expose provider-specific data via `ModelResponse.provider_details` or `TextPart.provider_details` — prevents API bloat and maintains consistent provider integration patterns — Keeps the core response interface clean while allowing providers to expose logprobs, safety filters, content filtering, and usage metrics without breaking consistency across integrations
<!-- rule:26 -->
- Don't add preemptive client-side guards that reject provider-namespaced settings (`google_*`, `openai_*`, …) based on assumed capability limits; forward the setting the user opted into and let the provider API surface the actual incompatibility — the API is the authority on what it currently supports, so a client-side guard degrades functionality on outdated assumptions
<!-- rule:478 -->
- Token counting must mirror actual request parameters (`tools`, `system_prompt`, configs) and use identical message formatting — Ensures token count estimates match actual API usage, preventing billing surprises and quota errors
- Per-request injections or mutations of request content (message blocks, tool defs, instructions, cache breakpoints) must land on a set of positions chosen by message identity, not by history length (e.g. every user message). Never anchor to a position *defined as* the tail — "the last message", or any length-based index — because that position moves every turn, which shifts the cacheable prefix, so the provider silently re-processes the tail instead of reading from cache, a cost/latency regression that surfaces no error. Covering the last message is correct when it falls out of an identity rule that covers the others too; what is forbidden is making the tail itself the anchor. Stability is necessary but not sufficient: pinning to the first user message alone is stable and still wrong when the wire needs the injection later in the conversation, which is how `container_upload` blocks stopped reaching a fresh container (https://github.com/pydantic/pydantic-ai/issues/7775). Cover every position the API acts on *and will accept the injection at*, then check each one is identity-anchored. Those are different sets: Anthropic acts on a `container_upload` in a user message holding only `tool_result` blocks, and rejects the request for it, so that position is excluded on purpose

## Error Handling

<!-- rule:562 -->
- Raise explicit errors for unsupported model features (e.g. function tools, JSON/native output modes) that can't be formed for a given model — never silently skip or degrade — makes capability limits discoverable at runtime; unsupported settings are governed by the settings rules above, and unrepresentable content/message-part types by the rule below
<!-- rule:65 -->
- Use exhaustive pattern matching for message part/content types in model adapters; raise explicit errors for unsupported types instead of filtering or assertions — Prevents silent data loss during message mapping and provides clear feedback when model APIs don't support certain content types (e.g., `FileContent`), making integration failures debuggable rather than mysterious
<!-- rule:433 -->
- Return `ModelResponse` with empty `parts=[]` but populated metadata (`finish_reason`, `timestamp`, `provider_response_id`) for recoverable API failures (content filters, empty content) — enables graceful degradation instead of cascading errors — Allows the system to handle provider-level failures gracefully by preserving response metadata for observability while signaling no usable content, preventing unnecessary exception propagation in model adapters

## Type System

<!-- rule:73 -->
- Use typed settings classes (e.g., `OpenAISettings`, `AnthropicSettings`) with provider-prefixed fields instead of `extra_body` or dict literals — Enables type checking and autocomplete for provider-specific config, preventing runtime errors from typos or invalid values
<!-- rule:972 -->
- Define Pydantic models to validate API responses — avoids `.get()` fragility and catches schema changes early — Prevents runtime errors from missing/malformed fields and provides type safety when parsing external API data

## General

<!-- rule:9 -->
- Place provider-specific code in `models/{provider}.py`, not shared modules — add functions consistently across all providers even if some are simple — Maintains clear architectural boundaries and prevents shared compatibility layers from accumulating provider-specific logic that becomes hard to maintain
- Anthropic-only helpers with a long background live in `_anthropic_*.py` siblings (`_anthropic_containers.py`, `_anthropic_bedrock_count_tokens.py`) so a reader of `anthropic.py` is not forced through them

A model should read reveal modes through `self.tool_deferral_mode` and
`self.tool_addition_mode`, never directly from the corresponding profile keys. An adapter that
implements a reveal renderer must declare its supported values in
`supported_tool_deferral_modes` and `supported_tool_addition_modes`; the inherited empty sets are
the safe default for adapters with no renderer.

An adapter that honors `CompactionPart`s on the wire declares
`compaction_requires_encrypted_content` and `compaction_retains_standing_prompt` and calls
`self._trim_before_compaction()` from its own message-prep step — never `_trim_messages_before_compaction`
directly, and never restating what those declarations imply. Each states only what the API does
(does it need the encrypted blob to honor an item; does the item keep serving the window's system
items); turning that into trim behavior belongs to the one helper. The two are independent — today's
two adapters happen to answer both the same way, so don't infer one from the other when adding a
third. They belong on the adapter, not the profile: eight providers route a profile of their own
through `OpenAIResponsesModel`, so a profile key would be absent exactly where the wire format is
most certain. Where in a request build the trim belongs stays adapter mechanics (OpenAI Responses
resolves server-side state from the *untrimmed* history, so it keeps a separate trimmed view).

### Third-party model fallback

Custom `Model` subclasses that continue reading `tool_defs` degrade gracefully: all tools are fully
declared and availability deltas fall back to a system-text announcement. Deferral is not withheld
on that model. Reading `declared_tool_defs` and `visibility_of()` opts the adapter into withholding.

<!-- /braindump -->
- When a model forwards a generic `ModelSettings` field, add it to that field's `Supported by:` list in `pydantic_ai/settings.py`, and give a new `Model` class a case in `tests/models/test_model_settings_support.py` — that test probes each class's outgoing request and fails when a list and the wire disagree.
- Narrow a capability by client class, never by the client's `base_url`. Where the gateway serves a model it must behave exactly as the provider's canonical API does, and the Pydantic AI Gateway — like an ordinary corporate proxy — reaches that API through the provider's normal SDK client carrying a proxy base URL. A host test therefore splits those callers off from the transport they actually reach and silently degrades them, which is why no capability here is decided that way. Genuinely separate transports (`AsyncAnthropicBedrock`, `AsyncAnthropicVertex`, `AsyncAnthropicFoundry`, …) are distinct client classes and earn their own gates — that is the line `isinstance` already draws. Probe the gateway leg (`Model('<id>', provider='gateway')`) rather than reasoning about it; a model the gateway genuinely does not serve belongs in `UNSUPPORTED_GATEWAY_MODEL_NAMES`, not in a carve-out that leaves the id advertised and degraded.
