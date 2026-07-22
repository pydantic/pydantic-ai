<!-- braindump: rules extracted from PR review patterns -->

# pydantic_ai_slim/pydantic_ai/models/ Guidelines

## API Design

<!-- rule:912 -->
- Silently ignore unsupported generic tuning settings (`temperature`, sampling params, penalties, …) at runtime and document them in docstrings — a model that simply no-ops an unsupported knob keeps client code portable across models; failing noisily would break that portability (provider-namespaced settings like `google_*`/`openai_*` are the next rule, not this one)
<!-- rule:81 -->
- Apply identical response processing to both `request()` and `request_stream()` — if `request()` calls `_process_response()`, `request_stream()` must apply it to each chunk — Ensures streaming and non-streaming code paths support the same message types (`ToolCallPart`, `NativeToolCallPart`, `TextPart`, etc.) with consistent behavior, preventing bugs where features work in one mode but fail in the other
<!-- rule:598 -->
- Expose provider-specific data via `ModelResponse.provider_details` or `TextPart.provider_details` — prevents API bloat and maintains consistent provider integration patterns — Keeps the core response interface clean while allowing providers to expose logprobs, safety filters, content filtering, and usage metrics without breaking consistency across integrations
<!-- rule:26 -->
- Don't add preemptive client-side guards that reject provider-namespaced settings (`google_*`, `openai_*`, …) based on assumed capability limits; forward the setting the user opted into and let the provider API surface the actual incompatibility — the API is the authority on what it currently supports, so a client-side guard degrades functionality on outdated assumptions
<!-- rule:478 -->
- Token counting must mirror actual request parameters (`tools`, `system_prompt`, configs) and use identical message formatting — Ensures token count estimates match actual API usage, preventing billing surprises and quota errors
- Per-request injections or mutations of request content (message blocks, tool defs, instructions, cache breakpoints) must anchor to a position that doesn't move with history length (e.g. the first user message or a fixed index), never the last message or a length-based index — anchoring to a moving position shifts the cacheable prefix every turn, so the provider silently re-processes the tail instead of reading from cache, a cost/latency regression that surfaces no error

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

<!-- /braindump -->
