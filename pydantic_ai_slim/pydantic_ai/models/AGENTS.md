<!-- braindump: rules extracted from PR review patterns -->

# pydantic_ai_slim/pydantic_ai/models/ Guidelines

## API Design

<!-- rule:912 -->
- Document unsupported model settings in docstrings and silently ignore at runtime — prevents breaking client code when models have different capabilities — Different model providers support different features; failing noisily when a setting is unsupported would break code portability across models, while silent degradation with clear documentation lets users make informed choices
<!-- rule:81 -->
- Apply identical response processing to both `request()` and `request_stream()` — if `request()` calls `_process_response()`, `request_stream()` must apply it to each chunk — Ensures streaming and non-streaming code paths support the same message types (`ToolCallPart`, `BuiltinToolCallPart`, `TextPart`, etc.) with consistent behavior, preventing bugs where features work in one mode but fail in the other
<!-- rule:598 -->
- Expose provider-specific data via `ModelResponse.provider_details` or `TextPart.provider_details` — prevents API bloat and maintains consistent provider integration patterns — Keeps the core response interface clean while allowing providers to expose logprobs, safety filters, content filtering, and usage metrics without breaking consistency across integrations
<!-- rule:26 -->
- Verify provider limitations through testing before implementing workarounds in `pydantic_ai/models/` — defer validation to runtime API responses rather than preemptive client-side checks — Prevents degrading functionality with unnecessary workarounds based on outdated assumptions, and lets the underlying API return clear error messages about actual incompatibilities
<!-- rule:478 -->
- Token counting must mirror actual request parameters (`tools`, `system_prompt`, configs) and use identical message formatting — Ensures token count estimates match actual API usage, preventing billing surprises and quota errors

## Error Handling

<!-- rule:562 -->
- Raise explicit errors for unsupported model features/content/parameters — never silently skip or degrade — Prevents silent failures and makes capability limits discoverable to users at runtime rather than producing unexpected behavior
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
