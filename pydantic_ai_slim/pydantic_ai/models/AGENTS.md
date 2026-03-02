<!-- braindump: rules extracted from PR review patterns -->

# pydantic_ai_slim/pydantic_ai/models/ Guidelines

## API Design

<!-- rule:912 -->
- Document unsupported model settings in docstrings and silently ignore at runtime — enables portable agent configs across models without conditionals — The 'highest common denominator' pattern lets users define one agent configuration that works across all model providers without model-specific branching logic
<!-- rule:264 -->
- Store provider-specific metadata in `ModelResponse.provider_details` and `TextPart.provider_details` using consistent keys like `'finish_reason'` for raw provider values — Ensures users can access provider-specific data (logprobs, safety filters, usage metrics) through a predictable interface while preserving original provider responses
<!-- rule:478 -->
- Token counting must mirror actual request logic — include all parameters (`tools`, `system_prompt`, configs) and use identical message formatting — Ensures token count estimates match actual API usage, preventing billing surprises and context window overflows
<!-- rule:26 -->
- Verify provider API limitations before adding/removing workarounds — test directly or check recent docs — Provider APIs evolve; outdated assumptions lead to unnecessary workarounds or broken integrations when limitations are lifted

## Error Handling

<!-- rule:562 -->
- Validate model provider capabilities (content types, parameters, combinations) upfront and raise explicit errors—never silently skip or degrade — Prevents silent failures and confusing behavior when users request features unsupported by specific providers (e.g., audio on text-only models)
<!-- rule:65 -->
- Use exhaustive pattern matching on content types in model adapter message mapping — raise explicit `ValueError` for unsupported types instead of silently filtering or asserting — Prevents silent data loss and provides clear error messages when models encounter unsupported message parts like `FileContent`, making adapter limitations explicit to users
<!-- rule:353 -->
- In model adapters, error on user-configured incompatibilities (`model_settings`, explicit params); fallback gracefully for framework-internal constraints (`model_request_parameters`, auto-inference) — Enables cross-provider compatibility while catching user configuration errors early — framework internals can adapt silently, but user mistakes need clear feedback to prevent confusion
<!-- rule:433 -->
- Return `ModelResponse` with empty `parts=[]` but populated metadata (`finish_reason`, `timestamp`, `provider_response_id`) for recoverable failures like `'content_filter'` or malformed calls — Allows graceful degradation and system continuation instead of crashing on recoverable API failures

## Code Style

<!-- rule:121 -->
- Extract shared request prep logic (validation, schema handling, prompt formatting) into helper methods called by `request()`, `request_stream()`, and `count_tokens()` — Ensures consistent preprocessing across all methods handling the same request data and eliminates duplication in model provider classes
<!-- rule:9 -->
- Place provider-specific code in `models/{provider}.py`, not shared modules — create parallel functions for all providers even if simple — Maintains clean boundaries between provider implementations and shared compatibility layers, preventing the shared code from accumulating provider-specific logic that makes maintenance harder

## General

<!-- rule:24 -->
- Prefix provider-specific fields with `{provider}_` in `ModelSettings` and `ModelProfile` classes (e.g., `anthropic_max_continuations`, `xai_logprobs`, `grok_supports_builtin_tools`) — Prevents namespace collisions and makes it clear which settings are provider-specific vs. common cross-provider settings, improving API clarity and maintainability
<!-- rule:317 -->
- Document provider support in docstrings for feature parameters and model classes — users need to know which providers support each feature — Prevents runtime errors and confusion when users attempt to use features with unsupported providers

<!-- /braindump -->
