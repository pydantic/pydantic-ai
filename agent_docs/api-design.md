# API Design & Public Interfaces

> Rules for designing clean, maintainable APIs including keyword-only parameters, return value conventions, private vs public interfaces, capability flags, provider-agnostic abstractions, and avoiding implicit defaults.

**When to check**: When designing new APIs, adding parameters to public functions, or deciding what to expose publicly

## Rules

<!-- rule:40 -->
- Use keyword-only parameters (place `*` after 1-2 clear positional args) for functions with 3+ params, optional params, and `__init__` methods — Prevents breakage when adding parameters and eliminates positional argument confusion in configuration-heavy APIs
<!-- rule:124 -->
- Use `_: KW_ONLY` before fields with defaults in dataclasses/Pydantic models — Prevents breakage when adding parameters — forces keyword arguments so positional call sites don't break when field order changes
<!-- rule:56 -->
- Use instance methods when accessing `self` attributes or enabling polymorphism; use standalone functions for stateless logic — Instance methods eliminate parameter passing overhead, provide encapsulation, and enable subclass overriding. For shared logic across classes, extract to private top-level helpers (e.g., `_is_valid_format()`) that multiple methods can call.
<!-- rule:775 -->
- Return new collections from transform functions instead of mutating inputs (unless named `update_*` or `*_inplace`) — prevents surprising side effects and makes code easier to reason about — Immutable transforms prevent bugs from unexpected mutations and make data flow explicit, improving testability and composability
<!-- rule:929 -->
- Require ≥2 provider support before adding fields to cross-provider abstractions — Prevents API bloat with provider-specific features and ensures abstractions remain truly cross-compatible
<!-- rule:587 -->
- Add boolean capability flags to model profile classes for provider-specific features; check before sending optional parameters — Prevents API errors when features aren't universally supported across models in a provider (e.g., `bedrock_supports_prompt_caching` in `BedrockModelProfile`). Only add flags for features that error; omit for silently-ignored features.
<!-- rule:302 -->
- In `BuiltinToolReturnPart.content`, avoid redundancy: no duplicate fields, no wrapper dicts with single keys, no single-item lists, no repeating `return_value` data — Prevents API bloat and confusion — keeps the structure flat and ensures each field has a distinct purpose
<!-- rule:673 -->
- Avoid accessing `_private` attributes from external code — respect encapsulation and prevent breakage from internal changes — Private attributes can change without notice; using public APIs ensures backward compatibility and clear interfaces
<!-- rule:711 -->
- Avoid implicit defaults for context-specific API config (model names, endpoint URLs, API versions) — require explicit values or use sentinel values like `None` to signal user attention needed — Prevents silent failures when credentials don't match assumed defaults and makes cross-provider APIs predictable; document any side effects in config options explicitly
<!-- rule:33 -->
- Design provider-agnostic abstractions (e.g., `programmatically_callable`) in framework APIs and classes like `ToolDefinition` — avoid provider-specific fields (`allowed_callers`, beta headers, magic version strings) that couple interfaces to one implementation — Prevents tight coupling to specific providers, enables framework evolution without breaking changes, and keeps user-facing APIs consistent regardless of which provider they use
