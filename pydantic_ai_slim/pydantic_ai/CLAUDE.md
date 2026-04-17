<!-- braindump: rules extracted from PR review patterns -->

# pydantic_ai_slim/pydantic_ai/ Guidelines

## API Design

<!-- rule:587 -->
- Add capability flags to model profile classes instead of inline `isinstance()` checks — prevents scattered feature detection logic — Centralizing feature support as boolean flags (e.g., `bedrock_supports_prompt_caching`) in profile classes makes capability detection maintainable and prevents duplicating provider/model type checks across usage sites.
<!-- rule:716 -->
- Configure provider-specific API features in `Provider.model_profile()`, not in model profile functions — model profiles should contain only intrinsic model characteristics — Keeps provider-agnostic model traits separate from provider-specific API behaviors, enabling models to work consistently across different providers
<!-- rule:264 -->
- Store provider-specific metadata in structured `provider_details` or `provider_metadata` fields, not in `id`, `content`, or `args` — Prevents semantic field overloading and enables consistent provider behavior interpretation while keeping main fields normalized across providers
<!-- rule:17 -->
- In `_otel_*.py` modules, implement only spec-defined features — no custom additions from internal concepts or other standards — Prevents spec drift and ensures compatibility with external tooling that expects standard-compliant telemetry data
<!-- rule:266 -->
- Store provider metadata (`provider_details`, `provider_name`, `id`, `signature`) in dedicated `provider_metadata` fields, not encoded in string fields — Prevents data loss during JSON serialization/deserialization cycles across storage, UI exchange, and provider round-trips — structured metadata survives where encoded strings lose type information
<!-- rule:987 -->
- Extend `WrapperToolset` for cross-cutting toolset behavior — don't modify base classes or individual toolset implementations — Composable wrappers (like `ApprovalRequiredToolset`, `DeferredLoadingToolset`) apply features to any toolset without coupling or duplication

## Type System

<!-- rule:185 -->
- End exhaustive `if`/`elif` chains with `else: assert_never(value)` for typed unions — Catches unhandled union variants at type-check time when unions are extended, preventing runtime errors from missing cases
<!-- rule:60 -->
- Avoid `Any` type annotations — use `Union`, `Protocol`, `TypeVar`, or schema-derived types for precision — Precise types catch bugs at type-check time and improve IDE autocomplete; when `Any` is unavoidable due to external constraints, document expected structure in docstrings
<!-- rule:238 -->
- Use `TypedDict` or dataclass instead of `dict[str, Any]` when structure is known — Enables static type checking, eliminates `cast()` calls, provides runtime validation, and self-documents expected structure

## Code Style

<!-- rule:552 -->
- Consolidate methods with duplicated logic using helpers, delegation, or type overloads — Reduces maintenance burden and prevents bugs from divergent implementations of the same control flow
<!-- rule:41 -->
- Order required fields before optional fields in dataclasses — Python requires non-default args before defaults — This prevents syntax errors since Python's dataclass implementation enforces that fields without defaults must precede fields with defaults

## General

<!-- rule:40 -->
- Use keyword-only params (place `*` after first 1-2 positional args) for optional/config parameters — Prevents breakage when adding parameters and eliminates positional argument confusion in functions with multiple optional parameters
<!-- rule:71 -->
- Prefix provider-specific fields with `{provider}_` in `ModelSettings` subclasses and profiles — prevents confusion about which provider supports which parameters — Clear provider namespacing prevents users from assuming a field works across providers when it's actually provider-specific, reducing API misuse
<!-- rule:894 -->
- Consolidate `try`/`except` blocks catching the same exception into one block — reduces duplication and simplifies control flow — When multiple operations throw the same exception type with similar handling logic, merging the try blocks prevents code duplication and makes error handling easier to maintain and update consistently.


<!-- /braindump -->
