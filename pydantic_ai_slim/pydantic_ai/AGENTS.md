<!-- braindump: rules extracted from PR review patterns -->

# pydantic_ai_slim/pydantic_ai/ Guidelines

## API Design

<!-- rule:267 -->
- Store provider-specific metadata in `provider_details` or `provider_metadata`, not in string IDs, URIs, or semantic fields like `id` — Keeps main content fields (`content`, `args`) normalized and consistent across providers, preventing semantic field overloading and brittle string-encoded metadata
<!-- rule:716 -->
- Configure provider-specific API features (like `thinking_field_name`) in `Provider.model_profile()`, not in model profile functions — model profiles should contain only model-intrinsic characteristics — Maintains separation of concerns between model characteristics (consistent across providers) and provider API implementation details (varies by provider), preventing profile pollution and enabling proper provider abstraction
<!-- rule:17 -->
- In `_otel_*.py` modules, implement only what's in the spec — no custom extensions or mixing standards — Prevents divergence from external standards like OpenTelemetry, ensuring compatibility with spec-compliant tooling and avoiding confusion between official and custom conventions
<!-- rule:811 -->
- Always set `provider_name` when setting provider-specific fields (`id`, `signature`, `provider_details`, file IDs) on message parts (`ThinkingPart`, `ToolCallPart`, `TextPart`, etc.) — Maintains the architectural invariant that objects without `provider_name` are portable across providers, preventing accidental cross-provider incompatibilities in both streaming and non-streaming paths
<!-- rule:266 -->
- In serialization methods (e.g., `dump_messages`/`load_messages`), preserve provider-specific fields (`provider_details`, `provider_name`, `id`, `signature`) in dedicated metadata structures — prevents data loss in round-trip conversions and maintains provider-specific behavior — Encoding provider data into existing string fields (like `id: 'openai:123:xyz'`) loses structure after JSON deserialization; dedicated metadata fields ensure full round-trip preservation and support conditional logic that depends on provider-specific information
<!-- rule:415 -->
- Follow `ModelSettings` architecture: common options in base class, provider-specific options in provider subclasses (e.g., `CohereEmbeddingSettings`) with provider prefix (e.g., `cohere_`). Support both generic params and prefixed overrides with documented precedence, map standard params to provider-specific ones automatically, keep prefixed versions for backward compatibility — Prevents base class pollution with provider-specific options, maintains consistent API across providers, ensures backward compatibility when migrating from prefixed to generic parameters
<!-- rule:987 -->
- Use `WrapperToolset` subclasses for cross-cutting toolset behavior — compose functionality instead of modifying base classes or individual toolsets — Prevents duplication and fragile inheritance; allows stacking behaviors like approval, deferred loading, or search without touching every toolset implementation
<!-- rule:72 -->
- Promote settings to base `ModelSettings` when 2-3+ providers support them — avoids duplicate provider-prefixed fields and maintains backward compatibility — Reduces API duplication, prevents fragmentation across provider SDKs, and creates a consistent interface while preserving existing provider-prefixed fields for backward compatibility

## Code Style

<!-- rule:552 -->
- Consolidate methods with duplicated logic using helpers, delegation, or type overloads — Reduces maintenance burden and prevents logic drift when the same control flow appears in multiple methods
<!-- rule:14 -->
- Inline single-use helper methods that only wrap property access or delegate calls — reduces indirection and improves readability — Unnecessary abstraction layers make code harder to trace and understand without providing reuse, clarity, or meaningful encapsulation benefits
<!-- rule:345 -->
- Extract shared activity logic, types, and patterns to parent classes or utilities in `pydantic_ai/durable_exec/temporal/` — Prevents duplication across wrapper classes like `TemporalFunctionToolset` and `TemporalMCPServer`, ensuring consistent validation and easier maintenance
<!-- rule:21 -->
- Extract model profile logic into dedicated `{provider}_model_profile()` functions in `profiles/{provider}.py` — keeps provider classes thin and profile logic testable/reusable — Separating profile logic from provider classes prevents bloat, makes profiles independently testable, and allows reuse across different provider implementations.

## General

<!-- rule:468 -->
- Document provider-specific feature support in docstrings with `Supported by:` section — provider name and link on same line — Standardizes provider capability documentation so developers can quickly identify compatibility across Anthropic, OpenAI, etc.

<!-- /braindump -->
