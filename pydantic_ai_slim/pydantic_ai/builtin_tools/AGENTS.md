# builtin_tools/ Guidelines

## Adding a new builtin tool

- Every builtin tool must have a corresponding capability extending `BuiltinOrLocalTool` — capabilities are the primary user-facing API for enabling tool features on agents; a builtin tool without a capability is undiscoverable for users working with the capabilities list
  - Vendor-specific tools: capability lives in the provider's model file (e.g., `XSearch` in `models/xai.py`)
  - Cross-provider tools: capability lives in `capabilities/` (e.g., `WebSearch`, `ImageGeneration`)
- When a provider's API has request-level parameters controlling raw tool output inclusion (e.g., xAI `include`, OpenAI `include`), expose the tool-specific ones as fields on the tool class — not just in model settings — users configuring `XSearchTool(...)` should discover all relevant options there; model settings remain as an alternative for backward compat
- Provider support must be documented in three places: the tool class docstring 'Supported by' list, `docs/builtin-tools.md` provider table, and field-level docstrings for provider-specific semantics
- When a tool field maps directly to a provider API field name, prefer that name — users may have provider docs open alongside pydantic-ai docs
- Validate mutual exclusivity and limits in `__post_init__` — fail fast with clear messages (e.g., 'Cannot specify both allowed_x_handles and excluded_x_handles')
- Builtin tool names in pydantic-ai must round-trip through provider APIs — if the API uses a different function name (e.g., xAI sends `x_keyword_search` not `x_search`), preserve the original name when replaying history
