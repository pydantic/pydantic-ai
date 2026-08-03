# native_tools/ Guidelines

## Adding a new native tool

- Native tools that represent a cross-provider feature should have a corresponding capability extending `NativeOrLocalTool` in `capabilities/` — capabilities are the primary user-facing API for enabling provider-adaptive tool features on agents
  - Local fallback (e.g., `WebSearch`, `WebFetch`): capability falls back to a function tool on providers without native support
  - Subagent fallback (e.g., `ImageGeneration`, `XSearch`): capability delegates to a subagent running another provider's model via `fallback_model`
- Keep provider-specific tools with no credible cross-provider abstraction as native tools wrapped in `NativeTool`; don't add a thin provider-agnostic capability until the feature has support or meaningful fallback semantics across providers
- When a provider's API has request-level parameters controlling raw tool output inclusion (e.g., xAI `include`, OpenAI `include`), expose the tool-specific ones as fields on the tool class — not just in model settings — users configuring `XSearchTool(...)` should discover all relevant options there; model settings remain as an alternative for backward compat
- Provider support must be documented in three places: the tool class docstring 'Supported by' list, `docs/native-tools.md` provider table, and field-level docstrings for provider-specific semantics
- When a tool field maps directly to a provider API field name, prefer that name — users may have provider docs open alongside pydantic-ai docs
- Validate mutual exclusivity and limits in `__post_init__` — fail fast with clear messages (e.g., 'Cannot specify both allowed_x_handles and excluded_x_handles')
- Native tool names in pydantic-ai must round-trip through provider APIs — if the API uses a different function name (e.g., xAI sends `x_keyword_search` not `x_search`), preserve the original name when replaying history
