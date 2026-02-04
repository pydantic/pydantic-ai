<!-- braindump: rules extracted from PR review patterns -->

# pydantic_ai/models/ Guidelines

## API Design

- Mirror patterns across provider implementations — ensures consistent APIs and prevents fragmentation — Users expect all providers to behave similarly; consistent patterns reduce learning curve and make providers interchangeable
- Expose provider-specific metadata via `provider_details` fields, always setting `provider_name` — Keeps the core API provider-agnostic while surfacing valuable metadata like logprobs, safety ratings, and token counts in a consistent, discoverable way
- Use `snake_case` versions of provider's official API parameter names in config classes — Ensures provider parameters match official documentation, making them discoverable and reducing confusion when users reference vendor docs

## General

- Silently ignore unsupported `ModelSettings` when docs already state compatibility — prevents warning noise — Users already know from documentation which models support which settings; runtime warnings are redundant and clutter logs unnecessarily
- Keep provider-specific code in provider modules (e.g., `models/openai.py`) not shared utils — Prevents coupling between providers and maintains clear module boundaries, making code easier to maintain and extend
- Prefix provider-specific config fields with provider name (e.g., `anthropic_cache_tools`, `openrouter_engine`) — Distinguishes provider-specific settings from generic cross-provider options, preventing configuration ambiguity

<!-- /braindump -->
