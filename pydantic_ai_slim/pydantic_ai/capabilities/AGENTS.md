# capabilities/ Guidelines

Capabilities are the composable home for cross-cutting agent behavior.

- Prefer a capability over a new `Agent` constructor kwarg when behavior contributes instructions, settings, tools, native tools, wrappers, lifecycle hooks, or event/history processing.
- Keep capabilities provider-agnostic unless the capability is explicitly modeling a provider-native feature; provider-specific facts belong in providers/profiles or provider-native tool classes.
- Preserve composition order. If a capability wraps model/tool/output/event behavior, check how it interacts with `CombinedCapability` and adjacent capabilities.
- For user-facing capabilities, update docs and examples so users discover the capability as the primary API, not an implementation detail.
- Check durable execution, agent specs, and serialized configuration before adding non-serializable state or hidden runtime dependencies.

## Prompt-cache prefix stability

- Never mutate an already-cached prefix during a run. Prefix changes silently discard the provider's cached work.
- Put per-turn dynamic content in the user message, tool results, or after the last `CachePoint`. This preserves the reusable prefix.
- Keep tool definitions and their ordering stable across steps. Providers serialize tools before instructions and messages.
- Use date-only timestamps in instructions. Finer-grained timestamps invalidate an otherwise stable prefix.
