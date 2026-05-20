# capabilities/ Guidelines

Capabilities are the composable home for cross-cutting agent behavior.

- Prefer a capability over a new `Agent` constructor kwarg when behavior contributes instructions, settings, tools, native tools, wrappers, lifecycle hooks, or event/history processing.
- Keep capabilities provider-agnostic unless the capability is explicitly modeling a provider-native feature; provider-specific facts belong in providers/profiles or provider-native tool classes.
- Preserve composition order. If a capability wraps model/tool/output/event behavior, check how it interacts with `CombinedCapability` and adjacent capabilities.
- For user-facing capabilities, update docs and examples so users discover the capability as the primary API, not an implementation detail.
- Check durable execution, agent specs, and serialized configuration before adding non-serializable state or hidden runtime dependencies.
