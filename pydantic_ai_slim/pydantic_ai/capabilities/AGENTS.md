# capabilities/ Guidelines

Capabilities are the composable home for cross-cutting agent behavior.

- Prefer a capability over a new `Agent` constructor kwarg when behavior contributes instructions, settings, tools, native tools, wrappers, lifecycle hooks, or event/history processing.
- Keep capabilities provider-agnostic unless the capability is explicitly modeling a provider-native feature; provider-specific facts belong in providers/profiles or provider-native tool classes.
- Preserve composition order. If a capability wraps model/tool/output/event behavior, check how it interacts with `CombinedCapability` and adjacent capabilities.
- Treat each lifecycle-hook dispatch as an activity epoch: it reads the active-capability set when the hook chain starts, and a hook that activates another deferred capability does not add it to the chain already in progress. Gate function tools in `prepare_tools`, not `before_model_request`; its result controls both what the model sees and what `ToolManager` can execute.
- For user-facing capabilities, update docs and examples so users discover the capability as the primary API, not an implementation detail.
- Check durable execution, agent specs, and serialized configuration before adding non-serializable state or hidden runtime dependencies. Anything a capability reads from `RunContext` at tool-call time must survive the durable boundary: inside a Temporal activity an uncarried field raises, reads as `None`, or reads as a stated default — `tool_manager` reads as `None`, so `ctx.tools` empties to `{}` instead of raising — and state keyed by `id(...)` or another process-local address misses entirely, because the object it keyed lives in the workflow process. `_GUARDED_FIELDS`, `_NONE_UNLESS_ATTACHED` and `_DEFAULTED_UNLESS_CARRIED` in `durable_exec/temporal/_run_context.py` say which a field is; prove the behavior with a test in `tests/durable_exec/`.
