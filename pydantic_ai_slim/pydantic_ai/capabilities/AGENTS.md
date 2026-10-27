# capabilities/ Guidelines

Capabilities are the composable home for cross-cutting agent behavior.

- Prefer a capability over a new `Agent` constructor kwarg when behavior contributes instructions, settings, tools, native tools, wrappers, lifecycle hooks, or event/history processing.
- Keep capabilities provider-agnostic unless the capability is explicitly modeling a provider-native feature; provider-specific facts belong in providers/profiles or provider-native tool classes.
- Preserve composition order. If a capability wraps model/tool/output/event behavior, check how it interacts with `CombinedCapability` and adjacent capabilities.
- Don't warn when two different capabilities contribute to the same thing, such as model settings. Composition order decides the result, and the user chose the order.
- Don't warn when two capabilities that share an `id` disagree. `combine` resolves the disagreement by its documented rule, or raises `UserError` when no rule fits. The default merge unions collections and otherwise takes the later value; a capability whose fields must agree overrides `combine` and raises, as harness `SubAgents` does.
- For user-facing capabilities, update docs and examples so users discover the capability as the primary API, not an implementation detail.
- Check durable execution, agent specs, and serialized configuration before adding non-serializable state or hidden runtime dependencies. Anything a capability reads from `RunContext` at tool-call time must survive the durable boundary: inside a Temporal activity an uncarried field raises, reads as `None`, or reads as a stated default — `tool_manager` reads as `None`, so `ctx.tools` empties to `{}` instead of raising — and state keyed by `id(...)` or another process-local address misses entirely, because the object it keyed lives in the workflow process. `_GUARDED_FIELDS`, `_NONE_UNLESS_ATTACHED` and `_DEFAULTED_UNLESS_CARRIED` in `durable_exec/_run_context.py` say which a field is (`validation_context` is the exception: it's rebuilt from the agent on first read); prove the behavior with a test in `tests/durable_exec/`.

## Prompt-cache prefix stability

- Follow [Message History & Prompt Caching](../../../agent_docs/message-history-and-caching.md): history is append-only across steps and runs, per-request nudges are enqueued `SystemPromptPart(scope='request')`s, and content goes in the channel that matches who wrote it.
- Keep tool definitions and their ordering stable across steps. Providers serialize tools before instructions and messages.
- Use date-only timestamps in instructions. Finer-grained timestamps invalidate an otherwise stable prefix.
