# durable_exec/ Guidelines

Durable execution integrations are first-class compatibility targets.

- Treat Temporal, DBOS, Prefect, Restate, and similar engines as compatibility checks for core agent semantics, not peripheral adapters.
- Preserve run context, dependencies, message history, retries, model/profile selection, and toolset lifecycle across durable boundaries.
- Avoid hidden ordering assumptions, non-serializable state, and runtime-only closures unless the durable wrapper explicitly owns them.
- Prefer generic capabilities/toolsets/models extension points over engine-specific escape hatches.
- When changing graph/tool/output/streaming/MCP behavior, check whether durable wrappers need matching updates and add workflow-level tests where external runtime behavior matters.
- Sync stream (`run_stream_sync`) inherits via `self.run_stream`; missing wrapper override is intentional — workflows are async and SyncStreamBridge rejects a running event loop.

## Building a new durable execution engine

Build new integrations on the public `pydantic_ai.durable_exec` surface. Subclass
`BaseDurabilityCapability` for the agent-facing capability and provide a `DurableOperationBackend`
from `_build_operation_backend`. Do not copy the model, toolset, event, or capability-operation
collection machinery into the integration.

Choose the backend tier from the engine SDK's execution model:

- Subclass `CallableOperationBackend` when the SDK accepts an async callback at invocation time.
  Implement `_execute` to run that callback in one named durable unit. The base owns parameter and
  result transport, cache identity, naming, and config resolution.
- Subclass `RegisteredOperationBackend` when handlers must be registered before a worker starts.
  Implement `_register` to return the bound caller and all SDK registration handles. Expose those
  registrations from the engine capability without rebuilding or reordering them.

Set every declarative field deliberately: `_codec`, `_unsupported_runtime_toolset_kinds`,
`_wrapped_toolset_kinds`, `_toolset_lifecycles`, `_tool_call_result_upgrade_lenient`,
`_journal_discovery`, `_force_sequential_tools_in_durable_context`, and
`_allow_inline_mcp_in_durable_context`. Also define the durable unit and container nouns used in
errors. Use `IDENTITY_CODEC` when the engine SDK serializes Python values itself. Use `JSON_CODEC`
when the integration must reduce values to JSON-compatible payloads before journaling them.

Persisted operation names are compatibility data, independent of Python class names. Prefer
`JournalOperationNamer` when its convention fits. Otherwise implement `DurableOperationNamer` and
keep its output stable. Before refactoring agent, model, toolset, or capability identity, pin the
complete name set using the `tests/test_durable_exec_compat.py` pattern. Never update those pins as
part of an implementation rename unless the migration of in-flight executions is intentional and
reviewed.

Resolve base and per-tool configuration with `OperationConfigRole` and `DurableOperationId`.
Exhaustively handle every ID variant; the union includes model requests, suspended-response
cancellation, message compaction, event handling, discovery, validation, calls, and
`CapabilityOperationId`. Operations declared by capabilities arrive through the same backend and
config resolver as framework operations, so do not maintain a second registration path for them.

Assume a durable unit may execute more than once if the process fails after the side effect but
before its checkpoint commits. Document the engine's guarantees and require idempotency or expose
an engine-native at-most-once option where available. Keep workflow-side code deterministic. Enter
and close toolset resources according to `_toolset_lifecycles`, including failure and cancellation
paths, and verify that resources created inside a durable unit do not escape it. Test replay,
teardown, control-flow exceptions, persisted-output upgrades, and behavior outside the durable
context in the engine's own suite.
