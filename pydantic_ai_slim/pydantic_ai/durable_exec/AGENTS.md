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
from `get_durable_operation_backend`. Do not copy the model, toolset, event, or capability-operation
collection machinery into the integration.

Choose the backend tier from the engine SDK's execution model:

- Subclass `CallableOperationBackend` when the SDK accepts an async callback at invocation time.
  Implement `execute` to run that callback in one named durable unit. The base owns parameter and
  result transport, cache identity, naming, and config resolution.
- Subclass `RegisteredOperationBackend` when handlers must be registered before a worker starts.
  Implement `register` to return the bound caller and all SDK registration handles. Expose those
  registrations from the engine capability without rebuilding or reordering them. The base binds
  the four model operations during agent assembly, so `registrations()` is complete before worker
  start; pass the collected registrations to the engine SDK when creating the worker.

When durable units can run in another process (Temporal activities, Render child tasks), give the
backend a `ParameterTransports` subclass that moves the public `*Params` types across the boundary,
and serialize and rebuild their `RunContext` with `DurableRunContext` (subclass it for
engine-specific wording or extra fields). Do not copy its field lists, guards or workspace
restoration into the integration.

Set the public `engine_spec` once using `DurabilityEngineSpec`. Declare the engine name, durable
unit noun, durable container noun, codec, unsupported runtime toolset kinds, wrapped toolset kinds,
toolset lifecycles, tool-call result upgrade policy, discovery policy, sequential tool policy, and
tool config key deliberately. Use `IDENTITY_CODEC` when the engine SDK serializes Python values
itself. Use `JSON_CODEC` when the integration must reduce values to JSON-compatible payloads before
journaling them.

Persisted operation names are compatibility data, independent of Python class names. Prefer
`JournalOperationNamer` when its convention fits. Otherwise implement `DurableOperationNamer` and
keep its output stable. Before refactoring agent, model, toolset, or capability identity, pin the
complete name set using the `tests/durable_exec/test_durable_exec_compat.py` pattern. Never update
those pins as part of an implementation rename unless the migration of in-flight executions is
intentional and reviewed.

Resolve base and per-tool configuration with `OperationConfigRole` and `DurableOperationId`.
Handle known ID variants and retain a default branch because the public union grows in minor
releases. The union includes model requests, suspended-response
cancellation, message compaction, event handling, discovery, validation, calls, and
`CapabilityOperationId`. Capability methods marked with `@durable_operation` arrive through the
same backend and config resolver as framework operations, so do not maintain a second registration
path for them.

A toolset the run holds reaches the durable unit through the run context, so an engine that
serializes it (or dispatches the unit elsewhere) sees none and the unit enters its own, as every unit
did before. When a unit does reach one, it enters that toolset on first use and leaves it entered for
the run, which the durable container closes: that is the deliberate exception to the rule below, and
it is what lets an `MCPToolset` hold one session per run instead of one per unit. Declare it with the
`'enter-in-durable-unit'` lifecycle, which is only for engines whose units run in the container's own
process; connecting still happens inside a unit, so the engine retries it. A `DynamicToolset` built
with `per_run_step=False` is held the same way, except that the run resolves it first: that factory
runs in container code, so it carries the same determinism requirement as every other factory that
runs there.

Workspaces are routed by the base, not by engines: when a construction-time capability overrides
`get_workspace`, `_bind_workspace_operation` binds one operation, `CapabilityOperationId('workspace',
operation='call')`, configured and retried like any capability operation (bind nothing otherwise, so
agents without a supplier keep their pinned names). `_prepare_workspace` installs a `DurableWorkspace`
around the selected workspace inside the container; each of its calls, `ensure` included, is one
`WorkspaceCall`. The `WorkspaceEnsurer` companion runs `ensure` before the run body, with a locked lazy
fallback on first call, so parallel units share one environment. Inside a unit, `_unit_workspace`
gives the call the plain workspace: the live one in-process, or on Temporal the one rebuilt from the
serialized ref. Expected workspace errors cross as data and are re-raised with their original type;
Temporal lists `WorkspaceError` as a workflow-failure type; an uncaught builtin error keeps Temporal's
default and fails the workflow task.

Assume a durable unit may execute more than once if the process fails after the side effect but
before its checkpoint commits. Document the engine's guarantees and require idempotency or expose
an engine-native at-most-once option where available. Keep workflow-side code deterministic. Enter
and close toolset resources according to `engine_spec.toolset_lifecycles`, including failure and cancellation
paths, and verify that resources created inside a durable unit do not escape it. Test replay,
teardown, control-flow exceptions, persisted-output upgrades, and behavior outside the durable
context in the engine's own suite.

## Core changes that engines depend on

`scripts/durable_exec_contract.json` lists the core definitions the durable engines import,
override, read or project, each with the engine sites that depend on it. A manifest hit means your
core change touches one of those definitions. Open each listed site and confirm it still holds:
serialized field lists, overridden signatures, projection fields. Add a workflow-level test where
the engine's runtime behavior depends on the change. A missing entry does not prove an engine is
unaffected: the generator's docstring lists the dependency forms and the modules the walk does not
record.

An entry whose only kind is `annotation_only` is a definition the engines name only in annotations,
such as a type in a mirrored signature. It stays in the manifest, and `gates` leaves it out of review
routing.

CI labels a pull request `needs-durable-exec-review` while its changed lines touch an entry for which
`gates` holds, and removes the label once a push no longer does. A changed line counts for the
innermost definition around it, a comment, docstring or blank line for none, and a removed or renamed
core file for all of its entries; `touched_entries` in the generator has the rules. For such a pull
request, CI Review gets this file and the touched entries with their engine sites, and checks each
site. The label job (`durable-exec-label` in `.github/workflows/bots.yml`) and CI Review run the
generator and the manifest from the base branch, never the pull request's copies, so an edit to
either takes effect only once merged. To list the touched entries yourself, run
`uv run python scripts/durable_exec_contract.py --touched <owner/repo> <pr> <base sha> <head sha>`.

`tests/durable_exec/test_durable_exec_contract_manifest.py` fails when the manifest is stale.
Regenerate it with `uv run python scripts/durable_exec_contract.py --write` and commit the diff.
Read the diff: a new entry is a new dependency an engine took on core.

`tests/durable_exec/test_workflow_side_determinism.py` fails when a module in its workflow-side set
directly calls a clock, a random source, a sleep, a thread, a task-completion wait, an environment
read or I/O that its `_DENYLIST` names. Prefer the replay-safe alternative the durability capability
supplies. For a delay, await `_agent_graph_sleep`, which Temporal and DBOS route through
`set_agent_graph_sleep` (Prefect and the Harness engines do not yet:
[#11178](https://github.com/pydantic/pydantic-ai/issues/11178)). When the call is safe, mark it
with `# workflow-nondeterminism-ok: <reason>` and name why replay cannot diverge. The test
docstring is the reference for the workflow-side modules, the pragma rules, and what the test does
not see, which needs a workflow-level test.
