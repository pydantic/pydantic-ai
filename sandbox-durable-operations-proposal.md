# Sandbox durable operations conversion

## Review patch

Commit `b3516e6fa5aa444d75a9ab856cb67a4bf05d5d1d` is the lifecycle conversion patch on top of the
merged durable-operations base. `sandbox-durable-operations-patch.diff` is the focused view of
what the sandbox-owned portion of PR #6492 becomes relative to `5a161b8dc`.

## Lifecycle operations

`AbstractCapability.create_sandbox` and `AbstractCapability.destroy_sandbox` are tier-1 durable
operations. Overrides inherit the marker and use the same generic capability dispatch as other
contributed operations. DBOS, Prefect, and Temporal do not have sandbox-specific lifecycle
activities, supplier indexes, or answer-final routing.

`get_sandbox` is permanently non-durable because it returns a live handle. It runs worker-side
after a durable boundary has carried a `SandboxRef` there. Decorating it is rejected when the
durability capability binds.

The `Agent.iter` whole-run preparation context is entered before model ID resolution, preserving
the ordering from PR #6492. Sandbox creation still occurs after the bootstrap model exists because
generic capability dispatch carries the model-backed `RunContext` through the durable unit.

A capability present when the agent binds may return a run-specific supplier from `for_run()`.
Worker recovery uses its stable capability ID and calls `for_run(ctx)` again before invoking the
lifecycle method. A new supplier passed only in `Agent.run(capabilities=...)` remains unsupported
because no durable operation was registered for it at bind time.

Sandbox teardown remains at least once. The destroy operation must be idempotent, and a provider
idle timeout remains the backstop when cancellation or infrastructure failure prevents teardown.

## Durable sandbox facade blocker

The proposed `SandboxOperationId(capability_id, method)` requires the stable ID of the capability
that can reopen a `SandboxRef`. The current `SandboxRef` carries only `provider` and `sandbox_id`.
The current capability contract discovers the recognizing capability by calling each
`get_sandbox(ctx, ref)` until one returns a live backend. That discovery is itself I/O and cannot
run in workflow code before an activity, step, or task is named.

Consequently, a sound facade needs one additional identity channel before framework sandbox
operations can be registered and named. Viable designs include carrying the stable capability ID
in `SandboxRef`, returning it alongside the ref from lifecycle resolution, or adding a pure
capability hook that claims a ref without opening it. Inferring it from `ref.provider` or capability
ordering is not sound because neither is required to equal the capability ID and recognition may
be dynamic.

Once that identity is available, `run`, `read_text`, `write_bytes`, and `exists` form the proof
subset. `read_bytes`, `write_text`, `stat`, `list_dir`, `make_dir`, `remove`, `resolve`, and
`working_dir` are mechanical repetitions. `start`, `backend`, and `fs` remain live-handle members
and must raise a `UserError` in durable container code. The public `DurableOperationId` union must
gain `SandboxOperationId`, and backend guidance must require a default match branch because the
union may grow in minor releases.
