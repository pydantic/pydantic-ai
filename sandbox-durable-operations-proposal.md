# Sandbox lifecycle as contributed durable operations

## Conclusion

The experiment supports the narrow thesis, but not the broad one.

For capabilities installed on a normal `Agent`, contributed operations replace the durability capability's sandbox impersonation. Each capability that overrides `create_sandbox` or `destroy_sandbox` is now registered automatically. The normal capability walk remains in workflow code, while each supplier hook crosses the engine boundary independently. This works with DBOS, Prefect, and Temporal.

It does not make all of PR #6492's sandbox durability machinery unnecessary. The deprecated durable-agent wrappers do not install a `BaseDurabilityCapability`, so they cannot use contributed operations. They still need static sandbox discovery and rejection. `_sandbox.py` also retains live-handle safety checks. Runtime-created sandbox suppliers cannot be registered when the agent is bound, and therefore remain unsupported.

There is also an integration regression in this proposal branch: sandbox preparation begins after model resolution in the newer `Agent.iter` implementation. PR #6492 deliberately moved the preparation boundary outside model resolution. `test_wrap_entire_run_observes_model_resolution_error` now records no lifecycle entry for a model-resolution failure. This branch is useful evidence, but is not merge-ready until that ordering is restored.

## Lifecycle rule

`create_sandbox` and `destroy_sandbox` are tier 1 operations. An override is inherently durable and is registered without a decorator. `get_sandbox` is permanently non-durable because it returns a live connection. Decorating it fails while binding the agent with an error that states that a live sandbox connection cannot cross a durable boundary.

This matches the data boundary already expressed by the sandbox API:

* create and destroy exchange a credential-free, journalable `SandboxRef` descriptor;
* get turns that descriptor into a process-local live handle inside the durable unit that needs it.

Every durable supplier needs an explicit stable capability `id`, because that id is part of persisted operation identity.

## What happened to PR #6492's machinery

| Machinery | Result | Reason |
|---|---|---|
| `durable_exec/_sandbox.py` | Kept, reduced in responsibility | Deprecated DBOS and Temporal agent wrappers still use `contributes_sandbox` to reject suppliers they cannot route. The module also contains live-sandbox guards. The durability capability no longer uses it to discover or impersonate a supplier. |
| `_supports_run_owned_sandbox` | Deleted | All three capability-based engines route tier 1 lifecycle hooks through their ordinary contributed-operation backend. There is no longer a Temporal-only opt-in. |
| `_create_sandbox_answer_is_final(ctx)` | Deleted | The durability capability no longer joins the supplier walk. A supplier's `None` has its ordinary meaning and the walk continues. A returned ref stops the walk once, without suppressing a second execution. |
| Deprecated wrapper rejection | Kept | Those wrappers do not install the contributor registry or operation dispatcher. Removing the check would execute provisioning in workflow code. Contributed operations do not replace that path. |

The dedicated Temporal sandbox activity methods, supplier-index routing, and durability capability `create_sandbox` and `destroy_sandbox` impersonation were removed.

## Conflict picture

The merge produced 13 real conflicts, recorded before resolution in `/tmp/sandbox-merge-conflicts.txt`:

* `pydantic_ai_slim/pydantic_ai/agent/__init__.py`
* `pydantic_ai_slim/pydantic_ai/agent/abstract.py`
* `pydantic_ai_slim/pydantic_ai/agent/wrapper.py`
* `pydantic_ai_slim/pydantic_ai/durable_exec/_base.py`
* `pydantic_ai_slim/pydantic_ai/durable_exec/dbos/_agent.py`
* `pydantic_ai_slim/pydantic_ai/durable_exec/dbos/_durability.py`
* `pydantic_ai_slim/pydantic_ai/durable_exec/prefect/_agent.py`
* `pydantic_ai_slim/pydantic_ai/durable_exec/prefect/_cache_policies.py`
* `pydantic_ai_slim/pydantic_ai/durable_exec/prefect/_durability.py`
* `pydantic_ai_slim/pydantic_ai/durable_exec/temporal/_agent.py`
* `pydantic_ai_slim/pydantic_ai/durable_exec/temporal/_durability.py`
* `tests/test_prefect.py`
* `tests/test_temporal.py`

The three engine durability files and `_base.py` were semantic conflicts, not mechanical overlaps. Both branches replace operation registration, execution boundaries, toolset wrapping, and capability composition. The agent conflict was similarly architectural because PR #6492 and the durable-operations branch start lifecycle scopes at different points in the run.

Merge order will not avoid the hard work. If #6492 merges first, #6696 must port its operation registry through the sandbox lifecycle refactor. If #6696 merges first, #6492 should be rewritten directly in terms of tier 1 operations instead of merging its impersonation layer and deleting it afterward. The latter order should produce a smaller conceptual patch, but the `Agent.iter` preparation boundary still requires an explicit maintainer decision and implementation.

## Teardown guarantees

Contributed operations improve routing and engine coverage. They do not guarantee that the workflow reaches `destroy_sandbox`. All engines still need provider idle expiry or an external reaper as the backstop. Destroy implementations must tolerate an already-gone sandbox.

### Temporal

Create and destroy are activities. A completed create result is recorded in workflow history and is reused on replay. If a worker creates the external sandbox and crashes before activity completion is recorded, Temporal can execute create again. A create-or-reuse operation keyed by `ctx.run_id` is therefore still required.

If the workflow unwinds normally, destroy is scheduled as its own activity. A worker crash after the external deletion but before activity completion can cause a second destroy call. The hook catches and logs supplier exceptions so a cleanup failure does not replace the run result, matching PR #6492's behavior, but this also means the platform cannot retry a returned cleanup failure. Workflow termination or a path that never resumes can skip destroy entirely.

### DBOS

Create and destroy are steps. A completed step result is recovered from the journal. A crash after an external side effect but before step completion is persisted can repeat either operation. Create must be create-or-reuse and destroy must be idempotent. Recovery can continue from a recorded create and eventually schedule destroy, but termination or a workflow that is never recovered provides no cleanup guarantee.

### Prefect

Create and destroy are tasks with persisted cache identities. A task crash between its external side effect and persisted completion can repeat the side effect. A later flow execution may reuse a cached create, but cache reuse is not a cleanup guarantee. A crashed or cancelled flow that never resumes can skip destroy. Destroy can also run twice around the side-effect-to-persistence crash window.

## Retry semantics

The existing `create_sandbox` guidance remains correct and necessary: provisioning should be idempotent create-or-reuse keyed by `ctx.run_id`.

PR #6492 put the entire supplier-resolution walk in one Temporal activity. This proposal gives every supplier its own operation. A supplier that declines with `None` and a later supplier that provisions are journaled separately, so each supplier can retry or replay independently. This is a cleaner ownership boundary, but it does not make external side effects exactly once. Each overriding supplier must meet the idempotency contract for the engine behavior described above.

## Limitations and regressions

* Per-run capability suppliers are rejected because contributed operations are registered when the agent is bound. Supporting them needs a dynamic registration or a stable generic dispatch operation.
* Deprecated durable-agent wrappers still cannot route contributed operations and retain their rejection behavior.
* Transparent wrapper forwarding is excluded from automatic registration so a wrapper does not falsely claim ownership of its wrapped supplier's operation. A complete wrapper-owned lifecycle design needs more work.
* Destroy is at least once if scheduled, not exactly once, and is not guaranteed to be scheduled.
* Cleanup exceptions are logged and swallowed. This preserves the original run result, but prevents engine retries for an ordinary supplier exception.
* The model-resolution preparation ordering regressed, as described above.

## Line-count comparison

These counts compare this proposal branch with `origin/sandbox-concept`. They include the contributed-operation prerequisite from #6696, so they describe the actual branch a reviewer would compare, not only the small lifecycle adaptation.

| File | Added | Removed |
|---|---:|---:|
| `docs/sandbox.md` | 18 | 10 |
| `capabilities/abstract.py` | 12 | 13 |
| `durable_exec/_base.py` | 1,367 | 127 |
| `durable_exec/_capability_operation.py` | 324 | 0 |
| `durable_exec/dbos/_durability.py` | 75 | 140 |
| `durable_exec/prefect/_durability.py` | 59 | 170 |
| `durable_exec/temporal/_durability.py` | 222 | 415 |
| `tests/test_capability_durable_operation.py` | 487 | 0 |
| `tests/test_dbos.py` | 280 | 43 |
| `tests/test_prefect.py` | 540 | 22 |
| `tests/test_temporal.py` | 474 | 28 |
| `tests/test_wrap_entire_run.py` | 2 | 2 |

`durable_exec/_sandbox.py` has no net comparison row because the retained file is identical to PR #6492 at this point. Its durability-layer consumers were removed elsewhere.

## Verification record

The focused capability, sandbox, wrapper, compatibility, and public-interface tests passed. The combined DBOS, Prefect, and Temporal suites ran with `-n auto --dist=loadgroup`: 583 passed, 2 skipped, and exactly 7 pre-existing Prefect failures remained. Targeted lifecycle tests passed on all three engines, including failed-run cleanup and Temporal cleanup-error logging. `command make lint` and `command make typecheck` passed.

The coverage gate did not pass. A focused run correctly failed because it did not execute the entire durable-execution surface. A subsequent full-suite xdist coverage run reached 99 percent test progress but produced broad unrelated concurrency failures and then stalled in the final Temporal group. It was interrupted after bounded waits. The partial combined report reached 86.07 percent for `*/durable_exec/*`, not the required 100 percent. This result must not be represented as successful coverage verification.
