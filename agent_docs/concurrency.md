# Async & Concurrency

> Rules for `asyncio`/`anyio` code, almost all of them paid for by a real bug in this repo

**When to check**: Whenever you write or review code that spawns a task, opens a task group or cancel scope, creates a lock/event/stream, writes an async context manager or async generator, crosses a thread or event-loop boundary, or tests any of those

## Rules

### Ownership

- For every task, scope, lock, stream, span, and connection you create, name the scope that guarantees it is torn down. "The garbage collector" or "the caller remembers" is a bug, not a design.
- Iteration owns no teardown. `async for ... break` does not close a generic async iterator, and when asyncio later finalizes an abandoned async generator its cleanup runs in a fresh task with a fresh context — so anything whose correctness depends on that cleanup having already happened is broken. A task first created during iteration must therefore be stored on, and drained by, the surrounding async context manager. `RealtimeSession` starts its pump lazily on first use and `__aexit__` cancels and drains it; iterator cleanup only resets iterator-local state.
- Prefer structured concurrency: an `anyio` task group whose `async with` encloses everything the children touch. Reach for something looser only when that structure genuinely doesn't fit.
- Avoid `asyncio.gather(..., return_exceptions=False)` when one failure should stop the batch — it propagates the first failure while the siblings keep running (it does retrieve their later exceptions). Use `_utils.gather`, which cancels and drains them. `asyncio.gather(..., return_exceptions=True)` is fine for a cleanup-only drain, as in `_utils.cancel_and_drain`.
- Use `asyncio.create_task` only when the task must outlive the frame that starts it. Then hold a strong reference (the task registry is a `WeakSet`), pass `name=`, and cancel *and await* it through `_utils.cancel_and_drain` in the owner's `finally` — `cancel()` alone requests, it does not tear down.

### Cancellation: level vs. edge

- Know which one you're in. `asyncio` is edge-triggered: catching the delivered `CancelledError` lets execution continue until something cancels again. An effectively cancelled `anyio` scope is level-triggered: every later cancellation checkpoint raises again unless shielded, so async cleanup inside one needs a shield to finish.
- Shield cleanup that must complete under an outer `anyio` cancel; `_utils.cancel_and_drain` is the ready-made task drain. Your own `finally` and each child's cleanup are unprotected unless they shield themselves. Don't add a shield around task-group exit alone — once cancellation reaches that wait, the parent's remaining wait for children is already protected.
- Keep cancellation bookkeeping at one edge. `RunCancellation.resolve()` consumes only controller-issued cancels via `Task.uncancel()`; `_utils.raise_if_cancelling()` separately re-asserts a cancellation a completed step swallowed. Both degrade on Python 3.10, where `Task.cancelling()`/`Task.uncancel()` don't exist, and the backstop is a no-op under Trio.
- One owner per deadline. For agent-owned function tools, `FunctionToolset.call_tool` picks the per-tool timeout when set and otherwise its toolset/agent fallback, then enforces exactly one scope — so a longer per-tool value replaces the agent default rather than being capped by it. `ToolManager` adds no timeout; MCP, custom, and external toolsets own deadlines at their own transport.
- A deadline cannot interrupt blocking sync work. `anyio.to_thread.run_sync` shields its wait by default, so a `fail_after` around only that await can return late without raising. `_utils.abandon_threads_on_cancel()` lets the wait time out, but the worker still runs to completion and its result is discarded (`toolsets/function.py`, `capabilities/hooks.py`).
- When a cancellation handler needs run state, call `RunCancelled.from_cancellation()` and re-raise.
- Enter and exit an `anyio.CancelScope` in the same task and in strict LIFO order. A scope may span a `yield` only when one persistent task performs every resume and finalization — a per-item `anext()` bridge can straddle tasks. anyio checks this at scope exit, not at the yield.
- Unwrap only an accidental single-child `BaseExceptionGroup` before it reaches a public API; preserve a genuine multi-failure group. On Python 3.10 the name comes from the backport re-exported by `_utils`, not builtins.
- Give partial streamed parts a valid replay form — cancellation leaves them in history for the next request. Anthropic starts a `ThinkingPart(signature='')`, so its mapper requires a *truthy* signature and falls back to tagged text for incomplete thinking; an `is not None` guard there sends `signature=""` back and earns a 400.

### Threads and event loops

- Async work driven by a sync entry point stays on the caller's loop. The `BlockingPortal` implementation was reverted because pooled transports bind per connection, and `SyncStreamBridge` keeps its owner and pump tasks on the caller's loop. Nested `run_sync()`/`run_stream_sync()` is rejected inside any synchronous callback Pydantic AI dispatches (`_utils.check_no_nested_sync_run()`) — make the callback async and await the nested run.
- Preserve `_sync_stream._run_task_to_completion()` when driving the caller's stopped loop: a completed future holding `KeyboardInterrupt`/`SystemExit` can leave `run_until_complete()` spinning, and an interrupted drive can strand a queued `loop.stop()` that then aborts the next unrelated sync call.
- Cross-loop reuse can duplicate an attempt, not just hang. A stale pooled connection can deliver a request before its read fails; if the SDK retries on a fresh connection the server may receive it twice. A live loop is not proof that pooled transports belong to it.
- Defer shared-object entry locks with the `_enter_lock` `cached_property` pattern (`agent/__init__.py`, `providers/__init__.py`, `mcp.py`, `models/fallback.py`). First entry selects the anyio backend, and deferring avoids constructing the lock during import, object initialization, or Temporal sandboxing. It is not a promise that an entered object is reusable from a later loop.
- `ContextVar` propagation is copy-down, not write-back: task creation and `anyio.to_thread.run_sync` copy the caller's values, child and worker writes don't escape, and raw threads or bare executors copy nothing. Sync hooks dispatched to a worker therefore lose `ContextVar` writes; inline `disable_threads` execution does not.

### Locks

- Design the lock out first. A critical section with no `await` in it is already atomic against other tasks on the same loop, so compute first and mutate in one unbroken stretch where you can. A lock is only needed when the section suspends — or when the state is also touched from a worker thread, which a task-only argument doesn't cover.
- `async with` on a shared object is not concurrency-safe by default. Guard entry with the `_enter_lock` plus an entered-count (`_entered_count` in `providers/__init__.py`, `_running_count` in `mcp.py`), or give each task its own instance.

### Testing it

- Prove the regression test fails against the unfixed implementation, then assert the concurrency fact itself rather than the output it produces. Fixture and interpreter-global state can quietly remove the trigger and leave the test green; use a clean subprocess when event-loop policy or similar global state is the subject.
- Exercise the public syntax — `async with`, `async for ... break` — not `__aenter__` or `agen.aclose()` by hand. The realtime early-break tests called `aclose()` themselves and passed while the shipped syntax leaked its tasks.
- Order steps with `Event`s, not sleeps, and wait on them with the module-level `READINESS_WAIT_TIMEOUT` rather than a one-second timeout — short waits flake under `xdist` (<https://github.com/pydantic/pydantic-ai/issues/5399>).
- Prove ownership directly: diff `asyncio.all_tasks()` for ordinary leak checks; for a GC fallback, drop the last strong reference, `gc.collect()`, let scheduled finalization run, then assert the captured owner and pump tasks are done (`tests/test_streaming.py`).
- Reach the real trigger. Loop affinity needs one async client reused across consecutive sync entry points plus assertions on the actual loop identities (`tests/test_sync_stream_loop_affinity.py`); level-cancellation behavior needs a real outer `anyio` cancel scope, not a bare `CancelledError` raise.
