# Async & Concurrency

> Rules for `asyncio`/`anyio` code, most of them paid for by a real bug in this repo

**When to check**: Whenever you write or review code that spawns a task, opens a task group or cancel scope, creates a lock/event/stream, writes an async context manager or async generator, crosses a thread or event-loop boundary, or tests any of those

Most rules name the symbol, file, or test that proves them; a rule with no anchor is judgment, not evidence. Check the anchor before you argue with a rule, and before you extend one — the usual way to get this wrong is to state a true general mechanism more broadly than the code supports, or to describe a design that was proposed but never shipped. If the code has moved, update the rule.

Before adding any of this, name the scope that guarantees teardown for every task, scope, lock, stream, span, and connection you create. "The garbage collector" or "the caller remembers" is a bug, not a design.

## Rules

### Ownership

- Iteration owns no teardown. `async for ... break` doesn't close an async iterator, and asyncio finalizes an abandoned async generator in *a different task, under an unrelated copied context* (`loop._asyncgen_finalizer_hook`), so nothing may depend on that cleanup having run. A task born during iteration must still be stored on and drained by the enclosing context manager — `RealtimeSession._start_pump` is lazy and `__aexit__` drains it (`realtime/_session.py`).
- Prefer a task group, whose `async with` encloses everything the children touch, over loose tasks. Avoid `asyncio.gather(..., return_exceptions=False)` when one failure should stop the batch — it propagates the first failure while siblings keep running. Use `_utils.gather`, which cancels and drains them. `return_exceptions=True` is fine for a cleanup-only drain (`_utils.cancel_and_drain`).
- Use `asyncio.create_task` only for a task that must outlive the frame starting it. Then keep the handle where the owner can reach it at teardown (`SyncStreamBridge._pump_tasks`, `RealtimeSession._background_tasks`, both added on create and discarded in a done-callback) and cancel *and await* it there via `_utils.cancel_and_drain` — `cancel()` requests, it does not tear down. Pass `name=` when there are many of a kind (`_tool_execution.py` names each by tool).

### Cancellation: level vs. edge

`asyncio` is edge-triggered: catching the delivered `CancelledError` resumes execution until something cancels again. A cancelled `anyio` scope is level-triggered: every later cancellation checkpoint raises again unless shielded, so async cleanup inside one cannot finish. Know which you are in before writing cleanup.

- Shield cleanup that must complete under an outer `anyio` cancel — `_utils.cancel_and_drain` is the ready-made task drain. Your `finally` and each child's cleanup are unprotected unless they shield themselves. Don't shield task-group exit alone: `TaskGroup.__aexit__` already shields the parent's *remaining* wait once the first cancel reaches it (anyio #695).
- Keep cancellation bookkeeping at one edge: `RunCancellation.resolve()` consumes only controller-issued cancels via `Task.uncancel()`, while `_utils.raise_if_cancelling()` separately re-asserts a cancel a completed step swallowed (`_cancel.py`, `run.py`). `Task.cancelling()`/`uncancel()` are 3.11+: on 3.10 `resolve()` can't disambiguate the race so first-party wins, while `raise_if_cancelling()` is a bare `return`, so an absorbed *external* cancel is lost outright. Under Trio `RunCancellation.bind()` never binds, so first-party cancellation doesn't arm at all.
- One owner per deadline. `FunctionToolset.call_tool` takes the per-tool timeout when set, else its toolset/agent fallback, and enforces exactly one scope — so a longer per-tool value *replaces* the agent default instead of being capped by it. `ToolManager` adds no timeout; MCP, custom, and external toolsets own deadlines at their own transport.
- A deadline can't interrupt blocking sync work: `anyio.to_thread.run_sync` shields its wait, so an enclosing `fail_after` returns late and raises only if a checkpoint follows inside the scope. `_utils.abandon_threads_on_cancel()` lets the wait time out, but the worker still runs to completion and its result is discarded (`toolsets/function.py`, `capabilities/hooks.py`).
- Enter and exit an `anyio.CancelScope` in the same task, in strict LIFO order. A scope may span a `yield` only if one persistent task performs every resume and finalization — a per-item `anext()` bridge can straddle tasks. anyio checks this at scope exit, not at the yield — `_sync_stream.py`'s module docstring names the exact error, and `capabilities/process_event_stream.py` hit it too.
- Unwrap only an accidental single-child `BaseExceptionGroup` before a public API; preserve a genuine multi-failure group (`_utils.gather`). On 3.10 the name comes from the backport re-exported by `_utils`, not builtins, and `except*` is 3.11+ *syntax* that the backport cannot provide — match on `BaseExceptionGroup` and use `.split()`/`.subgroup()`.
- Give partial streamed parts a valid replay form — a cancel leaves them in history for the next request. Anthropic starts a `ThinkingPart(signature='')`, so its mapper requires a *truthy* signature and falls back to tagged text; an `is not None` guard there ships `signature=""` and earns a 400 (`test_anthropic_model_empty_thinking_signature_sent_as_text`).

### Threads and event loops

- Async work driven by a sync entry point stays on the caller's loop. The `BlockingPortal` implementation (#6199) was reverted (#6454) because pooled transports bind per connection; `SyncStreamBridge` keeps its owner and pump tasks on the caller's loop. Nested `run_sync()`/`run_stream_sync()` is rejected inside any callback dispatched through `_utils.run_in_executor` (`_utils.check_no_nested_sync_run()`) — make the callback async instead.
- Defer shared-object entry locks with the `_enter_lock` `cached_property` pattern (`agent/__init__.py`, `providers/__init__.py`, `mcp.py`, `models/fallback.py`). First use binds the lock to that loop and backend; deferring keeps it out of `__init__` and Temporal's sandbox. It does not make an entered object reusable from a later loop.
- Sync callbacks are dispatched off-thread, and that costs `ContextVar` writes — an accepted tradeoff, since a `def` callback is assumed to block. `_utils.run_in_executor` copies the caller's context *in* (reads work) and discards writes, and `asyncio.get_running_loop()` raises there, silently breaking libraries that keep state in context variables such as tracing and logging. Make the callback `async` if it needs any of that. Covers `def` tools, output functions and output validators, `system_prompt`/`instructions` functions, hooks, and history processors (`docs/tools-advanced.md`, `docs/hooks.md`; `test_sync_before_run_hook_contextvar_does_not_propagate`).
- Not every sync callback is dispatched. `Tool.prepare`, `PreparedToolset.prepare_func`, `FallbackModel` handlers, and model-id resolvers are awaited inline via `_utils.await_maybe`: they block the loop, their `ContextVar` writes stick, and `_utils.check_no_nested_sync_run()` never fires for them. `_utils.disable_threads()` (Temporal, emscripten) puts *every* callback in that lane. Choose the lane deliberately when adding a sync-callable extension point.

### Locks

- Ask whether you need a lock at all before adding one. A critical section with no `await` in it is already atomic against other tasks on the same loop, so a problem you can restructure to compute first and mutate in one unbroken stretch needs no lock. You need one when the section suspends, or when a worker thread touches the same state — the no-`await` argument covers neither.
- `async with` on a shared object is not concurrency-safe by default. Guard entry with `_enter_lock` plus an entered-count (`_entered_count` in `providers/__init__.py`, `_running_count` in `mcp.py`), or give each task its own instance.

### Testing it

- Assert the concurrency fact itself, not the output it produces. Fixture and interpreter-global state can quietly remove the trigger and leave the test green — use a clean subprocess when event-loop policy or similar global state is the subject.
- Exercise the public syntax — `async with`, `async for ... break` — not `__aenter__` or `agen.aclose()` by hand. The realtime early-break tests called `aclose()` themselves and passed while the shipped syntax leaked its tasks; `test_early_break_cancels_pump` is the version that actually exercises it.
- Order steps with `Event`s, not sleeps, and wait on them with a module-level `READINESS_WAIT_TIMEOUT` (`tests/test_agent.py`, `tests/test_run_cancellation.py`), not a one-second timeout — short waits flake under `xdist` (<https://github.com/pydantic/pydantic-ai/issues/5399>).
- Prove ownership directly: diff `asyncio.all_tasks()` for ordinary leak checks; for a GC fallback, capture the owner and pump tasks, drop the last strong reference, `gc.collect()`, then assert both are done (`test_sync_stream_bridge_finalizes_with_unclosed_iterator`).
- Reach the real trigger. Loop affinity needs one async client reused across consecutive sync entry points plus assertions on the actual loop identities (`tests/test_sync_stream_loop_affinity.py`); level-cancellation behavior needs a real outer `anyio` cancel scope, not a bare `CancelledError` raise.
