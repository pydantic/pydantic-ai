# Async & Concurrency

Rules for `asyncio`/`anyio` code, ported from Pydantic AI core's guide
(pydantic/pydantic-ai#8097, `agent_docs/concurrency.md`) and re-anchored to
this repository. Most of the core rules were paid for by a real bug there; the
harness inherits the same runtime, so it inherits the same rules.

When to check: whenever you write or review code that spawns a task, opens a
task group or cancel scope, creates a lock, event, or stream, writes an async
context manager or async generator, spawns or waits on a subprocess, crosses a
thread or event-loop boundary, or tests any of those.

Most rules name the symbol, file, or test that proves them. A rule with no
anchor is judgment, not evidence. Check the anchor before you argue with a rule
and before you extend one: the usual way to get this wrong is to state a true
general mechanism more broadly than the code supports, or to describe a design
that was proposed but never shipped. If the code has moved, update the rule.
Anchors marked "core" live in `pydantic_ai`; read the installed package source,
not a contributor's checkout.

Before adding any of this, name the scope that guarantees teardown for every
task, scope, lock, stream, subprocess, span, and connection you create. "The
garbage collector" or "the caller remembers" is a bug, not a design.

## Rules

### Ownership

- Iteration owns no teardown. `async for ... break` does not close an async
  iterator, and asyncio finalizes an abandoned async generator in a different
  task under an unrelated copied context (`loop._asyncgen_finalizer_hook`), so
  nothing may depend on that cleanup having run. A task born during iteration
  must still be stored on, and drained by, the enclosing context manager (core:
  `RealtimeSession._start_pump` is lazy and `__aexit__` drains it).
- Prefer a task group, whose `async with` encloses everything the children
  touch, over loose tasks. `ShellToolset.run_command` reads stdout and stderr
  as two children of one group inside one `fail_after`, so a timeout cancels
  both readers together (`shell/_toolset.py`). Avoid
  `asyncio.gather(..., return_exceptions=False)` when one failure should stop
  the batch: it propagates the first failure while siblings keep running.
  `return_exceptions=True` is fine for a cleanup-only drain.
- Use `asyncio.create_task` only for a task that must outlive the frame that
  starts it, then keep the handle where the owner can reach it at teardown and
  cancel and await it there. `cancel()` requests, it does not tear down.
  `aws_lambda/_bridge.py` drives its own loop on a worker thread, keeps the
  handle of the task it schedules there, and on teardown cancels it and waits
  for its cleanup with a bounded timeout rather than returning while the task
  still holds the shared loop. Pass `name=` when there are many of a kind.
- A subprocess is a resource like any other. Every `anyio.open_process` needs
  an owner that waits on it, closes its pipes, and kills its process group on
  the failure path. `ShellToolset` tracks background processes in a dict that
  `__aexit__` terminates and cleans up (`shell/_toolset.py`), and
  `LocalStackContainer` pairs a startup `fail_after` with a shielded teardown
  (`localstack/_container.py`).

### Cancellation: level vs. edge

`asyncio` is edge-triggered: catching the delivered `CancelledError` resumes
execution until something cancels again. A cancelled `anyio` scope is
level-triggered: every later cancellation checkpoint raises again unless
shielded, so async cleanup inside one cannot finish. Know which you are in
before writing cleanup.

- Shield cleanup that must complete under an outer `anyio` cancel. Your
  `finally` and each child's cleanup are unprotected unless they shield
  themselves. `ShellToolset` waits for a killed process and drains its pipes
  under `anyio.CancelScope(shield=True)` (`shell/_toolset.py`),
  `ModalSandboxSession` shields both creation and teardown and bounds each with
  `move_on_after` so a shield can never hang (`modal_sandbox/_session.py`), and
  `_monty_exec.py` shields the interpreter's cleanup. Do not shield task-group
  exit alone: `TaskGroup.__aexit__` already shields the parent's remaining wait
  once the first cancel reaches it (anyio #695).
- Keep cancellation bookkeeping at one edge. Core owns run cancellation
  (`RunCancellation.resolve()` consumes controller-issued cancels via
  `Task.uncancel()`; `_utils.raise_if_cancelling()` re-asserts a cancel a
  completed step swallowed). Harness code must not call `uncancel()` or swallow
  `CancelledError`; if a capability needs to survive a cancel it shields the
  specific cleanup and re-raises. `Task.cancelling()`/`uncancel()` are 3.11+
  and this package supports 3.10, so never build on them here.
- One owner per deadline. Core's `FunctionToolset.call_tool` enforces exactly
  one scope for the per-tool timeout, so a longer per-tool value replaces the
  agent default instead of being capped by it. A harness toolset that owns a
  transport owns its own deadline at that transport: `ShellToolset` applies
  `timeout_seconds` around the readers, `LocalStackContainer` applies
  `_startup_timeout` around readiness, `SubAgentToolset` applies
  `timeout_seconds` with `asyncio.wait_for` around the child run. Do not stack
  a second scope over one of these.
- A deadline cannot interrupt blocking sync work. `anyio.to_thread.run_sync`
  shields its wait, so an enclosing `fail_after` returns late and raises only
  if a checkpoint follows inside the scope. The SQLite stores in
  `step_persistence/_store.py`, `media/_store.py`, `memory/_store.py`, and
  `planning/_store.py` run every statement through `to_thread`; keep those
  statements short, because a cancel cannot end one early.
- Enter and exit an `anyio.CancelScope` in the same task, in strict LIFO
  order. A scope may span a `yield` only if one persistent task performs every
  resume and finalization; a per-item `anext()` bridge can straddle tasks. anyio
  checks this at scope exit, not at the yield (core: `_sync_stream.py`'s module
  docstring names the exact error).
- Unwrap only an accidental single-child `BaseExceptionGroup` before a public
  API; preserve a genuine multi-failure group. On 3.10 the name comes from the
  `exceptiongroup` backport, and `except*` is 3.11+ syntax the backport cannot
  provide: match on `BaseExceptionGroup` and use `.split()`/`.subgroup()`.
- Decide which exceptions a containment boundary must never absorb, and name
  them once. `SubAgentToolset.delegate_task` re-raises core's control-flow
  exceptions (`CallDeferred`, `ApprovalRequired`, `SkipModelRequest`, ...)
  from a single `_ALWAYS_PROPAGATE` tuple, and re-raises a shared
  `UsageLimitExceeded`, before its `contain_errors` branch turns anything else
  into a `ModelRetry` (`subagents/_toolset.py`). `CancelledError` is a
  `BaseException`, so `except Exception` never catches it; the file says so at
  the top, so a reader does not go looking for a missing branch.

### Threads and event loops

- Async work driven by a sync entry point stays on the caller's loop. Core's
  `BlockingPortal` implementation (#6199) was reverted (#6454) because pooled
  transports bind per connection. Nested `run_sync()`/`run_stream_sync()` is
  rejected inside any callback core dispatches through `_utils.run_in_executor`;
  make the callback async instead.
- Defer shared-object entry locks with a `cached_property` (core:
  `_enter_lock` in `agent/__init__.py`, `providers/__init__.py`, `mcp.py`).
  First use binds the lock to that loop and backend; deferring keeps it out of
  `__init__` and out of Temporal's sandbox. It does not make an entered object
  reusable from a later loop. `OpenAICodexProvider._refresh_lock` in core is the
  same pattern applied to credential refresh.
- Sync callbacks are dispatched off-thread, and that costs `ContextVar` writes.
  Core's `_utils.run_in_executor` copies the caller's context in (reads work)
  and discards writes, and `asyncio.get_running_loop()` raises there. This
  covers `def` tools, output functions and validators, instructions functions,
  hooks, and history processors.
- Harness callbacks take the other lane. `SpendLimits.on_spend`,
  `ReportContextUsage.on_usage`, and `PromptInjectionDefender.on_detection` are
  called inline from the hook and awaited only if they return an awaitable;
  `SystemReminders.on_fire` is typed `Callable[[str], None]` and called bare,
  so an `async def` there produces a coroutine that is never run. Either way a
  sync callback blocks the loop and its `ContextVar` writes stick. That is
  deliberate (they are cheap notifications) but it is a contract: do not move
  one to `to_thread` without saying so, and prefer emitting a
  `CapabilityEvent` over adding another callback (#714 deprecates those four
  toward events).
- Not every sync callback is dispatched. Core awaits `Tool.prepare`,
  `PreparedToolset.prepare_func`, `FallbackModel` handlers, and model-id
  resolvers inline via `_utils.await_maybe`: they block the loop and their
  `ContextVar` writes stick. `_utils.disable_threads()` (Temporal, emscripten)
  puts every callback in that lane. Choose the lane deliberately when adding a
  sync-callable extension point.

### Locks

- Ask whether you need a lock at all before adding one. A critical section
  with no `await` in it is already atomic against other tasks on the same loop,
  so a problem you can restructure to compute first and mutate in one unbroken
  stretch needs no lock. You need one when the section suspends, or when a
  worker thread touches the same state; the no-`await` argument covers neither.
- `async with` on a shared object is not concurrency-safe by default. Guard
  entry with a deferred lock plus an entered-count (core: `_entered_count` in
  `providers/__init__.py`, `_running_count` in `mcp.py`), or give each run its
  own instance. `ShellToolset.for_run` takes the second route: a fresh instance
  per run so two concurrent runs cannot share `_cwd` or `_background`
  (`shell/_toolset.py`). Prefer that when the state is genuinely per run.
- A lock created in `__init__` binds to whichever backend and loop constructs
  it. `aws_lambda/_bridge.py` creates `asyncio.Lock()` in `__init__`
  deliberately because the bridge is asyncio-only and single-loop by contract;
  anything that can run under Trio or across loops uses the deferred pattern
  above.

### Testing it

- Assert the concurrency fact itself, not the output it produces. Fixture and
  interpreter-global state can quietly remove the trigger and leave the test
  green; use a clean subprocess when event-loop policy or similar global state
  is the subject.
- Exercise the public syntax (`async with`, `async for ... break`), not
  `__aenter__` or `agen.aclose()` by hand. Core's realtime early-break tests
  called `aclose()` themselves and passed while the shipped syntax leaked its
  tasks.
- Order steps with `Event`s, not sleeps, and wait on them with a module-level
  readiness timeout, not a one-second literal: short waits flake under `xdist`
  (pydantic/pydantic-ai#5399). `tests/shell/test_shell.py` and
  `tests/guardrails/test_input_guardrail.py` already carry sleeps; when you touch
  one, convert it.
- Prove ownership directly: diff `asyncio.all_tasks()` for ordinary leak
  checks; for a subprocess, assert the process group is gone after the timeout
  path, not just that the tool returned a timeout string.
- Reach the real trigger. Level-cancellation behavior needs a real outer
  `anyio` cancel scope, not a bare `CancelledError` raise; Trio behavior needs
  the `trio` parametrization this suite already runs, not a mental model of it.
- Know which backend a suite runs under before you trust it. anyio's pytest
  plugin parametrizes `anyio_backend` over every installed backend, and
  `anyio[trio]` is in the dev group, so the default is asyncio and Trio:
  `tests/shell` and `tests/filesystem` take that default (collect them and
  every test id ends in `[asyncio]` or `[trio]`), with individual tests opting
  out via `@pytest.mark.anyio(backends=['asyncio'])`. A suite goes asyncio-only
  by overriding the `anyio_backend` fixture at module level to return
  `'asyncio'`, which `tests/subagents` and `tests/filesystem/test_events.py`
  do. That override is why `SubAgentToolset.delegate_task` can use
  `asyncio.wait_for` today. Reach for `anyio` primitives by default
  (`fail_after`, `Lock`, task groups); before you remove an `anyio_backend`
  override, grep the package for `asyncio.` and convert or skip each hit, and
  keep `aws_lambda` asyncio-only because its bridge owns a real asyncio loop.
