"""Shared Monty execution loop for code-execution capabilities.

Drives a Monty REPL via the snapshot API (`feed_start`/`resume`), dispatching external
function calls back to a host-supplied async callback.

Two capabilities build on this:

- `code_mode`: the dispatch callback runs the agent's own tools.
- `dynamic_workflow`: the dispatch callback runs sub-agents.

The snapshot API (rather than `feed_run`) is used deliberately: it exposes each suspension
to this host-controlled loop, which owns sequential barriers, dispatch cancellation, and
trace context. Workers come from Monty's async bindings (`MontyRunState`); inside a Temporal
workflow each call into them goes through a blocking portal (see `call_monty`).
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from collections.abc import Awaitable, Callable, Container, Coroutine
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, TypeVar

import anyio
from anyio.from_thread import BlockingPortal, start_blocking_portal
from opentelemetry import context as otel_context
from typing_extensions import TypeVarTuple, Unpack

try:
    from pydantic_monty import (
        NOT_HANDLED,
        AsyncFunctionSnapshot,
        AsyncFutureSnapshot,
        AsyncMonty,
        AsyncMontySession,
        AsyncMontyWebsocket,
        AsyncNameLookupSnapshot,
        AsyncSnapshot,
        CollectString,
        ExternalException,
        ExternalFuture,
        ExternalReturnValue,
        ExternalSettledResult,
        MontyComplete,
        OsFunction,
        OsHandler,
        OSPolicy,
        ResourceLimits,
    )
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'pydantic-monty is required for code-execution capabilities. Install it with: '
        'pip install "pydantic-ai-harness[code-mode]" or "pydantic-ai-harness[dynamic-workflow]"'
    ) from _import_error

# Dispatch callback: given the sandbox function name and keyword arguments,
# perform the host-side work (tool call or sub-agent run) and return the result.
DispatchFn = Callable[[str, dict[str, Any]], Coroutine[Any, Any, Any]]

_T = TypeVar('_T')
_Args = TypeVarTuple('_Args')


def in_temporal_workflow() -> bool:
    """Whether this code runs in a Temporal workflow, where Monty must be called through the portal.

    Reads `sys.modules` so the optional extra is never imported.
    """
    workflow = sys.modules.get('temporalio.workflow')
    return workflow is not None and workflow.in_workflow()


# Running Monty inside a Temporal workflow
# ----------------------------------------
# Monty's bindings are async and complete each awaited call from Monty's own I/O thread by waking
# the event loop the `await` started on (`loop.call_soon_threadsafe`). A Temporal workflow's event
# loop cannot be woken that way, so the call would never complete. Inside a workflow, the
# helpers below therefore route every Monty call through an `anyio` blocking portal: a helper
# thread running a normal asyncio loop. The workflow thread blocks until the sandbox suspends or
# completes, exactly as it did with Monty's former sync bindings, and control is back in the
# workflow between calls, where nested tools run as activities. Outside a workflow the portal is
# `None` and every helper is a plain `await`.


async def call_monty(
    portal: BlockingPortal | None, fn: Callable[[Unpack[_Args]], Awaitable[_T]], *args: Unpack[_Args]
) -> _T:
    """Await one call into Monty's async bindings, `fn(*args)`, through `portal` when there is one."""
    if portal is None:
        return await fn(*args)
    return portal.call(fn, *args)


async def _enter_monty(
    stack: AsyncExitStack, resource: AbstractAsyncContextManager[_T], portal: BlockingPortal | None
) -> _T:
    """Enter a Monty pool or session on `stack`, through `portal` when there is one."""
    if portal is None:
        return await stack.enter_async_context(resource)
    return stack.enter_context(portal.wrap_async_context_manager(resource))


async def _release_monty(stack: AsyncExitStack) -> None:
    """Exit the Monty resources on `stack`, even while the run is being cancelled.

    Cancellation is delivered again at every suspension point while an enclosing cancel scope
    stays cancelled, which would abandon the session or pool exit half way and leak the worker.
    The exit waits for a snippet that is still running, so it is bounded by
    `max_feed_duration_secs` (and, for a remote worker, the transport's per-turn deadline); Monty
    offers no way to interrupt a running feed.
    """
    with anyio.CancelScope(shield=True):
        await stack.aclose()


# Extra time the WebSocket transport allows on top of `max_feed_duration_secs`, so the sandbox's own
# limit fires first and the model sees a time-limit error rather than a dropped connection.
_REMOTE_TURN_SLACK_SECS = 10.0

# Route the clock and unseeded randomness to the `os=` handler, as before Monty 1.0: without one they are
# unavailable, which keeps sandbox code deterministic when a Temporal workflow replays it. Sleeps come back to
# `MontyExecutor`, which waits on the run's own event loop (a durable timer inside a Temporal workflow).
_OS_POLICY: OSPolicy = {'datetime': 'call_host', 'sleep': 'call_host', 'random_start': 'call_host'}


@dataclass
class MontyRunState:
    """A Monty worker pool and its checked-out REPL session, opened lazily and closed together.

    Workers are local subprocesses from `AsyncMonty`, or remote ones dialed through
    `AsyncMontyWebsocket` when `monty_sandbox_url` is set. Inside a Temporal workflow every Monty
    call goes through `portal`, opened with the pool and closed after it.
    """

    monty_sandbox_url: str | None = None
    pool: AsyncMonty | AsyncMontyWebsocket | None = None
    session: AsyncMontySession | None = None
    portal: BlockingPortal | None = None
    has_executed_feed: bool = False
    _pool_stack: AsyncExitStack = field(default_factory=AsyncExitStack, repr=False)
    _session_stack: AsyncExitStack = field(default_factory=AsyncExitStack, repr=False)

    async def get_session(
        self,
        *,
        type_check: bool,
        type_check_stubs: str | None,
        limits: ResourceLimits,
        in_temporal_workflow: bool = False,
    ) -> AsyncMontySession:
        """Return the live REPL session, spawning or dialing the pool on first use."""
        if self.pool is None:
            try:
                if in_temporal_workflow:
                    self.portal = self._pool_stack.enter_context(start_blocking_portal())
                if self.monty_sandbox_url is None:
                    pool = AsyncMonty()
                else:
                    max_feed_duration_secs = limits.get('max_feed_duration_secs')
                    timeout = (
                        None if max_feed_duration_secs is None else max_feed_duration_secs + _REMOTE_TURN_SLACK_SECS
                    )
                    pool = AsyncMontyWebsocket(self.monty_sandbox_url, request_timeout=timeout)
                self.pool = await _enter_monty(self._pool_stack, pool, self.portal)
            except BaseException:
                await self._release_pool()  # a failed spawn or dial must not leave the portal thread behind
                raise
        if self.session is None:
            checkout = self.pool.checkout(
                limits=limits, type_check=type_check, type_check_stubs=type_check_stubs, os_policy=_OS_POLICY
            )
            self.session = await _enter_monty(self._session_stack, checkout, self.portal)
        return self.session

    async def reset(self) -> None:
        """Return the current worker and make the next call start a fresh REPL."""
        # Detach before awaiting the exit, so a session checked out meanwhile is not dropped.
        stack, self._session_stack = self._session_stack, AsyncExitStack()
        self.session = None
        self.has_executed_feed = False
        await _release_monty(stack)

    async def close(self) -> None:
        """Return the checked-out worker, then close the owning pool (and portal) even if that fails."""
        try:
            await self.reset()
        finally:
            await self._release_pool()

    async def _release_pool(self) -> None:
        stack, self._pool_stack = self._pool_stack, AsyncExitStack()
        self.pool = None
        self.portal = None
        await _release_monty(stack)


@dataclass
class PendingCall:
    """A dispatched call and the suspension context under which it executes."""

    call: asyncio.Task[Any] | Coroutine[Any, Any, Any]
    context: otel_context.Context


def is_sandbox_panic(exc: BaseException) -> bool:
    """Whether `exc` is a Rust-side sandbox panic surfacing through pyo3.

    pyo3 raises `pyo3_runtime.PanicException`, a `BaseException` (not `Exception`) subclass
    from a module that cannot be imported, so it is matched by name. A panic can surface from
    monty's host-side bindings, so callers should convert it to a retry rather than let it
    tear down the whole agent run.
    """
    return type(exc).__name__ == 'PanicException'


class PrintCapture:
    """Collects bounded print output from a Monty REPL."""

    def __init__(self) -> None:
        self.callback = CollectString()

    @property
    def joined(self) -> str:
        return self.callback.output

    def prepend_to(self, error_message: str) -> str:
        """Prefix captured stdout to an error message, so the model sees what printed before the error."""
        printed = self.joined.rstrip('\n')
        if not printed:
            return error_message
        return f'[stdout before error]\n{printed}\n[/stdout before error]\n{error_message}'


@dataclass
class MontyExecutor:
    """Drives a Monty REPL to completion, dispatching external calls to a host callback.

    Single-use: it accumulates per-run state in `_pending`/`_pre_resolved`, so construct a
    fresh executor for each `run` rather than reusing or sharing one across concurrent runs.

    External calls are handled by execution mode:

    - **Parallel** (`async def`): deferred via `resume({'future': ...})` and eagerly
      scheduled as `asyncio.Task`s. Resolved at `FutureSnapshot` via `asyncio.gather`.
    - **Per-call sequential** (`def`, name in `sequential_names`): resolved inline at
      `FunctionSnapshot`. Any pending parallel tasks are awaited first (barrier).
    - **Global sequential** (when selected by the run context): all calls deferred but
      stored as bare coroutines and awaited one-at-a-time to prevent interleaving.
    """

    dispatch: DispatchFn
    valid_names: Container[str]
    sequential_names: set[str] = field(default_factory=set[str])
    global_sequential: bool = False
    # Set inside a Temporal workflow; see `call_monty`.
    portal: BlockingPortal | None = None
    # Total seconds the code may sleep, or `None` for no cap. Sleep time does not count toward
    # Monty's execution-time limit, so it gets the same allowance separately.
    max_sleep_secs: float | None = None
    # Replaced in tests, to observe sleeps without waiting.
    sleep: Callable[[float], Coroutine[Any, Any, None]] = asyncio.sleep
    # CodeMode's `os_access`. Only needed here to answer host-state calls inside a Temporal workflow.
    os_handler: OsHandler | None = None

    _slept_secs: float = field(default=0.0, init=False)
    # Parallel calls deferred but not yet resolved, keyed by Monty call id.
    _pending: dict[int, PendingCall] = field(default_factory=dict[int, PendingCall], init=False)
    # Parallel results awaited early at a sequential barrier, before their FutureSnapshot is reached.
    _pre_resolved: dict[int, ExternalSettledResult] = field(
        default_factory=dict[int, ExternalSettledResult], init=False
    )

    async def run(self, feed_start: Callable[[], Awaitable[AsyncSnapshot]]) -> MontyComplete:
        """Drive the REPL from `feed_start` (a bound `AsyncMontySession.feed_start`) until it completes."""
        try:
            state = await call_monty(self.portal, feed_start)
            while not isinstance(state, MontyComplete):
                if isinstance(state, AsyncNameLookupSnapshot):
                    # Leave the name undefined so the sandbox raises `NameError`.
                    state = await call_monty(self.portal, state.resume)
                elif isinstance(state, AsyncFunctionSnapshot):
                    state = await self._handle_function(state)
                else:
                    state = await self._resolve_futures(state)
        finally:
            cancelled: list[asyncio.Task[Any]] = []
            for pending in self._pending.values():
                call = pending.call
                if isinstance(call, asyncio.Task):
                    call.cancel()
                    cancelled.append(call)
                else:
                    call.close()
            if cancelled:
                # `cancel()` only schedules a `CancelledError` at each task's next suspension
                # point; await them so dispatched work (e.g. sub-agent runs mutating shared
                # usage) has fully unwound before this returns. `return_exceptions=True` keeps
                # one task's teardown error from masking the original exception, and the
                # results are deliberately discarded. Shielded: run cancellation can land here
                # with an enclosing anyio scope already cancelled, and that scope re-cancels
                # its tasks on every event-loop cycle -- each delivery either aborts this await
                # outright (abandoning the tasks mid-unwind) or is forwarded through the
                # `gather` into every task, breaking any await their cleanup performs. The
                # shield holds for anyio-scope cancellation; a raw second `Task.cancel()` can
                # still pierce it.
                with anyio.CancelScope(shield=True):
                    await asyncio.gather(*cancelled, return_exceptions=True)
        return state

    async def _handle_function(self, snapshot: AsyncFunctionSnapshot) -> AsyncSnapshot:
        """Dispatch (or defer) a single external function call."""
        if snapshot.is_os_function and snapshot.function_name in ('time.sleep', 'asyncio.sleep'):
            return await self._sleep(snapshot)
        if snapshot.is_os_function:
            if self.portal is not None and self.os_handler is not None:
                match snapshot.function_name:
                    case (
                        'os.getenv' | 'os.environ' | 'date.today' | 'datetime.now' | 'os.urandom' | 'time.time' as name
                    ):
                        return await self._answer_os_call(snapshot, self.os_handler, name)
                    case _:  # file calls: Monty answers them from the mounts first
                        pass
            # OS calls (env, clock, filesystem) are answered from the feed's mounts and the
            # `os=` handler captured at `feed_start`, falling back to monty's unhandled default.
            return await call_monty(self.portal, snapshot.resume_auto)

        name = snapshot.function_name
        if name not in self.valid_names:
            return await self._raise_in_sandbox(snapshot, NameError(f'Unknown function: {name}'))

        if snapshot.args:
            return await self._raise_in_sandbox(
                snapshot, TypeError(f'{name}() does not accept positional arguments; use keyword arguments')
            )

        if name in self.sequential_names:
            # Rendered as `def` (sync), so the sandbox code doesn't `await` the result --
            # resolve inline. Await pending parallel tasks first (barrier) for ordering.
            # The dispatch coroutine is created only after the barrier: it is not in
            # `_pending`, so if it existed while the barrier awaits and we were cancelled
            # there, `run`'s cleanup would never close it.
            for cid in list(self._pending):
                self._pre_resolved[cid] = await _await_external(self._pending.pop(cid))
            try:
                call = self._dispatch(snapshot, parallel=False)
            except Exception as exc:
                return await self._raise_in_sandbox(snapshot, exc)
            # The wrapped outcome (`{'return_value': ...}` / `{'exception': ...}`) is already
            # exactly the payload `resume` expects.
            return await call_monty(self.portal, snapshot.resume, await _await_external(call))

        # Deferred execution -- resolved later at FutureSnapshot.
        try:
            call = self._dispatch(snapshot, parallel=not self.global_sequential)
        except Exception as exc:
            # `dispatch` refused the call before building its coroutine (e.g. an exhausted
            # per-snippet budget). Deliver the error at the sandbox call site, the same way a
            # failure raised inside the coroutine is delivered, rather than letting it abort the
            # feed: calls that already completed keep the results the host recorded for them, and
            # the snippet can still return them. Nothing was scheduled, so there is no task to
            # clean up and no further work is admitted.
            return await self._raise_in_sandbox(snapshot, exc)
        self._pending[snapshot.call_id] = call
        return await call_monty(self.portal, snapshot.resume, ExternalFuture(future=...))

    async def _sleep(self, snapshot: AsyncFunctionSnapshot) -> AsyncSnapshot:
        """Wait for a sandbox sleep here rather than in Monty, charging it to `max_sleep_secs`.

        Monty has already validated the duration: `args` is one non-negative float.
        """
        secs: float = snapshot.args[0]
        if self.max_sleep_secs is not None and self._slept_secs + secs > self.max_sleep_secs:
            remaining = self.max_sleep_secs - self._slept_secs
            return await self._raise_in_sandbox(
                snapshot,
                TimeoutError(
                    f'sleeping {secs:g}s would exceed the {self.max_sleep_secs:g}s this code may sleep '
                    f'(max_duration_secs); {remaining:g}s left'
                ),
            )
        self._slept_secs += secs
        if snapshot.function_name == 'time.sleep':
            await self.sleep(secs)
            return await call_monty(self.portal, snapshot.resume, ExternalReturnValue(return_value=None))
        # `asyncio.sleep` is awaitable, so defer it like a parallel call and gathered sleeps overlap.
        sleep = self.sleep(secs)
        parallel = not self.global_sequential
        task = asyncio.ensure_future(sleep) if parallel else sleep
        self._pending[snapshot.call_id] = PendingCall(task, otel_context.get_current())
        return await call_monty(self.portal, snapshot.resume, ExternalFuture(future=...))

    async def _answer_os_call(
        self, snapshot: AsyncFunctionSnapshot, handler: OsHandler, name: OsFunction
    ) -> AsyncSnapshot:
        """Call `os_access` on the workflow's own thread, so it can use `temporalio.workflow` APIs.

        Through the portal Monty would call it from another thread. File calls are not routed here:
        they go to Monty, which answers them from the mounts first.
        """
        token = otel_context.attach(snapshot.trace_context())
        try:
            value = handler(name=name, args=snapshot.args, kwargs=snapshot.kwargs, is_async=True)
            if inspect.isawaitable(value):
                value = await value
        except Exception as exc:
            return await self._raise_in_sandbox(snapshot, exc)
        finally:
            otel_context.detach(token)
        if value is NOT_HANDLED:
            return await call_monty(self.portal, snapshot.resume_not_handled)
        return await call_monty(self.portal, snapshot.resume, ExternalReturnValue(return_value=value))

    async def _raise_in_sandbox(self, snapshot: AsyncFunctionSnapshot, exc: Exception) -> AsyncSnapshot:
        """Resume the suspended call by raising `exc` at its sandbox call site."""
        return await call_monty(self.portal, snapshot.resume, ExternalException(exception=exc))

    def _dispatch(self, snapshot: AsyncFunctionSnapshot, *, parallel: bool) -> PendingCall:
        context = snapshot.trace_context()
        token = otel_context.attach(context)
        try:
            call = self.dispatch(snapshot.function_name, snapshot.kwargs)
            # Tasks inherit the active context; bare coroutines need it restored when awaited.
            return PendingCall(asyncio.ensure_future(call) if parallel else call, context)
        finally:
            otel_context.detach(token)

    async def _resolve_futures(self, snapshot: AsyncFutureSnapshot) -> AsyncSnapshot:
        """Resolve the deferred calls a `FutureSnapshot` is waiting on."""
        pending_ids = snapshot.pending_call_ids
        results: dict[int, ExternalSettledResult] = {}
        for cid in pending_ids:
            if cid in self._pre_resolved:
                results[cid] = self._pre_resolved.pop(cid)
            elif self.global_sequential:
                results[cid] = await _await_external(self._pending.pop(cid))

        # Gather any remaining parallel tasks concurrently. They stay in `_pending` until
        # gather returns, so the cleanup in `run` can still cancel them if this is cancelled.
        gather_ids = [cid for cid in pending_ids if cid not in results]
        if gather_ids:
            settled = await asyncio.gather(*(self._pending[cid].call for cid in gather_ids), return_exceptions=True)
            for cid, outcome in zip(gather_ids, settled):
                del self._pending[cid]
                results[cid] = _wrap_gathered(outcome)

        return await call_monty(self.portal, snapshot.resume, results)


async def _await_external(call: PendingCall) -> ExternalReturnValue | ExternalException:
    """Await a single deferred call and wrap its outcome for Monty."""
    token = otel_context.attach(call.context)
    try:
        result = await call.call
    except Exception as exc:
        return ExternalException(exception=exc)
    finally:
        otel_context.detach(token)
    return ExternalReturnValue(return_value=result)


def _wrap_gathered(outcome: Any) -> ExternalReturnValue | ExternalException:
    """Wrap an `asyncio.gather(return_exceptions=True)` outcome for Monty."""
    if isinstance(outcome, Exception):
        return ExternalException(exception=outcome)
    if isinstance(outcome, BaseException):  # pragma: no cover
        raise outcome
    return ExternalReturnValue(return_value=outcome)
