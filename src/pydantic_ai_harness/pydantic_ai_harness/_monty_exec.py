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
from functools import partial
from typing import Any, TypeVar, assert_never

import anyio
from anyio.from_thread import BlockingPortal, start_blocking_portal
from opentelemetry import context as otel_context
from typing_extensions import TypeVarTuple, Unpack

from pydantic_ai_harness._typeguards import is_object_tuple, is_str_keyed_dict

try:
    from pydantic_monty import (
        NOT_HANDLED,
        AsyncFunctionSnapshot,
        AsyncFutureSnapshot,
        AsyncMonty,
        AsyncMontyRemote,
        AsyncMontySession,
        AsyncNameLookupSnapshot,
        AsyncSnapshot,
        CollectString,
        ExternalException,
        ExternalFuture,
        ExternalResult,
        ExternalReturnValue,
        ExternalSettledResult,
        MontyComplete,
        MontyRefusedError,
        OsFunction,
        OsHandler,
        OSPolicy,
        ResourceLimits,
    )

    from pydantic_ai_harness._monty_turns import (
        SLEEP_FUNCTIONS,
        MontyFailure,
        MontyTurn,
        TurnCall,
        TurnRequest,
        TurnResume,
        from_wire,
        oversized,
        settled_data,
    )
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'pydantic-monty is required for code-execution capabilities. Install it with: '
        'pip install "pydantic-ai-harness[code-mode]" or "pydantic-ai-harness[dynamic-workflow]"'
    ) from _import_error

DispatchFn = Callable[[str, dict[str, Any]], Coroutine[Any, Any, Any]]
"""Given a sandbox function name and its keyword arguments, do the host-side work (a tool call or sub-agent run)."""

_T = TypeVar('_T')
_Args = TypeVarTuple('_Args')


def in_temporal_workflow() -> bool:
    """Whether this code runs in a Temporal workflow, where Monty must be called through the portal.

    Reads `sys.modules` so the optional extra is never imported.
    """
    workflow = sys.modules.get('temporalio.workflow')
    return workflow is not None and workflow.in_workflow()


def in_durable_task_orchestration() -> bool:
    """Whether this code runs in an Azure Durable Task orchestration, where Monty must be called through the portal.

    Reads `sys.modules` so the optional extra is never imported.
    """
    bridge = sys.modules.get('pydantic_ai_harness.azure_durable_task._bridge')
    return bridge is not None and bridge.current_loop() is not None


# Running Monty inside a Temporal workflow or a Durable Task orchestration
# ------------------------------------------------------------------------
# Monty's bindings are async and complete each awaited call from Monty's own I/O thread by waking
# the event loop the `await` started on (`loop.call_soon_threadsafe`). Neither a Temporal workflow's
# event loop nor the loop that drives an Azure Durable Task orchestrator can be woken that way, so
# the call would never complete. Inside a workflow or orchestration, the
# helpers below therefore route every Monty call through an `anyio` blocking portal: a helper
# thread running a normal asyncio loop. The workflow thread blocks until the sandbox suspends or
# completes, exactly as it did with Monty's former sync bindings, and control is back in the
# workflow between calls, where nested tools run as activities. Outside a workflow the portal is
# `None` and every helper is a plain `await`.


async def call_monty(
    portal: BlockingPortal | None, func: Callable[[Unpack[_Args]], Awaitable[_T]], *args: *_Args
) -> _T:
    """Await one call into Monty's async bindings, `func(*args)`, through `portal` when there is one."""
    if portal is None:
        return await func(*args)
    return portal.call(func, *args)


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


# Extra time the remote transport allows on top of `max_feed_duration_secs`, so the sandbox's own
# limit fires first and the model sees a time-limit error rather than a dropped connection.
_REMOTE_TURN_SLACK_SECONDS = 10.0

# Route the clock and unseeded randomness to the `os=` handler, as before Monty 1.0: without one they are
# unavailable, which keeps sandbox code deterministic when a Temporal workflow replays it. Sleeps come back to
# `MontyExecutor`, which waits on the run's own event loop (a durable timer inside a Temporal workflow).
# `MontyRunState.os_policy` overrides these key by key.
_OS_POLICY: OSPolicy = {'datetime': 'call_host', 'sleep': 'call_host', 'random_start': 'call_host'}


def _remote_turn_timeout(limits: ResourceLimits) -> float | None:
    """The WebSocket transport's per-turn deadline: no deadline when the feed duration is unlimited."""
    max_feed_duration_seconds = limits.get('max_feed_duration_secs')
    return None if max_feed_duration_seconds is None else max_feed_duration_seconds + _REMOTE_TURN_SLACK_SECONDS


def session_os_policy(os_policy: OSPolicy | None) -> OSPolicy:
    """The policies a session is checked out with: `os_policy` merged over `_OS_POLICY`, key by key."""
    return _OS_POLICY | (os_policy or {})


async def build_remote_pool(
    sandbox: str | Callable[[], AsyncMontyRemote | Awaitable[AsyncMontyRemote]], limits: ResourceLimits
) -> AsyncMontyRemote:
    """The remote pool `sandbox` names: dialed from a URL, or built by a sync or async factory."""
    if isinstance(sandbox, str):
        return AsyncMontyRemote(sandbox, request_timeout=_remote_turn_timeout(limits))
    pool = sandbox()
    return await pool if inspect.isawaitable(pool) else pool


@dataclass
class MontyRunState:
    """A Monty worker pool and its checked-out REPL session, opened lazily and closed together.

    Workers are local subprocesses from `AsyncMonty`, or remote ones dialed through
    `AsyncMontyRemote` when `monty_sandbox_url` or `pool_factory` is set. Inside a Temporal workflow or
    a Durable Task orchestration every Monty call goes through `portal`, opened with the pool and closed after it.
    """

    monty_sandbox_url: str | None = field(default=None, repr=False)
    """May carry credentials, so it stays out of `repr`."""
    os_policy: OSPolicy | None = None
    """Session policies merged over `_OS_POLICY` at checkout, key by key."""
    pool_factory: Callable[[], AsyncMontyRemote | Awaitable[AsyncMontyRemote]] | None = None
    """Builds the remote pool, sync or async, in place of `monty_sandbox_url`, with its own URL, metadata, and timeouts."""
    pool: AsyncMonty | AsyncMontyRemote | None = None
    session: AsyncMontySession | None = None
    portal: BlockingPortal | None = None
    has_executed_feed: bool = False
    stored_state: str | None = None
    """The server's record of the session when each turn runs as a durable operation (see `TurnFeed`)."""
    stores_sessions: bool = True
    """False once a turn found that the server stores no sessions; the run's turns then stay in the workflow."""
    suspensions: int = 0
    """The pauses the session's turns made, which count toward its `max_suspensions` (see `TurnFeed`)."""
    _pool_stack: AsyncExitStack = field(default_factory=AsyncExitStack, repr=False)
    _session_stack: AsyncExitStack = field(default_factory=AsyncExitStack, repr=False)

    async def get_session(
        self,
        *,
        type_check: bool,
        type_check_stubs: str | None,
        limits: ResourceLimits,
        use_portal: bool = False,
    ) -> AsyncMontySession:
        """Return the live REPL session, spawning or dialing the pool on first use.

        Retry a capacity refusal once after a short backoff. Other setup errors propagate unchanged.
        """
        if self.pool is None:
            try:
                if use_portal:
                    self.portal = self._pool_stack.enter_context(start_blocking_portal())
                remote = self.pool_factory if self.pool_factory is not None else self.monty_sandbox_url
                pool = AsyncMonty() if remote is None else await build_remote_pool(remote, limits)
                self.pool = await _enter_monty(self._pool_stack, pool, self.portal)
            except BaseException:
                await self._release_pool()  # a failed spawn or dial must not leave the portal thread behind
                raise
        if self.session is None:
            checkout = partial(
                self.pool.checkout,
                limits=limits,
                type_check=type_check,
                type_check_stubs=type_check_stubs,
                os_policy=session_os_policy(self.os_policy),
            )
            self.session = await self._checkout_with_retry(checkout)
        return self.session

    async def _checkout_with_retry(
        self, checkout: Callable[[], AbstractAsyncContextManager[AsyncMontySession]]
    ) -> AsyncMontySession:
        try:
            return await _enter_monty(self._session_stack, checkout(), self.portal)
        except MontyRefusedError as error:
            if error.cause not in ('caller_cap', 'capacity', 'worker_unavailable'):
                raise
            await call_monty(self.portal, anyio.sleep, 0.1)
            return await _enter_monty(self._session_stack, checkout(), self.portal)

    async def reset(self) -> None:
        """Return the current worker and make the next call start a fresh REPL."""
        # Detach before awaiting the exit, so a session checked out meanwhile is not dropped.
        stack, self._session_stack = self._session_stack, AsyncExitStack()
        self.session = None
        self.stored_state = None
        self.suspensions = 0
        self.has_executed_feed = False
        await _release_monty(stack)

    async def aclose(self) -> None:
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


_PRINT_LIMIT_BYTES = 10 * 1024 * 1024
"""The most a feed may print, `CollectString`'s default cap; Monty raises `MemoryError` past it."""


class PrintCapture:
    """Collects bounded print output from a Monty REPL."""

    def __init__(self) -> None:
        self.callback = CollectString(max_bytes=_PRINT_LIMIT_BYTES)
        self._relayed = ''
        self._relayed_bytes = 0

    @property
    def joined(self) -> str:
        return self.callback.output + self._relayed

    def relay(self, printed: str, printed_bytes: int) -> None:
        """Add output a turn printed in another process (see `TurnFeed`), of which `printed` may be a cut."""
        self._relayed += printed
        self._relayed_bytes += printed_bytes

    @property
    def unused_bytes(self) -> int:
        """How much more the next turn may print, so the whole feed stays under the cap."""
        return _PRINT_LIMIT_BYTES - self._relayed_bytes

    def prepend_to(self, error_message: str) -> str:
        """Prefix captured stdout to an error message, so the model sees what printed before the error."""
        printed = self.joined.rstrip('\n')
        if not printed:
            return error_message
        return f'[stdout before error]\n{printed}\n[/stdout before error]\n{error_message}'


class MontyTurnFailed(Exception):
    """A turn ended in a sandbox error, described by `failure`."""

    def __init__(self, failure: MontyFailure) -> None:
        super().__init__(failure.kind)
        self.failure = failure


class MontyStoresNoSessions(Exception):
    """The first turn of a run found that the server stores no sessions, before it ran any code.

    The snippet then runs on a live session in the workflow, as it did before turns ran elsewhere.
    """


class MontyTurnUnavailable(Exception):
    """The durable operation that runs a turn failed, so the turn's outcome is unknown.

    CodeMode resets the session and re-raises `error`, so the run fails as it does when any other
    activity fails.
    """

    def __init__(self, error: Exception) -> None:
        super().__init__('The durable operation running a Monty turn failed.')
        self.error = error


_MONTY_DEFAULT_MAX_SUSPENSIONS = 1000
"""The `max_suspensions` Monty applies when none is set; it cannot be turned off (see `ResourceLimits`)."""


def _unsendable_turn(type_name: str, message: str, *, resuming: bool) -> MontyTurnFailed:
    """A turn the host could not send, failing the snippet with `type_name` as a live session would.

    When `resuming`, the stored record is the paused snippet, which cannot continue, so the session is
    reset. A snippet that could not start ran nothing, so the session keeps its state.
    """
    failure = MontyFailure(
        'runtime',
        display=f'{type_name}: {message}',
        message=message,
        type_name=type_name,
        exception_types=(type_name, 'Exception', 'BaseException'),
        session_lost=resuming,
    )
    return MontyTurnFailed(failure)


@dataclass
class TurnFeed:
    """One feed of a remote session, run one turn at a time through `run_turn`, a durable operation.

    `MontyExecutor` drives it exactly as it drives a live session, through the `Turn*` snapshots
    below, so the host handles calls, sleeps, and deferred results in the workflow as before. Each
    turn starts from `run_state.stored_state` and replaces it with the record the turn stored.
    """

    run_turn: Callable[[TurnRequest], Awaitable[MontyTurn]]
    run_state: MontyRunState
    capture: PrintCapture
    limits: ResourceLimits

    async def start(self, code: str, type_check_stubs: str | None) -> TurnSnapshot:
        state = self.run_state.stored_state
        limits = self._limits(restored_pauses=0)
        request = TurnRequest(state, limits, self.capture.unused_bytes, code=code, type_check_stubs=type_check_stubs)
        return await self._turn(request)

    async def resume(self, encode: Callable[[], TurnResume]) -> TurnSnapshot:
        """Resume with what `encode` returns: the results, as data a turn can carry."""
        try:
            resume = encode()
        # A value the wire cannot carry, such as a `bytearray`. This runs in workflow code, whose lines coverage
        # does not record; `test_remote_turns_carry_values_as_a_live_session_does[bytearray_tool_result]` runs it.
        except TypeError as error:  # pragma: lax no cover
            raise _unsendable_turn('TypeError', str(error), resuming=True) from None
        limits = self._limits(restored_pauses=1)
        request = TurnRequest(self.run_state.stored_state, limits, self.capture.unused_bytes, resume=resume)
        return await self._turn(request)

    def _limits(self, *, restored_pauses: int) -> ResourceLimits:
        """`limits`, with `max_suspensions` lowered by the pauses the session made before this turn.

        Monty counts `max_suspensions` across a session's feeds, but a loaded record restarts the
        count: at 0 for a session, and at 1 for a snapshot, counting the pause it resumes.
        """
        # Monty keeps its default cap when the key is missing or `None`, as under `'unlimited'`.
        max_suspensions = self.limits.get('max_suspensions')
        if max_suspensions is None:
            max_suspensions = _MONTY_DEFAULT_MAX_SUSPENSIONS
        return {**self.limits, 'max_suspensions': max_suspensions - self.run_state.suspensions + restored_pauses}

    async def _turn(self, request: TurnRequest) -> TurnSnapshot:
        if (message := oversized(request)) is not None:
            raise _unsendable_turn('ValueError', message, resuming=request.resume is not None)
        try:
            turn = await self.run_turn(request)
        except Exception as error:
            raise MontyTurnUnavailable(error) from error
        self.capture.relay(turn.prints, turn.printed_bytes)
        self.run_state.suspensions += turn.suspensions
        # A turn that stored nothing failed in a way that resets the session, which clears this again.
        self.run_state.stored_state = turn.state
        match turn.kind:
            case 'call':
                assert turn.call is not None
                return TurnFunctionSnapshot.from_call(self, turn.call)
            case 'waiting':
                return TurnFutureSnapshot(self, turn.pending_call_ids)
            case 'complete':
                assert turn.output is not None
                return TurnComplete(from_wire(turn.output))
            case 'no_storage':
                raise MontyStoresNoSessions
            case 'failed':
                assert turn.failure is not None
                raise MontyTurnFailed(turn.failure)
            case _:
                assert_never(turn.kind)


@dataclass
class TurnFunctionSnapshot:
    """An `AsyncFunctionSnapshot` stand-in for a turn that paused at an external call."""

    feed: TurnFeed
    function_name: str
    args: tuple[object, ...]
    kwargs: dict[str, object]
    call_id: int
    is_os_function: bool

    @classmethod
    def from_call(cls, feed: TurnFeed, call: TurnCall) -> TurnFunctionSnapshot:
        args, kwargs = from_wire(call.args), from_wire(call.kwargs)
        assert is_object_tuple(args) and is_str_keyed_dict(kwargs), 'a turn call carries a tuple and a dict'
        return cls(feed, call.function_name, args, kwargs, call.call_id, call.is_os_function)

    def trace_context(self) -> otel_context.Context:
        return otel_context.get_current()

    async def resume(self, result: ExternalResult) -> TurnSnapshot:
        return await self.feed.resume(partial(TurnResume.from_result, result))


@dataclass
class TurnFutureSnapshot:
    """An `AsyncFutureSnapshot` stand-in for a turn that paused awaiting deferred calls."""

    feed: TurnFeed
    pending_call_ids: list[int]

    async def resume(self, results: dict[int, ExternalSettledResult]) -> TurnSnapshot:
        return await self.feed.resume(
            lambda: TurnResume(
                'settled', settled={call_id: settled_data(result) for call_id, result in results.items()}
            )
        )


@dataclass
class TurnComplete:
    """A `MontyComplete` stand-in for a turn that finished the snippet."""

    output: object


TurnSnapshot = TurnFunctionSnapshot | TurnFutureSnapshot | TurnComplete
FunctionSnapshot = AsyncFunctionSnapshot | TurnFunctionSnapshot
"""A sandbox paused at an external call, live or from a durable turn."""
Snapshot = AsyncSnapshot | TurnSnapshot


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
    - **Global sequential** (selected by the caller): all calls deferred but
      stored as bare coroutines and awaited one-at-a-time to prevent interleaving.
    """

    dispatch: DispatchFn
    valid_names: Container[str]
    sequential_names: set[str] = field(default_factory=set[str])
    global_sequential: bool = False
    # Set inside a Temporal workflow or a Durable Task orchestration; see `call_monty`.
    portal: BlockingPortal | None = None
    # Total seconds the code may sleep, or `None` for no cap. Sleep time does not count toward
    # Monty's execution-time limit, so it gets the same allowance separately.
    max_sleep_seconds: float | None = None
    # Replaced in tests, to observe sleeps without waiting.
    sleep: Callable[[float], Coroutine[Any, Any, None]] = anyio.sleep
    # CodeMode's `os_access`. Only needed here to answer host-state calls inside a Temporal workflow.
    os_handler: OsHandler | None = None
    # Awaited before deferred calls are resolved. Every call dispatched so far has started by then, so
    # a host can run the calls it collected from them as one batch, in dispatch order. That holds for
    # parallel dispatch only: under `global_sequential`, calls start after this.
    flush: Callable[[], Awaitable[None]] | None = None

    _slept_seconds: float = field(default=0.0, init=False)
    # Parallel calls deferred but not yet resolved, keyed by Monty call id.
    _pending: dict[int, PendingCall] = field(default_factory=dict[int, PendingCall], init=False)
    # Parallel results awaited early at a sequential barrier, before their FutureSnapshot is reached.
    _pre_resolved: dict[int, ExternalSettledResult] = field(
        default_factory=dict[int, ExternalSettledResult], init=False
    )

    async def run(self, feed_start: Callable[[], Awaitable[Snapshot]]) -> MontyComplete | TurnComplete:
        """Drive the REPL from `feed_start` (a bound `AsyncMontySession.feed_start` or `TurnFeed.start`) until it completes."""
        try:
            state = await call_monty(self.portal, feed_start)
            while not isinstance(state, MontyComplete | TurnComplete):
                if isinstance(state, AsyncNameLookupSnapshot):
                    # Leave the name undefined so the sandbox raises `NameError`.
                    state = await call_monty(self.portal, state.resume)
                elif isinstance(state, AsyncFunctionSnapshot | TurnFunctionSnapshot):
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

    async def _handle_function(self, snapshot: FunctionSnapshot) -> Snapshot:
        """Dispatch (or defer) a single external function call."""
        if snapshot.is_os_function:
            match snapshot.function_name:
                case name if name in SLEEP_FUNCTIONS:
                    return await self._sleep(snapshot)
                case (
                    'os.getenv' | 'os.environ' | 'date.today' | 'datetime.now' | 'os.urandom' | 'time.time' as name
                ) if (
                    isinstance(snapshot, AsyncFunctionSnapshot)
                    and self.portal is not None
                    and self.os_handler is not None
                ):
                    return await self._answer_os_call(snapshot, self.os_handler, name)
                case _:  # file calls: Monty answers them from the mounts first
                    # OS calls (env, clock, filesystem) are answered from the feed's mounts and the
                    # `os=` handler captured at `feed_start`, falling back to monty's unhandled default.
                    assert isinstance(snapshot, AsyncFunctionSnapshot), 'a turn answers these calls itself'
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

    async def _sleep(self, snapshot: FunctionSnapshot) -> Snapshot:
        """Wait for a sandbox sleep here rather than in Monty, charging it to `max_sleep_seconds`.

        Monty has already validated the duration: `args` is one non-negative number.
        """
        seconds = snapshot.args[0]
        assert isinstance(seconds, int | float), 'Monty validates the sleep duration'
        if self.max_sleep_seconds is not None and self._slept_seconds + seconds > self.max_sleep_seconds:
            remaining = self.max_sleep_seconds - self._slept_seconds
            return await self._raise_in_sandbox(
                snapshot,
                TimeoutError(
                    f'sleeping {seconds:g}s would exceed the {self.max_sleep_seconds:g}s this code may sleep '
                    f'(max_duration_seconds); {remaining:g}s left'
                ),
            )
        self._slept_seconds += seconds
        if snapshot.function_name in ('time.sleep', 'system.sleep'):
            await self.sleep(seconds)
            return await call_monty(self.portal, snapshot.resume, ExternalReturnValue(return_value=None))
        # `asyncio.sleep` is awaitable, so defer it like a parallel call and gathered sleeps overlap.
        sleep = self.sleep(seconds)
        parallel = not self.global_sequential
        task = asyncio.ensure_future(sleep) if parallel else sleep
        self._pending[snapshot.call_id] = PendingCall(task, otel_context.get_current())
        return await call_monty(self.portal, snapshot.resume, ExternalFuture(future=...))

    async def _answer_os_call(self, snapshot: AsyncFunctionSnapshot, handler: OsHandler, name: OsFunction) -> Snapshot:
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

    async def _raise_in_sandbox(self, snapshot: FunctionSnapshot, exc: Exception) -> Snapshot:
        """Resume the suspended call by raising `exc` at its sandbox call site."""
        return await call_monty(self.portal, snapshot.resume, ExternalException(exception=exc))

    def _dispatch(self, snapshot: FunctionSnapshot, *, parallel: bool) -> PendingCall:
        context = snapshot.trace_context()
        token = otel_context.attach(context)
        try:
            call = self.dispatch(snapshot.function_name, snapshot.kwargs)
            # Tasks inherit the active context; bare coroutines need it restored when awaited.
            return PendingCall(asyncio.ensure_future(call) if parallel else call, context)
        finally:
            otel_context.detach(token)

    async def _resolve_futures(self, snapshot: AsyncFutureSnapshot | TurnFutureSnapshot) -> Snapshot:
        """Resolve the deferred calls a `FutureSnapshot` is waiting on."""
        if self.flush is not None:
            await self.flush()
        pending_ids = snapshot.pending_call_ids
        results: dict[int, ExternalSettledResult] = {}
        for cid in pending_ids:
            if cid in self._pre_resolved:
                results[cid] = self._pre_resolved.pop(cid)

        # Sequential modes resume the deferred calls in dispatch order: `_pending` is
        # insertion-ordered, but `pending_call_ids` may list the ids in any order.
        if self.global_sequential:
            for cid in [cid for cid in self._pending if cid in pending_ids]:
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
    if isinstance(outcome, BaseException):
        raise outcome
    return ExternalReturnValue(return_value=outcome)
