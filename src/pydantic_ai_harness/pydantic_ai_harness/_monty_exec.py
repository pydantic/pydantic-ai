"""Shared Monty execution loop for code-execution capabilities.

Drives a Monty REPL via the synchronous snapshot API (`feed_start`/`resume`),
dispatching external function calls back to a host-supplied async callback.

Two capabilities build on this:

- `code_mode`: the dispatch callback runs the agent's own tools.
- `dynamic_workflow`: the dispatch callback runs sub-agents.

The synchronous snapshot API (rather than `feed_run_async`) is used deliberately:
it avoids background threads and `call_soon_threadsafe`, so the loop is safe inside
restricted event loops such as Temporal's workflow sandbox.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Container, Coroutine
from dataclasses import dataclass, field
from typing import Any

try:
    from pydantic_monty import (
        AbstractOS,
        ExternalException,
        ExternalResult,
        ExternalReturnValue,
        FunctionSnapshot,
        FutureSnapshot,
        MontyComplete,
        MountDir,
        NameLookupSnapshot,
        OsFunction,
    )
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'pydantic-monty is required for code-execution capabilities. Install it with: '
        'pip install "pydantic-ai-harness[code-mode]" or "pydantic-ai-harness[dynamic-workflow]"'
    ) from _import_error

# Dispatch callback: given the sandbox function name and keyword arguments,
# perform the host-side work (tool call or sub-agent run) and return the result.
DispatchFn = Callable[[str, dict[str, Any]], Coroutine[Any, Any, Any]]

MontyState = FunctionSnapshot | FutureSnapshot | NameLookupSnapshot | MontyComplete

# OS handler and host-directory mounts that route environment, clock, and filesystem
# calls from inside the sandbox. A raw callback or a ready-made `AbstractOS` is accepted.
MontyOSCallback = Callable[[OsFunction, tuple[object, ...], dict[str, object]], object]
MontyOS = AbstractOS | MontyOSCallback
MontyMount = MountDir | list[MountDir]

# A coroutine not yet scheduled on the event loop, or its running Task.
PendingCall = asyncio.Task[Any] | Coroutine[Any, Any, Any]


def is_sandbox_panic(exc: BaseException) -> bool:
    """Whether `exc` is a Rust-side sandbox panic surfacing through pyo3.

    pyo3 raises `pyo3_runtime.PanicException`, a `BaseException` (not `Exception`) subclass
    from a module that cannot be imported, so it is matched by name. The model can provoke a
    panic from inside the sandbox (e.g. awaiting the same external call twice in one
    `asyncio.gather`), so callers should convert it to a retry rather than let it tear down
    the whole agent run.
    """
    return type(exc).__name__ == 'PanicException'


class PrintCapture:
    """Accumulates print-callback chunks from a Monty REPL."""

    def __init__(self) -> None:
        self._chunks: list[str] = []

    def __call__(self, _stream: str, text: str) -> None:
        self._chunks.append(text)

    @property
    def joined(self) -> str:
        return ''.join(self._chunks)

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
    - **Global sequential** (DBOS/Temporal): all calls deferred but stored as bare
      coroutines and awaited one-at-a-time to prevent interleaving.
    """

    dispatch: DispatchFn
    valid_names: Container[str]
    sequential_names: set[str] = field(default_factory=set[str])
    global_sequential: bool = False
    # OS handler and mounts. Monty auto-dispatches OS calls only while every `resume`
    # carries them, so they are threaded through each resume below, not just `feed_start`.
    os_access: MontyOS | None = None
    mount: MontyMount | None = None

    # Parallel calls deferred but not yet resolved, keyed by Monty call id.
    _pending: dict[int, PendingCall] = field(default_factory=dict[int, PendingCall], init=False)
    # Parallel results awaited early at a sequential barrier, before their FutureSnapshot is reached.
    _pre_resolved: dict[int, ExternalResult] = field(default_factory=dict[int, ExternalResult], init=False)

    async def run(self, state: MontyState) -> MontyComplete:
        """Drive the REPL from `state` until it completes."""
        try:
            while not isinstance(state, MontyComplete):
                if isinstance(state, NameLookupSnapshot):
                    state = state.resume(os=self.os_access, mount=self.mount)
                elif isinstance(state, FunctionSnapshot):
                    state = await self._handle_function(state)
                else:
                    state = await self._resolve_futures(state)
        finally:
            cancelled: list[asyncio.Task[Any]] = []
            for call in self._pending.values():
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
                # results are deliberately discarded.
                await asyncio.gather(*cancelled, return_exceptions=True)
        return state

    async def _handle_function(self, snapshot: FunctionSnapshot) -> MontyState:
        """Dispatch (or defer) a single external function call."""
        name = snapshot.function_name
        if name not in self.valid_names:
            return snapshot.resume(
                {'exception': NameError(f'Unknown function: {name}')}, os=self.os_access, mount=self.mount
            )
        if snapshot.args:
            return snapshot.resume(
                {'exception': TypeError(f'{name}() does not accept positional arguments; use keyword arguments')},
                os=self.os_access,
                mount=self.mount,
            )

        if name in self.sequential_names:
            # Rendered as `def` (sync), so the sandbox code doesn't `await` the result --
            # resolve inline. Await pending parallel tasks first (barrier) for ordering.
            # The dispatch coroutine is created only after the barrier: it is not in
            # `_pending`, so if it existed while the barrier awaits and we were cancelled
            # there, `run`'s cleanup would never close it.
            for cid in list(self._pending):
                self._pre_resolved[cid] = await _await_external(self._pending.pop(cid))
            # The wrapped outcome (`{'return_value': ...}` / `{'exception': ...}`) is already
            # exactly the payload `resume` expects.
            return snapshot.resume(
                await _await_external(self.dispatch(name, snapshot.kwargs)), os=self.os_access, mount=self.mount
            )

        # Deferred execution -- resolved later at FutureSnapshot.
        call = self.dispatch(name, snapshot.kwargs)
        if self.global_sequential:
            # Keep the bare coroutine unscheduled; it's awaited one-at-a-time to avoid interleaving.
            self._pending[snapshot.call_id] = call
        else:
            # Schedule now as a Task so concurrently-deferred calls actually run in parallel.
            self._pending[snapshot.call_id] = asyncio.ensure_future(call)
        return snapshot.resume({'future': ...}, os=self.os_access, mount=self.mount)

    async def _resolve_futures(self, snapshot: FutureSnapshot) -> MontyState:
        """Resolve the deferred calls a `FutureSnapshot` is waiting on."""
        pending_ids = snapshot.pending_call_ids
        results: dict[int, ExternalResult] = {}
        for cid in pending_ids:
            if cid in self._pre_resolved:
                results[cid] = self._pre_resolved.pop(cid)
            elif self.global_sequential:
                results[cid] = await _await_external(self._pending.pop(cid))

        # Gather any remaining parallel tasks concurrently. They stay in `_pending` until
        # gather returns, so the cleanup in `run` can still cancel them if this is cancelled.
        gather_ids = [cid for cid in pending_ids if cid not in results]
        if gather_ids:
            settled = await asyncio.gather(*(self._pending[cid] for cid in gather_ids), return_exceptions=True)
            for cid, outcome in zip(gather_ids, settled):
                del self._pending[cid]
                results[cid] = _wrap_gathered(outcome)

        return snapshot.resume(results=results, os=self.os_access, mount=self.mount)


async def _await_external(call: PendingCall) -> ExternalReturnValue | ExternalException:
    """Await a single deferred call and wrap its outcome for Monty."""
    try:
        result = await call
    except Exception as exc:
        return ExternalException(exception=exc)
    return ExternalReturnValue(return_value=result)


def _wrap_gathered(outcome: Any) -> ExternalReturnValue | ExternalException:
    """Wrap an `asyncio.gather(return_exceptions=True)` outcome for Monty."""
    if isinstance(outcome, Exception):
        return ExternalException(exception=outcome)
    if isinstance(outcome, BaseException):  # pragma: no cover
        raise outcome
    return ExternalReturnValue(return_value=outcome)
