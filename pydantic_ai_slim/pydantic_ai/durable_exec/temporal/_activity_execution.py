"""Cancellation-safe activity execution for workflow code that runs under anyio cancel scopes.

Awaiting `workflow.execute_activity()` directly from a task inside an anyio cancel scope can
livelock the workflow when that scope is cancelled (e.g. by `asyncio.wait_for` cancelling an
in-flight agent run — the graph engine runs its steps as anyio task-group children):

- anyio delivers scope cancellation level-triggered: it re-arms `task.cancel()` via `call_soon`
  for as long as a task remains in the cancelled scope, deduplicating only on `task._must_cancel`
  or a done `_fut_waiter`.
- `task.cancel()` on a task awaiting an activity handle DELEGATES to the handle task, whose
  `run_activity` loop swallows the `CancelledError`, emits a `request_cancel_activity` command,
  and re-parks on a shielded result future — leaving the scoped task uncancelled with an undone
  waiter, so anyio cancels it again on the next loop iteration, appending another cancel command.
- The activity's resolution can only arrive in a later activation, which can never start because
  the event loop never goes idle: the spin continues until Temporal's deadlock detector fails the
  workflow task, which then retries identically forever.

This executor keeps the cancellation delivery on OUR task instead (by awaiting the handle through
`asyncio.wait`, which neither cancels it nor, unlike `asyncio.shield`, logs its error), forwards
exactly one cancellation to Temporal's graceful machinery, and waits for the configured
`ActivityCancellationType` resolution inside an anyio-shielded scope so re-delivery stops and the
activation can complete. Re-raising the original `CancelledError` also keeps standard asyncio
semantics at the caller: `asyncio.wait_for` produces `TimeoutError` instead of leaking the
activity's `ActivityError`, and cancelling the Temporal workflow still ends it as *Cancelled*.
The handle's own error is read before re-raising, so asyncio doesn't log it as never retrieved.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Any

import anyio
from temporalio import workflow
from temporalio.workflow import ActivityConfig, ChildWorkflowConfig
from typing_extensions import Unpack

from pydantic_ai.models._request_timeout import request_deadline_passed

CancelledResultHandler = Callable[[asyncio.CancelledError, Any], None]
"""Receives the cancellation and the result of an activity that completed after it was cancelled."""


async def execute_activity(
    activity: Any,
    *,
    args: Sequence[Any],
    on_cancelled_result: CancelledResultHandler | None = None,
    **config: Unpack[ActivityConfig],
) -> Any:
    """Drop-in replacement for `workflow.execute_activity()` — see the module docstring for why.

    An activity that completes after the cancellation was sent, which `WAIT_CANCELLATION_COMPLETED` waits for, has
    its result passed to `on_cancelled_result` with the cancellation, which is then still raised. A request deadline
    that cancels it doesn't keep that result, so it doesn't wait for it either.
    """
    return await _await_with_single_cancel(
        workflow.start_activity(activity, args=args, **config), on_cancelled_result=on_cancelled_result
    )


async def execute_child_workflow(
    workflow_: Any, *, args: Sequence[Any], result_type: Any, **config: Unpack[ChildWorkflowConfig]
) -> Any:
    """Drop-in replacement for `workflow.execute_child_workflow()`, for the same reason as `execute_activity`."""
    handle = await workflow.start_child_workflow(workflow_, args=args, result_type=result_type, **config)
    return await _await_with_single_cancel(handle)


async def _await_with_single_cancel(
    handle: asyncio.Future[Any], *, on_cancelled_result: CancelledResultHandler | None = None
) -> Any:
    try:
        await asyncio.wait([handle])
    except asyncio.CancelledError as cancellation:
        # The cancellation hit this task because `asyncio.wait` kept the Temporal handle alive.
        # Delegate exactly one cancellation to Temporal, then wait for its configured
        # cancellation behavior without anyio redelivering cancellation on every loop turn.
        # The handle is already done when the cancel lands in the same tick it resolves.
        if not handle.done():
            handle.cancel()
            if on_cancelled_result is not None and request_deadline_passed():
                # The request's deadline cancelled it, which raises `ModelRequestTimeout` instead, so there's nothing
                # to hand over: leave the activity to acknowledge the cancellation in its own time, as a
                # `TRY_CANCEL` activity does, rather than hold up the timeout and any fallback by a heartbeat.
                handle.add_done_callback(_retrieve_error)
                raise
            with anyio.CancelScope(shield=True):
                await asyncio.wait([handle])
        # Mark the handle's error (the cancelled activity's `ActivityError`) as retrieved,
        # or asyncio logs it when the handle is garbage collected. An activity that completed
        # instead hands its result over with the cancellation, which is still raised.
        if not handle.cancelled() and handle.exception() is None and on_cancelled_result is not None:
            on_cancelled_result(cancellation, handle.result())
        raise
    return handle.result()


def _retrieve_error(handle: asyncio.Future[Any]) -> None:
    """Mark the error a handle nothing awaits resolved with as retrieved, so asyncio doesn't log it."""
    if not handle.cancelled():
        handle.exception()
