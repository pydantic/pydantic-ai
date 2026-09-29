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

This executor keeps the cancellation delivery on OUR task instead (via `asyncio.shield`), forwards
exactly one cancellation to Temporal's graceful machinery, and waits for the configured
`ActivityCancellationType` resolution inside an anyio-shielded scope so re-delivery stops and the
activation can complete. Re-raising the original `CancelledError` also keeps standard asyncio
semantics at the caller: `asyncio.wait_for` produces `TimeoutError` instead of leaking the
activity's `ActivityError`, and cancelling the Temporal workflow still ends it as *Cancelled*.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import anyio
from temporalio import workflow
from temporalio.workflow import ActivityConfig
from typing_extensions import Unpack


async def execute_activity(activity: Any, *, args: Sequence[Any], **config: Unpack[ActivityConfig]) -> Any:
    """Drop-in replacement for `workflow.execute_activity()` — see the module docstring for why."""
    handle = workflow.start_activity(activity, args=args, **config)
    try:
        return await asyncio.shield(handle)
    except asyncio.CancelledError:
        # The cancellation hit this task because the shield kept the activity handle alive.
        # Delegate exactly one cancellation to Temporal, then wait for its configured
        # cancellation behavior without anyio redelivering cancellation on every loop turn.
        # The already-done arm is a real race (cancel landing in the same tick the activity
        # resolves) that cannot be timed deterministically through the workflow API.
        if not handle.done():  # pragma: no branch
            handle.cancel()
            with anyio.CancelScope(shield=True):
                await asyncio.wait([handle])
        raise
