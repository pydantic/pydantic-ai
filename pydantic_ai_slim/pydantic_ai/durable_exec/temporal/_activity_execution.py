"""Cancellation-safe activity/child-workflow execution for workflow code under anyio cancel scopes.

Awaiting `workflow.execute_activity()` or `workflow.execute_child_workflow()` directly from a task
inside an anyio cancel scope can livelock the workflow when that scope is cancelled (e.g. by
`asyncio.wait_for` cancelling an in-flight agent run — the graph engine runs its steps as anyio
task-group children):

- anyio delivers scope cancellation level-triggered: it re-arms `task.cancel()` via `call_soon`
  for as long as a task remains in the cancelled scope, deduplicating only on `task._must_cancel`
  or a done `_fut_waiter`.
- `task.cancel()` on a task awaiting an activity or child-workflow handle DELEGATES to the handle
  task, whose run loop swallows the `CancelledError`, emits a `request_cancel_activity` (or child
  workflow cancel) command, and re-parks on a shielded result future — leaving the scoped task
  uncancelled with an undone waiter, so anyio cancels it again on the next loop iteration, appending
  another cancel command. Both `ActivityHandle` and `ChildWorkflowHandle` share the same
  `_AsyncioTask`/`asyncio.Task` base and so both exhibit this exact shape.
- The resolution can only arrive in a later activation, which can never start because the event
  loop never goes idle: the spin continues until Temporal's deadlock detector fails the workflow
  task, which then retries identically forever.

These executors keep the cancellation delivery on OUR task instead (via `asyncio.shield`), forward
exactly one cancellation to Temporal's graceful machinery, and wait for the configured cancellation
type's resolution inside an anyio-shielded scope so re-delivery stops and the activation can
complete. Re-raising the original `CancelledError` also keeps standard asyncio semantics at the
caller: `asyncio.wait_for` produces `TimeoutError` instead of leaking the underlying `ActivityError`/
`ChildWorkflowError`, and cancelling the Temporal workflow still ends it as *Cancelled*.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import anyio
from temporalio import workflow
from temporalio.workflow import ActivityConfig, ChildWorkflowConfig
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


async def execute_child_workflow(
    workflow_: Any, *, args: Sequence[Any], result_type: Any = None, **config: Unpack[ChildWorkflowConfig]
) -> Any:
    """Drop-in replacement for `workflow.execute_child_workflow()` — see the module docstring for why.

    Only the wait on the returned `ChildWorkflowHandle` needs shielding, mirroring
    `execute_activity`: starting the child (the `await start_child_workflow(...)` below) is a single
    bounded round-trip, not the long-lived, cancel-and-rearm-prone wait the handle itself represents.

    `result_type` isn't part of `ChildWorkflowConfig` (unlike the rest of this function's keyword
    arguments) — it's `start_child_workflow`'s own parameter, kept separate here for the same reason.
    Typed `Any` rather than matching the SDK's own (too-narrow) `type | None`: a discriminated
    union like `CallToolResult` (`Annotated[Union[...], Discriminator(...)]`) is a legitimate
    `result_type` at runtime — Temporal's Pydantic-based converter handles `Annotated` types fine —
    but doesn't satisfy `type | None` statically.
    """
    handle = await workflow.start_child_workflow(workflow_, args=args, result_type=result_type, **config)
    try:
        return await asyncio.shield(handle)
    except asyncio.CancelledError:
        if not handle.done():  # pragma: no branch
            handle.cancel()
            with anyio.CancelScope(shield=True):
                await asyncio.wait([handle])
        raise
