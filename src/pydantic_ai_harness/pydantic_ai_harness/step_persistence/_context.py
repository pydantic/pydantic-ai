"""Shared async-context state for `StepPersistence` cross-capability coordination."""

from __future__ import annotations

from contextvars import ContextVar

from pydantic_ai.messages import ModelMessage

current_run_id: ContextVar[str | None] = ContextVar(
    'pydantic_ai_harness.step_persistence.current_run_id',
    default=None,
)
"""Async-context-local pointer to the active `StepPersistence` `run_id`.

Set by `StepPersistence.wrap_run` for the duration of a run; read by a
nested capability's `for_run` to auto-fill `parent_run_id`, and by
`annotate_tool_effect` to find the in-flight tool's run scope.

Module-level rather than a class attribute so the helpers in `_helpers.py`
and the capability in `_capability.py` can share it without a circular
import.
"""

snapshot_saved: ContextVar[int] = ContextVar(
    'pydantic_ai_harness.step_persistence.snapshot_saved',
    default=0,
)
"""Async-context-local message count of the newest snapshot `after_node_run` saved this run.

Set to `0` in `wrap_run` and to `len(messages)` whenever `after_node_run` saves
a `CallToolsNode` snapshot. `after_run` compares its final history against it
and saves only when the run ended past that boundary, which keeps the common
case free of a redundant terminal snapshot: the final `CallToolsNode` already
captured the tail with the correct `step_index`, whereas `after_run` runs with
`ctx.run_step` reset to 0.

A count rather than a boolean because a `CallToolsNode` snapshot is no longer
evidence that the *final* history was captured. `Agent.run_stream` ends through
`SetFinalResult`, and its closing response lands in the history after the last
node boundary, so only `after_run` ever sees the whole run. Task-isolated like
`current_run_id`, so concurrent runs don't interfere.
"""

live_run_history: ContextVar[tuple[list[ModelMessage], int] | None] = ContextVar(
    'pydantic_ai_harness.step_persistence.live_run_history',
    default=None,
)
"""Async-context-local `(live message list, run_step)` refreshed at each node boundary.

The list is held by *reference*, not copied: `after_node_run` re-stashes
`ctx.messages` at every boundary, and `on_run_error` reads the reference's
current content to persist the at-failure history. The `RunContext` passed to
`on_run_error` carries the start-of-run history (a stale reference the graph
rebinds in `UserPromptNode.run`), not the live message list, so the hook cannot
read the partial history from its own `ctx.messages`. See the stash site in
`_capability.py` for the pydantic-ai invariant this leans on.

`run_step` is the last completed boundary's step, so an error-path snapshot's
`step_index` can lag the failing request by one.

Reset to `None` in `wrap_run` so a run never inherits a prior run's tail.
Task-isolated like `current_run_id`.
"""
