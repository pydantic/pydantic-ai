"""Shared async-context state for `StepPersistence` cross-capability coordination."""

from __future__ import annotations

from contextvars import ContextVar

current_run_id: ContextVar[str | None] = ContextVar(
    'pydantic_ai_harness.experimental.step_persistence.current_run_id',
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

snapshot_saved: ContextVar[bool] = ContextVar(
    'pydantic_ai_harness.experimental.step_persistence.snapshot_saved',
    default=False,
)
"""Async-context-local flag: did `after_node_run` already save a snapshot this run?

Set `False` in `wrap_run`, flipped `True` whenever `after_node_run` saves a
`CallToolsNode` snapshot. `after_run` reads it to skip a redundant terminal
snapshot -- the final `CallToolsNode` already captured the provider-valid tail
with the correct `step_index`, whereas `after_run` runs with `ctx.run_step`
reset to 0. Task-isolated like `current_run_id`, so concurrent runs don't
interfere.
"""
