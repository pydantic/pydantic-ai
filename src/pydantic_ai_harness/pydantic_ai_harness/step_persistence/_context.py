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

snapshot_saved: ContextVar[bool] = ContextVar(
    'pydantic_ai_harness.step_persistence.snapshot_saved',
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

latest_node_history: ContextVar[tuple[list[ModelMessage], int] | None] = ContextVar(
    'pydantic_ai_harness.step_persistence.latest_node_history',
    default=None,
)
"""Async-context-local `(messages, step_index)` captured at the last completed node.

`after_node_run` refreshes this at every node boundary; `on_run_error` reads it
to rescue a provider-valid resume point that a node reached but that no
`after_node_run` persisted -- e.g. a text response whose subsequent
`CallToolsNode` raises inside output validation. The `RunContext` passed to
`on_run_error` carries the start-of-run history (a stale reference the graph
rebinds during the run), not the live message list, so the hook cannot read the
partial history from `ctx.messages` directly.

Reset to `None` in `wrap_run` so a run never inherits a prior run's tail.
Task-isolated like `current_run_id`.
"""
