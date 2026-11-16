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

live_run_history: ContextVar[tuple[list[ModelMessage], int] | None] = ContextVar(
    'pydantic_ai_harness.step_persistence.live_run_history',
    default=None,
)
"""Async-context-local `(live message list, run_step)` refreshed at each node boundary.

The list is held by reference so `on_run_error` can persist the at-failure
history; `StepPersistence._stash_live_history` explains why and the core
invariant this relies on.

Reset to `None` in `wrap_run` so a run never inherits a prior run's tail.
Task-isolated like `current_run_id`.
"""
