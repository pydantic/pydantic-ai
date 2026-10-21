"""Run-local Render Workflows task context."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

from render import TaskContext

_task_contexts: ContextVar[tuple[tuple[object, TaskContext], ...]] = ContextVar(
    'pydantic_ai_harness_render_task_contexts', default=()
)


def current_task_context(owner_token: object) -> TaskContext | None:
    """Return the Render context active for one capability instance, if any."""
    for owner, context in reversed(_task_contexts.get()):
        if owner is owner_token:
            return context
    return None


@contextmanager
def activate_task_context(owner_token: object, context: TaskContext) -> Generator[None, None, None]:
    """Make a task context available to nested operations owned by one capability."""
    token = _task_contexts.set((*_task_contexts.get(), (owner_token, context)))
    try:
        yield
    finally:
        _task_contexts.reset(token)
