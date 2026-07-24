"""Run-scoped cancellation controller for first-party run cancellation.

First-party cancellation (`AgentRun.cancel()`, `RunContext.cancel_run()`) is implemented by
cancelling the asyncio task that drives the run: that wakes whatever the run is blocked on (a
model stream, tool tasks, a suspended-job poll) and reuses the exact same teardown machinery as
external cancellation — streams are closed, in-flight tool tasks are cancelled and drained,
suspended server-side jobs are best-effort cancelled, and completed work is recorded to message
history. At the outer edge of [`Agent.iter()`][pydantic_ai.agent.Agent.iter], after teardown, the
resulting `CancelledError` is translated back into
[`RunCancelled`][pydantic_ai.exceptions.RunCancelled] — but only if the cancellation was ours:

- The controller counts every `Task.cancel()` it issues. On catching `CancelledError`, the
  outer edge consumes exactly that many cancellations via `Task.uncancel()` (mirroring what
  `asyncio.timeout()` does for its own cancellation).
- If `Task.cancelling()` is still positive afterwards, an *external* cancellation raced in; it
  wins, and the `CancelledError` keeps propagating as itself.

On Python 3.10, `Task.cancelling()`/`Task.uncancel()` don't exist, so the race cannot be
disambiguated: a requested first-party cancellation is translated to `RunCancelled` even if an
external cancellation arrived at the same time (documented degraded behavior).

The controller is runtime-only state: it holds a live task reference and is never serialized.
"""

from __future__ import annotations as _annotations

import asyncio
import sys

__all__ = ('RunCancellation',)


class RunCancellation:
    """Tracks first-party cancellation of a single agent run.

    One instance per run, shared (by reference) between the run's public handles and its
    internals. The task driving the run binds itself with [`bind`][pydantic_ai._cancel.RunCancellation.bind]
    at each step boundary, so `cancel()` always cancels the task currently doing the work.
    """

    def __init__(self) -> None:
        self._owner: asyncio.Task[object] | None = None
        self._issued = 0
        self._issued_to: asyncio.Task[object] | None = None
        self._requested = False
        self._finished = False

    @property
    def cancel_requested(self) -> bool:
        """Whether a first-party cancellation has been requested (and not yet resolved)."""
        return self._requested

    def bind(self, task: asyncio.Task[object] | None = None) -> None:
        """Bind the task that is currently driving the run.

        Called at run start and at each step boundary, so manual `AgentRun.next()` driving from
        a different task than the one that started the run still gets cancelled correctly.
        If a cancellation was requested before any task was bound (e.g. `cancel()` on a
        lazily-started run) or was issued to a previous driving task, it is (re-)delivered to
        this one.
        """
        task = task or asyncio.current_task()
        if task is None:  # pragma: no cover — agent runs always execute inside a task
            return
        self._owner = task
        if self._issued_to is not None and self._issued_to is not task:
            # Cancellations issued to a previous driving task can never be resolved on this
            # one; forget them so they can't mis-consume this task's external cancellations.
            self._issued = 0
            self._issued_to = None
        if self._requested and not self._finished and self._issued == 0:
            # Deliver a request that arrived before this task was bound.
            self._issue(task)

    def cancel(self) -> None:
        """Request cancellation of the run. Idempotent; a no-op once the run has finished."""
        if self._finished or self._requested:
            return
        self._requested = True
        if self._owner is not None and not self._owner.done():
            self._issue(self._owner)

    def _issue(self, task: asyncio.Task[object]) -> None:
        self._issued += 1
        self._issued_to = task
        task.cancel()

    def finish(self) -> None:
        """Mark the run as finished: later `cancel()` calls become no-ops."""
        self._finished = True

    def resolve(self) -> bool:
        """Resolve a caught `CancelledError` at the run's outer edge: is it ours to translate?

        Consumes exactly the cancellations this controller issued via `Task.uncancel()`. Returns
        `True` if the cancellation was first-party and no external cancellation is still pending
        (translate to `RunCancelled`); `False` if it must keep propagating as `CancelledError`.

        Must be called on the task the cancellation was delivered to.
        """
        if not self._requested:
            return False
        if sys.version_info < (3, 11):  # pragma: lax no cover
            # No `Task.uncancel()`/`Task.cancelling()`: we can't tell whether an external
            # cancellation raced with ours, so a requested cancellation wins (documented).
            return True
        task = asyncio.current_task()
        if task is None:  # pragma: no cover — agent runs always execute inside a task
            return True
        while self._issued > 0 and task.cancelling() > 0:
            task.uncancel()
            self._issued -= 1
        # Anything left on the counter was issued externally and takes precedence.
        return task.cancelling() == 0
