"""A replay-safe clock for agent runs in a Hatchet durable task.

Hatchet replays a durable task by running its code again and checking every child task it spawns against
the one it recorded, input included, so the timestamps Pydantic AI stamps on messages must come out the
same on a replay. The clock starts at a time recorded with `DurableContext.aio_now()` and advances to each
child task's completion time, which its recorded output carries. A child task's result advances the clock
only in the context that awaited it, so parallel tool calls each read their own child's time, whatever
order a replay hands their results back in.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime

from pydantic_ai._utils import override_now_utc

_durable_now: ContextVar[datetime | None] = ContextVar('pydantic_ai.hatchet.durable_now', default=None)


def _now() -> datetime:
    now = _durable_now.get()
    assert now is not None, 'the durable clock is only installed while it is set'
    return now


@contextmanager
def durable_clock(start: datetime) -> Generator[None]:
    """Stamp times from the durable clock in this context, starting at `start`."""
    token = _durable_now.set(start)
    try:
        with override_now_utc(_now):
            yield
    finally:
        _durable_now.reset(token)


def advance_durable_clock(completed_at: datetime) -> None:
    """Move this context's durable clock forward to a child task's recorded completion time."""
    now = _durable_now.get()
    if now is not None and completed_at > now:
        _durable_now.set(completed_at)
