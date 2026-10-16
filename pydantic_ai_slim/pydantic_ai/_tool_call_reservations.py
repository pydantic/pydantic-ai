"""Count tool calls that runs sharing a `RunUsage` have approved but not finished against `tool_calls_limit`.

A run checks `tool_calls_limit` before it executes a batch of tool calls, by adding the batch to
`RunUsage.tool_calls`, while `RunUsage.tool_calls` only counts a call once it has succeeded. A run doesn't
check again until its batch has finished, so on its own a run never misses a call.

Runs that share one `RunUsage` overlap, though. A delegate agent run from a tool with `usage=ctx.usage`
checks while the delegating call is still running, and delegates run in parallel check while their
siblings' calls are still running. None of those calls are counted yet, so each run passes a check that
the runs together exceed, by as much as the width of the fan-out.

So each call of an approved batch is reserved here until it's counted or known not to be, and a run
checking a batch also counts the calls reserved against the same `RunUsage`. A run's own batch is never
reserved when it checks, so a run that doesn't share its `RunUsage` checks exactly as before, and the
reservations live here rather than on `RunUsage`, so the usage a run reports doesn't change.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from .messages import ToolCallPart
from .usage import RunUsage

__all__ = (
    'ToolCallReservation',
    'release_counted_tool_call',
    'reserved_enclosing_tool_calls',
    'reserved_tool_calls',
)

_reserved: dict[int, int] = {}
"""How many calls are reserved against each `RunUsage`, by `id`, while any are.

`RunUsage` compares by value, so it can't be a key itself. An entry only exists while a reservation holds
calls against it, and a reservation keeps its `RunUsage` alive, so the `id` can't be reused meanwhile.
"""


@dataclass(eq=False)
class _RunningCall:
    reservation: ToolCallReservation
    call: ToolCallPart
    enclosing: _RunningCall | None
    """The reserved call this one runs inside of, if any, like a delegating call around a delegate's call."""
    released: bool = False

    def release(self) -> None:
        if not self.released:
            self.released = True
            self.reservation.release(self.call.tool_call_id)


_running_call: ContextVar[_RunningCall | None] = ContextVar('pydantic_ai.running_tool_call', default=None)
"""The reserved call this task is executing, so it can be released at the moment it's counted.

Each call runs in its own task, so concurrent calls each see their own.
"""


def reserved_tool_calls(usage: RunUsage) -> int:
    """How many approved calls on `usage` have not been counted in `usage.tool_calls`, or found not to be, yet."""
    return _reserved.get(id(usage), 0)


def reserved_enclosing_tool_calls(usage: RunUsage) -> int:
    """How many of the calls reserved against `usage` are running the current task, like the call delegating to it.

    Unlike other reserved calls, these stay reserved for as long as the current task runs.
    """
    enclosing: list[_RunningCall] = []
    running = _running_call.get()
    while running is not None:
        enclosing.append(running)
        running = running.enclosing
    return sum(not call.released and call.reservation.usage is usage for call in enclosing)


class ToolCallReservation:
    """The calls of one approved batch that are still reserved against the batch's `RunUsage`.

    Calls are held by `tool_call_id`, which outlives the `ToolCallPart` an approval's `override_args` replaces.
    """

    def __init__(self, usage: RunUsage, calls: Sequence[ToolCallPart]) -> None:
        self.usage = usage
        self._held = Counter(call.tool_call_id for call in calls)
        _reserved[id(usage)] = reserved_tool_calls(usage) + len(calls)

    @contextmanager
    def running(self, call: ToolCallPart) -> Generator[None]:
        """Execute `call`, releasing its reservation once it's counted, or at the latest once it's done.

        A call this reservation doesn't hold (one approved inline after the batch deferred it) runs as is.
        """
        if not self._held[call.tool_call_id]:
            yield
            return
        running = _RunningCall(self, call, enclosing=_running_call.get())
        token = _running_call.set(running)
        try:
            yield
        finally:
            _running_call.reset(token)
            running.release()

    def release_all(self) -> None:
        """Release every call still reserved, once the batch is done."""
        for tool_call_id, count in list(self._held.items()):
            self.release(tool_call_id, count)

    def release(self, tool_call_id: str, count: int = 1) -> None:
        """Release up to `count` of the calls held under `tool_call_id`."""
        count = min(count, self._held[tool_call_id])
        if not count:
            return
        self._held[tool_call_id] -= count
        key = id(self.usage)
        if remaining := _reserved[key] - count:
            _reserved[key] = remaining
        else:
            del _reserved[key]


def release_counted_tool_call(call: ToolCallPart) -> None:
    """Release `call`'s reservation now that it's counted in `RunUsage.tool_calls`.

    Called right after the count, with no `await` in between, so no other run sees the call both reserved
    and counted. Only the call this task is executing is released: calls counted inside it, by a tool that
    executes other tools of its run or by a delegate agent run, are not that call.
    """
    running = _running_call.get()
    if running is not None and running.call is call:
        running.release()
