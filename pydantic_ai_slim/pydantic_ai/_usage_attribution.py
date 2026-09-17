"""Record run usage, attributing it to every agent run whose span is open.

An agent run span reports the usage of every model request in its span subtree, which is what
`docs/logfire.md` promises and what backends summing root agent-run spans rely on. `RunUsage` alone
can't deliver that: it is accumulated into in place, and the multi-agent delegation pattern hands
the *same* object to concurrent delegates (`usage=ctx.usage`), so neither the object's contents nor
an end-minus-start delta on it says which run added what — concurrent siblings absorb each other.

Attribution has to follow the task stack instead. [`accumulate`][] pushes a run's own `RunUsage` for
as long as its span is open; the `record_*` functions credit every accumulator on the stack, so an
increment made inside a delegate lands on the delegate *and* on the ancestors that contain it.
Asyncio copies the context when a task is created, so concurrent delegates each inherit the parent's
stack and extend it with their own accumulator, never with each other's.

This module owns the *only* in-place mutation of a run's usage. Incrementing a `RunUsage` field
directly leaves its tokens off every containing span, so `tests/test_usage_attribution.py` fails on
a bare `requests += `, `tool_calls += `, or `.incr(` that isn't marked `# usage-attribution: ok`
with a reason.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

from .usage import RequestUsage, RunUsage

__all__ = ('accumulate', 'credit_applied', 'record_request', 'record_tool_call', 'record_usage', 'watch')

_active: ContextVar[tuple[RunUsage, ...]] = ContextVar['tuple[RunUsage, ...]'](
    'pydantic_ai.usage_attribution', default=()
)

# Mirrors keyed by the object recorded into, not by the recording task: a run sharing its usage with
# a concurrent delegate sees that delegate's records land on the same object from another task, and
# a caller asking "what was recorded into this object" means all of it. See `watch`.
_mirrors: list[tuple[RunUsage, RunUsage]] = []


@contextmanager
def accumulate(run_usage: RunUsage) -> Generator[None]:
    """Credit `run_usage` with everything recorded in this context until the block exits."""
    token = _active.set((*_active.get(), run_usage))
    try:
        yield
    finally:
        _active.reset(token)


@contextmanager
def watch(target: RunUsage) -> Generator[RunUsage]:
    """Mirror everything recorded into `target` while the block runs.

    Unlike [`accumulate`][], this follows the *object* rather than the task, so it also sees what a
    concurrently running sibling records into the same shared `RunUsage`. That is what makes the
    difference from the object's own delta the part nothing recorded — a direct mutation.
    """
    mirror = RunUsage()
    entry = (target, mirror)
    _mirrors.append(entry)
    try:
        yield mirror
    finally:
        _mirrors.remove(entry)


def _mirror(usage: RunUsage, recorded: RunUsage | RequestUsage) -> None:
    for target, mirror in _mirrors:
        if target is usage:
            mirror.incr(recorded)


def record_request(usage: RunUsage) -> None:
    """Count one model request against this run's usage and every run containing it."""
    usage.requests += 1
    _mirror(usage, RunUsage(requests=1))
    for run_usage in _active.get():
        run_usage.requests += 1


def record_tool_call(usage: RunUsage) -> None:
    """Count one successful tool call against this run's usage and every run containing it."""
    usage.tool_calls += 1
    _mirror(usage, RunUsage(tool_calls=1))
    for run_usage in _active.get():
        run_usage.tool_calls += 1


def credit_applied(applied: RunUsage) -> None:
    """Credit usage already applied to a run's own object to the runs containing it.

    The durable-operation boundary is the one place that learns about usage after the fact: an
    operation executed in process adds to `ctx.usage` itself, while a replayed one reports a delta
    to fold in. Both have to reach the containing spans, or the same capability would report
    different numbers on a first run and a replay.
    """
    for run_usage in _active.get():
        run_usage.incr(applied)


def record_usage(usage: RunUsage, recorded: RunUsage | RequestUsage) -> None:
    """Add recorded usage to this run's usage and to every run containing it.

    `recorded` is one response's `RequestUsage`, or the `RunUsage` delta a durable operation
    accumulated across the boundary — which carries its own requests and tool calls.
    """
    usage.incr(recorded)
    _mirror(usage, recorded)
    for run_usage in _active.get():
        run_usage.incr(recorded)
