"""Record run usage, attributing it to the agent run that produced it.

An agent run span reports the usage of the requests *that run* made — not its nested runs', which
report their own. Summing the agent-run spans in a trace then gives the conversation's total,
matching the sum of the `chat` spans underneath them, which is the point of keeping agent-run usage
in its own attribute namespace: a backend can add these up without counting anything twice.

`RunUsage` alone can't say who produced what. It is accumulated into in place, and the multi-agent
delegation pattern hands the *same* object to concurrent delegates (`usage=ctx.usage`), so neither
the object's contents nor an end-minus-start delta on it distinguishes this run's requests from a
sibling's — concurrent delegates absorb each other.

Which run is producing is a property of the call stack, so that is what this follows. [`accumulate`]
[] makes a run's `RunUsage` the one credited for as long as its span is open, and restores the
enclosing run's on the way out; the `record_*` functions credit only that innermost run. Asyncio
copies the context when a task is created, so concurrent delegates each start from the parent's and
replace it with their own, never seeing each other's.

This module owns the *only* in-place mutation of a run's usage. Incrementing a `RunUsage` field
directly leaves its tokens off the run's span, so `tests/test_usage_attribution.py` fails on a bare
`requests += `, `tool_calls += `, or `.incr(` that isn't marked `# usage-attribution: ok` with a
reason.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

from .usage import RequestUsage, RunUsage

__all__ = ('accumulate', 'credit_applied', 'record_request', 'record_tool_call', 'record_usage', 'watch')

_active: ContextVar[RunUsage | None] = ContextVar['RunUsage | None']('pydantic_ai.usage_attribution', default=None)

# Mirrors keyed by the object recorded into, not by the recording task: a run sharing its usage with
# a concurrent delegate sees that delegate's records land on the same object from another task, and
# a caller asking "what was recorded into this object" means all of it. See `watch`.
#
# Keyed by `id(mirror)` rather than held in a list, because `RunUsage` compares by value: two runs
# watching usage that happens to hold the same numbers — two that have not recorded anything yet,
# say — would otherwise be indistinguishable to `list.remove`, which would drop whichever entry it
# found first and leave the other watcher to fail on the way out.
_mirrors: dict[int, tuple[RunUsage, RunUsage]] = {}


@contextmanager
def accumulate(run_usage: RunUsage) -> Generator[None]:
    """Credit `run_usage` with what is recorded in this context, until the block exits.

    A nested run replaces it for the length of its own span, so what the nested run records is its
    own; resetting on the way out hands crediting back to the enclosing run.
    """
    token = _active.set(run_usage)
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
    key = id(mirror)
    _mirrors[key] = (target, mirror)
    try:
        yield mirror
    finally:
        del _mirrors[key]


def _mirror(usage: RunUsage, recorded: RunUsage | RequestUsage) -> None:
    for target, mirror in list(_mirrors.values()):
        if target is usage:
            mirror.incr(recorded)


def record_request(usage: RunUsage) -> None:
    """Count one model request against this run's usage and against the run that made it."""
    usage.requests += 1
    _mirror(usage, RunUsage(requests=1))
    if (run_usage := _active.get()) is not None:
        run_usage.requests += 1


def record_tool_call(usage: RunUsage) -> None:
    """Count one successful tool call against this run's usage and against the run that made it."""
    usage.tool_calls += 1
    _mirror(usage, RunUsage(tool_calls=1))
    if (run_usage := _active.get()) is not None:
        run_usage.tool_calls += 1


def credit_applied(applied: RunUsage) -> None:
    """Credit usage already applied to a run's own object to the run that produced it.

    The durable-operation boundary is the one place that learns about usage after the fact: an
    operation executed in process adds to `ctx.usage` itself, while a replayed one reports a delta
    to fold in. Both have to reach the producing run, or the same capability would report
    different numbers on a first run and a replay.
    """
    if (run_usage := _active.get()) is not None:
        run_usage.incr(applied)


def record_usage(usage: RunUsage, recorded: RunUsage | RequestUsage) -> None:
    """Add recorded usage to this run's usage and to the run that produced it.

    `recorded` is one response's `RequestUsage`, or the `RunUsage` delta a durable operation
    accumulated across the boundary — which carries its own requests and tool calls.
    """
    usage.incr(recorded)
    _mirror(usage, recorded)
    if (run_usage := _active.get()) is not None:
        run_usage.incr(recorded)
