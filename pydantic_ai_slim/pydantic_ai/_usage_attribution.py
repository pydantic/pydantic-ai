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
enclosing run's on the way out; the `record_*` functions credit only that innermost run. Every run
enters it with `None` when it starts, so a run without a span credits no one rather than the
enclosing run. Asyncio copies the context when a task is created, so concurrent delegates each start
from the parent's and replace it with their own, never seeing each other's.

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

__all__ = ('accumulate', 'record_request', 'record_tool_call', 'record_usage')

_active: ContextVar[RunUsage | None] = ContextVar['RunUsage | None']('pydantic_ai.usage_attribution', default=None)


@contextmanager
def accumulate(run_usage: RunUsage | None) -> Generator[None]:
    """Credit `run_usage` with what is recorded in this context, until the block exits.

    A nested run replaces it for the length of its own span, so what the nested run records is its
    own; resetting on the way out hands crediting back to the enclosing run. `None` credits no run:
    each run enters that when it starts, so a run that never opens a span of its own doesn't have
    its usage reported on its caller's.
    """
    token = _active.set(run_usage)
    try:
        yield
    finally:
        _active.reset(token)


def record_request(usage: RunUsage) -> None:
    """Count one model request against this run's usage and against the run that made it."""
    usage.requests += 1
    if (run_usage := _active.get()) is not None:
        run_usage.requests += 1


def record_tool_call(usage: RunUsage) -> None:
    """Count one successful tool call against this run's usage and against the run that made it."""
    usage.tool_calls += 1
    if (run_usage := _active.get()) is not None:
        run_usage.tool_calls += 1


def record_usage(usage: RunUsage, recorded: RunUsage | RequestUsage) -> None:
    """Add recorded usage to this run's usage and to the run that produced it.

    `recorded` is one response's `RequestUsage`, or the `RunUsage` delta a durable operation
    accumulated across the boundary — which carries its own requests and tool calls.
    """
    usage.incr(recorded)
    if (run_usage := _active.get()) is not None:
        run_usage.incr(recorded)
