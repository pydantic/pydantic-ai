"""Keep telemetry from replayed durable work out of traces and metrics.

A durable engine resumes a run by replaying it: the agent loop runs again from the start, and every
durable unit that already completed returns its recorded result instead of running. The agent's
instrumentation runs again too, so without this module each replay re-emits the spans of work that
already happened, and records its token and cost metrics again.

An engine wraps each durable agent run in [`replay_scope()`][pydantic_ai.durable_exec.replay_scope]
and reports each durable unit with [`record_durable_unit()`][pydantic_ai.durable_exec.record_durable_unit],
or, if it knows instead whether it is replaying at a given moment, passes `is_replaying`. Every span
instrumentation opens keeps a tally of the units that completed under it, in the context of the task
that requested them, so concurrent tool calls are told apart. A span whose units were all replayed is
never ended, so never exported, and a model request's metrics are skipped the same way.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from opentelemetry.trace import Span, Tracer, use_span

__all__ = ('ReplaySnapshot', 'record_durable_unit', 'replay_scope', 'snapshot', 'start_as_current_span')


@dataclass
class _Tally:
    """The durable units that completed under one span, or under the run when no span is open."""

    parent: _Tally | None
    executed: int = 0
    replayed: int = 0

    def credit(self, *, executed: bool) -> None:
        tally: _Tally | None = self
        while tally is not None:
            if executed:
                tally.executed += 1
            else:
                tally.replayed += 1
            tally = tally.parent


@dataclass(frozen=True)
class _Scope:
    is_replaying: Callable[[], bool] | None


_scope: ContextVar[_Scope | None] = ContextVar('_durable_replay_scope', default=None)
_tally: ContextVar[_Tally | None] = ContextVar('_durable_replay_tally', default=None)


@contextmanager
def replay_scope(*, is_replaying: Callable[[], bool] | None = None) -> Generator[None]:
    """Keep the telemetry of work replayed during the code inside, typically one durable agent run, out of traces and metrics.

    Args:
        is_replaying: For an engine that knows whether it is replaying at a given moment, like Temporal:
            a span or model request that ends while it returns `True` is not reported. An engine that
            knows instead whether each durable unit ran reports them with
            [`record_durable_unit()`][pydantic_ai.durable_exec.record_durable_unit].
    """
    scope_token = _scope.set(_Scope(is_replaying))
    tally_token = _tally.set(_Tally(parent=None))
    try:
        yield
    finally:
        _tally.reset(tally_token)
        _scope.reset(scope_token)


def record_durable_unit(*, executed: bool) -> None:
    """Report that a durable unit completed: its body ran (`executed=True`), or its recorded result was returned.

    Call it in the context of the task that requested the unit, so it is credited to the spans that
    unit ran under. Outside a [`replay_scope()`][pydantic_ai.durable_exec.replay_scope], it does nothing.
    """
    tally = _tally.get()
    if tally is not None:
        tally.credit(executed=executed)


@dataclass(frozen=True)
class ReplaySnapshot:
    """The replay state when an operation started, to tell when it ends whether it was only replayed."""

    _scope: _Scope
    _tally: _Tally
    _executed: int
    _replayed: int

    def replayed(self) -> bool:
        """Whether everything durable since the snapshot was replayed, so its telemetry was already recorded."""
        if self._scope.is_replaying is not None and self._scope.is_replaying():
            return True
        return self._tally.executed == self._executed and self._tally.replayed > self._replayed


def snapshot() -> ReplaySnapshot | None:
    """The current replay state, or `None` outside a [`replay_scope()`][pydantic_ai.durable_exec.replay_scope]."""
    tally = _tally.get()
    if _scope.get() is None or tally is None:
        return None
    return _snapshot_of(tally)


@contextmanager
def start_as_current_span(tracer: Tracer, name: str, **kwargs: Any) -> Generator[Span]:
    """`tracer.start_as_current_span(...)`, but a span that covered only replayed work is never ended, so never exported.

    A span that is reported counts as work that ran for the spans around it, so they are reported too.
    """
    parent = _tally.get()
    if _scope.get() is None or parent is None:
        with tracer.start_as_current_span(name, **kwargs) as span:
            yield span
        return

    record_exception = kwargs.pop('record_exception', True)
    set_status_on_exception = kwargs.pop('set_status_on_exception', True)
    span = tracer.start_span(
        name, record_exception=record_exception, set_status_on_exception=set_status_on_exception, **kwargs
    )
    own = _Tally(parent=parent)
    state = _snapshot_of(own)
    tally_token = _tally.set(own)
    try:
        with use_span(
            span,
            end_on_exit=False,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
        ):
            yield span
    finally:
        _tally.reset(tally_token)
        if not state.replayed():
            span.end()
            parent.credit(executed=True)


def _snapshot_of(tally: _Tally) -> ReplaySnapshot:
    scope = _scope.get()
    assert scope is not None
    return ReplaySnapshot(scope, tally, tally.executed, tally.replayed)
