"""Keep telemetry from replayed durable work out of traces and metrics.

A durable engine resumes a run by replaying it: the agent loop runs again from the start, and every
durable unit that already completed returns its recorded result instead of running. The agent's
instrumentation runs again too, so without this module each replay re-emits the spans of work that
already happened, and records its token and cost metrics again.

An engine installs a [`DurableReplay`][pydantic_ai.durable_exec.DurableReplay] for the run with
[`replay_scope`][pydantic_ai.durable_exec.replay_scope]. Instrumentation takes a snapshot when it
opens a span and asks the replay at the end whether everything in between was served from the
journal: if so, it drops the span and skips the request's metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from opentelemetry.trace import Span, Tracer, use_span

__all__ = ('DurableReplay', 'ReplayCounter', 'replay_scope', 'current_replay', 'start_as_current_span')


class DurableReplay(ABC):
    """What a durable engine tells instrumentation about work replayed from its journal."""

    @abstractmethod
    def snapshot(self) -> Any:
        """Capture the replay state when instrumentation opens a span or starts a model request."""

    @abstractmethod
    def replayed_since(self, snapshot: Any) -> bool:
        """Whether all durable work since `snapshot` was served from the journal rather than run.

        Return `False` when nothing durable happened since the snapshot: that work ran for real.
        """


class ReplayCounter(DurableReplay):
    """A [`DurableReplay`][pydantic_ai.durable_exec.DurableReplay] for engines that know, per durable unit, whether it ran.

    Call [`unit_executed`][pydantic_ai.durable_exec.ReplayCounter.unit_executed] when a unit's body ran and
    [`unit_replayed`][pydantic_ai.durable_exec.ReplayCounter.unit_replayed] when its recorded result was
    returned instead. A span is replayed when units were replayed during it and none ran.
    """

    def __init__(self) -> None:
        self.executed = 0
        self.replayed = 0

    def unit_executed(self) -> None:
        """Record that a durable unit's body ran."""
        self.executed += 1

    def unit_replayed(self) -> None:
        """Record that a durable unit returned its recorded result without running."""
        self.replayed += 1

    def snapshot(self) -> tuple[int, int]:
        return self.executed, self.replayed

    def replayed_since(self, snapshot: tuple[int, int]) -> bool:
        executed, replayed = snapshot
        return self.executed == executed and self.replayed > replayed


_current_replay: ContextVar[DurableReplay | None] = ContextVar('_current_replay', default=None)


def current_replay() -> DurableReplay | None:
    """The replay installed for the current run, if a durable engine installed one."""
    return _current_replay.get()


@contextmanager
def replay_scope(replay: DurableReplay) -> Generator[DurableReplay]:
    """Install `replay` for the code inside, typically one durable agent run.

    Instrumentation running inside drops spans, and skips metrics, for work `replay` reports as replayed.
    """
    token = _current_replay.set(replay)
    try:
        yield replay
    finally:
        _current_replay.reset(token)


@contextmanager
def start_as_current_span(tracer: Tracer, name: str, **kwargs: Any) -> Generator[Span]:
    """`tracer.start_as_current_span(...)`, but a span that covered only replayed work is never ended, so never exported."""
    replay = _current_replay.get()
    if replay is None:
        with tracer.start_as_current_span(name, **kwargs) as span:
            yield span
        return

    snapshot = replay.snapshot()
    record_exception = kwargs.pop('record_exception', True)
    set_status_on_exception = kwargs.pop('set_status_on_exception', True)
    span = tracer.start_span(
        name, record_exception=record_exception, set_status_on_exception=set_status_on_exception, **kwargs
    )
    try:
        with use_span(
            span,
            end_on_exit=False,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
        ):
            yield span
    finally:
        if not replay.replayed_since(snapshot):
            span.end()
