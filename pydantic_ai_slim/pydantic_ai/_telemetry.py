"""The run's tracer and content policy, for code that has no `RunContext`.

Code with a `RunContext` traces on `ctx.tracer` and records content only when `ctx.trace_include_content`.
Code without one, such as a `Workspace` that runs share and that can outlive a run, reads the same
two values here, at call time: a value stored on such an object would apply one run's settings to
another's calls. The run publishes them in the context its body, tools and their tasks inherit; a
nested run publishes its own, and outside a run there is none, so nothing is traced.
"""

from __future__ import annotations as _annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from opentelemetry.trace import INVALID_SPAN, NoOpTracer, Span, Tracer
from opentelemetry.util.types import AttributeValue

from ._instrumentation import record_uncaught_errors

__all__ = (
    'RunTelemetry',
    'current_run_telemetry',
    'open_span',
    'publish_run_telemetry',
    'run_telemetry_scope',
    'set_content',
)


@dataclass(frozen=True, eq=False)
class RunTelemetry:
    """The tracer a run's spans go to, and whether they may record content.

    Not `InstrumentationSettings`: a durable worker combines its own settings with the run's
    serialized content flag. Compared by identity, so each publish is a distinct run's telemetry.
    """

    tracer: Tracer
    include_content: bool


_current: ContextVar[RunTelemetry | None] = ContextVar('pydantic_ai.run_telemetry', default=None)


def current_run_telemetry() -> RunTelemetry | None:
    """The calling run's telemetry, or `None` outside an instrumented run."""
    return _current.get()


def publish_run_telemetry(tracer: Tracer, *, include_content: bool) -> None:
    """Make `tracer` and `include_content` the current run's telemetry; a `NoOpTracer` means none.

    A plain set, for a context the run owns: the run's body may finish in a different `Context`
    than it started in, where resetting a token raises.
    """
    _current.set(_run_telemetry(tracer, include_content))


@contextmanager
def run_telemetry_scope(tracer: Tracer, *, include_content: bool) -> Generator[None]:
    """`publish_run_telemetry` for the duration of a block that starts and ends in one `Context`."""
    token = _current.set(_run_telemetry(tracer, include_content))
    try:
        yield
    finally:
        _current.reset(token)


def _run_telemetry(tracer: Tracer, include_content: bool) -> RunTelemetry | None:
    return None if isinstance(tracer, NoOpTracer) else RunTelemetry(tracer, include_content)


@contextmanager
def open_span(
    name: str,
    *,
    attributes: Mapping[str, AttributeValue] | None = None,
    not_errors: tuple[type[Exception], ...] = (),
) -> Generator[Span]:
    """Open `name` on the current run's tracer, recording uncaught errors per its content policy.

    Yields `INVALID_SPAN` outside an instrumented run. `attributes` are set at the start, so Logfire's
    console and pending spans show them; they must not hold content, which goes through
    `set_content`. An exception in `not_errors` (such as an expected `FileNotFoundError`) is not
    recorded as a failure.
    """
    if (telemetry := _current.get()) is None:
        yield INVALID_SPAN
        return
    with telemetry.tracer.start_as_current_span(
        name, attributes=attributes, record_exception=False, set_status_on_exception=False
    ) as span:
        with record_uncaught_errors(span, include_content=telemetry.include_content, not_errors=not_errors):
            yield span


def set_content(span: Span, key: str, value: AttributeValue) -> None:
    """Set an attribute that may hold content, such as a path or a command, only when the run captures content."""
    if (telemetry := _current.get()) is not None and telemetry.include_content:
        span.set_attribute(key, value)
