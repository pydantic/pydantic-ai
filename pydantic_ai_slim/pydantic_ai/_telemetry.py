"""Spans on a run's tracer that record content only when the run allows it.

- `open_span(ctx, name)`: a span around work that takes time and can fail.
- `record_decision(ctx, name)`: a decision that changed the run's course, currently a zero-duration span.
- `current_span(ctx)`: the span already open where the caller runs, such as a tool hook's `execute_tool`.
- `set_content_attribute(ctx, span, key, value)`: an attribute that can quote content, set only when
  the run captures content.

The first argument names the run: its `ctx`, whose `ctx.tracer` and `ctx.trace_include_content` are
used, or, for code that runs during a run but has no `RunContext` (such as a `Workspace` that runs
share and that can outlive a run), `current_run_telemetry()`, which the run publishes for the context
its body, tools and their tasks inherit. A value stored on such an object would apply one run's
settings to another's calls. `None` means no run: nothing is traced.

Pass `ctx` whenever there is one: the published value only exists once `wrap_run` starts, so a
`for_run` hook that read it would find nothing, or in a nested run its parent's.
"""

from __future__ import annotations as _annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from opentelemetry.trace import INVALID_SPAN, NoOpTracer, Span, Tracer, get_current_span
from opentelemetry.util.types import AttributeValue

from ._instrumentation import record_exception, record_uncaught_errors, set_error_status

if TYPE_CHECKING:
    from ._run_context import RunContext

__all__ = (
    'RunTelemetry',
    'current_run_telemetry',
    'current_span',
    'open_span',
    'publish_run_telemetry',
    'record_decision',
    'records_content',
    'run_telemetry_scope',
    'set_content_attribute',
)


@dataclass(frozen=True)
class RunTelemetry:
    """The tracer a run's spans go to, and whether they may record content.

    [`current_run_telemetry()`][pydantic_ai.telemetry.current_run_telemetry] returns the calling run's, for
    code that has no `RunContext`.
    """

    # Not `InstrumentationSettings`: in a durable unit both come from the rebuilt `ctx`, whose content flag
    # the engine narrowed to the worker's setting. One built from a `ctx` is used for a single call and never
    # published.
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
def run_telemetry_scope(telemetry: RunTelemetry | None) -> Generator[None]:
    """Publish `telemetry` for a block that starts and ends in one `Context`, for code that has no `RunContext`."""
    token = _current.set(_telemetry_of(telemetry))
    try:
        yield
    finally:
        _current.reset(token)


def _run_telemetry(tracer: Tracer, include_content: bool) -> RunTelemetry | None:
    return None if isinstance(tracer, NoOpTracer) else RunTelemetry(tracer, include_content)


@contextmanager
def open_span(
    ctx: RunContext[Any] | RunTelemetry | None,
    name: str,
    *,
    attributes: Mapping[str, AttributeValue] | None = None,
    content: Mapping[str, AttributeValue] | None = None,
    not_errors: tuple[type[BaseException], ...] = (),
) -> Generator[Span]:
    """Open `name` on the run's tracer.

    Yields `INVALID_SPAN` outside an instrumented run. `attributes` are always recorded; `content`,
    which can quote what the span is about, only when the run captures content. Both are set at the
    start, so Logfire's console and pending spans show them. An uncaught error, including a
    cancellation, marks the span ERROR and is recorded by its type only, unless the run captures
    content, as its message can quote content. An exception in `not_errors`, such as an expected
    `FileNotFoundError` or a cancellation the caller expects, is still recorded as an `exception`
    event, but the span's status is left unset rather than marked ERROR.
    """
    if (telemetry := _telemetry_of(ctx)) is None:
        yield INVALID_SPAN
        return
    with telemetry.tracer.start_as_current_span(
        name,
        attributes=_with_content(telemetry, attributes, content),
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        try:
            with record_uncaught_errors(span, include_content=telemetry.include_content, not_errors=not_errors):
                yield span
        except Exception:
            raise
        except BaseException as error:
            # `use_span` also marks a cancellation, which `record_uncaught_errors` leaves alone.
            record_exception(span, error, include_content=telemetry.include_content, escaped=False)
            if not isinstance(error, not_errors):
                set_error_status(span, error, include_content=telemetry.include_content)
            raise


def record_decision(
    ctx: RunContext[Any] | RunTelemetry | None,
    name: str,
    *,
    attributes: Mapping[str, AttributeValue] | None = None,
    content: Mapping[str, AttributeValue] | None = None,
) -> None:
    """Record that a capability decided something that changed the run's course, such as a guard blocking a value.

    The record is currently a zero-duration span named `name`, on the run's tracer. `attributes` are
    always recorded; `content`, which can quote what the decision was about, only when the run captures
    content.
    """
    if (telemetry := _telemetry_of(ctx)) is not None:
        telemetry.tracer.start_span(name, attributes=_with_content(telemetry, attributes, content)).end()


def set_content_attribute(
    ctx: RunContext[Any] | RunTelemetry | None, span: Span, key: str, value: AttributeValue
) -> None:
    """Set an attribute that can quote content, such as a path or a command, only when the run captures content.

    For content known when the span opens, pass `content=` to `open_span` instead.
    """
    if records_content(ctx):
        span.set_attribute(key, value)


def records_content(ctx: RunContext[Any] | RunTelemetry | None) -> bool:
    """Whether the run's spans record content, to skip building a content value that costs something or can raise."""
    return (telemetry := _telemetry_of(ctx)) is not None and telemetry.include_content


def current_span(ctx: RunContext[Any] | RunTelemetry | None) -> Span:
    """The span open where the caller runs, to add attributes to, or `INVALID_SPAN` when the run isn't traced.

    In a tool hook that is the call's `execute_tool` span, and in `before_model_request` the request's
    `chat` span. Unlike `opentelemetry.trace.get_current_span()`, it returns `INVALID_SPAN` when the run
    isn't traced, so an uninstrumented run doesn't write to the application's own span. In `for_run`, or
    in a capability ordered outside `Instrumentation`, the open span is the one around the run.
    """
    return get_current_span() if _telemetry_of(ctx) is not None else INVALID_SPAN


_TRACE_FIELDS = frozenset({'tracer', 'trace_include_content'})


def _telemetry_of(ctx: RunContext[Any] | RunTelemetry | None) -> RunTelemetry | None:
    if ctx is None:
        return None
    if isinstance(ctx, RunTelemetry):
        # Built by hand, as a durable worker does, it may wrap an uninstrumented run's `NoOpTracer`.
        return None if isinstance(ctx.tracer, NoOpTracer) else ctx
    # A context rebuilt inside a durable unit has only the fields its engine carried, and reading any
    # other raises. Work in a unit that didn't carry the tracer goes untraced rather than failing.
    if not _TRACE_FIELDS <= ctx.__dataclass_fields__.keys():
        return None
    # `ctx` is the only source: in a durable unit the engine already narrowed its flag to the worker's setting.
    return _run_telemetry(ctx.tracer, ctx.trace_include_content)


def _with_content(
    telemetry: RunTelemetry,
    attributes: Mapping[str, AttributeValue] | None,
    content: Mapping[str, AttributeValue] | None,
) -> Mapping[str, AttributeValue] | None:
    return {**(attributes or {}), **content} if content and telemetry.include_content else attributes
