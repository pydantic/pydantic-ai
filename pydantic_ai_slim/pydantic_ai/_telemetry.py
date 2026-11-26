"""Spans on a run's tracer that record content only when the run allows it.

- `open_span(ctx, name)`: a span around work that takes time and can fail.
- `record_decision(ctx, name)`: a decision that changed the run's course, recorded as a span that ends at once.
- `current_span(ctx)`: the span already open where the caller runs, such as a tool hook's `execute_tool`.
- `set_content_attribute(ctx, span, key, value)`: an attribute that can quote content, set only when
  the run captures content.

The first argument names the run: its `ctx`, whose `ctx.tracer` and `ctx.trace_include_content` are
used, or, for code that runs during a run but has no `RunContext` (such as a `Workspace` that runs
share and that can outlive a run), `current_run_telemetry()`, which the run publishes for the context
its body, tools and their tasks inherit. A value stored on such an object would apply one run's
settings to another's calls. `None` means no run: nothing is traced.

Pass `ctx` whenever there is one: the run publishes its telemetry only once its `agent run` span is
open, so a `for_run` hook, or a `wrap_run` ordered outside `Instrumentation`, that read it would find
nothing, even in a nested run.
"""

from __future__ import annotations as _annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

from opentelemetry.trace import INVALID_SPAN, NoOpTracer, Span, Tracer, get_current_span
from opentelemetry.util.types import AttributeValue

from ._durable_replay import start_as_current_span
from ._instrumentation import record_uncaught_errors

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
    """The calling run's telemetry, to pass to the helpers from code that has no `RunContext`.

    Get it from [`current_run_telemetry()`][pydantic_ai.telemetry.current_run_telemetry]; it has no public fields.
    """

    _active: _Active


class _Active(NamedTuple):
    """What the helpers use.

    Not `InstrumentationSettings`: in a durable unit both come from the rebuilt `ctx`, whose content flag
    the engine narrowed to the worker's setting.
    """

    tracer: Tracer
    include_content: bool


_current: ContextVar[RunTelemetry | None] = ContextVar('pydantic_ai.run_telemetry', default=None)


def current_run_telemetry() -> RunTelemetry | None:
    """The calling run's telemetry, or `None` outside an instrumented run."""
    return _current.get()


def publish_run_telemetry(ctx: RunContext[Any] | None) -> None:
    """Make `ctx`'s telemetry the current run's; `None`, or an uninstrumented run, means none.

    A plain set, for a context the run owns: the run's body may finish in a different `Context`
    than it started in, where resetting a token raises.
    """
    _current.set(_published(_telemetry_of(ctx)))


@contextmanager
def run_telemetry_scope(ctx: RunContext[Any] | RunTelemetry | None) -> Generator[None]:
    """Publish `ctx`'s telemetry for a block that starts and ends in one `Context`, for code that has no `RunContext`."""
    token = _current.set(_published(_telemetry_of(ctx)))
    try:
        yield
    finally:
        _current.reset(token)


def _published(active: _Active | None) -> RunTelemetry | None:
    return None if active is None else RunTelemetry(active)


@contextmanager
def open_span(
    ctx: RunContext[Any] | RunTelemetry | None,
    /,
    name: str,
    *,
    attributes: Mapping[str, AttributeValue] | None = None,
    content: Mapping[str, AttributeValue] | None = None,
) -> Generator[Span]:
    """Open `name` on the run's tracer.

    Yields `INVALID_SPAN` outside an instrumented run. `attributes` are always recorded; `content`,
    which can quote what the span is about, only when the run captures content. Both are set at the
    start, so Logfire's console and pending spans show them. An uncaught `Exception` marks the span
    ERROR and is recorded by its type only, unless the run captures content, as its message can
    quote content. A cancellation is not an error, as in OpenTelemetry's own spans: it leaves the
    status unset. In a durable run, a span that covered only replayed work is not exported again.
    """
    if (telemetry := _telemetry_of(ctx)) is None:
        yield INVALID_SPAN
        return
    with (
        start_as_current_span(
            telemetry.tracer,
            name,
            attributes=_with_content(telemetry, attributes, content),
            record_exception=False,
            set_status_on_exception=False,
        ) as span,
        record_uncaught_errors(span, include_content=telemetry.include_content),
    ):
        yield span


def record_decision(
    ctx: RunContext[Any] | RunTelemetry | None,
    /,
    name: str,
    *,
    attributes: Mapping[str, AttributeValue] | None = None,
    content: Mapping[str, AttributeValue] | None = None,
) -> None:
    """Record that a capability decided something that changed the run's course, such as a guard blocking a value.

    It is recorded as a span named `name`, on the run's tracer, that ends as soon as it starts. A later
    release may record it as a span event instead, under the same name and attributes. `attributes` are
    always recorded; `content`, which can quote what the decision was about, only when the run captures
    content. On a durable engine that reports when it is replaying, such as Temporal, a decision made
    again during replay is not exported again.
    """
    if (telemetry := _telemetry_of(ctx)) is not None:
        with start_as_current_span(telemetry.tracer, name, attributes=_with_content(telemetry, attributes, content)):
            pass


def set_content_attribute(
    ctx: RunContext[Any] | RunTelemetry | None, /, span: Span, key: str, value: AttributeValue
) -> None:
    """Set an attribute that can quote content, such as a path or a command, only when the run captures content.

    For content known when the span opens, pass `content=` to `open_span` instead.
    """
    if records_content(ctx):
        span.set_attribute(key, value)


def records_content(ctx: RunContext[Any] | RunTelemetry | None, /) -> bool:
    """Whether the run's spans record content, to skip building a content value that costs something or can raise."""
    return (telemetry := _telemetry_of(ctx)) is not None and telemetry.include_content


def current_span(ctx: RunContext[Any] | RunTelemetry | None, /) -> Span:
    """The span open where the caller runs, to add attributes to, or `INVALID_SPAN` when there is none of the run's.

    In a tool hook that is the call's `execute_tool` span, and in `before_model_request` the request's
    `chat` span. Unlike `opentelemetry.trace.get_current_span()`, it returns a span only inside the run's
    own `agent run` span, where the run publishes its telemetry, so a run doesn't write to a span it
    doesn't own: it returns `INVALID_SPAN` when the run isn't traced, in `for_run`, and in a `wrap_run`
    ordered outside `Instrumentation`. In a Temporal activity, which publishes the run's telemetry, it
    is the span open in the activity, such as the activity's own span, not the workflow's
    `execute_tool`. A sub-agent's `for_run` runs inside its parent's tool call, but with no telemetry
    published, so it doesn't get the parent's span either.
    """
    if (telemetry := _telemetry_of(ctx)) is None or _telemetry_of(_current.get()) != telemetry:
        return INVALID_SPAN
    return get_current_span()


_TRACE_FIELDS = frozenset({'tracer', 'trace_include_content'})


def _telemetry_of(ctx: RunContext[Any] | RunTelemetry | None) -> _Active | None:
    if ctx is None:
        return None
    if isinstance(ctx, RunTelemetry):
        # Built only from an instrumented run's telemetry, by `_published`.
        return ctx._active  # pyright: ignore[reportPrivateUsage]
    # A context rebuilt inside a durable unit has only the fields its engine carried, and reading any
    # other raises. Work in a unit that didn't carry the tracer goes untraced rather than failing.
    if not _TRACE_FIELDS <= ctx.__dataclass_fields__.keys():
        return None
    # `ctx` is the only source: in a durable unit the engine already narrowed its flag to the worker's setting.
    return None if isinstance(ctx.tracer, NoOpTracer) else _Active(ctx.tracer, ctx.trace_include_content)


def _with_content(
    telemetry: _Active,
    attributes: Mapping[str, AttributeValue] | None,
    content: Mapping[str, AttributeValue] | None,
) -> Mapping[str, AttributeValue] | None:
    return {**(attributes or {}), **content} if content and telemetry.include_content else attributes
