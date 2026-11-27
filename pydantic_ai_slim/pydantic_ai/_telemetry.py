"""Spans on a run's tracer that record content only when the run allows it.

Public through `pydantic_ai.capabilities`:

- `open_span(ctx, name)`: a span around work that takes time and can fail.
- `record_decision(ctx, name)`: a decision that changed the run's course, recorded as a span that ends at once.
- `tool_call_span(ctx)`: the `execute_tool` span of the tool call `ctx` belongs to, to add attributes to.

Core code that runs during a run but has no `RunContext`, such as a `Workspace` that runs share and that
can outlive a run, uses `open_ambient_span`. It reads the tracer and content flag the run publishes once
its `agent run` span is open, in the context its body, tools and their tasks inherit. A value stored on
such an object would apply one run's settings to another's calls.
"""

from __future__ import annotations as _annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, NamedTuple

from opentelemetry.trace import INVALID_SPAN, NoOpTracer, Span, Tracer, get_current_span
from opentelemetry.util.types import AttributeValue

from ._durable_replay import start_as_current_span
from ._instrumentation import exception_type_name, record_uncaught_errors

if TYPE_CHECKING:
    from ._run_context import RunContext

__all__ = (
    'open_ambient_span',
    'open_span',
    'publish_run_telemetry',
    'record_decision',
    'run_telemetry_scope',
    'tool_call_span',
)


class _TracedRun(NamedTuple):
    """The tracer and content flag of a run that is traced.

    Not `InstrumentationSettings`: in a durable unit both come from the rebuilt `ctx`, whose content flag
    the engine narrowed to the worker's setting.
    """

    tracer: Tracer
    include_content: bool


_published_traced_run: ContextVar[_TracedRun | None] = ContextVar('pydantic_ai.published_traced_run', default=None)
"""The traced run that code without a `RunContext` traces to, or `None` when there is none."""


def publish_run_telemetry(ctx: RunContext[Any] | None) -> None:
    """Make `ctx`'s run the one code without a `RunContext` traces to; `None`, or an untraced run, means none.

    A plain set, for a context the run owns: the run's body may finish in a different `Context`
    than it started in, where resetting a token raises.
    """
    _published_traced_run.set(_traced_run_of(ctx))


@contextmanager
def run_telemetry_scope(ctx: RunContext[Any] | None) -> Generator[None]:
    """Publish `ctx`'s run for a block that starts and ends in one `Context`."""
    token = _published_traced_run.set(_traced_run_of(ctx))
    try:
        yield
    finally:
        _published_traced_run.reset(token)


@contextmanager
def open_span(
    ctx: RunContext[Any],
    /,
    name: str,
    *,
    attributes: Mapping[str, AttributeValue] | None = None,
    content: Mapping[str, AttributeValue] | None = None,
) -> Generator[Span]:
    """Open `name` on the run's tracer.

    Yields `INVALID_SPAN` when the run isn't traced. `attributes` are always recorded; `content`, which
    can quote what the span is about, only when the run captures content. Both are set at the start, so
    Logfire's console and pending spans show them. An uncaught `Exception` marks the span ERROR, sets
    `error.type` to the exception's type, and is recorded by its type only, unless the run captures
    content, as its message can quote content. A cancellation is not an error, as in OpenTelemetry's own
    spans: it leaves the status unset. In a durable run, a span that covered only replayed work is not
    exported again.
    """
    with _open_span(_traced_run_of(ctx), name, attributes=attributes, content=content) as span:
        yield span


@contextmanager
def open_ambient_span(
    name: str,
    *,
    attributes: Mapping[str, AttributeValue] | None = None,
    content: Mapping[str, AttributeValue] | None = None,
) -> Generator[Span]:
    """`open_span` for core code with no `RunContext`, on the run published for the calling context.

    Yields `INVALID_SPAN` outside a traced run, and in a run's setup, before its `agent run` span opens.
    """
    with _open_span(_published_traced_run.get(), name, attributes=attributes, content=content) as span:
        yield span


@contextmanager
def _open_span(
    traced_run: _TracedRun | None,
    name: str,
    *,
    attributes: Mapping[str, AttributeValue] | None,
    content: Mapping[str, AttributeValue] | None,
) -> Generator[Span]:
    if traced_run is None:
        yield INVALID_SPAN
        return
    with (
        start_as_current_span(
            traced_run.tracer,
            name,
            attributes=_with_content(traced_run, attributes, content),
            record_exception=False,
            set_status_on_exception=False,
        ) as span,
        record_uncaught_errors(span, include_content=traced_run.include_content),
    ):
        try:
            yield span
        except Exception as error:
            # The type never quotes content, so it is set whether or not the run captures content.
            span.set_attribute('error.type', exception_type_name(error))
            raise


def record_decision(
    ctx: RunContext[Any],
    /,
    name: str,
    *,
    attributes: Mapping[str, AttributeValue] | None = None,
    content: Mapping[str, AttributeValue] | None = None,
) -> None:
    """Record that a capability decided something that changed the run's course, such as a guard blocking a value.

    It is recorded as a span named `name`, on the run's tracer, that ends as soon as it starts. `attributes`
    are always recorded; `content`, which can quote what the decision was about, only when the run captures
    content. On a durable engine that reports when it is replaying, such as Temporal, a decision made
    again during replay is not exported again.
    """
    if (traced_run := _traced_run_of(ctx)) is not None:
        with start_as_current_span(traced_run.tracer, name, attributes=_with_content(traced_run, attributes, content)):
            pass


def tool_call_span(ctx: RunContext[Any], /) -> Span:
    """The `execute_tool` span of the tool call `ctx` belongs to, to add attributes to, or `INVALID_SPAN`.

    It is the span only inside a tool call of a traced run: in the tool hooks, the tool's body and a
    toolset's `call_tool`. Everywhere else, such as `for_run`, `before_run` and `wrap_run`, it is
    `INVALID_SPAN`, so a capability never writes to a span the run doesn't own, such as the
    application's span around the run or a parent run's tool call. Take it before or after nested work,
    not while a child span may be current. An attribute that can quote content is set only
    `if ctx.trace_include_content`.
    """
    if _traced_run_of(ctx) is None or ctx.tool_call_id is None:
        return INVALID_SPAN
    return get_current_span()


_TRACE_FIELDS = frozenset({'tracer', 'trace_include_content'})


def _traced_run_of(ctx: RunContext[Any] | None) -> _TracedRun | None:
    if ctx is None:
        return None
    # A context rebuilt inside a durable unit has only the fields its engine carried, and reading any
    # other raises. Work in a unit that didn't carry the tracer goes untraced rather than failing.
    if not _TRACE_FIELDS <= ctx.__dataclass_fields__.keys():
        return None
    # `ctx` is the only source: in a durable unit the engine already narrowed its flag to the worker's setting.
    return None if isinstance(ctx.tracer, NoOpTracer) else _TracedRun(ctx.tracer, ctx.trace_include_content)


def _with_content(
    traced_run: _TracedRun,
    attributes: Mapping[str, AttributeValue] | None,
    content: Mapping[str, AttributeValue] | None,
) -> Mapping[str, AttributeValue] | None:
    return {**(attributes or {}), **content} if content and traced_run.include_content else attributes
