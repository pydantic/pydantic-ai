"""Spans on a run's tracer that record content only when the run allows it.

Public through `pydantic_ai.capabilities`:

- `open_span(ctx, name)`: a span around work that takes time and can fail.
- `record_decision(ctx, name)`: a decision that changed the run's course, recorded as a span that ends at once.
- `tool_call_span(ctx)`: the `execute_tool` span of the tool call `ctx` belongs to, to add attributes to.

Core code that runs during a run but has no `RunContext`, such as a `Workspace` that runs share and that
can outlive a run, uses `open_ambient_span`. It reads the tracer and content flag the run publishes while
its run span (or a realtime session's span) is open, in the context its body, tools and their tasks
inherit. A value stored on such an object would apply one run's settings to another's calls.
"""

from __future__ import annotations as _annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, NamedTuple

from opentelemetry.trace import INVALID_SPAN, NoOpTracer, Span, Tracer
from opentelemetry.util.types import AttributeValue

from ._durable_replay import start_as_current_span
from ._instrumentation import exception_type_name, record_uncaught_errors

if TYPE_CHECKING:
    from ._run_context import RunContext

__all__ = (
    'ambient_traced_run',
    'bind_tool_call_span',
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


class _PublishedRun(NamedTuple):
    """A traced run published for code without a `RunContext`, valid only while `owner_span` records.

    `owner_span` is the run's span, or a realtime session's span. A task that copied the context keeps
    the value after the span ends; checking the span keeps that task from tracing with a finished run.
    """

    traced_run: _TracedRun
    owner_span: Span


_published_run: ContextVar[_PublishedRun | None] = ContextVar('pydantic_ai.published_run', default=None)


class _BoundToolCall(NamedTuple):
    """The `execute_tool` span `Instrumentation` opened for one tool call of one run."""

    span: Span
    run_id: str | None


_bound_tool_call: ContextVar[_BoundToolCall | None] = ContextVar('pydantic_ai.bound_tool_call', default=None)


def publish_run_telemetry(tracer: Tracer | None, include_content: bool = False, owner_span: Span | None = None) -> None:
    """Publish a run for code without a `RunContext`, while `owner_span` records; `None` publishes none.

    A plain set, for a context the caller owns, such as a task it created.
    """
    published = (
        None if tracer is None or owner_span is None else _PublishedRun(_TracedRun(tracer, include_content), owner_span)
    )
    _published_run.set(published)


@contextmanager
def run_telemetry_scope(ctx: RunContext[Any] | None, owner_span: Span | None = None) -> Generator[None]:
    """Publish `ctx`'s run, while `owner_span` records, for a block that starts and ends in one `Context`.

    `None` for either publishes none, which hides a parent run's value from a nested run's setup.
    """
    traced_run = _traced_run_of(ctx)
    published = None if traced_run is None or owner_span is None else _PublishedRun(traced_run, owner_span)
    token = _published_run.set(published)
    try:
        yield
    finally:
        _published_run.reset(token)


@contextmanager
def bind_tool_call_span(span: Span, *, run_id: str | None) -> Generator[None]:
    """Make `span` what `tool_call_span` returns for this run's tool call, for the span's lifetime."""
    token = _bound_tool_call.set(_BoundToolCall(span, run_id))
    try:
        yield
    finally:
        _bound_tool_call.reset(token)


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

    Meant for tool and model request hooks and `before_run`. `for_run` runs before the run's tracing
    settings are final, so a span from there can use a tracer or content setting the run then drops.

    Yields `INVALID_SPAN` when the run isn't traced. `attributes` are always recorded; `content`, which
    can quote what the span is about, only when the run captures content. Both are set at the start, so
    Logfire's console and pending spans show them. An uncaught `Exception` marks the span ERROR, sets
    `error.type` to the exception's type, and is recorded by its type only, unless the run captures
    content, as its message can quote content. A cancellation is not an error, as in OpenTelemetry's own
    spans: it leaves the status unset. In a Temporal workflow, from `wrap_run` on, a span that covered only
    replayed work is not exported again.
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

    Yields `INVALID_SPAN` outside a traced run's span: in its setup, before the span opens, and after it ends.
    """
    with _open_span(ambient_traced_run(), name, attributes=attributes, content=content) as span:
        yield span


def ambient_traced_run() -> _TracedRun | None:
    """The run published for the calling context, or `None` outside a traced run's span.

    Each publish makes a new value, so comparing by identity tells one run from a run started inside it.
    """
    published = _published_run.get()
    return published.traced_run if published is not None and published.owner_span.is_recording() else None


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
    """Record that a capability decided something that changed the run's course, such as a guard blocking it.

    Meant for tool and model request hooks and `before_run`. `for_run` runs before the run's tracing
    settings are final, so a decision from there can use a tracer or content setting the run then drops.

    It is recorded as a span named `name`, on the run's tracer, that ends as soon as it starts. `attributes`
    are always recorded; `content`, which can quote what the decision was about, only when the run captures
    content. In a Temporal workflow, from `wrap_run` on, a decision made again during replay is not exported
    again.
    """
    if (traced_run := _traced_run_of(ctx)) is not None:
        with start_as_current_span(traced_run.tracer, name, attributes=_with_content(traced_run, attributes, content)):
            pass


def tool_call_span(ctx: RunContext[Any], /) -> Span:
    """The `execute_tool` span of the tool call `ctx` belongs to, to add attributes to, or `INVALID_SPAN`.

    It is the span only while `Instrumentation` has it open for this call: in `before_tool_execute`,
    `after_tool_execute`, a `wrap_tool_execute` ordered inside `Instrumentation`, the tool's body and a
    toolset's `call_tool`. Everywhere else, such as `for_run`, `before_run`, `wrap_run`,
    `before_tool_validate` or a task that outlives the call, it is `INVALID_SPAN`, so a capability never
    writes to a span the call doesn't own. An attribute that can
    quote content is set only `if ctx.trace_include_content`.
    """
    bound = _bound_tool_call.get()
    if bound is None or not bound.span.is_recording() or bound.run_id != ctx.run_id:
        return INVALID_SPAN
    return bound.span


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
