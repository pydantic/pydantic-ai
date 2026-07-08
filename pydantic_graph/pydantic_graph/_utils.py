from __future__ import annotations as _annotations

import asyncio
import inspect
import types
import warnings
from collections.abc import Awaitable, Generator
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar, get_args, get_origin

from logfire_api import Logfire, LogfireSpan
from typing_inspection import typing_objects
from typing_inspection.introspection import is_union_origin

if TYPE_CHECKING:
    from opentelemetry.trace import Span

_logfire = Logfire(otel_scope='pydantic-graph')

AbstractSpan: TypeAlias = 'LogfireSpan | Span'

try:
    from opentelemetry.trace import Span, set_span_in_context
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    TRACEPARENT_PROPAGATOR = TraceContextTextMapPropagator()
    TRACEPARENT_NAME = 'traceparent'
    assert TRACEPARENT_NAME in TRACEPARENT_PROPAGATOR.fields

    # Logic taken from logfire.experimental.annotations
    def get_traceparent(span: AbstractSpan) -> str | None:
        """Get a string representing the span context to use for annotating spans."""
        real_span: Span
        if isinstance(span, Span):
            real_span = span  # pragma: lax no cover
        else:
            real_span = span._span
            assert real_span
        context = set_span_in_context(real_span)
        carrier: dict[str, Any] = {}
        TRACEPARENT_PROPAGATOR.inject(carrier, context)
        return carrier.get(TRACEPARENT_NAME, '')

except ImportError:  # pragma: no cover

    def get_traceparent(span: AbstractSpan) -> str | None:
        # Opentelemetry wasn't installed, so we can't get the traceparent
        return None


def get_event_loop() -> asyncio.AbstractEventLoop:
    try:
        event_loop = asyncio.get_event_loop()
    except RuntimeError:
        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)
    return event_loop


_T = TypeVar('_T')


def run_until_complete(coro: Awaitable[_T]) -> _T:
    """Run `coro` to completion on the event loop, cleaning up after itself if interrupted.

    If the caller interrupts `loop.run_until_complete()` (e.g. by pressing Ctrl-C, raising
    `KeyboardInterrupt`) while `coro` is suspended, asyncio leaves its task pending with its
    `async with`/`finally` blocks un-run, leaking the task and any open connections. We cancel
    *our own* task and drive its cleanup to completion before re-raising, without touching any
    other tasks on the (caller-owned) loop.
    """
    loop = get_event_loop()
    task = asyncio.ensure_future(coro, loop=loop)
    try:
        return loop.run_until_complete(task)
    except BaseException:
        if not task.done():
            task.cancel()
            with suppress(BaseException):
                loop.run_until_complete(task)
        raise


def get_union_args(tp: Any) -> tuple[Any, ...]:
    """Extract the arguments of a Union type if `response_type` is a union, otherwise return an empty tuple."""
    # similar to `pydantic_ai_slim/pydantic_ai/_result.py:get_union_args`
    if typing_objects.is_typealiastype(tp):
        tp = tp.__value__  # pragma: no cover

    origin = get_origin(tp)
    if is_union_origin(origin):
        return get_args(tp)
    else:
        return (tp,)


def unpack_annotated(tp: Any) -> tuple[Any, list[Any]]:
    """Strip `Annotated` from the type if present.

    Returns:
        `(tp argument, ())` if not annotated, otherwise `(stripped type, annotations)`.
    """
    origin = get_origin(tp)
    if typing_objects.is_annotated(origin):
        inner_tp, *args = get_args(tp)
        return inner_tp, args
    else:
        return tp, []


def get_parent_namespace(frame: types.FrameType | None) -> dict[str, Any] | None:
    """Attempt to get the namespace where the graph was defined.

    If the graph is defined with generics `Graph[a, b]` then another frame is inserted, and we have to skip that
    to get the correct namespace.

    Args:
        frame: The frame to start searching from, or `None`.

    Returns:
        The local namespace dict of the defining frame, or `None` if the frame was `None`.
    """
    if frame is not None:  # pragma: no branch
        if back := frame.f_back:  # pragma: no branch
            if back.f_globals.get('__name__') == 'typing':  # pragma: no cover
                # If the class calling this function is generic, explicitly parameterizing the class
                # results in a `typing._GenericAlias` instance, which proxies instantiation calls to the
                # "real" class and thus adding an extra frame to the call. To avoid pulling anything
                # from the `typing` module, use the correct frame (the one before):
                return get_parent_namespace(back)
            else:
                return back.f_locals


class Unset:
    """A singleton to represent an unset value.

    Copied from pydantic_ai/_utils.py.
    """

    pass


UNSET = Unset()


try:
    from logfire._internal.config import (
        LogfireNotConfiguredWarning,  # pyright: ignore[reportAssignmentType]
    )
except ImportError:  # pragma: lax no cover

    class LogfireNotConfiguredWarning(UserWarning):
        pass


if TYPE_CHECKING:
    logfire_span = _logfire.span
else:

    @contextmanager
    def logfire_span(*args: Any, **kwargs: Any) -> Generator[LogfireSpan, None, None]:
        """Create a Logfire span without warning if logfire is not configured."""
        # TODO: Remove once Logfire has the ability to suppress this warning from non-user code
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=LogfireNotConfiguredWarning)
            with _logfire.span(*args, **kwargs) as span:
                yield span


def infer_obj_name(obj: Any, *, depth: int) -> str | None:
    """Infer the variable name of an object from the calling frame's scope.

    This function examines the call stack to find what variable name was used
    for the given object in the calling scope. This is useful for automatic
    naming of objects based on their variable names.

    Args:
        obj: The object whose variable name to infer.
        depth: Number of stack frames to traverse upward from the current frame.

    Returns:
        The inferred variable name if found, None otherwise.

    Example:
        Usage should generally look like `infer_name(self, depth=2)` or similar.
    """
    target_frame = inspect.currentframe()
    if target_frame is None:
        return None  # pragma: no cover
    for _ in range(depth):
        target_frame = target_frame.f_back
        if target_frame is None:
            return None

    for name, item in target_frame.f_locals.items():
        if item is obj:
            return name

    if target_frame.f_locals != target_frame.f_globals:  # pragma: no branch
        # if we couldn't find the agent in locals and globals are a different dict, try globals
        for name, item in target_frame.f_globals.items():
            if item is obj:
                return name

    return None
