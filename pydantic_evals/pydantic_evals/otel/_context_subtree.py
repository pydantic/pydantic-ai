from __future__ import annotations as _annotations

import typing
from contextlib import contextmanager

from ._errors import SpanTreeRecordingError
from .span_tree import SpanTree


# A plain function assigned to `context_subtree` below, rather than a `def context_subtree` inside the
# `except` branch, so that static analysis (griffe, which builds the API docs) doesn't see the same
# name as both an import alias and a function definition — it can't resolve that combination.
@contextmanager
def _context_subtree_fallback() -> typing.Generator[SpanTree | SpanTreeRecordingError]:
    """See the docstring for `pydantic_evals.otel._context_in_memory_span_exporter.context_subtree` for more detail.

    This is the fallback implementation that is used if you don't have opentelemetry installed.
    """
    exc = SpanTreeRecordingError(
        'To make use of the `span_tree` in an evaluator, you must install `logfire` or'
        ' `opentelemetry-sdk`,  configure appropriate instrumentations, and set a tracer provider (e.g. by calling'
        ' `logfire.configure()`).'
        ' For more information, refer to the documentation at https://pydantic.dev/docs/ai/evals/evaluators/span-based/.'
    )
    exc.__context__ = _IMPORT_ERROR
    yield exc


try:
    from ._context_in_memory_span_exporter import context_subtree
except ImportError as e:  # pragma: lax no cover
    _IMPORT_ERROR = e
    context_subtree = _context_subtree_fallback


__all__ = ('context_subtree',)
