from __future__ import annotations as _annotations

import typing
from contextlib import contextmanager

from ._errors import SpanTreeRecordingError
from .span_tree import SpanTree

try:
    from ._context_in_memory_span_exporter import context_subtree
except ImportError as e:  # pragma: lax no cover
    _IMPORT_ERROR = e

    @contextmanager
    def context_subtree() -> typing.Iterator[SpanTree | SpanTreeRecordingError]:
        """See the docstring for `pydantic_evals.otel._context_in_memory_span_exporter.context_subtree` for more detail.

        This is the fallback implementation that is used if you don't have opentelemetry installed.
        """
        exc = SpanTreeRecordingError(
            'To make use of the `span_tree` in an evaluator, you must install `logfire` or'
            ' `opentelemetry-sdk`,  configure appropriate instrumentations, and set a tracer provider (e.g. by calling'
            ' `logfire.configure()`).'
            ' For more information, refer to the documentation at https://ai.pydantic.dev/evals/#opentelemetry-integration.'
        )
        exc.__context__ = _IMPORT_ERROR
        yield exc


__all__ = ('context_subtree',)
