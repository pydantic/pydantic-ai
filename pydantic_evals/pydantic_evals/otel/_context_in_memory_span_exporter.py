from __future__ import annotations

import threading
import typing
import uuid
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from weakref import WeakValueDictionary

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import ProxyTracerProvider, get_tracer_provider

try:
    from logfire._internal.tracer import (
        ProxyTracerProvider as LogfireProxyTracerProvider,  # pyright: ignore[reportAssignmentType,reportPrivateImportUsage]
    )

    _LOGFIRE_IS_INSTALLED = True
except ImportError:  # pragma: no cover
    _LOGFIRE_IS_INSTALLED = False  # pyright: ignore[reportConstantRedefinition]

    # Ensure that we can do an isinstance check without erroring
    class LogfireProxyTracerProvider:
        @property
        def provider(self):
            return None


from ._errors import SpanTreeRecordingError
from .span_tree import SpanTree

_EXPORTER_CONTEXT_ID = ContextVar['str | None']('_EXPORTER_CONTEXT_ID', default=None)


# Note: It may be a good idea to upstream this whole file to `logfire`
@contextmanager
def context_subtree() -> typing.Iterator[SpanTree | SpanTreeRecordingError]:
    """Context manager that yields a `SpanTree` containing all spans collected during the context.

    The tree will be empty until the context is exited.

    If no TracerProvider has been configured, a `SpanTreeRecordingError` will be yielded instead of the SpanTree.
    """
    tree = SpanTree()
    with _context_subtree_spans() as spans:
        if isinstance(spans, SpanTreeRecordingError):
            yield spans
            return
        yield tree
    tree.add_readable_spans(spans)


@contextmanager
def _context_subtree_spans() -> typing.Iterator[list[ReadableSpan] | SpanTreeRecordingError]:
    """Context manager that yields a list of spans that are collected during the context.

    The list will be empty until the context is exited.
    """
    exporter = _add_context_span_exporter()

    if isinstance(exporter, SpanTreeRecordingError):
        yield exporter
        return

    spans: list[ReadableSpan] = []
    with _set_exporter_context_id() as context_id:
        yield spans
    result = exporter.get_finished_spans(context_id)
    exporter.clear(context_id)
    spans.extend(result)


@contextmanager
def _set_exporter_context_id(context_id: str | None = None) -> typing.Iterator[str]:
    context_id = context_id or str(uuid.uuid4())
    token = _EXPORTER_CONTEXT_ID.set(context_id)
    try:
        yield context_id
    finally:
        _EXPORTER_CONTEXT_ID.reset(token)


class _ContextInMemorySpanExporter(SpanExporter):
    def __init__(self) -> None:
        self._finished_spans: dict[str, list[ReadableSpan]] = defaultdict(list)
        self._stopped = False
        self._lock = threading.Lock()

    def clear(self, context_id: str | None = None) -> None:
        """Clear list of collected spans."""
        with self._lock:
            if context_id is None:  # pragma: no cover
                self._finished_spans.clear()
            else:
                self._finished_spans.pop(context_id, None)

    def get_finished_spans(self, context_id: str | None = None) -> tuple[ReadableSpan, ...]:
        """Get list of collected spans."""
        with self._lock:
            if context_id is None:  # pragma: no cover
                all_finished_spans: list[ReadableSpan] = []
                for finished_spans in self._finished_spans.values():
                    all_finished_spans.extend(finished_spans)
                return tuple(all_finished_spans)
            else:
                return tuple(self._finished_spans.get(context_id, []))

    def export(self, spans: typing.Sequence[ReadableSpan]) -> SpanExportResult:
        """Stores a list of spans in memory."""
        if self._stopped:  # pragma: no cover
            return SpanExportResult.FAILURE
        with self._lock:
            context_id = _EXPORTER_CONTEXT_ID.get()
            if context_id is not None:
                self._finished_spans[context_id].extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """Shut downs the exporter.

        Calls to export after the exporter has been shut down will fail.
        """
        self._stopped = True

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # pragma: no cover
        return True


# This cache is mostly just necessary for testing
# When running in "real" code, the tracer provider won't be reset
_context_in_memory_providers: WeakValueDictionary[int, _ContextInMemorySpanExporter] = WeakValueDictionary()


def _add_context_span_exporter() -> _ContextInMemorySpanExporter | SpanTreeRecordingError:
    tracer_provider = get_tracer_provider()
    if isinstance(tracer_provider, LogfireProxyTracerProvider):
        cache_id = id(tracer_provider.provider)
    else:
        cache_id = id(tracer_provider)
    if (cached_exporter := _context_in_memory_providers.get(cache_id)) is not None:
        return cached_exporter

    # `tracer_provider` should generally be an `opentelemetry.sdk.trace.TracerProvider` or
    # `logfire._internal.tracer.ProxyTracerProvider`, in which case the `add_span_processor` method will be present
    if not hasattr(tracer_provider, 'add_span_processor'):
        if isinstance(tracer_provider, ProxyTracerProvider):
            required_call = (
                'logfire.configure(...)' if _LOGFIRE_IS_INSTALLED else 'opentelemetry.trace.set_tracer_provider(...)'
            )
            return SpanTreeRecordingError(
                f'To make use of the `span_tree` in an evaluator, you need to call `{required_call}` before running an'
                f' evaluation.'
                f' For more information, refer to the documentation at https://ai.pydantic.dev/evals/#opentelemetry-integration.'
            )
        else:
            # In this case, the user is using some kind of custom OpenTelemetry TracerProvider. It may or may not
            # be easy to handle this better, but given it's hard to know, we just raise an error telling the user
            # to notify us if they want help.
            raise TypeError(
                'Expected `tracer_provider` to have an `add_span_processor` method;'
                f' got an instance of {type(tracer_provider)}.'
                f' For help resolving this, please create an issue at https://github.com/pydantic/pydantic-ai/issues.'
            )

    exporter = _ContextInMemorySpanExporter()
    _context_in_memory_providers[cache_id] = exporter
    processor = SimpleSpanProcessor(exporter)
    tracer_provider.add_span_processor(processor)  # type: ignore
    return exporter
