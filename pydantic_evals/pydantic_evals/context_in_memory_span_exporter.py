import threading
import typing
import uuid
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import ProxyTracerProvider, get_tracer_provider

_EXPORTER_CONTEXT_ID = ContextVar[str | None]('_EXPORTER_CONTEXT_ID', default=None)


@contextmanager
def context_subtree_spans() -> typing.Iterator[list[ReadableSpan]]:
    """Context manager that yields a list of spans that are collected during the context.

    The list will be empty until the context is exited.
    """
    exporter = _add_context_span_exporter()
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


@cache
def _add_context_span_exporter():
    tracer_provider = get_tracer_provider()
    # `tracer_provider` should generally be an `opentelemetry.sdk.trace.TracerProvider` or
    # `logfire._internal.tracer.ProxyTracerProvider`, in which case the `add_span_processor` method will be present
    if not hasattr(tracer_provider, 'add_span_processor'):
        if isinstance(tracer_provider, ProxyTracerProvider):
            # TODO: Should not require logfire or opentelemetry as  dependencies
            raise TypeError(
                'A tracer provider has not been set. You need to call `logfire.configure(...)` or `opentelemetry.trace.set_tracer_provider(...)` before reaching this point'
            )
        else:
            raise TypeError(
                'Expected `tracer_provider` to have an `add_span_processor` method; '
                f'got an instance of {type(tracer_provider)}.'
            )

    exporter = _ContextInMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    tracer_provider.add_span_processor(processor)  # type: ignore
    return exporter


class _ContextInMemorySpanExporter(SpanExporter):
    def __init__(self) -> None:
        self._finished_spans: dict[str, list[ReadableSpan]] = defaultdict(list)
        self._stopped = False
        self._lock = threading.Lock()

    def clear(self, context_id: str | None = None) -> None:
        """Clear list of collected spans."""
        with self._lock:
            if context_id is None:
                self._finished_spans.clear()
            else:
                self._finished_spans.pop(context_id, None)

    def get_finished_spans(self, context_id: str | None = None) -> tuple[ReadableSpan, ...]:
        """Get list of collected spans."""
        with self._lock:
            if context_id is None:
                all_finished_spans: list[ReadableSpan] = []
                for finished_spans in self._finished_spans.values():
                    all_finished_spans.extend(finished_spans)
                return tuple(all_finished_spans)
            else:
                return tuple(self._finished_spans.get(context_id, []))

    def export(self, spans: typing.Sequence[ReadableSpan]) -> SpanExportResult:
        """Stores a list of spans in memory."""
        if self._stopped:
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

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
