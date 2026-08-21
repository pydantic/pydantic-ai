from __future__ import annotations

import threading
import typing
import uuid
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from weakref import ref

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import ProxyTracerProvider, TracerProvider, get_tracer_provider

try:
    from logfire._internal.tracer import (
        ProxyTracerProvider as LogfireProxyTracerProvider,  # pyright: ignore[reportAssignmentType]
    )

    _LOGFIRE_IS_INSTALLED = True
except ImportError:  # pragma: lax no cover
    _LOGFIRE_IS_INSTALLED = False  # pyright: ignore[reportConstantRedefinition]

    # Ensure that we can do an isinstance check without erroring
    class LogfireProxyTracerProvider:
        provider: TracerProvider


from ._errors import SpanTreeRecordingError
from .span_tree import SpanTree

_EXPORTER_CONTEXT_ID = ContextVar['str | None']('_EXPORTER_CONTEXT_ID', default=None)


# Note: It may be a good idea to upstream this whole file to `logfire`
@contextmanager
def context_subtree() -> typing.Generator[SpanTree | SpanTreeRecordingError]:
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
def _context_subtree_spans() -> typing.Generator[list[ReadableSpan] | SpanTreeRecordingError]:
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
def _set_exporter_context_id(context_id: str | None = None) -> typing.Generator[str]:
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

    def force_flush(self, timeout_millis: int = 30000) -> bool:  # pragma: no cover
        return True


# Caching the exporter per provider keeps `context_subtree()` from attaching another span processor on every
# call: a long-lived provider would otherwise accumulate one per evaluation. Each entry pairs the exporter with the
# provider it is attached to, keyed by `id()` and matched by identity. Identity is what matters here, not equality:
# a provider need not be hashable, and two distinct providers that compare equal each need their own exporter, since
# an exporter only ever receives spans from the provider it was attached to. The provider is held weakly wherever it
# can be, which is what makes an `id()` key safe -- `id()`s are recycled, so an entry outliving its provider must be
# rejected rather than handed to whatever was later allocated at the same address.
_context_in_memory_providers: dict[int, tuple[ref[TracerProvider] | TracerProvider, _ContextInMemorySpanExporter]] = {}
_context_in_memory_providers_lock = threading.Lock()


def _add_context_span_exporter() -> _ContextInMemorySpanExporter | SpanTreeRecordingError:
    tracer_provider = get_tracer_provider()

    # `logfire.configure()` reuses one `ProxyTracerProvider` and swaps the provider it wraps, so the wrapped provider
    # -- not the proxy -- is what owns the span processors. Resolve it once and then both key on it and attach to it:
    # going through the proxy would re-resolve `.provider` under logfire's own lock at attach time, and a concurrent
    # `logfire.configure()` in that window would leave the entry keyed on a provider we never attached to.
    if isinstance(tracer_provider, LogfireProxyTracerProvider):
        provider = tracer_provider.provider
    else:
        provider = tracer_provider

    # `provider` should generally be an `opentelemetry.sdk.trace.TracerProvider`, in which case the
    # `add_span_processor` method will be present.
    # Checked before the cache lookup so a provider we are going to reject never becomes a cache key.
    if not hasattr(provider, 'add_span_processor'):
        if isinstance(tracer_provider, ProxyTracerProvider):
            required_call = (
                'logfire.configure(...)' if _LOGFIRE_IS_INSTALLED else 'opentelemetry.trace.set_tracer_provider(...)'
            )
            return SpanTreeRecordingError(
                f'To make use of the `span_tree` in an evaluator, you need to call `{required_call}` before running an'
                f' evaluation.'
                f' For more information, refer to the documentation at https://ai.pydantic.dev/evals/evaluators/span-based.'
            )
        else:
            # Custom TracerProvider (e.g. ddtrace) without add_span_processor - degrade gracefully.
            return SpanTreeRecordingError(
                f'The current TracerProvider ({type(tracer_provider).__qualname__}) does not support'
                f' `add_span_processor`, so span tree recording is not available.'
                f' Evaluation will still work, but `span_tree` will not be populated in evaluator results.'
            )

    cache_id = id(provider)

    # Attaching the processor is inside the lock, not just the cache write: two threads racing here would
    # otherwise each attach one, and only the winner's exporter would be reachable through the cache. The
    # loser's would stay attached to the provider, collecting spans under every context id that nothing ever
    # clears. Locked once per `context_subtree()` rather than per span, so a plain uncontended acquire is
    # cheap enough not to need double-checked locking.
    with _context_in_memory_providers_lock:
        if (cached := _context_in_memory_providers.get(cache_id)) is not None:
            cached_provider, cached_exporter = cached
            if isinstance(cached_provider, ref):
                cached_provider = cached_provider()
            # A provider keeps its identity across `shutdown()`, which stops the exporter attached to it, and
            # a stopped exporter silently drops every span it is handed. The dead processor stays attached, so
            # a provider shut down repeatedly without being replaced accumulates one per shutdown -- bounded
            # in practice because `logfire.configure()` allocates a new provider each time.
            # Recovering by attaching to an already-shut-down provider relies on `opentelemetry-sdk` letting a
            # new processor receive spans after `shutdown()`; the OTel specification says an SDK SHOULD hand
            # out a no-op tracer instead, and nothing here pins the SDK version.
            if cached_provider is provider and not cached_exporter._stopped:  # pyright: ignore[reportPrivateUsage]
                return cached_exporter

        exporter = _ContextInMemorySpanExporter()
        try:
            # The eviction callback releases the entry with its provider. CPython runs it during the provider's
            # deallocation, before that address can be reused, so it can never evict a newer entry keyed on a
            # recycled `id()`. It deliberately does not take the lock: it can fire on a thread already holding
            # this non-reentrant one, and `dict.pop` needs no lock of its own.
            stored: ref[TracerProvider] | TracerProvider = ref(
                provider, lambda _: _context_in_memory_providers.pop(cache_id, None)
            )
        except TypeError:
            # A provider that cannot be weakly referenced is pinned instead. Pinning one provider is bounded,
            # where leaving it uncached would attach a fresh span processor on every call and leave every
            # orphaned exporter collecting spans that nothing ever clears. A pinned provider can never be
            # freed, so its `id()` can never be recycled either.
            stored = provider

        processor = SimpleSpanProcessor(exporter)
        # Cached only once the attach has succeeded: an entry written first would claim an attachment that
        # never happened, and every later call would return an exporter no provider feeds -- a silently empty
        # `SpanTree`, which is the failure this cache exists to avoid.
        provider.add_span_processor(processor)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
        _context_in_memory_providers[cache_id] = (stored, exporter)
        return exporter
