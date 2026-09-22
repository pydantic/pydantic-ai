from __future__ import annotations

import threading
import typing
import uuid
import weakref
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar

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
_context_in_memory_providers: dict[
    int, tuple[weakref.ref[TracerProvider] | TracerProvider, _ContextInMemorySpanExporter]
] = {}
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
                f' For more information, refer to the documentation at https://pydantic.dev/docs/ai/evals/evaluators/span-based/.'
            )
        else:
            # Custom TracerProvider (e.g. ddtrace) without add_span_processor - degrade gracefully.
            return SpanTreeRecordingError(
                f'The current TracerProvider ({type(provider).__qualname__}) does not support'
                f' `add_span_processor`, so span tree recording is not available.'
                f' Evaluation will still work, but `span_tree` will not be populated in evaluator results.'
            )

    cache_id = id(provider)

    # Attaching the processor is inside the lock, not just the cache write: two threads racing here would
    # otherwise each attach one, and only the winner's exporter would be reachable through the cache. The
    # loser's would stay attached to the provider, collecting spans under every context id that nothing ever
    # clears.
    # The consequence is that `add_span_processor` runs under a non-reentrant lock, so a provider that called
    # back into `context_subtree()` from it deadlocks, where the implementation this replaced returned
    # -- it published its entry before attaching, so the inner call found one. That is accepted rather than
    # fixed: swapping in an `RLock` alone would let the inner call attach a second processor -- the very leak
    # this lock exists to prevent -- and moving the attach outside the lock reopens the race. Publishing the
    # entry provisionally under an `RLock` does hold both, at the cost of a rollback path on a failed attach;
    # it was implemented and dropped, because no supported provider re-enters: `opentelemetry-sdk`'s
    # `TracerProvider.add_span_processor` and logfire's `ProxyTracerProvider.add_span_processor` both only
    # append under their own lock, without calling back into user code.
    with _context_in_memory_providers_lock:
        if (cached := _context_in_memory_providers.get(cache_id)) is not None:
            cached_provider, cached_exporter = cached
            if isinstance(cached_provider, weakref.ref):
                cached_provider = cached_provider()
            # A provider keeps its identity across `shutdown()`, which stops the exporter attached to it, and
            # a stopped exporter silently drops every span it is handed. The dead processor stays attached, so
            # a provider shut down repeatedly without being replaced accumulates one per shutdown -- bounded
            # in practice because `logfire.configure()` allocates a new provider each time.
            # Recovering by attaching to an already-shut-down provider relies on `opentelemetry-sdk` letting a
            # new processor receive spans after `shutdown()`, which two separate OTel `SHOULD`s say it need
            # not: `TracerProvider.Shutdown` says an SDK SHOULD hand out a no-op tracer afterwards, and
            # `SpanProcessor.Shutdown` says it SHOULD ignore later `OnEnd` calls. The second one starves the
            # newly attached processor whichever tracer produced the span -- `shutdown()` stops the composite
            # processor that `add_span_processor` then appends to -- so it exposes the logfire path just as
            # much as the plain-SDK one. Nothing here pins the SDK version.
            if cached_provider is provider and not cached_exporter._stopped:  # pyright: ignore[reportPrivateUsage]
                return cached_exporter

        exporter = _ContextInMemorySpanExporter()
        # The eviction callback releases the entry with its provider. CPython runs it during the provider's
        # deallocation, before that address can be reused, so it can never evict a newer entry keyed on a
        # recycled `id()`. It deliberately does not take the lock: it can fire on a thread already holding
        # this non-reentrant one, and `dict.pop` needs no lock of its own.
        try:
            stored: weakref.ref[TracerProvider] | TracerProvider = weakref.ref(
                provider, lambda _: _context_in_memory_providers.pop(cache_id, None)
            )
        except TypeError:
            # A provider that cannot be weakly referenced is pinned instead. That retains the provider and
            # everything it owns -- its span processors and their exporters -- for the life of the process, but
            # it is one entry per such provider, where leaving it uncached attaches a fresh span processor on
            # every call and leaves every orphaned exporter collecting spans that nothing ever clears. A pinned
            # provider can never be freed, so its `id()` can never be recycled either.
            stored = provider

        processor = SimpleSpanProcessor(exporter)
        # Cached only once the attach has succeeded, so a raising `add_span_processor` cannot leave an entry
        # claiming an attachment that never happened.
        provider.add_span_processor(processor)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
        _context_in_memory_providers[cache_id] = (stored, exporter)
        return exporter
