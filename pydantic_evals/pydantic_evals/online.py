"""Online evaluation — attach evaluators to live functions for automatic background evaluation.

This module provides the infrastructure for running evaluators on production (or staging) traffic.
The same `Evaluator` instances used with `Dataset.evaluate()` work here, the difference is in how
they are wired up (decorator vs dataset) rather than what they are.

Example:
```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.online import evaluate


@dataclass
class IsNonEmpty(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return bool(ctx.output)


@evaluate(IsNonEmpty())
async def my_function(x: int) -> int:
    return x
```
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import random
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import anyio
import sniffio
from typing_extensions import ParamSpec, TypeVar

from ._utils import UNSET, Unset, logfire_span
from .evaluators._run_evaluator import run_evaluator
from .evaluators.context import EvaluatorContext
from .evaluators.evaluator import EvaluationResult, Evaluator, EvaluatorFailure
from .otel._context_subtree import context_subtree
from .otel.span_tree import SpanTree

__all__ = (
    'CallbackSink',
    'DEFAULT_CONFIG',
    'EvaluationSink',
    'EvaluatorContextData',
    'EvaluatorContextSource',
    'OnlineEvalConfig',
    'OnlineEvaluator',
    'SinkCallback',
    'SpanReference',
    'configure',
    'disable_evaluation',
    'evaluate',
    'rebuild_context',
    'rebuild_contexts',
    'run_evaluators',
    'wait_for_evaluations',
)

logger = logging.getLogger('pydantic_evals.online')

_P = ParamSpec('_P')
_R = TypeVar('_R')

# Strong references to background tasks/threads to prevent garbage collection
# and enable deterministic waiting via wait_for_evaluations().
# Protected by _background_lock for thread-safety (free-threaded Python, multiple sync callers).
_background_lock = threading.Lock()
_background_tasks: set[asyncio.Task[Any]] = set()
_background_threads: set[threading.Thread] = set()


def _remove_background_task(task: asyncio.Task[Any]) -> None:
    """Callback to remove a completed task from the tracking set (thread-safe)."""
    with _background_lock:
        _background_tasks.discard(task)


async def _spawn_background_task(coro: Any) -> None:
    """Spawn a fire-and-forget background task using the current async backend.

    Uses sniffio to detect the backend (asyncio or trio) and dispatches accordingly.
    The task is tracked in _background_tasks for deterministic cleanup via wait_for_evaluations().
    """
    try:
        library = sniffio.current_async_library()
    except sniffio.AsyncLibraryNotFoundError:  # pragma: no cover
        logger.warning('No async library detected — cannot dispatch background evaluation')
        return

    if library == 'trio':  # pragma: no cover
        try:
            import trio.lowlevel  # pyright: ignore[reportMissingImports]

            trio.lowlevel.spawn_system_task(coro)  # pyright: ignore[reportUnknownMemberType]
        except ImportError:
            logger.warning('trio detected but not installed — cannot dispatch background evaluation')
    else:
        # asyncio (or any asyncio-compatible backend)
        loop = asyncio.get_running_loop()
        task = loop.create_task(coro)
        with _background_lock:
            _background_tasks.add(task)
        task.add_done_callback(_remove_background_task)


# ============================================================================
# Context variable for disabling evaluation
# ============================================================================

_EVALUATION_DISABLED: ContextVar[bool] = ContextVar('_evaluation_disabled', default=False)


@contextmanager
def disable_evaluation():
    """Context manager to disable all online evaluation in the current context.

    When active, decorated functions still execute normally but no evaluators are dispatched.
    """
    token = _EVALUATION_DISABLED.set(True)
    try:
        yield
    finally:
        _EVALUATION_DISABLED.reset(token)


# ============================================================================
# SpanReference
# ============================================================================


@dataclass(kw_only=True)
class SpanReference:
    """Identifies a span that evaluation results should be associated with.

    Used by sinks to associate evaluation results with the original function execution span.
    """

    trace_id: str
    """The trace ID of the span."""
    span_id: str
    """The span ID of the span."""


# ============================================================================
# EvaluationSink protocol and implementations
# ============================================================================

SinkCallback = Callable[
    [Sequence[EvaluationResult], Sequence[EvaluatorFailure], EvaluatorContext],
    None | Awaitable[None],
]
"""Type alias for bare callables accepted wherever an `EvaluationSink` is expected.

Auto-wrapped in `CallbackSink` when passed as a `sink` parameter.
"""


@runtime_checkable
class EvaluationSink(Protocol):
    """Protocol for evaluation result destinations.

    Implementations receive evaluation results and can send them to any backend
    (Logfire annotations, custom callback, stdout, etc.).
    """

    async def submit(
        self,
        *,
        results: Sequence[EvaluationResult],
        failures: Sequence[EvaluatorFailure],
        context: EvaluatorContext,
        span_reference: SpanReference | None,
    ) -> None:
        """Submit evaluation results to the sink.

        Args:
            results: Evaluation results from successful evaluator runs.
            failures: Failures from evaluator runs that raised exceptions.
            context: The full evaluator context for the function call.
            span_reference: Reference to the OTel span for the function call, if available.
        """
        ...


class CallbackSink:
    """An `EvaluationSink` that delegates to a user-provided callable.

    The callback receives the results, failures, and context. The span_reference is not
    passed to the callback — use a custom `EvaluationSink` implementation if you need it.
    """

    def __init__(self, callback: SinkCallback) -> None:
        self.callback = callback

    async def submit(
        self,
        *,
        results: Sequence[EvaluationResult],
        failures: Sequence[EvaluatorFailure],
        context: EvaluatorContext,
        span_reference: SpanReference | None,
    ) -> None:
        _ = span_reference  # Not passed to callback; use a custom EvaluationSink if needed
        result = self.callback(results, failures, context)
        if inspect.isawaitable(result):
            await result


# ============================================================================
# OnlineEvaluator
# ============================================================================


@dataclass(kw_only=True)
class OnlineEvaluator:
    """Wraps an `Evaluator` with per-evaluator online configuration.

    Different evaluators often need different settings — a cheap heuristic should
    run on 100% of traffic while an expensive LLM judge might run on only 1%.

    Args:
        evaluator: The evaluator to run.
        sample_rate: Probability of running this evaluator (0.0–1.0), or a callable returning
            a float or bool. Defaults to `UNSET`, which uses the config's `default_sample_rate`
            at each call. Set explicitly to override.
        sink: Override sink(s) for this evaluator. If None, the config's default_sink is used.
        max_concurrency: Maximum number of concurrent evaluations for this evaluator.
        gate: Optional predicate that receives the `EvaluatorContext` and returns whether the
            evaluator should run. Called only for sampled requests. Can be sync or async.
    """

    evaluator: Evaluator
    sample_rate: float | Callable[[], float | bool] | Unset = UNSET
    sink: EvaluationSink | Sequence[EvaluationSink] | SinkCallback | None = None
    max_concurrency: int = 10
    gate: Callable[[EvaluatorContext], bool | Awaitable[bool]] | None = None
    semaphore: threading.Semaphore = field(init=False, repr=False)
    """Thread-safe semaphore for per-evaluator concurrency limiting.

    Uses `threading.Semaphore` (not `anyio.Semaphore`) because evaluators may be
    dispatched from multiple background threads when decorating sync functions.
    """

    def __post_init__(self) -> None:
        self.semaphore = threading.Semaphore(self.max_concurrency)


# ============================================================================
# EvaluatorContextData and EvaluatorContextSource
# ============================================================================


@dataclass(kw_only=True)
class EvaluatorContextData:
    """All the data needed to construct an `EvaluatorContext`, fetched in one shot.

    Used by `EvaluatorContextSource` implementations to return context data
    from stored traces without requiring re-execution of the original function.
    """

    inputs: Any
    """The inputs that were provided to the function."""
    output: Any
    """The output that was returned by the function."""
    metadata: dict[str, Any] | None
    """Optional metadata associated with the function call."""
    duration: float
    """The duration of the function execution in seconds."""
    span_tree: SpanTree
    """The span tree captured during the function execution."""


class EvaluatorContextSource(Protocol):
    """Protocol for retrieving stored evaluation context data.

    Implementations fetch all data needed to reconstruct an `EvaluatorContext`
    from stored traces (e.g., Logfire). The batch method allows fetching data
    for multiple spans in a single call.
    """

    async def fetch(self, span: SpanReference) -> EvaluatorContextData:
        """Fetch context data for a single span.

        Args:
            span: Reference to the span to fetch context for.

        Returns:
            The context data for the span.
        """
        ...

    async def fetch_many(self, spans: Sequence[SpanReference]) -> list[EvaluatorContextData]:
        """Fetch context data for multiple spans in a single batch.

        Args:
            spans: References to the spans to fetch context for.

        Returns:
            Context data for each span, in the same order as the input.
        """
        ...


# ============================================================================
# Standalone run_evaluators
# ============================================================================


async def run_evaluators(
    evaluators: Sequence[Evaluator],
    context: EvaluatorContext,
) -> tuple[list[EvaluationResult], list[EvaluatorFailure]]:
    """Run evaluators on a context and return results.

    Useful for re-running evaluators from stored data (via `rebuild_context()`).

    Args:
        evaluators: The evaluators to run.
        context: The evaluator context to evaluate against.

    Returns:
        A tuple of (results, failures).
    """
    all_results: list[EvaluationResult] = []
    all_failures: list[EvaluatorFailure] = []

    async with anyio.create_task_group() as tg:
        results_by_index: dict[int, list[EvaluationResult] | EvaluatorFailure] = {}

        async def _run(idx: int, evaluator: Evaluator) -> None:
            results_by_index[idx] = await run_evaluator(evaluator, context)

        for i, evaluator in enumerate(evaluators):
            tg.start_soon(_run, i, evaluator)

    for i in range(len(evaluators)):
        result = results_by_index[i]
        if isinstance(result, EvaluatorFailure):
            all_failures.append(result)
        else:
            all_results.extend(result)

    return all_results, all_failures


# ============================================================================
# Context rebuilding
# ============================================================================


async def rebuild_context(
    source: EvaluatorContextSource,
    span: SpanReference,
) -> EvaluatorContext:
    """Build an `EvaluatorContext` from stored data via a single fetch.

    Args:
        source: The context source to fetch data from.
        span: Reference to the span to rebuild context for.

    Returns:
        A reconstructed EvaluatorContext.
    """
    data = await source.fetch(span)
    return EvaluatorContext(
        name=None,
        inputs=data.inputs,
        output=data.output,
        expected_output=None,
        metadata=data.metadata,
        duration=data.duration,
        _span_tree=data.span_tree,
        attributes={},
        metrics={},
    )


async def rebuild_contexts(
    source: EvaluatorContextSource,
    spans: Sequence[SpanReference],
) -> list[EvaluatorContext]:
    """Build `EvaluatorContext`s for multiple spans in a single batch fetch.

    Args:
        source: The context source to fetch data from.
        spans: References to the spans to rebuild context for.

    Returns:
        Reconstructed EvaluatorContexts in the same order as the input spans.
    """
    all_data = await source.fetch_many(spans)
    return [
        EvaluatorContext(
            name=None,
            inputs=d.inputs,
            output=d.output,
            expected_output=None,
            metadata=d.metadata,
            duration=d.duration,
            _span_tree=d.span_tree,
            attributes={},
            metrics={},
        )
        for d in all_data
    ]


# ============================================================================
# Internal helpers
# ============================================================================


def _resolve_sample_rate_field(
    online_eval: OnlineEvaluator,
    config: OnlineEvalConfig,
) -> float | Callable[[], float | bool]:
    """Resolve an OnlineEvaluator's sample_rate, falling back to config default if UNSET."""
    if isinstance(online_eval.sample_rate, Unset):
        return config.default_sample_rate
    return online_eval.sample_rate


def _resolve_sample_rate(rate: float | Callable[[], float | bool]) -> float | bool:
    """Resolve a sample rate value, calling it if it's a callable."""
    if callable(rate):
        return rate()
    return rate


def _should_evaluate(rate: float | Callable[[], float | bool], global_enabled: bool) -> bool:
    """Determine whether an evaluator should run based on sampling configuration."""
    if not global_enabled:  # pragma: no cover
        return False
    if _EVALUATION_DISABLED.get():  # pragma: no cover
        return False

    try:
        resolved = _resolve_sample_rate(rate)
    except Exception:
        logger.exception('Error resolving sample rate — skipping evaluator')
        return False

    # Callable can return bool (True = always, False = never)
    if isinstance(resolved, bool):
        return resolved

    # Float: probability
    if resolved >= 1.0:
        return True
    if resolved <= 0.0:
        return False
    return random.random() < resolved


async def _check_gate(gate: Callable[[EvaluatorContext], bool | Awaitable[bool]], ctx: EvaluatorContext) -> bool:
    """Check a gate condition, handling both sync and async gates."""
    result = gate(ctx)
    if inspect.isawaitable(result):
        return await result
    return result


def _resolve_sinks(
    evaluator_sink: EvaluationSink | Sequence[EvaluationSink] | SinkCallback | None,
    default_sink: EvaluationSink | Sequence[EvaluationSink] | SinkCallback | None,
) -> list[EvaluationSink]:
    """Resolve the sinks to use for an evaluator, following the resolution order."""
    raw = evaluator_sink if evaluator_sink is not None else default_sink
    if raw is None:
        return []
    return _normalize_sinks(raw)


def _normalize_sinks(
    sink: EvaluationSink | Sequence[EvaluationSink] | SinkCallback,
) -> list[EvaluationSink]:
    """Normalize a sink specification to a list of EvaluationSink instances."""
    if isinstance(sink, EvaluationSink):
        return [sink]
    if callable(sink):
        return [CallbackSink(sink)]
    return [_normalize_single_sink(s) for s in sink]


def _normalize_single_sink(sink: EvaluationSink | SinkCallback) -> EvaluationSink:
    if isinstance(sink, EvaluationSink):
        return sink
    return CallbackSink(sink)  # pragma: no cover


async def _dispatch_single_evaluator(
    online_eval: OnlineEvaluator,
    context: EvaluatorContext,
    span_reference: SpanReference | None,
    sinks: list[EvaluationSink],
) -> None:
    """Run a single evaluator's gate check, evaluation, and sink submission."""
    # Check gate in the background — never blocks the caller
    if online_eval.gate is not None:
        try:
            if not await _check_gate(online_eval.gate, context):
                return
        except Exception:
            logger.exception('Gate check failed for %r', online_eval.evaluator)
            return

    if not online_eval.semaphore.acquire(blocking=False):
        logger.warning(
            'Evaluation dropped: max concurrency (%d) reached for %r',
            online_eval.max_concurrency,
            online_eval.evaluator,
        )
        return

    try:
        raw_result = await run_evaluator(online_eval.evaluator, context)

        if isinstance(raw_result, EvaluatorFailure):
            results: Sequence[EvaluationResult] = []
            failures: Sequence[EvaluatorFailure] = [raw_result]
        else:
            results = raw_result
            failures = []

        for sink in sinks:
            try:
                await sink.submit(
                    results=results,
                    failures=failures,
                    context=context,
                    span_reference=span_reference,
                )
            except Exception:
                logger.exception('Error submitting evaluation results to sink %r', sink)
    finally:
        online_eval.semaphore.release()


async def _dispatch_evaluators(
    online_evaluators: list[OnlineEvaluator],
    context: EvaluatorContext,
    span_reference: SpanReference | None,
    config: OnlineEvalConfig,
) -> None:
    """Run all selected evaluators concurrently and submit results to their sinks."""
    async with anyio.create_task_group() as tg:
        for online_eval in online_evaluators:
            sinks = _resolve_sinks(online_eval.sink, config.default_sink)
            tg.start_soon(
                _dispatch_single_evaluator,
                online_eval,
                context,
                span_reference,
                sinks,
            )


def _capture_inputs(func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Capture function inputs as a dictionary."""
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


# ============================================================================
# OnlineEvalConfig
# ============================================================================


@dataclass(kw_only=True)
class OnlineEvalConfig:
    """Holds cross-evaluator defaults for online evaluation.

    Create instances for different evaluation configurations, or use the global
    `DEFAULT_CONFIG` via the module-level `evaluate()` and `configure()` functions.
    """

    default_sink: EvaluationSink | Sequence[EvaluationSink] | SinkCallback | None = None
    """Default sink(s) for evaluators that don't specify their own."""
    default_sample_rate: float | Callable[[], float | bool] = 1.0
    """Default sample rate for evaluators that don't specify their own."""
    enabled: bool = True
    """Whether online evaluation is enabled for this config."""
    metadata: dict[str, Any] | None = None
    """Optional metadata to include in evaluator contexts."""

    def evaluate(
        self,
        *evaluators: Evaluator | OnlineEvaluator,
    ) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
        """Decorator to attach online evaluators to a function.

        Bare `Evaluator` instances are auto-wrapped in `OnlineEvaluator` at decoration time
        (so concurrency semaphores are shared across calls). Their `sample_rate` defaults to
        `UNSET`, which resolves to the config's `default_sample_rate` at each call — so
        changes to the config after decoration take effect.

        Args:
            *evaluators: Evaluators to attach. Can be `Evaluator` or `OnlineEvaluator` instances.

        Returns:
            A decorator that wraps the function with online evaluation.
        """
        online_evals = [e if isinstance(e, OnlineEvaluator) else OnlineEvaluator(evaluator=e) for e in evaluators]

        def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
            if inspect.iscoroutinefunction(func):
                return _wrap_async(func, online_evals, self)  # pyright: ignore[reportReturnType]
            else:
                return _wrap_sync(func, online_evals, self)

        return decorator


def _wrap_async(
    func: Callable[_P, Awaitable[_R]],
    online_evals: list[OnlineEvaluator],
    config: OnlineEvalConfig,
) -> Callable[_P, Awaitable[_R]]:
    """Wrap an async function with online evaluation."""

    @functools.wraps(func)
    async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        # If evaluation is globally disabled, just run the function
        if not config.enabled or _EVALUATION_DISABLED.get():
            return await func(*args, **kwargs)

        # Determine which evaluators are sampled (before running the function)
        sampled = [
            oe for oe in online_evals if _should_evaluate(_resolve_sample_rate_field(oe, config), config.enabled)
        ]
        if not sampled:
            return await func(*args, **kwargs)

        # Capture inputs
        inputs = _capture_inputs(func, args, kwargs)

        # Run the function with span tree capture
        with logfire_span('evaluate {func_name}', func_name=func.__qualname__) as span, context_subtree() as span_tree:
            t0 = time.perf_counter()
            result = await func(*args, **kwargs)
            duration = time.perf_counter() - t0

        # Build context
        context = EvaluatorContext(
            name=func.__qualname__,
            inputs=inputs,
            output=result,
            expected_output=None,
            metadata=config.metadata,
            duration=duration,
            _span_tree=span_tree,
            attributes={},
            metrics={},
        )

        # Extract span reference from the logfire span
        span_reference = _extract_span_reference(span)

        # Dispatch all sampled evaluators to the background — gate checks happen there
        await _spawn_background_task(_dispatch_evaluators(sampled, context, span_reference, config))

        return result

    return wrapper


def _wrap_sync(
    func: Callable[_P, _R],
    online_evals: list[OnlineEvaluator],
    config: OnlineEvalConfig,
) -> Callable[_P, _R]:
    """Wrap a sync function with online evaluation."""

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        # If evaluation is globally disabled, just run the function
        if not config.enabled or _EVALUATION_DISABLED.get():
            return func(*args, **kwargs)

        # Determine which evaluators are sampled
        sampled = [
            oe for oe in online_evals if _should_evaluate(_resolve_sample_rate_field(oe, config), config.enabled)
        ]
        if not sampled:
            return func(*args, **kwargs)

        # Capture inputs
        inputs = _capture_inputs(func, args, kwargs)

        # Run the function with span tree capture
        with logfire_span('evaluate {func_name}', func_name=func.__qualname__) as span, context_subtree() as span_tree:
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            duration = time.perf_counter() - t0

        # Build context
        context = EvaluatorContext(
            name=func.__qualname__,
            inputs=inputs,
            output=result,
            expected_output=None,
            metadata=config.metadata,
            duration=duration,
            _span_tree=span_tree,
            attributes={},
            metrics={},
        )

        # Extract span reference
        span_reference = _extract_span_reference(span)

        # Dispatch all sampled evaluators to the background — gate checks happen there
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(_dispatch_evaluators(sampled, context, span_reference, config))
            with _background_lock:
                _background_tasks.add(task)
            task.add_done_callback(_remove_background_task)
        except RuntimeError:
            # No running loop — run in a background thread
            def _thread_target() -> None:
                try:
                    asyncio.run(_dispatch_evaluators(sampled, context, span_reference, config))
                finally:
                    with _background_lock:
                        _background_threads.discard(thread)

            thread = threading.Thread(target=_thread_target, daemon=True)
            with _background_lock:
                _background_threads.add(thread)
            thread.start()

        return result

    return wrapper


def _extract_span_reference(span: Any) -> SpanReference | None:
    """Extract a SpanReference from an OTel-compatible span, if available.

    Works with any span that implements `get_span_context()` (the standard
    OpenTelemetry Span interface), including LogfireSpan, OTel SDK spans,
    and any other ReadableSpan implementation.

    Returns None if the span doesn't have a valid context (e.g., when
    instrumentation is not configured and trace/span IDs are zero).
    """
    get_span_context = getattr(span, 'get_span_context', None)
    if get_span_context is None:  # pragma: no cover
        return None
    try:
        ctx = get_span_context()
    except Exception:  # pragma: no cover
        return None
    if ctx is not None and ctx.trace_id and ctx.span_id:
        return SpanReference(
            trace_id=format(ctx.trace_id, '032x'),
            span_id=format(ctx.span_id, '016x'),
        )
    return None  # pragma: lax no cover


# ============================================================================
# Global default config and module-level convenience functions
# ============================================================================

DEFAULT_CONFIG = OnlineEvalConfig()
"""The global default `OnlineEvalConfig` instance.

Module-level functions like `evaluate()` and `configure()` delegate to this instance.
"""


def evaluate(*evaluators: Evaluator | OnlineEvaluator) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Decorator to attach online evaluators to a function using the global default config.

    Equivalent to `DEFAULT_CONFIG.evaluate(...)`.

    Args:
        *evaluators: Evaluators to attach. Can be `Evaluator` or `OnlineEvaluator` instances.

    Returns:
        A decorator that wraps the function with online evaluation.

    Example:
    ```python
    from dataclasses import dataclass

    from pydantic_evals.evaluators import Evaluator, EvaluatorContext
    from pydantic_evals.online import evaluate


    @dataclass
    class IsNonEmpty(Evaluator):
        def evaluate(self, ctx: EvaluatorContext) -> bool:
            return bool(ctx.output)


    @evaluate(IsNonEmpty())
    async def my_function(x: int) -> int:
        return x
    ```
    """
    return DEFAULT_CONFIG.evaluate(*evaluators)


def configure(
    *,
    default_sink: EvaluationSink | Sequence[EvaluationSink] | SinkCallback | None | Unset = UNSET,
    default_sample_rate: float | Callable[[], float | bool] | Unset = UNSET,
    enabled: bool | Unset = UNSET,
    metadata: dict[str, Any] | None | Unset = UNSET,
) -> None:
    """Configure the global default `OnlineEvalConfig`.

    Only provided values are updated; unset arguments are ignored.
    Pass `None` explicitly to clear `default_sink` or `metadata`.

    Args:
        default_sink: Default sink(s) for evaluators. Pass `None` to clear.
        default_sample_rate: Default sample rate for evaluators.
        enabled: Whether online evaluation is enabled.
        metadata: Metadata to include in evaluator contexts. Pass `None` to clear.
    """
    if not isinstance(default_sink, Unset):
        DEFAULT_CONFIG.default_sink = default_sink
    if not isinstance(default_sample_rate, Unset):
        DEFAULT_CONFIG.default_sample_rate = default_sample_rate
    if not isinstance(enabled, Unset):
        DEFAULT_CONFIG.enabled = enabled
    if not isinstance(metadata, Unset):
        DEFAULT_CONFIG.metadata = metadata


async def wait_for_evaluations(*, timeout: float = 30.0) -> None:
    """Wait for all pending background evaluation tasks and threads to complete.

    This is useful in tests to deterministically wait for background evaluators
    to finish instead of relying on timing-based sleeps.

    For async tasks (dispatched from async decorated functions), this awaits them directly.
    For background threads (dispatched from sync decorated functions called outside an
    async context), this joins them with the given timeout.

    Args:
        timeout: Maximum seconds to wait for each background thread. Defaults to 30.
    """
    # Snapshot under lock, then await/join without holding the lock
    with _background_lock:
        tasks_snapshot = list(_background_tasks)
        threads_snapshot = list(_background_threads)

    # Await all pending async tasks (using anyio for backend compatibility)
    if tasks_snapshot:
        async with anyio.create_task_group() as tg:
            for task in tasks_snapshot:
                tg.start_soon(_wait_task, task)
    # Join all pending background threads (from sync function dispatch)
    for thread in threads_snapshot:
        thread.join(timeout=timeout)
        if thread.is_alive():  # pragma: no cover
            logger.warning('Background evaluation thread did not complete within %.1fs timeout', timeout)


async def _wait_task(task: asyncio.Task[Any]) -> None:
    """Await an asyncio.Task, suppressing exceptions (they're already logged by the task)."""
    try:
        await task
    except asyncio.CancelledError:  # pragma: no cover
        pass  # Expected during shutdown
    except BaseException:  # pragma: no cover
        logger.warning('Unexpected exception in background evaluation task', exc_info=True)
