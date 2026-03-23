"""Online evaluation — attach evaluators to live functions for automatic background evaluation.

This module provides the infrastructure for running evaluators on production (or staging) traffic.
The same `Evaluator` instances used with `Dataset.evaluate()` work here, the difference is in how
they are wired up (decorator vs dataset) rather than what they are.

Example:
```python {test="skip" lint="skip"}
from pydantic_evals.online import evaluate

@evaluate(Equals(value=42))
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
import time
from collections.abc import Awaitable, Callable, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import anyio
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
)

logger = logging.getLogger('pydantic_evals.online')

_P = ParamSpec('_P')
_R = TypeVar('_R')

# Strong references to background tasks to prevent garbage collection.
# See: https://docs.python.org/3/library/asyncio-task.html#creating-tasks
_background_tasks: set[asyncio.Task[Any]] = set()

# ============================================================================
# Context variable for disabling evaluation
# ============================================================================

_EVALUATION_DISABLED: ContextVar[bool] = ContextVar('_evaluation_disabled', default=False)


@contextmanager
def disable_evaluation():
    """Context manager to disable all online evaluation in the current context.

    When active, decorated functions still execute normally but no evaluators are dispatched.

    Example:
    ```python {test="skip" lint="skip"}
    with disable_evaluation():
        result = await my_evaluated_function("test input")
        # No evaluators run
    ```
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

    Example:
    ```python {test="skip" lint="skip"}
    async def my_callback(results, failures, context):
        for result in results:
            print(f"{result.name}: {result.value}")

    sink = CallbackSink(my_callback)
    ```
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
            a float or bool. Callables enable integration with dynamic configuration systems.
        sink: Override sink(s) for this evaluator. If None, the config's default_sink is used.
        max_concurrency: Maximum number of concurrent evaluations for this evaluator.
        gate: Optional predicate that receives the `EvaluatorContext` and returns whether the
            evaluator should run. Called only for sampled requests. Can be sync or async.

    Example:
    ```python {test="skip" lint="skip"}
    OnlineEvaluator(
        LLMJudge(rubric="Is the response helpful?"),
        sample_rate=0.01,
        max_concurrency=5,
    )
    ```
    """

    evaluator: Evaluator
    sample_rate: float | Callable[[], float | bool] = 1.0
    sink: EvaluationSink | Sequence[EvaluationSink] | SinkCallback | None = None
    max_concurrency: int = 10
    gate: Callable[[EvaluatorContext], bool | Awaitable[bool]] | None = None
    semaphore: anyio.Semaphore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.semaphore = anyio.Semaphore(self.max_concurrency)


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


def _resolve_sample_rate(rate: float | Callable[[], float | bool]) -> float | bool:
    """Resolve a sample rate value, calling it if it's a callable."""
    if callable(rate):
        return rate()
    return rate


def _should_evaluate(rate: float | Callable[[], float | bool], global_enabled: bool) -> bool:
    """Determine whether an evaluator should run based on sampling configuration."""
    if not global_enabled:
        return False
    if _EVALUATION_DISABLED.get():
        return False

    resolved = _resolve_sample_rate(rate)

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
    return CallbackSink(sink)


async def _dispatch_single_evaluator(
    online_eval: OnlineEvaluator,
    context: EvaluatorContext,
    span_reference: SpanReference | None,
    sinks: list[EvaluationSink],
) -> None:
    """Run a single evaluator and submit results to sinks."""
    try:
        online_eval.semaphore.acquire_nowait()
    except anyio.WouldBlock:
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

    Example:
    ```python {test="skip" lint="skip"}
    from pydantic_evals.online import OnlineEvalConfig

    my_eval = OnlineEvalConfig(
        default_sink=my_sink,
        default_sample_rate=0.1,
    )

    @my_eval.evaluate(LLMJudge(rubric="Is the response helpful?"))
    async def my_function(query: str) -> str:
        ...
    ```
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

        Bare `Evaluator` instances are auto-wrapped in `OnlineEvaluator` with this config's defaults.
        The decorated function's signature and return type are preserved.

        Args:
            *evaluators: Evaluators to attach. Can be `Evaluator` or `OnlineEvaluator` instances.

        Returns:
            A decorator that wraps the function with online evaluation.
        """
        online_evals = [
            e if isinstance(e, OnlineEvaluator) else OnlineEvaluator(evaluator=e, sample_rate=self.default_sample_rate)
            for e in evaluators
        ]

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
        sampled = [oe for oe in online_evals if _should_evaluate(oe.sample_rate, config.enabled)]
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

        # Check gates for sampled evaluators
        gated: list[OnlineEvaluator] = []
        for oe in sampled:
            if oe.gate is not None:
                try:
                    if not await _check_gate(oe.gate, context):
                        continue
                except Exception:
                    logger.exception('Gate check failed for %r', oe.evaluator)
                    continue
            gated.append(oe)

        if gated:
            # Dispatch evaluators in the background
            try:
                loop = asyncio.get_running_loop()
                task = loop.create_task(_dispatch_evaluators(gated, context, span_reference, config))
                _background_tasks.add(task)
                task.add_done_callback(_background_tasks.discard)
            except RuntimeError:
                # No running loop (shouldn't happen for async but be defensive)
                logger.warning('No running event loop for background evaluation dispatch')

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
        sampled = [oe for oe in online_evals if _should_evaluate(oe.sample_rate, config.enabled)]
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

        # Check gates synchronously (async gates not supported for sync functions)
        gated: list[OnlineEvaluator] = []
        for oe in sampled:
            if oe.gate is not None:
                try:
                    gate_result = oe.gate(context)
                    if inspect.isawaitable(gate_result):
                        logger.warning('Async gate on sync function %r — skipping evaluator %r', func, oe.evaluator)
                        continue
                    if not gate_result:
                        continue
                except Exception:
                    logger.exception('Gate check failed for %r', oe.evaluator)
                    continue
            gated.append(oe)

        if gated:
            # Try to dispatch to an existing event loop, or start a background thread
            try:
                loop = asyncio.get_running_loop()
                task = loop.create_task(_dispatch_evaluators(gated, context, span_reference, config))
                _background_tasks.add(task)
                task.add_done_callback(_background_tasks.discard)
            except RuntimeError:
                # No running loop — run in a background thread
                import threading

                thread = threading.Thread(
                    target=lambda: asyncio.run(_dispatch_evaluators(gated, context, span_reference, config)),
                    daemon=True,
                )
                thread.start()

        return result

    return wrapper


def _extract_span_reference(span: Any) -> SpanReference | None:
    """Extract a SpanReference from a logfire span, if available."""
    try:
        # logfire spans expose the underlying OTel span context
        otel_span = getattr(span, '_span', None) or getattr(span, 'span', None)
        if otel_span is not None:
            ctx = otel_span.get_span_context()
            if ctx is not None and ctx.trace_id and ctx.span_id:
                return SpanReference(
                    trace_id=format(ctx.trace_id, '032x'),
                    span_id=format(ctx.span_id, '016x'),
                )
    except Exception:
        pass
    return None


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
    ```python {test="skip" lint="skip"}
    from pydantic_evals.online import evaluate
    from pydantic_evals.evaluators import Equals

    @evaluate(Equals(value=42))
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
