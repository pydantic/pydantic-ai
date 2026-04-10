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
import threading
import time
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import anyio
from typing_extensions import ParamSpec, TypeVar

from . import _online as _online_internal, _task_run
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
    'EvaluatorContextSource',
    'OnErrorCallback',
    'OnErrorLocation',
    'OnMaxConcurrencyCallback',
    'OnSamplingErrorCallback',
    'OnlineEvalConfig',
    'OnlineEvaluator',
    'SamplingContext',
    'SamplingMode',
    'SinkCallback',
    'SpanReference',
    'configure',
    'disable_evaluation',
    'evaluate',
    'run_evaluators',
    'wait_for_evaluations',
)

OnErrorLocation = Literal['sink', 'on_max_concurrency']
"""The location within the online evaluation pipeline where an error occurred."""

SamplingMode = Literal['independent', 'correlated']
"""Controls how per-evaluator sample rates interact across evaluators for a single call.

- `'independent'` (default): Each evaluator flips its own coin. With N evaluators each at
  rate *r*, the probability of *any* evaluation overhead is ``1 − (1−r)^N``.
- `'correlated'`: A single random seed is generated per call and shared across evaluators.
  An evaluator runs when ``call_seed < rate``, so lower-rate evaluators' calls are always
  a subset of higher-rate ones. The probability of *any* overhead equals ``max(rate_i)``.
"""


@dataclass(kw_only=True)
class SamplingContext:
    """Context available when deciding whether to sample an evaluator.

    Contains the information available *before* the decorated function runs — the evaluator
    instance, function inputs, config metadata, and a per-call random seed. The function's
    output and duration are not yet available at sampling time.
    """

    evaluator: Evaluator
    """The evaluator being sampled."""
    inputs: Any
    """The inputs to the decorated function."""
    metadata: dict[str, Any] | None
    """Metadata from the [`OnlineEvalConfig`][pydantic_evals.online.OnlineEvalConfig], if set."""
    call_seed: float
    """A uniform random value in [0, 1) generated once per decorated function call.

    Shared across all evaluators for the same call. In `'correlated'` sampling mode this is
    used automatically; in `'independent'` mode it is available for custom `sample_rate`
    callables that want to implement their own correlated logic.
    """


OnMaxConcurrencyCallback = Callable[[EvaluatorContext], None | Awaitable[None]]
"""Callback invoked when an evaluation is dropped due to concurrency limits.

Receives the `EvaluatorContext` that would have been evaluated. Can be sync or async.
"""

OnSamplingErrorCallback = Callable[[Exception, Evaluator], None]
"""Callback invoked when a `sample_rate` callable raises an exception.

Called synchronously before the decorated function runs. Receives the exception
and the evaluator whose `sample_rate` failed. Must be sync (not async).
If set, the evaluator is skipped. If not set, the exception propagates to the caller.
"""

OnErrorCallback = Callable[
    [Exception, EvaluatorContext, Evaluator, OnErrorLocation],
    None | Awaitable[None],
]
"""Callback invoked when an exception occurs in the online evaluation pipeline.

Receives the exception, the evaluator context, the evaluator instance, and a
location string indicating where the error occurred. Can be sync or async.
"""

_P = ParamSpec('_P')
_R = TypeVar('_R')
_EVALUATION_DISABLED = _online_internal.EVALUATION_DISABLED


@contextmanager
def disable_evaluation() -> Iterator[None]:
    """Context manager to disable all online evaluation in the current context.

    When active, decorated functions still execute normally but no evaluators are dispatched.
    """
    token = _EVALUATION_DISABLED.set(True)
    try:
        yield
    finally:
        _EVALUATION_DISABLED.reset(token)


@dataclass(kw_only=True)
class SpanReference:
    """Identifies a span that evaluation results should be associated with.

    Used by sinks to associate evaluation results with the original function execution span.
    """

    trace_id: str
    """The trace ID of the span."""
    span_id: str
    """The span ID of the span."""


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


@dataclass(kw_only=True)
class OnlineEvaluator:
    """Wraps an `Evaluator` with per-evaluator online configuration.

    Different evaluators often need different settings — a cheap heuristic should
    run on 100% of traffic while an expensive LLM judge might run on only 1%.
    """

    evaluator: Evaluator
    """The evaluator to run."""
    sample_rate: float | Callable[[SamplingContext], float | bool] | None = None
    """Probability of running this evaluator (0.0–1.0), or a callable returning a float or bool.

    When a callable, it receives a [`SamplingContext`][pydantic_evals.online.SamplingContext]
    with the function inputs, config metadata, and evaluator name — but not the output or
    duration (which aren't available yet at sampling time).

    Defaults to `None`, which uses the config's `default_sample_rate` at each call.
    Set explicitly to override.
    """
    max_concurrency: int = 10
    """Maximum number of concurrent evaluations for this evaluator."""

    sink: EvaluationSink | Sequence[EvaluationSink | SinkCallback] | SinkCallback | None = None
    """Override sink(s) for this evaluator. If `None`, the config's `default_sink` is used."""

    on_max_concurrency: OnMaxConcurrencyCallback | None = None
    """Called when an evaluation is dropped because `max_concurrency` was reached.

    Receives the `EvaluatorContext` that would have been evaluated. Can be sync or async.
    If `None` (the default), dropped evaluations are silently ignored.
    """
    on_sampling_error: OnSamplingErrorCallback | None = None
    """Called synchronously when a `sample_rate` callable raises an exception.

    Receives the exception and the evaluator. Must be sync (not async), since sampling
    runs before the decorated function. If set, the evaluator is skipped. If `None`,
    uses the config's `on_sampling_error` default. If neither is set, the exception
    propagates to the caller.
    """
    on_error: OnErrorCallback | None = None
    """Called when an exception occurs in a sink or on_max_concurrency callback.

    Receives the exception, evaluator context, evaluator instance, and a location string
    (`'sink'` or `'on_max_concurrency'`). Can be sync or async.
    If `None`, uses the config's `on_error` default. If neither is set, exceptions are
    silently suppressed.
    """

    def __post_init__(self) -> None:
        self.semaphore = threading.Semaphore(self.max_concurrency)


class EvaluatorContextSource(Protocol):
    """Protocol for retrieving stored evaluator contexts.

    Implementations reconstruct [`EvaluatorContext`][pydantic_evals.evaluators.EvaluatorContext]
    objects from stored traces (e.g., Logfire). The batch method allows fetching contexts
    for multiple spans in a single call.
    """

    async def fetch(self, span: SpanReference) -> EvaluatorContext:
        """Fetch an evaluator context for a single span.

        Args:
            span: Reference to the span to fetch context for.

        Returns:
            The evaluator context for the span.
        """
        return (await self.fetch_many([span]))[0]

    async def fetch_many(self, spans: Sequence[SpanReference]) -> list[EvaluatorContext]:
        """Fetch evaluator contexts for multiple spans in a single batch.

        Args:
            spans: References to the spans to fetch context for.

        Returns:
            Evaluator contexts in the same order as the input spans.
        """
        ...


async def run_evaluators(
    evaluators: Sequence[Evaluator],
    context: EvaluatorContext,
) -> tuple[list[EvaluationResult], list[EvaluatorFailure]]:
    """Run evaluators on a context and return results.

    Useful for re-running evaluators from stored data.

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


def _capture_inputs(sig: inspect.Signature, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Capture function inputs as a dictionary using a pre-computed signature."""
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


def _build_sampling_context(
    evaluator: Evaluator,
    inputs: Any,
    metadata: dict[str, Any] | None,
    call_seed: float,
) -> SamplingContext:
    return SamplingContext(
        evaluator=evaluator,
        inputs=inputs,
        metadata=metadata,
        call_seed=call_seed,
    )


@dataclass(kw_only=True)
class OnlineEvalConfig:
    """Holds cross-evaluator defaults for online evaluation.

    Create instances for different evaluation configurations, or use the global
    `DEFAULT_CONFIG` via the module-level `evaluate()` and `configure()` functions.
    """

    default_sink: EvaluationSink | Sequence[EvaluationSink | SinkCallback] | SinkCallback | None = None
    """Default sink(s) for evaluators that don't specify their own."""
    default_sample_rate: float | Callable[[SamplingContext], float | bool] = 1.0
    """Default sample rate for evaluators that don't specify their own."""
    sampling_mode: SamplingMode = 'independent'
    """Controls how per-evaluator sample rates interact for a single call.

    - `'independent'` (default): each evaluator decides independently.
    - `'correlated'`: a shared random seed is used so that lower-rate evaluators'
      calls are a subset of higher-rate ones, minimising total overhead.

    See [`SamplingMode`][pydantic_evals.online.SamplingMode] for details.
    """
    enabled: bool = True
    """Whether online evaluation is enabled for this config."""
    metadata: dict[str, Any] | None = None
    """Optional metadata to include in evaluator contexts."""
    on_max_concurrency: OnMaxConcurrencyCallback | None = None
    """Default handler called when an evaluation is dropped because `max_concurrency` was reached.

    Receives the `EvaluatorContext` that would have been evaluated. Can be sync or async.
    If `None` (the default), dropped evaluations are silently ignored.
    Per-evaluator `OnlineEvaluator.on_max_concurrency` overrides this default.
    """
    on_sampling_error: OnSamplingErrorCallback | None = None
    """Default handler called synchronously when a `sample_rate` callable raises.

    Receives the exception and the evaluator. Must be sync (not async).
    If set, the evaluator is skipped. If `None` (the default), the exception
    propagates to the caller.
    Per-evaluator `OnlineEvaluator.on_sampling_error` overrides this default.
    """
    on_error: OnErrorCallback | None = None
    """Default handler called when an exception occurs in a sink or on_max_concurrency callback.

    Receives the exception, evaluator context, evaluator instance, and a location string
    (`'sink'` or `'on_max_concurrency'`). Can be sync or async.
    If `None` (the default), exceptions are silently suppressed.
    Per-evaluator `OnlineEvaluator.on_error` overrides this default.
    """

    def evaluate(
        self,
        *evaluators: Evaluator | OnlineEvaluator,
    ) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
        """Decorator to attach online evaluators to a function.

        Bare `Evaluator` instances are auto-wrapped in `OnlineEvaluator` at decoration time
        (so concurrency semaphores are shared across calls). Their `sample_rate` defaults to
        `None`, which resolves to the config's `default_sample_rate` at each call — so
        changes to the config after decoration take effect.

        Args:
            *evaluators: Evaluators to attach. Can be `Evaluator` or `OnlineEvaluator` instances.

        Returns:
            A decorator that wraps the function with online evaluation.
        """
        online_evals = [e if isinstance(e, OnlineEvaluator) else OnlineEvaluator(evaluator=e) for e in evaluators]

        def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
            if inspect.iscoroutinefunction(func):
                # ParamSpec can't distinguish async from sync return types — _wrap_async returns
                # Callable[_P, Awaitable[_R]] but the decorator signature expects Callable[_P, _R]
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
    sig = inspect.signature(func)

    @functools.wraps(func)
    async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        # If evaluation is globally disabled, or we're already inside an evaluation
        # context (e.g. Dataset.evaluate), just run the function
        if not config.enabled or _EVALUATION_DISABLED.get() or _task_run.CURRENT_TASK_RUN.get() is not None:
            return await func(*args, **kwargs)

        # Capture inputs early so sample_rate callables can use them
        inputs = _capture_inputs(sig, args, kwargs)

        # Determine which evaluators are sampled (before running the function)
        sampled = _online_internal.sample_evaluators(
            online_evals,
            config,
            inputs,
            build_sampling_context=_build_sampling_context,
        )
        if not sampled:
            return await func(*args, **kwargs)

        # Run the function with span tree capture and attribute/metric tracking
        task_run = _task_run.TaskRun()
        token = _task_run.CURRENT_TASK_RUN.set(task_run)
        try:
            with (
                logfire_span('evaluate {func_name}', func_name=func.__qualname__) as span,
                context_subtree() as span_tree,
            ):
                t0 = time.perf_counter()
                result = await func(*args, **kwargs)
                duration = time.perf_counter() - t0
        finally:
            _task_run.CURRENT_TASK_RUN.reset(token)

        # Extract standard metrics (requests, cost, token usage) from the span tree
        if isinstance(span_tree, SpanTree):  # pragma: no branch
            _task_run.extract_span_tree_metrics(task_run, span_tree)

        # Build context
        metadata = dict(config.metadata) if config.metadata is not None else None
        context = EvaluatorContext(
            name=None,
            inputs=inputs,
            output=result,
            expected_output=None,
            metadata=metadata,
            duration=duration,
            _span_tree=span_tree,
            attributes=task_run.attributes,
            metrics=task_run.metrics,
        )

        # Extract span reference from the logfire span
        span_reference = _extract_span_reference(span)

        # Dispatch evaluators on the caller's event loop — preserves ContextVars
        _online_internal.dispatch_async(_online_internal.dispatch_evaluators(sampled, context, span_reference, config))

        return result

    return wrapper


def _wrap_sync(
    func: Callable[_P, _R],
    online_evals: list[OnlineEvaluator],
    config: OnlineEvalConfig,
) -> Callable[_P, _R]:
    """Wrap a sync function with online evaluation."""
    sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        # If evaluation is globally disabled, or we're already inside an evaluation
        # context (e.g. Dataset.evaluate), just run the function
        if not config.enabled or _EVALUATION_DISABLED.get() or _task_run.CURRENT_TASK_RUN.get() is not None:
            return func(*args, **kwargs)

        # Capture inputs early so sample_rate callables can use them
        inputs = _capture_inputs(sig, args, kwargs)

        # Determine which evaluators are sampled
        sampled = _online_internal.sample_evaluators(
            online_evals,
            config,
            inputs,
            build_sampling_context=_build_sampling_context,
        )
        if not sampled:
            return func(*args, **kwargs)

        # Run the function with span tree capture and attribute/metric tracking
        task_run = _task_run.TaskRun()
        token = _task_run.CURRENT_TASK_RUN.set(task_run)
        try:
            with (
                logfire_span('evaluate {func_name}', func_name=func.__qualname__) as span,
                context_subtree() as span_tree,
            ):
                t0 = time.perf_counter()
                result = func(*args, **kwargs)
                duration = time.perf_counter() - t0
        finally:
            _task_run.CURRENT_TASK_RUN.reset(token)

        # Extract standard metrics (requests, cost, token usage) from the span tree
        if isinstance(span_tree, SpanTree):  # pragma: no branch
            _task_run.extract_span_tree_metrics(task_run, span_tree)

        # Build context
        metadata = dict(config.metadata) if config.metadata is not None else None
        context = EvaluatorContext(
            name=None,
            inputs=inputs,
            output=result,
            expected_output=None,
            metadata=metadata,
            duration=duration,
            _span_tree=span_tree,
            attributes=task_run.attributes,
            metrics=task_run.metrics,
        )

        # Extract span reference
        span_reference = _extract_span_reference(span)

        # If there's a running event loop (sync function called from async context),
        # dispatch on that loop. Otherwise, spawn a background thread with its own loop.
        try:
            asyncio.get_running_loop()
            has_running_loop = True
        except RuntimeError:
            has_running_loop = False

        coro = _online_internal.dispatch_evaluators(sampled, context, span_reference, config)
        if has_running_loop:
            _online_internal.dispatch_async(coro)
        else:
            _online_internal.dispatch_in_background_thread(coro)

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
    if (
        ctx is not None
        and isinstance(ctx.trace_id, int)
        and isinstance(ctx.span_id, int)
        and ctx.trace_id
        and ctx.span_id
    ):
        return SpanReference(
            trace_id=format(ctx.trace_id, '032x'),
            span_id=format(ctx.span_id, '016x'),
        )
    return None  # pragma: lax no cover


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
    default_sink: EvaluationSink | Sequence[EvaluationSink | SinkCallback] | SinkCallback | None | Unset = UNSET,
    default_sample_rate: float | Callable[[SamplingContext], float | bool] | Unset = UNSET,
    sampling_mode: SamplingMode | Unset = UNSET,
    enabled: bool | Unset = UNSET,
    metadata: dict[str, Any] | None | Unset = UNSET,
    on_max_concurrency: OnMaxConcurrencyCallback | None | Unset = UNSET,
    on_sampling_error: OnSamplingErrorCallback | None | Unset = UNSET,
    on_error: OnErrorCallback | None | Unset = UNSET,
) -> None:
    """Configure the global default `OnlineEvalConfig`.

    Only provided values are updated; unset arguments are ignored.
    Pass `None` explicitly to clear `default_sink`, `metadata`, `on_max_concurrency`,
    `on_sampling_error`, or `on_error`.

    Args:
        default_sink: Default sink(s) for evaluators. Pass `None` to clear.
        default_sample_rate: Default sample rate for evaluators.
        sampling_mode: Sampling mode (`'independent'` or `'correlated'`).
        enabled: Whether online evaluation is enabled.
        metadata: Metadata to include in evaluator contexts. Pass `None` to clear.
        on_max_concurrency: Default handler for dropped evaluations. Pass `None` to clear.
        on_sampling_error: Default handler for sample_rate exceptions. Pass `None` to clear.
        on_error: Default handler for pipeline exceptions. Pass `None` to clear.
    """
    if not isinstance(default_sink, Unset):
        DEFAULT_CONFIG.default_sink = default_sink
    if not isinstance(default_sample_rate, Unset):
        DEFAULT_CONFIG.default_sample_rate = default_sample_rate
    if not isinstance(sampling_mode, Unset):
        DEFAULT_CONFIG.sampling_mode = sampling_mode
    if not isinstance(enabled, Unset):
        DEFAULT_CONFIG.enabled = enabled
    if not isinstance(metadata, Unset):
        DEFAULT_CONFIG.metadata = metadata
    if not isinstance(on_max_concurrency, Unset):
        DEFAULT_CONFIG.on_max_concurrency = on_max_concurrency
    if not isinstance(on_sampling_error, Unset):
        DEFAULT_CONFIG.on_sampling_error = on_sampling_error
    if not isinstance(on_error, Unset):
        DEFAULT_CONFIG.on_error = on_error


async def wait_for_evaluations(*, timeout: float = 30.0) -> None:
    """Wait for all pending background evaluation tasks and threads to complete.

    This is useful in tests to deterministically wait for background evaluators
    to finish instead of relying on timing-based sleeps.

    For async decorated functions, evaluators run as tasks on the caller's event loop
    and are awaited directly. For sync decorated functions, evaluators run in background
    threads which are joined with the given timeout.

    Args:
        timeout: Maximum seconds to wait for each background thread. Defaults to 30.
    """
    await _online_internal.wait_for_evaluations(timeout=timeout)
