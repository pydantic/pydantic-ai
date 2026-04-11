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
import contextvars
import functools
import inspect
import random
import threading
import time
import warnings
from collections.abc import Awaitable, Callable, Coroutine, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import anyio
import sniffio
from anyio.to_thread import run_sync
from typing_extensions import ParamSpec, TypeVar

from ._utils import UNSET, Unset, logfire_span
from .dataset import (
    _CURRENT_TASK_RUN as _CURRENT_TASK_RUN,  # pyright: ignore[reportPrivateUsage]
    _extract_span_tree_metrics as _extract_span_tree_metrics,  # pyright: ignore[reportPrivateUsage]
    _TaskRun as _TaskRun,  # pyright: ignore[reportPrivateUsage]
)
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

# Protected by _background_lock for thread-safety (free-threaded Python, concurrent callers).
_background_lock = threading.Lock()
_background_tasks: set[asyncio.Task[Any]] = set()
_background_events: set[anyio.Event] = set()  # For trio system tasks (no native handle)
_background_threads: set[threading.Thread] = set()


def _remove_background_task(task: asyncio.Task[Any]) -> None:
    """Callback to remove a completed task from the tracking set (thread-safe)."""
    with _background_lock:
        _background_tasks.discard(task)


def _dispatch_async(coro: Coroutine[Any, Any, None]) -> None:
    """Dispatch an evaluation coroutine on the caller's event loop.

    Uses sniffio to detect the backend and dispatches accordingly:
    - asyncio: asyncio.get_running_loop().create_task() — ContextVars propagate
    - trio: trio.lowlevel.spawn_system_task()

    The task runs on the caller's event loop, NOT in a separate thread,
    so ContextVars from the caller's context are preserved.
    """
    library = sniffio.current_async_library()

    if library == 'trio':  # pragma: no cover
        import trio.lowlevel  # pyright: ignore[reportMissingImports]

        done_event = anyio.Event()
        with _background_lock:
            _background_events.add(done_event)

        async def _trio_task() -> None:
            try:
                await coro
            finally:
                done_event.set()
                with _background_lock:
                    _background_events.discard(done_event)

        trio.lowlevel.spawn_system_task(_trio_task)  # pyright: ignore[reportUnknownMemberType]
    else:
        # asyncio (or any asyncio-compatible backend)
        loop = asyncio.get_running_loop()
        task = loop.create_task(coro)
        with _background_lock:
            _background_tasks.add(task)
        task.add_done_callback(_remove_background_task)


def _dispatch_in_background_thread(coro: Coroutine[Any, Any, None]) -> None:
    """Dispatch an async coroutine to a background daemon thread.

    Used for sync decorated functions where there's no event loop to schedule on.
    The thread runs its own event loop via anyio.run().

    Captures the caller's contextvars so that evaluators running in the background
    thread can access context set by the caller (e.g. request IDs, auth context).
    """
    # Capture caller's context before spawning the thread — background threads
    # don't inherit contextvars, so we snapshot and run within it.
    ctx = contextvars.copy_context()

    async def _run() -> None:
        await coro

    def _thread_target() -> None:
        try:
            ctx.run(anyio.run, _run)
        finally:
            with _background_lock:
                _background_threads.discard(thread)

    thread = threading.Thread(target=_thread_target, daemon=True)
    with _background_lock:
        _background_threads.add(thread)
    try:
        thread.start()
    except Exception:  # pragma: no cover
        with _background_lock:
            _background_threads.discard(thread)


_EVALUATION_DISABLED: ContextVar[bool] = ContextVar('_evaluation_disabled', default=False)


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


def _resolve_sample_rate_field(
    online_eval: OnlineEvaluator,
    config: OnlineEvalConfig,
) -> float | Callable[[SamplingContext], float | bool]:
    """Resolve an OnlineEvaluator's sample_rate, falling back to config default if None."""
    if online_eval.sample_rate is None:
        return config.default_sample_rate
    return online_eval.sample_rate


def _resolve_sample_rate(
    rate: float | Callable[[SamplingContext], float | bool],
    sampling_context: SamplingContext,
) -> float | bool:
    """Resolve a sample rate value, calling it if it's a callable."""
    if callable(rate):
        return rate(sampling_context)
    return rate


def _should_evaluate(
    rate: float | Callable[[SamplingContext], float | bool],
    global_enabled: bool,
    sampling_context: SamplingContext,
    sampling_mode: SamplingMode,
) -> bool:
    """Determine whether an evaluator should run based on sampling configuration."""
    if not global_enabled:  # pragma: no cover
        return False
    if _EVALUATION_DISABLED.get():  # pragma: no cover
        return False

    resolved = _resolve_sample_rate(rate, sampling_context)

    # Callable can return bool (True = always, False = never)
    if isinstance(resolved, bool):
        return resolved

    # Float: probability
    if resolved >= 1.0:
        return True
    if resolved <= 0.0:
        return False

    if sampling_mode == 'correlated':
        # Use the shared per-call seed so all evaluators correlate
        return sampling_context.call_seed < resolved
    else:
        # Independent: each evaluator rolls its own random check
        return random.random() < resolved


def _sample_evaluators(
    online_evals: list[OnlineEvaluator],
    config: OnlineEvalConfig,
    inputs: dict[str, Any],
) -> list[OnlineEvaluator]:
    """Determine which evaluators should run, handling sample_rate exceptions.

    If a sample_rate callable raises and the evaluator (or config) has an
    `on_sampling_error` callback, the error is reported there and the evaluator
    is skipped. If no callback is configured, the exception propagates to the caller.
    """
    call_seed = random.random()
    sampled: list[OnlineEvaluator] = []
    for oe in online_evals:
        sampling_ctx = SamplingContext(
            evaluator=oe.evaluator,
            inputs=inputs,
            metadata=config.metadata,
            call_seed=call_seed,
        )
        try:
            if _should_evaluate(
                _resolve_sample_rate_field(oe, config), config.enabled, sampling_ctx, config.sampling_mode
            ):
                sampled.append(oe)
        except Exception as exc:
            handler = oe.on_sampling_error if oe.on_sampling_error is not None else config.on_sampling_error
            if handler is not None:
                try:
                    handler(exc, oe.evaluator)
                except Exception:
                    pass  # Handler itself failed — suppress to protect other evaluators
            else:
                raise
    return sampled


def _resolve_sinks(
    evaluator_sink: EvaluationSink | Sequence[EvaluationSink | SinkCallback] | SinkCallback | None,
    default_sink: EvaluationSink | Sequence[EvaluationSink | SinkCallback] | SinkCallback | None,
) -> list[EvaluationSink]:
    """Resolve the sinks to use for an evaluator, following the resolution order."""
    raw = evaluator_sink if evaluator_sink is not None else default_sink
    if raw is None:
        return []
    return _normalize_sinks(raw)


def _normalize_sinks(
    sink: EvaluationSink | Sequence[EvaluationSink | SinkCallback] | SinkCallback,
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


async def _call_on_error(
    on_error: OnErrorCallback | None,
    exc: Exception,
    context: EvaluatorContext,
    evaluator: Evaluator,
    location: OnErrorLocation,
) -> None:
    """Invoke the on_error callback, suppressing any exception it raises."""
    if on_error is None:
        return
    try:
        result = on_error(exc, context, evaluator, location)
        if inspect.isawaitable(result):
            await result
    except Exception:
        pass  # Handler itself failed — suppress to protect sibling evaluators


async def _submit_to_sink(
    sink: EvaluationSink,
    results: Sequence[EvaluationResult],
    failures: Sequence[EvaluatorFailure],
    context: EvaluatorContext,
    span_reference: SpanReference | None,
    on_error: OnErrorCallback | None,
    evaluator: Evaluator,
) -> None:
    """Submit results to a single sink, routing exceptions to on_error."""
    try:
        await sink.submit(results=results, failures=failures, context=context, span_reference=span_reference)
    except Exception as exc:
        await _call_on_error(on_error, exc, context, evaluator, 'sink')


async def _dispatch_single_evaluator(
    online_eval: OnlineEvaluator,
    context: EvaluatorContext,
    span_reference: SpanReference | None,
    sinks: list[EvaluationSink],
    on_max_concurrency: Callable[[EvaluatorContext], Any] | None,
    on_error: OnErrorCallback | None,
) -> None:
    """Run a single evaluator's evaluation and sink submission."""
    evaluator = online_eval.evaluator

    if not online_eval.semaphore.acquire(blocking=False):
        if on_max_concurrency is not None:
            try:
                result = on_max_concurrency(context)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                await _call_on_error(on_error, exc, context, evaluator, 'on_max_concurrency')
        return

    try:
        raw_result = await run_evaluator(evaluator, context)

        if isinstance(raw_result, EvaluatorFailure):
            results: Sequence[EvaluationResult] = []
            failures: Sequence[EvaluatorFailure] = [raw_result]
        else:
            results = raw_result
            failures = []

        async with anyio.create_task_group() as tg:
            for sink in sinks:
                tg.start_soon(_submit_to_sink, sink, results, failures, context, span_reference, on_error, evaluator)

    finally:
        online_eval.semaphore.release()


async def _dispatch_evaluators(
    online_evaluators: list[OnlineEvaluator],
    context: EvaluatorContext,
    span_reference: SpanReference | None,
    config: OnlineEvalConfig,
) -> None:
    """Run all selected evaluators concurrently and submit results to their sinks.

    Evaluators with no resolved sinks are skipped entirely — there's nowhere
    to send results, so running the evaluator would be wasted work.
    """
    async with anyio.create_task_group() as tg:
        for online_eval in online_evaluators:
            sinks = _resolve_sinks(online_eval.sink, config.default_sink)
            if not sinks:
                continue
            on_max_concurrency = online_eval.on_max_concurrency
            if on_max_concurrency is None:
                on_max_concurrency = config.on_max_concurrency
            on_error = online_eval.on_error
            if on_error is None:
                on_error = config.on_error
            tg.start_soon(
                _dispatch_single_evaluator,
                online_eval,
                context,
                span_reference,
                sinks,
                on_max_concurrency,
                on_error,
            )


def _capture_inputs(sig: inspect.Signature, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Capture function inputs as a dictionary using a pre-computed signature."""
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


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
        if not config.enabled or _EVALUATION_DISABLED.get() or _CURRENT_TASK_RUN.get() is not None:
            return await func(*args, **kwargs)

        # Capture inputs early so sample_rate callables can use them
        inputs = _capture_inputs(sig, args, kwargs)

        # Determine which evaluators are sampled (before running the function)
        sampled = _sample_evaluators(online_evals, config, inputs)
        if not sampled:
            return await func(*args, **kwargs)

        # Run the function with span tree capture and attribute/metric tracking
        task_run = _TaskRun()
        token = _CURRENT_TASK_RUN.set(task_run)
        try:
            with (
                logfire_span('evaluate {func_name}', func_name=func.__qualname__) as span,
                context_subtree() as span_tree,
            ):
                t0 = time.perf_counter()
                result = await func(*args, **kwargs)
                duration = time.perf_counter() - t0
        finally:
            _CURRENT_TASK_RUN.reset(token)

        # Extract standard metrics (requests, cost, token usage) from the span tree
        if isinstance(span_tree, SpanTree):  # pragma: no branch
            _extract_span_tree_metrics(task_run, span_tree)

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
        _dispatch_async(_dispatch_evaluators(sampled, context, span_reference, config))

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
        if not config.enabled or _EVALUATION_DISABLED.get() or _CURRENT_TASK_RUN.get() is not None:
            return func(*args, **kwargs)

        # Capture inputs early so sample_rate callables can use them
        inputs = _capture_inputs(sig, args, kwargs)

        # Determine which evaluators are sampled
        sampled = _sample_evaluators(online_evals, config, inputs)
        if not sampled:
            return func(*args, **kwargs)

        # Run the function with span tree capture and attribute/metric tracking
        task_run = _TaskRun()
        token = _CURRENT_TASK_RUN.set(task_run)
        try:
            with (
                logfire_span('evaluate {func_name}', func_name=func.__qualname__) as span,
                context_subtree() as span_tree,
            ):
                t0 = time.perf_counter()
                result = func(*args, **kwargs)
                duration = time.perf_counter() - t0
        finally:
            _CURRENT_TASK_RUN.reset(token)

        # Extract standard metrics (requests, cost, token usage) from the span tree
        if isinstance(span_tree, SpanTree):  # pragma: no branch
            _extract_span_tree_metrics(task_run, span_tree)

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

        coro = _dispatch_evaluators(sampled, context, span_reference, config)
        if has_running_loop:
            _dispatch_async(coro)
        else:
            _dispatch_in_background_thread(coro)

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
    with _background_lock:
        tasks_snapshot = list(_background_tasks)
        events_snapshot = list(_background_events)
        threads_snapshot = list(_background_threads)

    # Await async tasks (from async decorated functions on asyncio)
    for task in tasks_snapshot:
        try:
            await task
        except BaseException:  # pragma: no cover
            pass  # Exceptions are handled inside _dispatch_single_evaluator

    # Await trio events (from async decorated functions on trio)
    for event in events_snapshot:
        await event.wait()  # pragma: no cover

    # Join background threads (from sync decorated functions) without blocking the event loop
    if threads_snapshot:

        def _join_threads() -> None:
            for thread in threads_snapshot:
                thread.join(timeout=timeout)
                if thread.is_alive():  # pragma: no cover
                    warnings.warn(f'Background evaluation thread did not complete within {timeout:.1f}s timeout')

        await run_sync(_join_threads)
