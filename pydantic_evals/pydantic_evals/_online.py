"""Private helpers shared by the online-eval decorator and agent capability."""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import random
import threading
import warnings
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from contextvars import ContextVar
from typing import Any, Literal, Protocol, cast, runtime_checkable

import anyio
import sniffio
from anyio.to_thread import run_sync

from .evaluators._run_evaluator import run_evaluator
from .evaluators.context import EvaluatorContext
from .evaluators.evaluator import EvaluationResult, Evaluator, EvaluatorFailure

SinkCallback = Callable[
    [Sequence[EvaluationResult], Sequence[EvaluatorFailure], EvaluatorContext],
    None | Awaitable[None],
]
SamplingMode = Literal['independent', 'correlated']
OnErrorLocation = Literal['sink', 'on_max_concurrency']
OnSamplingErrorCallback = Callable[[Exception, Evaluator], None]
OnMaxConcurrencyCallback = Callable[[EvaluatorContext], None | Awaitable[None]]
OnErrorCallback = Callable[
    [Exception, EvaluatorContext, Evaluator, OnErrorLocation],
    None | Awaitable[None],
]
SamplingContextBuilder = Callable[[Evaluator, Any, dict[str, Any] | None, float], Any]


@runtime_checkable
class EvaluationSink(Protocol):
    async def submit(
        self,
        *,
        results: Sequence[EvaluationResult],
        failures: Sequence[EvaluatorFailure],
        context: EvaluatorContext,
        span_reference: Any | None,
    ) -> None: ...


class OnlineEvaluatorLike(Protocol):
    evaluator: Evaluator
    sample_rate: float | Callable[[Any], float | bool] | None
    max_concurrency: int
    semaphore: threading.Semaphore
    sink: Any
    on_max_concurrency: OnMaxConcurrencyCallback | None
    on_sampling_error: OnSamplingErrorCallback | None
    on_error: OnErrorCallback | None


class OnlineEvalConfigLike(Protocol):
    default_sink: Any
    default_sample_rate: float | Callable[[Any], float | bool]
    sampling_mode: SamplingMode
    enabled: bool
    metadata: dict[str, Any] | None
    on_max_concurrency: OnMaxConcurrencyCallback | None
    on_sampling_error: OnSamplingErrorCallback | None
    on_error: OnErrorCallback | None


class _CallbackSink:
    def __init__(self, callback: SinkCallback) -> None:
        self.callback = callback

    async def submit(
        self,
        *,
        results: Sequence[EvaluationResult],
        failures: Sequence[EvaluatorFailure],
        context: EvaluatorContext,
        span_reference: Any | None,
    ) -> None:
        _ = span_reference
        result = self.callback(results, failures, context)
        if inspect.isawaitable(result):
            await result


EVALUATION_DISABLED: ContextVar[bool] = ContextVar('_evaluation_disabled', default=False)

_background_lock = threading.Lock()
_background_tasks: set[asyncio.Task[Any]] = set()
_background_events: set[anyio.Event] = set()
_background_threads: set[threading.Thread] = set()


def _remove_background_task(task: asyncio.Task[Any]) -> None:
    with _background_lock:
        _background_tasks.discard(task)


def dispatch_async(coro: Coroutine[Any, Any, None]) -> None:
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
        loop = asyncio.get_running_loop()
        task = loop.create_task(coro)
        with _background_lock:
            _background_tasks.add(task)
        task.add_done_callback(_remove_background_task)


def dispatch_in_background_thread(coro: Coroutine[Any, Any, None]) -> None:
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


def _resolve_sample_rate_field(
    online_eval: OnlineEvaluatorLike,
    config: OnlineEvalConfigLike,
) -> float | Callable[[Any], float | bool]:
    if online_eval.sample_rate is None:
        return config.default_sample_rate
    return online_eval.sample_rate


def _resolve_sample_rate(
    rate: float | Callable[[Any], float | bool],
    sampling_context: Any,
) -> float | bool:
    if callable(rate):
        return rate(sampling_context)
    return rate


def _should_evaluate(
    rate: float | Callable[[Any], float | bool],
    global_enabled: bool,
    sampling_context: Any,
    sampling_mode: SamplingMode,
) -> bool:
    if not global_enabled:  # pragma: no cover
        return False
    if EVALUATION_DISABLED.get():  # pragma: no cover
        return False

    resolved = _resolve_sample_rate(rate, sampling_context)
    if isinstance(resolved, bool):
        return resolved
    if resolved >= 1.0:
        return True
    if resolved <= 0.0:
        return False

    if sampling_mode == 'correlated':
        return sampling_context.call_seed < resolved
    return random.random() < resolved


def sample_evaluators(
    online_evals: Sequence[OnlineEvaluatorLike],
    config: OnlineEvalConfigLike,
    inputs: Any,
    *,
    build_sampling_context: SamplingContextBuilder,
) -> list[OnlineEvaluatorLike]:
    call_seed = random.random()
    sampled: list[OnlineEvaluatorLike] = []
    for online_eval in online_evals:
        sampling_context = build_sampling_context(online_eval.evaluator, inputs, config.metadata, call_seed)
        try:
            if _should_evaluate(
                _resolve_sample_rate_field(online_eval, config),
                config.enabled,
                sampling_context,
                config.sampling_mode,
            ):
                sampled.append(online_eval)
        except Exception as exc:
            handler = (
                online_eval.on_sampling_error if online_eval.on_sampling_error is not None else config.on_sampling_error
            )
            if handler is not None:
                try:
                    handler(exc, online_eval.evaluator)
                except Exception:
                    pass
            else:
                raise
    return sampled


def _resolve_sinks(
    evaluator_sink: Any,
    default_sink: Any,
) -> list[EvaluationSink]:
    raw = evaluator_sink if evaluator_sink is not None else default_sink
    if raw is None:
        return []
    return _normalize_sinks(cast(EvaluationSink | Sequence[EvaluationSink | SinkCallback] | SinkCallback, raw))


def _normalize_sinks(
    sink: EvaluationSink | Sequence[EvaluationSink | SinkCallback] | SinkCallback,
) -> list[EvaluationSink]:
    if isinstance(sink, EvaluationSink):
        return [sink]
    if callable(sink):
        return [_CallbackSink(sink)]
    return [_normalize_single_sink(single_sink) for single_sink in sink]


def _normalize_single_sink(sink: EvaluationSink | SinkCallback) -> EvaluationSink:
    if isinstance(sink, EvaluationSink):
        return sink
    return _CallbackSink(sink)


async def _call_on_error(
    on_error: OnErrorCallback | None,
    exc: Exception,
    context: EvaluatorContext,
    evaluator: Evaluator,
    location: OnErrorLocation,
) -> None:
    if on_error is None:
        return
    try:
        result = on_error(exc, context, evaluator, location)
        if inspect.isawaitable(result):
            await result
    except Exception:
        pass


async def _submit_to_sink(
    sink: EvaluationSink,
    results: Sequence[EvaluationResult],
    failures: Sequence[EvaluatorFailure],
    context: EvaluatorContext,
    span_reference: Any | None,
    on_error: OnErrorCallback | None,
    evaluator: Evaluator,
) -> None:
    try:
        await sink.submit(results=results, failures=failures, context=context, span_reference=span_reference)
    except Exception as exc:
        await _call_on_error(on_error, exc, context, evaluator, 'sink')


async def _dispatch_single_evaluator(
    online_eval: OnlineEvaluatorLike,
    context: EvaluatorContext,
    span_reference: Any | None,
    sinks: list[EvaluationSink],
    on_max_concurrency: OnMaxConcurrencyCallback | None,
    on_error: OnErrorCallback | None,
) -> None:
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

        async with anyio.create_task_group() as task_group:
            for sink in sinks:
                task_group.start_soon(
                    _submit_to_sink,
                    sink,
                    results,
                    failures,
                    context,
                    span_reference,
                    on_error,
                    evaluator,
                )
    finally:
        online_eval.semaphore.release()


async def dispatch_evaluators(
    online_evaluators: Sequence[OnlineEvaluatorLike],
    context: EvaluatorContext,
    span_reference: Any | None,
    config: OnlineEvalConfigLike,
) -> None:
    async with anyio.create_task_group() as task_group:
        for online_eval in online_evaluators:
            sinks = _resolve_sinks(online_eval.sink, config.default_sink)
            if not sinks:
                continue
            on_max_concurrency = (
                online_eval.on_max_concurrency
                if online_eval.on_max_concurrency is not None
                else config.on_max_concurrency
            )
            on_error = online_eval.on_error if online_eval.on_error is not None else config.on_error
            task_group.start_soon(
                _dispatch_single_evaluator,
                online_eval,
                context,
                span_reference,
                sinks,
                on_max_concurrency,
                on_error,
            )


async def wait_for_evaluations(*, timeout: float = 30.0) -> None:
    with _background_lock:
        tasks_snapshot = list(_background_tasks)
        events_snapshot = list(_background_events)
        threads_snapshot = list(_background_threads)

    for task in tasks_snapshot:
        try:
            await task
        except BaseException:  # pragma: no cover
            pass

    for event in events_snapshot:
        await event.wait()  # pragma: no cover

    if threads_snapshot:

        def _join_threads() -> None:
            for thread in threads_snapshot:
                thread.join(timeout=timeout)
                if thread.is_alive():  # pragma: no cover
                    warnings.warn(f'Background evaluation thread did not complete within {timeout:.1f}s timeout')

        await run_sync(_join_threads)
