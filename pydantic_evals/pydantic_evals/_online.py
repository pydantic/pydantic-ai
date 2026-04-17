"""Private helpers shared by the online-eval decorator and agent capability."""

from __future__ import annotations

import asyncio
import contextvars
import functools
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
from opentelemetry import context as otel_context

from ._otel_emit import build_parent_context, emit_otel_events
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
        target: str,
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
    emit_otel_events: bool
    include_baggage: bool
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
        target: str,
    ) -> None:
        _ = span_reference, target  # custom sink only
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
        return [_ensure_target_compat(sink)]
    if callable(sink):
        return [_CallbackSink(sink)]
    return [_normalize_single_sink(single_sink) for single_sink in sink]


def _normalize_single_sink(sink: EvaluationSink | SinkCallback) -> EvaluationSink:
    if isinstance(sink, EvaluationSink):
        return _ensure_target_compat(sink)
    return _CallbackSink(sink)


# ---------------------------------------------------------------------------
# Back-compat shim for EvaluationSink.submit added `target: str` kwarg.
#
# TODO(v2): delete this whole section. In v2, `EvaluationSink.submit` will
# require `target: str` unconditionally — drop `_submit_accepts_target`,
# `_ensure_target_compat`, `_LegacyTargetShim`, and `_warned_legacy_sink_ids`,
# and inline the return values in `_normalize_sinks` / `_normalize_single_sink`.
# ---------------------------------------------------------------------------
_warned_legacy_sink_ids: set[int] = set()


def _submit_accepts_target(sink: EvaluationSink) -> bool:
    """Check whether a sink's `submit` accepts the `target` kwarg (directly or via **kwargs)."""
    try:
        params = inspect.signature(sink.submit).parameters
    except (TypeError, ValueError):  # pragma: no cover — some C-implemented callables
        return True  # can't introspect — assume modern signature, don't warn
    if 'target' in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


class _LegacyTargetShim:
    """Wraps a pre-`target` sink, dropping the kwarg before delegating."""

    def __init__(self, inner: EvaluationSink) -> None:
        self._inner = inner

    async def submit(
        self,
        *,
        results: Sequence[EvaluationResult],
        failures: Sequence[EvaluatorFailure],
        context: EvaluatorContext,
        span_reference: Any | None,
        target: str,
    ) -> None:
        _ = target  # dropped for legacy sinks
        # Pyright knows `EvaluationSink.submit` now requires `target`; the whole point
        # of this shim is to call sinks whose real signatures predate that kwarg.
        await self._inner.submit(  # pyright: ignore[reportCallIssue]
            results=results,
            failures=failures,
            context=context,
            span_reference=span_reference,
        )


def _ensure_target_compat(sink: EvaluationSink) -> EvaluationSink:
    """If `sink.submit` predates the `target` kwarg, wrap it and warn once per class."""
    if _submit_accepts_target(sink):
        return sink
    sink_cls = type(sink)
    cls_id = id(sink_cls)
    if cls_id not in _warned_legacy_sink_ids:
        _warned_legacy_sink_ids.add(cls_id)
        warnings.warn(
            f'{sink_cls.__module__}.{sink_cls.__qualname__}.submit() is missing the '
            "'target: str' keyword argument added to EvaluationSink.submit. Update the "
            'signature to accept `target` (or `**kwargs`); this compatibility shim will '
            'be removed in pydantic-evals v2.',
            DeprecationWarning,
            stacklevel=4,
        )
    return _LegacyTargetShim(sink)


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
    target: str,
    on_error: OnErrorCallback | None,
    evaluator: Evaluator,
) -> None:
    try:
        await sink.submit(
            results=results,
            failures=failures,
            context=context,
            span_reference=span_reference,
            target=target,
        )
    except Exception as exc:
        await _call_on_error(on_error, exc, context, evaluator, 'sink')


async def _dispatch_single_evaluator(
    online_eval: OnlineEvaluatorLike,
    context: EvaluatorContext,
    span_reference: Any | None,
    target: str,
    sinks: list[EvaluationSink],
    emit_events: bool,
    include_baggage: bool,
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

    # Attach the call's span as the current OTel parent for the whole evaluator
    # run. This nests the `evaluator: {name}` span (created inside `run_evaluator`)
    # under the decorated function's call span, and parents the emitted events
    # the same way without each emit having to attach independently.
    parent_ctx = build_parent_context(span_reference)
    parent_token = otel_context.attach(parent_ctx) if parent_ctx is not None else None
    try:
        raw_result = await run_evaluator(evaluator, context)

        if isinstance(raw_result, EvaluatorFailure):
            results: Sequence[EvaluationResult] = []
            failures: Sequence[EvaluatorFailure] = [raw_result]
        else:
            results = raw_result
            failures = []

        # Default OTel event emission. Unconditional unless the config opts out.
        # If no OTel SDK is configured in the process, `get_logger()` returns a
        # no-op logger and this is effectively free.
        if emit_events:
            try:
                emit_otel_events(
                    results=results,
                    failures=failures,
                    target=target,
                    include_baggage=include_baggage,
                )
            except Exception as exc:  # pragma: no cover - defensive
                await _call_on_error(on_error, exc, context, evaluator, 'sink')

        async with anyio.create_task_group() as task_group:
            for sink in sinks:
                task_group.start_soon(
                    _submit_to_sink,
                    sink,
                    results,
                    failures,
                    context,
                    span_reference,
                    target,
                    on_error,
                    evaluator,
                )
    finally:
        if parent_token is not None:
            otel_context.detach(parent_token)
        online_eval.semaphore.release()


async def dispatch_evaluators(
    online_evaluators: Sequence[OnlineEvaluatorLike],
    context: EvaluatorContext,
    span_reference: Any | None,
    target: str,
    config: OnlineEvalConfigLike,
) -> None:
    async with anyio.create_task_group() as task_group:
        for online_eval in online_evaluators:
            # Additional user sinks (if any) — OTel event emission happens
            # unconditionally inside `_dispatch_single_evaluator` when
            # `config.emit_otel_events` is True.
            sinks = _resolve_sinks(online_eval.sink, config.default_sink)
            on_max_concurrency = (
                online_eval.on_max_concurrency
                if online_eval.on_max_concurrency is not None
                else config.on_max_concurrency
            )
            on_error = online_eval.on_error if online_eval.on_error is not None else config.on_error
            task_group.start_soon(
                functools.partial(
                    _dispatch_single_evaluator,
                    online_eval,
                    context,
                    span_reference,
                    target,
                    sinks,
                    emit_events=config.emit_otel_events,
                    include_baggage=config.include_baggage,
                    on_max_concurrency=on_max_concurrency,
                    on_error=on_error,
                )
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
