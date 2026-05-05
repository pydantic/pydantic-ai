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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, runtime_checkable

import anyio
import sniffio
from anyio.to_thread import run_sync
from opentelemetry import context as otel_context

from ._otel_emit import build_parent_context, emit_otel_events
from .evaluators._run_evaluator import run_evaluator
from .evaluators.context import EvaluatorContext
from .evaluators.evaluator import EvaluationResult, Evaluator, EvaluatorFailure

if TYPE_CHECKING:
    # Imported only for type annotations; `online` imports from this module at runtime.
    from .online import OnlineEvalConfig, OnlineEvaluator, SpanReference

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


@dataclass(kw_only=True, frozen=True)
class SinkPayload:
    """Container passed to [`EvaluationSink.submit`][pydantic_evals.online.EvaluationSink.submit].

    !!! warning "Do not instantiate directly"
        `SinkPayload` is constructed internally by pydantic-evals. We reserve the right
        to add fields in any release — if you build your own instances, a future version
        may break your code. Sink implementations should accept the payload as-is and read
        only the fields they need.
    """

    results: Sequence[EvaluationResult]
    """Evaluation results from the evaluator run."""

    failures: Sequence[EvaluatorFailure]
    """Failures from the evaluator run if it raised."""

    context: EvaluatorContext
    """The full evaluator context for the function call."""

    span_reference: SpanReference | None
    """Reference to the OTel span for the function call, if available."""

    target: str
    """Identifies the function/agent being evaluated, supplied by the
    `@evaluate` decorator (defaults resolved at decoration time)."""


@runtime_checkable
class EvaluationSink(Protocol):
    """Protocol for **additional** evaluation result destinations.

    By default, online evaluation emits `gen_ai.evaluation.result` OTel events
    for every evaluator run — no sink registration required. Sinks are the
    escape hatch for custom handling *in addition to* OTel emission: in-memory
    test capture, fan-out to Slack/DB, non-OTel backends, alerting pipelines,
    etc. See [`OnlineEvalConfig.default_sink`][pydantic_evals.online.OnlineEvalConfig.default_sink].

    To disable the default OTel emission (e.g. in tests that only want to
    assert on a custom sink), set
    [`emit_otel_events=False`][pydantic_evals.online.OnlineEvalConfig.emit_otel_events]
    on the config.
    """

    async def submit(self, payload: SinkPayload) -> None:
        """Submit evaluation results to the sink.

        The payload may include results from one or more evaluators that ran for
        a given function call — when multiple evaluators share this sink, their
        results are batched into a single `submit()` call. Each result carries
        enough metadata (name, evaluator version, source) to be attributed
        downstream; the exact batching behavior is an implementation detail and
        may change.

        Args:
            payload: A [`SinkPayload`][pydantic_evals.online.SinkPayload] bundling
                results, failures, context, span reference, and target. Sinks
                should read only the fields they need; new fields may be added
                in future releases.
        """
        ...


class CallbackSink:
    """An `EvaluationSink` that delegates to a user-provided callable.

    The callback receives the results, failures, and context. Other fields on
    the [`SinkPayload`][pydantic_evals.online.SinkPayload] (such as
    `span_reference` and `target`) are not passed — use a custom
    `EvaluationSink` implementation if you need them.
    """

    def __init__(self, callback: SinkCallback) -> None:
        self.callback = callback

    async def submit(self, payload: SinkPayload) -> None:
        result = self.callback(payload.results, payload.failures, payload.context)
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

    thread = threading.Thread(target=_thread_target, daemon=False)
    with _background_lock:
        _background_threads.add(thread)
    try:
        thread.start()
    except Exception:  # pragma: no cover
        with _background_lock:
            _background_threads.discard(thread)


def _resolve_sample_rate_field(
    online_eval: OnlineEvaluator,
    config: OnlineEvalConfig,
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
    sampling_context: Any,
    sampling_mode: SamplingMode,
) -> bool:
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
    online_evals: Sequence[OnlineEvaluator],
    config: OnlineEvalConfig,
    inputs: Any,
) -> list[OnlineEvaluator]:
    from .online import SamplingContext

    call_seed = random.random()
    sampled: list[OnlineEvaluator] = []
    for online_eval in online_evals:
        sampling_context = SamplingContext(
            evaluator=online_eval.evaluator,
            inputs=inputs,
            metadata=config.metadata,
            call_seed=call_seed,
        )
        try:
            if _should_evaluate(
                _resolve_sample_rate_field(online_eval, config),
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


def _normalize_sinks(
    sink: EvaluationSink | Sequence[EvaluationSink | SinkCallback] | SinkCallback,
) -> list[EvaluationSink]:
    if isinstance(sink, EvaluationSink):
        return [_ensure_payload_compat(sink)]
    if callable(sink):
        return [CallbackSink(sink)]
    return [_normalize_single_sink(single_sink) for single_sink in sink]


def _normalize_single_sink(sink: EvaluationSink | SinkCallback) -> EvaluationSink:
    if isinstance(sink, EvaluationSink):
        return _ensure_payload_compat(sink)
    return CallbackSink(sink)


# ---------------------------------------------------------------------------
# Back-compat shim for EvaluationSink.submit signature change.
#
# The original signature was
#   async def submit(*, results, failures, context, span_reference)
# and is now
#   async def submit(payload: SinkPayload)
# Sinks that still use the old kwargs are detected via signature introspection
# and wrapped in `_LegacyKwargsShim`, which unpacks the payload for them.
#
# TODO(v2): delete this whole section. In v2, `EvaluationSink.submit` will
# require `(payload: SinkPayload)` unconditionally — drop `_is_legacy_submit`,
# `_ensure_payload_compat`, `_LegacyKwargsShim`, and `_warned_legacy_sink_ids`,
# and inline the return values in `_normalize_sinks` / `_normalize_single_sink`.
# ---------------------------------------------------------------------------
_warned_legacy_sink_ids: set[int] = set()


def _is_legacy_submit(sink: EvaluationSink) -> bool:
    """Return whether a sink's `submit` uses the pre-`SinkPayload` kwargs signature.

    The bound signature of a modern sink (`async def submit(self, payload)`)
    has exactly one non-`**kwargs` parameter. Everything else — 4-kwarg legacy,
    `**kwargs`-only, zero-arg — gets the shim. Using arity rather than
    parameter names means a modern sink that happens to name its argument
    `results` still resolves correctly, and an unusual keyword-only
    `submit(self, *, payload)` is routed to the modern call path rather than
    being shimmed and emitting a spurious deprecation warning.
    """
    try:
        params = list(inspect.signature(sink.submit).parameters.values())
    except (TypeError, ValueError):  # pragma: no cover — some C-implemented callables
        return False  # can't introspect — assume modern signature, don't warn
    non_var_kwargs = [p for p in params if p.kind is not inspect.Parameter.VAR_KEYWORD]
    return len(non_var_kwargs) != 1


class _LegacyKwargsShim:
    """Wraps a legacy sink using the pre-`SinkPayload` kwargs signature.

    Unpacks the `SinkPayload` into `(results, failures, context, span_reference)`
    kwargs and delegates.
    """

    def __init__(self, inner: EvaluationSink) -> None:
        self._inner = inner

    async def submit(self, payload: SinkPayload) -> None:
        # Cast to Any: the shim exists specifically to call sinks whose real signatures
        # predate the `payload: SinkPayload` change, so pyright's protocol-based view
        # of `_inner.submit` doesn't match the kwargs we need to pass.
        await cast(Any, self._inner).submit(
            results=payload.results,
            failures=payload.failures,
            context=payload.context,
            span_reference=payload.span_reference,
        )


def _ensure_payload_compat(sink: EvaluationSink) -> EvaluationSink:
    """If `sink.submit` uses the pre-`SinkPayload` signature, wrap it and warn once per class."""
    if not _is_legacy_submit(sink):
        return sink
    sink_cls = type(sink)
    cls_id = id(sink_cls)
    if cls_id not in _warned_legacy_sink_ids:
        _warned_legacy_sink_ids.add(cls_id)
        warnings.warn(
            f'{sink_cls.__module__}.{sink_cls.__qualname__}.submit() uses the deprecated '
            'kwargs signature (results, failures, context, span_reference). Update it to '
            '`async def submit(self, payload: SinkPayload)` — this compatibility shim will '
            'be removed in pydantic-evals v2.',
            DeprecationWarning,
            stacklevel=4,
        )
    return _LegacyKwargsShim(sink)


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


@dataclass
class _SinkGroup:
    """Evaluators that share a sink source, batched into one `submit` call per sink.

    `sinks` is the normalized list for this group's raw source; all member
    evaluators route through the same sinks. `outcomes` is populated as each
    evaluator finishes — after all evaluators in the group complete, their
    results are flattened into a single `SinkPayload`.
    """

    sinks: list[EvaluationSink]
    evaluators: list[OnlineEvaluator]
    outcomes: list[tuple[OnlineEvaluator, Sequence[EvaluationResult], Sequence[EvaluatorFailure]]]


async def _run_and_collect(
    online_eval: OnlineEvaluator,
    context: EvaluatorContext,
    span_reference: SpanReference | None,
    target: str,
    config: OnlineEvalConfig,
    group: _SinkGroup,
) -> None:
    """Run a single evaluator, emit its OTel events, and stash the outcome on the group.

    Semaphore acquisition, `on_max_concurrency`, and OTel emission remain per
    evaluator. Sink submission is deferred to the group-level batch.
    """
    evaluator = online_eval.evaluator
    on_error = online_eval.on_error if online_eval.on_error is not None else config.on_error
    on_max_concurrency = (
        online_eval.on_max_concurrency if online_eval.on_max_concurrency is not None else config.on_max_concurrency
    )

    if not online_eval.semaphore.acquire(blocking=False):
        if on_max_concurrency is not None:
            try:
                result = on_max_concurrency(context)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                await _call_on_error(on_error, exc, context, evaluator, 'on_max_concurrency')
        return

    # Pair the acquire with a try/finally that always releases. The parent-context
    # setup is inside the `try` so a stray failure in `otel_context.attach` (or a
    # pathological `build_parent_context`) can't leak the semaphore slot.
    parent_token = None
    try:
        # Attach the call's span as the current OTel parent for the whole evaluator
        # run. This nests the `evaluator: {evaluator_name}` span (created inside
        # `run_evaluator`) under the decorated function's call span, and parents the
        # emitted events the same way without each emit having to attach independently.
        parent_ctx = build_parent_context(span_reference)
        parent_token = otel_context.attach(parent_ctx) if parent_ctx is not None else None
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
        if config.emit_otel_events:
            try:
                emit_otel_events(
                    results=results,
                    failures=failures,
                    target=target,
                    include_baggage=config.include_baggage,
                )
            except Exception as exc:  # pragma: no cover - defensive
                # Report OTel emission failures under the `'sink'` location: it's the
                # catch-all for "something went wrong delivering results downstream",
                # which default-sink users and OTel users can both reason about. See
                # `OnErrorLocation` docstring in `online.py` for the contract.
                await _call_on_error(on_error, exc, context, evaluator, 'sink')

        group.outcomes.append((online_eval, results, failures))
    finally:
        if parent_token is not None:
            otel_context.detach(parent_token)
        online_eval.semaphore.release()


async def _submit_group_to_sink(
    sink: EvaluationSink,
    payload: SinkPayload,
    group: _SinkGroup,
    config: OnlineEvalConfig,
) -> None:
    """Submit a batched payload to one sink, routing errors to each evaluator's on_error.

    If the sink raises, every on_error handler represented in the group fires
    once (deduped by handler identity) — the common-case single-evaluator group
    still routes exactly like pre-batching.
    """
    try:
        await sink.submit(payload)
    except Exception as exc:
        seen: set[int] = set()
        for online_eval, _, _ in group.outcomes:
            handler = online_eval.on_error if online_eval.on_error is not None else config.on_error
            if handler is None:
                continue
            hid = id(handler)
            if hid in seen:
                continue
            seen.add(hid)
            await _call_on_error(handler, exc, payload.context, online_eval.evaluator, 'sink')


async def dispatch_evaluators(
    online_evaluators: Sequence[OnlineEvaluator],
    context: EvaluatorContext,
    span_reference: SpanReference | None,
    target: str,
    config: OnlineEvalConfig,
) -> None:
    # Group evaluators by raw sink source so evaluators sharing a source land a
    # single batched `submit` call per sink. Keyed by `id()` of the raw source
    # (default_sink or per-evaluator `sink=` override) so user-visible identity
    # drives the grouping without silently coalescing distinct sink instances.
    groups: dict[int, _SinkGroup] = {}
    for online_eval in online_evaluators:
        raw = online_eval.sink if online_eval.sink is not None else config.default_sink
        key = id(raw)
        group = groups.get(key)
        if group is None:
            sinks = _normalize_sinks(raw) if raw is not None else []
            # Skip evaluators whose results would have nowhere to go: both OTel
            # emission disabled and no sinks attached. Avoids spending semaphore
            # slots and evaluator work to produce results we'd immediately drop.
            if not config.emit_otel_events and not sinks:
                continue
            group = _SinkGroup(sinks=sinks, evaluators=[], outcomes=[])
            groups[key] = group
        group.evaluators.append(online_eval)

    if not groups:
        return

    # Phase 1: run every evaluator in parallel, collecting outcomes into their group.
    async with anyio.create_task_group() as eval_tg:
        for group in groups.values():
            for online_eval in group.evaluators:
                eval_tg.start_soon(
                    functools.partial(
                        _run_and_collect,
                        online_eval,
                        context,
                        span_reference,
                        target,
                        config,
                        group,
                    )
                )

    # Phase 2: one batched `submit` call per (group, sink), fanned out in parallel.
    async with anyio.create_task_group() as sink_tg:
        for group in groups.values():
            if not group.outcomes or not group.sinks:
                continue
            batched_results: list[EvaluationResult] = []
            batched_failures: list[EvaluatorFailure] = []
            for _, r, f in group.outcomes:
                batched_results.extend(r)
                batched_failures.extend(f)
            if not batched_results and not batched_failures:
                continue
            payload = SinkPayload(
                results=batched_results,
                failures=batched_failures,
                context=context,
                span_reference=span_reference,
                target=target,
            )
            for sink in group.sinks:
                sink_tg.start_soon(
                    functools.partial(
                        _submit_group_to_sink,
                        sink,
                        payload,
                        group,
                        config,
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
