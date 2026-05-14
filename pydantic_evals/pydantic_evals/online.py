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
from collections.abc import Awaitable, Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import anyio
from opentelemetry import trace
from typing_extensions import LiteralString, ParamSpec, TypeVar

from . import _online as _online_internal, _task_run
from ._online import CallbackSink, EvaluationSink, SinkPayload
from ._utils import UNSET, Unset, logfire_span
from .evaluators._run_evaluator import run_evaluator
from .evaluators.context import EvaluatorContext
from .evaluators.evaluator import EvaluationResult, Evaluator, EvaluatorFailure

try:
    import logfire as _logfire  # pyright: ignore[reportUnusedImport]  # noqa: F401

    _LOGFIRE_INSTALLED = True
except ImportError:  # pragma: lax no cover
    _LOGFIRE_INSTALLED = False  # pyright: ignore[reportConstantRedefinition]

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
    'SinkPayload',
    'SpanReference',
    'configure',
    'disable_evaluation',
    'evaluate',
    'run_evaluators',
    'wait_for_evaluations',
)


OnErrorLocation = Literal['sink', 'on_max_concurrency']
"""The location within the online evaluation pipeline where an error occurred.

- `'sink'` — something went wrong delivering results downstream. This is most often
  an exception raised by a registered [`EvaluationSink.submit`][pydantic_evals.online.EvaluationSink.submit],
  but it's also used as a catch-all for failures in the default OTel event emission
  path (which is rare in practice; the OTel SDK rarely raises during `emit()`).
- `'on_max_concurrency'` — the evaluator's `on_max_concurrency` callback itself raised
  while being notified about a dropped evaluation.
"""

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


@dataclass(kw_only=True)
class OnlineEvaluator:
    """Wraps an `Evaluator` with per-evaluator online configuration.

    Different evaluators often need different settings — a cheap heuristic should
    run on 100% of traffic while an expensive LLM judge might run on only 1%.
    """

    evaluator: Evaluator
    """The evaluator to run.

    To version an evaluator, set `evaluator_version` as a class attribute on the
    `Evaluator` subclass itself (see `Evaluator` docstring). The framework reads it
    via `getattr` at dispatch time and propagates it to sinks alongside each result.
    """
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
    """Override additional sink(s) for this evaluator. If `None`, the config's
    `default_sink` is used.

    Sinks are *additive* to the default OTel event emission — not replacements.
    See [`EvaluationSink`][pydantic_evals.online.EvaluationSink]."""

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
    (see [`OnErrorLocation`][pydantic_evals.online.OnErrorLocation]). Can be sync or async.
    `'sink'` covers both custom sink failures and the rarer default OTel event emission
    failures — the value is intentionally broad.
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


_ExtractedArgs = Literal[False, True] | tuple[str, ...]
"""Resolved `extract_args` setting: `False` = record none, `True` = record all,
or a tuple of explicit argument names to record."""


def _resolve_extract_args(
    func: Callable[..., Any],
    sig: inspect.Signature,
    extract_args: bool | Iterable[str],
) -> _ExtractedArgs:
    """Normalise `extract_args` and validate any explicit argument names."""
    if extract_args is False or extract_args is True:
        return extract_args
    if isinstance(extract_args, str):
        names: tuple[str, ...] = (extract_args,)
    else:
        names = tuple(extract_args)
    if not names:
        return False
    unknown = [name for name in names if name not in sig.parameters]
    if unknown:
        raise ValueError(f'extract_args references parameters not in {func.__qualname__}: {sorted(unknown)}')
    return names


def _select_recorded_inputs(
    inputs: dict[str, Any],
    extract_args: _ExtractedArgs,
) -> dict[str, Any] | None:
    """Return the subset of inputs to record on the span, or `None` if disabled."""
    if extract_args is False:
        return None
    if extract_args is True:
        return inputs
    return {name: inputs[name] for name in extract_args if name in inputs}


def _default_call_span_name(func: Callable[..., Any]) -> str:
    """Build a default span name/msg_template following `@logfire.instrument`'s convention."""
    qualname = getattr(func, '__qualname__', getattr(func, '__name__', repr(func)))
    module = inspect.getmodule(func)
    module_name = getattr(module, '__name__', None)
    if module_name:
        return f'Calling {module_name}.{qualname}'
    return f'Calling {qualname}'  # pragma: no cover


@contextmanager
def _open_call_span(
    msg_template: str,
    span_name: str | None,
    recorded_inputs: dict[str, Any] | None,
) -> Iterator[Any]:
    """Open the span that represents the decorated function call.

    When logfire is installed, uses `logfire.span` so argument and return
    attributes get JSON-schema serialization. Otherwise falls back to a raw
    OTel span via the configured tracer provider — preserving span parenting
    for evaluator events even when logfire is not available.
    """
    if _LOGFIRE_INSTALLED:
        attrs = recorded_inputs or {}
        with logfire_span(msg_template, _span_name=span_name, **attrs) as span:
            yield span
    else:
        tracer = trace.get_tracer('pydantic-evals')
        with tracer.start_as_current_span(span_name or msg_template) as span:
            yield span


@dataclass(kw_only=True)
class OnlineEvalConfig:
    """Holds cross-evaluator defaults for online evaluation.

    Create instances for different evaluation configurations, or use the global
    `DEFAULT_CONFIG` via the module-level `evaluate()` and `configure()` functions.
    """

    default_sink: EvaluationSink | Sequence[EvaluationSink | SinkCallback] | SinkCallback | None = None
    """Additional sink(s) to receive results, for evaluators that don't specify their own.

    Sinks run *in addition to* the default `gen_ai.evaluation.result` OTel event
    emission — they are the escape hatch for custom destinations (in-memory test
    capture, fan-out to Slack/DB, non-OTel backends). To disable OTel emission
    itself, set [`emit_otel_events=False`][pydantic_evals.online.OnlineEvalConfig.emit_otel_events].
    """
    default_sample_rate: float | Callable[[SamplingContext], float | bool] = 1.0
    """Default sample rate for evaluators that don't specify their own."""
    emit_otel_events: bool = True
    """Whether to emit `gen_ai.evaluation.result` OTel events for every evaluator run.

    When `True` (the default), dispatch emits one OTel log event per `EvaluationResult`
    or `EvaluatorFailure`, following the [OTel GenAI evaluation semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/#event-gen_aievaluationresult).
    If no OTel SDK is configured in the process, emission is a cheap no-op.

    Set to `False` to disable — useful for tests that want to assert on a custom
    sink alone, or in environments where OTel emission is undesirable. Custom
    sinks registered via `default_sink` still run regardless of this flag. With
    `emit_otel_events=False` AND no sinks configured, dispatch short-circuits
    entirely (the evaluator never runs) since results would have nowhere to go.
    """
    include_baggage: bool = True
    """Whether to copy OTel baggage entries onto every emitted evaluation event.

    When `True` (the default), each emitted `gen_ai.evaluation.result` event also
    carries the keys present in the current OTel baggage as attributes — useful
    for propagating tenant/user/request identifiers from the calling context.
    Standard `gen_ai.*` and `error.type` attributes always win on conflict, so
    baggage cannot accidentally overwrite the semantic-convention attributes.

    Set to `False` to skip the baggage snapshot per event.
    """
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
    (see [`OnErrorLocation`][pydantic_evals.online.OnErrorLocation]). Can be sync or async.
    `'sink'` covers both custom sink failures and the rarer default OTel event emission
    failures — the value is intentionally broad.
    If `None` (the default), exceptions are silently suppressed.
    Per-evaluator `OnlineEvaluator.on_error` overrides this default.
    """

    def evaluate(
        self,
        *evaluators: Evaluator | OnlineEvaluator,
        target: str | None = None,
        msg_template: LiteralString | None = None,
        span_name: str | None = None,
        extract_args: bool | Iterable[str] = False,
        record_return: bool = False,
    ) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
        """Decorator to attach online evaluators to a function.

        Each decorated call opens a dedicated span representing the function
        invocation — evaluator events are parented to this span, and the span
        itself appears in the user's configured OTel/logfire traces.

        Bare `Evaluator` instances are auto-wrapped in `OnlineEvaluator` at decoration time
        (so concurrency semaphores are shared across calls). Their `sample_rate` defaults to
        `None`, which resolves to the config's `default_sample_rate` at each call — so
        changes to the config after decoration take effect.

        To version an evaluator, set `evaluator_version` on the `Evaluator` subclass
        itself — the framework reads it at dispatch time and records it on every
        [`EvaluationResult`][pydantic_evals.evaluators.EvaluationResult] and
        [`EvaluatorFailure`][pydantic_evals.evaluators.EvaluatorFailure] the evaluator emits:

        ```python
        from dataclasses import dataclass

        from pydantic_evals.evaluators import Evaluator, EvaluatorContext
        from pydantic_evals.online import evaluate


        @dataclass
        class Tone(Evaluator):
            evaluator_version = 'v2'

            def evaluate(self, ctx: EvaluatorContext) -> str:
                return 'neutral'


        @evaluate(Tone())
        async def summarize(text: str) -> str:
            return text
        ```

        Args:
            *evaluators: Evaluators to attach. Can be `Evaluator` or `OnlineEvaluator` instances.
            target: Name of the thing being evaluated. Written to sinks and emitted
                OTel events as `gen_ai.evaluation.target`. Defaults to the decorated
                function's `__name__` when omitted.
            msg_template: Template for the call span's message. Defaults to
                `"Calling {module}.{qualname}"` like `@logfire.instrument`.
                When logfire is installed, `{arg=}`-style placeholders in the
                template are formatted against the function's arguments.
            span_name: Override for the call span's name. Defaults to `msg_template`.
            extract_args: Whether to record function arguments as span attributes.
                `False` (default) records nothing; `True` records all bound arguments;
                an iterable of names records only those arguments. Requires logfire
                to be installed so arguments are serialised with their JSON schema —
                raises `RuntimeError` at decoration time otherwise.
            record_return: Whether to record the function's return value as a `return`
                span attribute. Requires logfire for the same reason as `extract_args`.

        Returns:
            A decorator that wraps the function with online evaluation.
        """
        online_evals = [e if isinstance(e, OnlineEvaluator) else OnlineEvaluator(evaluator=e) for e in evaluators]

        if (extract_args or record_return) and not _LOGFIRE_INSTALLED:
            raise RuntimeError(
                'extract_args and record_return require logfire to be installed for argument and '
                'return-value serialization. Install `pydantic-evals[logfire]` (or disable both '
                'options) to use them.'
            )

        # These options are intentionally decorator-only for now so they stay close to
        # `@logfire.instrument`'s shape; we can lift them to `OnlineEvalConfig` as
        # defaults if users ask for it.

        def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
            resolved_target = target if target is not None else func.__name__
            sig = inspect.signature(func)
            resolved_extract_args = _resolve_extract_args(func, sig, extract_args)
            resolved_msg_template = msg_template if msg_template is not None else _default_call_span_name(func)
            call_span = _CallSpanSpec(
                msg_template=resolved_msg_template,
                span_name=span_name,
                extract_args=resolved_extract_args,
                record_return=record_return,
            )
            if inspect.iscoroutinefunction(func):
                # ParamSpec can't distinguish async from sync return types — _wrap_async returns
                # Callable[_P, Awaitable[_R]] but the decorator signature expects Callable[_P, _R]
                return _wrap_async(func, sig, online_evals, self, resolved_target, call_span)  # pyright: ignore[reportReturnType]
            else:
                return _wrap_sync(func, sig, online_evals, self, resolved_target, call_span)

        return decorator

    def should_evaluate(self) -> bool:
        """Whether evaluators with this config should run, based on the current settings and context."""
        return self.enabled and not _EVALUATION_DISABLED.get() and _task_run.CURRENT_TASK_RUN.get() is None


@dataclass(kw_only=True, frozen=True)
class _CallSpanSpec:
    """Resolved configuration for the span that wraps a decorated call."""

    msg_template: str
    span_name: str | None
    extract_args: _ExtractedArgs
    record_return: bool


def _wrap_async(
    func: Callable[_P, Awaitable[_R]],
    sig: inspect.Signature,
    online_evals: list[OnlineEvaluator],
    config: OnlineEvalConfig,
    target: str,
    call_span: _CallSpanSpec,
) -> Callable[_P, Awaitable[_R]]:
    """Wrap an async function with online evaluation."""

    @functools.wraps(func)
    async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        # If evaluation is globally disabled, or we're already inside an evaluation
        # context (e.g. Dataset.evaluate), just run the function
        if not config.should_evaluate():
            return await func(*args, **kwargs)

        # Capture inputs early so sample_rate callables can use them
        inputs = _capture_inputs(sig, args, kwargs)

        # Determine which evaluators are sampled (before running the function)
        sampled = _online_internal.sample_evaluators(
            online_evals,
            config,
            inputs,
        )
        if not sampled:
            return await func(*args, **kwargs)

        recorded_inputs = _select_recorded_inputs(inputs, call_span.extract_args)

        # Run the function with span tree capture and attribute/metric tracking
        with (
            _open_call_span(call_span.msg_template, call_span.span_name, recorded_inputs) as span,
            _task_run.run_task() as get_eval_context_kwargs,
        ):
            result = await func(*args, **kwargs)
            if call_span.record_return:
                # Swallow attribute-set failures so an exotic return value (e.g. one
                # whose repr raises during logfire's JSON-schema serialisation) can't
                # mask the function's real return. `record_return=True` is opt-in for
                # observability, not a contract to fail the call.
                try:
                    span.set_attribute('return', result)
                except Exception:  # pragma: no cover - defensive
                    pass

        # Build context
        metadata = dict(config.metadata) if config.metadata is not None else None
        context = EvaluatorContext(
            name=None,
            inputs=inputs,
            output=result,
            expected_output=None,
            metadata=metadata,
            **get_eval_context_kwargs(),
        )

        # Extract span reference from the logfire span
        span_reference = _extract_span_reference(span)

        # Dispatch evaluators on the caller's event loop — preserves ContextVars
        _online_internal.dispatch_async(
            _online_internal.dispatch_evaluators(sampled, context, span_reference, target, config)
        )

        return result

    return wrapper


def _wrap_sync(
    func: Callable[_P, _R],
    sig: inspect.Signature,
    online_evals: list[OnlineEvaluator],
    config: OnlineEvalConfig,
    target: str,
    call_span: _CallSpanSpec,
) -> Callable[_P, _R]:
    """Wrap a sync function with online evaluation."""

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        # If evaluation is globally disabled, or we're already inside an evaluation
        # context (e.g. Dataset.evaluate), just run the function
        if not config.should_evaluate():
            return func(*args, **kwargs)

        # Capture inputs early so sample_rate callables can use them
        inputs = _capture_inputs(sig, args, kwargs)

        # Determine which evaluators are sampled
        sampled = _online_internal.sample_evaluators(
            online_evals,
            config,
            inputs,
        )
        if not sampled:
            return func(*args, **kwargs)

        recorded_inputs = _select_recorded_inputs(inputs, call_span.extract_args)

        # Run the function with span tree capture and attribute/metric tracking
        with (
            _open_call_span(call_span.msg_template, call_span.span_name, recorded_inputs) as span,
            _task_run.run_task() as get_eval_context_kwargs,
        ):
            result = func(*args, **kwargs)
            if call_span.record_return:
                # Swallow attribute-set failures so an exotic return value (e.g. one
                # whose repr raises during logfire's JSON-schema serialisation) can't
                # mask the function's real return. `record_return=True` is opt-in for
                # observability, not a contract to fail the call.
                try:
                    span.set_attribute('return', result)
                except Exception:  # pragma: no cover - defensive
                    pass

        # Build context
        metadata = dict(config.metadata) if config.metadata is not None else None
        context = EvaluatorContext(
            name=None,
            inputs=inputs,
            output=result,
            expected_output=None,
            metadata=metadata,
            **get_eval_context_kwargs(),
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

        coro = _online_internal.dispatch_evaluators(sampled, context, span_reference, target, config)
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


def evaluate(
    *evaluators: Evaluator | OnlineEvaluator,
    target: str | None = None,
    msg_template: LiteralString | None = None,
    span_name: str | None = None,
    extract_args: bool | Iterable[str] = False,
    record_return: bool = False,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Decorator to attach online evaluators to a function using the global default config.

    Equivalent to `DEFAULT_CONFIG.evaluate(...)`.

    Args:
        *evaluators: Evaluators to attach. Can be `Evaluator` or `OnlineEvaluator` instances.
        target: Name of the thing being evaluated. Written to sinks and emitted
            OTel events as `gen_ai.evaluation.target`. Defaults to the decorated
            function's `__name__` when omitted.
        msg_template: Template for the call span's message. Defaults to
            `"Calling {module}.{qualname}"` like `@logfire.instrument`.
        span_name: Override for the call span's name. Defaults to `msg_template`.
        extract_args: Whether to record function arguments as span attributes.
            `False` (default) records nothing; `True` records all bound arguments;
            an iterable of names records only those arguments. Requires logfire
            to be installed — raises `RuntimeError` at decoration time otherwise.
        record_return: Whether to record the function's return value as a `return`
            span attribute. Requires logfire for the same reason as `extract_args`.

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
    return DEFAULT_CONFIG.evaluate(
        *evaluators,
        target=target,
        msg_template=msg_template,
        span_name=span_name,
        extract_args=extract_args,
        record_return=record_return,
    )


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
    emit_otel_events: bool | Unset = UNSET,
    include_baggage: bool | Unset = UNSET,
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
        emit_otel_events: Whether to emit `gen_ai.evaluation.result` OTel events.
        include_baggage: Whether to copy current OTel baggage onto every emitted event.
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
    if not isinstance(emit_otel_events, Unset):
        DEFAULT_CONFIG.emit_otel_events = emit_otel_events
    if not isinstance(include_baggage, Unset):
        DEFAULT_CONFIG.include_baggage = include_baggage


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
