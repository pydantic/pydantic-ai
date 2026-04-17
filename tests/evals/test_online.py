"""Tests for pydantic_evals.online — online evaluation infrastructure."""

from __future__ import annotations as _annotations

import asyncio
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_evals.dataset import increment_eval_metric, set_eval_attribute
    from pydantic_evals.evaluators import EvaluationResult, Evaluator, EvaluatorContext, EvaluatorFailure
    from pydantic_evals.evaluators.evaluator import EvaluatorOutput
    from pydantic_evals.online import (
        DEFAULT_CONFIG,
        CallbackSink,
        OnErrorLocation,
        OnlineEvalConfig,
        OnlineEvaluator,
        SamplingContext,
        SpanReference,
        configure,
        disable_evaluation,
        evaluate,
        run_evaluators,
        wait_for_evaluations,
    )
    from pydantic_evals.otel.span_tree import SpanTree

with try_import() as logfire_import_successful:
    from logfire.testing import CaptureLogfire

pytestmark = pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed')

needs_logfire = pytest.mark.skipif(not logfire_import_successful(), reason='logfire not installed')


if TYPE_CHECKING or imports_successful():

    @dataclass
    class AlwaysTrue(Evaluator):
        """Simple evaluator that always returns True."""

        def evaluate(self, ctx: EvaluatorContext) -> EvaluatorOutput:
            return True

    @dataclass
    class AlwaysFalse(Evaluator):
        """Simple evaluator that always returns False."""

        def evaluate(self, ctx: EvaluatorContext) -> EvaluatorOutput:
            return False

    @dataclass
    class OutputEquals(Evaluator):
        """Evaluator that checks if output equals a value."""

        value: Any

        def evaluate(self, ctx: EvaluatorContext) -> EvaluatorOutput:
            return ctx.output == self.value

    @dataclass
    class FailingEvaluator(Evaluator):
        """Evaluator that always raises an exception."""

        def evaluate(self, ctx: EvaluatorContext) -> EvaluatorOutput:
            raise ValueError('Simulated evaluator failure')

    @dataclass
    class AsyncEvaluator(Evaluator):
        """Async evaluator for testing."""

        async def evaluate(self, ctx: EvaluatorContext) -> EvaluatorOutput:
            await asyncio.sleep(0)
            return True

    @dataclass
    class MultiResultEvaluator(Evaluator):
        """Evaluator that returns multiple results."""

        def evaluate(self, ctx: EvaluatorContext) -> EvaluatorOutput:
            return {'accuracy': True, 'score': 0.95, 'label': 'good'}

    def _make_context(
        *,
        inputs: Any = None,
        output: Any = None,
        expected_output: Any = None,
        metadata: Any = None,
        duration: float = 0.0,
    ) -> EvaluatorContext[Any, Any, Any]:
        """Create an EvaluatorContext for testing."""
        return EvaluatorContext(
            name='test',
            inputs=inputs,
            output=output,
            expected_output=expected_output,
            metadata=metadata,
            duration=duration,
            _span_tree=SpanTree(),
            attributes={},
            metrics={},
        )

    class Collector:
        """Collects sink submissions for test assertions."""

        def __init__(self) -> None:
            self.calls: list[
                tuple[list[EvaluationResult[Any]], list[EvaluatorFailure], EvaluatorContext[Any, Any, Any]]
            ] = []

        async def __call__(
            self,
            results: Sequence[EvaluationResult[Any]],
            failures: Sequence[EvaluatorFailure],
            context: EvaluatorContext[Any, Any, Any],
        ) -> None:
            self.calls.append((list(results), list(failures), context))

        @property
        def result_count(self) -> int:
            return sum(len(c[0]) for c in self.calls)

    class MockContextSource:
        """Mock implementation of EvaluatorContextSource for testing."""

        def __init__(self, data: dict[str, EvaluatorContext[Any, Any, Any]]) -> None:
            self._data = data

        async def fetch(self, span: SpanReference) -> EvaluatorContext[Any, Any, Any]:
            return self._data[span.span_id]

        async def fetch_many(self, spans: Sequence[SpanReference]) -> list[EvaluatorContext[Any, Any, Any]]:
            return [self._data[s.span_id] for s in spans]


@pytest.mark.anyio
async def test_callback_sink_sync():
    """CallbackSink works with sync callbacks."""
    collected: list[tuple[list[Any], list[Any], Any]] = []

    def callback(
        results: Sequence[EvaluationResult[Any]],
        failures: Sequence[EvaluatorFailure],
        context: EvaluatorContext[Any, Any, Any],
    ) -> None:
        collected.append((list(results), list(failures), context))

    sink = CallbackSink(callback)
    ctx = _make_context(output='hello')
    results = [EvaluationResult(name='test', value=True, reason=None, source=AlwaysTrue().as_spec())]

    await sink.submit(
        results=results,
        failures=[],
        context=ctx,
        span_reference=None,
        target='t',
    )

    assert len(collected) == 1
    assert collected[0][0] == results
    assert collected[0][1] == []
    assert collected[0][2] is ctx


@pytest.mark.anyio
async def test_callback_sink_async():
    """CallbackSink works with async callbacks."""
    collector = Collector()
    sink = CallbackSink(collector)
    ctx = _make_context(output='hello')
    results = [EvaluationResult(name='test', value=True, reason=None, source=AlwaysTrue().as_spec())]

    await sink.submit(
        results=results,
        failures=[],
        context=ctx,
        span_reference=None,
        target='t',
    )
    assert len(collector.calls) == 1


@pytest.mark.anyio
async def test_callback_sink_ignores_span_reference():
    """CallbackSink does not pass span_reference to the callback."""
    collector = Collector()
    sink = CallbackSink(collector)
    ctx = _make_context(output='hello')
    span_ref = SpanReference(trace_id='abc', span_id='def')

    await sink.submit(
        results=[],
        failures=[],
        context=ctx,
        span_reference=span_ref,
        target='t',
    )
    assert len(collector.calls) == 1
    assert collector.result_count == 0


@pytest.mark.anyio
async def test_legacy_sink_without_target_kwarg_is_wrapped_with_deprecation_warning():
    """Sinks whose `submit()` predates the `target` kwarg still work via a back-compat shim.

    TODO(v2): delete this test alongside the shim in pydantic_evals/_online.py.
    """
    calls: list[dict[str, Any]] = []

    class LegacySink:
        async def submit(
            self,
            *,
            results: Sequence[EvaluationResult],
            failures: Sequence[EvaluatorFailure],
            context: EvaluatorContext[Any, Any, Any],
            span_reference: SpanReference | None,
        ) -> None:
            calls.append(
                {
                    'results': list(results),
                    'failures': list(failures),
                    'context': context,
                    'span_reference': span_reference,
                }
            )

    @dataclass
    class LegacyEvaluator(Evaluator):
        def evaluate(self, ctx: EvaluatorContext) -> bool:
            return True

    # Cast: the point of this test is that LegacySink intentionally doesn't
    # satisfy the current EvaluationSink protocol (missing `target` kwarg).
    config = OnlineEvalConfig(default_sink=cast(Any, LegacySink()), emit_otel_events=False)

    @config.evaluate(LegacyEvaluator())
    async def run(x: int) -> int:
        return x

    with pytest.warns(DeprecationWarning, match=r"missing the 'target: str' keyword argument"):
        await run(1)
        await wait_for_evaluations()

    assert len(calls) == 1
    # The legacy sink receives the original four kwargs — `target` is dropped by the shim.
    assert set(calls[0]) == {'results', 'failures', 'context', 'span_reference'}


async def test_legacy_sink_warning_fires_once_per_class():
    """The back-compat shim warns the first time it wraps a given class, not every time.

    Exercises the compat shim directly rather than the full dispatch pipeline,
    so parallel tests touching the module-level `_warned_legacy_sink_ids` set
    can't flake this assertion via `id()` reuse.

    TODO(v2): delete this test alongside the shim in pydantic_evals/_online.py.
    """
    from pydantic_evals._online import (
        _ensure_target_compat,  # pyright: ignore[reportPrivateUsage]
        _warned_legacy_sink_ids,  # pyright: ignore[reportPrivateUsage]
    )

    class OnceLegacySink:
        async def submit(
            self,
            *,
            results: Sequence[EvaluationResult],
            failures: Sequence[EvaluatorFailure],
            context: EvaluatorContext[Any, Any, Any],
            span_reference: SpanReference | None,
        ) -> None:
            pass

    # Defensive: drop any stale id(cls) collision from an earlier GC'd class.
    _warned_legacy_sink_ids.discard(id(OnceLegacySink))

    sink = cast(Any, OnceLegacySink())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', DeprecationWarning)
        _ensure_target_compat(sink)
        _ensure_target_compat(sink)

    legacy_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(legacy_warnings) == 1


@pytest.mark.anyio
async def test_sink_with_var_keyword_is_treated_as_modern():
    """A sink whose `submit` uses **kwargs is assumed to accept `target` — no shim, no warning."""
    calls: list[dict[str, Any]] = []

    class KwargsSink:
        async def submit(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    @dataclass
    class E(Evaluator):
        def evaluate(self, ctx: EvaluatorContext) -> bool:
            return True

    config = OnlineEvalConfig(default_sink=KwargsSink(), emit_otel_events=False)

    @config.evaluate(E(), target='my_target')
    async def run(x: int) -> int:
        return x

    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        await run(1)
        await wait_for_evaluations()

    assert len(calls) == 1
    assert calls[0]['target'] == 'my_target'


@pytest.mark.anyio
async def test_span_reference():
    """SpanReference stores trace and span IDs."""
    ref = SpanReference(trace_id='abc123', span_id='def456')
    assert ref.trace_id == 'abc123'
    assert ref.span_id == 'def456'


@pytest.mark.anyio
async def test_online_evaluator_defaults():
    """OnlineEvaluator has sensible defaults."""
    evaluator = AlwaysTrue()
    online = OnlineEvaluator(evaluator=evaluator)
    assert online.evaluator is evaluator
    assert online.sample_rate is None
    assert online.sink is None
    assert online.max_concurrency == 10


@pytest.mark.anyio
async def test_online_evaluator_custom_config():
    """OnlineEvaluator accepts custom configuration."""
    evaluator = AlwaysTrue()
    collector = Collector()
    sink = CallbackSink(collector)
    online = OnlineEvaluator(
        evaluator=evaluator,
        sample_rate=0.5,
        sink=sink,
        max_concurrency=5,
    )
    assert online.sample_rate == 0.5
    assert online.sink is sink
    assert online.max_concurrency == 5


@pytest.mark.anyio
async def test_run_evaluators_success():
    """run_evaluators returns results from all evaluators."""
    ctx = _make_context(output=42)
    results, failures = await run_evaluators([AlwaysTrue(), OutputEquals(value=42)], ctx)

    assert len(results) == 2
    assert len(failures) == 0
    assert results[0].value is True
    assert results[1].value is True


@pytest.mark.anyio
async def test_run_evaluators_with_failure():
    """run_evaluators collects failures separately from results."""
    ctx = _make_context(output=42)
    results, failures = await run_evaluators([AlwaysTrue(), FailingEvaluator()], ctx)

    assert len(results) == 1
    assert results[0].value is True
    assert len(failures) == 1
    assert 'Simulated evaluator failure' in failures[0].error_message


@pytest.mark.anyio
async def test_run_evaluators_empty():
    """run_evaluators handles empty evaluator list."""
    ctx = _make_context(output=42)
    results, failures = await run_evaluators([], ctx)
    assert results == []
    assert failures == []


@pytest.mark.anyio
async def test_run_evaluators_multi_result():
    """run_evaluators handles evaluators that return multiple results."""
    ctx = _make_context(output=42)
    results, failures = await run_evaluators([MultiResultEvaluator()], ctx)

    assert len(results) == 3
    assert len(failures) == 0
    result_names = {r.name for r in results}
    assert result_names == {'accuracy', 'score', 'label'}


@pytest.mark.anyio
async def test_run_evaluators_async_evaluator():
    """run_evaluators works with async evaluators."""
    ctx = _make_context(output=42)
    results, failures = await run_evaluators([AsyncEvaluator()], ctx)

    assert len(results) == 1
    assert results[0].value is True
    assert len(failures) == 0


@pytest.mark.anyio
async def test_evaluate_decorator_async_basic():
    """evaluate() decorator runs evaluators on async function calls."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(AlwaysTrue())
    async def my_func(x: int) -> int:
        return x * 2

    result = await my_func(21)
    assert result == 42

    # Wait for background task to complete
    await wait_for_evaluations()

    assert len(collector.calls) == 1
    results, _, ctx = collector.calls[0]
    assert len(results) == 1
    assert results[0].value is True
    assert ctx.output == 42
    assert ctx.inputs == {'x': 21}


@pytest.mark.anyio
async def test_evaluate_decorator_async_preserves_signature():
    """evaluate() decorator preserves the function's name and docs."""

    @evaluate(AlwaysTrue())
    async def my_func(x: int) -> int:
        """My docstring."""
        return x

    assert my_func.__name__ == 'my_func'
    assert my_func.__doc__ == 'My docstring.'
    assert await my_func(42) == 42


@pytest.mark.anyio
async def test_evaluate_decorator_multiple_evaluators():
    """evaluate() decorator runs multiple evaluators."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(AlwaysTrue(), OutputEquals(value=42))
    async def my_func(x: int) -> int:
        return x * 2

    result = await my_func(21)
    assert result == 42

    await wait_for_evaluations()

    assert len(collector.calls) >= 1
    assert collector.result_count == 2


@pytest.mark.anyio
async def test_evaluate_decorator_with_failure():
    """evaluate() decorator handles evaluator failures gracefully."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(FailingEvaluator())
    async def my_func(x: int) -> int:
        return x

    result = await my_func(42)
    assert result == 42

    await wait_for_evaluations()

    assert len(collector.calls) == 1
    results, failures, _ = collector.calls[0]
    assert len(results) == 0
    assert len(failures) == 1
    assert 'Simulated evaluator failure' in failures[0].error_message


@pytest.mark.anyio
async def test_sample_rate_zero_skips_evaluation():
    """sample_rate=0.0 skips all evaluations."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(OnlineEvaluator(evaluator=AlwaysTrue(), sample_rate=0.0))
    async def my_func(x: int) -> int:
        return x

    result = await my_func(42)
    assert result == 42

    await wait_for_evaluations()
    assert len(collector.calls) == 0


@pytest.mark.anyio
async def test_sample_rate_one_always_evaluates():
    """sample_rate=1.0 always evaluates."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(OnlineEvaluator(evaluator=AlwaysTrue(), sample_rate=1.0))
    async def my_func(x: int) -> int:
        return x

    for _ in range(5):
        await my_func(42)

    await wait_for_evaluations()
    assert len(collector.calls) == 5


@pytest.mark.anyio
async def test_sample_rate_callable():
    """sample_rate as a callable is evaluated each time."""
    call_count = 0
    collector = Collector()

    def dynamic_rate(ctx: SamplingContext) -> float:
        nonlocal call_count
        call_count += 1
        return 1.0

    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(OnlineEvaluator(evaluator=AlwaysTrue(), sample_rate=dynamic_rate))
    async def my_func(x: int) -> int:
        return x

    await my_func(42)
    await wait_for_evaluations()

    assert call_count >= 1
    assert len(collector.calls) == 1


@pytest.mark.anyio
async def test_sample_rate_callable_returning_bool():
    """sample_rate callable can return bool."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(OnlineEvaluator(evaluator=AlwaysTrue(), sample_rate=lambda ctx: False))
    async def my_func(x: int) -> int:
        return x

    await my_func(42)
    await wait_for_evaluations()
    assert len(collector.calls) == 0


@pytest.mark.anyio
async def test_disable_evaluation_context_manager():
    """disable_evaluation() suppresses all evaluators."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(AlwaysTrue())
    async def my_func(x: int) -> int:
        return x

    with disable_evaluation():
        result = await my_func(42)
        assert result == 42

    await wait_for_evaluations()
    assert len(collector.calls) == 0


@pytest.mark.anyio
async def test_disable_evaluation_restores():
    """disable_evaluation() restores evaluation after exiting."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(AlwaysTrue())
    async def my_func(x: int) -> int:
        return x

    with disable_evaluation():
        await my_func(42)

    # After exiting, evaluations should resume
    await my_func(42)
    await wait_for_evaluations()
    assert len(collector.calls) == 1


@pytest.mark.anyio
async def test_config_enabled_false():
    """OnlineEvalConfig.enabled=False disables all evaluation."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector, enabled=False)

    @config.evaluate(AlwaysTrue())
    async def my_func(x: int) -> int:
        return x

    result = await my_func(42)
    assert result == 42

    await wait_for_evaluations()
    assert len(collector.calls) == 0


@pytest.mark.anyio
async def test_per_evaluator_sink_override():
    """OnlineEvaluator.sink overrides config's default_sink."""
    default_collector = Collector()
    override_collector = Collector()

    config = OnlineEvalConfig(default_sink=default_collector)

    @config.evaluate(
        AlwaysTrue(),  # uses default sink
        OnlineEvaluator(evaluator=AlwaysFalse(), sink=override_collector),  # uses override
    )
    async def my_func(x: int) -> int:
        return x

    await my_func(42)
    await wait_for_evaluations()

    assert len(default_collector.calls) == 1
    assert len(override_collector.calls) == 1


@pytest.mark.anyio
@needs_logfire
async def test_no_sink_still_emits_otel_events(capfire: CaptureLogfire):
    """When no sink is configured, evaluators still run and emit OTel events.

    Out-of-the-box OTel event emission matches how offline evals produce spans
    via `logfire_span` — users don't need to register a sink to see results.
    """
    config = OnlineEvalConfig()  # no user sinks

    @config.evaluate(AlwaysTrue())
    async def my_func(x: int) -> int:
        return x

    result = await my_func(42)
    assert result == 42
    await wait_for_evaluations()

    finished = capfire.log_exporter.get_finished_logs()
    assert len(finished) == 1
    attrs = dict(finished[0].log_record.attributes or {})
    assert attrs['gen_ai.evaluation.name'] == 'AlwaysTrue'
    assert attrs['gen_ai.evaluation.score.label'] == 'pass'


@pytest.mark.anyio
@needs_logfire
async def test_emit_otel_events_false_disables_emission(capfire: CaptureLogfire):
    """`emit_otel_events=False` suppresses the default OTel emission."""
    calls: list[int] = []

    def sink_cb(
        results: Sequence[EvaluationResult],
        failures: Sequence[EvaluatorFailure],
        context: EvaluatorContext[Any, Any, Any],
    ) -> None:
        calls.append(len(results))

    config = OnlineEvalConfig(default_sink=sink_cb, emit_otel_events=False)

    @config.evaluate(AlwaysTrue())
    async def my_func(x: int) -> int:
        return x

    await my_func(42)
    await wait_for_evaluations()

    # User sink still ran.
    assert calls == [1]
    # But no OTel events were emitted.
    assert list(capfire.log_exporter.get_finished_logs()) == []


@pytest.mark.anyio
async def test_configure_updates_default_config():
    """configure() updates the global DEFAULT_CONFIG."""
    original_enabled = DEFAULT_CONFIG.enabled
    original_sink = DEFAULT_CONFIG.default_sink
    original_rate = DEFAULT_CONFIG.default_sample_rate

    original_on_max = DEFAULT_CONFIG.on_max_concurrency
    original_emit = DEFAULT_CONFIG.emit_otel_events

    try:
        configure(enabled=False, default_sample_rate=0.5)
        assert DEFAULT_CONFIG.enabled is False
        assert DEFAULT_CONFIG.default_sample_rate == 0.5
        assert DEFAULT_CONFIG.default_sink is original_sink

        def handler(ctx: EvaluatorContext[Any, Any, Any]) -> None:
            pass

        configure(on_max_concurrency=handler)
        assert DEFAULT_CONFIG.on_max_concurrency is handler

        configure(emit_otel_events=False)
        assert DEFAULT_CONFIG.emit_otel_events is False

    finally:
        DEFAULT_CONFIG.enabled = original_enabled
        DEFAULT_CONFIG.default_sink = original_sink
        DEFAULT_CONFIG.default_sample_rate = original_rate
        DEFAULT_CONFIG.on_max_concurrency = original_on_max
        DEFAULT_CONFIG.emit_otel_events = original_emit


@pytest.mark.anyio
async def test_configure_can_reset_to_none():
    """configure() can explicitly set fields to None to clear them."""
    collector = Collector()
    original_sink = DEFAULT_CONFIG.default_sink
    original_metadata = DEFAULT_CONFIG.metadata

    try:
        configure(default_sink=collector, metadata={'key': 'value'})
        assert DEFAULT_CONFIG.default_sink is collector
        assert DEFAULT_CONFIG.metadata == {'key': 'value'}

        # Explicitly passing None should clear the values
        configure(default_sink=None, metadata=None)
        assert DEFAULT_CONFIG.default_sink is None
        assert DEFAULT_CONFIG.metadata is None
    finally:
        DEFAULT_CONFIG.default_sink = original_sink
        DEFAULT_CONFIG.metadata = original_metadata


@pytest.mark.anyio
async def test_module_level_evaluate():
    """Module-level evaluate() delegates to DEFAULT_CONFIG."""
    collector = Collector()
    original_sink = DEFAULT_CONFIG.default_sink
    try:
        DEFAULT_CONFIG.default_sink = collector

        @evaluate(AlwaysTrue())
        async def my_func(x: int) -> int:
            return x

        result = await my_func(42)
        assert result == 42

        await wait_for_evaluations()
        assert len(collector.calls) == 1
    finally:
        DEFAULT_CONFIG.default_sink = original_sink


@pytest.mark.anyio
async def test_context_source_fetch():
    """EvaluatorContextSource.fetch retrieves stored context data."""
    source = MockContextSource(
        {
            'span1': _make_context(
                inputs={'query': 'hello'},
                output='world',
                metadata={'service': 'test'},
                duration=1.5,
            ),
        }
    )

    ctx = await source.fetch(SpanReference(trace_id='trace1', span_id='span1'))

    assert ctx.inputs == {'query': 'hello'}
    assert ctx.output == 'world'
    assert ctx.expected_output is None
    assert ctx.metadata == {'service': 'test'}
    assert ctx.duration == 1.5


@pytest.mark.anyio
async def test_context_source_fetch_many():
    """EvaluatorContextSource.fetch_many retrieves multiple contexts in batch."""
    source = MockContextSource(
        {
            'span1': _make_context(inputs={'q': '1'}, output='a', duration=1.0),
            'span2': _make_context(inputs={'q': '2'}, output='b', duration=2.0),
        }
    )

    spans = [
        SpanReference(trace_id='t', span_id='span1'),
        SpanReference(trace_id='t', span_id='span2'),
    ]
    contexts = await source.fetch_many(spans)

    assert len(contexts) == 2
    assert contexts[0].inputs == {'q': '1'}
    assert contexts[0].output == 'a'
    assert contexts[1].inputs == {'q': '2'}
    assert contexts[1].output == 'b'


@pytest.mark.anyio
async def test_fetch_and_run_evaluators():
    """EvaluatorContextSource.fetch + run_evaluators works end-to-end."""
    source = MockContextSource(
        {
            'span1': _make_context(output=42, duration=0.1),
        }
    )

    ctx = await source.fetch(SpanReference(trace_id='t', span_id='span1'))
    results, failures = await run_evaluators([OutputEquals(value=42), AlwaysTrue()], ctx)

    assert len(results) == 2
    assert len(failures) == 0
    assert all(r.value is True for r in results)


@pytest.mark.anyio
async def test_config_metadata_passed_to_context():
    """OnlineEvalConfig.metadata is included in the EvaluatorContext."""
    collected_contexts: list[EvaluatorContext[Any, Any, Any]] = []

    async def sink_callback(
        results: Sequence[EvaluationResult[Any]],
        failures: Sequence[EvaluatorFailure],
        context: EvaluatorContext[Any, Any, Any],
    ) -> None:
        collected_contexts.append(context)

    config = OnlineEvalConfig(
        default_sink=sink_callback,
        metadata={'service': 'test-app', 'version': '1.0'},
    )

    @config.evaluate(AlwaysTrue())
    async def my_func(x: int) -> int:
        return x

    await my_func(42)
    await wait_for_evaluations()

    assert len(collected_contexts) == 1
    assert collected_contexts[0].metadata == {'service': 'test-app', 'version': '1.0'}


@pytest.mark.anyio
async def test_max_concurrency_respected():
    """OnlineEvaluator.max_concurrency limits concurrent evaluations."""
    active = 0
    max_active = 0
    completed = 0

    @dataclass
    class SlowEvaluator(Evaluator):
        async def evaluate(self, ctx: EvaluatorContext) -> EvaluatorOutput:
            nonlocal active, max_active, completed
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.05)
            active -= 1
            completed += 1
            return True

    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(OnlineEvaluator(evaluator=SlowEvaluator(), max_concurrency=2, sample_rate=1.0))
    async def my_func(x: int) -> int:
        return x

    # Fire off several calls rapidly
    tasks = [my_func(i) for i in range(5)]
    await asyncio.gather(*tasks)

    # Wait for all background evaluations
    await wait_for_evaluations()

    # The semaphore should have limited concurrency to 2
    assert max_active <= 2


@pytest.mark.anyio
async def test_custom_sink_protocol():
    """Custom EvaluationSink implementations work."""

    class MySink:
        def __init__(self) -> None:
            self.submissions: list[tuple[list[EvaluationResult[Any]], SpanReference | None]] = []

        async def submit(
            self,
            *,
            results: Sequence[EvaluationResult[Any]],
            failures: Sequence[EvaluatorFailure],
            context: EvaluatorContext[Any, Any, Any],
            span_reference: SpanReference | None,
            target: Any,
        ) -> None:
            self.submissions.append((list(results), span_reference))

    sink = MySink()
    config = OnlineEvalConfig(default_sink=sink)

    @config.evaluate(AlwaysTrue())
    async def my_func(x: int) -> int:
        return x

    await my_func(42)
    await wait_for_evaluations()

    assert len(sink.submissions) == 1
    results, _ = sink.submissions[0]
    assert len(results) == 1
    assert results[0].value is True


@pytest.mark.anyio
async def test_bare_evaluator_uses_config_defaults():
    """Bare Evaluator passed to evaluate() uses config's default_sample_rate."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector, default_sample_rate=0.0)

    @config.evaluate(AlwaysTrue())  # bare evaluator, inherits sample_rate=0.0
    async def my_func(x: int) -> int:
        return x

    await my_func(42)
    await wait_for_evaluations()
    assert len(collector.calls) == 0


@pytest.mark.anyio
async def test_bare_evaluator_late_binds_config_defaults():
    """Config defaults are resolved at call time, not decoration time."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector, default_sample_rate=0.0)

    @config.evaluate(AlwaysTrue())  # bare evaluator, sample_rate resolved at call time
    async def my_func(x: int) -> int:
        return x

    # Initially sample_rate=0.0 — no evaluations
    await my_func(42)
    await wait_for_evaluations()
    assert len(collector.calls) == 0

    # Change config after decoration — should take effect
    config.default_sample_rate = 1.0
    await my_func(42)
    await wait_for_evaluations()
    assert len(collector.calls) == 1

    # OnlineEvaluator with explicit sample_rate is NOT affected by config changes
    collector2 = Collector()
    config2 = OnlineEvalConfig(default_sink=collector2, default_sample_rate=1.0)

    @config2.evaluate(OnlineEvaluator(evaluator=AlwaysTrue(), sample_rate=0.0))
    async def my_func2(x: int) -> int:
        return x

    config2.default_sample_rate = 1.0  # this should NOT override the explicit 0.0
    await my_func2(42)
    await wait_for_evaluations()
    assert len(collector2.calls) == 0  # still 0 because OnlineEvaluator has explicit sample_rate=0.0


@pytest.mark.anyio
async def test_multiple_sinks():
    """Multiple sinks receive all results."""
    collector1 = Collector()
    collector2 = Collector()

    config = OnlineEvalConfig(default_sink=[CallbackSink(collector1), CallbackSink(collector2)])

    @config.evaluate(AlwaysTrue())
    async def my_func(x: int) -> int:
        return x

    await my_func(42)
    await wait_for_evaluations()

    assert len(collector1.calls) == 1
    assert len(collector2.calls) == 1


@pytest.mark.anyio
async def test_fractional_sample_rate():
    """Fractional sample_rate evaluates a subset of calls."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(OnlineEvaluator(evaluator=AlwaysTrue(), sample_rate=0.5))
    async def my_func(x: int) -> int:
        return x

    # Run many times — with 0.5 rate, we should get some but not all
    for _ in range(50):
        await my_func(42)

    await wait_for_evaluations()
    # Statistically, should get roughly 25 ± some variance, but definitely not 0 or 50
    assert 5 < len(collector.calls) < 45


@pytest.mark.anyio
async def test_sample_rate_callable_exception_propagates():
    """Exception in sample_rate callable propagates — it's the user's responsibility."""
    collector = Collector()

    def bad_rate(ctx: SamplingContext) -> float:
        raise ValueError('rate error')

    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(OnlineEvaluator(evaluator=AlwaysTrue(), sample_rate=bad_rate))
    async def my_func(x: int) -> int:
        return x  # pragma: no cover

    with pytest.raises(ValueError, match='rate error'):
        await my_func(42)


@pytest.mark.anyio
async def test_sample_rate_callable_exception_calls_on_sampling_error():
    """When on_sampling_error is set, sample_rate exceptions are routed there instead of propagating."""
    errors: list[tuple[Exception, Evaluator]] = []

    def on_sampling_error(exc: Exception, evaluator: Evaluator) -> None:
        errors.append((exc, evaluator))

    collector = Collector()

    def bad_rate(ctx: SamplingContext) -> float:
        raise ValueError('rate error')

    config = OnlineEvalConfig(default_sink=collector, on_sampling_error=on_sampling_error)

    @config.evaluate(OnlineEvaluator(evaluator=AlwaysTrue(), sample_rate=bad_rate))
    async def my_func(x: int) -> int:
        return x

    result = await my_func(42)
    assert result == 42

    assert len(errors) == 1
    assert isinstance(errors[0][0], ValueError)
    assert str(errors[0][0]) == 'rate error'


@pytest.mark.anyio
async def test_on_sampling_error_handler_exception_suppressed():
    """If on_sampling_error itself raises, the exception is suppressed and the evaluator is skipped."""
    collector = Collector()

    def bad_handler(exc: Exception, evaluator: Evaluator) -> None:
        raise RuntimeError('handler boom')

    def bad_rate(ctx: SamplingContext) -> float:
        raise ValueError('rate error')

    config = OnlineEvalConfig(default_sink=collector, on_sampling_error=bad_handler)

    @config.evaluate(OnlineEvaluator(evaluator=AlwaysTrue(), sample_rate=bad_rate))
    async def my_func(x: int) -> int:
        return x

    result = await my_func(42)
    assert result == 42
    assert len(collector.calls) == 0


@pytest.mark.anyio
async def test_sink_exception_does_not_propagate():
    """Exception in a sink is logged but does not break other sinks."""

    class FailingSink:
        async def submit(self, **kwargs: Any) -> None:
            raise ValueError('sink error')

    collector = Collector()
    config = OnlineEvalConfig(default_sink=[FailingSink(), CallbackSink(collector)])

    @config.evaluate(AlwaysTrue())
    async def my_func(x: int) -> int:
        return x

    result = await my_func(42)
    assert result == 42

    await wait_for_evaluations()
    # The second sink should still have received results despite the first failing
    assert len(collector.calls) == 1


@pytest.mark.anyio
async def test_sync_function_from_async_context():
    """Sync decorated function called from async context dispatches via background thread."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(AlwaysTrue())
    def my_func(x: int) -> int:
        return x * 2

    result = my_func(21)
    assert result == 42

    await wait_for_evaluations()

    assert len(collector.calls) == 1
    results, _, ctx = collector.calls[0]
    assert len(results) == 1
    assert results[0].value is True
    assert ctx.output == 42


@needs_logfire
@pytest.mark.anyio
async def test_span_reference_with_configured_logfire(capfire: CaptureLogfire):
    """Decorator produces valid SpanReference when logfire is configured."""
    span_refs: list[SpanReference | None] = []

    class SpanCaptureSink:
        async def submit(
            self,
            *,
            results: Sequence[EvaluationResult[Any]],
            failures: Sequence[EvaluatorFailure],
            context: EvaluatorContext[Any, Any, Any],
            span_reference: SpanReference | None,
            target: Any,
        ) -> None:
            span_refs.append(span_reference)

    config = OnlineEvalConfig(default_sink=SpanCaptureSink())

    @config.evaluate(AlwaysTrue())
    async def my_func(x: int) -> int:
        return x

    await my_func(42)
    await wait_for_evaluations()

    assert len(span_refs) == 1
    ref = span_refs[0]
    assert ref is not None
    assert isinstance(ref, SpanReference)
    assert len(ref.trace_id) == 32
    assert len(ref.span_id) == 16
    assert int(ref.trace_id, 16) != 0
    assert int(ref.span_id, 16) != 0


@pytest.mark.anyio
async def test_sync_decorated_function_dispatch():
    """Sync decorated function dispatches evaluators when called from async context."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(AlwaysTrue())
    def my_func(x: int) -> int:
        return x * 2

    result = my_func(21)
    assert result == 42

    await wait_for_evaluations()

    assert len(collector.calls) == 1
    results, _, ctx = collector.calls[0]
    assert len(results) == 1
    assert results[0].value is True
    assert ctx.output == 42
    assert ctx.inputs == {'x': 21}


@pytest.mark.anyio
async def test_sync_decorated_function_disabled():
    """Disabled config doesn't dispatch evaluators for sync decorated functions."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector, enabled=False)

    @config.evaluate(AlwaysTrue())
    def my_func(x: int) -> int:
        return x

    result = my_func(42)
    assert result == 42

    await wait_for_evaluations()
    assert len(collector.calls) == 0


@pytest.mark.anyio
async def test_sync_decorated_function_sample_rate_zero():
    """sample_rate=0 doesn't dispatch evaluators for sync decorated functions."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(OnlineEvaluator(evaluator=AlwaysTrue(), sample_rate=0.0))
    def my_func(x: int) -> int:
        return x

    result = my_func(42)
    assert result == 42

    await wait_for_evaluations()
    assert len(collector.calls) == 0


@pytest.mark.anyio
async def test_sync_function_no_event_loop():
    """Sync decorated function called without an event loop dispatches via background thread."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(AlwaysTrue())
    def my_func(x: int) -> int:
        return x * 2

    # Call from a thread with no running event loop to exercise _dispatch_in_background_thread
    from anyio.to_thread import run_sync

    result = await run_sync(my_func, 21)
    assert result == 42

    await wait_for_evaluations()

    assert len(collector.calls) == 1
    results, _, ctx = collector.calls[0]
    assert len(results) == 1
    assert results[0].value is True
    assert ctx.output == 42
    assert ctx.inputs == {'x': 21}


@pytest.mark.anyio
async def test_mixed_list_sink():
    """A list containing both a bare callable and a CallbackSink exercises _normalize_single_sink."""
    collector1 = Collector()
    collector2 = Collector()

    # Passing a list with a bare callable alongside a CallbackSink triggers
    # _normalize_single_sink for the callable element.
    config = OnlineEvalConfig(default_sink=[collector1, CallbackSink(collector2)])

    @config.evaluate(AlwaysTrue())
    async def my_func(x: int) -> int:
        return x

    await my_func(42)
    await wait_for_evaluations()

    assert len(collector1.calls) == 1
    assert len(collector2.calls) == 1


@pytest.mark.anyio
async def test_on_max_concurrency_callback():
    """on_max_concurrency is called when evaluations are dropped."""
    dropped_contexts: list[EvaluatorContext[Any, Any, Any]] = []

    @dataclass
    class SlowEvaluator(Evaluator):
        async def evaluate(self, ctx: EvaluatorContext) -> EvaluatorOutput:
            await asyncio.sleep(0.1)
            return True

    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(
        OnlineEvaluator(
            evaluator=SlowEvaluator(),
            max_concurrency=1,
            sample_rate=1.0,
            on_max_concurrency=lambda ctx: dropped_contexts.append(ctx),
        )
    )
    async def my_func(x: int) -> int:
        return x

    # Fire off several calls — only 1 can run concurrently, rest should be dropped
    tasks = [my_func(i) for i in range(5)]
    await asyncio.gather(*tasks)

    await wait_for_evaluations()

    # At least some evaluations should have been dropped
    assert len(dropped_contexts) > 0
    # Total dropped + completed should equal 5
    assert len(dropped_contexts) + len(collector.calls) == 5


@pytest.mark.anyio
async def test_on_max_concurrency_async_callback():
    """on_max_concurrency works with async callbacks."""
    dropped_count = 0

    @dataclass
    class SlowEvaluator(Evaluator):
        async def evaluate(self, ctx: EvaluatorContext) -> EvaluatorOutput:
            await asyncio.sleep(0.1)
            return True

    async def on_drop(ctx: EvaluatorContext[Any, Any, Any]) -> None:
        nonlocal dropped_count
        dropped_count += 1

    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(
        OnlineEvaluator(
            evaluator=SlowEvaluator(),
            max_concurrency=1,
            sample_rate=1.0,
            on_max_concurrency=on_drop,
        )
    )
    async def my_func(x: int) -> int:
        return x

    tasks = [my_func(i) for i in range(5)]
    await asyncio.gather(*tasks)

    await wait_for_evaluations()

    assert dropped_count > 0
    assert dropped_count + len(collector.calls) == 5


@pytest.mark.anyio
async def test_on_max_concurrency_config_default():
    """OnlineEvalConfig.on_max_concurrency is used when OnlineEvaluator doesn't set one."""
    dropped_contexts: list[EvaluatorContext[Any, Any, Any]] = []

    @dataclass
    class SlowEvaluator(Evaluator):
        async def evaluate(self, ctx: EvaluatorContext) -> EvaluatorOutput:
            await asyncio.sleep(0.1)
            return True

    collector = Collector()
    config = OnlineEvalConfig(
        default_sink=collector,
        on_max_concurrency=lambda ctx: dropped_contexts.append(ctx),
    )

    @config.evaluate(OnlineEvaluator(evaluator=SlowEvaluator(), max_concurrency=1, sample_rate=1.0))
    async def my_func(x: int) -> int:
        return x

    tasks = [my_func(i) for i in range(5)]
    await asyncio.gather(*tasks)

    await wait_for_evaluations()

    assert len(dropped_contexts) > 0
    assert len(dropped_contexts) + len(collector.calls) == 5


@pytest.mark.anyio
async def test_on_max_concurrency_evaluator_overrides_config():
    """OnlineEvaluator.on_max_concurrency overrides the config default."""
    config_drops: list[EvaluatorContext[Any, Any, Any]] = []
    evaluator_drops: list[EvaluatorContext[Any, Any, Any]] = []

    @dataclass
    class SlowEvaluator(Evaluator):
        async def evaluate(self, ctx: EvaluatorContext) -> EvaluatorOutput:
            await asyncio.sleep(0.1)
            return True

    collector = Collector()
    config = OnlineEvalConfig(
        default_sink=collector,
        on_max_concurrency=lambda ctx: config_drops.append(ctx),
    )

    @config.evaluate(
        OnlineEvaluator(
            evaluator=SlowEvaluator(),
            max_concurrency=1,
            sample_rate=1.0,
            on_max_concurrency=lambda ctx: evaluator_drops.append(ctx),
        )
    )
    async def my_func(x: int) -> int:
        return x

    tasks = [my_func(i) for i in range(5)]
    await asyncio.gather(*tasks)

    await wait_for_evaluations()

    # Config handler should NOT have been called — evaluator handler overrides it
    assert len(config_drops) == 0
    assert len(evaluator_drops) > 0
    assert len(evaluator_drops) + len(collector.calls) == 5


# --- on_error tests ---


@pytest.mark.anyio
async def test_on_error_sink_exception():
    """on_error is called with 'sink' location when sink raises."""
    errors: list[tuple[Exception, OnErrorLocation]] = []

    def on_error(
        exc: Exception,
        ctx: EvaluatorContext[Any, Any, Any],
        evaluator: Evaluator,
        location: OnErrorLocation,
    ) -> None:
        errors.append((exc, location))

    class FailingSink:
        async def submit(self, **kwargs: Any) -> None:
            raise ValueError('sink boom')

    good_collector = Collector()
    config = OnlineEvalConfig(default_sink=[FailingSink(), CallbackSink(good_collector)], on_error=on_error)

    @config.evaluate(AlwaysTrue())
    async def my_func(x: int) -> int:
        return x

    await my_func(42)
    await wait_for_evaluations()

    assert len(errors) == 1
    assert errors[0][1] == 'sink'
    # The other sink should still have received results
    assert len(good_collector.calls) == 1


@pytest.mark.anyio
async def test_on_error_on_max_concurrency_exception():
    """on_error is called with 'on_max_concurrency' when on_max_concurrency callback raises."""
    errors: list[tuple[Exception, OnErrorLocation]] = []

    def on_error(
        exc: Exception,
        ctx: EvaluatorContext[Any, Any, Any],
        evaluator: Evaluator,
        location: OnErrorLocation,
    ) -> None:
        errors.append((exc, location))

    def bad_callback(ctx: EvaluatorContext[Any, Any, Any]) -> None:
        raise ValueError('callback boom')

    @dataclass
    class SlowEvaluator(Evaluator):
        async def evaluate(self, ctx: EvaluatorContext) -> EvaluatorOutput:
            await asyncio.sleep(0.1)
            return True

    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector, on_error=on_error)

    @config.evaluate(
        OnlineEvaluator(
            evaluator=SlowEvaluator(),
            max_concurrency=1,
            sample_rate=1.0,
            on_max_concurrency=bad_callback,
        )
    )
    async def my_func(x: int) -> int:
        return x

    tasks = [my_func(i) for i in range(5)]
    await asyncio.gather(*tasks)
    await wait_for_evaluations()

    # At least some should have been dropped and triggered the bad callback
    assert len(errors) > 0
    assert all(loc == 'on_max_concurrency' for _, loc in errors)


@pytest.mark.anyio
async def test_on_error_handler_exception_suppressed():
    """on_error handler that raises is silently suppressed."""

    class FailingSink:
        async def submit(self, **kwargs: Any) -> None:
            raise ValueError('sink boom')

    good_collector = Collector()

    def bad_on_error(
        exc: Exception,
        ctx: EvaluatorContext[Any, Any, Any],
        evaluator: Evaluator,
        location: OnErrorLocation,
    ) -> None:
        raise RuntimeError('handler boom')

    config = OnlineEvalConfig(default_sink=[FailingSink(), CallbackSink(good_collector)], on_error=bad_on_error)

    @config.evaluate(AlwaysTrue())
    async def my_func(x: int) -> int:
        return x

    result = await my_func(42)
    assert result == 42

    await wait_for_evaluations()

    # Good sink should still have received results despite on_error raising
    assert len(good_collector.calls) == 1


@pytest.mark.anyio
async def test_on_error_per_evaluator_overrides_config():
    """Per-evaluator on_error overrides the config default."""
    evaluator_errors: list[OnErrorLocation] = []

    class FailingSink:
        async def submit(self, **kwargs: Any) -> None:
            raise ValueError('sink boom')

    def config_on_error(
        exc: Exception,
        ctx: EvaluatorContext[Any, Any, Any],
        evaluator: Evaluator,
        location: OnErrorLocation,
    ) -> None:
        pytest.fail('config on_error should not be called when per-evaluator on_error is set')  # pragma: no cover

    def evaluator_on_error(
        exc: Exception,
        ctx: EvaluatorContext[Any, Any, Any],
        evaluator: Evaluator,
        location: OnErrorLocation,
    ) -> None:
        evaluator_errors.append(location)

    config = OnlineEvalConfig(on_error=config_on_error)

    @config.evaluate(
        OnlineEvaluator(evaluator=AlwaysTrue(), sink=FailingSink(), on_error=evaluator_on_error),
    )
    async def my_func(x: int) -> int:
        return x

    await my_func(42)
    await wait_for_evaluations()

    assert len(evaluator_errors) == 1
    assert evaluator_errors[0] == 'sink'


@pytest.mark.anyio
async def test_on_error_async_callback():
    """Async on_error callback works."""
    errors: list[OnErrorLocation] = []

    class FailingSink:
        async def submit(self, **kwargs: Any) -> None:
            raise ValueError('sink boom')

    async def async_on_error(
        exc: Exception,
        ctx: EvaluatorContext[Any, Any, Any],
        evaluator: Evaluator,
        location: OnErrorLocation,
    ) -> None:
        await asyncio.sleep(0)
        errors.append(location)

    config = OnlineEvalConfig(on_error=async_on_error)

    @config.evaluate(OnlineEvaluator(evaluator=AlwaysTrue(), sink=FailingSink()))
    async def my_func(x: int) -> int:
        return x

    await my_func(42)
    await wait_for_evaluations()

    assert len(errors) == 1
    assert errors[0] == 'sink'


@pytest.mark.anyio
async def test_configure_on_sampling_error():
    """configure() can set on_sampling_error on DEFAULT_CONFIG."""
    original = DEFAULT_CONFIG.on_sampling_error
    try:

        def handler(exc: Exception, evaluator: Evaluator) -> None:
            pass

        configure(on_sampling_error=handler)
        assert DEFAULT_CONFIG.on_sampling_error is handler

        configure(on_sampling_error=None)
        assert DEFAULT_CONFIG.on_sampling_error is None
    finally:
        DEFAULT_CONFIG.on_sampling_error = original


@pytest.mark.anyio
async def test_configure_on_error():
    """configure() can set on_error on DEFAULT_CONFIG."""
    original = DEFAULT_CONFIG.on_error
    try:

        def handler(
            exc: Exception,
            ctx: EvaluatorContext[Any, Any, Any],
            evaluator: Evaluator,
            location: OnErrorLocation,
        ) -> None:
            pass

        configure(on_error=handler)
        assert DEFAULT_CONFIG.on_error is handler

        configure(on_error=None)
        assert DEFAULT_CONFIG.on_error is None
    finally:
        DEFAULT_CONFIG.on_error = original


# --- SamplingContext tests ---


@pytest.mark.anyio
async def test_sampling_context_passed_to_callable():
    """SamplingContext is passed to sample_rate callables with correct data."""
    captured_contexts: list[SamplingContext] = []

    def capture_rate(ctx: SamplingContext) -> bool:
        captured_contexts.append(ctx)
        return True

    collector = Collector()
    config = OnlineEvalConfig(
        default_sink=collector,
        metadata={'service': 'test'},
    )

    @config.evaluate(OnlineEvaluator(evaluator=AlwaysTrue(), sample_rate=capture_rate))
    async def my_func(x: int, y: str = 'hello') -> int:
        return x

    await my_func(42, y='world')
    await wait_for_evaluations()

    assert len(captured_contexts) == 1
    ctx = captured_contexts[0]
    assert ctx.inputs == {'x': 42, 'y': 'world'}
    assert ctx.metadata == {'service': 'test'}
    assert isinstance(ctx.evaluator, AlwaysTrue)
    assert 0.0 <= ctx.call_seed < 1.0


@pytest.mark.anyio
async def test_sampling_context_call_seed_shared_across_evaluators():
    """call_seed is the same for all evaluators in a single call."""
    captured_seeds: list[float] = []

    def capture_rate(ctx: SamplingContext) -> bool:
        captured_seeds.append(ctx.call_seed)
        return True

    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(
        OnlineEvaluator(evaluator=AlwaysTrue(), sample_rate=capture_rate),
        OnlineEvaluator(evaluator=AlwaysFalse(), sample_rate=capture_rate),
    )
    async def my_func(x: int) -> int:
        return x

    await my_func(42)
    await wait_for_evaluations()

    assert len(captured_seeds) == 2
    assert captured_seeds[0] == captured_seeds[1]


@pytest.mark.anyio
async def test_sampling_context_input_based_sampling():
    """sample_rate callable can use inputs to decide whether to evaluate."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    def only_large_inputs(ctx: SamplingContext) -> bool:
        return ctx.inputs.get('x', 0) > 10

    @config.evaluate(OnlineEvaluator(evaluator=AlwaysTrue(), sample_rate=only_large_inputs))
    async def my_func(x: int) -> int:
        return x

    await my_func(5)  # should be skipped
    await wait_for_evaluations()
    assert len(collector.calls) == 0

    await my_func(20)  # should be evaluated
    await wait_for_evaluations()
    assert len(collector.calls) == 1


# --- correlated sampling tests ---


@pytest.mark.anyio
async def test_correlated_sampling_subset_property():
    """In correlated mode, lower-rate evaluator calls are a subset of higher-rate ones."""
    collector_high = Collector()
    collector_low = Collector()
    config = OnlineEvalConfig(sampling_mode='correlated')

    @config.evaluate(
        OnlineEvaluator(evaluator=AlwaysTrue(), sample_rate=0.5, sink=collector_high),
        OnlineEvaluator(evaluator=AlwaysFalse(), sample_rate=0.1, sink=collector_low),
    )
    async def my_func(x: int) -> int:
        return x

    for i in range(100):
        await my_func(i)

    await wait_for_evaluations()

    # In correlated mode, every call that ran the low-rate evaluator (0.1)
    # must also have run the high-rate evaluator (0.5)
    assert len(collector_low.calls) <= len(collector_high.calls)
    # Sanity: we should have gotten some evaluations
    assert len(collector_high.calls) > 0
    assert len(collector_low.calls) > 0


@pytest.mark.anyio
async def test_correlated_sampling_max_overhead():
    """In correlated mode, total overhead probability equals max(rate_i)."""
    collector1 = Collector()
    collector2 = Collector()
    collector3 = Collector()
    config = OnlineEvalConfig(sampling_mode='correlated')

    @config.evaluate(
        OnlineEvaluator(evaluator=AlwaysTrue(), sample_rate=0.1, sink=collector1),
        OnlineEvaluator(evaluator=AlwaysFalse(), sample_rate=0.1, sink=collector2),
        OnlineEvaluator(evaluator=OutputEquals(value=0), sample_rate=0.1, sink=collector3),
    )
    async def my_func(x: int) -> int:
        return x

    for i in range(200):
        await my_func(i)

    await wait_for_evaluations()

    # All three should have run on exactly the same calls (same rate, same seed)
    assert len(collector1.calls) == len(collector2.calls) == len(collector3.calls)
    # Should be roughly 10% of 200 = ~20, not ~27% like independent mode
    assert 5 < len(collector1.calls) < 40


@pytest.mark.anyio
async def test_independent_sampling_is_default():
    """Independent mode is the default — evaluators sample independently."""
    collector1 = Collector()
    collector2 = Collector()
    config = OnlineEvalConfig()  # default is 'independent'

    @config.evaluate(
        OnlineEvaluator(evaluator=AlwaysTrue(), sample_rate=0.5, sink=collector1),
        OnlineEvaluator(evaluator=AlwaysFalse(), sample_rate=0.5, sink=collector2),
    )
    async def my_func(x: int) -> int:
        return x

    for i in range(100):
        await my_func(i)

    await wait_for_evaluations()

    # Both should have roughly 50 each, but they should NOT be identical
    # (extremely unlikely with independent sampling over 100 trials)
    assert len(collector1.calls) > 0
    assert len(collector2.calls) > 0


@pytest.mark.anyio
async def test_configure_sampling_mode():
    """configure() can set sampling_mode on DEFAULT_CONFIG."""
    original = DEFAULT_CONFIG.sampling_mode
    try:
        configure(sampling_mode='correlated')
        assert DEFAULT_CONFIG.sampling_mode == 'correlated'

        configure(sampling_mode='independent')
        assert DEFAULT_CONFIG.sampling_mode == 'independent'
    finally:
        DEFAULT_CONFIG.sampling_mode = original


# --- attributes/metrics tests ---


@pytest.mark.anyio
async def test_set_eval_attribute_in_async_function():
    """set_eval_attribute in an async decorated function propagates to EvaluatorContext."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(AlwaysTrue())
    async def my_func(x: int) -> int:
        set_eval_attribute('model', 'gpt-4o')
        set_eval_attribute('region', 'us-east-1')
        return x * 2

    result = await my_func(21)
    assert result == 42

    await wait_for_evaluations()

    assert len(collector.calls) == 1
    _, _, ctx = collector.calls[0]
    assert ctx.attributes == {'model': 'gpt-4o', 'region': 'us-east-1'}


@pytest.mark.anyio
async def test_increment_eval_metric_in_async_function():
    """increment_eval_metric in an async decorated function propagates to EvaluatorContext."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(AlwaysTrue())
    async def my_func(x: int) -> int:
        increment_eval_metric('tokens', 150)
        increment_eval_metric('tokens', 50)
        increment_eval_metric('requests', 1)
        return x

    result = await my_func(42)
    assert result == 42

    await wait_for_evaluations()

    assert len(collector.calls) == 1
    _, _, ctx = collector.calls[0]
    assert ctx.metrics == {'tokens': 200, 'requests': 1}


@pytest.mark.anyio
async def test_set_eval_attribute_in_sync_function():
    """set_eval_attribute in a sync decorated function propagates to EvaluatorContext."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(AlwaysTrue())
    def my_func(x: int) -> int:
        set_eval_attribute('model', 'gpt-4o')
        return x * 2

    result = my_func(21)
    assert result == 42

    await wait_for_evaluations()

    assert len(collector.calls) == 1
    _, _, ctx = collector.calls[0]
    assert ctx.attributes == {'model': 'gpt-4o'}


@pytest.mark.anyio
async def test_increment_eval_metric_in_sync_function():
    """increment_eval_metric in a sync decorated function propagates to EvaluatorContext."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(AlwaysTrue())
    def my_func(x: int) -> int:
        increment_eval_metric('tokens', 100)
        return x

    result = my_func(42)
    assert result == 42

    await wait_for_evaluations()

    assert len(collector.calls) == 1
    _, _, ctx = collector.calls[0]
    assert ctx.metrics == {'tokens': 100}


@pytest.mark.anyio
async def test_attributes_and_metrics_empty_by_default():
    """When no attributes/metrics are set, context has empty dicts."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(AlwaysTrue())
    async def my_func(x: int) -> int:
        return x

    await my_func(42)
    await wait_for_evaluations()

    assert len(collector.calls) == 1
    _, _, ctx = collector.calls[0]
    assert ctx.attributes == {}
    assert ctx.metrics == {}


@pytest.mark.anyio
async def test_online_eval_suppressed_inside_task_run():
    """Online evaluation is suppressed when already inside `CURRENT_TASK_RUN`."""
    from pydantic_evals._task_run import CURRENT_TASK_RUN, TaskRun

    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(AlwaysTrue())
    async def my_func(x: int) -> int:
        return x

    # Simulate being inside Dataset.evaluate by setting CURRENT_TASK_RUN.
    outer_task_run = TaskRun()
    token = CURRENT_TASK_RUN.set(outer_task_run)
    try:
        result = await my_func(42)
        assert result == 42
    finally:
        CURRENT_TASK_RUN.reset(token)

    await wait_for_evaluations()
    # Online evaluation should have been suppressed
    assert len(collector.calls) == 0


@pytest.mark.anyio
async def test_metadata_not_shared_between_contexts():
    """config.metadata is copied so sinks can't corrupt the config."""
    collected_contexts: list[EvaluatorContext[Any, Any, Any]] = []

    async def capture_sink(
        results: Sequence[EvaluationResult[Any]],
        failures: Sequence[EvaluatorFailure],
        context: EvaluatorContext[Any, Any, Any],
    ) -> None:
        collected_contexts.append(context)
        # Mutate the context's metadata — should NOT affect the config
        if context.metadata is not None:  # pragma: no branch
            context.metadata['injected'] = True

    config = OnlineEvalConfig(
        default_sink=capture_sink,
        metadata={'service': 'test'},
    )

    @config.evaluate(AlwaysTrue())
    async def my_func(x: int) -> int:
        return x

    await my_func(1)
    await wait_for_evaluations()
    await my_func(2)
    await wait_for_evaluations()

    assert len(collected_contexts) == 2
    # Both contexts' metadata were mutated by the sink
    assert collected_contexts[0].metadata == {'service': 'test', 'injected': True}
    assert collected_contexts[1].metadata == {'service': 'test', 'injected': True}
    # But config metadata should be untouched — the copies are independent
    assert config.metadata == {'service': 'test'}


# --- Call span / instrument-style recording ---------------------------------


@needs_logfire
@pytest.mark.anyio
async def test_call_span_default_name_and_no_args(capfire: CaptureLogfire):
    """Each decorated call opens a span named after the function; args/return are not recorded by default."""
    config = OnlineEvalConfig(emit_otel_events=False)

    @config.evaluate(AlwaysTrue())
    async def my_func(x: int) -> int:
        return x

    await my_func(42)
    await wait_for_evaluations()

    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    call_spans = [s for s in spans if 'my_func' in s['name']]
    assert len(call_spans) == 1
    attrs = call_spans[0]['attributes']
    assert 'x' not in attrs
    assert 'return' not in attrs


@needs_logfire
@pytest.mark.anyio
async def test_call_span_extract_args_true_records_all(capfire: CaptureLogfire):
    """`extract_args=True` records every bound argument."""
    config = OnlineEvalConfig(emit_otel_events=False)

    @config.evaluate(AlwaysTrue(), extract_args=True)
    async def my_func(x: int, label: str = 'default') -> int:
        return x

    await my_func(42, label='hello')
    await wait_for_evaluations()

    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    call_spans = [s for s in spans if 'my_func' in s['name']]
    assert len(call_spans) == 1
    attrs = call_spans[0]['attributes']
    assert attrs['x'] == 42
    assert attrs['label'] == 'hello'


@needs_logfire
@pytest.mark.anyio
async def test_call_span_extract_args_subset(capfire: CaptureLogfire):
    """Passing a list to `extract_args` records only the named arguments."""
    config = OnlineEvalConfig(emit_otel_events=False)

    @config.evaluate(AlwaysTrue(), extract_args=['x'])
    async def my_func(x: int, secret: str) -> int:
        return x

    await my_func(42, secret='shh')
    await wait_for_evaluations()

    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    call_spans = [s for s in spans if 'my_func' in s['name']]
    assert len(call_spans) == 1
    attrs = call_spans[0]['attributes']
    assert attrs['x'] == 42
    assert 'secret' not in attrs


@needs_logfire
@pytest.mark.anyio
async def test_call_span_record_return(capfire: CaptureLogfire):
    """`record_return=True` records the function's return value on the span."""
    config = OnlineEvalConfig(emit_otel_events=False)

    @config.evaluate(AlwaysTrue(), record_return=True)
    async def my_func(x: int) -> int:
        return x * 2

    await my_func(21)
    await wait_for_evaluations()

    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    call_spans = [s for s in spans if 'my_func' in s['name']]
    assert len(call_spans) == 1
    assert call_spans[0]['attributes']['return'] == 42


@needs_logfire
@pytest.mark.anyio
async def test_call_span_msg_template_and_span_name(capfire: CaptureLogfire):
    """`msg_template`/`span_name` override the default span name."""
    config = OnlineEvalConfig(emit_otel_events=False)

    @config.evaluate(AlwaysTrue(), msg_template='running my task', span_name='task.run')
    async def my_func(x: int) -> int:
        return x

    await my_func(1)
    await wait_for_evaluations()

    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    names = [s['name'] for s in spans]
    assert 'task.run' in names


@needs_logfire
@pytest.mark.anyio
async def test_evaluation_events_parented_to_call_span(capfire: CaptureLogfire):
    """Emitted `gen_ai.evaluation.result` events parent to the decorated call's span."""
    config = OnlineEvalConfig()  # default emit_otel_events=True

    @config.evaluate(AlwaysTrue())
    async def my_func(x: int) -> int:
        return x

    await my_func(1)
    await wait_for_evaluations()

    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    call_spans = [s for s in spans if 'my_func' in s['name']]
    assert len(call_spans) == 1
    call_span_id = call_spans[0]['context']['span_id']

    finished = capfire.log_exporter.get_finished_logs()
    assert len(finished) == 1
    # Event's span_context points back to the decorated call's span id,
    # so the evaluator event appears nested under the function call in traces.
    assert finished[0].log_record.span_id == call_span_id


def test_extract_args_without_logfire_raises(monkeypatch: pytest.MonkeyPatch):
    """Opting into arg/return recording without logfire installed raises at decoration time."""
    from pydantic_evals import online as online_module

    monkeypatch.setattr(online_module, '_LOGFIRE_INSTALLED', False)

    with pytest.raises(RuntimeError, match='logfire'):

        @online_module.evaluate(AlwaysTrue(), extract_args=True)
        async def f(x: int) -> int:
            return x

    with pytest.raises(RuntimeError, match='logfire'):

        @online_module.evaluate(AlwaysTrue(), record_return=True)
        async def g(x: int) -> int:
            return x


@needs_logfire
def test_extract_args_unknown_parameter_raises():
    """Naming an unknown parameter in `extract_args` fails at decoration time."""
    with pytest.raises(ValueError, match='not in'):

        @evaluate(AlwaysTrue(), extract_args=['nonexistent'])
        async def f(x: int) -> int:
            return x


@pytest.mark.anyio
async def test_sync_call_span_with_extract_args(capfire: CaptureLogfire):
    """Sync decorated functions also open a span and honour `extract_args`."""
    if not logfire_import_successful():
        pytest.skip('logfire not installed')

    config = OnlineEvalConfig(emit_otel_events=False)

    @config.evaluate(AlwaysTrue(), extract_args=True, record_return=True)
    def my_func(x: int) -> int:
        return x * 2

    assert my_func(21) == 42
    await wait_for_evaluations()

    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    call_spans = [s for s in spans if 'my_func' in s['name']]
    assert len(call_spans) == 1
    attrs = call_spans[0]['attributes']
    assert attrs['x'] == 21
    assert attrs['return'] == 42


# --- Evaluator span: parenting and result attributes -----------------------


@needs_logfire
@pytest.mark.anyio
async def test_evaluator_span_nested_under_call_span(capfire: CaptureLogfire):
    """The `evaluator: {name}` span created in `run_evaluator` parents to the call span."""
    config = OnlineEvalConfig(emit_otel_events=False)

    @config.evaluate(AlwaysTrue())
    async def my_func(x: int) -> int:
        return x

    await my_func(1)
    await wait_for_evaluations()

    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    call_spans = [s for s in spans if 'my_func' in s['name']]
    evaluator_spans = [s for s in spans if s['name'] == 'evaluator: {evaluator_name}']
    assert len(call_spans) == 1
    assert len(evaluator_spans) == 1
    assert evaluator_spans[0]['parent']['span_id'] == call_spans[0]['context']['span_id']


@needs_logfire
@pytest.mark.anyio
async def test_evaluator_span_carries_score_attributes(capfire: CaptureLogfire):
    """Single-result evaluators expose their score on the evaluator span."""
    config = OnlineEvalConfig(emit_otel_events=False)

    @config.evaluate(AlwaysTrue())
    async def my_func(x: int) -> int:
        return x

    await my_func(1)
    await wait_for_evaluations()

    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    eval_spans = [s for s in spans if s['name'] == 'evaluator: {evaluator_name}']
    assert len(eval_spans) == 1
    attrs = eval_spans[0]['attributes']
    assert attrs['gen_ai.evaluation.name'] == 'AlwaysTrue'
    assert attrs['gen_ai.evaluation.score.value'] == 1.0
    assert attrs['gen_ai.evaluation.score.label'] == 'pass'
    assert 'gen_ai.evaluation.evaluator_source' in attrs


@needs_logfire
@pytest.mark.anyio
async def test_evaluator_span_multi_result_summary(capfire: CaptureLogfire):
    """Multi-result evaluators record a count and the result names on the span."""
    config = OnlineEvalConfig(emit_otel_events=False)

    @config.evaluate(MultiResultEvaluator())
    async def my_func(x: int) -> int:
        return x

    await my_func(1)
    await wait_for_evaluations()

    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    eval_spans = [s for s in spans if s['name'] == 'evaluator: {evaluator_name}']
    assert len(eval_spans) == 1
    attrs = eval_spans[0]['attributes']
    assert attrs['gen_ai.evaluation.result_count'] == 3
    assert set(attrs['gen_ai.evaluation.names']) == {'accuracy', 'score', 'label'}
    assert 'gen_ai.evaluation.score.value' not in attrs


# --- Baggage propagation ---------------------------------------------------


@needs_logfire
@pytest.mark.anyio
async def test_baggage_attached_to_evaluation_event(capfire: CaptureLogfire):
    """Baggage set in the calling context propagates onto emitted evaluation events."""
    from opentelemetry import baggage as ot_baggage, context as ot_context

    config = OnlineEvalConfig()  # emit_otel_events=True

    @config.evaluate(AlwaysTrue())
    async def my_func(x: int) -> int:
        return x

    bag_ctx = ot_baggage.set_baggage('tenant', 'acme')
    token = ot_context.attach(bag_ctx)
    try:
        await my_func(1)
        await wait_for_evaluations()
    finally:
        ot_context.detach(token)

    finished = capfire.log_exporter.get_finished_logs()
    assert len(finished) == 1
    attrs = dict(finished[0].log_record.attributes or {})
    assert attrs['tenant'] == 'acme'


@needs_logfire
@pytest.mark.anyio
async def test_baggage_disabled_via_config(capfire: CaptureLogfire):
    """`include_baggage=False` keeps baggage out of emitted events."""
    from opentelemetry import baggage as ot_baggage, context as ot_context

    config = OnlineEvalConfig(include_baggage=False)

    @config.evaluate(AlwaysTrue())
    async def my_func(x: int) -> int:
        return x

    bag_ctx = ot_baggage.set_baggage('tenant', 'acme')
    token = ot_context.attach(bag_ctx)
    try:
        await my_func(1)
        await wait_for_evaluations()
    finally:
        ot_context.detach(token)

    attrs = dict(capfire.log_exporter.get_finished_logs()[0].log_record.attributes or {})
    assert 'tenant' not in attrs
