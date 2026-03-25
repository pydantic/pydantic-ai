"""Tests for pydantic_evals.online — online evaluation infrastructure."""

from __future__ import annotations as _annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_evals.evaluators import EvaluationResult, Evaluator, EvaluatorContext, EvaluatorFailure
    from pydantic_evals.evaluators.evaluator import EvaluatorOutput
    from pydantic_evals.online import (
        DEFAULT_CONFIG,
        CallbackSink,
        OnErrorLocation,
        OnlineEvalConfig,
        OnlineEvaluator,
        SpanReference,
        configure,
        disable_evaluation,
        evaluate,
        run_evaluators,
        wait_for_evaluations,
    )
    from pydantic_evals.otel.span_tree import SpanTree

pytestmark = pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed')


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

    await sink.submit(results=results, failures=[], context=ctx, span_reference=None)

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

    await sink.submit(results=results, failures=[], context=ctx, span_reference=None)
    assert len(collector.calls) == 1


@pytest.mark.anyio
async def test_callback_sink_ignores_span_reference():
    """CallbackSink does not pass span_reference to the callback."""
    collector = Collector()
    sink = CallbackSink(collector)
    ctx = _make_context(output='hello')
    span_ref = SpanReference(trace_id='abc', span_id='def')

    await sink.submit(results=[], failures=[], context=ctx, span_reference=span_ref)
    assert len(collector.calls) == 1
    assert collector.result_count == 0


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
    assert online.gate is None


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

    def dynamic_rate() -> float:
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

    @config.evaluate(OnlineEvaluator(evaluator=AlwaysTrue(), sample_rate=lambda: False))
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
async def test_gate_prevents_evaluation():
    """Gate returning False prevents evaluation."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(OnlineEvaluator(evaluator=AlwaysTrue(), gate=lambda ctx: False))
    async def my_func(x: int) -> int:
        return x

    await my_func(42)
    await wait_for_evaluations()
    assert len(collector.calls) == 0


@pytest.mark.anyio
async def test_gate_allows_evaluation():
    """Gate returning True allows evaluation."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(OnlineEvaluator(evaluator=AlwaysTrue(), gate=lambda ctx: True))
    async def my_func(x: int) -> int:
        return x

    await my_func(42)
    await wait_for_evaluations()
    assert len(collector.calls) == 1


@pytest.mark.anyio
async def test_gate_async():
    """Async gate functions work."""
    collector = Collector()

    async def my_gate(ctx: EvaluatorContext[Any, Any, Any]) -> bool:
        return ctx.output > 10

    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(OnlineEvaluator(evaluator=AlwaysTrue(), gate=my_gate))
    async def my_func(x: int) -> int:
        return x

    await my_func(5)  # output=5, gate should block
    await wait_for_evaluations()
    assert len(collector.calls) == 0

    await my_func(20)  # output=20, gate should allow
    await wait_for_evaluations()
    assert len(collector.calls) == 1


@pytest.mark.anyio
async def test_gate_exception_skips_evaluator():
    """Gate that raises an exception skips the evaluator gracefully."""
    collector = Collector()

    def bad_gate(ctx: EvaluatorContext[Any, Any, Any]) -> bool:
        raise ValueError('gate error')

    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(OnlineEvaluator(evaluator=AlwaysTrue(), gate=bad_gate))
    async def my_func(x: int) -> int:
        return x

    result = await my_func(42)
    assert result == 42

    await wait_for_evaluations()
    assert len(collector.calls) == 0


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
async def test_no_sink_skips_evaluators():
    """When no sink is configured, evaluators are skipped entirely."""
    config = OnlineEvalConfig()  # no sink

    @config.evaluate(AlwaysTrue())
    async def my_func(x: int) -> int:
        return x

    result = await my_func(42)
    assert result == 42
    await wait_for_evaluations()


@pytest.mark.anyio
async def test_configure_updates_default_config():
    """configure() updates the global DEFAULT_CONFIG."""
    original_enabled = DEFAULT_CONFIG.enabled
    original_sink = DEFAULT_CONFIG.default_sink
    original_rate = DEFAULT_CONFIG.default_sample_rate

    original_on_max = DEFAULT_CONFIG.on_max_concurrency

    try:
        configure(enabled=False, default_sample_rate=0.5)
        assert DEFAULT_CONFIG.enabled is False
        assert DEFAULT_CONFIG.default_sample_rate == 0.5
        assert DEFAULT_CONFIG.default_sink is original_sink

        def handler(ctx: EvaluatorContext[Any, Any, Any]) -> None:
            pass

        configure(on_max_concurrency=handler)
        assert DEFAULT_CONFIG.on_max_concurrency is handler
    finally:
        DEFAULT_CONFIG.enabled = original_enabled
        DEFAULT_CONFIG.default_sink = original_sink
        DEFAULT_CONFIG.default_sample_rate = original_rate
        DEFAULT_CONFIG.on_max_concurrency = original_on_max


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
async def test_sample_rate_callable_exception_skips_evaluator():
    """Exception in sample_rate callable skips the evaluator without breaking the function."""
    collector = Collector()

    def bad_rate() -> float:
        raise ValueError('rate error')

    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(OnlineEvaluator(evaluator=AlwaysTrue(), sample_rate=bad_rate))
    async def my_func(x: int) -> int:
        return x

    result = await my_func(42)
    assert result == 42

    await wait_for_evaluations()
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


@pytest.mark.anyio
async def test_span_reference_with_configured_logfire(capfire: Any):
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
async def test_sync_decorated_function_gate():
    """Sync gates work when called from async context."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(OnlineEvaluator(evaluator=AlwaysTrue(), gate=lambda ctx: ctx.output > 10))
    def my_func(x: int) -> int:
        return x

    my_func(5)  # gate blocks
    await wait_for_evaluations()
    assert len(collector.calls) == 0

    my_func(20)  # gate allows
    await wait_for_evaluations()
    assert len(collector.calls) == 1


@pytest.mark.anyio
async def test_sync_decorated_function_async_gate():
    """Async gates work with sync decorated functions."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    async def async_gate(ctx: EvaluatorContext[Any, Any, Any]) -> bool:
        return ctx.output > 10

    @config.evaluate(OnlineEvaluator(evaluator=AlwaysTrue(), gate=async_gate))
    def my_func(x: int) -> int:
        return x

    my_func(5)  # gate blocks
    await wait_for_evaluations()
    assert len(collector.calls) == 0

    my_func(20)  # gate allows
    await wait_for_evaluations()
    assert len(collector.calls) == 1


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
async def test_sync_decorated_function_gate_exception():
    """Gate exception skips evaluator gracefully for sync decorated functions."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    def bad_gate(ctx: EvaluatorContext[Any, Any, Any]) -> bool:
        raise ValueError('gate error')

    @config.evaluate(OnlineEvaluator(evaluator=AlwaysTrue(), gate=bad_gate))
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
    config = OnlineEvalConfig(default_sink=[collector1, CallbackSink(collector2)])  # type: ignore[arg-type]

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
async def test_on_error_gate_exception():
    """on_error is called with 'gate' location when gate raises."""
    errors: list[tuple[Exception, OnErrorLocation]] = []

    def on_error(
        exc: Exception,
        ctx: EvaluatorContext[Any, Any, Any],
        evaluator: Evaluator,
        location: OnErrorLocation,
    ) -> None:
        errors.append((exc, location))

    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector, on_error=on_error)

    def bad_gate(ctx: EvaluatorContext[Any, Any, Any]) -> bool:
        raise ValueError('gate boom')

    @config.evaluate(OnlineEvaluator(evaluator=AlwaysTrue(), gate=bad_gate))
    async def my_func(x: int) -> int:
        return x

    result = await my_func(42)
    assert result == 42

    await wait_for_evaluations()

    assert len(errors) == 1
    assert isinstance(errors[0][0], ValueError)
    assert str(errors[0][0]) == 'gate boom'
    assert errors[0][1] == 'gate'
    # Evaluator should not have run
    assert len(collector.calls) == 0


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
    collector = Collector()

    def bad_on_error(
        exc: Exception,
        ctx: EvaluatorContext[Any, Any, Any],
        evaluator: Evaluator,
        location: OnErrorLocation,
    ) -> None:
        raise RuntimeError('handler boom')

    def bad_gate(ctx: EvaluatorContext[Any, Any, Any]) -> bool:
        raise ValueError('gate boom')

    config = OnlineEvalConfig(default_sink=collector, on_error=bad_on_error)

    @config.evaluate(
        AlwaysTrue(),  # this one should still run
        OnlineEvaluator(evaluator=AlwaysFalse(), gate=bad_gate),  # gate fails, on_error fails
    )
    async def my_func(x: int) -> int:
        return x

    result = await my_func(42)
    assert result == 42

    await wait_for_evaluations()

    # AlwaysTrue should still have run despite the other evaluator's error chain
    assert len(collector.calls) >= 1


@pytest.mark.anyio
async def test_on_error_per_evaluator_overrides_config():
    """Per-evaluator on_error overrides the config default."""
    evaluator_errors: list[OnErrorLocation] = []

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

    def bad_gate(ctx: EvaluatorContext[Any, Any, Any]) -> bool:
        raise ValueError('gate boom')

    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector, on_error=config_on_error)

    @config.evaluate(
        OnlineEvaluator(evaluator=AlwaysTrue(), gate=bad_gate, on_error=evaluator_on_error),
    )
    async def my_func(x: int) -> int:
        return x

    await my_func(42)
    await wait_for_evaluations()

    assert len(evaluator_errors) == 1
    assert evaluator_errors[0] == 'gate'


@pytest.mark.anyio
async def test_on_error_async_callback():
    """Async on_error callback works."""
    errors: list[OnErrorLocation] = []

    async def async_on_error(
        exc: Exception,
        ctx: EvaluatorContext[Any, Any, Any],
        evaluator: Evaluator,
        location: OnErrorLocation,
    ) -> None:
        await asyncio.sleep(0)
        errors.append(location)

    def bad_gate(ctx: EvaluatorContext[Any, Any, Any]) -> bool:
        raise ValueError('gate boom')

    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector, on_error=async_on_error)

    @config.evaluate(OnlineEvaluator(evaluator=AlwaysTrue(), gate=bad_gate))
    async def my_func(x: int) -> int:
        return x

    await my_func(42)
    await wait_for_evaluations()

    assert len(errors) == 1
    assert errors[0] == 'gate'


@pytest.mark.anyio
async def test_gate_exception_does_not_cancel_sibling_evaluators():
    """A gate exception in one evaluator doesn't prevent siblings from running."""
    collector = Collector()

    def bad_gate(ctx: EvaluatorContext[Any, Any, Any]) -> bool:
        raise ValueError('gate boom')

    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(
        OnlineEvaluator(evaluator=AlwaysFalse(), gate=bad_gate),  # gate will fail
        AlwaysTrue(),  # should still run
    )
    async def my_func(x: int) -> int:
        return x

    await my_func(42)
    await wait_for_evaluations()

    # AlwaysTrue should have run despite the sibling's gate failure
    assert len(collector.calls) >= 1
    assert any(r.value is True for results, _, _ in collector.calls for r in results)


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
