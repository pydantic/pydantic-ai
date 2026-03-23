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
        EvaluatorContextData,
        OnlineEvalConfig,
        OnlineEvaluator,
        SpanReference,
        _extract_span_reference,  # pyright: ignore[reportPrivateUsage]
        configure,
        disable_evaluation,
        evaluate,
        rebuild_context,
        rebuild_contexts,
        run_evaluators,
        wait_for_evaluations,
    )
    from pydantic_evals.otel.span_tree import SpanTree

pytestmark = [pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed'), pytest.mark.anyio]


if TYPE_CHECKING or imports_successful():
    # ============================================================================
    # Test evaluators
    # ============================================================================

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

    # ============================================================================
    # Helpers
    # ============================================================================

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

        def __init__(self, data: dict[str, EvaluatorContextData]) -> None:
            self._data = data

        async def fetch(self, span: SpanReference) -> EvaluatorContextData:
            return self._data[span.span_id]

        async def fetch_many(self, spans: Sequence[SpanReference]) -> list[EvaluatorContextData]:
            return [self._data[s.span_id] for s in spans]


# ============================================================================
# Test CallbackSink
# ============================================================================


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


async def test_callback_sink_async():
    """CallbackSink works with async callbacks."""
    collector = Collector()
    sink = CallbackSink(collector)
    ctx = _make_context(output='hello')
    results = [EvaluationResult(name='test', value=True, reason=None, source=AlwaysTrue().as_spec())]

    await sink.submit(results=results, failures=[], context=ctx, span_reference=None)
    assert len(collector.calls) == 1


async def test_callback_sink_ignores_span_reference():
    """CallbackSink does not pass span_reference to the callback."""
    collector = Collector()
    sink = CallbackSink(collector)
    ctx = _make_context(output='hello')
    span_ref = SpanReference(trace_id='abc', span_id='def')

    await sink.submit(results=[], failures=[], context=ctx, span_reference=span_ref)
    assert len(collector.calls) == 1
    assert collector.result_count == 0


# ============================================================================
# Test SpanReference
# ============================================================================


async def test_span_reference():
    """SpanReference stores trace and span IDs."""
    ref = SpanReference(trace_id='abc123', span_id='def456')
    assert ref.trace_id == 'abc123'
    assert ref.span_id == 'def456'


# ============================================================================
# Test OnlineEvaluator
# ============================================================================


async def test_online_evaluator_defaults():
    """OnlineEvaluator has sensible defaults."""
    evaluator = AlwaysTrue()
    online = OnlineEvaluator(evaluator=evaluator)
    assert online.evaluator is evaluator
    assert online.sample_rate == 1.0
    assert online.sink is None
    assert online.max_concurrency == 10
    assert online.gate is None


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


# ============================================================================
# Test run_evaluators
# ============================================================================


async def test_run_evaluators_success():
    """run_evaluators returns results from all evaluators."""
    ctx = _make_context(output=42)
    results, failures = await run_evaluators([AlwaysTrue(), OutputEquals(value=42)], ctx)

    assert len(results) == 2
    assert len(failures) == 0
    assert results[0].value is True
    assert results[1].value is True


async def test_run_evaluators_with_failure():
    """run_evaluators collects failures separately from results."""
    ctx = _make_context(output=42)
    results, failures = await run_evaluators([AlwaysTrue(), FailingEvaluator()], ctx)

    assert len(results) == 1
    assert results[0].value is True
    assert len(failures) == 1
    assert 'Simulated evaluator failure' in failures[0].error_message


async def test_run_evaluators_empty():
    """run_evaluators handles empty evaluator list."""
    ctx = _make_context(output=42)
    results, failures = await run_evaluators([], ctx)
    assert results == []
    assert failures == []


async def test_run_evaluators_multi_result():
    """run_evaluators handles evaluators that return multiple results."""
    ctx = _make_context(output=42)
    results, failures = await run_evaluators([MultiResultEvaluator()], ctx)

    assert len(results) == 3
    assert len(failures) == 0
    result_names = {r.name for r in results}
    assert result_names == {'accuracy', 'score', 'label'}


async def test_run_evaluators_async_evaluator():
    """run_evaluators works with async evaluators."""
    ctx = _make_context(output=42)
    results, failures = await run_evaluators([AsyncEvaluator()], ctx)

    assert len(results) == 1
    assert results[0].value is True
    assert len(failures) == 0


# ============================================================================
# Test evaluate() decorator — async functions
# ============================================================================


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


async def test_evaluate_decorator_async_preserves_signature():
    """evaluate() decorator preserves the function's name and docs."""

    @evaluate(AlwaysTrue())
    async def my_func(x: int) -> int:
        """My docstring."""
        return x

    assert my_func.__name__ == 'my_func'
    assert my_func.__doc__ == 'My docstring.'
    assert await my_func(42) == 42


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


# ============================================================================
# Test evaluate() decorator — sync functions
# ============================================================================


async def test_evaluate_decorator_sync_basic():
    """evaluate() decorator works with sync functions."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(AlwaysTrue())
    def my_func(x: int) -> int:
        return x * 2

    result = my_func(21)
    assert result == 42

    # Give background task time to complete
    await wait_for_evaluations()

    assert len(collector.calls) == 1
    results, _, ctx = collector.calls[0]
    assert len(results) == 1
    assert results[0].value is True
    assert ctx.output == 42


# ============================================================================
# Test sampling
# ============================================================================


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


# ============================================================================
# Test disable_evaluation
# ============================================================================


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


# ============================================================================
# Test gating
# ============================================================================


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


# ============================================================================
# Test config enabled flag
# ============================================================================


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


# ============================================================================
# Test per-evaluator sink override
# ============================================================================


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


# ============================================================================
# Test no sink configured
# ============================================================================


async def test_no_sink_runs_without_error():
    """When no sink is configured, evaluators run but results are dropped silently."""
    config = OnlineEvalConfig()  # no sink

    @config.evaluate(AlwaysTrue())
    async def my_func(x: int) -> int:
        return x

    result = await my_func(42)
    assert result == 42
    await wait_for_evaluations()


# ============================================================================
# Test configure() module-level function
# ============================================================================


async def test_configure_updates_default_config():
    """configure() updates the global DEFAULT_CONFIG."""
    original_enabled = DEFAULT_CONFIG.enabled
    original_sink = DEFAULT_CONFIG.default_sink
    original_rate = DEFAULT_CONFIG.default_sample_rate

    try:
        configure(enabled=False, default_sample_rate=0.5)
        assert DEFAULT_CONFIG.enabled is False
        assert DEFAULT_CONFIG.default_sample_rate == 0.5
        assert DEFAULT_CONFIG.default_sink is original_sink
    finally:
        DEFAULT_CONFIG.enabled = original_enabled
        DEFAULT_CONFIG.default_sink = original_sink
        DEFAULT_CONFIG.default_sample_rate = original_rate


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


# ============================================================================
# Test module-level evaluate()
# ============================================================================


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


# ============================================================================
# Test rebuild_context and rebuild_contexts
# ============================================================================


async def test_rebuild_context():
    """rebuild_context builds an EvaluatorContext from stored data."""
    source = MockContextSource(
        {
            'span1': EvaluatorContextData(
                inputs={'query': 'hello'},
                output='world',
                metadata={'service': 'test'},
                duration=1.5,
                span_tree=SpanTree(),
            ),
        }
    )

    ctx = await rebuild_context(source, SpanReference(trace_id='trace1', span_id='span1'))

    assert ctx.inputs == {'query': 'hello'}
    assert ctx.output == 'world'
    assert ctx.expected_output is None
    assert ctx.metadata == {'service': 'test'}
    assert ctx.duration == 1.5
    assert ctx.attributes == {}
    assert ctx.metrics == {}


async def test_rebuild_contexts_batch():
    """rebuild_contexts builds multiple contexts in a single batch."""
    source = MockContextSource(
        {
            'span1': EvaluatorContextData(
                inputs={'q': '1'},
                output='a',
                metadata=None,
                duration=1.0,
                span_tree=SpanTree(),
            ),
            'span2': EvaluatorContextData(
                inputs={'q': '2'},
                output='b',
                metadata=None,
                duration=2.0,
                span_tree=SpanTree(),
            ),
        }
    )

    spans = [
        SpanReference(trace_id='t', span_id='span1'),
        SpanReference(trace_id='t', span_id='span2'),
    ]
    contexts = await rebuild_contexts(source, spans)

    assert len(contexts) == 2
    assert contexts[0].inputs == {'q': '1'}
    assert contexts[0].output == 'a'
    assert contexts[1].inputs == {'q': '2'}
    assert contexts[1].output == 'b'


async def test_rebuild_and_run_evaluators():
    """rebuild_context + run_evaluators works end-to-end."""
    source = MockContextSource(
        {
            'span1': EvaluatorContextData(
                inputs={},
                output=42,
                metadata=None,
                duration=0.1,
                span_tree=SpanTree(),
            ),
        }
    )

    ctx = await rebuild_context(source, SpanReference(trace_id='t', span_id='span1'))
    results, failures = await run_evaluators([OutputEquals(value=42), AlwaysTrue()], ctx)

    assert len(results) == 2
    assert len(failures) == 0
    assert all(r.value is True for r in results)


# ============================================================================
# Test OnlineEvalConfig metadata
# ============================================================================


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


# ============================================================================
# Test max_concurrency
# ============================================================================


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


# ============================================================================
# Test EvaluationSink protocol
# ============================================================================


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


# ============================================================================
# Test bare evaluator auto-wrapping
# ============================================================================


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


# ============================================================================
# Test multiple sinks
# ============================================================================


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


# ============================================================================
# Test fractional sample rate
# ============================================================================


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


# ============================================================================
# Test sample_rate callable exception
# ============================================================================


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


# ============================================================================
# Test sink exception handling
# ============================================================================


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


# ============================================================================
# Test sync function edge cases
# ============================================================================


async def test_sync_disabled_config():
    """Sync function with disabled config runs without evaluation."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector, enabled=False)

    @config.evaluate(AlwaysTrue())
    def my_func(x: int) -> int:
        return x * 2

    result = my_func(21)
    assert result == 42

    await wait_for_evaluations()
    assert len(collector.calls) == 0


async def test_sync_sample_rate_zero():
    """Sync function with sample_rate=0 runs without evaluation."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(OnlineEvaluator(evaluator=AlwaysTrue(), sample_rate=0.0))
    def my_func(x: int) -> int:
        return x * 2

    result = my_func(21)
    assert result == 42

    await wait_for_evaluations()
    assert len(collector.calls) == 0


async def test_sync_gate():
    """Sync function gate works correctly."""
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


async def test_sync_async_gate_works():
    """Async gate on sync function works — gates run in background async context."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    async def async_gate(ctx: EvaluatorContext[Any, Any, Any]) -> bool:
        return ctx.output > 10

    @config.evaluate(OnlineEvaluator(evaluator=AlwaysTrue(), gate=async_gate))
    def my_func(x: int) -> int:
        return x

    my_func(5)  # gate blocks (output=5 < 10)
    await wait_for_evaluations()
    assert len(collector.calls) == 0

    my_func(20)  # gate allows (output=20 > 10)
    await wait_for_evaluations()
    assert len(collector.calls) == 1


async def test_sync_gate_exception():
    """Sync function gate exception skips evaluator gracefully."""
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


# ============================================================================
# Test _extract_span_reference
# ============================================================================


async def test_extract_span_reference_with_logfire_span():
    """_extract_span_reference works with LogfireSpan (which is a ReadableSpan)."""
    from pydantic_evals._utils import logfire_span

    with logfire_span('test') as span:
        ref = _extract_span_reference(span)

    # LogfireSpan always has get_span_context(); when logfire is not configured,
    # trace_id and span_id are 0, so ref should be None.
    # When logfire IS configured, ref should be a SpanReference.
    # Either way, the method should not error.
    if ref is not None:
        assert isinstance(ref, SpanReference)
        assert len(ref.trace_id) == 32
        assert len(ref.span_id) == 16


async def test_extract_span_reference_with_otel_span():
    """_extract_span_reference works with a standard OTel SDK span."""
    from opentelemetry.sdk.trace import TracerProvider

    provider = TracerProvider()
    tracer = provider.get_tracer('test')

    with tracer.start_as_current_span('test') as span:
        ref = _extract_span_reference(span)

    assert ref is not None
    assert isinstance(ref, SpanReference)
    assert len(ref.trace_id) == 32
    assert len(ref.span_id) == 16


async def test_extract_span_reference_with_no_context():
    """_extract_span_reference returns None for objects without get_span_context."""
    assert _extract_span_reference(None) is None
    assert _extract_span_reference('not a span') is None
    assert _extract_span_reference(42) is None


async def test_extract_span_reference_with_zero_ids():
    """_extract_span_reference returns None when trace_id and span_id are zero."""

    class FakeSpan:
        def get_span_context(self):
            from opentelemetry.trace import SpanContext

            return SpanContext(trace_id=0, span_id=0, is_remote=False)

    assert _extract_span_reference(FakeSpan()) is None
