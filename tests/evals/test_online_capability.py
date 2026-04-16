"""Tests for OnlineEvaluation capability — agent integration for online evaluators."""

from __future__ import annotations as _annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel
    from pydantic_evals.evaluators import EvaluationResult, Evaluator, EvaluatorContext, EvaluatorFailure
    from pydantic_evals.evaluators.evaluator import EvaluatorOutput
    from pydantic_evals.online import (
        OnlineEvalConfig,
        OnlineEvaluator,
        SamplingContext,
        SpanReference,
        configure,
        disable_evaluation,
        wait_for_evaluations,
    )
    from pydantic_evals.online_capability import OnlineEvaluation

pytestmark = pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed')


if TYPE_CHECKING or imports_successful():

    @dataclass
    class AlwaysTrue(Evaluator):
        def evaluate(self, ctx: EvaluatorContext) -> EvaluatorOutput:
            return True

    @dataclass
    class OutputEquals(Evaluator):
        value: Any

        def evaluate(self, ctx: EvaluatorContext) -> EvaluatorOutput:
            return ctx.output == self.value

    @dataclass
    class FailingEvaluator(Evaluator):
        def evaluate(self, ctx: EvaluatorContext) -> EvaluatorOutput:
            raise ValueError('Simulated evaluator failure')

    class Collector:
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

    class SpanCollector:
        def __init__(self) -> None:
            self.span_refs: list[SpanReference | None] = []

        async def submit(
            self,
            *,
            results: Sequence[EvaluationResult[Any]],
            failures: Sequence[EvaluatorFailure],
            context: EvaluatorContext[Any, Any, Any],
            span_reference: SpanReference | None,
            target: Any,
        ) -> None:
            self.span_refs.append(span_reference)


@pytest.mark.anyio
async def test_basic_dispatch():
    """OnlineEvaluation dispatches evaluators after agent.run()."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    agent = Agent(
        TestModel(),
        capabilities=[OnlineEvaluation(evaluators=[AlwaysTrue()], config=config)],
    )

    result = await agent.run('hello')
    await wait_for_evaluations()

    assert result.output == 'success (no tool calls)'
    assert len(collector.calls) == 1
    results, failures, ctx = collector.calls[0]
    assert len(results) == 1
    assert results[0].value is True
    assert len(failures) == 0
    assert ctx.output == 'success (no tool calls)'


@pytest.mark.anyio
async def test_evaluator_context_fields():
    """EvaluatorContext is populated with correct agent run data."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector, metadata={'env': 'test'})

    agent = Agent(
        TestModel(),
        name='my-agent',
        capabilities=[
            OnlineEvaluation(evaluators=[AlwaysTrue()], config=config),
        ],
    )

    result = await agent.run('what is 2+2?')
    await wait_for_evaluations()

    assert len(collector.calls) == 1
    _, _, ctx = collector.calls[0]
    # Context name defaults to the agent run id (a ULID-ish string).
    assert isinstance(ctx.name, str) and ctx.name
    assert ctx.inputs == 'what is 2+2?'
    assert ctx.output == result.output
    assert ctx.expected_output is None
    assert ctx.duration > 0
    assert ctx.metadata is not None
    assert ctx.metadata['env'] == 'test'


@pytest.mark.anyio
async def test_usage_metrics(capfire: Any):
    """Span tree metrics are extracted into EvaluatorContext when instrumented."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    agent = Agent(
        TestModel(),
        capabilities=[OnlineEvaluation(evaluators=[AlwaysTrue()], config=config)],
        instrument=True,
    )

    await agent.run('hello')
    await wait_for_evaluations()

    assert len(collector.calls) == 1
    _, _, ctx = collector.calls[0]
    assert ctx.metrics.get('requests', 0) > 0


@pytest.mark.anyio
async def test_sampling_zero_rate():
    """Evaluators with sample_rate=0.0 are never dispatched."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    agent = Agent(
        TestModel(),
        capabilities=[
            OnlineEvaluation(
                evaluators=[OnlineEvaluator(evaluator=AlwaysTrue(), sample_rate=0.0)],
                config=config,
            ),
        ],
    )

    await agent.run('hello')
    await wait_for_evaluations()

    assert len(collector.calls) == 0


@pytest.mark.anyio
async def test_sampling_context_inputs_match_evaluator_inputs():
    """Sampling and evaluation see the same agent inputs value."""
    collector = Collector()
    sampled_inputs: list[Any] = []
    config = OnlineEvalConfig(default_sink=collector)

    def capture_rate(ctx: SamplingContext) -> bool:
        sampled_inputs.append(ctx.inputs)
        return True

    @dataclass
    class InputsMatchSampled(Evaluator):
        def evaluate(self, ctx: EvaluatorContext) -> EvaluatorOutput:
            return ctx.inputs == sampled_inputs[0]

    agent = Agent(
        TestModel(),
        capabilities=[
            OnlineEvaluation(
                evaluators=[OnlineEvaluator(evaluator=InputsMatchSampled(), sample_rate=capture_rate)],
                config=config,
            ),
        ],
    )

    await agent.run('hello')
    await wait_for_evaluations()

    assert sampled_inputs == ['hello']
    assert len(collector.calls) == 1
    results, failures, ctx = collector.calls[0]
    assert len(results) == 1
    assert results[0].value is True
    assert len(failures) == 0
    assert ctx.inputs == 'hello'


@pytest.mark.anyio
async def test_disable_evaluation():
    """disable_evaluation() context manager prevents dispatch."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    agent = Agent(
        TestModel(),
        capabilities=[OnlineEvaluation(evaluators=[AlwaysTrue()], config=config)],
    )

    with disable_evaluation():
        await agent.run('hello')
    await wait_for_evaluations()

    assert len(collector.calls) == 0


@pytest.mark.anyio
async def test_config_disabled():
    """Config with enabled=False prevents dispatch."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector, enabled=False)

    agent = Agent(
        TestModel(),
        capabilities=[OnlineEvaluation(evaluators=[AlwaysTrue()], config=config)],
    )

    result = await agent.run('hello')
    await wait_for_evaluations()

    assert result.output == 'success (no tool calls)'
    assert len(collector.calls) == 0


@pytest.mark.anyio
async def test_multiple_evaluators():
    """Multiple evaluators all dispatch concurrently."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    agent = Agent(
        TestModel(),
        capabilities=[
            OnlineEvaluation(
                evaluators=[AlwaysTrue(), OutputEquals(value='success (no tool calls)')],
                config=config,
            ),
        ],
    )

    await agent.run('hello')
    await wait_for_evaluations()

    assert len(collector.calls) == 2
    assert all(len(r) == 1 for r, _, _ in collector.calls)
    assert all(r[0].value is True for r, _, _ in collector.calls)


@pytest.mark.anyio
async def test_failing_evaluator_does_not_crash_agent():
    """Evaluator exceptions don't crash the agent run."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    agent = Agent(
        TestModel(),
        capabilities=[
            OnlineEvaluation(evaluators=[FailingEvaluator()], config=config),
        ],
    )

    result = await agent.run('hello')
    await wait_for_evaluations()

    assert result.output == 'success (no tool calls)'
    assert len(collector.calls) == 1
    results, failures, _ = collector.calls[0]
    assert len(results) == 0
    assert len(failures) == 1


@pytest.mark.anyio
async def test_metadata_merging():
    """Config metadata and run metadata are merged."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector, metadata={'config_key': 'config_val'})

    agent = Agent(
        TestModel(),
        capabilities=[OnlineEvaluation(evaluators=[AlwaysTrue()], config=config)],
    )

    await agent.run('hello', metadata={'run_key': 'run_val'})
    await wait_for_evaluations()

    assert len(collector.calls) == 1
    _, _, ctx = collector.calls[0]
    assert ctx.metadata is not None
    assert ctx.metadata['config_key'] == 'config_val'
    assert ctx.metadata['run_key'] == 'run_val'


@pytest.mark.anyio
async def test_empty_config_metadata_is_preserved():
    """An empty config metadata dict stays an empty dict instead of becoming None."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector, metadata={})

    agent = Agent(
        TestModel(),
        capabilities=[OnlineEvaluation(evaluators=[AlwaysTrue()], config=config)],
    )

    await agent.run('hello')
    await wait_for_evaluations()

    assert len(collector.calls) == 1
    _, _, ctx = collector.calls[0]
    assert ctx.metadata == {}


@pytest.mark.anyio
async def test_name_defaults_to_run_id():
    """EvaluatorContext name defaults to run_id when no name is provided."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    agent = Agent(
        TestModel(),
        capabilities=[OnlineEvaluation(evaluators=[AlwaysTrue()], config=config)],
    )

    result = await agent.run('hello')
    await wait_for_evaluations()

    assert len(collector.calls) == 1
    _, _, ctx = collector.calls[0]
    assert ctx.name == result.run_id


@pytest.mark.anyio
async def test_default_config_fallback():
    """OnlineEvaluation uses DEFAULT_CONFIG when no config is provided."""
    collector = Collector()

    configure(default_sink=collector)
    try:
        agent = Agent(
            TestModel(),
            capabilities=[OnlineEvaluation(evaluators=[AlwaysTrue()])],
        )

        await agent.run('hello')
        await wait_for_evaluations()

        assert len(collector.calls) == 1
    finally:
        configure(default_sink=None)


def test_serialization_name_is_none():
    """OnlineEvaluation opts out of spec serialization."""
    assert OnlineEvaluation.get_serialization_name() is None


@pytest.mark.anyio
async def test_streaming_dispatches_after_context_exit():
    """Streaming runs dispatch evaluations only after the stream context exits."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    agent = Agent(
        TestModel(),
        capabilities=[OnlineEvaluation(evaluators=[AlwaysTrue()], config=config)],
    )

    async with agent.run_stream('hello') as result:
        assert await result.get_output() == 'success (no tool calls)'
        assert len(collector.calls) == 0

    await wait_for_evaluations()

    assert len(collector.calls) == 1
    results, failures, ctx = collector.calls[0]
    assert len(results) == 1
    assert len(failures) == 0
    assert ctx.output == 'success (no tool calls)'


@pytest.mark.anyio
async def test_iter_dispatches_after_context_exit():
    """Iter-based runs dispatch evaluations only after the context exits."""
    collector = Collector()
    config = OnlineEvalConfig(default_sink=collector)

    agent = Agent(
        TestModel(),
        capabilities=[OnlineEvaluation(evaluators=[AlwaysTrue()], config=config)],
    )

    async with agent.iter('hello') as agent_run:
        async for _ in agent_run:
            pass
        assert agent_run.result is not None
        assert agent_run.result.output == 'success (no tool calls)'
        assert len(collector.calls) == 0

    await wait_for_evaluations()

    assert len(collector.calls) == 1
    results, failures, ctx = collector.calls[0]
    assert len(results) == 1
    assert len(failures) == 0
    assert ctx.output == 'success (no tool calls)'


@pytest.mark.anyio
async def test_span_reference_with_logfire(capfire: Any):
    """OnlineEvaluation produces a valid SpanReference when logfire is configured."""
    collector = SpanCollector()
    config = OnlineEvalConfig(default_sink=collector)

    agent = Agent(
        TestModel(),
        capabilities=[OnlineEvaluation(evaluators=[AlwaysTrue()], config=config)],
        instrument=True,
    )

    await agent.run('hello')
    await wait_for_evaluations()

    assert len(collector.span_refs) == 1
    ref = collector.span_refs[0]
    assert ref is not None
    assert isinstance(ref, SpanReference)
    assert len(ref.trace_id) == 32
    assert len(ref.span_id) == 16


@pytest.mark.parametrize(
    'traceparent',
    (
        'malformed',
        f'00-{"0" * 32}-1234567890abcdef-01',
        f'00-1234567890abcdef1234567890abcdef-{"0" * 16}-01',
    ),
)
@pytest.mark.anyio
async def test_malformed_traceparent_yields_no_span_reference(monkeypatch: pytest.MonkeyPatch, traceparent: str):
    """Malformed traceparents do not produce span references."""

    def fake_traceparent(self: Any, *, required: bool = True) -> str | None:
        return traceparent

    monkeypatch.setattr('pydantic_ai.run.AgentRunResult._traceparent', fake_traceparent)

    collector = SpanCollector()
    config = OnlineEvalConfig(default_sink=collector)

    agent = Agent(
        TestModel(),
        capabilities=[OnlineEvaluation(evaluators=[AlwaysTrue()], config=config)],
    )

    await agent.run('hello')
    await wait_for_evaluations()

    assert collector.span_refs == [None]
