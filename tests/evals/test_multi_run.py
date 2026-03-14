"""Tests for multi-run aggregation (repeat parameter)."""

from __future__ import annotations as _annotations

from dataclasses import dataclass

import pytest

from .._inline_snapshot import snapshot
from ..conftest import try_import

with try_import() as imports_successful:
    from logfire.testing import CaptureLogfire

    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import EvaluationResult, Evaluator, EvaluatorContext, EvaluatorOutput
    from pydantic_evals.reporting import (
        EvaluationReport,
        ReportCase,
        ReportCaseAggregate,
        ReportCaseFailure,
    )

    @dataclass
    class MockScoreEvaluator(Evaluator[object, object, object]):
        """Returns a score based on the output."""

        evaluation_name: str = 'score'

        def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluatorOutput:
            return float(ctx.output)  # type: ignore[arg-type]  # pragma: no cover

    @dataclass
    class AlwaysPass(Evaluator[object, object, object]):
        """Always passes."""

        def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluatorOutput:
            return True


pytestmark = [pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed'), pytest.mark.anyio]


@pytest.fixture(autouse=True)
def use_logfire(capfire: CaptureLogfire):
    assert capfire


async def test_repeat_1_produces_identical_behavior():
    """repeat=1 (default) should produce identical behavior to current — no source_case_name, case_groups() returns None."""
    call_count = 0

    async def task(inputs: str) -> str:
        nonlocal call_count
        call_count += 1
        return inputs.upper()

    dataset = Dataset(
        cases=[
            Case(name='case1', inputs='hello'),
            Case(name='case2', inputs='world'),
        ]
    )
    report = await dataset.evaluate(task, name='test', progress=False)

    assert call_count == 2
    assert len(report.cases) == 2
    assert all(c.source_case_name is None for c in report.cases)
    assert report.case_groups() is None


async def test_repeat_3_produces_3x_cases():
    """repeat=3 should produce 3x cases, each with run-indexed names and source_case_name set."""
    call_count = 0

    async def task(inputs: str) -> str:
        nonlocal call_count
        call_count += 1
        return inputs.upper()

    dataset = Dataset(
        cases=[
            Case(name='case1', inputs='hello'),
            Case(name='case2', inputs='world'),
        ]
    )
    report = await dataset.evaluate(task, name='test', progress=False, repeat=3)

    assert call_count == 6  # 2 cases * 3 repeats
    assert len(report.cases) == 6

    # Check naming
    case_names = sorted(c.name for c in report.cases)
    assert case_names == sorted(
        [
            'case1 [1/3]',
            'case1 [2/3]',
            'case1 [3/3]',
            'case2 [1/3]',
            'case2 [2/3]',
            'case2 [3/3]',
        ]
    )

    # Check source_case_name
    assert all(c.source_case_name is not None for c in report.cases)
    case1_runs = [c for c in report.cases if c.source_case_name == 'case1']
    case2_runs = [c for c in report.cases if c.source_case_name == 'case2']
    assert len(case1_runs) == 3
    assert len(case2_runs) == 3


async def test_repeat_with_unnamed_cases():
    """repeat should work with cases that don't have explicit names."""

    async def task(inputs: str) -> str:
        return inputs.upper()

    dataset = Dataset(
        cases=[
            Case(inputs='hello'),
            Case(inputs='world'),
        ]
    )
    report = await dataset.evaluate(task, name='test', progress=False, repeat=2)

    assert len(report.cases) == 4
    case_names = sorted(c.name for c in report.cases)
    assert case_names == sorted(
        [
            'Case 1 [1/2]',
            'Case 1 [2/2]',
            'Case 2 [1/2]',
            'Case 2 [2/2]',
        ]
    )
    assert all(c.source_case_name is not None for c in report.cases)


async def test_repeat_invalid_value():
    """repeat < 1 should raise ValueError."""

    async def task(inputs: str) -> str:
        return inputs  # pragma: no cover

    dataset = Dataset(cases=[Case(inputs='hello')])
    with pytest.raises(ValueError, match='repeat must be >= 1'):
        await dataset.evaluate(task, name='test', progress=False, repeat=0)


async def test_case_groups_correctly_groups():
    """case_groups() should correctly group runs by source_case_name."""

    async def task(inputs: str) -> str:
        return inputs.upper()

    dataset = Dataset(
        cases=[
            Case(name='case1', inputs='hello'),
            Case(name='case2', inputs='world'),
        ]
    )
    report = await dataset.evaluate(task, name='test', progress=False, repeat=2)

    groups = report.case_groups()
    assert groups is not None
    assert len(groups) == 2

    group_names = [g.name for g in groups]
    assert 'case1' in group_names
    assert 'case2' in group_names

    for group in groups:
        assert len(group.runs) == 2
        assert len(group.failures) == 0
        assert group.summary is not None
        assert group.summary.name == 'Averages'


async def test_case_groups_returns_none_for_single_run():
    """case_groups() should return None when no cases have source_case_name (single-run experiment)."""

    async def task(inputs: str) -> str:
        return inputs.upper()

    dataset = Dataset(cases=[Case(name='case1', inputs='hello')])
    report = await dataset.evaluate(task, name='test', progress=False, repeat=1)

    assert report.case_groups() is None


def test_averages_with_multi_run():
    """averages() with multi-run should use two-level aggregation via average_from_aggregates."""
    # Use deterministic scores: case1 gets [0.2, 0.4], case2 gets [0.9]
    # case2 has one failure to create unequal group sizes (distinguishes flat vs two-level).
    # Two-level: mean(mean(0.2, 0.4), mean(0.9)) = mean(0.3, 0.9) = 0.6
    # Flat would be: mean(0.2, 0.4, 0.9) = 0.5
    mock_evaluator_spec = MockScoreEvaluator().as_spec()
    case1_run1 = ReportCase(
        name='case1 [1/2]',
        inputs='hello',
        output=0.2,
        expected_output=None,
        metadata=None,
        metrics={},
        attributes={},
        scores={'score': EvaluationResult(name='score', value=0.2, reason=None, source=mock_evaluator_spec)},
        labels={},
        assertions={},
        task_duration=1.0,
        total_duration=1.0,
        source_case_name='case1',
    )
    case1_run2 = ReportCase(
        name='case1 [2/2]',
        inputs='hello',
        output=0.4,
        expected_output=None,
        metadata=None,
        metrics={},
        attributes={},
        scores={'score': EvaluationResult(name='score', value=0.4, reason=None, source=mock_evaluator_spec)},
        labels={},
        assertions={},
        task_duration=1.0,
        total_duration=1.0,
        source_case_name='case1',
    )
    case2_run1 = ReportCase(
        name='case2 [1/2]',
        inputs='world',
        output=0.9,
        expected_output=None,
        metadata=None,
        metrics={},
        attributes={},
        scores={'score': EvaluationResult(name='score', value=0.9, reason=None, source=mock_evaluator_spec)},
        labels={},
        assertions={},
        task_duration=1.0,
        total_duration=1.0,
        source_case_name='case2',
    )
    case2_failure = ReportCaseFailure(
        name='case2 [2/2]',
        inputs='world',
        metadata=None,
        expected_output=None,
        error_message='failed',
        error_stacktrace='Traceback ...',
        source_case_name='case2',
    )

    report = EvaluationReport(
        name='test',
        cases=[case1_run1, case1_run2, case2_run1],
        failures=[case2_failure],
    )
    averages = report.averages()
    assert averages is not None
    # Two-level: mean(mean(0.2, 0.4), mean(0.9)) = mean(0.3, 0.9) = 0.6
    assert averages.scores['score'] == pytest.approx(0.6)  # pyright: ignore[reportUnknownMemberType]


def test_average_from_aggregates_computation():
    """Test ReportCaseAggregate.average_from_aggregates() correctness."""
    agg1 = ReportCaseAggregate(
        name='Averages',
        scores={'s1': 0.5, 's2': 0.25},
        labels={'l1': {'a': 0.5, 'b': 0.5}},
        metrics={'m1': 10.0},
        assertions=1.0,
        task_duration=1.0,
        total_duration=2.0,
    )
    agg2 = ReportCaseAggregate(
        name='Averages',
        scores={'s1': 0.5, 's2': 0.75},
        labels={'l1': {'a': 0.25, 'b': 0.75}},
        metrics={'m1': 20.0},
        assertions=0.5,
        task_duration=3.0,
        total_duration=4.0,
    )

    result = ReportCaseAggregate.average_from_aggregates([agg1, agg2])

    assert result.name == 'Averages'
    assert result.scores['s1'] == 0.5
    assert result.scores['s2'] == 0.5
    assert result.metrics['m1'] == 15.0
    assert result.assertions == 0.75
    assert result.task_duration == 2.0
    assert result.total_duration == 3.0
    # Labels: average the distributions
    assert result.labels['l1']['a'] == 0.375
    assert result.labels['l1']['b'] == 0.625


def test_average_from_aggregates_empty():
    """average_from_aggregates with empty list should return empty aggregate."""
    result = ReportCaseAggregate.average_from_aggregates([])
    assert result.scores == {}
    assert result.labels == {}
    assert result.metrics == {}
    assert result.assertions is None
    assert result.task_duration == 0.0
    assert result.total_duration == 0.0


def test_average_from_aggregates_partial_keys():
    """average_from_aggregates should handle aggregates with different keys for scores, metrics, and labels."""
    agg1 = ReportCaseAggregate(
        name='Averages',
        scores={'s1': 1.0},
        labels={'sentiment': {'positive': 0.8, 'negative': 0.2}},
        metrics={'m1': 10.0},
        assertions=1.0,
        task_duration=1.0,
        total_duration=1.0,
    )
    agg2 = ReportCaseAggregate(
        name='Averages',
        scores={'s2': 2.0},
        labels={'topic': {'science': 0.6, 'arts': 0.4}},
        metrics={'m2': 20.0},
        assertions=None,
        task_duration=3.0,
        total_duration=3.0,
    )

    result = ReportCaseAggregate.average_from_aggregates([agg1, agg2])
    # s1 only present in agg1, so average is 1.0/1 = 1.0
    assert result.scores['s1'] == 1.0
    # s2 only present in agg2, so average is 2.0/1 = 2.0
    assert result.scores['s2'] == 2.0
    # m1 only in agg1, m2 only in agg2
    assert result.metrics['m1'] == 10.0
    assert result.metrics['m2'] == 20.0
    # Labels: each label key only in one aggregate
    assert result.labels['sentiment'] == {'positive': 0.8, 'negative': 0.2}
    assert result.labels['topic'] == {'science': 0.6, 'arts': 0.4}
    # Only one aggregate has assertions
    assert result.assertions == 1.0


async def test_otel_spans_have_correct_attributes(capfire: CaptureLogfire):
    """OTel spans should have repeat and source_case_name attributes set when repeat > 1."""

    async def task(inputs: str) -> str:
        return inputs.upper()

    dataset = Dataset(cases=[Case(name='case1', inputs='hello')])
    await dataset.evaluate(task, name='test', progress=False, repeat=2)

    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)

    # The evaluate span should have the repeat attribute
    eval_spans = [s for s in spans if s['name'] == 'evaluate {name}']
    assert len(eval_spans) == 1
    assert eval_spans[0]['attributes']['logfire.experiment.repeat'] == 2

    # Each case span should have source_case_name
    case_spans = [s for s in spans if s['name'] == 'case: {case_name}']
    assert len(case_spans) == 2
    for span in case_spans:
        assert span['attributes']['logfire.experiment.source_case_name'] == 'case1'


async def test_repeat_with_evaluators():
    """Evaluators should run on each repeated run."""

    async def task(inputs: str) -> str:
        return inputs.upper()

    dataset = Dataset(
        cases=[Case(name='case1', inputs='hello')],
        evaluators=(AlwaysPass(),),
    )
    report = await dataset.evaluate(task, name='test', progress=False, repeat=3)

    assert len(report.cases) == 3
    # Each case should have assertions from the evaluator
    for case in report.cases:
        assert len(case.assertions) > 0


def test_repeat_rendering():
    """Multi-run report should render correctly."""
    mock_evaluator_spec = AlwaysPass().as_spec()

    def _make_case(name: str, source: str, duration: float) -> ReportCase:
        return ReportCase(
            name=name,
            inputs='hello',
            output='HELLO',
            expected_output=None,
            metadata=None,
            metrics={},
            attributes={},
            scores={'score': EvaluationResult(name='score', value=0.8, reason=None, source=mock_evaluator_spec)},
            labels={},
            assertions={
                'AlwaysPass': EvaluationResult(name='AlwaysPass', value=True, reason=None, source=mock_evaluator_spec)
            },
            task_duration=duration,
            total_duration=duration,
            source_case_name=source,
        )

    report = EvaluationReport(
        name='test',
        cases=[
            _make_case('case1 [1/2]', 'case1', 0.1),
            _make_case('case1 [2/2]', 'case1', 0.2),
            _make_case('case2 [1/2]', 'case2', 0.3),
            _make_case('case2 [2/2]', 'case2', 0.4),
        ],
    )

    rendered = report.render(width=200)
    assert rendered == snapshot("""\
               Evaluation Summary: test               \n\
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Case ID     ┃ Scores       ┃ Assertions ┃ Duration ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━┩
│ case1 [1/2] │ score: 0.800 │ ✔          │  100.0ms │
├─────────────┼──────────────┼────────────┼──────────┤
│ case1 [2/2] │ score: 0.800 │ ✔          │  200.0ms │
├─────────────┼──────────────┼────────────┼──────────┤
│ case2 [1/2] │ score: 0.800 │ ✔          │  300.0ms │
├─────────────┼──────────────┼────────────┼──────────┤
│ case2 [2/2] │ score: 0.800 │ ✔          │  400.0ms │
├─────────────┼──────────────┼────────────┼──────────┤
│ Averages    │ score: 0.800 │ 100.0% ✔   │  250.0ms │
└─────────────┴──────────────┴────────────┴──────────┘
""")


def test_report_case_group_fields():
    """Test that ReportCaseGroup has expected fields."""
    mock_evaluator_spec = AlwaysPass().as_spec()
    case1 = ReportCase(
        name='case1 [1/2]',
        inputs='hello',
        output='HELLO',
        expected_output=None,
        metadata=None,
        metrics={},
        attributes={},
        scores={},
        labels={},
        assertions={
            'AlwaysPass': EvaluationResult(name='AlwaysPass', value=True, reason=None, source=mock_evaluator_spec)
        },
        task_duration=0.1,
        total_duration=0.2,
        source_case_name='case1',
    )
    case2 = ReportCase(
        name='case1 [2/2]',
        inputs='hello',
        output='HELLO',
        expected_output=None,
        metadata=None,
        metrics={},
        attributes={},
        scores={},
        labels={},
        assertions={
            'AlwaysPass': EvaluationResult(name='AlwaysPass', value=True, reason=None, source=mock_evaluator_spec)
        },
        task_duration=0.15,
        total_duration=0.25,
        source_case_name='case1',
    )

    report = EvaluationReport(name='test', cases=[case1, case2])
    groups = report.case_groups()
    assert groups is not None
    assert len(groups) == 1

    group = groups[0]
    assert group.name == 'case1'
    assert group.inputs == 'hello'
    assert group.metadata is None
    assert group.expected_output is None
    assert len(group.runs) == 2
    assert len(group.failures) == 0
    assert group.summary.task_duration == 0.125


def test_case_groups_with_failures():
    """case_groups() should correctly group failures by source_case_name."""
    mock_evaluator_spec = AlwaysPass().as_spec()
    case1 = ReportCase(
        name='case1 [1/2]',
        inputs='hello',
        output='HELLO',
        expected_output=None,
        metadata=None,
        metrics={},
        attributes={},
        scores={},
        labels={},
        assertions={
            'AlwaysPass': EvaluationResult(name='AlwaysPass', value=True, reason=None, source=mock_evaluator_spec)
        },
        task_duration=0.1,
        total_duration=0.2,
        source_case_name='case1',
    )
    failure1 = ReportCaseFailure(
        name='case1 [2/2]',
        inputs='hello',
        metadata=None,
        expected_output=None,
        error_message='something went wrong',
        error_stacktrace='Traceback ...',
        source_case_name='case1',
    )

    report = EvaluationReport(name='test', cases=[case1], failures=[failure1])
    groups = report.case_groups()
    assert groups is not None
    assert len(groups) == 1

    group = groups[0]
    assert group.name == 'case1'
    assert len(group.runs) == 1
    assert len(group.failures) == 1
    assert group.failures[0].error_message == 'something went wrong'
