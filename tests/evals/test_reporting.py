from __future__ import annotations as _annotations

from dataclasses import dataclass

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from ..conftest import try_import
from .utils import render_table

with try_import() as imports_successful:
    from pydantic_evals.evaluators import EvaluationResult, Evaluator, EvaluatorContext
    from pydantic_evals.reporting import (
        EvaluationRenderer,
        EvaluationReport,
        ReportCase,
        ReportCaseAggregate,
    )

pytestmark = [pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed'), pytest.mark.anyio]


class TaskInput(BaseModel):
    query: str


class TaskOutput(BaseModel):
    answer: str


class TaskMetadata(BaseModel):
    difficulty: str


@pytest.fixture
def mock_evaluator() -> Evaluator[TaskInput, TaskOutput, TaskMetadata]:
    class MockEvaluator(Evaluator[TaskInput, TaskOutput, TaskMetadata]):
        def evaluate(self, ctx: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]) -> bool:
            raise NotImplementedError

    return MockEvaluator()


@pytest.fixture
def sample_assertion(mock_evaluator: Evaluator[TaskInput, TaskOutput, TaskMetadata]) -> EvaluationResult[bool]:
    return EvaluationResult(
        name='MockEvaluator',
        value=True,
        reason=None,
        source=mock_evaluator,
    )


@pytest.fixture
def sample_score(mock_evaluator: Evaluator[TaskInput, TaskOutput, TaskMetadata]) -> EvaluationResult[float]:
    return EvaluationResult(
        name='MockEvaluator',
        value=2.5,
        reason=None,
        source=mock_evaluator,
    )


@pytest.fixture
def sample_label(mock_evaluator: Evaluator[TaskInput, TaskOutput, TaskMetadata]) -> EvaluationResult[str]:
    return EvaluationResult(
        name='MockEvaluator',
        value='hello',
        reason=None,
        source=mock_evaluator,
    )


@pytest.fixture
def sample_report_case(
    sample_assertion: EvaluationResult[bool], sample_score: EvaluationResult[float], sample_label: EvaluationResult[str]
) -> ReportCase:
    return ReportCase(
        name='test_case',
        inputs={'query': 'What is 2+2?'},
        output={'answer': '4'},
        expected_output={'answer': '4'},
        metadata={'difficulty': 'easy'},
        metrics={'accuracy': 0.95},
        attributes={},
        scores={'score1': sample_score},
        labels={'label1': sample_label},
        assertions={sample_assertion.name: sample_assertion},
        task_duration=0.1,
        total_duration=0.2,
        trace_id='test-trace-id',
        span_id='test-span-id',
    )


@pytest.fixture
def sample_report(sample_report_case: ReportCase) -> EvaluationReport:
    return EvaluationReport(
        cases=[sample_report_case],
        name='test_report',
    )


async def test_evaluation_renderer_basic(sample_report: EvaluationReport):
    """Test basic functionality of EvaluationRenderer."""
    renderer = EvaluationRenderer(
        include_input=True,
        include_output=True,
        include_metadata=True,
        include_expected_output=True,
        include_durations=True,
        include_total_duration=True,
        include_removed_cases=False,
        include_averages=True,
        input_config={},
        metadata_config={},
        output_config={},
        score_configs={},
        label_configs={},
        metric_configs={},
        duration_config={},
    )

    table = renderer.build_table(sample_report)
    assert render_table(table) == snapshot("""\
                                                                              Evaluation Summary: test_report
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Case ID   ┃ Inputs                    ┃ Metadata               ┃ Expected Output ┃ Outputs         ┃ Scores       ┃ Labels                 ┃ Metrics         ┃ Assertions ┃    Durations ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ test_case │ {'query': 'What is 2+2?'} │ {'difficulty': 'easy'} │ {'answer': '4'} │ {'answer': '4'} │ score1: 2.50 │ label1: hello          │ accuracy: 0.950 │ ✔          │  task: 0.100 │
│           │                           │                        │                 │                 │              │                        │                 │            │ total: 0.200 │
├───────────┼───────────────────────────┼────────────────────────┼─────────────────┼─────────────────┼──────────────┼────────────────────────┼─────────────────┼────────────┼──────────────┤
│ Averages  │                           │                        │                 │                 │ score1: 2.50 │ label1: {'hello': 1.0} │ accuracy: 0.950 │ 100.0% ✔   │  task: 0.100 │
│           │                           │                        │                 │                 │              │                        │                 │            │ total: 0.200 │
└───────────┴───────────────────────────┴────────────────────────┴─────────────────┴─────────────────┴──────────────┴────────────────────────┴─────────────────┴────────────┴──────────────┘
""")


async def test_evaluation_renderer_with_baseline(sample_report: EvaluationReport):
    """Test EvaluationRenderer with baseline comparison."""
    baseline_report = EvaluationReport(
        cases=[
            ReportCase(
                name='test_case',
                inputs={'query': 'What is 2+2?'},
                output={'answer': '4'},
                expected_output={'answer': '4'},
                metadata={'difficulty': 'easy'},
                metrics={'accuracy': 0.90},
                attributes={},
                scores={
                    'score1': EvaluationResult(
                        name='MockEvaluator',
                        value=2.5,
                        reason=None,
                        source=sample_report.cases[0].scores['score1'].source,
                    )
                },
                labels={
                    'label1': EvaluationResult(
                        name='MockEvaluator',
                        value='hello',
                        reason=None,
                        source=sample_report.cases[0].labels['label1'].source,
                    )
                },
                assertions={},
                task_duration=0.15,
                total_duration=0.25,
                trace_id='test-trace-id',
                span_id='test-span-id',
            )
        ],
        name='baseline_report',
    )

    renderer = EvaluationRenderer(
        include_input=True,
        include_metadata=True,
        include_expected_output=True,
        include_output=True,
        include_durations=True,
        include_total_duration=True,
        include_removed_cases=False,
        include_averages=True,
        input_config={},
        metadata_config={},
        output_config={},
        score_configs={},
        label_configs={},
        metric_configs={},
        duration_config={},
    )

    table = renderer.build_diff_table(sample_report, baseline_report)
    assert render_table(table) == snapshot("""\
                                                                                                                               Evaluation Diff: baseline_report → test_report
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Case ID   ┃ Inputs                    ┃ Metadata               ┃ Expected Output ┃ Outputs         ┃ Scores       ┃ Labels                                                                              ┃ Metrics                                 ┃ Assertions   ┃                             Durations ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ test_case │ {'query': 'What is 2+2?'} │ {'difficulty': 'easy'} │ {'answer': '4'} │ {'answer': '4'} │ score1: 2.50 │ label1: EvaluationResult(name='MockEvaluator', value='hello', reason=None,          │ accuracy: 0.900 → 0.950 (+0.05 / +5.6%) │  → ✔         │  task: 0.150 → 0.100 (-0.05 / -33.3%) │
│           │                           │                        │                 │                 │              │ source=mock_evaluator.<locals>.MockEvaluator())                                     │                                         │              │ total: 0.250 → 0.200 (-0.05 / -20.0%) │
├───────────┼───────────────────────────┼────────────────────────┼─────────────────┼─────────────────┼──────────────┼─────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────┼──────────────┼───────────────────────────────────────┤
│ Averages  │                           │                        │                 │                 │ score1: 2.50 │ label1: {'hello': 1.0}                                                              │ accuracy: 0.900 → 0.950 (+0.05 / +5.6%) │ - → 100.0% ✔ │  task: 0.150 → 0.100 (-0.05 / -33.3%) │
│           │                           │                        │                 │                 │              │                                                                                     │                                         │              │ total: 0.250 → 0.200 (-0.05 / -20.0%) │
└───────────┴───────────────────────────┴────────────────────────┴─────────────────┴─────────────────┴──────────────┴─────────────────────────────────────────────────────────────────────────────────────┴─────────────────────────────────────────┴──────────────┴───────────────────────────────────────┘
""")


async def test_evaluation_renderer_with_removed_cases(sample_report: EvaluationReport):
    """Test EvaluationRenderer with removed cases."""
    baseline_report = EvaluationReport(
        cases=[
            ReportCase(
                name='removed_case',
                inputs={'query': 'What is 3+3?'},
                output={'answer': '6'},
                expected_output={'answer': '6'},
                metadata={'difficulty': 'medium'},
                metrics={'accuracy': 0.85},
                attributes={},
                scores={},
                labels={},
                assertions={},
                task_duration=0.1,
                total_duration=0.15,
                trace_id='test-trace-id-2',
                span_id='test-span-id-2',
            )
        ],
        name='baseline_report',
    )

    renderer = EvaluationRenderer(
        include_input=True,
        include_metadata=True,
        include_expected_output=True,
        include_output=True,
        include_durations=True,
        include_total_duration=True,
        include_removed_cases=True,
        include_averages=True,
        input_config={},
        metadata_config={},
        output_config={},
        score_configs={},
        label_configs={},
        metric_configs={},
        duration_config={},
    )

    table = renderer.build_diff_table(sample_report, baseline_report)
    assert render_table(table) == snapshot("""\
                                                                                                                Evaluation Diff: baseline_report → test_report
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Case ID        ┃ Inputs                    ┃ Metadata                 ┃ Expected Output ┃ Outputs         ┃ Scores                   ┃ Labels                             ┃ Metrics                                 ┃ Assertions   ┃                             Durations ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ + Added Case   │ {'query': 'What is 2+2?'} │ {'difficulty': 'easy'}   │ {'answer': '4'} │ {'answer': '4'} │ score1: 2.50             │ label1: hello                      │ accuracy: 0.950                         │ ✔            │                           task: 0.100 │
│ test_case      │                           │                          │                 │                 │                          │                                    │                                         │              │                          total: 0.200 │
├────────────────┼───────────────────────────┼──────────────────────────┼─────────────────┼─────────────────┼──────────────────────────┼────────────────────────────────────┼─────────────────────────────────────────┼──────────────┼───────────────────────────────────────┤
│ - Removed Case │ {'query': 'What is 3+3?'} │ {'difficulty': 'medium'} │ {'answer': '6'} │ {'answer': '6'} │ -                        │ -                                  │ accuracy: 0.850                         │ -            │                           task: 0.100 │
│ removed_case   │                           │                          │                 │                 │                          │                                    │                                         │              │                          total: 0.150 │
├────────────────┼───────────────────────────┼──────────────────────────┼─────────────────┼─────────────────┼──────────────────────────┼────────────────────────────────────┼─────────────────────────────────────────┼──────────────┼───────────────────────────────────────┤
│ Averages       │                           │                          │                 │                 │ score1: <missing> → 2.50 │ label1: <missing> → {'hello': 1.0} │ accuracy: 0.850 → 0.950 (+0.1 / +11.8%) │ - → 100.0% ✔ │                           task: 0.100 │
│                │                           │                          │                 │                 │                          │                                    │                                         │              │ total: 0.150 → 0.200 (+0.05 / +33.3%) │
└────────────────┴───────────────────────────┴──────────────────────────┴─────────────────┴─────────────────┴──────────────────────────┴────────────────────────────────────┴─────────────────────────────────────────┴──────────────┴───────────────────────────────────────┘
""")


async def test_evaluation_renderer_with_custom_configs(sample_report: EvaluationReport):
    """Test EvaluationRenderer with custom render configurations."""
    renderer = EvaluationRenderer(
        include_input=True,
        include_metadata=True,
        include_expected_output=True,
        include_output=True,
        include_durations=True,
        include_total_duration=True,
        include_removed_cases=False,
        include_averages=True,
        input_config={'value_formatter': lambda x: str(x)},
        metadata_config={'value_formatter': lambda x: str(x)},
        output_config={'value_formatter': lambda x: str(x)},
        score_configs={
            'score1': {
                'value_formatter': '{:.2f}',
                'diff_formatter': '{:+.2f}',
                'diff_atol': 0.01,
                'diff_rtol': 0.05,
                'diff_increase_style': 'bold green',
                'diff_decrease_style': 'bold red',
            }
        },
        label_configs={'label1': {'value_formatter': lambda x: str(x)}},
        metric_configs={
            'accuracy': {
                'value_formatter': '{:.1%}',
                'diff_formatter': '{:+.1%}',
                'diff_atol': 0.01,
                'diff_rtol': 0.05,
                'diff_increase_style': 'bold green',
                'diff_decrease_style': 'bold red',
            }
        },
        duration_config={
            'value_formatter': '{:.3f}s',
            'diff_formatter': '{:+.3f}s',
            'diff_atol': 0.001,
            'diff_rtol': 0.05,
            'diff_increase_style': 'bold red',
            'diff_decrease_style': 'bold green',
        },
    )

    table = renderer.build_table(sample_report)
    assert render_table(table) == snapshot("""\
                                                                               Evaluation Summary: test_report
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Case ID   ┃ Inputs                    ┃ Metadata               ┃ Expected Output ┃ Outputs         ┃ Scores       ┃ Labels                 ┃ Metrics         ┃ Assertions ┃     Durations ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ test_case │ {'query': 'What is 2+2?'} │ {'difficulty': 'easy'} │ {'answer': '4'} │ {'answer': '4'} │ score1: 2.50 │ label1: hello          │ accuracy: 95.0% │ ✔          │  task: 0.100s │
│           │                           │                        │                 │                 │              │                        │                 │            │ total: 0.200s │
├───────────┼───────────────────────────┼────────────────────────┼─────────────────┼─────────────────┼──────────────┼────────────────────────┼─────────────────┼────────────┼───────────────┤
│ Averages  │                           │                        │                 │                 │ score1: 2.50 │ label1: {'hello': 1.0} │ accuracy: 95.0% │ 100.0% ✔   │  task: 0.100s │
│           │                           │                        │                 │                 │              │                        │                 │            │ total: 0.200s │
└───────────┴───────────────────────────┴────────────────────────┴─────────────────┴─────────────────┴──────────────┴────────────────────────┴─────────────────┴────────────┴───────────────┘
""")


async def test_report_case_aggregate_average():
    """Test ReportCaseAggregate.average() method."""

    @dataclass
    class MockEvaluator(Evaluator[TaskInput, TaskOutput, TaskMetadata]):
        def evaluate(self, ctx: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]) -> float:
            raise NotImplementedError

    cases = [
        ReportCase(
            name='case1',
            inputs={'query': 'What is 2+2?'},
            output={'answer': '4'},
            expected_output={'answer': '4'},
            metadata={'difficulty': 'easy'},
            metrics={'accuracy': 0.95},
            attributes={},
            scores={
                'score1': EvaluationResult(
                    name='MockEvaluator',
                    value=0.8,
                    reason=None,
                    source=MockEvaluator(),
                )
            },
            labels={
                'label1': EvaluationResult(
                    name='MockEvaluator',
                    value='good',
                    reason=None,
                    source=MockEvaluator(),
                )
            },
            assertions={
                'assert1': EvaluationResult(
                    name='MockEvaluator',
                    value=True,
                    reason=None,
                    source=MockEvaluator(),
                )
            },
            task_duration=0.1,
            total_duration=0.2,
            trace_id='test-trace-id-1',
            span_id='test-span-id-1',
        ),
        ReportCase(
            name='case2',
            inputs={'query': 'What is 3+3?'},
            output={'answer': '6'},
            expected_output={'answer': '6'},
            metadata={'difficulty': 'medium'},
            metrics={'accuracy': 0.85},
            attributes={},
            scores={
                'score1': EvaluationResult(
                    name='MockEvaluator',
                    value=0.7,
                    reason=None,
                    source=MockEvaluator(),
                )
            },
            labels={
                'label1': EvaluationResult(
                    name='MockEvaluator',
                    value='good',
                    reason=None,
                    source=MockEvaluator(),
                )
            },
            assertions={
                'assert1': EvaluationResult(
                    name='MockEvaluator',
                    value=False,
                    reason=None,
                    source=MockEvaluator(),
                )
            },
            task_duration=0.15,
            total_duration=0.25,
            trace_id='test-trace-id-2',
            span_id='test-span-id-2',
        ),
    ]

    aggregate = ReportCaseAggregate.average(cases)

    assert aggregate.name == 'Averages'
    assert aggregate.scores['score1'] == 0.75  # (0.8 + 0.7) / 2
    assert aggregate.labels['label1']['good'] == 1.0  # Both cases have 'good' label
    assert abs(aggregate.metrics['accuracy'] - 0.90) < 1e-10  # floating-point error  # (0.95 + 0.85) / 2
    assert aggregate.assertions == 0.5  # 1 passing out of 2 assertions
    assert aggregate.task_duration == 0.125  # (0.1 + 0.15) / 2
    assert aggregate.total_duration == 0.225  # (0.2 + 0.25) / 2


async def test_report_case_aggregate_empty():
    """Test ReportCaseAggregate.average() with empty cases list."""
    assert ReportCaseAggregate.average([]).model_dump() == {
        'assertions': None,
        'labels': {},
        'metrics': {},
        'name': 'Averages',
        'scores': {},
        'task_duration': 0.0,
        'total_duration': 0.0,
    }
