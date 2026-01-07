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
        source=mock_evaluator.as_spec(),
    )


@pytest.fixture
def sample_score(mock_evaluator: Evaluator[TaskInput, TaskOutput, TaskMetadata]) -> EvaluationResult[float]:
    return EvaluationResult(
        name='MockEvaluator',
        value=2.5,
        reason='my reason',
        source=mock_evaluator.as_spec(),
    )


@pytest.fixture
def sample_label(mock_evaluator: Evaluator[TaskInput, TaskOutput, TaskMetadata]) -> EvaluationResult[str]:
    return EvaluationResult(
        name='MockEvaluator',
        value='hello',
        reason=None,
        source=mock_evaluator.as_spec(),
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
        include_reasons=False,
        include_error_message=False,
        include_error_stacktrace=False,
        include_evaluator_failures=True,
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


async def test_evaluation_renderer_with_reasons(sample_report: EvaluationReport):
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
        include_reasons=True,
        include_error_message=False,
        include_error_stacktrace=False,
        include_evaluator_failures=True,
    )

    table = renderer.build_table(sample_report)
    assert render_table(table) == snapshot("""\
                                                                                     Evaluation Summary: test_report
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Case ID   ┃ Inputs                    ┃ Metadata               ┃ Expected Output ┃ Outputs         ┃ Scores              ┃ Labels                 ┃ Metrics         ┃ Assertions       ┃    Durations ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ test_case │ {'query': 'What is 2+2?'} │ {'difficulty': 'easy'} │ {'answer': '4'} │ {'answer': '4'} │ score1: 2.50        │ label1: hello          │ accuracy: 0.950 │ MockEvaluator: ✔ │  task: 0.100 │
│           │                           │                        │                 │                 │   Reason: my reason │                        │                 │                  │ total: 0.200 │
│           │                           │                        │                 │                 │                     │                        │                 │                  │              │
├───────────┼───────────────────────────┼────────────────────────┼─────────────────┼─────────────────┼─────────────────────┼────────────────────────┼─────────────────┼──────────────────┼──────────────┤
│ Averages  │                           │                        │                 │                 │ score1: 2.50        │ label1: {'hello': 1.0} │ accuracy: 0.950 │ 100.0% ✔         │  task: 0.100 │
│           │                           │                        │                 │                 │                     │                        │                 │                  │ total: 0.200 │
└───────────┴───────────────────────────┴────────────────────────┴─────────────────┴─────────────────┴─────────────────────┴────────────────────────┴─────────────────┴──────────────────┴──────────────┘
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
        include_reasons=False,
        include_error_message=False,
        include_error_stacktrace=False,
        include_evaluator_failures=True,
    )

    table = renderer.build_diff_table(sample_report, baseline_report)
    assert render_table(table) == snapshot("""\
                                                                                                Evaluation Diff: baseline_report → test_report
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Case ID   ┃ Inputs                    ┃ Metadata               ┃ Expected Output ┃ Outputs         ┃ Scores       ┃ Labels                 ┃ Metrics                                 ┃ Assertions   ┃                             Durations ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ test_case │ {'query': 'What is 2+2?'} │ {'difficulty': 'easy'} │ {'answer': '4'} │ {'answer': '4'} │ score1: 2.50 │ label1: hello          │ accuracy: 0.900 → 0.950 (+0.05 / +5.6%) │  → ✔         │  task: 0.150 → 0.100 (-0.05 / -33.3%) │
│           │                           │                        │                 │                 │              │                        │                                         │              │ total: 0.250 → 0.200 (-0.05 / -20.0%) │
├───────────┼───────────────────────────┼────────────────────────┼─────────────────┼─────────────────┼──────────────┼────────────────────────┼─────────────────────────────────────────┼──────────────┼───────────────────────────────────────┤
│ Averages  │                           │                        │                 │                 │ score1: 2.50 │ label1: {'hello': 1.0} │ accuracy: 0.900 → 0.950 (+0.05 / +5.6%) │ - → 100.0% ✔ │  task: 0.150 → 0.100 (-0.05 / -33.3%) │
│           │                           │                        │                 │                 │              │                        │                                         │              │ total: 0.250 → 0.200 (-0.05 / -20.0%) │
└───────────┴───────────────────────────┴────────────────────────┴─────────────────┴─────────────────┴──────────────┴────────────────────────┴─────────────────────────────────────────┴──────────────┴───────────────────────────────────────┘
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
        include_reasons=False,
        include_error_message=False,
        include_error_stacktrace=False,
        include_evaluator_failures=True,
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
        include_reasons=False,
        include_error_message=False,
        include_error_stacktrace=False,
        include_evaluator_failures=True,
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
                    source=MockEvaluator().as_spec(),
                )
            },
            labels={
                'label1': EvaluationResult(
                    name='MockEvaluator',
                    value='good',
                    reason=None,
                    source=MockEvaluator().as_spec(),
                )
            },
            assertions={
                'assert1': EvaluationResult(
                    name='MockEvaluator',
                    value=True,
                    reason=None,
                    source=MockEvaluator().as_spec(),
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
                    source=MockEvaluator().as_spec(),
                )
            },
            labels={
                'label1': EvaluationResult(
                    name='MockEvaluator',
                    value='good',
                    reason=None,
                    source=MockEvaluator().as_spec(),
                )
            },
            assertions={
                'assert1': EvaluationResult(
                    name='MockEvaluator',
                    value=False,
                    reason=None,
                    source=MockEvaluator().as_spec(),
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


async def test_evaluation_renderer_with_failures(sample_report_case: ReportCase):
    """Test EvaluationRenderer with task failures."""
    from pydantic_evals.reporting import ReportCaseFailure

    failure = ReportCaseFailure(
        name='failed_case',
        inputs={'query': 'What is 10/0?'},
        metadata={'difficulty': 'impossible'},
        expected_output={'answer': 'undefined'},
        error_message='Division by zero',
        error_stacktrace='Traceback (most recent call last):\n  File "test.py", line 1\n    10/0\nZeroDivisionError: division by zero',
        trace_id='test-trace-failure',
        span_id='test-span-failure',
    )

    report = EvaluationReport(
        cases=[sample_report_case],
        failures=[failure],
        name='test_report_with_failures',
    )

    # Test with include_error_message=True, include_error_stacktrace=False
    failures_table = report.failures_table(
        include_input=True,
        include_metadata=True,
        include_expected_output=True,
        include_error_message=True,
        include_error_stacktrace=False,
    )

    assert render_table(failures_table) == snapshot("""\
                                                     Case Failures
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Case ID     ┃ Inputs                     ┃ Metadata                     ┃ Expected Output         ┃ Error Message    ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ failed_case │ {'query': 'What is 10/0?'} │ {'difficulty': 'impossible'} │ {'answer': 'undefined'} │ Division by zero │
└─────────────┴────────────────────────────┴──────────────────────────────┴─────────────────────────┴──────────────────┘
""")

    # Test with both include_error_message=True and include_error_stacktrace=True
    failures_table_with_stacktrace = report.failures_table(
        include_input=False,
        include_metadata=False,
        include_expected_output=False,
        include_error_message=False,
        include_error_stacktrace=True,
    )

    assert render_table(failures_table_with_stacktrace) == snapshot("""\
                    Case Failures
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Case ID     ┃ Error Stacktrace                    ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ failed_case │ Traceback (most recent call last):  │
│             │   File "test.py", line 1            │
│             │     10/0                            │
│             │ ZeroDivisionError: division by zero │
└─────────────┴─────────────────────────────────────┘
""")


async def test_evaluation_renderer_with_evaluator_failures(
    sample_assertion: EvaluationResult[bool], sample_score: EvaluationResult[float], sample_label: EvaluationResult[str]
):
    """Test EvaluationRenderer with evaluator failures."""
    from pydantic_evals.evaluators.evaluator import EvaluatorFailure

    case_with_evaluator_failures = ReportCase(
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
        evaluator_failures=[
            EvaluatorFailure(
                name='CustomEvaluator',
                error_message='Failed to evaluate: timeout',
                error_stacktrace='Timeout stacktrace',
                source=sample_score.source,
            ),
            EvaluatorFailure(
                name='AnotherEvaluator',
                error_message='Connection refused',
                error_stacktrace='Connection refused stacktrace',
                source=sample_label.source,
            ),
        ],
    )

    report = EvaluationReport(
        cases=[case_with_evaluator_failures],
        name='test_report_with_evaluator_failures',
    )

    # Test with include_evaluator_failures=True (default)
    renderer = EvaluationRenderer(
        include_input=True,
        include_metadata=False,
        include_expected_output=False,
        include_output=True,
        include_durations=True,
        include_total_duration=False,
        include_removed_cases=False,
        include_averages=True,
        include_error_message=False,
        include_error_stacktrace=False,
        include_evaluator_failures=True,
        input_config={},
        metadata_config={},
        output_config={},
        score_configs={},
        label_configs={},
        metric_configs={},
        duration_config={},
        include_reasons=False,
    )

    table = renderer.build_table(report)
    assert render_table(table) == snapshot("""\
                                                                  Evaluation Summary: test_report_with_evaluator_failures
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Case ID   ┃ Inputs                    ┃ Outputs         ┃ Scores       ┃ Labels                 ┃ Metrics         ┃ Assertions ┃ Evaluator Failures                           ┃ Duration ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ test_case │ {'query': 'What is 2+2?'} │ {'answer': '4'} │ score1: 2.50 │ label1: hello          │ accuracy: 0.950 │ ✔          │ CustomEvaluator: Failed to evaluate: timeout │    0.100 │
│           │                           │                 │              │                        │                 │            │ AnotherEvaluator: Connection refused         │          │
├───────────┼───────────────────────────┼─────────────────┼──────────────┼────────────────────────┼─────────────────┼────────────┼──────────────────────────────────────────────┼──────────┤
│ Averages  │                           │                 │ score1: 2.50 │ label1: {'hello': 1.0} │ accuracy: 0.950 │ 100.0% ✔   │                                              │    0.100 │
└───────────┴───────────────────────────┴─────────────────┴──────────────┴────────────────────────┴─────────────────┴────────────┴──────────────────────────────────────────────┴──────────┘
""")

    # Test with include_evaluator_failures=False
    renderer_no_failures = EvaluationRenderer(
        include_input=True,
        include_metadata=False,
        include_expected_output=False,
        include_output=True,
        include_durations=True,
        include_total_duration=False,
        include_removed_cases=False,
        include_averages=True,
        include_error_message=False,
        include_error_stacktrace=False,
        include_evaluator_failures=False,
        input_config={},
        metadata_config={},
        output_config={},
        score_configs={},
        label_configs={},
        metric_configs={},
        duration_config={},
        include_reasons=False,
    )

    table_no_failures = renderer_no_failures.build_table(report)
    assert render_table(table_no_failures) == snapshot("""\
                                           Evaluation Summary: test_report_with_evaluator_failures
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Case ID   ┃ Inputs                    ┃ Outputs         ┃ Scores       ┃ Labels                 ┃ Metrics         ┃ Assertions ┃ Duration ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━┩
│ test_case │ {'query': 'What is 2+2?'} │ {'answer': '4'} │ score1: 2.50 │ label1: hello          │ accuracy: 0.950 │ ✔          │    0.100 │
├───────────┼───────────────────────────┼─────────────────┼──────────────┼────────────────────────┼─────────────────┼────────────┼──────────┤
│ Averages  │                           │                 │ score1: 2.50 │ label1: {'hello': 1.0} │ accuracy: 0.950 │ 100.0% ✔   │    0.100 │
└───────────┴───────────────────────────┴─────────────────┴──────────────┴────────────────────────┴─────────────────┴────────────┴──────────┘
""")


async def test_evaluation_renderer_with_evaluator_failures_diff(
    sample_assertion: EvaluationResult[bool], sample_score: EvaluationResult[float], sample_label: EvaluationResult[str]
):
    """Test EvaluationRenderer with evaluator failures in diff table."""
    from pydantic_evals.evaluators.evaluator import EvaluatorFailure

    # Create baseline case with one evaluator failure
    baseline_case = ReportCase(
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
        evaluator_failures=[
            EvaluatorFailure(
                name='BaselineEvaluator',
                error_message='Baseline error',
                error_stacktrace='Baseline stacktrace',
                source=sample_score.source,
            ),
        ],
    )

    # Create new case with different evaluator failures
    new_case = ReportCase(
        name='test_case',
        inputs={'query': 'What is 2+2?'},
        output={'answer': '4'},
        expected_output={'answer': '4'},
        metadata={'difficulty': 'easy'},
        metrics={'accuracy': 0.97},
        attributes={},
        scores={'score1': sample_score},
        labels={'label1': sample_label},
        assertions={sample_assertion.name: sample_assertion},
        task_duration=0.09,
        total_duration=0.19,
        trace_id='test-trace-id-new',
        span_id='test-span-id-new',
        evaluator_failures=[
            EvaluatorFailure(
                name='NewEvaluator',
                error_message='New error',
                error_stacktrace='New stacktrace',
                source=sample_label.source,
            ),
        ],
    )

    baseline_report = EvaluationReport(
        cases=[baseline_case],
        name='baseline_report',
    )

    new_report = EvaluationReport(
        cases=[new_case],
        name='new_report',
    )

    # Test diff table with evaluator failures
    renderer = EvaluationRenderer(
        include_input=False,
        include_metadata=False,
        include_expected_output=False,
        include_output=False,
        include_durations=True,
        include_total_duration=False,
        include_removed_cases=False,
        include_averages=True,
        include_error_message=False,
        include_error_stacktrace=False,
        include_evaluator_failures=True,
        input_config={},
        metadata_config={},
        output_config={},
        score_configs={},
        label_configs={},
        metric_configs={},
        duration_config={},
        include_reasons=False,
    )

    diff_table = renderer.build_diff_table(new_report, baseline_report)
    assert render_table(diff_table) == snapshot("""\
                                                          Evaluation Diff: baseline_report → new_report
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Case ID   ┃ Scores       ┃ Labels                 ┃ Metrics                 ┃ Assertions ┃ Evaluator Failures                ┃                        Duration ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ test_case │ score1: 2.50 │ label1: hello          │ accuracy: 0.950 → 0.970 │ ✔          │ BaselineEvaluator: Baseline error │ 0.100 → 0.0900 (-0.01 / -10.0%) │
│           │              │                        │                         │            │ →                                 │                                 │
│           │              │                        │                         │            │ NewEvaluator: New error           │                                 │
├───────────┼──────────────┼────────────────────────┼─────────────────────────┼────────────┼───────────────────────────────────┼─────────────────────────────────┤
│ Averages  │ score1: 2.50 │ label1: {'hello': 1.0} │ accuracy: 0.950 → 0.970 │ 100.0% ✔   │                                   │ 0.100 → 0.0900 (-0.01 / -10.0%) │
└───────────┴──────────────┴────────────────────────┴─────────────────────────┴────────────┴───────────────────────────────────┴─────────────────────────────────┘
""")


async def test_evaluation_renderer_failures_without_error_message(sample_report_case: ReportCase):
    """Test failures table without error message."""
    from pydantic_evals.reporting import ReportCaseFailure

    # Create failure without error message
    failure = ReportCaseFailure(
        name='failed_case',
        inputs={'query': 'What is 10/0?'},
        metadata={'difficulty': 'impossible'},
        expected_output={'answer': 'undefined'},
        error_message='',  # Empty error message
        error_stacktrace='Traceback',
        trace_id='test-trace-failure',
        span_id='test-span-failure',
    )

    report = EvaluationReport(
        cases=[sample_report_case],
        failures=[failure],
        name='test_report_with_failures',
    )

    # Test with include_error_message=True even though message is empty
    failures_table = report.failures_table(
        include_input=True,
        include_metadata=False,
        include_expected_output=False,
        include_error_message=True,
        include_error_stacktrace=False,
    )

    # The test ensures build_failure_row covers the empty error_message branch
    assert render_table(failures_table) == snapshot("""\
                       Case Failures
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Case ID     ┃ Inputs                     ┃ Error Message ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ failed_case │ {'query': 'What is 10/0?'} │ -             │
└─────────────┴────────────────────────────┴───────────────┘
""")


async def test_evaluation_renderer_evaluator_failures_without_message():
    """Test evaluator failures without error messages."""
    from pydantic_evals.evaluators.evaluator import Evaluator, EvaluatorFailure

    @dataclass
    class MockEvaluator(Evaluator[TaskInput, TaskOutput, TaskMetadata]):
        def evaluate(self, ctx: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]) -> float:
            raise NotImplementedError

    source = MockEvaluator().as_spec()

    # Create case with evaluator failure that has no error message
    case_with_no_message_failure = ReportCase(
        name='test_case',
        inputs={'query': 'What is 2+2?'},
        output={'answer': '4'},
        expected_output={'answer': '4'},
        metadata={'difficulty': 'easy'},
        metrics={'accuracy': 0.95},
        attributes={},
        scores={},
        labels={},
        assertions={},
        task_duration=0.1,
        total_duration=0.2,
        trace_id='test-trace-id',
        span_id='test-span-id',
        evaluator_failures=[
            EvaluatorFailure(
                name='EmptyMessageEvaluator',
                error_message='',  # Empty error message
                error_stacktrace='Some stacktrace',
                source=source,
            ),
        ],
    )

    report = EvaluationReport(
        cases=[case_with_no_message_failure],
        name='test_report',
    )

    renderer = EvaluationRenderer(
        include_input=False,
        include_metadata=False,
        include_expected_output=False,
        include_output=False,
        include_durations=False,
        include_total_duration=False,
        include_removed_cases=False,
        include_averages=False,
        include_error_message=False,
        include_error_stacktrace=False,
        include_evaluator_failures=True,
        input_config={},
        metadata_config={},
        output_config={},
        score_configs={},
        label_configs={},
        metric_configs={},
        duration_config={},
        include_reasons=False,
    )

    table = renderer.build_table(report)
    assert render_table(table) == snapshot("""\
            Evaluation Summary: test_report
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Case ID   ┃ Metrics         ┃ Evaluator Failures    ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│ test_case │ accuracy: 0.950 │ EmptyMessageEvaluator │
└───────────┴─────────────────┴───────────────────────┘
""")


async def test_evaluation_renderer_no_evaluator_failures_column():
    """Test that evaluator failures column is omitted when no failures exist even if flag is True."""

    case_without_evaluator_failures = ReportCase(
        name='test_case',
        inputs={'query': 'What is 2+2?'},
        output={'answer': '4'},
        expected_output={'answer': '4'},
        metadata={'difficulty': 'easy'},
        metrics={'accuracy': 0.95},
        attributes={},
        scores={},
        labels={},
        assertions={},
        task_duration=0.1,
        total_duration=0.2,
        trace_id='test-trace-id',
        span_id='test-span-id',
        evaluator_failures=[],  # No evaluator failures
    )

    report = EvaluationReport(
        cases=[case_without_evaluator_failures],
        name='test_report_no_evaluator_failures',
    )

    # Even with include_evaluator_failures=True, column should not appear if no failures exist
    renderer = EvaluationRenderer(
        include_input=True,
        include_metadata=False,
        include_expected_output=False,
        include_output=True,
        include_durations=True,
        include_total_duration=False,
        include_removed_cases=False,
        include_averages=False,
        include_error_message=False,
        include_error_stacktrace=False,
        include_evaluator_failures=True,  # True, but no failures exist
        input_config={},
        metadata_config={},
        output_config={},
        score_configs={},
        label_configs={},
        metric_configs={},
        duration_config={},
        include_reasons=False,
    )

    table = renderer.build_table(report)
    # The Evaluator Failures column should not be present
    assert render_table(table) == snapshot("""\
                 Evaluation Summary: test_report_no_evaluator_failures
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Case ID   ┃ Inputs                    ┃ Outputs         ┃ Metrics         ┃ Duration ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ test_case │ {'query': 'What is 2+2?'} │ {'answer': '4'} │ accuracy: 0.950 │    0.100 │
└───────────┴───────────────────────────┴─────────────────┴─────────────────┴──────────┘
""")


async def test_evaluation_renderer_with_experiment_metadata(sample_report_case: ReportCase):
    """Test EvaluationRenderer with experiment metadata."""
    report = EvaluationReport(
        cases=[sample_report_case],
        name='test_report',
        experiment_metadata={'model': 'gpt-4o', 'temperature': 0.7, 'prompt_version': 'v2'},
    )

    output = report.render(
        include_input=True,
        include_metadata=False,
        include_expected_output=False,
        include_output=False,
        include_durations=True,
        include_total_duration=False,
        include_removed_cases=False,
        include_averages=True,
        include_errors=False,
        include_error_stacktrace=False,
        include_evaluator_failures=True,
        input_config={},
        metadata_config={},
        output_config={},
        score_configs={},
        label_configs={},
        metric_configs={},
        duration_config={},
        include_reasons=False,
    )

    assert output == snapshot("""\
╭─ Evaluation Summary: test_report ─╮
│ model: gpt-4o                     │
│ temperature: 0.7                  │
│ prompt_version: v2                │
╰───────────────────────────────────╯
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Case ID   ┃ Inputs                    ┃ Scores       ┃ Labels                 ┃ Metrics         ┃ Assertions ┃ Duration ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━┩
│ test_case │ {'query': 'What is 2+2?'} │ score1: 2.50 │ label1: hello          │ accuracy: 0.950 │ ✔          │  100.0ms │
├───────────┼───────────────────────────┼──────────────┼────────────────────────┼─────────────────┼────────────┼──────────┤
│ Averages  │                           │ score1: 2.50 │ label1: {'hello': 1.0} │ accuracy: 0.950 │ 100.0% ✔   │  100.0ms │
└───────────┴───────────────────────────┴──────────────┴────────────────────────┴─────────────────┴────────────┴──────────┘
""")


async def test_evaluation_renderer_with_long_experiment_metadata(sample_report_case: ReportCase):
    """Test EvaluationRenderer with very long experiment metadata."""
    report = EvaluationReport(
        cases=[sample_report_case],
        name='test_report',
        experiment_metadata={
            'model': 'gpt-4o-2024-08-06',
            'temperature': 0.7,
            'prompt_version': 'v2.1.5',
            'system_prompt': 'You are a helpful assistant',
            'max_tokens': 1000,
            'top_p': 0.9,
            'frequency_penalty': 0.1,
            'presence_penalty': 0.1,
        },
    )

    output = report.render(
        include_input=False,
        include_metadata=False,
        include_expected_output=False,
        include_output=False,
        include_durations=True,
        include_total_duration=False,
        include_removed_cases=False,
        include_averages=False,
        include_errors=False,
        include_error_stacktrace=False,
        include_evaluator_failures=True,
        input_config={},
        metadata_config={},
        output_config={},
        score_configs={},
        label_configs={},
        metric_configs={},
        duration_config={},
        include_reasons=False,
    )

    assert output == snapshot("""\
╭─ Evaluation Summary: test_report ──────────╮
│ model: gpt-4o-2024-08-06                   │
│ temperature: 0.7                           │
│ prompt_version: v2.1.5                     │
│ system_prompt: You are a helpful assistant │
│ max_tokens: 1000                           │
│ top_p: 0.9                                 │
│ frequency_penalty: 0.1                     │
│ presence_penalty: 0.1                      │
╰────────────────────────────────────────────╯
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Case ID   ┃ Scores       ┃ Labels        ┃ Metrics         ┃ Assertions ┃ Duration ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━┩
│ test_case │ score1: 2.50 │ label1: hello │ accuracy: 0.950 │ ✔          │  100.0ms │
└───────────┴──────────────┴───────────────┴─────────────────┴────────────┴──────────┘
""")


async def test_evaluation_renderer_diff_with_experiment_metadata(sample_report_case: ReportCase):
    """Test EvaluationRenderer diff table with experiment metadata."""
    baseline_report = EvaluationReport(
        cases=[sample_report_case],
        name='baseline_report',
        experiment_metadata={'model': 'gpt-4', 'temperature': 0.5},
    )

    new_report = EvaluationReport(
        cases=[sample_report_case],
        name='new_report',
        experiment_metadata={'model': 'gpt-4o', 'temperature': 0.7},
    )

    output = new_report.render(
        baseline=baseline_report,
        include_input=False,
        include_metadata=False,
        include_expected_output=False,
        include_output=False,
        include_durations=True,
        include_total_duration=False,
        include_removed_cases=False,
        include_averages=True,
        include_errors=False,
        include_error_stacktrace=False,
        include_evaluator_failures=True,
        input_config={},
        metadata_config={},
        output_config={},
        score_configs={},
        label_configs={},
        metric_configs={},
        duration_config={},
        include_reasons=False,
    )

    assert output == snapshot("""\
╭─ Evaluation Diff: baseline_report → new_report ─╮
│ model: gpt-4 → gpt-4o                           │
│ temperature: 0.5 → 0.7                          │
╰─────────────────────────────────────────────────╯
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Case ID   ┃ Scores       ┃ Labels                 ┃ Metrics         ┃ Assertions ┃ Duration ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━┩
│ test_case │ score1: 2.50 │ label1: hello          │ accuracy: 0.950 │ ✔          │  100.0ms │
├───────────┼──────────────┼────────────────────────┼─────────────────┼────────────┼──────────┤
│ Averages  │ score1: 2.50 │ label1: {'hello': 1.0} │ accuracy: 0.950 │ 100.0% ✔   │  100.0ms │
└───────────┴──────────────┴────────────────────────┴─────────────────┴────────────┴──────────┘
""")


async def test_evaluation_renderer_diff_with_only_new_metadata(sample_report_case: ReportCase):
    """Test EvaluationRenderer diff table where only new report has metadata."""
    baseline_report = EvaluationReport(
        cases=[sample_report_case],
        name='baseline_report',
        experiment_metadata=None,  # No metadata
    )

    new_report = EvaluationReport(
        cases=[sample_report_case],
        name='new_report',
        experiment_metadata={'model': 'gpt-4o', 'temperature': 0.7},
    )

    output = new_report.render(
        baseline=baseline_report,
        include_input=False,
        include_metadata=False,
        include_expected_output=False,
        include_output=False,
        include_durations=True,
        include_total_duration=False,
        include_removed_cases=False,
        include_averages=False,
        include_errors=False,
        include_error_stacktrace=False,
        include_evaluator_failures=True,
        input_config={},
        metadata_config={},
        output_config={},
        score_configs={},
        label_configs={},
        metric_configs={},
        duration_config={},
        include_reasons=False,
    )

    assert output == snapshot("""\
╭─ Evaluation Diff: baseline_report → new_report ─╮
│ + model: gpt-4o                                 │
│ + temperature: 0.7                              │
╰─────────────────────────────────────────────────╯
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Case ID   ┃ Scores       ┃ Labels        ┃ Metrics         ┃ Assertions ┃ Duration ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━┩
│ test_case │ score1: 2.50 │ label1: hello │ accuracy: 0.950 │ ✔          │  100.0ms │
└───────────┴──────────────┴───────────────┴─────────────────┴────────────┴──────────┘
""")


async def test_evaluation_renderer_diff_with_only_baseline_metadata(sample_report_case: ReportCase):
    """Test EvaluationRenderer diff table where only baseline report has metadata."""
    baseline_report = EvaluationReport(
        cases=[sample_report_case],
        name='baseline_report',
        experiment_metadata={'model': 'gpt-4', 'temperature': 0.5},
    )

    new_report = EvaluationReport(
        cases=[sample_report_case],
        name='new_report',
        experiment_metadata=None,  # No metadata
    )

    output = new_report.render(
        baseline=baseline_report,
        include_input=False,
        include_metadata=False,
        include_expected_output=False,
        include_output=False,
        include_durations=True,
        include_total_duration=False,
        include_removed_cases=False,
        include_averages=False,
        include_errors=False,
        include_error_stacktrace=False,
        include_evaluator_failures=True,
        input_config={},
        metadata_config={},
        output_config={},
        score_configs={},
        label_configs={},
        metric_configs={},
        duration_config={},
        include_reasons=False,
    )

    assert output == snapshot("""\
╭─ Evaluation Diff: baseline_report → new_report ─╮
│ - model: gpt-4                                  │
│ - temperature: 0.5                              │
╰─────────────────────────────────────────────────╯
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Case ID   ┃ Scores       ┃ Labels        ┃ Metrics         ┃ Assertions ┃ Duration ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━┩
│ test_case │ score1: 2.50 │ label1: hello │ accuracy: 0.950 │ ✔          │  100.0ms │
└───────────┴──────────────┴───────────────┴─────────────────┴────────────┴──────────┘
""")


async def test_evaluation_renderer_diff_with_same_metadata(sample_report_case: ReportCase):
    """Test EvaluationRenderer diff table where both reports have the same metadata."""
    metadata = {'model': 'gpt-4o', 'temperature': 0.7}

    baseline_report = EvaluationReport(
        cases=[sample_report_case],
        name='baseline_report',
        experiment_metadata=metadata,
    )

    new_report = EvaluationReport(
        cases=[sample_report_case],
        name='new_report',
        experiment_metadata=metadata,
    )

    output = new_report.render(
        include_input=False,
        include_metadata=False,
        include_expected_output=False,
        include_output=False,
        include_durations=True,
        include_total_duration=False,
        include_removed_cases=False,
        include_averages=False,
        include_error_stacktrace=False,
        include_evaluator_failures=True,
        input_config={},
        metadata_config={},
        output_config={},
        score_configs={},
        label_configs={},
        metric_configs={},
        duration_config={},
        include_reasons=False,
        baseline=baseline_report,
        include_errors=False,  # Prevent failures table from being added
    )
    assert output == snapshot("""\
╭─ Evaluation Diff: baseline_report → new_report ─╮
│ model: gpt-4o                                   │
│ temperature: 0.7                                │
╰─────────────────────────────────────────────────╯
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Case ID   ┃ Scores       ┃ Labels        ┃ Metrics         ┃ Assertions ┃ Duration ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━┩
│ test_case │ score1: 2.50 │ label1: hello │ accuracy: 0.950 │ ✔          │  100.0ms │
└───────────┴──────────────┴───────────────┴─────────────────┴────────────┴──────────┘
""")


async def test_evaluation_renderer_diff_with_changed_metadata(sample_report_case: ReportCase):
    """Test EvaluationRenderer diff table where both reports have the same metadata."""

    baseline_report = EvaluationReport(
        cases=[sample_report_case],
        name='baseline_report',
        experiment_metadata={
            'updated-key': 'original value',
            'preserved-key': 'preserved value',
            'old-key': 'old value',
        },
    )

    new_report = EvaluationReport(
        cases=[sample_report_case],
        name='new_report',
        experiment_metadata={
            'updated-key': 'updated value',
            'preserved-key': 'preserved value',
            'new-key': 'new value',
        },
    )

    output = new_report.render(
        include_input=False,
        include_metadata=False,
        include_expected_output=False,
        include_output=False,
        include_durations=True,
        include_total_duration=False,
        include_removed_cases=False,
        include_averages=False,
        include_error_stacktrace=False,
        include_evaluator_failures=True,
        input_config={},
        metadata_config={},
        output_config={},
        score_configs={},
        label_configs={},
        metric_configs={},
        duration_config={},
        include_reasons=False,
        baseline=baseline_report,
        include_errors=False,  # Prevent failures table from being added
    )
    assert output == snapshot("""\
╭─ Evaluation Diff: baseline_report → new_report ─╮
│ + new-key: new value                            │
│ - old-key: old value                            │
│ preserved-key: preserved value                  │
│ updated-key: original value → updated value     │
╰─────────────────────────────────────────────────╯
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Case ID   ┃ Scores       ┃ Labels        ┃ Metrics         ┃ Assertions ┃ Duration ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━┩
│ test_case │ score1: 2.50 │ label1: hello │ accuracy: 0.950 │ ✔          │  100.0ms │
└───────────┴──────────────┴───────────────┴─────────────────┴────────────┴──────────┘
""")


async def test_evaluation_renderer_diff_with_no_metadata(sample_report_case: ReportCase):
    """Test EvaluationRenderer diff table where both reports have the same metadata."""

    baseline_report = EvaluationReport(
        cases=[sample_report_case],
        name='baseline_report',
    )

    new_report = EvaluationReport(
        cases=[sample_report_case],
        name='new_report',
    )

    output = new_report.render(
        include_input=False,
        include_metadata=False,
        include_expected_output=False,
        include_output=False,
        include_durations=True,
        include_total_duration=False,
        include_removed_cases=False,
        include_averages=False,
        include_error_stacktrace=False,
        include_evaluator_failures=True,
        input_config={},
        metadata_config={},
        output_config={},
        score_configs={},
        label_configs={},
        metric_configs={},
        duration_config={},
        include_reasons=False,
        baseline=baseline_report,
        include_errors=False,  # Prevent failures table from being added
    )
    assert output == snapshot("""\
                    Evaluation Diff: baseline_report → new_report                     \n\
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Case ID   ┃ Scores       ┃ Labels        ┃ Metrics         ┃ Assertions ┃ Duration ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━┩
│ test_case │ score1: 2.50 │ label1: hello │ accuracy: 0.950 │ ✔          │  100.0ms │
└───────────┴──────────────┴───────────────┴─────────────────┴────────────┴──────────┘
""")
