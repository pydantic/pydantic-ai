from __future__ import annotations as _annotations

from dataclasses import dataclass, replace
from io import BytesIO, TextIOWrapper

import pytest
from pydantic import BaseModel
from rich.console import Console

from .._inline_snapshot import snapshot
from ..conftest import try_import
from .utils import render_table, trim_trailing_whitespace

with try_import() as imports_successful:
    from pydantic_evals.evaluators import EvaluationResult, Evaluator, EvaluatorContext
    from pydantic_evals.evaluators.evaluator import EvaluatorFailure
    from pydantic_evals.reporting import (
        EvaluationRenderer,
        EvaluationReport,
        ReportCase,
        ReportCaseAggregate,
        ReportCaseFailure,
    )

pytestmark = [pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed')]


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
        width=300,
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
        width=300,
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
        width=300,
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
        width=300,
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
        width=300,
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
        width=300,
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
        width=300,
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
        width=300,
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


async def test_render_shows_bracketed_text_verbatim(mock_evaluator: Evaluator[TaskInput, TaskOutput, TaskMetadata]):
    """Case, task and evaluator text containing square brackets is shown as written, not parsed as Rich markup.

    A closing tag with no opening tag, like the `[/INST]` of a leaked chat template, used to make rendering raise
    `MarkupError`, and text that looks like a tag, like `[int]` or pydantic's `[type=...]` error details, was dropped.
    """
    source = mock_evaluator.as_spec()
    case = ReportCase(
        name='parse list[int]',
        inputs='[INST] What is 2+2? [/INST]',
        metadata={'source': '[docs](https://example.com)'},
        expected_output='list[int]',
        output='4 [/INST]',
        metrics={'cost[usd]': 0.5},
        attributes={},
        scores={
            'match[strict]': EvaluationResult(
                name='match[strict]', value=1.0, reason='answer is followed by [/INST]', source=source
            )
        },
        labels={'category': EvaluationResult(name='category', value='[/INST]', reason=None, source=source)},
        assertions={
            'is_json[strict]': EvaluationResult(
                name='is_json[strict]', value=False, reason='found [/INST] after the answer', source=source
            )
        },
        task_duration=0.1,
        total_duration=0.2,
        evaluator_failures=[
            EvaluatorFailure(
                name='Judge[gpt]',
                error_message='judge replied [/INST] instead of a verdict',
                error_stacktrace='',
                source=source,
            )
        ],
    )
    failure = ReportCaseFailure(
        name='parse list[str]',
        inputs='[INST] Name a city [/INST]',
        metadata=None,
        expected_output=None,
        error_message='ValidationError: Input should be a valid string [type=string_type, input_value=1, input_type=int]',
        error_stacktrace='Traceback (most recent call last):\nValueError: model emitted [/INST]',
    )
    report = EvaluationReport(cases=[case], failures=[failure], name='test_report')

    output = report.render(
        width=300,
        include_input=True,
        include_metadata=True,
        include_expected_output=True,
        include_output=True,
        include_reasons=True,
        include_error_stacktrace=True,
    )
    assert trim_trailing_whitespace(output) == snapshot("""\
                                                                                                                                      Evaluation Summary: test_report
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Case ID         ┃ Inputs                      ┃ Metadata                              ┃ Expected Output ┃ Outputs   ┃ Scores                                 ┃ Labels                     ┃ Metrics          ┃ Assertions                            ┃ Evaluator Failures                     ┃ Duration ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ parse list[int] │ [INST] What is 2+2? [/INST] │ {'source':                            │ list[int]       │ 4 [/INST] │ match[strict]: 1.00                    │ category: [/INST]          │ cost[usd]: 0.500 │ is_json[strict]: ✗                    │ Judge[gpt]: judge replied [/INST]      │  100.0ms │
│                 │                             │ '[docs](https://example.com)'}        │                 │           │   Reason: answer is followed by        │                            │                  │   Reason: found [/INST] after the     │ instead of a verdict                   │          │
│                 │                             │                                       │                 │           │ [/INST]                                │                            │                  │ answer                                │                                        │          │
│                 │                             │                                       │                 │           │                                        │                            │                  │                                       │                                        │          │
│                 │                             │                                       │                 │           │                                        │                            │                  │                                       │                                        │          │
├─────────────────┼─────────────────────────────┼───────────────────────────────────────┼─────────────────┼───────────┼────────────────────────────────────────┼────────────────────────────┼──────────────────┼───────────────────────────────────────┼────────────────────────────────────────┼──────────┤
│ Averages        │                             │                                       │                 │           │ match[strict]: 1.00                    │ category: {'[/INST]': 1.0} │ cost[usd]: 0.500 │ 0.0% ✔                                │                                        │  100.0ms │
└─────────────────┴─────────────────────────────┴───────────────────────────────────────┴─────────────────┴───────────┴────────────────────────────────────────┴────────────────────────────┴──────────────────┴───────────────────────────────────────┴────────────────────────────────────────┴──────────┘
                                                                                                     Case Failures
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Case ID         ┃ Inputs                     ┃ Metadata  ┃ Expected Output ┃ Error Message                                                                                     ┃ Error Stacktrace                   ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ parse list[str] │ [INST] Name a city [/INST] │ <missing> │ <missing>       │ ValidationError: Input should be a valid string [type=string_type, input_value=1, input_type=int] │ Traceback (most recent call last): │
│                 │                            │           │                 │                                                                                                   │ ValueError: model emitted [/INST]  │
└─────────────────┴────────────────────────────┴───────────┴─────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────┴────────────────────────────────────┘
""")


async def test_render_diff_shows_bracketed_text_verbatim(sample_report_case: ReportCase):
    """A diff between values that differ only inside square brackets still shows both values."""
    baseline_case = replace(
        sample_report_case,
        name='parse list[int]',
        output='list[int]',
        labels={'kind[raw]': replace(sample_report_case.labels['label1'], value='list[int]')},
    )
    new_case = replace(
        baseline_case,
        output='list[str]',
        labels={'kind[raw]': replace(sample_report_case.labels['label1'], value='[/INST]')},
    )
    baseline_report = EvaluationReport(cases=[baseline_case], name='baseline_report')
    new_report = EvaluationReport(cases=[new_case], name='new_report')

    output = new_report.render(width=300, baseline=baseline_report, include_output=True)
    assert trim_trailing_whitespace(output) == snapshot("""\
                                                     Evaluation Diff: baseline_report → new_report
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Case ID         ┃ Outputs               ┃ Scores       ┃ Labels                                           ┃ Metrics         ┃ Assertions ┃ Duration ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━┩
│ parse list[int] │ list[int] → list[str] │ score1: 2.50 │ kind[raw]: list[int] → [/INST]                   │ accuracy: 0.950 │ ✔          │  100.0ms │
├─────────────────┼───────────────────────┼──────────────┼──────────────────────────────────────────────────┼─────────────────┼────────────┼──────────┤
│ Averages        │                       │ score1: 2.50 │ kind[raw]: {'list[int]': 1.0} → {'[/INST]': 1.0} │ accuracy: 0.950 │ 100.0% ✔   │  100.0ms │
└─────────────────┴───────────────────────┴──────────────┴──────────────────────────────────────────────────┴─────────────────┴────────────┴──────────┘
""")


async def test_render_diff_shows_custom_formatter_text_verbatim(sample_report_case: ReportCase):
    """Text returned by a custom `value_formatter` or `diff_formatter` is shown as written, not parsed as Rich markup."""
    case = replace(sample_report_case, scores={}, labels={}, metrics={}, assertions={})
    baseline_report = EvaluationReport(cases=[replace(case, output='a')], name='baseline_report')
    new_report = EvaluationReport(cases=[replace(case, output='b')], name='new_report')

    output = new_report.render(
        width=300,
        baseline=baseline_report,
        include_output=True,
        include_averages=False,
        output_config={
            'value_formatter': lambda value: f'{value} [/INST]',
            'diff_formatter': lambda old, new: f'was {old}: list[int] [/INST]',
        },
    )
    assert trim_trailing_whitespace(output) == snapshot("""\
               Evaluation Diff: baseline_report → new_report
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Case ID   ┃ Outputs                                          ┃ Duration ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ test_case │ a [/INST] → b [/INST] (was a: list[int] [/INST]) │  100.0ms │
└───────────┴──────────────────────────────────────────────────┴──────────┘
""")


async def test_render_shows_trailing_backslashes_as_written(sample_report_case: ReportCase):
    """Text ending in backslashes is shown as written, both at the end of a cell and inside the diff style tag."""
    case = replace(sample_report_case, scores={}, labels={}, metrics={}, assertions={})
    baseline_report = EvaluationReport(
        cases=[replace(case, name='dir\\', output='x'), replace(case, name='c2', output='x\\')], name='baseline'
    )
    new_report = EvaluationReport(
        cases=[replace(case, name='dir\\', output='C:\\temp\\'), replace(case, name='c2', output='x\\\\')], name='new'
    )

    output = new_report.render(width=300, include_output=True, include_averages=False)
    assert trim_trailing_whitespace(output) == snapshot("""\
     Evaluation Summary: new
┏━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┓
┃ Case ID ┃ Outputs  ┃ Duration ┃
┡━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━┩
│ dir\\    │ C:\\temp\\ │  100.0ms │
├─────────┼──────────┼──────────┤
│ c2      │ x\\\\      │  100.0ms │
└─────────┴──────────┴──────────┘
""")

    output = new_report.render(width=300, baseline=baseline_report, include_output=True, include_averages=False)
    assert trim_trailing_whitespace(output) == snapshot("""\
   Evaluation Diff: baseline → new
┏━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Case ID ┃ Outputs      ┃ Duration ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ c2      │ x\\ → x\\\\     │  100.0ms │
├─────────┼──────────────┼──────────┤
│ dir\\    │ x → C:\\temp\\ │  100.0ms │
└─────────┴──────────────┴──────────┘
""")


def render_to_non_utf8_console(
    report: EvaluationReport, baseline: EvaluationReport | None = None, *, encoding: str = 'cp1252'
) -> str:
    """Print a report to a console whose stream can't encode Unicode report text, and return what was written.

    This is Windows with stdout redirected to a file or a pipe: Python uses UTF-8 for the console
    itself, but falls back to the ANSI code page for a redirected stream.
    """
    buffer = BytesIO()
    stream = TextIOWrapper(buffer, encoding=encoding, errors='strict', newline='')
    report.print(baseline=baseline, console=Console(file=stream, width=150))
    stream.flush()
    return trim_trailing_whitespace(buffer.getvalue().decode(encoding))


async def test_print_falls_back_to_ascii_glyphs_on_non_utf8_console(
    mock_evaluator: Evaluator[TaskInput, TaskOutput, TaskMetadata], sample_report_case: ReportCase
):
    """A report printed to a stream that can't encode `✔`/`✗` renders ASCII markers instead of raising."""
    failed_assertion = EvaluationResult(
        name='FailingEvaluator', value=False, reason=None, source=mock_evaluator.as_spec()
    )
    case = replace(
        sample_report_case, assertions={**sample_report_case.assertions, 'FailingEvaluator': failed_assertion}
    )
    report = EvaluationReport(cases=[case], name='test_report')

    assert render_to_non_utf8_console(report) == snapshot("""\
                                Evaluation Summary: test_report
+---------------------------------------------------------------------------------------------+
| Case ID   | Scores       | Labels                 | Metrics         | Assertions | Duration |
|-----------+--------------+------------------------+-----------------+------------+----------|
| test_case | score1: 2.50 | label1: hello          | accuracy: 0.950 | vx         |  100.0ms |
|-----------+--------------+------------------------+-----------------+------------+----------|
| Averages  | score1: 2.50 | label1: {'hello': 1.0} | accuracy: 0.950 | 50.0% v    |  100.0ms |
+---------------------------------------------------------------------------------------------+
""")


async def test_print_falls_back_to_ascii_duration_units_on_cp932_console(sample_report_case: ReportCase):
    """Default duration values and diffs use ASCII units on a console that can't encode the micro sign."""
    baseline_report = EvaluationReport(
        cases=[replace(sample_report_case, task_duration=0.0001)], name='baseline_report'
    )
    report = EvaluationReport(cases=[replace(sample_report_case, task_duration=0.0002)], name='test_report')

    assert render_to_non_utf8_console(report, baseline=baseline_report, encoding='cp932') == snapshot("""\
                                    Evaluation Diff: baseline_report -> test_report
+----------------------------------------------------------------------------------------------------------------------+
| Case ID   | Scores       | Labels                 | Metrics         | Assertions |                          Duration |
|-----------+--------------+------------------------+-----------------+------------+-----------------------------------|
| test_case | score1: 2.50 | label1: hello          | accuracy: 0.950 | v          | 100us -> 200us (+100us / +100.0%) |
|-----------+--------------+------------------------+-----------------+------------+-----------------------------------|
| Averages  | score1: 2.50 | label1: {'hello': 1.0} | accuracy: 0.950 | 100.0% v   | 100us -> 200us (+100us / +100.0%) |
+----------------------------------------------------------------------------------------------------------------------+
""")


async def test_print_diff_falls_back_to_ascii_glyphs_on_non_utf8_console(
    mock_evaluator: Evaluator[TaskInput, TaskOutput, TaskMetadata], sample_report_case: ReportCase
):
    """Every `→` in a diff report degrades too: metadata panel, value/number diffs, and cell diffs."""
    failed_assertion = EvaluationResult(name='MockEvaluator', value=False, reason=None, source=mock_evaluator.as_spec())
    baseline_case = replace(
        sample_report_case,
        assertions={'MockEvaluator': failed_assertion},
        metrics={'accuracy': 0.95},
        evaluator_failures=[
            EvaluatorFailure(
                name='BaselineEvaluator',
                error_message='Baseline error',
                error_stacktrace='Baseline stacktrace',
                source=mock_evaluator.as_spec(),
            )
        ],
    )
    new_case = replace(
        sample_report_case,
        labels={'label1': replace(sample_report_case.labels['label1'], value='goodbye')},
        metrics={'accuracy': 0.97},
        evaluator_failures=[
            EvaluatorFailure(
                name='NewEvaluator',
                error_message='New error',
                error_stacktrace='New stacktrace',
                source=mock_evaluator.as_spec(),
            )
        ],
    )
    baseline_report = EvaluationReport(
        cases=[baseline_case], name='baseline_report', experiment_metadata={'model': 'gpt-4'}
    )
    new_report = EvaluationReport(cases=[new_case], name='new_report', experiment_metadata={'model': 'gpt-4o'})

    assert render_to_non_utf8_console(new_report, baseline=baseline_report) == snapshot("""\
+- Evaluation Diff: baseline_report -> new_report -+
| model: gpt-4 -> gpt-4o                           |
+--------------------------------------------------+
+----------------------------------------------------------------------------------------------------------------------------------------------------+
| Case ID   | Scores       | Labels                        | Metrics                  | Assertions         | Evaluator Failures           | Duration |
|-----------+--------------+-------------------------------+--------------------------+--------------------+------------------------------+----------|
| test_case | score1: 2.50 | label1: hello -> goodbye      | accuracy: 0.950 -> 0.970 | x -> v             | BaselineEvaluator: Baseline  |  100.0ms |
|           |              |                               |                          |                    | error                        |          |
|           |              |                               |                          |                    | ->                           |          |
|           |              |                               |                          |                    | NewEvaluator: New error      |          |
|-----------+--------------+-------------------------------+--------------------------+--------------------+------------------------------+----------|
| Averages  | score1: 2.50 | label1: {'hello': 1.0} ->     | accuracy: 0.950 -> 0.970 | 0.0% v -> 100.0% v |                              |  100.0ms |
|           |              | {'goodbye': 1.0}              |                          |                    |                              |          |
+----------------------------------------------------------------------------------------------------------------------------------------------------+
""")


async def test_print_diff_title_falls_back_to_ascii_glyphs_on_non_utf8_console(sample_report_case: ReportCase):
    """Without an experiment-metadata panel the diff name lands in the table title, which degrades as well."""
    baseline_report = EvaluationReport(cases=[sample_report_case], name='baseline_report')
    new_report = EvaluationReport(cases=[replace(sample_report_case, metrics={'accuracy': 0.97})], name='new_report')

    assert render_to_non_utf8_console(new_report, baseline=baseline_report) == snapshot("""\
                             Evaluation Diff: baseline_report -> new_report
+------------------------------------------------------------------------------------------------------+
| Case ID   | Scores       | Labels                 | Metrics                  | Assertions | Duration |
|-----------+--------------+------------------------+--------------------------+------------+----------|
| test_case | score1: 2.50 | label1: hello          | accuracy: 0.950 -> 0.970 | v          |  100.0ms |
|-----------+--------------+------------------------+--------------------------+------------+----------|
| Averages  | score1: 2.50 | label1: {'hello': 1.0} | accuracy: 0.950 -> 0.970 | 100.0% v   |  100.0ms |
+------------------------------------------------------------------------------------------------------+
""")


async def test_console_table_renders_ascii_glyphs_when_asked(
    mock_evaluator: Evaluator[TaskInput, TaskOutput, TaskMetadata], sample_report_case: ReportCase
):
    """A caller holding its own non-UTF-8 console can ask `console_table` for the fallback and print it."""
    failed_assertion = EvaluationResult(
        name='FailingEvaluator', value=False, reason=None, source=mock_evaluator.as_spec()
    )
    case = replace(
        sample_report_case, assertions={**sample_report_case.assertions, 'FailingEvaluator': failed_assertion}
    )
    report = EvaluationReport(cases=[replace(case, task_duration=0.0001)], name='test_report')

    buffer = BytesIO()
    stream = TextIOWrapper(buffer, encoding='cp932', errors='strict', newline='')
    console = Console(file=stream, width=150)
    console.print(report.console_table(ascii_only=True, duration_config={}))
    stream.flush()

    assert trim_trailing_whitespace(buffer.getvalue().decode('cp932')) == snapshot("""\
                                Evaluation Summary: test_report
+---------------------------------------------------------------------------------------------+
| Case ID   | Scores       | Labels                 | Metrics         | Assertions | Duration |
|-----------+--------------+------------------------+-----------------+------------+----------|
| test_case | score1: 2.50 | label1: hello          | accuracy: 0.950 | vx         |    100us |
|-----------+--------------+------------------------+-----------------+------------+----------|
| Averages  | score1: 2.50 | label1: {'hello': 1.0} | accuracy: 0.950 | 50.0% v    |    100us |
+---------------------------------------------------------------------------------------------+
""")


async def test_console_table_keeps_custom_duration_formatters_on_ascii_only(sample_report_case: ReportCase):
    """ASCII fallback applies only to renderer-supplied duration formatters."""
    report = EvaluationReport(cases=[replace(sample_report_case, task_duration=0.0001)], name='test_report')

    table = report.console_table(
        ascii_only=True,
        duration_config={'value_formatter': lambda _: 'custom µs'},
    )

    assert 'custom µs' in render_table(table)
