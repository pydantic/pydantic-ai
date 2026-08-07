from __future__ import annotations as _annotations

import io
from typing import Any

import pytest
from pydantic import BaseModel

from .._inline_snapshot import snapshot
from ..conftest import try_import

with try_import() as imports_successful:
    from rich.console import Console

    from pydantic_evals.evaluators import EvaluationResult, Evaluator, EvaluatorContext
    from pydantic_evals.evaluators.evaluator import EvaluatorFailure
    from pydantic_evals.reporting import (
        EvaluationReport,
        EvaluationReportAdapter,
        RenderNumberConfig,
        RenderValueConfig,
        ReportCase,
        ReportCaseAdapter,
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
def sample_evaluator_output() -> dict[str, Any]:
    return {'correct': True, 'confidence': 0.95}


@pytest.fixture
def mock_evaluator() -> Evaluator[TaskInput, TaskOutput, TaskMetadata]:
    class MockEvaluator(Evaluator[TaskInput, TaskOutput, TaskMetadata]):
        def evaluate(self, ctx: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]) -> bool:
            raise NotImplementedError

    return MockEvaluator()


@pytest.fixture
def sample_evaluation_result(
    sample_evaluator_output: dict[str, Any], mock_evaluator: Evaluator[TaskInput, TaskOutput, TaskMetadata]
) -> EvaluationResult[bool]:
    return EvaluationResult(
        name='MockEvaluator',
        value=True,
        reason=None,
        source=mock_evaluator.as_spec(),
    )


@pytest.fixture
def sample_report_case(sample_evaluation_result: EvaluationResult[bool]) -> ReportCase:
    return ReportCase(
        name='test_case',
        inputs={'query': 'What is 2+2?'},
        output={'answer': '4'},
        expected_output={'answer': '4'},
        metadata={'difficulty': 'easy'},
        metrics={},
        attributes={},
        scores={},
        labels={},
        assertions={sample_evaluation_result.name: sample_evaluation_result},
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


async def test_report_init(sample_report_case: ReportCase):
    """Test EvaluationReport initialization."""
    report = EvaluationReport(
        cases=[sample_report_case],
        name='test_report',
    )

    assert report.name == 'test_report'
    assert len(report.cases) == 1


async def test_report_add_case(
    sample_report: EvaluationReport,
    sample_report_case: ReportCase,
    mock_evaluator: Evaluator[TaskInput, TaskOutput, TaskMetadata],
):
    """Test adding cases to a report."""
    initial_case_count = len(sample_report.cases)

    # Create a new case
    new_case = ReportCase(
        name='new_case',
        inputs={'query': 'What is 3+3?'},
        output={'answer': '6'},
        expected_output={'answer': '6'},
        metadata={'difficulty': 'medium'},
        metrics={},
        attributes={},
        scores={},
        labels={},
        assertions={},
        task_duration=0.1,
        total_duration=0.15,
        trace_id='test-trace-id-2',
        span_id='test-span-id-2',
    )

    # Add the case
    sample_report.cases.append(new_case)

    # Check that the case was added
    assert len(sample_report.cases) == initial_case_count + 1
    assert sample_report.cases[-1].name == 'new_case'


async def test_report_case_aggregate():
    """Test ReportCaseAggregate functionality."""
    # Create a case aggregate
    aggregate = ReportCaseAggregate(
        name='test_aggregate',
        scores={'test_evaluator': 0.75},
        labels={'test_label': {'value': 0.75}},
        metrics={'accuracy': 0.75},
        assertions=0.75,
        task_duration=0.1,
        total_duration=0.2,
    )

    assert aggregate.name == 'test_aggregate'
    assert aggregate.scores['test_evaluator'] == 0.75
    assert aggregate.labels['test_label']['value'] == 0.75
    assert aggregate.metrics['accuracy'] == 0.75
    assert aggregate.assertions == 0.75
    assert aggregate.task_duration == 0.1
    assert aggregate.total_duration == 0.2


async def test_report_serialization(sample_report: EvaluationReport):
    """Test serializing a report to dict."""
    # Serialize the report
    serialized = EvaluationReportAdapter.dump_python(sample_report)

    # Check the serialized structure
    assert 'cases' in serialized
    assert 'name' in serialized

    # Check the values
    assert serialized['name'] == 'test_report'
    assert len(serialized['cases']) == 1


async def test_report_with_error(mock_evaluator: Evaluator[TaskInput, TaskOutput, TaskMetadata]):
    """Test a report with error in one of the cases."""
    # Create an evaluator output
    error_output = EvaluationResult[bool](
        name='error_evaluator',
        value=False,  # No result
        reason='Test error message',
        source=mock_evaluator.as_spec(),
    )

    # Create a case
    error_case = ReportCase(
        name='error_case',
        inputs={'query': 'What is 1/0?'},
        output=None,
        expected_output={'answer': 'Error'},
        metadata={'difficulty': 'hard'},
        metrics={},
        attributes={'error': 'Division by zero'},
        scores={},
        labels={},
        assertions={error_output.name: error_output},
        task_duration=0.05,
        total_duration=0.1,
        trace_id='test-error-trace-id',
        span_id='test-error-span-id',
    )

    # Create a report with the error case
    report = EvaluationReport(
        cases=[error_case],
        name='error_report',
    )

    assert ReportCaseAdapter.dump_python(report.cases[0]) == snapshot(
        {
            'assertions': {
                'error_evaluator': {
                    'name': 'error_evaluator',
                    'reason': 'Test error message',
                    'source': {'arguments': None, 'name': 'MockEvaluator'},
                    'value': False,
                    'evaluator_version': None,
                }
            },
            'attributes': {'error': 'Division by zero'},
            'evaluator_failures': [],
            'expected_output': {'answer': 'Error'},
            'inputs': {'query': 'What is 1/0?'},
            'labels': {},
            'metadata': {'difficulty': 'hard'},
            'metrics': {},
            'name': 'error_case',
            'output': None,
            'scores': {},
            'span_id': 'test-error-span-id',
            'source_case_name': None,
            'task_duration': 0.05,
            'total_duration': 0.1,
            'trace_id': 'test-error-trace-id',
        }
    )


async def test_render_config():
    """Test render configuration objects."""
    # Test RenderNumberConfig
    number_config: RenderNumberConfig = {
        'value_formatter': '{:.0%}',
        'diff_formatter': '{:+.0%}',
        'diff_atol': 0.01,
        'diff_rtol': 0.05,
        'diff_increase_style': 'green',
        'diff_decrease_style': 'red',
    }

    # Assert the dictionary has the expected keys
    assert 'value_formatter' in number_config
    assert 'diff_formatter' in number_config
    assert 'diff_atol' in number_config
    assert 'diff_rtol' in number_config
    assert 'diff_increase_style' in number_config
    assert 'diff_decrease_style' in number_config

    # Test RenderValueConfig
    value_config: RenderValueConfig = {
        'value_formatter': '{value}',
        'diff_checker': lambda x, y: x != y,
        'diff_formatter': None,
        'diff_style': 'magenta',
    }

    # Assert the dictionary has the expected keys
    assert 'value_formatter' in value_config
    assert 'diff_checker' in value_config
    assert 'diff_formatter' in value_config
    assert 'diff_style' in value_config


def _print_with_encoding(report: EvaluationReport, encoding: str, **kwargs: Any) -> str:
    """Render `report` through a `Console` whose stream can't encode non-ASCII characters.

    Simulates e.g. a Windows console with output redirected to a non-UTF-8 stream.
    """
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding=encoding)
    console = Console(width=200, file=stream)
    report.print(console=console, **kwargs)
    stream.flush()
    return buffer.getvalue().decode(encoding)


async def test_print_ascii_fallback_for_non_utf8_console(
    mock_evaluator: Evaluator[TaskInput, TaskOutput, TaskMetadata],
):
    """Regression test for https://github.com/pydantic/pydantic-ai/issues/7281.

    `print()` used to raise `UnicodeEncodeError` whenever the target console's stream couldn't
    encode the `✔`/`✗` assertion glyphs (e.g. Windows with stdout redirected to a non-UTF-8
    stream). It should fall back to ASCII markers instead.
    """
    case = ReportCase(
        name='case1',
        inputs={'query': 'What is 2+2?'},
        output={'answer': '4'},
        expected_output={'answer': '4'},
        metadata={'difficulty': 'easy'},
        metrics={},
        attributes={},
        scores={},
        labels={},
        assertions={
            'pass': EvaluationResult(name='pass', value=True, reason=None, source=mock_evaluator.as_spec()),
            'fail': EvaluationResult(name='fail', value=False, reason=None, source=mock_evaluator.as_spec()),
        },
        task_duration=0.1,
        total_duration=0.2,
        trace_id='test-trace-id',
        span_id='test-span-id',
    )
    report = EvaluationReport(cases=[case], name='report')

    output = _print_with_encoding(report, 'cp1252')

    assert '✔' not in output
    assert '✗' not in output
    assert 'v' in output
    assert 'x' in output


async def test_print_diff_ascii_fallback_for_non_utf8_console(
    mock_evaluator: Evaluator[TaskInput, TaskOutput, TaskMetadata],
):
    """Regression test for https://github.com/pydantic/pydantic-ai/issues/7281, for diff reports.

    Every `→` used to join before/after values in a diff report (assertions, evaluator failures,
    scores/metrics/durations, and experiment metadata) is subject to the same `UnicodeEncodeError`
    as the assertion glyphs, so this exercises them together via `print(baseline=...)`.
    """
    baseline_case = ReportCase(
        name='case1',
        inputs={'query': 'old input'},
        output={'answer': 'old'},
        expected_output={'answer': 'old'},
        metadata={'difficulty': 'easy'},
        metrics={'accuracy': 0.5},
        attributes={},
        scores={},
        labels={},
        assertions={'a': EvaluationResult(name='a', value=False, reason=None, source=mock_evaluator.as_spec())},
        task_duration=0.2,
        total_duration=0.3,
        trace_id='baseline-trace-id',
        span_id='baseline-span-id',
        evaluator_failures=[
            EvaluatorFailure(
                name='OldEvaluator',
                error_message='old failure',
                error_stacktrace='old stacktrace',
                source=mock_evaluator.as_spec(),
            )
        ],
    )
    new_case = ReportCase(
        name='case1',
        inputs={'query': 'new input'},
        output={'answer': 'new'},
        expected_output={'answer': 'new'},
        metadata={'difficulty': 'hard'},
        metrics={'accuracy': 0.9},
        attributes={},
        scores={},
        labels={},
        assertions={
            'a': EvaluationResult(name='a', value=True, reason=None, source=mock_evaluator.as_spec()),
            'b': EvaluationResult(name='b', value=False, reason=None, source=mock_evaluator.as_spec()),
        },
        task_duration=0.1,
        total_duration=0.15,
        trace_id='new-trace-id',
        span_id='new-span-id',
        evaluator_failures=[
            EvaluatorFailure(
                name='NewEvaluator',
                error_message='new failure',
                error_stacktrace='new stacktrace',
                source=mock_evaluator.as_spec(),
            )
        ],
    )
    baseline_report = EvaluationReport(
        cases=[baseline_case], name='baseline_report', experiment_metadata={'run': 'a', 'stable': 'x'}
    )
    new_report = EvaluationReport(cases=[new_case], name='new_report', experiment_metadata={'run': 'b', 'stable': 'x'})

    output = _print_with_encoding(new_report, 'cp1252', baseline=baseline_report)

    assert '✔' not in output
    assert '✗' not in output
    assert '→' not in output
    assert '->' in output
