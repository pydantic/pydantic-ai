from __future__ import annotations

import random

import pytest

from tests.conftest import try_import

with try_import() as imports_successful:
    from pydantic_evals.evaluators import PrecisionRecallEvaluator, ReportEvaluatorContext, ROCAUCEvaluator
    from pydantic_evals.reporting import EvaluationReport, ReportCase
    from pydantic_evals.reporting.analyses import LinePlot, PrecisionRecall, ScalarResult

pytestmark = [
    pytest.mark.benchmark(max_time=15),
    pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed'),
]


@pytest.fixture
def blockbuster_enabled() -> bool:
    return False


@pytest.fixture(params=[(256, 32), (4096, 2048)], ids=['256-cases-32-buckets', '4096-cases-2048-buckets'])
def curve_context(request: pytest.FixtureRequest) -> tuple[ReportEvaluatorContext[None, bool, None], int]:
    n_cases, buckets = request.param
    cases = [
        ReportCase[None, bool, None](
            name=f'case-{i}',
            inputs=None,
            metadata=None,
            expected_output=bool(i % 2),
            output=False,
            metrics={'score': (i // (n_cases // buckets)) / buckets},
            attributes={},
            scores={},
            labels={},
            assertions={},
            task_duration=0.0,
            total_duration=0.0,
        )
        for i in range(n_cases)
    ]
    random.Random(0).shuffle(cases)
    report = EvaluationReport(name='balanced-tied-scores', cases=cases)
    return ReportEvaluatorContext(name=report.name, report=report, experiment_metadata=None), buckets


@pytest.fixture(params=['precision-recall', 'roc-auc'])
def curve_evaluator(
    request: pytest.FixtureRequest,
    curve_context: tuple[ReportEvaluatorContext[None, bool, None], int],
) -> PrecisionRecallEvaluator | ROCAUCEvaluator:
    evaluator_type = PrecisionRecallEvaluator if request.param == 'precision-recall' else ROCAUCEvaluator
    evaluator = evaluator_type(
        score_from='metrics', score_key='score', positive_from='expected_output', n_thresholds=16
    )
    evaluator.evaluate(curve_context[0])
    return evaluator


def test_eval_curves(
    curve_context: tuple[ReportEvaluatorContext[None, bool, None], int],
    curve_evaluator: PrecisionRecallEvaluator | ROCAUCEvaluator,
) -> None:
    ctx, buckets = curve_context
    chart, scalar = curve_evaluator.evaluate(ctx)
    assert isinstance(scalar, ScalarResult)
    assert isinstance(scalar.value, float)
    assert scalar.title == f'{curve_evaluator.title} AUC'
    if isinstance(curve_evaluator, PrecisionRecallEvaluator):
        assert isinstance(chart, PrecisionRecall)
        assert len(chart.curves) == 1
        assert chart.curves[0].auc == 0.5 + 0.25 / buckets
        assert scalar.value == chart.curves[0].auc
    else:
        assert isinstance(chart, LinePlot)
        assert len(chart.curves) == 2
        assert scalar.value == 0.5
        assert chart.curves[1].style == 'dashed'
    assert len(chart.curves[0].points) == 16
