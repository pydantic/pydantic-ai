from __future__ import annotations

import json
import random
from typing import Literal

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_evals.evaluators import (
        EvaluationResult,
        PrecisionRecallEvaluator,
        ReportEvaluatorContext,
        ROCAUCEvaluator,
    )
    from pydantic_evals.evaluators.spec import EvaluatorSpec
    from pydantic_evals.reporting import EvaluationReport, ReportCase
    from pydantic_evals.reporting.analyses import (
        LinePlot,
        LinePlotCurve,
        LinePlotPoint,
        PrecisionRecall,
        PrecisionRecallCurve,
        PrecisionRecallPoint,
        ReportAnalysis,
        ScalarResult,
    )

pytestmark = pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed')


def _curve_context(
    rows: list[tuple[float | None, bool | None]],
    score_from: Literal['scores', 'metrics'],
    positive_from: Literal['expected_output', 'assertions', 'labels'],
) -> ReportEvaluatorContext[None, bool, None]:
    source = EvaluatorSpec(name='test', arguments=None)
    cases: list[ReportCase[None, bool, None]] = []
    for i, (score, positive) in enumerate(rows):
        cases.append(
            ReportCase[None, bool, None](
                name=f'case-{i}',
                inputs=None,
                metadata=None,
                expected_output=positive if positive_from == 'expected_output' else None,
                output=False,
                metrics={'score': score} if score_from == 'metrics' and score is not None else {},
                attributes={},
                scores={'score': EvaluationResult[int | float](name='score', value=score, reason=None, source=source)}
                if score_from == 'scores' and score is not None
                else {},
                labels={
                    'positive': EvaluationResult(
                        name='positive', value='yes' if positive else '', reason=None, source=source
                    )
                }
                if positive_from == 'labels' and positive is not None
                else {},
                assertions={'positive': EvaluationResult(name='positive', value=positive, reason=None, source=source)}
                if positive_from == 'assertions' and positive is not None
                else {},
                task_duration=0.0,
                total_duration=0.0,
            )
        )
    report = EvaluationReport(name='curve-counts', cases=cases)
    return ReportEvaluatorContext(name=report.name, report=report, experiment_metadata=None)


def _quadratic_auc(points: list[tuple[float, float]]) -> float:
    auc = 0.0
    for i in range(1, len(points)):
        auc += abs(points[i][0] - points[i - 1][0]) * (points[i][1] + points[i - 1][1]) / 2
    return auc


def _quadratic_results(
    scored_cases: list[tuple[float, bool]],
    evaluator: PrecisionRecallEvaluator | ROCAUCEvaluator,
    name: str,
) -> list[ReportAnalysis]:
    """Reproduce the original scans, anchors, AUC arithmetic, and display sampling."""
    total_positives = sum(1 for _, positive in scored_cases if positive)
    total_negatives = len(scored_cases) - total_positives
    thresholds = sorted({score for score, _ in scored_cases}, reverse=True)

    if isinstance(evaluator, PrecisionRecallEvaluator):
        if not scored_cases:
            return [
                PrecisionRecall(title=evaluator.title, curves=[]),
                ScalarResult(title=f'{evaluator.title} AUC', value=float('nan')),
            ]
        pr_points = [PrecisionRecallPoint(threshold=thresholds[0], precision=1.0, recall=0.0)]
        for threshold in thresholds:
            tp = sum(1 for score, positive in scored_cases if score >= threshold and positive)
            fp = sum(1 for score, positive in scored_cases if score >= threshold and not positive)
            fn = total_positives - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall = tp / (fn + tp) if (fn + tp) > 0 else 0.0
            pr_points.append(PrecisionRecallPoint(threshold=threshold, precision=precision, recall=recall))
        auc = _quadratic_auc([(point.recall, point.precision) for point in pr_points])
        if len(pr_points) > evaluator.n_thresholds and evaluator.n_thresholds > 1:
            indices = sorted(
                {int(i * (len(pr_points) - 1) / (evaluator.n_thresholds - 1)) for i in range(evaluator.n_thresholds)}
            )
            pr_points = [pr_points[i] for i in indices]
        return [
            PrecisionRecall(title=evaluator.title, curves=[PrecisionRecallCurve(name=name, points=pr_points, auc=auc)]),
            ScalarResult(title=f'{evaluator.title} AUC', value=auc),
        ]

    roc_points: list[tuple[float, float]] = [(0.0, 0.0)]
    curves: list[LinePlotCurve] = []
    auc = float('nan')
    if scored_cases and total_positives > 0 and total_negatives > 0:
        for threshold in thresholds:
            tp = sum(1 for score, positive in scored_cases if score >= threshold and positive)
            fp = sum(1 for score, positive in scored_cases if score >= threshold and not positive)
            tpr = tp / total_positives
            fpr = fp / total_negatives
            roc_points.append((fpr, tpr))
        roc_points.sort()
        auc = _quadratic_auc(roc_points)
        if len(roc_points) > evaluator.n_thresholds and evaluator.n_thresholds > 1:
            indices = sorted(
                {int(i * (len(roc_points) - 1) / (evaluator.n_thresholds - 1)) for i in range(evaluator.n_thresholds)}
            )
            roc_points = [roc_points[i] for i in indices]
        curves = [
            LinePlotCurve(name=f'{name} (AUC: {auc:.3f})', points=[LinePlotPoint(x=x, y=y) for x, y in roc_points]),
            LinePlotCurve(name='Random', points=[LinePlotPoint(x=0, y=0), LinePlotPoint(x=1, y=1)], style='dashed'),
        ]
    return [
        LinePlot(
            title=evaluator.title,
            x_label='False Positive Rate',
            y_label='True Positive Rate',
            x_range=(0, 1),
            y_range=(0, 1),
            curves=curves,
        ),
        ScalarResult(title=f'{evaluator.title} AUC', value=auc),
    ]


def _check_curve(
    rows: list[tuple[float | None, bool | None]],
    evaluator: PrecisionRecallEvaluator | ROCAUCEvaluator,
) -> None:
    ctx = _curve_context(rows, evaluator.score_from, evaluator.positive_from)
    scored_cases = [(float(score), positive) for score, positive in rows if score is not None and positive is not None]
    expected = _quadratic_results(scored_cases, evaluator, ctx.name)
    actual = evaluator.evaluate(ctx)
    assert [analysis.model_dump_json() for analysis in actual] == [analysis.model_dump_json() for analysis in expected]
    # Standard JSON encoding distinguishes NaN, infinities, and signed zero without approximate equality.
    assert json.dumps([analysis.model_dump() for analysis in actual]) == json.dumps(
        [analysis.model_dump() for analysis in expected]
    )


_NAN = float('nan')


@pytest.mark.parametrize('evaluator_type', [PrecisionRecallEvaluator, ROCAUCEvaluator] if imports_successful() else [])
@pytest.mark.parametrize('n_thresholds', [-5, 0, 1, 2, 3, 4, 7, 1000])
@pytest.mark.parametrize(
    'rows',
    [
        pytest.param([], id='empty'),
        pytest.param([(1.0, True)], id='single-positive'),
        pytest.param([(1.0, False)], id='single-negative'),
        pytest.param([(0.9, True), (0.1, True)], id='all-positive'),
        pytest.param([(0.9, False), (0.1, False)], id='all-negative'),
        pytest.param([(0.5, False), (0.5, True), (0.5, False)], id='all-tied'),
        pytest.param([(0.2, False), (0.9, True), (0.2, True), (0.9, False), (0.5, True)], id='mixed-ties'),
        pytest.param([(-0.0, False), (0.0, True)], id='negative-zero-first'),
        pytest.param([(0.0, True), (-0.0, False)], id='positive-zero-first'),
        pytest.param([(float('inf'), True), (float('inf'), False)], id='positive-infinity-tie'),
        pytest.param([(float('-inf'), False), (float('-inf'), True)], id='negative-infinity-tie'),
        pytest.param(
            [(float('-inf'), True), (0.5, False), (float('inf'), False), (float('inf'), True), (-1.0, True)],
            id='infinities-and-finite',
        ),
        pytest.param([(_NAN, True)], id='single-nan'),
        pytest.param([(_NAN, True), (_NAN, False)], id='repeated-nan'),
        pytest.param([(float('nan'), True), (float('nan'), False)], id='distinct-nans'),
        pytest.param([(_NAN, False), (0.9, True), (0.2, False), (_NAN, True)], id='nan-first'),
        pytest.param([(0.9, True), (0.2, False), (_NAN, False)], id='nan-last'),
        pytest.param(
            [
                (float('inf'), True),
                (_NAN, True),
                (float('-inf'), False),
                (float('nan'), False),
                (-0.0, True),
                (0.0, False),
            ],
            id='nans-infinities-and-zeros',
        ),
        pytest.param([(None, True), (0.9, None), (0.5, True), (0.2, False)], id='missing-data'),
        pytest.param([(None, True), (0.5, None)], id='all-missing'),
    ],
)
def test_curve_counts_edge_cases(
    rows: list[tuple[float | None, bool | None]],
    evaluator_type: type[PrecisionRecallEvaluator] | type[ROCAUCEvaluator],
    n_thresholds: int,
) -> None:
    _check_curve(
        rows,
        evaluator_type(
            score_key='score', score_from='metrics', positive_from='expected_output', n_thresholds=n_thresholds
        ),
    )


@pytest.mark.parametrize('evaluator_type', [PrecisionRecallEvaluator, ROCAUCEvaluator] if imports_successful() else [])
@pytest.mark.parametrize('score_from', ['scores', 'metrics'])
@pytest.mark.parametrize('positive_from', ['expected_output', 'assertions', 'labels'])
def test_curve_counts_seeded_differential(
    evaluator_type: type[PrecisionRecallEvaluator] | type[ROCAUCEvaluator],
    score_from: Literal['scores', 'metrics'],
    positive_from: Literal['expected_output', 'assertions', 'labels'],
) -> None:
    for seed in (0, 42):
        rng = random.Random(seed)
        for size in (2, 17, 64):
            for scores in (
                [rng.uniform(-10, 10) for _ in range(size)],
                [-1.0, -0.0, 0.0, 0.25, 0.5, 1.0, float('inf'), float('-inf')],
                [-0.0, 0.0, 0.5, float('inf'), float('-inf'), _NAN, _NAN, float('nan')],
            ):
                rows: list[tuple[float | None, bool | None]] = [
                    (rng.choice(scores), bool(rng.getrandbits(1))) for _ in range(size)
                ]
                rng.shuffle(rows)
                rows.extend([(None, True), (0.75, None)])
                full_resolution = (
                    len({score for score, positive in rows if score is not None and positive is not None}) + 1
                )
                for n_thresholds in (-1, 0, 1, 3, full_resolution):
                    _check_curve(
                        rows,
                        evaluator_type(
                            score_key='score',
                            score_from=score_from,
                            positive_from=positive_from,
                            positive_key='positive' if positive_from != 'expected_output' else None,
                            n_thresholds=n_thresholds,
                        ),
                    )
