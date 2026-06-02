from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Literal, cast

from typing_extensions import assert_never

from ..reporting import ReportCase
from ..reporting.analyses import (
    ConfusionMatrix,
    LinePlot,
    LinePlotCurve,
    LinePlotPoint,
    PrecisionRecall,
    PrecisionRecallCurve,
    PrecisionRecallPoint,
    ReportAnalysis,
    ScalarResult,
)
from .report_evaluator import ReportEvaluator, ReportEvaluatorContext

__all__ = (
    'ConfusionMatrixEvaluator',
    'KolmogorovSmirnovEvaluator',
    'PrecisionRecallEvaluator',
    'ROCAUCEvaluator',
    'DEFAULT_REPORT_EVALUATORS',
)


# --- Shared helpers for binary classification evaluators ---


def _get_score(
    case: ReportCase[Any, Any, Any],
    score_key: str,
    score_from: Literal['scores', 'metrics'],
) -> float | None:
    if score_from == 'scores':
        result = case.scores.get(score_key)
        return float(result.value) if result else None
    elif score_from == 'metrics':
        val = case.metrics.get(score_key)
        return float(val) if val is not None else None
    assert_never(score_from)


def _get_positive(
    case: ReportCase[Any, Any, Any],
    positive_from: Literal['expected_output', 'assertions', 'labels'],
    positive_key: str | None,
) -> bool | None:
    if positive_from == 'expected_output':
        return bool(case.expected_output) if case.expected_output is not None else None
    elif positive_from == 'assertions':
        if positive_key is None:
            raise ValueError("'positive_key' is required when positive_from='assertions'")
        assertion = case.assertions.get(positive_key)
        return assertion.value if assertion else None
    elif positive_from == 'labels':
        if positive_key is None:
            raise ValueError("'positive_key' is required when positive_from='labels'")
        label = case.labels.get(positive_key)
        return bool(label.value) if label else None
    assert_never(positive_from)


def _extract_scored_cases(
    cases: list[ReportCase[Any, Any, Any]],
    score_key: str,
    score_from: Literal['scores', 'metrics'],
    positive_from: Literal['expected_output', 'assertions', 'labels'],
    positive_key: str | None,
) -> list[tuple[float, bool]]:
    """Extract (score, is_positive) pairs from report cases, skipping cases with missing data."""
    scored_cases: list[tuple[float, bool]] = []
    for case in cases:
        score = _get_score(case, score_key, score_from)
        is_positive = _get_positive(case, positive_from, positive_key)
        if score is None or is_positive is None:
            continue
        scored_cases.append((score, is_positive))
    return scored_cases


def _downsample(points: list[tuple[float, ...]], n: int) -> list[tuple[float, ...]]:
    """Downsample a list of points to at most n evenly spaced entries."""
    if len(points) <= n or n <= 1:
        return points
    indices = sorted({int(i * (len(points) - 1) / (n - 1)) for i in range(n)})
    return [points[i] for i in indices]


def _trapezoidal_auc(xy_points: list[tuple[float, float]]) -> float:
    """Compute area under curve using the trapezoidal rule on (x, y) points."""
    auc = 0.0
    for i in range(1, len(xy_points)):
        auc += abs(xy_points[i][0] - xy_points[i - 1][0]) * (xy_points[i][1] + xy_points[i - 1][1]) / 2
    return auc


# --- Evaluators ---


@dataclass(repr=False)
class ConfusionMatrixEvaluator(ReportEvaluator):
    """Computes a confusion matrix from case data."""

    predicted_from: Literal['expected_output', 'output', 'metadata', 'labels'] = 'output'
    predicted_key: str | None = None

    expected_from: Literal['expected_output', 'output', 'metadata', 'labels'] = 'expected_output'
    expected_key: str | None = None

    title: str = 'Confusion Matrix'

    def evaluate(self, ctx: ReportEvaluatorContext[Any, Any, Any]) -> ConfusionMatrix:
        predicted: list[str] = []
        expected: list[str] = []

        for case in ctx.report.cases:
            pred = self._extract(case, self.predicted_from, self.predicted_key)
            exp = self._extract(case, self.expected_from, self.expected_key)
            if pred is None or exp is None:
                continue
            predicted.append(pred)
            expected.append(exp)

        all_labels = sorted(set(predicted) | set(expected))
        label_to_idx = {label: i for i, label in enumerate(all_labels)}
        matrix = [[0] * len(all_labels) for _ in all_labels]

        for e, p in zip(expected, predicted):
            matrix[label_to_idx[e]][label_to_idx[p]] += 1

        return ConfusionMatrix(
            title=self.title,
            class_labels=all_labels,
            matrix=matrix,
        )

    @staticmethod
    def _extract(
        case: ReportCase[Any, Any, Any],
        from_: Literal['expected_output', 'output', 'metadata', 'labels'],
        key: str | None,
    ) -> str | None:
        if from_ == 'expected_output':
            return str(case.expected_output) if case.expected_output is not None else None
        elif from_ == 'output':
            return str(case.output) if case.output is not None else None
        elif from_ == 'metadata':
            if key is not None:
                if isinstance(case.metadata, dict):
                    metadata_dict = cast(dict[str, Any], case.metadata)  # pyright: ignore[reportUnknownMemberType]
                    val = metadata_dict.get(key)
                    return str(val) if val is not None else None
                return None  # key requested but metadata isn't a dict — skip this case
            return str(case.metadata) if case.metadata is not None else None
        elif from_ == 'labels':
            if key is None:
                raise ValueError("'key' is required when from_='labels'")
            label_result = case.labels.get(key)
            return label_result.value if label_result else None
        assert_never(from_)


@dataclass(repr=False)
class PrecisionRecallEvaluator(ReportEvaluator):
    """Computes a precision-recall curve from case data.

    Returns both a `PrecisionRecall` chart and a `ScalarResult` with the AUC value.
    The AUC is computed at full resolution (every unique score threshold) for accuracy,
    while the chart points are downsampled to `n_thresholds` for display.
    """

    score_key: str
    positive_from: Literal['expected_output', 'assertions', 'labels']
    positive_key: str | None = None

    score_from: Literal['scores', 'metrics'] = 'scores'

    title: str = 'Precision-Recall Curve'
    n_thresholds: int = 100

    def evaluate(self, ctx: ReportEvaluatorContext[Any, Any, Any]) -> list[ReportAnalysis]:
        scored_cases = _extract_scored_cases(
            ctx.report.cases, self.score_key, self.score_from, self.positive_from, self.positive_key
        )

        if not scored_cases:
            return [
                PrecisionRecall(title=self.title, curves=[]),
                ScalarResult(title=f'{self.title} AUC', value=float('nan')),
            ]

        total_positives = sum(1 for _, p in scored_cases if p)

        # Compute precision/recall at every unique score for exact AUC
        unique_thresholds = sorted({s for s, _ in scored_cases}, reverse=True)
        # Start with anchor at (recall=0, precision=1) — the "no predictions" point
        max_score = unique_thresholds[0]
        all_points: list[PrecisionRecallPoint] = [PrecisionRecallPoint(threshold=max_score, precision=1.0, recall=0.0)]
        for threshold in unique_thresholds:
            tp = sum(1 for s, p in scored_cases if s >= threshold and p)
            fp = sum(1 for s, p in scored_cases if s >= threshold and not p)
            fn = total_positives - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall = tp / (fn + tp) if (fn + tp) > 0 else 0.0
            all_points.append(PrecisionRecallPoint(threshold=threshold, precision=precision, recall=recall))

        # Exact AUC from the full-resolution points (anchor included)
        auc_points = [(p.recall, p.precision) for p in all_points]
        auc = _trapezoidal_auc(auc_points)

        # Downsample for display
        if len(all_points) <= self.n_thresholds or self.n_thresholds <= 1:
            display_points = all_points
        else:
            indices = sorted(
                {int(i * (len(all_points) - 1) / (self.n_thresholds - 1)) for i in range(self.n_thresholds)}
            )
            display_points = [all_points[i] for i in indices]

        curve = PrecisionRecallCurve(name=ctx.name, points=display_points, auc=auc)
        return [
            PrecisionRecall(title=self.title, curves=[curve]),
            ScalarResult(title=f'{self.title} AUC', value=auc),
        ]


@dataclass(repr=False)
class ROCAUCEvaluator(ReportEvaluator):
    """Computes an ROC curve and AUC from case data.

    Returns a `LinePlot` with the ROC curve (plus a dashed random-baseline diagonal)
    and a `ScalarResult` with the AUC value.
    """

    score_key: str
    positive_from: Literal['expected_output', 'assertions', 'labels']
    positive_key: str | None = None

    score_from: Literal['scores', 'metrics'] = 'scores'

    title: str = 'ROC Curve'
    n_thresholds: int = 100

    def evaluate(self, ctx: ReportEvaluatorContext[Any, Any, Any]) -> list[ReportAnalysis]:
        scored_cases = _extract_scored_cases(
            ctx.report.cases, self.score_key, self.score_from, self.positive_from, self.positive_key
        )

        empty_result: list[ReportAnalysis] = [
            LinePlot(
                title=self.title,
                x_label='False Positive Rate',
                y_label='True Positive Rate',
                x_range=(0, 1),
                y_range=(0, 1),
                curves=[],
            ),
            ScalarResult(title=f'{self.title} AUC', value=float('nan')),
        ]
        if not scored_cases:
            return empty_result

        total_positives = sum(1 for _, p in scored_cases if p)
        total_negatives = len(scored_cases) - total_positives

        if total_positives == 0 or total_negatives == 0:
            return empty_result

        # Compute TPR/FPR at every unique score for exact AUC
        unique_thresholds = sorted({s for s, _ in scored_cases}, reverse=True)
        all_fpr_tpr: list[tuple[float, float]] = [(0.0, 0.0)]
        for threshold in unique_thresholds:
            tp = sum(1 for s, p in scored_cases if s >= threshold and p)
            fp = sum(1 for s, p in scored_cases if s >= threshold and not p)
            tpr = tp / total_positives
            fpr = fp / total_negatives
            all_fpr_tpr.append((fpr, tpr))
        all_fpr_tpr.sort()

        # Exact AUC
        auc = _trapezoidal_auc(all_fpr_tpr)

        # Downsample for display
        downsampled = _downsample(all_fpr_tpr, self.n_thresholds)

        roc_curve = LinePlotCurve(
            name=f'{ctx.name} (AUC: {auc:.3f})',
            points=[LinePlotPoint(x=fpr, y=tpr) for fpr, tpr in downsampled],
        )
        baseline = LinePlotCurve(
            name='Random',
            points=[LinePlotPoint(x=0, y=0), LinePlotPoint(x=1, y=1)],
            style='dashed',
        )

        return [
            LinePlot(
                title=self.title,
                x_label='False Positive Rate',
                y_label='True Positive Rate',
                x_range=(0, 1),
                y_range=(0, 1),
                curves=[roc_curve, baseline],
            ),
            ScalarResult(title=f'{self.title} AUC', value=auc),
        ]


@dataclass(repr=False)
class KolmogorovSmirnovEvaluator(ReportEvaluator):
    """Computes a Kolmogorov-Smirnov plot and statistic from case data.

    Plots the empirical CDFs of the score distribution for positive and negative cases,
    and computes the KS statistic (maximum vertical distance between the two CDFs).

    Returns a `LinePlot` with the two CDF curves and a `ScalarResult` with the KS statistic.
    """

    score_key: str
    positive_from: Literal['expected_output', 'assertions', 'labels']
    positive_key: str | None = None

    score_from: Literal['scores', 'metrics'] = 'scores'

    title: str = 'KS Plot'
    n_thresholds: int = 100

    def evaluate(self, ctx: ReportEvaluatorContext[Any, Any, Any]) -> list[ReportAnalysis]:
        scored_cases = _extract_scored_cases(
            ctx.report.cases, self.score_key, self.score_from, self.positive_from, self.positive_key
        )

        empty_result: list[ReportAnalysis] = [
            LinePlot(
                title=self.title,
                x_label='Score',
                y_label='Cumulative Probability',
                y_range=(0, 1),
                curves=[],
            ),
            ScalarResult(title='KS Statistic', value=float('nan')),
        ]
        if not scored_cases:
            return empty_result

        pos_scores = sorted(s for s, p in scored_cases if p)
        neg_scores = sorted(s for s, p in scored_cases if not p)

        if not pos_scores or not neg_scores:
            return empty_result

        # Compute CDFs at all unique scores using binary search
        all_scores = sorted({s for s, _ in scored_cases})
        # Start both CDFs at y=0 at the minimum score
        pos_cdf: list[tuple[float, float]] = [(all_scores[0], 0.0)]
        neg_cdf: list[tuple[float, float]] = [(all_scores[0], 0.0)]
        ks_stat = 0.0

        for score in all_scores:
            pos_val = bisect_right(pos_scores, score) / len(pos_scores)
            neg_val = bisect_right(neg_scores, score) / len(neg_scores)
            pos_cdf.append((score, pos_val))
            neg_cdf.append((score, neg_val))
            ks_stat = max(ks_stat, abs(pos_val - neg_val))

        # Downsample for display
        display_pos = _downsample(pos_cdf, self.n_thresholds)
        display_neg = _downsample(neg_cdf, self.n_thresholds)

        pos_curve = LinePlotCurve(
            name='Positive',
            points=[LinePlotPoint(x=s, y=v) for s, v in display_pos],
            step='end',
        )
        neg_curve = LinePlotCurve(
            name='Negative',
            points=[LinePlotPoint(x=s, y=v) for s, v in display_neg],
            step='end',
        )

        return [
            LinePlot(
                title=self.title,
                x_label='Score',
                y_label='Cumulative Probability',
                y_range=(0, 1),
                curves=[pos_curve, neg_curve],
            ),
            ScalarResult(title='KS Statistic', value=ks_stat),
        ]


DEFAULT_REPORT_EVALUATORS: tuple[type[ReportEvaluator[Any, Any, Any]], ...] = (
    ConfusionMatrixEvaluator,
    KolmogorovSmirnovEvaluator,
    PrecisionRecallEvaluator,
    ROCAUCEvaluator,
)
