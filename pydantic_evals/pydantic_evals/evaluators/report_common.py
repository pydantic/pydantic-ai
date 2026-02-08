from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from typing_extensions import assert_never

from ..reporting import ReportCase
from ..reporting.analyses import (
    ConfusionMatrix,
    PrecisionRecall,
    PrecisionRecallCurve,
    PrecisionRecallPoint,
)
from .report_evaluator import ReportEvaluator, ReportEvaluatorContext

__all__ = (
    'ConfusionMatrixEvaluator',
    'PrecisionRecallEvaluator',
    'DEFAULT_REPORT_EVALUATORS',
)


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
    """Computes a precision-recall curve from case data."""

    score_key: str
    positive_from: Literal['expected_output', 'assertions', 'labels']
    positive_key: str | None = None

    score_from: Literal['scores', 'metrics'] = 'scores'

    title: str = 'Precision-Recall Curve'
    n_thresholds: int = 100

    def evaluate(self, ctx: ReportEvaluatorContext[Any, Any, Any]) -> PrecisionRecall:
        scored_cases: list[tuple[float, bool]] = []

        for case in ctx.report.cases:
            score = self._get_score(case)
            is_positive = self._get_positive(case)
            if score is None or is_positive is None:
                continue
            scored_cases.append((score, is_positive))

        scored_cases.sort(key=lambda x: -x[0])

        if not scored_cases:
            return PrecisionRecall(title=self.title, curves=[])

        scores = [s for s, _ in scored_cases]
        min_score, max_score = min(scores), max(scores)
        if min_score == max_score:
            thresholds = [min_score]
        else:
            step = (max_score - min_score) / self.n_thresholds
            thresholds = [min_score + i * step for i in range(self.n_thresholds + 1)]

        points: list[PrecisionRecallPoint] = []
        for threshold in thresholds:
            tp = sum(1 for s, p in scored_cases if s >= threshold and p)
            fp = sum(1 for s, p in scored_cases if s >= threshold and not p)
            fn = sum(1 for s, p in scored_cases if s < threshold and p)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            points.append(PrecisionRecallPoint(threshold=threshold, precision=precision, recall=recall))

        # Compute AUC using trapezoidal rule
        auc = 0.0
        for i in range(1, len(points)):
            auc += abs(points[i].recall - points[i - 1].recall) * (points[i].precision + points[i - 1].precision) / 2

        curve = PrecisionRecallCurve(name=ctx.name, points=points, auc=auc)
        return PrecisionRecall(title=self.title, curves=[curve])

    def _get_score(self, case: ReportCase[Any, Any, Any]) -> float | None:
        if self.score_from == 'scores':
            result = case.scores.get(self.score_key)
            return float(result.value) if result else None
        elif self.score_from == 'metrics':
            val = case.metrics.get(self.score_key)
            return float(val) if val is not None else None
        assert_never(self.score_from)

    def _get_positive(self, case: ReportCase[Any, Any, Any]) -> bool | None:
        if self.positive_from == 'expected_output':
            return bool(case.expected_output) if case.expected_output is not None else None
        elif self.positive_from == 'assertions':
            if self.positive_key is None:
                raise ValueError("'positive_key' is required when positive_from='assertions'")
            assertion = case.assertions.get(self.positive_key)
            return assertion.value if assertion else None
        elif self.positive_from == 'labels':
            if self.positive_key is None:
                raise ValueError("'positive_key' is required when positive_from='labels'")
            label = case.labels.get(self.positive_key)
            return bool(label.value) if label else None
        assert_never(self.positive_from)


DEFAULT_REPORT_EVALUATORS: tuple[type[ReportEvaluator[Any, Any, Any]], ...] = (
    ConfusionMatrixEvaluator,
    PrecisionRecallEvaluator,
)
