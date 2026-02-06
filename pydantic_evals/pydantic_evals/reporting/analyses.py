from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Discriminator


class ConfusionMatrix(BaseModel):
    """A confusion matrix comparing expected vs predicted labels across cases."""

    type: Literal['confusion_matrix'] = 'confusion_matrix'
    title: str = 'Confusion Matrix'
    description: str | None = None
    class_labels: list[str]
    """Ordered list of class labels (used for both axes)."""
    matrix: list[list[int]]
    """matrix[expected_idx][predicted_idx] = count of cases."""
    normalize: bool = False
    """If True, values are normalized to [0, 1] per row."""


class PrecisionRecallPoint(BaseModel):
    """A single point on a precision-recall curve."""

    threshold: float
    precision: float
    recall: float


class PrecisionRecallCurve(BaseModel):
    """A single precision-recall curve."""

    name: str
    """Name of this curve (e.g., experiment name or evaluator name)."""
    points: list[PrecisionRecallPoint]
    """Points on the curve, ordered by threshold."""
    auc: float | None = None
    """Area under the precision-recall curve."""


class PrecisionRecall(BaseModel):
    """Precision-recall curve data across cases."""

    type: Literal['precision_recall'] = 'precision_recall'
    title: str = 'Precision-Recall Curve'
    description: str | None = None
    curves: list[PrecisionRecallCurve]
    """One or more curves."""


class ScalarResult(BaseModel):
    """A single scalar statistic (e.g., F1 score, accuracy, BLEU)."""

    type: Literal['scalar'] = 'scalar'
    title: str
    description: str | None = None
    value: float | int
    unit: str | None = None
    """Optional unit label (e.g., '%', 'ms')."""


class TableResult(BaseModel):
    """A generic table of data (fallback for custom analyses)."""

    type: Literal['table'] = 'table'
    title: str
    description: str | None = None
    columns: list[str]
    """Column headers."""
    rows: list[list[str | int | float | bool | None]]
    """Row data, one list per row."""


ReportAnalysis = Annotated[
    ConfusionMatrix | PrecisionRecall | ScalarResult | TableResult,
    Discriminator('type'),
]
"""Discriminated union of all report-level analysis types."""
