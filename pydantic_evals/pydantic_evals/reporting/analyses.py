from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Discriminator

__all__ = (
    'ConfusionMatrix',
    'LinePlot',
    'LinePlotCurve',
    'LinePlotPoint',
    'PrecisionRecall',
    'PrecisionRecallCurve',
    'PrecisionRecallPoint',
    'ReportAnalysis',
    'ScalarResult',
    'TableResult',
)


class ConfusionMatrix(BaseModel):
    """A confusion matrix comparing expected vs predicted labels across cases."""

    type: Literal['confusion_matrix'] = 'confusion_matrix'
    title: str = 'Confusion Matrix'
    description: str | None = None
    class_labels: list[str]
    """Ordered list of class labels (used for both axes)."""
    matrix: list[list[int]]
    """matrix[expected_idx][predicted_idx] = count of cases."""


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


class LinePlotPoint(BaseModel):
    """A single point on a line plot."""

    x: float
    y: float


class LinePlotCurve(BaseModel):
    """A single curve on a line plot."""

    name: str
    """Name of this curve (shown in legend)."""
    points: list[LinePlotPoint]
    """Points on the curve, ordered by x value."""
    style: Literal['solid', 'dashed'] = 'solid'
    """Line style for rendering."""
    step: Literal['start', 'middle', 'end'] | None = None
    """Step interpolation mode. Use `'end'` for empirical CDFs (right-continuous step functions)."""


class LinePlot(BaseModel):
    """A generic XY line plot with labeled axes, supporting multiple curves.

    Use this for ROC curves, KS plots, calibration curves, or any custom
    line chart that doesn't fit the specific PrecisionRecall type.
    """

    type: Literal['line_plot'] = 'line_plot'
    title: str
    description: str | None = None
    x_label: str
    """Label for the x-axis."""
    y_label: str
    """Label for the y-axis."""
    x_range: tuple[float, float] | None = None
    """Optional fixed range for x-axis (min, max)."""
    y_range: tuple[float, float] | None = None
    """Optional fixed range for y-axis (min, max)."""
    curves: list[LinePlotCurve]
    """One or more curves to plot."""


ReportAnalysis = Annotated[
    ConfusionMatrix | PrecisionRecall | ScalarResult | TableResult | LinePlot,
    Discriminator('type'),
]
"""Discriminated union of all report-level analysis types."""
