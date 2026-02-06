"""A toolkit for evaluating the execution of arbitrary "stochastic functions", such as LLM calls.

This package provides functionality for:
- Creating and loading test datasets with structured inputs and outputs
- Evaluating model performance using various metrics and evaluators
- Generating reports for evaluation results
"""

from .dataset import Case, Dataset, increment_eval_metric, set_eval_attribute
from .evaluators import ConfusionMatrixEvaluator, PrecisionRecallEvaluator, ReportEvaluator, ReportEvaluatorContext
from .reporting.analyses import (
    ConfusionMatrix,
    PrecisionRecall,
    PrecisionRecallCurve,
    PrecisionRecallPoint,
    ReportAnalysis,
    ScalarResult,
    TableResult,
)

__all__ = (
    'Case',
    'Dataset',
    'increment_eval_metric',
    'set_eval_attribute',
    # Report evaluators
    'ReportEvaluator',
    'ReportEvaluatorContext',
    'ConfusionMatrixEvaluator',
    'PrecisionRecallEvaluator',
    # Analysis types
    'ConfusionMatrix',
    'PrecisionRecall',
    'PrecisionRecallCurve',
    'PrecisionRecallPoint',
    'ScalarResult',
    'TableResult',
    'ReportAnalysis',
)
