"""A toolkit for evaluating the execution of arbitrary "stochastic functions", such as LLM calls.

This package provides functionality for:
- Creating and loading test datasets with structured inputs and outputs
- Evaluating model performance using various metrics and evaluators
- Generating reports for evaluation results
"""

from .dataset import Case, Dataset, increment_eval_metric, set_eval_attribute

__all__ = (
    'Case',
    'Dataset',
    'increment_eval_metric',
    'set_eval_attribute',
)
