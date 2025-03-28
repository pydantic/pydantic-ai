"""A toolkit for evaluating the execution of arbitrary "stochastic functions", such as LLM calls.

This package provides functionality for:
- Creating and loading test datasets with structured inputs and outputs
- Evaluating model performance using various metrics and evaluators
- Generating reports for evaluation results

TODO(DavidM): Implement serialization of reports for later comparison, and add git hashes etc.
  Note: I made pydantic_ai.evals.reports.EvalReport a BaseModel specifically to make this easier
TODO(DavidM): Add commit hash, timestamp, and other metadata to reports (like pytest-speed does), possibly in a dedicated struct
TODO(DavidM): Implement a CLI with some pytest-like filtering API to make it easier to run only specific cases
"""

# TODO: Question: How should we decorate functions to make it possible to eval them later?
#  E.g., could use some kind of `eval_function` decorator, which ensures that calls to the function send eval-review-compatible data to logfire
#  Basically we need to record the inputs and output. @logfire.instrument might be enough if we make it possible to record the output

from .dataset import Case, Dataset, increment_eval_metric
from .reporting import RenderNumberConfig, RenderValueConfig

__all__ = (
    'Case',
    'Dataset',
    'increment_eval_metric',
    'RenderNumberConfig',
    'RenderValueConfig',
)
