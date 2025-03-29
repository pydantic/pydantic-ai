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

from .dataset import Case, Dataset

__all__ = (
    'Case',
    'Dataset',
)
