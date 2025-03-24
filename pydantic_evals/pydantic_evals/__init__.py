"""Pydantic evals module.

TODO(DavidM): Implement serialization of reports for later comparison, and add git hashes etc.
  Note: I made pydantic_ai.evals.reports.EvalReport a BaseModel specifically to make this easier
TODO(DavidM): Add commit hash, timestamp, and other metadata to reports (like pytest-speed does), possibly in a dedicated struct
TODO(DavidM): Implement a CLI with some pytest-like filtering API to make it easier to run only specific cases
"""

# TODO: Question: How should we decorate functions to make it possible to eval them later?
#  E.g., could use some kind of `eval_function` decorator, which ensures that calls to the function send eval-review-compatible data to logfire
#  Basically we need to record the inputs and output. @logfire.instrument might be enough if we make it possible to record the output

# TODO: Rework so that:
#   * EvaluatorFunction becomes EvaluatorFactory, which returns what is currently called a BoundEvaluatorFunction
#       * the EvaluatorSpec "call" is used to look up the EvaluatorFactory, and its kwargs are used to call it and produce the BoundEvaluatorFunction
#   * BoundEvaluatorFunction gets renamed to Evaluator
#   * The EvaluatorSpec class becomes purely a (private?) means for serializing/deserializing Evaluator instances (with registered EvaluatorFactories)
#   * The thing currently called Evaluator goes away and we assume that serializing/deserializing between Evaluator and EvaluatorSpec is idempotent
from .dataset import Case, Dataset, increment_eval_metric
from .reporting import RenderNumberConfig, RenderValueConfig

__all__ = (
    'Case',
    'Dataset',
    'increment_eval_metric',
    'RenderNumberConfig',
    'RenderValueConfig',
)
