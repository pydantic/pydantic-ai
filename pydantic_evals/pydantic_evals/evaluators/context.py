"""This module defines the EvaluatorContext.

The EvaluatorContext is the context object that is passed to evaluators, containing all the information
needed to evaluate a task execution, including inputs, outputs, metadata, telemetry data, and more.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic

from typing_extensions import TypeVar

from ..otel._errors import SpanTreeRecordingError
from ..otel.span_tree import SpanTree

__all__ = ('EvaluatorContext',)

# ScoringContext needs to be covariant
InputsT = TypeVar('InputsT', default=Any, covariant=True)
"""Type variable for the inputs to the task being evaluated."""

OutputT = TypeVar('OutputT', default=Any, covariant=True)
"""Type variable for the output of the task being evaluated."""

MetadataT = TypeVar('MetadataT', default=Any, covariant=True)
"""Type variable for the metadata associated with the task being evaluated."""


@dataclass(kw_only=True)
class EvaluatorContext(Generic[InputsT, OutputT, MetadataT]):
    """Context for evaluating a task execution.

    An instance of this class is the sole input to all Evaluators. It contains all the information
    needed to evaluate the task execution, including inputs, outputs, metadata, and telemetry data.

    Evaluators use this context to access the task inputs, actual output, expected output, and other
    information when evaluating the result of the task execution.

    Example:
    ```python
    from dataclasses import dataclass

    from pydantic_evals.evaluators import Evaluator, EvaluatorContext


    @dataclass
    class ExactMatch(Evaluator):
        def evaluate(self, ctx: EvaluatorContext) -> bool:
            # Use the context to access task inputs, outputs, and expected outputs
            return ctx.output == ctx.expected_output
    ```
    """

    name: str | None
    """The name of the case."""
    inputs: InputsT
    """The inputs provided to the task for this case."""
    metadata: MetadataT | None
    """Metadata associated with the case, if provided. May be None if no metadata was specified."""
    expected_output: OutputT | None
    """The expected output for the case, if provided. May be None if no expected output was specified."""

    output: OutputT
    """The actual output produced by the task for this case."""
    duration: float
    """The duration of the task run for this case."""
    _span_tree: SpanTree | SpanTreeRecordingError = field(repr=False)
    """The span tree for the task run for this case.

    This will be `None` if `logfire.configure` has not been called.
    """

    attributes: dict[str, Any]
    """Attributes associated with the task run for this case.

    These can be set by calling `pydantic_evals.dataset.set_eval_attribute` in any code executed
    during the evaluation task."""
    metrics: dict[str, int | float]
    """Metrics associated with the task run for this case.

    These can be set by calling `pydantic_evals.dataset.increment_eval_metric` in any code executed
    during the evaluation task."""

    @property
    def span_tree(self) -> SpanTree:
        """Get the `SpanTree` for this task execution.

        The span tree is a graph where each node corresponds to an OpenTelemetry span recorded during the task
        execution, including timing information and any custom spans created during execution.

        Returns:
            The span tree for the task execution.

        Raises:
            SpanTreeRecordingError: If spans were not captured during execution of the task, e.g. due to not having
                the necessary dependencies installed.
        """
        if isinstance(self._span_tree, SpanTreeRecordingError):
            # In this case, there was a reason we couldn't record the SpanTree. We raise that now
            raise self._span_tree
        return self._span_tree
