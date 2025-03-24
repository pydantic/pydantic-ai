from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic

from typing_extensions import TypeVar

from ..otel._errors import SpanTreeRecordingError
from ..otel.span_tree import SpanTree

# ScoringContext needs to be covariant
InputsT = TypeVar('InputsT', default=dict[str, Any], covariant=True)
OutputT = TypeVar('OutputT', default=dict[str, Any], covariant=True)
MetadataT = TypeVar('MetadataT', default=dict[str, Any], covariant=True)


@dataclass
class EvaluatorContext(Generic[InputsT, OutputT, MetadataT]):
    """Context for scoring an evaluation case.

    An instance of this class is the sole input to all Evaluator functions.
    """

    name: str | None
    """The name of the case."""
    inputs: InputsT
    """The inputs to the task run for this case."""
    metadata: MetadataT | None
    """Metadata associated with the case, if provided."""
    expected_output: OutputT | None
    """The expected output for the case, if provided."""

    output: OutputT
    """The output of the task run for this case."""
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
        if isinstance(self._span_tree, SpanTreeRecordingError):
            # In this case, there was a reason we couldn't record the SpanTree. We raise that now
            raise self._span_tree
        return self._span_tree
