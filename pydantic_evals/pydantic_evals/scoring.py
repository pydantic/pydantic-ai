from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from typing import Any, Generic

from typing_extensions import TypeVar

__all__ = ('ScoringContext',)

from pydantic_evals.span_tree import SpanTree

InputsT = TypeVar('InputsT', default=dict[str, Any])
OutputT = TypeVar('OutputT', default=dict[str, Any])
MetadataT = TypeVar('MetadataT', default=dict[str, Any])


@dataclass
class ScoringContext(Generic[InputsT, OutputT, MetadataT]):
    """Context for scoring an evaluation case."""

    name: str
    inputs: InputsT
    metadata: MetadataT
    expected_output: OutputT | None

    output: OutputT
    duration: float
    span_tree: SpanTree

    attributes: dict[str, Any]
    metrics: dict[str, int | float]

    # TODO: Add the set of otel spans that were created during the case as an attribute that can be accessed here
    #   This would be useful for things like checking for specific agent behaviors in a more extended behavior.
    # TODO: Allow storing a reason with the scores
    scores: dict[str, int | float] = field(init=False, default_factory=dict)
    # TODO: Allow storing a reason with the labels
    labels: dict[str, bool | str] = field(init=False, default_factory=dict)

    def __post_init__(self):
        print('-----')
        print(repr(self.span_tree))
        print('-----')

    def record_score(self, name: str, value: int | float) -> None:
        self.scores[name] = value

    def record_label(self, name: str, value: bool | str) -> None:
        self.labels[name] = value
