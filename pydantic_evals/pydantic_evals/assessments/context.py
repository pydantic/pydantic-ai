from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic

from typing_extensions import TypeVar

from ..otel.span_tree import SpanTree

# ScoringContext needs to be covariant
InputsT = TypeVar('InputsT', default=dict[str, Any], covariant=True)
OutputT = TypeVar('OutputT', default=dict[str, Any], covariant=True)
MetadataT = TypeVar('MetadataT', default=dict[str, Any], covariant=True)


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
