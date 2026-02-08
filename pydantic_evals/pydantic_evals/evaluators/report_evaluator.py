from __future__ import annotations

import inspect
from abc import abstractmethod
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, cast

from typing_extensions import TypeVar

from ..reporting.analyses import ReportAnalysis
from ._base import BaseEvaluator

if TYPE_CHECKING:
    from pydantic_evals.reporting import EvaluationReport

__all__ = (
    'ReportEvaluator',
    'ReportEvaluatorContext',
)

InputsT = TypeVar('InputsT', default=Any, contravariant=True)
OutputT = TypeVar('OutputT', default=Any, contravariant=True)
MetadataT = TypeVar('MetadataT', default=Any, contravariant=True)


@dataclass(kw_only=True)
class ReportEvaluatorContext(Generic[InputsT, OutputT, MetadataT]):
    """Context for report-level evaluation, containing the full experiment results."""

    name: str
    """The experiment name."""
    report: EvaluationReport[InputsT, OutputT, MetadataT]
    """The full evaluation report."""
    experiment_metadata: dict[str, Any] | None
    """Experiment-level metadata."""


@dataclass(repr=False)
class ReportEvaluator(BaseEvaluator, Generic[InputsT, OutputT, MetadataT]):
    """Base class for experiment-wide evaluators that analyze full reports.

    Unlike case-level Evaluators which assess individual task outputs,
    ReportEvaluators see all case results together and produce
    experiment-wide analyses like confusion matrices, precision-recall curves,
    or scalar statistics.
    """

    @abstractmethod
    def evaluate(
        self, ctx: ReportEvaluatorContext[InputsT, OutputT, MetadataT]
    ) -> ReportAnalysis | list[ReportAnalysis] | Awaitable[ReportAnalysis | list[ReportAnalysis]]:
        """Evaluate the full report and return experiment-wide analysis/analyses."""
        ...

    async def evaluate_async(
        self, ctx: ReportEvaluatorContext[InputsT, OutputT, MetadataT]
    ) -> ReportAnalysis | list[ReportAnalysis]:
        """Evaluate, handling both sync and async implementations."""
        output = self.evaluate(ctx)
        if inspect.iscoroutine(output):
            return await output
        return cast('ReportAnalysis | list[ReportAnalysis]', output)
