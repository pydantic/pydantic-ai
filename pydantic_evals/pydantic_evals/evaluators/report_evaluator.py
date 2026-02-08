from __future__ import annotations

import inspect
from abc import abstractmethod
from collections.abc import Awaitable
from dataclasses import MISSING, dataclass, fields
from typing import TYPE_CHECKING, Any, Generic, cast

from pydantic import ConfigDict, model_serializer
from pydantic_core import to_jsonable_python
from pydantic_core.core_schema import SerializationInfo
from typing_extensions import TypeVar

from pydantic_ai import _utils

from ..reporting.analyses import ReportAnalysis
from .spec import EvaluatorSpec

if TYPE_CHECKING:
    from pydantic_evals.reporting import EvaluationReport

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
class ReportEvaluator(Generic[InputsT, OutputT, MetadataT]):
    """Base class for experiment-wide evaluators that analyze full reports.

    Unlike case-level Evaluators which assess individual task outputs,
    ReportEvaluators see all case results together and produce
    experiment-wide analyses like confusion matrices, precision-recall curves,
    or scalar statistics.
    """

    __pydantic_config__ = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def get_serialization_name(cls) -> str:
        """Get the name used for serialization."""
        return cls.__name__

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

    @model_serializer(mode='plain')
    def serialize(self, info: SerializationInfo) -> Any:
        """Serialize this ReportEvaluator to a JSON-serializable form."""
        return to_jsonable_python(
            self.as_spec(),
            context=info.context,
            serialize_unknown=True,
        )

    def as_spec(self) -> EvaluatorSpec:
        raw_arguments = self.build_serialization_arguments()

        arguments: None | tuple[Any,] | dict[str, Any]
        if len(raw_arguments) == 0:
            arguments = None
        elif len(raw_arguments) == 1:
            # Only use the compact tuple form if the single non-default field is the first
            # dataclass field, since the tuple form passes the value as the first positional arg.
            first_field_name = fields(self)[0].name
            key = next(iter(raw_arguments))
            if key == first_field_name:
                arguments = (raw_arguments[key],)
            else:
                arguments = raw_arguments
        else:
            arguments = raw_arguments

        return EvaluatorSpec(name=self.get_serialization_name(), arguments=arguments)

    def build_serialization_arguments(self) -> dict[str, Any]:
        """Build the arguments for serialization, excluding defaults."""
        raw_arguments: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if field.default is not MISSING:
                if value == field.default:
                    continue
            if field.default_factory is not MISSING:
                if value == field.default_factory():  # pragma: no branch
                    continue
            raw_arguments[field.name] = value
        return raw_arguments

    __repr__ = _utils.dataclasses_no_defaults_repr
