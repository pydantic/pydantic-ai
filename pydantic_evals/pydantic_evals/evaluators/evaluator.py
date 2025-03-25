from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, dataclass, fields
from typing import Any, Generic, cast

from pydantic import (
    TypeAdapter,
    ValidationError,
    model_serializer,
)
from pydantic.alias_generators import to_snake
from pydantic_core import to_jsonable_python
from typing_extensions import TypeVar

from .._utils import run_until_complete
from ._spec import EvaluatorSpec
from .context import EvaluatorContext

EvaluationScalar = bool | int | float | str


@dataclass
class EvaluationReason:
    """The result of running an evaluator."""

    value: EvaluationScalar
    reason: str | None = None


EvaluatorOutput = EvaluationScalar | EvaluationReason | Mapping[str, EvaluationScalar | EvaluationReason]


# TODO(DavidM): Add bound=EvaluationScalar to the following typevar after we upgrade to pydantic 2.11
EvaluationScalarT = TypeVar('EvaluationScalarT', default=EvaluationScalar, covariant=True)
T = TypeVar('T')


@dataclass
class EvaluationResult(Generic[EvaluationScalarT]):
    """The details of an individual evaluation result."""

    name: str
    value: EvaluationScalarT
    reason: str | None
    source: Evaluator

    def downcast(self, *value_types: type[T]) -> EvaluationResult[T] | None:
        if isinstance(self.value, value_types):
            return cast(EvaluationResult[T], self)
        return None


# Evaluators are contravariant in all of its parameters.
InputsT = TypeVar('InputsT', default=Any, contravariant=True)
OutputT = TypeVar('OutputT', default=Any, contravariant=True)
MetadataT = TypeVar('MetadataT', default=Any, contravariant=True)


@dataclass
class Evaluator(Generic[InputsT, OutputT, MetadataT]):
    """Base class for all evaluators."""

    def __post_init__(self):
        if type(self) is Evaluator:  # pragma: no cover
            raise TypeError('You should not instantiate Evaluator directly.')

    def __init_subclass__(cls):
        # Ensure that at least one of `evaluate` and `evaluate_async` is implemented on this class
        # and not inherited from the base class.
        # It's okay if it's on a parent class as long as the class isn't BaseEvaluator.
        if cls.evaluate is Evaluator.evaluate and cls.evaluate_async is Evaluator.evaluate_async:
            raise TypeError(f'{cls.__name__} must implement either `evaluate` or `evaluate_async`')

    @classmethod
    def name(cls) -> str:
        """Return the 'name' of this Evaluator to use during serialization."""
        return to_snake(cls.__name__)

    def evaluate(self, ctx: EvaluatorContext[InputsT, OutputT, MetadataT]) -> EvaluatorOutput:
        return run_until_complete(self.evaluate_async(ctx))

    async def evaluate_async(self, ctx: EvaluatorContext[InputsT, OutputT, MetadataT]) -> EvaluatorOutput:
        # If you need to prevent this from blocking, you can override this with:
        # return await anyio.to_thread.run_sync(self.evaluate, ctx)
        return self.evaluate(ctx)

    @model_serializer(mode='plain')
    def serialize(self) -> Any:
        """Serialize this Evaluator to a JSON-serializable form."""
        raw_arguments: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            # always exclude defaults:
            if field.default is not MISSING:
                if value == field.default:
                    continue
            if field.default_factory is not MISSING:
                if value == field.default_factory():
                    continue
            raw_arguments[field.name] = value

        arguments: None | tuple[Any,] | dict[str, Any]
        if len(raw_arguments) == 0:
            arguments = None
        elif len(raw_arguments) == 1:
            arguments = (next(iter(raw_arguments.values())),)
        else:
            arguments = raw_arguments
        return to_jsonable_python(EvaluatorSpec(name=self.name(), arguments=arguments))


async def run_evaluator(
    evaluator: Evaluator[InputsT, OutputT, MetadataT], ctx: EvaluatorContext[InputsT, OutputT, MetadataT]
) -> list[EvaluationResult]:
    """Run an evaluator and return the results."""
    raw_results = await evaluator.evaluate_async(ctx)

    try:
        results = _EVALUATOR_OUTPUT_ADAPTER.validate_python(raw_results)
    except ValidationError as e:
        raise ValueError(f'{evaluator!r}.evaluate returned a value of an invalid type: {raw_results!r}.') from e

    results = _convert_to_mapping(results, scalar_name=type(evaluator).__name__)

    details: list[EvaluationResult] = []
    for name, result in results.items():
        if not isinstance(result, EvaluationReason):
            result = EvaluationReason(value=result)
        details.append(EvaluationResult(name=name, value=result.value, reason=result.reason, source=evaluator))

    return details


_EVALUATOR_OUTPUT_ADAPTER = TypeAdapter[EvaluatorOutput](EvaluatorOutput)


def _convert_to_mapping(
    result: EvaluatorOutput, *, scalar_name: str
) -> Mapping[str, EvaluationScalar | EvaluationReason]:
    if isinstance(result, Mapping):
        return result
    return {scalar_name: result}
